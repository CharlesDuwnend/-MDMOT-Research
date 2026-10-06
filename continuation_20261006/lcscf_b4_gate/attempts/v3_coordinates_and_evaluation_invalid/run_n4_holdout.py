#!/usr/bin/env python3
"""Resumable train-pair holdout runner for the SCI-ID representation gate.

The script deliberately has no val/test path and no MOT evaluator import. It
trains each of B1--B4/P1 on configured MDMT train pairs, evaluates on a
disjoint configured train pair, and records event predictions plus pair-level
macro metrics. It is suitable for a long run only after the short dry-run has
been inspected.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

# Keep direct script execution independent of the caller's working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sci_id.data.episodes import PackedFrameEpisodeDataset, collate_packed_frame_episodes
from sci_id.models.full_model import build_arm_model
from train_n4_holdout import intervention_loss


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "source_data/train_episodes_v1"
ARMS = ("B1", "B2", "B3", "B4", "P1")
CONFIGURED_TRAIN_PAIRS = {
    "23", "25", "27", "28", "29", "30", "32", "39", "42", "44", "45",
    "50", "51", "53", "54", "58", "63", "64", "65", "66", "69", "70",
    "74", "76", "78",
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _batch(
    dataset,
    order: Sequence[int],
    cursor: int,
    batch_size: int,
    image_size: Tuple[int, int],
    mean: Sequence[float],
    std: Sequence[float],
    value_scale: float,
    channel_order: str,
):
    selected = [dataset[order[(cursor + offset) % len(order)]] for offset in range(batch_size)]
    return collate_packed_frame_episodes(
        selected,
        image_size=image_size,
        mean=mean,
        std=std,
        value_scale=value_scale,
        channel_order=channel_order,
    )


def _maps_and_output(model, batch):
    target_maps, candidate_maps = model.encode_packed(
        batch["frame_images"],
        batch["target_frame_index"],
        batch["source_frame_index"],
        batch["target_boxes"],
        batch["candidate_boxes"],
        batch["mask"],
    )
    return target_maps, candidate_maps, model.forward_maps(
        target_maps, candidate_maps, batch["mask"]
    )


def _intervention_outputs(model, batch, target_maps, candidate_maps):
    positive_remove_mask = batch["mask"] & ~batch["positive_mask"]
    control_remove_mask = batch["mask"] & ~batch["remove_control_mask"]
    irrelevant_remove_mask = batch["mask"].clone()
    for row in range(irrelevant_remove_mask.shape[0]):
        indices = batch["known_negative_mask"][row].nonzero(as_tuple=False).flatten()
        if indices.numel():
            irrelevant_remove_mask[row, indices[0]] = False
    return (
        model.forward_maps(target_maps, candidate_maps, positive_remove_mask),
        model.forward_maps(target_maps, candidate_maps, control_remove_mask),
        model.forward_maps(target_maps, candidate_maps, irrelevant_remove_mask),
    )


def _train_step(model, optimizer, criterion, batch):
    optimizer.zero_grad(set_to_none=True)
    target_maps, candidate_maps, output = _maps_and_output(model, batch)
    remove_output, control_output, drop_output = _intervention_outputs(
        model, batch, target_maps, candidate_maps
    )
    loss, terms = intervention_loss(
        criterion, batch, output, remove_output, control_output, drop_output
    )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("training step produced no parameter gradient")
    optimizer.step()
    return output, float(loss.detach().cpu()), terms, float(
        sum(gradient.detach().abs().sum().cpu() for gradient in gradients)
    )


def _empty_pair_stats() -> Dict[str, float]:
    return {
        "events": 0,
        "candidate_present": 0,
        "rank1": 0,
        "mrr_sum": 0.0,
        "margin_sum": 0.0,
        "margin_count": 0,
        "no_match": 0,
        "dustbin_brier_sum": 0.0,
        "dustbin_shift_remove_sum": 0.0,
        "dustbin_shift_control_sum": 0.0,
        "intervention_count": 0,
    }


def _evaluate(
    model,
    dataset,
    device,
    image_size,
    batch_size,
    max_events: Optional[int],
    log_handle,
    mean,
    std,
    value_scale,
    channel_order,
):
    model.eval()
    pair_stats: Dict[str, Dict[str, float]] = {}
    eval_order = list(range(len(dataset)))
    seen = 0
    cursor = 0
    with torch.no_grad():
        while seen < len(dataset) and (max_events is None or seen < max_events):
            current_size = min(batch_size, len(dataset) - seen)
            if max_events is not None:
                current_size = min(current_size, max_events - seen)
            batch = _to_device(
                _batch(
                    dataset,
                    eval_order,
                    cursor,
                    current_size,
                    image_size,
                    mean,
                    std,
                    value_scale,
                    channel_order,
                ),
                device,
            )
            cursor = (cursor + current_size) % len(dataset)
            _, _, output = _maps_and_output(model, batch)
            probabilities = torch.softmax(output.logits, dim=-1)
            known_candidate_mask = batch["mask"] & ~batch["unmapped_mask"]
            positive_indices = batch["positive_mask"] & known_candidate_mask
            valid = batch["supervision_valid"]
            known_negative = batch["known_negative_mask"]
            for row in range(current_size):
                pair = str(batch["pair_ids"][row])
                stats = pair_stats.setdefault(pair, _empty_pair_stats())
                if not bool(valid[row]):
                    continue
                stats["events"] += 1
                candidate_scores = output.candidate_logits[row]
                candidate_mask = known_candidate_mask[row]
                positives = positive_indices[row] & candidate_mask
                dustbin_probability = float(probabilities[row, -1].cpu())
                if bool(positives.any()):
                    stats["candidate_present"] += 1
                    ranking = torch.argsort(candidate_scores, descending=True)
                    positive_set = set(positives.nonzero(as_tuple=False).flatten().tolist())
                    rank = next(
                        rank_index
                        for rank_index, index in enumerate(ranking.tolist(), 1)
                        if index in positive_set
                    )
                    stats["rank1"] += int(rank == 1)
                    stats["mrr_sum"] += 1.0 / rank
                    negatives = known_negative[row] & candidate_mask
                    if bool(negatives.any()):
                        stats["margin_sum"] += float(
                            candidate_scores[positives].max()
                            - candidate_scores[negatives].max()
                        )
                        stats["margin_count"] += 1
                    target_dustbin = 0.0
                else:
                    stats["no_match"] += 1
                    target_dustbin = 1.0
                stats["dustbin_brier_sum"] += (dustbin_probability - target_dustbin) ** 2
                log_handle.write(
                    json.dumps(
                        {
                            "event": "eval",
                            "pair_id": pair,
                            "episode_id": str(batch["episode_ids"][row]),
                            "candidate_present": bool(positives.any()),
                            "rank1": bool(positives.any() and rank == 1) if bool(positives.any()) else None,
                            "mrr": 1.0 / rank if bool(positives.any()) else None,
                            "dustbin_probability": dustbin_probability,
                            "dustbin_target": target_dustbin,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            seen += current_size
    model.train()

    pair_metrics = {}
    macro = {
        "pairs": len(pair_stats),
        "candidate_present_recall1": [],
        "candidate_present_mrr": [],
        "candidate_present_margin": [],
        "dustbin_brier": [],
        "no_match_events": 0,
    }
    for pair, stats in sorted(pair_stats.items()):
        present = stats["candidate_present"]
        metrics = {
            "events": int(stats["events"]),
            "candidate_present_events": int(present),
            "candidate_present_recall1": stats["rank1"] / present if present else None,
            "candidate_present_mrr": stats["mrr_sum"] / present if present else None,
            "candidate_present_margin": stats["margin_sum"] / stats["margin_count"] if stats["margin_count"] else None,
            "no_match_events": int(stats["no_match"]),
            "dustbin_brier": stats["dustbin_brier_sum"] / stats["events"] if stats["events"] else None,
        }
        pair_metrics[pair] = metrics
        if metrics["candidate_present_recall1"] is not None:
            macro["candidate_present_recall1"].append(metrics["candidate_present_recall1"])
            macro["candidate_present_mrr"].append(metrics["candidate_present_mrr"])
        if metrics["candidate_present_margin"] is not None:
            macro["candidate_present_margin"].append(metrics["candidate_present_margin"])
        if metrics["dustbin_brier"] is not None:
            macro["dustbin_brier"].append(metrics["dustbin_brier"])
        macro["no_match_events"] += metrics["no_match_events"]
    for key in ("candidate_present_recall1", "candidate_present_mrr", "candidate_present_margin", "dustbin_brier"):
        values = macro[key]
        macro[key] = float(sum(values) / len(values)) if values else None
    return {"pair_metrics": pair_metrics, "pair_macro": macro, "evaluated_events": seen}


def _atomic_torch_save(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


def _save_checkpoint(run_dir, arm_index, step, cursor, model, optimizer, args):
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "arm_index": arm_index,
        "arm": ARMS[arm_index],
        "step": step,
        "cursor": cursor,
        "shuffle_seed": 12000 + arm_index,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _atomic_torch_save(payload, checkpoint_dir / "latest.pt")
    # Periodic archives can consume tens of GB over a multi-pair study. Keep
    # latest.pt for recovery and always preserve the final state of each arm;
    # intermediate archives are opt-in.
    if args.keep_periodic_checkpoints or step == args.steps_per_arm:
        _atomic_torch_save(payload, checkpoint_dir / f"{ARMS[arm_index]}_step_{step:06d}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pairs", nargs="+", required=True)
    parser.add_argument("--eval-pairs", nargs="+", required=True)
    parser.add_argument("--steps-per-arm", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--prototypes", type=int, default=4)
    parser.add_argument("--roi-size", type=int, default=7)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-eval-events", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--keep-periodic-checkpoints", action="store_true")
    parser.add_argument("--backbone-init", type=Path, default=None)
    parser.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument("--value-scale", type=float, default=255.0)
    parser.add_argument("--channel-order", choices=("rgb", "bgr"), default="rgb")
    args = parser.parse_args()
    if args.steps_per_arm < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        raise ValueError("steps-per-arm, batch-size and checkpoint-every must be positive")
    if args.value_scale <= 0:
        raise ValueError("value-scale must be positive")
    if args.backbone_init is not None:
        args.backbone_init = args.backbone_init.expanduser().resolve()
        if not args.backbone_init.is_file():
            raise FileNotFoundError(f"backbone checkpoint not found: {args.backbone_init}")
    train_pairs, eval_pairs = set(args.train_pairs), set(args.eval_pairs)
    if not train_pairs or not eval_pairs or train_pairs & eval_pairs:
        raise ValueError("train/eval pairs must be non-empty and disjoint")
    invalid = sorted((train_pairs | eval_pairs) - CONFIGURED_TRAIN_PAIRS)
    if invalid:
        raise ValueError(f"only configured train pairs are allowed: {invalid}")
    # The output path is explicit and may be a pytest/temp or external volume;
    # data access remains restricted to DATA_ROOT and configured train pairs.
    run_dir = args.run_dir.expanduser().resolve()
    if run_dir == Path(run_dir.anchor):
        raise ValueError("run-dir must not be a filesystem root")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    image_size = (args.image_height, args.image_width)
    train_dataset = PackedFrameEpisodeDataset(DATA_ROOT, pairs=sorted(train_pairs))
    eval_dataset = PackedFrameEpisodeDataset(DATA_ROOT, pairs=sorted(eval_pairs))
    meta = {
        "status": "RUNNING",
        "interpretation": "train-only pair holdout; no official-val/test access and no MOT metrics",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "train_pairs": sorted(train_pairs),
        "eval_pairs": sorted(eval_pairs),
        "steps_per_arm": args.steps_per_arm,
        "batch_size": args.batch_size,
        "image_size": list(image_size),
        "channels": args.channels,
        "prototypes": args.prototypes,
        "roi_size": args.roi_size,
        "device": str(device),
        "arms": list(ARMS),
        "shuffle_seed_base": 12000,
        "checkpoint_every": args.checkpoint_every,
        "keep_periodic_checkpoints": args.keep_periodic_checkpoints,
        "backbone_init": str(args.backbone_init) if args.backbone_init is not None else None,
        "backbone_init_sha256": hashlib.sha256(args.backbone_init.read_bytes()).hexdigest() if args.backbone_init is not None else None,
        "mean": list(args.mean),
        "std": list(args.std),
        "value_scale": args.value_scale,
        "channel_order": args.channel_order,
    }
    metadata_path = run_dir / "run_meta.json"
    if not args.resume:
        metadata_path.write_text(json.dumps(meta, indent=2) + "\n")
    else:
        if not metadata_path.is_file():
            raise FileNotFoundError("--resume requires an existing run_meta.json")
        old_meta = json.loads(metadata_path.read_text())
        for key in ("train_pairs", "eval_pairs", "steps_per_arm", "batch_size", "image_size", "channels", "prototypes", "roi_size", "shuffle_seed_base", "checkpoint_every", "keep_periodic_checkpoints", "backbone_init", "mean", "std", "value_scale", "channel_order"):
            if old_meta.get(key) != meta[key]:
                raise ValueError(f"resume configuration mismatch for {key}")

    log_path = run_dir / "events.jsonl"
    checkpoint_path = run_dir / "checkpoints/latest.pt"
    resume_payload = None
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError("--resume requires checkpoints/latest.pt")
        resume_payload = torch.load(checkpoint_path, map_location=device)
    start_arm_index = int(resume_payload["arm_index"]) if resume_payload is not None else 0
    start_step = int(resume_payload["step"]) if resume_payload is not None else 0
    start_cursor = int(resume_payload["cursor"]) if resume_payload is not None else 0
    criterion = __import__("sci_id.losses.composition_intervention", fromlist=["CompositionInterventionLoss"]).CompositionInterventionLoss()
    with log_path.open("a", encoding="utf-8") as log_handle:
        for arm_index in range(start_arm_index, len(ARMS)):
            arm = ARMS[arm_index]
            # Keep episode order shuffled but exactly reproducible for resume.
            train_order = list(range(len(train_dataset)))
            random.Random(12000 + arm_index).shuffle(train_order)
            if arm_index == start_arm_index and resume_payload is not None:
                if int(resume_payload.get("shuffle_seed", -1)) != 12000 + arm_index:
                    raise ValueError("resume checkpoint shuffle seed mismatch")
                model = build_arm_model(arm, channels=args.channels, prototypes=args.prototypes, output_size=args.roi_size, backbone_init=args.backbone_init).to(device).train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                model.load_state_dict(resume_payload["model"])
                optimizer.load_state_dict(resume_payload["optimizer"])
                step_start, cursor = start_step, start_cursor
            else:
                _set_seed(9000 + sum(ord(char) for char in arm))
                model = build_arm_model(arm, channels=args.channels, prototypes=args.prototypes, output_size=args.roi_size, backbone_init=args.backbone_init).to(device).train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                step_start, cursor = 0, 0
            for step in range(step_start, args.steps_per_arm):
                batch = _to_device(_batch(train_dataset, train_order, cursor, args.batch_size, image_size, args.mean, args.std, args.value_scale, args.channel_order), device)
                cursor = (cursor + args.batch_size) % len(train_dataset)
                _, loss, terms, gradient_l1 = _train_step(model, optimizer, criterion, batch)
                log_handle.write(json.dumps({
                    "event": "train",
                    "arm": arm,
                    "step": step,
                    "loss": loss,
                    "terms": terms,
                    "gradient_l1": gradient_l1,
                    "valid_events": int(batch["supervision_valid"].sum()),
                    "unique_frames": int(batch["frame_images"].shape[0]),
                    "tensor_k": int(batch["mask"].shape[1]),
                }, ensure_ascii=False) + "\n")
                log_handle.flush()
                if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps_per_arm:
                    _save_checkpoint(run_dir, arm_index, step + 1, cursor, model, optimizer, args)
            metrics = _evaluate(
                model,
                eval_dataset,
                device,
                image_size,
                args.batch_size,
                args.max_eval_events,
                log_handle,
                args.mean,
                args.std,
                args.value_scale,
                args.channel_order,
            )
            log_handle.write(json.dumps({"event": "eval_summary", "arm": arm, **metrics}, ensure_ascii=False) + "\n")
            log_handle.flush()
            (run_dir / f"metrics_{arm}.json").write_text(json.dumps(metrics, indent=2) + "\n")
            if arm_index + 1 < len(ARMS):
                resume_payload = None
                start_step = 0
                start_cursor = 0
        meta["status"] = "COMPLETE"
        meta["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"status": meta["status"], "run_dir": str(run_dir), "log": str(log_path)}))


if __name__ == "__main__":
    main()
