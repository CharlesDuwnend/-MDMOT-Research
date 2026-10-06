#!/usr/bin/env python3
"""AutoAssign-initialized SCI-ID diagnostic on the frozen N4 holdouts.

This is a diagnostic comparison against the already completed random-init N4
run.  It keeps the original optimizer, pair holdouts, image contract, and
metrics, but runs only the three relevant arms (B1/B4/P1) with the strictly
compatible AutoAssign Caffe-BGR backbone initialization.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_n4_holdout as base  # noqa: E402
from sci_id.losses.composition_intervention import CompositionInterventionLoss  # noqa: E402
from sci_id.models.lcscf import transport_contract_loss  # noqa: E402


ARMS = ("B4", "LCSCF_B4")
CONFIGURED = base.CONFIGURED_TRAIN_PAIRS
ORIGINAL_ARM_INDEX = {"B4": 3, "LCSCF_B4": 3}
SEED_ARM = {"B4": "B4", "LCSCF_B4": "B4"}


def _mask_unknown_candidates(output, allowed):
    """Remove unmapped candidates from every supervised logit denominator."""
    result = copy.copy(output)
    negative_inf = torch.finfo(output.candidate_logits.dtype).min
    result.candidate_logits = output.candidate_logits.masked_fill(~allowed, negative_inf)
    result.logits = torch.cat((result.candidate_logits, output.dustbin_logit[:, None]), dim=1)
    result.event_candidates = output.event_candidates * allowed[:, :, None].to(output.event_candidates.dtype)
    return result



def _train_step_lcscf(model, optimizer, criterion, batch):
    optimizer.zero_grad(set_to_none=True)
    target_maps, candidate_maps, output = base._maps_and_output(model, batch)
    remove_output, control_output, drop_output = base._intervention_outputs(
        model, batch, target_maps, candidate_maps
    )
    allowed = batch["mask"] & ~batch["unmapped_mask"]
    output = _mask_unknown_candidates(output, allowed)
    remove_output = _mask_unknown_candidates(remove_output, allowed)
    control_output = _mask_unknown_candidates(control_output, allowed)
    drop_output = _mask_unknown_candidates(drop_output, allowed)
    safe_batch = dict(batch)
    safe_batch["mask"] = allowed
    loss, terms = base.intervention_loss(
        criterion, safe_batch, output, remove_output, control_output, drop_output
    )
    transport_loss = loss.detach() * 0.0
    transport_terms = {"cycle_gap": 0.0, "mass_gap": 0.0, "positive_entropy": 0.0}
    if getattr(model, "last_lcscf_stats", None) is not None:
        transport_loss, transport_terms = transport_contract_loss(
            model.last_lcscf_stats, batch
        )
        loss = loss + 0.20 * transport_loss
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("training step produced no parameter gradient")
    if not torch.isfinite(loss).item() or any(not torch.isfinite(g).all().item() for g in gradients):
        raise FloatingPointError("non-finite loss/gradient before optimizer update")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    if not torch.isfinite(norm).item():
        raise FloatingPointError("non-finite total gradient norm")
    optimizer.step()
    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
        raise FloatingPointError("non-finite parameter after optimizer update")
    terms = dict(terms)
    terms["transport_contract"] = float(transport_loss.detach().cpu())
    terms.update({"transport_" + key: value for key, value in transport_terms.items()})
    return output, float(loss.detach().cpu()), terms, float(
        sum(gradient.detach().abs().sum().cpu() for gradient in gradients)
    )

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


def verify_gpu(device: torch.device):
    if device.type != "cuda":
        return None
    binding = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not binding.startswith("GPU-"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be a physical GPU UUID")
    properties = torch.cuda.get_device_properties(device)
    if "A100" not in properties.name or properties.total_memory < 40000000000:
        raise RuntimeError("refusing non-40GB-A100 GPU")
    return {
        "cuda_visible_devices": binding,
        "logical_index": device.index,
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pairs", nargs="+", required=True)
    parser.add_argument("--eval-pairs", nargs="+", required=True)
    parser.add_argument("--steps-per-arm", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=800)
    parser.add_argument("--image-width", type=int, default=1333)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--prototypes", type=int, default=2)
    parser.add_argument("--roi-size", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backbone-init", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-eval-events", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--mean", nargs=3, type=float, default=[102.9801, 115.9465, 122.7717])
    parser.add_argument("--std", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--channel-order", choices=("rgb", "bgr"), default="bgr")
    return parser.parse_args()


def main():
    global ARMS
    args = parse_args()
    ARMS = tuple(args.arms)
    train_pairs, eval_pairs = set(args.train_pairs), set(args.eval_pairs)
    if not train_pairs or not eval_pairs or train_pairs & eval_pairs:
        raise ValueError("train/eval pairs must be non-empty and disjoint")
    if (train_pairs | eval_pairs) - CONFIGURED:
        raise ValueError("pairs outside frozen configured train set")
    args.backbone_init = args.backbone_init.expanduser().resolve()
    if not args.backbone_init.is_file():
        raise FileNotFoundError(args.backbone_init)
    if args.steps_per_arm < 1 or args.batch_size < 1:
        raise ValueError("steps-per-arm and batch-size must be positive")

    # The imported helper functions read this module-global list.
    base.ARMS = ARMS
    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("diagnostic run directory is non-empty: %s" % run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    gpu = verify_gpu(device)

    image_size = (args.image_height, args.image_width)
    train_dataset = base.PackedFrameEpisodeDataset(base.DATA_ROOT, pairs=sorted(train_pairs))
    eval_dataset = base.PackedFrameEpisodeDataset(base.DATA_ROOT, pairs=sorted(eval_pairs))
    meta = {
        "status": "RUNNING",
        "classification": "initialization_diagnostic_not_confirmatory",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "train_pairs": sorted(train_pairs), "eval_pairs": sorted(eval_pairs),
        "steps_per_arm": args.steps_per_arm, "batch_size": args.batch_size,
        "image_size": list(image_size), "channels": args.channels,
        "prototypes": args.prototypes, "roi_size": args.roi_size,
        "device": str(device), "arms": list(ARMS),
        "backbone_init": str(args.backbone_init),
        "backbone_init_sha256": sha256_file(args.backbone_init),
        "mean": list(args.mean), "std": list(args.std),
        "value_scale": args.value_scale, "channel_order": args.channel_order,
        "official_val_access": False, "test_access": False,
        "seed_offset": args.seed_offset,
        "transport_loss_weight": 0.20,
        "gradient_clip_norm": 10.0,
        "preprocess_contract": "MMCV/cv2 keep_ratio (1333,800); Caffe BGR mean [102.9801,115.9465,122.7717], std 1; normalize then pad32 and batch pad; no flip",
        "supervision_scope": "known-label subset of frozen train candidates; unknown candidates remain unlabeled model context, excluded from supervised logit denominators and ranks",
        "finite_checks": "loss/all gradients before clip; total norm; all parameters after update",
        "pair_is_independent_unit": True,
        "gpu": gpu,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    criterion = CompositionInterventionLoss()
    event_log = run_dir / "events.jsonl"

    with event_log.open("w", encoding="utf-8") as log_handle:
        for arm in ARMS:
            base._set_seed(9000 + args.seed_offset + sum(ord(char) for char in SEED_ARM.get(arm, arm)))
            order = list(range(len(train_dataset)))
            random.Random(12000 + ORIGINAL_ARM_INDEX[arm]).shuffle(order)
            model = base.build_arm_model(
                arm, channels=args.channels, prototypes=args.prototypes,
                output_size=args.roi_size, backbone_init=args.backbone_init,
            ).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            cursor = 0
            for step in range(args.steps_per_arm):
                batch = base._to_device(base._batch(
                    train_dataset, order, cursor, args.batch_size, image_size,
                    args.mean, args.std, args.value_scale, args.channel_order,
                ), device)
                cursor = (cursor + args.batch_size) % len(train_dataset)
                _output, loss, terms, gradient_l1 = _train_step_lcscf(model, optimizer, criterion, batch)
                log_handle.write(json.dumps({
                    "event": "train", "arm": arm, "step": step + 1,
                    "loss": loss, "terms": terms, "gradient_l1": gradient_l1,
                    "valid_events": int(batch["supervision_valid"].sum()),
                    "unique_frames": int(batch["frame_images"].shape[0]),
                    "tensor_k": int(batch["mask"].shape[1]),
                }, ensure_ascii=False) + "\n")
                if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps_per_arm:
                    atomic_save({"arm": arm, "step": step + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, run_dir / "checkpoints" / (arm + "_latest.pt"))
                log_handle.flush()
            metrics = base._evaluate(
                model, eval_dataset, device, image_size, args.batch_size,
                args.max_eval_events, log_handle, args.mean, args.std,
                args.value_scale, args.channel_order,
            )
            (run_dir / ("metrics_%s.json" % arm)).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            log_handle.write(json.dumps({"event": "eval_summary", "arm": arm, **metrics}, ensure_ascii=False) + "\n")
            log_handle.flush()
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    meta["status"] = "COMPLETE"
    meta["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["event_log_sha256"] = sha256_file(event_log)
    meta["metric_files"] = {arm: sha256_file(run_dir / ("metrics_%s.json" % arm)) for arm in ARMS}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": meta["status"], "run_dir": str(run_dir), "arms": list(ARMS), "eval_pairs": sorted(eval_pairs)}))


if __name__ == "__main__":
    main()
