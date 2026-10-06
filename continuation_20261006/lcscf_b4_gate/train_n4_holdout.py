#!/usr/bin/env python3
"""Train-only SCI-ID/B1--B4 representation gate.

This command is intentionally explicit about its split: it accepts configured
train pairs only and never imports val/test XML or official metrics. The
default is a tiny wiring run; a real holdout run must be requested with an
explicit step count and records all provenance in its output JSON.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Mapping

import torch

# Keep direct script execution independent of the caller's working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sci_id.data.episodes import PackedFrameEpisodeDataset, collate_packed_frame_episodes
from sci_id.losses.composition_intervention import CompositionInterventionLoss
from sci_id.models.full_model import build_arm_model


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "source_data/train_episodes_v1"
ARMS = ("B1", "B2", "B3", "B4", "P1")


def _select_rows(valid: torch.Tensor) -> torch.Tensor:
    return valid.nonzero(as_tuple=False).flatten()


def _assignment_loss(criterion, output, target, valid):
    rows = _select_rows(valid)
    if rows.numel() == 0:
        return output.dustbin_logit.sum() * 0.0
    return criterion.assignment(output.logits[rows], target[rows])


def _intervention_target(batch_target: torch.Tensor, rows: torch.Tensor, dustbin: bool = False):
    target = torch.zeros_like(batch_target[rows])
    if dustbin:
        target[:, -1] = True
    else:
        target.copy_(batch_target[rows])
    return target


def _masked_common_values(output, drop_output, common):
    factual_logits = output.candidate_logits[common].reshape(-1)
    drop_logits = drop_output.candidate_logits[common].reshape(-1)
    factual_maps = output.event_candidates[common].reshape(-1, output.event_candidates.shape[-1])
    drop_maps = drop_output.event_candidates[common].reshape(-1, drop_output.event_candidates.shape[-1])
    return factual_logits, drop_logits, factual_maps, drop_maps


def intervention_loss(
    criterion: CompositionInterventionLoss,
    batch: Mapping[str, object],
    output,
    remove_output,
    control_output,
    drop_output,
):
    valid = batch["supervision_valid"]
    factual = _assignment_loss(criterion, output, batch["factual_target"], valid)

    positive_rows = valid & batch["positive_mask"].any(dim=1)
    remove_target = _intervention_target(batch["factual_target"], _select_rows(positive_rows), dustbin=True)
    removed = criterion.assignment(
        remove_output.logits[_select_rows(positive_rows)], remove_target
    ) if bool(positive_rows.any()) else factual * 0.0

    control_rows = valid & batch["remove_control_mask"].any(dim=1)
    control_target = _intervention_target(batch["factual_target"], _select_rows(control_rows))
    control = criterion.assignment(
        control_output.logits[_select_rows(control_rows)], control_target
    ) if bool(control_rows.any()) else factual * 0.0

    positive_index = batch["positive_mask"].float().argmax(dim=1)
    negative_scores = output.candidate_logits.masked_fill(
        ~batch["known_negative_mask"], torch.finfo(output.candidate_logits.dtype).min
    )
    hard_index = negative_scores.argmax(dim=1)
    hard_rows = valid & batch["positive_mask"].any(dim=1) & batch["known_negative_mask"].any(dim=1)
    hard = factual * 0.0
    if bool(hard_rows.any()):
        hard = criterion.margin * 0.0 + torch.relu(
            criterion.margin
            - (
                output.candidate_logits[hard_rows, :].gather(1, positive_index[hard_rows, None]).squeeze(1)
                - output.candidate_logits[hard_rows, :].gather(1, hard_index[hard_rows, None]).squeeze(1)
            )
        ).mean()

    irrelevant = factual * 0.0
    map_irrelevant = factual * 0.0
    if bool(batch["known_negative_mask"].any()):
        # Only compare members retained by the drop batch. This is an explicit
        # no-shortcut consistency term; exact member choice is deterministic.
        retained = batch["mask"] & ~(
            batch["known_negative_mask"]
            & (batch["known_negative_mask"].cumsum(dim=1) == 1)
        )
        factual_logits, drop_logits, factual_maps, drop_maps = _masked_common_values(
            output, drop_output, retained
        )
        if factual_logits.numel():
            irrelevant = torch.nn.functional.mse_loss(factual_logits, drop_logits)
            map_irrelevant = torch.nn.functional.mse_loss(factual_maps, drop_maps)

    total = (
        factual
        + criterion.remove_weight * removed
        + criterion.control_weight * control
        + criterion.hard_weight * hard
        + criterion.irrelevant_weight * irrelevant
        + criterion.map_weight * map_irrelevant
    )
    return total, {
        "factual": float(factual.detach().cpu()),
        "remove_positive": float(removed.detach().cpu()),
        "remove_negative_control": float(control.detach().cpu()),
        "hard_negative": float(hard.detach().cpu()),
        "irrelevant_logit": float(irrelevant.detach().cpu()),
        "irrelevant_map": float(map_irrelevant.detach().cpu()),
    }


def _batch(dataset, step: int, batch_size: int, image_size, start_index: int):
    selected = [
        dataset[(start_index + step * batch_size + offset) % len(dataset)]
        for offset in range(batch_size)
    ]
    return collate_packed_frame_episodes(selected, image_size=image_size)


def _to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pairs", nargs="+", default=["23"])
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--prototypes", type=int, default=2)
    parser.add_argument("--roi-size", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/n4_holdout_smoke.json")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    dataset = PackedFrameEpisodeDataset(DATA_ROOT, pairs=args.train_pairs)
    device = torch.device(args.device)
    criterion = CompositionInterventionLoss()
    results = []
    for arm in ARMS:
        torch.manual_seed(9000 + sum(ord(char) for char in arm))
        model = build_arm_model(
            arm, channels=args.channels, prototypes=args.prototypes, output_size=args.roi_size
        ).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        history = []
        for step in range(args.steps):
            batch = _to_device(
                _batch(
                    dataset,
                    step,
                    args.batch_size,
                    (args.image_height, args.image_width),
                    args.start_index,
                ),
                device,
            )
            target_index = batch["target_frame_index"]
            source_index = batch["source_frame_index"]
            optimizer.zero_grad(set_to_none=True)
            target_maps, candidate_maps = model.encode_packed(
                batch["frame_images"], target_index, source_index,
                batch["target_boxes"], batch["candidate_boxes"], batch["mask"]
            )
            output = model.forward_maps(target_maps, candidate_maps, batch["mask"])

            positive_remove_mask = batch["mask"] & ~batch["positive_mask"]
            control_remove_mask = batch["mask"] & ~batch["remove_control_mask"]
            irrelevant_remove_mask = batch["mask"].clone()
            for row in range(irrelevant_remove_mask.shape[0]):
                indices = batch["known_negative_mask"][row].nonzero(as_tuple=False).flatten()
                if indices.numel():
                    irrelevant_remove_mask[row, indices[0]] = False

            remove_output = model.forward_maps(target_maps, candidate_maps, positive_remove_mask)
            control_output = model.forward_maps(target_maps, candidate_maps, control_remove_mask)
            drop_output = model.forward_maps(target_maps, candidate_maps, irrelevant_remove_mask)
            loss, terms = intervention_loss(
                criterion, batch, output, remove_output, control_output, drop_output
            )
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients:
                raise RuntimeError(f"{arm} produced no gradient")
            optimizer.step()
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "gradient_l1": float(sum(g.detach().abs().sum().cpu() for g in gradients)),
                    "unique_frames": int(batch["frame_images"].shape[0]),
                    "tensor_k": int(batch["mask"].shape[1]),
                    "valid_events": int(batch["supervision_valid"].sum()),
                    "terms": terms,
                }
            )
        results.append(
            {
                "arm": arm,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "history": history,
            }
        )
    payload = {
        "status": "PASS_N4_TRAINING_WIRING_ONLY",
        "interpretation": "train-only tiny holdout wiring; no official-val access and no performance claim",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "train_pairs": args.train_pairs,
        "steps": args.steps,
        "start_index": args.start_index,
        "batch_size": args.batch_size,
        "image_size": [args.image_height, args.image_width],
        "channels": args.channels,
        "roi_size": args.roi_size,
        "device": str(device),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
