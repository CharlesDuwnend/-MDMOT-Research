#!/usr/bin/env python3
"""Independent post-run audit for the AutoAssign-initialized CVSGM gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_checkpoint(path: Path) -> bool:
    payload = torch.load(str(path), map_location="cpu")
    state = payload.get("model", payload)
    return all(torch.isfinite(value).all().item() for value in state.values() if isinstance(value, torch.Tensor))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.run_dir / "run_meta.json").read_text())
    arms = list(meta["arms"])
    metrics = {arm: json.loads((args.run_dir / ("metrics_%s.json" % arm)).read_text()) for arm in arms}
    pair_sets = {arm: sorted(metrics[arm]["pair_metrics"]) for arm in arms}
    event_counts = {
        arm: {pair: metrics[arm]["pair_metrics"][pair]["events"] for pair in pair_sets[arm]}
        for arm in arms
    }
    equal_pairs = len(set(tuple(value) for value in pair_sets.values())) == 1
    equal_events = len(set(tuple(sorted(value.items())) for value in event_counts.values())) == 1
    wins = {}
    for reference in ("B1", "B4"):
        wins[reference] = sum(
            metrics["CVSGM"]["pair_metrics"][pair]["candidate_present_recall1"]
            > metrics[reference]["pair_metrics"][pair]["candidate_present_recall1"]
            for pair in pair_sets["CVSGM"]
        )
    macro = {
        arm: metrics[arm]["pair_macro"] for arm in arms
    }
    checkpoints = {}
    for arm in arms:
        path = args.run_dir / "checkpoints" / (arm + "_latest.pt")
        checkpoints[arm] = {
            "path": str(path),
            "sha256": sha256(path),
            "finite": finite_checkpoint(path),
        }
    checks = {
        "run_complete": meta.get("status") == "COMPLETE",
        "same_backbone_sha": len({meta.get("backbone_init_sha256")}) == 1,
        "same_pair_set": equal_pairs,
        "same_pair_event_denominators": equal_events,
        "all_expected_steps": meta.get("steps_per_arm") == 600,
        "all_checkpoints_finite": all(item["finite"] for item in checkpoints.values()),
        "physical_gpu_is_A100_40GB": meta.get("gpu", {}).get("name", "").find("A100") >= 0 and meta.get("gpu", {}).get("total_memory_bytes", 0) >= 40_000_000_000,
        "gpu2_not_used": meta.get("gpu", {}).get("cuda_visible_devices") != "GPU-6f385048-7771-5048-9c46-067bf301ea27",
        "official_val_test_locked": meta.get("official_val_access") is False and meta.get("test_access") is False,
        "cvsgm_wins_at_least_three_pairs_vs_b1": wins["B1"] >= 3,
        "cvsgm_wins_at_least_three_pairs_vs_b4": wins["B4"] >= 3,
        "cvsgm_macro_r1_above_b1_b4": macro["CVSGM"]["candidate_present_recall1"] > macro["B1"]["candidate_present_recall1"] and macro["CVSGM"]["candidate_present_recall1"] > macro["B4"]["candidate_present_recall1"],
    }
    status = "PASS_SHORT_GATE_ADVANCE_SEED17" if all(checks.values()) else "HOLD_SHORT_GATE_AUDIT"
    payload = {
        "status": status,
        "run_dir": str(args.run_dir.resolve()),
        "checks": checks,
        "pair_event_counts": event_counts,
        "pair_recall1_wins_vs_reference": wins,
        "pair_macro": macro,
        "checkpoints": checkpoints,
        "interpretation": "single-seed short train-only representation gate; not official-val, tracker, or paper performance",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if status != "PASS_SHORT_GATE_ADVANCE_SEED17":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
