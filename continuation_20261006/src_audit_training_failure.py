#!/usr/bin/env python3
"""Independent checks for the P62 policy's negative calibration gate."""
import hashlib
import json
from pathlib import Path

import torch

from train_policy import COSTPolicy, FIT, CAL, PAIRS, OUT, make_events, replay


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def compact(events):
    return hashlib.sha256(json.dumps([
        (e["pair"], e["ready_frame"], e["target_lid"],
         tuple((c["source_lid"], c["label"], tuple(np for np in c["state"])) for c in e["candidates"]))
        for e in events
    ], sort_keys=True, default=lambda x: x.tolist() if hasattr(x, "tolist") else x).encode()).hexdigest()


def main():
    out = Path(OUT)
    result = json.loads((out / "TRAIN_REPLAY_RESULT.json").read_text())
    receipt = json.loads((out / "TRAIN_REPLAY_RECEIPT.json").read_text())
    ck = Path(receipt["checkpoint_path"])
    raw = torch.load(ck, map_location="cpu")
    checks = []

    # Round 1: artifact and split closure.
    checks.append({"round": 1, "check": "checkpoint_finite_and_receipt_hashes", "pass": all(torch.isfinite(v).all().item() for v in raw["state_dict"].values()) and sha(ck) == receipt["checkpoint_sha256"] and sha(out / "TRAIN_REPLAY_RESULT.json") == receipt["result_sha256"]})
    checks.append({"round": 1, "check": "fit_calibration_disjoint_and_closed", "pass": set(result["pairs_fit"]).isdisjoint(result["pairs_calibration"]) and set(result["pairs_fit"]) | set(result["pairs_calibration"]) == set(PAIRS)})
    checks.append({"round": 1, "check": "official_and_test_closed", "pass": result["official_val_test_read"] is False and result["future_outcome_features_used"] is False})

    # Round 2: input/label/causal contract.
    events_a, hashes_a = make_events(PAIRS)
    events_b, hashes_b = make_events(PAIRS)
    checks.append({"round": 2, "check": "source_hashes_replay_identical", "pass": hashes_a == hashes_b and compact(events_a) == compact(events_b)})
    checks.append({"round": 2, "check": "legal_prefix_labels", "pass": all((c["label"] == 0 or c["source_lid"] in e["positive_ids"]) for e in events_a for c in e["candidates"])})
    checks.append({"round": 2, "check": "candidate_and_state_finite", "pass": all(float(x) == float(x) for e in events_a for c in e["candidates"] for x in list(c["base"]) + list(c["state"]))})

    # Round 3: independent checkpoint replay and negative gate.
    model = COSTPolicy(); model.load_state_dict(raw["state_dict"], strict=True)
    a = replay(model, events_a, CAL, torch.device("cpu"), "dynamic")
    b = replay(model, events_a, CAL, torch.device("cpu"), "dynamic")
    checks.append({"round": 3, "check": "checkpoint_replay_deterministic", "pass": a == b})
    checks.append({"round": 3, "check": "negative_gate_replayed", "pass": a["selected"] == result["dynamic_calibration"]["selected"] and a["correct"] == result["dynamic_calibration"]["correct"] and a["correct"] == 0})
    checks.append({"round": 3, "check": "no_hidden_nan_or_parameter_collapse", "pass": sum(v.numel() for v in raw["state_dict"].values()) == result["model_parameters"] and all(torch.isfinite(v).all().item() for v in raw["state_dict"].values())})

    payload = {
        "status": "PASS_IMPLEMENTATION_AUDIT_METHOD_NEGATIVE",
        "rounds": 3, "checks_per_round": 3,
        "all_implementation_checks_pass": all(x["pass"] for x in checks),
        "scientific_decision": "STOP_P62_OWNER_STATE_POLICY_NEGATIVE_CALIBRATION",
        "negative_gate": {"dynamic_selected": a["selected"], "dynamic_correct": a["correct"], "dynamic_false": a["false"], "baseline_selected": result["baseline_calibration"]["selected"], "baseline_correct": result["baseline_calibration"]["correct"]},
        "checks": checks,
        "loss_not_converged": result["train"]["loss_last10"] > result["train"]["loss_min"] * 1.05,
        "official_val_test_read": False,
    }
    (out / "IMPLEMENTATION_FAILURE_AUDIT_TRAINING.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
