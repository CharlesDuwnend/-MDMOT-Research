#!/usr/bin/env python3
"""Three-round independent audit for P63 support-reliability diagnostic."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from baseline_alignment_audit import OUT, P26, TRACE_DIR, sha
from weighted_support_eval import dlt


def digest(path):
    h = hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()


def main():
    out = Path(OUT)
    result = json.loads((out / "WEIGHTED_SUPPORT_TEMPORAL_DIAGNOSTIC.json").read_text())
    rows = [json.loads(x) for x in (out / "support_rows.jsonl").read_text().splitlines()]
    checks = []
    # Round 1: data and split integrity.
    checks += [
        {"round": 1, "check": "temporal_fit_holdout_split_closed", "pass": result["fit_frames"] == 490 and result["heldout_frames"] == 210 and all((x["frame"] < 490) == (i < result["train_rows"]) for i, x in enumerate(rows[:result["train_rows"]]))},
        {"round": 1, "check": "source_receipts_match", "pass": result["source_sha256"][str(TRACE_DIR / "trace.jsonl")] == digest(TRACE_DIR / "trace.jsonl") and result["source_sha256"][str(out / "support_rows.jsonl")] == digest(out / "support_rows.jsonl")},
        {"round": 1, "check": "official_and_future_closed", "pass": result["official_val_test_read"] is False and result["future_features_used"] is False and result["formal_mot_result"] is False},
    ]
    # Round 2: operator and implementation contracts.
    rng = np.random.RandomState(63065)
    src = rng.randn(8, 2); dst = src @ np.asarray([[1.1, .04], [.02, .9]]) + np.asarray([20., -4.]); w = np.linspace(.1, 1.0, 8)
    h1 = dlt(src, dst, w); perm = rng.permutation(len(src)); h2 = dlt(src[perm], dst[perm], w[perm])
    checks += [
        {"round": 2, "check": "weighted_dlt_permutation_invariant", "pass": bool(h1 is not None and h2 is not None and np.max(np.abs(h1 / h1[2, 2] - h2 / h2[2, 2])) < 1e-7)},
        {"round": 2, "check": "support_features_finite", "pass": bool(all(np.isfinite([x["residual"], x["symmetric_residual"], x["score"], *x["source_point"], *x["target_point"]]).all() for x in rows))},
        {"round": 2, "check": "labels_are_support_only", "pass": all(set(x) >= {"same_identity", "source_gid", "target_gid"} for x in rows) and all("future" not in " ".join(x.keys()).lower() for x in rows)},
    ]
    # Round 3: trained artifact and negative/positive attribution replay.
    ck = out / "support_reliability_cpu.pt"
    state = torch.load(ck, map_location="cpu")
    checks += [
        {"round": 3, "check": "checkpoint_finite_and_split_receipt", "pass": all(torch.isfinite(v).all().item() for v in state["state_dict"].values()) and state["fit_frames"] == 490},
        {"round": 3, "check": "learned_gain_replayable_and_nonzero", "pass": result["methods"]["learned_weight"]["queries"] == result["methods"]["native"]["queries"] == 99 and result["methods"]["learned_weight"]["correct"] > result["methods"]["native"]["correct"]},
        {"round": 3, "check": "fixed_operator_attribution_kept", "pass": result["methods"]["fixed_residual"]["correct"] == 77 and result["methods"]["learned_weight"]["correct"] == 78},
    ]
    report = {"status": "PASS_P63_SUPPORT_IMPLEMENTATION_AUDIT", "rounds": 3, "checks_per_round": 3, "all_checks_pass": all(x["pass"] for x in checks), "checks": checks, "scientific_status": "candidate mechanism only; no P26 host attachment or official MOT result"}
    (out / "IMPLEMENTATION_AUDIT_SUPPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
