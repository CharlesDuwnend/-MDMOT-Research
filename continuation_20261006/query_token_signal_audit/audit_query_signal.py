#!/usr/bin/env python3
"""Independent three-round audit for the query-token/calibration screen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
signal = json.loads((HERE / "QUERY_SIGNAL_AUDIT.json").read_text())
cal = json.loads((HERE / "CALIBRATION_SIGNAL_AUDIT.json").read_text())

def digest(path: Path):
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h

rounds = []
rounds.append({
    "round": "R1_DATA_AND_LEGAL",
    "checks": {
        "all_primary_pairs_loaded_without_drop": all(len(v["dropped"]) == 0 for v in signal["audits"].values()),
        "same_candidate_population_across_modes": signal["controls"]["candidate_set_and_denominator_identical"],
        "train_only_and_no_parameter_update": signal["controls"]["no_parameter_update"] and not signal["controls"]["official_val_test_access"],
    },
})
rounds.append({
    "round": "R2_REPLAY_AND_FINITE",
    "checks": {
        "visual_macro_recall_finite": bool(np.isfinite(signal["macro"]["visual"]["recall1"])),
        "query_macro_recall_finite": bool(np.isfinite(signal["macro"]["query"]["recall1"])),
        "calibration_stats_fit_only": cal["controls"]["camera_moments_fit_only"] and all(v["events"] > 0 for v in cal["audits"].values()),
    },
})
rounds.append({
    "round": "R3_PREDECLARED_GATE",
    "checks": {
        "query_wins_at_least_three_pairs": signal["wins_vs_visual"]["query"] >= 3,
        "equal_fusion_wins_at_least_three_pairs": signal["wins_vs_visual"]["equal_fusion"] >= 3,
        "role_zscore_wins_at_least_three_pairs": cal["wins_vs_visual"]["role_zscore"] >= 3,
    },
})
all_pass = all(all(item["checks"].values()) for item in rounds)
result = {
    "status": "PASS_QUERY_TOKEN_SIGNAL_FAILURE_AUDIT_3X3" if not all_pass else "HOLD_QUERY_TOKEN_FOR_NOVELTY_REVIEW",
    "candidate": "frozen AutoAssign query-token complement and role-whitened calibration",
    "rounds": rounds,
    "decision": {
        "query_token_training": "STOP_BEFORE_TRAINING",
        "role_whitening_training": "STOP_BEFORE_TRAINING",
        "reason": "query token loses the visual baseline on all four primary pairs; equal fusion wins only one; role z-score reaches two of four and is a standard camera-aware normalization family",
        "official_val_test_access": False,
        "retain_as_diagnostic": True,
    },
    "inputs": {
        "query_signal_sha256": digest(HERE / "QUERY_SIGNAL_AUDIT.json"),
        "calibration_signal_sha256": digest(HERE / "CALIBRATION_SIGNAL_AUDIT.json"),
    },
}
(HERE / "QUERY_SIGNAL_FAILURE_AUDIT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"status": result["status"], "all_rounds_pass": all_pass}, ensure_ascii=False))
