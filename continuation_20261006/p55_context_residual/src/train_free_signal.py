#!/usr/bin/env python3
"""Evaluate frozen target/context residual ranking on a disjoint fit holdout."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
P23 = ROOT / "continuation_20261004/p23_visual/data"
OUT = ROOT / "continuation_20261006/p55_context_residual"
HOLDOUT = {"51", "53", "78"}


def one_direction(q, qy, c, cy):
    valid_q = qy >= 0
    valid_c = cy >= 0
    if not valid_c.any():
        return []
    sims = q[valid_q] @ c[valid_c].T
    qlabels = qy[valid_q]
    clabels = cy[valid_c]
    ranks = []
    for row, label in zip(sims, qlabels):
        pos = np.flatnonzero(clabels == label)
        if len(pos) == 0:
            continue
        ranks.append(1 + int(np.sum(row > row[pos[0]])))
    return ranks


def main():
    x = np.load(P23 / "inputs.npz", allow_pickle=False)
    target = np.asarray(np.load(P23 / "frozen_dino.npy", mmap_mode="r"), dtype=np.float32)
    context = np.asarray(np.load(ROOT / "continuation_20261006/p55_context_residual/context_dino.npy", mmap_mode="r"), dtype=np.float32)
    labels = np.asarray(np.load(P23 / "offline_pid.npy", mmap_mode="r"), dtype=np.int64)
    groups = json.loads((P23 / "GROUPS.json").read_text())
    methods = {
        "target_only": target,
        "context_only": context,
        "context_residual": target - context,
        "equal_context_fusion": target + context,
    }
    for k, v in methods.items():
        methods[k] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    per_pair = {p: {m: [] for m in methods} for p in HOLDOUT}
    swap_pair = {p: [] for p in HOLDOUT}
    for g in groups:
        p = str(g["pair"])
        if p not in HOLDOUT:
            continue
        ia, ib = np.asarray(g["sides"][0]), np.asarray(g["sides"][1])
        for name, feat in methods.items():
            ranks = one_direction(feat[ia], labels[ia], feat[ib], labels[ib])
            ranks += one_direction(feat[ib], labels[ib], feat[ia], labels[ia])
            per_pair[p][name].extend(ranks)
        # Candidate-context counterfactual: preserve target features and labels,
        # cyclically exchange only the candidate context within the opposite
        # view. This detects a score that ignores its declared context input.
        if len(ib) > 1:
            q = target[ia] - context[np.roll(ia, 1)]
            c = target[ib] - context[np.roll(ib, 1)]
            q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
            c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
            swap_pair[p].extend(one_direction(q, labels[ia], c, labels[ib]))
    summary = {}
    for p, values in per_pair.items():
        summary[p] = {m: {"queries": len(r), "r1": float(np.mean(np.asarray(r) == 1)) if r else None,
                          "mrr": float(np.mean(1/np.asarray(r))) if r else None}
                      for m, r in values.items()}
    macro = {m: float(np.mean([summary[p][m]["r1"] for p in HOLDOUT])) for m in methods}
    wins = {m: int(sum(summary[p][m]["r1"] > summary[p]["target_only"]["r1"] for p in HOLDOUT)) for m in methods}
    swap = {p: (float(np.mean(np.asarray(v) == 1)) if v else None) for p, v in swap_pair.items()}
    out = {"status": "P55_TRAIN_FREE_INTERNAL_HOLDOUT", "holdout_pairs": sorted(HOLDOUT),
           "readout": "positive-present conditional R@1/MRR; fit-only diagnostic",
           "per_pair": summary, "macro_r1": macro, "wins_vs_target_only": wins,
           "candidate_context_swap_r1": swap,
           "interpretation": "context-only is a shortcut diagnostic; residual must be evaluated with context-swap controls before training",
           "official_val_test_read": False, "calibration_pairs_read": False}
    (OUT / "TRAIN_FREE_SIGNAL.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": out["status"], "macro_r1": macro, "wins": wins}))


if __name__ == "__main__": main()
