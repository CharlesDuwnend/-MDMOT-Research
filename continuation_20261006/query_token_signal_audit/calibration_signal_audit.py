#!/usr/bin/env python3
"""Unsupervised cross-view feature-calibration signal audit.

The candidate is deliberately kept analytic and frozen: camera-role moments
are estimated from the fit feature cache without labels, then applied to the
same train-only episode ranking contract.  This is a diagnostic, not a
learned ranker or an official metric.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
import query_signal_audit as q


def moments(pairs):
    vals = {"1": [], "2": []}
    for pair in pairs:
        for uav in ("1", "2"):
            with np.load(q.BASE / f"{pair}-{uav}.npz", allow_pickle=False) as z:
                x = z["visual_feature"].astype(np.float64)
            # Bounded sampling keeps this audit memory-safe while preserving
            # all sequence roles and does not use identity labels.
            stride = max(1, len(x) // 20000)
            vals[uav].append(x[::stride])
    out = {}
    for uav in vals:
        x = np.concatenate(vals[uav], axis=0)
        out[uav] = {"mean": x.mean(0).astype(np.float32), "std": np.maximum(x.std(0), 1e-4).astype(np.float32), "rows": int(len(x))}
    return out


def cosine(a, c):
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    return c @ a


def rank(scores, positives):
    order = np.argsort(-scores, kind="stable")
    for k, j in enumerate(order, 1):
        if int(j) in set(positives):
            return k
    return None


def score(events, stats, mode):
    ranks = []
    for e in events:
        a, c = e["a_visual"].copy(), e["c_visual"].copy()
        role_a, role_c = e["target_uav"], e["source_uav"]
        if mode == "role_center":
            a = a - stats[role_a]["mean"]
            c = c - stats[role_c]["mean"]
        elif mode == "role_zscore":
            a = (a - stats[role_a]["mean"]) / stats[role_a]["std"]
            c = (c - stats[role_c]["mean"]) / stats[role_c]["std"]
        elif mode == "event_set_center":
            if len(c):
                center = c.mean(0)
                a, c = a - center, c - center
        elif mode == "event_set_zscore":
            if len(c):
                center = c.mean(0)
                scale = np.maximum(c.std(0), 1e-4)
                a, c = (a - center) / scale, (c - center) / scale
        elif mode != "visual":
            raise ValueError(mode)
        r = rank(cosine(a, c), e["positive"])
        if r is not None:
            ranks.append(r)
    return {"events": len(ranks), "recall1": float(np.mean(np.asarray(ranks) == 1)), "mrr": float(np.mean(1.0 / np.asarray(ranks))), "median_rank": float(np.median(ranks))}


def main():
    pairs = tuple(dict.fromkeys(q.FIT + q.EVAL))
    events = {}
    audits = {}
    for pair in pairs:
        events[pair], audits[pair] = q.build_pair(pair)
    stats = moments(q.FIT)
    modes = ("visual", "role_center", "role_zscore", "event_set_center", "event_set_zscore")
    per_pair = {m: {p: score(events[p], stats, m) for p in q.EVAL} for m in modes}
    macro = {m: {k: float(np.mean([per_pair[m][p][k] for p in q.EVAL])) for k in ("recall1", "mrr", "median_rank")} for m in modes}
    wins = {m: sum(per_pair[m][p]["recall1"] > per_pair["visual"][p]["recall1"] for p in q.EVAL) for m in modes if m != "visual"}
    result = {"status": "COMPLETE_UNSUPERVISED_CALIBRATION_SIGNAL_AUDIT", "fit_pairs": list(q.FIT), "eval_pairs": list(q.EVAL), "modes": modes, "per_pair": per_pair, "macro": macro, "wins_vs_visual": wins, "stats": {u: {"rows": v["rows"], "mean_norm": float(np.linalg.norm(v["mean"])), "std_min": float(v["std"].min()), "std_max": float(v["std"].max())} for u, v in stats.items()}, "controls": {"labels_used_only_for_rank": True, "parameter_update": False, "official_val_test_access": False, "camera_moments_fit_only": True}, "audits": {p: {"events": a["events"], "dropped": len(a["dropped"])} for p, a in audits.items()}}
    (HERE / "CALIBRATION_SIGNAL_AUDIT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "macro": macro, "wins_vs_visual": wins}, ensure_ascii=False))


if __name__ == "__main__":
    main()
