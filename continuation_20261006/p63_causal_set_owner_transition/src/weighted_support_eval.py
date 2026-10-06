#!/usr/bin/env python3
"""Temporal fit/held-out diagnostic for learned support-weighted homography."""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from baseline_alignment_audit import P26, TRACE_DIR, OUT, sha
from support_signal import read_gt, row_labels

TRACE = TRACE_DIR / "trace.jsonl"


class Reliability(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 24), nn.LayerNorm(24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def dlt(src, dst, weights):
    if len(src) < 4:
        return None
    rows = []
    for (x, y), (u, v), w in zip(src, dst, weights):
        rows.extend([[ -x, -y, -1, 0, 0, 0, u*x, u*y, u], [0, 0, 0, -x, -y, -1, v*x, v*y, v]])
    A = np.asarray(rows, dtype=np.float64)
    ww = np.repeat(np.sqrt(np.maximum(np.asarray(weights, dtype=np.float64), 1e-8)), 2)
    _, _, vh = np.linalg.svd(A * ww[:, None], full_matrices=False)
    h = vh[-1]
    if not np.isfinite(h).all() or abs(h[-1]) < 1e-10:
        return None
    H = (h / h[-1]).reshape(3, 3)
    return H if np.isfinite(H).all() else None


def center(row):
    r = np.asarray(row, dtype=np.float64)
    return (r[1:3] + r[3:5]) * 0.5


def eval_h(H, source, dst, source_label, dst_labels):
    if H is None or not len(dst):
        return None
    p = cv2.perspectiveTransform(center(source).reshape(1, 1, 2).astype(np.float32), H).reshape(2)
    values = [float(np.linalg.norm(p - np.rint(center(x)))) for x in dst]
    chosen = int(np.argmin(values))
    return bool(dst_labels[chosen] == source_label), bool(any(x == source_label for x in dst_labels)), values[chosen]


def feature(row):
    return np.asarray([row["score"], row["residual"], row["symmetric_residual"], row["source_point"][0] / 1920.0, row["source_point"][1] / 1080.0, row["target_point"][0] / 1920.0], dtype=np.float32)


def main():
    torch.manual_seed(63064); np.random.seed(63064)
    gt = {1: read_gt(P26 / "xml/23-1.xml"), 2: read_gt(P26 / "xml/23-2.xml")}
    trace = [json.loads(line) for line in TRACE.open()]
    by = {(int(x["frame"]), x["stage"]): x for x in trace}
    support = [json.loads(line) for line in (OUT / "support_rows.jsonl").open()]
    support_by = defaultdict(list)
    for x in support:
        support_by[(x["frame"], x["stage"])].append(x)
    # Strict temporal split: first 490 frames fit; held-out frames are untouched
    # until metric calculation.
    train = [x for x in support if x["frame"] < 490]
    X = torch.tensor(np.stack([feature(x) for x in train]), dtype=torch.float32)
    y = torch.tensor([int(x["same_identity"]) for x in train], dtype=torch.float32)
    mean, std = X.mean(0), X.std(0).clamp_min(1e-4)
    model = Reliability(); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_weight = torch.tensor(float((y == 0).sum() / max(1, (y == 1).sum())))
    loss_hist = []
    for step in range(500):
        logits = model((X - mean) / std)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        opt.zero_grad(); loss.backward(); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()); opt.step(); loss_hist.append(float(loss.detach()))
    assert loss_hist[-1] < loss_hist[0]
    model.eval()
    rows = []
    # Only actual new-ID refresh events are evaluated; this matches the P26
    # intervention point and leaves the protected wrapper untouched.
    for frame in trace:
        if frame["stage"] not in ("before_A_new", "before_B_new") or frame["frame"] < 490 or frame["H"] is None:
            continue
        f, stage = int(frame["frame"]), frame["stage"]
        pre = by[(f, "pre_mia")]
        a, b = pre["view1"], pre["view2"]
        src, dst = (a, b) if stage == "before_A_new" else (b, a)
        src_v, dst_v = (1, 2) if stage == "before_A_new" else (2, 1)
        src_labels, dst_labels = row_labels(src, gt[src_v][f]), row_labels(dst, gt[dst_v][f])
        native = np.asarray(frame["H"], dtype=np.float64)
        supp = support_by[(f, stage)]
        if len(supp) < 4: continue
        sx = torch.tensor(np.stack([feature(x) for x in supp]), dtype=torch.float32)
        with torch.no_grad(): weights = torch.sigmoid(model((sx - mean) / std)).numpy()
        learned = dlt(np.asarray([x["source_point"] for x in supp]), np.asarray([x["target_point"] for x in supp]), weights)
        fixed = dlt(np.asarray([x["source_point"] for x in supp]), np.asarray([x["target_point"] for x in supp]), np.exp(-np.asarray([x["score"] for x in supp])))
        for source_id in frame["extra"].get("source_ids", []):
            candidates = [i for i, row in enumerate(src) if int(row[0]) == int(source_id)]
            if len(candidates) != 1 or src_labels[candidates[0]] is None: continue
            source_label = src_labels[candidates[0]]
            for name, H in (("native", native), ("fixed_residual", fixed), ("learned_weight", learned)):
                q = eval_h(H, src[candidates[0]], dst, source_label, dst_labels)
                if q is not None and q[1]: rows.append({"frame": f, "stage": stage, "method": name, "correct": q[0], "distance": q[2]})
    methods = {}
    for name in ("native", "fixed_residual", "learned_weight"):
        vals = [r for r in rows if r["method"] == name]
        methods[name] = {"queries": len(vals), "correct": int(sum(x["correct"] for x in vals)), "top1": float(np.mean([x["correct"] for x in vals])) if vals else 0.0, "median_distance": float(np.median([x["distance"] for x in vals])) if vals else None}
    report = {"status": "PASS_P63_WEIGHTED_SUPPORT_TEMPORAL_DIAGNOSTIC", "fit_frames": 490, "heldout_frames": 210, "train_rows": len(train), "train_positive": int(y.sum()), "model_parameters": sum(p.numel() for p in model.parameters()), "loss_first": loss_hist[0], "loss_last": loss_hist[-1], "methods": methods, "future_features_used": False, "official_val_test_read": False, "formal_mot_result": False, "source_sha256": {str(TRACE): sha(TRACE), str(OUT / "support_rows.jsonl"): sha(OUT / "support_rows.jsonl")}, "script_sha256": sha(Path(__file__))}
    (OUT / "WEIGHTED_SUPPORT_TEMPORAL_DIAGNOSTIC.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "fit_frames": 490}, OUT / "support_reliability_cpu.pt")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
