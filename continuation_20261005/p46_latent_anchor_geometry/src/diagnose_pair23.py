#!/usr/bin/env python3
"""Labelled fit diagnostic for the P46 geometry observable.

This reads only fit pair23 detector/track outputs and fit XML labels. It does
not train, alter P26 output, or report MOT metrics. The purpose is to decide
whether a support-line observable has signal before any learned head exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path("/home/chenhc/claude_try_MDMOT")
P26 = ROOT / "continuation_20261005/p26_mia_baseline/runs/fit23/json/mia_baseline"
XML = Path("/raid/datasets/chc_data/MDMT/new_xml")
OUT = ROOT / "continuation_20261005/p46_latent_anchor_geometry/runs/pair23"
IOU_GATE = 0.30
ALPHAS = np.linspace(0.0, 0.4, 9, dtype=np.float64)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-12)


def parse_gt(path: Path):
    root = ET.parse(path).getroot()
    by_frame = defaultdict(list)
    for tr in root.findall("track"):
        tid = int(tr.attrib["id"])
        for bx in tr.findall("box"):
            if int(bx.attrib["outside"]) == 1:
                continue
            frame = int(bx.attrib["frame"])
            b = np.array([float(bx.attrib["xtl"]), float(bx.attrib["ytl"]),
                          float(bx.attrib["xbr"]), float(bx.attrib["ybr"])], dtype=np.float64)
            by_frame[frame].append((tid, b))
    return by_frame


def pred_box(row):
    return np.asarray(row[1:5], dtype=np.float64)


def match_predictions(rows, gt_rows):
    if not rows or not gt_rows:
        return {}, {}
    mat = np.asarray([[iou(pred_box(r), g[1]) for g in gt_rows] for r in rows], dtype=np.float64)
    rr, cc = linear_sum_assignment(-mat)
    row_to_gt, gt_to_row = {}, {}
    for r, c in zip(rr, cc):
        if mat[r, c] >= IOU_GATE:
            tid = int(gt_rows[c][0])
            row_to_gt[int(r)] = tid
            gt_to_row[tid] = int(r)
    return row_to_gt, gt_to_row


def center(b):
    return np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float64)


def anchor(b, alpha):
    # alpha=0 is bottom center; alpha=.5 is box center.
    return np.array([(b[0] + b[2]) * 0.5, b[3] - alpha * (b[3] - b[1])], dtype=np.float64)


def project(H, p):
    q = cv2.perspectiveTransform(np.asarray(p, dtype=np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)[0]
    if not np.isfinite(q).all():
        return np.array([np.nan, np.nan], dtype=np.float64)
    return q.astype(np.float64)


def diag(b):
    return max(float(np.hypot(b[2] - b[0], b[3] - b[1])), 1.0)


def costs(H, a, b):
    scale = diag(b)
    c = float(np.linalg.norm(project(H, center(a)) - center(b)) / scale)
    bot = float(np.linalg.norm(project(H, anchor(a, 0.0)) - anchor(b, 0.0)) / scale)
    vals = []
    for alpha in ALPHAS:
        vals.append(np.linalg.norm(project(H, anchor(a, float(alpha))) - anchor(b, float(alpha))) / scale)
    line = float(np.min(vals))
    return {"center": c, "bottom": bot, "line": line}


def initial_h(a, b):
    ma = {int(round(float(r[0]))): center(pred_box(r)) for r in a}
    mb = {int(round(float(r[0]))): center(pred_box(r)) for r in b}
    keys = sorted(set(ma) & set(mb))
    if len(keys) < 4:
        return np.eye(3, dtype=np.float64), len(keys)
    h, status = cv2.findHomography(np.asarray([ma[k] for k in keys], np.float32),
                                   np.asarray([mb[k] for k in keys], np.float32), cv2.RANSAC, 5.0)
    if h is None or not np.isfinite(h).all():
        return np.eye(3, dtype=np.float64), len(keys)
    return h.astype(np.float64), len(keys)


def update_h(h, a, b):
    # Reconstruct the frozen host's geometry-only transport update from its
    # current shared IDs. No XML labels are used here.
    ma = {int(round(float(r[0]))): center(pred_box(r)) for r in a}
    mb = {int(round(float(r[0]))): center(pred_box(r)) for r in b}
    keys = sorted(set(ma) & set(mb))
    if len(keys) < 4:
        return h
    nh, status = cv2.findHomography(np.asarray([ma[k] for k in keys], np.float32),
                                    np.asarray([mb[k] for k in keys], np.float32), cv2.RANSAC, 5.0)
    if nh is not None and np.isfinite(nh).all():
        return nh.astype(np.float64)
    return h


def summarize(rows):
    out = {}
    for kind in ("center", "bottom", "line"):
        vals = [r for r in rows if np.isfinite(r[kind]["true"])]
        ranks, margins, wins = [], [], 0
        for r in vals:
            true = r[kind]["true"]
            wrong = [x for x in r[kind]["all"] if x[0] != r["true_bj"] and np.isfinite(x[1])]
            rank = 1 + sum(float(c) < true - 1e-12 for _, c in r[kind]["all"])
            ranks.append(rank)
            if wrong:
                margins.append(float(min(c for _, c in wrong) - true))
            wins += int(rank == 1)
        out[kind] = {
            "n": len(vals),
            "rank1": float(wins / len(vals)) if vals else None,
            "median_rank": float(np.median(ranks)) if ranks else None,
            "median_margin_wrong_minus_true": float(np.median(margins)) if margins else None,
            "positive_margin_rate": float(np.mean(np.asarray(margins) > 0)) if margins else None,
        }
    return out


def main():
    pred = {v: json.loads((P26 / f"23-{v}.json").read_text()) for v in (1, 2)}
    gt = {v: parse_gt(XML / str(v) / f"23-{v}.xml") for v in (1, 2)}
    h, shared0 = initial_h(pred[1]["frame=0"], pred[2]["frame=0"])
    rows = []
    frames_with_pairs = 0
    for f in range(700):
        a, b = pred[1][f"frame={f}"], pred[2][f"frame={f}"]
        ra, _ = match_predictions(a, gt[1].get(f, []))
        rb, _ = match_predictions(b, gt[2].get(f, []))
        b_by_tid = {tid: j for j, tid in rb.items()}
        for ai, tid in ra.items():
            if tid not in b_by_tid:
                continue
            bj = b_by_tid[tid]
            all_costs = {kind: [] for kind in ("center", "bottom", "line")}
            for j, br in enumerate(b):
                cc = costs(h, pred_box(a[ai]), pred_box(br))
                for kind in all_costs:
                    all_costs[kind].append((j, cc[kind]))
            true_cost = {kind: next(c for j, c in all_costs[kind] if j == bj) for kind in all_costs}
            rows.append({"frame": f, "true_tid": int(tid), "true_bj": int(bj),
                         **{kind: {"true": true_cost[kind], "all": all_costs[kind]} for kind in all_costs}})
        frames_with_pairs += int(any(tid in b_by_tid for tid in ra.values()))
        if f > 0:
            h = update_h(h, a, b)
    result = {
        "status": "COMPLETE_P46_PAIR23_GEOMETRY_DIAGNOSTIC",
        "iou_gate": IOU_GATE,
        "alpha_grid": ALPHAS.tolist(),
        "initial_shared_ids": shared0,
        "rows_with_cross_view_gt_pair": len(rows),
        "frames_with_cross_view_gt_pair": frames_with_pairs,
        "read": {"p26_json": True, "fit_xml": True, "official_test": False},
        "summary": summarize(rows),
        "inputs_sha256": {f"p26_{v}": sha(P26 / f"23-{v}.json") for v in (1, 2)},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "diagnose.exit").write_text("0\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
