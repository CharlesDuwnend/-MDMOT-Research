#!/usr/bin/env python3
"""Measure whether native support residuals predict false support pairs."""
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "continuation_20261006/p63_causal_set_owner_transition"
TRACE = ROOT / "continuation_20261005/p48_cross_view_kinematic_transport/runs/trace_pair23_v1/trace.jsonl"
P26 = ROOT / "continuation_20261005/p26_mia_baseline"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_gt(path):
    out = defaultdict(list)
    for track in ET.parse(path).getroot().findall("track"):
        gid, label = int(track.attrib["id"]), track.attrib["label"]
        for box in track.findall("box"):
            if box.attrib.get("outside", "0") == "1":
                continue
            out[int(box.attrib["frame"])].append((gid, label, np.asarray([float(box.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr")], dtype=np.float64)))
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-12)


def row_labels(rows, gt):
    labels = []
    for row in rows:
        best = max(((iou(row[1:5], box), gid, label) for gid, label, box in gt), default=(0.0, None, None))
        labels.append((best[1], best[2]) if best[0] >= 0.5 else (None, None))
    return labels


def center(row):
    return (np.asarray(row[1:3], dtype=np.float64) + np.asarray(row[3:5], dtype=np.float64)) * 0.5


def frame_pair(frame, stage, view_a, view_b, H, labels_a, labels_b):
    src = view_a if stage == "before_A_new" else view_b
    dst = view_b if stage == "before_A_new" else view_a
    src_labels = labels_a if stage == "before_A_new" else labels_b
    dst_labels = labels_b if stage == "before_A_new" else labels_a
    try:
        inverse = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return []
    supports = frame["extra"].get("support_src", [])
    targets = frame["extra"].get("support_dst", [])
    result = []
    for point_a, point_b in zip(np.asarray(supports, dtype=np.float64).reshape(-1, 2), np.asarray(targets, dtype=np.float64).reshape(-1, 2)):
        if not len(src) or not len(dst):
            continue
        src_i = int(np.argmin([np.linalg.norm(center(r) - point_a) for r in src]))
        dst_i = int(np.argmin([np.linalg.norm(center(r) - point_b) for r in dst]))
        source_gid, source_class = src_labels[src_i]
        target_gid, target_class = dst_labels[dst_i]
        if source_gid is None or target_gid is None:
            continue
        projected = cv2.perspectiveTransform(point_a.reshape(1, 1, 2).astype(np.float32), H).reshape(2)
        back = cv2.perspectiveTransform(point_b.reshape(1, 1, 2).astype(np.float32), inverse).reshape(2)
        src_scale = max(float(np.linalg.norm(np.asarray(src[src_i][3:5], dtype=np.float64) - np.asarray(src[src_i][1:3], dtype=np.float64))), 1.0)
        dst_scale = max(float(np.linalg.norm(np.asarray(dst[dst_i][3:5], dtype=np.float64) - np.asarray(dst[dst_i][1:3], dtype=np.float64))), 1.0)
        residual = float(np.linalg.norm(projected - point_b) / dst_scale)
        symmetric = float(np.linalg.norm(back - point_a) / src_scale)
        result.append({"frame": int(frame["frame"]), "stage": stage, "source_gid": int(source_gid), "target_gid": int(target_gid), "same_identity": bool(source_gid == target_gid and source_class == target_class), "residual": residual, "symmetric_residual": symmetric, "score": residual + symmetric, "source_point": point_a.tolist(), "target_point": point_b.tolist(), "source_center": center(src[src_i]).tolist(), "target_center": center(dst[dst_i]).tolist()})
    return result


def main():
    gt = {1: read_gt(P26 / "xml/23-1.xml"), 2: read_gt(P26 / "xml/23-2.xml")}
    rows = [json.loads(line) for line in TRACE.open()]
    by = {(int(x["frame"]), x["stage"]): x for x in rows}
    all_supports = []
    for key, frame in by.items():
        if frame["stage"] not in ("before_A_new", "before_B_new") or frame["H"] is None:
            continue
        if not np.isfinite(np.asarray(frame["H"], dtype=np.float64)).all():
            continue
        f = int(frame["frame"]); a, b = by[(f, "pre_mia")]["view1"], by[(f, "pre_mia")]["view2"]
        all_supports.extend(frame_pair(frame, frame["stage"], a, b, np.asarray(frame["H"], dtype=np.float64), row_labels(a, gt[1][f]), row_labels(b, gt[2][f])))
    y = np.asarray([x["same_identity"] for x in all_supports], dtype=np.int64)
    score = np.asarray([x["score"] for x in all_supports], dtype=np.float64)
    assert y.any() and (~y.astype(bool)).any()
    # Lower residual should imply correctness; report AUC of the inverted score.
    auc = float(roc_auc_score(y, -score))
    split = np.asarray([x["frame"] < 490 for x in all_supports])
    report = {"status": "PASS_P63_SUPPORT_SIGNAL_DIAGNOSTIC", "rows": len(all_supports), "positive_rows": int(y.sum()), "negative_rows": int((1 - y).sum()), "positive_rate": float(y.mean()), "residual_auc": auc, "train_rows": int(split.sum()), "heldout_rows": int((~split).sum()), "train_positive": int(y[split].sum()), "heldout_positive": int(y[~split].sum()), "positive_median_score": float(np.median(score[y == 1])), "negative_median_score": float(np.median(score[y == 0])), "future_features_used": False, "official_val_test_read": False, "formal_mot_result": False, "source_sha256": {str(TRACE): sha(TRACE), str(P26 / "xml/23-1.xml"): sha(P26 / "xml/23-1.xml"), str(P26 / "xml/23-2.xml"): sha(P26 / "xml/23-2.xml")}, "script_sha256": sha(Path(__file__))}
    (OUT / "SUPPORT_SIGNAL.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (OUT / "support_rows.jsonl").open("w") as f:
        for row in all_supports:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
