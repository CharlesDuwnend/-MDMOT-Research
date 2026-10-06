#!/usr/bin/env python3
"""Fit23-only audit: localize a failure in protected P26 before selecting a model.

The P48 trace is used as evidence only after its final predictions are shown to
equal P26's. GT labels annotate diagnostics, never enter a proposed policy. The
oracle clean-support homography is explicitly an offline input intervention,
not a learned method or a full tracker replay.
"""
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "continuation_20261006/p63_causal_set_owner_transition"
P26 = ROOT / "continuation_20261005/p26_mia_baseline"
TRACE_DIR = ROOT / "continuation_20261005/p48_cross_view_kinematic_transport/runs/trace_pair23_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def truth(path):
    frames = defaultdict(list)
    for track in ET.parse(path).getroot().findall("track"):
        identity = (int(track.attrib["id"]), track.attrib["label"])
        for box in track.findall("box"):
            if box.attrib.get("outside", "0") == "1":
                continue
            frames[int(box.attrib["frame"])].append((identity, np.asarray([
                float(box.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr")
            ])))
    return frames


def ious(boxes, gt_boxes):
    if not len(boxes) or not len(gt_boxes):
        return np.zeros((len(boxes), len(gt_boxes)))
    a, b = np.asarray(boxes)[:, None], np.asarray(gt_boxes)[None]
    intersection = np.maximum(0, np.minimum(a[..., 2:], b[..., 2:]) - np.maximum(a[..., :2], b[..., :2])).prod(-1)
    area_a = np.maximum(0, a[..., 2:] - a[..., :2]).prod(-1)
    area_b = np.maximum(0, b[..., 2:] - b[..., :2]).prod(-1)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def labels(rows, gt, conflicting_ids):
    """One-to-one IoU match; ambiguous or conflicting identities remain unknown."""
    boxes = [r[1:5] for r in rows]
    matrix = ious(boxes, [b for _, b in gt])
    out = [None] * len(rows)
    if matrix.size:
        rr, cc = linear_sum_assignment(np.where(matrix >= 0.5, -matrix, 1e6))
        for r, c in zip(rr, cc):
            identity = gt[c][0]
            competitors = {gt[j][0] for j in np.where(matrix[r] >= 0.5)[0]}
            if matrix[r, c] >= 0.5 and identity[0] not in conflicting_ids and len(competitors) == 1:
                out[r] = identity
    return out


def centers(rows):
    x = np.asarray(rows, dtype=np.float64)
    return (x[:, 1:3] + x[:, 3:5]) * 0.5 if len(x) else np.empty((0, 2))


def project(H, points):
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), np.asarray(H, np.float64)).reshape(-1, 2)


def nearest_quality(H, row, dst_rows, dst_labels, identity):
    """Static native-geometry diagnostic, not a full owner decision replay."""
    points = centers([row])
    q = project(H, points)[0]
    # P26 casts target centers to integer pixels before computing distance.
    distance = np.linalg.norm(centers(dst_rows).astype(int) - q, axis=1)
    best = int(np.argmin(distance))
    true = [i for i, label in enumerate(dst_labels) if label == identity]
    corners = project(H, [[row[1], row[2]], [row[3], row[4]]])
    inside = bool(np.isfinite(corners).all() and corners[:, 0].min() >= 0 and corners[:, 1].min() >= 0
                  and corners[:, 0].max() <= 1920 and corners[:, 1].max() <= 1080)
    return {
        "correct_top1": dst_labels[best] == identity,
        "chosen_unknown": dst_labels[best] is None,
        "true_in_gate": bool(any(distance[j] < 80 for j in true)),
        "min_true_distance": float(min(distance[j] for j in true)),
        "best_distance": float(distance[best]),
        "projected_box_inside": inside,
        "projection_error_to_true_center": float(min(np.linalg.norm(centers(dst_rows)[j] - q) for j in true)),
    }


def main():
    trace_path = TRACE_DIR / "trace.jsonl"
    rows = [json.loads(line) for line in trace_path.open()]
    xml = {v: truth(P26 / f"xml/23-{v}.xml") for v in (1, 2)}
    equality = {}
    sources = {str(trace_path): sha(trace_path)}
    for v in (1, 2):
        a = P26 / f"runs/fit23/json/mia_baseline/23-{v}.json"
        b = TRACE_DIR / f"json/traced_baseline/23-{v}.json"
        equality[str(v)] = json.loads(a.read_text()) == json.loads(b.read_text())
        assert equality[str(v)]
        sources.update({str(a): sha(a), str(b): sha(b), str(P26 / f"xml/23-{v}.xml"): sha(P26 / f"xml/23-{v}.xml")})
    # Prefix conflict set: no annotation later than the event enters label eligibility.
    conflicts = {}
    seen_labels = defaultdict(set)
    for f in range(700):
        for v in (1, 2):
            for identity, _ in xml[v].get(f, []):
                seen_labels[identity[0]].add(identity[1])
        conflicts[f] = {gid for gid, values in seen_labels.items() if len(values) > 1}
    stats = defaultdict(Counter)
    supports = Counter()
    interventions = Counter()
    residuals = {k: [] for k in ("baseline", "oracle_clean_support")}
    local_rows = []
    for row in rows:
        if row["stage"] not in ("before_A_new", "before_B_new"):
            continue
        f = row["frame"]
        direction = row["stage"]
        src_v, dst_v = (1, 2) if direction == "before_A_new" else (2, 1)
        src, dst = row[f"view{src_v}"], row[f"view{dst_v}"]
        sl = labels(src, xml[src_v].get(f, []), conflicts[f])
        dl = labels(dst, xml[dst_v].get(f, []), conflicts[f])
        if row["H"] is None or not np.isfinite(np.asarray(row["H"])).all():
            stats[direction]["invalid_H_frames"] += 1
            continue
        new_ids = (row.get("extra") or {}).get("source_ids", [])
        clean_H = None
        # A trace records the native supports used for the actual current H.
        if direction == "before_A_new":
            a = np.asarray(row["extra"]["support_src"]).reshape(-1, 2)
            b = np.asarray(row["extra"]["support_dst"]).reshape(-1, 2)
            keep = []
            for k, (pa, pb) in enumerate(zip(a, b)):
                if not len(src) or not len(dst):
                    supports["unknown"] += 1
                    continue
                ai = int(np.argmin(np.linalg.norm(centers(src) - pa, axis=1)))
                bi = int(np.argmin(np.linalg.norm(centers(dst) - pb, axis=1)))
                if np.linalg.norm(centers(src)[ai] - pa) > 1e-3 or np.linalg.norm(centers(dst)[bi] - pb) > 1e-3 or sl[ai] is None or dl[bi] is None:
                    supports["unknown"] += 1
                elif sl[ai] == dl[bi]:
                    supports["known_correct"] += 1
                    keep.append(k)
                else:
                    supports["known_false"] += 1
            if len(keep) >= 5:
                cv2.setRNGSeed(0)
                clean_H, _ = cv2.findHomography(a[keep], b[keep], cv2.RANSAC, 5.0)
                interventions["frames_eligible_clean_support"] += 1
        for source_id in new_ids:
            stats[direction]["new_source_events"] += 1
            indexes = [i for i, r in enumerate(src) if int(r[0]) == int(source_id)]
            if len(indexes) != 1:
                stats[direction]["source_id_not_unique"] += 1
                continue
            index = indexes[0]
            identity = sl[index]
            if identity is None:
                stats[direction]["source_label_unknown"] += 1
                continue
            if identity not in [x[0] for x in xml[dst_v].get(f, [])]:
                stats[direction]["counterpart_not_visible"] += 1
                continue
            if identity not in dl:
                stats[direction]["counterpart_not_matched_in_tracks"] += 1
                continue
            q = nearest_quality(row["H"], src[index], dst, dl, identity)
            stats[direction]["evaluable_new_events"] += 1
            for name in ("correct_top1", "chosen_unknown", "true_in_gate", "projected_box_inside"):
                stats[direction][name] += int(q[name])
            residuals["baseline"].append(q["projection_error_to_true_center"])
            record = {"frame": f, "direction": direction, "source_track_id": int(source_id), "baseline": q}
            if clean_H is not None and np.isfinite(clean_H).all():
                alt = nearest_quality(clean_H, src[index], dst, dl, identity)
                interventions["events_evaluable_both_H"] += 1
                interventions["baseline_correct_top1"] += int(q["correct_top1"])
                interventions["oracle_correct_top1"] += int(alt["correct_top1"])
                interventions["fixed_wrong_top1"] += int(not q["correct_top1"] and alt["correct_top1"])
                interventions["lost_correct_top1"] += int(q["correct_top1"] and not alt["correct_top1"])
                residuals["oracle_clean_support"].append(alt["projection_error_to_true_center"])
                record["oracle_clean_support"] = alt
            local_rows.append(record)
    for d, s in stats.items():
        denom = s["evaluable_new_events"]
        s["top1_rate"] = s["correct_top1"] / max(1, denom)
        s["true_gate_recall"] = s["true_in_gate"] / max(1, denom)
    report = {
        "status": "COMPLETE_P26_FIT23_ALIGNMENT_AUDIT",
        "baseline_trace_output_equal": equality,
        "directions": {k: dict(v) for k, v in stats.items()},
        "native_H_supports": dict(supports),
        "oracle_clean_support_input_intervention": dict(interventions),
        "support_false_rate_among_known": supports["known_false"] / max(1, supports["known_false"] + supports["known_correct"]),
        "scope": "pair23 fit diagnostic, actual new-ID refresh events only; static geometry, not complete owner replay",
        "identity_protocol": "P26 paired XML protocol, one-to-one IoU >= 0.5; prefix class-conflict and ambiguous overlaps masked",
        "source_hashes": sources,
        "script_sha256": sha(Path(__file__)),
        "official_val_test_read": False,
        "training_run": False,
        "formal_MOT_result": False,
        "oracle_supervision_is_policy_input": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "P26_ALIGNMENT_AUDIT.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    # Local detailed event stream is ignored by Git; only aggregate evidence is archived.
    with (OUT / "p26_alignment_events.jsonl").open("w") as f:
        for item in local_rows:
            f.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_hashes"}, indent=2))


if __name__ == "__main__":
    main()
