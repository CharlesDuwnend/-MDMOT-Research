"""Frame/row alignment between fresh P26 track arrays and sealed P14 rows."""

from __future__ import annotations

import numpy as np


def iou(a, b):
    x1, y1 = max(float(a[1]), float(b[1])), max(float(a[2]), float(b[2]))
    x2, y2 = min(float(a[3]), float(b[3])), min(float(a[4]), float(b[4]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[3]) - float(a[1])) * max(0.0, float(a[4]) - float(a[2]))
    bb = max(0.0, float(b[3]) - float(b[1])) * max(0.0, float(b[4]) - float(b[2]))
    return inter / max(aa + bb - inter, 1e-12)


def align(current_rows, reference_rows, ref_desc, ref_valid):
    """Map descriptors by unique same-frame IoU; never silently use row order."""
    pairs = []
    for i, a in enumerate(current_rows):
        for j, b in enumerate(reference_rows):
            pairs.append((iou(a, b), i, j))
    used_i, used_j, mapping = set(), set(), {}
    for score, i, j in sorted(pairs, key=lambda x: (-x[0], x[1], x[2])):
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j); mapping[i] = j
    out = np.zeros((len(current_rows), ref_desc.shape[1]), dtype=np.float32)
    valid = np.zeros(len(current_rows), dtype=bool)
    scores = []
    for i, j in mapping.items():
        out[i] = ref_desc[j]
        valid[i] = bool(ref_valid[j])
        scores.append(float(max(0.0, min(1.0, iou(current_rows[i], reference_rows[j])))))
    return out, valid, {"current_rows": len(current_rows), "reference_rows": len(reference_rows),
                        "mapped_rows": len(mapping), "coverage": len(mapping) / max(len(current_rows), 1),
                        "mean_iou": float(np.mean(scores)) if scores else None,
                        "min_iou": float(np.min(scores)) if scores else None}
