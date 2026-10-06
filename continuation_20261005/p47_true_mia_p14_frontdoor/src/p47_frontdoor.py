"""Descriptor-aware replacement for the three MIA refresh calls.

The function keeps the P26 input/output contract and only changes candidate
selection. It is intentionally small so that the wrapper can log every edge
and audit the host integration separately from the P14 representation.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def _center(row):
    return np.array([(row[1] + row[3]) * 0.5, (row[2] + row[4]) * 0.5], dtype=np.float64)


def _diag(row):
    return max(float(np.hypot(row[3] - row[1], row[4] - row[2])), 1.0)


def _project(H, point):
    if H is None or not np.isfinite(H).all():
        return np.array([np.nan, np.nan], dtype=np.float64)
    out = cv2.perspectiveTransform(np.asarray(point, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)[0]
    return out.astype(np.float64)


def _build_cost(source_rows, target_rows, source_desc, source_valid, target_desc, target_valid,
                source_indices, H, use_appearance):
    C = np.full((len(source_indices), len(target_rows)), 1e6, dtype=np.float64)
    G = np.full_like(C, np.nan)
    A = np.full_like(C, np.nan)
    for ii, si in enumerate(source_indices):
        projected = _project(H, _center(source_rows[si]))
        if not np.isfinite(projected).all():
            continue
        for tj, row in enumerate(target_rows):
            geom = float(np.linalg.norm(projected - _center(row)) / _diag(row))
            G[ii, tj] = geom
            if geom > 2.0:
                continue
            cost = geom / 2.0
            if use_appearance and source_valid[si] and target_valid[tj]:
                cos = float(np.dot(source_desc[si], target_desc[tj]))
                A[ii, tj] = cos
                cost = 0.7 * geom / 2.0 + 0.3 * (1.0 - cos) / 2.0
            C[ii, tj] = cost
    return C, G, A


def refresh(source_ids, source_rows, target_rows, source_desc, source_valid, target_desc, target_valid,
            H, matched_ids, coID, use_appearance=True):
    """Apply one-to-one proposals in the same place as the original refresh.

    `source_ids` are the source-side new/old-unmatched IDs. The lower ID is
    retained exactly as in P26. No output after this call is relabelled.
    """
    if len(source_ids) == 0 or len(target_rows) == 0:
        return matched_ids, coID, {"proposals": [], "accepted": [], "skipped": 0}
    source_indices = []
    for sid in source_ids:
        ix = np.where(np.asarray(source_rows)[:, 0] == sid)[0]
        if len(ix):
            source_indices.append(int(ix[0]))
    if not source_indices:
        return matched_ids, coID, {"proposals": [], "accepted": [], "skipped": 0}
    C, G, A = _build_cost(source_rows, target_rows, source_desc, source_valid, target_desc, target_valid,
                          source_indices, H, use_appearance)
    rr, cc = linear_sum_assignment(C)
    proposals, accepted = [], []
    used_target = {int(round(float(x[0]))) for x in target_rows}
    confirmed = set(int(x) for x in matched_ids) | set(int(x) for x in coID)
    skipped = 0
    for r, j in zip(rr, cc):
        if C[r, j] >= 1e5:
            continue
        si = source_indices[r]
        sid = int(round(float(source_rows[si, 0])))
        tid = int(round(float(target_rows[j, 0])))
        proposals.append({"source_index": si, "target_index": int(j), "source_id": sid,
                          "target_id": tid, "cost": float(C[r, j]),
                          "geometry": float(G[r, j]),
                          "cosine": None if not np.isfinite(A[r, j]) else float(A[r, j])})
        new_id = int(min(sid, tid))
        if tid in confirmed or new_id in confirmed:
            skipped += 1
            continue
        # Within-frame injectivity: the target side cannot receive two owners.
        if any(int(x["target_index"]) == int(j) for x in accepted):
            skipped += 1
            continue
        source_rows[si, 0] = float(new_id)
        target_rows[j, 0] = float(new_id)
        matched_ids.append(new_id)
        coID.append(new_id)
        confirmed.add(new_id)
        accepted_item = dict(proposals[-1])
        accepted_item["new_id"] = new_id
        accepted.append(accepted_item)
    return matched_ids, coID, {"proposals": proposals, "accepted": accepted, "skipped": skipped}
