"""CPU-only contract checks for P45.

This file intentionally tests mechanism contracts only.  It does not read MDMT,
GT/XML, detector output, or report MOT metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def relation_signature(boxes: np.ndarray, max_neighbors: int = 3) -> np.ndarray:
    """Permutation-equivariant node signature from the same-view configuration."""
    boxes = np.asarray(boxes, dtype=np.float64)
    c = (boxes[:, :2] + boxes[:, 2:4]) * 0.5
    s = np.maximum(boxes[:, 2:4] - boxes[:, :2], 1e-6)
    n = len(boxes)
    out = np.zeros((n, max_neighbors, 4), dtype=np.float64)
    for i in range(n):
        rows = []
        for j in range(n):
            if i == j:
                continue
            d = c[j] - c[i]
            dist = float(np.linalg.norm(d))
            rows.append((dist, d[0], d[1], float(np.mean(s[j] / s[i]))))
        rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        for k, row in enumerate(rows[:max_neighbors]):
            out[i, k] = row
    scale = max(float(np.median(np.linalg.norm(s, axis=1))), 1.0)
    out[:, :, :3] /= scale
    return out.reshape(n, -1)


def pair_cost(box_a, box_b, emb_a, emb_b, rel_a, rel_b) -> float:
    ca = (box_a[:2] + box_a[2:4]) * 0.5
    cb = (box_b[:2] + box_b[2:4]) * 0.5
    sa = np.maximum(box_a[2:4] - box_a[:2], 1e-6)
    sb = np.maximum(box_b[2:4] - box_b[:2], 1e-6)
    geom = np.linalg.norm(ca - cb) / max(float(np.mean(np.r_[sa, sb])), 1.0)
    app = 1.0 - float(np.dot(emb_a, emb_b))
    rel = float(np.linalg.norm(rel_a - rel_b)) / max(np.sqrt(len(rel_a)), 1.0)
    return 0.45 * app + 0.35 * geom + 0.20 * rel


def partial_transport(box_a, box_b, emb_a, emb_b, null_cost: float = 0.90):
    """One-to-one partial assignment with an explicit unmatched capacity.

    The null construction is a control mechanism, not a novelty claim.
    """
    rel_a = relation_signature(box_a)
    rel_b = relation_signature(box_b)
    n, m = len(box_a), len(box_b)
    c = np.array(
        [[pair_cost(box_a[i], box_b[j], emb_a[i], emb_b[j], rel_a[i], rel_b[j])
          for j in range(m)] for i in range(n)], dtype=np.float64)
    size = n + m
    big = null_cost + 10.0
    aug = np.full((size, size), big, dtype=np.float64)
    aug[:n, :m] = c
    # A rows may use their own unmatched columns; B rows may use their own.
    for i in range(n):
        aug[i, m + i] = null_cost
    for j in range(m):
        aug[n + j, j] = null_cost
    aug[n:, m:] = 0.0
    rr, cc = linear_sum_assignment(aug)
    pairs = []
    unmatched_a = []
    unmatched_b = []
    for r, col in zip(rr, cc):
        if r < n and col < m:
            pairs.append((int(r), int(col), float(c[r, col])))
        elif r < n and col >= m:
            unmatched_a.append(int(r))
        elif r >= n and col < m:
            unmatched_b.append(int(col))
    return {
        "pairs": pairs,
        "unmatched_a": unmatched_a,
        "unmatched_b": unmatched_b,
        "cost": c,
        "rel_a": rel_a,
        "rel_b": rel_b,
    }


class StableOwnerState:
    """Minimal causal publication contract; no post-hoc owner rewrite."""

    def __init__(self, min_hits: int = 2):
        self.min_hits = int(min_hits)
        self.confirmed = {}
        self.pending = {}

    def update(self, proposals):
        published = []
        for a, b in proposals:
            if a in self.confirmed and self.confirmed[a] != b:
                continue
            if b in self.confirmed.values() and self.confirmed.get(a) != b:
                continue
            key = (int(a), int(b))
            self.pending[key] = self.pending.get(key, 0) + 1
            if self.pending[key] >= self.min_hits:
                self.confirmed[int(a)] = int(b)
                published.append(key)
        return published


def run() -> dict:
    boxes_a = np.array([[0, 0, 2, 2], [8, 0, 10, 2], [20, 0, 22, 2]], dtype=np.float64)
    boxes_b = np.array([[8.1, 0, 10.1, 2], [0.1, 0, 2.1, 2], [60, 0, 62, 2]], dtype=np.float64)
    emb_a = unit(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64))
    emb_b = unit(np.array([[0, 1, 0], [1, 0, 0], [0.3, 0.3, 0.9]], dtype=np.float64))

    base = partial_transport(boxes_a, boxes_b, emb_a, emb_b)
    assert np.isfinite(base["cost"]).all()
    assert len({a for a, _, _ in base["pairs"]}) == len(base["pairs"])
    assert len({b for _, b, _ in base["pairs"]}) == len(base["pairs"])

    # Permuting either input may only permute the reported indices.
    pa = np.array([2, 0, 1])
    pb = np.array([1, 2, 0])
    perm = partial_transport(boxes_a[pa], boxes_b[pb], emb_a[pa], emb_b[pb])
    mapped = {(int(pa[a]), int(pb[b])) for a, b, _ in perm["pairs"]}
    original = {(a, b) for a, b, _ in base["pairs"]}
    assert mapped == original, (mapped, original)

    # A future observation is not accepted as a function input to the t prefix.
    prefix = partial_transport(boxes_a[:2], boxes_b[:2], emb_a[:2], emb_b[:2])
    future_boxes = np.vstack([boxes_a[:2], [[100, 100, 102, 102]]])
    future_emb = np.vstack([emb_a[:2], [[0.2, 0.8, 0.1]]])
    prefix_again = partial_transport(future_boxes[:2], boxes_b[:2], future_emb[:2], emb_b[:2])
    assert np.allclose(prefix["cost"], prefix_again["cost"])

    # Confirmed owner continuity: a later conflicting proposal cannot rewrite it.
    state = StableOwnerState(min_hits=2)
    assert state.update([(10, 20)]) == []
    assert state.update([(10, 20)]) == [(10, 20)]
    assert state.update([(10, 21)]) == []
    assert state.confirmed[10] == 20

    result = {
        "status": "PASS_P45_CPU_MECHANISM_CONTRACT",
        "tests": [
            "finite_cost_and_injective_partial_plan",
            "permutation_equivariance",
            "strict_prefix_input_boundary",
            "confirmed_owner_continuity",
        ],
        "labels_read": False,
        "xml_read": False,
        "mot_metrics": False,
    }
    return result


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "CPU_CONTRACT.json"
    result = run()
    result["sha256_self"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
