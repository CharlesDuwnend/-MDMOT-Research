#!/usr/bin/env python3
"""CPU mechanism contract for P63's partial set transition.

This synthetic test checks tensor/operator contracts only.  It is not an MOT
metric and cannot authorize host training or official evaluation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "continuation_20261006/p63_causal_set_owner_transition"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def log_sinkhorn(scores, iterations=30):
    """Square entropic partial assignment with a learned dustbin row/column."""
    n = scores.shape[-1]
    log_mu = scores.new_full((n,), -np.log(n))
    log_nu = scores.new_full((n,), -np.log(n))
    u = scores.new_zeros(n)
    v = scores.new_zeros(n)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(scores + v[None, :], dim=1)
        v = log_nu - torch.logsumexp(scores + u[:, None], dim=0)
    return torch.exp(scores + u[:, None] + v[None, :])


class SetOwnerTransition(nn.Module):
    def __init__(self, dim=14, hidden=32):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.dust_row = nn.Parameter(torch.zeros(1))
        self.dust_col = nn.Parameter(torch.zeros(1))

    def forward(self, features):
        # features: [targets, sources, dim].  The final row/column are dustbin.
        n, m, _ = features.shape
        size = max(n, m) + 1
        scores = features.new_full((size, size), -8.0)
        scores[:n, :m] = self.edge(features).squeeze(-1)
        scores[n:, :m] = self.dust_col
        scores[:n, m:] = self.dust_row
        scores[n:, m:] = 0.0
        return scores, log_sinkhorn(scores)


def owner_transition(prob, n, m, threshold=0.04):
    """Greedy conflict-free hard transition used only for the contract."""
    edges = []
    for i, j in sorted(((i, j) for i in range(n) for j in range(m)), key=lambda x: (-float(prob[x[0], x[1]]), x)):
        if float(prob[i, j]) < threshold:
            continue
        if any(a == i or b == j for a, b in edges):
            continue
        edges.append((i, j))
    return edges


def main():
    torch.manual_seed(63063)
    np.random.seed(63063)
    n, m, d = 3, 4, 14
    features = torch.randn(n, m, d, dtype=torch.float32)
    target = {(0, 1), (1, 3)}
    model = SetOwnerTransition(d)
    scores, prob = model(features)
    # Every row/column has the prescribed entropic marginal, including dustbin.
    marginal_error = max(float((prob.sum(1) - 1.0 / prob.shape[0]).abs().max()), float((prob.sum(0) - 1.0 / prob.shape[1]).abs().max()))
    row_losses = []
    for i in range(n):
        row = prob[i, :m + 1]
        row = row / row.sum()
        j = next((b for a, b in target if a == i), None)
        row_losses.append(-torch.log(row[j] + 1e-8) if j is not None else -torch.log(row[m] + 1e-8))
    # A soft conflict term is differentiable and zero for the synthetic target.
    conflict = sum(prob[i, j] * prob[i2, j] for i in range(n) for i2 in range(i + 1, n) for j in range(m))
    loss = torch.stack(row_losses).mean() + 0.2 * conflict
    loss.backward()
    finite_gradients = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.isfinite(loss) and finite_gradients

    row_perm = torch.tensor([2, 0, 1]); col_perm = torch.tensor([3, 1, 0, 2])
    with torch.no_grad():
        _, perm_prob = model(features[row_perm][:, col_perm])
    # Undo permutations on real rows/columns; dustbin is excluded.
    restored = perm_prob[:n, :m][torch.argsort(row_perm)][:, torch.argsort(col_perm)]
    permutation_error = float((restored - prob[:n, :m]).abs().max())
    edges = owner_transition(prob.detach(), n, m)
    assert edges, "synthetic transition must exercise a nonempty owner update"
    assert len({i for i, _ in edges}) == len(edges) and len({j for _, j in edges}) == len(edges)
    report = {
        "status": "PASS_P63_CPU_SET_OWNER_CONTRACT",
        "parameters": sum(p.numel() for p in model.parameters()),
        "loss": float(loss.detach()),
        "finite_gradients": bool(finite_gradients),
        "sinkhorn_marginal_max_error": marginal_error,
        "set_permutation_max_error": permutation_error,
        "conflict_free_edges": [list(x) for x in edges],
        "owner_transition_contract": True,
        "official_val_test_read": False,
        "source_sha256": sha(Path(__file__)),
        "prereg_sha256": sha(OUT / "PREREG.json"),
    }
    assert marginal_error < 1e-6 and permutation_error < 1e-6
    (OUT / "CPU_CONTRACT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
