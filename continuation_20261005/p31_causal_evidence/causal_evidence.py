"""P31 causal support-set evidence operator.

This module deliberately operates on already extracted target observations.  It
does not alter detector boxes or fabricate observations.  A support set is a
strictly causal collection for one local track/candidate; the operator returns
an order-invariant representation, per-support leave-one-out influence, a
quality scalar, and a pair logit with explicit null logits.

The implementation is a mechanism prototype.  It is not an effectiveness
claim and has no dataset, XML, ID, or tracker dependency.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


def _validate(tokens: Tensor, ages: Tensor, mask: Tensor) -> Tuple[int, int, int]:
    if tokens.ndim != 3:
        raise ValueError("tokens must be [B,N,D]")
    if ages.shape != tokens.shape[:2] or mask.shape != tokens.shape[:2]:
        raise ValueError("ages/mask must be [B,N]")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be bool")
    if not torch.isfinite(tokens).all() or not torch.isfinite(ages).all():
        raise ValueError("tokens and ages must be finite")
    if (ages < 0).any():
        raise ValueError("age must be non-negative; future offsets are illegal")
    return tuple(tokens.shape)


def _masked_weights(logits: Tensor, mask: Tensor) -> Tensor:
    """Finite masked softmax, including an all-invalid empty-support case."""
    masked = logits.masked_fill(~mask, -1e9)
    weights = torch.softmax(masked, dim=1)
    weights = weights * mask.to(dtype=weights.dtype)
    denom = weights.sum(dim=1, keepdim=True)
    return torch.where(denom > 0, weights / denom.clamp_min(1e-12), torch.zeros_like(weights))


class CausalEvidenceEncoder(nn.Module):
    """Encode one strict-past support set without depending on token order."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64,
                 max_age: float = 32.0):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.max_age = float(max_age)
        self.token = nn.Sequential(
            nn.Linear(self.input_dim + 1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.weight = nn.Linear(hidden_dim, 1)
        self.value = nn.Linear(hidden_dim, output_dim)
        self.quality = nn.Sequential(
            nn.Linear(output_dim + 2, hidden_dim // 2 or 1), nn.GELU(),
            nn.Linear(hidden_dim // 2 or 1, 1))

    def forward(self, tokens: Tensor, ages: Tensor, mask: Tensor) -> Dict[str, Tensor]:
        _validate(tokens, ages, mask)
        age = (ages / self.max_age).clamp(0.0, 1.0).unsqueeze(-1)
        h = self.token(torch.cat([tokens, age], dim=-1))
        raw = self.weight(h).squeeze(-1)
        alpha = _masked_weights(raw, mask)
        values = self.value(h)
        z = (alpha.unsqueeze(-1) * values).sum(dim=1)

        # Leave-one-support-out influence.  It is defined for masked tokens as
        # zero and remains finite when a set has only one valid observation.
        total = z.unsqueeze(1)
        remaining_mass = (1.0 - alpha).clamp_min(1e-6).unsqueeze(-1)
        loo = (total - alpha.unsqueeze(-1) * values) / remaining_mass
        influence = torch.linalg.vector_norm(z.unsqueeze(1) - loo, dim=-1)
        influence = influence * mask.to(dtype=influence.dtype)
        valid_count = mask.sum(dim=1).to(dtype=z.dtype)
        mean_influence = influence.sum(dim=1) / valid_count.clamp_min(1.0)
        spread = ((alpha.unsqueeze(-1) * (values - z.unsqueeze(1)) ** 2).sum(dim=1)
                  .mean(dim=-1))
        quality = torch.sigmoid(self.quality(torch.cat([z, mean_influence[:, None], spread[:, None]], dim=-1)))
        return {
            "embedding": z,
            "quality": quality.squeeze(-1),
            "weights": alpha,
            "influence": influence,
            "valid_count": valid_count,
        }


class CausalEvidenceIntervention(nn.Module):
    """Pair two support sets and expose match/null evidence."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64,
                 max_age: float = 32.0):
        super().__init__()
        self.left = CausalEvidenceEncoder(input_dim, hidden_dim, output_dim, max_age)
        self.right = CausalEvidenceEncoder(input_dim, hidden_dim, output_dim, max_age)
        pair_dim = output_dim * 4 + 4
        self.match = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.null_left = nn.Linear(output_dim + 1, 1)
        self.null_right = nn.Linear(output_dim + 1, 1)

    def forward(self, left_tokens: Tensor, left_ages: Tensor, left_mask: Tensor,
                right_tokens: Tensor, right_ages: Tensor, right_mask: Tensor) -> Dict[str, Tensor]:
        a = self.left(left_tokens, left_ages, left_mask)
        b = self.right(right_tokens, right_ages, right_mask)
        za, zb = a["embedding"], b["embedding"]
        pair = torch.cat([
            za, zb, torch.abs(za - zb), za * zb,
            a["quality"][:, None], b["quality"][:, None],
            a["valid_count"][:, None], b["valid_count"][:, None],
        ], dim=-1)
        return {
            "match_logit": self.match(pair).squeeze(-1),
            "null_left_logit": self.null_left(torch.cat([za, a["quality"][:, None]], dim=-1)).squeeze(-1),
            "null_right_logit": self.null_right(torch.cat([zb, b["quality"][:, None]], dim=-1)).squeeze(-1),
            "left": a,
            "right": b,
        }


def sinkhorn_with_dustbin(edge_logits: Tensor, null_left: Tensor, null_right: Tensor,
                          iters: int = 8) -> Tensor:
    """Return a finite soft assignment with one explicit null row and column.

    This is a differentiable assignment primitive for the later host port.  It
    is not an evaluator and does not decide IDs by itself.
    """
    if edge_logits.ndim != 3:
        raise ValueError("edge_logits must be [B,Na,Nb]")
    b, na, nb = edge_logits.shape
    if null_left.shape != (b, na) or null_right.shape != (b, nb):
        raise ValueError("null logits have incompatible shapes")
    row = torch.cat([edge_logits, null_left.unsqueeze(-1)], dim=-1)
    null_corner = torch.zeros((b, 1, 1), dtype=edge_logits.dtype, device=edge_logits.device)
    col = torch.cat([null_right.unsqueeze(1), null_corner], dim=2)
    logits = torch.cat([row, col], dim=1)
    log_p = logits
    for _ in range(int(iters)):
        log_p = torch.log_softmax(log_p, dim=2)
        log_p = torch.log_softmax(log_p, dim=1)
    return torch.exp(log_p)


def parameter_count(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())

