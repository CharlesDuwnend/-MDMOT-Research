"""Cross-view temporal differential co-movement field prototype.

The module consumes detector-derived ROI features and support validity only;
tracker IDs and candidate-set context are intentionally absent from its API.
"""
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn

class CrossViewTemporalDifferentialField(nn.Module):
    def __init__(self, dim: int, hidden: int = 32, output_dim: int | None = None) -> None:
        super().__init__()
        self.dim = int(dim)
        self.output_dim = int(output_dim or dim)
        self.field = nn.Sequential(
            nn.LayerNorm(2 * self.dim + 2),
            nn.Linear(2 * self.dim + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.output_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * self.dim + 2),
            nn.Linear(2 * self.dim + 2, 1),
        )
        self.base = nn.Sequential(
            nn.LayerNorm(2 * self.dim),
            nn.Linear(2 * self.dim, self.output_dim),
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        current_target: Tensor,
        current_source: Tensor,
        support_target: Tensor,
        support_source: Tensor,
        support_offsets: Tensor,
        support_valid: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if current_target.ndim != 2 or current_source.shape != current_target.shape:
            raise ValueError("current views must be [B,D] and matching")
        if support_target.ndim != 3 or support_source.shape != support_target.shape:
            raise ValueError("support views must be [B,S,D] and matching")
        b, s, d = support_target.shape
        if d != self.dim or current_target.shape != (b, d):
            raise ValueError("feature dimension mismatch")
        if support_offsets.shape != (b, s) or support_valid.shape != (b, s) or support_valid.dtype != torch.bool:
            raise ValueError("offsets/valid must be [B,S]")
        if s < 1 or not torch.isfinite(support_offsets).all():
            raise ValueError("support offsets must be finite and S>=1")
        if not torch.isfinite(current_target).all() or not torch.isfinite(current_source).all():
            raise ValueError("current features must be finite")
        expanded_valid = support_valid[:, :, None].expand_as(support_target)
        if (not torch.isfinite(support_target[expanded_valid]).all()
                or not torch.isfinite(support_source[expanded_valid]).all()):
            raise ValueError("observed support features must be finite")
        # Evidence is chronological; an explicit mask may remove missing rows,
        # but it may not silently reorder the observed support sequence.
        if s > 1 and bool(((support_offsets[:, 1:] - support_offsets[:, :-1]) < 0).any()):
            raise ValueError("support offsets must be nondecreasing")
        # Masked placeholders are not observations. Remove them before any
        # nonlinear operation so NaN*0 cannot contaminate a valid output.
        target = F.normalize(support_target.masked_fill(~expanded_valid, 0.), dim=-1, eps=1e-6)
        source = F.normalize(support_source.masked_fill(~expanded_valid, 0.), dim=-1, eps=1e-6)
        gaps = support_offsets[:, 1:] - support_offsets[:, :-1]
        pair_valid = support_valid[:, 1:] & support_valid[:, :-1] & (gaps > 0)
        dt = (target[:, 1:] - target[:, :-1]) / gaps.clamp_min(1e-6).unsqueeze(-1)
        ds = (source[:, 1:] - source[:, :-1]) / gaps.clamp_min(1e-6).unsqueeze(-1)
        dt = dt.masked_fill(~pair_valid[:, :, None], 0.)
        ds = ds.masked_fill(~pair_valid[:, :, None], 0.)
        if s == 1:
            field_tokens = support_target.new_zeros((b, 0, 2 * d + 2))
        else:
            gap_token = gaps.log1p().unsqueeze(-1)
            elapsed_token = (support_offsets[:, -1:] - support_offsets[:, 1:]).log1p().unsqueeze(-1)
            field_tokens = torch.cat((dt * ds, (dt - ds).abs(), gap_token, elapsed_token), dim=-1)
        if field_tokens.shape[1]:
            weights = self.gate(field_tokens).squeeze(-1).masked_fill(~pair_valid, -1e4)
            weights = torch.softmax(weights, dim=1).masked_fill(~pair_valid, 0.0)
            field = (self.field(field_tokens) * weights.unsqueeze(-1)).sum(dim=1)
            valid_pair_count = pair_valid.sum(dim=1)
        else:
            weights = support_target.new_zeros((b, 0))
            field = support_target.new_zeros((b, self.output_dim))
            valid_pair_count = torch.zeros(b, dtype=torch.long, device=support_valid.device)
        base = self.base(torch.cat((F.normalize(current_target, dim=-1), F.normalize(current_source, dim=-1)), dim=-1))
        output = F.normalize(base + torch.tanh(self.residual_scale) * field, dim=-1, eps=1e-6)
        stats = {
            "valid_pair_count": valid_pair_count,
            "support_valid_count": support_valid.sum(dim=1),
            "field_norm": field.norm(dim=-1),
            "residual_norm": (torch.tanh(self.residual_scale) * field).norm(dim=-1),
            "weights": weights,
        }
        return output, stats
