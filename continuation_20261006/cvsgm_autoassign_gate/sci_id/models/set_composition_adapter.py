"""Candidate-composition adapter for pre-pooling spatial identity maps.

The production forward path never materializes a ``[B,K,M,C,H,W]`` tensor.
It computes one prototype slice at a time, so the dominant spatial activation
is linear in the number of candidates K and ambiguity prototypes M.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class AdapterOutput:
    universal_source: Tensor
    universal_candidates: Tensor
    event_source: Tensor
    event_candidates: Tensor
    assignment: Tensor
    prototypes: Tensor
    leave_one_out_context: Tensor
    exclusive_context: Tensor
    mask: Tensor


class SetCompositionAdapter(nn.Module):
    """Build member-specific maps from the current mutually exclusive set.

    ``event_source[:, j]`` and ``event_candidates[:, j]`` are both modulated by
    candidate j relative to all other feasible candidates. No candidate index,
    GT identity, tracker ID, future frame, or geometry enters this module.
    """

    def __init__(
        self,
        channels: int = 256,
        prototypes: int = 4,
        enabled: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if prototypes < 1:
            raise ValueError("prototypes must be positive")
        self.channels = channels
        self.prototypes = prototypes
        self.enabled = enabled
        self.eps = eps
        self.psi = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.prototype_logits = nn.Linear(channels, prototypes, bias=False)
        self.relevance_log_scale = nn.Parameter(torch.tensor(1.4))
        self.relevance_bias = nn.Parameter(torch.tensor(0.0))
        self.channel_residual = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, channels)
        )
        self.spatial_residual = nn.Conv2d(channels, 1, kernel_size=1)

    def _validate(self, source: Tensor, candidates: Tensor, mask: Tensor) -> None:
        if source.ndim != 4 or candidates.ndim != 5:
            raise ValueError("source=[B,C,H,W], candidates=[B,K,C,H,W]")
        if (
            source.shape[0] != candidates.shape[0]
            or source.shape[1] != candidates.shape[2]
            or source.shape[-2:] != candidates.shape[-2:]
        ):
            raise ValueError("source/candidate shapes are incompatible")
        if candidates.shape[1] < 1 or candidates.shape[1] > 64:
            raise ValueError("candidate K must satisfy 1 <= K <= 64")
        if mask.shape != candidates.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be bool [B,K]")
        # A padded row may contain no valid member.  Such rows are real K=0
        # events in the frozen MDMT ledger and supervise the learned dustbin.
        # The tensor width itself remains >=1 so the batched K+1 layout is
        # stable; all masked map/logit reductions below evaluate to zero.

    def visual_relevance(self, source: Tensor, candidates: Tensor, mask: Tensor) -> Tensor:
        """Self-contained visual relevance; no externally supplied zero gate."""
        src = F.normalize(source.mean(dim=(-1, -2)), dim=-1, eps=self.eps)
        cand = F.normalize(candidates.mean(dim=(-1, -2)), dim=-1, eps=self.eps)
        cosine = (src[:, None] * cand).sum(dim=-1)
        scale = F.softplus(self.relevance_log_scale) + 1.0
        relevance = torch.sigmoid(scale * cosine + self.relevance_bias)
        return relevance * mask.to(dtype=candidates.dtype)

    def _base_statistics(
        self, source: Tensor, candidates: Tensor, mask: Tensor
    ) -> Dict[str, Tensor]:
        self._validate(source, candidates, mask)
        b, k, c, h, w = candidates.shape
        pooled = candidates.mean(dim=(-1, -2))
        proto_prob = torch.softmax(self.prototype_logits(pooled), dim=-1)
        relevance = self.visual_relevance(source, candidates, mask)
        assignment = proto_prob * relevance.unsqueeze(-1)
        psi = self.psi(candidates.reshape(b * k, c, h, w)).reshape(b, k, c, h, w)
        # Avoid an explicit [B,K,M,C,H,W] weighted tensor.
        sums = torch.einsum("bkm,bkchw->bmchw", assignment, psi)
        mass = assignment.sum(dim=1)
        prototypes = sums / (mass[..., None, None, None] + self.eps)
        return {
            "assignment": assignment,
            "proto_prob": proto_prob,
            "relevance": relevance,
            "psi": psi,
            "sum": sums,
            "mass": mass,
            "prototypes": prototypes,
        }

    def _streaming_leave_one_out_context(self, stats: Dict[str, Tensor]) -> Tensor:
        """Return [B,K,C,H,W] without materializing all M LOO maps."""
        assignment = stats["assignment"]
        proto_prob = stats["proto_prob"]
        psi = stats["psi"]
        sums = stats["sum"]
        mass = stats["mass"]
        context = torch.zeros_like(psi)
        for m in range(self.prototypes):
            weight = proto_prob[:, :, m, None, None, None]
            a_m = assignment[:, :, m, None, None, None]
            numerator = sums[:, None, m] - a_m * psi
            denominator = mass[:, None, m, None, None, None] - a_m
            loo_m = numerator / (denominator + self.eps)
            context = context + weight * loo_m
        return context

    def materialize_leave_one_out(
        self, source: Tensor, candidates: Tensor, mask: Tensor
    ) -> Tensor:
        """Debug/test-only exact [B,K,M,C,H,W] leave-one-out maps."""
        stats = self._base_statistics(source, candidates, mask)
        assignment = stats["assignment"]
        psi = stats["psi"]
        loo = []
        for m in range(self.prototypes):
            a_m = assignment[:, :, m, None, None, None]
            numerator = stats["sum"][:, None, m] - a_m * psi
            denominator = stats["mass"][:, None, m, None, None, None] - a_m
            loo.append(numerator / (denominator + self.eps))
        return torch.stack(loo, dim=2)

    def prototype_statistics(
        self, candidates: Tensor, mask: Tensor, source: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Compatibility/debug API with exact materialized LOO output."""
        if source is None:
            source = candidates[:, 0]
        stats = self._base_statistics(source, candidates, mask)
        return {
            **stats,
            "leave_one_out": self.materialize_leave_one_out(source, candidates, mask),
        }

    def _modulate(self, maps: Tensor, context: Tensor) -> Tensor:
        cgate = torch.tanh(
            self.channel_residual(context.mean(dim=(-1, -2)))
        )[..., None, None]
        flat = context.reshape(-1, self.channels, *context.shape[-2:])
        sgate = torch.tanh(self.spatial_residual(flat)).reshape(
            *context.shape[:2], 1, *context.shape[-2:]
        )
        return maps * (1.0 + 0.10 * cgate + 0.10 * sgate)

    def forward(self, source: Tensor, candidates: Tensor, mask: Tensor) -> AdapterOutput:
        self._validate(source, candidates, mask)
        stats = self._base_statistics(source, candidates, mask)
        _, k, _, _, _ = candidates.shape
        source_per_candidate = source[:, None].expand(-1, k, -1, -1, -1)
        if not self.enabled:
            zeros = torch.zeros_like(candidates)
            return AdapterOutput(
                source,
                candidates,
                source_per_candidate,
                candidates,
                stats["assignment"],
                stats["prototypes"],
                zeros,
                zeros,
                mask,
            )
        loo_context = self._streaming_leave_one_out_context(stats)
        exclusive = stats["psi"] - loo_context
        event_source = self._modulate(source_per_candidate, exclusive)
        event_candidates = self._modulate(candidates, exclusive)
        valid = mask[:, :, None, None, None].to(candidates.dtype)
        return AdapterOutput(
            source,
            candidates,
            event_source * valid,
            event_candidates * valid,
            stats["assignment"],
            stats["prototypes"],
            loo_context,
            exclusive,
            mask,
        )

    def operation_count(self, batch: int, k: int, height: int, width: int) -> int:
        if not 1 <= k <= 64:
            raise ValueError("K must be in [1,64]")
        return int(batch * k * self.prototypes * self.channels * height * width)
