"""Strong N3 association-head controls sharing the SCI-ID map contract."""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .assignment_head import AssignmentOutput, MultiScaleSCIIDHead


LEVELS = ("p2_id", "p3_id", "p4_id")


def _pool(maps: Tensor) -> Tensor:
    return F.normalize(maps.mean(dim=(-1, -2)), dim=-1, eps=1e-6)


def _masked_mean(features: Tensor, mask: Tensor) -> Tensor:
    denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(features.dtype)
    return (features * mask[:, :, None].to(features.dtype)).sum(dim=1) / denominator


def _validate(source_maps: Mapping[str, Tensor], candidate_maps: Mapping[str, Tensor], mask: Tensor) -> None:
    if set(source_maps) != set(LEVELS) or set(candidate_maps) != set(LEVELS):
        raise ValueError("source/candidate maps must contain p2_id, p3_id and p4_id")
    first = candidate_maps[LEVELS[0]]
    if first.ndim != 5 or mask.shape != first.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("candidate maps must be [B,K,C,H,W] with bool mask [B,K]")
    if not 1 <= first.shape[1] <= 64:
        raise ValueError("tensor candidate width must be in [1,64]")


class B1UniversalMultiScaleHead(MultiScaleSCIIDHead):
    """Ordinary multiscale head with the SCI-ID adapter causally disabled.

    It intentionally retains the identical module skeleton/parameter budget;
    adapter parameters receive no gradient. This is the direct adapter-off
    control, not the only universal-ReID baseline used in final experiments.
    """

    def __init__(self, channels: int = 256, prototypes: int = 4) -> None:
        super().__init__(channels, prototypes, adapter_enabled=False)


class _FusionAndDustbin(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        fused = channels * len(LEVELS)
        self.fuse = nn.Sequential(
            nn.LayerNorm(fused), nn.Linear(fused, channels), nn.SiLU()
        )
        self.dustbin = nn.Sequential(
            nn.Linear(channels * 3, channels), nn.SiLU(), nn.Linear(channels, 1)
        )

    def pooled(
        self, source_maps: Mapping[str, Tensor], candidate_maps: Mapping[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        source = F.normalize(
            self.fuse(torch.cat([_pool(source_maps[level]) for level in LEVELS], dim=-1)),
            dim=-1,
        )
        candidates = F.normalize(
            self.fuse(torch.cat([_pool(candidate_maps[level]) for level in LEVELS], dim=-1)),
            dim=-1,
        )
        return source, candidates

    def dustbin_logit(
        self, source: Tensor, candidates: Tensor, event_set: Tensor, mask: Tensor
    ) -> Tensor:
        return self.dustbin(
            torch.cat((source, _masked_mean(candidates, mask), event_set), dim=-1)
        ).squeeze(-1)


class B2PairwiseSpatialHead(nn.Module):
    """QAConv-like pairwise local correspondence without set-conditioned maps."""

    def __init__(self, channels: int = 256, projection_channels: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.common = _FusionAndDustbin(channels)
        self.spatial_projection = nn.Conv2d(channels, projection_channels, 1, bias=False)
        self.level_logits = nn.Parameter(torch.zeros(len(LEVELS)))
        self.spatial_scale = nn.Parameter(torch.tensor(1.0))

    def _level_score(self, source: Tensor, candidates: Tensor) -> Tensor:
        b, k, c, h, w = candidates.shape
        source_projected = F.normalize(
            self.spatial_projection(source).flatten(2), dim=1, eps=1e-6
        )
        candidate_projected = F.normalize(
            self.spatial_projection(candidates.reshape(b * k, c, h, w))
            .reshape(b, k, -1, h * w),
            dim=2,
            eps=1e-6,
        )
        correspondence = torch.einsum(
            "bcn,bkcm->bknm", source_projected, candidate_projected
        )
        source_to_candidate = correspondence.max(dim=-1).values.mean(dim=-1)
        candidate_to_source = correspondence.max(dim=-2).values.mean(dim=-1)
        return 0.5 * (source_to_candidate + candidate_to_source)

    def forward(
        self,
        source_maps: Mapping[str, Tensor],
        candidate_maps: Mapping[str, Tensor],
        mask: Tensor,
    ) -> AssignmentOutput:
        _validate(source_maps, candidate_maps, mask)
        source, candidates = self.common.pooled(source_maps, candidate_maps)
        level_weight = torch.softmax(self.level_logits, dim=0)
        spatial = sum(
            level_weight[index]
            * self._level_score(source_maps[level], candidate_maps[level])
            for index, level in enumerate(LEVELS)
        )
        candidate_logits = (source[:, None] * candidates).sum(-1)
        candidate_logits = candidate_logits + self.spatial_scale * spatial
        candidate_logits = candidate_logits.masked_fill(
            ~mask, torch.finfo(candidate_logits.dtype).min
        )
        finite_spatial = spatial.masked_fill(~mask, -1e4)
        weights = torch.softmax(finite_spatial, dim=1) * mask.to(spatial.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        event_set = (candidates * weights[:, :, None]).sum(dim=1)
        dustbin = self.common.dustbin_logit(source, candidates, event_set, mask)
        source_per_candidate = source[:, None].expand_as(candidates)
        return AssignmentOutput(
            candidate_logits,
            dustbin,
            torch.cat((candidate_logits, dustbin[:, None]), dim=1),
            source,
            candidates,
            source_per_candidate,
            candidates,
            {},
        )


class B3PooledSetRelativeHead(nn.Module):
    """Post-pooling DeepSets/leave-other context control."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.channels = channels
        self.common = _FusionAndDustbin(channels)
        self.source_update = nn.Sequential(
            nn.LayerNorm(channels * 3), nn.Linear(channels * 3, channels), nn.SiLU()
        )
        self.candidate_update = nn.Sequential(
            nn.LayerNorm(channels * 3), nn.Linear(channels * 3, channels), nn.SiLU()
        )
        self.update_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        source_maps: Mapping[str, Tensor],
        candidate_maps: Mapping[str, Tensor],
        mask: Tensor,
    ) -> AssignmentOutput:
        _validate(source_maps, candidate_maps, mask)
        source, candidates = self.common.pooled(source_maps, candidate_maps)
        valid = mask[:, :, None].to(candidates.dtype)
        candidate_sum = (candidates * valid).sum(dim=1, keepdim=True)
        other_count = (mask.sum(dim=1, keepdim=True) - 1).clamp_min(1)
        leave_other = (candidate_sum - candidates * valid) / other_count[:, :, None].to(candidates.dtype)
        source_expand = source[:, None].expand_as(candidates)
        event_source = F.normalize(
            source_expand
            + self.update_scale
            * self.source_update(torch.cat((source_expand, candidates, leave_other), dim=-1)),
            dim=-1,
        )
        event_candidates = F.normalize(
            candidates
            + self.update_scale
            * self.candidate_update(torch.cat((candidates, source_expand, leave_other), dim=-1)),
            dim=-1,
        )
        event_source = event_source * valid
        event_candidates = event_candidates * valid
        candidate_logits = (event_source * event_candidates).sum(-1).masked_fill(
            ~mask, torch.finfo(candidates.dtype).min
        )
        event_set = _masked_mean(0.5 * (event_source + event_candidates), mask)
        dustbin = self.common.dustbin_logit(source, candidates, event_set, mask)
        return AssignmentOutput(
            candidate_logits,
            dustbin,
            torch.cat((candidate_logits, dustbin[:, None]), dim=1),
            source,
            candidates,
            event_source,
            event_candidates,
            {},
        )


class B4PooledExactCompositionHead(nn.Module):
    """Exact SCI-ID operation after spatial pooling, parameter-matched to P1."""

    def __init__(self, channels: int = 256, prototypes: int = 4) -> None:
        super().__init__()
        self.head = MultiScaleSCIIDHead(channels, prototypes, adapter_enabled=True)

    def forward(
        self,
        source_maps: Mapping[str, Tensor],
        candidate_maps: Mapping[str, Tensor],
        mask: Tensor,
    ) -> AssignmentOutput:
        _validate(source_maps, candidate_maps, mask)
        pooled_source = {
            level: source_maps[level].mean(dim=(-1, -2), keepdim=True)
            for level in LEVELS
        }
        pooled_candidates = {
            level: candidate_maps[level].mean(dim=(-1, -2), keepdim=True)
            for level in LEVELS
        }
        return self.head(pooled_source, pooled_candidates, mask)
