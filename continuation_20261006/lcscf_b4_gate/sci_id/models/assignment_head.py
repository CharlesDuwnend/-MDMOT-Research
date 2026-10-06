"""Single- and multi-scale universal/event K+1 association heads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .set_composition_adapter import AdapterOutput, SetCompositionAdapter


@dataclass
class AssignmentOutput:
    candidate_logits: Tensor
    dustbin_logit: Tensor
    logits: Tensor
    universal_source: Tensor
    universal_candidates: Tensor
    event_source: Tensor
    event_candidates: Tensor
    level_adapters: Dict[str, AdapterOutput]


def _pool(x: Tensor) -> Tensor:
    return F.normalize(x.mean(dim=(-1, -2)), dim=-1, eps=1e-6)


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
    return (x * mask[:, :, None].to(x.dtype)).sum(dim=1) / denominator


class SCIIDAssignmentHead(nn.Module):
    """Single-scale diagnostic head retained for focused adapter tests."""

    def __init__(
        self, channels: int = 256, prototypes: int = 4, adapter_enabled: bool = True
    ) -> None:
        super().__init__()
        self.adapter_enabled = adapter_enabled
        self.adapter = SetCompositionAdapter(channels, prototypes, enabled=adapter_enabled)
        self.event_scale = nn.Parameter(torch.tensor(1.0))
        self.dustbin = nn.Sequential(
            nn.Linear(channels * 3, channels), nn.SiLU(), nn.Linear(channels, 1)
        )

    def forward(
        self, source_maps: Tensor, candidate_maps: Tensor, mask: Tensor
    ) -> AssignmentOutput:
        adapted = self.adapter(source_maps, candidate_maps, mask)
        us = _pool(adapted.universal_source)
        uc = _pool(adapted.universal_candidates)
        es = _pool(adapted.event_source)
        ec = _pool(adapted.event_candidates)
        universal_score = (us[:, None] * uc).sum(-1)
        event_score = (es * ec).sum(-1)
        candidate_logits = universal_score
        if self.adapter_enabled:
            candidate_logits = candidate_logits + self.event_scale * event_score
        candidate_logits = candidate_logits.masked_fill(
            ~mask, torch.finfo(candidate_logits.dtype).min
        )
        event_set = _masked_mean(0.5 * (es + ec), mask)
        universal_set = _masked_mean(uc, mask)
        dustbin_logit = self.dustbin(
            torch.cat((us, universal_set, event_set), dim=-1)
        ).squeeze(-1)
        return AssignmentOutput(
            candidate_logits,
            dustbin_logit,
            torch.cat((candidate_logits, dustbin_logit[:, None]), dim=1),
            us,
            uc,
            es,
            ec,
            {"single": adapted},
        )


class MultiScaleSCIIDHead(nn.Module):
    """Jointly adapts P2/P3/P4 maps before any identity pooling.

    Adapter parameters are shared across scales. Scale fusion is learned only
    after each level has already undergone the same candidate-composition
    operation, keeping the novelty core singular and testable.
    """

    levels = ("p2_id", "p3_id", "p4_id")

    def __init__(
        self, channels: int = 256, prototypes: int = 4, adapter_enabled: bool = True
    ) -> None:
        super().__init__()
        self.channels = channels
        self.adapter_enabled = adapter_enabled
        self.adapter = SetCompositionAdapter(channels, prototypes, enabled=adapter_enabled)
        fused_channels = channels * len(self.levels)
        self.universal_fuse = nn.Sequential(
            nn.LayerNorm(fused_channels), nn.Linear(fused_channels, channels), nn.SiLU()
        )
        self.event_fuse = nn.Sequential(
            nn.LayerNorm(fused_channels), nn.Linear(fused_channels, channels), nn.SiLU()
        )
        self.event_scale = nn.Parameter(torch.tensor(1.0))
        self.dustbin = nn.Sequential(
            nn.Linear(channels * 3, channels), nn.SiLU(), nn.Linear(channels, 1)
        )

    def _validate_levels(
        self,
        source_maps: Mapping[str, Tensor],
        candidate_maps: Mapping[str, Tensor],
    ) -> None:
        if set(source_maps) != set(self.levels) or set(candidate_maps) != set(self.levels):
            raise ValueError("source/candidate maps must contain p2_id, p3_id and p4_id")

    def forward(
        self,
        source_maps: Mapping[str, Tensor],
        candidate_maps: Mapping[str, Tensor],
        mask: Tensor,
    ) -> AssignmentOutput:
        self._validate_levels(source_maps, candidate_maps)
        adapted: Dict[str, AdapterOutput] = {}
        us_levels = []
        uc_levels = []
        es_levels = []
        ec_levels = []
        for level in self.levels:
            level_output = self.adapter(source_maps[level], candidate_maps[level], mask)
            adapted[level] = level_output
            us_levels.append(_pool(level_output.universal_source))
            uc_levels.append(_pool(level_output.universal_candidates))
            es_levels.append(_pool(level_output.event_source))
            ec_levels.append(_pool(level_output.event_candidates))

        us = F.normalize(self.universal_fuse(torch.cat(us_levels, dim=-1)), dim=-1)
        uc = F.normalize(self.universal_fuse(torch.cat(uc_levels, dim=-1)), dim=-1)
        es = F.normalize(self.event_fuse(torch.cat(es_levels, dim=-1)), dim=-1)
        ec = F.normalize(self.event_fuse(torch.cat(ec_levels, dim=-1)), dim=-1)
        universal_score = (us[:, None] * uc).sum(-1)
        event_score = (es * ec).sum(-1)
        candidate_logits = universal_score
        if self.adapter_enabled:
            candidate_logits = candidate_logits + self.event_scale * event_score
        candidate_logits = candidate_logits.masked_fill(
            ~mask, torch.finfo(candidate_logits.dtype).min
        )
        universal_set = _masked_mean(uc, mask)
        event_set = _masked_mean(0.5 * (es + ec), mask)
        dustbin_logit = self.dustbin(
            torch.cat((us, universal_set, event_set), dim=-1)
        ).squeeze(-1)
        return AssignmentOutput(
            candidate_logits,
            dustbin_logit,
            torch.cat((candidate_logits, dustbin_logit[:, None]), dim=1),
            us,
            uc,
            es,
            ec,
            adapted,
        )
