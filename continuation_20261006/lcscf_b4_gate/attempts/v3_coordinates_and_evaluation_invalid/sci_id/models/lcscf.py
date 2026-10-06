"""Latent local cross-view spatial transport for paired identity RoI maps.

This module deliberately avoids geometry, tracker state, candidate-set context,
and dense correspondence labels.  It computes a small fixed local transport
plan in both directions and exposes cycle/mass/entropy diagnostics to the
train-only gate.  The learned residual scale starts at zero so the B4 control
is exactly recovered at step zero.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


LEVELS = ("p2_id", "p3_id", "p4_id")
_OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
_OPPOSITE = tuple(_OFFSETS.index((-dy, -dx)) for dy, dx in _OFFSETS)


def _shift(x: Tensor, dy: int, dx: int) -> Tensor:
    """Replicate-padded local shift; output samples x[i+dy,j+dx]."""
    if x.ndim != 4:
        raise ValueError("shift expects [N,C,H,W]")
    _, _, h, w = x.shape
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return padded[:, :, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]


class LCSCFTransport(nn.Module):
    """Shared 3x3 bidirectional transport with per-scale zero-init residual."""

    def __init__(self, channels: int, levels: Tuple[str, ...] = LEVELS) -> None:
        super().__init__()
        self.channels = int(channels)
        self.levels = tuple(levels)
        if self.levels != LEVELS:
            raise ValueError("LCSCF expects p2_id, p3_id and p4_id")
        self.log_temperature = nn.Parameter(torch.tensor(-1.5))
        self.residual_scale = nn.Parameter(torch.zeros(len(self.levels)))
        self.visibility_bias = nn.Parameter(torch.tensor(0.0))

    def _validate(self, target: Mapping[str, Tensor], candidates: Mapping[str, Tensor], mask: Tensor) -> None:
        if tuple(target) != LEVELS or tuple(candidates) != LEVELS:
            raise ValueError("target/candidate levels must be p2_id,p3_id,p4_id")
        first = candidates[LEVELS[0]]
        if first.ndim != 5 or first.shape[2] != self.channels:
            raise ValueError("candidate maps must be [B,K,C,H,W]")
        if mask.shape != first.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be bool [B,K]")
        if not 1 <= first.shape[1] <= 64:
            raise ValueError("tensor K must be in [1,64]")
        for level in LEVELS:
            t, c = target[level], candidates[level]
            if t.ndim != 4 or c.ndim != 5 or c.shape[0] != t.shape[0] or c.shape[2] != t.shape[1] or c.shape[-2:] != t.shape[-2:]:
                raise ValueError("target/candidate map shapes are incompatible")

    @staticmethod
    def _local_neighbours(x: Tensor) -> Tensor:
        # [N,C,H,W] -> [N,9,C,H,W].
        return torch.stack([_shift(x, dy, dx) for dy, dx in _OFFSETS], dim=1)

    @staticmethod
    def _valid_neighbours(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return [9,H,W] masks; invalid border offsets never enter softmax."""
        rows = torch.arange(height, device=device)[:, None]
        cols = torch.arange(width, device=device)[None, :]
        masks = []
        for dy, dx in _OFFSETS:
            masks.append(((rows + dy >= 0) & (rows + dy < height) & (cols + dx >= 0) & (cols + dx < width)))
        return torch.stack(masks, dim=0).to(dtype)

    def _one_level(self, target: Tensor, candidates: Tensor, mask: Tensor, level_index: int):
        b, k, c, h, w = candidates.shape
        target_norm = F.normalize(target, dim=1, eps=1e-6)
        candidates_flat = candidates.reshape(b * k, c, h, w)
        candidate_norm = F.normalize(candidates_flat, dim=1, eps=1e-6).reshape(b, k, c, h, w)
        target_neighbours = self._local_neighbours(target)  # [B,9,C,H,W]
        candidate_neighbours = self._local_neighbours(candidates_flat).reshape(b, k, 9, c, h, w)
        target_neighbours_norm = F.normalize(target_neighbours, dim=2, eps=1e-6)
        candidate_neighbours_norm = F.normalize(candidate_neighbours, dim=3, eps=1e-6)
        valid_offsets = self._valid_neighbours(h, w, target.device, target.dtype)
        invalid = valid_offsets[None, None] <= 0
        temperature = F.softplus(self.log_temperature.clamp(-5.0, 5.0)) + 0.03

        # target query -> candidate source local transport.
        corr_t2s = (target_norm[:, None, None] * candidate_neighbours_norm).sum(dim=3)
        scaled_t2s = (corr_t2s / temperature).masked_fill(invalid, -1.0e4)
        plan_t2s = torch.softmax(scaled_t2s, dim=2)
        target_to_source = (plan_t2s[:, :, :, None] * candidate_neighbours).sum(dim=2)

        # candidate query -> target source local transport.
        corr_s2t = (candidate_norm[:, :, None] * target_neighbours_norm[:, None]).sum(dim=3)
        scaled_s2t = (corr_s2t / temperature).masked_fill(invalid, -1.0e4)
        plan_s2t = torch.softmax(scaled_s2t, dim=2)
        source_to_target = (plan_s2t[:, :, :, None] * target_neighbours[:, None]).sum(dim=2)

        target_expand = target[:, None].expand(-1, k, -1, -1, -1)
        feature_cycle = 0.5 * (
            F.cosine_similarity(source_to_target, target_expand, dim=2, eps=1e-6)
            + F.cosine_similarity(target_to_source, candidates, dim=2, eps=1e-6)
        )
        # A true local round trip: target->source offset d must be followed by
        # source->target offset -d. This is a probability, not a same-map proxy.
        round_trip = (plan_t2s * plan_s2t[:, :, _OPPOSITE]).sum(dim=2)
        cycle = (0.5 * round_trip + 0.25 * (feature_cycle + 1.0)).clamp(0.0, 1.0)
        # Convert the signed cosine to a stable [0,1] visibility/transport gate.
        visibility = torch.sigmoid(5.0 * (cycle - self.visibility_bias)).unsqueeze(2)
        delta = target_to_source - candidates
        residual = torch.tanh(self.residual_scale[level_index]) * visibility * delta
        updated = candidates + residual

        entropy = -(plan_t2s.clamp_min(1e-8) * plan_t2s.clamp_min(1e-8).log()).sum(dim=2) / torch.log(
            torch.tensor(9.0, device=plan_t2s.device, dtype=plan_t2s.dtype)
        )
        mass = plan_t2s.max(dim=2).values
        valid = mask[:, :, None, None].to(cycle.dtype)
        stats = {
            "cycle": cycle,
            "visibility": visibility.squeeze(2),
            "entropy": entropy,
            "mass": mass,
            "delta_norm": delta.abs().mean(dim=(2, 3, 4)),
            "residual_norm": residual.abs().mean(dim=(2, 3, 4)),
            "valid": valid,
        }
        return updated * mask[:, :, None, None, None].to(updated.dtype), stats

    def forward(self, target: Mapping[str, Tensor], candidates: Mapping[str, Tensor], mask: Tensor):
        self._validate(target, candidates, mask)
        updated: Dict[str, Tensor] = {}
        stats: Dict[str, Dict[str, Tensor]] = {}
        for index, level in enumerate(LEVELS):
            updated[level], stats[level] = self._one_level(target[level], candidates[level], mask, index)
        return dict(target), updated, stats


def transport_contract_loss(stats: Mapping[str, Mapping[str, Tensor]], batch: Mapping[str, object]) -> Tuple[Tensor, Dict[str, float]]:
    """Weak pair-label transport contract; no dense correspondence is assumed."""
    valid = batch["supervision_valid"]
    positive = batch["positive_mask"]
    known_negative = batch["known_negative_mask"]
    terms = []
    cycle_gaps, mass_gaps, positive_entropy = [], [], []
    for level in LEVELS:
        current = stats[level]
        cycle = current["cycle"]
        mass = current["mass"]
        entropy = current["entropy"]
        positive_cells = (positive[:, :, None, None] & valid[:, None, None, None]).expand_as(cycle)
        negative_cells = (known_negative[:, :, None, None] & valid[:, None, None, None]).expand_as(cycle)
        if bool(positive_cells.any()) and bool(negative_cells.any()):
            pos_cycle = cycle.masked_select(positive_cells).mean()
            neg_cycle = cycle.masked_select(negative_cells).mean()
            pos_mass = mass.masked_select(positive_cells).mean()
            neg_mass = mass.masked_select(negative_cells).mean()
            gap_cycle = torch.relu(0.05 - pos_cycle + neg_cycle)
            gap_mass = torch.relu(0.02 - pos_mass + neg_mass)
            terms.extend((gap_cycle, gap_mass))
            cycle_gaps.append(pos_cycle - neg_cycle)
            mass_gaps.append(pos_mass - neg_mass)
        if bool(positive_cells.any()):
            pos_entropy = entropy.masked_select(positive_cells).mean()
            # A uniform 3x3 plan has entropy 1; do not force a point estimate,
            # but reject a completely uniform positive transport.
            terms.append(torch.relu(pos_entropy - 0.97))
            positive_entropy.append(pos_entropy)
    if not terms:
        zero = next(iter(stats.values()))["cycle"].sum() * 0.0
        return zero, {"cycle_gap": 0.0, "mass_gap": 0.0, "positive_entropy": 0.0}
    loss = torch.stack(terms).mean()
    return loss, {
        "cycle_gap": float(torch.stack(cycle_gaps).mean().detach().cpu()) if cycle_gaps else 0.0,
        "mass_gap": float(torch.stack(mass_gaps).mean().detach().cpu()) if mass_gaps else 0.0,
        "positive_entropy": float(torch.stack(positive_entropy).mean().detach().cpu()) if positive_entropy else 0.0,
    }
