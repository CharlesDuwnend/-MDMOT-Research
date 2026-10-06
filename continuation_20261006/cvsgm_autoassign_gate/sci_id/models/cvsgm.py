"""Cross-view, scale-consistent gated modulation for paired identity maps.

The operation is deliberately placed after the shared C2/C3/C4 identity
pyramid and before RoIAlign.  A gate is computed from the paired target/source
maps at one scale, while the gate MLP and spatial residual convolution are
shared by P2/P3/P4.  The residual convolution is zero initialised, so the
module is an exact step-zero identity for a controlled B1 comparison.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


LEVELS = ("p2_id", "p3_id", "p4_id")


class CrossViewScaleGatedModulation(nn.Module):
    """Symmetric paired-map modulation shared across the three identity scales."""

    def __init__(self, channels: int, gate_hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = gate_hidden or max(16, channels // 2)
        self.channels = channels
        self.gate = nn.Sequential(
            nn.LayerNorm(channels * 3),
            nn.Linear(channels * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.spatial_residual = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.spatial_residual.weight)
        # Non-zero gain lets the zero residual learn on the first update while
        # retaining exact step-zero equivalence to the B1 identity pyramid.
        self.scale_gain = nn.Parameter(torch.full((len(LEVELS),), 0.1))
        self.scale_embedding = nn.Parameter(torch.zeros(len(LEVELS), channels))

    def _validate(self, target: Mapping[str, Tensor], source: Mapping[str, Tensor]) -> None:
        if tuple(target) != LEVELS or tuple(source) != LEVELS:
            raise ValueError("target/source maps must contain p2_id, p3_id and p4_id")
        for level in LEVELS:
            t, s = target[level], source[level]
            if t.ndim != 4 or s.shape != t.shape:
                raise ValueError("paired maps must share [B,C,H,W] shape per scale")
            if t.shape[1] != self.channels:
                raise ValueError("map channel count does not match modulation channels")

    def forward(
        self,
        target: Mapping[str, Tensor],
        source: Mapping[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        self._validate(target, source)
        out_target: Dict[str, Tensor] = {}
        out_source: Dict[str, Tensor] = {}
        for index, level in enumerate(LEVELS):
            t, s = target[level], source[level]
            t_stat = t.mean(dim=(-1, -2))
            s_stat = s.mean(dim=(-1, -2))
            delta_stat = (t - s).abs().mean(dim=(-1, -2))
            gate_input = torch.cat((t_stat, s_stat, delta_stat), dim=-1)
            gate_input = gate_input + torch.cat(
                (self.scale_embedding[index], self.scale_embedding[index], self.scale_embedding[index]),
                dim=0,
            )
            gate = self.gate(gate_input).unsqueeze(-1).unsqueeze(-1)
            t_delta = self.spatial_residual(t)
            s_delta = self.spatial_residual(s)
            gain = self.scale_gain[index]
            out_target[level] = t + gain * gate * t_delta
            out_source[level] = s + gain * gate * s_delta
        return out_target, out_source


def modulation_parameter_count(channels: int, gate_hidden: Optional[int] = None) -> int:
    """Return trainable parameter count for audit and matched-control reports."""
    return sum(parameter.numel() for parameter in CrossViewScaleGatedModulation(channels, gate_hidden).parameters())
