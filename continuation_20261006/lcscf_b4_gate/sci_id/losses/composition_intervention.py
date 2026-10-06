"""Losses implementing the declared candidate-composition interventions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class InterventionLosses:
    total: Tensor
    factual: Tensor
    remove_positive_dustbin: Tensor
    remove_negative_control: Tensor
    hard_negative_margin: Tensor
    irrelevant_logit_consistency: Tensor
    irrelevant_map_consistency: Tensor

    @property
    def irrelevant_consistency(self) -> Tensor:
        """Backward-compatible aggregate for callers that only need one value."""
        return self.irrelevant_logit_consistency + self.irrelevant_map_consistency


class CompositionInterventionLoss(nn.Module):
    """K+1 factual/remove-positive CE plus explicit hard/irrelevant contracts."""

    def __init__(
        self,
        remove_weight: float = 0.30,
        hard_weight: float = 0.20,
        irrelevant_weight: float = 0.10,
        margin: float = 0.20,
        control_weight: float = 0.30,
        map_weight: float = 0.10,
    ) -> None:
        super().__init__()
        self.remove_weight = remove_weight
        self.control_weight = control_weight
        self.hard_weight = hard_weight
        self.irrelevant_weight = irrelevant_weight
        self.map_weight = map_weight
        self.margin = margin

    @staticmethod
    def assignment(logits: Tensor, target: Tensor) -> Tensor:
        """Single-index CE or multi-positive set negative log-likelihood.

        A boolean target has the same shape as ``logits`` and marks every
        assignment that is valid for the event.  This avoids treating a second
        local-track fragment mapped to the same train identity as a negative.
        """
        if target.ndim == 1 and target.dtype == torch.long:
            return F.cross_entropy(logits, target)
        if target.shape != logits.shape or target.dtype != torch.bool:
            raise ValueError("target must be long [B] or bool [B,K+1]")
        if not bool(torch.all(target.any(dim=1))):
            raise ValueError("every supervised event needs at least one positive")
        negative_inf = torch.finfo(logits.dtype).min
        positive_logits = logits.masked_fill(~target, negative_inf)
        return (torch.logsumexp(logits, dim=-1) - torch.logsumexp(positive_logits, dim=-1)).mean()

    def forward(
        self,
        factual_logits: Tensor,
        factual_target: Tensor,
        remove_logits: Optional[Tensor] = None,
        remove_target: Optional[Tensor] = None,
        control_logits: Optional[Tensor] = None,
        control_target: Optional[Tensor] = None,
        positive_logit: Optional[Tensor] = None,
        hard_negative_logit: Optional[Tensor] = None,
        retained_factual: Optional[Tensor] = None,
        retained_drop: Optional[Tensor] = None,
        retained_factual_maps: Optional[Tensor] = None,
        retained_drop_maps: Optional[Tensor] = None,
    ) -> InterventionLosses:
        factual = self.assignment(factual_logits, factual_target)
        zero = factual * 0.0
        removed = self.assignment(remove_logits, remove_target) if remove_logits is not None and remove_target is not None else zero
        control = self.assignment(control_logits, control_target) if control_logits is not None and control_target is not None else zero
        hard = F.relu(self.margin - (positive_logit - hard_negative_logit)).mean() if positive_logit is not None and hard_negative_logit is not None else zero
        irrelevant_logits = F.mse_loss(retained_factual, retained_drop) if retained_factual is not None and retained_drop is not None else zero
        irrelevant_maps = F.mse_loss(retained_factual_maps, retained_drop_maps) if retained_factual_maps is not None and retained_drop_maps is not None else zero
        total = (
            factual
            + self.remove_weight * removed
            + self.control_weight * control
            + self.hard_weight * hard
            + self.irrelevant_weight * irrelevant_logits
            + self.map_weight * irrelevant_maps
        )
        return InterventionLosses(
            total,
            factual,
            removed,
            control,
            hard,
            irrelevant_logits,
            irrelevant_maps,
        )
