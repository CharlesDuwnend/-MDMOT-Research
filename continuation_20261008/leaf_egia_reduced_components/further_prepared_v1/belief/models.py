"""Small explicit latent-source models, not a hidden change of tracker policy."""
import numpy as np
import torch
from torch import nn


class SourceModel(nn.Module):
    """Three outcomes, with EXISTING marginalized over individual owners.

    Uniform conditional owner priors prevent duplicated pool cardinality from
    automatically increasing evidence. Source IDs are not features. Real
    duplicate tracker IDs must be deduplicated by the caller; separate tracks
    with unknown/common GT remain separate online hypotheses.
    """
    def __init__(self, candidate_dim=16, owner_dim=16, contextual_null=False):
        super().__init__()
        self.contextual_null = contextual_null
        self.unary = nn.Sequential(nn.Linear(candidate_dim+(16 if contextual_null else 0), 16), nn.Tanh(), nn.Linear(16, 2))
        self.owner = nn.Sequential(nn.Linear(candidate_dim+owner_dim, 32), nn.Tanh(),
                                   nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, candidate, owners, mask):
        expanded = candidate[:, None, :].expand(-1, owners.shape[1], -1)
        pair_input = torch.cat([expanded, owners], -1)
        latent = self.owner[:-1](pair_input)
        energy = self.owner[-1](latent).squeeze(-1)
        energy = energy.masked_fill(~mask, -1e9)
        if self.contextual_null:
            # v2: null hypotheses see the same posterior-weighted, paired
            # source context. v1 restricted NEW/CLUTTER odds to candidate-only
            # inputs, an observable fit-set underfitting limitation.
            responsibility = energy.softmax(-1) * mask.float()
            context = (responsibility[..., None] * latent).sum(1)
            unary = self.unary(torch.cat([candidate, context], -1))
        else:
            unary = self.unary(candidate)
        count = mask.sum(-1)
        existing = torch.logsumexp(energy, -1) - count.clamp(min=1).float().log()
        existing = existing.masked_fill(count == 0, -1e9)
        classes = torch.stack([unary[:, 0], existing, unary[:, 1]], -1)
        return classes, energy


class SummaryMLP(nn.Module):
    """Capacity control restricted to the original ten-feature information."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(10, 64), nn.Tanh(), nn.Linear(64, 32),
                                     nn.Tanh(), nn.Linear(32, 3))

    def forward(self, features):
        return self.network(features)


def outcome_loss(class_logits, labels):
    """-1 means known foreground with unresolved NEW-versus-EXISTING source."""
    if ((labels < -1) | (labels > 2)).any():
        raise ValueError('unsupported outcome label')
    log_prob = class_logits.log_softmax(-1)
    known_loss = -log_prob.gather(1, labels.clamp(min=0)[:, None]).squeeze(1)
    foreground_loss = -torch.logsumexp(log_prob[:, :2], -1)
    return torch.where(labels == -1, foreground_loss, known_loss)


def source_loss(class_logits, owner_energy, labels, compatible, owner_known,
                weights):
    class_loss = outcome_loss(class_logits, labels)
    # Partial conditional source labels: unknown owners are not negative labels.
    eligible = (labels == 1) & compatible.any(-1)
    if eligible.any():
        denom = torch.logsumexp(owner_energy.masked_fill(~owner_known, -1e9), -1)
        numer = torch.logsumexp(owner_energy.masked_fill(~compatible, -1e9), -1)
        class_loss = class_loss + torch.where(eligible, denom-numer, torch.zeros_like(denom))
    return (class_loss * weights).sum() / weights.sum().clamp(min=1e-9)


def semantic_update(prior, likelihood_ratio, responsibility):
    """A normalized contamination-mixture update for the committed observation.

    The no-information component has likelihood ratio 1. This is an assumed
    observation model; a learned responsibility is not an exact global JPDA
    posterior. No threshold, temporal decay or identity feature memory update
    is performed here.
    """
    prior = np.asarray(prior, float)
    ratio = np.asarray(likelihood_ratio, float)
    r = float(responsibility)
    if prior.shape != (2,) or ratio.shape != (2,) or not 0 <= r <= 1:
        raise ValueError('invalid semantic filter inputs')
    if not np.isfinite(prior).all() or not np.isfinite(ratio).all() or (ratio < 0).any():
        raise ValueError('invalid likelihood')
    posterior = prior * ((1-r) + r*ratio)
    total = posterior.sum()
    if total <= 0:
        raise ValueError('zero posterior mass')
    return posterior/total


def semantic_penalty(track_probability, likelihood_ratio):
    """Evidence against same-owner semantics, neutral at the training prior.

    The likelihood ratio is normalized against the population prior. A Bayes
    factor below one adds a bounded penalty on already legal host edges.
    """
    factor = np.asarray(track_probability) @ np.asarray(likelihood_ratio).T
    return np.clip(1-factor, 0., 1.)
