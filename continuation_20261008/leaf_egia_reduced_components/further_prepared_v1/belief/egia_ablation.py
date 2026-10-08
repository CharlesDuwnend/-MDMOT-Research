"""Fit-only neutral posteriors and neutral-cue degeneration controls."""
import numpy as np
from scipy.special import softmax
from belief.coherence import CoherentSourceModel


class ConstantBinaryPrior:
    classes_ = np.asarray([0, 1], dtype=int)

    def __init__(self, probability):
        self.probability = float(probability)
        if not 0. < self.probability < 1.:
            raise ValueError('prior must retain both outcomes')
        self.n_features_in_ = 10

    def predict_proba(self, x):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != 10 or not np.isfinite(x).all():
            raise ValueError('invalid contextual features')
        return np.tile([1. - self.probability, self.probability], (len(x), 1))


class NeutralCueSourceModel(CoherentSourceModel):
    """A missing cue has unit likelihood ratio and exactly zero co-reference."""
    def log_evidence(self, owners):
        owners = np.asarray(owners, float)
        if len(owners) < 2:
            return 0., np.zeros(len(owners))
        prior_odds = np.log(self.cue_prior / (1. - self.cue_prior))
        if self.neutral_cue == 'geometry':
            remaining = np.asarray(self.appearance_model.decision_function(owners[:, 1:2]), float)
            remaining -= prior_odds
            remaining[owners[:, 2] == 0] = 0.
        elif self.neutral_cue == 'appearance':
            remaining = np.asarray(self.geometry_model.decision_function(owners[:, 0:1]), float)
            remaining -= prior_odds
        else:
            raise ValueError('unknown neutral cue')
        # log(n) + LSE(remaining) - LSE(zeros) - LSE(remaining) = 0.
        return 0., softmax(remaining)
