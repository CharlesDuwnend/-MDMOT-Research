"""Hierarchical action head for imbalanced UAVDT coverage actions.

The action space has a natural factorization: foreground versus clutter, then
uncovered versus covered foreground.  This avoids forcing the rare covered
foreground class to compete directly with the much more frequent clutter
class in one flat softmax.
"""
import numpy as np


class HierarchicalActionAnchor:
    """Expose a sklearn-like three-class posterior from two binary heads."""

    classes_ = np.asarray([0, 1, 2], dtype=int)

    def __init__(self, foreground_model, coverage_model):
        self.foreground_model = foreground_model
        self.coverage_model = coverage_model

    @staticmethod
    def _positive(model, x):
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError('hierarchical head is missing positive class')
        p = np.asarray(model.predict_proba(x), dtype=float)
        return p[:, classes.index(1)]

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        p_foreground = np.clip(self._positive(self.foreground_model, x), 0., 1.)
        p_covered = np.clip(self._positive(self.coverage_model, x), 0., 1.)
        p = np.column_stack((
            p_foreground * (1. - p_covered),
            p_foreground * p_covered,
            1. - p_foreground,
        ))
        return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)


def class_balanced_weights(base, labels, exponent=0.5):
    """Balance class mass while retaining within-class causal group weights.

    ``exponent=0`` recovers the audited causal weights; 0.5 is a conservative
    square-root correction; 1.0 gives equal total mass to every class.
    """
    base = np.asarray(base, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(base) != len(labels):
        raise ValueError('weight/label length mismatch')
    present = np.unique(labels)
    masses = {int(c): float(base[labels == c].sum()) for c in present}
    if any(v <= 0. for v in masses.values()):
        raise ValueError('class has no positive training mass')
    target = 1. / len(present)
    factors = {c: (target / mass) ** float(exponent)
               for c, mass in masses.items()}
    out = base * np.asarray([factors[int(c)] for c in labels], dtype=float)
    return out * (base.sum() / max(out.sum(), 1e-12))
