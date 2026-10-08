"""Functional head deletion; no empirical-prior replacement or refitting.

Deleting FG collapses the decision to U/E (f=1). Deleting coverage collapses
it to U/C (c=0). These deliberately remove an action function, rather than
compare equivalent model parameterizations. The separate paired flat versus
hierarchical comparison retains all three actions and common fit weights.
"""
import numpy as np
from scipy.special import softmax
from belief.features import legacy_vector,source_vectors
from belief.hierarchical import HierarchicalActionAnchor
from belief.coherence import CoherentSourceModel


class DeletedHeadAnchor(HierarchicalActionAnchor):
    def __init__(self, original, remove):
        if remove not in ('foreground', 'coverage'):
            raise ValueError(remove)
        super().__init__(None if remove == 'foreground' else original.foreground_model,
                         None if remove == 'coverage' else original.coverage_model)
        self.remove = remove
        self.calls = {'foreground': 0, 'coverage': 0, 'predictions': 0}

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        self.calls['predictions'] += 1
        if self.foreground_model is None:
            f = np.ones(len(x))
        else:
            self.calls['foreground'] += 1
            f = np.clip(self._positive(self.foreground_model, x), 0., 1.)
        if self.coverage_model is None:
            c = np.zeros(len(x))
        else:
            self.calls['coverage'] += 1
            c = np.clip(self._positive(self.coverage_model, x), 0., 1.)
        p = np.column_stack((f * (1. - c), f * c, 1. - f))
        return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)


class StructuralSourceModel(CoherentSourceModel):
    """Keep source computation; prohibit epsilon revival of deleted actions."""
    def __init__(self, original, mode, forbidden_classes=()):
        super().__init__(original.anchor, original.geometry_model,
                         original.appearance_model, original.cue_prior, original.weight)
        self.structural_mode = mode
        self.forbidden_classes = tuple(forbidden_classes)
        self.structural_calls = {'events': 0, 'rows': 0, 'forbidden_nonzero': 0}

    def _restrict(self, p):
        if not self.forbidden_classes:
            return p  # exact original operator path, without a renormalization
        p = np.asarray(p, dtype=float).copy()
        p[..., list(self.forbidden_classes)] = 0.
        p /= np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
        self.structural_calls['forbidden_nonzero'] += int(
            np.count_nonzero(p[..., list(self.forbidden_classes)]))
        if not np.isfinite(p).all():
            raise ValueError('invalid structural posterior')
        return p

    def predict_event(self, event, shuffled=False, intervention_counts=None):
        self.structural_calls['events'] += 1
        if self.forbidden_classes:
            _,owners,ids=source_vectors(event,shuffle_pairs=shuffled,
                                       intervention_counts=intervention_counts)
            p=self.anchor.predict_proba(legacy_vector(event)[None])[0]
            evidence,_=self.log_evidence(owners)
            logits=np.log(np.clip(p,1e-12,1.))
            logits[1]+=self.weight*evidence
            logits[list(self.forbidden_classes)]=-np.inf
            return self._restrict(softmax(logits)),ids,evidence
        p, ids, evidence = super().predict_event(event, shuffled, intervention_counts)
        return self._restrict(p), ids, evidence

    def predict_rows(self, rows, shuffled=False):
        self.structural_calls['rows'] += len(rows)
        if self.forbidden_classes:
            x=np.stack([r['legacy'] for r in rows])
            p=self.anchor.predict_proba(x)
            evidence=np.array([self.log_evidence(r['shuffled' if shuffled else 'owners'])[0] for r in rows])
            logits=np.log(np.clip(p,1e-12,1.))
            logits[:,1]+=self.weight*evidence
            logits[:,list(self.forbidden_classes)]=-np.inf
            return self._restrict(softmax(logits,axis=-1)),evidence
        p, evidence = super().predict_rows(rows, shuffled)
        return self._restrict(p), evidence

    def structural_report(self):
        return dict(mode=self.structural_mode, forbidden_classes=list(self.forbidden_classes),
                    source_calls=dict(self.structural_calls),
                    head_calls=dict(getattr(self.anchor, 'calls', {})),
                    foreground_head_deleted=getattr(self.anchor, 'remove', None) == 'foreground',
                    coverage_head_deleted=getattr(self.anchor, 'remove', None) == 'coverage')


class WithoutSourceCues(StructuralSourceModel):
    """Physically remove both cue estimators and their agreement correction."""
    def __init__(self, original):
        super().__init__(original, 'delete_source_cues')
        self.geometry_model = None
        self.appearance_model = None
        self.weight = 0.
        self.cue_calls = 0

    def log_evidence(self, owners):
        self.cue_calls += 1
        n = len(owners)
        return 0., np.full(n, 1. / n) if n else np.zeros(0)

    def structural_report(self):
        report = super().structural_report()
        report.update(geometry_head_deleted=self.geometry_model is None,
                      appearance_head_deleted=self.appearance_model is None,
                      coherence_weight=self.weight, cue_estimator_calls=0,
                      bypassed_cue_evidence_calls=self.cue_calls)
        return report
