"""Cue-source co-reference evidence with an explicit independence control.

For n owner hypotheses and uniform owner priors, the Bayes factor comparing
one shared owner with independently selected geometry/appearance owners is
n * sum_i p_geom(i) p_app(i). Its log is exactly invariant to reordering whole
owners, zero with one owner or an uninformative cue, and changes when one cue
is permuted while all cue marginals are preserved.
"""
import numpy as np
from scipy.special import logsumexp, softmax
from belief.features import legacy_vector, source_vectors


class CoherentSourceModel:
    def __init__(self, anchor, geometry_model, appearance_model, cue_prior, weight):
        self.anchor=anchor
        self.geometry_model=geometry_model
        self.appearance_model=appearance_model
        self.cue_prior=float(cue_prior)
        self.weight=float(weight)
        if self.weight<0:raise ValueError('negative co-reference authority')

    def log_evidence(self, owners):
        owners=np.asarray(owners,float)
        if len(owners)<2:return 0.,np.zeros(len(owners))
        a=np.asarray(self.geometry_model.decision_function(owners[:,0:1]),float)
        b=np.asarray(self.appearance_model.decision_function(owners[:,1:2]),float)
        # Discriminative odds divided by the class prior are cue likelihood
        # ratios. Missing appearance is uninformative (log ratio zero).
        prior_odds=np.log(self.cue_prior/(1-self.cue_prior))
        a-=prior_odds;b-=prior_odds
        b[owners[:,2]==0]=0.
        posterior=a+b
        coherence=float(np.log(len(owners))+logsumexp(posterior)-logsumexp(a)-logsumexp(b))
        return coherence,softmax(posterior)

    def predict_rows(self, rows, shuffled=False):
        x=np.stack([r['legacy'] for r in rows])
        p=np.asarray(self.anchor.predict_proba(x),float)
        assert list(self.anchor.classes_)==[0,1,2]
        logits=np.log(np.clip(p,1e-12,1.))
        evidence=np.array([self.log_evidence(r['shuffled' if shuffled else 'owners'])[0] for r in rows])
        logits[:,1]+=self.weight*evidence
        return softmax(logits,axis=-1),evidence

    def predict_event(self,event,shuffled=False,intervention_counts=None):
        _,owners,ids=source_vectors(event,shuffle_pairs=shuffled,
                                    intervention_counts=intervention_counts)
        p=self.anchor.predict_proba(legacy_vector(event)[None])[0]
        logits=np.log(np.clip(p,1e-12,1.))
        evidence,_=self.log_evidence(owners)
        logits[1]+=self.weight*evidence
        return softmax(logits),ids,evidence
