"""Compare corrected real seams against immutable pre-factorial runtime code.

The original reference consumes the already verified detection identities.
It runs on an isolated same-state clone and cannot commit host track updates.
"""
import copy
import hashlib
import importlib.util
import pickle
from pathlib import Path

import numpy as np


SOURCES = {
    True: Path('/home/chenhc/leaf_v5_evidence_completion_20261006_v1/experiment/belief/runtime.py'),
    False: Path('/home/chenhc/leaf_v5_evidence_completion_20261006_v1/experiment/'
                'internal_path_ablation_20261007_v1/belief/runtime.py'),
}
_references = {}


def original_runtime(cost_enabled):
    if cost_enabled not in _references:
        spec = importlib.util.spec_from_file_location(
            '_leaf_immutable_reference_%s' % int(cost_enabled), str(SOURCES[cost_enabled]))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _references[cost_enabled] = module
    return _references[cost_enabled]


def snapshot(tracker):
    # These fields could otherwise be changed by flushing or a shared counter.
    return (dict(tracker.counts), dict(tracker.module_seconds), tracker.stage,
            pickle.dumps(tracker.observed, protocol=4),
            tuple((k, np.asarray(v).dtype.str, np.asarray(v).shape,
                   np.asarray(v).tobytes()) for k, v in sorted(tracker.beliefs.items())),
            dict(tracker.det_ordinals), len(tracker.pending),
            tracker.responsibility_sum, tracker.belief_change_l1)


def assignment_key(result):
    return (tuple(map(tuple, np.asarray(result[0], int).reshape(-1, 2))),
            tuple(np.asarray(result[1], int)), tuple(np.asarray(result[2], int)))


def oracle_for_seam(tracker, trks, dets, thresh, fuse_score, iou_only):
    assert not tracker.pending, 'REFERENCE_ORACLE_REQUIRES_FLUSHED_PENDING'
    assert tracker.cost_factorial
    assert ((tracker.assignment_mask_enabled and tracker.semantic_penalty_enabled) or
            (not tracker.assignment_mask_enabled and not tracker.semantic_penalty_enabled))
    assert not tracker.clutter_match_policy and tracker.association_edge_model is None
    before = snapshot(tracker)
    reference = original_runtime(bool(tracker.semantic_penalty_enabled))
    clone = object.__new__(reference.BeliefTracker)
    clone.__dict__ = tracker.__dict__.copy()
    for key, value in tracker.__dict__.items():
        if isinstance(value, (dict, list, set)):
            clone.__dict__[key] = copy.copy(value)
    clone.pending = []
    clone.observed = copy.deepcopy(tracker.observed)
    clone.beliefs = {k: np.asarray(v).copy() for k, v in tracker.beliefs.items()}
    clone.model = copy.copy(tracker.model)
    # Prediction models are small CPU heads. Isolate even instrumentation on a
    # head, rather than relying solely on sklearn predict_proba being read-only.
    clone.model.observation = copy.deepcopy(tracker.model.observation)
    clone.model.responsibility_model = copy.deepcopy(tracker.model.responsibility_model)
    clone.detection_identity_matches = tracker.detection_identity_matches
    # Deep-copy the complete objects exposed to the association operator,
    # preserving aliases between the two lists while protecting their features
    # and Kalman states. Do not copy the unrelated removed-track history.
    clone_trks, clone_dets = copy.deepcopy((trks, dets))
    aliases = {id(original): copied
               for originals, copies in [(trks, clone_trks), (dets, clone_dets)]
               for original, copied in zip(originals, copies)}
    clone.det_ordinals = tracker.det_ordinals.copy()
    for original, copied in zip(dets, clone_dets):
        del clone.det_ordinals[id(original)]
        clone.det_ordinals[id(copied)] = tracker.det_ordinals[id(original)]
    for key in ['tracked_stracks', 'lost_stracks', 'removed_stracks']:
        if hasattr(clone, key):
            setattr(clone, key, [aliases.get(id(t), t) for t in getattr(clone, key)])
    clone.stage = tracker.stage - 1  # original associate increments it itself
    clone.src_cost_enabled = bool(tracker.semantic_penalty_enabled)
    clone.src_state_enabled = True
    clone.src_update_enabled = True
    clone.use_src = True
    clone.src_shadow_only = False
    clone.hard_update = False
    result = clone.associate(clone_trks, clone_dets, thresh, fuse_score, iou_only)
    assert snapshot(tracker) == before, 'REFERENCE_ORACLE_MUTATED_ACTUAL_STATE'
    return {'assignment': result, 'pending': clone.pending,
            'stage': clone.stage,
            'track_identity': {id(copied): original
                               for original, copied in zip(trks, clone_trks)},
            'source': str(SOURCES[bool(tracker.semantic_penalty_enabled)])}


def verify_reference(tracker, actual_result, reference):
    assert assignment_key(actual_result) == assignment_key(reference['assignment']), \
        'CORRECTED_REFERENCE_ASSIGNMENT_MISMATCH'
    assert tracker.stage == reference['stage'], 'CORRECTED_REFERENCE_STAGE_MISMATCH'
    expected = reference['pending']
    assert len(tracker.pending) == len(expected), 'CORRECTED_REFERENCE_PENDING_COUNT_MISMATCH'
    for actual, oracle in zip(tracker.pending, expected):
        a_track, a_obs, a_ratio, a_r = actual
        b_track, b_obs, b_ratio, b_r = oracle
        assert a_track is reference['track_identity'][id(b_track)], \
            'CORRECTED_REFERENCE_TRACK_IDENTITY_MISMATCH'
        assert a_obs == b_obs, 'CORRECTED_REFERENCE_OBSERVATION_MISMATCH'
        assert np.array_equal(a_ratio, b_ratio), 'CORRECTED_REFERENCE_RATIO_MISMATCH'
        assert a_r == b_r, 'CORRECTED_REFERENCE_RESPONSIBILITY_MISMATCH'
    tracker.counts['corrected_reference_seams_verified'] += 1
