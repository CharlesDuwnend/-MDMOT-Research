"""CPU contracts for the frozen mask/penalty factorial and input ledger."""
from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from belief import runtime
from belief.online import OnlineBeliefTracker
from belief.runtime import BeliefTracker, FACTORIAL_ARMS
from yolox.tracker.u2mot_tracker import DefaultArgs


class ObservationModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features):
        return np.asarray([[.95, .05] if np.argmax(row[:10]) == 0 else [.05, .95]
                           for row in features])


class ResponsibilityModel:
    classes_ = np.asarray([0, 1])

    def __init__(self):
        self.inputs = []

    def predict_proba(self, features):
        self.inputs.append(np.asarray(features).copy())
        return np.tile([.25, .75], (len(features), 1))


def make_tracker(arm):
    args = DefaultArgs()
    args.continuous_semantic_reliability = True
    args.egia_model = '/dev/null'
    args.egia_capture_dir = ''
    responsibility = ResponsibilityModel()
    bundle = {
        'observation_model': ObservationModel(),
        'responsibility_model': responsibility,
        'group_prior': np.asarray([.5, .5]),
        'source_model': object(),
        'source_architecture': 'coverage_hierarchical_coherence_v5',
        'egia_fusion_policy': True, 'egia_fusion_weight': .4,
        'selective_birth_policy': True, 'selective_birth_margin': .2,
    }
    with patch.object(runtime.pickle, 'load', return_value={}):
        tracker = BeliefTracker(args, arm=arm, bundle=bundle)
    tracker.frame_id = 7
    tracker.stage = 0
    tracks = [SimpleNamespace(track_id=11, frame_id=6, cls=0,
                              tlbr=np.asarray([0., 0., 10., 10.])),
              SimpleNamespace(track_id=12, frame_id=6, cls=3,
                              tlbr=np.asarray([5., 0., 15., 10.]))]
    dets = [SimpleNamespace(cls=0, tlwh=np.asarray([0., 0., 10., 10.]),
                            tlbr=np.asarray([0., 0., 10., 10.])),
            SimpleNamespace(cls=3, tlwh=np.asarray([5., 0., 10., 10.]),
                            tlbr=np.asarray([5., 0., 15., 10.]))]
    tracker.beliefs = {11: np.asarray([.9, .1]), 12: np.asarray([.1, .9])}
    tracker.det_ordinals = {id(det): i for i, det in enumerate(dets)}
    tracker.det_record = lambda det: {
        'frame': 7, 'detection_ordinal': tracker.det_ordinals[id(det)],
        'predicted_class': det.cls, 'class_confidence': .9,
    }
    return tracker, tracks, dets, responsibility


GEOMETRY = np.asarray([[.05, .2], [.2, .05]])
APPEARANCE = np.asarray([[.1, .01], [.01, .1]])


class CostFactorContracts(unittest.TestCase):
    def run_seam(self, arm, geometry=GEOMETRY, appearance=APPEARANCE,
                 thresh=.8, iou_only=False, fuse_score=False):
        tracker, tracks, dets, responsibility = make_tracker(arm)
        solver_inputs = []
        original_lap = runtime.matching.linear_assignment

        def logged_lap(cost, thresh):
            solver_inputs.append(np.asarray(cost).copy())
            return original_lap(cost, thresh=thresh)

        with patch.object(runtime.matching, 'iou_distance',
                          side_effect=lambda *_: geometry.copy()), \
                patch.object(runtime.matching, 'embedding_distance',
                             side_effect=lambda *_: appearance.copy()), \
                patch.object(runtime.matching, 'linear_assignment', side_effect=logged_lap):
            result = tracker.associate(tracks, dets, thresh,
                                       fuse_score=fuse_score, iou_only=iou_only)
        return tracker, tracks, dets, responsibility, result, solver_inputs

    def test_real_constructor_preserves_state_and_frozen_context_in_every_cell(self):
        expected = {'fc_control': (False, False), 'fc_mask_only': (True, False),
                    'fc_penalty_only': (False, True), 'fc_full': (True, True)}
        for arm, factors in expected.items():
            with self.subTest(arm=arm):
                tracker, _, _, _ = make_tracker(arm)
                self.assertEqual((tracker.assignment_mask_enabled,
                                  tracker.semantic_penalty_enabled), factors)
                self.assertTrue(tracker.src_state_enabled)
                self.assertTrue(tracker.src_update_enabled)
                self.assertTrue(tracker.use_egia)
                self.assertFalse(tracker.src_shadow_only)
                self.assertEqual(tracker.egia_fusion_weight, .4)
                self.assertEqual(tracker.selective_birth_margin, .2)
                np.testing.assert_array_equal(tracker.super_cls,
                                              [0, 0, 2, 1, 1, 3, 4, 4, 3, 2])
        self.assertEqual(set(FACTORIAL_ARMS), set(expected))

    def test_each_factor_only_changes_its_assignment_operator(self):
        observed = {}
        for arm in FACTORIAL_ARMS:
            tracker, _, _, _, _, costs = self.run_seam(arm)
            native = np.minimum(GEOMETRY, APPEARANCE)
            masked = GEOMETRY.copy()
            assignment = masked if tracker.assignment_mask_enabled else native
            expected = assignment.copy()
            if tracker.semantic_penalty_enabled:
                penalty = runtime.semantic_penalty(np.asarray([[.9, .1], [.1, .9]]),
                                                    np.asarray([[.95, .05], [.05, .95]])/.5)
                expected += (1.-expected)*penalty
            np.testing.assert_array_equal(costs[0], native)
            np.testing.assert_array_equal(costs[-1], expected)
            observed[arm] = costs[-1]
            self.assertEqual(tracker.counts['src_semantic_penalty_calls'],
                             int(tracker.semantic_penalty_enabled))
            self.assertEqual(tracker.counts['factorial_mask_changed_edges'],
                             2 if tracker.assignment_mask_enabled else 0)
        np.testing.assert_array_equal(observed['fc_mask_only'], GEOMETRY)
        np.testing.assert_array_equal(observed['fc_control'],
                                      np.minimum(GEOMETRY, APPEARANCE))

    def test_all_cells_use_original_masked_reference_for_responsibility(self):
        for arm in FACTORIAL_ARMS:
            with self.subTest(arm=arm):
                tracker, _, _, model, result, _ = self.run_seam(arm)
                self.assertEqual(len(model.inputs), 1)
                pairs = np.asarray(result[0], int).reshape(-1, 2)
                expected = np.asarray([runtime.responsibility_features(
                    GEOMETRY, APPEARANCE, GEOMETRY, i, j, .8, False)
                    for i, j in pairs])
                np.testing.assert_array_equal(model.inputs[0], expected)
                self.assertEqual(tracker.counts['src_responsibility_rows'], len(pairs))
                np.testing.assert_array_equal(GEOMETRY, [[.05, .2], [.2, .05]])

    def test_off_off_returns_actual_native_assignment_and_counts_real_seam(self):
        tracker, _, _, _, result, costs = self.run_seam('fc_control')
        np.testing.assert_array_equal(result[0], [[0, 1], [1, 0]])
        # Actual host LAP plus independent reconstruction; no third SRC LAP.
        self.assertEqual(len(costs), 2)
        self.assertEqual(tracker.counts['native_assignment_control_seams'], 1)
        self.assertEqual(tracker.counts['factorial_native_assignment_calls'], 1)
        self.assertEqual(tracker.counts['factorial_assignment_verified_seams'], 1)
        self.assertEqual(tracker.counts['committed_changed_vs_native_seams'], 0)

    def test_mismatched_native_seam_stops_instead_of_faking_a_control(self):
        tracker, tracks, dets, _ = make_tracker('fc_control')
        wrong = (np.asarray([[0, 0], [1, 1]]), (), ())
        with patch.object(runtime.matching, 'iou_distance', return_value=GEOMETRY.copy()), \
                patch.object(runtime.matching, 'embedding_distance', return_value=APPEARANCE.copy()), \
                patch.object(runtime.U2MOTTracker, 'associate', return_value=wrong):
            with self.assertRaisesRegex(AssertionError, 'NATIVE_BASE_CONTRACT_MISMATCH'):
                tracker.associate(tracks, dets, .8)
        self.assertFalse(tracker.pending)

    def test_penalty_can_reject_but_never_create_a_threshold_illegal_edge(self):
        geometry = np.asarray([[.9, .7], [.7, .9]])
        appearance = np.ones((2, 2))
        for arm in FACTORIAL_ARMS:
            with self.subTest(arm=arm):
                _, _, _, _, result, costs = self.run_seam(
                    arm, geometry=geometry, appearance=appearance)
                native, cost = costs[0], costs[-1]
                self.assertFalse(np.any((native > .8) & (cost <= .8)))
                self.assertTrue(np.all(cost >= native))
                for i, j in np.asarray(result[0], int).reshape(-1, 2):
                    self.assertLessEqual(cost[i, j], .8)

    def test_invalid_negative_penalty_is_stopped(self):
        with patch.object(runtime, 'semantic_penalty', return_value=np.full((2, 2), -1.)):
            with self.assertRaisesRegex(AssertionError, 'CREATED_ILLEGAL_BASE_EDGE'):
                self.run_seam('fc_full')

    def test_iou_only_has_no_mask_effect_and_retains_reference_features(self):
        for arm in FACTORIAL_ARMS:
            tracker, _, _, model, result, costs = self.run_seam(arm, iou_only=True)
            self.assertEqual(tracker.counts['factorial_mask_changed_edges'], 0)
            np.testing.assert_array_equal(costs[0], GEOMETRY)
            for feature in model.inputs[0]:
                self.assertEqual(feature[-1], 1.)
            self.assertEqual(len(result[0]), 2)

    def test_empty_seams_are_native_and_never_call_semantic_models(self):
        for arm in FACTORIAL_ARMS:
            for empty_tracks in (True, False):
                with self.subTest(arm=arm, empty_tracks=empty_tracks):
                    tracker, tracks, dets, model = make_tracker(arm)
                    tracks, dets = ([], dets) if empty_tracks else (tracks, [])
                    result = tracker.associate(tracks, dets, .8)
                    self.assertEqual(len(result[0]), 0)
                    self.assertEqual(tracker.counts['factorial_empty_seams'], 1)
                    self.assertEqual(tracker.counts['factorial_native_assignment_calls'], 1)
                    self.assertEqual(tracker.counts['factorial_assignment_verified_seams'], 1)
                    self.assertEqual(tracker.counts['src_observation_probability_calls'], 0)
                    self.assertEqual(tracker.counts['src_responsibility_calls'], 0)
                    self.assertFalse(model.inputs)

    def test_update_waits_for_host_commit_and_preserves_frozen_formula(self):
        tracker, tracks, _, _, _, _ = self.run_seam('fc_full')
        before = {tid: value.copy() for tid, value in tracker.beliefs.items()}
        with self.assertRaisesRegex(AssertionError, 'BEFORE_HOST_COMMIT'):
            tracker._flush_committed()
        for tid, value in before.items():
            np.testing.assert_array_equal(tracker.beliefs[tid], value)
        pending = list(tracker.pending)
        for track in tracks:
            track.frame_id = 7
        tracker._flush_committed()
        for track, observation, ratio, responsibility in pending:
            tid = track.track_id
            np.testing.assert_array_equal(tracker.beliefs[tid],
                                          runtime.semantic_update(before[tid], ratio, responsibility))
            self.assertEqual(tracker.observed[tid], observation)
        self.assertEqual(tracker.counts['belief_updates'], len(pending))
        self.assertEqual(tracker.counts['committed_observations'], len(pending))
        self.assertFalse(tracker.pending)

    def test_unresolved_detection_ordinal_stops_before_model_or_assignment(self):
        tracker, tracks, dets, model = make_tracker('fc_full')
        tracker.det_ordinals = {}
        tracker.frame_observations = {
            0: {'bbox_tlwh': [100., 100., 10., 10.]},
            1: {'bbox_tlwh': [200., 200., 10., 10.]},
        }
        with self.assertRaisesRegex(AssertionError, 'DETECTION_ORDINAL_UNRESOLVED'):
            tracker.associate(tracks, dets, .8)
        self.assertFalse(model.inputs)
        self.assertFalse(tracker.pending)


class OnlineInputLedgerContracts(unittest.TestCase):
    @staticmethod
    def ledger_tracker():
        tracker = OnlineBeliefTracker.__new__(OnlineBeliefTracker)
        tracker.sequence_name = 'synthetic_sequence'
        return tracker

    def test_digest_is_order_dtype_shape_and_value_sensitive(self):
        detections = np.arange(14, dtype=np.float32).reshape(2, 7)
        embeddings = np.arange(8, dtype=np.float16).reshape(2, 4)
        digests = []
        for det, emb in ((detections, embeddings), (detections.copy(), embeddings.copy()),
                         (detections[::-1], embeddings),
                         (detections.astype(np.float64), embeddings),
                         (detections, embeddings.reshape(4, 2)),
                         (detections, embeddings + np.float16(1))):
            tracker = self.ledger_tracker()
            tracker._record_online_inputs(det, emb, (100, 200), (896, 1600), 1)
            digests.append(tracker._input_stream_hasher.hexdigest())
        self.assertEqual(digests[0], digests[1])
        self.assertEqual(len(set(digests)), 5)
        np.testing.assert_array_equal(detections, np.arange(14, dtype=np.float32).reshape(2, 7))
        np.testing.assert_array_equal(embeddings, np.arange(8, dtype=np.float16).reshape(2, 4))

    def test_per_frame_ledger_is_serializable_and_rejects_repeated_frames(self):
        import json
        detections = np.arange(14, dtype=np.float32).reshape(2, 7)
        embeddings = np.arange(8, dtype=np.float16).reshape(2, 4)
        tracker = self.ledger_tracker()
        tracker._record_online_inputs(detections, embeddings, (100, 200), (896, 1600), 1)
        tracker._record_online_inputs(detections, embeddings, (100, 200), (896, 1600), 2)
        records = json.loads(json.dumps(tracker._input_stream_records))
        self.assertEqual([record['frame'] for record in records], [1, 2])
        self.assertEqual(records[0]['embedding_dtype'], embeddings.dtype.str)
        self.assertEqual(records[0]['detector_sha256'], records[1]['detector_sha256'])
        self.assertNotEqual(records[0]['frame_input_sha256'], records[1]['frame_input_sha256'])
        self.assertEqual(records[-1]['rolling_input_sha256'], tracker._input_stream_hasher.hexdigest())
        with self.assertRaisesRegex(AssertionError, 'FRAME_ORDER'):
            tracker._record_online_inputs(detections, embeddings, (100, 200), (896, 1600), 2)

    def test_hash_ledger_leaves_host_inputs_and_output_order_unchanged(self):
        tracker = self.ledger_tracker()
        tracker.track_high_thresh = .5
        tracker.track_low_thresh = .1
        tracker.src_state_enabled = False
        tracker.observed = {}
        detections = np.asarray([[20, 5, 30, 15, .9, .8, 3],
                                 [1, 2, 5, 6, .8, .9, 0]], dtype=np.float32)
        embeddings = np.asarray([[3, 2], [1, 4]], dtype=np.float16)
        original_det, original_emb = detections.copy(), embeddings.copy()
        host_result = [object(), object()]

        def host_update(det, info, size, **kwargs):
            np.testing.assert_array_equal(det, original_det)
            np.testing.assert_array_equal(kwargs['embeddings'], original_emb)
            self.assertEqual(det.dtype, original_det.dtype)
            self.assertEqual(kwargs['embeddings'].dtype, original_emb.dtype)
            return host_result

        with patch.object(tracker, 'update', side_effect=host_update), \
                patch.object(tracker, '_flush_committed'):
            result = tracker.update_online(detections, (100, 200), (896, 1600), embeddings, 1)
        self.assertIs(result, host_result)
        np.testing.assert_array_equal(detections, original_det)
        np.testing.assert_array_equal(embeddings, original_emb)
        self.assertEqual(len(tracker._input_stream_records), 1)

    def test_birth_initialization_and_report_are_preserved_in_all_four_cells(self):
        import json
        detections = np.asarray([[0, 0, 10, 10, .9, .8, 0]], dtype=np.float32)
        embeddings = np.asarray([[1., 0.]], dtype=np.float16)
        for arm in FACTORIAL_ARMS:
            with self.subTest(arm=arm):
                base_tracker, _, _, _ = make_tracker(arm)
                tracker = self.ledger_tracker()
                tracker.__dict__.update(base_tracker.__dict__)
                tracker.beliefs = {}
                born = SimpleNamespace(track_id=101, start_frame=1)

                def host_update(*_, **kwargs):
                    tracker.frame_id = 1
                    tracker.new_candidates.append((born, tracker.frame_observations[0]))
                    return [born]

                with patch.object(tracker, 'update', side_effect=host_update):
                    result = tracker.update_online(detections, (100, 200),
                                                   (896, 1600), embeddings, 1)
                self.assertEqual(result, [born])
                np.testing.assert_array_equal(tracker.beliefs[101], [.95, .05])
                self.assertEqual(tracker.counts['belief_initializations'], 1)
                report = json.loads(json.dumps(tracker.runtime_report()))
                self.assertEqual(report['input_frames_hashed'], 1)
                self.assertEqual(report['input_stream_sha256'],
                                 report['input_records'][-1]['rolling_input_sha256'])
                self.assertTrue(report['cost_factorial']['src_state_birth_update_enabled'])


if __name__ == '__main__':
    unittest.main()
