import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from belief import runtime
from belief.runtime import BeliefTracker, PATH_ARMS


class InternalPathContract(unittest.TestCase):
    def test_path_arm_switches_are_explicit(self):
        self.assertEqual(
            PATH_ARMS,
            {
                'path_no_src': {
                    'src_cost_enabled': False,
                    'src_update_enabled': False,
                    'use_egia': True,
                    'src_shadow_only': False,
                },
                'path_update_only': {
                    'src_cost_enabled': False,
                    'src_update_enabled': True,
                    'use_egia': True,
                    'src_shadow_only': False,
                },
                'path_cost_only': {
                    'src_cost_enabled': True,
                    'src_update_enabled': False,
                    'use_egia': True,
                    'src_shadow_only': False,
                },
                'path_full': {
                    'src_cost_enabled': True,
                    'src_update_enabled': True,
                    'use_egia': True,
                    'src_shadow_only': False,
                },
            },
        )

    @staticmethod
    def tracker_for_flush(update_enabled):
        tracker = BeliefTracker.__new__(BeliefTracker)
        tracker.frame_id = 7
        tracker.src_update_enabled = update_enabled
        tracker.hard_update = False
        tracker.pending = []
        tracker.observed = {}
        tracker.beliefs = {11: np.asarray([0.5, 0.5], dtype=float)}
        tracker.counts = Counter()
        tracker.module_seconds = Counter()
        tracker.responsibility_sum = 0.0
        tracker.belief_change_l1 = 0.0
        track = SimpleNamespace(track_id=11, frame_id=7)
        observation = {'frame': 7, 'detection_ordinal': 0}
        tracker.pending.append((track, observation, np.asarray([2.0, 0.5]), 0.5))
        return tracker

    def test_update_only_commits_a_semantic_update(self):
        tracker = self.tracker_for_flush(update_enabled=True)
        BeliefTracker._flush_committed(tracker)
        self.assertEqual(tracker.counts['belief_updates'], 1)
        self.assertEqual(tracker.counts['committed_observations'], 1)
        self.assertEqual(tracker.observed[11], {'frame': 7, 'detection_ordinal': 0})
        self.assertFalse(np.allclose(tracker.beliefs[11], [0.5, 0.5]))
        self.assertEqual(tracker.pending, [])

    def test_cost_only_commits_observation_without_update(self):
        tracker = self.tracker_for_flush(update_enabled=False)
        before = tracker.beliefs[11].copy()
        BeliefTracker._flush_committed(tracker)
        self.assertEqual(tracker.counts['committed_observations'], 1)
        self.assertEqual(tracker.counts.get('belief_updates', 0), 0)
        np.testing.assert_array_equal(tracker.beliefs[11], before)
        self.assertEqual(tracker.pending, [])

    @staticmethod
    def report_tracker(arm):
        config = PATH_ARMS[arm]
        tracker = BeliefTracker.__new__(BeliefTracker)
        tracker.arm = arm
        tracker.src_cost_enabled = config['src_cost_enabled']
        tracker.src_update_enabled = config['src_update_enabled']
        tracker.src_state_enabled = tracker.src_cost_enabled or tracker.src_update_enabled
        tracker.src_shadow_only = config['src_shadow_only']
        tracker.counts = Counter()
        tracker.module_seconds = Counter()
        tracker.responsibility_sum = 0.0
        tracker.belief_change_l1 = 0.0
        tracker.beliefs = {}
        tracker.observed = {}
        tracker.pure_v5_policy = False
        tracker.egia_model_payload = object()
        tracker.egia_fusion_policy = False
        tracker.shuffle_pairs = False
        tracker.hard_update = False
        return tracker

    def test_report_exposes_assignment_path(self):
        expected = {
            'path_no_src': 'native',
            'path_update_only': 'native',
            'path_cost_only': 'src_cost',
            'path_full': 'src_cost',
        }
        for arm, assignment_source in expected.items():
            with self.subTest(arm=arm):
                self.assertEqual(
                    BeliefTracker.runtime_report(self.report_tracker(arm))[
                    'assignment_source'],
                    assignment_source,
                )

    @staticmethod
    def association_tracker(arm, track, detection):
        config = PATH_ARMS[arm]
        tracker = BeliefTracker.__new__(BeliefTracker)
        tracker.arm = arm
        tracker.frame_id = 7
        tracker.stage = 0
        tracker.src_cost_enabled = config['src_cost_enabled']
        tracker.src_update_enabled = config['src_update_enabled']
        tracker.src_state_enabled = tracker.src_cost_enabled or tracker.src_update_enabled
        tracker.use_egia = config['use_egia']
        tracker.src_shadow_only = config['src_shadow_only']
        tracker.shuffle_pairs = False
        tracker.hard_update = False
        tracker.det_ordinals = {id(detection): 0}
        tracker.frame_observations = {0: {
            'bbox_tlwh': detection._tlwh.tolist(),
            'predicted_class': detection.cls, 'score': detection.score,
            'class_confidence': detection.semantic_score,
        }}
        tracker.pending = []
        tracker.observed = {}
        tracker.beliefs = {int(track.track_id): np.asarray([0.9, 0.1])}
        tracker.counts = Counter()
        tracker.module_seconds = Counter()
        tracker.responsibility_sum = 0.0
        tracker.belief_change_l1 = 0.0
        tracker.clutter_match_policy = False
        tracker.association_edge_model = None
        tracker.association_edge_lambda = 0.15
        tracker.model = SimpleNamespace(
            prior=np.asarray([0.5, 0.5]),
            observation_probabilities=lambda observations: np.asarray([[0.1, 0.9]]),
            responsibilities=lambda features: np.asarray([0.5]),
        )
        tracker.det_record = lambda det: {'frame': 7, 'detection_ordinal': 0}
        tracker._base_cost = lambda *args: (
            np.asarray([[0.2]]), np.asarray([[0.3]]), np.asarray([[0.2]])
        )
        return tracker

    def test_association_path_counters_and_assignment_source(self):
        track = SimpleNamespace(track_id=11, frame_id=7)
        detection = SimpleNamespace(tlwh=np.asarray([1., 2., 3., 4.]),
                                    _tlwh=np.asarray([1., 2., 3., 4.]),
                                    cls=0, score=.9, semantic_score=.9)
        pair = (np.asarray([[0, 0]], dtype=int),
                np.asarray([], dtype=int), np.asarray([], dtype=int))
        no_pair = (np.empty((0, 2), dtype=int),
                   np.asarray([0], dtype=int), np.asarray([0], dtype=int))

        for arm in PATH_ARMS:
            with self.subTest(arm=arm):
                tracker = self.association_tracker(arm, track, detection)
                assignment_calls = []

                def assignment(cost, thresh):
                    assignment_calls.append(cost.copy())
                    # The first call is the native control.  For a cost path,
                    # make the second call's outcome explicit so the test
                    # exercises both assignment branches without relying on
                    # a particular frozen model score.
                    if len(assignment_calls) == 1:
                        return pair
                    return pair if tracker.src_update_enabled else no_pair

                patches = [
                    patch.object(runtime.matching, 'linear_assignment', side_effect=assignment),
                    patch.object(runtime.matching,
                                 'continuous_human_vehicle_semantic_cost',
                                 side_effect=lambda cost, tracks, detections: (cost.copy(), None)),
                    patch.object(runtime, 'responsibility_features', return_value=np.asarray([0.])),
                ]
                if arm in ('path_no_src', 'path_update_only'):
                    patches.append(patch.object(runtime.U2MOTTracker,
                                                'associate', return_value=pair))
                with patches[0], patches[1], patches[2]:
                    if arm in ('path_no_src', 'path_update_only'):
                        with patches[3]:
                            result = BeliefTracker.associate(
                                tracker, [track], [detection], 0.5)
                    else:
                        result = BeliefTracker.associate(
                            tracker, [track], [detection], 0.5)
                counts = tracker.counts
                if arm == 'path_no_src':
                    self.assertEqual(counts.get('src_observation_probability_calls', 0), 0)
                    self.assertEqual(counts.get('src_cost_evaluation_calls', 0), 0)
                    self.assertEqual(counts.get('src_responsibility_calls', 0), 0)
                    np.testing.assert_array_equal(result[0], pair[0])
                elif arm == 'path_update_only':
                    self.assertEqual(counts['src_observation_probability_calls'], 1)
                    self.assertEqual(counts.get('src_cost_evaluation_calls', 0), 0)
                    self.assertEqual(counts.get('src_semantic_penalty_calls', 0), 0)
                    self.assertEqual(counts['src_responsibility_calls'], 1)
                    self.assertEqual(counts.get('committed_changed_vs_native_seams', 0), 0)
                    self.assertEqual(counts['native_assignment_control_seams'], 1)
                    np.testing.assert_array_equal(result[0], pair[0])
                elif arm == 'path_cost_only':
                    self.assertEqual(counts['src_observation_probability_calls'], 1)
                    self.assertEqual(counts['src_cost_evaluation_calls'], 1)
                    self.assertEqual(counts['src_semantic_penalty_calls'], 1)
                    self.assertEqual(counts.get('src_responsibility_calls', 0), 0)
                    self.assertEqual(counts.get('belief_updates', 0), 0)
                    np.testing.assert_array_equal(result[0], no_pair[0])
                else:
                    self.assertEqual(counts['src_observation_probability_calls'], 1)
                    self.assertEqual(counts['src_cost_evaluation_calls'], 1)
                    self.assertEqual(counts['src_semantic_penalty_calls'], 1)
                    self.assertEqual(counts['src_responsibility_calls'], 1)
                    np.testing.assert_array_equal(result[0], pair[0])


if __name__ == '__main__':
    unittest.main()
