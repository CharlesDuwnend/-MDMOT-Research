"""Nine independent CPU contracts for detector-row identity, in three rounds."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import MethodType
import unittest
from unittest.mock import patch

import numpy as np

from belief import runtime
from belief.online import OnlineBeliefTracker
from capture_belief_evidence import STrack
from test_cost_factors import make_tracker


def frame_rows(raw, frame=7):
    boxes = raw[:, :4].copy()
    boxes[:, 2:] -= boxes[:, :2]
    return {i: {'frame': frame, 'source_frame': frame,
                'detection_ordinal': i, 'bbox_tlwh': boxes[i].astype(float).tolist(),
                'predicted_class': int(row[6]), 'score': float(row[4]),
                'class_confidence': float(row[5])}
            for i, row in enumerate(raw)}


def real_detections(raw):
    return [STrack(STrack.tlbr_to_tlwh(row[:4]), row[4], row[6],
                   semantic_score=row[5]) for row in raw]


def identity_tracker(raw, arm='fc_full', online=False):
    tracker, _, _, responsibility = make_tracker(arm)
    del tracker.det_record  # Actual method, rather than the arithmetic-test mock.
    tracker.det_ordinals = {}
    tracker.frame_observations = frame_rows(raw)
    tracker.sequence_name = 'cpu_identity'
    tracker.source_frame = 7
    tracker.pending = []
    tracker.beliefs = {}
    tracker.observed = {}
    tracker.track_high_thresh = .5
    if online:
        result = OnlineBeliefTracker.__new__(OnlineBeliefTracker)
        result.__dict__.update(tracker.__dict__)
        result.frame_id = 0
        return result, responsibility
    return tracker, responsibility


LOW_FIRST = np.asarray([[1, 2, 11, 12, .4, .9, 0],
                        [1, 2, 11, 12, .9, .8, 3]], dtype=np.float16)


class InputIdentityContracts(unittest.TestCase):
    def setUp(self):
        # Identity is checked against raw inputs, separately from the frozen
        # seam oracle. Small heads and CPU fixtures are not reference evidence.
        oracle = patch('belief.reference_oracle.oracle_for_seam', return_value=None)
        oracle.start()
        self.addCleanup(oracle.stop)

    def assert_record(self, tracker, det, raw, ordinal):
        got = tracker.det_record(det)
        expected = frame_rows(raw)[ordinal]
        self.assertEqual(got['detection_ordinal'], ordinal)
        for field in ('bbox_tlwh', 'predicted_class', 'score', 'class_confidence'):
            self.assertEqual(got[field], expected[field])
        self.assertEqual(got['predicted_class'], int(det.cls))
        self.assertEqual(got['score'], float(det.score))
        self.assertEqual(got['class_confidence'], float(det.semantic_score))

    def test_round1_1_cross_class_same_box_high_low_raw_order(self):
        for raw in (LOW_FIRST, LOW_FIRST[::-1].copy()):
            for classes in ((0, 3), (3, 0)):
                raw = raw.copy()
                raw[:, 6] = classes
                with self.subTest(rows=raw.tolist()):
                    tracker, model = identity_tracker(raw)
                    dets = real_detections(raw)
                    high, low, _ = tracker._split_detection_masks(raw[:, 4], raw[:, 5])
                    for mask in (high, low):
                        selected = np.flatnonzero(mask)
                        tracker.associate([], [dets[i] for i in selected], .8)
                        for i in selected:
                            self.assert_record(tracker, dets[i], raw, int(i))
                    self.assertEqual(len(set(tracker.det_ordinals.values())), 2)
                    self.assertFalse(model.inputs)

    def test_round1_2_same_class_scores_confidences_and_threshold_boundary(self):
        raw = np.asarray([[1, 2, 11, 12, .4, .9, 3],
                          [1, 2, 11, 12, .9, .8, 3],
                          [1, 2, 11, 12, .5, .7, 3],
                          [1, 2, 11, 12, .9, .6, 3]], dtype=np.float16)
        tracker, _ = identity_tracker(raw)
        dets = real_detections(raw)
        high, low, _ = tracker._split_detection_masks(raw[:, 4], raw[:, 5])
        self.assertEqual(np.flatnonzero(high).tolist(), [1, 3])
        self.assertEqual(np.flatnonzero(low).tolist(), [0, 2])
        for mask in (high, low):
            selected = np.flatnonzero(mask)
            tracker.associate([], [dets[i] for i in selected], .8)
            for i in selected:
                self.assert_record(tracker, dets[i], raw, int(i))

    def test_round1_3_exact_metadata_duplicates_are_deterministic_unused_ordinals(self):
        raw = np.tile(LOW_FIRST[1], (3, 1))
        for _ in range(3):
            tracker, model = identity_tracker(raw)
            tracker.frame_observations = dict(reversed(list(tracker.frame_observations.items())))
            dets = real_detections(raw)
            tracker.associate([], dets, .8)
            self.assertEqual([tracker.det_ordinals[id(d)] for d in dets], [0, 1, 2])
            for i, det in enumerate(dets):
                self.assert_record(tracker, det, raw, i)
            self.assertEqual(tracker.counts['src_observation_probability_calls'], 0)
            self.assertFalse(model.inputs)

    def test_round2_1_cached_identity_is_checked_by_seam_and_record(self):
        raw = LOW_FIRST[1:2].copy()
        for field, value in (('predicted_class', 0), ('score', .4),
                             ('class_confidence', .2), ('bbox_tlwh', [9, 2, 10, 10])):
            for entry in ('associate', 'det_record'):
                with self.subTest(field=field, entry=entry):
                    tracker, _ = identity_tracker(raw)
                    det = real_detections(raw)[0]
                    tracker.det_ordinals[id(det)] = 0
                    tracker.frame_observations[0][field] = value
                    with patch.object(tracker.model, 'observation_probabilities') as model, \
                            patch.object(runtime.U2MOTTracker, 'associate') as native:
                        with self.assertRaisesRegex(AssertionError, 'DETECTION_ORDINAL_(METADATA|GEOMETRY)_MISMATCH'):
                            if entry == 'associate':
                                tracker.associate([], [det], .8)
                            else:
                                tracker.det_record(det)
                        model.assert_not_called()
                        native.assert_not_called()

    def test_round2_2_unresolved_identity_has_no_geometry_or_arbitrary_fallback(self):
        raw = LOW_FIRST[1:2].copy()
        for field, value in (('predicted_class', 0), ('score', .4),
                             ('class_confidence', float('nan')),
                             ('bbox_tlwh', [1.00001, 2, 10, 10])):
            with self.subTest(field=field):
                tracker, _ = identity_tracker(raw)
                tracker.frame_observations[0][field] = value
                det = real_detections(raw)[0]
                with patch.object(tracker.model, 'observation_probabilities') as model, \
                        patch.object(runtime.U2MOTTracker, 'associate') as native:
                    with self.assertRaisesRegex(AssertionError, 'DETECTION_ORDINAL_UNRESOLVED'):
                        tracker.associate([], [det], .8)
                    model.assert_not_called()
                    native.assert_not_called()
        tracker, _ = identity_tracker(raw)
        first, second = real_detections(np.vstack([raw, raw]))
        tracker.associate([], [first], .8)
        with self.assertRaisesRegex(AssertionError, 'DETECTION_ORDINAL_UNRESOLVED'):
            tracker.associate([], [second], .8)

    def test_round2_3_bijection_reused_seam_and_online_frame_reset(self):
        raw = np.tile(LOW_FIRST[1], (2, 1))
        tracker, _ = identity_tracker(raw)
        dets = real_detections(raw)
        tracker.associate([], dets, .8)
        initial = tracker.det_ordinals.copy()
        tracker.associate([], [dets[1], dets[0]], .8)
        self.assertEqual(tracker.det_ordinals, initial)
        tracker.det_ordinals[id(dets[1])] = 0
        with self.assertRaisesRegex(AssertionError, 'DETECTION_ORDINAL_REUSED'):
            tracker.associate([], dets, .8)

        online, _ = identity_tracker(raw, online=True)
        seen = []
        def only_seams(this, incoming, info, size, **kwargs):
            this.frame_id += 1
            det = real_detections(incoming)[0]
            this.associate([], [det], .8)
            seen.append(this.det_record(det))
            return []
        online.update = MethodType(only_seams, online)
        second = raw[:1].copy()
        second[0, 6] = 0
        for frame, incoming in enumerate((raw[:1], second), 1):
            online.update_online(incoming, (20, 20), (20, 20), np.ones((1, 4)), frame)
        self.assertEqual([r['predicted_class'] for r in seen], [3, 0])
        self.assertEqual([r['detection_ordinal'] for r in seen], [0, 0])

    @staticmethod
    def online_with_small_heads(arm='fc_full'):
        tracker, responsibility = identity_tracker(LOW_FIRST, arm=arm, online=True)
        source_events = []
        def source(event, *args, **kwargs):
            source_events.append(copy.deepcopy(event))
            return np.asarray([.99, .005, .005]), []
        tracker.model.source_probabilities = source
        tracker._legacy_egia_probabilities = lambda *args: np.asarray([.99, .005, .005])
        return tracker, responsibility, source_events

    def test_round3_1_real_online_birth_source_and_postcommit_memory(self):
        tracker, _, sources = self.online_with_small_heads()
        embeddings = np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.]], dtype=np.float16)
        initial_raw, initial_emb = LOW_FIRST.copy(), embeddings.copy()
        result = tracker.update_online(LOW_FIRST, (20, 20), (20, 20), embeddings, 1)
        self.assertEqual(len(result), 1)
        tid = int(result[0].track_id)
        self.assertEqual(tracker.observed[tid]['predicted_class'], 3)
        self.assertEqual(tracker.observed[tid]['detection_ordinal'], 1)
        self.assertEqual(sources[0]['candidate']['predicted_class'], 3)
        self.assertEqual(sources[0]['candidate']['detection_ordinal'], 1)
        np.testing.assert_array_equal(tracker.beliefs[tid], [.05, .95])
        before = tracker.beliefs[tid].copy()
        actual_update = STrack.update
        committed = []
        def checked_update(track, det, *args, **kwargs):
            committed.append(int(track.track_id))
            self.assertEqual(tracker.observed[tid]['frame'], 1)
            np.testing.assert_array_equal(tracker.beliefs[tid], before)
            self.assertEqual(len(tracker.pending), 1)
            self.assertEqual(tracker.pending[0][1]['predicted_class'], 3)
            return actual_update(track, det, *args, **kwargs)
        with patch.object(STrack, 'update', side_effect=checked_update, autospec=True):
            tracker.update_online(LOW_FIRST, (20, 20), (20, 20), embeddings, 2)
        self.assertEqual(committed, [tid])
        self.assertEqual(tracker.observed[tid]['frame'], 2)
        self.assertEqual(tracker.observed[tid]['predicted_class'], 3)
        self.assertEqual(tracker.observed[tid]['detection_ordinal'], 1)
        self.assertEqual(tracker.counts['belief_updates'], 1)
        self.assertFalse(tracker.pending)
        np.testing.assert_array_equal(LOW_FIRST, initial_raw)
        np.testing.assert_array_equal(embeddings, initial_emb)

    def test_round3_2_all_factorial_cells_keep_hashes_and_native_control(self):
        raw = np.asarray([[1, 2, 11, 12, .9, .8, 3]], dtype=np.float16)
        embeddings = np.asarray([[1., 0., 0., 0.]], dtype=np.float16)
        reports = {}
        for arm in runtime.FACTORIAL_ARMS:
            tracker, model, _ = self.online_with_small_heads(arm)
            high, low = tracker.track_high_thresh, tracker.track_low_thresh
            outputs = []
            for frame in (1, 2):
                tracks = tracker.update_online(raw, (20, 20), (20, 20), embeddings, frame)
                outputs.append([(t.tlwh.tolist(), int(t.cls), float(t.score)) for t in tracks])
            report = tracker.runtime_report()
            self.assertEqual(tracker.track_high_thresh, high)
            self.assertEqual(tracker.track_low_thresh, low)
            np.testing.assert_array_equal(tracker.super_cls, [0, 0, 2, 1, 1, 3, 4, 4, 3, 2])
            self.assertEqual(len(model.inputs), 1)
            report['test_outputs'] = outputs
            reports[arm] = report
            if arm == 'fc_control':
                self.assertGreater(report['counts']['native_assignment_control_seams'], 0)
                self.assertEqual(report['counts']['src_semantic_penalty_calls'], 0)
        reference = reports['fc_full']
        for report in reports.values():
            self.assertEqual(report['input_frames_hashed'], 2)
            self.assertEqual(report['input_stream_sha256'], reference['input_stream_sha256'])
            self.assertEqual(report['input_records'], reference['input_records'])
            self.assertEqual(report['test_outputs'], reference['test_outputs'])

    def test_round3_3_nine_byte_verified_real_frames_all_detection_records(self):
        audit_path = Path('/home/chenhc/leaf_internal_mechanism_20261008_v1/review/'
                          'DETECTION_ORDINAL_AMBIGUITY_AUDIT_20261008.json')
        self.assertEqual(hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                         'e9d895097b4e0d1ffda63becd3bedc0daba878f8362021682ccafe16ab827ce3')
        audit = json.loads(audit_path.read_text())
        frames = audit['exact_frozen_code_CPU_operator_reproduction'][
            'all_nine_frames_actual_code_reproduction']['checked_byte_verified_real_frames']
        source_paths = {Path(s['npz_path']).stem: s for s in audit['source_artifacts']}
        caches = {}
        results = []
        for entry in frames:
            sequence, frame = entry['sequence'], entry['frame']
            with self.subTest(sequence=sequence, frame=frame):
                if sequence not in caches:
                    source = source_paths[sequence]
                    self.assertEqual(hashlib.sha256(Path(source['npz_path']).read_bytes()).hexdigest(),
                                     source['npz_SHA256'])
                    with np.load(source['npz_path']) as npz:
                        caches[sequence] = {k: npz[k] for k in ('frame_ids', 'offsets', 'detections',
                                                               'frame_height', 'frame_width')}
                cache = caches[sequence]
                position = int(np.flatnonzero(cache['frame_ids'] == frame)[0])
                start, stop = cache['offsets'][position:position+2]
                raw = cache['detections'][start:stop].copy()
                raw = raw[raw[:, 4].astype(np.float16)*raw[:, 5].astype(np.float16) >= np.float16(.09)]
                height, width = int(cache['frame_height'][0]), int(cache['frame_width'][0])
                scale = min(896/float(height), 1600/float(width))
                raw[:, :4] *= scale
                raw = raw.astype(np.float16).astype(np.float32)
                raw = raw[np.argsort(raw[:, 0])]
                self.assertEqual(OnlineBeliefTracker._array_input_digest(raw), entry['detector_SHA256'])
                boxes = raw[:, :4].copy()
                boxes /= scale
                boxes[:, 2:] -= boxes[:, :2]

                tracker, _, _ = self.online_with_small_heads()
                tracker.frame_id = frame - 1
                original_associate = tracker.associate
                checked = Counter()
                def checked_seam(trks, dets, *args, **kwargs):
                    result = original_associate(trks, dets, *args, **kwargs)
                    for det in dets:
                        row = tracker.det_record(det)
                        ordinal = row['detection_ordinal']
                        self.assertEqual(row['predicted_class'], int(det.cls))
                        self.assertEqual(row['score'], float(det.score))
                        self.assertEqual(row['class_confidence'], float(det.semantic_score))
                        np.testing.assert_array_equal(row['bbox_tlwh'], det._tlwh)
                        np.testing.assert_array_equal(row['bbox_tlwh'], boxes[ordinal].astype(float))
                        self.assertEqual(row['predicted_class'], int(raw[ordinal, 6]))
                        self.assertEqual(row['score'], float(raw[ordinal, 4]))
                        self.assertEqual(row['class_confidence'], float(raw[ordinal, 5]))
                        checked[ordinal] += 1
                    return result
                tracker.associate = checked_seam
                embeddings = np.tile(np.asarray([1., 0., 0., 0.], dtype=np.float16), (len(raw), 1))
                saved_raw, saved_emb = raw.copy(), embeddings.copy()
                tracker.update_online(raw, (height, width), (896, 1600), embeddings, frame)
                eligible = set(np.flatnonzero(raw[:, 4] > tracker.track_low_thresh).tolist())
                self.assertEqual(set(checked), eligible)
                for example in audit['examples']:
                    if example['sequence'] == sequence and example['frame'] == frame:
                        self.assertIn(example['host_detection_ordinal'], checked)
                np.testing.assert_array_equal(raw, saved_raw)
                np.testing.assert_array_equal(embeddings, saved_emb)
                self.assertEqual(tracker.runtime_report()['input_frames_hashed'], 1)
                results.append(dict(entry, eligible_detections_checked=len(checked),
                                    wrong_metadata_detections=0, dummy_embedding_and_heads=True))
        self.assertEqual(len(results), 9)
        type(self).real_frame_results = results


if __name__ == '__main__':
    unittest.main()
