"""Nine CPU contracts using real tracks, frozen heads and whole old methods.

No matching, detector-record, estimator or oracle method is mocked. The only
shared source change is the explicitly declared corrected Capture det_record.
Run this file directly to seal the nine checks in CORRECTED_REFERENCE_CPU.json.
"""
import ast
import copy
import hashlib
import inspect
import json
import os
import pickle
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMMUTABLE = Path('/home/chenhc/leaf_v5_evidence_completion_20261006_v1/experiment')
os.environ['BELIEF_HOST'] = str(ROOT / 'host')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(ROOT))

import numpy as np
from belief import runtime
from belief import reference_oracle as oracle
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.u2mot_tracker import DefaultArgs, STrack

CHECKS = []
SOURCE_HASHES = {
    str(IMMUTABLE / 'belief/runtime.py'):
        '731f6be2e71fa1ec8b537da7e8b13966d527492889fa474b7e67e2f3569ac200',
    str(IMMUTABLE / 'internal_path_ablation_20261007_v1/belief/runtime.py'):
        '619b4f354c475c117d09f769c0fa2968abd7bd05ed1b9a981831f352332c4d28',
    str(IMMUTABLE / 'host/yolox/tracker/u2mot_tracker.py'):
        'c658a5e110e6b40c8b83ea8f17e2104dcfd5c5054535bc65a1172d0a5b9349c6',
    str(ROOT / 'models/F00_full.pkl'):
        'ee903030e48f73fe87411b6e9aa60d52bbf2ee2d895aebfa8085d260779702ac',
    str(ROOT / 'models/summary_frozen.pkl'):
        'aa48c972a3ddbe4c5e4b82c5f2a64f9032978d7c15a863a5a3caefe67ac04ce8',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def finish(round_number, check, name, **evidence):
    CHECKS.append({'round': round_number, 'check': check, 'name': name,
                   'status': 'PASS', 'evidence': evidence})
    print('[round %d check %d] PASS %s' % (round_number, check, name), flush=True)


def raw_rows(raw, frame=7):
    boxes = raw[:, :4].copy()
    boxes[:, 2:] -= boxes[:, :2]
    return {i: {'frame': frame, 'source_frame': frame, 'detection_ordinal': i,
                'bbox_tlwh': boxes[i].astype(float).tolist(),
                'predicted_class': int(row[6]), 'score': float(row[4]),
                'class_confidence': float(row[5])}
            for i, row in enumerate(raw)}


RAW = np.asarray([[10, 20, 30, 60, .9, .8, 0],
                  [100, 20, 120, 60, .85, .75, 3]], dtype=np.float32)
DUPLICATE = np.asarray([[10, 20, 30, 60, .4, .9, 0],
                        [10, 20, 30, 60, .9, .8, 3]], dtype=np.float16)


def make(arm='fc_full', raw=RAW):
    args = DefaultArgs()
    args.continuous_semantic_reliability = True
    args.egia_model = str(ROOT / 'models/summary_frozen.pkl')
    args.egia_capture_dir = ''
    with (ROOT / 'models/F00_full.pkl').open('rb') as stream:
        bundle = pickle.load(stream)
    tracker = runtime.BeliefTracker(args, arm=arm, bundle=bundle)
    tracker.frame_id, tracker.stage, tracker.source_frame = 7, 0, 7
    tracker.sequence_name = 'cpu_corrected_reference'
    tracker._egia_frame_height, tracker._egia_frame_width = 200., 300.
    tracker.frame_observations = raw_rows(raw)
    dets, tracks = [], []
    for i, row in enumerate(raw):
        feature = np.eye(4, dtype=float)[i % 4].copy()
        det = STrack(STrack.tlbr_to_tlwh(row[:4]), row[4], row[6],
                     feat=feature.copy(), semantic_score=row[5])
        track = STrack(det._tlwh.copy(), row[4], row[6],
                       feat=feature.copy(), semantic_score=row[5])
        track.activate(tracker.kalman_filter, 6)
        track.track_id = 101 + i
        track.is_activated = True
        tracks.append(track)
        dets.append(det)
        tracker.beliefs[track.track_id] = tracker.model.prior.copy()
    tracker.tracked_stracks = tracks.copy()
    tracker.lost_stracks = [tracks[-1]] if tracks else []
    return tracker, tracks, dets


def bind_independently(tracker, dets):
    """Exact raw tuple checker independent of the implementation helper."""
    used = set()
    for det in dets:
        matches = [i for i, row in tracker.frame_observations.items()
                   if i not in used and
                   np.array_equal(det._tlwh, row['bbox_tlwh']) and
                   int(det.cls) == row['predicted_class'] and
                   float(det.score) == row['score'] and
                   float(det.semantic_score) == row['class_confidence']]
        assert matches, 'INDEPENDENT_RAW_IDENTITY_UNRESOLVED'
        ordinal = min(matches)
        tracker.det_ordinals[id(det)] = ordinal
        used.add(ordinal)


def independent_original(tracker, tracks, dets, threshold=.99,
                         fuse_score=False, iou_only=False):
    """Direct whole-source call, without using oracle_for_seam to make it."""
    cost = bool(tracker.semantic_penalty_enabled)
    original = oracle.original_runtime(cost)
    # GMC has no role in associate; exclude its external state explicitly.
    state = {k: v for k, v in tracker.__dict__.items() if k != 'gmc'}
    cloned_state, cloned_tracks, cloned_dets = copy.deepcopy((state, tracks, dets))
    clone = object.__new__(original.BeliefTracker)
    clone.__dict__ = cloned_state
    clone.gmc = None
    clone.detection_identity_matches = runtime.BeliefTracker.detection_identity_matches
    clone.use_src = clone.src_state_enabled = clone.src_update_enabled = True
    clone.src_cost_enabled = cost
    clone.src_shadow_only = clone.hard_update = False
    clone.det_ordinals = {}
    bind_independently(clone, cloned_dets)
    result = original.BeliefTracker.associate(
        clone, cloned_tracks, cloned_dets, threshold, fuse_score, iou_only)
    return result, clone, dict(zip(map(id, cloned_tracks), tracks))


def assert_whole(test, actual, expected, tracker, clone, identity):
    test.assertEqual(oracle.assignment_key(actual), oracle.assignment_key(expected))
    test.assertEqual(tracker.stage, clone.stage)
    test.assertEqual(len(tracker.pending), len(clone.pending))
    for (a_track, a_obs, a_ratio, a_r), (b_track, b_obs, b_ratio, b_r) in zip(
            tracker.pending, clone.pending):
        test.assertIs(a_track, identity[id(b_track)])
        test.assertEqual(a_obs, b_obs)
        np.testing.assert_array_equal(a_ratio, b_ratio)
        test.assertEqual(a_r, b_r)
        test.assertTrue(np.isfinite(a_ratio).all())
        test.assertTrue(0. <= a_r <= 1.)


def critical_state(tracker, tracks, dets):
    return (oracle.snapshot(tracker), digest([t.__dict__ for t in tracks]),
            digest([d.__dict__ for d in dets]), digest(tracker.model),
            digest(tracker.frame_observations),
            tuple(map(id, tracker.tracked_stracks)),
            tuple(map(id, tracker.lost_stracks)), BaseTrack._count)


def ast_method(path, class_name, method_name):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    return ast.dump(fn, include_attributes=False)


def ast_function(path, name):
    tree = ast.parse(Path(path).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(fn, include_attributes=False)


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Unsupported receipt value: ' + type(value).__name__)


def provider_evidence():
    evidence = []
    for cost in (True, False):
        module = oracle.original_runtime(cost)
        providers = {}
        for name in ('semantic_penalty', 'semantic_update', 'responsibility_features', 'semantic_features'):
            method = getattr(module, name)
            path = Path(inspect.getsourcefile(method)).resolve()
            providers[name] = {'path': str(path), 'sha256': sha(path),
                               'function_AST_sha256': hashlib.sha256(
                                   ast_function(path, name).encode()).hexdigest()}
        for name, method in [('host_associate', module.U2MOTTracker.associate),
                             ('matching_lap', module.matching.linear_assignment),
                             ('shared_corrected_det_record', module.BeliefTracker.det_record)]:
            path = Path(inspect.getsourcefile(method)).resolve()
            providers[name] = {'path': str(path), 'sha256': sha(path)}
        evidence.append({'cost_enabled': cost, 'runtime': str(module.__file__),
                         'runtime_sha256': sha(module.__file__),
                         'associate_code_file': module.BeliefTracker.associate.__code__.co_filename,
                         'MRO': [c.__module__ + '.' + c.__qualname__ for c in module.BeliefTracker.__mro__],
                         'providers': providers})
    return evidence


class CorrectedReferenceCPU(unittest.TestCase):
    def test_round1_1_frozen_sources_actual_providers_mro_and_head_methods(self):
        for path, expected in SOURCE_HASHES.items():
            self.assertEqual(sha(path), expected)
        for cost in (True, False):
            original = oracle.original_runtime(cost)
            self.assertEqual(Path(original.__file__), oracle.SOURCES[cost])
            self.assertEqual(Path(original.BeliefTracker.associate.__code__.co_filename),
                             oracle.SOURCES[cost])
            self.assertIs(original.BeliefTracker.__mro__[1], runtime.U2MOTTracker)
            self.assertIs(original.BeliefTracker.det_record, runtime.CaptureTracker.det_record)
            for method_name in ('observation_probabilities', 'responsibilities'):
                self.assertEqual(ast_method(ROOT / 'belief/runtime.py', 'BundleAdapter', method_name),
                                 ast_method(oracle.SOURCES[cost], 'BundleAdapter', method_name))
            for name, relative in [('semantic_penalty', 'belief/models.py'),
                                   ('semantic_update', 'belief/models.py'),
                                   ('responsibility_features', 'belief/features.py'),
                                   ('semantic_features', 'belief/features.py')]:
                provider = Path(inspect.getsourcefile(getattr(original, name))).resolve()
                if relative == 'belief/models.py':
                    self.assertEqual(sha(provider), sha(IMMUTABLE / relative))
                else:
                    self.assertEqual(ast_function(provider, name),
                                     ast_function(IMMUTABLE / relative, name))
                    current = ast.parse(provider.read_text())
                    old = ast.parse((IMMUTABLE / relative).read_text())
                    changed = [a.name for a, b in zip(current.body, old.body)
                               if ast.dump(a, include_attributes=False) !=
                               ast.dump(b, include_attributes=False)]
                    self.assertEqual(len(current.body), len(old.body))
                    self.assertEqual(changed, ['source_vectors'])
            self.assertEqual(sha(inspect.getsourcefile(original.U2MOTTracker)),
                             SOURCE_HASHES[str(IMMUTABLE / 'host/yolox/tracker/u2mot_tracker.py')])
            self.assertEqual(sha(inspect.getsourcefile(original.matching.linear_assignment)),
                             sha(IMMUTABLE / 'host/yolox/tracker/matching.py'))
        finish(1, 1, 'immutable providers and MRO', providers=provider_evidence(),
               shared_corrected_capture_declared=True,
               feature_file_difference='Only unused source_vectors pair_shuffle instrumentation differs; called functions AST identical',
               old_feature_file_sha256=sha(IMMUTABLE / 'belief/features.py'),
               current_feature_file_sha256=sha(ROOT / 'belief/features.py'))

    def test_round1_2_real_frozen_estimators_and_summary_prediction(self):
        tracker, tracks, dets = make()
        bind_independently(tracker, dets)
        self.assertTrue(type(tracker.model.observation).__module__.startswith('sklearn.'))
        self.assertTrue(type(tracker.model.responsibility_model).__module__.startswith('sklearn.'))
        self.assertEqual(type(tracker.model.source).__name__, 'CoherentSourceModel')
        probabilities = tracker.model.observation_probabilities([tracker.det_record(d) for d in dets])
        self.assertEqual(probabilities.shape, (2, 2))
        np.testing.assert_allclose(probabilities.sum(1), 1., atol=1e-15, rtol=0.)
        summary = tracker._legacy_egia_probabilities(dets[0], tracks, [])
        self.assertEqual(summary.shape, (3,))
        self.assertTrue(np.isfinite(summary).all())
        self.assertAlmostEqual(summary.sum(), 1.)
        self.assertFalse(runtime.torch.cuda.is_initialized())
        finish(1, 2, 'real F00 and summary CPU predictions',
               observation_probabilities=probabilities.tolist(), summary=summary.tolist(),
               cuda_initialized=False)

    def test_round1_3_C11_original_complete_associate_tuple_pending_stage(self):
        tracker, tracks, dets = make('fc_full')
        expected, clone, identity = independent_original(tracker, tracks, dets)
        result = tracker.associate(tracks, dets, .99)
        assert_whole(self, result, expected, tracker, clone, identity)
        self.assertGreater(len(tracker.pending), 0)
        self.assertEqual(tracker.counts['corrected_reference_seams_verified'], 1)
        finish(1, 3, 'C11 direct original whole-method equality',
               assignment=oracle.assignment_key(result), pending=len(tracker.pending), stage=tracker.stage)

    def test_round2_1_C00_original_update_only_tuple_pending_argument_variants(self):
        details = []
        for fuse_score, iou_only in ((False, False), (True, False), (False, True)):
            tracker, tracks, dets = make('fc_control')
            expected, clone, identity = independent_original(
                tracker, tracks, dets, fuse_score=fuse_score, iou_only=iou_only)
            result = tracker.associate(tracks, dets, .99, fuse_score=fuse_score, iou_only=iou_only)
            assert_whole(self, result, expected, tracker, clone, identity)
            self.assertGreater(len(tracker.pending), 0)
            self.assertEqual(tracker.counts['corrected_reference_seams_verified'], 1)
            details.append({'fuse_score': fuse_score, 'iou_only': iou_only,
                            'assignment': oracle.assignment_key(result), 'pending': len(tracker.pending)})
        finish(2, 1, 'C00 direct update-only whole-method variants', variants=details)

    def test_round2_2_empty_whole_seams_both_original_references(self):
        checks = 0
        for arm in ('fc_full', 'fc_control'):
            for empty in ('tracks', 'detections', 'both'):
                tracker, tracks, dets = make(arm)
                if empty in ('tracks', 'both'):
                    tracks = []
                if empty in ('detections', 'both'):
                    dets = []
                expected, clone, identity = independent_original(tracker, tracks, dets)
                result = tracker.associate(tracks, dets, .99)
                assert_whole(self, result, expected, tracker, clone, identity)
                self.assertFalse(tracker.pending)
                self.assertEqual(tracker.counts['corrected_reference_seams_verified'], 1)
                self.assertEqual(tracker.counts['src_observation_probability_calls'], 0)
                self.assertEqual(tracker.counts['src_responsibility_calls'], 0)
                checks += 1
        finish(2, 2, 'all empty combinations use complete original methods', cases=checks)

    def test_round2_3_same_box_cross_class_high_low_cascade_metadata(self):
        observations = []
        for arm in ('fc_full', 'fc_control'):
            for raw in (DUPLICATE, DUPLICATE[::-1].copy()):
                tracker, tracks, dets = make(arm, raw)
                high = [i for i, d in enumerate(dets) if d.score > tracker.track_high_thresh]
                low = [i for i, d in enumerate(dets) if d.score <= tracker.track_high_thresh]
                self.assertEqual((len(high), len(low)), (1, 1))
                for selected in (high, low):
                    subset = [dets[i] for i in selected]
                    expected, clone, identity = independent_original(tracker, tracks, subset, iou_only=True)
                    result = tracker.associate(tracks, subset, .99, iou_only=True)
                    assert_whole(self, result, expected, tracker, clone, identity)
                    for i in selected:
                        row = tracker.det_record(dets[i])
                        self.assertEqual(row['detection_ordinal'], i)
                        self.assertEqual(row['predicted_class'], int(raw[i, 6]))
                        self.assertEqual(row['score'], float(raw[i, 4]))
                        self.assertEqual(row['class_confidence'], float(raw[i, 5]))
                        observations.append({'arm': arm, 'ordinal': i, 'class': row['predicted_class'],
                                             'score': row['score'], 'confidence': row['class_confidence']})
                    for track, _, _, _ in tracker.pending:
                        track.frame_id = tracker.frame_id
                    tracker._flush_committed()
                self.assertEqual(len(set(tracker.det_ordinals.values())), 2)
                self.assertEqual(tracker.counts['corrected_reference_seams_verified'], 2)
        finish(2, 3, 'duplicate bbox class/score high-low identity', observations=observations)

    def test_round3_1_oracle_whole_graph_alias_isolation_and_actual_state(self):
        cases = []
        for arm in ('fc_full', 'fc_control'):
            tracker, tracks, dets = make(arm)
            bind_independently(tracker, dets)
            tracker.stage = 1  # Runtime calls its oracle after increment/binding.
            before = critical_state(tracker, tracks, dets)
            reference = oracle.oracle_for_seam(tracker, tracks, dets, .99, False, False)
            self.assertEqual(before, critical_state(tracker, tracks, dets))
            self.assertTrue(reference['pending'])
            for cloned_track, observation, ratio, responsibility in reference['pending']:
                original_track = reference['track_identity'][id(cloned_track)]
                self.assertIsNot(cloned_track, original_track)
                self.assertFalse(np.shares_memory(cloned_track.curr_feat, original_track.curr_feat))
                self.assertFalse(np.shares_memory(cloned_track.mean, original_track.mean))
                cloned_track.curr_feat[0] += 13.
                cloned_track.mean[0] += 17.
                observation['score'] = -3.
                ratio[0] += 7.
            self.assertEqual(before, critical_state(tracker, tracks, dets))
            # A real track used simultaneously in both lists remains one clone.
            alias_tracker, alias_tracks, _ = make(arm)
            alias_det = alias_tracks[0]
            alias_tracker.frame_observations = {0: raw_rows(RAW)[0]}
            alias_tracker.det_ordinals = {id(alias_det): 0}
            alias_tracker.stage = 1
            original = critical_state(alias_tracker, [alias_det], [alias_det])
            alias_ref = oracle.oracle_for_seam(alias_tracker, [alias_det], [alias_det], .99, False, False)
            self.assertEqual(original, critical_state(alias_tracker, [alias_det], [alias_det]))
            self.assertEqual(alias_ref['stage'], 1)
            cases.append({'arm': arm, 'pending': len(reference['pending']), 'alias_case_checked': True})
        finish(3, 1, 'whole-seam clone isolation including aliased real STrack', cases=cases)

    def test_round3_2_committed_update_matches_original_semantic_update(self):
        evidence = []
        for arm in ('fc_full', 'fc_control'):
            tracker, tracks, dets = make(arm)
            expected, clone, identity = independent_original(tracker, tracks, dets)
            actual = tracker.associate(tracks, dets, .99)
            assert_whole(self, actual, expected, tracker, clone, identity)
            before = {k: v.copy() for k, v in tracker.beliefs.items()}
            pending = list(tracker.pending)
            with self.assertRaisesRegex(AssertionError, 'BEFORE_HOST_COMMIT'):
                tracker._flush_committed()
            for tid, value in before.items():
                np.testing.assert_array_equal(tracker.beliefs[tid], value)
            original = oracle.original_runtime(arm == 'fc_full')
            for (track, obs, ratio, responsibility), (copy_track, _, _, _) in zip(pending, clone.pending):
                track.frame_id = copy_track.frame_id = tracker.frame_id
                expected_belief = original.semantic_update(before[track.track_id], ratio, responsibility)
                self.assertTrue(np.isfinite(expected_belief).all())
            tracker._flush_committed()
            original.BeliefTracker._flush_committed(clone)
            for track, obs, ratio, responsibility in pending:
                np.testing.assert_array_equal(tracker.beliefs[track.track_id],
                    original.semantic_update(before[track.track_id], ratio, responsibility))
                np.testing.assert_array_equal(tracker.beliefs[track.track_id], clone.beliefs[track.track_id])
                self.assertEqual(tracker.observed[track.track_id], clone.observed[track.track_id])
            self.assertFalse(tracker.pending)
            self.assertEqual(tracker.responsibility_sum, clone.responsibility_sum)
            self.assertEqual(tracker.belief_change_l1, clone.belief_change_l1)
            evidence.append({'arm': arm, 'commits': len(pending)})
        finish(3, 2, 'host-commit boundary and original semantic update', cases=evidence)

    def test_round3_3_wrong_full_tuple_responsibility_ratio_and_stage_reject(self):
        rejected = []
        for arm in ('fc_full', 'fc_control'):
            tracker, tracks, dets = make(arm)
            expected, clone, identity = independent_original(tracker, tracks, dets)
            result = tracker.associate(tracks, dets, .99)
            reference = {'assignment': expected, 'pending': clone.pending, 'stage': clone.stage,
                         'track_identity': identity}
            self.assertTrue(tracker.pending)
            mutations = [
                ('unmatched', 'ASSIGNMENT_MISMATCH'),
                ('responsibility', 'RESPONSIBILITY_MISMATCH'),
                ('ratio', 'RATIO_MISMATCH'),
                ('stage', 'STAGE_MISMATCH'),
                ('metadata', 'OBSERVATION_MISMATCH'),
            ]
            for kind, code in mutations:
                changed = copy.copy(reference)
                changed['pending'] = [tuple(x) for x in reference['pending']]
                wrong_result = result
                if kind == 'unmatched':
                    wrong_result = (result[0], tuple(result[1]) + (len(tracks),), result[2])
                elif kind == 'stage':
                    changed['stage'] += 1
                else:
                    track, obs, ratio, responsibility = changed['pending'][0]
                    obs, ratio = copy.deepcopy(obs), np.asarray(ratio).copy()
                    if kind == 'responsibility':
                        responsibility = float(responsibility) + .03125
                    elif kind == 'ratio':
                        ratio[0] += .125
                    else:
                        obs['class_confidence'] = float(obs['class_confidence']) + .125
                    changed['pending'][0] = track, obs, ratio, responsibility
                verified_before = tracker.counts['corrected_reference_seams_verified']
                with self.assertRaisesRegex(AssertionError, code):
                    oracle.verify_reference(tracker, wrong_result, changed)
                self.assertEqual(tracker.counts['corrected_reference_seams_verified'], verified_before)
                rejected.append(arm + ':' + kind)
        finish(3, 3, 'deliberate tuple/pending/stage defects must reject', rejected=rejected)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CorrectedReferenceCPU)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    complete = (result.wasSuccessful() and len(CHECKS) == 9 and
                {(x['round'], x['check']) for x in CHECKS} ==
                {(r, c) for r in range(1, 4) for c in range(1, 4)} and
                not runtime.torch.cuda.is_initialized())
    artifact = ROOT / 'artifacts/CORRECTED_REFERENCE_CPU.json'
    log = ROOT / 'artifacts/CORRECTED_REFERENCE_CPU.log'
    production_sources = [
        'tests/test_corrected_reference.py', 'belief/reference_oracle.py',
        'belief/runtime.py', 'belief/features.py', 'belief/models.py',
        'tools/capture_belief_evidence.py', 'tools/belief_data_common.py',
        'host/yolox/tracker/u2mot_tracker.py', 'host/yolox/tracker/matching.py',
        'host/yolox/tracker/basetrack.py', 'host/yolox/tracker/kalman_filter.py',
    ]
    sources = {**{p: sha(p) for p in SOURCE_HASHES},
               **{str(ROOT / p): sha(ROOT / p) for p in production_sources}}
    payload = {'status': 'PASS_NINE_CORRECTED_REFERENCE_CPU_CONTRACTS' if complete else 'FAIL_CPU_CONTRACTS',
               'scope': 'Real STrack/F00/summary heads, whole immutable associate methods under shared corrected metadata',
               'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'rounds': 3, 'checks_per_round': 3, 'checks': CHECKS,
               'test_path': str(Path(__file__).resolve()), 'test_sha256': sha(__file__),
               'source_files_sha256': {p: sha(p) for p in SOURCE_HASHES},
               'source_sha256': sources, 'source_hash_key_scope': 'absolute filesystem paths',
               'oracle_path': str(ROOT / 'belief/reference_oracle.py'),
               'oracle_sha256': sha(ROOT / 'belief/reference_oracle.py'),
               'runtime_sha256': sha(ROOT / 'belief/runtime.py'),
               'providers': provider_evidence(), 'cuda_initialized': runtime.torch.cuda.is_initialized(),
               'mocked_methods': [], 'formal_online_accuracy_evidence': False,
               'limits': ['CPU seam fixtures do not replace all-seam real-online reference coverage',
                          'Shared corrected metadata guard is declared, not independently frozen old code',
                          'features.py differs only in unused source_vectors instrumentation; called function ASTs equal the immutable versions',
                          'Source head is loaded but not executed by association when clutter/edge policies are off',
                          'Input-state deepcopy isolates seam objects; unrelated removed history is not copied by the oracle'],
               'command': 'BELIEF_HOST=%s PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /home/chenhc/.conda/envs/u2mot/bin/python -B tests/test_corrected_reference.py' % (ROOT / 'host'),
               'log_path': str(log),
               'initial_attempt': {
                   'log_path': str(ROOT / 'artifacts/CORRECTED_REFERENCE_CPU_attempt1.log'),
                   'log_sha256': sha(ROOT / 'artifacts/CORRECTED_REFERENCE_CPU_attempt1.log'),
                   'tests': 9, 'passed': 8,
                   'failure': 'Overbroad features.py whole-file identity assertion: only unused source_vectors instrumentation differs',
                   'receipt_error': 'numpy int64 JSON encoding corrected in test recorder; no runtime/oracle modification'}}
    artifact.write_text(json.dumps(payload, indent=2, allow_nan=False, default=json_scalar) + '\n')
    sys.exit(0 if complete else 1)
