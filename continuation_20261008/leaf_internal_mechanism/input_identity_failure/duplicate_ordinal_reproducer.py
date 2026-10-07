"""Independent CPU reproducer; does not patch frozen tracker source."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from belief import runtime
from capture_belief_evidence import CaptureTracker, STrack
from yolox.tracker.u2mot_tracker import DefaultArgs


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


canonical_path = Path('/home/chenhc/src_egia_belief_20260923/tools/capture_belief_evidence.py')
canonical = module('canonical_capture_ordinal_audit', canonical_path)
contracts = module('frozen_cost_contract_fixture', ROOT / 'tests/test_cost_factors.py')


def rows_for(raw):
    boxes = raw[:, :4].copy()
    boxes[:, 2:] -= boxes[:, :2]
    return {i: {'sequence': 'synthetic_no_gpu', 'frame': 7, 'source_frame': 7,
                'detection_ordinal': i, 'bbox_tlwh': boxes[i].astype(float).tolist(),
                'score': float(row[4]), 'class_confidence': float(row[5]),
                'predicted_class': int(row[6])}
            for i, row in enumerate(raw)}


def fresh(kind, raw):
    if kind == 'current_frozen_belief':
        tracker, _, _, _ = contracts.make_tracker('fc_full')
        del tracker.det_record  # Use the actual inherited CaptureTracker implementation.
    else:
        cls = canonical.CaptureTracker if kind == 'canonical_capture' else CaptureTracker
        tracker = cls(DefaultArgs())
    tracker.frame_id = 7
    tracker.stage = 0
    tracker.source_frame = 7
    tracker.sequence_name = 'synthetic_no_gpu'
    tracker.streams = {'association': io.StringIO()}
    tracker.det_ordinals = {}
    tracker.frame_observations = rows_for(raw)
    if hasattr(tracker, '_build_ordinal_index'):
        tracker._build_ordinal_index()
    return tracker


def objects(raw):
    return [STrack(STrack.tlbr_to_tlwh(row[:4]), row[4], row[6],
                   semantic_score=row[5]) for row in raw]


def run_case(kind, raw, groups):
    tracker = fresh(kind, raw)
    dets = objects(raw)
    results = []
    for group in groups:
        tracker.associate([], [dets[i] for i in group], .8)
        for i in group:
            got = tracker.det_record(dets[i])
            results.append({'actual_ordinal': i,
                            'bound_ordinal': got['detection_ordinal'],
                            'actual_class': int(dets[i].cls),
                            'bound_class': got['predicted_class'],
                            'actual_objectness': float(dets[i].score),
                            'bound_objectness': got['score'],
                            'actual_class_confidence': float(dets[i].semantic_score),
                            'bound_class_confidence': got['class_confidence'],
                            'bbox_exact_assertion_passed': True})
    return results


def robust_reference_candidates(det, rows, used):
    return [i for i, row in rows.items()
            if i not in used
            and np.array_equal(np.asarray(row['bbox_tlwh']), det._tlwh)
            and int(row['predicted_class']) == int(det.cls)
            and float(row['score']) == float(det.score)
            and float(np.clip(row['class_confidence'], 0., 1.)) == float(det.semantic_score)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    low_first = np.asarray([[1, 2, 11, 12, .4, .9, 0],
                            [1, 2, 11, 12, .9, .8, 3]], dtype=np.float16)
    high_first = low_first[::-1].copy()
    same_pool = np.asarray([[1, 2, 11, 12, .9, .9, 0],
                            [1, 2, 11, 12, .9, .8, 3]], dtype=np.float16)
    results = {}
    for kind in ('copied_capture', 'canonical_capture', 'current_frozen_belief'):
        results[kind] = {
            'duplicate_low_before_high': run_case(kind, low_first, [[1], [0]]),
            'duplicate_high_before_low': run_case(kind, high_first, [[0], [1]]),
            'duplicate_same_pool_stable_order': run_case(kind, same_pool, [[0, 1]]),
        }
        bad = results[kind]['duplicate_low_before_high']
        assert [r['bound_ordinal'] for r in bad] == [0, 1]
        assert [r['actual_ordinal'] for r in bad] == [1, 0]
        assert all(r['bound_class'] != r['actual_class'] for r in bad)
        for key in ('duplicate_high_before_low', 'duplicate_same_pool_stable_order'):
            assert all(r['bound_ordinal'] == r['actual_ordinal'] for r in results[kind][key])

    metadata_proof = []
    for raw in (low_first, high_first, same_pool):
        dets = objects(raw)
        rows = rows_for(raw)
        used = set()
        for i in reversed(range(len(dets))):
            matches = robust_reference_candidates(dets[i], rows, used)
            assert matches == [i], (i, matches)
            used.add(matches[0])
            metadata_proof.append({'actual_ordinal': i, 'unique_metadata_candidates': matches})

    det = objects(low_first)[1]
    tracker = fresh('current_frozen_belief', low_first)
    tracker.associate([], [det], .8)
    wrong = tracker.model.observation_probabilities([tracker.det_record(det)])[0]
    correct = tracker.model.observation_probabilities([rows_for(low_first)[1]])[0]
    assert not np.array_equal(wrong, correct)

    full_duplicate = np.vstack([low_first[0], low_first[0]])
    ambiguous = robust_reference_candidates(objects(full_duplicate)[0], rows_for(full_duplicate), set())
    assert ambiguous == [0, 1]

    result = {
        'status': 'SYNTHETIC_CURRENT_SOURCE_CORRUPTION_REPRODUCED',
        'scope': 'Actual frozen/canonical associate and det_record methods, real STrack, synthetic FP16 rows, CPU; no inference, GPU, frozen edits or live-stream claim',
        'source_sha256': {str(p): sha(p) for p in [canonical_path,
                         ROOT / 'tools/capture_belief_evidence.py', ROOT / 'belief/runtime.py',
                         ROOT / 'belief/online.py', ROOT / 'host/yolox/tracker/u2mot_tracker.py']},
        'actual_methods': results,
        'wrong_observation_probabilities': wrong.tolist(),
        'correct_observation_probabilities': correct.tolist(),
        'synthetic_observation_model_only': True,
        'metadata_reference_proof': metadata_proof,
        'full_tuple_duplicate_remains_ambiguous': ambiguous,
        'raw_dtype': str(low_first.dtype),
        'raw_rows_low_first': low_first.astype(float).tolist(),
    }
    target = ROOT / 'review/DUPLICATE_ORDINAL_CPU_RECEIPT.json'
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'receipt': str(target),
                      'receipt_sha256': sha(target), 'actual_method_cases': 9,
                      'silent_swaps': 6, 'unambiguous_metadata_reference_rows': 6,
                      'full_duplicate_ambiguity_detected': True}, sort_keys=True))


if __name__ == '__main__':
    main()
