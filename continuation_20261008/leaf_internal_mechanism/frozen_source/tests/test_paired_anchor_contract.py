"""Three audit rounds with three independent paired-anchor checks each."""
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from test_egia_modes import canonical, changed_fields, file_sha256, object_sha256


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def audit():
    prereg = json.loads((ROOT / 'PHASE2_HEAD_PREREGISTRATION.json').read_text())
    receipt = json.loads((ROOT / 'artifacts/PHASE2_HEAD_TRAIN_RECEIPT.json').read_text())
    data = pickle.loads(Path(prereg['prepared_path']).read_bytes())
    base = pickle.loads((ROOT / 'models/F00_full.pkl').read_bytes())
    bundles = {name: pickle.loads((ROOT / 'models' / (name + '.pkl')).read_bytes())
               for name in ['Paired_Flat', 'Paired_Hierarchical']}
    return prereg, receipt, data, base, bundles


def test_round1_exact_preregistered_sources_and_no_test_fit(audit):
    prereg, _, _, _, _ = audit
    for path, expected in prereg['source_files_sha256'].items():
        assert file_sha256(path) == expected
    assert not prereg['testdev_selection']
    assert not prereg['parameter_search']
    assert not prereg['detector_or_ReID_training']


def test_round1_actual_training_and_calibration_membership(audit):
    prereg, _, data, _, _ = audit
    assert sorted({row['sequence'] for row in data['fit']}) == sorted(prereg['fit_sequences'])
    assert sorted({row['sequence'] for row in data['calibration']}) == sorted(prereg['calibration_sequences'])
    assert not ({row['flight'] for row in data['fit']} &
                {row['flight'] for row in data['calibration']})
    assert (len(data['fit']), len(data['calibration'])) == (8024, 8411)


def test_round1_actual_features_and_labels(audit):
    prereg, _, data, _, _ = audit
    x = np.stack([row['legacy'] for row in data['fit']]).astype(np.float64)
    y = np.asarray([row['label'] for row in data['fit']], dtype=int)
    assert x.shape == (8024, 10) and np.isfinite(x).all()
    assert object_sha256(x) == prereg['fit_feature_array_sha256']
    assert object_sha256(y) == prereg['fit_labels_sha256']
    assert np.bincount(y, minlength=3).tolist() == [3985, 2108, 1931]


def test_round2_independent_causal_and_common_weight_construction(audit):
    prereg, _, data, _, _ = audit
    keys = []
    flight_groups = defaultdict(set)
    for row in data['fit']:
        key = ((row['flight'], 2, row['sequence'], int(row['source_frame']) // 30)
               if row['label'] == 2 else
               (row['flight'], row['label'], row['candidate_gt_id']))
        keys.append(key)
        flight_groups[row['flight']].add(key)
    counts = Counter(keys)
    w = np.asarray([1. / len(flight_groups) / len(flight_groups[key[0]]) / counts[key]
                    for key in keys])
    y = np.asarray([row['label'] for row in data['fit']])
    assert object_sha256(w) == prereg['event_weights']['base_sha256']
    masses = {label: w[y == label].sum() for label in range(3)}
    common = w * np.asarray([((1. / 3.) / masses[int(label)]) ** .5 for label in y])
    common *= w.sum() / common.sum()
    assert object_sha256(common) == prereg['event_weights']['common_sha256']
    assert float(common[y != 2].sum()) == prereg['event_weights']['coverage_subset_weight']


def test_round2_common_scalers_are_full_fit_float64(audit):
    _, _, data, _, bundles = audit
    x = np.stack([row['legacy'] for row in data['fit']]).astype(np.float64)
    hierarchy = bundles['Paired_Hierarchical']['source_model'].anchor
    pipelines = [bundles['Paired_Flat']['source_model'].anchor,
                 hierarchy.foreground_model, hierarchy.coverage_model]
    for model in pipelines:
        scaler = model.named_steps['standardscaler']
        assert int(scaler.n_samples_seen_) == 8024
        np.testing.assert_array_equal(scaler.mean_, x.mean(axis=0))
        np.testing.assert_allclose(scaler.var_, x.var(axis=0), rtol=1e-14, atol=0)
    assert len({object_sha256(model.named_steps['standardscaler']) for model in pipelines}) == 1


def test_round2_common_fit_parameters_and_posterior_factorization(audit):
    prereg, _, data, _, bundles = audit
    x = np.stack([row['legacy'] for row in data['calibration'][::128]]).astype(np.float64)
    flat = bundles['Paired_Flat']['source_model'].anchor
    hierarchy = bundles['Paired_Hierarchical']['source_model'].anchor
    for model in [flat, hierarchy.foreground_model, hierarchy.coverage_model]:
        parameters = model.named_steps['logisticregression'].get_params()
        assert all(parameters[key] == value
                   for key, value in prereg['logistic_regression_parameters'].items())
    pf = hierarchy.foreground_model.predict_proba(x)[:, 1]
    pc = hierarchy.coverage_model.predict_proba(x)[:, 1]
    manual = np.column_stack((pf * (1-pc), pf * pc, 1-pf))
    manual /= manual.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(hierarchy.predict_proba(x), manual)
    assert flat.classes_.tolist() == hierarchy.classes_.tolist() == [0, 1, 2]


def test_round3_only_nested_anchor_differs_from_frozen_f00(audit):
    _, _, _, base, bundles = audit
    for bundle in bundles.values():
        assert changed_fields(base, bundle) == ['source_model']
        assert changed_fields(base['source_model'].__dict__,
                              bundle['source_model'].__dict__) == ['anchor']


def test_round3_gt_free_online_event_contract(audit):
    _, _, _, _, bundles = audit
    event = {
        'sequence': 'contract', 'frame': 10, 'detection_ordinal': 0,
        'candidate': {'predicted_class': 3},
        'native_features': dict(zip(
            ['score', 'class_confidence', 'max_active_iou', 'max_lost_iou',
             'max_active_reid_similarity', 'max_lost_reid_similarity',
             'active_track_count', 'lost_track_count', 'bbox_area_ratio',
             'border_distance_ratio'], [.9, .8, .4, 0., .8, 0., 2, 0, .01, .1])),
        'sources': [dict(track_id=i, paired_iou=iou, reid_similarity=reid,
                         active=True, lost=False, age=5, gap=0,
                         semantic_evidence=[0.2, 0.8], last_observation=None)
                    for i, iou, reid in [(1, .4, .8), (2, .2, .3)]],
    }
    assert 'label' not in event and 'candidate_gt_id' not in event
    for bundle in bundles.values():
        p, ids, evidence = bundle['source_model'].predict_event(event)
        assert ids == [1, 2]
        assert np.isfinite(p).all() and np.isfinite(evidence)
        np.testing.assert_allclose(p.sum(), 1., atol=1e-12)


def test_round3_readback_receipt_models_and_diagnostic_boundary(audit):
    _, receipt, _, _, _ = audit
    assert receipt['preregistration_sha256'] == file_sha256(ROOT / 'PHASE2_HEAD_PREREGISTRATION.json')
    for model in receipt['models']:
        assert file_sha256(model['path']) == model['sha256']
    assert file_sha256(receipt['calibration_metrics_path']) == receipt['calibration_metrics_sha256']
    assert not receipt['formal_metrics']
    assert not receipt['calibration_used_for_fit_or_selection']
    assert not receipt['testdev_used_for_fit_or_selection']
    assert not receipt['GPU_inference_started']
    assert len(receipt['checks']) == 9
    for round_ in [1, 2, 3]:
        assert sum(item['round'] == round_ for item in receipt['checks']) == 3
