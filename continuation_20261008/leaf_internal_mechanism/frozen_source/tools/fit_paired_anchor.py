"""Predeclare and fit the common-weight flat/hierarchical anchor comparison.

Only two small EGIA anchors are fitted. All other F00 model attributes and
deployment policies stay frozen. Calibration is diagnostic, never selection.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from belief.features import CLASS_NAMES, LEGACY_FEATURES
from belief.hierarchical import HierarchicalActionAnchor, class_balanced_weights
from train_coverage_egia_v4 import weights


DATA = Path('/home/chenhc/src_egia_belief_20260923/artifacts/'
            'prepare_coverage_v4_fit39_cal8_v2/prepared.pkl')
ORIGINAL_TRAIN = Path('/home/chenhc/src_egia_belief_20260923/artifacts/'
                      'train_visdrone_hierarchical_v5_beta05_fit39_cal8_20260926_v1')
F00_SHA = 'ee903030e48f73fe87411b6e9aa60d52bbf2ee2d895aebfa8085d260779702ac'
DATA_SHA = 'e126f0ebcbb284985eb8bfd43c6bcaec4dc401fe7e507960518ea59fafde055d'
TEACHER_SHA = 'aa48c972a3ddbe4c5e4b82c5f2a64f9032978d7c15a863a5a3caefe67ac04ce8'
PREREG = ROOT / 'PHASE2_HEAD_PREREGISTRATION.json'
RECEIPT = ROOT / 'artifacts/PHASE2_HEAD_TRAIN_RECEIPT.json'
METRICS = ROOT / 'artifacts/phase2_head_calibration_metrics.json'
LR_PARAMS = dict(C=1., penalty='l2', solver='lbfgs', max_iter=2000,
                 tol=1e-4, random_state=20260923, fit_intercept=True,
                 class_weight=None)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    if isinstance(value, np.ndarray):
        data = canonical(value.tolist()) if value.dtype.hasobject else value.tobytes().hex()
        return ['array', str(value.dtype), list(value.shape), data]
    if isinstance(value, np.generic):
        return ['scalar', str(value.dtype), value.item()]
    if value is None or isinstance(value, (str, int, float, bool)):
        return [type(value).__name__, value]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [canonical(item) for item in value]]
    if isinstance(value, dict):
        return ['dict', [[str(key), canonical(item)] for key, item in
                         sorted(value.items(), key=lambda pair: str(pair[0]))]]
    if hasattr(value, '__dict__'):
        return ['object', type(value).__module__ + '.' + type(value).__qualname__,
                canonical(value.__dict__)]
    raise TypeError(type(value))


def objsha(value):
    raw = json.dumps(canonical(value), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


def changed(reference, candidate):
    return sorted(key for key in set(reference) | set(candidate)
                  if key not in reference or key not in candidate or
                  objsha(reference[key]) != objsha(candidate[key]))


def write_exclusive(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def independent_base_weights(rows):
    """Reconstruct equal-flight/group weights without the imported helper."""
    counts = defaultdict(int)
    flight_groups = defaultdict(set)
    row_groups = []
    for row in rows:
        if row['label'] == 2:
            key = (row['flight'], 2, row['sequence'], int(row['source_frame']) // 30)
        else:
            key = (row['flight'], row['label'], row['candidate_gt_id'])
        row_groups.append(key)
        counts[key] += 1
        flight_groups[row['flight']].add(key)
    return np.asarray([1. / len(flight_groups) / len(flight_groups[key[0]]) /
                       counts[key] for key in row_groups], dtype=float)


def inspect_data():
    assert sha(DATA) == DATA_SHA
    assert sha(ROOT / 'models/F00_full.pkl') == F00_SHA
    assert sha(ROOT / 'models/summary_frozen.pkl') == TEACHER_SHA
    data = pickle.loads(DATA.read_bytes())
    fit, cal = data['fit'], data['calibration']
    # HierarchicalActionAnchor and live legacy_vector use float64. Use that
    # same numerical transform for both heads and independent comparator paths.
    x = np.stack([row['legacy'] for row in fit]).astype(np.float64)
    y = np.asarray([row['label'] for row in fit], dtype=int)
    raw_weights = weights(fit)
    shared_weights = class_balanced_weights(raw_weights, y, exponent=.5)
    prep = json.loads((DATA.parent / 'prepare_receipt.json').read_text())
    assert (len(fit), len(cal), x.shape) == (8024, 8411, (8024, 10))
    assert np.bincount(y, minlength=3).tolist() == [3985, 2108, 1931]
    assert np.bincount([row['label'] for row in cal], minlength=3).tolist() == [6309, 467, 1635]
    assert sorted({row['sequence'] for row in fit}) == sorted(prep['fit_sequences'])
    assert sorted({row['sequence'] for row in cal}) == sorted(prep['cal_sequences'])
    assert len(prep['fit_sequences']) == 39 and len(prep['cal_sequences']) == 8
    assert not ({row['flight'] for row in fit} & {row['flight'] for row in cal})
    assert np.isfinite(x).all() and (shared_weights > 0).all()
    np.testing.assert_array_equal(raw_weights, independent_base_weights(fit))
    masses = np.asarray([raw_weights[y == label].sum() for label in range(3)])
    independently_balanced = raw_weights * np.sqrt((1. / 3.) / masses[y])
    independently_balanced *= raw_weights.sum() / independently_balanced.sum()
    np.testing.assert_allclose(shared_weights, independently_balanced, rtol=1e-14, atol=0)
    return data, prep, x, y, raw_weights, shared_weights


def registration():
    data, prep, x, y, raw_weights, shared_weights = inspect_data()
    base = pickle.loads((ROOT / 'models/F00_full.pkl').read_bytes())
    original_receipt = json.loads((ORIGINAL_TRAIN / 'receipt.json').read_text())
    assert original_receipt['prepared_sha256'] == DATA_SHA
    assert sha(ORIGINAL_TRAIN / 'model.pkl') == original_receipt['model_sha256']
    original = pickle.loads((ORIGINAL_TRAIN / 'model.pkl').read_bytes())
    assert objsha(original['source_model']) == objsha(base['source_model'])
    sources = [Path(__file__), ROOT / 'belief/hierarchical.py',
               ROOT / 'belief/coherence.py', ROOT / 'belief/features.py',
               ROOT / 'tools/train_coverage_egia_v4.py',
               ROOT / 'tools/train_uavdt_hierarchical_v5.py',
               ROOT / 'models/F00_full.pkl', ROOT / 'models/summary_frozen.pkl',
               DATA, DATA.parent / 'prepare_receipt.json',
               ORIGINAL_TRAIN / 'model.pkl', ORIGINAL_TRAIN / 'receipt.json',
               ORIGINAL_TRAIN / 'metrics.json']
    return {
        'status': 'PREREGISTERED_BEFORE_VALID_PAIRED_HEAD_FIT_REVISION2',
        'prior_failed_attempt': str(ROOT / 'artifacts/phase2_failed_attempt1/failure.json'),
        'prior_failed_attempt_sha256': sha(ROOT / 'artifacts/phase2_failed_attempt1/failure.json'),
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_files_sha256': {str(path): sha(path) for path in sources},
        'sklearn_version': sklearn.__version__,
        'dataset': 'VisDrone2019 original train annotations via frozen prepared rows',
        'prepared_path': str(DATA), 'prepared_sha256': DATA_SHA,
        'fit_sequences': prep['fit_sequences'], 'calibration_sequences': prep['cal_sequences'],
        'fit_rows': len(data['fit']), 'calibration_rows': len(data['calibration']),
        'fit_flights': 32, 'calibration_flights': 8, 'flight_overlap': [],
        'fit_class_counts': [3985, 2108, 1931], 'calibration_class_counts': [6309, 467, 1635],
        'feature_field': 'legacy', 'feature_names': LEGACY_FEATURES,
        'feature_storage_dtype': 'float32', 'computation_dtype': 'float64',
        'fit_feature_array_sha256': objsha(x), 'fit_labels_sha256': objsha(y),
        'label_order': CLASS_NAMES,
        'label_semantics': ['uncovered foreground', 'covered foreground', 'clutter'],
        'unknown_row_mask': 'Reuse exact prepared row membership; no relabeling',
        'scaler': {'class': 'StandardScaler', 'copy': True, 'with_mean': True,
                   'with_std': True, 'fit_on': 'All 8024 fit rows only, unweighted',
                   'shared_transform': 'One fitted scaler copied into all three fitted pipelines'},
        'event_weights': {'base_rule': 'equal flight, equal (flight,label,GT identity) foreground group or 30-frame clutter block, equal rows within group',
                          'base_sha256': objsha(raw_weights),
                          'balance_function': 'belief.hierarchical.class_balanced_weights',
                          'balance_labels': 'original three-class labels', 'balance_exponent': .5,
                          'common_sha256': objsha(shared_weights),
                          'total_weight': float(shared_weights.sum()),
                          'coverage_subset_weight': float(shared_weights[y != 2].sum()),
                          'normalization': 'Keep absolute weights; do not normalize coverage subset or rescale to row count'},
        'logistic_regression_parameters': LR_PARAMS,
        'arms': [
            {'name': 'Paired_Flat', 'anchor': 'Pipeline(StandardScaler, LogisticRegression)',
             'classes': [0, 1, 2], 'objective': 'multinomial weighted three-class NLL',
             'multi_class': 'multinomial', 'fit_rows': 8024},
            {'name': 'Paired_Hierarchical', 'anchor': 'HierarchicalActionAnchor',
             'foreground_target': 'label != 2', 'foreground_fit_rows': 8024,
             'coverage_target': 'label == 1 among label != 2', 'coverage_fit_rows': 6093,
             'binary_multi_class': 'auto',
             'posterior': '[p_fg*(1-p_cov), p_fg*p_cov, 1-p_fg]'}],
        'frozen_downstream': {'base_model_sha256': F00_SHA,
                              'only_replacement': 'bundle.source_model.anchor',
                              'source_gamma': base['source_model'].weight,
                              'source_pool': base.get('source_pool_mode', 'legacy_all'),
                              'fusion_weight': base['egia_fusion_weight'],
                              'selective_margin': base['selective_birth_margin'],
                              'SRC_and_cue_heads_and_teacher': 'All unchanged'},
        'external_reference': 'Canonical F00; paired refit hierarchy is not canonical F00',
        'original_recipe_difference': 'F00 has binary-specific class balance and a foreground-only coverage scaler; both new arms replace these with shared three-class weights and one full-fit scaler',
        'analysis': 'Paired factorization comparison under common features, transform and per-event weights; no architecture-only F00-flat attribution',
        'calibration_use': 'Locked diagnostic weighted NLL and per-class values only; no fitting, selection or tuning',
        'parameter_search': False, 'testdev_selection': False,
        'detector_or_ReID_training': False, 'GPU_inference_started': False,
        'formal_metrics': False,
    }


def probabilities_metrics(rows, probabilities):
    y = np.asarray([row['label'] for row in rows], dtype=int)
    w = weights(rows)
    p = np.clip(probabilities, 1e-12, 1.)
    p /= p.sum(axis=1, keepdims=True)
    losses = -np.log(p[np.arange(len(y)), y])
    return {'rows': len(rows), 'class_counts': np.bincount(y, minlength=3).tolist(),
            'flight_weighted_nll': float(np.average(losses, weights=w)),
            'weighted_accuracy': float(np.average(p.argmax(1) == y, weights=w)),
            'per_class_weighted_nll': [float(np.average(losses[y == c], weights=w[y == c]))
                                       for c in range(3)],
            'posterior_sha256': objsha(probabilities)}


def fit():
    prereg_sha = sha(PREREG)
    prereg = json.loads(PREREG.read_text())
    for path, expected in prereg['source_files_sha256'].items():
        assert sha(path) == expected, ('STOP_PREREG_SOURCE_CHANGED', path)
    assert not RECEIPT.exists() and not METRICS.exists()
    for name in ['Paired_Flat', 'Paired_Hierarchical']:
        assert not (ROOT / 'models' / (name + '.pkl')).exists()
    data, prep, x, y, raw_weights, common_weights = inspect_data()
    base = pickle.loads((ROOT / 'models/F00_full.pkl').read_bytes())
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    foreground_mask = y != 2
    flat_lr = LogisticRegression(multi_class='multinomial', **LR_PARAMS)
    flat_lr.fit(z, y, sample_weight=common_weights)
    fg_lr = LogisticRegression(**LR_PARAMS)
    fg_lr.fit(z, foreground_mask.astype(int), sample_weight=common_weights)
    cov_lr = LogisticRegression(**LR_PARAMS)
    cov_lr.fit(z[foreground_mask], (y[foreground_mask] == 1).astype(int),
               sample_weight=common_weights[foreground_mask])
    pipeline = lambda lr: Pipeline([('standardscaler', copy.deepcopy(scaler)),
                                    ('logisticregression', lr)])
    anchors = {'Paired_Flat': pipeline(flat_lr),
               'Paired_Hierarchical': HierarchicalActionAnchor(pipeline(fg_lr), pipeline(cov_lr))}
    checks = []
    def checked(round_, name, detail):
        checks.append(dict(round=round_, name=name, status='PASS', detail=detail))
    checked(1, 'source_hashes_sealed_before_fit', {'prereg_sha256': prereg_sha, 'sources': len(prereg['source_files_sha256'])})
    checked(1, 'actual_membership_and_label_support', {'fit_rows': len(y), 'fit_sequences': 39, 'calibration_rows': 8411, 'calibration_sequences': 8, 'flight_overlap': []})
    checked(1, 'independent_weight_reconstruction', {'common_weight_sha256': objsha(common_weights), 'no_subset_normalization': True})
    pipelines = [anchors['Paired_Flat'], anchors['Paired_Hierarchical'].foreground_model,
                 anchors['Paired_Hierarchical'].coverage_model]
    for model in pipelines:
        np.testing.assert_array_equal(model.named_steps['standardscaler'].mean_, scaler.mean_)
        np.testing.assert_array_equal(model.named_steps['standardscaler'].scale_, scaler.scale_)
        assert int(model.named_steps['standardscaler'].n_samples_seen_) == 8024
        assert all(model.named_steps['logisticregression'].get_params()[key] == value
                   for key, value in LR_PARAMS.items())
        assert np.max(model.named_steps['logisticregression'].n_iter_) < 2000
    checked(2, 'shared_fit_only_scaler_in_all_three_pipelines', {'scaler_object_sha256': objsha(scaler), 'fit_rows': 8024})
    checked(2, 'fixed_estimator_parameters_and_convergence', {'parameters': LR_PARAMS, 'n_iter': [model.named_steps['logisticregression'].n_iter_.tolist() for model in pipelines]})
    np.testing.assert_array_equal(anchors['Paired_Flat'].predict_proba(x), flat_lr.predict_proba(z))
    pfg, pcov = fg_lr.predict_proba(z)[:, 1], cov_lr.predict_proba(z)[:, 1]
    manual = np.column_stack((pfg * (1-pcov), pfg * pcov, 1-pfg))
    manual /= manual.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(anchors['Paired_Hierarchical'].predict_proba(x), manual, rtol=0, atol=1e-15)
    checked(2, 'posterior_matches_independent_transformed_estimator_path', {'flat_max_error': 0., 'hierarchical_absolute_tolerance': 1e-15})
    metrics = {'formal_metrics': False, 'calibration_used_for_fit_or_selection': False,
               'calibration_data_scope': 'Original train cal8 prepared rows; historical current-output source pool, no closed-loop benchmark trajectories', 'arms': {}}
    model_records = []
    for name, anchor in anchors.items():
        bundle = copy.deepcopy(base)
        bundle['source_model'].anchor = anchor
        assert changed(base, bundle) == ['source_model']
        assert changed(base['source_model'].__dict__, bundle['source_model'].__dict__) == ['anchor']
        path = ROOT / 'models' / (name + '.pkl')
        with path.open('xb') as stream:
            pickle.dump(bundle, stream, protocol=4)
        loaded = pickle.loads(path.read_bytes())
        assert changed(base, loaded) == ['source_model']
        assert changed(base['source_model'].__dict__, loaded['source_model'].__dict__) == ['anchor']
        metrics['arms'][name] = {}
        for split in ['fit', 'calibration']:
            rows = data[split]
            xx = np.stack([row['legacy'] for row in rows]).astype(np.float64)
            ap = loaded['source_model'].anchor.predict_proba(xx)
            cp, _ = loaded['source_model'].predict_rows(rows)
            assert np.isfinite(ap).all() and np.isfinite(cp).all()
            np.testing.assert_allclose(ap.sum(axis=1), 1., rtol=0, atol=1e-12)
            np.testing.assert_allclose(cp.sum(axis=1), 1., rtol=0, atol=1e-12)
            metrics['arms'][name][split] = {'anchor': probabilities_metrics(rows, ap),
                                           'with_frozen_source_gamma': probabilities_metrics(rows, cp)}
        model_records.append({'name': name, 'path': str(path), 'sha256': sha(path),
                              'anchor_sha256': objsha(anchor), 'only_changed_nested_field': 'source_model.anchor'})
    checked(3, 'serialized_bundle_only_changes_source_anchor', {'models': model_records})
    checked(3, 'fit_and_calibration_posteriors_valid_without_selection', {'calibration_rows': 8411, 'selection_performed': False})
    for path, expected in prereg['source_files_sha256'].items():
        assert sha(path) == expected, ('STOP_PROTECTED_SOURCE_CHANGED_AFTER_FIT', path)
    assert sha(PREREG) == prereg_sha
    checked(3, 'preregistration_and_protected_sources_unchanged', {'prereg_sha256': prereg_sha})
    write_exclusive(METRICS, metrics)
    receipt = {'status': 'COMPLETE_PAIRED_ANCHOR_FIT_AND_NINE_CPU_CONTRACTS',
               'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'preregistration_path': str(PREREG), 'preregistration_sha256': prereg_sha,
               'training_program_sha256': sha(__file__), 'prepared_sha256': DATA_SHA,
               'base_model_sha256': F00_SHA, 'models': model_records,
               'calibration_metrics_path': str(METRICS), 'calibration_metrics_sha256': sha(METRICS),
               'checks': checks, 'fit_rows': 8024, 'coverage_fit_rows': 6093,
               'calibration_rows': 8411, 'training_data_only': True,
               'calibration_used_for_fit_or_selection': False, 'testdev_used_for_fit_or_selection': False,
               'detector_or_ReID_training': False, 'GPU_inference_started': False,
               'formal_metrics': False}
    assert len(checks) == 9 and all(sum(c['round'] == k for c in checks) == 3 for k in [1, 2, 3])
    write_exclusive(RECEIPT, receipt)
    print(json.dumps({'receipt': str(RECEIPT), 'receipt_sha256': sha(RECEIPT),
                      'models': model_records, 'checks': 9}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        if PREREG.exists():
            archived = ROOT / 'artifacts/phase2_failed_attempt1/PHASE2_HEAD_PREREGISTRATION.json'
            assert archived.exists() and sha(PREREG) == sha(archived)
            PREREG.unlink()
        write_exclusive(PREREG, registration())
        print(json.dumps({'preregistration': str(PREREG), 'sha256': sha(PREREG)}))
    else:
        fit()


if __name__ == '__main__':
    main()
