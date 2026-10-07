#!/usr/bin/env python3
"""Independent label, numerical and completeness checks for plotted evidence."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import ast
import csv
import gzip
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
OLD = Path('/home/chenhc/src_egia_belief_20260923/artifacts')
RUNTIME = Path('/home/chenhc/leaf_v5_evidence_completion_20261006_v1/runtime')
sys.path.insert(0, str(RUNTIME))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def rows(p):
    with gzip.open(p, 'rt') as f:
        return [json.loads(line) for line in f]


def main():
    r = json.loads((HERE / 'ANALYSIS_RECEIPT.json').read_text())
    assert r['status'] == 'COMPLETE_FROZEN_CALIBRATION_INTERNAL_DIAGNOSTIC'
    assert len(r['sequences']) == 8
    for name, digest in r['source_sha256'].items():
        assert sha(Path(name)) == digest, name
    bundle_path = Path('/home/chenhc/leaf_v5_evidence_completion_20261006_v1/experiment/models/F00_full.pkl')
    bundle = pickle.load(bundle_path.open('rb'))
    head = bundle['observation_model']
    source = Path('/home/chenhc/leaf_v5_evidence_completion_20261006_v1/experiment/tools/label_belief_evidence.py')
    tree = ast.parse(source.read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in ('ground_truth', 'overlap', 'label_frame')]
    namespace = {'np': np, 'defaultdict': defaultdict, 'IOU_MIN': .5,
        'IOU_MARGIN': .05, 'CLUTTER_IOU_MAX': .2, 'IGNORE_IOA_MIN': .5}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), namespace)
    labels_receipt = json.loads((OLD / 'labels_fitcal_v1/receipt.json').read_text())
    sample_n = 0
    sequence_audits = {}
    for seq in r['sequences']:
        cr = json.loads((OLD / 'capture_fitcal_v1' / seq / 'receipt.json').read_text())
        counts = r['per_sequence_audit_only'][seq]
        assert counts['association_records'] == cr['counts']['association']
        assert counts['association_records'] == 3 * cr['counts']['frames']
        assert counts.get('left_censored_track_states_excluded', 0) == 0
        observations = rows(OLD / 'capture_fitcal_v1' / seq / 'observations.jsonl.gz')
        labels = rows(OLD / 'labels_fitcal_v1' / seq / 'observation_labels.jsonl.gz')
        assert len(observations) == len(labels) == cr['counts']['observations']
        gt_path = Path('/raid/datasets/chc_data/VisDrone2019/train/annotations') / (seq + '.txt')
        assert sha(gt_path) == labels_receipt['gt_sha256'][str(gt_path)]
        gt = namespace['ground_truth'](gt_path)
        local_n = 0
        step = max(1, len(observations) // 32)
        for i in range(0, len(observations), step):
            observation, stored = observations[i], labels[i]
            actual = namespace['label_frame']([observation], gt.get(observation['source_frame'], []))[0]
            for key in ('known', 'kind', 'gt_id', 'gt_class', 'group', 'reason'):
                assert actual[key] == stored[key], (seq, i, key)
            assert abs(actual['gt_iou'] - stored['gt_iou']) < 1e-12
            cls, confidence = observation['predicted_class'], observation['class_confidence']
            indicator = np.eye(10)[cls]
            logit = np.log(max(confidence, 1e-5) / max(1-confidence, 1e-5))
            features = np.r_[indicator, indicator * logit]
            scaler, logistic = head.named_steps['standardscaler'], head.named_steps['logisticregression']
            standardized = (features - scaler.mean_) / scaler.scale_
            p_vehicle = expit(float(logistic.coef_[0] @ standardized + logistic.intercept_[0]))
            independent = np.array([1-p_vehicle, p_vehicle])
            assert np.allclose(head.predict_proba(features[None])[0], independent, atol=1e-14)
            local_n += 1
        sample_n += local_n
        sequence_audits[seq] = {'systematic_observation_samples': local_n,
            'gt_sha256': sha(gt_path), 'complete_association_records': counts['association_records'],
            'left_censored_states': 0}
    example = r['representative_operator_check']
    prior = np.array(r['group_prior'])
    b, z, base = map(np.array, [example['beliefs'], example['observations'], example['base']])
    independent_cost = np.empty(base.shape)
    for i in range(len(b)):
        for j in range(len(z)):
            factor = sum(float(b[i, k]) * float(z[j, k]) / float(prior[k]) for k in range(2))
            penalty = min(1., max(0., 1-factor))
            independent_cost[i, j] = base[i, j] + (1-base[i, j]) * penalty
    assert np.allclose(independent_cost, example['src'], atol=1e-14)
    table = list(csv.DictReader((HERE / 'COST_CDF.csv').open()))
    for kind, summary in r['cost_summary'].items():
        for name in ('base', 'src'):
            rr = [row for row in table if row['kind'] == kind and row['operator'] == name]
            assert all(int(row['n']) == summary['n'] for row in rr)
            cdf = np.array([float(row['cdf']) for row in rr])
            assert np.all(np.diff(cdf) >= 0) and cdf[-1] == 1
            threshold = next(float(row['cdf']) for row in rr if abs(float(row['cost'])-.8) < 1e-12)
            expected = 1 if name == 'base' else summary['remaining_eligible_fraction']
            assert abs(threshold - expected) < 1e-12
    checks = {
        'source_hashes_unchanged': True, 'all_eight_calibration_sequences_included': True,
        'all_saved_association_records_consumed': True, 'all_beliefs_have_recorded_birth_state': True,
        'current_annotation_hashes_match_frozen_labels': True,
        'systematic_label_spot_check_against_current_original_gt': True,
        'independent_manual_head_probability_reconstruction': True,
        'independent_scalar_cost_reconstruction': True, 'empirical_cdf_and_threshold_counts_consistent': True}
    out = {'status': 'PASS_INDEPENDENT_INTERNAL_MECHANISM_AUDIT',
        'checks': checks, 'systematic_sample_n': sample_n, 'per_sequence': sequence_audits,
        'label_source_sha256': sha(source), 'analysis_receipt_sha256': sha(HERE / 'ANALYSIS_RECEIPT.json'),
        'audit_script_sha256': sha(Path(__file__)), 'formal_mot_metrics': False, 'gpu_used': False}
    (HERE / 'INDEPENDENT_AUDIT.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'status': out['status'], 'checks': len(checks), 'samples': sample_n}))


if __name__ == '__main__':
    main()
