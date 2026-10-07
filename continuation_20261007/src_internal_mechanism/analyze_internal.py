#!/usr/bin/env python3
"""Frozen calibration diagnostics of the current SRC heads and operators.

No fitting, tracker rerun, threshold search, GPU use or benchmark claim.
Beliefs are causally reconstructed on the saved H2 trajectory. Labels are
applied after cost evaluation and never enter the operator state.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import gzip
import hashlib
import json
import pickle
import sys
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = Path('/home/chenhc/src_egia_belief_20260923/artifacts')
RUNTIME = Path('/home/chenhc/leaf_v5_evidence_completion_20261006_v1/runtime')
BUNDLE = Path('/home/chenhc/leaf_v5_evidence_completion_20261006_v1/experiment/models/F00_full.pkl')
sys.path.insert(0, str(RUNTIME))
from belief.features import semantic_features_batch, responsibility_features
from belief.models import semantic_update, semantic_penalty


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def save_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def metrics(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-9, 1.)
    p /= p.sum(1, keepdims=True)
    return {'n': len(y), 'nll': float(-np.log(p[np.arange(len(y)), y]).mean()),
            'brier_two_class': float(np.mean(np.sum((p - np.eye(2)[y]) ** 2, axis=1))),
            'accuracy': float(np.mean(p.argmax(1) == y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit-sequences', type=int)
    args = ap.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    source_sha = {}

    def record(path):
        source_sha[str(path)] = sha(path)
        return source_sha[str(path)]

    bundle = pickle.load(BUNDLE.open('rb'))
    original = pickle.load((OLD / 'train_belief_v2/model.pkl').open('rb'))
    for key in ('observation_model', 'responsibility_model'):
        for step in ('standardscaler', 'logisticregression'):
            current, old = bundle[key].named_steps[step], original[key].named_steps[step]
            for attr in ('mean_', 'scale_', 'coef_', 'intercept_', 'classes_'):
                if hasattr(current, attr):
                    assert np.array_equal(getattr(current, attr), getattr(old, attr)), (key, step, attr)
    assert np.array_equal(bundle['group_prior'], original['group_prior'])
    head, responsibility = bundle['observation_model'], bundle['responsibility_model']
    prior = np.asarray(bundle['group_prior'], float)
    assert np.array_equal(head.classes_, [0, 1])
    assert np.array_equal(responsibility.classes_, [0, 1])
    for p in (BUNDLE, OLD / 'train_belief_v2/model.pkl',
              OLD / 'train_belief_v2/receipt.json', OLD / 'train_belief_v2/prepare_receipt.json',
              OLD / 'capture_fitcal_v1/receipt.json', OLD / 'labels_fitcal_v1/receipt.json',
              RUNTIME / 'belief/features.py', RUNTIME / 'belief/models.py',
              RUNTIME / 'belief/runtime.py', Path(__file__)):
        record(p)
    labels_receipt = json.loads((OLD / 'labels_fitcal_v1/receipt.json').read_text())
    capture_receipt = json.loads((OLD / 'capture_fitcal_v1/receipt.json').read_text())
    assert labels_receipt['capture_receipt_sha256'] == record(OLD / 'capture_fitcal_v1/receipt.json')
    assert capture_receipt['gt_read'] is False and capture_receipt['heldout_read'] is False
    sequences = [r['sequence'] for r in labels_receipt['results'] if r['split'] == 'calibration']
    assert len(sequences) == 8
    if args.limit_sequences:
        sequences = sequences[:args.limit_sequences]
    prepared_path = OLD / 'train_belief_v2/prepared.pkl'
    prepared_receipt = json.loads((OLD / 'train_belief_v2/prepare_receipt.json').read_text())
    assert record(prepared_path) == prepared_receipt['prepared_sha256']
    prepared = pickle.load(prepared_path.open('rb'))
    cal = prepared['calibration']['observation']
    fit_flights = {r[2] for r in prepared['fit']['observation']}
    cal_flights = {r[2] for r in cal}
    assert fit_flights.isdisjoint(cal_flights)
    x = np.stack([r[0] for r in cal])
    y = np.array([r[1] for r in cal])
    raw = np.stack([r[3] for r in cal])
    learned = head.predict_proba(x)
    reference = json.loads((OLD / 'train_belief_v2/metrics.json').read_text())['semantic']
    evidence_metrics = {'raw': metrics(y, raw), 'learned': metrics(y, learned)}
    assert abs(evidence_metrics['learned']['nll'] - reference['learned']['nll']) < 1e-12
    assert abs(evidence_metrics['raw']['nll'] - reference['raw_confidence']['nll']) < 1e-12
    reliability_rows = []
    for name, prob in [('raw', raw), ('learned', learned)]:
        h = prob[:, 0]
        bins = np.minimum((h * 10).astype(int), 9)
        for bi in range(10):
            mask = bins == bi
            if mask.any():
                reliability_rows.append({'model': name, 'bin': bi, 'n': int(mask.sum()),
                    'mean_predicted_human': float(h[mask].mean()),
                    'observed_human_fraction': float(np.mean(y[mask] == 0))})
    with (HERE / 'SEMANTIC_RELIABILITY.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(reliability_rows[0]))
        writer.writeheader(); writer.writerows(reliability_rows)
    del prepared, cal, x, raw, learned

    classes = ['same_owner', 'different_owner_same_group', 'different_owner_cross_group']
    values = {kind: {'base': [], 'src': [], 'penalty': []} for kind in classes}
    stats = Counter()
    per_sequence = {}
    representative = None
    for seq in sequences:
        capture = OLD / 'capture_fitcal_v1' / seq
        labels = OLD / 'labels_fitcal_v1' / seq
        cr = json.loads((capture / 'receipt.json').read_text())
        lr = json.loads((labels / 'receipt.json').read_text())
        assert lr['split'] == 'calibration'
        assert lr['capture_receipt_sha256'] == record(capture / 'receipt.json')
        for filename, digest in cr['files_sha256'].items():
            assert record(capture / filename) == digest
        for filename, digest in lr['files_sha256'].items():
            assert record(labels / filename) == digest
        observations = list(rows(capture / 'observations.jsonl.gz'))
        probabilities = head.predict_proba(semantic_features_batch(observations))
        p_index = {(int(r['source_frame']), int(r['detection_ordinal'])): p
                   for r, p in zip(observations, probabilities)}
        label_index = {(int(r['source_frame']), int(r['detection_ordinal'])): r
                       for r in rows(labels / 'observation_labels.jsonl.gz')}
        assert len(p_index) == len(observations)
        del observations, probabilities
        belief = {}
        exact_state = {}
        local = Counter()
        for event, label in zip_longest(rows(capture / 'association.jsonl.gz'),
                                       rows(labels / 'association_labels.jsonl.gz')):
            assert event is not None and label is not None
            assert all(event[k] == label[k] for k in ('sequence', 'frame', 'source_frame', 'stage'))
            tracks, detections = event['tracks'], event['detections']
            assert [t['track_id'] for t in tracks] == label['track_ids']
            assert [d['detection_ordinal'] for d in detections] == label['detection_ordinals']
            local['association_records'] += 1
            for track in tracks:
                tid = int(track['track_id'])
                if tid not in belief:
                    last = track.get('last_observation')
                    exact = last is not None and int(last['frame']) == int(track['start_frame'])
                    # Only a recorded birth observation establishes exact state.
                    if exact:
                        belief[tid] = p_index[(int(last['source_frame']), int(last['detection_ordinal']))].copy()
                        exact_state[tid] = True
                        local['birth_states_reconstructed'] += 1
                    else:
                        belief[tid] = np.array([.5, .5])
                        exact_state[tid] = False
                        local['left_censored_track_states_excluded'] += 1
            if not tracks or not detections:
                assert not event['matches']
                continue
            base = np.asarray(event['base_cost'], float)
            assert base.shape == (len(tracks), len(detections))
            assert np.isfinite(base).all() and np.all((base >= 0) & (base <= 1))
            z = np.array([p_index[(int(d['source_frame']), int(d['detection_ordinal']))] for d in detections])
            b = np.array([belief[int(t['track_id'])] for t in tracks])
            u = semantic_penalty(b, z / prior)
            cost = base + (1 - base) * u
            assert np.all(cost >= base) and np.all(cost <= 1.)
            assert np.allclose(cost[u == 0], base[u == 0])
            # Cost/state above are computed before reading any evaluation label.
            if event['stage'] == 0:
                tau = float(event['threshold'])
                assert abs(tau - .8) < 1e-12
                local['first_stage_calls'] += 1
                for ti, track in enumerate(tracks):
                    if not exact_state[int(track['track_id'])]:
                        continue
                    last = track.get('last_observation')
                    if last is None:
                        continue
                    tl = label_index[(int(last['source_frame']), int(last['detection_ordinal']))]
                    if tl['kind'] != 'foreground':
                        continue
                    assert tl['gt_id'] == label['track_gt_ids'][ti]
                    for di, det in enumerate(detections):
                        dl = label_index[(int(det['source_frame']), int(det['detection_ordinal']))]
                        local['first_stage_edges_from_known_owner'] += 1
                        if dl['kind'] != 'foreground':
                            local['unknown_or_clutter_detections_excluded'] += 1
                            continue
                        if base[ti, di] > tau:
                            local['already_ineligible_base_edges_excluded'] += 1
                            continue
                        if tl['gt_id'] == dl['gt_id']:
                            kind = 'same_owner'
                        elif tl['group'] == dl['group']:
                            kind = 'different_owner_same_group'
                        else:
                            kind = 'different_owner_cross_group'
                        for k, matrix in [('base', base), ('src', cost), ('penalty', u)]:
                            values[kind][k].append(float(matrix[ti, di]))
                        local[kind] += 1
                # A reproducible example is selected by first occurrence, not gain.
                if representative is None:
                    known = [i for i, t in enumerate(tracks)
                        if exact_state[int(t['track_id'])] and t.get('last_observation') is not None]
                    if len(known) >= 2 and len(detections) >= 2:
                        ti = known[:2]; di = list(range(2))
                        representative = {'sequence': seq, 'frame': event['source_frame'],
                            'track_ids': [tracks[i]['track_id'] for i in ti],
                            'detection_ordinals': [detections[j]['detection_ordinal'] for j in di],
                            'beliefs': b[ti].tolist(), 'observations': z[di].tolist(),
                            'base': base[np.ix_(ti, di)].tolist(),
                            'src': cost[np.ix_(ti, di)].tolist(), 'penalty': u[np.ix_(ti, di)].tolist()}
            pairs = np.asarray(event['matches'], int).reshape(-1, 2)
            if len(pairs):
                rx = np.array([responsibility_features(base, event['appearance_distance'],
                    event['geometry_cost'], ti, di, event['threshold'], event['iou_only']) for ti, di in pairs])
                r = responsibility.predict_proba(rx)[:, 1]
                for (ti, di), responsibility_value in zip(pairs, r):
                    tid = int(tracks[ti]['track_id'])
                    belief[tid] = semantic_update(belief[tid], z[di] / prior, responsibility_value)
                    local['recorded_committed_updates_replayed'] += 1
        per_sequence[seq] = dict(local)
        stats.update(local)
        print(json.dumps({'sequence': seq, 'counts': dict(local)}), flush=True)

    summaries = {}
    cdf_rows = []
    delta_rows = []
    grid = np.unique(np.r_[np.linspace(0, 1, 101), .8])
    for kind in classes:
        a = {k: np.asarray(v, float) for k, v in values[kind].items()}
        assert len(a['base']) > 0, kind
        summaries[kind] = {'n': len(a['base']),
            'base_median': float(np.median(a['base'])), 'src_median': float(np.median(a['src'])),
            'mean_added_cost': float((a['src'] - a['base']).mean()),
            'unchanged_fraction': float(np.mean(a['src'] == a['base'])),
            'remaining_eligible_fraction': float(np.mean(a['src'] <= .8)),
            'ineligible_fraction': float(np.mean(a['src'] > .8)),
            'penalty_quantiles': np.quantile(a['penalty'], [0, .25, .5, .75, 1]).tolist()}
        for name in ['base', 'src']:
            sorted_values = np.sort(a[name])
            for g in grid:
                cdf_rows.append({'kind': kind, 'operator': name, 'cost': float(g),
                    'cdf': float(np.searchsorted(sorted_values, g, side='right') / len(sorted_values)),
                    'n': len(sorted_values)})
        delta = np.sort(a['src'] - a['base'])
        for q in np.linspace(0, 1, 101):
            delta_rows.append({'kind': kind, 'quantile': float(q), 'added_cost': float(np.quantile(delta, q)), 'n': len(delta)})
    for filename, table in [('COST_CDF.csv', cdf_rows), ('COST_DELTA_QUANTILES.csv', delta_rows)]:
        with (HERE / filename).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    receipt = {'status': 'COMPLETE_FROZEN_CALIBRATION_INTERNAL_DIAGNOSTIC' if len(sequences) == 8 else 'PARTIAL_SMOKE_ONLY',
        'model_parameter_identity': 'Current F00 SRC heads equal frozen train_belief_v2 heads exactly',
        'group_prior': prior.tolist(), 'calibration_observation_metrics': evidence_metrics,
        'semantic_calibration_rows': len(y), 'fit_calibration_flights_disjoint': True,
        'sequences': sequences, 'cost_summary': summaries, 'counts': dict(stats),
        'per_sequence_audit_only': per_sequence, 'representative_operator_check': representative,
        'source_sha256': source_sha,
        'contracts': {'stage': 0, 'association_threshold': .8, 'base_legal_edges_only': True,
            'known_foreground_both_endpoints_only': True, 'groups': ['human', 'vehicle'],
            'owner_label': 'last committed detection with unique class-agnostic GT geometry match',
            'belief_initialization': 'recorded birth observation only; all left-censored states excluded',
            'belief_update': 'exact current operator on recorded H2 committed edges, before next seam',
            'cost_evaluation_before_gt_label': True, 'bounded_non_decreasing_cost_checked': True,
            'no_fitting_or_parameter_selection': True, 'gpu_used': False,
            'raw_probability': 'confidence mass on predicted coarse group plus uniform residual mass',
            'cdf_weighting': 'each eligible edge once; empirical descriptive CDF, no IID or significance claim'},
        'limitations': [
            'Counterfactual SRC operator on frozen H2 visited states; no changed assignment or trajectory is executed.',
            'Owner correspondence follows last accepted observation; this is not a global identity oracle.',
            'Eligibility changes do not establish correctness of a global one-to-one assignment.',
            'Semantic evidence separates coarse human/vehicle groups and is not fine-grained identity evidence.',
            'Calibration diagnostics establish input/operator behavior, not formal tracking improvement.',
            'Responsibility-weighted temporal update is used for state reconstruction; its independent effectiveness is not claimed.']}
    save_json('ANALYSIS_RECEIPT.json', receipt)
    print(json.dumps({'status': receipt['status'], 'cost_summary': summaries, 'semantic': evidence_metrics}), flush=True)


if __name__ == '__main__':
    main()
