#!/usr/bin/env python3
"""Find auditable small assignment components on frozen calibration states."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import numpy as np
import lap
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'src_internal_mechanism'
sys.path.insert(0, str(PREVIOUS))
from analyze_internal import OLD, BUNDLE, RUNTIME, rows, sha
from belief.features import semantic_features_batch, responsibility_features
from belief.models import semantic_penalty, semantic_update


def assign(cost, threshold):
    _, x, y = lap.lapjv(np.asarray(cost, float), extend_cost=True, cost_limit=threshold)
    return {(i, int(j)) for i, j in enumerate(x) if j >= 0}


def known_score(pairs, tracks, detections):
    counts = Counter()
    for i, j in pairs:
        t, d = tracks[i], detections[j]
        if t is None or d is None or t['kind'] != 'foreground' or d['kind'] != 'foreground':
            counts['unknown_pairs'] += 1
        elif t['gt_id'] == d['gt_id']:
            counts['same_owner_pairs'] += 1
        else:
            counts['different_owner_pairs'] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--limit-sequences', type=int)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    previous = json.loads((PREVIOUS / 'ANALYSIS_RECEIPT.json').read_text())
    for path, digest in previous['source_sha256'].items():
        assert sha(Path(path)) == digest
    bundle = pickle.load(BUNDLE.open('rb'))
    head, responsibility, prior = bundle['observation_model'], bundle['responsibility_model'], np.asarray(bundle['group_prior'], float)
    sequences = previous['sequences'][:args.limit_sequences] if args.limit_sequences else previous['sequences']
    selected = {}
    totals = Counter(); per_sequence = {}
    for seq in sequences:
        cd, ld = OLD / 'capture_fitcal_v1' / seq, OLD / 'labels_fitcal_v1' / seq
        observations = list(rows(cd / 'observations.jsonl.gz'))
        z_all = head.predict_proba(semantic_features_batch(observations))
        p_index = {(int(row['source_frame']), int(row['detection_ordinal'])): p for row, p in zip(observations, z_all)}
        labels = {(int(row['source_frame']), int(row['detection_ordinal'])): row for row in rows(ld / 'observation_labels.jsonl.gz')}
        del observations, z_all
        belief = {}; local = Counter()
        for event, annotation in zip_longest(rows(cd / 'association.jsonl.gz'), rows(ld / 'association_labels.jsonl.gz')):
            assert event is not None and annotation is not None
            assert all(event[k] == annotation[k] for k in ('sequence', 'frame', 'source_frame', 'stage'))
            tracks, detections = event['tracks'], event['detections']
            local['association_records'] += 1
            for t in tracks:
                tid = int(t['track_id'])
                if tid not in belief:
                    last = t.get('last_observation')
                    assert last is not None and last['frame'] == t['start_frame'], 'UNOBSERVED_BIRTH_STATE'
                    belief[tid] = p_index[(int(last['source_frame']), int(last['detection_ordinal']))].copy()
            if not tracks or not detections:
                continue
            base = np.asarray(event['base_cost'], float)
            z = np.array([p_index[(int(d['source_frame']), int(d['detection_ordinal']))] for d in detections])
            b = np.array([belief[int(t['track_id'])] for t in tracks])
            u = semantic_penalty(b, z / prior)
            src = base + (1-base) * u
            if event['stage'] == 0:
                tau = float(event['threshold']); assert tau == .8
                pb, ps = assign(base, tau), assign(src, tau)
                local['first_stage_calls'] += 1
                if pb != ps:
                    local['base_vs_src_assignment_changed_calls'] += 1
                    tlabels = [labels[(int(t['last_observation']['source_frame']), int(t['last_observation']['detection_ordinal']))]
                               if t.get('last_observation') is not None else None for t in tracks]
                    dlabels = [labels[(int(d['source_frame']), int(d['detection_ordinal']))] for d in detections]
                    legal = csr_matrix(base <= tau)
                    graph = bmat([[None, legal], [legal.T, None]], format='csr')
                    nc, component = connected_components(graph, directed=False)
                    for ci in range(nc):
                        ti = np.flatnonzero(component[:len(tracks)] == ci).tolist()
                        di = np.flatnonzero(component[len(tracks):] == ci).tolist()
                        if not ti or not di:
                            continue
                        bs = {(i, j) for i, j in pb if i in ti and j in di}
                        ss = {(i, j) for i, j in ps if i in ti and j in di}
                        if bs == ss:
                            continue
                        local['changed_legal_components'] += 1
                        old_score = known_score(bs, tlabels, dlabels)
                        new_score = known_score(ss, tlabels, dlabels)
                        gain = new_score.get('same_owner_pairs', 0) - old_score.get('same_owner_pairs', 0)
                        false_reduction = old_score.get('different_owner_pairs', 0) - new_score.get('different_owner_pairs', 0)
                        fully_known = all(tlabels[i] is not None and tlabels[i]['kind'] == 'foreground' for i in ti) and all(dlabels[j]['kind'] == 'foreground' for j in di)
                        beneficial = fully_known and gain >= 0 and false_reduction >= 0 and (gain > 0 or false_reduction > 0)
                        if beneficial:
                            local['fully_known_beneficial_components'] += 1
                        elif fully_known and (gain < 0 or false_reduction < 0):
                            local['fully_known_adverse_or_tradeoff_components'] += 1
                        else:
                            local['unknown_or_no_owner_score_change_components'] += 1
                        if not beneficial or max(len(ti), len(di)) > 3:
                            continue
                        key = f'{len(ti)}x{len(di)}'
                        if key in selected:
                            continue
                        bm, sm, um = base[np.ix_(ti, di)], src[np.ix_(ti, di)], u[np.ix_(ti, di)]
                        li, lj = {v: k for k, v in enumerate(ti)}, {v: k for k, v in enumerate(di)}
                        before = {(li[i], lj[j]) for i, j in bs}
                        after = {(li[i], lj[j]) for i, j in ss}
                        assert before == assign(bm, tau) and after == assign(sm, tau)
                        assert not np.any(base[np.ix_(ti, [j for j in range(len(detections)) if j not in di])] <= tau)
                        assert not np.any(base[np.ix_([i for i in range(len(tracks)) if i not in ti], di)] <= tau)
                        selected[key] = {
                            'sequence': seq, 'frame': int(event['source_frame']), 'tracker_frame': int(event['frame']), 'stage': 0,
                            'full_pool_shape': list(base.shape), 'threshold': tau, 'prior': prior.tolist(),
                            'global_row_indices': ti, 'global_column_indices': di,
                            'tracks': [tracks[i] for i in ti], 'detections': [detections[j] for j in di],
                            'track_labels': [tlabels[i] for i in ti], 'detection_labels': [dlabels[j] for j in di],
                            'beliefs': b[ti].tolist(), 'observation_probabilities': z[di].tolist(),
                            'base_cost': bm.tolist(), 'semantic_penalty': um.tolist(), 'src_cost': sm.tolist(),
                            'added_cost': (sm-bm).tolist(),
                            'base_assignment': sorted(map(list, before)), 'src_assignment': sorted(map(list, after)),
                            'base_owner_score': old_score, 'src_owner_score': new_score,
                            'closed_legal_component': True, 'standalone_and_full_pool_assignments_identical': True,
                            'selection': 'first fully known beneficial closed legal component of this shape in sorted calibration streams'}
                        print(json.dumps({'found': key, 'sequence': seq, 'frame': event['source_frame'],
                            'base_assignment': selected[key]['base_assignment'], 'src_assignment': selected[key]['src_assignment'],
                            'base_score': old_score, 'src_score': new_score}), flush=True)
            pairs = np.asarray(event['matches'], int).reshape(-1, 2)
            if len(pairs):
                rx = np.array([responsibility_features(base, event['appearance_distance'], event['geometry_cost'], i, j,
                                                      event['threshold'], event['iou_only']) for i, j in pairs])
                rr = responsibility.predict_proba(rx)[:, 1]
                for (i, j), rv in zip(pairs, rr):
                    tid = int(tracks[i]['track_id'])
                    belief[tid] = semantic_update(belief[tid], z[j] / prior, rv)
        expected = previous['per_sequence_audit_only'][seq]['association_records']
        assert local['association_records'] == expected
        totals.update(local); per_sequence[seq] = dict(local)
        print(json.dumps({'completed': seq, 'counts': dict(local)}), flush=True)
    receipt = {'status': 'COMPLETE_ALL_CALIBRATION_CASE_SEARCH' if len(sequences) == 8 else 'PARTIAL_CASE_SEARCH',
        'sequences': sequences, 'counts': dict(totals), 'per_sequence_audit_only': per_sequence,
        'cases': selected, 'source_receipt_sha256': sha(PREVIOUS / 'ANALYSIS_RECEIPT.json'),
        'source_sha256': previous['source_sha256'], 'script_sha256': sha(Path(__file__)),
        'assignment_solver': 'same host objective: lap.lapjv, extend_cost=True, cost_limit=0.80',
        'limits': ['Costs, heads and beliefs follow the frozen calibration operator diagnostic.',
            'The comparison is pre-semantic base C versus current SRC C_tilde, not the historical native semantic rule.',
            'Assignments are same-state counterfactuals, not a new online tracking run.',
            'Case selection explicitly requests a favorable illustrative example; counts retain adverse/tradeoff and unknown components.',
            'Only closed legal components are eligible: cropping cannot change the displayed assignment.'],
        'training': False, 'gpu_used': False, 'formal_metrics': False}
    (HERE / 'CASE_SEARCH_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'cases': list(selected), 'counts': dict(totals)}), flush=True)


if __name__ == '__main__':
    main()
