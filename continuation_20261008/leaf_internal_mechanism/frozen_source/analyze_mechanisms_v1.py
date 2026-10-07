"""Closed-loop mechanism tables; reject incomplete runs before writing results.

This author-review layer does not alter inference, model selection or the paper.
Historical controls are rehashed and labelled as reused evidence.
"""
import csv
import hashlib
import json
from pathlib import Path

from analyze_completed import ROOT, METRICS, main as primary_analysis, sha


INVENTORY = Path('/home/chenhc/claude_try_MDMOT/continuation_20261008/'
                 'leaf_internal_mechanism/EXISTING_EVIDENCE_INVENTORY.json')
POLICY_AUDIT = ROOT/'artifacts/egia_existing_controls_audit.json'
ACTIVITY = [
    'frames', 'association_seams', 'factorial_eligible_seams',
    'factorial_empty_seams', 'factorial_assignment_verified_seams',
    'association_edges', 'association_finite_edges',
    'factorial_mask_changed_edges', 'src_nonzero_penalty_edges',
    'src_semantic_penalty_calls', 'src_responsibility_calls',
    'belief_initializations', 'belief_updates',
    'committed_changed_vs_native_seams',
    'association_added_vs_native_pairs', 'association_removed_vs_native_pairs',
    'factorial_illegal_edge_recoveries', 'source_events', 'source_nodes',
    'egia_fusion_events', 'coherence_nonzero_events',
    'selective_native_kept', 'selective_native_rejected',
    'selective_native_abstained', 'selective_new_certificate',
]


def verified_metric(path, expected_sha, reference):
    path = Path(path)
    assert sha(path) == expected_sha, 'METRIC_HASH_CHANGED: ' + str(path)
    value = json.loads(path.read_text())
    assert value['groundtruth_files_sha256'] == reference['groundtruth_files_sha256']
    assert value['expected_sequences'] == 17 and value['eval_split'] == 'test-dev'
    assert set(value['result_files_sha256']) == set(reference['result_files_sha256'])
    folder = Path(value['results_folder'])
    assert all(sha(folder/(s+'.txt')) == h
               for s, h in value['result_files_sha256'].items())
    for s in list(reference['result_files_sha256']) + ['OVERALL']:
        a = value['metrics'][s]
        assert a['num_objects'] == reference['metrics'][s]['num_objects']
        expected_mota = 1 - (a['num_false_positives'] + a['num_misses'] +
                             a['num_switches']) / a['num_objects']
        assert abs(a['mota'] - expected_mota) < 1e-12
    return value


def delta(a, b):
    return {'mota_pp': 100*(a['mota']-b['mota']),
            'idf1_pp': 100*(a['idf1']-b['idf1']),
            **{k: a[k]-b[k] for k in METRICS[2:]}}


def csv_write(path, rows):
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_row(label, m):
    return '| %s | %.4f | %.4f | %d | %d | %d | %d |' % (
        label, 100*m['mota'], 100*m['idf1'],
        *[m[k] for k in METRICS[2:]])


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    correctness = json.loads((ROOT/'artifacts/INPUT_IDENTITY_AUDIT_GATE.json').read_text())
    assert correctness['status'] == 'RESOLVED_DETECTION_METADATA_IDENTITY', \
        'UNRESOLVED_DETECTION_METADATA_IDENTITY'
    for path, expected in correctness['evidence_sha256'].items():
        assert sha(path) == expected, 'IDENTITY_AUDIT_EVIDENCE_CHANGED: ' + path
    receipt = json.loads((ROOT/'artifacts/receipt.json').read_text())
    assert receipt['status'] == 'PASS_PREDECLARED_ONLINE_MECHANISM_ADDENDUM_COMPLETE'
    assert receipt['manifest_sha256'] == sha(ROOT/'manifest.json')
    assert (ROOT/'artifacts/launcher.exit').read_text().strip() == '0'
    assert {a['name'] for a in receipt['arms']} == {a['name'] for a in manifest['arms']}
    assert len(receipt['arms']) == 7
    assert not (ROOT/'artifacts/stop.json').exists()
    for path, expected in manifest['frozen_files_sha256'].items():
        assert sha(path) == expected, 'FROZEN_FILE_CHANGED: ' + path
    reference = json.loads(Path(manifest['full_reference_metrics']).read_text())
    assert all(sha(Path(manifest['gt_root'])/(s+'.txt')) == h
               for s, h in reference['groundtruth_files_sha256'].items())
    new, provenance, activity = {}, {}, []
    for arm, r in zip(manifest['arms'], receipt['arms']):
        name = arm['name']
        assert name == r['name'] and r['status'] == 'PASS_SCORED_ARM_COMPLETE'
        assert r['frames'] == 6635 and r['sequence_count'] == 17
        assert r['gpu_uuid'] == manifest['gpu_uuid']
        assert r['same_process_gpu_verification']['cuda_uuid'] == manifest['gpu_uuid']
        assert r['gt_guard']['blocked_gt_attempts'] == 0
        assert r['gt_guard']['status'] == 'PASS_NO_GT_READ'
        if arm['reference_metrics']:
            assert r['reference_prediction_parity'] == {'matched': 17, 'mismatches': []}
        assert r['input_sha256'] == receipt['arms'][0]['input_sha256']
        path = ROOT/'YOLOX_outputs'/name/'tracking_metrics.json'
        m = verified_metric(path, r['metrics_sha256'], reference)
        assert m['result_files_sha256'] == r['result_sha256']
        assert m['metrics']['OVERALL'] == r['metrics']
        reports = {s: json.loads((path.parent/'track_res'/(s+'.txt.runtime.json')).read_text())
                   for s in manifest['sequence_frames']}
        assert all(reports[s]['frames_seen'] == f for s, f in manifest['sequence_frames'].items())
        totals = {k: sum(x['counts'].get(k, 0) for x in reports.values())
                  for k in set(k for x in reports.values() for k in x['counts'])}
        assert totals == r['runtime_totals']
        assert totals.get('factorial_illegal_edge_recoveries', 0) == 0
        new[name] = m
        provenance[name] = {'path': str(path), 'sha256': sha(path),
                            'kind': 'fresh_online_inference', 'frames': 6635,
                            'prediction_files': 17}
        activity.append({'arm': name, **{k: totals.get(k, 0) for k in ACTIVITY}})
    policy = json.loads(POLICY_AUDIT.read_text())
    assert policy['status'] == 'VERIFIED_EXISTING_CURRENT_F00_POLICY_CONTROLS'
    old_policy = {}
    for a in policy['controls']:
        assert sha(a['receipt_path']) == a['receipt_sha256']
        assert sha(a['model_path']) == a['model_sha256']
        m = verified_metric(a['tracking_metrics_path'], a['tracking_metrics_sha256'], reference)
        assert m['metrics']['OVERALL'] == a['metrics']
        old_policy[a['name']] = m
        provenance[a['name']] = {'path': a['tracking_metrics_path'],
                                 'sha256': a['tracking_metrics_sha256'],
                                 'kind': 'reused_verified_online_control',
                                 'frames': 6635, 'prediction_files': 17}
    assert old_policy['A06_full']['result_files_sha256'] == new['C11_full_reference']['result_files_sha256']
    inventory = json.loads(INVENTORY.read_text())
    history = {}
    inventory_rows = []
    for family in inventory['families']:
        assert sha(family['receipt']['path']) == family['receipt']['sha256']
        for arm in family['arms']:
            src = arm['metrics_source']
            m = verified_metric(src['path'], src['sha256'], reference)
            assert all(m['metrics']['OVERALL'][k] == v for k, v in arm['metrics'].items())
            key = family['family'] + '/' + arm['arm']
            history[key] = m
            provenance[key] = {'path': src['path'], 'sha256': src['sha256'],
                               'kind': 'reused_verified_online_control',
                               'frames': 6635, 'prediction_files': 17}
            inventory_rows.append({'family': family['family'], 'arm': arm['arm'],
                                   'evidence_kind': 'reused_verified_online_control',
                                   **{k: m['metrics']['OVERALL'][k] for k in METRICS}})
    # No inference settings or selection rules depend on these observed results.
    primary_analysis()
    all_metrics = {**new, **old_policy, **history}
    pairs = [
        ('mask_without_penalty', 'C10_mask_only', 'C00_nativecost_reference'),
        ('penalty_without_mask', 'C01_penalty_only', 'C00_nativecost_reference'),
        ('penalty_with_mask', 'C11_full_reference', 'C10_mask_only'),
        ('mask_with_penalty', 'C11_full_reference', 'C01_penalty_only'),
        ('SRC_joint_cost_operators', 'C11_full_reference', 'C00_nativecost_reference'),
        ('selective_with_fusion', 'C11_full_reference', 'E10_fusion_no_selective'),
        ('fusion_without_selective', 'E10_fusion_no_selective', 'A04_coherence'),
        ('fusion_with_selective', 'C11_full_reference', 'A05_selective'),
        ('selective_without_fusion', 'A05_selective', 'A04_coherence'),
        ('paired_hierarchy_minus_flat', 'H_paired_hierarchical', 'H_paired_flat'),
        ('responsibility_soft_minus_hard', 'frozen_SRC_SOE_factorial/F00_full',
         'operator_controls/hard_update'),
        ('recurrence_with_SRC_cost', 'SRC_paths_with_EGIA/F00_replay',
         'SRC_paths_with_EGIA/cost_only'),
        ('recurrence_without_SRC_cost', 'SRC_paths_with_EGIA/update_only',
         'SRC_paths_with_EGIA/no_src'),
        ('source_pairing_intact_minus_shuffled', 'frozen_SRC_SOE_factorial/F00_full',
         'operator_controls/pair_shuffle'),
        ('native_birth_class_gate', 'SRC_native_N0_N5/N1', 'SRC_native_N0_N5/N0'),
        ('SRC_conditional_on_native_birth_gate', 'SRC_native_N0_N5/N4', 'SRC_native_N0_N5/N1'),
    ]
    contrasts = {}
    for label, treatment, control in pairs:
        contrasts[label] = {'treatment': treatment, 'control': control,
                            **delta(all_metrics[treatment]['metrics']['OVERALL'],
                                    all_metrics[control]['metrics']['OVERALL'])}
    interactions = {}
    for label, plus, minus in [
            ('mask_x_penalty', 'penalty_with_mask', 'penalty_without_mask'),
            ('fusion_x_selective', 'selective_with_fusion', 'selective_without_fusion')]:
        interactions[label] = {
            'definition': plus + ' minus ' + minus,
            **{k: contrasts[plus][k]-contrasts[minus][k]
               for k in ['mota_pp', 'idf1_pp'] + METRICS[2:]}}
    csv_write(ROOT/'artifacts/mechanism_activity.csv', activity)
    csv_write(ROOT/'artifacts/retained_historical_controls.csv', inventory_rows)
    csv_write(ROOT/'artifacts/conditional_mechanism_contrasts.csv',
              [{'contrast': k, **v} for k, v in contrasts.items()])
    result = {'status': 'COMPLETE_INDEPENDENTLY_REHASHED_MECHANISM_ANALYSIS',
              'manifest_sha256': sha(ROOT/'manifest.json'),
              'receipt_sha256': sha(ROOT/'artifacts/receipt.json'),
              'analysis_sha256': sha(__file__), 'inventory_sha256': sha(INVENTORY),
              'provenance': provenance, 'contrasts': contrasts,
              'descriptive_factorial_interactions': interactions,
              'selection_performed': False, 'formal_significance_claim': False,
              'GT_guard_scope': 'Python builtins.open/io.open/os.open on designated GT root',
              'interpretation_limits': [
                  'Closed-loop conditional effects at one frozen operating point',
                  'Activity counts alone do not establish accuracy or utility',
                  'Mask affects appearance before min fusion; it is not a whole-edge hard gate',
                  'Responsibility feature definition is fixed, not its state-dependent numeric values',
                  'Paired heads share C and fitting recipe; different L2 parameter geometry remains',
                  'Historical null results are retained; no test-dev winner selection',
                  'Detection metadata identity requires the separate resolved input audit gate',
              ]}
    (ROOT/'artifacts/mechanism_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    header = ['| Cell | MOTA | IDF1 | FP | FN | IDs | FM |',
              '|---|---:|---:|---:|---:|---:|---:|']
    lines = ['', '## SRC mask × learned penalty', '',
             'All four cells retain the same memory initialization/update definition and EGIA. '
             'C00 is the native assignment-cost control within this system.', '', *header]
    for name in ['C00_nativecost_reference', 'C10_mask_only', 'C01_penalty_only', 'C11_full_reference']:
        lines.append(metric_row(name, new[name]['metrics']['OVERALL']))
    lines += ['', '## Fusion × selective policy', '',
              'Coherence gamma is held fixed in every cell. A04/A05 are rehashed existing '
              'online controls; E10/C11 are fresh runs.', '', *header]
    for name in ['A04_coherence', 'A05_selective', 'E10_fusion_no_selective', 'C11_full_reference']:
        lines.append(metric_row(name, all_metrics[name]['metrics']['OVERALL']))
    lines += ['', '## Retained mechanism controls', '',
              '| Contrast | MOTA delta (pp) | IDF1 delta (pp) | FP delta | FN delta | IDs delta |',
              '|---|---:|---:|---:|---:|---:|']
    for name, d in contrasts.items():
        lines.append('| %s | %+.4f | %+.4f | %+d | %+d | %+d |' %
                     (name, d['mota_pp'], d['idf1_pp'], d['num_false_positives'],
                      d['num_misses'], d['num_switches']))
    lines += ['', 'Factorial interaction is the difference between two conditional effects; '
              'it is a descriptive closed-loop contrast, not a significance test. '
              'Mechanism activity CSV counts actual calls and changes against the same-state '
              'native solver; they are not correctness counts. Frames, candidate edges and '
              'tracks are not independent experiment replicates.', '',
              'The paired anchor result tests hierarchical versus flat parameterization under '
              'the common declared fitting recipe. Shared C does not equate L2 geometry. '
              'F00 retains its original recipe and is not used to isolate this contrast. '
              'The manuscript and canonical method are unchanged.', '']
    with (ROOT/'artifacts/RESULTS_FOR_AUTHOR.md').open('a') as f:
        f.write('\n'.join(lines))
    print(json.dumps({'status': result['status'], 'contrasts': len(contrasts),
                      'historical_rows': len(inventory_rows), 'fresh_arms': 7}))


if __name__ == '__main__':
    main()
