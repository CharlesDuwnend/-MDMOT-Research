"""Analyze corrected cells only; archive historical scores outside scientific contrasts."""
import json
import pickle
from pathlib import Path

from analyze_completed import (
    ROOT, METRICS, SUMMARY_METRICS, PAIRS, compare_summary, csv_write, delta,
    read, require, report_completed, sha, validate_completed, verified_metric,
)

INVENTORY = Path('/home/chenhc/claude_try_MDMOT/continuation_20261008/'
                 'leaf_internal_mechanism/EXISTING_EVIDENCE_INVENTORY.json')
POLICY_AUDIT = ROOT / 'artifacts/egia_existing_controls_audit.json'
HISTORICAL_KIND = 'historical_metadata_contaminated'
ACTIVITY = [
    'association_seams', 'corrected_reference_seams_verified',
    'factorial_native_assignment_calls', 'factorial_assignment_verified_seams',
    'factorial_eligible_seams', 'factorial_empty_seams',
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
HISTORICAL_NULL_PAIRS = [
    ('responsibility_soft_minus_hard', 'frozen_SRC_SOE_factorial/F00_full',
     'operator_controls/hard_update'),
    ('recurrence_with_SRC_cost', 'SRC_paths_with_EGIA/F00_replay',
     'SRC_paths_with_EGIA/cost_only'),
    ('recurrence_without_SRC_cost', 'SRC_paths_with_EGIA/update_only',
     'SRC_paths_with_EGIA/no_src'),
]


def metric_row(label, metric):
    return '| %s | %.4f | %.4f | %d | %d | %d | %d |' % (
        label, 100 * metric['mota'], 100 * metric['idf1'],
        *[metric[k] for k in METRICS[2:]])


def verify_paired_recipe(manifest):
    """Inspect frozen scaler/estimator payloads; never fit or select a model."""
    import numpy as np
    path = Path(manifest['phase2_fitting_receipt'])
    require(str(path) in manifest['frozen_files_sha256'], 'PAIRED_FIT_RECEIPT_NOT_FROZEN')
    receipt = read(path)
    require(receipt['status'] == 'COMPLETE_PAIRED_ANCHOR_FIT_AND_NINE_CPU_CONTRACTS',
            'PAIRED_FIT_CONTRACT_INCOMPLETE')
    require(receipt['testdev_used_for_fit_or_selection'] is False and
            receipt['calibration_used_for_fit_or_selection'] is False,
            'PAIRED_FIT_SELECTION_BOUNDARY_CHANGED')
    require(receipt['fit_rows'] == 8024 and receipt['coverage_fit_rows'] == 6093,
            'PAIRED_FIT_ROW_CONTRACT_CHANGED')
    checks = {check['name']: check for check in receipt['checks']}
    require(len(checks) == 9 and all(c['status'] == 'PASS' for c in checks.values()),
            'PAIRED_FIT_NINE_CONTRACTS_NOT_CLOSED')
    weights = checks['independent_weight_reconstruction']['detail']
    require(weights['no_subset_normalization'] is True, 'COVERAGE_WEIGHTS_RENORMALIZED')
    models = {}
    for name in ['Paired_Hierarchical', 'Paired_Flat']:
        model_path = ROOT / 'models' / (name + '.pkl')
        recorded = next(m for m in receipt['models'] if m['name'] == name)
        require(sha(model_path) == recorded['sha256'], 'PAIRED_HEAD_PAYLOAD_CHANGED')
        with model_path.open('rb') as stream:
            models[name] = pickle.load(stream)
    hierarchical = models['Paired_Hierarchical']['source_model'].anchor
    flat = models['Paired_Flat']['source_model'].anchor
    pipelines = [hierarchical.foreground_model, hierarchical.coverage_model, flat]
    scalers = [pipeline.steps[0][1] for pipeline in pipelines]
    first = scalers[0]
    for scaler in scalers:
        for field in ['mean_', 'scale_', 'var_', 'n_samples_seen_', 'n_features_in_']:
            require(np.array_equal(np.asarray(getattr(scaler, field)),
                                   np.asarray(getattr(first, field))),
                    'PAIRED_FIT_COMMON_SCALER_CHANGED: ' + field)
        require(int(scaler.n_samples_seen_) == 8024, 'SCALER_NOT_FIT_ON_COMMON_FIT_ROWS')
    scaler_hash = object_sha(first)
    require(all(object_sha(scaler) == scaler_hash for scaler in scalers),
            'PAIRED_SCALER_PAYLOADS_DIFFER')
    require(scaler_hash == checks['shared_fit_only_scaler_in_all_three_pipelines']
            ['detail']['scaler_object_sha256'], 'PAIRED_SCALER_RECEIPT_CHANGED')
    parameters = checks['fixed_estimator_parameters_and_convergence']['detail']['parameters']
    for pipeline in pipelines:
        estimator = pipeline.steps[-1][1]
        for key, expected in parameters.items():
            require(getattr(estimator, key) == expected, 'PAIRED_ESTIMATOR_RECIPE_CHANGED: ' + key)
    return {'receipt_path': str(path), 'receipt_sha256': sha(path),
            'common_scaler_object_sha256': scaler_hash,
            'common_event_weights_sha256': weights['common_weight_sha256'],
            'common_weights_verification_scope': 'Frozen pre-fit source and independently reconstructed training-receipt evidence; online analysis does not refit weights.',
            'fit_rows': 8024, 'coverage_fit_rows': 6093,
            'metadata_repair_refit': False, 'testdev_selection': False}


def object_sha(value):
    """Use the frozen fitting receipt's canonical object encoding."""
    import hashlib
    import numpy as np

    def canonical(item):
        if isinstance(item, np.ndarray):
            data = canonical(item.tolist()) if item.dtype.hasobject else item.tobytes().hex()
            return ['array', str(item.dtype), list(item.shape), data]
        if isinstance(item, np.generic):
            return ['scalar', str(item.dtype), item.item()]
        if item is None or isinstance(item, (str, int, float, bool)):
            return [type(item).__name__, item]
        if isinstance(item, (list, tuple)):
            return [type(item).__name__, [canonical(v) for v in item]]
        if isinstance(item, dict):
            return ['dict', [[str(k), canonical(v)] for k, v in
                             sorted(item.items(), key=lambda pair: str(pair[0]))]]
        if hasattr(item, '__dict__'):
            return ['object', type(item).__module__ + '.' + type(item).__qualname__,
                    canonical(item.__dict__)]
        raise TypeError(type(item))

    encoded = json.dumps(canonical(value), sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_history(reference):
    """Preserve twenty old score rows, never merge them into corrected cells."""
    policy = read(POLICY_AUDIT)
    inventory = read(INVENTORY)
    history, provenance, rows = {}, {}, []
    for control in policy['controls']:
        require(sha(control['receipt_path']) == control['receipt_sha256'],
                'HISTORICAL_POLICY_RECEIPT_CHANGED')
        require(sha(control['model_path']) == control['model_sha256'],
                'HISTORICAL_POLICY_MODEL_CHANGED')
        value = verified_metric(control['tracking_metrics_path'],
                                control['tracking_metrics_sha256'], reference)
        compare_summary(value['metrics']['OVERALL'], control['metrics'], control['name'])
        key = 'historical_policy/' + control['name']
        history[key] = value
        provenance[key] = {'path': control['tracking_metrics_path'],
                           'sha256': control['tracking_metrics_sha256'],
                           'kind': HISTORICAL_KIND, 'scientific_use': False}
        rows.append({'family': 'historical_policy', 'arm': control['name'],
                     'evidence_kind': HISTORICAL_KIND, 'scientific_use': False,
                     **{k: value['metrics']['OVERALL'][k] for k in SUMMARY_METRICS}})
    for family in inventory['families']:
        require(sha(family['receipt']['path']) == family['receipt']['sha256'],
                'HISTORICAL_FAMILY_RECEIPT_CHANGED')
        for arm in family['arms']:
            source = arm['metrics_source']
            value = verified_metric(source['path'], source['sha256'], reference)
            compare_summary(value['metrics']['OVERALL'], arm['metrics'], arm['arm'])
            key = family['family'] + '/' + arm['arm']
            history[key] = value
            provenance[key] = {'path': source['path'], 'sha256': source['sha256'],
                               'kind': HISTORICAL_KIND, 'scientific_use': False}
            rows.append({'family': family['family'], 'arm': arm['arm'],
                         'evidence_kind': HISTORICAL_KIND, 'scientific_use': False,
                         **{k: value['metrics']['OVERALL'][k] for k in SUMMARY_METRICS}})
    require(len(rows) == len(history) == 20, 'HISTORICAL_TWENTY_ROW_INVENTORY_CHANGED')
    nulls = {}
    for label, treatment, control in HISTORICAL_NULL_PAIRS:
        a, b = history[treatment], history[control]
        value = delta(a['metrics']['OVERALL'], b['metrics']['OVERALL'])
        nulls[label] = {'treatment': treatment, 'control': control,
                        'evidence_kind': HISTORICAL_KIND, 'scientific_use': False,
                        'zero_in_recorded_summary': all(x == 0 for x in value.values()),
                        'identical_prediction_files': sum(
                            a['result_files_sha256'][s] == b['result_files_sha256'][s]
                            for s in a['result_files_sha256']), **value}
    return {'rows': rows, 'provenance': provenance, 'null_observations': nulls,
            'inventory_sha256': sha(INVENTORY), 'policy_audit_sha256': sha(POLICY_AUDIT),
            'scientific_use': False, 'kind': HISTORICAL_KIND}


def main():
    # Validate every current and retained artifact before producing completion reports.
    context = validate_completed()
    recipe = verify_paired_recipe(context['manifest'])
    historical = verify_history(context['reference'])
    summary = report_completed(context)
    contrasts = summary['contrasts']
    interactions = {}
    for label, plus, minus in [
        ('mask_x_penalty', 'penalty_with_mask', 'penalty_without_mask'),
        ('fusion_x_selective', 'selective_with_fusion', 'selective_without_fusion'),
    ]:
        interactions[label] = {
            'definition': plus + ' minus ' + minus,
            'evidence_kind': 'corrected_metadata_new_cells_only', 'scientific_use': True,
            **{k: contrasts[plus][k] - contrasts[minus][k]
               for k in ['mota_pp', 'idf1_pp'] + METRICS[2:]}}
    activity = []
    for arm in context['manifest']['arms']:
        name = arm['name']
        totals = context['runtime_totals'][name]
        activity.append({'arm': name, 'frames': 6635,
                         'corrected_oracle_required': arm['runtime_arm'] in ('fc_full', 'fc_control'),
                         **{k: totals.get(k, 0) for k in ACTIVITY}})
    csv_write(ROOT / 'artifacts/mechanism_activity.csv', activity)
    csv_write(ROOT / 'artifacts/retained_historical_controls.csv', historical['rows'])
    csv_write(ROOT / 'artifacts/conditional_mechanism_contrasts.csv',
              [{'contrast': k, **v} for k, v in contrasts.items()])
    result = {
        'status': 'COMPLETE_VERIFIED_CORRECTED_METADATA_MECHANISM_ANALYSIS',
        'manifest_sha256': sha(ROOT / 'manifest.json'),
        'receipt_sha256': sha(ROOT / 'artifacts/receipt.json'),
        'analysis_sha256': sha(__file__),
        'primary_analysis_sha256': sha(ROOT / 'analyze_completed.py'),
        'identity_gate': context['identity_gate'],
        'scientific_scope': 'Frozen models with corrected online metadata, all nine fresh cells, same 17 test-dev sequences / 6635 frames, same GT/scorer and identical ordered detector/ReID inputs.',
        'corrected_provenance': context['provenance'],
        'corrected_contrasts': contrasts,
        'descriptive_factorial_interactions': interactions,
        'paired_head_recipe': recipe,
        'historical_archive': historical,
        'historical_prediction_parity_required': False,
        'selection_performed': False, 'formal_significance_claim': False,
        'GT_guard_scope': 'Python builtins.open/io.open/os.open on the designated GT root',
        'interpretation_limits': [
            'Conditional closed-loop effects at the declared frozen operating point; zero and negative results are retained.',
            'Activity counts establish execution, not accuracy or utility.',
            'Mask operates on appearance before minimum fusion; it is not a whole-edge hard gate.',
            'Responsibility definitions are fixed while their state-dependent numerical values may vary.',
            'C11/C00 and fc_full verify tuples plus pending responsibilities against the frozen corrected same-state reference; C10/C01 verify actual native seams and candidate legality.',
            'Paired heads share the declared scaler and event weights; shared C does not equate L2 parameter geometry.',
            'The metadata repair holds all model weights fixed; it does not establish that historical fitting metadata was corrected.',
            'All twenty historical rows and their null observations are metadata-contaminated archival evidence with scientific_use=false; they are absent from corrected contrasts.',
            'Frames, candidate edges and tracks are not independent experiment replicates.',
        ]}
    (ROOT / 'artifacts/mechanism_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    header = ['| Cell | MOTA | IDF1 | FP | FN | IDs | FM |',
              '|---|---:|---:|---:|---:|---:|---:|']
    lines = ['', '## SRC mask × learned penalty', '',
             'Every cell uses corrected metadata and frozen models; C00 retains the system\'s initialization/update and EGIA while using native assignment cost.', '', *header]
    for name in ['C00_nativecost_reference', 'C10_mask_only', 'C01_penalty_only', 'C11_full_reference']:
        lines.append(metric_row(name, context['metrics'][name]['metrics']['OVERALL']))
    lines += ['', '## Fusion × selective policy', '',
              'All four cells are newly inferred with corrected metadata. Coherence and all fitted weights are fixed; E00/E01 reuse frozen A04/A05 weights, not their historical scores.', '', *header]
    for name in ['E00_fusion_off_no_selective', 'E01_fusion_off_selective',
                 'E10_fusion_no_selective', 'C11_full_reference']:
        lines.append(metric_row(name, context['metrics'][name]['metrics']['OVERALL']))
    lines += ['', '## Historical archive — scientific_use=false', '',
              'The twenty historical score rows are rehashed and retained in retained_historical_controls.csv as historical_metadata_contaminated. None enters the corrected tables or contrasts. Their hard-update/recurrence null observations remain recorded below and do not establish corrected null effects.', '',
              '| Historical observation | MOTA delta (pp) | IDF1 delta (pp) | Identical files | Scientific use |',
              '|---|---:|---:|---:|---|']
    for label, value in historical['null_observations'].items():
        lines.append('| %s | %+.4f | %+.4f | %d/17 | false |' %
                     (label, value['mota_pp'], value['idf1_pp'], value['identical_prediction_files']))
    lines += ['', 'Factorial interactions are descriptive differences of conditional effects, with no significance claim. The paired hierarchy/flat comparison retains its declared fitting recipe; canonical F00 is not an isolated hierarchy-versus-flat fitting control. The frozen-weight metadata repair leaves historical fitting lineage unchanged.', '']
    with (ROOT / 'artifacts/RESULTS_FOR_AUTHOR.md').open('a') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps({'status': result['status'], 'corrected_arms': 9,
                      'corrected_contrasts': len(contrasts), 'historical_rows': 20,
                      'historical_scientific_use': False}))


if __name__ == '__main__':
    main()
