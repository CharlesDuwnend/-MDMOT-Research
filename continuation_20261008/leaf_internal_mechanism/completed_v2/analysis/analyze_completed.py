"""Verify all nine corrected online cells before writing paired result reports."""
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
METRICS = ['mota', 'idf1', 'num_false_positives', 'num_misses',
           'num_switches', 'num_fragmentations']
SUMMARY_METRICS = METRICS + ['num_objects']
EXPECTED_ARMS = {
    'C11_full_reference', 'C00_nativecost_reference', 'C10_mask_only',
    'C01_penalty_only', 'E10_fusion_no_selective',
    'E00_fusion_off_no_selective', 'E01_fusion_off_selective',
    'H_paired_hierarchical', 'H_paired_flat',
}
EXPECTED_FACTORS = {
    'C11_full_reference': (True, True, True, True, 'fc_full'),
    'C00_nativecost_reference': (False, False, True, True, 'fc_control'),
    'C10_mask_only': (True, False, True, True, 'fc_mask_only'),
    'C01_penalty_only': (False, True, True, True, 'fc_penalty_only'),
    'E10_fusion_no_selective': (True, True, True, False, 'fc_full'),
    'E00_fusion_off_no_selective': (True, True, False, False, 'fc_full'),
    'E01_fusion_off_selective': (True, True, False, True, 'fc_full'),
    'H_paired_hierarchical': (True, True, True, True, 'fc_full'),
    'H_paired_flat': (True, True, True, True, 'fc_full'),
}
PAIRS = [
    ('mask_without_penalty', 'C10_mask_only', 'C00_nativecost_reference'),
    ('penalty_without_mask', 'C01_penalty_only', 'C00_nativecost_reference'),
    ('penalty_with_mask', 'C11_full_reference', 'C10_mask_only'),
    ('mask_with_penalty', 'C11_full_reference', 'C01_penalty_only'),
    ('SRC_joint_cost_operators', 'C11_full_reference', 'C00_nativecost_reference'),
    ('selective_with_fusion', 'C11_full_reference', 'E10_fusion_no_selective'),
    ('fusion_without_selective', 'E10_fusion_no_selective', 'E00_fusion_off_no_selective'),
    ('fusion_with_selective', 'C11_full_reference', 'E01_fusion_off_selective'),
    ('selective_without_fusion', 'E01_fusion_off_selective', 'E00_fusion_off_no_selective'),
    ('paired_hierarchy_minus_flat', 'H_paired_hierarchical', 'H_paired_flat'),
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expected_bundle_sha(manifest, arm):
    """Resolve model identity from the unchanged pre-inference freeze."""
    path = ROOT / arm['bundle']
    require(str(path) in manifest['frozen_files_sha256'],
            'MODEL_NOT_IN_ORIGINAL_FREEZE: ' + arm['name'])
    expected = manifest['frozen_files_sha256'][str(path)]
    require(arm.get('bundle_sha256', expected) == expected,
            'ARM_MODEL_HASH_DISAGREES_WITH_FREEZE: ' + arm['name'])
    require(sha(path) == expected, 'FROZEN_MODEL_CHANGED: ' + arm['name'])
    return expected


def compatibility_provenance():
    return {
        'scope': 'Postprocessing only: model hash metadata for E00/E01 is resolved from the original frozen_files_sha256; experiment source, manifest and settings are unchanged.',
        'frozen_analysis_sha256': {
            name: sha(ROOT / name)
            for name in ['analyze_completed.py', 'analyze_mechanisms.py']},
        'executed_analysis_sha256': {
            name: sha(Path(__file__).with_name(name))
            for name in ['analyze_completed.py', 'analyze_mechanisms.py']},
    }


def compare_summary(actual, recorded, label):
    # Older receipts intentionally contain only these seven summary fields.
    for field in SUMMARY_METRICS:
        require(field in recorded and actual[field] == recorded[field],
                label + ': SUMMARY_FIELD_CHANGED: ' + field)


def csv_write(path, rows):
    require(bool(rows), 'EMPTY_REPORT: ' + str(path))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def delta(a, b):
    return {'mota_pp': 100 * (a['mota'] - b['mota']),
            'idf1_pp': 100 * (a['idf1'] - b['idf1']),
            **{field: a[field] - b[field] for field in METRICS[2:]}}


def verified_metric(path, expected_sha, reference, expected_folder=None):
    path = Path(path)
    require(sha(path) == expected_sha, 'METRIC_HASH_CHANGED: ' + str(path))
    value = read(path)
    sequences = set(reference['groundtruth_files_sha256'])
    require(value['eval_split'] == 'test-dev' and value['expected_sequences'] == 17,
            'WRONG_EVALUATION_PROTOCOL: ' + str(path))
    require(set(value['sequence_names']) == sequences, 'METRIC_SEQUENCE_SET_CHANGED')
    require(value['groundtruth_files_sha256'] == reference['groundtruth_files_sha256'],
            'GT_HASH_SET_CHANGED')
    require(set(value['result_files_sha256']) == sequences, 'PREDICTION_SEQUENCE_SET_CHANGED')
    folder = Path(value['results_folder']).resolve()
    if expected_folder is not None:
        require(folder == Path(expected_folder).resolve(), 'NONCURRENT_RESULTS_FOLDER')
    require({p.stem for p in folder.glob('*.txt')} == sequences,
            'INCOMPLETE_OR_EXTRA_PREDICTION_FILES')
    for sequence, expected in value['result_files_sha256'].items():
        require(sha(folder / (sequence + '.txt')) == expected,
                'PREDICTION_HASH_CHANGED: ' + sequence)
    require(set(value['metrics']) == sequences | {'OVERALL'}, 'METRIC_ROWS_CHANGED')
    for sequence in sorted(sequences) + ['OVERALL']:
        row = value['metrics'][sequence]
        require(row['num_objects'] == reference['metrics'][sequence]['num_objects'],
                'GT_OBJECT_COUNT_CHANGED: ' + sequence)
        for field in SUMMARY_METRICS:
            require(isinstance(row[field], (int, float)) and math.isfinite(row[field]),
                    'NONFINITE_METRIC: ' + sequence + ':' + field)
        expected_mota = 1 - (row['num_false_positives'] + row['num_misses'] +
                             row['num_switches']) / row['num_objects']
        require(abs(row['mota'] - expected_mota) < 1e-12, 'MOTA_ARITHMETIC_CHANGED')
    return value


def verify_input_ledger(report, sequence, frames):
    require(report['sequence'] == sequence and report['frames_seen'] == frames,
            'RUNTIME_SEQUENCE_OR_FRAME_COUNT_CHANGED')
    require(report['input_hash_schema'] == 'leaf-online-input-v1', 'INPUT_SCHEMA_CHANGED')
    require(report['input_frames_hashed'] == frames, 'INCOMPLETE_INPUT_HASH_LEDGER')
    records = report['input_records']
    require(len(records) == frames, 'INCOMPLETE_INPUT_RECORDS')
    rolling = hashlib.sha256(b'leaf-online-input-v1\0')
    for frame, record in enumerate(records, 1):
        require(record['frame'] == frame and record['sequence'] == sequence,
                'INPUT_SEQUENCE_OR_FRAME_ORDER_CHANGED')
        detector, embedding = record['detector_shape'], record['embedding_shape']
        require(len(detector) == len(embedding) == 2 and detector[1] == 7 and
                detector[0] == embedding[0] and embedding[1] > 0,
                'DETECTOR_REID_ROW_CONTRACT_CHANGED')
        for key in ['detector_sha256', 'embedding_sha256']:
            require(len(record[key]) == 64 and len(bytes.fromhex(record[key])) == 32,
                    'INVALID_ARRAY_INPUT_HASH')
        payload = {k: v for k, v in record.items()
                   if k not in ('frame_input_sha256', 'rolling_input_sha256')}
        digest = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        require(digest == record['frame_input_sha256'], 'FRAME_INPUT_HASH_CHANGED')
        rolling.update(bytes.fromhex(digest))
        require(rolling.hexdigest() == record['rolling_input_sha256'],
                'ROLLING_INPUT_HASH_CHANGED')
    require(rolling.hexdigest() == report['input_stream_sha256'],
            'FINAL_INPUT_STREAM_HASH_CHANGED')
    return records


def verify_identity_gate(manifest):
    path = ROOT / 'artifacts/INPUT_IDENTITY_AUDIT_GATE.json'
    gate = read(path)
    require(gate['status'] == 'RESOLVED_DETECTION_METADATA_IDENTITY',
            'UNRESOLVED_DETECTION_METADATA_IDENTITY')
    require(bool(gate.get('evidence_sha256')), 'EMPTY_IDENTITY_AUDIT_EVIDENCE')
    require(gate.get('sealed') is True, 'IDENTITY_GATE_NOT_SEALED')
    require(gate.get('manifest_sha256') == sha(ROOT / 'manifest.json'),
            'IDENTITY_GATE_MANIFEST_CHANGED')
    for evidence, expected in gate['evidence_sha256'].items():
        require(sha(evidence) == expected, 'IDENTITY_AUDIT_EVIDENCE_CHANGED: ' + evidence)
    return {'path': str(path), 'sha256': sha(path), 'status': gate['status'],
            'sealed': True, 'manifest_sha256': gate['manifest_sha256'],
            'evidence_sha256': gate['evidence_sha256']}


def validate_completed():
    """Read-only verification; a failed gate cannot produce a complete report."""
    manifest = read(ROOT / 'manifest.json')
    names = [arm['name'] for arm in manifest['arms']]
    require(len(names) == len(set(names)) == 9 and set(names) == EXPECTED_ARMS,
            'NINE_CORRECTED_ARMS_REQUIRED')
    frames = manifest['sequence_frames']
    require(len(frames) == 17 and sum(frames.values()) == 6635,
            'FULL_TESTDEV_17_SEQUENCE_6635_FRAME_PROTOCOL_REQUIRED')
    require(manifest.get('testdev_parameter_selection') is False,
            'TESTDEV_PARAMETER_SELECTION_NOT_PERMITTED')
    gate = verify_identity_gate(manifest)
    receipt_path = ROOT / 'artifacts/receipt.json'
    receipt = read(receipt_path)
    require(receipt['status'] == 'PASS_PREDECLARED_ONLINE_MECHANISM_ADDENDUM_COMPLETE',
            'ONLINE_RUN_NOT_COMPLETE')
    require(receipt['manifest_sha256'] == sha(ROOT / 'manifest.json'), 'MANIFEST_CHANGED')
    require((ROOT / 'artifacts/launcher.exit').read_text().strip() == '0',
            'LAUNCHER_DID_NOT_EXIT_SUCCESSFULLY')
    require(not (ROOT / 'artifacts/stop.json').exists(), 'STOP_RECEIPT_PRESENT')
    records = {arm['name']: arm for arm in receipt['arms']}
    require(len(receipt['arms']) == len(records) == 9 and set(records) == set(names),
            'COMPLETE_RECEIPT_ARM_SET_CHANGED')
    for path, expected in manifest['frozen_files_sha256'].items():
        require(sha(path) == expected, 'FROZEN_FILE_CHANGED: ' + path)
    scorer = ROOT / 'host/tools/utils/eval_visdrone.py'
    require(str(scorer) in manifest['frozen_files_sha256'], 'SCORER_NOT_FROZEN')
    scorer_hash = sha(scorer)
    # The historical score contributes immutable GT metadata only here.
    reference_path = Path(manifest['full_reference_metrics'])
    require(str(reference_path) in manifest['frozen_files_sha256'], 'GT_REFERENCE_NOT_FROZEN')
    reference = read(reference_path)
    require(set(reference['groundtruth_files_sha256']) == set(frames), 'GT_SEQUENCE_SET_CHANGED')
    for sequence, expected in reference['groundtruth_files_sha256'].items():
        require(sha(Path(manifest['gt_root']) / (sequence + '.txt')) == expected,
                'ACTUAL_GT_HASH_CHANGED: ' + sequence)
    metrics, provenance, reports_by_arm, totals_by_arm = {}, {}, {}, {}
    for arm in manifest['arms']:
        name = arm['name']
        require(tuple(arm[k] for k in ('mask', 'penalty', 'fusion', 'selective',
                                      'runtime_arm')) == EXPECTED_FACTORS[name],
                'PREREGISTERED_CELL_FACTORS_CHANGED: ' + name)
        require(arm['corrected_reference_oracle'] ==
                (arm['runtime_arm'] in ('fc_full', 'fc_control')),
                'CORRECTED_ORACLE_ARM_CONTRACT_CHANGED: ' + name)
        output = ROOT / 'YOLOX_outputs' / name
        local_path = output / 'receipt.json'
        local, global_record = read(local_path), records[name]
        for record in [local, global_record]:
            require(record['name'] == name and record['status'] == 'PASS_SCORED_ARM_COMPLETE',
                    'ARM_NOT_SCORED_COMPLETE: ' + name)
            require(record['frames'] == 6635 and record['sequence_count'] == 17,
                    'PARTIAL_OR_SMOKE_RESULT: ' + name)
            require(record['gpu_uuid'] == manifest['gpu_uuid'] and
                    record['same_process_gpu_verification']['cuda_uuid'] == manifest['gpu_uuid'],
                    'GPU_RECEIPT_CHANGED: ' + name)
            require(record['gt_guard']['status'] == 'PASS_NO_GT_READ' and
                    record['gt_guard']['blocked_gt_attempts'] == 0,
                    'GT_GUARD_FAILED: ' + name)
            require(record['bundle_sha256'] == expected_bundle_sha(manifest, arm),
                    'FROZEN_MODEL_CHANGED: ' + name)
        metric_path = output / 'tracking_metrics.json'
        value = verified_metric(metric_path, local['metrics_sha256'], reference,
                                output / 'track_res')
        for record in [local, global_record]:
            require(record['metrics_sha256'] == sha(metric_path), 'ARM_METRIC_RECEIPT_CHANGED')
            require(record['result_sha256'] == value['result_files_sha256'],
                    'ARM_PREDICTION_RECEIPT_CHANGED')
            compare_summary(value['metrics']['OVERALL'], record['metrics'], name)
        reports = {s: read(output / 'track_res' / (s + '.txt.runtime.json')) for s in frames}
        for sequence, report in reports.items():
            verify_input_ledger(report, sequence, frames[sequence])
            require(report['arm'] == arm['runtime_arm'], 'RUNTIME_ARM_CHANGED: ' + name)
            require(report['gt_read'] is False, 'RUNTIME_GT_ACCESS')
            require(report['src_state_enabled'] and report['src_update_enabled'] and
                    report['use_egia'], 'MODULE_STATE_CONTRACT_CHANGED')
            require(report['cost_factorial']['assignment_mask_enabled'] == arm['mask'] and
                    report['cost_factorial']['semantic_penalty_enabled'] == arm['penalty'],
                    'MASK_PENALTY_ARM_CONTRACT_CHANGED')
            counts = report['counts']
            seams = counts['association_seams']
            require(seams > 0 and counts.get('factorial_illegal_edge_recoveries', 0) == 0,
                    'ILLEGAL_OR_MISSING_ASSOCIATION_SEAMS')
            require(counts.get('factorial_assignment_verified_seams', 0) == seams and
                    counts.get('factorial_native_assignment_calls', 0) == seams,
                    'ACTUAL_NATIVE_SEAM_OR_CANDIDATE_LEGALITY_NOT_CLOSED')
            if arm['runtime_arm'] in ('fc_full', 'fc_control'):
                require(counts.get('corrected_reference_seams_verified', 0) == seams,
                        'CORRECTED_REFERENCE_TUPLE_AND_RESPONSIBILITY_PARITY_NOT_CLOSED')
            else:
                require(name in ('C10_mask_only', 'C01_penalty_only'),
                        'UNDECLARED_NONORACLE_RUNTIME_ARM')
                require(counts.get('corrected_reference_seams_verified', 0) == 0,
                        'NONORACLE_CELL_CLAIMS_REFERENCE_PARITY')
            require(counts.get('src_semantic_penalty_calls', 0) ==
                    counts.get('src_cost_evaluation_calls', 0),
                    'PENALTY_CALL_COUNT_CONTRACT_CHANGED')
            if not arm['penalty']:
                require(counts.get('src_semantic_penalty_calls', 0) == 0,
                        'PENALTY_OFF_CELL_CALLED_PENALTY')
            if not arm['selective']:
                require(all(v == 0 for k, v in counts.items() if k.startswith('selective_')),
                        'SELECTIVE_OFF_CELL_CALLED_SELECTIVE_POLICY')
            fusion = arm['fusion']
            require(report['egia_fusion_policy'] == fusion, 'FUSION_ARM_CONTRACT_CHANGED')
            require(report['legacy_egia_loaded'] == fusion, 'TEACHER_LOADING_CONTRACT_CHANGED')
            if not fusion:
                require(counts.get('egia_fusion_events', 0) == 0,
                        'FUSION_OFF_CELL_EXECUTED_FUSION')
        totals = dict(Counter({k: sum(r['counts'].get(k, 0) for r in reports.values())
                              for k in {k for r in reports.values() for k in r['counts']}}))
        inputs = {s: r['input_stream_sha256'] for s, r in reports.items()}
        for record in [local, global_record]:
            require(record['runtime_totals'] == totals, 'RUNTIME_TOTALS_CHANGED: ' + name)
            require(record['input_sha256'] == inputs, 'INPUT_RECEIPT_CHANGED: ' + name)
        metrics[name], reports_by_arm[name], totals_by_arm[name] = value, reports, totals
        provenance[name] = {'path': str(metric_path), 'sha256': sha(metric_path),
                            'receipt_path': str(local_path), 'receipt_sha256': sha(local_path),
                            'kind': 'corrected_metadata_fresh_online_inference',
                            'scientific_use': True, 'frames': 6635, 'prediction_files': 17,
                            'runtime_sha256': {s: sha(output / 'track_res' /
                                                      (s + '.txt.runtime.json')) for s in frames}}
    baseline = reports_by_arm['C11_full_reference']
    for name, reports in reports_by_arm.items():
        for sequence, report in reports.items():
            require(report['input_records'] == baseline[sequence]['input_records'] and
                    report['input_stream_sha256'] == baseline[sequence]['input_stream_sha256'],
                    'DETECTOR_REID_ORDERED_ARRAY_INPUTS_DIFFER: ' + name + ':' + sequence)
    return {'manifest': manifest, 'receipt': receipt, 'metrics': metrics,
            'provenance': provenance, 'runtime_totals': totals_by_arm,
            'identity_gate': gate, 'reference': reference,
            'scorer': {'path': str(scorer), 'sha256': scorer_hash},
            'groundtruth_sha256': reference['groundtruth_files_sha256']}


def report_completed(context):
    manifest, metrics = context['manifest'], context['metrics']
    rows, flights, contrasts = [], defaultdict(list), {}
    for label, treatment, control in PAIRS:
        a, b = metrics[treatment]['metrics'], metrics[control]['metrics']
        contrasts[label] = {'treatment': treatment, 'control': control,
                            'evidence_kind': 'corrected_metadata_new_cells_only',
                            'scientific_use': True, **delta(a['OVERALL'], b['OVERALL'])}
        for sequence in manifest['sequence_frames']:
            row = {'contrast': label, 'sequence': sequence,
                   'flight_prefix': sequence.split('_')[0], **delta(a[sequence], b[sequence])}
            rows.append(row)
            flights[(label, row['flight_prefix'])].append(row)
    csv_write(ROOT / 'artifacts/paired_sequence_deltas.csv', rows)
    flight_rows = [{'contrast': label, 'flight_prefix': flight, 'clips': len(group),
                    'mean_clip_delta_mota_pp': sum(x['mota_pp'] for x in group) / len(group),
                    'mean_clip_delta_idf1_pp': sum(x['idf1_pp'] for x in group) / len(group),
                    **{k: sum(x[k] for x in group) for k in METRICS[2:]}}
                   for (label, flight), group in sorted(flights.items())]
    csv_write(ROOT / 'artifacts/paired_flight_descriptive_deltas.csv', flight_rows)
    summary = {'status': 'VERIFIED_COMPLETE_CORRECTED_METADATA_PAIRED_CONTRASTS',
               'manifest_sha256': sha(ROOT / 'manifest.json'),
               'receipt_sha256': sha(ROOT / 'artifacts/receipt.json'),
               'analysis_sha256': sha(__file__),
               'postprocessing_compatibility': compatibility_provenance(),
               'identity_gate': context['identity_gate'],
               'provenance': context['provenance'], 'scorer': context['scorer'],
               'groundtruth_sha256': context['groundtruth_sha256'], 'contrasts': contrasts,
               'sequence_count': 17, 'frames': 6635, 'corrected_arms': 9,
               'flight_count': len({s.split('_')[0] for s in manifest['sequence_frames']}),
               'analysis_scope': 'Frozen models with corrected online metadata; conditional closed-loop contrasts at the declared operating point. Flight rows average clip deltas, not pooled flight IDF1.',
               'frame_level_significance': False, 'testdev_selection': False,
               'models_refitted_for_metadata_repair': False,
               'historical_prediction_parity_required': False}
    (ROOT / 'artifacts/paired_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Corrected online internal mechanism results', '',
             'All nine cells use fresh corrected-metadata inference over the same 17 sequences / 6635 frames. Frozen source/model hashes, actual prediction and GT hashes, the scorer, ordered detector/ReID input ledgers and seam contracts were verified. The sealed identity gate is resolved. Historical prediction-file parity is not a completion gate.', '',
             '| Arm | MOTA | IDF1 | FP | FN | IDs | FM |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm in manifest['arms']:
        name = arm['name']
        m = metrics[name]['metrics']['OVERALL']
        lines.append('| %s | %.4f | %.4f | %s | %s | %s | %s |' %
                     (name, 100 * m['mota'], 100 * m['idf1'],
                      *[m[k] for k in METRICS[2:]]))
    lines += ['', '| Conditional contrast | MOTA delta (pp) | IDF1 delta (pp) | FP delta | FN delta | IDs delta | FM delta |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for label, d in contrasts.items():
        lines.append('| %s | %+.4f | %+.4f | %+d | %+d | %+d | %+d |' %
                     (label, d['mota_pp'], d['idf1_pp'], *[d[k] for k in METRICS[2:]]))
    lines += ['', 'Every corrected contrast uses the nine new cells only. Zero and negative values are retained. C11/C00 and the other fc_full cells verify exact association tuples and pending responsibilities against the frozen same-state corrected-seam oracle; C10/C01 verify actual native seams and candidate legality.', '',
              'The hierarchy/flat contrast uses the declared paired fitting scaler and event weights, with frozen heads. It is separate from F00\'s original fitting recipe; shared C does not equate L2 parameter geometry. No refitting for the metadata repair or test-dev winner selection is implied. Activity counts are diagnostics, not tracking accuracy.', '']
    (ROOT / 'artifacts/RESULTS_FOR_AUTHOR.md').write_text('\n'.join(lines))
    return summary


def main():
    summary = report_completed(validate_completed())
    print(json.dumps({'status': summary['status'], 'corrected_arms': 9,
                      'contrasts': summary['contrasts']}, indent=2))


if __name__ == '__main__':
    main()
