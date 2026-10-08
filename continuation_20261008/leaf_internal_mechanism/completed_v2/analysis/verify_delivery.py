"""Audit the completed experiment and deliverable against the declared scope."""
import csv
import datetime
import json
import subprocess
from pathlib import Path

from analyze_completed import ROOT, METRICS, delta, read, require, sha, validate_completed
from analyze_mechanisms import verify_history, verify_paired_recipe

REPO = Path('/home/chenhc/claude_try_MDMOT')


def main():
    c = validate_completed()
    m = c['manifest']
    recipe = verify_paired_recipe(m)
    history = verify_history(c['reference'])
    s = read(ROOT / 'artifacts/mechanism_summary.json')
    require(s['status'] == 'COMPLETE_VERIFIED_CORRECTED_METADATA_MECHANISM_ANALYSIS',
            'ANALYSIS_NOT_COMPLETE')
    require(not s['selection_performed'] and not s['formal_significance_claim'],
            'UNDECLARED_SELECTION_OR_INFERENCE')
    p = read(ROOT / 'artifacts/progress.json')
    require(p['status'] == 'COMPLETE' and p['active_arm'] is None and
            len(p['completed_arms']) == 9, 'FULL_PROGRESS_NOT_TERMINAL')
    for name in ['launcher.exit', 'pipeline.exit', 'smoke_launcher.exit']:
        require((ROOT / 'artifacts' / name).read_text().strip() == '0',
                'TERMINAL_EXIT_FAILED: ' + name)
    require(not (ROOT / 'artifacts/live.lock').exists() and
            not (ROOT / 'artifacts/stop.json').exists(), 'RUN_NOT_CLOSED')
    tmux = subprocess.run(['tmux', 'has-session', '-t', 'leaf_mechanism_corrected_20261008'],
                          capture_output=True)
    require(tmux.returncode != 0, 'PIPELINE_TMUX_STILL_LIVE')
    cells = {}
    for arm in c['receipt']['arms']:
        name = arm['name']
        guard = read(ROOT / 'artifacts' / (name + '_gt_guard.json'))
        inventory = read(ROOT / 'artifacts' / (name + '_hardware.json'))
        hardware = arm['same_process_gpu_verification']
        require(guard == arm['gt_guard'] and guard['checked_file_open_calls'] > 0 and
                guard['blocked_gt_attempts'] == 0, 'ACTUAL_GUARD_RECEIPT_DIFFERS')
        require(inventory['selected'] == hardware['selected'],
                'PRELAUNCH_AND_INFERENCE_GPU_SELECTION_DIFFERS')
        logged_hardware, _ = json.JSONDecoder().raw_decode(
            (ROOT / 'logs' / (name + '.log')).read_text().lstrip())
        require(logged_hardware == hardware, 'INFERENCE_LOG_AND_GPU_RECEIPT_DIFFERS')
        require(hardware['selected']['index'] == 1 and
                hardware['cuda_total_memory_bytes'] == 42298834944 and
                hardware['cuda_uuid'] == m['gpu_uuid'], 'WRONG_PHYSICAL_DEVICE')
        metric = c['metrics'][name]['metrics']
        for field in ['num_objects'] + METRICS[2:]:
            require(sum(metric[seq][field] for seq in m['sequence_frames']) ==
                    metric['OVERALL'][field], 'COUNT_AGGREGATION_DIFFERS: ' + field)
        log = ROOT / 'logs' / (name + '_evaluation.log')
        text = log.read_text()
        require('Found 17 groundtruths and 17 test files.' in text and
                'Saved structured metrics to ' in text and 'Completed' in text,
                'SCORER_EXECUTION_NOT_PROVEN: ' + name)
        cells[name] = {
            'frames': arm['frames'], 'sequence_count': arm['sequence_count'],
            'hardware_sha256': sha(ROOT / 'artifacts' / (name + '_hardware.json')),
            'gt_guard_sha256': sha(ROOT / 'artifacts' / (name + '_gt_guard.json')),
            'scorer_log_sha256': sha(log), 'bundle_sha256': arm['bundle_sha256'],
            'metrics_sha256': arm['metrics_sha256'],
            'association_seams': arm['runtime_totals']['association_seams'],
            'corrected_reference_seams_verified': arm['runtime_totals'].get(
                'corrected_reference_seams_verified', 0),
            'input_stream_sha256': arm['input_sha256'],
            'predictions_sha256': arm['result_sha256'],
        }
    rows = list(csv.DictReader((ROOT / 'artifacts/results.csv').open()))
    require(len(rows) == 9 and {x['arm'] for x in rows} == set(cells),
            'RESULT_TABLE_COVERAGE_DIFFERS')
    for row in rows:
        actual = c['metrics'][row['arm']]['metrics']['OVERALL']
        for field in METRICS:
            require(float(row[field]) == actual[field], 'RESULT_TABLE_VALUE_DIFFERS')
    require(len(s['corrected_contrasts']) == 10, 'CONDITIONAL_CONTRAST_MISSING')
    for label, d in s['corrected_contrasts'].items():
        actual = delta(c['metrics'][d['treatment']]['metrics']['OVERALL'],
                       c['metrics'][d['control']]['metrics']['OVERALL'])
        require(all(d[k] == v for k, v in actual.items()), 'DELTA_DIFFERS: ' + label)
        require(d['scientific_use'] and d['evidence_kind'] ==
                'corrected_metadata_new_cells_only', 'HISTORICAL_SCORE_IN_NEW_CONTRAST')
    require(not history['scientific_use'] and len(history['rows']) == 20,
            'HISTORICAL_BOUNDARY_DIFFERS')
    figure = read(ROOT / 'figures/completed_mechanisms/FIGURE_RECEIPT.json')
    require(figure['summary_sha256'] == sha(ROOT / 'artifacts/mechanism_summary.json'),
            'FIGURES_FROM_DIFFERENT_ANALYSIS')
    for path, expected in figure['files_sha256'].items():
        require(sha(path) == expected, 'FIGURE_FILE_CHANGED: ' + path)
    protected = read(REPO / 'continuation_20261008/leaf_internal_mechanism/PROTECTED_ARTIFACTS.json')
    for path, expected in protected['sha256'].items():
        require(sha(path) == expected, 'PROTECTED_ARTIFACT_CHANGED: ' + path)
    import hashlib
    diff = subprocess.run(['git', 'diff', '--cached', '--binary'], cwd=REPO,
                          capture_output=True, check=True).stdout
    index_sha = hashlib.sha256(diff).hexdigest()
    require(index_sha == 'b03d208f880c4bde3f50d3d5a56f29d068e668f1ba1e634b6d07cd506e52db3c',
            'UNRELATED_P63_STAGING_CHANGED')
    result = {
        'status': 'PASS_COMPLETE_NINE_ARM_MECHANISM_DELIVERY',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'manifest_sha256': sha(ROOT / 'manifest.json'),
        'receipt_sha256': sha(ROOT / 'artifacts/receipt.json'),
        'analysis_sha256': sha(ROOT / 'artifacts/mechanism_summary.json'),
        'verifier_sha256': sha(__file__),
        'requirements_verified': [
            'Nine preregistered corrected cells, each 17 sequences / 6635 frames',
            'Frozen source/model/scorer hashes and actual GT/prediction hashes',
            'Identical ordered detector/ReID input ledgers across all cells',
            'Resolved metadata gate and appropriate per-seam reference/legality gates',
            'Actual Python GT guards, model policy switches and physical A100 GPU1',
            'Executed scorer logs, GT denominator and count aggregation',
            'Ten conditional contrasts, zero/negative results and two descriptive interactions',
            'Paired fit recipe preserved, without metadata-repair refit or test-dev selection',
            'Twenty historical rows isolated from new scientific contrasts',
            'Terminal pipeline/smoke exit codes and absence of live pipeline session',
            'Mechanism figures bound to verified scores',
            'Protected manuscript/baseline receipts and unrelated P63 staging unchanged',
        ],
        'cells': cells, 'paired_recipe': recipe,
        'protected_sha256': protected['sha256'], 'unrelated_staged_diff_sha256': index_sha,
        'limitations': [
            'One frozen operating point; conditional closed-loop contrasts are descriptive.',
            'GT guard covers Python opens on the designated annotation root, not all native I/O.',
            'Runtime diagnostic hashes are sealed at analysis time; no producer-side seal of every diagnostic byte.',
            'No corrected independent support yet for recurrence, soft-versus-hard responsibility or source-owner pairing.',
            'Paired hierarchy result is negative; matching C does not equalize L2 geometry.',
            'Canonical fitting lineage stays unchanged; frozen heads were not refitted for the runtime metadata repair.',
            'Aggregate counts alone do not identify GT-correct edge or birth interventions.',
        ],
    }
    (ROOT / 'artifacts/FINAL_DELIVERY_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], len(cells), len(result['requirements_verified']))


if __name__ == '__main__':
    main()
