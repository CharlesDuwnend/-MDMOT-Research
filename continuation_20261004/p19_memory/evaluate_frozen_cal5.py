#!/usr/bin/env python3
"""Evaluate only five existing P19 cal5 runs; never run inference or read dev5.

The historical evaluator is imported unchanged. Only its output directory and
metadata writer are redirected; its metric computations are not modified.
"""
import sys
sys.dont_write_bytecode = True

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
OLD = Path('/home/chenhc/mdmot_research_20261002')
P18 = OLD / 'p18_merge_verifier'
RUN_ROOT = Path('/raid/datasets/chc_data/mdmot_research_cache_20261004/p18_runs')
PAIRS = ['27', '32', '42', '64', '65']
THRESHOLDS = ['0.8918', '0.85', '0.80', '0.75', '0.70']
TAGS = ['p19_cal_mem' + value + '_sf0.70' for value in THRESHOLDS]
ATTRIBUTION = ('Observed response of the existing joint implementation: the memory threshold '
               'and the same-frame score shift both change. Same-frame admissibility stays at '
               'cosine >= 0.70, but a varying shift can change Hungarian assignment cardinality. '
               'This is not an isolated causal estimate of memory-threshold effects.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    if OUT not in path.resolve().parents:
        raise ValueError('all output must remain in the new p19_memory directory')
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + '\n')


def protected_inputs():
    paths = [P18 / name for name in ('evaluate_cal5.py', 'data_io_cal5.py', 'run_host_merge.py')]
    paths += [OLD / 'p1e/evaluation' / name for name in ('metrics.py', 'crosslink.py', 'RULES.json')]
    paths += [OLD / 'p19_merge_geometry' / name for name in ('run_p19_mem.sh', 'P19_PREREG.json', 'p19mem.log')]
    paths += [RUN_ROOT / tag / name for tag in TAGS for name in ('PREDICTIONS_FROZEN.json', 'runner.exit')]
    return {str(path): sha(path) for path in paths}


def precheck():
    if (OUT / 'PRECHECK.json').exists():
        raise ValueError('precheck already exists; use --finalize for an already evaluated set')
    checks = []
    for tag, value in zip(TAGS, THRESHOLDS):
        run = RUN_ROOT / tag
        freeze = json.loads((run / 'PREDICTIONS_FROZEN.json').read_text())
        assert (run / 'runner.exit').read_text().strip() == '0'
        assert freeze['status'] == 'PREDICTIONS_FROZEN_BEFORE_GT'
        assert freeze['pairs'] == PAIRS and freeze['encoder_arm'] == 'anchor_p1'
        assert freeze['arms'] == ['current_future_merge'] and freeze['tag'] == tag
        assert freeze['gt_read'] is False and freeze['gpu_used'] is False
        assert freeze['official_val_access'] is False and freeze['test_access'] is False
        assert freeze['threshold'] == float(value)
        # All recorded run outputs are checked before any raw GT read.
        for item in freeze['files']:
            path = Path(item['path']).resolve()
            assert run.resolve() in path.parents
            assert sha(path) == item['sha256'], str(path)
        checks.append({'tag': tag, 'runtime_threshold': freeze['threshold'],
                       'same_frame_threshold': 0.70,
                       'same_frame_threshold_provenance': 'launch shell and recorded historical command; absent from freeze schema',
                       'same_frame_shift': freeze['threshold'] - 0.70,
                       'prediction_freeze_sha256': sha(run / 'PREDICTIONS_FROZEN.json'),
                       'verified_frozen_run_files': len(freeze['files']),
                       'input_counts': freeze['counts']['input'],
                       'runner_exit': 0})
    launch = (OLD / 'p19_merge_geometry/run_p19_mem.sh').read_text()
    log = (OLD / 'p19_merge_geometry/p19mem.log').read_text()
    assert '--same-frame-threshold 0.70' in launch
    assert 'P19MEM_DONE\nEXIT=0' in log
    result = {'status': 'PASS_FIVE_EXISTING_CAL5_RUNS_FROZEN', 'checks': checks,
              'pairs': PAIRS, 'protected_input_sha256': protected_inputs(),
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'raw_GT_read': False, 'dev5_read': False, 'official_val_or_test_read': False,
              'attribution_limit': ATTRIBUTION}
    dump(OUT / 'PRECHECK.json', result)
    return result


def evaluate_one(tag):
    assert tag in TAGS
    assert (OUT / 'PRECHECK.json').exists()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    sys.path.insert(0, str(P18))
    spec = importlib.util.spec_from_file_location('historical_p19_cal5_evaluator', P18 / 'evaluate_cal5.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.P18 = OUT
    assert module.PAIRS == PAIRS
    import data_io_cal5
    assert data_io_cal5.PAIR_ALLOWLIST == set(PAIRS)
    upstream_dump = module.dump

    def redirected_dump(path, value):
        path = Path(path)
        assert OUT in path.resolve().parents
        if isinstance(value, dict) and value.get('status') == 'PASS_FIXED_ONE_PASS_CAL5_COMPLETE_OUTPUT_EVALUATION':
            value = dict(value)
            value['inherited_scope_text_correction'] = value.get('scope')
            value['scope'] = 'previously used official-train internal cal5; evaluation of an existing five-threshold calibration grid'
            value['dev5_read_this_evaluation'] = False
            value['same_frame_threshold'] = 0.70
            value['same_frame_threshold_provenance'] = 'historical launch command and run_p19_mem.sh; not bound by original freeze schema'
            value['same_frame_score_shift'] = value['threshold'] - 0.70
            value['attribution_limit'] = ATTRIBUTION
        upstream_dump(path, value)

    module.dump = redirected_dump
    sys.argv = [str(P18 / 'evaluate_cal5.py'), '--arm', 'anchor_p1', '--tag', tag]
    module.main()


def finalize():
    pre = json.loads((OUT / 'PRECHECK.json').read_text())
    assert protected_inputs() == pre['protected_input_sha256'], 'historical protected input changed'
    rows = []
    common_pair_evidence = {}
    common_pair_iou = {}
    max_mean_iou_drift = 0.0
    for check in pre['checks']:
        tag = check['tag']
        result = json.loads((OUT / ('RESULTS_' + tag + '.json')).read_text())
        assert (OUT / ('evaluation_' + tag + '.exit')).read_text().strip() == '0'
        assert result['status'] == 'PASS_FIXED_ONE_PASS_CAL5_COMPLETE_OUTPUT_EVALUATION'
        freeze = json.loads((RUN_ROOT / tag / 'PREDICTIONS_FROZEN.json').read_text())
        per_pair = {}
        for pair in PAIRS:
            artifact = json.loads((OUT / ('RESULTS_' + tag + '_PAIR_' + pair + '.json')).read_text())
            p = artifact['results']['current_future_merge']
            spatial_exact = {k: v for k, v in p['spatial'].items() if k != 'mean_matched_iou'}
            evidence = {'spatial': spatial_exact, 'support_sha256': p['support_sha256'],
                        'source_stream_sha256': artifact['source_stream_sha256'],
                        'source_GT_sha256': artifact['source_GT_sha256']}
            if pair in common_pair_evidence:
                assert evidence == common_pair_evidence[pair], 'cross-arm support or spatial metrics drifted'
            else:
                common_pair_evidence[pair] = evidence
                common_pair_iou[pair] = p['spatial']['mean_matched_iou']
            iou_drift = abs(p['spatial']['mean_matched_iou'] - common_pair_iou[pair])
            max_mean_iou_drift = max(max_mean_iou_drift, iou_drift)
            assert iou_drift < 1e-12, 'mean IoU drift is larger than roundoff tolerance'
            assert p['clock']['clock_boundary_check'] == 'PASS'
            routes = freeze['counts']['pair_results'][pair]['routes_and_events']['current_future_merge']
            per_pair[pair] = {'CA_IDF1': p['idf1'], 'crosslinks': p['crosslinks'],
                              'spatial': p['spatial'], 'clock': p['clock']['clock_boundary_check'],
                              'current_future_gid_merge': routes.get('current_future_gid_merge', 0),
                              'past_memory_link': routes.get('past_memory_link', 0)}
        summary = result['summary']['current_future_merge']
        assert abs(summary['pair_macro_CA_IDF1'] - sum(per_pair[p]['CA_IDF1'] for p in PAIRS) / 5) < 1e-12
        assert summary['micro_CA_IDF1'] == 2 * summary['idtp'] / (2 * summary['idtp'] + summary['idfp'] + summary['idfn'])
        for key in ('tp', 'fp', 'fn'):
            assert summary['crosslinks_micro'][key] == sum(per_pair[p]['crosslinks'][key] for p in PAIRS)
        rows.append(dict(check, summary=summary, per_pair=per_pair,
                         result_sha256=sha(OUT / ('RESULTS_' + tag + '.json'))))
    # Pre-existing protocol selects cal5 pair-macro argmax; ties use original grid order.
    best = max(rows, key=lambda row: row['summary']['pair_macro_CA_IDF1'])
    base = rows[0]
    receipt = {'status': 'PASS_EXISTING_P19_CAL5_GRID_EVALUATED_AND_SELECTED',
               'dataset': 'MDMT', 'metric': 'CA_IDF1_conventional_pseudo', 'official_metric': False,
               'identity_semantics': 'same_label_equal_id annotation-convention pseudo identity',
               'rows': rows, 'pairs': PAIRS,
               'selection_rule': 'argmax cal5 pair_macro_CA_IDF1 over the five historical settings; first grid entry breaks exact ties',
               'selected_tag': best['tag'], 'selected_runtime_threshold': best['runtime_threshold'],
               'selected_same_frame_threshold': 0.70,
               'selected_score': best['summary']['pair_macro_CA_IDF1'],
               'delta_vs_first_grid_setting': best['summary']['pair_macro_CA_IDF1'] - base['summary']['pair_macro_CA_IDF1'],
               'per_pair_wins_vs_first_grid_setting': sum(best['per_pair'][p]['CA_IDF1'] > base['per_pair'][p]['CA_IDF1'] for p in PAIRS),
               'dev5_read': False, 'official_val_or_test_read': False, 'new_inference': False,
               'training': False, 'gpu_used': False, 'max_parallel_evaluations': 2,
               'attribution_limit': ATTRIBUTION,
               'historical_metadata_corrections': [
                   'freeze dev_access=true is a copied label; actual pairs are internal cal5',
                   'freeze threshold_origin refers to P16; actual settings belong to the P19 memory grid',
                   'freeze hyperparameter_search=false describes an individual fixed run; the collection is a calibration sweep',
                   'same-frame threshold is missing from the freeze schema and is derived from launch evidence'],
               'protected_inputs_unchanged': True,
               'verification': {'common_support_spatial_GT_and_stream_hashes_across_five_settings': True,
                                'all_25_pair_clock_checks_pass': True,
                                'macro_micro_and_crosslink_aggregation_crosschecked': True,
                                'all_180_frozen_run_files_verified_before_raw_GT': True,
                                'mean_matched_iou_max_abs_roundoff': max_mean_iou_drift,
                                'mean_matched_iou_tolerance': 1e-12},
               'verification_lookback': 'Initial exact dict comparison failed only for mean_matched_iou at floating summation roundoff; all integer counts, derived P/R/F1, support, source and GT hashes matched exactly. Mean IoU now uses 1e-12 tolerance; raw evaluation results remain untouched.',
               'precheck_sha256': sha(OUT / 'PRECHECK.json'),
               'reproducer_sha256': sha(Path(__file__)),
               'completed_utc': datetime.now(timezone.utc).isoformat()}
    dump(OUT / 'P19_MEMORY_CAL5_RECEIPT.json', receipt)
    sealed = [p for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'SHA256SUMS' and p.suffix != '.log']
    (OUT / 'SHA256SUMS').write_text(''.join(sha(p) + '  ' + p.name + '\n' for p in sealed))
    print(json.dumps({k: receipt[k] for k in ('status', 'selected_tag', 'selected_score', 'delta_vs_first_grid_setting',
                                             'per_pair_wins_vs_first_grid_setting', 'dev5_read')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--one', choices=TAGS)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    if args.one:
        evaluate_one(args.one)
        return
    if args.finalize:
        finalize()
        return
    precheck()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def child(tag):
        with (OUT / (tag + '.log')).open('w') as log:
            process = subprocess.run([sys.executable, '-B', str(Path(__file__)), '--one', tag],
                                     env=env, cwd=str(OUT), stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError('evaluation failed; inspect implementation and ' + tag + '.log before retry')
        print('completed ' + tag, flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(child, TAGS))
    finalize()


if __name__ == '__main__':
    main()
