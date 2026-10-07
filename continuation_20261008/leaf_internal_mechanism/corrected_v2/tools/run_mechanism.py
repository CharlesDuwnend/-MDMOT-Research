"""Run predeclared online controls, preserving source/input/solver contracts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/chenhc/.conda/envs/u2mot/bin/python'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def run(command, env, log):
    print(json.dumps({'command': command, 'log': str(log)}), flush=True)
    with Path(log).open('x') as out:
        rc = subprocess.run(command, env=env, cwd=ROOT, stdout=out,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError('COMMAND_FAILED_%d: %s' % (rc, log))


def device_receipt(log):
    raw = Path(log).read_text(errors='replace')
    decoder = json.JSONDecoder()
    for offset, c in enumerate(raw):
        if c != '{':
            continue
        try:
            obj, _ = decoder.raw_decode(raw[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get('status') == 'VERIFIED_PHYSICAL_GPU':
            return obj
    raise RuntimeError('MISSING_SAME_PROCESS_DEVICE_RECEIPT')


def check_frozen(manifest):
    for name, expected in manifest['frozen_files_sha256'].items():
        if sha(name) != expected:
            raise RuntimeError('FROZEN_FILE_CHANGED: ' + name)
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise RuntimeError('LESS_THAN_2GIB_FREE_DISK')


def launch_gpu(manifest, name):
    from verified_gpu import inventory, select_allowed
    selected = select_allowed(inventory(), manifest['gpu_uuid'])
    memory = subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    rows = [s.split(',') for s in memory.splitlines()]
    row = next(z for z in rows if z[0].strip() == manifest['gpu_uuid'])
    free = int(row[1].strip())
    if free < 8192:
        raise RuntimeError('LESS_THAN_8192MIB_FREE_GPU')
    write(ROOT/'artifacts'/('%s_hardware.json' % name), {
        'selected': selected, 'free_mib': free, 'used_mib': int(row[2].strip()),
        'utilization_percent': int(row[3].strip()),
        'compute_processes': subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
            '--format=csv,noheader'], text=True),
    })


def command(manifest, arm, name, smoke=False):
    data = manifest['data_root']
    if smoke:
        data = str(ROOT/'review/smoke_sequences')
    cmd = [PYTHON, str(ROOT/'tools/verified_gpu.py'), '--uuid', manifest['gpu_uuid'],
           '--', str(ROOT/'tools/inference_without_gt.py'),
           str(ROOT/'tools/track_dynamic_online.py'),
           '-f', str(ROOT/'host/exps/example/u2mot/yolox_x_u2mot_visdrone.py'),
           '-c', manifest['detector'], '-d', '0', '--fp16', '--fuse', data,
           '--benchmark', 'VisDrone', '--cmc-method', 'file',
           '--cmc-file-dir', 'VisDrone/test-dev', '--new-track-thresh', '0.6',
           '--new-track-min-class-confidence', '0.7', '--min-box-area', '1',
           '--aspect-ratio-thresh', '100000', '--experiment-name', name,
           '--belief-arm', arm['runtime_arm'], '--belief-bundle', str(ROOT/arm['bundle']),
           '--belief-egia-model', str(ROOT/'models/summary_frozen.pkl')]
    if smoke:
        cmd += ['--belief-max-frames', str(manifest['smoke_frames'])]
    return cmd


def audit_runtime(reports, arm, smoke=False):
    mask, penalty = arm['mask'], arm['penalty']
    for sequence, r in reports.items():
        c = r['counts']
        f = r['cost_factorial']
        assert f['assignment_mask_enabled'] == mask
        assert f['semantic_penalty_enabled'] == penalty
        assert r['src_state_enabled'] and r['src_update_enabled'] and r['use_egia']
        assert r['egia_fusion_policy'] == arm['fusion']
        assert r['legacy_egia_loaded'] == arm['fusion']
        assert r['gt_read'] is False
        assert c.get('belief_initializations', 0) > 0 and c.get('belief_updates', 0) > 0
        assert c.get('src_responsibility_calls', 0) > 0
        assert c.get('egia_source_probability_calls', 0) > 0
        if arm['fusion']:
            assert c.get('egia_fusion_events', 0) > 0
        else:
            assert c.get('egia_fusion_events', 0) == 0
        assert c.get('factorial_illegal_edge_recoveries', 0) == 0
        assert c.get('factorial_assignment_verified_seams', 0) > 0
        if arm['corrected_reference_oracle']:
            assert c.get('corrected_reference_seams_verified', 0) == c['association_seams']
        else:
            assert c.get('corrected_reference_seams_verified', 0) == 0
        assert c.get('src_semantic_penalty_calls', 0) == (c.get('src_cost_evaluation_calls', 0))
        if penalty:
            assert c.get('src_semantic_penalty_calls', 0) > 0
        else:
            assert c.get('src_semantic_penalty_calls', 0) == 0
        if not mask and not penalty:
            assert c.get('committed_changed_vs_native_seams', 0) == 0
            assert c.get('native_assignment_control_seams', 0) > 0
        if not arm['selective']:
            assert all(v == 0 for k, v in c.items() if k.startswith('selective_'))
        if not smoke:
            assert r.get('input_stream_sha256') and len(r['input_stream_sha256']) == 64
        assert r['input_frames_hashed'] == r['frames_seen']
        assert [z['frame'] for z in r['input_records']] == list(range(1, r['frames_seen']+1))


def run_arm(manifest, arm, env, smoke=False):
    check_frozen(manifest)
    name = ('SMOKE_' if smoke else '') + arm['name']
    output = ROOT/'YOLOX_outputs'/name
    if output.exists():
        raise RuntimeError('EXCLUSIVE_OUTPUT_ALREADY_EXISTS: ' + str(output))
    launch_gpu(manifest, name)
    started = time.time()
    cmd = command(manifest, arm, name, smoke)
    write(ROOT/'configs'/(name+'_command.json'), {'command': cmd, 'arm': arm,
          'environment': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
          'UAVDT_ONLINE_HOST', 'BELIEF_HOST', 'PYTHONPATH')}})
    log = ROOT/'logs'/(name+'.log')
    run(cmd, env, log)
    hardware = device_receipt(log)
    assert hardware['cuda_uuid'] == manifest['gpu_uuid']
    files = {p.stem: p for p in (output/'track_res').glob('*.txt')}
    expected = {manifest['smoke_sequence']: manifest['smoke_frames']} if smoke else manifest['sequence_frames']
    assert set(files) == set(expected), 'INCOMPLETE_SEQUENCE_SET'
    reports = {s: json.loads(Path(str(f)+'.runtime.json').read_text()) for s, f in files.items()}
    assert all(reports[s]['frames_seen'] == frames for s, frames in expected.items())
    audit_runtime(reports, arm, smoke)
    guard = json.loads((ROOT/'artifacts'/(name+'_gt_guard.json')).read_text())
    assert guard['status'] == 'PASS_NO_GT_READ' and guard['blocked_gt_attempts'] == 0
    hashes = {s: sha(f) for s, f in files.items()}
    record = {'name': name, 'status': 'PASS_ENGINEERING_SMOKE' if smoke else 'PASS_ONLINE_INFERENCE',
              'elapsed_seconds': time.time()-started, 'sequence_count': len(files),
              'frames': sum(expected.values()), 'gpu_uuid': manifest['gpu_uuid'],
              'same_process_gpu_verification': hardware, 'result_sha256': hashes,
              'bundle_sha256': sha(ROOT/arm['bundle']), 'gt_guard': guard,
              'runtime_totals': {k: sum(r['counts'].get(k, 0) for r in reports.values())
                                 for k in set(k for r in reports.values() for k in r['counts'])},
              'input_sha256': {s: r['input_stream_sha256'] for s, r in reports.items()}}
    if not smoke and arm['name'] != 'C11_full_reference':
        reference_inputs = json.loads((ROOT/'YOLOX_outputs/C11_full_reference/receipt.json').read_text())
        assert record['input_sha256'] == reference_inputs['input_sha256'], 'DETECTOR_EMBEDDING_INPUTS_CHANGED'
    if smoke:
        if arm.get('historical_reference_metrics'):
            ref = Path(json.loads(Path(arm['historical_reference_metrics']).read_text())['results_folder'])/next(iter(files.values())).name
            expected_lines = [z for z in ref.read_text().splitlines()
                              if int(z.split(',')[0]) <= manifest['smoke_frames']]
            record['historical_prediction_comparison'] = {
                'byte_equal': files[manifest['smoke_sequence']].read_text().splitlines() == expected_lines,
                'role': 'diagnostic only; historical metadata binding defect is corrected'}
        write(output/'receipt.json', record)
        return record
    if arm.get('historical_reference_metrics'):
        ref = json.loads(Path(arm['historical_reference_metrics']).read_text())
        mismatch = [s for s, h in hashes.items() if h != ref['result_files_sha256'][s]]
        record['historical_prediction_comparison'] = {
            'matched': 17-len(mismatch), 'mismatches': mismatch,
            'role': 'diagnostic only; historical metadata binding defect is corrected'}
    elif arm['name'] != 'C11_full_reference':
        reference = json.loads((ROOT/'YOLOX_outputs/C11_full_reference/receipt.json').read_text())
        assert record['input_sha256'] == reference['input_sha256'], 'DETECTOR_EMBEDDING_INPUTS_CHANGED'
    evaluation_env = env.copy()
    evaluation_env.update(CUDA_VISIBLE_DEVICES='', VISDRONE_EVAL_SPLIT='test-dev',
                          VISDRONE_GT_FOLDER=manifest['gt_root'],
                          VISDRONE_RESULTS_FOLDER=str(output/'track_res'),
                          VISDRONE_METRICS_JSON=str(output/'tracking_metrics.json'))
    run([PYTHON, str(ROOT/'host/tools/utils/eval_visdrone.py')], evaluation_env,
        ROOT/'logs'/(name+'_evaluation.log'))
    metric = json.loads((output/'tracking_metrics.json').read_text())
    reference = json.loads(Path(manifest['full_reference_metrics']).read_text())
    assert metric['result_files_sha256'] == hashes
    assert metric['groundtruth_files_sha256'] == reference['groundtruth_files_sha256']
    assert metric['expected_sequences'] == 17 and metric['eval_split'] == 'test-dev'
    assert all(metric['metrics'][s]['num_objects'] == reference['metrics'][s]['num_objects']
               for s in list(expected)+['OVERALL'])
    record.update(status='PASS_SCORED_ARM_COMPLETE', metrics=metric['metrics']['OVERALL'],
                  metrics_sha256=sha(output/'tracking_metrics.json'))
    write(output/'receipt.json', record)
    check_frozen(manifest)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'manifest.json').read_text())
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=manifest['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
               U2MOT_GPU_LOCK_ACTIVE='1', UAVDT_ONLINE_HOST=str(ROOT/'host'),
               BELIEF_HOST=str(ROOT/'host'), PYTHONPATH=str(ROOT),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               LEAF_GT_GUARD_ROOT=manifest['gt_root'])
    if args.smoke_only:
        arms = manifest['arms']
        records = []
        for arm in arms:
            env['LEAF_GT_GUARD_RECEIPT'] = str(ROOT/'artifacts'/('SMOKE_'+arm['name']+'_gt_guard.json'))
            records.append(run_arm(manifest, arm, env, True))
        assert all(r['input_sha256'] == records[0]['input_sha256'] for r in records)
        write(ROOT/'artifacts/smoke_receipt.json', {
            'status': 'PASS_CORRECTED_200_FRAME_ENGINEERING_SMOKE',
            'manifest_sha256': sha(ROOT/'manifest.json'), 'arms': records})
        return
    smoke = json.loads((ROOT/'artifacts/smoke_receipt.json').read_text())
    assert smoke['status'] == 'PASS_CORRECTED_200_FRAME_ENGINEERING_SMOKE'
    assert smoke['manifest_sha256'] == sha(ROOT/'manifest.json')
    assert len(smoke['arms']) == len(manifest['arms']) == 9
    assert (ROOT/'artifacts/smoke_launcher.exit').read_text().strip() == '0'
    identity = json.loads((ROOT/'artifacts/INPUT_IDENTITY_AUDIT_GATE.json').read_text())
    assert identity['status'] == 'RESOLVED_DETECTION_METADATA_IDENTITY' and identity['sealed'] is True
    assert identity['manifest_sha256'] == sha(ROOT/'manifest.json')
    assert identity['evidence_sha256']
    for path, expected in identity['evidence_sha256'].items():
        assert sha(path) == expected, 'IDENTITY_AUDIT_EVIDENCE_CHANGED'
    lock = ROOT/'artifacts/live.lock'
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, ('pid=%d\n' % os.getpid()).encode()); os.close(fd)
    records = []
    for arm in manifest['arms']:
        write(ROOT/'artifacts/progress.json', {'status': 'RUNNING', 'active_arm': arm['name'],
              'completed_arms': records, 'updated_utc': datetime.datetime.utcnow().isoformat()+'Z'})
        env['LEAF_GT_GUARD_RECEIPT'] = str(ROOT/'artifacts'/(arm['name']+'_gt_guard.json'))
        record = run_arm(manifest, arm, env)
        records.append(record)
        write(ROOT/'artifacts/progress.json', {'status': 'RUNNING', 'active_arm': None, 'completed_arms': records})
    fields = ['mota', 'idf1', 'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations']
    lines = ['arm,'+','.join(fields)]
    for r in records:
        lines.append(r['name']+','+','.join(str(r['metrics'][k]) for k in fields))
    (ROOT/'artifacts/results.csv').write_text('\n'.join(lines)+'\n')
    write(ROOT/'artifacts/receipt.json', {'status': 'PASS_PREDECLARED_ONLINE_MECHANISM_ADDENDUM_COMPLETE',
          'manifest_sha256': sha(ROOT/'manifest.json'), 'arms': records, 'testdev_selection': False})
    write(ROOT/'artifacts/progress.json', {'status': 'COMPLETE', 'active_arm': None, 'completed_arms': records})
    lock.unlink()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(ROOT/'artifacts/stop.json', {'status': 'STOP', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise
