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
           '--new-track-min-class-confidence', str(arm['birth_class_gate']), '--min-box-area', '1',
           '--aspect-ratio-thresh', '100000', '--experiment-name', name,
           '--belief-arm', arm['runtime_arm'], '--belief-bundle', str(ROOT/arm['bundle'])]
    if smoke:
        cmd += ['--belief-max-frames', str(manifest['smoke_frames'])]
    return cmd


def audit_runtime(reports, arm, smoke=False):
    from grid_audit import audit_grid_runtime
    return audit_grid_runtime(reports, arm, smoke)


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
    if not smoke:
        reference_inputs = json.loads(Path(manifest['input_reference_receipt']).read_text())
        assert record['input_sha256'] == reference_inputs['input_sha256'], 'DETECTOR_EMBEDDING_INPUTS_CHANGED'
    if smoke:
        if arm.get('historical_reference_metrics'):
            ref = Path(json.loads(Path(arm['historical_reference_metrics']).read_text())['results_folder'])/next(iter(files.values())).name
            expected_lines = [z for z in ref.read_text().splitlines()
                              if int(z.split(',')[0]) <= manifest['smoke_frames']]
            record['historical_prediction_comparison'] = {
                'byte_equal': files[manifest['smoke_sequence']].read_text().splitlines() == expected_lines,
                'role': 'hard parity requirement against completed corrected C11'}
            assert record['historical_prediction_comparison']['byte_equal'], 'FULL_REFERENCE_SMOKE_PREDICTIONS_CHANGED'
        write(output/'receipt.json', record)
        return record
    if arm.get('historical_reference_metrics'):
        ref = json.loads(Path(arm['historical_reference_metrics']).read_text())
        mismatch = [s for s, h in hashes.items() if h != ref['result_files_sha256'][s]]
        record['historical_prediction_comparison'] = {
            'matched': 17-len(mismatch), 'mismatches': mismatch,
            'role': 'hard parity requirement against completed corrected C11'}
        assert not mismatch, 'FULL_REFERENCE_PREDICTIONS_CHANGED'
        parent_receipt = json.loads((Path(arm['historical_reference_metrics']).parent/'receipt.json').read_text())
        assert record['input_sha256'] == parent_receipt['input_sha256'], 'FULL_REFERENCE_INPUTS_CHANGED'
    else:
        reference = json.loads(Path(manifest['input_reference_receipt']).read_text())
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


if __name__ == '__main__':
    raise SystemExit('Use run_gate_grid.py; this module only exports verified single-arm execution.')
