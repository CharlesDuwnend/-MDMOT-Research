"""User-authorized scheduling amendment; frozen inference operators unchanged."""
import ast
import concurrent.futures
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_mechanism as original

SERIAL_PID=859315
INFERENCE_PID=918776
ALLOCATIONS={
 'D07_no_geometry_cue':{'index':0,'uuid':'GPU-2a55197e-2c19-45ff-cf33-13031951f9d0'},
 'D08_no_appearance_cue':{'index':3,'uuid':'GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a'}}

def read(path):return json.loads(Path(path).read_text())

def environment(manifest,name):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=manifest['gpu_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
        U2MOT_GPU_LOCK_ACTIVE='1',UAVDT_ONLINE_HOST=str(ROOT/'host'),BELIEF_HOST=str(ROOT/'host'),
        PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        LEAF_GT_GUARD_ROOT=manifest['gt_root'],
        LEAF_GT_GUARD_RECEIPT=str(ROOT/'artifacts'/(name+'_gt_guard.json')))
    return env

def complete_current(manifest):
    # Wait only for the already-launched inference child. Its paused parent
    # cannot launch a duplicate arm; the child and all data workers continue.
    exit_verified=False
    while Path(f'/proc/{INFERENCE_PID}').exists():
        stat=Path(f'/proc/{INFERENCE_PID}/stat').read_text()
        fields=stat.rsplit(')',1)[1].split()
        if fields[0]=='Z':
            assert int(fields[49])==0,'IN_FLIGHT_INFERENCE_EXITED_NONZERO'
            exit_verified=True
            break
        time.sleep(5)
    assert exit_verified,'INFERENCE_EXIT_STATUS_NOT_VERIFIABLE'
    tree=ast.parse((ROOT/'tools/run_mechanism.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_arm')
    start=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='hardware' for t in n.targets))
    function.name='complete_previously_launched_arm'
    function.args=ast.parse('def f(manifest,arm,env,started,name,output,log,smoke=False): pass').body[0].args
    function.body=function.body[start:]
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    exec(compile(module,'<unchanged_run_arm_post_inference_suffix>','exec'),original.__dict__)
    name='D06_broken_source_pairing'
    arm=next(a for a in manifest['arms'] if a['name']==name)
    started=(ROOT/'artifacts'/(name+'_hardware.json')).stat().st_mtime
    record=original.complete_previously_launched_arm(manifest,arm,environment(manifest,name),
        started,name,ROOT/'YOLOX_outputs'/name,ROOT/'logs'/(name+'.log'))
    original.write(ROOT/'review/EXISTING_INFERENCE_COMPLETION_RECEIPT.json',dict(
        status='ORIGINAL_INFERENCE_COMPLETED_WITH_UNCHANGED_VALIDATION_AND_SCORING',
        inference_pid=INFERENCE_PID,serial_pid=SERIAL_PID,
        suffix_ast_sha256=hashlib.sha256(ast.unparse(module).encode()).hexdigest(),
        source_sha256=original.sha(ROOT/'tools/run_mechanism.py'),
        elapsed_time_origin='hardware receipt modification time; scheduler metadata only',
        metrics_sha256=record['metrics_sha256']))
    return record

def launch_remaining(manifest,arm):
    assigned=copy.deepcopy(manifest)
    assigned['gpu_uuid']=ALLOCATIONS[arm['name']]['uuid']
    smoke_arm=copy.deepcopy(arm)
    smoke_arm['name']=arm['name']+'_parallel_device_check'
    smoke_name='SMOKE_'+smoke_arm['name']
    smoke=original.run_arm(assigned,smoke_arm,environment(assigned,smoke_name),True)
    reference=read(ROOT/'YOLOX_outputs'/('SMOKE_'+arm['name'])/'receipt.json')
    assert smoke['input_sha256']==reference['input_sha256'],'CROSS_GPU_SMOKE_INPUTS_DIFFER'
    assert smoke['result_sha256']==reference['result_sha256'],'CROSS_GPU_SMOKE_PREDICTIONS_DIFFER'
    original.write(ROOT/'review'/(arm['name']+'_CROSS_GPU_SMOKE.json'),dict(
        status='PASS_BYTE_IDENTICAL_200_FRAME_CROSS_GPU_SMOKE',
        original_receipt_sha256=original.sha(ROOT/'YOLOX_outputs'/('SMOKE_'+arm['name'])/'receipt.json'),
        new_receipt_sha256=original.sha(ROOT/'YOLOX_outputs'/smoke_name/'receipt.json'),
        allocation=ALLOCATIONS[arm['name']],smoke_record=smoke))
    return original.run_arm(assigned,arm,environment(assigned,arm['name']))

def retire_serial():
    assert b'tools/run_mechanism.py' in Path(f'/proc/{SERIAL_PID}/cmdline').read_bytes()
    os.kill(SERIAL_PID,signal.SIGTERM)
    os.kill(SERIAL_PID,signal.SIGCONT)
    while not (ROOT/'artifacts/pipeline.exit').exists():time.sleep(1)
    for name in ['launcher.exit','pipeline.exit']:
        p=ROOT/'artifacts'/name
        assert p.read_text().strip()=='143','UNEXPECTED_SERIAL_HANDOFF_EXIT'
        p.rename(ROOT/'review'/('original_serial_'+name))
    lock=ROOT/'artifacts/live.lock'
    assert lock.read_text().strip()==f'pid={SERIAL_PID}'
    lock.unlink()
    original.write(ROOT/'review/SERIAL_SCHEDULER_HANDOFF.json',dict(
        status='CONTROLLED_SCHEDULER_HANDOFF_NO_INFERENCE_TERMINATED',
        original_scheduler_exit=143,all_started_inference_finished=True,
        old_live_lock=f'pid={SERIAL_PID}',
        reason='User requested remaining arms on separate physical GPUs'))

def main():
    manifest=read(ROOT/'manifest.json')
    original.check_frozen(manifest)
    assert not (ROOT/'artifacts/pipeline.exit').exists()
    assert Path(f'/proc/{SERIAL_PID}/stat').read_text().rsplit(')',1)[1].split()[0]=='T'
    for name in ALLOCATIONS:assert not (ROOT/'YOLOX_outputs'/name).exists()
    source_paths=['review/parallel_coordinator.py','review/analyze_detailed.py',
        'review/finalize_when_complete.py','review/monitor_progress.py','review/launch_parallel.sh']
    allocation={a['name']:{'index':1,'uuid':manifest['gpu_uuid']} for a in manifest['arms']}
    allocation.update(ALLOCATIONS)
    amendment=dict(status='USER_AUTHORIZED_GPU_SCHEDULING_AMENDMENT',
        declared_utc=datetime.datetime.utcnow().isoformat()+'Z',
        user_instruction='多个实验arm可以放在不同的GPU上',
        original_manifest_sha256=original.sha(ROOT/'manifest.json'),
        all_inference_source_model_and_operator_hashes_unchanged=True,
        all_declared_cells_and_operating_points_unchanged=True,
        actual_allocation=allocation,serial_parent_only_paused=True,
        in_flight_inference_continues=True,
        source_sha256={p:original.sha(ROOT/p) for p in source_paths},
        storage_relocation_receipt_sha256=original.sha(ROOT/'review/STORAGE_RELOCATION_RECEIPT.json'),
        scientific_scope='No new outcome-selected control or parameter; cross-GPU input/prediction gates mandatory')
    original.write(ROOT/'review/GPU_SCHEDULING_AMENDMENT.json',amendment)
    (ROOT/'review/parallel.live.lock').write_text(f'pid={os.getpid()}\n')
    original.write(ROOT/'review/parallel_progress.json',dict(status='RUNNING',completed=[],
        active=['D06_broken_source_pairing',*ALLOCATIONS]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(complete_current,manifest):'D06_broken_source_pairing'}
        for arm in manifest['arms']:
            if arm['name'] in ALLOCATIONS:futures[pool.submit(launch_remaining,manifest,arm)]=arm['name']
        completed=[]
        for future in concurrent.futures.as_completed(futures):
            record=future.result();completed.append(record['name'])
            original.write(ROOT/'review/parallel_progress.json',dict(status='RUNNING',completed=completed,
                active=[n for n in futures.values() if n not in completed]))
    retire_serial()
    records=[read(ROOT/'YOLOX_outputs'/a['name']/'receipt.json') for a in manifest['arms']]
    assert all(r['status']=='PASS_SCORED_ARM_COMPLETE' for r in records)
    assert all(r['input_sha256']==records[0]['input_sha256'] for r in records)
    fields=['mota','idf1','num_false_positives','num_misses','num_switches','num_fragmentations']
    lines=['arm,'+','.join(fields)]+[r['name']+','+','.join(str(r['metrics'][k]) for k in fields) for r in records]
    (ROOT/'artifacts/results.csv').write_text('\n'.join(lines)+'\n')
    original.write(ROOT/'artifacts/receipt.json',dict(status='PASS_PREDECLARED_ONLINE_MECHANISM_ADDENDUM_COMPLETE',
        manifest_sha256=original.sha(ROOT/'manifest.json'),arms=records,testdev_selection=False,
        gpu_scheduling_amendment_sha256=original.sha(ROOT/'review/GPU_SCHEDULING_AMENDMENT.json')))
    original.write(ROOT/'artifacts/progress.json',dict(status='COMPLETE',active_arm=None,completed_arms=records))
    original.write(ROOT/'review/parallel_progress.json',dict(status='COMPLETE',completed=completed,active=[]))
    (ROOT/'review/parallel.live.lock').unlink()

if __name__=='__main__':
    try:main()
    except BaseException as error:
        original.write(ROOT/'review/PARALLEL_STOP.json',dict(status='STOP_NO_COMPLETE_CLAIM',
            error=str(error),traceback=traceback.format_exc()))
        raise
