"""Permitted multi-GPU independent-arm queues; all controls retained."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_mechanism import run_arm,sha,write,check_frozen

def env_for(manifest,arm,smoke):
    name=('SMOKE_' if smoke else '')+arm['name']
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=arm['gpu_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
        U2MOT_GPU_LOCK_ACTIVE='1',UAVDT_ONLINE_HOST=str(ROOT/'host'),BELIEF_HOST=str(ROOT/'host'),
        PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        LEAF_GT_GUARD_ROOT=manifest['gt_root'],LEAF_GT_GUARD_RECEIPT=str(ROOT/'artifacts'/(name+'_gt_guard.json')))
    return env

def main():
    smoke='--smoke-only' in sys.argv
    manifest=json.loads((ROOT/'manifest.json').read_text())
    check_frozen(manifest)
    marker='smoke' if smoke else 'full'
    lock=ROOT/'artifacts'/(marker+'.live.lock')
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,('pid=%d\n'%os.getpid()).encode());os.close(fd)
    if not smoke:
        receipt=json.loads((ROOT/'artifacts/smoke_grid_receipt.json').read_text())
        assert receipt['status']=='PASS_ALL_4_ACTUAL_SMOKES_INPUT_AND_REFERENCE_PARITY'
        assert receipt['manifest_sha256']==sha(ROOT/'manifest.json')
        assert (ROOT/'artifacts/smoke.exit').read_text().strip()=='0'
    records={}
    active={}
    import threading
    state_lock=threading.Lock()
    def progress():
        write(ROOT/'artifacts'/(marker+'_progress.json'),dict(status='RUNNING',active=dict(active),
            completed_names=sorted(records),completed=len(records),total=len(manifest['arms']),
            updated_utc=datetime.datetime.utcnow().isoformat()+'Z'))
    def execute(arm):
        assigned=dict(manifest,gpu_uuid=arm['gpu_uuid'],gpu_physical_index=arm['gpu_physical_index'])
        with state_lock:active[str(arm['gpu_physical_index'])]=arm['name'];progress()
        record=run_arm(assigned,arm,env_for(manifest,arm,smoke),smoke)
        with state_lock:
            records[arm['name']]=record
            active.pop(str(arm['gpu_physical_index']))
            progress()
        return record
    # Reproduce full source before allowing other full inference cells.
    execute(manifest['arms'][0])
    def queue(index):
        for arm in manifest['arms'][1:]:
            if arm['gpu_physical_index']==index:execute(arm)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(queue,index) for index in [0,1,3]]
        for future in concurrent.futures.as_completed(futures):future.result()
    ordered=[records[a['name']] for a in manifest['arms']]
    assert len(ordered)==4
    assert all(r['input_sha256']==ordered[0]['input_sha256'] for r in ordered)
    if smoke:
        parent=Path(manifest['smoke_input_reference_receipt'])
        assert ordered[0]['input_sha256']==json.loads(parent.read_text())['input_sha256']
        status='PASS_ALL_4_ACTUAL_SMOKES_INPUT_AND_REFERENCE_PARITY'
    else:status='PASS_ALL_4_SCORED_REDUCED_CELLS_COMPLETE'
    write(ROOT/'artifacts'/('smoke_grid_receipt.json' if smoke else 'grid_receipt.json'),dict(
        status=status,manifest_sha256=sha(ROOT/'manifest.json'),arms=ordered,testdev_selection=False))
    write(ROOT/'artifacts'/(marker+'_progress.json'),dict(status='COMPLETE',active={},
        completed_names=[a['name'] for a in manifest['arms']],completed=4,total=4))
    lock.unlink()

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'artifacts/grid_stop.json',dict(status='STOP_NO_FORMAL_COMPLETION_CLAIM',
            phase='smoke' if '--smoke-only' in sys.argv else 'full',error=str(error),traceback=traceback.format_exc()))
        raise
