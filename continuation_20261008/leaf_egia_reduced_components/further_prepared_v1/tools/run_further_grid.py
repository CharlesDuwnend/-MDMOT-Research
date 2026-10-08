"""One pure full reference smoke, three new independent GPU full runs."""
import concurrent.futures,datetime,json,os,sys,threading,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_mechanism import run_arm,sha,write,check_frozen

def environment(m,a,smoke):
    name=('SMOKE_' if smoke else '')+a['name']
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=a['gpu_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
        U2MOT_GPU_LOCK_ACTIVE='1',UAVDT_ONLINE_HOST=str(ROOT/'host'),BELIEF_HOST=str(ROOT/'host'),
        PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        LEAF_GT_GUARD_ROOT=m['gt_root'],LEAF_GT_GUARD_RECEIPT=str(ROOT/'artifacts'/(name+'_gt_guard.json')))
    return env

def main():
    smoke='--smoke-only' in sys.argv
    m=json.loads((ROOT/'manifest.json').read_text());check_frozen(m)
    marker='smoke' if smoke else 'full'
    lock=ROOT/'artifacts'/(marker+'.live.lock')
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,('pid=%d\n'%os.getpid()).encode());os.close(fd)
    if not smoke:
        receipt=json.loads((ROOT/'artifacts/smoke_grid_receipt.json').read_text())
        assert receipt['status']=='PASS_PURE_REFERENCE_AND_THREE_NEW_SMOKES'
        assert receipt['manifest_sha256']==sha(ROOT/'manifest.json')
        assert (ROOT/'artifacts/smoke.exit').read_text().strip()=='0'
    records={};active={};state_lock=threading.Lock()
    def progress():
        write(ROOT/'artifacts'/(marker+'_progress.json'),dict(status='RUNNING',active=dict(active),
            completed_names=sorted(records),completed=len(records),total=len(m['arms']),
            updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    def execute(a):
        assigned=dict(m,gpu_uuid=a['gpu_uuid'],gpu_physical_index=a['gpu_physical_index'])
        with state_lock:active[str(a['gpu_physical_index'])]=a['name'];progress()
        result=run_arm(assigned,a,environment(m,a,smoke),smoke)
        with state_lock:
            records[a['name']]=result;active.pop(str(a['gpu_physical_index']));progress()
        return result
    reference=execute(m['smoke_reference']) if smoke else None
    if smoke:records.clear()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(execute,a) for a in m['arms']]
        for f in concurrent.futures.as_completed(futures):f.result()
    ordered=[records[a['name']] for a in m['arms']]
    assert len(ordered)==3 and all(r['input_sha256']==ordered[0]['input_sha256'] for r in ordered)
    if smoke:
        expected=json.loads(Path(m['smoke_input_reference_receipt']).read_text())['input_sha256']
        assert reference['input_sha256']==expected==ordered[0]['input_sha256']
        status='PASS_PURE_REFERENCE_AND_THREE_NEW_SMOKES'
    else:status='PASS_ALL_THREE_FURTHER_PURE_ARMS_SCORED'
    write(ROOT/'artifacts'/('smoke_grid_receipt.json' if smoke else 'grid_receipt.json'),dict(
        status=status,manifest_sha256=sha(ROOT/'manifest.json'),arms=ordered,
        reference_smoke=reference,testdev_selection=False))
    write(ROOT/'artifacts'/(marker+'_progress.json'),dict(status='COMPLETE',active={},
        completed_names=[a['name'] for a in m['arms']],completed=3,total=3))
    lock.unlink()

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'artifacts/grid_stop.json',dict(status='STOP_NO_FORMAL_COMPLETION_CLAIM',
            phase='smoke' if '--smoke-only' in sys.argv else 'full',error=str(error),traceback=traceback.format_exc()))
        raise
