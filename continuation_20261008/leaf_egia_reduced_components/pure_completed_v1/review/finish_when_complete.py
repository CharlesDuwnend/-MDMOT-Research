"""Finish verified tables and scoped Git backup only after all runs terminate."""
import datetime,json,os,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/home/chenhc/claude_try_MDMOT')
PREFIX='continuation_20261008/leaf_egia_reduced_components'
PYTHON='/home/chenhc/.conda/envs/u2mot/bin/python'
def write(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def main():
    while not (ROOT/'artifacts/pipeline.exit').exists():
        if (ROOT/'artifacts/grid_stop.json').exists():
            raise RuntimeError('INFERENCE_STOPPED; preserve partial evidence, no final result')
        time.sleep(30)
    assert (ROOT/'artifacts/pipeline.exit').read_text().strip()=='0'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',BELIEF_HOST=str(ROOT/'host'),
        UAVDT_ONLINE_HOST=str(ROOT/'host'),PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (ROOT/'review/finalization.log').open('x') as f:
        subprocess.run([PYTHON,'-B',str(ROOT/'review/finalize_reduced.py')],cwd=ROOT,env=env,
                       stdout=f,stderr=subprocess.STDOUT,check=True)
    dest=REPO/PREFIX/'pure_completed_v1';dest.mkdir(exist_ok=False)
    for p in (ROOT/'review/results').iterdir():shutil.copy2(p,dest/p.name)
    for name in ['manifest.json','PREREGISTRATION.json','EXPERIMENT_PLAN.md',
                 'review/finalize_reduced.py','review/finish_when_complete.py','review/status.py',
                 'artifacts/grid_receipt.json','artifacts/smoke_grid_receipt.json',
                 'artifacts/smoke.exit','artifacts/full.exit','artifacts/pipeline.exit']:
        out=dest/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,out)
    import hashlib
    frozen=json.loads((ROOT/'manifest.json').read_text())
    for a in frozen['arms']:
        folder=dest/'scores'/a['name'];folder.mkdir(parents=True)
        for name in ['receipt.json','tracking_metrics.json']:
            shutil.copy2(ROOT/'YOLOX_outputs'/a['name']/name,folder/name)
    # Reused Native baseline is distinguishable and independently hash checked.
    folder=dest/'scores/native_completed_reference';folder.mkdir(parents=True)
    for name in ['receipt.json','tracking_metrics.json']:
        shutil.copy2(Path('/home/chenhc/leaf_egia_src_off_gate_20261008_v1/YOLOX_outputs/S0_G0_disabled')/name,folder/name)
    inventory={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
    write(dest/'ARCHIVE_RECEIPT.json',dict(status='ALL_FOUR_NEW_PURE_V5_ARMS_COMPLETE_AND_AUDITED',
        files_sha256=inventory,raw_predictions_or_weights_in_git=False,runtime_root=str(ROOT)))
    command=['python3',str(REPO/PREFIX/'archive_phase.py'),PREFIX,
             'leaf_egia_reduced_pure_completed_20261008_v1',
             'LEAF EGIA: complete pure v5 component deletion with no SRC, gate or H2 classifier']
    result=subprocess.run(command,cwd=REPO,capture_output=True,text=True,check=True)
    snapshot=json.loads(result.stdout)
    write(ROOT/'review/FINALIZATION_RECEIPT.json',dict(status='COMPLETE_ACTUAL_COMPONENT_REDUCTION_AND_TABLES',
        completed_utc=datetime.datetime.utcnow().isoformat()+'Z',
        table=str(dest/'RESULTS_ZH.md'),runtime_table=str(ROOT/'review/results/RESULTS_ZH.md'),
        archive_root=str(dest),**snapshot))
    (ROOT/'review/finalization.exit').write_text('0\n')
if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'review/FINALIZATION_RECEIPT.json',dict(status='FINALIZATION_FAILED_NO_COMPLETE_CLAIM',
            error=str(error),traceback=traceback.format_exc()))
        (ROOT/'review/finalization.exit').write_text('1\n')
        raise
