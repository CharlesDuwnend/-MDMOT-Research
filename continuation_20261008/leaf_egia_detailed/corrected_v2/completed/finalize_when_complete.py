"""Finalize tables and exact-scope local Git evidence after terminal completion."""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/home/chenhc/claude_try_MDMOT')
PYTHON='/home/chenhc/.conda/envs/u2mot/bin/python'
PREFIX='continuation_20261008/leaf_egia_detailed/corrected_v2/completed/'
STAGE='leaf_egia_detailed_all9_complete_20261008_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    p=Path(path);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(p)


def git(*args,env=None):
    return subprocess.check_output(['git',*args],cwd=REPO,env=env)


def archive():
    dest=REPO/PREFIX
    dest.mkdir(exist_ok=False)
    for p in (ROOT/'artifacts/completed_detailed').iterdir():
        assert p.is_file() and p.suffix in ['.json','.md','.csv']
        shutil.copy2(p,dest/p.name)
    for name in ['analyze_detailed.py','finalize_when_complete.py','monitor_progress.py']:
        shutil.copy2(ROOT/'review'/name,dest/name)
    for name in ['parallel_coordinator.py','launch_parallel.sh','GPU_SCHEDULING_AMENDMENT.json',
        'STORAGE_RELOCATION_RECEIPT.json','relocate_closed_artifacts.py','SERIAL_SCHEDULER_HANDOFF.json',
        'EXISTING_INFERENCE_COMPLETION_RECEIPT.json','original_serial_launcher.exit',
        'original_serial_pipeline.exit','D07_no_geometry_cue_CROSS_GPU_SMOKE.json',
        'D08_no_appearance_cue_CROSS_GPU_SMOKE.json']:
        shutil.copy2(ROOT/'review'/name,dest/name)
    preserved=dest/'postprocessing_serial_pre_parallel'
    shutil.copytree(ROOT/'review/postprocessing_serial_pre_parallel',preserved)
    for name in ['manifest.json','PREREGISTRATION.json']:
        shutil.copy2(ROOT/name,dest/name)
    for name in ['receipt.json','INPUT_IDENTITY_AUDIT_GATE.json','smoke_receipt.json',
                 'DETAILED_CPU_GATE.json','PRIOR_AND_BUNDLE_RECEIPT.json',
                 'V2_PREFLIGHT_RESOLUTION_AUDIT.json','pipeline.exit','launcher.exit','smoke_launcher.exit']:
        shutil.copy2(ROOT/'artifacts'/name,dest/name)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for arm in manifest['arms']:
        target=dest/'scores'/arm['name'];target.mkdir(parents=True)
        for name in ['tracking_metrics.json','receipt.json']:
            shutil.copy2(ROOT/'YOLOX_outputs'/arm['name']/name,target/name)
    for name in ['stop.json','pipeline.exit','smoke_launcher.exit']:
        p=Path('/home/chenhc/leaf_egia_detailed_20261008_v1/artifacts')/name
        if p.exists():
            target=dest/'initial_engineering_attempt';target.mkdir(exist_ok=True)
            shutil.copy2(p,target/name)
    write(dest/'DELIVERY_ARCHIVE_RECEIPT.json',{
        'status':'COMPLETE_NINE_FULL_SCORED_CONTROLS_AND_VERIFIED_TABLES',
        'runtime_root':str(ROOT),'raw_predictions_or_model_binaries_in_git':False,
        'manifest_sha256':sha(ROOT/'manifest.json'),
        'files_sha256':{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()},
        'formal_scope':'Original-GT VisDrone2019 test-dev; 17 sequences x 6635 frames x 9 cells',
        'remote_write_dependency':'GitHub repository write denied to authenticated account luwnend; local-only snapshot'})
    (dest/'README.md').write_text('# Completed full-SRC EGIA detailed ablation\n\nNine declared cells completed actual detector/ReID inference and original-GT evaluation. The full reference matches corrected C11 in all 17 prediction hashes; all detector/ReID ordered-array ledgers match. Source/model/input/solver/GT/device/terminal checks are recorded in FINAL_AUDIT.json. Tables retain every result. Neutral geometry and appearance controls are algebraically equivalent to zero coherence and verified byte-identical; do not count them as independent gains.\n\nBinary weights, raw predictions/runtime ledgers and rendered artifacts remain local at '+str(ROOT)+'. Only compact scores, receipts, hashes, source and tables are archived. Current authenticated GitHub identity lacks remote repository write access; the exact-scope snapshot is local-only.\n')
    eligible=set(p.decode() for p in git('ls-files','--cached','--others','--exclude-standard','-z','--',PREFIX).split(b'\0') if p)
    expected=set(str(p.relative_to(REPO)) for p in dest.rglob('*') if p.is_file())
    assert eligible==expected and len(eligible)>30 and all(p.startswith(PREFIX) for p in eligible)
    before=git('diff','--cached','--binary')
    oldhead=git('rev-parse','HEAD').strip()
    fd,index_path=tempfile.mkstemp(prefix='leaf_egia_final_exact_index_',dir='/tmp')
    os.close(fd);os.unlink(index_path)
    env=os.environ.copy();env['GIT_INDEX_FILE']=index_path
    try:
        git('read-tree','HEAD',env=env)
        cmd=['python3','scripts/stage_snapshot.py','--stage',STAGE,
             '--message','LEAF EGIA: complete and verify nine full-SRC frozen-head ablations; retain all signed results',
             '--local-only']
        for name in sorted(eligible):cmd+=['--include',name]
        subprocess.run(cmd,cwd=REPO,env=env,check=True)
    finally:
        newhead=git('rev-parse','HEAD').strip()
        if newhead!=oldhead:
            changes=git('diff','--name-only',oldhead.decode(),newhead.decode()).decode().splitlines()
            assert all(p.startswith(PREFIX) or p.startswith('versioning/stages/'+STAGE+'/') for p in changes)
            git('restore','--staged','--source=HEAD','--',*changes)
        assert git('diff','--cached','--binary')==before,'UNRELATED_STAGED_DIFF_CHANGED'
        if Path(index_path).exists():Path(index_path).unlink()
    return dict(local_head=newhead.decode(),stage=STAGE,archive_root=str(dest),
                preserved_staged_diff_sha256=hashlib.sha256(before).hexdigest(),remote_backup=False)


def main():
    (ROOT/'review/finalization.lock').open('x').close()
    write(ROOT/'review/finalization_progress.json',{'status':'WAITING_FOR_ALL_NINE_FULL_RUNS'})
    while not (ROOT/'artifacts/pipeline.exit').exists():
        time.sleep(30)
    assert (ROOT/'artifacts/pipeline.exit').read_text().strip()=='0','INFERENCE_PIPELINE_FAILED'
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',BELIEF_HOST=str(ROOT/'host'),UAVDT_ONLINE_HOST=str(ROOT/'host'),
               PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+':'+str(ROOT/'tools'))
    with (ROOT/'review/analysis.log').open('x') as out:
        subprocess.run([PYTHON,str(ROOT/'review/analyze_detailed.py')],cwd=ROOT,env=env,
                       stdout=out,stderr=subprocess.STDOUT,check=True)
    archival=archive()
    write(ROOT/'review/FINALIZATION_RECEIPT.json',dict(status='COMPLETE_VERIFIED_TABLES_AND_LOCAL_SNAPSHOT',
        completed_utc=datetime.datetime.utcnow().isoformat()+'Z',
        final_audit_sha256=sha(ROOT/'artifacts/completed_detailed/FINAL_AUDIT.json'),
        author_table=str(ROOT/'artifacts/completed_detailed/RESULTS_FOR_AUTHOR_ZH.md'),**archival))
    (ROOT/'review/finalization.exit').write_text('0\n')
    (ROOT/'review/finalization.lock').unlink()


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        write(ROOT/'review/FINALIZATION_RECEIPT.json',dict(status='FINALIZATION_FAILED_NO_COMPLETE_CLAIM',
            error=str(error),traceback=traceback.format_exc()))
        (ROOT/'review/finalization.exit').write_text('1\n')
        raise
