"""Complete side-by-side presentation and exact-scope archive after final audit."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/home/chenhc/claude_try_MDMOT')
PREFIX='continuation_20261008/leaf_egia_src_off_gate/paired_presentation_v1/'
STAGE='leaf_egia_src_off_gate_verified_paired_tables_20261008_v1'
PYTHON='/home/chenhc/.conda/envs/u2mot/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=REPO,env=env)

def main():
    while not (ROOT/'review/finalization.exit').exists():
        stop=ROOT/'artifacts/grid_stop.json'
        if stop.exists():raise RuntimeError('GRID_STOP_REQUIRES_REPAIR_BEFORE_DELIVERY')
        time.sleep(30)
    assert (ROOT/'review/finalization.exit').read_text().strip()=='0','FINAL_AUDIT_OR_ARCHIVE_FAILED'
    seal=json.loads((ROOT/'review/PAIRED_PRESENTATION_PREREGISTRATION.json').read_text())
    assert sha(ROOT/'review/paired_gate_table.py')==seal['source_sha256']
    assert seal['all_primary_cells']==20 and seal['presentation_only']
    result=subprocess.run([PYTHON,str(ROOT/'review/paired_gate_table.py')],cwd=ROOT,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,check=True)
    out=ROOT/'artifacts/completed_grid'
    final=json.loads((ROOT/'review/FINALIZATION_RECEIPT.json').read_text())
    assert final['status']=='COMPLETE_VERIFIED_23_CELLS_AND_BOTH_GATE_TABLES'
    assert sha(out/'FINAL_AUDIT.json')==final['final_audit_sha256']
    audit=json.loads((out/'FINAL_AUDIT.json').read_text())
    assert audit['cells']==23 and audit['prediction_files_rehashed']==391
    dest=REPO/PREFIX;dest.mkdir(exist_ok=False)
    for name in ['PAIRED_GATE_TABLE_ZH.md','paired_gate_results.csv']:shutil.copy2(out/name,dest/name)
    for name in ['paired_gate_table.py','PAIRED_PRESENTATION_PREREGISTRATION.json',
                 'deliver_paired_when_complete.py','FINALIZATION_RECEIPT.json']:
        shutil.copy2(ROOT/'review'/name,dest/name)
    inventory={p.name:sha(p) for p in dest.iterdir()}
    (dest/'PRESENTATION_RECEIPT.json').write_text(json.dumps(dict(
        status='ALL_TWENTY_PRIMARY_CELLS_PAIRED_AFTER_ALL_23_VERIFIED',
        files_sha256=inventory,source_final_audit_sha256=sha(out/'FINAL_AUDIT.json'),
        presentation_output=json.loads(result.stdout),remote_backup=False),indent=2)+'\n')
    expected={str(p.relative_to(REPO)) for p in dest.iterdir()}
    eligible={p.decode() for p in git('ls-files','--cached','--others','--exclude-standard','-z','--',PREFIX).split(b'\0') if p}
    assert expected==eligible and len(eligible)==7
    before=git('diff','--cached','--binary');old=git('rev-parse','HEAD').decode().strip()
    fd,index=tempfile.mkstemp(prefix='leaf_paired_gate_table_index_',dir='/tmp');os.close(fd);os.unlink(index)
    env=os.environ.copy();env['GIT_INDEX_FILE']=index
    try:
        git('read-tree','HEAD',env=env)
        cmd=['python3','scripts/stage_snapshot.py','--stage',STAGE,'--message',
            'LEAF EGIA: deliver both gate states side by side after verifying every declared full run','--local-only']
        for p in sorted(eligible):cmd+=['--include',p]
        subprocess.run(cmd,cwd=REPO,env=env,check=True)
    finally:
        head=git('rev-parse','HEAD').decode().strip()
        if head!=old:
            names=git('diff','--name-only',old,head).decode().splitlines()
            assert all(p.startswith(PREFIX) or p.startswith('versioning/stages/'+STAGE+'/') for p in names)
            git('restore','--staged','--source=HEAD','--',*names)
        assert git('diff','--cached','--binary')==before
        if Path(index).exists():Path(index).unlink()
    receipt=dict(status='COMPLETE_GRID_AND_PAIRED_TABLES_WITH_LOCAL_ARCHIVE',
        paired_table=str(dest/'PAIRED_GATE_TABLE_ZH.md'),all_results=str(Path(final['archive_root'])/'RESULTS_FOR_AUTHOR_ZH.md'),
        local_head=head,remote_backup=False,preserved_staged_diff_sha256=hashlib.sha256(before).hexdigest())
    (ROOT/'review/PAIRED_DELIVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'review/paired_delivery.exit').write_text('0\n')

if __name__=='__main__':
    try:main()
    except BaseException as error:
        (ROOT/'review/PAIRED_DELIVERY_RECEIPT.json').write_text(json.dumps(dict(
            status='PAIRED_DELIVERY_FAILED_NO_COMPLETE_CLAIM',error=str(error),traceback=traceback.format_exc()),indent=2)+'\n')
        (ROOT/'review/paired_delivery.exit').write_text('1\n')
        raise
