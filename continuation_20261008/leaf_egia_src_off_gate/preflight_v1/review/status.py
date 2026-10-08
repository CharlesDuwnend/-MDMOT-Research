"""Read the actual live workers, sequence receipts and terminal states."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
result={'runtime_root':str(ROOT),'free_gib':round(shutil.disk_usage(ROOT).free/2**30,2),
    'session_live':subprocess.run(['tmux','has-session','-t','leaf_src_off_gate_grid_20261008_v1'],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0}
for phase in ['smoke','full']:
    p=ROOT/'artifacts'/(phase+'_progress.json')
    if not p.exists():continue
    d=json.loads(p.read_text())
    active={}
    for gpu,name in d['active'].items():
        output=('SMOKE_' if phase=='smoke' else '')+name
        files=list((ROOT/'YOLOX_outputs'/output/'track_res').glob('*.txt.runtime.json'))
        item={'name':name,'gpu':int(gpu),'completed_sequences':len(files),
            'completed_frames':sum(json.loads(p.read_text())['frames_seen'] for p in files)}
        log=ROOT/'logs'/(output+'.log')
        if log.exists():
            lines=[z for z in log.read_text(errors='replace').splitlines() if z.startswith('{"sequence":')]
            if lines:item['latest_frame']=json.loads(lines[-1])
        active[gpu]=item
    result[phase]=dict(status=d['status'],completed=d['completed'],total=d['total'],active=active)
    lock=ROOT/'artifacts'/(phase+'.live.lock')
    if lock.exists():
        pid=int(lock.read_text().split('=')[1]);result[phase]['coordinator_pid']=pid
        proc=Path('/proc')/str(pid)
        result[phase]['process_present']=proc.exists()
        if proc.exists():result[phase]['process_cmdline']=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
for n in ['smoke.exit','full.exit','pipeline.exit','grid_stop.json']:
    p=ROOT/'artifacts'/n
    if p.exists():result[n]=json.loads(p.read_text()) if p.suffix=='.json' else p.read_text().strip()
for n in ['FINALIZATION_RECEIPT.json','finalization.exit']:
    p=ROOT/'review'/n
    if p.exists():result[n]=json.loads(p.read_text()) if p.suffix=='.json' else p.read_text().strip()
print(json.dumps(result,ensure_ascii=False))
