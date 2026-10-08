"""Small read-only status report for the named persistent experiment."""
import json
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    result={'root':str(ROOT),'free_gib':round(shutil.disk_usage(ROOT).free/1024**3,2)}
    result['tmux_active']=subprocess.run(['tmux','has-session','-t',
        'leaf_egia_detailed_20261008_v2'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
    progress=ROOT/'artifacts/progress.json'
    if progress.exists():
        p=json.loads(progress.read_text())
        result.update(status=p['status'],active=p['active_arm'],completed=len(p['completed_arms']))
        result['completed_names']=[r['name'] for r in p['completed_arms']]
        if p['active_arm']:
            output=ROOT/'YOLOX_outputs'/p['active_arm']
            runtimes=list((output/'track_res').glob('*.txt.runtime.json'))
            result['active_sequences_finished']=len(runtimes)
            result['active_frames_finished']=sum(json.loads(f.read_text())['frames_seen'] for f in runtimes)
            path=ROOT/'logs'/(p['active_arm']+'.log')
            if path.exists():
                entries=[line for line in path.read_text(errors='replace').splitlines()
                         if line.startswith('{"sequence":')]
                if entries:
                    result['last_logged_frame']=json.loads(entries[-1])
    for name in ['stop.json','pipeline.exit','launcher.exit','smoke_launcher.exit']:
        p=ROOT/'artifacts'/name
        if p.exists():
            result[name]=json.loads(p.read_text()) if name.endswith('.json') else p.read_text().strip()
    for name in ['finalization.exit','FINALIZATION_RECEIPT.json']:
        p=ROOT/'review'/name
        if p.exists():
            result[name]=json.loads(p.read_text()) if name.endswith('.json') else p.read_text().strip()
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
