"""Read-only lower-bound inference progress, completion markers and live tmux."""
import datetime,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'manifest.json').read_text())
expected=manifest['sequence_frames']
state={'time_utc':datetime.datetime.utcnow().isoformat()+'Z'}
for phase in ['smoke','full']:
    p=ROOT/'artifacts'/(phase+'_progress.json')
    if p.exists():state[phase]=json.loads(p.read_text())
cells=[]
for a in manifest['arms']:
    folder=ROOT/'YOLOX_outputs'/a['name']
    if not folder.exists():continue
    complete={p.name[:-len('.txt.runtime.json')] for p in (folder/'track_res').glob('*.txt.runtime.json')}
    frames=sum(expected[s] for s in complete)
    latest=None
    log=ROOT/'logs'/(a['name']+'.log')
    if log.exists():
        for line in reversed(log.read_text(errors='replace').splitlines()):
            if line.startswith('{"sequence":'):
                latest=json.loads(line);break
    if latest and latest['sequence'] not in complete:frames+=latest['frame']
    cells.append(dict(name=a['name'],gpu=a['gpu_physical_index'],finished_sequences=len(complete),
                      total_sequences=17,frames_at_least=frames,total_frames=6635,
                      latest_progress=latest,scored=(folder/'receipt.json').exists()))
state['cells']=cells
state['terminal_markers']={p.name:p.read_text().strip() for p in (ROOT/'artifacts').glob('*.exit')}
try:state['tmux']=subprocess.check_output(['tmux','list-panes','-t','leaf_egia_reduced_pure_20261008_v1','-F','#{pane_pid} #{pane_current_command} #{pane_dead}'],text=True).strip()
except subprocess.CalledProcessError:state['tmux']='session absent; inspect terminal markers'
stop=ROOT/'artifacts/grid_stop.json'
if stop.exists():state['stop']=json.loads(stop.read_text())
print(json.dumps(state,ensure_ascii=False))
