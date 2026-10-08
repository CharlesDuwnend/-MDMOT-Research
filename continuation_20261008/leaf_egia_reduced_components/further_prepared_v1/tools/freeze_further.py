"""Freeze after CPU checks, including all reused complete evidence."""
import datetime, json, re
from pathlib import Path
from run_mechanism import sha
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('/home/chenhc/leaf_egia_reduced_pure_20261008_v1')
d=json.loads((ROOT/'design.json').read_text())
r=json.loads((ROOT/'PREREGISTRATION.json').read_text())
assert d['arms']==r['arms'] and len(d['arms'])==3
log=(ROOT/'logs/cpu_contracts.log').read_text()
assert re.search(r'\d+ passed',log) and not re.search(r'\d+ (failed|error)',log)
frozen=json.loads((PARENT/'manifest.json').read_text())['frozen_files_sha256'].copy()
for p,h in frozen.items():assert sha(p)==h
for directory in ['belief','tools','tests','host']:
    for p in (ROOT/directory).rglob('*'):
        if p.is_file() and p.suffix in ('.py','.sh'):frozen[str(p)]=sha(p)
for p in (ROOT/'models').glob('*.pkl'):frozen[str(p)]=sha(p)
for name in ['AGENTS.md','EXPERIMENT_PLAN.md','PREREGISTRATION.json','design.json',
             'logs/cpu_contracts.log','logs/cpu_round1.log','logs/cpu_round2.log',
             'artifacts/CPU_FAILURE_RESOLUTION.json','review/STORAGE_RECEIPT.json']:
    frozen[str(ROOT/name)]=sha(ROOT/name)
for p in [PARENT/'manifest.json',PARENT/'review/results/FINAL_AUDIT.json']:
    frozen[str(p)]=sha(p)
for arm in ['R00_full','R03_no_source_cues','R02_no_coverage_head']:
    folder=PARENT/'YOLOX_outputs'/arm
    for p in [folder/'tracking_metrics.json',folder/'receipt.json']:
        frozen[str(p)]=sha(p)
    for p in (folder/'track_res').glob('*.txt'):frozen[str(p)]=sha(p)
for p in [Path(d['smoke_input_reference_receipt'])]:frozen[str(p)]=sha(p)
d.update(status='FROZEN_THREE_FURTHER_PURE_EGIA_ARMS',
         frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         frozen_files_sha256=frozen)
with (ROOT/'manifest.json').open('x') as f:
    json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(cells=3,frozen_dependencies=len(frozen),manifest_sha256=sha(ROOT/'manifest.json'))))
