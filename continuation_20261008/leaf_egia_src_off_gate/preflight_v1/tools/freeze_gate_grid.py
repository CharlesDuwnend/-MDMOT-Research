"""Freeze source closure after explicit CPU contracts pass."""
import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('/home/chenhc/leaf_egia_detailed_20261008_v2')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

design=json.loads((ROOT/'design.json').read_text())
registration=json.loads((ROOT/'PREREGISTRATION.json').read_text())
cpu=(ROOT/'logs/cpu_contracts_raid.log').read_text()
assert re.search(r'85 passed',cpu) and not re.search(r'\d+ failed',cpu),cpu[-1000:]
assert design['arms']==registration['arms'] and len(design['arms'])==23
parent=json.loads((PARENT/'manifest.json').read_text())
frozen={}
for name,h in parent['frozen_files_sha256'].items():
    p=Path(name);assert sha(p)==h
    if PARENT in p.parents:
        new=ROOT/p.relative_to(PARENT)
        if new.is_file():frozen[str(new)]=sha(new)
    else:frozen[name]=h
for directory in ['belief','tools','tests','host']:
    for p in (ROOT/directory).rglob('*'):
        if p.is_file() and p.suffix in ['.py','.sh']:
            frozen[str(p)]=sha(p)
for p in (ROOT/'models').glob('*.pkl'):frozen[str(p)]=sha(p)
for name in ['AGENTS.md','EXPERIMENT_PLAN.md','PREREGISTRATION.json','design.json',
    'logs/cpu_contracts.log','logs/cpu_contracts_raid.log','artifacts/PREFLIGHT_RESOLUTION_AUDIT.json',
    'review/NEW_RUNTIME_STORAGE_RECEIPT.json']:
    p=ROOT/name;frozen[str(p)]=sha(p)
for p in [PARENT/'manifest.json',PARENT/'YOLOX_outputs/D00_full_reference/receipt.json',
          PARENT/'YOLOX_outputs/D00_full_reference/tracking_metrics.json',
          PARENT/'YOLOX_outputs/D01_EGIA_bypass/receipt.json']:
    frozen[str(p)]=sha(p)
design.update(status='FROZEN_SRC_OFF_EGIA_X_GATE_23_CELL_PROTOCOL',
    frozen_utc=datetime.datetime.utcnow().isoformat()+'Z',frozen_files_sha256=frozen)
with (ROOT/'manifest.json').open('x') as f:json.dump(design,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'frozen_dependencies':len(frozen),'cells':23,'manifest_sha256':sha(ROOT/'manifest.json')}))
