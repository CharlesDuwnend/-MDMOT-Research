"""Freeze the four preregistered arms after actual CPU contracts."""
import datetime,json,re
from pathlib import Path
from run_mechanism import sha
ROOT=Path(__file__).resolve().parents[1]
parent=Path('/home/chenhc/leaf_egia_src_off_gate_20261008_v1')
d=json.loads((ROOT/'design.json').read_text())
r=json.loads((ROOT/'PREREGISTRATION.json').read_text())
assert d['arms']==r['arms'] and len(d['arms'])==4
log=(ROOT/'logs/cpu_contracts.log').read_text()
assert re.search(r'\d+ passed',log) and not re.search(r'\d+ (failed|error)',log)
frozen=json.loads((parent/'manifest.json').read_text())['frozen_files_sha256'].copy()
for name,h in frozen.items():assert sha(name)==h
for directory in ['belief','tools','tests','host']:
    for p in (ROOT/directory).rglob('*'):
        if p.is_file() and p.suffix in ['.py','.sh']:frozen[str(p)]=sha(p)
for p in (ROOT/'models').glob('*.pkl'):frozen[str(p)]=sha(p)
for name in ['AGENTS.md','EXPERIMENT_PLAN.md','PREREGISTRATION.json','design.json',
             'logs/cpu_contracts.log','review/STORAGE_RECEIPT.json']:
    frozen[str(ROOT/name)]=sha(ROOT/name)
for p in [parent/'manifest.json',Path(d['full_reference_metrics']),
          Path(d['input_reference_receipt']),Path(d['smoke_input_reference_receipt'])]:
    frozen[str(p)]=sha(p)
d.update(status='FROZEN_FOUR_REDUCED_EGIA_COMPONENT_ARMS_SRC_OFF_GATE_OFF',
         frozen_utc=datetime.datetime.utcnow().isoformat()+'Z',frozen_files_sha256=frozen)
with (ROOT/'manifest.json').open('x') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(cells=4,frozen_dependencies=len(frozen),manifest_sha256=sha(ROOT/'manifest.json'))))
