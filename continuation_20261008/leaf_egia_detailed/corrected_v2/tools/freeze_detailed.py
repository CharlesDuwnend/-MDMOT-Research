"""Seal passed CPU evidence, model differences and complete source lineage."""
import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_mechanism import sha, write


def main():
    parent=Path('/home/chenhc/leaf_internal_mechanism_20261008_v2_metadata')
    registration=json.loads((ROOT/'PREREGISTRATION.json').read_text())
    assert sha(parent/'manifest.json')==registration['parent_manifest_sha256']
    old=json.loads((parent/'manifest.json').read_text())
    for p,h in old['frozen_files_sha256'].items():
        assert sha(p)==h, 'PARENT_FROZEN_SOURCE_CHANGED: '+p
    log=ROOT/'logs/detailed_cpu_contracts.log'
    raw=log.read_text()
    assert re.search(r'59 passed',raw) and not re.search(r'\d+ failed',raw)
    reference=ROOT/'artifacts/CORRECTED_REFERENCE_CPU.json'
    r=json.loads(reference.read_text())
    assert r['tests_run']==9 and r['failures']==r['errors']==0 and len(r['checks'])==9
    assert not r['cuda_initialized'] and not r['mocked_methods']
    prior=ROOT/'artifacts/PRIOR_AND_BUNDLE_RECEIPT.json'
    receipt=json.loads(prior.read_text())
    for p,h in receipt['source_files_sha256'].items():
        assert sha(p)==h
    manifest={k:old[k] for k in ['gpu_uuid','gpu_physical_index','data_root','gt_root',
                                'detector','fixed','sequence_frames','smoke_sequence','smoke_frames',
                                'corrected_reference_sources']}
    full=parent/'YOLOX_outputs/C11_full_reference/tracking_metrics.json'
    manifest.update(status='FROZEN_FULL_SRC_DETAILED_EGIA_NINE_CELL_PROTOCOL',
        full_reference_metrics=str(full),parent_root=str(parent),
        parent_manifest_sha256=sha(parent/'manifest.json'),testdev_parameter_selection=False,
        training='none; neutral priors computed on canonical fit39 only',
        frozen_utc=datetime.datetime.utcnow().isoformat()+'Z',arms=[])
    for cell in receipt['cells']:
        manifest['arms'].append(dict(name=cell['name'],mode=cell['mode'],
            runtime_arm='fc_full',mask=True,penalty=True,fusion=True,selective=True,
            use_egia=cell['mode']!='disabled',corrected_reference_oracle=True,
            bundle=cell['bundle'],bundle_sha256=cell['bundle_sha256'],
            historical_reference_metrics=str(full) if cell['mode']=='full' else None))
    frozen={}
    # Map the entire parent dependency closure to this isolated clone, with
    # originals retained only when the path is an external immutable dependency.
    for p,h in old['frozen_files_sha256'].items():
        original=Path(p)
        if str(original).startswith(str(parent)+'/'):
            new=ROOT/original.relative_to(parent)
            if new.is_file():
                frozen[str(new)]=sha(new)
        else:
            frozen[p]=h
    for directory in ['belief','tools','tests']:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and p.suffix in ['.py','.sh']:
                frozen[str(p)]=sha(p)
    for p in [ROOT/'AGENTS.md',ROOT/'PROTOCOL.md',ROOT/'PREREGISTRATION.json',
              prior,reference,log,parent/'manifest.json',full,full.parent/'receipt.json']:
        frozen[str(p)]=sha(p)
    if registration.get('prior_engineering_attempt'):
        prior_attempt=Path(registration['prior_engineering_attempt'])
        assert sha(prior_attempt/'manifest.json')==registration['prior_manifest_sha256']
        for p in [prior_attempt/'manifest.json',prior_attempt/'artifacts/stop.json']:
            frozen[str(p)]=sha(p)
    for p in list((ROOT/'models').glob('*.pkl')):
        frozen[str(p)]=sha(p)
    for p,h in receipt['source_files_sha256'].items():
        frozen[p]=h
    gate=ROOT/'artifacts/DETAILED_CPU_GATE.json'
    write(gate,dict(status='PASS_DETAILED_EGIA_CPU_CONTRACTS',tests_passed=59,
        exit_code=0,formal_MOT_results=False,parent_source_files_verified=len(old['frozen_files_sha256']),
        source_fields_preserved=True,full_SRC_all_cells=True,
        evidence_sha256={str(p):sha(p) for p in [log,prior,reference]}))
    frozen[str(gate)]=sha(gate)
    manifest['frozen_files_sha256']=frozen
    assert not (ROOT/'manifest.json').exists()
    write(ROOT/'manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],arms=len(manifest['arms']),
                         files_frozen=len(frozen),manifest_sha256=sha(ROOT/'manifest.json'))))


if __name__=='__main__':
    main()
