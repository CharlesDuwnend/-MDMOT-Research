#!/usr/bin/env python3
"""Bind final diagnostic outputs and verified pre-existing inputs."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(name, value):
    (HERE/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


summary=json.loads((HERE/'SUMMARY.json').read_text())
verify=json.loads((HERE/'VERIFY.json').read_text())
inputs=json.loads((HERE/'INPUTS.json').read_text())
assert summary['status']=='PASS_FIT_OFFLINE_FLOW_CENTER_DIAGNOSTIC'
assert verify['status']=='PASS_INDEPENDENT_SAVED_ROW_VERIFICATION'
assert verify['source_sha256']==sha(HERE/'verify_saved.py')
assert (HERE/'REPORT.md').is_file() and (HERE/'STRATA.csv').is_file()
for path,digest in inputs.items():
    assert sha(path)==digest, 'Changed input: '+path
dump('EXECUTION.json',dict(
    producer=dict(command='/home/chenhc/.conda/envs/remdet_paper/bin/python -B audit_motion.py > run.log 2>&1',
                  cwd=str(HERE),tool_session_id=21140,observed_terminal_exit_code=0,
                  elapsed_seconds=summary['elapsed_seconds'],completed_groups=summary['groups'],comparisons=summary['comparisons'],
                  evidence='Prior tool session completed with exit 0; final run.log JSON agrees with SUMMARY.json.'),
    verifier=dict(command='/home/chenhc/.conda/envs/remdet_paper/bin/python -B verify_saved.py > verify.log 2>&1',
                  cwd=str(HERE),tool_session_id=28027,observed_terminal_exit_code=0,
                  elapsed_seconds=verify['elapsed_seconds'],evidence='Current tool session completed with exit 0; VERIFY.json PASS.'),
    python_version=platform.python_version(),GPU_used=False,images_read=False,model_inference=False,non_FIT_GT_read=False,
    output_scope=str(HERE)))
outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(HERE.iterdir())
         if p.is_file() and p.name not in {'RECEIPT.json','SHA256SUMS'}}
receipt=dict(status='PASS_P28_FIT_OFFLINE_MOTION_LOOKBACK_COMPLETE',sealed_utc=datetime.now(timezone.utc).isoformat(),
             scope='Fixed FIT240 cached current-to-past RAFT motion diagnostic only',
             output_root=str(HERE),outputs=outputs,input_hashes=inputs,
             input_count=len(inputs),producer_exit_code=0,verifier_exit_code=0,
             groups=summary['groups'],comparisons=summary['comparisons'],lags=[1,4,8],
             checks=dict(synthetic_contract='PASS',source_direction='PASS',saved_rows='PASS',all_bound_inputs_unchanged=True),
             conclusion='Useful FIT center-motion signal versus zero; no gross direction/scale/halfpixel error observed.',
             limitation='Not FPN fusion validation, failure causality, a temporal STOP, or detection/tracking performance.')
dump('RECEIPT.json',receipt)
files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='SHA256SUMS']
(HERE/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files))
for name,info in outputs.items():
    assert sha(HERE/name)==info['sha256']
print(json.dumps(dict(status=receipt['status'],receipt_sha256=sha(HERE/'RECEIPT.json'),
                      SHA256SUMS_sha256=sha(HERE/'SHA256SUMS'),sealed_files=len(files))))
