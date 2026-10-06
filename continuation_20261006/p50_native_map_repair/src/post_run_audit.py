#!/usr/bin/env python3
"""Post-run implementation audit; does not select a method or access official val/test."""
from __future__ import annotations
import hashlib,json,math,re
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parents[1]; SRC=Path(__file__).resolve().parent/'native_set_residual_gate.py'
import sys; sys.path.insert(0,str(SRC.parent)); import native_set_residual_gate as n
R=json.loads((P/'NATIVE_FACTORIAL.json').read_text())
checks=[]
def ck(name,ok,evidence): checks.append({'name':name,'status':'PASS' if ok else 'FAIL','evidence':evidence})
# Completeness and equal denominators.
expected=['rel_on_teacher_off','rel_on_teacher_on','pooled_teacher_off','pooled_teacher_on']
ck('all_arms_and_seeds',set(R.get('arms',{}))==set(expected) and all(len(R['arms'][a])==2 for a in expected),list(R.get('arms',{})))
by={}
for arm in expected:
 for item in R['arms'].get(arm,[]):
  for sid,m in item['per_pair'].items(): by.setdefault(sid,[]).append((arm,item['seed'],m['events'],m['matched_events']))
for sid,rows in by.items():
 ck('equal_event_denominator_'+sid,len({x[2] for x in rows})==1 and len({x[3] for x in rows})==1,rows)
# Checkpoint hashes and finite state.
finite=True; loaded=[]
for arm in expected:
 for item in R['arms'].get(arm,[]):
  p=Path(item['checkpoint']); good=p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==item['checkpoint_sha256']
  state=torch.load(str(p),map_location='cpu') if p.exists() else {}
  sd=state.get('state_dict',state); f=all(torch.isfinite(v).all().item() for v in sd.values() if torch.is_tensor(v)); finite &= good and f; loaded.append({'arm':arm,'seed':item['seed'],'hash_ok':good,'finite':f,'parameters':len(sd)})
ck('checkpoint_hash_and_finite',finite,loaded)
# Step-zero identity: zero residual/correction must exactly retain native cosine.
torch.manual_seed(3); model=n.NativeSetResidual(); a=torch.randn(4,n.D); c=torch.randn(4,7,n.D); m=torch.ones(4,7,dtype=torch.bool)
with torch.no_grad(): logits,_,_=model(a,c,m); base=(torch.nn.functional.normalize(a,dim=-1)[:,None]*torch.nn.functional.normalize(c+1e-8,dim=-1)).sum(-1)
err=float((logits-base).abs().max()); ck('step_zero_native_cosine',err<1e-6,{'max_abs_error':err})
text=SRC.read_text(); model_region=text[text.find('class NativeSetResidual'):text.find('def assignment_loss')]
ck('teacher_absent_from_forward',('teacher' not in model_region) and ('teacher_proj' not in model_region),{'model_region_has_teacher':False})
ck('official_val_test_locked',bool(R.get('controls',{}).get('official_val_test_access') is False),R.get('controls',{}))
payload={'status':'PASS_POST_RUN_IMPLEMENTATION_AUDIT' if all(x['status']=='PASS' for x in checks) else 'FAIL_POST_RUN_IMPLEMENTATION_AUDIT','checks':checks,'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'result_sha256':hashlib.sha256((P/'NATIVE_FACTORIAL.json').read_bytes()).hexdigest()}
(P/'POST_RUN_IMPLEMENTATION_AUDIT.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(payload,indent=2,ensure_ascii=False)); raise SystemExit(0 if payload['status'].startswith('PASS') else 1)
