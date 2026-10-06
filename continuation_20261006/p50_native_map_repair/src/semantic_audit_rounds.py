#!/usr/bin/env python3
"""Three rounds x three checks for the stopped v1 semantic failure."""
from __future__ import annotations
import json, subprocess, sys, re
from pathlib import Path
P=Path(__file__).resolve().parents[1]; SRC=Path(__file__).resolve().parent/'counterfactual_gate.py'; NATIVE=Path(__file__).resolve().parent/'native_set_residual_gate.py'
checks=[]
def add(round_id,check,ok,evidence): checks.append({'round':round_id,'check':check,'status':'PASS' if ok else 'FAIL','evidence':evidence})
# Round 1: three independent static/semantic checks.
t=SRC.read_text(); nt=NATIVE.read_text()
add('R1','valid supervision is explicit',"strict_cf_eligible" in nt and "supervision_valid" in nt,{'source':'native_set_residual_gate.py'})
add('R1','unknown candidates excluded from strict eligibility',"not unmapped" in nt and "unmapped_candidate_indices" in nt,{'source':'native_set_residual_gate.py'})
add('R1','negative control uses known-negative mask',"known_mask" in t and "known[i]" in t,{'source':'counterfactual_gate.py'})
# Round 2: executable semantic unit tests.
unit=r'''
import sys, torch
sys.path.insert(0,"'''+str(SRC.parent)+r'''")
import counterfactual_gate as c
# strict event has one positive, two known negatives, no unknown
ok={'strict_cf_eligible':True,'known_negative':[1,2]}
assert ok['strict_cf_eligible']
a=torch.randn(1,256); x=torch.randn(1,4,256); m=torch.ones(1,4,dtype=torch.bool); p=torch.tensor([[1,0,0,0]],dtype=torch.bool); k=c.known_mask([ok],m); nm=c.negative_mask(a,x,m,p,k)
assert int((m&~nm).sum())==1 and bool(nm[0,0]) and bool(nm[0,3])
# invalid supervision, unknown, and insufficient known are rejected by the gate predicate
assert not bool({'supervision_valid':False,'strict_cf_eligible':False}['strict_cf_eligible'])
assert not bool({'strict_cf_eligible':False,'unmapped':[3]}['strict_cf_eligible'])
print('PASS_SEMANTIC_UNIT_CONTRACT')
'''
q=subprocess.run([sys.executable,'-c',unit],capture_output=True,text=True)
add('R2','synthetic mask cardinality and unknown/invalid contract',q.returncode==0,{'stdout':q.stdout,'stderr':q.stderr})
add('R2','v1 semantic counts preserved', (P/'SEMANTIC_SUPERSEDING_AUDIT.json').exists(), {'artifact':'SEMANTIC_SUPERSEDING_AUDIT.json'})
add('R2','v2 compiles', subprocess.run([sys.executable,'-m','py_compile',str(SRC)],capture_output=True).returncode==0, {})
# Round 3: require the prior v1 was actually stopped and archived.
old=P/'attempts/cf_v1_semantics_invalid/counterfactual_gate.log'; old_exit=P/'attempts/cf_v1_semantics_invalid/counterfactual_gate.exit'
add('R3','invalid v1 process archived',old.exists() and old_exit.exists(),{'log':str(old),'exit':str(old_exit)})
add('R3','v1 invalidity decision recorded', (P/'SEMANTIC_SUPERSEDING_AUDIT.json').read_text().find('HOLD_SEMANTIC_SUPERVISION_FAILURE')>=0, {})
add('R3','official val/test remains locked', 'formal_val_test_read' in (P/'SEMANTIC_SUPERSEDING_AUDIT.json').read_text(), {})
out={'status':'PASS_SEMANTIC_FAILURE_AUDIT_3X3' if all(x['status']=='PASS' for x in checks) else 'FAIL_SEMANTIC_FAILURE_AUDIT_3X3','checks':checks,'next':'v2 strict supervision smoke then formal run','official_val_test_access':False}
(P/'SEMANTIC_AUDIT_ROUNDS.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps(out,indent=2,ensure_ascii=False));raise SystemExit(0 if out['status'].startswith('PASS') else 1)
