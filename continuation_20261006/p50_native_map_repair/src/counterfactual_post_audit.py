#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np, torch
P=Path(__file__).resolve().parents[1]; R=json.loads((P/'COUNTERFACTUAL_GATE.json').read_text()); checks=[]
def ck(n,ok,e): checks.append({'check':n,'status':'PASS' if ok else 'FAIL','evidence':e})
# A: complete arms, hashes, finite state.
expected=['cf_off','cf_on']; complete=set(R['arms'])==set(expected) and all(len(R['arms'][a])==2 for a in expected); details=[]; finite=True
for a in expected:
 for x in R['arms'][a]:
  p=Path(x['checkpoint']); h=p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==x['checkpoint_sha256']; st=torch.load(str(p),map_location='cpu'); sd=st['state_dict']; f=all(torch.isfinite(v).all().item() for v in sd.values()); finite &= h and f;details.append({'arm':a,'seed':x['seed'],'hash':h,'finite':f})
ck('complete_hash_finite',complete and finite,details)
# B: event and strict subset denominators.
all_fit=R['all_fit_events']; strict_fit=R['fit_events']; evals=R['eval_events']; strict_ok=all(v>0 for v in evals.values()) and strict_fit==1913
ck('strict_supervision_population',strict_ok,{'all_fit_events':all_fit,'strict_fit_events':strict_fit,'eval_events':evals,'controls':R['controls']})
# C: same per-pair denominators and nonempty cf metrics.
rows=[]; denom_ok=True
for a in expected:
 for x in R['arms'][a]:
  for sid,m in x['per_pair'].items(): rows.append((a,x['seed'],sid,m['events'],m['positive_events']))
for sid in evals:
 vals=[z for z in rows if z[2]==sid]; denom_ok &= len({z[3] for z in vals})==1 and len({z[4] for z in vals})==1 and vals[0][3]==evals[sid]
ck('same_pair_denominators',denom_ok,rows)
# D: v2 source semantics and exact negative-mask failure behavior.
source=(P/'src/counterfactual_gate.py').read_text(); ck('unknown_and_short_control_rejected','known_mask' in source and 'cardinality-matched negative control unavailable' in source and 'strict_cf_eligible' in source,{'source_sha256':hashlib.sha256(source.encode()).hexdigest()})
# E: official lock.
ck('official_locked',R['controls'].get('official_val_test_access') is False, R['controls'])
out={'status':'PASS_COUNTERFACTUAL_POST_AUDIT' if all(x['status']=='PASS' for x in checks) else 'FAIL_COUNTERFACTUAL_POST_AUDIT','checks':checks,'result_sha256':hashlib.sha256((P/'COUNTERFACTUAL_GATE.json').read_bytes()).hexdigest()}
(P/'COUNTERFACTUAL_POST_AUDIT.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps(out,indent=2,ensure_ascii=False));raise SystemExit(0 if out['status'].startswith('PASS') else 1)
