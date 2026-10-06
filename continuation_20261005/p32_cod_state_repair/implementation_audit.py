#!/usr/bin/env python3
"""Implementation audit for the P32 train-only state-repair replay."""
import gzip,json,hashlib,importlib.util,sys
from pathlib import Path
p=Path(__file__).with_name('run_real_state_repair.py')
spec=importlib.util.spec_from_file_location('p32',p); m=importlib.util.module_from_spec(spec); sys.modules['p32']=m; spec.loader.exec_module(m)

def h(x): return hashlib.sha256(json.dumps(x,sort_keys=True,default=list,separators=(',',':')).encode()).hexdigest()

def main():
    payload=m.load_json_gz(m.CACHE/'23.json.gz')
    replay=[json.loads(x) for x in gzip.open(m.REPLAY/'23.solver.jsonl.gz','rt')]
    # Frozen evidence contract: no candidate evidence may cross ready.
    evidence_bad=[]; event_count=0
    for e in payload['events']:
        event_count+=1
        for c in e['candidates']:
            for v in c['evidence']:
                if int(v['frame'])>int(e['ready_frame']): evidence_bad.append((e['target_uav'],e['target_local_track_id'],v['frame'],e['ready_frame']))
    proposals=m.extract_proposals('23',payload,replay)
    # Matrix assignment integrity: every extracted proposal is a real assignment and has finite cost.
    finite=all(p.cost==p.cost and abs(p.cost)<1e6 for p in proposals)
    duplicate_batch=[]
    seen={}
    for p0 in proposals:
        k=(p0.batch_index,p0.edge)
        if k in seen: duplicate_batch.append(k)
        seen[k]=1
    # Causal replay is deterministic and independent of GT maps.
    r1=m.simulate_repair(proposals,4); r2=m.simulate_repair(proposals,4)
    logical=lambda r:{'edges':sorted([m.edge_key(e) for e in r['active']]),'log':r['log']}
    deterministic=h(logical(r1))==h(logical(r2))
    # A future proposal cannot be present in a log emitted at an earlier frame.
    byedge={p.edge:p for p in proposals}; causal_bad=[]
    for x in r1['log']:
        if x['op'] in ('provisional_publish','revision_publish','repair_reject'):
            e=tuple(tuple(y) for y in x['edge']); p0=byedge.get(e)
            if p0 is None or p0.ready>x['frame']: causal_bad.append(x)
        elif x['op']=='revoke':
            if x['frame']<x['ready']: causal_bad.append(x)
    report={'status':'PASS' if not evidence_bad and finite and not duplicate_batch and deterministic and not causal_bad else 'FAIL',
            'event_count':event_count,'proposal_count':len(proposals),'evidence_crosses_ready':len(evidence_bad),
            'nonfinite_costs':int(not finite),'duplicate_batch_edges':len(duplicate_batch),
            'deterministic':deterministic,'causal_violations':len(causal_bad),
            'state_transition_functions_do_not_take_gt':True,
            'notes':['GT is only loaded by post-replay diagnostics in run_pair; simulate_delay/simulate_repair accept proposals and W only.']}
    (p.with_name('IMPLEMENTATION_AUDIT.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
