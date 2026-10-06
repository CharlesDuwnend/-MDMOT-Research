#!/usr/bin/env python3
import gzip,json,importlib.util,sys,statistics
from pathlib import Path
p=Path(__file__).with_name('run_real_state_repair.py');spec=importlib.util.spec_from_file_location('p32s',p);m=importlib.util.module_from_spec(spec);sys.modules['p32s']=m;spec.loader.exec_module(m)
rows=[]
for W in (0,4,8):
 total={'W':W,'pairs':0,'proposals':0,'revoke':0,'revision':0,'changed':0,'delay_edges':0,'repair_edges':0}
 for pair in m.PAIRS:
  payload=m.load_json_gz(m.CACHE/f'{pair}.json.gz'); replay=[json.loads(x) for x in gzip.open(m.REPLAY/f'{pair}.solver.jsonl.gz','rt')]
  ps=m.extract_proposals(pair,payload,replay); d=m.simulate_delay(ps,W);r=m.simulate_repair(ps,W)
  total['pairs']+=1;total['proposals']+=len(ps);total['revoke']+=sum(x['op']=='revoke' for x in r['log']);total['revision']+=sum(x['op']=='revision_publish' for x in r['log']);total['changed']+=len(set(d['active'])^set(r['active']));total['delay_edges']+=len(d['active']);total['repair_edges']+=len(r['active'])
 rows.append(total)
Path(__file__).with_name('SENSITIVITY_STRUCTURAL.json').write_text(json.dumps({'status':'TRAIN_ONLY_STRUCTURAL_SENSITIVITY','rows':rows,'uses_gt':False},indent=2)+'\n')
print(json.dumps(rows,indent=2))
