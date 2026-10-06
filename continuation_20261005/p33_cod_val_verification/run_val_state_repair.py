#!/usr/bin/env python3
import gzip,json,hashlib,importlib.util,sys,datetime as dt,statistics
from pathlib import Path
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('p32val',Path('/home/chenhc/claude_try_MDMOT/continuation_20261005/p32_cod_state_repair/run_real_state_repair.py'))
m=importlib.util.module_from_spec(spec);sys.modules['p32val']=m;spec.loader.exec_module(m)
m.ROOT=HERE
m.CACHE=Path('/home/chenhc/mdmot_global_host_stage15_20260905/cache/val')
m.REPLAY=Path('/home/chenhc/mdmot_global_host_stage15_20260905/association_ledger/B0')
m.STREAM=Path('/home/chenhc/mdmot_strong_host_stage1_20260905/streams/val')
m.PAIRS=['22','36','46','49','72']
m.XML=Path('/raid/datasets/chc_data/MDMT/new_xml')
W=4

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

out=HERE/'runs/W4';out.mkdir(parents=True,exist_ok=True);rows=[];inputs={}
for pair in m.PAIRS:
 payload=m.load_json_gz(m.CACHE/f'{pair}.json.gz')
 replay=[json.loads(x) for x in gzip.open(m.REPLAY/f'{pair}.solver.jsonl.gz','rt')]
 inputs[f'cache/{pair}.json.gz']=sha(m.CACHE/f'{pair}.json.gz');inputs[f'replay/{pair}.solver.jsonl.gz']=sha(m.REPLAY/f'{pair}.solver.jsonl.gz')
 ps=m.extract_proposals(pair,payload,replay);d=m.simulate_delay(ps,W);r=m.simulate_repair(ps,W);b=m.b0_edges(ps)
 streams={u:m.load_stream(pair,u) for u in ('1','2')};maps=m.local_gid_maps(pair,streams)
 s=m.summarize(pair,ps,d,r,b,streams,maps,W);rows.append(s)
 (out/f'{pair}.json').write_text(json.dumps({'summary':s,'proposals':[m.asdict(p) for p in ps],'delay_log':d['log'],'repair_log':r['log']},indent=2,default=list)+'\n')
 print(json.dumps({k:s[k] for k in ['pair','proposal_count','delay_final_edges','repair_final_edges','repair_revokes','repair_changes_vs_delay','delay_diagnostics','repair_diagnostics']}),flush=True)
agg={'status':'OFFICIAL_VAL_PROTOCOL_VERIFICATION_COMPLETE','split':'official_val','W':W,'pairs':m.PAIRS,'pair_count':len(rows),'proposal_count':sum(x['proposal_count'] for x in rows),'repair_revoke_count':sum(x['repair_revokes'] for x in rows),'repair_revision_count':sum(x['repair_revisions'] for x in rows),'repair_changed_pairs':sum(x['repair_changes_vs_delay']>0 for x in rows),'pair_macro':{mode:{k:{'mean':statistics.mean(x[f'{mode}_diagnostics'][k] for x in rows),'median':statistics.median(x[f'{mode}_diagnostics'][k] for x in rows),'sum':sum(x[f'{mode}_diagnostics'][k] for x in rows)} for k in ['correct_edges','false_edges','unknown_edges','future_wrong_coobserved_frames','future_coobserved_frames']} for mode in ['b0','delay','repair']},'summaries':rows,'input_sha256':inputs,'runtime_decision_uses_gt':False,'gt_usage':'post-replay val diagnostic labels only','official_test_accessed':False,'training':False,'threshold_search':False,'prereg_sha256':sha(HERE/'PREREG.json')}
(out/'AGGREGATE.json').write_text(json.dumps(agg,indent=2)+'\n');print(json.dumps({'aggregate':str(out/'AGGREGATE.json'),'status':agg['status']}))
