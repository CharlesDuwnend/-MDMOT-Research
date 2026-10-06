#!/usr/bin/env python3
"""Train-free signal audit for cross-view temporal differential co-movement."""
from __future__ import annotations
import gzip,json,math,hashlib,importlib.util,collections
from pathlib import Path
import numpy as np
STAGE=Path('/home/chenhc/mdmot_global_host_stage15_20260905'); EP=Path('/home/chenhc/mdmot_sci_id_20260906/data/train_episodes_v1'); OUT=Path(__file__).parent
PAIRS=['23','25','27','28','29','30','32','39']
spec=importlib.util.spec_from_file_location('frozen',STAGE/'scripts/stage15_global_host.py'); frozen=importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)

def cosine(a,b):
 na=np.linalg.norm(a); nb=np.linalg.norm(b)
 return float(np.dot(a,b)/(na*nb+1e-8))

def load_det_features(pair,uav):
 stream,sp=frozen.load_stage1_stream('train',pair,uav); feat,_,_=frozen.load_features('train',pair,uav)
 by_frame,_,_,_,audit=frozen.derive_track_rows(stream,feat)
 out=collections.defaultdict(dict)
 for frame,rows in by_frame.items():
  for row in rows.values():
   if row['feature'] is not None: out[frame][int(row['detector_index'])]=np.asarray(row['feature'],dtype=np.float32)
 return out,audit

def candidate_signal(event,candidate,maps):
 tu=int(event['target_uav']); su=int(event['source_uav']); pairs=[]
 for ev in candidate.get('evidence',[]):
  f=int(ev['frame']); t=maps[tu].get(f,{}).get(int(ev['target_detector_index'])); s=maps[su].get(f,{}).get(int(ev['source_detector_index']))
  if t is not None and s is not None: pairs.append((f,t,s))
 pairs.sort(key=lambda x:x[0])
 static=[cosine(t,s) for _,t,s in pairs]
 dc=[]; dd=[]
 for (_,t0,s0),(_,t1,s1) in zip(pairs,pairs[1:]):
  dt=t1-t0; ds=s1-s0; nt=np.linalg.norm(dt); ns=np.linalg.norm(ds)
  if nt>1e-6 and ns>1e-6:
   dc.append(cosine(dt,ds)); dd.append(float(np.linalg.norm(dt/(nt+1e-8)-ds/(ns+1e-8))))
 return {'support_pairs':len(pairs),'static_cos':float(np.mean(static)) if static else None,'delta_cos':float(np.mean(dc)) if dc else None,'delta_discrepancy':float(np.mean(dd)) if dd else None,'valid_delta_pairs':len(dc)}

def main():
 pair_rows={}; all_rows=[]; map_audits={}
 for pair in PAIRS:
  maps={}
  for u in (1,2): maps[u],map_audits[f'{pair}-{u}']=load_det_features(pair,u)
  cache=json.loads(next(gzip.open(STAGE/f'cache/train/{pair}.json.gz','rt')))
  episodes=[json.loads(line) for line in gzip.open(EP/f'{pair}.jsonl.gz','rt')]
  if len(cache['events'])!=len(episodes): raise AssertionError(f'event count mismatch {pair}')
  rows=[]; event_count=0
  for event,episode in zip(cache['events'],episodes):
   if event.get('candidate_set_sha256')!=episode.get('candidate_set_sha256'): raise AssertionError(f'candidate hash mismatch {pair}')
   pos=set(episode.get('same_id_candidate_indices',[])); neg=set(episode.get('known_negative_indices',[]))
   candidate_metrics=[]
   for i,c in enumerate(event.get('candidates',[])):
    sig=candidate_signal(event,c,maps)
    label='positive' if i in pos else ('negative' if i in neg else 'unknown')
    if label in ('positive','negative') and sig['delta_cos'] is not None:
     x=dict(sig,label=label,pair=pair,event=event_count,candidate_index=i); rows.append(x); all_rows.append(x)
    candidate_metrics.append(sig)
   event_count+=1
  pair_rows[pair]=rows
 # aggregate candidate-level signal and event-level rank diagnostic
 metrics={}
 for pair,rows in pair_rows.items():
  metrics[pair]={'rows':len(rows)}
  for field in ('static_cos','delta_cos','delta_discrepancy'):
   pos=[r[field] for r in rows if r['label']=='positive' and r[field] is not None]; neg=[r[field] for r in rows if r['label']=='negative' and r[field] is not None]
   metrics[pair][field]={'positive_n':len(pos),'negative_n':len(neg),'positive_mean':float(np.mean(pos)) if pos else None,'negative_mean':float(np.mean(neg)) if neg else None,'delta_pos_minus_neg':float(np.mean(pos)-np.mean(neg)) if pos and neg else None}
 # Macro over pairs, with direction where larger is expected.
 macro={}
 for field in ('static_cos','delta_cos','delta_discrepancy'):
  vals=[metrics[p][field]['delta_pos_minus_neg'] for p in PAIRS if metrics[p][field]['delta_pos_minus_neg'] is not None]
  macro[field]={'pairs':len(vals),'mean_delta':float(np.mean(vals)) if vals else None,'pair_positive':sum(v>0 for v in vals),'pair_nonnegative':sum(v>=0 for v in vals)}
 payload={'status':'PASS_CVTDCF_TRAIN_FREE_SIGNAL_AUDIT' if macro['delta_cos']['pair_positive']>=5 and macro['delta_cos']['mean_delta']>0 else 'HOLD_CVTDCF_WEAK_TRAIN_FREE_SIGNAL','candidate':'Cross-View Temporal Differential Co-Movement Field','pairs':PAIRS,'metrics_by_pair':metrics,'macro':macro,'rows':len(all_rows),'feature_alignment_audit':map_audits,'protocol':'train-only frozen detector/query features; labels used only for diagnostic partition; no parameter updates; no val/test','official_val_test_access':False,'source_sha256':{'stage15_script':hashlib.sha256((STAGE/'scripts/stage15_global_host.py').read_bytes()).hexdigest(),'stage15_manifest':hashlib.sha256((STAGE/'manifest.json').read_bytes()).hexdigest()}}
 (OUT/'TEMPORAL_EVIDENCE_SIGNAL_AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
