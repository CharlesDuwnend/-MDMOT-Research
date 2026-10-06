#!/usr/bin/env python3
"""Independent data-contract round: three checks separate from the trainer."""
from __future__ import annotations
import gzip,json,hashlib,re
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; SCI=Path('/home/chenhc/mdmot_sci_id_20260906'); BASE=Path('/home/chenhc/cross_uav_query_mamba_20260830/features/train')
FIT=('23','25','27','28','29','30','32','39','42','44','45','50','51','53','54'); EVAL=('69','70','74','76','78'); IDS=FIT+EVAL
R=json.loads((ROOT/'NATIVE_FACTORIAL.json').read_text()); checks=[]
def ck(name,ok,e): checks.append({'name':name,'status':'PASS' if ok else 'FAIL','evidence':e})
# D1: independent episode line count and trainer pair audit.
manifest_counts={};
for sid in IDS:
 p=SCI/'data/train_episodes_v1'/f'{sid}.jsonl.gz'; count=0
 with gzip.open(p,'rt') as f:
  for _ in f: count+=1
 manifest_counts[sid]=count
reported={s:int(R['pair_audit'][s]['counts']['events']) for s in IDS}
drops={s:len(R['pair_audit'][s]['dropped']) for s in IDS}
ck('D1_episode_count_and_no_drop',manifest_counts==reported and all(v==0 for v in drops.values()),{'manifest':manifest_counts,'reported':reported,'dropped':drops})
# D2: zero-candidate rows are explicitly present, not filtered.
zero_expected={}
for sid in IDS:
 z=0
 with gzip.open(SCI/'data/train_episodes_v1'/f'{sid}.jsonl.gz','rt') as f:
  for line in f:
   if json.loads(line).get('candidate_count')==0:z+=1
 zero_expected[sid]=z
zero_reported={s:int(R['pair_audit'][s]['counts']['zero_candidate']) for s in IDS}
ck('D2_zero_candidate_preserved',zero_expected==zero_reported,{'expected':zero_expected,'reported':zero_reported,'total':sum(zero_expected.values())})
# D3: all 40 native feature shards are structurally complete and inference-safe.
forbidden={'gt_id','global_id','target_train_gt_id','source_train_gt_id','evaluator_global_id','track_id'}; bad=[]; shard_summary={}
for sid in IDS:
 for dev in ('1','2'):
  ip=BASE/f'{sid}-{dev}.jsonl'; npz=BASE/f'{sid}-{dev}.npz'
  rows=0; maxrow=-1; shape=None
  try:
   with ip.open() as f:
    for line in f:
     if not line.strip():continue
     r=json.loads(line); rows+=1; maxrow=max(maxrow,int(r['feature_row']))
     if forbidden.intersection(r): bad.append({'file':str(ip),'forbidden':sorted(forbidden.intersection(r))})
     if list(r.get('source_image_shape',[])) != [1080,1920]: bad.append({'file':str(ip),'shape':r.get('source_image_shape')})
   with np.load(npz,allow_pickle=False) as z: shape=list(z['visual_feature'].shape)
   if shape != [rows,256] or maxrow != rows-1: bad.append({'file':str(ip),'rows':rows,'maxrow':maxrow,'npz_shape':shape})
   shard_summary[f'{sid}-{dev}']={'rows':rows,'shape':shape}
  except Exception as e: bad.append({'file':str(ip),'error':str(e)})
ck('D3_native_shards_inference_safe',not bad,{'bad':bad[:10],'bad_count':len(bad),'shards':len(shard_summary),'total_rows':sum(x['rows'] for x in shard_summary.values())})
payload={'status':'PASS_DATA_CONTRACT_ROUND' if all(x['status']=='PASS' for x in checks) else 'FAIL_DATA_CONTRACT_ROUND','round':'R2','checks':checks,'official_val_test_access':False}
(ROOT/'DATA_CONTRACT_AUDIT.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(payload,indent=2,ensure_ascii=False)); raise SystemExit(0 if payload['status'].startswith('PASS') else 1)
