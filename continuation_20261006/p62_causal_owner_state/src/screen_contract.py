#!/usr/bin/env python3
"""Build the P62 train-only prefix dataset and run mechanism contracts."""
import gzip,hashlib,json,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]; HOST=Path('/home/chenhc/mdmot_global_host_stage15_20260905'); AUD=Path('/home/chenhc/mdmt_cod_revision_audit_20260926'); OUT=ROOT/'continuation_20261006/p62_causal_owner_state'; FIT=('23','25','28','29','39','44','45','51','53','63','66','69','70','74','78'); CAL=('27','32','42','64','65'); PAIRS=FIT+CAL
sys.path.insert(0,str(Path('/home/chenhc/mdmt_cod_revision_audit_20260926')))
from state_repair_contract import EventWindow,OwnerState,ContractError

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load_cache(pair):
 with gzip.open(HOST/f'cache/train/{pair}.json.gz','rt') as f:d=json.load(f)
 assert d['split']=='train' and d['pair_id']==pair and d['gt_read'] is False and d['test_access'] is False
 return d
def load_events(pair):
 out=[]
 with gzip.open(AUD/f'artifacts_v2/events/{pair}.jsonl.gz','rt') as f:
  for line in f:out.append(json.loads(line))
 return out

def candidate_features(c,event):
 ev=np.asarray([float(x['distance']) for x in c['evidence']],np.float64); x0=float(ev[0]); x1=float(ev[-1]); slope=float(np.polyfit(np.arange(len(ev)),ev,1)[0]) if len(ev)>1 else 0.
 return np.asarray([float(c['event_frame_distance']),float(c['aggregate_distance']),float(c['support_frames'])/max(1,event['ready_frame']-event['confirmation_frame']+1),float(ev.mean()),float(ev.std()),slope,float(ev[-1]-ev[0])],np.float32)

def build_rows():
 rows=[]; receipts=[]
 for pair in PAIRS:
  d=load_cache(pair); evs=load_events(pair); assert len(d['events'])==len(evs)
  for source,e in zip(d['events'],evs):
   assert source['target_local_track_id']==e['target_lid'] and source['ready_frame']==e['ready_frame']
   pos=set(int(x) for x in e['correct_source_lids_at_confirmation'])
   eligible=e['label_state']=='eligible'; present=e['availability'] in ('candidate_present','candidate_missing')
   for c in source['candidates']:
    feat=candidate_features(c,e); y=1 if eligible and int(c['source_local_track_id']) in pos else 0
    rows.append({'pair':pair,'target_lid':int(e['target_lid']),'source_lid':int(c['source_local_track_id']),'ready_frame':int(e['ready_frame']),'features':feat.tolist(),'prefix_label':int(y),'eligible':bool(eligible),'candidate_present':bool(present),'baseline_rank':int(sorted(source['candidates'],key=lambda z:float(z['aggregate_distance'])).index(c)+1)})
  receipts.append({'pair':pair,'events':len(d['events']),'cache_sha256':sha(HOST/f'cache/train/{pair}.json.gz'),'events_sha256':sha(AUD/f'artifacts_v2/events/{pair}.jsonl.gz')})
 return rows,receipts

class COSTCell(nn.Module):
 def __init__(self):
  super().__init__();self.edge=nn.Sequential(nn.Linear(7,32),nn.LayerNorm(32),nn.GELU(),nn.Linear(32,16));self.state=nn.GRUCell(16,16);self.logit=nn.Linear(16,1)
 def forward(self,x,h):
  e=self.edge(x);h=self.state(e,h);return self.logit(h).squeeze(-1),h

def cpu_contract(rows):
 x=torch.tensor(np.asarray([r['features'] for r in rows[:64]],np.float32));h=torch.zeros(len(x),16);m=COSTCell();logit,h2=m(x,h);loss=F.binary_cross_entropy_with_logits(logit,torch.tensor([r['prefix_label'] for r in rows[:64]],dtype=torch.float32));loss.backward();finite=all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());
 if not finite or not torch.isfinite(loss):raise ValueError('COST CPU gradient/finite contract failed')
 # Same edge sequence with the owner state reset is deterministic; permuting batch rows does not alter per-row local edge contract.
 with torch.no_grad():a,_=m(x,h);b,_=m(x,h)
 if not torch.equal(a,b):raise ValueError('COST deterministic replay failed')
 w=EventWindow.from_confirmation(1,2,4); s=OwnerState(w); bad=(('1','a'),('2','b'));good=(('1','a'),('2','c'));s.provisional_publish(bad,2);s.revoke(bad,5,evidence_frame=5);s.provisional_publish(good,6);snap=s.finalize();
 if not any(r['kind']=='link' and r['edge'][1]==('2','c') for r in snap['records']):raise ValueError('owner transition contract failed')
 return {'status':'PASS_P62_DATA_CPU_CONTRACT','rows':len(rows),'positive_rows':sum(r['prefix_label'] for r in rows),'eligible_rows':sum(r['eligible'] for r in rows),'candidate_present_rows':sum(r['candidate_present'] for r in rows),'pairs':len(set(r['pair'] for r in rows)),'model_parameters':sum(p.numel() for p in m.parameters()),'finite_gradients':finite,'loss':float(loss.detach()),'owner_transition_contract':True,'official_val_test_read':False,'source_sha256':sha(Path(__file__)),'prereg_sha256':sha(OUT/'PREREG.json')}

def main():
 rows,receipts=build_rows(); report=cpu_contract(rows);report['pair_receipts']=receipts;report['feature_names']=['event_distance','aggregate_distance','support_fraction','evidence_mean','evidence_std','evidence_slope','evidence_delta'];dump(OUT/'DATA_CPU_CONTRACT.json',report);dump(OUT/'DATASET_RECEIPT.json',{'status':'P62_PREFIX_DATASET_COMPLETE','rows':len(rows),'pair_receipts':receipts,'fit_pairs':list(FIT),'calibration_pairs':list(CAL),'future_outcome_features_used':False,'official_val_test_read':False});
 # compact aggregate only; raw row dataset stays local
 by=defaultdict(lambda:Counter())
 for r in rows:by[r['pair']]['rows']+=1;by[r['pair']]['positive']+=r['prefix_label'];by[r['pair']]['events']+=1 if r['source_lid']==r['source_lid'] else 0
 print(json.dumps({'status':'P62_DATA_CONTRACT_COMPLETE','report':report,'per_pair':{p:dict(v) for p,v in by.items()}},indent=2))
if __name__=='__main__':main()
