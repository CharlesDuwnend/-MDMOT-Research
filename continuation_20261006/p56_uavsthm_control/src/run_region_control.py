#!/usr/bin/env python3
"""Run the UAVST-HM local-region appearance control on P23 calibration rows."""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3]; OLD=Path('/home/chenhc/mdmot_research_20261002'); OUT=ROOT/'continuation_20261006/p56_uavsthm_control'
sys.path[:0]=[str(OLD/'p2/src'),str(OLD/'p4_diagnosis'),str(ROOT/'continuation_20261006/p55_context_residual/src')]
from data import split_assignments
import compare_crop32 as p4
from train_head_init import BaseHead, INIT

def ranks(feat,row):
 out=[]
 for i in range(len(feat)):
  if row['row_pid_offline'][i]<0: continue
  same=(row['row_view']!=row['row_view'][i])&(row['row_frame']==row['row_frame'][i])&(row['row_class']==row['row_class'][i])
  c=np.flatnonzero(same & (row['row_pid_offline']>=0)); pos=np.flatnonzero(row['row_pid_offline'][c]==row['row_pid_offline'][i])
  if len(c) and len(pos): out.append(1+int(np.sum(feat[i] @ feat[c].T > float(feat[i] @ feat[c[pos[0]]]))))
 return out

def main():
 frame=json.loads((OLD/'p4_diagnosis/crop32/FRAME_MANIFEST.json').read_text()); split=split_assignments(); p2=json.loads((OLD/'p2/cache/MANIFEST.json').read_text()); inputs={r['sequence']:r for r in p2['sequences']}; head=BaseHead(); head.load_state_dict(torch.load(INIT,map_location='cpu',weights_only=False)['head_state'],strict=True);head.eval(); per={}
 for rec in frame['pairs']:
  if rec['role']!='calibration':continue
  pair=rec['pair']; row,features,refs,cnt=p4.align_pair(rec,OLD/'p4_diagnosis/crop32',inputs)
  with np.load(OLD/'p4_diagnosis/crop32'/f'{pair}.npz',allow_pickle=False) as a: bbox=a['bbox']
  with torch.inference_mode(): base=F.normalize(head.raw(torch.from_numpy(features['fpn256']),torch.from_numpy(features['dino384'])),dim=-1).numpy()
  region=base.copy()
  for i in range(len(base)):
   same=(row['row_view']==row['row_view'][i])&(row['row_frame']==row['row_frame'][i])
   idx=np.flatnonzero(same); idx=idx[idx!=i]
   if len(idx):
    c=(bbox[idx,:2]+bbox[idx,2:])/2; q=(bbox[i,:2]+bbox[i,2:])/2; nn=idx[np.argsort(np.linalg.norm(c-q,axis=1))[:2]]; region[i]=F.normalize(torch.from_numpy(base[i]+base[nn].mean(0)),dim=0).numpy()
  br=ranks(base,row); rr=ranks(region,row); per[pair]={'target_only':{'queries':len(br),'r1':float(np.mean(np.asarray(br)==1))},'two_neighbour_region':{'queries':len(rr),'r1':float(np.mean(np.asarray(rr)==1))}}
 macro={m:float(np.mean([per[p][m]['r1'] for p in per])) for m in ('target_only','two_neighbour_region')};wins=sum(per[p]['two_neighbour_region']['r1']>per[p]['target_only']['r1'] for p in per);out={'status':'P56_REGION_CONTROL_COMPLETE','pairs':sorted(per),'per_pair':per,'macro_r1':macro,'delta':macro['two_neighbour_region']-macro['target_only'],'pair_wins':wins,'official_val_test_read':False,'formal_mot_claim':False,'source':'https://www.mdpi.com/2504-446X/8/12/704'}; (OUT/'CONTROL_RESULT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':out['status'],'macro_r1':macro,'delta':out['delta'],'pair_wins':wins}))

if __name__=='__main__':main()
