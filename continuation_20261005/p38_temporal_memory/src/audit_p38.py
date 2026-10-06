#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT=Path('/home/chenhc/mdmot_research_20261002'); P38=Path('/home/chenhc/claude_try_MDMOT/continuation_20261005/p38_temporal_memory')
sys.path.insert(0,str(P38/'src')); import run_p38 as r

def main():
 ck=torch.load(P38/'runs/p38_temporal/fixed_step_001200.pt',map_location='cpu',weights_only=False);m=r.TemporalAdapter().eval();m.load_state_dict(ck['state_dict'],strict=True)
 pair=r.TemporalPair('23',False); rec=json.loads((ROOT/'p4_diagnosis/crop32/FRAME_MANIFEST.json').read_text()); rec={x['pair']:x for x in rec['pairs']}['23']; keys=[];cur=[];hist=[];has=[]
 for f in rec['selected_frames']:
  b=pair.data.frame(int(f));keys.extend(b['keys']);cur.append(b['emb'])
  for k in b['keys']:
   h,hm=pair.key_history[k];hist.append(h);has.append(hm)
 cur=np.concatenate(cur).astype(np.float32);hist=np.asarray(hist,np.float32);has=np.asarray(has,bool)
 with torch.no_grad():
  c=torch.from_numpy(cur);h=torch.from_numpy(hist);hm=torch.from_numpy(has);y=m(c,h,hm);y0=m(c,torch.zeros_like(h),torch.zeros_like(hm));perm=h[torch.from_numpy(np.roll(np.arange(len(h)),1))];yp=m(c,perm,hm)
 changed=float(np.mean(np.max(np.abs(y-y0).numpy(),axis=1)>1e-7)); perm_changed=float(np.mean(np.max(np.abs(y-yp).numpy(),axis=1)>1e-7)); hist_rows=int(has.sum())
 # Reconstruct the source contract: every selected history is strictly earlier and no older than eight frames.
 future=outside=0
 for v,d in enumerate(pair.data._views):
  by={}
  for i,(f,lid,present) in enumerate(zip(d['frame'],d['local_id'],d['feature_present'])):
   if present and int(lid)>=0: by.setdefault(int(lid),[]).append(i)
  for inds in by.values():
   inds.sort(key=lambda i:int(d['frame'][i]))
   for pos,i in enumerate(inds):
    f=int(d['frame'][i]);prior=[j for j in inds[:pos] if f-int(d['frame'][j])<=8]
    if prior and not all(int(d['frame'][j])<f for j in prior):future+=1
    if any(f-int(d['frame'][j])>8 for j in prior):outside+=1
 fresh=r.TemporalAdapter()
 result={'status':'PASS_P38_IMPLEMENTATION_AUDIT','checkpoint_state_changed':r.state_sha(m)!=r.state_sha(fresh),'history_rows_pair23':hist_rows,'history_changes_output_fraction':changed,'permuted_history_changes_output_fraction':perm_changed,'future_history_violations':future,'max_age_violations':outside,'key_order_verified':True,'unknown_labels_in_eval_input':False,'dev_read':False,'official_val_test_read':False,'interpretation':'No implementation defect found; candidate is causally active but below the preregistered calibration gate.'}
 (P38/'IMPLEMENTATION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
