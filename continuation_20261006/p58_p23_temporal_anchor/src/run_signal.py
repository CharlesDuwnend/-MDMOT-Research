#!/usr/bin/env python3
"""P23 head-only strict-causal temporal anchor signal on P23 calibration pairs."""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3]
OLD=Path('/home/chenhc/mdmot_research_20261002')
OUT=ROOT/'continuation_20261006/p58_p23_temporal_anchor'
sys.path[:0]=[str(OLD/'p2/src'),str(OLD/'p4_diagnosis'),str(ROOT/'continuation_20261006/p55_context_residual/src')]
from data import PairData, split_assignments
import compare_crop32 as p4
from train_head_init import BaseHead, INIT

CAL=('27','32','42','64','65')

def unit(x):
    x=np.asarray(x,np.float64); n=np.linalg.norm(x,axis=1,keepdims=True)
    if not np.isfinite(x).all() or np.any(n<=0): raise ValueError('nonfinite/zero embedding')
    return x/n

def causal_anchor(base,row):
    out=np.zeros_like(base,np.float32); hist_counts=[]
    common=np.asarray(row['row_common_support'],bool)
    for i in np.flatnonzero(common):
        view=int(row['row_view'][i]); local=int(row['local_id_metadata_only'][i]); frame=int(row['row_frame'][i])
        eligible=np.flatnonzero(common & (row['row_view']==view) &
                                 (row['local_id_metadata_only']==local) &
                                 (row['row_frame']<frame) &
                                 (row['row_frame']>=frame-8))
        eligible=eligible[np.argsort(row['row_frame'][eligible],kind='stable')][-4:]
        vals=np.concatenate((base[i:i+1],base[eligible]),axis=0)
        out[i]=unit(vals).mean(axis=0)
        out[i]=unit(out[i:i+1])[0].astype(np.float32)
        hist_counts.append(len(eligible))
    out[common]=unit(out[common]).astype(np.float32)
    return out, {'queries':int(common.sum()),'with_history':int(sum(x>0 for x in hist_counts)),'mean_history':float(np.mean(hist_counts)),'max_history':int(max(hist_counts) if hist_counts else 0)}

def main():
    records={r['pair']:r for r in json.loads((OLD/'p4_diagnosis/crop32/FRAME_MANIFEST.json').read_text())['pairs']}
    inputs={r['sequence']:r for r in json.loads((OLD/'p2/cache/MANIFEST.json').read_text())['sequences']}
    head=BaseHead(); ck=torch.load(INIT,map_location='cpu',weights_only=False); head.load_state_dict(ck['head_state'],strict=True); head.eval()
    per={}; history={}
    for pair in CAL:
        rec=records[pair]
        if rec['role']!='calibration': raise ValueError('role mismatch')
        row,features,refs,cov=p4.align_pair(rec,OLD/'p4_diagnosis/crop32',inputs)
        pdata=PairData(pair,with_labels=False)
        local_by_key={}
        for vi,d in enumerate(pdata._views):
            for f,det,lid in zip(d['frame'],d['detector_index'],d['local_id']):
                local_by_key[f'{pair}-{vi+1}:{int(f)}:{int(det)}']=int(lid)
        row['local_id_metadata_only']=np.asarray([local_by_key[str(k)] for k in row['row_key']],np.int64)
        with torch.inference_mode():
            base=F.normalize(head.raw(torch.from_numpy(features['fpn256']),torch.from_numpy(features['dino384'])),dim=-1).numpy()
        anchor,h=causal_anchor(base,row)
        # Non-common rows are ignored by the frozen P4 evaluator; zero placeholders preserve row shape.
        p4.METHODS=('target_only','temporal_anchor')
        arrays=p4.evaluate_arrays(row,{'target_only':base,'temporal_anchor':anchor})
        summary=p4.summaries(arrays)['all']['methods']
        per[pair]={'target_only':summary['target_only']['all_candidates']['known_positive_r1_tie_averaged'],
                   'temporal_anchor':summary['temporal_anchor']['all_candidates']['known_positive_r1_tie_averaged'],
                   'queries':summary['target_only']['all_candidates']['eligible_queries']}
        history[pair]=h
    target=float(np.mean([per[p]['target_only'] for p in CAL])); anchor=float(np.mean([per[p]['temporal_anchor'] for p in CAL]))
    delta=anchor-target; wins=sum(per[p]['temporal_anchor']>per[p]['target_only'] for p in CAL)
    out={'status':'P58_TEMPORAL_ANCHOR_SIGNAL_COMPLETE','pairs':list(CAL),'per_pair':per,'history':history,
         'macro_r1':{'target_only':target,'temporal_anchor':anchor},'delta':delta,'pair_wins':wins,
         'operator':'L2-normalized mean of current plus up to four strict-past P23 embeddings',
         'official_val_test_read':False,'training_authorized':False}
    (OUT/'SIGNAL.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'status':out['status'],'macro_r1':out['macro_r1'],'delta':delta,'pair_wins':wins}))

if __name__=='__main__':main()
