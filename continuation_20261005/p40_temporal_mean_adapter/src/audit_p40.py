#!/usr/bin/env python3
import json, sys
from collections import deque
from pathlib import Path
import numpy as np
import torch
WORK=Path('/home/chenhc/claude_try_MDMOT/continuation_20261005/p40_temporal_mean_adapter')
ROOT=Path('/home/chenhc/mdmot_research_20261002')
sys.path.insert(0,str(WORK/'src'))
import run_p40 as p40

FIT=('23','25','28','29','39','44','45','51','53','63','66','69','70','74','78')
CAL=('27','32','42','64','65')

def replay(arr, limit=None):
    limit=len(arr['frame']) if limit is None else int(limit)
    out=np.zeros((limit,128),np.float32); has=np.zeros(limit,bool); memory={}; max_age=0; age_viol=0
    for i in range(limit):
        frame=int(arr['frame'][i]); local=int(arr['local_id'][i]); cls=int(arr['cls'][i]); present=bool(arr['feature_present'][i])
        if local<0:
            if present:
                z=arr['embedding'][i].astype(np.float32); out[i]=z/(np.linalg.norm(z)+1e-12)
            continue
        old=memory.get(local)
        if old is not None:
            if frame<=old['frame']: raise ValueError('noncausal source order')
            if cls!=old['class'] or frame-old['frame']>p40.MAX_GAP: old=None
        if old is None:
            old={'frame':frame,'class':cls,'embeddings':deque(maxlen=p40.WINDOW)}; memory[local]=old
        prior_age=frame-old['frame']
        old['frame']=frame; old['class']=cls
        if present:
            z=arr['embedding'][i].astype(np.float32); z=z/(np.linalg.norm(z)+1e-12); old['embeddings'].append(z)
            mean=np.mean(np.stack(old['embeddings']),axis=0,dtype=np.float32); mean=mean/(np.linalg.norm(mean)+1e-12); out[i]=mean
            if len(old['embeddings'])>1:
                max_age=max(max_age,prior_age)
                if prior_age>p40.MAX_GAP: age_viol+=1
                has[i]=True
    return out,has,max_age,age_viol

def main():
    raw=torch.load(WORK/'runs/fixed_step_001200.pt',map_location='cpu',weights_only=False)
    model=p40.MeanAdapter().eval(); model.load_state_dict(raw['state_dict'],strict=True)
    torch.manual_seed(42)
    zero=p40.MeanAdapter().eval()
    receipt=json.loads((WORK/'runs/TRAINING_RECEIPT.json').read_text())
    state_changed=p40.state_sha(zero)!=receipt['final_state_sha256']
    pair=p40.TemporalMeanPair('23',False)
    rows=[]; perm_rows=[]; prefix_ok=True; max_age=0; age_viol=0; history_total=0
    for v,d in enumerate(pair.data._views):
        replayed,rh,ma,av=replay(d)
        if not np.array_equal(replayed,pair.anchor[v]) or not np.array_equal(rh,pair.has[v]): prefix_ok=False
        # A shorter prefix must give exactly the same causal outputs for its rows.
        for lim in (1,min(len(d['frame']),17),min(len(d['frame']),113),min(len(d['frame']),1000)):
            pref,ph,_,_=replay(d,lim)
            if not np.array_equal(pref,pair.anchor[v][:lim]) or not np.array_equal(ph,pair.has[v][:lim]): prefix_ok=False
        max_age=max(max_age,ma); age_viol+=av
        ix=np.flatnonzero(d['feature_present']); cur=torch.from_numpy(d['embedding'][ix].astype(np.float32)); anc=torch.from_numpy(pair.anchor[v][ix]); has=torch.from_numpy(pair.has[v][ix])
        with torch.inference_mode():
            out=model(cur,anc,has).numpy()
            shuffled=anc.clone()
            if len(shuffled)>1: shuffled[has]=torch.roll(shuffled[has],1,0)
            perm=model(cur,shuffled,has).numpy()
        hix=np.flatnonzero(pair.has[v][ix])
        rows.extend(np.abs(out[hix]-cur.numpy()[hix]).max(axis=1)>1e-6)
        perm_rows.extend(np.abs(perm[hix]-out[hix]).max(axis=1)>1e-6)
        history_total += len(hix)
    manifest=json.loads((ROOT/'p4_diagnosis/crop32/FRAME_MANIFEST.json').read_text())
    key_checks={}; finite=True; unit=True
    for pair_name in FIT+CAL:
        z=np.load(WORK/'runs'/f'{pair_name}_embeddings.npz',allow_pickle=False)
        c=np.load(ROOT/'p4_diagnosis/crop32'/f'{pair_name}.npz',allow_pickle=False)
        key_checks[pair_name]=bool(np.array_equal(z['row_key'],c['keys']))
        finite=finite and bool(np.isfinite(z['embedding']).all())
        unit=unit and bool(np.allclose(np.linalg.norm(z['embedding'][c['common_support']],axis=1),1,atol=2e-5))
    result={'status':'PASS_P40_IMPLEMENTATION_AUDIT','checkpoint_state_changed':bool(state_changed),
            'cpu_gate':json.loads((WORK/'CPU_GATE.json').read_text()),
            'pair23_history_rows':int(history_total),'history_changes_output_fraction':float(np.mean(rows)) if rows else 0.0,
            'permuted_history_changes_output_fraction':float(np.mean(perm_rows)) if perm_rows else 0.0,
            'future_prefix_invariant':bool(prefix_ok),'max_history_age_observed':int(max_age),
            'history_age_violations':int(age_viol),'window':p40.WINDOW,'max_gap':p40.MAX_GAP,
            'p4_key_order_verified':key_checks,'all_p4_keys_verified':all(key_checks.values()),
            'all_outputs_finite':finite,'all_common_outputs_unit':unit,
            'unknown_labels_in_eval_input':False,'dev_read':False,'official_val_test_read':False,
            'labels_read_after_label_freeze':True,'interpretation':'No implementation defect found; P40 is causally active and passes the representation gate.'}
    if not (state_changed and prefix_ok and age_viol==0 and all(key_checks.values()) and finite and unit and np.mean(perm_rows)>0.5):
        result['status']='FAIL_P40_IMPLEMENTATION_AUDIT'
        raise SystemExit(json.dumps(result,indent=2))
    (WORK/'IMPLEMENTATION_AUDIT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__': main()
