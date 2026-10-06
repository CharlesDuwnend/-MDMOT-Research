#!/usr/bin/env python3
import json,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[3]; OLD=Path('/home/chenhc/mdmot_research_20261002')
sys.path[:0]=[str(OLD/'p2/src'),str(OLD/'p4_diagnosis'),str(ROOT/'continuation_20261004/p23_visual'),str(ROOT/'continuation_20261006/p55_context_residual/src')]
from data import sha256
import compare_crop32 as p4
from train_head_init import BaseHead
CAL=('27','32','42','64','65')

def sinkhorn(scores,eps=.10,iters=50):
    # scores shape n x m; uniform balanced marginals. Work in log domain.
    if scores.size==0:return scores
    a=scores.astype(np.float64)/eps
    a-=np.max(a)
    log_r=-np.log(scores.shape[0]); log_c=-np.log(scores.shape[1])
    u=np.zeros(scores.shape[0]); v=np.zeros(scores.shape[1])
    for _ in range(iters):
        u=log_r-np.logaddexp.reduce(a+v[None,:],axis=1)
        v=log_c-np.logaddexp.reduce(a+u[:,None],axis=0)
    p=np.exp(a+u[:,None]+v[None,:])
    if not np.isfinite(p).all(): raise ValueError('nonfinite transport')
    return p

def score_pair(row,z,tau=.10):
    common=row['row_common_support']; idx=np.flatnonzero(common); pid=row['row_pid_offline']; vals=np.asarray(z,np.float64); vals/=np.linalg.norm(vals,axis=1,keepdims=True)
    qrows=[]; cand=[]; scores=[]
    groups={}
    for i in idx:
        key=(int(row['row_frame'][i]),int(row['row_class'][i]),int(row['row_view'][i]));groups.setdefault(key,[]).append(int(i))
    for key,left in groups.items():
        if key[2]!=0:continue
        right=groups.get((key[0],key[1],1),[])
        if not right:continue
        left=np.asarray(left,np.int32);right=np.asarray(right,np.int32); sim=vals[left]@vals[right].T; tr=sinkhorn(sim,eps=tau)
        for k,i in enumerate(left):qrows.append(i);cand.append(right.copy());scores.append(tr[k].copy())
        for k,j in enumerate(right):qrows.append(j);cand.append(left.copy());scores.append(tr[:,k].copy())
    qrows=np.asarray(qrows,np.int32); result={}; per=[]
    for q,rows,s in zip(qrows,cand,scores):
        top=np.max(s); ties=np.flatnonzero(s==top); positive=int(np.sum(pid[rows[ties]]==pid[q])); unknown=int(np.sum(pid[rows[ties]]<0)); eligible=bool(pid[q]>=0 and np.any(pid[rows]==pid[q])); result.setdefault('eligible',[]).append(eligible); result.setdefault('credit',[]).append(positive/len(ties)); result.setdefault('unknown',[]).append(unknown/len(ties)); result.setdefault('qpair',[]).append(str(row['row_key'][q]).split('-')[0])
    return result

def main():
    p2=json.loads((OLD/'p2/cache/MANIFEST.json').read_text()); inputs={r['sequence']:r for r in p2['sequences']}; ck=ROOT/'continuation_20261004/p23_visual/runs/head_only/fixed_step_001200.pt'; head=BaseHead(); raw=torch.load(ck,map_location='cpu',weights_only=False); head.load_state_dict(raw['head_state'],strict=True); head.eval(); out={}; base={}
    for pair in CAL:
        rec={r['pair']:r for r in json.loads((OLD/'p4_diagnosis/crop32/FRAME_MANIFEST.json').read_text())['pairs']}[pair]; row,features,refs,cov=p4.align_pair(rec,OLD/'p4_diagnosis/crop32',inputs)
        with torch.inference_mode():z=torch.nn.functional.normalize(head.raw(torch.from_numpy(features['fpn256']),torch.from_numpy(features['dino384'])),dim=-1).numpy()
        # baseline exact cosine through p4 implementation
        p4.METHODS=('target_only',); arr=p4.evaluate_arrays(row,{'target_only':z}); rep=p4.summaries(arr)['all']['methods']['target_only']['all_candidates']; b=float(rep['known_positive_r1_tie_averaged']); base[pair]=b
        s=score_pair(row,z); elig=np.asarray(s['eligible'],bool); c=float(np.mean(np.asarray(s['credit'])[elig])); unknown=float(np.mean(np.asarray(s['unknown'])[elig])); out[pair]={'target_only':b,'transport':c,'delta':c-b,'eligible':int(elig.sum()),'unknown_top_mass':unknown,'queries':len(elig)}
        print(json.dumps({'pair':pair,**out[pair]}),flush=True)
    macro_base=float(np.mean([v['target_only'] for v in out.values()])); macro=float(np.mean([v['transport'] for v in out.values()])); wins=sum(v['transport']>v['target_only'] for v in out.values()); result={'status':'P60_FROZEN_SIGNAL_COMPLETE','operator':'balanced_sinkhorn_eps_0.10_iters_50','per_pair':out,'macro':{'target_only':macro_base,'transport':macro,'delta':macro-macro_base},'pair_wins':wins,'gate':{'minimum_macro_delta':.005,'minimum_pair_wins':3,'minimum_calibration_macro':.48412456211781995,'passed':bool(macro-macro_base>=.005 and wins>=3 and macro>=.48412456211781995)},'official_val_test_read':False,'training':False,'source_sha256':sha256(Path(__file__)),'prereg_sha256':sha256(ROOT/'continuation_20261006/p60_setwise_transport/PREREG.json')};(ROOT/'continuation_20261006/p60_setwise_transport/SIGNAL.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
