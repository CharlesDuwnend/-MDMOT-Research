#!/usr/bin/env python3
"""P60 Setwise Entropic Identity Transport: CPU contract, train, evaluate."""
import argparse, hashlib, json, random, time, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
OLD=Path('/home/chenhc/mdmot_research_20261002')
P23=ROOT/'continuation_20261004/p23_visual/data'
OUT=ROOT/'continuation_20261006/p60_setwise_transport'
INIT=ROOT/'continuation_20261004/p23_visual/runs/head_only/fixed_step_001200.pt'
UUID='GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a'
FIT=('23','25','28','29','39','44','45','51','53','63','66','69','70','74','78')
CAL=('27','32','42','64','65')
sys.path[:0]=[str(OLD/'p2/src'),str(OLD/'p4_diagnosis'),str(ROOT/'continuation_20261006/p55_context_residual/src')]
from data import sha256
import compare_crop32 as p4
from train_head_init import BaseHead

EPS=0.10; SINK_ITERS=50

def digest(p): return sha256(p)
def dump(p,o): Path(p).write_text(json.dumps(o,indent=2,sort_keys=True,allow_nan=False)+'\n')

def load_base():
    x=np.load(P23/'inputs.npz',allow_pickle=False); fpn=np.load(P23/'raw_fpn.npy',mmap_mode='r'); dino=np.load(P23/'frozen_dino.npy',mmap_mode='r'); labels=np.load(P23/'offline_pid.npy',mmap_mode='r'); groups=json.loads((P23/'GROUPS.json').read_text())
    ck=torch.load(INIT,map_location='cpu',weights_only=False); h=BaseHead(); h.load_state_dict(ck['head_state'],strict=True); h.eval()
    with torch.inference_mode(): z=F.normalize(h.raw(torch.from_numpy(np.asarray(fpn,np.float32).copy()),torch.from_numpy(np.asarray(dino,np.float32).copy())),dim=-1).numpy()
    if len(z)!=len(labels) or len(groups)!=1200: raise ValueError('P23 shape/schedule contract')
    return x,z,labels,groups

def log_sinkhorn(scores,eps=EPS,iters=SINK_ITERS):
    """Differentiable balanced transport with uniform row/column marginals."""
    n,m=scores.shape
    a=scores/eps
    u=torch.zeros(n,device=scores.device,dtype=scores.dtype); v=torch.zeros(m,device=scores.device,dtype=scores.dtype)
    log_r=-float(np.log(n)); log_c=-float(np.log(m))
    for _ in range(iters):
        u=log_r-torch.logsumexp(a+v[None,:],dim=1)
        v=log_c-torch.logsumexp(a+u[:,None],dim=0)
    return torch.exp(a+u[:,None]+v[None,:])

def np_sinkhorn(scores,eps=EPS,iters=SINK_ITERS):
    a=np.asarray(scores,np.float64)/eps; a-=a.max(); n,m=a.shape; u=np.zeros(n);v=np.zeros(m)
    lr=-np.log(n);lc=-np.log(m)
    for _ in range(iters):
        u=lr-np.logaddexp.reduce(a+v[None,:],axis=1);v=lc-np.logaddexp.reduce(a+u[:,None],axis=0)
    p=np.exp(a+u[:,None]+v[None,:])
    if not np.isfinite(p).all(): raise ValueError('nonfinite numpy transport')
    return p

class SEIT(nn.Module):
    def __init__(self):
        super().__init__(); self.residual=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128)); nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)
    def forward(self,z): return F.normalize(z+self.residual(z),dim=-1)

def group_loss(ea,ya,eb,yb):
    sa=ea@eb.T; sb=sa.T; pa=log_sinkhorn(sa); pb=log_sinkhorn(sb)
    vals=[]; counts=0
    for p,y,other in ((pa,ya,yb),(pb,yb,ya)):
        known_q=y>=0; known_c=other>=0
        for i in torch.where(known_q)[0].tolist():
            pos=torch.where(known_c & (other==y[i]))[0]
            if len(pos):
                # row mass is 1/n; convert to conditional row probability.
                prob=p[i,pos].sum()*p.shape[0]
                vals.append(-torch.log(prob.clamp_min(1e-8))); counts+=1
    if not vals: return (ea.sum()+eb.sum())*0.,0
    return torch.stack(vals).mean(),counts

def cpu_contract():
    x,z,labels,groups=load_base(); m=SEIT().eval(); g=next(g for g in groups if len(g['sides'][0]) and len(g['sides'][1])); ia=np.asarray(g['sides'][0]);ib=np.asarray(g['sides'][1]); za=torch.from_numpy(np.asarray(z[ia],np.float32));zb=torch.from_numpy(np.asarray(z[ib],np.float32));ya=torch.from_numpy(labels[ia].astype(np.int64));yb=torch.from_numpy(labels[ib].astype(np.int64))
    with torch.no_grad(): ea=m(za);eb=m(zb); zero=float((ea-za).abs().max().item())
    ea,eb=m(za),m(zb); loss,n=group_loss(ea,ya,eb,yb)
    with torch.no_grad():
        ref_loss=float(group_loss(ea.detach(),ya,eb.detach(),yb)[0]); perm=torch.randperm(len(ib)); perm_loss=float(group_loss(ea.detach(),ya,eb.detach()[perm],yb[perm])[0])
    # Permuting rows and labels together must preserve the setwise loss.
    loss.backward(); finite=all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());
    if not finite: raise ValueError('nonfinite/missing gradient')
    if abs(ref_loss-perm_loss)>1e-7: raise ValueError('permutation contract failed')
    out={'status':'PASS_P60_CPU_CONTRACT','schedule_groups':len(groups),'fit_rows':len(z),'parameters':sum(p.numel() for p in m.parameters()),'zero_init_max_error':zero,'finite_gradients':bool(finite),'gradient_norm':float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)),'permutation_invariance':True,'loss':float(loss.detach()),'queries':n,'official_val_test_read':False,'source_sha256':digest(Path(__file__)),'prereg_sha256':digest(OUT/'PREREG.json')};dump(OUT/'SUPPORT_CPU.json',out);print(json.dumps(out,indent=2))

def train():
    cpu_contract();x,z,labels,groups=load_base()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0) or torch.cuda.get_device_properties(0).total_memory<30*1024**3: raise RuntimeError('GPU verification failed')
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42);m=SEIT().cuda().train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);hist=[];started=time.monotonic()
    for step,g in enumerate(groups,1):
        ia=np.asarray(g['sides'][0]);ib=np.asarray(g['sides'][1]);za=torch.from_numpy(np.asarray(z[ia],np.float32)).cuda();zb=torch.from_numpy(np.asarray(z[ib],np.float32)).cuda();ya=torch.from_numpy(labels[ia].astype(np.int64)).cuda();yb=torch.from_numpy(labels[ib].astype(np.int64)).cuda();ea=m(za);eb=m(zb);loss,n=group_loss(ea,ya,eb,yb)
        if not torch.isfinite(loss): raise ValueError('nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        if not torch.isfinite(gn): raise ValueError('nonfinite gradient')
        opt.step()
        if step==1 or step%100==0: hist.append({'step':step,'loss':float(loss.detach()),'queries':n,'gradient_norm':float(gn)})
    outdir=OUT/'runs';outdir.mkdir(parents=True,exist_ok=False); ck=outdir/'fixed_step_001200.pt';torch.save({'state_dict':{k:v.detach().cpu() for k,v in m.state_dict().items()},'step':len(groups),'init_sha256':digest(INIT),'prereg_sha256':digest(OUT/'PREREG.json')},ck);rec={'status':'PASS_P60_TRAINING','updates':len(groups),'checkpoint_sha256':digest(ck),'gpu_uuid':UUID,'gpu_name':torch.cuda.get_device_name(0),'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'history':hist,'elapsed_seconds':time.monotonic()-started,'official_val_test_read':False};dump(outdir/'TRAINING_RECEIPT.json',rec);(outdir/'training.exit').write_text('0\n');print(json.dumps(rec,indent=2))

def evaluate():
    x,z,labels,groups=load_base();raw=torch.load(OUT/'runs/fixed_step_001200.pt',map_location='cpu',weights_only=False);m=SEIT().eval();m.load_state_dict(raw['state_dict'],strict=True)
    inputs={r['sequence']:r for r in json.loads((OLD/'p2/cache/MANIFEST.json').read_text())['sequences']}; per={}
    for pair in CAL:
        rec={r['pair']:r for r in json.loads((OLD/'p4_diagnosis/crop32/FRAME_MANIFEST.json').read_text())['pairs']}[pair];row,features,refs,cov=p4.align_pair(rec,OLD/'p4_diagnosis/crop32',inputs)
        with torch.inference_mode(): zz=F.normalize(BaseHead().raw(torch.from_numpy(features['fpn256']),torch.from_numpy(features['dino384'])),dim=-1)
        # load the same frozen head weights used in load_base
        base=BaseHead();base.load_state_dict(torch.load(INIT,map_location='cpu',weights_only=False)['head_state'],strict=True);base.eval()
        with torch.inference_mode(): zz=F.normalize(base.raw(torch.from_numpy(features['fpn256']),torch.from_numpy(features['dino384'])),dim=-1); ee=m(zz).numpy(); zz=zz.numpy()
        # raw cosine baseline from exact P4 evaluator; transport candidate score from identical rows.
        p4.METHODS=('target_only',);arr=p4.evaluate_arrays(row,{'target_only':zz}); b=p4.summaries(arr)['all']['methods']['target_only']['all_candidates']['known_positive_r1_tie_averaged']; common=row['row_common_support']; idx=np.flatnonzero(common); pid=row['row_pid_offline']; vals=ee.astype(np.float64);vals/=np.linalg.norm(vals,axis=1,keepdims=True);groups2={}
        for i in idx:groups2.setdefault((int(row['row_frame'][i]),int(row['row_class'][i]),int(row['row_view'][i])),[]).append(int(i))
        credits=[];elig=[]
        for key,left in groups2.items():
            if key[2]!=0:continue
            right=groups2.get((key[0],key[1],1),[])
            if not right:continue
            left=np.asarray(left,np.int32);right=np.asarray(right,np.int32);tr=np_sinkhorn(vals[left]@vals[right].T)
            for q,rows,s in [(left[k],right,tr[k]) for k in range(len(left))]+[(right[k],left,tr[:,k]) for k in range(len(right))]:
                ok=pid[q]>=0 and np.any(pid[rows]==pid[q]);elig.append(ok)
                if ok:
                    top=s.max();ties=np.flatnonzero(s==top);credits.append(np.mean(pid[rows[ties]]==pid[q]))
        cand=float(np.mean(credits));per[pair]={'target_only':float(b),'seit':cand,'delta':cand-float(b),'pair_wins':cand>float(b),'eligible':int(sum(elig))}
        print(json.dumps({'pair':pair,**per[pair]}),flush=True)
    base=float(np.mean([v['target_only'] for v in per.values()]));cand=float(np.mean([v['seit'] for v in per.values()]));delta=cand-base;wins=sum(v['pair_wins'] for v in per.values());result={'status':'P60_CALIBRATION_GATE_COMPLETE','per_pair':per,'macro':{'target_only':base,'seit':cand,'delta':delta},'pair_wins':wins,'gate':{'minimum_macro_delta':.005,'minimum_pair_wins':3,'minimum_calibration_macro':.48412456211781995,'passed':bool(delta>=.005 and wins>=3 and cand>=.48412456211781995)},'official_val_test_read':False,'training_completed':True,'source_sha256':digest(Path(__file__)),'checkpoint_sha256':digest(OUT/'runs/fixed_step_001200.pt')};dump(OUT/'SIGNAL_TRAINED.json',result);print(json.dumps(result,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['cpu','train','evaluate'],required=True);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);{'cpu':cpu_contract,'train':train,'evaluate':evaluate}[a.mode]()
if __name__=='__main__':main()
