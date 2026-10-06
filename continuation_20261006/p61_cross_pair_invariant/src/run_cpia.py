#!/usr/bin/env python3
"""P61 Cross-Pair Invariant Identity Adapter: CPU, train, and calibration gate."""
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
OLD=Path('/home/chenhc/mdmot_research_20261002')
P23=ROOT/'continuation_20261004/p23_visual/data'
OUT=ROOT/'continuation_20261006/p61_cross_pair_invariant'
INIT=ROOT/'continuation_20261004/p23_visual/runs/head_only/fixed_step_001200.pt'
UUID='GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a'
FIT=('23','25','28','29','39','44','45','51','53','63','66','69','70','74','78')
CAL=('27','32','42','64','65')
TEMP=0.10; LAMBDA=0.05
sys.path[:0]=[str(OLD/'p2/src'),str(OLD/'p4_diagnosis'),str(ROOT/'continuation_20261006/p55_context_residual/src')]
from data import sha256
import compare_crop32 as p4
from train_head_init import BaseHead

def sha(p): return sha256(p)
def dump(p,o): Path(p).write_text(json.dumps(o,indent=2,sort_keys=True,allow_nan=False)+'\n')

def load_data():
    x=np.load(P23/'inputs.npz',allow_pickle=False); fpn=np.load(P23/'raw_fpn.npy',mmap_mode='r'); dino=np.load(P23/'frozen_dino.npy',mmap_mode='r'); labels=np.load(P23/'offline_pid.npy',mmap_mode='r'); groups=json.loads((P23/'GROUPS.json').read_text())
    ck=torch.load(INIT,map_location='cpu',weights_only=False);h=BaseHead();h.load_state_dict(ck['head_state'],strict=True);h.eval()
    with torch.inference_mode():z=F.normalize(h.raw(torch.from_numpy(np.asarray(fpn,np.float32).copy()),torch.from_numpy(np.asarray(dino,np.float32).copy())),dim=-1).numpy()
    if len(x['keys'])!=len(z) or len(z)!=len(labels) or len(groups)!=1200:raise ValueError('P23 shape/schedule mismatch')
    return x,z,labels,groups

class CPIA(nn.Module):
    def __init__(self):
        super().__init__();self.residual=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,256),nn.GELU(),nn.Linear(256,128));nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def forward(self,z):return F.normalize(z+self.residual(z),dim=-1)

def symmetric_ce(ea,ya,eb,yb):
    vals=[]; n=0
    for q,qy,c,cy in ((ea,ya,eb,yb),(eb,yb,ea,ya)):
        known_c=cy>=0
        for i in torch.where(qy>=0)[0].tolist():
            pos=torch.where(known_c & (cy==qy[i]))[0]
            if len(pos)==0:continue
            logits=(q[i:i+1]@c[known_c].T).flatten()/TEMP
            target=(cy[known_c]==qy[i]).nonzero(as_tuple=False)[0,0]
            vals.append(F.cross_entropy(logits[None],target[None]));n+=1
    if not vals:return (ea.sum()+eb.sum())*0.,0
    return torch.stack(vals).mean(),n

def group_tensors(z,labels,g,device='cpu'):
    ia=np.asarray(g['sides'][0]);ib=np.asarray(g['sides'][1]);return (torch.from_numpy(np.asarray(z[ia],np.float32)).to(device),torch.from_numpy(np.asarray(labels[ia],np.int64)).to(device),torch.from_numpy(np.asarray(z[ib],np.float32)).to(device),torch.from_numpy(np.asarray(labels[ib],np.int64)).to(device))

def meta_objective(m,a,b):
    ea,ya,eb,yb=a; fa,fy,fb,fz=b
    la,na=symmetric_ce(m(ea),ya,m(eb),yb); lb,nb=symmetric_ce(m(fa),fy,m(fb),fz)
    params=tuple(p for p in m.parameters() if p.requires_grad)
    ga=torch.autograd.grad(la,params,create_graph=True,retain_graph=True,allow_unused=False)
    gb=torch.autograd.grad(lb,params,create_graph=True,retain_graph=True,allow_unused=False)
    va=torch.cat([g.reshape(-1) for g in ga]);vb=torch.cat([g.reshape(-1) for g in gb]);align=1-F.cosine_similarity(va[None,:],vb[None,:],dim=1).mean()
    return 0.5*(la+lb)+LAMBDA*align,{'support_loss':float(la.detach()),'query_loss':float(lb.detach()),'alignment':float(align.detach()),'support_queries':na,'query_queries':nb}

def base_group_stats(z,labels,groups):
    out={p:[] for p in FIT}
    for g in groups:
        p=str(g['pair']);
        ia=np.asarray(g['sides'][0]);ib=np.asarray(g['sides'][1]);
        vals=[];n=0
        for q,qy,c,cy in ((ia,labels[ia],ib,labels[ib]),(ib,labels[ib],ia,labels[ia])):
            kc=cy>=0
            for i in range(len(q)):
                if qy[i]<0:continue
                known_c=c[kc]; known_y=cy[kc]; pos=np.flatnonzero(known_y==qy[i])
                if len(pos):
                    logits=(z[q[i]]@z[known_c].T)/TEMP; logits=logits-logits.max()
                    vals.append(float(-np.log(np.exp(logits[pos[0]])/np.exp(logits).sum()))); n+=1
        if vals:out[p].append(float(np.mean(vals)))
    return {p:{'groups':len(v),'mean_ce':float(np.mean(v)) if v else None,'std_ce':float(np.std(v)) if v else None} for p,v in out.items()}

def cpu():
    x,z,labels,groups=load_data();by={p:[] for p in FIT}
    for g in groups:by[str(g['pair'])].append(g)
    m=CPIA().eval();g0=groups[0];a=group_tensors(z,labels,g0);other=next(g for g in groups if str(g['pair'])!=str(g0['pair']));b=group_tensors(z,labels,other)
    with torch.no_grad():zero=float((m(a[0])-a[0]).abs().max().item())
    m.train();obj,metrics=meta_objective(m,a,b);obj.backward();finite=all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.0))
    if zero>=1e-6 or not finite or not np.isfinite(gn):raise ValueError('CPIA CPU contract failed')
    if set(by)!=set(FIT) or any(not v for v in by.values()):raise ValueError('pair environment coverage failed')
    out={'status':'PASS_P61_CPU_CONTRACT','fit_rows':len(z),'schedule_groups':len(groups),'groups_by_pair':{p:len(v) for p,v in by.items()},'zero_init_max_error':zero,'finite_gradients':finite,'gradient_norm':gn,'meta_objective':metrics,'lambda':LAMBDA,'temperature':TEMP,'fit_pair_stats':base_group_stats(z,labels,groups),'official_val_test_read':False,'source_sha256':sha(Path(__file__)),'prereg_sha256':sha(OUT/'PREREG.json')};dump(OUT/'SUPPORT_CPU.json',out);print(json.dumps(out,indent=2))

def train():
    cpu();x,z,labels,groups=load_data();by={p:[] for p in FIT}
    for g in groups:by[str(g['pair'])].append(g)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0) or torch.cuda.get_device_properties(0).total_memory<30*1024**3:raise RuntimeError('GPU verification failed')
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42);m=CPIA().cuda().train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);hist=[];start=time.monotonic()
    for step,g in enumerate(groups,1):
        p=str(g['pair']);q=FIT[FIT.index(p)+1 if FIT.index(p)+1<len(FIT) else 0]; source=g; target=by[q][(step*7)%len(by[q])];
        a=group_tensors(z,labels,source,'cuda');b=group_tensors(z,labels,target,'cuda');obj,met=meta_objective(m,a,b)
        if not torch.isfinite(obj):raise ValueError('nonfinite CPIA objective')
        opt.zero_grad(set_to_none=True);obj.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        if not torch.isfinite(gn):raise ValueError('nonfinite CPIA gradient')
        opt.step()
        if step==1 or step%100==0:hist.append({'step':step,'objective':float(obj.detach()),'gradient_norm':float(gn),**met})
    od=OUT/'runs';od.mkdir(parents=True,exist_ok=False);ck=od/'fixed_step_001200.pt';torch.save({'state_dict':{k:v.detach().cpu() for k,v in m.state_dict().items()},'step':len(groups),'init_sha256':sha(INIT),'prereg_sha256':sha(OUT/'PREREG.json')},ck);rec={'status':'PASS_P61_TRAINING','updates':len(groups),'checkpoint_sha256':sha(ck),'gpu_uuid':UUID,'gpu_name':torch.cuda.get_device_name(0),'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'history':hist,'elapsed_seconds':time.monotonic()-start,'official_val_test_read':False};dump(od/'TRAINING_RECEIPT.json',rec);(od/'training.exit').write_text('0\n');print(json.dumps(rec,indent=2))

def evaluate():
    _,_,_,_=load_data();raw=torch.load(OUT/'runs/fixed_step_001200.pt',map_location='cpu',weights_only=False);m=CPIA().eval();m.load_state_dict(raw['state_dict'],strict=True);head=BaseHead();head.load_state_dict(torch.load(INIT,map_location='cpu',weights_only=False)['head_state'],strict=True);head.eval();inputs={r['sequence']:r for r in json.loads((OLD/'p2/cache/MANIFEST.json').read_text())['sequences']};per={}
    for pair in CAL:
        rec={r['pair']:r for r in json.loads((OLD/'p4_diagnosis/crop32/FRAME_MANIFEST.json').read_text())['pairs']}[pair];row,features,refs,cov=p4.align_pair(rec,OLD/'p4_diagnosis/crop32',inputs)
        with torch.inference_mode():z=F.normalize(head.raw(torch.from_numpy(features['fpn256']),torch.from_numpy(features['dino384'])),dim=-1);e=m(z).numpy();z=z.numpy()
        p4.METHODS=('target_only','cpia');arr=p4.evaluate_arrays(row,{'target_only':z,'cpia':e});s=p4.summaries(arr)['all']['methods'];b=s['target_only']['all_candidates']['known_positive_r1_tie_averaged'];c=s['cpia']['all_candidates']['known_positive_r1_tie_averaged'];per[pair]={'target_only':float(b),'cpia':float(c),'delta':float(c-b),'pair_wins':bool(c>b),'queries':s['cpia']['all_candidates']['eligible_queries']};print(json.dumps({'pair':pair,**per[pair]}),flush=True)
    base=float(np.mean([v['target_only'] for v in per.values()]));cand=float(np.mean([v['cpia'] for v in per.values()]));delta=cand-base;wins=sum(v['pair_wins'] for v in per.values());result={'status':'P61_CALIBRATION_GATE_COMPLETE','per_pair':per,'macro':{'target_only':base,'cpia':cand,'delta':delta},'pair_wins':wins,'gate':{'minimum_macro_delta':.005,'minimum_pair_wins':3,'minimum_calibration_macro':.48412456211781995,'passed':bool(delta>=.005 and wins>=3 and cand>=.48412456211781995)},'official_val_test_read':False,'training_completed':True,'source_sha256':sha(Path(__file__)),'checkpoint_sha256':sha(OUT/'runs/fixed_step_001200.pt')};dump(OUT/'SIGNAL.json',result);print(json.dumps(result,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['cpu','train','evaluate'],required=True);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);{'cpu':cpu,'train':train,'evaluate':evaluate}[a.mode]()
if __name__=='__main__':main()
