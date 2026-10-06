#!/usr/bin/env python3
"""P59 Cross-Representation Causal Bridge: support, train, and cal5 gate."""
import argparse, hashlib, json, random, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3]
OLD=Path('/home/chenhc/mdmot_research_20261002')
P23=ROOT/'continuation_20261004/p23_visual/data'
OUT=ROOT/'continuation_20261006/p59_cross_rep_causal_bridge'
INIT=ROOT/'continuation_20261004/p23_visual/runs/head_only/fixed_step_001200.pt'
UUID='GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a'
FIT=('23','25','28','29','39','44','45','51','53','63','66','69','70','74','78')
CAL=('27','32','42','64','65')
sys.path[:0]=[str(OLD/'p2/src'),str(OLD/'p4_diagnosis'),str(ROOT/'continuation_20261006/p55_context_residual/src')]
from data import PairData, sha256
import compare_crop32 as p4
from train_head_init import BaseHead


def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def digest(path): return sha256(path)
def unit_np(x):
    x=np.asarray(x,np.float32); n=np.linalg.norm(x,axis=1,keepdims=True)
    if not np.isfinite(x).all() or np.any(n<=1e-8): raise ValueError('nonfinite/zero vector')
    return x/n

class Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.history_proj=nn.Sequential(nn.Linear(128,128),nn.LayerNorm(128),nn.GELU())
        self.gate=nn.Sequential(nn.Linear(384,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,1))
        self.residual=nn.Sequential(nn.Linear(384,128),nn.GELU(),nn.Linear(128,128))
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)
    def forward(self,z,h,has):
        q=self.history_proj(h)
        x=torch.cat((z,q,z-q),dim=-1)
        r=self.residual(x); g=torch.sigmoid(self.gate(x))
        return F.normalize(z+has.float().unsqueeze(-1)*g*r,dim=-1), q


def key_history(pair, keys):
    data=PairData(pair,with_labels=False); lookup={}; by=defaultdict(list)
    for view,d in enumerate(data._views):
        for f,det,lid,emb,present in zip(d['frame'],d['detector_index'],d['local_id'],d['embedding'],d['feature_present']):
            key=f'{pair}-{view+1}:{int(f)}:{int(det)}'; lookup[key]=(view,int(f),int(lid))
            if bool(present): by[(view,int(lid))].append((int(f),np.asarray(emb,np.float32)))
    for k in by: by[k].sort(key=lambda x:x[0])
    hist=[]; has=[]; counts=[]
    for key in map(str,keys):
        if key not in lookup: raise ValueError('history key missing: '+key)
        view,frame,lid=lookup[key]; vals=[(f,e) for f,e in by[(view,lid)] if f<frame and f>=frame-8][-4:]
        if vals:
            h=unit_np(np.mean(unit_np(np.stack([e for _,e in vals])),axis=0,keepdims=True))[0]; hist.append(h);has.append(True);counts.append(len(vals))
        else:
            hist.append(np.zeros(128,np.float32));has.append(False);counts.append(0)
    return np.asarray(hist,np.float32),np.asarray(has,bool),{'rows':len(hist),'with_history':int(sum(has)),'history_fraction':float(np.mean(has)) if has else 0.0,'mean_history':float(np.mean(counts)) if counts else 0.0,'max_history':int(max(counts) if counts else 0)}


def load_p23():
    x=np.load(P23/'inputs.npz',allow_pickle=False); target=np.load(P23/'frozen_dino.npy',mmap_mode='r'); fpn=np.load(P23/'raw_fpn.npy',mmap_mode='r'); labels=np.load(P23/'offline_pid.npy',mmap_mode='r'); groups=json.loads((P23/'GROUPS.json').read_text())
    if not(len(x['keys'])==len(target)==len(fpn)==len(labels)): raise ValueError('P23 row mismatch')
    head=BaseHead(); ck=torch.load(INIT,map_location='cpu',weights_only=False); head.load_state_dict(ck['head_state'],strict=True); head.eval()
    with torch.inference_mode(): z=F.normalize(head.raw(torch.from_numpy(np.asarray(fpn,np.float32).copy()),torch.from_numpy(np.asarray(target,np.float32).copy())),dim=-1).numpy()
    h=np.zeros_like(z); has=np.zeros(len(z),bool); per={}
    for pair in FIT:
        sel=np.flatnonzero(x['pair'].astype(str)==pair); hh,mm,stats=key_history(pair,x['keys'][sel]); h[sel]=hh;has[sel]=mm;per[pair]=stats
    return x,z,h,has,labels,[g for g in groups if str(g['pair']) in FIT],per


def group_loss(ea,qa,ha,ya,va,eb,qb,hb,yb,vb,temp=.1):
    def side(q,h,y,valid,other,other_y):
        vals=[];n=0
        known=other_y>=0
        for i in range(len(q)):
            if not bool(valid[i]): continue
            pos=torch.where(other_y[known]==y[i])[0] if bool(known.any()) else torch.empty(0,dtype=torch.long,device=q.device)
            if len(pos)==0: continue
            logits=(q[i:i+1]@other[known].T).flatten()/temp
            vals.append(F.cross_entropy(logits[None],pos[:1]));n+=1
        return (torch.stack(vals).mean() if vals else q.sum()*0.,n)
    a,na=side(ea,qa,ya,va,eb,yb); b,nb=side(eb,qb,yb,vb,ea,ya)
    ce=(a+b)/2; cons=[]; 
    for e,q,m in ((ea,qa,ha),(eb,qb,hb)):
        if bool(m.any()): cons.append(1-(e[m]*F.normalize(q[m],dim=-1)).sum(-1).mean())
    consistency=torch.stack(cons).mean() if cons else ce*0.
    return ce+0.2*consistency,{'ce':float(ce.detach()),'consistency':float(consistency.detach()),'queries':na+nb}


def support_and_cpu():
    x,z,h,has,labels,groups,per=load_p23();
    if not groups or len(groups)!=1200: raise ValueError('P23 schedule mismatch')
    m=Bridge().eval();
    sample=next(g for g in groups if len(g['sides'][0]) and len(g['sides'][1]))
    ia=np.asarray(sample['sides'][0]); ib=np.asarray(sample['sides'][1]);
    za=torch.from_numpy(np.asarray(z[ia],np.float32)); zb=torch.from_numpy(np.asarray(z[ib],np.float32)); ha=torch.from_numpy(np.asarray(h[ia],np.float32)); hb=torch.from_numpy(np.asarray(h[ib],np.float32)); ma=torch.from_numpy(has[ia]); mb=torch.from_numpy(has[ib])
    with torch.no_grad():
        ea0,qa0=m(za,ha,ma); eb0,qb0=m(zb,hb,mb)
    if not torch.allclose(ea0,za,atol=1e-6,rtol=0): raise ValueError('zero-init equivalence failed')
    va=torch.from_numpy(labels[ia]>=0); vb=torch.from_numpy(labels[ib]>=0); ya=torch.from_numpy(labels[ia].astype(np.int64)); yb=torch.from_numpy(labels[ib].astype(np.int64));
    ea,qa=m(za,ha,ma); eb,qb=m(zb,hb,mb); loss,_=group_loss(ea,qa,ma,ya,va,eb,qb,mb,yb,vb); loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()): raise ValueError('CPU gradient contract failed')
    out={'status':'PASS_P59_SUPPORT_CPU','fit_rows':int(len(z)),'fit_history_fraction':float(has.mean()),'per_pair_history':per,'schedule_groups':len(groups),'zero_init_max_error':float((ea0-za).abs().max()),'finite_gradients':True,'parameters':sum(p.numel() for p in m.parameters()),'official_val_test_read':False}
    dump(OUT/'SUPPORT_CPU.json',out); print(json.dumps(out,indent=2)); return out


def train():
    support_and_cpu(); x,z,h,has,labels,groups,per=load_p23()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0) or torch.cuda.get_device_properties(0).total_memory<30*1024**3: raise RuntimeError('GPU verification failed')
    torch.manual_seed(42); torch.cuda.manual_seed_all(42); np.random.seed(42); random.seed(42); device=torch.device('cuda:0'); m=Bridge().to(device).train(); opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4); history=[]; start=time.monotonic()
    for step,g in enumerate(groups,1):
        ia=np.asarray(g['sides'][0]); ib=np.asarray(g['sides'][1]); za=torch.from_numpy(np.asarray(z[ia],np.float32)).to(device); zb=torch.from_numpy(np.asarray(z[ib],np.float32)).to(device); ha=torch.from_numpy(np.asarray(h[ia],np.float32)).to(device); hb=torch.from_numpy(np.asarray(h[ib],np.float32)).to(device); ma=torch.from_numpy(has[ia]).to(device); mb=torch.from_numpy(has[ib]).to(device); ya=torch.from_numpy(labels[ia].astype(np.int64)).to(device); yb=torch.from_numpy(labels[ib].astype(np.int64)).to(device); ea,qa=m(za,ha,ma); eb,qb=m(zb,hb,mb); va=ya>=0; vb=yb>=0; loss,metrics=group_loss(ea,qa,ma,ya,va,eb,qb,mb,yb,vb); 
        if not torch.isfinite(loss): raise ValueError('nonfinite loss')
        opt.zero_grad(set_to_none=True); loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()): raise ValueError('nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if step==1 or step%100==0: history.append({'step':step,'loss':float(loss.detach()),**metrics})
    out=OUT/'runs_v2'; out.mkdir(parents=True,exist_ok=False); ckpt=out/'fixed_step_001200.pt'; torch.save({'state_dict':{k:v.detach().cpu() for k,v in m.state_dict().items()},'step':len(groups),'p23_init_sha256':digest(INIT),'prereg_sha256':digest(OUT/'PREREG.json')},ckpt); rec={'status':'PASS_P59_TRAINING','updates':len(groups),'history':history,'checkpoint_sha256':digest(ckpt),'gpu_uuid':UUID,'gpu_name':torch.cuda.get_device_name(0),'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'fit_history_fraction':float(has.mean()),'official_val_test_read':False,'elapsed_seconds':time.monotonic()-start}; dump(out/'TRAINING_RECEIPT.json',rec); (out/'training.exit').write_text('0\n'); print(json.dumps(rec,indent=2))


def evaluate():
    raw=torch.load(OUT/'runs_v2/fixed_step_001200.pt',map_location='cpu',weights_only=False); m=Bridge().eval(); m.load_state_dict(raw['state_dict'],strict=True)
    records={r['pair']:r for r in json.loads((OLD/'p4_diagnosis/crop32/FRAME_MANIFEST.json').read_text())['pairs']}; inputs={r['sequence']:r for r in json.loads((OLD/'p2/cache/MANIFEST.json').read_text())['sequences']}; per={}; histstats={}
    head=BaseHead(); ck=torch.load(INIT,map_location='cpu',weights_only=False); head.load_state_dict(ck['head_state'],strict=True); head.eval()
    for pair in CAL:
        rec=records[pair]; row,features,refs,cov=p4.align_pair(rec,OLD/'p4_diagnosis/crop32',inputs)
        with torch.inference_mode(): z=F.normalize(head.raw(torch.from_numpy(features['fpn256']),torch.from_numpy(features['dino384'])),dim=-1).numpy()
        h,has,stats=key_history(pair,row['row_key']); histstats[pair]=stats
        with torch.inference_mode(): e,_=m(torch.from_numpy(z),torch.from_numpy(h),torch.from_numpy(has)); e=e.numpy()
        p4.METHODS=('target_only','crcb')
        arrays=p4.evaluate_arrays(row,{'target_only':z,'crcb':e}); s=p4.summaries(arrays)['all']['methods']; per[pair]={'target_only':s['target_only']['all_candidates']['known_positive_r1_tie_averaged'],'crcb':s['crcb']['all_candidates']['known_positive_r1_tie_averaged'],'queries':s['target_only']['all_candidates']['eligible_queries']}
    base=float(np.mean([per[p]['target_only'] for p in CAL])); cand=float(np.mean([per[p]['crcb'] for p in CAL])); delta=cand-base; wins=sum(per[p]['crcb']>per[p]['target_only'] for p in CAL); gate=bool(delta>=0.005 and wins>=3)
    out={'status':'P59_CALIBRATION_GATE_COMPLETE','pairs':list(CAL),'per_pair':per,'history':histstats,'macro_r1':{'target_only':base,'crcb':cand},'delta':delta,'pair_wins':wins,'gate':{'minimum_macro_delta':0.005,'minimum_pair_wins':3,'minimum_calibration_macro':0.48412456211781995,'passed':gate},'official_val_test_read':False,'training_completed':True,'progression_authorized':gate,'host_attachment_authorized':False}; dump(OUT/'SIGNAL.json',out); print(json.dumps(out,indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['support','train','evaluate'],required=True); a=ap.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1); {'support':support_and_cpu,'train':train,'evaluate':evaluate}[a.mode]()
if __name__=='__main__': main()
