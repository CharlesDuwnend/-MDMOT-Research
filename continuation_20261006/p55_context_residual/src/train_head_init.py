#!/usr/bin/env python3
"""P55 matched short run initialized from the verified P23 head-only adapter."""
import hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3]; P23=ROOT/'continuation_20261004/p23_visual/data'; OUT=ROOT/'continuation_20261006/p55_context_residual/runs_v4'
FIT_PAIRS={"23","25","28","29","39","44","45","51","53","63","66","69","70","74","78"}
INIT=ROOT/'continuation_20261004/p23_visual/runs/head_only/fixed_step_001200.pt'

class BaseHead(nn.Module):
    def __init__(self):
        super().__init__(); self.projection=nn.Sequential(nn.Linear(256,128),nn.LayerNorm(128),nn.GELU()); self.ffn=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,128)); self.norm=nn.LayerNorm(128); self.adapter=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,128))
    def raw(self, fpn, target):
        h=self.projection(fpn)+self.adapter(target); return self.norm(h+self.ffn(h))

class CVCRv2(nn.Module):
    def __init__(self):
        super().__init__(); self.base=BaseHead(); ck=torch.load(INIT,map_location='cpu',weights_only=False); self.base.load_state_dict(ck['head_state'],strict=True); self.base.requires_grad_(False)
        self.context=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,128),nn.GELU(),nn.Linear(128,128)); nn.init.zeros_(self.context[-1].weight); nn.init.zeros_(self.context[-1].bias)
        self.dust=nn.Linear(128,1); nn.init.zeros_(self.dust.weight); nn.init.zeros_(self.dust.bias)
    def forward(self,target,context,fpn,context_enabled=True):
        b=self.base.raw(fpn,target)
        r=self.context(torch.cat((context,target-context),dim=-1)) if context_enabled else torch.zeros_like(b)
        return F.normalize(b+r,dim=-1), self.dust(b+r).squeeze(-1)

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def group_loss(za,da,ya,zb,db,yb,temp=.07):
 def one(q,qd,qy,c,cy):
  vals=[]; dust=0
  for i in range(len(q)):
   if int(qy[i])<0:continue
   valid=cy>=0; logits=torch.cat(((q[i:i+1]@c[valid].T).flatten()/temp,qd[i:i+1])) if bool(valid.any()) else qd[i:i+1]
   pos=torch.where(cy[valid]==qy[i])[0] if bool(valid.any()) else torch.empty(0,dtype=torch.long)
   target=int(pos[0]) if len(pos) else len(logits)-1; dust+=int(target==len(logits)-1); vals.append(F.cross_entropy(logits[None],torch.tensor([target],device=logits.device)))
  return (torch.stack(vals).mean() if vals else q.sum()*0.,len(vals),dust)
 a=one(za,da,ya,zb,yb); b=one(zb,db,yb,za,ya); return (a[0]+b[0])/2,a[1]+b[1],a[2]+b[2]

def load():
 x=np.load(P23/'inputs.npz',allow_pickle=False); target=np.load(P23/'frozen_dino.npy',mmap_mode='r'); context=np.load(ROOT/'continuation_20261006/p55_context_residual/context_dino.npy',mmap_mode='r'); fpn=np.load(P23/'raw_fpn.npy',mmap_mode='r'); labels=np.load(P23/'offline_pid.npy',mmap_mode='r'); groups=json.loads((P23/'GROUPS.json').read_text())
 if not(len(target)==len(context)==len(fpn)==len(labels)==len(x['keys'])):raise ValueError('row mismatch')
 return target,context,fpn,labels,[g for g in groups if str(g['pair']) in FIT_PAIRS]

def run(name,target,context,fpn,labels,groups):
 torch.manual_seed(42);np.random.seed(42);random.seed(42);m=CVCRv2().cuda();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);hist=[];start=time.monotonic()
 for step in range(1200):
  g=groups[step%len(groups)]; ia=np.asarray(g['sides'][0]);ib=np.asarray(g['sides'][1]); ta=torch.from_numpy(np.asarray(target[ia],np.float32)).cuda();tb=torch.from_numpy(np.asarray(target[ib],np.float32)).cuda();ca=torch.from_numpy(np.asarray(context[ia],np.float32)).cuda();cb=torch.from_numpy(np.asarray(context[ib],np.float32)).cuda();fa=torch.from_numpy(np.asarray(fpn[ia],np.float32)).cuda();fb=torch.from_numpy(np.asarray(fpn[ib],np.float32)).cuda();ya=torch.from_numpy(np.asarray(labels[ia],np.int64)).cuda();yb=torch.from_numpy(np.asarray(labels[ib],np.int64)).cuda();za,da=m(ta,ca,fa,context_enabled=name=='context_residual');zb,db=m(tb,cb,fb,context_enabled=name=='context_residual');loss,n,d=group_loss(za,da,ya,zb,db,yb);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  if step==0 or (step+1)%100==0:hist.append({'step':step+1,'loss':float(loss.detach().cpu()),'queries':n,'dust_targets':d})
 out=OUT/name;out.mkdir(parents=True,exist_ok=False);torch.save({'name':name,'step':1200,'state_dict':m.state_dict()},out/'fixed_step_001200.pt');receipt={'status':'COMPLETE_P55_HEAD_INIT_SHORT','arm':name,'updates':1200,'fit_pairs':sorted(FIT_PAIRS),'history':hist,'checkpoint_sha256':sha(out/'fixed_step_001200.pt'),'p23_init_sha256':sha(INIT),'official_val_test_read':False,'elapsed_seconds':time.monotonic()-start};(out/'TRAINING_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');(out/'training.exit').write_text('0\n');return receipt

def main():
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('verified GPU required')
 target,context,fpn,labels,groups=load(); rs=[run(n,target,context,fpn,labels,groups) for n in ('target_only','context_residual')]; out={'status':'P55_HEAD_INIT_SHORT_COMPLETE','receipts':rs,'gpu_name':torch.cuda.get_device_name(0),'gpu_total_memory':torch.cuda.get_device_properties(0).total_memory,'official_val_test_read':False};(OUT/'TRAINING_RECEIPT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':out['status'],'gpu':out['gpu_name']}))

if __name__=='__main__':main()
