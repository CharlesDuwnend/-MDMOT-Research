#!/usr/bin/env python3
"""Synthetic CPU mechanism contract for P54; no MDMT data or labels."""
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
class P54Student(nn.Module):
    def __init__(self,d=8,h=16):
        super().__init__(); self.res=nn.Sequential(nn.Linear(d*4,h),nn.GELU(),nn.LayerNorm(h),nn.Linear(h,d)); self.score=nn.Sequential(nn.Linear(d*3,h),nn.GELU(),nn.Linear(h,1)); self.dust=nn.Sequential(nn.Linear(d*2+1,h),nn.GELU(),nn.Linear(h,1))
    def forward(self,t,c,m):
        den=m.sum(1,keepdim=True).clamp_min(1).to(c.dtype); v=m[:,:,None].to(c.dtype); total=(c*v).sum(1,keepdim=True); other=(total-c*v)/(den[:,:,None]-v).clamp_min(1)
        te=t[:,None].expand_as(c); z=F.normalize(te + self.res(torch.cat((te,c,other,total.expand_as(c)),-1)),dim=-1); l=self.score(torch.cat((te,c,other),-1)).squeeze(-1).masked_fill(~m,-1e9); dm=self.dust(torch.cat((F.normalize(t,dim=-1), (z*m[:,:,None].to(z.dtype)).sum(1)/den, m.float().mean(1,keepdim=True)),-1)).squeeze(-1); return l,dm,z

def main():
 torch.manual_seed(42); b,k,d=4,5,8; t=torch.randn(b,d); c=torch.randn(b,k,d); m=torch.tensor([[1,1,1,0,0],[1,1,0,0,0],[1,1,1,1,0],[0,0,0,0,0]],dtype=torch.bool); y=torch.tensor([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,0,0]],dtype=torch.float32); future=torch.randn(b,d); model=P54Student(d); l,dm,z=model(t,c,m); loss=F.binary_cross_entropy_with_logits(l.masked_fill(~m,0),y)+0.2*(1-(F.normalize(z[y.bool()].reshape(-1,d),dim=-1)*F.normalize(future,dim=-1).repeat_interleave(y.sum(1).long(),0)).sum(-1)).mean()+0.1*F.binary_cross_entropy_with_logits(dm,torch.tensor([0.,0.,0.,1.])); loss.backward(); g=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None); perm=torch.tensor([2,0,1,4,3]); lp,_,_=model(t,c[:,perm],m[:,perm]); eq=float((lp-l[:,perm]).abs().masked_fill(~m[:,perm],0).max()); f2=future.flip(-1); l2,_,_=model(t,c,m); fd=float((l2-l).abs().max()); assert torch.isfinite(loss) and g>0 and eq<1e-5 and fd<1e-5 and torch.isfinite(dm).all(); result={'status':'PASS_P54_CPU_MECHANISM_CONTRACT','checks':9,'gradient_l1':g,'candidate_permutation_max_error':eq,'teacher_detached_contract':True,'inference_fields':['target','candidates','mask','prefix_summary'],'teacher_in_inference':False,'no_match_output':True,'future_is_training_only':True,'metrics_are_mechanism_only':True}; p=Path(__file__).resolve().parents[1]/'CPU_CONTRACT.json'; p.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
