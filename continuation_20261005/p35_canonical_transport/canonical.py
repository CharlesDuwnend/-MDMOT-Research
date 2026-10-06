import torch
import numpy as np

class AnchorHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.Sequential(torch.nn.Linear(389,128),torch.nn.GELU(),torch.nn.Linear(128,2))
        torch.nn.init.zeros_(self.layers[-1].weight);torch.nn.init.zeros_(self.layers[-1].bias)
    def forward(self,feat,boxes):
        wh=(boxes[:,2:]-boxes[:,:2]).clamp_min(1e-6)
        cent=(boxes[:,:2]+boxes[:,2:])/2
        shape=torch.cat((cent,torch.log(wh.clamp_min(1e-5)),torch.log((wh[:,0]/wh[:,1]).clamp_min(1e-5))[:,None]),dim=1)
        offset=.45*torch.tanh(self.layers(torch.cat((feat,shape),1)))
        return cent+offset*wh,offset

def dlt(src,dst):
    # Solve normalized image coordinates with h33=1. Target objects excluded by caller.
    src=src.to(torch.float64);dst=dst.to(torch.float64)
    x,y=src.unbind(1);u,v=dst.unbind(1);one=torch.ones_like(x);zero=torch.zeros_like(x)
    A=torch.stack((torch.stack((x,y,one,zero,zero,zero,-u*x,-u*y),1),torch.stack((zero,zero,zero,x,y,one,-v*x,-v*y),1)),1).reshape(-1,8)
    b=torch.stack((u,v),1).reshape(-1,1)
    # Explicit ridge avoids rank-deficient tiny-support gradients; fixed regularization.
    h=torch.linalg.solve(A.T@A+1e-7*torch.eye(8,dtype=A.dtype,device=A.device),A.T@b).ravel()
    return torch.cat((h,one[:1])).reshape(3,3)

def project(H,xy):
    p=torch.cat((xy.to(H.dtype),torch.ones((len(xy),1),dtype=H.dtype,device=xy.device)),1)@H.T
    den=p[:,2:];den=torch.where(den.abs()<1e-7,torch.where(den<0,-1e-7,1e-7),den)
    return p[:,:2]/den

def crossfit_error(pa,pb,fold,scale):
    support=fold!=0;target=fold==0
    H=dlt(pa[support],pb[support]); G=dlt(pb[support],pa[support])
    ea=(project(H,pa[target])-pb[target])*scale
    eb=(project(G,pb[target])-pa[target])*scale
    return ea,eb
