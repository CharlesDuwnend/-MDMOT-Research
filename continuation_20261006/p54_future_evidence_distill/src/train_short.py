#!/usr/bin/env python3
"""P54 fixed short train-only representation gate on grouped MDMT train pairs."""
import hashlib, json, os, random, sys, time, subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import data_audit as data
FIT=data.FIT; EVAL=data.EVAL
SEED=42; STEPS=1200; BATCH=128; LIMIT_PER_FIT_PAIR=2000
EXPECTED_UUID='GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a'

class CandidateResidual(nn.Module):
    """Permutation-equivariant candidate-set residual with explicit no-match."""
    def __init__(self,d=256,h=192):
        super().__init__()
        self.residual=nn.Sequential(nn.Linear(d*4,h),nn.GELU(),nn.LayerNorm(h),nn.Linear(h,d))
        self.score=nn.Sequential(nn.Linear(d*4,h),nn.GELU(),nn.LayerNorm(h),nn.Linear(h,1))
        self.dustbin=nn.Sequential(nn.Linear(d*2+1,h),nn.GELU(),nn.Linear(h,1))
    def forward(self,target,candidates,mask):
        den=mask.sum(1,keepdim=True).clamp_min(1).to(candidates.dtype)
        valid=mask[:,:,None].to(candidates.dtype)
        c=F.normalize(candidates,dim=-1); t=F.normalize(target,dim=-1)
        total=(c*valid).sum(1,keepdim=True)
        other=(total-c*valid)/(den[:,:,None]-valid).clamp_min(1.0)
        setmean=(total/den[:,:,None])[:,0]
        te=t[:,None].expand_as(c); sm=setmean[:,None].expand_as(c)
        x=torch.cat((te,c,other,sm),-1)
        z=F.normalize(c+self.residual(x),dim=-1)
        logits=self.score(x).squeeze(-1) + (te*z).sum(-1)
        logits=logits.masked_fill(~mask,torch.finfo(logits.dtype).min)
        dust=self.dustbin(torch.cat((t,setmean,mask.float().mean(1,keepdim=True)),-1)).squeeze(-1)
        return logits,dust,z

def pad_events(events,device):
    b=len(events); k=max(1,max(e['valid'] for e in events)); d=256
    t=torch.zeros(b,d,device=device); c=torch.zeros(b,k,d,device=device); m=torch.zeros(b,k,dtype=torch.bool,device=device); y=torch.zeros(b,k,device=device); teach=torch.zeros(b,d,device=device); has=torch.zeros(b,dtype=torch.bool,device=device); no=torch.zeros(b,dtype=torch.float32,device=device)
    for i,e in enumerate(events):
        t[i]=torch.from_numpy(e['target']).to(device); n=e['valid']; c[i,:n]=torch.from_numpy(e['candidates']).to(device); m[i,:n]=True; y[i,e['positive']]=1.; no[i]=float(len(e['positive'])==0)
        if e['teacher'] is not None: teach[i]=torch.from_numpy(e['teacher']).to(device); has[i]=True
    return t,c,m,y,teach,has,no

def event_metrics(model,events,device):
    model.eval(); ranks=[]; no_scores=[]; base=[]
    with torch.no_grad():
      for st in range(0,len(events),256):
        sel=events[st:st+256]; t,c,m,y,teach,has,no=pad_events(sel,device); l,dust,z=model(t,c,m)
        base_score=(F.normalize(t,dim=-1)[:,None]*F.normalize(c,dim=-1)).sum(-1).masked_fill(~m,-1e9)
        for i,e in enumerate(sel):
          if not e['positive']: continue
          pos=set(e['positive']); order=torch.argsort(l[i],descending=True).tolist(); bo=torch.argsort(base_score[i],descending=True).tolist(); ranks.append(next(r for r,j in enumerate(order,1) if j in pos)); base.append(next(r for r,j in enumerate(bo,1) if j in pos))
        no_scores.extend((torch.sigmoid(dust)[no.bool()].detach().cpu().tolist()))
    return {'events':len(ranks),'student_r1':float(np.mean(np.asarray(ranks)==1)) if ranks else None,'baseline_r1':float(np.mean(np.asarray(base)==1)) if base else None,'student_mrr':float(np.mean(1/np.asarray(ranks))) if ranks else None,'baseline_mrr':float(np.mean(1/np.asarray(base))) if base else None,'no_match_score_mean':float(np.mean(no_scores)) if no_scores else None}

def main():
  random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
  if not torch.cuda.is_available(): raise SystemExit('CUDA required for P54 short gate')
  device=torch.device('cuda'); prop=torch.cuda.get_device_properties(device); uuid=torch.cuda.get_device_name(device); torch.cuda.reset_peak_memory_stats(device)
  gpu_uuid=os.environ.get('P54_GPU_UUID','')
  try:
    snap=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip()
  except Exception:
    snap=''
  train=[]; build_counts={}
  for sid in FIT:
    ev,c=data.build_pair(sid,limit=LIMIT_PER_FIT_PAIR,include_no_positive=True); train.extend(ev); build_counts[sid]=c
  eval_by={sid:data.build_pair(sid)[0] for sid in EVAL}
  model=CandidateResidual().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
  history=[]; start=time.time()
  for step in range(1,STEPS+1):
    ix=torch.randint(0,len(train),(min(BATCH,len(train)),),device=device); sel=[train[int(j)] for j in ix.cpu().tolist()]; t,c,m,y,teach,has,no=pad_events(sel,device); logits,dust,z=model(t,c,m)
    # Valid candidate BCE; no-match target for candidate-empty-positive events.
    assign=F.binary_cross_entropy_with_logits(logits[m],y[m])
    dust_loss=F.binary_cross_entropy_with_logits(dust,no)
    align_terms=[]
    for i in range(len(sel)):
      if bool(has[i]) and sel[i]['positive']:
        p=torch.tensor(sel[i]['positive'],device=device,dtype=torch.long); align_terms.append(1-(z[i,p].mean(0)*F.normalize(teach[i],dim=-1)).sum())
    align=torch.stack(align_terms).mean() if align_terms else assign*0
    loss=assign+0.15*dust_loss+0.30*align
    opt.zero_grad(set_to_none=True); loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.0)); assert np.isfinite(grad); opt.step()
    if step in (1,300,600,900,1200): history.append({'step':step,'loss':float(loss.detach()),'assign':float(assign.detach()),'dustbin':float(dust_loss.detach()),'align':float(align.detach()),'grad_norm':grad})
  ck=HERE.parent/'runs'; ck.mkdir(exist_ok=True); ck=ck/'fixed_step_001200.pt'; torch.save(model.state_dict(),ck)
  results={sid:event_metrics(model,ev,device) for sid,ev in eval_by.items()}
  cal=[r for r in results.values() if r['student_r1'] is not None]; student=float(np.mean([r['student_r1'] for r in cal])); base=float(np.mean([r['baseline_r1'] for r in cal])); wins=sum(r['student_r1']>r['baseline_r1'] for r in cal)
  receipt={'status':'PASS_P54_SHORT_TRAINING','steps':STEPS,'fit_events':len(train),'fit_pairs':list(FIT),'eval_pairs':list(EVAL),'seed':SEED,'gpu_name':str(prop.name),'gpu_uuid':str(gpu_uuid),'gpu_visible_name':uuid,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES',''),'gpu_snapshot':snap,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'elapsed_seconds':time.time()-start,'checkpoint':str(ck),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'teacher_in_inference':False,'official_val_test_access':False,'build_counts':build_counts}
  out={'status':'PASS_P54_SHORT_GATE_COMPLETE','calibration_pair_macro':{'student_r1':student,'baseline_r1':base,'delta':student-base,'wins':wins,'minimum_delta':0.01,'minimum_wins':3,'pass':student-base>=0.01 and wins>=3},'per_pair':results,'training_receipt':receipt,'history':history,'metrics_are_train_only':True}
  (HERE.parent/'runs'/'TRAINING_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n'); (HERE.parent/'SHORT_GATE.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
