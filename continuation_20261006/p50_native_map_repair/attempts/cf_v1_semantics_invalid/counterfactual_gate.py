#!/usr/bin/env python3
"""Counterfactual positive-removal objective on the corrected native gate.

All arms share the same NativeSetResidual architecture and candidate data. The
control arm receives the same cardinality-matched negative-removal assignment
loss; only the counterfactual positive-removal dustbin loss is added in the
experimental arm. This is a train-only mechanism gate.
"""
from __future__ import annotations
import hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0,str(Path(__file__).resolve().parent))
import native_set_residual_gate as n

FIT=n.FIT; EVAL=n.EVAL; OUT=P

def negative_mask(a,c,m,p):
    # Remove as many highest baseline-similarity negatives as there are
    # positives.  The removal cardinality is label-derived only in training.
    out=m.clone(); base=(F.normalize(a,dim=-1)[:,None]*F.normalize(c+1e-8,dim=-1)).sum(-1).masked_fill(~m,-1e9)
    for i in range(m.shape[0]):
        q=int((p[i]&m[i]).sum())
        neg=(m[i]&~p[i]).nonzero().flatten().tolist()
        if q and neg:
            order=sorted(neg,key=lambda j:float(base[i,j]),reverse=True)[:min(q,len(neg))]
            out[i,order]=False
    return out

def cf_dustbin_loss(logits,dust,mask,active):
    all_l=torch.cat([logits,dust[:,None]],1); losses=[]
    for i in active.nonzero().flatten().tolist():
        losses.append(all_l[i,-1]-torch.logsumexp(all_l[i],0))
    return torch.stack(losses).neg().mean() if losses else all_l.sum()*0

def evaluate_cf(model,events,device,batch_size=128):
    model.eval(); total=0; cf_dust=[]; ctrl_r1=[]; orig_r1=[]
    with torch.no_grad():
      for st in range(0,len(events),batch_size):
        es=events[st:st+batch_size]; a,c,m,p,t,tm=n.pad_batch(es,device)
        l,d,z=model(a,c,m); nm=negative_mask(a,c,m,p); nl,nd,nz=model(a,c,nm); rm=m&~p; rl,rd,rz=model(a,c,rm)
        for i,e in enumerate(es):
          pos=(p[i]&m[i]);
          if not bool(pos.any()): continue
          total+=1; cf_dust.append(float(rd[i].item()> (rl[i].max().item() if bool(rm[i].any()) else -1e9)))
          co=(p[i]&nm[i]); oo=torch.argsort(l[i],descending=True).tolist(); cc=torch.argsort(nl[i],descending=True).tolist()
          pp=pos.nonzero().flatten().tolist();
          orig_r1.append(float(next((q for q,j in enumerate(oo,1) if j in pp),999)==1))
          ctrl_r1.append(float(next((q for q,j in enumerate(cc,1) if j in co.nonzero().flatten().tolist()),999)==1))
    return {'positive_events':total,'positive_removal_dustbin_rate':float(np.mean(cf_dust)) if cf_dust else None,'negative_removal_retain_r1':float(np.mean(ctrl_r1)) if ctrl_r1 else None,'original_r1_positive_subset':float(np.mean(orig_r1)) if orig_r1 else None}

def train_arm(train,evals,arm,seed,steps,device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=n.NativeSetResidual(relational=True).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); hist=[]; bs=64
    for step in range(steps):
      es=[train[(step*bs+j)%len(train)] for j in range(bs)]; a,c,m,p,t,tm=n.pad_batch(es,device)
      l,d,z=model(a,c,m); assign=n.assignment_loss(l,d,p,m); nm=negative_mask(a,c,m,p); nl,nd,nz=model(a,c,nm); control=n.assignment_loss(nl,nd,p,nm); rm=m&~p; rl,rd,rz=model(a,c,rm); cf=cf_dustbin_loss(rl,rd,rm,p.any(1))
      loss=assign+0.35*control+(0.35*cf if arm=='cf_on' else 0.)
      opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.); opt.step()
      if step in (0,steps//4,steps//2,3*steps//4,steps-1): hist.append({'step':step+1,'loss':float(loss.detach()),'assign':float(assign.detach()),'control':float(control.detach()),'cf':float(cf.detach())})
    results={sid:{**n.evaluate(model,ev,device),**evaluate_cf(model,ev,device)} for sid,ev in evals.items()}
    return model,hist,results

def main():
    steps=int(os.environ.get('P52_STEPS','2400')); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');t0=time.time(); ids=list(FIT+EVAL);data={}; audits={}
    for sid in ids:
      data[sid],audits[sid]=n.build_events(sid,include_teacher=False); print(json.dumps({'load_pair':sid,'events':len(data[sid]),'dropped':len(audits[sid]['dropped'])}),flush=True)
    train=sum((data[s] for s in FIT),[]); evals={s:data[s] for s in EVAL}; arms={}
    for arm in ('cf_off','cf_on'):
      arms[arm]=[]
      for seed in (7,17):
       print(json.dumps({'arm':arm,'seed':seed,'steps':steps,'device':str(device)}),flush=True); model,hist,res=train_arm(train,evals,arm,seed,steps,device); ck=OUT/f'counterfactual_{arm}_seed{seed}.pt'; torch.save({'state_dict':model.state_dict(),'arm':arm,'seed':seed,'steps':steps},ck); arms[arm].append({'seed':seed,'history':hist,'per_pair':res,'checkpoint':str(ck),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest()}); del model; torch.cuda.empty_cache()
    payload={'status':'COUNTERFACTUAL_GATE_COMPLETE','version':'2026-10-06-v1','device':str(device),'steps_per_arm':steps,'seeds':[7,17],'fit_pairs':list(FIT),'eval_pairs':list(EVAL),'fit_events':len(train),'pair_audit':audits,'arms':arms,'controls':{'native_features_frozen':True,'teacher_in_inference':False,'official_val_test_access':False,'negative_removal_cardinality_matched':True,'cf_loss_only_difference':'positive-removal dustbin term'},'elapsed_seconds':time.time()-t0}
    (OUT/'COUNTERFACTUAL_GATE.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({'status':payload['status'],'fit_events':len(train),'elapsed_seconds':payload['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
