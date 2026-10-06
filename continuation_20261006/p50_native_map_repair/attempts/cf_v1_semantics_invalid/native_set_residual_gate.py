#!/usr/bin/env python3
"""Native-feature causal gate for a candidate-conditioned residual module.

The detector and its stage-3 feature cache are frozen.  The module operates on
native 256-D ROI features and candidate sets, with no image resize or new FPN.
Training-only teachers are fixed raw source features from strict past frames.
All valid episodes, including zero-positive/no-candidate rows, remain in the
population.  No val/test data are used for training or model selection.
"""
from __future__ import annotations
import gzip, hashlib, json, math, os, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SCI=Path('/home/chenhc/mdmot_sci_id_20260906')
ROOT=Path('/home/chenhc/cross_uav_query_mamba_20260830')
sys.path.insert(0,str(ROOT))
import stage3_runner as s3

FIT=('23','25','27','28','29','30','32','39','42','44','45','50','51','53','54')
EVAL=('69','70','74','76','78')
BASE=ROOT/'features/train'
OUT=Path(__file__).resolve().parents[1]
D=256; KMAX=64


def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def load_store(sid, dev):
    idx=[]
    with (BASE/f'{sid}-{dev}.jsonl').open() as f:
        idx=[json.loads(x) for x in f if x.strip()]
    arr=np.load(BASE/f'{sid}-{dev}.npz',allow_pickle=False)['visual_feature'].astype(np.float32)
    assert len(idx)==len(arr)
    # The episode cache and stage3 cache use different detector-index scopes in
    # a few historical sequences.  Keep both the metadata and an IoU resolver;
    # matching the frozen detector box is the stable cross-cache contract.
    by_frame=defaultdict(list)
    for r in idx:
        by_frame[(int(r['frame_id']),int(r['class_id']))].append((r['bbox'],arr[int(r['feature_row'])],int(r['detection_index'])))
    return by_frame, idx

def resolve(store, fr, cls, bbox, threshold=0.5):
    candidates=store.get((int(fr),int(cls)),[])
    if not candidates: return None
    best=max(((s3.iou(bbox,x[0]),x[1]) for x in candidates),key=lambda x:x[0])
    return best[1] if best[0] >= threshold else None

def frame_maps(sid):
    maps={}
    for rec in s3.records('train',(sid,)):
        dev=str(rec['device']); paths=s3.frame_paths(Path(rec['image_dir']))
        maps[dev]={str(p.resolve()):i for i,p in enumerate(paths)}
    return maps

def gid_for(feature_row, gt):
    best=max(((s3.iou(feature_row['bbox'],g['bbox']),g) for g in gt),key=lambda x:x[0]) if gt else (0.,None)
    return int(best[1]['gt_id']) if best[0] >= .5 else None

def build_prefix(sid, dev, rows):
    gt=s3.read_gt(sid,dev); history=defaultdict(list)
    for r in rows:
        fr=int(r['frame_id']); gid=gid_for(r,gt.get(fr,[]))
        if gid is not None: history[gid].append((fr,int(r['class_id']),list(r['bbox'])))
    return {g:sorted(v) for g,v in history.items()}

def build_events(sid, include_teacher=True):
    stores={}; rows_by={}; prefix={}; fmap=frame_maps(sid)
    for dev in ('1','2'):
        stores[dev],rows_by[dev]=load_store(sid,dev)
        if include_teacher: prefix[dev]=build_prefix(sid,dev,rows_by[dev])
    path=SCI/'data/train_episodes_v1'/f'{sid}.jsonl.gz'
    events=[]; dropped=[]; counts=defaultdict(int)
    with gzip.open(path,'rt') as f:
      for line_no,line in enumerate(f,1):
        r=json.loads(line); target_path=str(Path(r['target_image']).resolve()); source_path=str(Path(r['source_image']).resolve())
        if target_path not in fmap[str(r['target_uav'])] or source_path not in fmap[str(r['source_uav'])]:
            dropped.append({'line':line_no,'reason':'image_path_not_in_stage3_frame_manifest'}); continue
        fr=fmap[str(r['target_uav'])][target_path]; source_fr=fmap[str(r['source_uav'])][source_path]
        if fr != source_fr: dropped.append({'line':line_no,'reason':'paired_frame_index_mismatch','target_frame':fr,'source_frame':source_fr}); continue
        td=str(r['target_uav']); sd=str(r['source_uav']); tc=int(r['class'])
        target=resolve(stores[td],fr,tc,r['target_bbox_xyxy'])
        if target is None:
            dropped.append({'line':line_no,'reason':'target_feature_missing','frame':fr,'det':r['target_detector_index']}); continue
        cand=[]; missing=[]
        for c in r['candidates'][:KMAX]:
            x=resolve(stores[sd],source_fr,int(c['class']),c['bbox_xyxy'])
            if x is None: missing.append(int(c['detector_index']))
            else: cand.append(x)
        if missing:
            # Dropping a malformed cache row is an integrity failure, not a
            # silent negative; record it and exclude only that event.
            dropped.append({'line':line_no,'reason':'candidate_feature_missing','missing':missing,'frame':fr}); continue
        pos=[int(i) for i in r.get('same_id_candidate_indices',[]) if int(i)<len(cand)]
        # Keep all rows: zero candidates and zero positive rows are explicit
        # dustbin/no-match supervision under the episode contract.
        teacher=None; gid=r.get('target_train_gt_id')
        if include_teacher and gid is not None:
            past=[resolve(stores[sd],pf,pcls,pbox) for pf,pcls,pbox in prefix[sd].get(int(gid),[]) if pf < fr]
            past=[x for x in past if x is not None]
            if past: teacher=np.mean(np.stack(past),0).astype(np.float32)
        events.append({'a':target.astype(np.float32),'c':np.asarray(cand,np.float32).reshape(-1,D),
                       'pos':pos,'teacher':teacher,'pair':str(sid),'frame':fr,
                       'episode_id':r['episode_id'],'candidate_count':len(cand),
                       'supervision_valid':bool(r.get('supervision_valid',False)),
                       'has_teacher':teacher is not None})
        counts['events']+=1; counts['zero_candidate']+=int(len(cand)==0); counts['positive']+=int(bool(pos)); counts['teacher']+=int(teacher is not None)
    return events, {'pair':sid,'counts':dict(counts),'dropped':dropped}

def pad_batch(events, device):
    n=len(events); km=max(1,max((len(e['c']) for e in events),default=0))
    a=torch.from_numpy(np.stack([e['a'] for e in events])).to(device)
    c=torch.zeros(n,km,D,device=device); mask=torch.zeros(n,km,dtype=torch.bool,device=device); pos=torch.zeros(n,km,dtype=torch.bool,device=device)
    teacher=torch.zeros(n,D,device=device); tmask=torch.zeros(n,dtype=torch.bool,device=device)
    for i,e in enumerate(events):
        k=len(e['c'])
        if k: c[i,:k]=torch.from_numpy(e['c']).to(device); mask[i,:k]=True
        if e['pos']: pos[i,[j for j in e['pos'] if j<k]]=True
        if e['teacher'] is not None: teacher[i]=torch.from_numpy(e['teacher']).to(device); tmask[i]=True
    return a,c,mask,pos,teacher,tmask

class NativeSetResidual(nn.Module):
    """Candidate-conditioned residual with leave-one-out set context.

    The frozen baseline cosine is retained as the step-zero candidate score.
    The residual is map-free here because the native cache contains pooled
    stage-3 ROI vectors; the set operator is still pre-score and candidate
    relational rather than a scalar threshold/ranker.
    """
    def __init__(self,d=D,h=128,relational=True):
      super().__init__(); self.relational=relational
      self.res=nn.Sequential(nn.Linear(4*d,h),nn.GELU(),nn.LayerNorm(h),nn.Linear(h,d))
      nn.init.zeros_(self.res[-1].weight); nn.init.zeros_(self.res[-1].bias)
      self.gate=nn.Sequential(nn.Linear(4*d,h),nn.GELU(),nn.Linear(h,1))
      self.score=nn.Sequential(nn.Linear(4*d,h),nn.GELU(),nn.Linear(h,1))
      nn.init.zeros_(self.score[-1].weight); nn.init.zeros_(self.score[-1].bias)
      self.dust=nn.Sequential(nn.Linear(2*d+2,h),nn.GELU(),nn.Linear(h,1))
    def forward(self,a,c,mask):
      b,k,d=c.shape; valid=mask.float(); den=valid.sum(1,keepdim=True).clamp_min(1).unsqueeze(-1)
      total=(c*valid.unsqueeze(-1)).sum(1,keepdim=True)
      loo=(total-c*valid.unsqueeze(-1))/(den-valid.unsqueeze(-1)).clamp_min(1.)
      if not self.relational: loo=torch.zeros_like(loo)
      aa=a[:,None].expand(-1,k,-1)
      x=torch.cat([aa,c,c-loo,c-aa],-1)
      residual=self.res(x); gate=torch.sigmoid(self.gate(x)-2.0)
      z=F.normalize(c+gate*residual,dim=-1)
      base=(F.normalize(a,dim=-1)[:,None]*z).sum(-1)
      corr=self.score(x).squeeze(-1)
      logits=(base+0.25*corr).masked_fill(~mask,-1e9)
      setvec=(c*valid.unsqueeze(-1)).sum(1)/den.squeeze(1)
      maxsim=torch.where(mask,(F.normalize(a,dim=-1)[:,None]*F.normalize(c+1e-8,dim=-1)).sum(-1),torch.full_like(base,-1.)).amax(1,keepdim=True)
      count=(valid.sum(1,keepdim=True)/KMAX)
      dust=self.dust(torch.cat([F.normalize(a,dim=-1),F.normalize(setvec,dim=-1),maxsim,count],-1)).squeeze(-1)
      return logits,dust,z

def assignment_loss(logits,dust,pos,mask):
    all_logits=torch.cat([logits,dust[:,None]],1)
    losses=[]
    for i in range(all_logits.shape[0]):
      p=pos[i]&mask[i]
      if bool(p.any()): losses.append(torch.logsumexp(logits[i,p],0)-torch.logsumexp(all_logits[i],0))
      else: losses.append(all_logits[i,-1]-torch.logsumexp(all_logits[i],0))
    return -torch.stack(losses).mean()

def evaluate(model, events, device, batch_size=128):
    model.eval(); ranks=[]; base_ranks=[]; dust=[]; dust_y=[]; matched=0; total=0; mrr=[]; base_mrr=[]
    with torch.no_grad():
      for st in range(0,len(events),batch_size):
       es=events[st:st+batch_size]; a,c,m,p,t,tm=pad_batch(es,device); logits,d,z=model(a,c,m)
       base=(F.normalize(a,dim=-1)[:,None]*F.normalize(c+1e-8,dim=-1)).sum(-1).masked_fill(~m,-1e9)
       for i,e in enumerate(es):
        positive=p[i]&m[i]; has=bool(positive.any()); total+=1
        order=torch.argsort(logits[i],descending=True).tolist() if bool(m[i].any()) else []
        bo=torch.argsort(base[i],descending=True).tolist() if bool(m[i].any()) else []
        if has:
          rank=next((q for q,j in enumerate(order,1) if j in positive.nonzero().flatten().tolist()),None)
          brank=next((q for q,j in enumerate(bo,1) if j in positive.nonzero().flatten().tolist()),None)
          if rank is not None: ranks.append(rank);mrr.append(1/rank)
          if brank is not None: base_ranks.append(brank);base_mrr.append(1/brank)
          matched+=1
        pred_d=float(d[i].item() > (logits[i].max().item() if bool(m[i].any()) else -1e9))
        dust.append(pred_d); dust_y.append(float(not has))
    return {'events':total,'matched_events':matched,'student_recall1':float(np.mean(np.asarray(ranks)==1)) if ranks else None,'base_recall1':float(np.mean(np.asarray(base_ranks)==1)) if base_ranks else None,'student_mrr':float(np.mean(mrr)) if mrr else None,'base_mrr':float(np.mean(base_mrr)) if base_mrr else None,'dustbin_accuracy':float(np.mean(np.asarray(dust)==np.asarray(dust_y))) if dust else None,'dustbin_rate':float(np.mean(dust)) if dust else None,'dustbin_label_rate':float(np.mean(dust_y)) if dust_y else None}

def train_arm(train, evals, arm, seed, steps, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # The pooled control has identical parameter shapes but receives no
    # leave-one-out candidate-set context.
    relational = not arm.startswith('pooled_')
    teacher_on=arm.endswith('teacher_on')
    model=NativeSetResidual(relational=relational).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    hist=[]; n=len(train); bs=64
    for step in range(steps):
      ix=[(step*bs+j)%n for j in range(bs)]
      a,c,m,p,t,tm=pad_batch([train[j] for j in ix],device)
      logits,d,z=model(a,c,m); loss=assignment_loss(logits,d,p,m)
      if teacher_on:
       per=[]
       for i in range(bs):
        inds=(p[i]&m[i]).nonzero().flatten()
        if bool(tm[i]) and inds.numel(): per.append(1-(F.normalize(z[i,inds].mean(0),dim=-1)*F.normalize(t[i],dim=-1)).sum())
       if per: loss=loss+0.20*torch.stack(per).mean()
      opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.); opt.step()
      if step in (0,steps//4,steps//2,3*steps//4,steps-1): hist.append({'step':step+1,'loss':float(loss.detach()),'assign':float(assignment_loss(logits,d,p,m).detach()),'teacher_rows':int(tm.sum())})
    results={sid:evaluate(model,ev,device) for sid,ev in evals.items()}
    return model,hist,results

def main():
    steps=int(os.environ.get('P50_STEPS','2400')); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fit_ids=tuple(x for x in os.environ.get('P50_FIT_PAIRS',','.join(FIT)).split(',') if x)
    eval_ids=tuple(x for x in os.environ.get('P50_EVAL_PAIRS',','.join(EVAL)).split(',') if x)
    t0=time.time(); allpairs=list(FIT+EVAL); datasets={}; audits={}
    allpairs=list(fit_ids+eval_ids)
    for sid in allpairs:
      datasets[sid],audits[sid]=build_events(sid, include_teacher=(sid in fit_ids))
      print(json.dumps({'load_pair':sid,'events':len(datasets[sid]),'audit':audits[sid]['counts'],'dropped':len(audits[sid]['dropped'])}),flush=True)
    train=sum((datasets[s] for s in fit_ids),[]); evals={s:datasets[s] for s in eval_ids}
    arms={}
    # Same model, ordering, batch and update budget; only relational context or
    # teacher loss changes. Two seeds make pair-level direction less seed-bound.
    for arm in ('rel_on_teacher_off','rel_on_teacher_on','pooled_teacher_off','pooled_teacher_on'):
      arms[arm]=[]
      for seed in (7,17):
       print(json.dumps({'arm':arm,'seed':seed,'steps':steps,'device':str(device)}),flush=True)
       model,hist,res=train_arm(train,evals,arm,seed,steps,device)
       path=OUT/f'{arm}_seed{seed}.pt'; torch.save({'state_dict':model.state_dict(),'arm':arm,'seed':seed,'steps':steps},path)
       arms[arm].append({'seed':seed,'history':hist,'per_pair':res,'checkpoint':str(path),'checkpoint_sha256':sha(path)})
       del model; torch.cuda.empty_cache()
    payload={'status':'NATIVE_SET_RESIDUAL_FACTORIAL_COMPLETE','version':'2026-10-06-v1','device':str(device),'steps_per_arm':steps,'seeds':[7,17],'fit_pairs':list(fit_ids),'eval_pairs':list(eval_ids),'fit_events':len(train),'eval_events':{s:len(datasets[s]) for s in eval_ids},'pair_audit':audits,'arms':arms,'controls':{'native_features_frozen':True,'native_detector_preprocessing_preserved':True,'teacher_projection_trainable':False,'teacher_in_inference':False,'zero_candidate_rows_retained':True,'official_val_test_access':False,'model_selection_on_eval':False,'baseline':'frozen native 256-D stage3 cosine, evaluated on exact same rows'},'elapsed_seconds':time.time()-t0}
    (OUT/'NATIVE_FACTORIAL.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({'status':payload['status'],'elapsed_seconds':payload['elapsed_seconds'],'fit_events':len(train)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
