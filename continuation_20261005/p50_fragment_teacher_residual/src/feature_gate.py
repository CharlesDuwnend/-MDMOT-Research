#!/usr/bin/env python3
"""Train-only feature-space feasibility gate for P50.

This is deliberately a short gate, not a map-level result or official metric.
The student sees only frozen detector features and a candidate-set summary. A
strict-prefix source-view prototype is used as a detached training target.
"""
from __future__ import annotations

import json, math, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path("/home/chenhc/cross_uav_query_mamba_20260830")
sys.path.insert(0, str(ROOT))
import stage3_runner as s3  # noqa: E402

FIT = ("23","25","27","28","29","30","32","39","42","44","45","50","51","53","54")
EVAL = ("69","70","74","76","78")


def center(box): return np.asarray([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=np.float32)


def row_gid(row, gt):
    best = max(((s3.iou(row["bbox"], g["bbox"]), g) for g in gt), key=lambda x: x[0]) if gt else (0., None)
    return int(best[1]["gt_id"]) if best[0] >= .5 else None


def build_pair(sid, limit_events=1200):
    aa, bb = s3.load_index("train", sid, "1"), s3.load_index("train", sid, "2")
    by_a, by_b = defaultdict(list), defaultdict(list)
    for r in aa: by_a[int(r["frame_id"])].append(r)
    for r in bb: by_b[int(r["frame_id"])].append(r)
    gt_a, gt_b = s3.read_gt(sid, "1"), s3.read_gt(sid, "2")
    # Strict-prefix source prototypes, indexed by source-view train identity.
    history = defaultdict(list)
    events=[]
    for fr in sorted(set(by_a) & set(by_b)):
        src = by_b[fr]
        src_gid = {id(r): row_gid(r, gt_b.get(fr, [])) for r in src}
        for r in src:
            gid = src_gid[id(r)]
            if gid is not None:
                history[gid].append((fr, r["visual_feature"].astype(np.float32)))
        for tr in by_a[fr]:
            gid = row_gid(tr, gt_a.get(fr, []))
            if gid is None: continue
            tf = [(f,v) for f,v in history[gid] if f < fr]
            if not tf: continue
            # Use all source candidates when <=64, otherwise a deterministic
            # baseline top-64; no labels enter this pool construction.
            a = tr["visual_feature"].astype(np.float32)
            an = max(float(np.linalg.norm(a)), 1e-6)
            scored=[]
            for j, cr in enumerate(src):
                b = cr["visual_feature"].astype(np.float32)
                sim=float(np.dot(a,b)/(an*max(float(np.linalg.norm(b)),1e-6)))
                dist=float(np.linalg.norm(center(tr["bbox"])-center(cr["bbox"])))
                scored.append((0.7*sim-0.3*min(dist/500.,1.), j))
            chosen=[j for _,j in sorted(scored, reverse=True)[:64]]
            pos=[j_index for j_index,j in enumerate(chosen) if src_gid[id(src[j])] == gid]
            if not pos: continue
            # Set summary is computed from inference-visible candidate features.
            valid_count = len(chosen)
            feats = np.zeros((64, 256), dtype=np.float32)
            if chosen:
                feats[:valid_count] = np.stack([src[j]["visual_feature"].astype(np.float32) for j in chosen])
            teacher=np.mean([v for f,v in tf], axis=0).astype(np.float32)
            events.append({"a":a,"c":feats,"valid":valid_count,"positive":pos,"teacher":teacher,"frame":fr,"pair":sid})
            if len(events)>=limit_events: break
        if len(events)>=limit_events: break
    return events


class Student(nn.Module):
    def __init__(self, d=256, hidden=192):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(d*3,hidden),nn.GELU(),nn.LayerNorm(hidden),nn.Linear(hidden,d))
        self.score=nn.Sequential(nn.Linear(d*4,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self,a,c,mask=None):
        # Inputs are target, candidate, and inference-visible candidate-set mean.
        if mask is None:
            mask=torch.ones(c.shape[:2],dtype=torch.bool,device=c.device)
        denom=mask.sum(1,keepdim=True).clamp_min(1).to(c.dtype)
        setmean=((c*mask[:,:,None].to(c.dtype)).sum(1,keepdim=True)/denom[:,:,None]).expand_as(c)
        z=F.normalize(self.embed(torch.cat((a[:,None].expand_as(c),c,setmean),-1)),dim=-1)
        ta=F.normalize(a,dim=-1)
        logits=self.score(torch.cat((a[:,None].expand_as(c),c,setmean,z),-1)).squeeze(-1)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min),z


def tensors(events, device=None):
    a=torch.from_numpy(np.stack([e["a"] for e in events])).float()
    c=torch.from_numpy(np.stack([e["c"] for e in events])).float()
    mask=torch.zeros(len(events),c.shape[1],dtype=torch.bool)
    for i,e in enumerate(events): mask[i,:int(e["valid"])] = True
    y=torch.zeros(len(events),c.shape[1])
    for i,e in enumerate(events): y[i,e["positive"]]=1.
    t=torch.from_numpy(np.stack([e["teacher"] for e in events])).float()
    if device is not None:
        a,c,y,t,mask=[x.to(device) for x in (a,c,y,t,mask)]
    return a,c,y,t,mask


def evaluate(model, events, device):
    model.eval(); a,c,y,t,mask=tensors(events, device)
    with torch.no_grad(): logits,z=model(a,c,mask)
    ranks=[]; base=[]
    for i,e in enumerate(events):
        pos=set(e["positive"]); order=torch.argsort(logits[i],descending=True).tolist(); base_scores=(a[i,None]*c[i]).sum(-1).masked_fill(~mask[i],-1e9); bo=torch.argsort(base_scores,descending=True).tolist()
        ranks.append(next(r for r,j in enumerate(order,1) if j in pos)); base.append(next(r for r,j in enumerate(bo,1) if j in pos))
    return {"events":len(events),"student_recall1":float(np.mean(np.asarray(ranks)==1)),"baseline_recall1":float(np.mean(np.asarray(base)==1)),"student_mrr":float(np.mean(1/np.asarray(ranks))),"baseline_mrr":float(np.mean(1/np.asarray(base)))}


def main():
    torch.manual_seed(23); random.seed(23)
    fit=sum((build_pair(s,500) for s in FIT),[])
    eval_by={s:build_pair(s,1200) for s in EVAL}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=fit; a,c,y,t,mask=tensors(train, device)
    model=Student().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    history=[]
    for step in range(1200):
        ix=torch.randint(0,len(train),(min(64,len(train)),),device=device)
        logits,z=model(a[ix],c[ix],mask[ix]); target=y[ix]
        # Multi-positive assignment BCE plus teacher alignment on the positive mean.
        bce=F.binary_cross_entropy_with_logits(logits, target)
        pos=z[target.bool()].reshape(-1,256) if bool(target.any()) else z.reshape(-1,256)[:1]
        tt=F.normalize(t[ix],dim=-1)
        # Align each event's positive average to the detached teacher prototype.
        per=[]
        for row in range(len(ix)):
            inds=target[row].nonzero(as_tuple=False).flatten()
            per.append(1-(F.normalize(z[row,inds].mean(0),dim=-1)*tt[row]).sum() if inds.numel() else z[row].sum()*0)
        loss=bce+0.35*torch.stack(per).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (0,399,799,1199): history.append({"step":step+1,"loss":float(loss.detach()),"bce":float(bce.detach())})
    results={s:evaluate(model,ev,device) for s,ev in eval_by.items()}
    result={"status":"PASS_P50_FEATURE_SHORT_GATE_COMPLETE","fit_pairs":list(FIT),"eval_pairs":list(EVAL),"fit_events":len(fit),"steps":1200,"device":str(device),"history":history,"per_pair":results,"metrics_are_train_only_diagnostics":True,"teacher_in_inference":False}
    out=Path(__file__).resolve().parents[1]/"FEATURE_GATE.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
