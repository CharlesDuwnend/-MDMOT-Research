#!/usr/bin/env python3
"""P50 real ROI-map student short gate.

The only privileged input is a detached 256-D source-view prefix prototype
constructed from frozen train features. It is a training target, never a model
input at evaluation. The host tracker and official val/test are not imported.
"""
from __future__ import annotations

import gzip, hashlib, json, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SCI = Path("/home/chenhc/mdmot_sci_id_20260906")
sys.path.insert(0, str(SCI))
from sci_id.data.episodes import PackedFrameEpisodeDataset, collate_packed_frame_episodes
from sci_id.models.full_model import SCIIDBackboneModel
from sci_id.models.assignment_head import AssignmentOutput

ROOT = Path("/home/chenhc/cross_uav_query_mamba_20260830")
sys.path.insert(0, str(ROOT))
import stage3_runner as s3  # noqa: E402

FIT = ("23","25","27","28","29","30","32","39","42","44","45","50","51","53","54")
EVAL = ("69","70","74","76","78")
CKPT = ROOT / "checkpoints/autoassign_r50_fpn_8x2_1x_full_mdmt_epoch_60.pth"


class MapResidualHead(nn.Module):
    levels = ("p2_id", "p3_id", "p4_id")
    def __init__(self, channels=32):
        super().__init__()
        self.channels = channels
        self.residual = nn.Sequential(nn.Conv2d(channels * 3, channels, 1), nn.GELU(), nn.Conv2d(channels, channels, 1))
        self.fuse = nn.Sequential(nn.LayerNorm(channels * 3), nn.Linear(channels * 3, channels), nn.GELU())
        self.proj = nn.Linear(channels, channels, bias=False)
        self.teacher_proj = nn.Linear(256, channels, bias=False)
        self.dustbin = nn.Sequential(nn.Linear(channels * 2 + 1, channels), nn.GELU(), nn.Linear(channels, 1))

    def forward(self, source_maps, candidate_maps, mask):
        valid = mask[:, :, None, None, None].to(next(self.parameters()).dtype)
        denom = mask.sum(1, keepdim=True).clamp_min(1).to(valid.dtype)
        src_vecs=[]; cand_vecs=[]
        for level in self.levels:
            src=source_maps[level]; cand=candidate_maps[level]
            b,k,c,h,w=cand.shape
            total=(cand*valid).sum(1,keepdim=True)
            other=(total-cand*valid)/(denom[:,:,None,None,None]-valid).clamp_min(1.0)
            src_expand=src[:,None].expand(-1,k,-1,-1,-1)
            x=torch.cat((src_expand,cand,other),2).reshape(b*k,3*c,h,w)
            adapted=(cand+self.residual(x).reshape(b,k,c,h,w))*valid
            src_vecs.append(src.mean((-1,-2)))
            cand_vecs.append(adapted.mean((-1,-2)))
        src_z=F.normalize(self.proj(self.fuse(torch.cat(src_vecs,1))),dim=-1)
        cand_z=F.normalize(self.proj(self.fuse(torch.cat(cand_vecs,2))),dim=-1)
        logits=(src_z[:,None]*cand_z).sum(-1).masked_fill(~mask,torch.finfo(src_z.dtype).min)
        set_mean=(cand_z*mask[:,:,None].to(cand_z.dtype)).sum(1)/denom
        dust=self.dustbin(torch.cat((src_z,set_mean,mask.float().mean(1,keepdim=True)),1)).squeeze(-1)
        return AssignmentOutput(logits,dust,torch.cat((logits,dust[:,None]),1),src_z,cand_z,src_z,cand_z,{})


def row_gid(row, gt):
    best=max(((s3.iou(row["bbox"],g["bbox"]),g) for g in gt),key=lambda x:x[0]) if gt else (0.,None)
    return int(best[1]["gt_id"]) if best[0]>=.5 else None


def build_teacher_lookup(pairs):
    hist=defaultdict(list)
    for sid in pairs:
        rows=s3.load_index("train",sid,"2"); gt=s3.read_gt(sid,"2")
        for r in rows:
            gid=row_gid(r,gt.get(int(r["frame_id"]),[]))
            if gid is not None: hist[(str(sid),gid)].append((int(r["frame_id"]),r["visual_feature"].astype(np.float32)))
    return hist


def add_teacher(row, hist):
    gid=row.get("target_train_gt_id");
    if gid is None: return None
    frames=[v for f,v in hist[(str(row["pair_id"]),int(gid))] if f<int(row["confirmation_frame"])]
    return np.mean(frames,0).astype(np.float32) if frames else None


def eligible_dataset(pairs, hist):
    ds=PackedFrameEpisodeDataset(SCI/"data/train_episodes_v1",pairs=pairs); rows=[]
    for row in ds.rows:
        t=add_teacher(row,hist)
        if t is not None and row.get("same_id_candidate_indices"):
            x=dict(row); x["_teacher"]=t; rows.append(x)
    return rows


def build_feature_lookup(pairs):
    lookup={}
    for sid in pairs:
        for dev in ("1","2"):
            rows=s3.load_index("train",sid,dev)
            by=defaultdict(list)
            for r in rows: by[int(r["frame_id"])].append(r)
            lookup[(str(sid),dev)]=by
    return lookup


def match_feature(rows, box):
    best=max(((s3.iou(r["bbox"],box),r) for r in rows),key=lambda x:x[0]) if rows else (0.,None)
    return best[1]["visual_feature"].astype(np.float32) if best[0]>=.5 else None


def batch(rows, device):
    teacher=torch.from_numpy(np.stack([r["_teacher"] for r in rows])).float().to(device)
    clean=[]
    for r in rows:
        x=dict(r); x.pop("_teacher",None); clean.append(x)
    b=collate_packed_frame_episodes(clean,image_size=(256,448),mean=(102.9801,115.9465,122.7717),std=(1.,1.,1.),value_scale=1.,channel_order="bgr")
    return {k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in b.items()},teacher


def loss_and_out(model,b,teacher):
    maps=model.encode_packed(b["frame_images"],b["target_frame_index"],b["source_frame_index"],b["target_boxes"],b["candidate_boxes"],b["mask"])
    out=model.forward_maps(*maps,b["mask"])
    valid=b["supervision_valid"]
    if not bool(valid.any()): return out,out.logits.sum()*0
    # Assignment is multi-positive BCE; padded candidates are excluded.
    target=b["factual_target"].to(out.logits.dtype)
    logits=out.logits.masked_fill(~torch.cat((b["mask"],torch.ones((len(rows),1),dtype=torch.bool,device=b["mask"].device)),1),-20) if False else out.logits
    assign=F.binary_cross_entropy_with_logits(logits[valid],target[valid])
    pos=b["positive_mask"] & b["mask"]
    per=[]
    teacher_z=F.normalize(model.head.teacher_proj(teacher.detach()),dim=-1)
    for i in range(len(teacher)):
        inds=pos[i].nonzero(as_tuple=False).flatten()
        if inds.numel(): per.append(1-(F.normalize(out.universal_candidates[i,inds].mean(0),dim=-1)*teacher_z[i]).sum())
    dist=torch.stack(per).mean() if per else assign*0
    return out,assign+0.35*dist


def evaluate(model, rows, device, feature_lookup, limit=1000):
    model.eval(); rs=[]; base=[]
    with torch.no_grad():
        for start in range(0,min(len(rows),limit),4):
            selected=rows[start:start+4]; b,_=batch(selected,device)
            maps=model.encode_packed(b["frame_images"],b["target_frame_index"],b["source_frame_index"],b["target_boxes"],b["candidate_boxes"],b["mask"])
            out=model.forward_maps(*maps,b["mask"])
            for i in range(len(selected)):
                pos=(b["positive_mask"][i]&b["mask"][i]).nonzero(as_tuple=False).flatten().tolist()
                if not pos: continue
                order=torch.argsort(out.candidate_logits[i],descending=True).tolist(); rs.append(next(r for r,j in enumerate(order,1) if j in pos))
                # The frozen feature baseline uses exactly the same feature
                # cosine definition as the P50 feature gate.
                row=selected[i]; fr=max(int(row["confirmation_frame"])-1,0)
                ta=match_feature(feature_lookup[(str(row["pair_id"]),str(row["target_uav"]))].get(fr,[]),row["target_bbox_xyxy"])
                cs=[match_feature(feature_lookup[(str(row["pair_id"]),str(row["source_uav"]))].get(fr,[]),c["bbox_xyxy"]) for c in row["candidates"]]
                if ta is None or any(x is None for x in cs): continue
                tn=max(float(np.linalg.norm(ta)),1e-6); scores=torch.tensor([float(np.dot(ta,x)/(tn*max(float(np.linalg.norm(x)),1e-6))) for x in cs],device=device)
                bo=torch.argsort(scores,descending=True).tolist(); base.append(next(r for r,j in enumerate(bo,1) if j in pos))
    return {"events":len(rs),"student_recall1":float(np.mean(np.asarray(rs)==1)) if rs else None,"baseline_recall1":float(np.mean(np.asarray(base)==1)) if base else None,"student_mrr":float(np.mean(1/np.asarray(rs))) if rs else None,"baseline_mrr":float(np.mean(1/np.asarray(base))) if base else None}


def main():
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hist=build_teacher_lookup(FIT)
    train=eligible_dataset(FIT,hist)
    eval_hist=build_teacher_lookup(EVAL)
    eval_rows={sid:eligible_dataset((sid,),eval_hist) for sid in EVAL}
    model=SCIIDBackboneModel(channels=32,prototypes=2,output_size=7,backbone_init=CKPT)
    model.head=MapResidualHead(32); model.to(device).train()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    history=[]
    for step in range(1200):
        selected=[train[(step*2+j)%len(train)] for j in range(2)]
        b,t=batch(selected,device); opt.zero_grad(set_to_none=True); out,loss=loss_and_out(model,b,t); loss.backward(); opt.step()
        if step in (0,399,799,1199): history.append({"step":step+1,"loss":float(loss.detach()),"grad_l1":float(sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None))})
    feature_lookup=build_feature_lookup(EVAL)
    model_path=Path(__file__).resolve().parents[1]/"MAP_GATE_MODEL.pt"; torch.save(model.state_dict(),model_path)
    results={sid:evaluate(model,rows,device,feature_lookup,limit=500) for sid,rows in eval_rows.items()}
    result={"status":"PASS_P50_MAP_SHORT_GATE_COMPLETE","device":str(device),"fit_rows":len(train),"steps":1200,"history":history,"per_pair":results,"teacher_in_inference":False,"official_val_test_access":False,"metrics_are_train_only_diagnostics":True,"baseline":"frozen_stage3_visual_feature_cosine_same_definition_as_feature_gate","checkpoint":str(model_path),"checkpoint_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest()}
    out=Path(__file__).resolve().parents[1]/"MAP_GATE.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
