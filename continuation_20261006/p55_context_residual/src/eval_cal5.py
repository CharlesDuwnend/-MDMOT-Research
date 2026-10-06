#!/usr/bin/env python3
"""Extract legal calibration context features and replay both P55 arms."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OLD = Path("/home/chenhc/mdmot_research_20261002")
P55 = ROOT / "continuation_20261006/p55_context_residual"
sys.path[:0] = [str(OLD / "p2/src"), str(OLD / "p4_diagnosis")]
from data import split_assignments
import compare_crop32 as p4
from train_head_init import CVCRv2

ASSET = OLD / "p4_diagnosis/assets"
WEIGHT = ASSET / "dinov2_vits14_pretrain.pth"


def ctx(arr, box):
    h, w = arr.shape[:2]; x1,y1,x2,y2 = [float(x) for x in box]
    cx, cy = (x1+x2)/2, (y1+y2)/2; bw,bh=max(x2-x1,2.),max(y2-y1,2.)
    ex=[max(0,int(np.floor(cx-1.5*bw))),max(0,int(np.floor(cy-1.5*bh))),min(w,int(np.ceil(cx+1.5*bw))),min(h,int(np.ceil(cy+1.5*bh)))]
    a=arr[ex[1]:ex[3],ex[0]:ex[2]].copy(); tx1,ty1=max(0,int(np.floor(x1))-ex[0]),max(0,int(np.floor(y1))-ex[1]); tx2,ty2=min(a.shape[1],int(np.ceil(x2))-ex[0]),min(a.shape[0],int(np.ceil(y2))-ex[1])
    keep=np.ones(a.shape[:2],bool); keep[ty1:ty2,tx1:tx2]=False; fill=np.median(a[keep],axis=0) if keep.any() else np.median(a.reshape(-1,3),axis=0); a[ty1:ty2,tx1:tx2]=np.asarray(fill,np.uint8)
    return np.asarray(Image.fromarray(a).resize((224,224),Image.Resampling.BICUBIC),np.uint8)


def context_features(pair, record, row):
    crop_rows=json.loads((OLD/'p4_diagnosis/crop32'/f'{pair}_images.json').read_text())
    images={r['image_ref']:r for r in crop_rows}; cache=OLD/'p4_diagnosis/crop32'
    out=np.zeros((len(row['row_key']),384),np.float32); model=torch.hub.load(str(ASSET/'dinov2'),'dinov2_vits14',source='local',pretrained=False); model.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True); model.eval().cuda()
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]; std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    arrs=[]; idxs=[]; current_ref=None; image=None
    for i, (ref, box) in enumerate(zip(row['image_ref'], row['_bbox'])):
        ref=str(ref)
        if ref != current_ref:
            with Image.open(images[ref]['image_path']) as h: image=np.asarray(h.convert('RGB'))
            current_ref=ref
        arrs.append(ctx(image,box)); idxs.append(i)
        if len(arrs)>=64 or i==len(row['row_key'])-1:
            x=torch.from_numpy(np.stack(arrs).transpose(0,3,1,2)).cuda().float()/255.
            with torch.inference_mode(): out[np.asarray(idxs)]=F.normalize(model.forward_features((x-mean)/std)['x_norm_clstoken'],dim=-1).float().cpu().numpy()
            arrs=[];idxs=[]
    return out


def score_model(model, enabled, target, context, fpn, labels, row):
    per=[]
    model.eval()
    with torch.inference_mode():
        for view in (0,1):
            q=np.flatnonzero(row['row_view']==view); c=np.flatnonzero(row['row_view']==1-view)
            tq=torch.from_numpy(target[q]).cuda(); cq=torch.from_numpy(context[q]).cuda(); fq=torch.from_numpy(fpn[q]).cuda(); yq=labels[q]
            zq,dq=model(tq,cq,fq,context_enabled=enabled)
            for i in range(len(q)):
                if yq[i] < 0: continue
                same = ((row['row_class'][c] == row['row_class'][q[i]]) &
                        (row['row_frame'][c] == row['row_frame'][q[i]]))
                ci = c[same]
                if not len(ci): continue
                tc=torch.from_numpy(target[ci]).cuda(); cc=torch.from_numpy(context[ci]).cuda(); fc=torch.from_numpy(fpn[ci]).cuda(); yc=labels[ci]
                zc,_=model(tc,cc,fc,context_enabled=enabled); valid=yc>=0
                if not valid.any(): continue
                sims=(zq[i:i+1]@zc[valid].T/.07).flatten(); pos=np.flatnonzero(yc[valid]==yq[i])
                if len(pos): per.append(1+int((sims>sims[pos[0]]).sum()))
    return {'queries':len(per),'r1':float(np.mean(np.asarray(per)==1)) if per else None,'mrr':float(np.mean(1/np.asarray(per))) if per else None}


def main():
    frame=json.loads((OLD/'p4_diagnosis/crop32/FRAME_MANIFEST.json').read_text()); split=split_assignments(); p2=json.loads((OLD/'p2/cache/MANIFEST.json').read_text()); inputs={r['sequence']:r for r in p2['sequences']}; models={}
    for name,enabled in [('target_only',False),('context_residual',True)]:
        ck=torch.load(P55/'runs_v4'/name/'fixed_step_001200.pt',map_location='cuda',weights_only=False); m=CVCRv2().cuda();m.load_state_dict(ck['state_dict'],strict=True);models[name]=(m,enabled)
    per={};
    for rec in frame['pairs']:
        if rec['role']!='calibration': continue
        pair=rec['pair']; row,features,refs,cnt=p4.align_pair(rec,OLD/'p4_diagnosis/crop32',inputs)
        with np.load(OLD/'p4_diagnosis/crop32'/f'{pair}.npz',allow_pickle=False) as archive:
            row['_bbox']=archive['bbox']; row['image_ref']=archive['image_ref']
        context=context_features(pair,rec,row); target=features['dino384']; fpn=features['fpn256']; labels=row['row_pid_offline']; per[pair]={}
        for name,(m,en) in models.items(): per[pair][name]=score_model(m,en,target,context,fpn,labels,row)
    macro={n:float(np.mean([per[p][n]['r1'] for p in per])) for n in models}; wins=sum(per[p]['context_residual']['r1']>per[p]['target_only']['r1'] for p in per)
    out={'status':'P55_CAL5_REPLAY_COMPLETE','pairs':sorted(per),'per_pair':per,'macro_r1':macro,'delta_macro_r1':macro['context_residual']-macro['target_only'],'pair_wins':wins,'official_val_test_read':False,'protocol':'P4 calibration five-pair positive-present readout; no dev/test'}
    (P55/'CAL5_GATE_V4.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'status':out['status'],'macro_r1':macro,'delta':out['delta_macro_r1'],'pair_wins':wins}))

if __name__=='__main__': main()
