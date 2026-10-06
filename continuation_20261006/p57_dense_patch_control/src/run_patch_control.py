#!/usr/bin/env python3
"""Frozen DINO patch correspondence readout on the P23 calibration protocol."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]; OLD=Path('/home/chenhc/mdmot_research_20261002'); OUT=ROOT/'continuation_20261006/p57_dense_patch_control'; ASSET=OLD/'p4_diagnosis/assets'; WEIGHT=ASSET/'dinov2_vits14_pretrain.pth'
sys.path[:0]=[str(OLD/'p2/src'),str(OLD/'p4_diagnosis'),str(ROOT/'continuation_20261006/p55_context_residual/src')]
from data import split_assignments
import compare_crop32 as p4
from train_head_init import BaseHead, INIT

def crop(im, box):
 h,w=im.shape[:2];x1,y1,x2,y2=[float(x) for x in box]; b=[max(0,int(np.floor(x1))),max(0,int(np.floor(y1))),min(w,int(np.ceil(x2))),min(h,int(np.ceil(y2)))]
 if b[2]<=b[0] or b[3]<=b[1]: raise ValueError('invalid crop')
 return np.asarray(Image.fromarray(im[b[1]:b[3],b[0]:b[2]]).resize((224,224),Image.Resampling.BICUBIC),np.uint8)

def tokens(model, refs, boxes):
 images={r['image_ref']:r for r in refs['images']}; out=[]; batch=[]; cache={}
 mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 for i,(ref,box) in enumerate(zip(refs['_row_ref'],boxes)):
  ref=str(ref)
  if ref not in cache:
   with Image.open(images[ref]['image_path']) as h: cache[ref]=np.asarray(h.convert('RGB'))
  batch.append(crop(cache[ref],box))
  if len(batch)>=64 or i==len(boxes)-1:
   x=torch.from_numpy(np.stack(batch).transpose(0,3,1,2)).cuda().float()/255.
   with torch.inference_mode(): out.append(F.normalize(model.forward_features((x-mean)/std)['x_norm_patchtokens'],dim=-1).float().cpu())
   batch=[]
 return torch.cat(out).numpy().astype(np.float16)

def main():
 start=time.monotonic(); prop=torch.cuda.get_device_properties(0); model=torch.hub.load(str(ASSET/'dinov2'),'dinov2_vits14',source='local',pretrained=False);model.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True);model.eval().cuda(); head=BaseHead();head.load_state_dict(torch.load(INIT,map_location='cpu',weights_only=False)['head_state'],strict=True);head.eval(); frame=json.loads((OLD/'p4_diagnosis/crop32/FRAME_MANIFEST.json').read_text());p2=json.loads((OLD/'p2/cache/MANIFEST.json').read_text());inputs={r['sequence']:r for r in p2['sequences']};per={}
 for rec in frame['pairs']:
  if rec['role']!='calibration':continue
  pair=rec['pair'];row,features,refs,cnt=p4.align_pair(rec,OLD/'p4_diagnosis/crop32',inputs)
  with np.load(OLD/'p4_diagnosis/crop32'/f'{pair}.npz',allow_pickle=False) as a: box=a['bbox']; image_ref=a['image_ref']
  # Keep image metadata separate from row arrays; the operator sees only crops.
  image_rows=json.loads((OLD/'p4_diagnosis/crop32'/f'{pair}_images.json').read_text()); meta={'_row_ref':image_ref,'images':image_rows};patch=tokens(model,meta,box)
  with torch.inference_mode(): base=F.normalize(head.raw(torch.from_numpy(features['fpn256']),torch.from_numpy(features['dino384'])),dim=-1).numpy()
  ranks_base=[];ranks_patch=[]
  for i in range(len(patch)):
   if row['row_pid_offline'][i]<0:continue
   same=(row['row_view']!=row['row_view'][i])&(row['row_frame']==row['row_frame'][i])&(row['row_class']==row['row_class'][i]); c=np.flatnonzero(same&(row['row_pid_offline']>=0));pos=np.flatnonzero(row['row_pid_offline'][c]==row['row_pid_offline'][i])
   if not len(c) or not len(pos):continue
   bsim=base[i]@base[c].T; ranks_base.append(1+int(np.sum(bsim>bsim[pos[0]])))
   q=torch.from_numpy(patch[i].astype(np.float32)).cuda(); cc=torch.from_numpy(patch[c].astype(np.float32)).cuda(); sim=q@cc.transpose(1,2); a=sim.max(2).values.topk(min(16,sim.shape[1]),dim=1).values.mean(1); b=sim.max(1).values.topk(min(16,sim.shape[2]),dim=1).values.mean(1); score=((a+b)/2).detach().cpu().numpy();ranks_patch.append(1+int(np.sum(score>score[pos[0]])))
  per[pair]={'target_only':{'queries':len(ranks_base),'r1':float(np.mean(np.asarray(ranks_base)==1))},'patch_partial':{'queries':len(ranks_patch),'r1':float(np.mean(np.asarray(ranks_patch)==1))}}
 macro={m:float(np.mean([per[p][m]['r1'] for p in per])) for m in ('target_only','patch_partial')};wins=sum(per[p]['patch_partial']['r1']>per[p]['target_only']['r1'] for p in per);out={'status':'P57_PATCH_SIGNAL_COMPLETE','pairs':sorted(per),'per_pair':per,'macro_r1':macro,'delta':macro['patch_partial']-macro['target_only'],'pair_wins':wins,'operator':'symmetric top-16 patch maxima','gpu':{'name':prop.name,'memory':prop.total_memory},'official_val_test_read':False,'training_authorized':False,'elapsed_seconds':time.monotonic()-start};(OUT/'SIGNAL.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':out['status'],'macro_r1':macro,'delta':out['delta'],'pair_wins':wins}))

if __name__=='__main__':main()
