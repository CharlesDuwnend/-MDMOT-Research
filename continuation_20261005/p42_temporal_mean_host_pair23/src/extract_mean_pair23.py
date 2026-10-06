#!/usr/bin/env python3
import hashlib,json
from collections import defaultdict,deque
from pathlib import Path
import numpy as np
ROOT=Path('/home/chenhc/claude_try_MDMOT'); P42=ROOT/'continuation_20261005/p42_temporal_mean_host_pair23'; P26=ROOT/'continuation_20261005/p26_mia_baseline/runs/fit23/json/mia_baseline'; P2=Path('/home/chenhc/mdmot_research_20261002/p2/cache/inputs')
WINDOW=8;MAX_GAP=30

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b"" ):h.update(b)
 return h.hexdigest()
def iou(a,b):
 x1=max(float(a[0]),float(b[0])); y1=max(float(a[1]),float(b[1])); x2=min(float(a[2]),float(b[2])); y2=min(float(a[3]),float(b[3])); inter=max(0.,x2-x1)*max(0.,y2-y1); aa=max(0.,float(a[2])-float(a[0]))*max(0.,float(a[3])-float(a[1])); bb=max(0.,float(b[2])-float(b[0]))*max(0.,float(b[3])-float(b[1])); return inter/(aa+bb-inter+1e-12)
def unit(x): return x/(np.linalg.norm(x)+1e-12)
def map_rows(pred,cache):
 by=defaultdict(list)
 for j,f in enumerate(cache['frame']):by[int(f)].append(j)
 mapping=[]; scores=[]
 for f0 in range(700):
  rows=pred[f'frame={f0}']; pairs=[]
  for i,r in enumerate(rows):
   for j in by.get(f0+1,[]):
    q=iou(r[1:5],cache['bbox'][j])
    if q>=.30:pairs.append((q,i,j))
  usedi=set();usedj=set();
  for q,i,j in sorted(pairs,key=lambda x:(-x[0],x[1],x[2])):
   if i in usedi or j in usedj:continue
   usedi.add(i);usedj.add(j);mapping.append((f0,i,int(j)));scores.append(q)
 return {(f,i):j for f,i,j in mapping},scores
def main():
 out=P42/'runs/pair23';out.mkdir(exist_ok=False); receipts=[]; total=[]
 for view in (1,2):
  pred=json.loads((P26/f'23-{view}.json').read_text()); c={k:v for k,v in np.load(P2/f'23-{view}.npz',allow_pickle=False).items()}; mapping,ious=map_rows(pred,c); n=sum(len(pred[f'frame={f}']) for f in range(700)); emb=np.zeros((n,128),np.float32); valid=np.zeros(n,bool); memory={}; cursor=0; hist_rows=0; resets=0
  for f0 in range(700):
   for i,r in enumerate(pred[f'frame={f0}']):
    j=mapping.get((f0,i));
    if j is None or not bool(c['feature_present'][j]): cursor+=1;continue
    frame=f0+1; lid=int(c['local_id'][j]); cls=int(c['cls'][j]); z=unit(c['embedding'][j].astype(np.float32))
    if lid<0: value=z
    else:
     old=memory.get(lid)
     if old is not None and (cls!=old['class'] or frame-old['frame']>MAX_GAP): old=None;resets+=1
     if old is None:old={'frame':frame,'class':cls,'embeddings':deque(maxlen=WINDOW)};memory[lid]=old
     old['frame']=frame;old['class']=cls;old['embeddings'].append(z);value=unit(np.mean(np.stack(old['embeddings']),axis=0,dtype=np.float32));hist_rows+=int(len(old['embeddings'])>1)
    emb[cursor]=value;valid[cursor]=True;cursor+=1
  path=out/f'23-{view}.npz';np.savez_compressed(path,embedding=emb,valid=valid)
  rec={'view':view,'rows':n,'mapped_rows':len(mapping),'mapping_coverage':len(mapping)/n,'valid_embeddings':int(valid.sum()),'mean_iou':float(np.mean(ious)),'min_iou':float(np.min(ious)),'history_used_rows':hist_rows,'resets':resets,'output_sha256':sha(path),'p26_sha256':sha(P26/f'23-{view}.json'),'p2_sha256':sha(P2/f'23-{view}.npz')};receipts.append(rec);print(json.dumps(rec),flush=True)
 result={'status':'COMPLETE_P42_P39_MEAN_EXTRACTION','window':WINDOW,'max_gap':MAX_GAP,'sequences':receipts,'labels_read':False,'xml_read':False,'official_test_read':False};(out/'RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');(out/'extract.exit').write_text('0\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
