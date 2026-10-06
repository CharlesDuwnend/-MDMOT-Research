#!/usr/bin/env python3
import json, math, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

ROOT=Path('/home/chenhc/claude_try_MDMOT')
P37=ROOT/'continuation_20261005/p42_temporal_mean_host_pair23'; P26=ROOT/'continuation_20261005/p26_mia_baseline/runs/fit23/json/mia_baseline'; OUT=P37/'runs/host_pair23'; EMB=P37/'runs/pair23'
OUT.mkdir(parents=True,exist_ok=False)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def box(r):return np.asarray(r[1:5],np.float64)
def center(r):
 b=box(r); return np.array([(b[0]+b[2])*.5,(b[1]+b[3])*.5],np.float64)
def diag(r):
 b=box(r); return max(float(np.hypot(b[2]-b[0],b[3]-b[1])),1.)
def project(H,p):return cv2.perspectiveTransform(np.asarray(p,np.float32).reshape(-1,1,2),H).reshape(-1,2)[0].astype(np.float64)
def clone(d):return {k:[list(x) for x in v] for k,v in d.items()}
def init_H(a,b):
 A={int(round(float(r[0]))):center(r) for r in a}; B={int(round(float(r[0]))):center(r) for r in b}; keys=sorted(set(A)&set(B))
 if len(keys)>=4:
  H,mask=cv2.findHomography(np.asarray([A[k] for k in keys],np.float32),np.asarray([B[k] for k in keys],np.float32),cv2.RANSAC,5.)
  if H is not None and np.isfinite(H).all():return H.astype(np.float64),len(keys)
 return np.eye(3),len(keys)
def assign(a,b,H,emb_a,va,emb_b,vb,appearance):
 if not a or not b:return []
 C=np.full((len(a),len(b)),1e6,np.float64); D=np.full_like(C,1e6); S=np.full_like(C,np.nan)
 for i,ra in enumerate(a):
  p=project(H,center(ra))
  for j,rb in enumerate(b):
   geom=float(np.linalg.norm(p-center(rb))/diag(rb)); D[i,j]=geom
   cos=np.nan
   if va[i] and vb[j]: cos=float(np.dot(emb_a[i],emb_b[j]))
   S[i,j]=cos
   if geom<=2.0:
    C[i,j]=geom/2. if not (appearance and np.isfinite(cos)) else .7*geom/2.+.3*(1.-cos)/2.
 ri,cj=linear_sum_assignment(C); out=[]
 for i,j in zip(ri,cj):
  if C[i,j]>=1e5:continue
  out.append({'ai':int(i),'bj':int(j),'cost':float(C[i,j]),'geom':float(D[i,j]),'cos':None if not np.isfinite(S[i,j]) else float(S[i,j])})
 return out
def apply(pred2, maps):
 out=clone(pred2); used_safe=skips=0
 for f,links in maps.items():
  rows=out.get(f'frame={f}',[]); used={int(round(float(r[0]))) for r in rows}
  for link in sorted(links,key=lambda x:(x['cost'],x['ai'],x['bj'])):
   j=link['bj']; target=int(link['target_id'])
   if j>=len(rows):continue
   old=int(round(float(rows[j][0])))
   if target==old or target not in used:
    rows[j][0]=float(target); used.discard(old); used.add(target); used_safe+=int(target!=old)
   else:skips+=1
  out[f'frame={f}']=rows
 return out,{'safe_relabels':used_safe,'collision_skips':skips}
def dupes(d):return sum(len(rs)!=len(set(int(round(float(r[0]))) for r in rs)) for rs in d.values())
def main():
 p1=json.loads((P26/'23-1.json').read_text());p2=json.loads((P26/'23-2.json').read_text())
 z1=np.load(EMB/'23-1.npz');z2=np.load(EMB/'23-2.npz');e1,v1=z1['embedding'],z1['valid'];e2,v2=z2['embedding'],z2['valid']
 cur1=cur2=0; maps={'mean8':{},'geometry':{}}; H,n0=init_H(p1['frame=0'],p2['frame=0']); assignment_changes=0; rows=[]; links_tot={'mean8':0,'geometry':0}
 for f in range(700):
  a=p1[f'frame={f}'];b=p2[f'frame={f}']; aa=e1[cur1:cur1+len(a)];bb=e2[cur2:cur2+len(b)]; va=v1[cur1:cur1+len(a)];vb=v2[cur2:cur2+len(b)]
  # frame 0 is the fixed GT-initialized host; no edit is allowed there.
  if f==0: lp=[];lg=[]
  else:
   lp=assign(a,b,H,aa,va,bb,vb,True); lg=assign(a,b,H,aa,va,bb,vb,False)
   pset={(x['ai'],x['bj']) for x in lp};gset={(x['ai'],x['bj']) for x in lg}; assignment_changes += len(pset.symmetric_difference(gset))
  maps['mean8'][f]=[dict(x,target_id=int(round(float(a[x['ai']][0])))) for x in lp]
  maps['geometry'][f]=[dict(x,target_id=int(round(float(a[x['ai']][0])))) for x in lg]
  links_tot['mean8']+=len(lp);links_tot['geometry']+=len(lg)
  # Shared transport control: homography is updated from geometry only, so appearance cannot move the motion model.
  if f>0 and len(lg)>=4:
   src=np.asarray([center(a[x['ai']]) for x in lg],np.float32);dst=np.asarray([center(b[x['bj']]) for x in lg],np.float32);h,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.)
   if h is not None and np.isfinite(h).all():H=h.astype(np.float64)
  if f in (0,1,100,300,699):rows.append({'frame':f,'view1_rows':len(a),'view2_rows':len(b),'mean8_links':len(lp),'geometry_links':len(lg)})
  cur1+=len(a);cur2+=len(b)
 out_mean,app_mean=apply(p2,maps['mean8']);out_geo,app_geo=apply(p2,maps['geometry'])
 for name,d in [('P42_mean8',out_mean),('P42_geometry',out_geo)]:
  for v,out in [(1,p1),(2,d)]: (OUT/f'{name}-23-{v}.json').write_text(json.dumps(out,indent=2)+'\n')
 result={'status':'COMPLETE_P37_PAIR23_HOST_DIAGNOSTIC','method':'P39 fixed causal temporal mean association','frames':700,'initial_H_shared_ids':n0,'links':links_tot,'assignment_symmetric_difference_events':assignment_changes,'frames_sampled':rows,'application':{'mean8':app_mean,'geometry':app_geo},'duplicate_frame_ids':{'mean8':dupes(out_mean),'geometry':dupes(out_geo)},'descriptor_coverage':{'view1':float(v1.mean()),'view2':float(v2.mean())},'labels_read':False,'xml_read':False,'official_test_read':False,'inputs_sha256':{'p26_view1':sha(P26/'23-1.json'),'p26_view2':sha(P26/'23-2.json'),'mean8_view1':sha(EMB/'23-1.npz'),'mean8_view2':sha(EMB/'23-2.npz')}}
 (OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n'); (OUT/'associate.exit').write_text('0\n'); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
