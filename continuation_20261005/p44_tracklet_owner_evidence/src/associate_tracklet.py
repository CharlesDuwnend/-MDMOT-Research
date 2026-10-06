#!/usr/bin/env python3
import json, math, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
ROOT=Path('/home/chenhc/claude_try_MDMOT');P44=ROOT/'continuation_20261005/p44_tracklet_owner_evidence';P26=ROOT/'continuation_20261005/p26_mia_baseline/runs/fit23/json/mia_baseline';EMB=Path('/home/chenhc/claude_try_MDMOT/continuation_20261005/p37_p14_host_transfer/runs/pair23');OUT=P44/'runs/host_pair23'
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def box(r):return np.asarray(r[1:5],np.float64)
def center(r):
 b=box(r);return np.array([(b[0]+b[2])*.5,(b[1]+b[3])*.5],np.float64)
def diag(r):
 b=box(r);return max(float(np.hypot(b[2]-b[0],b[3]-b[1])),1.)
def project(H,p):return cv2.perspectiveTransform(np.asarray(p,np.float32).reshape(-1,1,2),H).reshape(-1,2)[0].astype(np.float64)
def init_H(a,b):
 A={int(round(float(r[0]))):center(r) for r in a};B={int(round(float(r[0]))):center(r) for r in b};keys=sorted(set(A)&set(B))
 if len(keys)>=4:
  H,mask=cv2.findHomography(np.asarray([A[k] for k in keys],np.float32),np.asarray([B[k] for k in keys],np.float32),cv2.RANSAC,5.)
  if H is not None and np.isfinite(H).all():return H.astype(np.float64),len(keys)
 return np.eye(3),len(keys)
def assign(a,b,H,ea,va,eb,vb,appearance):
 if not a or not b:return []
 C=np.full((len(a),len(b)),1e6,np.float64);D=np.full_like(C,1e6);S=np.full_like(C,np.nan)
 for i,ra in enumerate(a):
  pp=project(H,center(ra))
  for j,rb in enumerate(b):
   geom=float(np.linalg.norm(pp-center(rb))/diag(rb));D[i,j]=geom
   cos=float(np.dot(ea[i],eb[j])) if va[i] and vb[j] else np.nan;S[i,j]=cos
   if geom<=2.:C[i,j]=geom/2. if not(appearance and np.isfinite(cos)) else .7*geom/2.+.3*(1.-cos)/2.
 ri,cj=linear_sum_assignment(C);out=[]
 for i,j in zip(ri,cj):
  if C[i,j]<1e5:out.append({'ai':int(i),'bj':int(j),'cost':float(C[i,j]),'geom':float(D[i,j]),'cos':None if not np.isfinite(S[i,j]) else float(S[i,j])})
 return out
def clone(d):return {k:[list(x) for x in v] for k,v in d.items()}
def make_segments(pred, gap=30):
 seg={};state={}
 for f in range(700):
  for j,r in enumerate(pred[f'frame={f}']):
   native=int(round(float(r[0])));old=state.get(native);nseg=(old[1]+1) if old is not None and f-old[0]>gap else (old[1] if old is not None else 0);seg[(f,j)]=(native,nseg);state[native]=(f,nseg)
 return seg
def apply(pred, mapping, segments):
 out=clone(pred);used_stats=defaultdict(int)
 for f in range(700):
  rows=out[f'frame={f}'];used={int(round(float(r[0]))) for r in rows}
  for j,r in enumerate(rows):
   if f==0:continue
   key=segments[(f,j)];desired=mapping.get(key)
   if desired is None:continue
   old=int(round(float(r[0])))
   if desired==old:continue
   if desired in used:used_stats['collision_skips']+=1;continue
   used.discard(old);used.add(desired);r[0]=float(desired);used_stats['relabels']+=1
  if len(rows)!=len({int(round(float(r[0]))) for r in rows}):raise ValueError('duplicate IDs after P44 apply')
 return out,dict(used_stats)
def main():
 OUT.mkdir(parents=True,exist_ok=False);p1=json.loads((P26/'23-1.json').read_text());p2=json.loads((P26/'23-2.json').read_text());z1=np.load(EMB/'23-1.npz');z2=np.load(EMB/'23-2.npz');e1,v1=z1['embedding'],z1['valid'];e2,v2=z2['embedding'],z2['valid'];segments=make_segments(p2);cur1=cur2=0;H,n0=init_H(p1['frame=0'],p2['frame=0']);agg={'mean8':defaultdict(list),'geometry':defaultdict(list)};frame_assignment_changes=0;links=defaultdict(int);sample=[]
 for f in range(700):
  a=p1[f'frame={f}'];b=p2[f'frame={f}'];aa=e1[cur1:cur1+len(a)];bb=e2[cur2:cur2+len(b)];va=v1[cur1:cur1+len(a)];vb=v2[cur2:cur2+len(b)]
  lp=[] if f==0 else assign(a,b,H,aa,va,bb,vb,True);lg=[] if f==0 else assign(a,b,H,aa,va,bb,vb,False)
  frame_assignment_changes+=len({(x['ai'],x['bj']) for x in lp}.symmetric_difference({(x['ai'],x['bj']) for x in lg}))
  for tag,arr in (('mean8',lp),('geometry',lg)):
   for x in arr:
    key=segments[(f,x['bj'])];target=int(round(float(a[x['ai']][0])));agg[tag][(key,target)].append(float(x['cost']));links[tag]+=1
  if len(lg)>=4:
   src=np.asarray([center(a[x['ai']]) for x in lg],np.float32);dst=np.asarray([center(b[x['bj']]) for x in lg],np.float32);h,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.)
   if h is not None and np.isfinite(h).all():H=h.astype(np.float64)
  if f in (0,1,100,300,699):sample.append({'frame':f,'mean8_links':len(lp),'geometry_links':len(lg)})
  cur1+=len(a);cur2+=len(b)
 mappings={}
 for tag in ('mean8','geometry'):
  candidates=defaultdict(list)
  for (key,target),costs in agg[tag].items():candidates[key].append((float(np.mean(costs)),target,len(costs)))
  for key,vals in candidates.items():mappings[(tag,key)]=sorted(vals,key=lambda x:(x[0],x[1]))[0][1]
 outputs={};apps={}
 for tag in ('mean8','geometry'):
  mp={k[1]:v for k,v in mappings.items() if k[0]==tag}; outputs[tag],apps[tag]=apply(p2,mp,segments)
  (OUT/f'P44_{tag}-23-1.json').write_text(json.dumps(p1,indent=2)+'\n');(OUT/f'P44_{tag}-23-2.json').write_text(json.dumps(outputs[tag],indent=2)+'\n')
 result={'status':'COMPLETE_P44_TRACKLET_OWNER_EVIDENCE','segment_gap_frames':30,'initial_H_shared_ids':n0,'links':dict(links),'frame_assignment_changes':frame_assignment_changes,'sample':sample,'segments':len(set(segments.values())),'mapped_segments':{tag:len({k[1] for k in mappings if k[0]==tag}) for tag in ('mean8','geometry')},'application':apps,'duplicate_frame_ids':{tag:sum(len(rs)!=len(set(int(round(float(r[0]))) for r in rs)) for rs in outputs[tag].values()) for tag in ('mean8','geometry')},'labels_read':False,'xml_read':False,'official_test_read':False,'inputs_sha256':{'p26_view1':sha(P26/'23-1.json'),'p26_view2':sha(P26/'23-2.json'),'p14_view1':sha(EMB/'23-1.npz'),'p14_view2':sha(EMB/'23-2.npz')}}
 (OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'associate.exit').write_text('0\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
