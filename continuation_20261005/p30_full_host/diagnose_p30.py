import json, statistics
from pathlib import Path
import numpy as np
ROOT=Path('/raid/datasets/chc_data/claude_try_MDMOT_p30_full_host_20261005')
PAIRS=['27','32','42','64','65']
def iou(a,b):
 x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]); return inter/max(aa+bb-inter,1e-9)
rows=[]
for pair in PAIRS:
 for view in [1,2]:
  aroot=ROOT/'calibration/overrides/native'/f'{pair}-{view}'; broot=ROOT/'calibration/overrides/past_fixed_offsets'/f'{pair}-{view}'
  files=sorted(aroot.glob('*.npy'))
  counts=[]; exact=0; matches=[]; score_delta=[]; box_delta=[]
  for p in files:
   a=np.load(p); b=np.load(broot/p.name)
   counts.append((len(a),len(b)))
   if a.shape==b.shape and np.array_equal(a,b): exact+=1
   # Vectorized row-wise best IoU; this is a diagnostic, not an evaluator.
   if len(a) and len(b):
    x1=np.maximum(a[:,None,0],b[None,:,0]); y1=np.maximum(a[:,None,1],b[None,:,1])
    x2=np.minimum(a[:,None,2],b[None,:,2]); y2=np.minimum(a[:,None,3],b[None,:,3])
    inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
    aa=np.maximum(0,a[:,2]-a[:,0])*np.maximum(0,a[:,3]-a[:,1])
    bb=np.maximum(0,b[:,2]-b[:,0])*np.maximum(0,b[:,3]-b[:,1])
    ov=inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-9)
    ix=np.argmax(ov,axis=1); best=ov[np.arange(len(a)),ix]; keep=best>=.5
    matches.extend(best[keep].tolist())
    score_delta.extend((a[keep,4]-b[ix[keep],4]).tolist())
    box_delta.extend(np.max(np.abs(a[keep,:4]-b[ix[keep],:4]),axis=1).tolist())
  rows.append({'pair':pair,'view':view,'frames':len(files),'native_rows':sum(x[0] for x in counts),'fixed_rows':sum(x[1] for x in counts),'mean_count_delta':statistics.mean(y-x for x,y in counts),'exact_frames':exact,'exact_first8':sum(1 for p in files[:8] if np.array_equal(np.load(p),np.load(broot/p.name))),'matched_iou_ge05':len(matches),'matched_iou_mean':statistics.mean(matches) if matches else None,'matched_score_abs_mean':statistics.mean(abs(x) for x in score_delta) if score_delta else None,'matched_box_abs_mean':statistics.mean(box_delta) if box_delta else None})
out={'status':'COMPLETE_P30_OPERATOR_DIAGNOSTIC','rows':rows,'note':'array-only frozen override comparison; no labels or XML read'}
(ROOT/'host/P30_OPERATOR_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
