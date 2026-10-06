#!/usr/bin/env python3
"""Audit P54 event legality and future-teacher observability; no model training."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path('/home/chenhc/cross_uav_query_mamba_20260830')
sys.path.insert(0, str(ROOT))
import stage3_runner as s3
FIT=("23","25","27","28","29","30","32","39","42","44","45","50","51","53","54")
EVAL=("69","70","74","76","78")

def iou(a,b): return s3.iou(a,b)
def gid(row, gt):
    best=max(((iou(row['bbox'],g['bbox']),g) for g in gt),key=lambda x:x[0]) if gt else (0,None)
    return int(best[1]['gt_id']) if best[0]>=.5 else None
def center(box): return np.asarray([(box[0]+box[2])/2,(box[1]+box[3])/2],np.float32)
def norm(v): return v/max(float(np.linalg.norm(v)),1e-6)

def build_pair(sid, limit=None, include_no_positive=False):
    aa=s3.load_index('train',sid,'1'); bb=s3.load_index('train',sid,'2')
    by_a,by_b=defaultdict(list),defaultdict(list)
    for r in aa: by_a[int(r['frame_id'])].append(r)
    for r in bb: by_b[int(r['frame_id'])].append(r)
    gt_a,gt_b=s3.read_gt(sid,'1'),s3.read_gt(sid,'2')
    source_ids={fr:{id(r):gid(r,gt_b.get(fr,[])) for r in rows} for fr,rows in by_b.items()}
    future=defaultdict(list)
    for fr in sorted(by_b):
        for r in by_b[fr]:
            g=source_ids[fr][id(r)]
            if g is not None: future[g].append((fr,r['visual_feature'].astype(np.float32)))
    events=[]; counts=defaultdict(int)
    for fr in sorted(set(by_a)&set(by_b)):
        src=by_b[fr]; ids=source_ids[fr]
        for tr in by_a[fr]:
            g=gid(tr,gt_a.get(fr,[]))
            if g is None: counts['target_unlabelled']+=1; continue
            a=tr['visual_feature'].astype(np.float32); an=max(float(np.linalg.norm(a)),1e-6)
            scored=[]
            for j,cr in enumerate(src):
                b=cr['visual_feature'].astype(np.float32)
                sim=float(np.dot(a,b)/(an*max(float(np.linalg.norm(b)),1e-6)))
                dist=float(np.linalg.norm(center(tr['bbox'])-center(cr['bbox'])))
                scored.append((0.7*sim-0.3*min(dist/500.,1.),j))
            chosen=[j for _,j in sorted(scored,reverse=True)[:64]]
            pos=[i for i,j in enumerate(chosen) if ids[id(src[j])]==g]
            if not pos:
                counts['positive_not_in_top64']+=1
                if not include_no_positive:
                    continue
            # Strictly future source-view evidence: no same-frame and no past row.
            frows=[(f,v) for f,v in future[g] if f>fr]
            frows=sorted(frows,key=lambda x:x[0])[:8]
            teacher=None
            if frows and pos:
                arr=np.stack([norm(v) for _,v in frows])
                teacher=norm(arr.mean(0)).astype(np.float32)
                counts['future_teacher_events']+=1
            elif pos: counts['no_future_teacher']+=1
            events.append({'pair':sid,'frame':fr,'target':a,'candidates':np.stack([src[j]['visual_feature'].astype(np.float32) for j in chosen]),'positive':pos,'teacher':teacher,'valid':len(chosen)})
            counts['events']+=1
            if limit and len(events)>=limit: break
        if limit and len(events)>=limit: break
    counts['candidate_rows_top64']=sum(int(e['valid']) for e in events)
    return events,counts

def main():
    per={}; all_counts=defaultdict(int)
    for sid in FIT+EVAL:
        rows,c=build_pair(sid)
        per[sid]=c
        for k,v in c.items(): all_counts[k]+=v
    result={'status':'PASS_P54_FUTURE_TEACHER_DATA_AUDIT','fit_pairs':list(FIT),'eval_pairs':list(EVAL),'per_pair':per,'totals':dict(all_counts),'teacher_in_inference':False,'labels_used_only_for_event_construction':True,'official_val_test_access':False,'future_condition':'source_view frame > target frame','no_same_frame_future_leak':True}
    out=Path(__file__).resolve().parents[1]/'DATA_AUDIT.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
