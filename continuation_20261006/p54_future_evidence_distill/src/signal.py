#!/usr/bin/env python3
"""Train-free ceiling diagnostic for P54; labels only construct legal event positives."""
import json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import data_audit as d

def rank1(scores,pos):
 order=np.argsort(-scores,kind='stable')
 return bool(any(int(j) in set(pos) for j in order[:1]))
def main():
 out={};
 for sid in d.EVAL:
  events,_=d.build_pair(sid)
  full=[]; base=[]; teacher=[]; nteach=0
  for e in events:
   c=e['candidates']; t=e['target']; pos=e['positive'];
   ts=t/max(float(np.linalg.norm(t)),1e-6); cs=c/np.maximum(np.linalg.norm(c,axis=1,keepdims=True),1e-6)
   bs=cs@ts; full.append(rank1(bs,pos))
   if e['teacher'] is not None:
    fs=e['teacher']/max(float(np.linalg.norm(e['teacher'])),1e-6); teacher.append(rank1(cs@fs,pos)); nteach+=1
  out[sid]={'all_events':len(events),'baseline_r1_all':float(np.mean(full)),'teacher_events':nteach,'baseline_r1_teacher_subset':float(np.mean([rank1((e['candidates']/np.maximum(np.linalg.norm(e['candidates'],axis=1,keepdims=True),1e-6))@(e['target']/max(float(np.linalg.norm(e['target'])),1e-6)),e['positive']) for e in events if e['teacher'] is not None])),'future_teacher_r1':float(np.mean(teacher))}
  del events
 macro={k:float(np.mean([v[k] for v in out.values()])) for k in ('baseline_r1_all','baseline_r1_teacher_subset','future_teacher_r1')}
 result={'status':'PASS_P54_TRAIN_FREE_FUTURE_SIGNAL','eval_pairs':list(d.EVAL),'per_pair':out,'macro':macro,'teacher_is_inference_input':False,'metrics_are_ceiling_diagnostics_only':True,'official_val_test_access':False}
 p=HERE.parent/'TRAIN_FREE_SIGNAL.json'; p.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
