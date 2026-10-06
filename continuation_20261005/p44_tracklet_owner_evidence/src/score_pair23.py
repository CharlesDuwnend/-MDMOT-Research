#!/usr/bin/env python3
import json, statistics, importlib.util
from pathlib import Path
import motmetrics as mm
ROOT=Path('/home/chenhc/claude_try_MDMOT'); P37=ROOT/'continuation_20261005/p44_tracklet_owner_evidence/runs/host_pair23'; BASE=ROOT/'continuation_20261005/p26_mia_baseline/runs/fit23/json/mia_baseline'; SCORE=P37/'scores'; SCORE.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('stage15','/home/chenhc/mdmot_global_host_stage15_20260905/scripts/stage15_global_host.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s); mango,_=s.load_official_mda()
gt={v:SCORE/f'gt-23-{v}.txt' for v in (1,2)}
for v,p in gt.items(): s.xml_to_mot('23',str(v),p)
def pred_txt(js_path,out):
 js=json.loads(Path(js_path).read_text())
 with Path(out).open('w') as f:
  for k,rs in js.items():
   fr=int(k.split('=')[1])+1
   for r in rs:f.write(f'{fr},{int(float(r[0]))},{r[1]},{r[2]},{float(r[3])-float(r[1])},{float(r[4])-float(r[2])},1,1,1\n')
def one(name,paths):
 mda,_=s.official_mda(mango,paths[1],paths[2],gt[1],gt[2]); vals=[]
 for v in (1,2):
  pred=SCORE/f'{name}-pred-{v}.txt'; pred_txt(paths[v],pred)
  g=mm.io.loadtxt(str(gt[v]),fmt='mot15-2D',min_confidence=1); t=mm.io.loadtxt(str(pred),fmt='mot15-2D'); acc=mm.utils.compare_to_groundtruth(g,t,'iou',distth=.5); vals.append(mm.metrics.create().compute(acc,metrics=['idf1','mota','num_switches']).iloc[0].to_dict())
 return {'method':name,'MDA':mda,'IDF1':statistics.mean(x['idf1'] for x in vals),'MOTA':statistics.mean(x['mota'] for x in vals),'num_switches':sum(x['num_switches'] for x in vals),'views':vals}
rows=[one('P26_baseline',{1:BASE/'23-1.json',2:BASE/'23-2.json'}),one('P44_geometry',{1:P37/'P44_geometry-23-1.json',2:P37/'P44_geometry-23-2.json'}),one('P44_mean8',{1:P37/'P44_mean8-23-1.json',2:P37/'P44_mean8-23-2.json'})]
(P37/'SCORES.json').write_text(json.dumps(rows,indent=2)+'\n'); print(json.dumps(rows,indent=2))
