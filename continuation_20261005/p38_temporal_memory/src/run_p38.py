#!/usr/bin/env python3
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path('/home/chenhc/mdmot_research_20261002'); P38=Path('/home/chenhc/claude_try_MDMOT/continuation_20261005/p38_temporal_memory')
sys.path.insert(0,str(ROOT/'p2/src')); sys.path.insert(0,str(ROOT/'p1/src')); sys.path.insert(0,str(ROOT/'p7r'))
from data import PairData, split_assignments, sha256
from train_association_probe import masked_matching_loss

UUID='GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a'; INIT=ROOT/'p1/runs/independent_s42_4000/fixed_step_004000.pt'
FIT=('23','25','28','29','39','44','45','51','53','63','66','69','70','74','78'); CAL=('27','32','42','64','65')

def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha(p): return sha256(p)
def state_sha(m):
 h=hashlib.sha256()
 for n,v in sorted(m.state_dict().items()): h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def unit(x): return F.normalize(x,dim=-1)

class TemporalAdapter(nn.Module):
 def __init__(self):
  super().__init__(); self.gate=nn.Sequential(nn.Linear(384,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,1)); self.delta=nn.Sequential(nn.Linear(384,128),nn.GELU(),nn.Linear(128,128)); nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
 def forward(self,current,history,has_history):
  x=torch.cat((current,history,current-history),dim=-1); g=torch.sigmoid(self.gate(x)); d=self.delta(x); return unit(current+has_history.float().unsqueeze(-1)*g*d)

class TemporalPair:
 def __init__(self,pair,with_labels):
  self.pair=str(pair); self.data=PairData(pair,with_labels=with_labels)
  if with_labels:self.data.require_fit()
  self.groups={}; self.hist=[]; self.has=[]; self.key_history={}
  for v,d in enumerate(self.data._views):
   n=len(d['frame']); h=np.zeros((n,128),np.float32); hm=np.zeros(n,bool)
   by=defaultdict(list)
   for i,(f,lid,present) in enumerate(zip(d['frame'],d['local_id'],d['feature_present'])):
    if present and int(lid)>=0: by[int(lid)].append(i)
   for lid,inds in by.items():
    inds.sort(key=lambda i:int(d['frame'][i]))
    for pos,i in enumerate(inds):
     f=int(d['frame'][i]); prior=[j for j in inds[:pos] if f-int(d['frame'][j])<=8]
     prior=prior[-4:]
     if prior:
      z=d['embedding'][np.asarray(prior)].astype(np.float32); z=z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12); z=z.mean(0); z=z/(np.linalg.norm(z)+1e-12); h[i]=z;hm[i]=True
   self.hist.append(h);self.has.append(hm)
   for i in range(n):
    key='%s-%d:%d:%d'%(self.pair,v+1,int(d['frame'][i]),int(d['detector_index'][i]));self.key_history[key]=(h[i].copy(),bool(hm[i]))
  for frame in range(1,self.data.frames+1):
   for cls in range(3):
    sides=[]
    for d,off in zip(self.data._views,self.data._offsets):
     ix=np.arange(int(off[frame-1]),int(off[frame]),dtype=np.int64);ix=ix[(d['cls'][ix]==cls)&d['feature_present'][ix]];sides.append(ix)
    if len(sides[0]) or len(sides[1]):self.groups[(frame,cls)]=tuple(sides)
 def batch(self,frame,cls,device):
  out=[]
  for v,ix in enumerate(self.groups[(int(frame),int(cls))]):
   d=self.data._views[v]; cur=torch.from_numpy(d['embedding'][ix].astype(np.float32)).to(device); hist=torch.from_numpy(self.hist[v][ix]).to(device); hm=torch.from_numpy(self.has[v][ix]).to(device); lab=torch.from_numpy(d['pid'][ix].copy()).to(device); valid=lab>=0;out.append((cur,hist,hm,lab,valid))
  return out

def load_init():
 ck=torch.load(INIT,map_location='cpu',weights_only=False)
 if ck.get('arm')!='independent' or ck.get('fixed_step')!=4000:raise ValueError('P1 init lineage mismatch')
 # The source digest is checked against the immutable P7R FROZEN manifest rather than a hand-copied literal.
 frozen=json.loads((ROOT/'p7r/FROZEN.json').read_text()); expected=frozen['init']['sha256']
 if sha(INIT)!=expected:raise ValueError('P1 checkpoint hash differs from P7R FROZEN')
 return ck,expected

def cpu_gate(pair):
 m=TemporalAdapter().eval(); selected=None
 for key in pair.groups:
  sides=pair.batch(*key,torch.device('cpu'))
  if len(sides)!=2: continue
  a,b=sides
  shared=set(int(x) for x in a[3][a[4]].tolist()) & set(int(x) for x in b[3][b[4]].tolist())
  if shared and (len(a[3][a[4]])>1 or len(b[3][b[4]])>1): selected=(key,sides); break
 if selected is None: raise ValueError('no real supervised CPU gate group')
 key,(a,b)=selected
 # zero-init equivalence on a real group
 with torch.no_grad():
  out=m(a[0],a[1],a[2])
  if not torch.allclose(out,unit(a[0]),atol=1e-6,rtol=0):raise ValueError('zero-init adapter is not current embedding')
 # unknown labels must not enter CE; history remains a causal input
 ma=m(a[0],a[1],a[2]);mb=m(b[0],b[1],b[2]); ce,_=masked_matching_loss(ma,mb,a[3],b[3],a[4],b[4],.1); la=a[3].clone();lb=b[3].clone();la[~a[4]]=918273;lb[~b[4]]=918273; ce2,_=masked_matching_loss(ma,mb,la,lb,a[4],b[4],.1)
 if not torch.allclose(ce,ce2,atol=1e-7,rtol=1e-7):raise ValueError('unknown-label invariance failed')
 (ce+0.2*(1-(ma[a[2]]*a[1][a[2]]).sum(-1).mean() if a[2].any() else ce*0)).backward();
 if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('gradient gate failed')
 return {'status':'PASS','zero_init_max_error':float((out-unit(a[0])).abs().max()),'unknown_label_invariant':True,'finite_gradients':True,'parameters':sum(p.numel() for p in m.parameters())}

def configure_gpu():
 snap=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True); lines=[x for x in snap.splitlines() if UUID in x]
 if len(lines)!=1 or lines[0].split(',')[0].strip()!='3' or 'A100' not in lines[0] or int(lines[0].split(',')[3])<30000:raise RuntimeError('GPU3 physical verification failed')
 os.environ['CUDA_VISIBLE_DEVICES']=UUID;torch.cuda.init();return torch.device('cuda:0'),snap

def train():
 ck,init_sha=load_init(); split=json.loads((ROOT/'p7r/SCHEDULE.json').read_text()); schedule=split
 data={p:TemporalPair(p,True) for p in FIT};
 gate=cpu_gate(data[FIT[0]]);dump(P38/'CPU_GATE.json',gate)
 device,snap=configure_gpu();torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);m=TemporalAdapter().to(device).train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);hist=[];start=time.monotonic()
 if len(schedule)!=1200:raise ValueError('P7R schedule length differs')
 for step,event in enumerate(schedule,1):
  pair=data[event['pair']]; sides=pair.batch(event['frame'],event['class'],device);(ca,ha,ma,la,va),(cb,hb,mb,lb,vb)=sides
  ea=m(ca,ha,ma);eb=m(cb,hb,mb);ce,count=masked_matching_loss(ea,eb,la,lb,va,vb,.1);tvals=[]
  for e,h,hm in ((ea,ha,ma),(eb,hb,mb)):
   if hm.any():tvals.append(1-(e[hm]*h[hm]).sum(-1).mean())
  tl=torch.stack(tvals).mean() if tvals else ce*0;loss=ce+0.2*tl
  if not torch.isfinite(loss):raise ValueError('nonfinite loss')
  opt.zero_grad(set_to_none=True);loss.backward()
  if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('nonfinite gradient')
  opt.step();hist.append({'step':step,'loss':float(loss.detach()),'ce':float(ce.detach()),'temporal':float(tl.detach()),'queries':count})
  if step==1 or step%200==0:print(json.dumps(hist[-1]),flush=True)
 torch.cuda.synchronize(); out=P38/'runs/p38_temporal';out.mkdir(parents=True,exist_ok=False); state=state_sha(m);torch.save({'state_dict':{k:v.detach().cpu() for k,v in m.state_dict().items()},'fixed_step':1200,'initial_checkpoint_sha256':init_sha,'final_state_sha256':state,'prereg_sha256':sha(P38/'PREREG.json')},out/'fixed_step_001200.pt');dump(out/'TRAINING_RECEIPT.json',{'status':'PASS_P38_FIXED_TRAINING','steps':len(hist),'first':hist[0],'last':hist[-1],'last100_mean':float(np.mean([x['loss'] for x in hist[-100:]])),'checkpoint_sha256':sha(out/'fixed_step_001200.pt'),'final_state_sha256':state,'gpu_snapshot':snap,'gpu_uuid':UUID,'gpu_name':'NVIDIA A100-SXM4-40GB','elapsed_seconds':time.monotonic()-start,'dev_read':False,'calibration_read':False});(out/'training.exit').write_text('0\n');(out/'launcher.exit').write_text('0\n');print(json.dumps({'status':'PASS_P38_FIXED_TRAINING','checkpoint_sha256':sha(out/'fixed_step_001200.pt')}),flush=True)

def build_label_free_eval(pair,record,model,device):
 import sys;sys.path.insert(0,str(ROOT/'p4_diagnosis'));import extract_dino_crop32 as ex
 data=TemporalPair(pair,False);cache=ROOT/'p4_diagnosis/crop32'; crop=np.load(cache/(pair+'.npz'),allow_pickle=False); keys=[];frames=[];views=[];classes=[];cur=[];hist=[];has=[]
 for frame in record['selected_frames']:
  b=data.data.frame(int(frame));keys.extend(b['keys']);frames.extend([r['frame'] for r in b['rows']]);views.extend([int(r['camera'])-1 for r in b['rows']]);classes.extend([r['class'] for r in b['rows']]);cur.append(b['emb'])
  for k in b['keys']:
   h,hm=data.key_history[k];hist.append(h);has.append(hm)
 cur=np.concatenate(cur).astype(np.float32); keys=np.asarray(keys); hist=np.asarray(hist,np.float32);has=np.asarray(has,bool);crop_keys=crop['keys']
 if not np.array_equal(keys,crop_keys):raise ValueError('P38 evaluation key order differs from frozen crop cache')
 valid=crop['common_support'];out=np.zeros((len(keys),128),np.float32)
 with torch.inference_mode():
  for s in range(0,len(keys),2048):
   ix=np.arange(s,min(s+2048,len(keys)));e=model(torch.from_numpy(cur[ix]).to(device),torch.from_numpy(hist[ix]).to(device),torch.from_numpy(has[ix]).to(device));out[ix]=e.cpu().numpy()
 if not np.isfinite(out).all() or not np.allclose(np.linalg.norm(out[valid],axis=1),1.,atol=2e-5):raise ValueError('P38 eval embedding contract failed')
 return {'row_key':keys,'embedding':out,'valid':valid}, {'rows':len(keys),'valid':int(valid.sum())}

def evaluate():
 import sys;sys.path.insert(0,str(ROOT/'p4_diagnosis'));import compare_crop32 as p4
 ck=json.loads((P38/'runs/p38_temporal/TRAINING_RECEIPT.json').read_text());model=TemporalAdapter().eval();raw=torch.load(P38/'runs/p38_temporal/fixed_step_001200.pt',map_location='cpu',weights_only=False);model.load_state_dict(raw['state_dict'],strict=True);model.eval()
 evalout=P38/'runs/evaluation';evalout.mkdir(exist_ok=False);frame_manifest=json.loads((ROOT/'p4_diagnosis/crop32/FRAME_MANIFEST.json').read_text());records={r['pair']:r for r in frame_manifest['pairs']}; allpairs=FIT+CAL;pred={}
 # Freeze all label-free outputs before opening the offline-label scorer.
 for pair in allpairs:
  z,meta=build_label_free_eval(pair,records[pair],model,torch.device('cpu'));np.savez_compressed(evalout/(pair+'_embeddings.npz'),**z);pred[pair]=meta;print(json.dumps({'pair':pair,**meta}),flush=True)
 p4.METHODS=('p1_independent128','temporal128'); reports=[];compact={'fit':[],'calibration':[]}; inputs={r['sequence']:r for r in json.loads((ROOT/'p2/cache/MANIFEST.json').read_text())['sequences']}
 for pair in allpairs:
  rec=records[pair];row,features,sources,cov=p4.align_pair(rec,ROOT/'p4_diagnosis/crop32',inputs);z=np.load(evalout/(pair+'_embeddings.npz'));assert np.array_equal(row['row_key'],z['row_key']);features['temporal128']=z['embedding'];arrays=p4.evaluate_arrays(row,features);report={'pair':pair,'role':rec['role'],'metrics':p4.summaries(arrays),'coverage':cov,'embedding':str(evalout/(pair+'_embeddings.npz'))};dump(evalout/(pair+'_RESULTS.json'),report);reports.append(report);compact[rec['role']].append(p4.compact(arrays))
 for role,chunks in compact.items():
  joined={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]};joined['query_row']=np.arange(len(joined['row_pid_offline']),dtype=np.int32); macro={m:float(np.mean([r['metrics']['all']['methods'][m]['all_candidates']['known_positive_r1_tie_averaged'] for r in reports if r['role']==role])) for m in p4.METHODS}; compact[role]={'pair_macro_r1':macro,'summary':p4.summaries(joined)}
 fit=compact['fit']['pair_macro_r1'];cal=compact['calibration']['pair_macro_r1'];wins=sum(reports[i]['metrics']['all']['methods']['temporal128']['all_candidates']['known_positive_r1_tie_averaged']>reports[i]['metrics']['all']['methods']['p1_independent128']['all_candidates']['known_positive_r1_tie_averaged'] for i in range(len(reports)) if reports[i]['role']=='calibration');result={'status':'COMPLETE_P38_TRAIN_ONLY_EVALUATION','training':ck,'by_role':compact,'pairs':reports,'gate':{'calibration_pair_macro_r1_min':.4641,'calibration_pair_wins_min':3,'fit15_drop_max':.05,'calibration_pair_wins':wins,'calibration_lift':cal['temporal128']-cal['p1_independent128'],'fit_drop':fit['temporal128']-fit['p1_independent128']},'dev_read':False,'official_val_test_read':False};dump(evalout/'RESULTS.json',result);keep=bool(cal['temporal128']>=.4641 and wins>=3 and fit['temporal128']>=fit['p1_independent128']-.05);dump(evalout/'DECISION.json',{'verdict':'KEEP_P38_REPRESENTATION_GATE' if keep else 'STOP_P38_REPRESENTATION_GATE','gate_pass':keep,'results':'RESULTS.json'});print(json.dumps({'status':result['status'],'fit':fit,'calibration':cal,'wins':wins,'keep':keep},indent=2),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['train','evaluate'],required=True);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1); train() if a.mode=='train' else evaluate()
if __name__=='__main__':main()
