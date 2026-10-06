#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path('/home/chenhc/mdmot_research_20261002')
WORK=Path('/home/chenhc/claude_try_MDMOT/continuation_20261005/p41_mean_adapter_fullcoverage')
P4=ROOT/'p4_diagnosis'
sys.path.insert(0,str(ROOT/'p2/src')); sys.path.insert(0,str(ROOT/'p1/src'))
from data import PairData, sha256
from train_association_probe import masked_matching_loss
UUID='GPU-1b297aba-ae7e-e326-5903-476f1bb683d7'
INIT=ROOT/'p1/runs/independent_s42_4000/fixed_step_004000.pt'
FIT=('23','25','28','29','39','44','45','51','53','63','66','69','70','74','78')
CAL=('27','32','42','64','65')
WINDOW=8; MAX_GAP=30

def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha(path): return sha256(path)
def unit(x):
    return F.normalize(x,dim=-1)
def state_sha(m):
    h=hashlib.sha256()
    for n,v in sorted(m.state_dict().items()): h.update(n.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class TemporalMeanPair:
    def __init__(self,pair,with_labels):
        self.pair=str(pair); self.data=PairData(pair,with_labels=with_labels)
        if with_labels:self.data.require_fit()
        self.anchor=[]; self.has=[]; self.key_history={}; self.groups={}
        for view,(d,off) in enumerate(zip(self.data._views,self.data._offsets)):
            anchor=np.zeros_like(d['embedding'],dtype=np.float32); has=np.zeros(len(d['frame']),dtype=bool)
            memory={}; counts=defaultdict(int)
            for i in range(len(d['frame'])):
                frame=int(d['frame'][i]); local=int(d['local_id'][i]); cls=int(d['cls'][i]); present=bool(d['feature_present'][i])
                if local<0:
                    if present: anchor[i]=d['embedding'][i]/(np.linalg.norm(d['embedding'][i])+1e-12)
                else:
                    old=memory.get(local)
                    if old is not None:
                        if frame<=old['frame']: raise ValueError('noncausal/duplicate local observation')
                        if cls!=old['class'] or frame-old['frame']>MAX_GAP: old=None
                    if old is None: old={'frame':frame,'class':cls,'embeddings':deque(maxlen=WINDOW)}; memory[local]=old
                    old['frame']=frame; old['class']=cls
                    if present:
                        z=d['embedding'][i].astype(np.float32); z=z/(np.linalg.norm(z)+1e-12); old['embeddings'].append(z)
                        mean=np.mean(np.stack(old['embeddings']),axis=0,dtype=np.float32); mean=mean/(np.linalg.norm(mean)+1e-12)
                        anchor[i]=mean; has[i]=len(old['embeddings'])>1
                key=f'{self.pair}-{view+1}:{frame}:{int(d["detector_index"][i])}'
                self.key_history[key]=(anchor[i].copy(),bool(has[i]))
            self.anchor.append(anchor); self.has.append(has)
        # Use all present rows grouped by frame/class for the fixed P7R schedule.
        for frame in range(1,self.data.frames+1):
            for cls in range(3):
                sides=[]
                for d,off in zip(self.data._views,self.data._offsets):
                    ix=np.arange(int(off[frame-1]),int(off[frame]),dtype=np.int64)
                    ix=ix[(d['cls'][ix]==cls)&d['feature_present'][ix]]
                    sides.append(ix)
                if len(sides[0]) or len(sides[1]): self.groups[(frame,cls)]=tuple(sides)
    def batch(self,frame,cls,device):
        out=[]
        for v,ix in enumerate(self.groups[(int(frame),int(cls))]):
            d=self.data._views[v]
            cur=torch.from_numpy(d['embedding'][ix].astype(np.float32)).to(device)
            anc=torch.from_numpy(self.anchor[v][ix]).to(device)
            has=torch.from_numpy(self.has[v][ix]).to(device)
            lab=torch.from_numpy(d['pid'][ix].copy()).to(device); valid=lab>=0
            out.append((cur,anc,has,lab,valid))
        return out

def load_init():
    ck=torch.load(INIT,map_location='cpu',weights_only=False)
    if ck.get('arm')!='independent' or ck.get('fixed_step')!=4000: raise ValueError('P1 init lineage mismatch')
    frozen=json.loads((ROOT/'p7r/FROZEN.json').read_text())
    if sha(INIT)!=frozen['init']['sha256']: raise ValueError('P1 checkpoint hash differs from P7R FROZEN')
    return frozen['init']['sha256']

class MeanAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(384,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,1))
        self.delta=nn.Sequential(nn.Linear(384,128),nn.GELU(),nn.Linear(128,128))
        nn.init.zeros_(self.delta[-1].weight); nn.init.zeros_(self.delta[-1].bias)
    def forward(self,current,anchor,has):
        base=torch.where(has.unsqueeze(-1),anchor,current)
        x=torch.cat((current,base,current-base),dim=-1)
        g=torch.sigmoid(self.gate(x)); d=self.delta(x)
        return unit(base+has.float().unsqueeze(-1)*g*d)

def configure_gpu():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    selected=[x for x in lines if UUID in x]
    if len(selected)!=1: raise RuntimeError('GPU UUID not found')
    parts=[x.strip() for x in selected[0].split(',')]
    if parts[0]!='1' or 'A100' not in parts[2] or int(parts[3])<30000: raise RuntimeError('physical GPU1 verification failed')
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'; os.environ['CUDA_VISIBLE_DEVICES']=UUID
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('UUID binding failed')
    prop=torch.cuda.get_device_properties(0)
    if 'A100' not in prop.name or prop.total_memory/1024**2<30000: raise RuntimeError('torch GPU mismatch')
    return torch.device('cuda:0'), selected[0]

def cpu_gate(pair):
    m=MeanAdapter().eval(); selected=None
    for key in pair.groups:
        sides=pair.batch(*key,torch.device('cpu'))
        if len(sides)!=2: continue
        a,b=sides; shared=set(int(x) for x in a[3][a[4]].tolist())&set(int(x) for x in b[3][b[4]].tolist())
        if shared and (len(a[3][a[4]])>1 or len(b[3][b[4]])>1): selected=(key,sides); break
    if selected is None: raise ValueError('no CPU gate group')
    key,(a,b)=selected
    with torch.no_grad():
        out=m(a[0],a[1],a[2]); expected=unit(torch.where(a[2].unsqueeze(-1),a[1],a[0]))
        if not torch.allclose(out,expected,atol=1e-6,rtol=0): raise ValueError('zero-init anchor equivalence failed')
    ma,mb=m(a[0],a[1],a[2]),m(b[0],b[1],b[2]); ce,_=masked_matching_loss(ma,mb,a[3],b[3],a[4],b[4],.1)
    la,lb=a[3].clone(),b[3].clone(); la[~a[4]]=918273; lb[~b[4]]=918273; ce2,_=masked_matching_loss(ma,mb,la,lb,a[4],b[4],.1)
    if not torch.allclose(ce,ce2,atol=1e-7,rtol=1e-7): raise ValueError('unknown-label invariance failed')
    (ce+0.2*(1-(ma[a[2]]*a[1][a[2]]).sum(-1).mean() if a[2].any() else ce*0)).backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()): raise ValueError('gradient gate failed')
    return {'status':'PASS_P41_CPU_GATE','selected_group':list(key),'zero_init_max_error':float((out-expected).abs().max()),'unknown_label_invariant':True,'finite_gradients':True,'parameters':sum(p.numel() for p in m.parameters())}

def build_schedule(pairs):
    eligible=[]
    for pair_name in FIT:
        pair=pairs[pair_name]
        for key in sorted(pair.groups):
            a,b=pair.batch(*key,torch.device('cpu'))
            shared=set(int(x) for x in a[3][a[4]].tolist()) & set(int(x) for x in b[3][b[4]].tolist())
            if shared and (len(a[3][a[4]])>1 or len(b[3][b[4]])>1):
                eligible.append({'pair':pair_name,'frame':int(key[0]),'class':int(key[1])})
    if len(eligible)!=18412: raise ValueError(f'eligible group count changed: {len(eligible)}')
    rng=np.random.default_rng(42)
    schedule=[]
    for epoch in range(3):
        order=rng.permutation(len(eligible))
        schedule.extend([eligible[int(i)] for i in order])
    return schedule, eligible

def train():
    if (WORK/'runs').exists(): raise RuntimeError('runs already exists; sealed outputs require a fresh workspace')
    init_sha=load_init(); pairs={p:TemporalMeanPair(p,True) for p in FIT}; gate=cpu_gate(pairs[FIT[0]]); dump(WORK/'CPU_GATE.json',gate)
    schedule,eligible=build_schedule(pairs); dump(WORK/'SCHEDULE.json',{'status':'PASS_P41_FULL_COVERAGE_SCHEDULE','epochs':3,'eligible_groups':len(eligible),'updates':len(schedule),'seed':42,'order':'fixed seeded permutation per epoch','pairs':list(FIT)})
    device,gpu=configure_gpu(); torch.manual_seed(42); torch.cuda.manual_seed_all(42); np.random.seed(42)
    m=MeanAdapter().to(device).train(); opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4); history=[]; start=time.monotonic()
    for step,event in enumerate(schedule,1):
        pair=pairs[event['pair']]; (ca,aa,ha,la,va),(cb,ab,hb,lb,vb)=pair.batch(event['frame'],event['class'],device)
        ea,eb=m(ca,aa,ha),m(cb,ab,hb); ce,count=masked_matching_loss(ea,eb,la,lb,va,vb,.1)
        vals=[]
        for e,a,h in ((ea,aa,ha),(eb,ab,hb)):
            if h.any(): vals.append(1-(e[h]*a[h]).sum(-1).mean())
        temporal=torch.stack(vals).mean() if vals else ce*0; loss=ce+0.2*temporal
        if not torch.isfinite(loss): raise ValueError('nonfinite loss')
        opt.zero_grad(set_to_none=True); loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()): raise ValueError('nonfinite gradient')
        opt.step(); history.append({'step':step,'loss':float(loss.detach()),'ce':float(ce.detach()),'temporal':float(temporal.detach()),'queries':count})
        if step==1 or step%1000==0: print(json.dumps(history[-1]),flush=True)
    torch.cuda.synchronize(); out=WORK/'runs'; out.mkdir()
    checkpoint=out/'fixed_step_055236.pt'; final_state=state_sha(m)
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in m.state_dict().items()},'fixed_step':len(schedule),'initial_checkpoint_sha256':init_sha,'final_state_sha256':final_state,'prereg_sha256':sha(WORK/'PREREG.json')},checkpoint)
    dump(out/'TRAINING_RECEIPT.json',{'status':'PASS_P41_FULL_COVERAGE_TRAINING','steps':len(schedule),'epochs':3,'eligible_groups':len(eligible),'first':history[0],'last':history[-1],'last100_mean':float(np.mean([x['loss'] for x in history[-100:]])),'checkpoint_sha256':sha(checkpoint),'final_state_sha256':final_state,'gpu_uuid':UUID,'gpu_name':'NVIDIA A100-SXM4-40GB','gpu_snapshot':gpu,'elapsed_seconds':time.monotonic()-start,'dev_read':False,'calibration_read':False,'schedule_sha256':sha(WORK/'SCHEDULE.json'),'init_sha256':init_sha})
    (out/'training.exit').write_text('0\n'); (out/'launcher.exit').write_text('0\n')
    print(json.dumps({'status':'PASS_P41_FULL_COVERAGE_TRAINING','checkpoint_sha256':sha(checkpoint),'final_state_sha256':final_state,'steps':len(schedule)}),flush=True)

def label_free_embeddings(pair, rec, model):
    d=TemporalMeanPair(pair,False); out=[]
    for frame in rec['selected_frames']:
        b=d.data.frame(int(frame))
        for row in b['rows']:
            # Use the exact frame/detector key; detector_index is metadata, not an array offset.
            view=int(row['camera'])-1; source=d.data._views[view]
            ix=np.flatnonzero((source['frame']==int(row['frame']))&(source['detector_index']==int(row['detector_index'])))
            if len(ix)!=1: raise ValueError('source row lookup is not unique')
            i=int(ix[0]); out.append((row['key'],source['embedding'][i],d.anchor[view][i],d.has[view][i],bool(source['feature_present'][i])))
    keys=np.asarray([x[0] for x in out]); current=np.asarray([x[1] for x in out],np.float32); anchor=np.asarray([x[2] for x in out],np.float32); has=np.asarray([x[3] for x in out],bool); present=np.asarray([x[4] for x in out],bool)
    result=np.zeros_like(current)
    with torch.inference_mode():
        for s in range(0,len(keys),2048):
            ix=np.arange(s,min(s+2048,len(keys))); result[ix]=model(torch.from_numpy(current[ix]),torch.from_numpy(anchor[ix]),torch.from_numpy(has[ix])).numpy()
    return keys,result,present

def evaluate():
    import sys as _sys; _sys.path.insert(0,str(P4)); import compare_crop32 as cmp
    allpairs=FIT+CAL; manifest=json.loads((P4/'crop32/FRAME_MANIFEST.json').read_text()); records={r['pair']:r for r in manifest['pairs']}; out=WORK/'runs'
    raw=torch.load(out/'fixed_step_055236.pt',map_location='cpu',weights_only=False); model=MeanAdapter().eval(); model.load_state_dict(raw['state_dict'],strict=True)
    frozen={}
    for pair in allpairs:
        keys,emb,present=label_free_embeddings(pair,records[pair],model)
        with np.load(P4/'crop32'/f'{pair}.npz',allow_pickle=False) as crop:
            if not np.array_equal(keys,crop['keys']): raise ValueError('P4 key alignment mismatch: '+pair)
            common=crop['common_support']
        if np.any(emb[~present]!=0) or not np.isfinite(emb).all() or not np.allclose(np.linalg.norm(emb[common],axis=1),1,atol=2e-5): raise ValueError('P40 embedding contract failed: '+pair)
        path=out/(pair+'_embeddings.npz'); np.savez_compressed(path,row_key=keys,embedding=emb,valid=common); frozen[pair]={'path':str(path),'sha256':sha(path),'rows':int(len(keys)),'common_support':int(common.sum())}
    inputs={r['sequence']:r for r in json.loads((ROOT/'p2/cache/MANIFEST.json').read_text())['sequences']}; cmp.METHODS=('p1_independent128','temporal_mean_adapter'); reports=[]
    for pair in allpairs:
        rec=records[pair]; row,features,refs,cov=cmp.align_pair(rec,P4/'crop32',inputs); z=np.load(out/(pair+'_embeddings.npz'),allow_pickle=False)
        if not np.array_equal(row['row_key'],z['row_key']): raise ValueError('row alignment changed: '+pair)
        features['temporal_mean_adapter']=z['embedding']; arr=cmp.evaluate_arrays(row,features); result={'pair':pair,'role':rec['role'],'metrics':cmp.summaries(arr),'coverage':cov,'embedding':frozen[pair]}; dump(out/(pair+'_RESULTS.json'),result); reports.append(result)
    by={}
    for role in ('fit','calibration'):
        rs=[r for r in reports if r['role']==role]; macro={m:float(np.mean([r['metrics']['all']['methods'][m]['all_candidates']['known_positive_r1_tie_averaged'] for r in rs])) for m in cmp.METHODS}; wins=int(sum(r['metrics']['all']['methods']['temporal_mean_adapter']['all_candidates']['known_positive_r1_tie_averaged']>r['metrics']['all']['methods']['p1_independent128']['all_candidates']['known_positive_r1_tie_averaged'] for r in rs)); by[role]={'pair_macro_r1':macro,'wins':wins,'pairs':[r['pair'] for r in rs]}
    fit,cal=by['fit']['pair_macro_r1'],by['calibration']['pair_macro_r1']; fixed=json.loads((Path('/home/chenhc/claude_try_MDMOT/continuation_20261005/p39_temporal_mean_anchor/runs/RESULTS.json')).read_text())['by_role']['calibration']['pair_macro_r1']['temporal_mean8']; gate={'calibration_pair_macro_r1_min':.4641,'relative_lift_min':.01,'calibration_pair_wins_min':3,'fit_drop_max':.05,'calibration_lift':cal['temporal_mean_adapter']-cal['p1_independent128'],'calibration_pair_wins':by['calibration']['wins'],'fit_drop':fit['temporal_mean_adapter']-fit['p1_independent128'],'fixed_mean_cal5':fixed,'learned_vs_fixed_mean':cal['temporal_mean_adapter']-fixed}; gate['pass']=bool(cal['temporal_mean_adapter']>=.4641 and gate['calibration_lift']>=.01 and gate['calibration_pair_wins']>=3 and gate['fit_drop']>=-.05 and gate['learned_vs_fixed_mean']>0)
    result={'status':'COMPLETE_P41_FULL_COVERAGE_EVALUATION','by_role':by,'gate':gate,'pairs':reports,'dev_read':False,'official_val_test_read':False,'labels_read_after_label_freeze':True}; dump(out/'RESULTS.json',result); verdict='KEEP_P41_FULL_COVERAGE_REPRESENTATION_DIRECTION' if gate['pass'] else 'STOP_P41_FULL_COVERAGE_ADAPTER'; dump(WORK/'DECISION.json',{'verdict':verdict,'gate_pass':gate['pass'],'no_host_transfer':True,'results':'runs/RESULTS.json'}); print(json.dumps({'fit':fit,'calibration':cal,'gate':gate,'verdict':verdict},indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=('train','evaluate'),required=True); a=ap.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1); train() if a.mode=='train' else evaluate()
if __name__=='__main__': main()
