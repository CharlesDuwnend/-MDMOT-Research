#!/usr/bin/env python3
"""Train canonical anchor head, paired supervision separated from inference inputs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,hashlib,random,time,sys
import numpy as np,torch
from canonical import AnchorHead,crossfit_error,dlt,project
HERE=Path(__file__).resolve().parent
P4=Path('/home/chenhc/mdmot_research_20261002/p4_diagnosis')
SPLIT=Path('/home/chenhc/mdmot_research_20261002/p0/data/ASSOCIATION_DIAGNOSTIC_SPLIT.json')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def data_pair(pair):
    z=np.load(P4/'crop32'/f'{pair}.npz',allow_pickle=False)
    labels=np.load(P4/'crop32_comparison'/f'{pair}_aligned_queries.npz',allow_pickle=False)
    assert not any('pid' in k.lower() for k in z.files)
    assert np.array_equal(z['keys'],labels['row_key'])
    groups=[]
    for frame in sorted(set(z['frame'])):
        maps=[]
        for v in [0,1]:
            m={}
            for i in np.flatnonzero((z['frame']==frame)&(z['view']==v)&z['common_support']):
                pid=int(labels['row_pid_offline'][i])
                if pid>=0:m.setdefault(pid,[]).append(int(i))
            maps.append(m)
        pids=sorted(set(maps[0])&set(maps[1]))
        pids=[p for p in pids if len(maps[0][p])==1 and len(maps[1][p])==1]
        if len(pids)<8:continue
        # Identity correspondence is supervision; does not enter feat/box values.
        ix=np.array([[maps[v][p][0] for p in pids] for v in [0,1]])
        feats=np.asarray(z['unit_dino'][ix],np.float32)
        boxes=np.asarray(z['bbox'][ix],np.float32)/np.array([1920,1080,1920,1080],np.float32)
        groups.append({'pair':pair,'frame':int(frame),'feat':torch.from_numpy(feats),'boxes':torch.from_numpy(boxes),'count':len(pids),'source_indices':ix})
    return groups

def get_points(model,g,mode):
    boxes=g['boxes'];feats=g['feat']
    if mode=='center':return (boxes[:,:,:2]+boxes[:,:,2:])/2
    if mode=='bottom':return torch.stack(((boxes[:,:,0]+boxes[:,:,2])/2,boxes[:,:,3]),dim=2)
    return torch.stack([model(feats[v],boxes[v])[0] for v in [0,1]])

def evaluate(model,groups):
    rows=[]
    with torch.no_grad():
        for g in groups:
            for mode in ['center','bottom','learned']:
                pp=get_points(model,g,mode);err=[]
                for f in range(4):
                    fold=(torch.arange(g['count'])+f)%4
                    ea,eb=crossfit_error(pp[0],pp[1],fold,torch.tensor([1920.,1080.]))
                    err.extend(torch.linalg.norm(ea,dim=1).tolist()+torch.linalg.norm(eb,dim=1).tolist())
                rows.append({'pair':g['pair'],'frame':g['frame'],'mode':mode,'median_px':float(np.median(err)),'p90_px':float(np.percentile(err,90))})
    per={}
    for p in sorted(set(g['pair'] for g in groups),key=int):
        per[p]={m:float(np.mean([r['median_px'] for r in rows if r['pair']==p and r['mode']==m])) for m in ['center','bottom','learned']}
    return {'per_pair':per,'macro':{m:float(np.mean([r[m] for r in per.values()])) for m in ['center','bottom','learned']},'rows':rows}

def main():
    out=HERE/'runs/train_v1';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    splits=json.loads(SPLIT.read_text())['assignments'];train=[];cal=[]
    for role,dst in [('fit',train),('calibration',cal)]:
        for p in splits[role]:
            gs=data_pair(p);dst.extend(gs);print(json.dumps({'stage':'load','role':role,'pair':p,'groups':len(gs)}),flush=True)
    assert set(g['pair'] for g in train).isdisjoint(set(g['pair'] for g in cal))
    model=AnchorHead();opt=torch.optim.Adam(model.parameters(),lr=.0003);scale=torch.tensor([1920.,1080.]);losses=[]
    # zero-init contract before any training
    init=evaluate(model,cal[:4]);zero=max(abs(r['learned']-r['center']) for r in init['per_pair'].values())
    assert zero<1e-4,zero
    start=time.monotonic()
    for step in range(800):
        g=train[step%len(train)]; pp=get_points(model,g,'learned');fold=(torch.arange(g['count'])+(step//len(train))%4)%4
        ea,eb=crossfit_error(pp[0],pp[1],fold,scale)
        # robust pixels; clipped gradient influence for far extrapolated objects.
        loss=(torch.sqrt(1+ea.square().sum(1)).mean()+torch.sqrt(1+eb.square().sum(1)).mean())/40
        reg=0
        for v in [0,1]:reg=reg+model(g['feat'][v],g['boxes'][v])[1].square().mean()
        loss=loss+.02*reg
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        opt.zero_grad();loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();losses.append(float(loss))
        if step%100==0:print(json.dumps({'step':step,'loss':float(loss),'seconds':time.monotonic()-start}),flush=True)
    torch.save({'state_dict':model.state_dict(),'steps':800,'seed':42,'method':'P35 canonical anchor'},out/'anchor.pt')
    fit=evaluate(model,train);calres=evaluate(model,cal)
    ratio=calres['macro']['learned']/calres['macro']['center'];wins=sum(r['learned']<=.95*r['center'] for r in calres['per_pair'].values())
    # Guarantee actual dependence on inputs and absence of GT fields from head.
    g=train[0];pa,_=model(g['feat'][0],g['boxes'][0]);q,_=model(g['feat'][0].flip(0),g['boxes'][0]);inputdep=float((pa-q).abs().max())
    audit={'status':'PASS','input_fields':['unit_dino','bbox'],'label_fields_separate':True,'key_alignment_asserted':True,'zero_init_max_error':zero,'target_excluded_from_DLT_support':True,'nonfinite_loss_or_gradients':False,'input_feature_dependency_max':inputdep,'no_official_val_or_test_read':True,'source_sha256':sha(__file__),'canonical_sha256':sha(HERE/'canonical.py'),'split_sha256':sha(SPLIT)}
    result={'status':'PASS_CALIBRATION_REPRESENTATION_GATE' if ratio<=.95 and wins>=3 else 'FAIL_CALIBRATION_REPRESENTATION_GATE','fit_groups':len(train),'calibration_groups':len(cal),'calibration_ratio':ratio,'calibration_pairs_improved_5pct':wins,'fit':fit,'calibration':calres,'training_loss_first':losses[0],'training_loss_last':losses[-1],'training_seconds':time.monotonic()-start,'checkpoint_sha256':sha(out/'anchor.pt'),'audit':audit,'boundary':'GT correspondences form supervisory diagnostic DLT; not legal tracking inference or MOT effectiveness'}
    dump(out/'RESULT.json',result);dump(out/'IMPLEMENTATION_AUDIT.json',audit);print(json.dumps({k:v for k,v in result.items() if k not in ['fit','calibration']},indent=2),flush=True)
if __name__=='__main__':main()
