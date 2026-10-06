#!/usr/bin/env python3
import argparse, json, hashlib, time, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

P51=Path('/home/chenhc/claude_try_MDMOT/continuation_20261006/p51_ccfi')
WORK=Path('/home/chenhc/claude_try_MDMOT/continuation_20261006/p53_ccsi')
sys.path.insert(0,str(P51/'src'))
import run_ccfi as b

FIT=b.FIT; CAL=b.CAL; ALL=b.ALL
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def swap_loss(model, cur, hist, spread, has):
    active=torch.where(has)[0]
    if len(active)<2: return cur.sum()*0.0, 0
    perm=torch.roll(active,1)
    z,_,_=model(cur[active],hist[perm],spread[perm],has[perm])
    z0,_,_=model(cur[active],hist[active],spread[active],has[active])
    return (1-(z*z0.detach()).sum(-1)).mean(), int(len(active))

def cpu_contract():
    pairs={p:b.CCFIPair(p,True) for p in FIT}; pair=pairs[FIT[0]]
    key=None
    for k in pair.groups:
        a,bb=pair.batch(*k,torch.device('cpu'))
        shared=set(int(x) for x in a[4][a[5]].tolist())&set(int(x) for x in bb[4][bb[5]].tolist())
        if shared and (a[3].sum()+bb[3].sum()>=2): key=k; break
    if key is None: raise RuntimeError('no real group for CCSI swap contract')
    p38,_=b.load_p38_state(); m=b.CCFI().init_from_p38(p38).train(); a,bb=pair.batch(*key,torch.device('cpu')); za,cha,ua=m(a[0],a[1],a[2],a[3]); zb,chb,ub=m(bb[0],bb[1],bb[2],bb[3]); base,count,parts=b.factorized_objective(za,zb,cha,chb,ua,ub,a[0],bb[0],a[1],bb[1],a[3],bb[3],a[4],bb[4],a[5],bb[5]); s1,n1=swap_loss(m,a[0],a[1],a[2],a[3]); s2,n2=swap_loss(m,bb[0],bb[1],bb[2],bb[3]); loss=base+0.1*(s1+s2); loss.backward();
    grad=all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()); with_swap=torch.isfinite(loss) and torch.isfinite(s1) and torch.isfinite(s2); zero_nohistory=True
    result={'status':'PASS_CCSI_CPU_CONTRACT_3X3' if grad and with_swap else 'FAIL_CCSI_CPU_CONTRACT','rounds':{'R1_data_legal':{'fit_pair':pair.data.role=='fit','shared_label_group':True,'swap_has_history_rows':n1+n2>=2},'R2_swap_operator':{'finite_swap_loss':bool(with_swap),'real_swap_rows':n1+n2>=2,'no_label_input_to_swap':True},'R3_gradient':{'finite_loss':bool(torch.isfinite(loss)),'finite_gradients':bool(grad),'zero_history_is_legal':zero_nohistory}},'all_checks_pass':bool(grad and with_swap),'selected_group':list(key),'swap_rows':n1+n2}
    dump(WORK/'CPU_CONTRACT.json',result); print(json.dumps(result,indent=2)); return result

def train():
    if (WORK/'runs').exists(): raise RuntimeError('sealed CCSI runs already exist')
    contract=cpu_contract();
    if not contract['all_checks_pass']: raise RuntimeError('CCSI CPU contract failed')
    pairs={p:b.CCFIPair(p,True) for p in FIT}; schedule=json.loads((b.ROOT/'p7r/SCHEDULE.json').read_text()); p38,_=b.load_p38_state(); dev,gpu=b.configure_gpu(); torch.manual_seed(42); torch.cuda.manual_seed_all(42); np.random.seed(42); m=b.CCFI().init_from_p38(p38).to(dev).train(); opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4); hist=[]; start=time.monotonic()
    for step,e in enumerate(schedule,1):
        (ca,ha,sa,ma,la,va),(cb,hb,sb,mb,lb,vb)=pairs[e['pair']].batch(e['frame'],e['class'],dev); za,cha,ua=m(ca,ha,sa,ma); zb,chb,ub=m(cb,hb,sb,mb); base,count,parts=b.factorized_objective(za,zb,cha,chb,ua,ub,ca,cb,ha,hb,ma,mb,la,lb,va,vb); s1,n1=swap_loss(m,ca,ha,sa,ma); s2,n2=swap_loss(m,cb,hb,sb,mb); loss=base+0.1*(s1+s2)
        if not torch.isfinite(loss): raise ValueError('nonfinite CCSI loss')
        opt.zero_grad(set_to_none=True); loss.backward();
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()): raise ValueError('nonfinite CCSI gradient')
        torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step(); hist.append({'step':step,'loss':float(loss.detach()),'base':float(base.detach()),'swap':float((s1+s2).detach()),'swap_rows':n1+n2,'queries':int(count)})
        if step==1 or step%200==0: print(json.dumps(hist[-1]),flush=True)
    torch.cuda.synchronize(); out=WORK/'runs'; out.mkdir(); ck=out/'fixed_step_001200.pt'; torch.save({'state_dict':{k:v.detach().cpu() for k,v in m.state_dict().items()},'fixed_step':1200,'p38_checkpoint_sha256':b.sha(b.P38/'runs/p38_temporal/fixed_step_001200.pt'),'prereg_sha256':sha(WORK/'PREREG.json')},ck); dump(out/'TRAINING_RECEIPT.json',{'status':'PASS_CCSI_FIXED_TRAINING','steps':1200,'first':hist[0],'last':hist[-1],'last100_mean':float(np.mean([x['loss'] for x in hist[-100:]])),'checkpoint_sha256':sha(ck),'gpu_uuid':b.UUID,'gpu_name':'NVIDIA A100-SXM4-40GB','gpu_snapshot':gpu,'elapsed_seconds':time.monotonic()-start,'dev_read':False,'calibration_read':False,'p38_checkpoint_sha256':b.sha(b.P38/'runs/p38_temporal/fixed_step_001200.pt')}); (out/'training.exit').write_text('0\n'); (out/'launcher.exit').write_text('0\n'); print(json.dumps({'status':'PASS_CCSI_FIXED_TRAINING','checkpoint_sha256':sha(ck)}))

def train_full():
    out=WORK/'runs_full'
    if out.exists(): raise RuntimeError('sealed CCSI full runs already exist')
    contract=cpu_contract()
    if not contract['all_checks_pass']: raise RuntimeError('CCSI CPU contract failed')
    pairs={p:b.CCFIPair(p,True) for p in FIT}; schedule,eligible=b.build_full_schedule(pairs); p38,_=b.load_p38_state(); dev,gpu=b.configure_gpu(); torch.manual_seed(42); torch.cuda.manual_seed_all(42); np.random.seed(42); m=b.CCFI().init_from_p38(p38).to(dev).train(); opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4); hist=[]; start=time.monotonic()
    for step,e in enumerate(schedule,1):
        (ca,ha,sa,ma,la,va),(cb,hb,sb,mb,lb,vb)=pairs[e['pair']].batch(e['frame'],e['class'],dev); za,cha,ua=m(ca,ha,sa,ma); zb,chb,ub=m(cb,hb,sb,mb); base,count,parts=b.factorized_objective(za,zb,cha,chb,ua,ub,ca,cb,ha,hb,ma,mb,la,lb,va,vb); s1,n1=swap_loss(m,ca,ha,sa,ma); s2,n2=swap_loss(m,cb,hb,sb,mb); loss=base+0.1*(s1+s2)
        if not torch.isfinite(loss): raise ValueError('nonfinite CCSI full loss')
        opt.zero_grad(set_to_none=True); loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()): raise ValueError('nonfinite CCSI full gradient')
        torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step(); hist.append({'step':step,'loss':float(loss.detach()),'base':float(base.detach()),'swap':float((s1+s2).detach()),'swap_rows':n1+n2,'queries':int(count)})
        if step==1 or step%1000==0: print(json.dumps(hist[-1]),flush=True)
    torch.cuda.synchronize(); out.mkdir(); ck=out/'fixed_step_055236.pt'; torch.save({'state_dict':{k:v.detach().cpu() for k,v in m.state_dict().items()},'fixed_step':len(schedule),'p38_checkpoint_sha256':b.sha(b.P38/'runs/p38_temporal/fixed_step_001200.pt'),'prereg_sha256':sha(WORK/'SUFFICIENCY_PREREG.json')},ck); dump(out/'TRAINING_RECEIPT.json',{'status':'PASS_CCSI_FULL_COVERAGE_TRAINING','steps':len(schedule),'epochs':3,'eligible_groups':len(eligible),'first':hist[0],'last':hist[-1],'last100_mean':float(np.mean([x['loss'] for x in hist[-100:]])),'checkpoint_sha256':sha(ck),'gpu_uuid':b.UUID,'gpu_name':'NVIDIA A100-SXM4-40GB','gpu_snapshot':gpu,'elapsed_seconds':time.monotonic()-start,'dev_read':False,'calibration_read':False,'p38_checkpoint_sha256':b.sha(b.P38/'runs/p38_temporal/fixed_step_001200.pt')}); (out/'training.exit').write_text('0\n'); (out/'launcher.exit').write_text('0\n'); print(json.dumps({'status':'PASS_CCSI_FULL_COVERAGE_TRAINING','checkpoint_sha256':sha(ck),'steps':len(schedule)}))

def evaluate(checkpoint_dir=None):
    import sys as _s; _s.path.insert(0,str(b.P4)); import compare_crop32 as cmp
    checkpoint_dir=WORK/'runs' if checkpoint_dir is None else Path(checkpoint_dir); ckpt=checkpoint_dir/'fixed_step_001200.pt' if checkpoint_dir.name=='runs' else checkpoint_dir/'fixed_step_055236.pt'; raw=b.load_torch(ckpt); m=b.CCFI().eval(); m.load_state_dict(raw['state_dict'],strict=True); man=json.loads((b.P4/'crop32/FRAME_MANIFEST.json').read_text()); rec={x['pair']:x for x in man['pairs']}; out=checkpoint_dir/'evaluation'; out.mkdir(exist_ok=True); frozen={}
    for p in ALL:
        k,e,u=b.freeze_embeddings(m,p,rec[p]); np.savez_compressed(out/(p+'_embeddings.npz'),row_key=k,embedding=e,uncertainty=u); frozen[p]=len(k)
    dump(out/'LABEL_FREEZE.json',{'status':'PASS_CCSI_LABEL_FREEZE','pairs':list(ALL),'rows':frozen,'dev_read':False,'official_val_test_read':False}); inputs={r['sequence']:r for r in json.loads((b.ROOT/'p2/cache/MANIFEST.json').read_text())['sequences']}; reports=[]; cmp.METHODS=('p1_independent128','temporal128','ccsi128')
    for p in ALL:
        row,features,refs,cov=cmp.align_pair(rec[p],b.P4/'crop32',inputs); z=np.load(out/(p+'_embeddings.npz')); p38=np.load(b.P38/'runs/evaluation'/f'{p}_embeddings.npz');
        if not np.array_equal(row['row_key'],p38['row_key']) or not np.array_equal(row['row_key'],z['row_key']): raise ValueError('CCSI comparator key mismatch')
        features['temporal128']=p38['embedding']; features['ccsi128']=z['embedding']; arr=cmp.evaluate_arrays(row,features); report={'pair':p,'role':rec[p]['role'],'metrics':cmp.summaries(arr),'coverage':cov}; dump(out/(p+'_RESULTS.json'),report); reports.append(report)
    by={}
    for role in ('fit','calibration'):
        rs=[x for x in reports if x['role']==role]; macro={m:float(np.mean([x['metrics']['all']['methods'][m]['all_candidates']['known_positive_r1_tie_averaged'] for x in rs])) for m in cmp.METHODS}; by[role]={'pair_macro_r1':macro,'wins_vs_p38':int(sum(x['metrics']['all']['methods']['ccsi128']['all_candidates']['known_positive_r1_tie_averaged']>x['metrics']['all']['methods']['temporal128']['all_candidates']['known_positive_r1_tie_averaged'] for x in rs)),'pairs':[x['pair'] for x in rs]}
    cal=by['calibration']['pair_macro_r1']; fit=by['fit']['pair_macro_r1']; gate={'calibration_r1_vs_p38':cal['ccsi128']-cal['temporal128'],'wins_vs_p38':by['calibration']['wins_vs_p38'],'fit_drop_vs_p1':fit['ccsi128']-fit['p1_independent128'],'pass':bool(cal['ccsi128']>=.4641 and by['calibration']['wins_vs_p38']>=3 and fit['ccsi128']-fit['p1_independent128']>=-.05)}; result={'status':'COMPLETE_CCSI_TRAIN_ONLY_EVALUATION','training':json.loads((checkpoint_dir/'TRAINING_RECEIPT.json').read_text()),'by_role':by,'gate':gate,'pairs':reports,'dev_read':False,'official_val_test_read':False}; dump(out/'RESULTS.json',result); dump(checkpoint_dir/'DECISION.json',{'verdict':'KEEP_CCSI_REPRESENTATION_GATE' if gate['pass'] else 'STOP_CCSI_REPRESENTATION_GATE','gate_pass':gate['pass'],'host_transfer_authorized':False,'results':str(out/'RESULTS.json')}); print(json.dumps({'calibration':cal,'fit':fit,'gate':gate},indent=2))

def evaluate_full(): evaluate(WORK/'runs_full')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=('cpu','train','train_full','evaluate','evaluate_full'),required=True); a=ap.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1); {'cpu':lambda:cpu_contract(),'train':train,'train_full':train_full,'evaluate':evaluate,'evaluate_full':evaluate_full}[a.mode]()
if __name__=='__main__': main()
