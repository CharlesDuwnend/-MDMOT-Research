#!/usr/bin/env python3
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import run_ccfi as r

def run():
    pairs={p:r.CCFIPair(p,True) for p in r.FIT}; schedule=r.build_schedule(pairs); st,_=r.load_p38_state(); dev,_=r.configure_gpu(); torch.manual_seed(42); torch.cuda.manual_seed_all(42); np.random.seed(42)
    m=r.CCFI().init_from_p38(st).to(dev).train(); opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4); first=None; trace=[]
    for step,event in enumerate(schedule,1):
        (ca,ha,sa,ma,la,va),(cb,hb,sb,mb,lb,vb)=pairs[event['pair']].batch(event['frame'],event['class'],dev)
        za,cha,ua=m(ca,ha,sa,ma); zb,chb,ub=m(cb,hb,sb,mb); loss,count,parts=r.factorized_objective(za,zb,cha,chb,ua,ub,ca,cb,ha,hb,ma,mb,la,lb,va,vb)
        opt.zero_grad(set_to_none=True); loss.backward()
        stats={n:{'finite':bool(p.grad is not None and torch.isfinite(p.grad).all()),'max':None if p.grad is None else float(torch.nan_to_num(p.grad.detach(),nan=0.0,posinf=0.0,neginf=0.0).abs().max().cpu())} for n,p in m.named_parameters()}
        bad=[n for n,x in stats.items() if not x['finite']]
        trace.append({'step':step,'pair':event['pair'],'frame':event['frame'],'class':event['class'],'loss':float(loss.detach().cpu()),'parts':{k:float(v.detach().cpu()) for k,v in parts.items()},'bad':bad,'stats':stats})
        if bad:
            first=trace[-1]; break
        torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
    out={'status':'FOUND_CCFI_NONFINITE_GRADIENT_V3','first_failure':first,'checked_steps':len(trace),'pre_failure_finite':all(not x['bad'] for x in trace[:-1]),'trace_tail':trace[-3:]}
    (r.WORK/'IMPLEMENTATION_FAILURE_V3_TRACE.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': run()
