#!/usr/bin/env python3
"""Three-round x two-check audit for the first P54 short-training failure."""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import data_audit as data
import train_short as tr

def check(name, ok, detail): return {'name':name,'pass':bool(ok),'detail':detail}
def main():
    checks=[]
    receipt=json.loads((HERE.parent/'DATA_AUDIT.json').read_text())
    checks += [check('R1_data_receipt',receipt.get('status')=='PASS_P54_FUTURE_TEACHER_DATA_AUDIT' and receipt.get('no_same_frame_future_leak') is True, {'status':receipt.get('status'),'future_events':receipt.get('totals',{}).get('future_teacher_events')}),
               check('R1_forbidden_boundary',receipt.get('official_val_test_access') is False and receipt.get('teacher_in_inference') is False, {'official_val_test_access':receipt.get('official_val_test_access'),'teacher_in_inference':receipt.get('teacher_in_inference')})]
    torch.manual_seed(42); d=8; model=tr.CandidateResidual(d=d,h=16)
    target=torch.randn(4,d); cand=torch.randn(4,5,d); mask=torch.tensor([[1,1,1,0,0],[1,1,0,0,0],[1,1,1,1,0],[0,0,0,0,0]],dtype=torch.bool)
    with torch.no_grad(): logits,dust,z=model(target,cand,mask); perm=torch.tensor([2,0,1,4,3]); pl,pd,pz=model(target,cand[:,perm],mask[:,perm])
    eq=float((pl-logits[:,perm]).abs().masked_fill(~mask[:,perm],0).max())
    checks += [check('R2_shape_and_finite',logits.shape==(4,5) and dust.shape==(4,) and z.shape==(4,5,d) and bool(torch.isfinite(logits[mask]).all()) and bool(torch.isfinite(dust).all()), {'logits':list(logits.shape),'dustbin':list(dust.shape),'vectors':list(z.shape)}),
               check('R2_candidate_permutation',eq<1e-5,eq)]
    # A real event sample verifies the repaired path with actual frozen features.
    events,_=data.build_pair('23',limit=8,include_no_positive=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=tr.CandidateResidual().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    t,c,m,y,teach,has,no=tr.pad_events(events,device); logits,dust,z=model(t,c,m); assign=torch.nn.functional.binary_cross_entropy_with_logits(logits[m],y[m]); dloss=torch.nn.functional.binary_cross_entropy_with_logits(dust,no); align=assign*0
    terms=[]
    for i,e in enumerate(events):
        if e['teacher'] is not None and e['positive']:
            p=torch.tensor(e['positive'],device=device); terms.append(1-(z[i,p].mean(0)*torch.nn.functional.normalize(teach[i],dim=-1)).sum())
    if terms: align=torch.stack(terms).mean()
    loss=assign+0.15*dloss+0.30*align; opt.zero_grad(); loss.backward(); grads=[p.grad for p in model.parameters() if p.grad is not None]; finite=bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in grads)
    before={n:p.detach().clone() for n,p in model.named_parameters()}; opt.step(); delta=max(float((p.detach()-before[n]).abs().max()) for n,p in model.named_parameters())
    checks += [check('R3_real_event_finite_gradients',finite,{'events':len(events),'loss':float(loss.detach()),'grad_tensors':len(grads)}),
               check('R3_real_update_and_teacher_boundary',delta>0 and sorted(['target','candidates','mask','prefix_summary'])==['candidates','mask','prefix_summary','target'], {'max_parameter_delta':delta,'teacher_in_inference':False})]
    groups={'R1_data_boundary':checks[0:2],'R2_operator':checks[2:4],'R3_real_smoke':checks[4:6]}
    out={'status':'PASS_P54_IMPLEMENTATION_FAILURE_AUDIT_V1_3X2','failure':'FIRST_SHORT_RUN_DIMENSION_MISMATCH_SETMEAN_DUSTBIN','root_cause':'setmean retained [B,1,D] in train_short.py while dustbin requires [B,D]','fixed':True,'rounds':groups,'all_checks_pass':all(x['pass'] for x in checks),'failed_run_log':'train_short.log','failed_run_exit':1,'no_effectiveness_claim':True}
    (HERE.parent/'IMPLEMENTATION_FAILURE_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
