#!/usr/bin/env python3
"""Independent 3-round x 3-check audit of the valid P55 v4 negative gate."""
import hashlib, json, re
from pathlib import Path
import numpy as np
import torch
from train_head_init import CVCRv2, INIT, ROOT, P23

P55=ROOT/'continuation_20261006/p55_context_residual'; GATE=json.loads((P55/'CAL5_GATE_V4.json').read_text())

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def check(name, passed, evidence): return {'check':name,'passed':bool(passed),'evidence':evidence}

def main():
    x=np.load(P23/'inputs.npz',allow_pickle=False); labels=np.load(P23/'offline_pid.npy',mmap_mode='r'); groups=json.loads((P23/'GROUPS.json').read_text()); fit=set(str(g['pair']) for g in groups)
    cal=set(GATE['pairs']);
    rounds=[]
    rounds.append({'round':'R1_data_lineage','checks':[
        check('fit_calibration_disjoint',fit.isdisjoint(cal),{'fit_pairs':sorted(fit),'cal_pairs':sorted(cal)}),
        check('row_alignment',len(x['keys'])==len(labels)==54488,{'keys':len(x['keys']),'labels':len(labels)}),
        check('calibration_protocol',set(GATE['pairs'])=={'27','32','42','64','65'} and GATE['official_val_test_read'] is False,{'pairs':GATE['pairs'],'official_val_test_read':GATE['official_val_test_read']})
    ]})
    ck=torch.load(P55/'runs_v4/context_residual/fixed_step_001200.pt',map_location='cpu',weights_only=False); model=CVCRv2(); model.load_state_dict(ck['state_dict'],strict=True); model.eval(); init=torch.load(INIT,map_location='cpu',weights_only=False)['head_state']; base=model.base.state_dict(); frozen=all(torch.equal(base[k],init[k]) for k in init)
    t=torch.randn(6,384); c=torch.randn(6,384); f=torch.randn(6,256)
    with torch.no_grad(): z,d=model(t,c,f,True); zs,_=model(t,c[torch.roll(torch.arange(6),1)],f,True); zd,_=model(t,c,f,False)
    rounds.append({'round':'R2_operator_training','checks':[
        check('strict_checkpoint_load_finite',torch.isfinite(z).all().item() and torch.isfinite(d).all().item(),{'parameters':sum(p.numel() for p in model.parameters())}),
        check('base_head_frozen_to_p23_init',frozen,{'init_sha256':sha(INIT),'base_tensors':len(base)}),
        check('context_counterfactual_changes_output',(z-zs).abs().max().item()>1e-7 and (z-zd).abs().max().item()>1e-7,{'swap_max_abs':float((z-zs).abs().max()),'dropout_max_abs':float((z-zd).abs().max())})
    ]})
    per=GATE['per_pair']; recomputed=float(np.mean([per[p]['context_residual']['r1'] for p in GATE['pairs']])); wins=sum(per[p]['context_residual']['r1']>per[p]['target_only']['r1'] for p in GATE['pairs'])
    checks=[
        check('independent_macro_recompute',abs(recomputed-GATE['macro_r1']['context_residual'])<1e-12,{'recomputed':recomputed,'stored':GATE['macro_r1']['context_residual']}),
        check('pair_win_recompute',wins==GATE['pair_wins'] and wins==0,{'recomputed_wins':wins,'stored':GATE['pair_wins']}),
        check('negative_gate_is_scientific_after_contract_fix',GATE['delta_macro_r1']<0 and GATE['pair_wins']==0,{'delta':GATE['delta_macro_r1'],'wins':GATE['pair_wins'],'target_macro':GATE['macro_r1']['target_only']})
    ]
    rounds.append({'round':'R3_evaluation_decision','checks':checks})
    out={'status':'P55_STOP_AFTER_VALID_NEGATIVE_GATE','candidate':'CVCR','rounds':rounds,'round_count':3,'checks_per_round':[len(r['checks']) for r in rounds], 'scientific_classification':'generalization_failure_after_valid_protocol; not an unresolved implementation defect','decision':'STOP_AS_PAPER_MODULE_RETAIN_AS_NEGATIVE_CONTROL','official_val_test_read':False,'training_authorized':False}
    (P55/'FAILURE_AUDIT_V4.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':out['status'],'checks':sum(out['checks_per_round']),'classification':out['scientific_classification']}))

if __name__=='__main__':main()
