#!/usr/bin/env python3
from __future__ import annotations
import json,numpy as np
from pathlib import Path
P=Path(__file__).resolve().parents[1]; R=json.loads((P/'NATIVE_FACTORIAL.json').read_text()); eval_ids=R['eval_pairs']
def arm(seed,name): return next(x for x in R['arms'][name] if x['seed']==seed)['per_pair']
summary={}
for seed in (7,17):
 rel=arm(seed,'rel_on_teacher_off'); pooled=arm(seed,'pooled_teacher_off'); ton=arm(seed,'rel_on_teacher_on')
 relb=np.array([rel[s]['student_recall1'] for s in eval_ids]); base=np.array([rel[s]['base_recall1'] for s in eval_ids]); pool=np.array([pooled[s]['student_recall1'] for s in eval_ids]); teach=np.array([ton[s]['student_recall1'] for s in eval_ids])
 summary[str(seed)]={'rel_vs_base':{'macro_r1_delta':float((relb-base).mean()),'wins':int((relb>base).sum()),'per_pair':(relb-base).tolist()},'rel_vs_pooled':{'macro_r1_delta':float((relb-pool).mean()),'wins':int((relb>pool).sum()),'per_pair':(relb-pool).tolist()},'teacher_on_vs_off':{'macro_r1_delta':float((teach-relb).mean()),'wins':int((teach>relb).sum()),'per_pair':(teach-relb).tolist()},'rel_dustbin':float(np.mean([rel[s]['dustbin_accuracy'] for s in eval_ids])),'pooled_dustbin':float(np.mean([pooled[s]['dustbin_accuracy'] for s in eval_ids]))}
keep_feasible=all(summary[str(s)]['rel_vs_base']['wins']>=4 for s in (7,17)) and all(summary[str(s)]['rel_vs_base']['macro_r1_delta']>0.05 for s in (7,17))
set_gate=all(summary[str(s)]['rel_vs_pooled']['wins']>=3 for s in (7,17))
teacher_gate=all(summary[str(s)]['teacher_on_vs_off']['macro_r1_delta']>=0.01 for s in (7,17))
payload={'status':'HOLD_P50_SET_TEACHER_NOVELTY_GATE','summary':summary,'pre_registered_gate':{'relational_operator_keep':set_gate,'teacher_additive_keep':teacher_gate,'native_residual_feasibility':keep_feasible},'decision':{'native_candidate_residual':'KEEP_AS_STRONG_TRAINED_ADAPTER_CONTROL','leave_one_out_set_claim':'HOLD_NOT_CAUSAL_IN_BOTH_SEEDS','privileged_teacher_claim':'HOLD_NO_ADDITIVE_GAIN','official_val_test_authorized':False,'next_research_step':'counterfactual positive-removal dustbin objective with cardinality-matched negative-removal control before any official-val integration'},'metrics_scope':'held-out train calibration only; no official MOT metric'}
(P/'GATE_DECISION.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n');print(json.dumps(payload,indent=2,ensure_ascii=False))
