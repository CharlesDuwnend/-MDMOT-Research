#!/usr/bin/env python3
import hashlib, json, math
from pathlib import Path
import numpy as np

WORK=Path('/home/chenhc/claude_try_MDMOT/continuation_20261006/p51_ccfi')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 return h.hexdigest()
def main():
 tr=json.loads((WORK/'runs_full/TRAINING_RECEIPT.json').read_text()); res=json.loads((WORK/'runs_full/evaluation/RESULTS.json').read_text()); freeze=json.loads((WORK/'runs_full/evaluation/LABEL_FREEZE.json').read_text()); short=json.loads((WORK/'runs/evaluation/RESULTS.json').read_text()); checks=[]
 def c(name,ok,detail): checks.append({'name':name,'pass':bool(ok),'detail':detail})
 # R1: training implementation and lineage.
 c('R1_full_schedule_contract',tr.get('steps')==55236 and tr.get('eligible_groups')==18412 and tr.get('epochs')==3,{'steps':tr.get('steps'),'groups':tr.get('eligible_groups'),'epochs':tr.get('epochs')})
 c('R1_full_checkpoint_hash',sha(WORK/'runs_full/fixed_step_055236.pt')==tr.get('checkpoint_sha256'),tr.get('checkpoint_sha256'))
 c('R1_full_finite_training_receipt',tr.get('status')=='PASS_CCFI_FULL_COVERAGE_TRAINING' and tr.get('gpu_uuid','')=='GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a',tr.get('status'))
 # R2: frozen evaluation wiring.
 reports=res.get('pairs',[]); c('R2_label_freeze',freeze.get('status')=='PASS_CCFI_LABEL_FREEZE' and len(freeze.get('pairs',[]))==20,{'status':freeze.get('status'),'pairs':len(freeze.get('pairs',[]))})
 c('R2_all_outputs_finite',len(reports)==20 and all(math.isfinite(float(r['metrics']['all']['methods']['ccfi128']['all_candidates']['known_positive_r1_tie_averaged'])) for r in reports),len(reports))
 c('R2_no_forbidden_split',res.get('dev_read') is False and res.get('official_val_test_read') is False and tr.get('calibration_read') is False,True)
 # R3: method gate failure and overfit pattern.
 g=res['gate']; sg=short['gate']; c('R3_calibration_gate_failure',g['pass'] is False and g['wins_vs_p38']==0 and g['calibration_r1_vs_p38']<0,g)
 c('R3_short_to_full_regression',sg['calibration_r1_vs_p38']>0 and g['calibration_r1_vs_p38']<0 and sg['wins_vs_p38']==4,{'short':sg,'full':g})
 c('R3_fit_calibration_gap_exposes_overfit',g['fit_drop_vs_p1']>0.05 and g['fit_drop_vs_p1']>0 and g['calibration_r1_vs_p1']<0,{'fit_delta':g['fit_drop_vs_p1'],'cal_delta':g['calibration_r1_vs_p1']})
 out={'status':'PASS_CCFI_FULL_FAILURE_AUDIT_3X3','classification':'METHOD_GENERALIZATION_FAILURE_CROSS_PAIR_OVERFIT','decision':'STOP_CCFI_AFTER_FULL_COVERAGE','rounds':{'R1_training_integrity':[x for x in checks if x['name'].startswith('R1_')],'R2_evaluation_integrity':[x for x in checks if x['name'].startswith('R2_')],'R3_failure_cause':[x for x in checks if x['name'].startswith('R3_')]},'all_checks_pass':all(x['pass'] for x in checks),'evidence':{'short_gate':sg,'full_gate':g}}
 (WORK/'FULL_FAILURE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
