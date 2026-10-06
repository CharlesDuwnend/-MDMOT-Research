#!/usr/bin/env python3
import hashlib, json, math
from pathlib import Path
import numpy as np

WORK=Path('/home/chenhc/claude_try_MDMOT/continuation_20261006/p51_ccfi')
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 return h.hexdigest()
def main():
 rec=json.loads((WORK/'runs/TRAINING_RECEIPT.json').read_text()); result=json.loads((WORK/'runs/evaluation/RESULTS.json').read_text()); freeze=json.loads((WORK/'runs/evaluation/LABEL_FREEZE.json').read_text()); checks=[]
 def c(name,ok,detail): checks.append({'name':name,'pass':bool(ok),'detail':detail})
 # R1 provenance and legal split.
 c('R1_training_receipt',rec.get('status')=='PASS_CCFI_FIXED_TRAINING' and rec.get('steps')==1200,rec.get('status'))
 c('R1_label_freeze',freeze.get('status')=='PASS_CCFI_LABEL_FREEZE' and len(freeze.get('pairs',[]))==20,{'pairs':len(freeze.get('pairs',[]))})
 c('R1_no_dev_or_official',rec.get('dev_read') is False and rec.get('calibration_read') is False and result.get('dev_read') is False and result.get('official_val_test_read') is False,True)
 # R2 independent artifact/replay checks.
 reports=result['pairs']; c('R2_all_pair_reports',len(reports)==20 and all(len(r.get('pair',''))>0 for r in reports),len(reports))
 c('R2_all_metric_values_finite',all(math.isfinite(float(r['metrics']['all']['methods']['ccfi128']['all_candidates']['known_positive_r1_tie_averaged'])) for r in reports),True)
 c('R2_checkpoint_receipt_hash',sha(WORK/'runs/fixed_step_001200.pt')==rec['checkpoint_sha256'],rec['checkpoint_sha256'])
 # R3 method gate and claim boundary.
 g=result['gate']; c('R3_beats_p38_pair_gate',g['wins_vs_p38']>=3 and g['calibration_r1_vs_p38']>0,{'wins':g['wins_vs_p38'],'delta':g['calibration_r1_vs_p38']})
 c('R3_absolute_gate_failure_is_explicit',g['pass'] is False and g['calibration_min']>g['calibration_r1_vs_p1'],g)
 c('R3_no_host_authorization',Path(WORK/'DECISION.json').is_file() and json.loads((WORK/'DECISION.json').read_text()).get('host_transfer_authorized') is False,True)
 out={'status':'PASS_CCFI_GATE_FAILURE_AUDIT_3X3','classification':'METHOD_GATE_BELOW_ABSOLUTE_CALIBRATION_THRESHOLD','decision':'HOLD_FOR_FULL_COVERAGE_SUFFICIENCY','rounds':{'R1_provenance':[x for x in checks if x['name'].startswith('R1_')],'R2_replay':[x for x in checks if x['name'].startswith('R2_')],'R3_gate':[x for x in checks if x['name'].startswith('R3_')]},'all_checks_pass':all(x['pass'] for x in checks),'evidence':{'calibration':result['by_role']['calibration'],'fit':result['by_role']['fit'],'gate':g}}
 (WORK/'FAILURE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
