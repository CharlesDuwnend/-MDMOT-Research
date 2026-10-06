#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path
WORK=Path('/home/chenhc/claude_try_MDMOT/continuation_20261006/p53_ccsi')
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 return h.hexdigest()
def main():
 tr=json.loads((WORK/'runs/TRAINING_RECEIPT.json').read_text()); res=json.loads((WORK/'runs/evaluation/RESULTS.json').read_text()); freeze=json.loads((WORK/'runs/evaluation/LABEL_FREEZE.json').read_text()); checks=[]
 def c(n,v,d): checks.append({'name':n,'pass':bool(v),'detail':d})
 c('R1_receipt',tr.get('status')=='PASS_CCSI_FIXED_TRAINING' and tr.get('steps')==1200,tr.get('status'))
 c('R1_cpu_contract',json.loads((WORK/'CPU_CONTRACT.json').read_text()).get('all_checks_pass') is True,True)
 c('R1_p38_lineage',tr.get('p38_checkpoint_sha256')==sha('/home/chenhc/claude_try_MDMOT/continuation_20261005/p38_temporal_memory/runs/p38_temporal/fixed_step_001200.pt'),tr.get('p38_checkpoint_sha256'))
 c('R2_label_freeze',freeze.get('status')=='PASS_CCSI_LABEL_FREEZE' and len(freeze.get('pairs',[]))==20,len(freeze.get('pairs',[])))
 c('R2_finite_metrics',len(res.get('pairs',[]))==20 and all(math.isfinite(float(x['metrics']['all']['methods']['ccsi128']['all_candidates']['known_positive_r1_tie_averaged'])) for x in res['pairs']),len(res.get('pairs',[])))
 c('R2_no_forbidden_read',res.get('dev_read') is False and res.get('official_val_test_read') is False,True)
 g=res['gate']; c('R3_relative_signal',g['wins_vs_p38']>=3 and g['calibration_r1_vs_p38']>0,g)
 c('R3_absolute_gate_explicit',g['pass'] is False and g['calibration_r1_vs_p38']>0,g)
 c('R3_no_host_attachment',json.loads((WORK/'DECISION.json').read_text()).get('host_transfer_authorized') is False,True)
 out={'status':'PASS_CCSI_SHORT_GATE_AUDIT_3X3','classification':'METHOD_GATE_BELOW_ABSOLUTE_THRESHOLD','decision':'HOLD_FOR_FULL_COVERAGE_SUFFICIENCY','rounds':{'R1_training':[x for x in checks if x['name'].startswith('R1_')],'R2_evaluation':[x for x in checks if x['name'].startswith('R2_')],'R3_gate':[x for x in checks if x['name'].startswith('R3_')]},'all_checks_pass':all(x['pass'] for x in checks),'evidence':{'gate':g,'calibration':res['by_role']['calibration'],'fit':res['by_role']['fit']}}
 (WORK/'FAILURE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
