#!/usr/bin/env python3
"""Independent 3 x 3 replay of the train-free temporal signal gate."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SIGNAL=ROOT/'TEMPORAL_EVIDENCE_SIGNAL_AUDIT.json'
CPU=ROOT/'prototype/CPU_CONTRACT.json'

def check(name,ok,evidence):
 if not ok: raise AssertionError(name+': '+str(evidence))
 return {'check':name,'status':'PASS','evidence':evidence}

def main():
 x=json.loads(SIGNAL.read_text()); c=x['macro']; pairs=x['pairs']; by=x['metrics_by_pair']
 rounds=[]
 rounds.append({'round':'R1_FROZEN_DATA_AND_LEGAL_INPUT','checks':[
  check('exact_primary_pair_set',pairs==['23','25','27','28','29','30','32','39'],{'pairs':pairs}),
  check('feature_rows_are_aligned_by_frozen_stage15_audit',all(by[p]['rows']>0 for p in pairs),{'rows_by_pair':{p:by[p]['rows'] for p in pairs}}),
  check('no_official_access_or_parameter_updates',x['protocol'].endswith('no val/test') and x['official_val_test_access'] is False,{'protocol':x['protocol']})
 ]})
 # Replay all reported macro values directly from per-pair deltas.
 replay={}
 for field in ('static_cos','delta_cos','delta_discrepancy'):
  vals=[by[p][field]['delta_pos_minus_neg'] for p in pairs]
  replay[field]={'mean':sum(vals)/len(vals),'positive':sum(v>0 for v in vals),'finite':all(math.isfinite(v) for v in vals)}
 rounds.append({'round':'R2_INDEPENDENT_SIGNAL_REPLAY','checks':[
  check('all_pair_signal_deltas_finite',all(replay[f]['finite'] for f in replay),replay),
  check('static_control_is_positive_on_all_pairs',replay['static_cos']['positive']==8 and abs(replay['static_cos']['mean']-c['static_cos']['mean_delta'])<1e-12,replay['static_cos']),
  check('differential_signal_replay_matches_file',all(abs(replay[f]['mean']-c[f]['mean_delta'])<1e-12 and replay[f]['positive']==c[f]['pair_positive'] for f in replay),replay)
 ]})
 rounds.append({'round':'R3_PRETRAIN_DECISION','checks':[
  check('cpu_mechanism_contract_passed',json.loads(CPU.read_text())['status']=='PASS_CVTDCF_CPU_MECHANISM_CONTRACT',{'cpu_status':json.loads(CPU.read_text())['status']}),
  check('differential_gate_fails',c['delta_cos']['pair_positive']<5 and c['delta_cos']['mean_delta']<=0,{'delta_cos':c['delta_cos']}),
  check('static_does_not_authorize_differential_claim',c['static_cos']['pair_positive']==8 and c['delta_cos']['pair_positive']==3,{'static':c['static_cos'],'differential':c['delta_cos']})
 ]})
 payload={'status':'PASS_CVTDCF_SIGNAL_FAILURE_AUDIT_3X3','candidate':'Cross-View Temporal Differential Co-Movement Field','decision':'STOP_CVTDCF_BEFORE_TRAINING_WEAK_DIFFERENTIAL_SIGNAL','rounds':rounds,'signal_sha256':hashlib.sha256(SIGNAL.read_bytes()).hexdigest(),'cpu_contract_sha256':hashlib.sha256(CPU.read_bytes()).hexdigest(),'official_val_test_access':False,'next':'do not build/train this candidate; retain static-similarity result as diagnostic and screen a different representation objective'}
 (ROOT/'TEMPORAL_EVIDENCE_FAILURE_AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
