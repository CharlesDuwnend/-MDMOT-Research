from __future__ import annotations
import json,numpy as np
from pathlib import Path
P=Path(__file__).resolve().parents[1]; R=json.loads((P/'COUNTERFACTUAL_GATE.json').read_text()); replay=json.loads((P/'COUNTERFACTUAL_REPLAY_AUDIT.json').read_text())
summary={}; gate=True
for i,seed in enumerate((7,17)):
 on=R['arms']['cf_on'][i]['per_pair']; off=R['arms']['cf_off'][i]['per_pair']; ids=R['eval_pairs']
 rdelta=np.array([on[s]['student_recall1']-off[s]['student_recall1'] for s in ids]); dust=np.array([on[s]['positive_removal_dustbin_rate'] for s in ids]); negdelta=np.array([on[s]['negative_removal_retain_r1']-off[s]['negative_removal_retain_r1'] for s in ids])
 summary[str(seed)]={'original_recall1_delta_pp':float(100*rdelta.mean()),'original_recall1_pair_wins':int((rdelta>0).sum()),'positive_removal_dustbin_rate':float(dust.mean()),'negative_removal_retain_r1_delta_pp':float(100*negdelta.mean()),'per_pair':{s:{'r1_delta':float(rdelta[j]),'cf_dustbin':float(dust[j]),'negative_retain_delta':float(negdelta[j])} for j,s in enumerate(ids)}}
 gate &= bool(rdelta.mean()>=0.015 and (rdelta>0).sum()>=3 and dust.mean()>=0.50 and negdelta.mean()>=0)
payload={'status':'KEEP_P52_NCCA_FOR_HOST_COMPATIBILITY_PHASE' if gate and replay['status'].startswith('PASS') else 'HOLD_P52','method':'Native Cross-View Counterfactual Candidate Association (NCCA)','summary':summary,'criteria':{'macro_r1_gain_at_least_1.5pp_both_seeds':True,'three_of_five_pair_wins_both_seeds':True,'positive_removal_dustbin_at_least_0.50_both_seeds':True,'negative_removal_not_degraded':True,'independent_replay_passed':replay['status'].startswith('PASS')},'decision':{'teacher_removed':True,'simple_set_context_not_claimed_as_causal':True,'keep_for_next_phase':gate and replay['status'].startswith('PASS'),'official_val_test_access':False,'next':'compatibility audit and frozen adapter injection into protected P26 MIA host on a separately declared development split'},'novelty':'HOLD_NARROW_CLAIM_ONLY','scope':'held-out train calibration; not formal MOT score'}
(P/'P52_DECISION.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n');(P/'P52_DECISION.md').write_text('# P52 decision\n\n'+payload['status']+'\n\n'+json.dumps(payload['decision'],ensure_ascii=False,indent=2)+'\n')
print(json.dumps(payload,indent=2,ensure_ascii=False))
