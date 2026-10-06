#!/usr/bin/env python3
"""Nine-check audit register: three rounds, three checks each."""
from __future__ import annotations
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[1]
R1=json.loads((P/'IMPLEMENTATION_FAILURE_AUDIT.json').read_text())
R2=json.loads((P/'DATA_CONTRACT_AUDIT.json').read_text())
R3=json.loads((P/'POST_RUN_IMPLEMENTATION_AUDIT.json').read_text())
rounds=[
 {'round':'R1_failure_fix','checks':[{'id':'R1.1','source':'IMPLEMENTATION_FAILURE_AUDIT.json','check':'stack root cause and source fix','status':R1['audits'][0]['status']},{'id':'R1.2','source':'IMPLEMENTATION_FAILURE_AUDIT.json','check':'K=0/1/19/64 finite gradient edge test','status':R1['audits'][1]['status']},{'id':'R1.3','source':'IMPLEMENTATION_FAILURE_AUDIT.json','check':'fresh two-pair eight-arm smoke exit 0 and zero drops','status':R1['audits'][2]['status']}]},
 {'round':'R2_data_contract','checks':[{'id':'R2.1','source':'DATA_CONTRACT_AUDIT.json','check':'episode line counts equal trainer events and zero drops','status':R2['checks'][0]['status']},{'id':'R2.2','source':'DATA_CONTRACT_AUDIT.json','check':'all zero-candidate rows retained','status':R2['checks'][1]['status']},{'id':'R2.3','source':'DATA_CONTRACT_AUDIT.json','check':'40 native shards inference-safe, 1,372,492 rows','status':R2['checks'][2]['status']}]},
 {'round':'R3_post_run','checks':[{'id':'R3.1','source':'POST_RUN_IMPLEMENTATION_AUDIT.json','check':'all eight checkpoint hashes and parameters finite','status':R3['checks'][2]['status']},{'id':'R3.2','source':'POST_RUN_IMPLEMENTATION_AUDIT.json','check':'same event and matched denominators across arms/seeds','status':'PASS' if all(c['name'].startswith('equal_event_denominator_') and c['status']=='PASS' for c in R3['checks'] if c['name'].startswith('equal_event_denominator_')) else 'FAIL'},{'id':'R3.3','source':'POST_RUN_IMPLEMENTATION_AUDIT.json','check':'step-zero cosine and teacher absent from forward','status':'PASS' if R3['checks'][-2]['status']=='PASS' and R3['checks'][-1]['status']=='PASS' else 'FAIL'}]}
]
allpass=all(c['status']=='PASS' for r in rounds for c in r['checks'])
out={'status':'PASS_THREE_ROUNDS_NINE_CHECKS' if allpass else 'FAIL_THREE_ROUNDS_NINE_CHECKS','rounds':rounds,'official_val_test_access':False}
(P/'AUDIT_ROUNDS.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps(out,indent=2,ensure_ascii=False));raise SystemExit(0 if allpass else 1)
