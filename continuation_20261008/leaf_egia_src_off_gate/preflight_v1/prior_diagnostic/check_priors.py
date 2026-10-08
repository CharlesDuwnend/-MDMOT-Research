"""Read-only binary-head diagnostics; no fitting or test-dev selection."""
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parent
EXPERIMENT=Path('/home/chenhc/leaf_egia_detailed_20261008_v2')
sys.path[:0]=[str(EXPERIMENT),str(EXPERIMENT/'tools')]
os.environ['BELIEF_HOST']=str(EXPERIMENT/'host')
os.environ['UAVDT_ONLINE_HOST']=str(EXPERIMENT/'host')
import numpy as np
from sklearn.metrics import roc_auc_score
from train_coverage_egia_v4 import weights
from run_mechanism import check_frozen

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

receipt=read(EXPERIMENT/'artifacts/PRIOR_AND_BUNDLE_RECEIPT.json')
data_path=Path('/home/chenhc/src_egia_belief_20260923/artifacts/prepare_coverage_v4_fit39_cal8_v2/prepared.pkl')
assert sha(data_path)==receipt['data_sha256']
check_frozen(read(EXPERIMENT/'manifest.json'))
bundle_path=EXPERIMENT/'models/F00_full.pkl'
bundle=pickle.loads(bundle_path.read_bytes())
data=pickle.loads(data_path.read_bytes())
assert not ({r['flight'] for r in data['fit']} & {r['flight'] for r in data['calibration']})

def metrics(y,p,w):
    p=np.clip(np.asarray(p,dtype=float),1e-12,1-1e-12)
    assert len(y)==len(p)==len(w) and set(np.unique(y))=={0,1}
    loss=-(y*np.log(p)+(1-y)*np.log1p(-p))
    return dict(rows=len(y),positives=int(y.sum()),positive_rate=float(np.mean(y)),
        mean_probability=float(np.mean(p)),std_probability=float(np.std(p)),
        ROC_AUC=float(roc_auc_score(y,p)),NLL=float(np.mean(loss)),
        Brier=float(np.mean((p-y)**2)),
        flight_group_weighted_ROC_AUC=float(roc_auc_score(y,p,sample_weight=w)),
        flight_group_weighted_NLL=float(np.average(loss,weights=w)),
        flight_group_weighted_Brier=float(np.average((p-y)**2,weights=w)))

records=[]
for split in ['fit','calibration']:
    rows=data[split]; x=np.stack([r['legacy'] for r in rows]).astype(np.float64)
    labels=np.asarray([r['label'] for r in rows],dtype=int);base=weights(rows)
    assert x.shape[1]==10
    for head_name,model,prior,mask,y in [
       ('foreground',bundle['source_model'].anchor.foreground_model,receipt['foreground_prior'],
        np.ones(len(rows),bool),(labels!=2).astype(int)),
       ('coverage_given_foreground',bundle['source_model'].anchor.coverage_model,receipt['coverage_prior'],
        labels!=2,(labels[labels!=2]==1).astype(int))]:
        probs=np.asarray(model.predict_proba(x[mask]),float)
        p=probs[:,list(model.classes_).index(1)]
        learned=metrics(y,p,base[mask]);constant=metrics(y,np.full(len(y),prior),base[mask])
        records.append(dict(split=split,head=head_name,learned=learned,constant_prior=constant))
        print(json.dumps(records[-1]),flush=True)

# Verify the interventions preserve the documented class order and only fix
# the selected head probability, rather than inverting foreground/coverage.
checks=[]
x=np.stack([r['legacy'] for r in data['calibration']]).astype(np.float64)
pf=bundle['source_model'].anchor.foreground_model.predict_proba(x)[:,1]
pc=bundle['source_model'].anchor.coverage_model.predict_proba(x)[:,1]
for name in ['D02_no_foreground_evidence','D03_no_coverage_evidence','D04_no_learned_context']:
    obj=pickle.loads((EXPERIMENT/'models'/(name+'.pkl')).read_bytes())['source_model'].anchor
    f=np.full(len(x),receipt['foreground_prior']) if name!='D03_no_coverage_evidence' else pf
    c=np.full(len(x),receipt['coverage_prior']) if name!='D02_no_foreground_evidence' else pc
    expected=np.column_stack([f*(1-c),f*c,1-f])
    np.testing.assert_allclose(obj.predict_proba(x),expected,rtol=1e-14,atol=1e-14)
    checks.append(dict(cell=name,class_order=['NEW_TRUE','EXISTING_TRUE','CLUTTER_FALSE'],all_8411_outputs_verified=True))

result=dict(status='COMPLETE_READ_ONLY_PRIOR_AND_HEAD_DIAGNOSTIC',
    source_sha256=sha(Path(__file__)),prepared_sha256=sha(data_path),bundle_sha256=sha(bundle_path),
    original_experiment_manifest_sha256=sha(EXPERIMENT/'manifest.json'),
    no_training_no_parameter_changes=True,no_testdev_prediction_or_GT_read=True,
    calibration_split_scope='Diagnostic; the existing cal8 split also served wider method calibration, not untouched final evaluation',
    metrics_scope='Binary head scores on recorded fit/calibration candidates; not MOT scores or a live test-dev causal diagnosis',
    foreground_prior=receipt['foreground_prior'],coverage_prior=receipt['coverage_prior'],
    prior_weighting='Original equal-flight/group weights and binary square-root class balancing beta=0.5 on fit39 only',
    metric_weighting='Raw rows and original flight/group weights; no recalculated calibration class balancing',
    class_mapping_and_intervention_checks=checks,records=records)
check_frozen(read(EXPERIMENT/'manifest.json'))
(ROOT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
