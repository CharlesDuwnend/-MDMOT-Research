"""Seal component reduction with the user's final SRC-off/gate-off baseline."""
import copy,datetime,json,os,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['BELIEF_HOST']=os.environ['UAVDT_ONLINE_HOST']=str(ROOT/'host')
import numpy as np
from fit_paired_anchor import sha,objsha
from belief.structural import DeletedHeadAnchor,StructuralSourceModel,WithoutSourceCues
PARENT=Path('/home/chenhc/leaf_egia_src_off_gate_20261008_v1')
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
parent=json.loads((PARENT/'manifest.json').read_text())
for p,h in parent['frozen_files_sha256'].items():assert sha(p)==h
base=pickle.loads((ROOT/'models/F00_full.pkl').read_bytes())
assert base['egia_fusion_policy'] and base['selective_birth_policy']
assert not any(base.get(k,False) for k in ['owner_selective_birth_policy','decision_specific_birth_policy','asymmetric_selective_birth_policy'])
base['egia_fusion_policy']=False
gpu={0:'GPU-2a55197e-2c19-45ff-cf33-13031951f9d0',1:'GPU-1b297aba-ae7e-e326-5903-476f1bb683d7',3:'GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a'}
specs=[('R00_full','full',0,()),('R01_no_foreground_head','delete_foreground',0,(2,)),
       ('R02_no_coverage_head','delete_coverage',1,(1,)),('R03_no_source_cues','delete_source_cues',3,())]
arms=[]
refmetrics=str(PARENT/'YOLOX_outputs/S0_G0_fusion_off/tracking_metrics.json')
for name,mode,index,forbidden in specs:
    b=copy.deepcopy(base)
    s=WithoutSourceCues(b['source_model']) if mode=='delete_source_cues' else StructuralSourceModel(b['source_model'],mode,forbidden)
    if mode.startswith('delete_') and mode!='delete_source_cues':
        s.anchor=DeletedHeadAnchor(s.anchor,'foreground' if mode=='delete_foreground' else 'coverage')
    b['source_model']=s
    assert all(objsha(b[k])==objsha(base[k]) for k in base if k!='source_model')
    if mode!='delete_source_cues':
        assert all(objsha(getattr(s,k))==objsha(getattr(base['source_model'],k)) for k in ['geometry_model','appearance_model','cue_prior','weight'])
    else:assert objsha(s.anchor)==objsha(base['source_model'].anchor)
    path=ROOT/'models'/(name+'.pkl')
    with path.open('xb') as f:pickle.dump(b,f,protocol=4)
    arms.append(dict(name=name,structural_mode=mode,forbidden_classes=list(forbidden),
                     control=mode,mode='full',runtime_arm='egia',src=False,mask=False,penalty=False,
                     gate=False,birth_class_gate=0.,use_egia=True,fusion=False,selective=True,
                     corrected_reference_oracle=False,bundle=str(path.relative_to(ROOT)),
                     bundle_sha256=sha(path),gpu_physical_index=index,gpu_uuid=gpu[index],
                     historical_reference_metrics=refmetrics if mode=='full' else None))
reg=dict(status='PREREGISTERED_BEFORE_ANY_NEW_REDUCED_COMPONENT_INFERENCE_OR_SCORE',
    created_utc=datetime.datetime.utcnow().isoformat()+'Z',new_scores_seen=0,arms=arms,
    latest_user_constraint='baseline without SRC and gate; pure v5 must not load or use H2 fusion classifier; reduce remaining EGIA parts',
    fixed=dict(src=False,birth_class_gate=0.,birth_detector_score=.6,fusion=False,selective=True),
    no_new_training=True,no_prior_replacement=True,no_testdev_configuration_selection=True,
    no_new_figures=True,all_results_retained=True,
    dropout_semantics={'foreground':'delete FG model, f=1, source U/E only',
        'coverage':'delete coverage model, c=0, source U/C only; E-only coherence authority also lost',
        'source_cues':'delete geometry and appearance models; evidence and weight zero'},
    inference_scope='No legacy classifier loaded; selective native fallback retained',
    hypotheses=['FG deletion may weaken clutter rejection','Coverage deletion may weaken redundant-target suppression',
                'Source-cue deletion tests a compact two-head EGIA without co-reference models'],
    outcomes=['MOTA','IDF1','FP','FN','IDs','FM','birth activations','head calls'],
    full_frames=sum(parent['sequence_frames'].values()),sequence_count=len(parent['sequence_frames']),
    paired_units='17 sequences, 14 flights; no frame independence or significance claims',
    parent_root=str(PARENT),parent_manifest_sha256=sha(PARENT/'manifest.json'),
    run_order='Fresh full reference first, then fixed GPU0/1/3 independent queues',
    inference_GT_reads_forbidden=True)
write(ROOT/'PREREGISTRATION.json',reg)
write(ROOT/'design.json',dict(**{k:parent[k] for k in ['data_root','gt_root','detector','fixed','sequence_frames','smoke_sequence','smoke_frames','corrected_reference_sources']},
    status='REDUCED_DESIGN_NOT_YET_FROZEN',parent_root=str(PARENT),arms=arms,
    full_reference_metrics=refmetrics,input_reference_receipt=str(Path(refmetrics).parent/'receipt.json'),
    smoke_input_reference_receipt=str(PARENT/'YOLOX_outputs/SMOKE_S0_G0_fusion_off/receipt.json'),
    gpu_uuid=gpu[0],gpu_physical_index=0,testdev_parameter_selection=False))
print(json.dumps(dict(arms=len(arms),SRC=False,gate=0.,fusion=False,selective=True,new_scores_seen=0)))
