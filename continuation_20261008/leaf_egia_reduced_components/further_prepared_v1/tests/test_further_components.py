"""Frozen weights, independent operators, true native runtime and policies."""
import ast, copy, hashlib, json, pickle
from pathlib import Path
import numpy as np
import pytest
from scipy.special import logsumexp
from belief.features import legacy_vector,LEGACY_FEATURES
from belief.online import OnlineBeliefTracker
from yolox.tracker.u2mot_tracker import DefaultArgs,STrack
from fit_paired_anchor import objsha
from test_operator_audit import event

ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('/home/chenhc/leaf_egia_reduced_pure_20261008_v1')
DESIGN=json.loads((ROOT/'design.json').read_text())
DATA=Path('/home/chenhc/src_egia_belief_20260923/artifacts/prepare_coverage_v4_fit39_cal8_v2/prepared.pkl')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bundle(arm):return pickle.loads((ROOT/arm['bundle']).read_bytes())
def base():return pickle.loads((ROOT/'models/R00_full.pkl').read_bytes())
def track(arm):
    a=DefaultArgs();a.egia_model='';a.egia_capture_dir=''
    a.new_track_min_class_confidence=0.;a.new_track_thresh=.6;a.track_buffer=15
    t=OnlineBeliefTracker(a,arm='egia',bundle=bundle(arm));t.sequence_name='cpu_further'
    return t
def complete_event():
    e=event('cpu_further',1,0)
    for k in LEGACY_FEATURES:e['native_features'].setdefault(k,0.)
    e['native_features'].update(max_active_iou=.9,max_lost_iou=.2,
        max_active_reid_similarity=.95,max_lost_reid_similarity=.1)
    return e

def test_round1_parent_complete_and_model_hashes():
    assert json.loads((PARENT/'review/results/FINAL_AUDIT.json').read_text())['new_full_runs']==4
    assert sha(ROOT/'models/R00_full.pkl')==sha(PARENT/'models/R00_full.pkl')
    assert sha(DATA)=='e126f0ebcbb284985eb8bfd43c6bcaec4dc401fe7e507960518ea59fafde055d'
    d=pickle.loads(DATA.read_bytes())
    assert len(d['fit'])==8024 and len(d['calibration'])==8411
    assert not ({r['flight'] for r in d['fit']} & {r['flight'] for r in d['calibration']})

def test_round1_exact_allowed_changes_and_no_priors():
    b0=base()
    for a in DESIGN['arms']:
        b=bundle(a)
        assert all(objsha(b[k])==objsha(b0[k]) for k in b0 if k not in ('source_model','selective_birth_policy'))
        assert not b['egia_fusion_policy'] and b['selective_birth_policy']==a['selective']
        assert b['selective_birth_margin']==.2
        s=b['source_model'];s0=b0['source_model']
        assert objsha(s.anchor.coverage_model)==objsha(s0.anchor.coverage_model)
        if a['structural_mode']=='coverage_only':
            assert s.anchor.foreground_model is None and s.geometry_model is None and s.appearance_model is None
        else:assert objsha(s.anchor.foreground_model)==objsha(s0.anchor.foreground_model)

def test_round1_native_algorithms_and_birth_policy_unchanged():
    for rel in ['host/yolox/tracker/u2mot_tracker.py','belief/online.py','belief/coherence.py','belief/structural.py']:
        assert sha(ROOT/rel)==sha(PARENT/rel)
    def method(p,name):
        tree=ast.parse(p.read_text())
        return ast.dump(next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)
    for name in ['_egia_allows_birth','_allow_new_track_activation','associate','_source_event']:
        assert method(ROOT/'belief/runtime.py',name)==method(PARENT/'belief/runtime.py',name)

def test_round2_only_coverage_probabilities_match_real_classifier():
    a=next(a for a in DESIGN['arms'] if a['structural_mode']=='coverage_only')
    s=bundle(a)['source_model']
    rows=pickle.loads(DATA.read_bytes())['calibration'][::41]
    x=np.stack([r['legacy'] for r in rows]).astype(np.float64)
    model=s.anchor.coverage_model;q=model.predict_proba(x)[:,list(model.classes_).index(1)]
    np.testing.assert_array_equal(s.anchor.predict_proba(x),np.column_stack((1-q,q,np.zeros(len(q)))))
    p,e=s.predict_rows(rows)
    np.testing.assert_allclose(p[:,:2],np.column_stack((1-q,q)),atol=1e-12,rtol=0.)
    assert np.count_nonzero(p[:,2])==0 and np.count_nonzero(e)==0

def test_round2_argmax_models_keep_exact_source_posteriors():
    rows=pickle.loads(DATA.read_bytes())['calibration'][::53]
    for name,reference in [('A05_two_heads_argmax','R03_no_source_cues.pkl'),('A06_full_argmax','R00_full.pkl')]:
        a=next(a for a in DESIGN['arms'] if a['name']==name)
        s=bundle(a)['source_model'];old=pickle.loads((ROOT/'models'/reference).read_bytes())['source_model']
        p,e=s.predict_rows(rows);p0,e0=old.predict_rows(rows)
        np.testing.assert_array_equal(p,p0);np.testing.assert_array_equal(e,e0)

def test_round2_single_cue_removal_and_coverage_deletion_are_degenerate():
    # Algebraic identity for arbitrary finite cue scores, independent of model code.
    rng=np.random.RandomState(20261008)
    for n in [2,3,17,101]:
        cue=rng.normal(size=n)*10
        for a,b in [(cue,np.zeros(n)),(np.zeros(n),cue)]:
            k=np.log(n)+logsumexp(a+b)-logsumexp(a)-logsumexp(b)
            np.testing.assert_allclose(k,0.,atol=1e-14)
    no_cov=pickle.loads((PARENT/'models/R02_no_coverage_head.pkl').read_bytes())['source_model']
    no_cov_cues=copy.deepcopy(no_cov);no_cov_cues.weight=0.
    rows=pickle.loads(DATA.read_bytes())['calibration'][::47]
    np.testing.assert_array_equal(no_cov.predict_rows(rows)[0],no_cov_cues.predict_rows(rows)[0])

@pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
def test_round3_actual_runtime_no_src_no_gate_no_h2(arm):
    t=track(arm)
    for frame in range(1,4):
        raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                        [100,20,120,60,.99,.65,3]],np.float32)
        t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
    r=t.runtime_report();c=r['counts'];s=r['structural_audit']
    assert r['pure_v5_policy'] and not r['legacy_egia_loaded'] and t.egia_model_payload is None
    assert not r['src_cost_enabled'] and not r['src_state_enabled'] and not r['src_update_enabled']
    assert r['assignment_source']=='native' and r['stored_beliefs']==0
    assert r['birth_class_confidence_threshold']==0. and r['birth_score_threshold']==.6
    assert c['native_reference_seams_verified']==c['association_seams']==9
    assert c['native_birth_gate_low_class_admitted']>0 and c.get('egia_fusion_events',0)==0
    assert r['selective_birth_policy']==arm['selective']
    if not arm['selective']:assert all(v==0 for k,v in c.items() if k.startswith('selective_'))
    assert s['mode']==arm['structural_mode'] and s['source_calls']['events']==c['birth_seams']>0
    if arm['structural_mode']=='coverage_only':
        assert s['head_calls']['foreground']==0 and s['head_calls']['coverage']>0
    if arm['structural_mode']!='full':assert c['coherence_nonzero_events']==0

@pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
def test_round3_outer_and_inner_gate_disabled_with_real_candidate(arm):
    t=track(arm);t.frame_id=1;t.source_frame=1
    candidate=STrack(np.asarray([10.,20.,20.,40.]),.9,0,semantic_score=.65)
    t.frame_observations={0:{'frame':1,'source_frame':1,'detection_ordinal':0,
        'bbox_tlwh':[10.,20.,20.,40.],'score':.9,'class_confidence':.65,
        'predicted_class':0,'frame_width':300,'frame_height':200}}
    t.det_ordinals[id(candidate)]=0
    # Native eligibility must admit below 0.7 before EGIA makes its own decision.
    outer=t._allow_new_track_activation(candidate)
    assert t.counts['native_birth_gate_admitted']==1
    assert t.counts['native_birth_gate_low_class_admitted']==1
    assert t.counts['local_h2_admitted']==1
    inner=t._egia_allows_birth(candidate,[],[])
    assert outer==inner and t.counts['local_h2_admitted']==2
    candidate.score=.1
    assert not t._allow_new_track_activation(candidate)

def test_round3_coverage_only_forbids_clutter_at_every_owner_count():
    a=next(a for a in DESIGN['arms'] if a['structural_mode']=='coverage_only')
    s=bundle(a)['source_model']
    for n in [0,1,2]:
        e=complete_event();e['sources']=e['sources'][:n]
        p,ids,k=s.predict_event(e)
        assert len(ids)==n and p[2]==0. and k==0.
        np.testing.assert_allclose(p.sum(),1.,atol=1e-15)
