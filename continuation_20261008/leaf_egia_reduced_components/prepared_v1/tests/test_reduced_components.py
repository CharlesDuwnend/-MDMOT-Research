"""Nine checks: provenance, actual deletion, operator path and frozen policy."""
import copy,json,pickle
from pathlib import Path
import numpy as np
import pytest
from scipy.special import softmax
from belief.features import legacy_vector,LEGACY_FEATURES
from test_egia_modes import file_sha256,object_sha256,PREPARED,F00_SHA256,PREPARED_SHA256
from test_operator_audit import event
from test_gate_grid import tracker,DESIGN
ROOT=Path(__file__).resolve().parents[1]

def complete_event():
    e=event('reduced_cpu',1,0)
    for key in LEGACY_FEATURES:
        e['native_features'].setdefault(key,0.)
    e['native_features'].update(max_active_iou=.9,max_lost_iou=.2,
                                max_active_reid_similarity=.95,max_lost_reid_similarity=.1)
    assert set(LEGACY_FEATURES)<=set(e['native_features'])
    return e

@pytest.fixture(scope='module')
def context():
    base=pickle.loads((ROOT/'models/F00_full.pkl').read_bytes())
    bundles={a['structural_mode']:pickle.loads((ROOT/a['bundle']).read_bytes()) for a in DESIGN['arms']}
    return base,bundles,pickle.loads(PREPARED.read_bytes())

def test_round1_actual_source_and_legitimate_fit_partition(context):
    assert file_sha256(ROOT/'models/F00_full.pkl')==F00_SHA256
    assert file_sha256(PREPARED)==PREPARED_SHA256
    _,_,data=context
    assert (len(data['fit']),len(data['calibration']))==(8024,8411)
    assert len({r['sequence'] for r in data['fit']})==39
    assert len({r['sequence'] for r in data['calibration']})==8
    assert not ({r['flight'] for r in data['fit']} & {r['flight'] for r in data['calibration']})

def test_round1_only_source_changed_and_all_outer_policies_fixed(context):
    base,bundles,_=context
    for b in bundles.values():
        assert all(object_sha256(b[k])==object_sha256(base[k]) for k in base if k!='source_model')
        assert b['egia_fusion_policy'] and b['selective_birth_policy']
        assert b['egia_fusion_weight']==.4 and b['selective_birth_margin']==.2
    assert all(a['src'] is False and a['birth_class_gate']==0. for a in DESIGN['arms'])

def test_round1_retained_fitted_estimators_identical(context):
    base,bundles,_=context;s0=base['source_model']
    for mode,b in bundles.items():
        s=b['source_model']
        if mode=='delete_foreground':
            assert s.anchor.foreground_model is None
            assert object_sha256(s.anchor.coverage_model)==object_sha256(s0.anchor.coverage_model)
        elif mode=='delete_coverage':
            assert s.anchor.coverage_model is None
            assert object_sha256(s.anchor.foreground_model)==object_sha256(s0.anchor.foreground_model)
        else:assert object_sha256(s.anchor)==object_sha256(s0.anchor)
        if mode!='delete_source_cues':
            for k in ['geometry_model','appearance_model','cue_prior','weight']:
                assert object_sha256(getattr(s,k))==object_sha256(getattr(s0,k))

def test_round2_full_operator_exact_probability_identity(context):
    base,bundles,data=context
    for split in ['fit','calibration']:
        rows=data[split][::97]
        p,e=bundles['full']['source_model'].predict_rows(rows)
        p0,e0=base['source_model'].predict_rows(rows)
        np.testing.assert_array_equal(p,p0);np.testing.assert_array_equal(e,e0)

def test_round2_head_deletion_matches_retained_real_classifier(context):
    _,bundles,data=context
    x=np.stack([r['legacy'] for r in data['calibration'][::83]]).astype(float)
    for mode,retained,column in [('delete_foreground','coverage',2),('delete_coverage','foreground',1)]:
        s=copy.deepcopy(bundles[mode]['source_model']);a=s.anchor
        pipeline=getattr(a,retained+'_model')
        q=pipeline.predict_proba(x)[:,list(pipeline.classes_).index(1)]
        expected=np.column_stack((1-q,q,np.zeros(len(q)))) if retained=='coverage' else np.column_stack((q,np.zeros(len(q)),1-q))
        np.testing.assert_array_equal(a.predict_proba(x),expected)
        removed='foreground' if retained=='coverage' else 'coverage'
        assert a.calls[removed]==0 and a.calls[retained]>0
        assert np.count_nonzero(a.predict_proba(x)[:,column])==0

def test_round2_source_cues_deleted_equal_zero_weight_control(context):
    base,bundles,data=context
    s=copy.deepcopy(bundles['delete_source_cues']['source_model'])
    control=copy.deepcopy(base['source_model']);control.weight=0.
    assert s.geometry_model is None and s.appearance_model is None and s.weight==0.
    rows=data['calibration'][::61]
    np.testing.assert_array_equal(s.predict_rows(rows)[0],control.predict_rows(rows)[0])
    assert np.count_nonzero(s.predict_rows(rows)[1])==0

def test_round3_all_actual_event_shapes_have_zero_deleted_actions(context):
    _,bundles,_=context
    for mode,b in bundles.items():
        s=copy.deepcopy(b['source_model'])
        for n in [0,1,2]:
            e=complete_event();e['sources']=e['sources'][:n]
            p,ids,evidence=s.predict_event(e)
            assert len(ids)==n and np.isfinite(p).all() and (p>=0).all()
            np.testing.assert_allclose(p.sum(),1.,atol=1e-15)
            assert np.count_nonzero(p[list(s.forbidden_classes)])==0
            if mode=='delete_coverage':
                f=s.anchor.foreground_model.predict_proba(legacy_vector(e)[None])[0,1]
                np.testing.assert_allclose(p,[f,0.,1-f],atol=1e-15)

def test_round3_no_epsilon_revival_even_extreme_coherence(context):
    _,bundles,_=context
    for mode in ['delete_foreground','delete_coverage']:
        s=copy.deepcopy(bundles[mode]['source_model'])
        s.weight=1e4
        s.log_evidence=lambda owners:(1.,np.ones(len(owners)))
        p,_,_=s.predict_event(complete_event())
        assert np.count_nonzero(p[list(s.forbidden_classes)])==0
        np.testing.assert_allclose(p.sum(),1.,atol=1e-15)

def test_round3_runtime_structural_report_tracks_actual_calls(context):
    for a in DESIGN['arms']:
        t=tracker(a)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.99,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();d=r['structural_audit'];c=r['counts']
        assert d['mode']==a['structural_mode']
        assert d['source_calls']['events']==c['birth_seams']>0
        assert d['source_calls']['forbidden_nonzero']==0
        assert c['egia_fusion_events']==c['birth_seams']
        assert not r['src_cost_enabled'] and not r['src_update_enabled'] and not r['src_state_enabled']
        assert c['native_reference_seams_verified']==c['association_seams']==9
        assert c['native_birth_gate_low_class_admitted']>0
        if a['structural_mode'].startswith('delete_') and a['structural_mode']!='delete_source_cues':
            removed='foreground' if a['structural_mode']=='delete_foreground' else 'coverage'
            assert d['head_calls'][removed]==0
