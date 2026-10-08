"""Actual source flags, native seams, and both birth-eligibility locations."""
import copy
import json
import pickle
from pathlib import Path
import numpy as np
import pytest
from belief.runtime import BeliefTracker
from belief.online import OnlineBeliefTracker
from belief.reference_oracle import assignment_key
from yolox.tracker.u2mot_tracker import DefaultArgs,STrack
from test_corrected_reference import make,bind_independently

ROOT=Path(__file__).resolve().parents[1]
DESIGN=json.loads((ROOT/'design.json').read_text())

def tracker(arm):
    args=DefaultArgs();args.continuous_semantic_reliability=True
    args.egia_model=str(ROOT/'models/summary_frozen.pkl');args.egia_capture_dir=''
    args.new_track_min_class_confidence=arm['birth_class_gate'];args.new_track_thresh=.6
    args.track_buffer=15
    bundle=pickle.loads((ROOT/arm['bundle']).read_bytes())
    t=OnlineBeliefTracker(args,arm=arm['runtime_arm'],bundle=bundle)
    t.sequence_name='cpu_gate_grid'
    return t

@pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
def test_declared_runtime_flags_and_actual_stream(arm):
    t=tracker(arm)
    for frame in range(1,4):
        raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                        [100,20,120,60,.99,.65,3]],np.float32)
        t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
    r=t.runtime_report();c=r['counts']
    assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
    assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
    assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
    assert c['association_seams']==9
    if not arm['src']:
        assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
        for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                  'src_observation_probability_calls','src_semantic_penalty_calls']:
            assert c.get(k,0)==0
    else:assert c['corrected_reference_seams_verified']==9
    if arm['gate']:assert c['native_birth_gate_low_class_admitted']==0
    else:assert c['native_birth_gate_low_class_admitted']>0

@pytest.mark.parametrize('gate',[0.,.7])
def test_outer_and_inner_birth_gates_agree_without_egia(gate):
    arm=next(a for a in DESIGN['arms'] if not a['src'] and a['control']=='disabled' and a['birth_class_gate']==gate)
    t=tracker(arm);t.frame_id=1;t.source_frame=1
    candidate=STrack(np.asarray([10.,20.,20.,40.]),.9,0,semantic_score=.65)
    t.frame_observations={0:{'frame':1,'source_frame':1,'detection_ordinal':0,'bbox_tlwh':[10.,20.,20.,40.],
      'score':.9,'class_confidence':.65,'predicted_class':0,'frame_width':300,'frame_height':200}}
    t.det_ordinals[id(candidate)]=0
    outer=t._allow_new_track_activation(candidate)
    inner=t._egia_allows_birth(candidate,[],[])
    assert outer==inner==(gate==0.)
    candidate.score=.1
    assert not t._allow_new_track_activation(candidate)
    assert not t._egia_allows_birth(candidate,[],[])

def test_model_source_flags_not_retrained_and_prior_head_mapping():
    for arm in DESIGN['arms']:
        obj=pickle.loads((ROOT/arm['bundle']).read_bytes())
        assert obj['egia_fusion_weight']==.4 and obj['selective_birth_margin']==.2
        assert list(obj['source_model'].anchor.classes_)==[0,1,2]
        assert obj['selective_birth_policy']==arm['selective']
        assert obj['egia_fusion_policy']==arm['fusion']
