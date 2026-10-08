"""Twelve distinct contracts for fit-only substitutions and actual runtime flags."""
import copy
import json
import pickle
from pathlib import Path
import numpy as np
import pytest
from scipy.special import logsumexp, softmax
from test_egia_modes import changed_fields, object_sha256, file_sha256, PREPARED
from belief.hierarchical import class_balanced_weights
from belief.egia_ablation import ConstantBinaryPrior
from belief.runtime import BeliefTracker
from yolox.tracker.u2mot_tracker import DefaultArgs, STrack

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def context():
    receipt = json.loads((ROOT/'artifacts/PRIOR_AND_BUNDLE_RECEIPT.json').read_text())
    data = pickle.loads(PREPARED.read_bytes())
    bundles = {r['mode']: pickle.loads((ROOT/r['bundle']).read_bytes()) for r in receipt['cells']}
    return receipt, data, bundles


def test_round1_sources_and_training_partition(context):
    receipt, data, _ = context
    for p, h in receipt['source_files_sha256'].items():
        assert file_sha256(p) == h
    assert sorted({r['sequence'] for r in data['fit']}) == sorted(receipt['fit_sequences'])
    assert not ({r['flight'] for r in data['fit']} & {r['flight'] for r in data['calibration']})
    assert not receipt['calibration_used_for_prior'] and not receipt['testdev_selection']
    assert not receipt['head_refit']


def test_round1_independent_binary_priors(context):
    from collections import Counter, defaultdict
    receipt, data, _ = context
    rows = data['fit']
    keys = [(r['flight'], 2, r['sequence'], int(r['source_frame'])//30) if r['label']==2
            else (r['flight'], r['label'], r['candidate_gt_id']) for r in rows]
    counts, groups = Counter(keys), defaultdict(set)
    for k in keys:
        groups[k[0]].add(k)
    base = np.asarray([1./len(groups)/len(groups[k[0]])/counts[k] for k in keys])
    assert object_sha256(base) == receipt['base_weight_sha256']
    y = np.asarray([r['label'] for r in rows])
    for label, w, prior in [((y!=2).astype(int), base, receipt['foreground_prior']),
                            ((y[y!=2]==1).astype(int), base[y!=2], receipt['coverage_prior'])]:
        masses = np.bincount(label, weights=w, minlength=2)
        independently_balanced = w * np.sqrt(.5/masses[label])
        independently_balanced *= w.sum()/independently_balanced.sum()
        np.testing.assert_allclose(class_balanced_weights(w, label, .5), independently_balanced, rtol=1e-14)
        assert np.isclose(np.average(label, weights=independently_balanced), prior, rtol=1e-14)


def test_round1_preserve_every_src_and_policy_field(context):
    receipt, _, bundles = context
    base = bundles['full']
    for r in receipt['cells']:
        candidate = bundles[r['mode']]
        assert file_sha256(ROOT/r['bundle']) == r['bundle_sha256']
        for k in base:
            if k != 'source_model':
                assert object_sha256(candidate[k]) == object_sha256(base[k])
        assert candidate['egia_fusion_weight'] == .4 and candidate['selective_birth_margin'] == .2
        assert candidate['egia_fusion_policy'] and candidate['selective_birth_policy']


def test_round2_named_heads_only(context):
    receipt, _, b = context
    base = b['full']['source_model']
    for mode in ['no_foreground', 'no_coverage', 'no_context']:
        source = b[mode]['source_model']
        for k in base.__dict__:
            if k != 'anchor':
                assert object_sha256(getattr(source,k)) == object_sha256(getattr(base,k))
        for name, prior_key, removed in [('foreground_model','foreground_prior',mode!='no_coverage'),
                                         ('coverage_model','coverage_prior',mode!='no_foreground')]:
            head = getattr(source.anchor,name)
            if removed:
                assert isinstance(head,ConstantBinaryPrior) and head.probability==receipt[prior_key]
            else:
                assert object_sha256(head)==object_sha256(getattr(base.anchor,name))


def test_round2_legal_and_input_independent_priors(context):
    _, data, b = context
    x = np.stack([r['legacy'] for r in data['fit'][::113]])
    for mode in ['no_foreground','no_coverage','no_context']:
        p = b[mode]['source_model'].anchor.predict_proba(x)
        assert np.isfinite(p).all() and (p>0).all()
        np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-14)
    p = b['no_context']['source_model'].anchor.predict_proba(x)
    np.testing.assert_array_equal(p,np.tile(p[0],(len(p),1)))


def test_round2_coherence_weight_only(context):
    _, _, b = context
    source = b['no_coherence']['source_model']
    assert source.weight==0.
    for k in source.__dict__:
        if k!='weight':
            assert object_sha256(getattr(source,k))==object_sha256(getattr(b['full']['source_model'],k))
    for mode in ['disabled','pair_shuffle']:
        assert object_sha256(b[mode]['source_model'])==object_sha256(b['full']['source_model'])


def test_round3_neutral_cue_algebra_and_owner_posterior(context):
    _, data, b = context
    source = b['full']['source_model']
    for row in data['fit'][::71]:
        owners = np.asarray(row['owners'], dtype=float)
        for mode,cue in [('no_geometry_cue','geometry'),('no_appearance_cue','appearance')]:
            evidence, owner = b[mode]['source_model'].log_evidence(owners)
            assert evidence==0.
            if len(owners)<2:
                np.testing.assert_array_equal(owner,np.zeros(len(owners)))
                continue
            remaining_model = source.appearance_model if cue=='geometry' else source.geometry_model
            column = 1 if cue=='geometry' else 0
            remaining = remaining_model.decision_function(owners[:,column:column+1]).astype(float)
            remaining -= np.log(source.cue_prior/(1-source.cue_prior))
            if cue=='geometry':
                remaining[owners[:,2]==0]=0.
            algebra = np.log(len(owners))+logsumexp(remaining)-logsumexp(np.zeros(len(owners)))-logsumexp(remaining)
            assert np.isclose(algebra,0.,atol=1e-12)
            np.testing.assert_array_equal(owner,softmax(remaining))


def test_round3_neutral_cue_predictions_equal_zero_weight(context):
    _, data, b = context
    rows = data['fit'][::37]
    baseline, _ = b['no_coherence']['source_model'].predict_rows(rows)
    for mode in ['no_geometry_cue','no_appearance_cue']:
        p,e = b[mode]['source_model'].predict_rows(rows)
        np.testing.assert_array_equal(p,baseline)
        assert (e==0).all()


def test_round3_neutral_cues_keep_fitted_estimators(context):
    _, _, b = context
    for mode in ['no_geometry_cue','no_appearance_cue']:
        source = b[mode]['source_model']
        for k in b['full']['source_model'].__dict__:
            assert object_sha256(getattr(source,k))==object_sha256(getattr(b['full']['source_model'],k))


def trackers(context):
    _,_,bundles=context
    for mode,bundle in bundles.items():
        args=DefaultArgs()
        args.continuous_semantic_reliability=True
        args.egia_model=str(ROOT/'models/summary_frozen.pkl')
        args.egia_capture_dir=''
        yield mode,BeliefTracker(args,arm='fc_full',bundle=copy.deepcopy(bundle))


def test_actual_runtime_keeps_full_src_in_all_cells(context):
    for mode,t in trackers(context):
        assert t.egia_ablation==mode
        assert t.src_state_enabled and t.src_update_enabled and t.src_cost_enabled
        assert t.assignment_mask_enabled and t.semantic_penalty_enabled
        assert t.use_src and not t.src_shadow_only and not t.hard_update


def test_actual_runtime_bypass_and_pairing_switches(context):
    for mode,t in trackers(context):
        assert t.use_egia==(mode!='disabled')
        assert t.shuffle_pairs==(mode=='pair_shuffle')
        assert t.egia_fusion_policy and t.selective_birth_policy
        assert t.egia_fusion_weight==.4 and t.selective_birth_margin==.2
        t.frame_id=t.source_frame=1
        t.sequence_name='cpu_birth_gate'
        t._egia_frame_height,t._egia_frame_width=200.,300.
        t.frame_observations={0:dict(frame=1,source_frame=1,detection_ordinal=0,
            bbox_tlwh=[1.,2.,20.,40.],predicted_class=0,score=.9,class_confidence=.9)}
        candidate=STrack(np.asarray([1.,2.,20.,40.]),.9,0,semantic_score=.9)
        matches=[i for i,row in t.frame_observations.items() if t.detection_identity_matches(candidate,row)]
        assert matches==[0]
        t.det_ordinals[id(candidate)]=matches[0]  # Same fixture binding used by whole-seam CPU checks.
        t.det_record(candidate)
        t._allow_new_track_activation(candidate)
        assert t.counts['native_birth_gate_calls']==1
        assert t.counts['native_birth_gate_admitted']==1
        assert t.counts['egia_source_probability_calls']==int(mode!='disabled')
        if mode=='disabled':
            assert len(t.new_candidates)==1
        rejected=STrack(np.asarray([1.,2.,20.,40.]),.01,0,semantic_score=.9)
        assert not t._allow_new_track_activation(rejected)
        assert t.counts['native_birth_gate_calls']==2 and t.counts['native_birth_gate_admitted']==1


def test_invalid_mode_stops_before_inference(context):
    _,_,b=context
    args=DefaultArgs()
    args.continuous_semantic_reliability=True
    args.egia_model=str(ROOT/'models/summary_frozen.pkl')
    args.egia_capture_dir=''
    bundle=copy.deepcopy(b['disabled'])
    bundle['egia_ablation']='undeclared'
    with pytest.raises(ValueError,match='unknown EGIA ablation'):
        BeliefTracker(args,arm='fc_full',bundle=bundle)
