#!/usr/bin/env python3
"""Independent coordinate, pixel, supervision and integration audit (3 x 3)."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import mmcv
import numpy as np
import torch
from mmdet.datasets.pipelines import Normalize, Pad, Resize
from torchvision.ops import roi_align

import run_n4_holdout as base
from run_lcscf_gate import _train_step_lcscf
from sci_id.data.episodes import _resize_to_tensor, _normalise_box
from sci_id.losses.composition_intervention import CompositionInterventionLoss
from sci_id.models.full_model import build_arm_model
from sci_id.models.identity_pyramid import roi_align_normalized
from sci_id.models.lcscf import LEVELS, LCSCFTransport

ROOT = Path(__file__).resolve().parent


def check(name, condition, evidence):
    if not bool(condition):
        raise AssertionError(name + ': ' + str(evidence))
    return {'check': name, 'status': 'PASS', 'evidence': evidence}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        assert 'A100' in props.name and props.total_memory >= 40000000000
    rounds = []

    # R1: independent, deterministic operator counterexamples.
    f = torch.zeros(1, 1, 9, 3, 4)
    r = torch.zeros_like(f)
    f[:, :, 5, 1, 1] = 1.; r[:, :, 3, 1, 2] = 1.
    trip = float(LCSCFTransport.round_trip_probability(f, r)[0, 0, 1, 1])
    t = torch.tensor([1., 0.]).reshape(1, 2, 1, 1)
    s = torch.tensor([0., 1.]).reshape(1, 1, 2, 1, 1)
    ct, cs = LCSCFTransport.cross_view_consistency(t, s, t[:, None], s)
    operator = LCSCFTransport(4).to(device)
    torch.manual_seed(81)
    target = {lev: torch.randn(2, 4, 3, 5, device=device, requires_grad=True) for lev in LEVELS}
    candidates = {lev: torch.randn(2, 3, 4, 3, 5, device=device, requires_grad=True) for lev in LEVELS}
    mask = torch.tensor([[True, True, True], [False, False, False]], device=device)
    grad_results = []
    border_max = 0.
    for tau in (-20., 0., 20.):
        operator.zero_grad(set_to_none=True)
        with torch.no_grad():
            operator.log_temperature.fill_(tau)
            operator.residual_scale.fill_(.1)
        _, got, stats = operator(target, candidates, mask)
        loss = sum(v.sum() for v in got.values()) + sum(stats[l]['cycle'].sum() for l in LEVELS)
        loss.backward()
        gradients = [p.grad for p in operator.parameters() if p.grad is not None]
        grad_results.append(all(torch.isfinite(g).all().item() for g in gradients))
        for lev in LEVELS:
            plan = stats[lev]['plan_t2s']
            assert torch.allclose(plan.sum(2), torch.ones_like(plan.sum(2)), atol=1e-6)
            invalid = ~operator._valid_neighbours(3, 5, device, torch.float32).bool()
            border_max = max(border_max, float(plan.masked_select(invalid[None, None].expand_as(plan)).abs().max().cpu()))
            assert got[lev][1].abs().max().item() == 0.
    rounds.append({'round':'R1_OPERATOR_COORDINATES', 'checks':[
        check('displaced_round_trip_samples_reverse_at_destination', trip == 1., {'expected':1., 'actual':trip, 'legacy_actual':0.}),
        check('transported_appearance_compared_with_destination', ct.item() == 1. and cs.item() == 1., {'target_consistency':ct.item(),'source_consistency':cs.item(),'legacy_actual':0.}),
        check('border_and_extreme_temperature_gradients', all(grad_results) and border_max == 0., {'finite_backward_at_log_tau':[-20,0,20], 'results':grad_results, 'invalid_border_mass':border_max})
    ]})

    # R2: native preprocessing and annotation-subset evaluator contract.
    dataset = base.PackedFrameEpisodeDataset(base.DATA_ROOT, pairs=['23','25','27','28'])
    path = Path(dataset[0]['target_image'])
    mean = [102.9801,115.9465,122.7717]; std = [1.,1.,1.]
    image, original, resized, _ = _resize_to_tensor(path, (800,1333), torch.tensor(mean), torch.tensor(std), 1., 'bgr')
    native = {'img':mmcv.imread(str(path)), 'img_fields':['img'], 'bbox_fields':[]}
    native = Resize(img_scale=(1333,800), keep_ratio=True)(native)
    native = Normalize(mean=mean, std=std, to_rgb=False)(native)
    native = Pad(size_divisor=32)(native)
    reference = torch.from_numpy(native['img'].transpose(2,0,1).copy())
    pixel_error = float((image-reference).abs().max())
    box = [13.,27.,233.,317.]
    norm = torch.tensor([[0.] + _normalise_box(box, original, resized, image.shape[-2:])])
    feature = torch.arange(16*24, dtype=torch.float32).reshape(1,1,16,24)
    rois = torch.tensor([[0.,.1,.2,.7,.8]])
    expected = rois.clone(); expected[:,[1,3]] *= 24; expected[:,[2,4]] *= 16
    roi_error = float((roi_align_normalized(feature, rois, 3) - roi_align(feature,expected,(3,3),1.,2,True)).abs().max())
    # Bbox pixel reconstruction is independent of the feature sampler.
    reconstructed = norm[0,1:] * torch.tensor([image.shape[-1],image.shape[-2],image.shape[-1],image.shape[-2]])
    expected_box = torch.tensor(box) * torch.tensor([resized[0]/original[0],resized[1]/original[1]]*2)
    bbox_error = float((reconstructed-expected_box).abs().max())
    dummy = SimpleNamespace(candidate_logits=torch.tensor([[1.,100.,0.]]), dustbin_logit=torch.tensor([.5]))
    known = torch.tensor([[True,False,True]])
    scores, probs = base.supervised_candidate_scores(dummy, known)
    idx = known[0].nonzero().flatten(); ranking = idx[torch.argsort(scores[0,idx],descending=True)]
    reference_prob = torch.softmax(torch.tensor([1.,0.,.5]),dim=0)[-1]
    eligible = [dataset[i] for i in range(len(dataset)) if not dataset[i]['supervision_valid']][:1]
    eligible += [dataset[i] for i in range(len(dataset)) if dataset[i]['candidate_count']==0][:1]
    eligible += [dataset[i] for i in range(len(dataset)) if dataset[i].get('unmapped_candidate_indices')][:1]
    eligible += [dataset[i] for i in range(len(dataset)) if len(dataset[i].get('same_id_candidate_indices',[]))>1][:1]
    assert len(eligible)==4
    batch = base.collate_packed_frame_episodes(eligible,image_size=(800,1333),mean=mean,std=std,value_scale=1.,channel_order='bgr')
    assert not (batch['positive_mask'] & batch['unmapped_mask']).any()
    assert not (batch['known_negative_mask'] & batch['unmapped_mask']).any()
    assert not (batch['positive_mask'] & batch['known_negative_mask']).any()
    assert batch['frame_images'].shape[-1] % 32 == batch['frame_images'].shape[-2] % 32 == 0
    rounds.append({'round':'R2_DATA_AND_METRICS', 'checks':[
        check('same_pixels_as_mmdetection_native_resize_normalize_pad',pixel_error<=1e-5,{'max_pixel_error':pixel_error,'padded_shape':list(image.shape),'path':str(path)}),
        check('box_scale_and_roi_extent_equal_absolute_reference',roi_error<=1e-5 and bbox_error<=1e-4,{'roi_error':roi_error,'bbox_error':bbox_error}),
        check('unknown_excluded_from_supervised_rank_and_probability',ranking[0].item()==0 and probs[0,1].item()==0 and torch.allclose(probs[0,-1],reference_prob),{'known_only_rank':1,'dustbin_prob':probs[0,-1].item(),'reference_prob':reference_prob.item(),'batch_edge_cases':['unlabeled_target','zero_candidates','unknown_candidate','multi_positive']})
    ]})

    # R3: real initialized model wiring, all gradient/optimizer states and labels.
    base._set_seed(9322)
    b4 = build_arm_model('B4', channels=8,prototypes=2,output_size=3,backbone_init=args.checkpoint).to(device).eval()
    base._set_seed(9322)
    lcs = build_arm_model('LCSCF_B4', channels=8,prototypes=2,output_size=3,backbone_init=args.checkpoint).to(device).eval()
    common = b4.state_dict()
    same_keys = [k for k in common if not torch.equal(common[k],lcs.state_dict()[k])]
    errors=[]
    images=torch.randn(2,3,64,96,device=device)
    ti=torch.tensor([0],device=device);si=torch.tensor([1],device=device)
    tb=torch.tensor([[.1,.1,.8,.8]],device=device)
    with torch.no_grad():
        for k,valid in ((1,0),(1,1),(64,64)):
            cb=tb[:,None].expand(1,k,4);m=torch.full((1,k),valid>0,device=device,dtype=torch.bool)
            ref=b4.forward_packed(images,ti,si,tb,cb,m);got=lcs.forward_packed(images,ti,si,tb,cb,m)
            errors.append(float((ref.logits-got.logits).abs().max().cpu()))
    rounds.append({'round':'R3_REAL_WIRING_AND_UPDATES','checks':[
        check('strict_initialization_and_same_common_seed',not same_keys and lcs.backbone_init_counts=={'loaded':258,'skipped':0},{'actual_backbone_load':lcs.backbone_init_counts,'common_keys_different':same_keys}),
        check('step_zero_preserves_b4_including_k0_k1_k64',max(errors)<=1e-5,{'logit_max_errors':errors,'valid_k':[0,1,64]})
    ]})
    del b4,lcs
    if device.type=='cuda':torch.cuda.empty_cache()
    # Use real native frames and the production training function, not only a
    # synthetic positive-logit loss. Three updates cover unusual label masks.
    base._set_seed(9322)
    lcs=build_arm_model('LCSCF_B4',channels=8,prototypes=2,output_size=3,backbone_init=args.checkpoint).to(device).train()
    optimizer=torch.optim.AdamW(lcs.parameters(),lr=1e-4,weight_decay=1e-4)
    production=[]
    for selected in ([eligible[0]], [eligible[1]], [eligible[2],eligible[3]]):
        real=base.collate_packed_frame_episodes(selected,image_size=(800,1333),mean=mean,std=std,value_scale=1.,channel_order='bgr')
        real=base._to_device(real,device)
        _,loss,terms,grad=_train_step_lcscf(lcs,optimizer,CompositionInterventionLoss(),real)
        state_finite=all(torch.isfinite(v).all().item() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
        production.append({'loss':loss,'gradient_l1':grad,'optimizer_finite':state_finite,'valid_rows':int(real['supervision_valid'].sum()),'candidate_count':real['mask'].sum(1).tolist()})
    rounds[-1]['checks'].append(check('production_loss_backward_optimizer_on_real_edge_cases',all(x['optimizer_finite'] and np.isfinite(x['loss']) for x in production),production))
    sources=['sci_id/models/lcscf.py','sci_id/models/identity_pyramid.py','sci_id/models/full_model.py','sci_id/data/episodes.py','run_n4_holdout.py','run_lcscf_gate.py','train_n4_holdout.py','audit_lcscf_v4.py']
    payload={'status':'PASS_LCSCF_V4_IMPLEMENTATION_AUDIT_3X3','rounds':rounds,'source_sha256':{r:hashlib.sha256((ROOT/r).read_bytes()).hexdigest() for r in sources},'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'official_val_test_access':False,'scope':'Mechanism, preprocessing and production gradient contracts; no effectiveness claim.'}
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
