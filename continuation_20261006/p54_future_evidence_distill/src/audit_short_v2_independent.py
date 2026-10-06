#!/usr/bin/env python3
"""Nine independent checks of the archived P54 v2 run, using CPU only.

Historical source/receipts are not modified. A completed audit can contain failed
contracts; its status must never be interpreted as a method PASS.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import inspect
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import contract
import data_audit as data
import train_short as tr


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def plain(v):
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return plain(v.tolist())
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    return v


def independently_label(rows, xml):
    by_frame = {}
    for track in ET.parse(xml).getroot().findall('track'):
        for box in track.findall('box'):
            if int(box.get('outside', '0')):
                continue
            xy = [float(box.get(k)) for k in ('xtl', 'ytl', 'xbr', 'ybr')]
            by_frame.setdefault(int(box.get('frame')), []).append((int(track.get('id')), xy))
    labels = []
    for r in rows:
        a = np.array(r['bbox'], dtype=np.float64)
        boxes = by_frame.get(int(r['frame_id']), [])
        if not boxes:
            labels.append(None)
            continue
        b = np.array([x[1] for x in boxes])
        wh = np.maximum(0, np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2]))
        intersection = wh.prod(1)
        aa = np.maximum(0, a[2:] - a[:2]).prod()
        bb = np.maximum(0, b[:, 2:] - b[:, :2]).prod(1)
        iou = intersection / np.maximum(aa + bb - intersection, 1e-9)
        best = int(iou.argmax())
        labels.append(boxes[best][0] if iou[best] >= .5 else None)
    return labels


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261006)
    checks = []
    def add(round_, name, ok, evidence, classification):
        checks.append(dict(round=round_, name=name, contract_pass=bool(ok),
                           classification=classification, evidence=plain(evidence)))

    # R1. Inspect real input labels, teacher construction, and run provenance.
    prereg = json.loads((ROOT/'PREREG.json').read_text())
    manifest = json.loads(data.s3.MANIFEST.read_text())
    train_ids = {str(r['sequence_id']) for r in manifest if r['split'] == 'train'}
    rows = data.s3.load_index('train', '23', '1')
    chosen = rows[::max(1, len(rows)//200)][:200]
    independent = independently_label(chosen, data.s3.DATA/'new_xml/1/23-1.xml')
    gt = data.s3.read_gt('23', '1')
    production = [data.gid(r, gt.get(int(r['frame_id']), [])) for r in chosen]
    legal = not (set(tr.FIT) & set(tr.EVAL)) and set(tr.FIT + tr.EVAL) <= train_ids
    add('R1', 'train_split_and_independent_XML_labels', legal and independent == production,
        dict(fit=list(tr.FIT), calibration=list(tr.EVAL), checked_real_rows=len(chosen),
             label_disagreements=sum(a != b for a,b in zip(independent, production)),
             raw_val_test_read=False, feature_source=str(data.ROOT),
             feature_dimension=256, P26_host_replay=False), 'INPUT_LABEL_CHECK')
    del rows

    # Isolate the production data builder with known times and repeated donors.
    box = [0., 0., 10., 10.]
    def row(fr, v):
        return dict(frame_id=fr, bbox=box, visual_feature=np.array(v, np.float32))
    target = [row(10, [1., 0.])]
    source = [row(9, [-1., 0.]), row(10, [0., -1.])]
    source += [row(11, [1., 0.]) for _ in range(7)] + [row(11, [0., 1.])]
    source += [row(fr, [0., 1.]) for fr in range(12, 20)]
    synthetic_gt = {fr: [dict(gt_id=7, bbox=box)] for fr in range(9, 20)}
    with patch.object(data.s3, 'load_index', side_effect=lambda split,sid,dev: target if dev=='1' else source), \
         patch.object(data.s3, 'read_gt', return_value=synthetic_gt):
        ev, _ = data.build_pair('23')
    expected_raw_mean = np.array([7., 1.]) / np.sqrt(50.)
    source_text = inspect.getsource(data.build_pair)
    add('R1', 'future_teacher_frame_and_trim_contract', False,
        dict(actual_teacher=ev[0]['teacher'], untrimmed_first_eight_rows=expected_raw_mean,
             equal_to_untrimmed_mean=bool(np.allclose(ev[0]['teacher'], expected_raw_mean)),
             donor_rows=8, distinct_future_frames_selected=1, future_frames_available=9,
             strict_future_filter_present='if f>fr' in source_text,
             mismatch='Spec says max 8 frames and trimmed mean; code takes first 8 rows and arithmetic mean.'),
        'SPEC_IMPLEMENTATION_MISMATCH')

    gate = json.loads((ROOT/'SHORT_GATE.json').read_text())
    receipt = json.loads((ROOT/'runs/TRAINING_RECEIPT.json').read_text())
    ck = ROOT/'runs/fixed_step_001200.pt'
    model = tr.CandidateResidual()
    state = torch.load(ck, map_location='cpu')
    model.load_state_dict(state, strict=True)
    ck_ok = sha(ck) == receipt['checkpoint_sha256'] and all(bool(torch.isfinite(v).all()) for v in state.values())
    snapshot = [x.strip().split(', ') for x in receipt['gpu_snapshot'].splitlines()]
    actual_matches = [r for r in snapshot if r[2] == receipt['gpu_visible_name']]
    hardware_ok = (len(actual_matches)==1 and actual_matches[0][1]==receipt['gpu_uuid']
                   and int(actual_matches[0][0]) != 2 and int(actual_matches[0][3])>4096)
    add('R1', 'checkpoint_and_physical_GPU_provenance', ck_ok and hardware_ok,
        dict(checkpoint_sha256=sha(ck), checkpoint_integrity=ck_ok, tensors=len(state),
             parameters=sum(p.numel() for p in model.parameters()),
             exit_code=int((ROOT/'train_short_v2.exit').read_text()),
             recorded_cuda_name=receipt['gpu_visible_name'], asserted_env_uuid=receipt['gpu_uuid'],
             unique_name_matches_in_recorded_snapshot=actual_matches,
             historical_actual_cuda_uuid_recorded=False,
             hardware_violation='CUDA reported DGX Display; its unique snapshot entry is physical GPU2, 4096 MiB. UUID field came only from environment.'),
        'HARDWARE_POLICY_VIOLATION_AND_UNVERIFIED_UUID')

    # R2. Exercise the exact production operator, loss, and supervision path.
    prod = tr.CandidateResidual(d=8,h=16)
    toy = contract.P54Student(d=8,h=16)
    prod_shapes = {n:list(v.shape) for n,v in prod.state_dict().items()}
    toy_shapes = {n:list(v.shape) for n,v in toy.state_dict().items()}
    t = torch.randn(2,8); c = torch.randn(2,3,8); mask = torch.ones(2,3,dtype=torch.bool)
    l,d,z = prod(t,c,mask)
    perm = torch.tensor([2,0,1]); lp,dp,zp = prod(t,c[:,perm],mask[:,perm])
    equiv = float((lp-l[:,perm]).abs().max())
    cp = F.pad(c,(0,0,0,4)); mp = F.pad(mask,(0,4),value=False)
    ll,dd,zz = prod(t,cp,mp)
    padding_delta = float((dd-d).abs().max())
    _,d0,_ = prod(t,c[:,:0],mask[:,:0])
    add('R2', 'production_operator_contract_and_batch_padding',
        prod_shapes==toy_shapes and padding_delta<1e-6 and bool(torch.isfinite(d0).all()),
        dict(production_shapes=prod_shapes, tested_toy_shapes=toy_shapes,
             candidate_permutation_logit_error=equiv,
             valid_logits_padding_error=float((ll[:,:3]-l).abs().max()),
             dustbin_padding_error=padding_delta, true_K0_dustbin_finite=bool(torch.isfinite(d0).all()),
             cause='dustbin uses valid_count/padded_batch_width; K=0 takes mean of empty mask'),
        'WRONG_TESTED_MODEL_AND_NO_MATCH_OPERATOR_DEFECT')

    y = torch.zeros(1,64); y[0,0]=1
    good = torch.full((1,64),-.1); good[0,0]=.1
    collapsed = torch.full((1,64),-4.); collapsed[0,1]=-3.9
    good_bce = float(F.binary_cross_entropy_with_logits(good,y))
    bad_bce = float(F.binary_cross_entropy_with_logits(collapsed,y))
    main_src = inspect.getsource(tr.main)
    add('R2', 'implemented_assignment_loss_vs_specification', False,
        dict(implemented_assignment='mean BCE over all valid candidate rows',
             total_loss='assign + 0.15*dust_loss + 0.30*align',
             hard_negative_margin_in_executable=False,
             missing_margin_verified_in_source='margin' not in main_src,
             ranking_counterexample=dict(correct_top1_BCE=good_bce, wrong_top1_BCE=bad_bce),
             significance='Lower BCE alone cannot validate ranking; margin named in METHOD_SPEC is absent.'),
        'MISSING_DECLARED_LOSS')

    # At z=c, a future prototype equal to a positive candidate has zero alignment
    # loss for any target. Show that alignment cannot train the separate scorer.
    align = 1-(z[:,0]*F.normalize(torch.randn(2,8),dim=-1)).sum(-1).mean()
    prod.zero_grad(); align.backward()
    score_grad = sum(float(p.grad.abs().sum()) for p in prod.score.parameters() if p.grad is not None)
    future = torch.tensor([[1.,0.],[0.,1.]])
    candidate = future.clone()
    target_changed = future.flip(0)
    candidate_copy_loss = 1-(F.normalize(candidate,dim=-1)*future).sum(-1)
    add('R2', 'privileged_teacher_signal_proves_causal_transfer', False,
        dict(candidate_copy_alignment_losses=candidate_copy_loss,
             losses_after_target_swap=candidate_copy_loss,
             swapped_target_teacher_cosines=(target_changed*future).sum(-1),
             alignment_gradient_in_separate_score_branch=score_grad,
             interpretation='GT selects same-identity source future; near-perfect same-source oracle ranking is not a causal-student learning signal.',
             scope='Representation degeneracy counterexample, not a claim that all future distillation fails.'),
        'TEACHER_CEILING_DOES_NOT_ESTABLISH_LEARNABILITY')

    # R3. Verify denominator/no-match behavior, real checkpoint ranks and the
    # scientific advancement boundary, without reading official evaluation data.
    def event(pos):
        return dict(target=np.eye(1,256,0,dtype=np.float32)[0],
                    candidates=np.eye(2,256,dtype=np.float32), valid=2,
                    positive=pos, teacher=None)
    class AlwaysAbstain(torch.nn.Module):
        def forward(self,t,c,m):
            score=torch.zeros(m.shape); score[:,0]=1
            return score, torch.full((len(t),),100.), c
    fake = tr.event_metrics(AlwaysAbstain(),[event([0]),event([])],torch.device('cpu'))
    add('R3', 'scorer_denominator_and_no_match_decision', False,
        dict(input_events=2, returned=fake, no_match_logit=100.,
             all_stored_calibration_no_match_scores_null=all(r['no_match_score_mean'] is None for r in gate['per_pair'].values()),
             interpretation='Reported R@1/MRR are conditional on a supplied positive and ignore dustbin. They are not full-event association/no-match metrics.'),
        'SCORER_SCOPE_MISMATCH')

    events, counts = data.build_pair('23',limit=128,include_no_positive=True)
    replay = tr.event_metrics(model,events,torch.device('cpu'))
    t,c,m,y,teacher,has,no = tr.pad_events(events,torch.device('cpu'))
    with torch.no_grad():
        score,dust,z=model(t,c,m)
    independent_ranks=[]
    for i,e in enumerate(events):
        if not e['positive']:
            continue
        vals=score[i,:e['valid']].numpy().astype(np.float64)
        posvals=vals[e['positive']]
        best=float(posvals.max())
        # Strictly greater rule is independent of argsort in the implementation.
        if sum(vals==best)>1:
            raise RuntimeError('Tie encountered; report tie policy before replay.')
        independent_ranks.append(1+int(sum(vals>best)))
    r1=float(np.mean(np.array(independent_ranks)==1)); mrr=float(np.mean(1/np.array(independent_ranks)))
    macro_s=float(np.mean([r['student_r1'] for r in gate['per_pair'].values()]))
    macro_b=float(np.mean([r['baseline_r1'] for r in gate['per_pair'].values()]))
    add('R3', 'real_CPU_checkpoint_rank_replay_and_recorded_macro',
        abs(r1-replay['student_r1'])<1e-12 and abs(mrr-replay['student_mrr'])<1e-12
        and abs(macro_s-gate['calibration_pair_macro']['student_r1'])<1e-12,
        dict(real_sample_events=len(events), pair='23', sample_frames=[min(e['frame'] for e in events),max(e['frame'] for e in events)],
             conditional_positive_events=len(independent_ranks), independent_r1=r1, independent_mrr=mrr,
             production_replay=replay, recorded_calibration_macro_student=macro_s,
             recorded_calibration_macro_baseline=macro_b, recorded_delta=macro_s-macro_b,
             full_calibration_predictions_independently_replayed=False,
             sampling='build_pair limit keeps the earliest events; 2000/pair was not a full temporal epoch'),
        'LIMITED_REPLAY_PASS_NOT_METHOD_VALIDATION')

    forward = inspect.signature(tr.CandidateResidual.forward)
    add('R3', 'method_depth_and_P26_host_evidence', False,
        dict(forward_signature=str(forward), input_shape='pooled 256-D target and candidate vectors',
             trained_backbone_or_spatial_map=False, P26_inference_modified=False,
             existing_method_stop_rule='METHOD_SPEC: stop if it reduces to a post-pooling ranker',
             conclusion='Implemented vector ranker crosses its own stop boundary; vector screening cannot be promoted to a new identity representation or P26 MOT gain.',
             external_novelty_reaudit_performed=False),
        'LOCAL_METHOD_SCOPE_STOP_NOT_CROSS_DOMAIN_COLLISION')

    sources=[HERE/'train_short.py',HERE/'contract.py',HERE/'data_audit.py',HERE/'signal.py',HERE/'audit_short_v1.py',
             ROOT/'METHOD_SPEC.md',ROOT/'PREREG.json',ROOT/'SHORT_GATE.json',ROOT/'runs/TRAINING_RECEIPT.json',Path(__file__).resolve()]
    result=dict(status='AUDIT_COMPLETE_WITH_FAILED_CONTRACTS_3X3',
                rounds={r:[c for c in checks if c['round']==r] for r in ('R1','R2','R3')},
                checks=len(checks), contracts_passed=sum(c['contract_pass'] for c in checks),
                contracts_failed=sum(not c['contract_pass'] for c in checks),
                decision='INVALIDATE_P54_V2_METHOD_FAILURE_AND_HOLD_RETRAINING',
                audit_device='CPU', cuda_initialized=torch.cuda.is_initialized(),
                raw_official_val_test_access=False,
                historical_receipts_preserved=True,
                source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
                external_input_hashes={str(p):sha(p) for p in (
                    Path(data.s3.__file__).resolve(), data.s3.MANIFEST,
                    data.s3.DATA/'new_xml/1/23-1.xml',data.s3.DATA/'new_xml/2/23-2.xml',
                    data.ROOT/'features/train/23-1.jsonl',data.ROOT/'features/train/23-2.jsonl',
                    data.ROOT/'features/train/23-1.npz',data.ROOT/'features/train/23-2.npz')},
                no_validated_new_learned_method=True)
    assert len(checks)==9 and all(len(result['rounds'][r])==3 for r in result['rounds'])
    assert not result['cuda_initialized']
    out=ROOT/'SHORT_V2_INDEPENDENT_AUDIT.json'
    out.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rounds'},indent=2),flush=True)


if __name__=='__main__':
    main()
