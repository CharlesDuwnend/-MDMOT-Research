#!/usr/bin/env python3
"""Train isolated adapters through the frozen, native AutoAssign loss."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
from export_fpn import P26, CKPT, DETECTOR_CFG, UUID, sha, dump, source_bindings
from temporal_module import TemporalAdapter, validate_temporal_group
import numpy as np
import torch

LABEL_MANIFEST = Path('/home/chenhc/mdmot_research_20261002/p0/data/sequences_train.json')
LABEL_PARSER = LABEL_MANIFEST.parent / 'build_detection_labels.py'


def model_init():
    import mmcv, mmdet, mmtrack
    from mdmt_compat_runtime import init_model_mdmt
    from export_fpn import environment
    runtime = environment(torch, np, mmcv, mmdet, mmtrack)
    model = init_model_mdmt(str(DETECTOR_CFG), str(CKPT), device='cuda:0')
    model.eval().requires_grad_(False)
    assert model.detector.bbox_head.prior_generator.offset == 0
    return model, runtime


def state_sha(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


class CachedGroups:
    def __init__(self, role, verify_archives=False):
        self.role = role
        self.feature = json.loads((HERE / role / 'FEATURE_INDEX.json').read_text())
        self.flow = json.loads((HERE / role / 'FLOW_INDEX.json').read_text())
        self.groups = [g for g in json.loads((HERE / 'GROUPS.json').read_text()) if g['role'] == role]
        self.by_key = {g['key']: g for g in self.groups}
        fr = json.loads((HERE / role / 'FEATURE_RECEIPT.json').read_text())
        fv = json.loads((HERE / role / 'FEATURE_VERIFY.json').read_text())
        rr = json.loads((HERE / role / 'FLOW_RECEIPT.json').read_text())
        assert fv['status'] == 'PASS_NATIVE_FPN_ARCHIVE_VERIFICATION'
        assert fv['receipt_sha256'] == sha(HERE / role / 'FEATURE_RECEIPT.json')
        assert fv['feature_index_sha256'] == sha(HERE / role / 'FEATURE_INDEX.json')
        assert rr['status'] == 'COMPLETE_FROZEN_CAUSAL_FLOW'
        assert rr['flow_index_sha256'] == sha(HERE / role / 'FLOW_INDEX.json')
        assert self.feature['role'] == self.flow['role'] == rr['role'] == role
        assert rr['groups'] == len(self.groups)
        assert set(self.flow['by_key']) == set(self.by_key)
        if verify_archives:
            for e in self.feature['entries']:
                assert sha(e['archive_path']) == e['archive_sha256'], e['key']
            for e in self.flow['entries']:
                assert sha(e['path']) == e['sha256'], e['key']
        for group in self.groups:
            records = [self.feature['entries'][self.feature['by_key'][k]] for k in group['image_keys']]
            assert records[0]['key'] == group['key'] and records[0]['frame'] == group['frame']
            validate_temporal_group(pair=group['pair'], view=group['view'], current_frame=group['frame'], reference_records=records[1:])
            flow = self.flow['entries'][self.flow['by_key'][group['key']]]
            assert flow['image_keys'] == group['image_keys']
            assert all(x['role'] == role for x in records)
        self.verified_archives = verify_archives

    def load(self, key, past=True):
        group = self.by_key[key]
        records = [self.feature['entries'][self.feature['by_key'][k]] for k in group['image_keys']]
        levels = []
        native = None
        for ri, e in enumerate(records if past else records[:1]):
            with np.load(e['archive_path'], allow_pickle=False) as z:
                levels.append([torch.from_numpy(z['f%d' % l]).cuda() for l in range(5)])
                if ri == 0: native = [z['det%d' % c].copy() for c in range(3)]
        meta = copy.deepcopy(records[0]['metadata'])
        meta['scale_factor'] = np.asarray(meta['scale_factor'], dtype=np.float32)
        geometry = dict(original_hw=meta['ori_shape'][:2], detector_img_hw=meta['img_shape'][:2], pad_hw=meta['pad_shape'][:2])
        ref = flow = None
        if past:
            ref = [torch.cat([levels[r][l] for r in range(1, 4)], dim=0) for l in range(5)]
            e = self.flow['entries'][self.flow['by_key'][key]]
            with np.load(e['path'], allow_pickle=False) as z:
                assert z['lags'].tolist() == [1,4,8]
                assert z['original_hw'].tolist() == geometry['original_hw']
                flow = torch.from_numpy(z['flow']).cuda()
                assert flow.dtype == torch.float32 and bool(torch.isfinite(flow).all())
        return levels[0], ref, flow, meta, geometry, native


def load_targets(groups, role):
    """Only call after input/prediction seal appropriate to role."""
    config = json.loads((HERE / 'CONFIG.json').read_text())
    allowed = config['fit_pairs' if role == 'fit' else 'calibration_pairs']
    assert {g['pair'] for g in groups} == set(allowed)
    spec = importlib.util.spec_from_file_location('p28_original_xml_parser', LABEL_PARSER)
    parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
    manifest = {(x['pair'],str(x['view'])):x for x in json.loads(LABEL_MANIFEST.read_text())}
    frames = {}; bindings = {str(LABEL_PARSER):sha(LABEL_PARSER), str(LABEL_MANIFEST):sha(LABEL_MANIFEST)}
    for pair in allowed:
        for view in ('1','2'):
            record = manifest[(pair, view)]
            value, _, path = parser.read_xml(pair, view, record['xml_sha256'])
            frames[(pair,view)] = value; bindings[str(path)] = record['xml_sha256']
    targets = {g['key']:[x for x in frames[(g['pair'],str(g['view']))].get(g['frame'],[]) if x['class'] in (0,1,2)] for g in groups}
    return targets, bindings


def target_tensors(rows, meta):
    boxes = np.asarray([r['bbox'] for r in rows], dtype=np.float32).reshape(-1,4)
    boxes *= meta['scale_factor']
    h,w = meta['img_shape'][:2]
    boxes[:,0::2] = np.clip(boxes[:,0::2],0,w)
    boxes[:,1::2] = np.clip(boxes[:,1::2],0,h)
    assert np.all(boxes[:,2:] > boxes[:,:2]), 'Degenerate target after native resize'
    labels = np.asarray([r['class'] for r in rows], dtype=np.int64)
    return torch.from_numpy(boxes).cuda(), torch.from_numpy(labels).cuda()


def make_adapter(mode, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    adapter = TemporalAdapter(mode=mode).cuda()
    if mode in ('single','uniform'): adapter.embedding.requires_grad_(False)
    return adapter


def loss_forward(head, features, meta, rows):
    boxes,labels = target_tensors(rows,meta)
    losses = head.forward_train(features,[meta],[boxes],[labels])
    assert set(losses) == {'loss_pos','loss_neg','loss_center'}
    total = sum(losses.values())
    assert bool(torch.isfinite(total))
    return total, {k:float(v.detach()) for k,v in losses.items()}


def gradient_norm(parameters):
    grads = [p.grad for p in parameters if p.grad is not None]
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    return float(torch.sqrt(sum(g.float().square().sum() for g in grads))) if grads else 0.0


def build_schedule(groups, protocol):
    rng = np.random.RandomState(protocol['seed']); schedule=[]
    pairs=sorted({g['pair'] for g in groups})
    for epoch in range(protocol['epochs']):
        order=list(rng.permutation(pairs))
        choices={p:list(rng.permutation([g['key'] for g in groups if g['pair']==p])) for p in pairs}
        assert {len(x) for x in choices.values()} == {16}
        for round_id in range(16):
            for p in order: schedule.append(str(choices[p][round_id]))
        assert len(set(schedule[-240:])) == 240
    assert len(schedule)==protocol['steps']
    return schedule


def smoke(model, dataset, targets, protocol, out):
    from mmdet.core import bbox2result
    head=model.detector.bbox_head
    key=dataset.groups[0]['key']
    current,past,flow,meta,geometry,native=dataset.load(key)
    before=state_sha(model)
    single=make_adapter('single',protocol['seed'])
    with torch.no_grad():
        ident=single(current)
        assert all(torch.equal(a,b) for a,b in zip(ident,current))
        det=head.simple_test(ident,[meta],rescale=True)[0]
        boxes=bbox2result(det[0],det[1],3)
        parity=[bool(np.array_equal(a,b)) for a,b in zip(boxes,native)]
    assert all(parity),'Native cache -> identity adapter -> head parity failed'
    record={'key':key,'native_exact_array_equal':parity,'real_gt_count':len(targets[key]),'arms':{}}
    for arm,mode in protocol['arms'].items():
        adapter=make_adapter(mode,protocol['seed']); adapter.train()
        initial=state_sha(adapter)
        opt=torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad],lr=protocol['lr'],weight_decay=protocol['weight_decay'])
        outputs=adapter(current,past,flow,**geometry)
        total,losses=loss_forward(head,outputs,meta,targets[key]);total.backward()
        value_grad=gradient_norm(adapter.value_adapter.parameters())
        embedding_grad=gradient_norm(adapter.embedding.parameters())
        assert value_grad>0 and (embedding_grad>0 if mode=='fgfa' else embedding_grad==0)
        assert all(p.grad is None for p in model.parameters())
        opt.step()
        assert state_sha(adapter)!=initial
        record['arms'][arm]=dict(losses=losses,value_gradient_norm=value_grad,embedding_gradient_norm=embedding_grad,parameter_counts=adapter.parameter_counts(),real_update_verified=True)
        del outputs,total,adapter,opt
    assert state_sha(model)==before
    record.update(status='PASS_REAL_GPU_SMOKE',frozen_model_state_sha256=before,labels_role='fit',synthetic_performance_claim=False)
    dump(out/'SMOKE.json',record)
    return before


def main():
    args=argparse.ArgumentParser();args.add_argument('--smoke-only',action='store_true');args=args.parse_args()
    torch.set_num_threads(2)
    protocol=json.loads((HERE/'TRAIN_PROTOCOL.json').read_text())
    out=HERE/('smoke' if args.smoke_only else 'training')
    out.mkdir(exist_ok=True)
    assert not (out/'PRE_TRAIN.json').exists(),'Preserve existing attempt'
    inputs=source_bindings()
    for p in [HERE/'TRAIN_PROTOCOL.json',HERE/'temporal_module.py',Path(__file__),HERE/'run_train.sh']:
        inputs[str(p)]=sha(p)
    for f in ['FEATURE_INDEX.json','FEATURE_RECEIPT.json','FEATURE_VERIFY.json','FEATURE_COMPLETION.json','FLOW_INDEX.json','FLOW_RECEIPT.json','FLOW_PROTOCOL.json']:
        p=HERE/'fit'/f;inputs[str(p)]=sha(p)
    dataset=CachedGroups('fit',verify_archives=True)
    schedule=build_schedule(dataset.groups,protocol)
    dump(out/'SCHEDULE.json',schedule);inputs[str(out/'SCHEDULE.json')]=sha(out/'SCHEDULE.json')
    dump(out/'INPUTS_VERIFIED_BEFORE_LABELS.json',dict(status='PASS',feature_archives=960,flow_archives=240,inputs=inputs))
    targets,gt_bindings=load_targets(dataset.groups,'fit')
    inputs.update(gt_bindings)
    model,runtime=model_init();head=model.detector.bbox_head
    dump(out/'PRE_TRAIN.json',dict(protocol=protocol,inputs=inputs,runtime=runtime,current_group_count=len(targets),target_count=sum(map(len,targets.values())),labels_only='fit',calibration_labels_read=False))
    frozen=smoke(model,dataset,targets,protocol,out)
    print(json.dumps({'event':'REAL_GPU_SMOKE_PASS'}),flush=True)
    if args.smoke_only:return
    checkpoints={};start=time.monotonic();shared_value=None
    for arm,mode in protocol['arms'].items():
        armout=out/arm;armout.mkdir(exist_ok=False)
        adapter=make_adapter(mode,protocol['seed']);adapter.train()
        value_init=state_sha(adapter.value_adapter)
        if shared_value is None:shared_value=value_init
        assert value_init==shared_value
        opt=torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad],lr=protocol['lr'],weight_decay=protocol['weight_decay'])
        init_sha=state_sha(adapter);t0=time.monotonic();running=[]
        with (armout/'LOSSES.jsonl').open('x') as log:
            for step,key in enumerate(schedule,1):
                current,past,flow,meta,geometry,_=dataset.load(key,past=mode!='single')
                opt.zero_grad(set_to_none=True)
                outputs=adapter(current,past,flow,**geometry)
                total,losses=loss_forward(head,outputs,meta,targets[key]);total.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad],protocol['gradient_clip_norm'])
                assert bool(torch.isfinite(norm))
                opt.step();running.append(float(total.detach()))
                row=dict(step=step,key=key,loss=float(total.detach()),losses=losses,preclip_gradient_norm=float(norm))
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if step%30==0 or step==1:
                    log.flush();progress=dict(arm=arm,step=step,total_steps=len(schedule),mean_recent_loss=float(np.mean(running[-30:])),seconds=time.monotonic()-t0,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    dump(out/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
                del current,past,flow,outputs,total
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert state_sha(model)==frozen,'Frozen detector changed'
        checkpoint=armout/'final.pth'
        torch.save(dict(state_dict=adapter.state_dict(),arm=arm,mode=mode,steps=len(schedule),protocol_sha256=sha(HERE/'TRAIN_PROTOCOL.json'),module_sha256=sha(HERE/'temporal_module.py'),schedule_sha256=sha(out/'SCHEDULE.json'),seed=protocol['seed']),checkpoint)
        checkpoints[arm]=dict(path=str(checkpoint),sha256=sha(checkpoint),parameter_counts=adapter.parameter_counts(),initial_state_sha256=init_sha,initial_value_sha256=value_init,final_state_sha256=state_sha(adapter),mean_first30=float(np.mean(running[:30])),mean_last30=float(np.mean(running[-30:])),seconds=time.monotonic()-t0)
        dump(armout/'RECEIPT.json',dict(status='COMPLETE_FIXED_FINAL',**checkpoints[arm],steps=len(schedule),frozen_model_state_sha256=frozen))
        del adapter,opt
    for path,expected in inputs.items():assert sha(path)==expected,'Bound input changed: '+path
    dump(out/'TRAIN_RECEIPT.json',dict(status='COMPLETE_THREE_ARM_FIT',checkpoints=checkpoints,steps_per_arm=len(schedule),groups=240,seconds=time.monotonic()-start,inputs=inputs,protocol_sha256=sha(HERE/'TRAIN_PROTOCOL.json'),frozen_model_state_sha256=frozen,calibration_labels_read=False,official_val_or_test_read=False,runtime=runtime))


if __name__=='__main__':
    main()
