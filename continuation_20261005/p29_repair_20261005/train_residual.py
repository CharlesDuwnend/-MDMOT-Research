#!/usr/bin/env python3
"""P29 fixed three-arm FIT training; frozen native AutoAssign supervision.

Execution is explicit. Import and syntax inspection never open annotation XML,
load checkpoints, construct models or initialize CUDA.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
P28 = HERE.parent / 'p28_causal_fgfa'
P28_TRAIN_RECEIPT_SHA = '15b7fadc79183132a8fc3386297e56902988258c29fb7cdfb04534131cc0bb1e'
MODES = ('current_only', 'past_fixed_offsets', 'past_learned_offsets')
PARAMETER_GROUPS = {'query_conditioner', 'value_projection', 'output_projection',
                    'sampling_offsets', 'attention_weights'}
SMOKE_UPDATES = 4

# Keep the imported helper's own HERE pointing at P28 so its sealed caches and
# original parser/config paths are reused without copying or editing that code.
sys.path.insert(0, str(P28))
_spec = importlib.util.spec_from_file_location('p29_p28_readonly_training_helpers', P28 / 'train_adapter.py')
_p28 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p28)
# Use repair-local cache metadata and allow any authorized A100 UUID.
_p28.HERE = HERE
_REPAIR_UUID = __import__('os').environ.get('MDMT_GPU_UUID', 'GPU-1b297aba-ae7e-e326-5903-476f1bb683d7')
_p28.UUID = _REPAIR_UUID
CachedGroups = _p28.CachedGroups
load_targets = _p28.load_targets
target_tensors = _p28.target_tensors
loss_forward = _p28.loss_forward
state_sha = _p28.state_sha
model_init = _p28.model_init
build_schedule = _p28.build_schedule
sha = _p28.sha
dump = _p28.dump
import numpy as np
import torch

def _portable_environment(torch_mod, np_mod, mmcv, mmdet, mmtrack):
    import os, subprocess
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == _REPAIR_UUID
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'], text=True)
    row = next(x for x in rows.splitlines() if _REPAIR_UUID in x)
    fields = [x.strip() for x in row.split(',')]
    prop = torch_mod.cuda.get_device_properties(0)
    assert fields[1] == _REPAIR_UUID and 'A100' in fields[2] and int(fields[3]) >= 40000
    assert 'A100' in prop.name and prop.total_memory > 39 * 2**30
    return dict(python=sys.version, python_executable=sys.executable, torch=torch_mod.__version__, numpy=np_mod.__version__,
        mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmtrack_path=mmtrack.__file__,
        physical_gpu=dict(index=int(fields[0]), uuid=fields[1], name=fields[2], memory_total_mib=int(fields[3]), memory_used_mib_at_start=int(fields[4])),
        logical_gpu=dict(index=0, name=prop.name, total_memory_bytes=prop.total_memory), torch_threads=torch_mod.get_num_threads(),
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], cudnn_benchmark=torch_mod.backends.cudnn.benchmark,
        cudnn_deterministic=torch_mod.backends.cudnn.deterministic, cudnn_allow_tf32=torch_mod.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch_mod.backends.cuda.matmul.allow_tf32)

def model_init():
    import mmcv, mmdet, mmtrack
    from mdmt_compat_runtime import init_model_mdmt
    runtime = _portable_environment(torch, np, mmcv, mmdet, mmtrack)
    model = init_model_mdmt(str(_p28.DETECTOR_CFG), str(_p28.CKPT), device='cuda:0')
    model.eval().requires_grad_(False)
    assert model.detector.bbox_head.prior_generator.offset == 0
    return model, runtime

# P29 remains importable by the prediction entrypoint without touching labels.
sys.path.insert(0, str(HERE))
from residual_deformable import ResidualDeformableAdapter


def make_adapter(mode, seed):
    assert mode in MODES
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return ResidualDeformableAdapter(mode=mode, channels=256, strides=(8, 16, 32, 64, 128)).cuda()


def native_capacity(head):
    cfg = head.test_cfg
    return dict(nms_pre=int(cfg.nms_pre), max_per_img=int(cfg.max_per_img),
                score_thr=float(cfg.score_thr), nms_iou=float(cfg.nms.iou_threshold))


def apply_native_capacity(head, protocol):
    expected = dict(nms_pre=2000, max_per_img=300, score_thr=.05, nms_iou=.6)
    assert protocol['native_detection_config'] == expected
    head.test_cfg = copy.deepcopy(head.test_cfg)
    head.test_cfg.nms_pre = expected['nms_pre']
    head.test_cfg.max_per_img = expected['max_per_img']
    head.test_cfg.score_thr = expected['score_thr']
    head.test_cfg.nms.iou_threshold = expected['nms_iou']
    assert native_capacity(head) == expected
    return expected


def validate_protocol(protocol, config):
    required = dict(seed=42, steps=1200, epochs=5, groups_per_epoch=240,
                    optimizer='AdamW', lr=1e-4, weight_decay=1e-4,
                    gradient_clip_norm=10.0, batch_size=1)
    for key, value in required.items():
        assert protocol[key] == value, ('Protocol changed', key, protocol[key], value)
    assert len(protocol['arms']) == 3 and set(protocol['arms'].values()) == set(MODES)
    assert protocol['physical_gpu_uuid'] == _p28.UUID
    assert config['native_capacity'] == protocol['native_detection_config']
    assert Path(config['cache_source']).resolve() == P28
    assert config['lags'] == [0, 1, 4, 8]
    assert SMOKE_UPDATES >= 4


def bind_path(inputs, path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None:
        assert digest == expected, 'Pinned input changed: ' + str(path)
    if str(path) in inputs:
        assert inputs[str(path)] == digest
    inputs[str(path)] = digest
    return path


def source_bindings():
    """All receipt keys are absolute, including imported __file__ values."""
    inputs = {}
    legacy = bind_path(inputs, P28 / 'training/TRAIN_RECEIPT.json', P28_TRAIN_RECEIPT_SHA)
    legacy_inputs = json.loads(legacy.read_text())['inputs']
    assert Path(_p28.__file__).resolve() == P28 / 'train_adapter.py'
    for name in ('train_adapter.py', 'export_fpn.py', 'temporal_module.py'):
        path = P28 / name
        expected = legacy_inputs.get(str(path), legacy_inputs.get(name))
        assert expected is not None, 'Missing sealed helper hash: ' + name
        bind_path(inputs, path, expected)
    for path, expected in _p28.source_bindings().items():
        bind_path(inputs, path, expected)
    for name in ('CONFIG.json', 'GROUPS.json', 'FRAMES.json', 'PLAN_INPUTS.json',
                 'TRAIN_PROTOCOL.json', 'run_train.sh'):
        bind_path(inputs, HERE / name)
    # Include the complete local operator port and any local prior-art/source
    # audit metadata present before this attempt. Runtime output directories are
    # excluded; prediction/report source may exist and is explicitly bound too.
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or any(part in {'training', 'smoke', '__pycache__'} for part in path.relative_to(HERE).parts):
            continue
        if path.suffix in {'.py', '.sh'}:
            bind_path(inputs, path)
    bind_path(inputs, Path(__file__).resolve())
    import residual_deformable as operator
    if hasattr(operator, 'source_bindings'):
        for path, expected in operator.source_bindings().items():
            bind_path(inputs, path, expected)
    for name in ('FEATURE_INDEX.json', 'FEATURE_RECEIPT.json', 'FEATURE_VERIFY.json',
                 'FEATURE_COMPLETION.json', 'FLOW_INDEX.json', 'FLOW_RECEIPT.json',
                 'FLOW_PROTOCOL.json', 'RESOLVED_DETECTOR_CONFIG.json'):
        bind_path(inputs, P28 / 'fit' / name)
    feature_receipt = json.loads((P28 / 'fit/FEATURE_RECEIPT.json').read_text())
    for path, expected in feature_receipt['imported_runtime_module_hashes'].items():
        bind_path(inputs, path, expected)
    for path in (_p28.LABEL_MANIFEST, _p28.LABEL_PARSER):
        bind_path(inputs, path, legacy_inputs[str(path)])
    assert all(Path(path).is_absolute() for path in inputs)
    return inputs, legacy_inputs


def parameter_contract(adapter, mode):
    groups = {k: tuple(v) for k, v in adapter.parameter_groups().items()}
    assert set(groups) == PARAMETER_GROUPS
    all_ids = [id(p) for values in groups.values() for p in values]
    assert len(all_ids) == len(set(all_ids)), 'Parameter groups overlap'
    assert set(all_ids) == {id(p) for p in adapter.parameters()}, 'Ungrouped parameters'
    for name, values in groups.items():
        assert values, ('Empty parameter group', name)
        active = not (mode == 'past_fixed_offsets' and name == 'sampling_offsets')
        assert all(p.requires_grad == active for p in values), (mode, name)
    return groups


def group_snapshot(groups):
    return {name: [p.detach().cpu().clone() for p in values] for name, values in groups.items()}


def group_delta(values, initial):
    return float(np.sqrt(sum(float((p.detach().cpu().double() - before.double()).square().sum())
                             for p, before in zip(values, initial))))


def gradient_norm(values):
    grads = [p.grad.detach() for p in values if p.grad is not None]
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    return float(torch.sqrt(sum(g.float().square().sum() for g in grads))) if grads else 0.0


def decode(head, features, meta):
    from mmdet.core import bbox2result
    det = head.simple_test(features, [copy.deepcopy(meta)], rescale=True)[0]
    return bbox2result(det[0], det[1], 3)


def exact_arrays(left, right):
    return [bool(a.shape == b.shape and np.array_equal(a, b)) for a, b in zip(left, right)]


def optimizer(adapter, protocol):
    parameters = [p for p in adapter.parameters() if p.requires_grad]
    return torch.optim.AdamW(parameters, lr=protocol['lr'], weight_decay=protocol['weight_decay'])


def assert_frozen(model):
    assert not model.training and not model.detector.training
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def smoke(model, dataset, targets, protocol, out):
    """Real FIT cache/head/gradient smoke; actual training starts from new objects."""
    head = model.detector.bbox_head
    key = dataset.groups[0]['key']
    current, past, flow, meta, geometry, cached_native = dataset.load(key, past=True)
    assert_frozen(model)
    frozen = state_sha(model)
    original_capacity = native_capacity(head)
    assert original_capacity == dict(nms_pre=1000, max_per_img=100, score_thr=.05, nms_iou=.6)
    with torch.no_grad():
        original_native = decode(head, current, meta)
    cache_parity = exact_arrays(original_native, cached_native)
    assert len(cache_parity) == 3 and all(cache_parity), 'Fresh cap100 head no longer matches sealed native cache'
    capacity = apply_native_capacity(head, protocol)
    with torch.no_grad():
        native300 = decode(head, current, meta)
    record = dict(key=key, labels_role='fit', real_gt_count=len(targets[key]),
                  native_cache_capacity=original_capacity, native_cached_cap100_exact_array_equal=cache_parity,
                  active_all_arm_capacity=capacity, native_cap100_count=sum(map(len, original_native)),
                  native_cap300_count=sum(map(len, native300)), updates_per_arm=SMOKE_UPDATES, arms={})
    shared_initial = None
    for arm, mode in protocol['arms'].items():
        adapter = make_adapter(mode, protocol['seed'])
        adapter.train()
        groups = parameter_contract(adapter, mode)
        initial = state_sha(adapter)
        if shared_initial is None:
            shared_initial = initial
        assert initial == shared_initial, 'Arms did not start from identical tensor initialization'
        before = group_snapshot(groups)
        with torch.no_grad():
            features = adapter(current, past, flow, **geometry)
            identity_features = [bool(torch.equal(a, b)) for a, b in zip(features, current)]
            identity_decode = exact_arrays(decode(head, features, meta), native300)
        assert len(identity_features) == 5 and all(identity_features), (mode, 'Non-identity zero-output initialization')
        assert len(identity_decode) == 3 and all(identity_decode), (mode, 'Identity cap300 head mismatch')
        opt = optimizer(adapter, protocol)
        steps = []
        max_gradients = {name: 0.0 for name in groups}
        for step in range(1, SMOKE_UPDATES + 1):
            opt.zero_grad(set_to_none=True)
            features = adapter(current, past, flow, **geometry)
            total, losses = loss_forward(head, features, meta, targets[key])
            total.backward()
            norms = {name: gradient_norm(values) for name, values in groups.items()}
            for name, norm in norms.items():
                max_gradients[name] = max(max_gradients[name], norm)
            clip = torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad],
                                                  protocol['gradient_clip_norm'])
            assert bool(torch.isfinite(clip))
            assert_frozen(model)
            opt.step()
            steps.append(dict(step=step, loss=float(total.detach()), losses=losses,
                              group_gradient_norms=norms, preclip_gradient_norm=float(clip)))
            del features, total
        deltas = {name: group_delta(values, before[name]) for name, values in groups.items()}
        for name in groups:
            active = not (mode == 'past_fixed_offsets' and name == 'sampling_offsets')
            if active:
                assert max_gradients[name] > 0 and deltas[name] > 0, (mode, name, 'Inactive after staged real updates')
            else:
                assert max_gradients[name] == 0 and deltas[name] == 0, 'Fixed offset parameters changed'
        current_independence = None
        if mode == 'current_only':
            with torch.no_grad():
                supplied = adapter(current, past, flow, **geometry)
                absent = adapter(current, None, None, **geometry)
            current_independence = [bool(torch.equal(a, b)) for a, b in zip(supplied, absent)]
            assert all(current_independence), 'current_only uses supplied past/flow after learning'
            del supplied, absent
        assert state_sha(adapter) != initial
        record['arms'][arm] = dict(mode=mode, initial_state_sha256=initial,
            identity_feature_equal=identity_features, identity_cap300_exact_array_equal=identity_decode,
            parameter_counts=adapter.parameter_counts(), updates=steps, maximum_group_gradient_norms=max_gradients,
            group_parameter_delta_l2=deltas, current_only_past_flow_independence=current_independence,
            active_groups_nonzero_update_verified=True, fixed_offsets_unchanged=(mode == 'past_fixed_offsets'),
            disposable_smoke_adapter=True)
        del adapter, opt, groups, before
    assert_frozen(model)
    assert state_sha(model) == frozen, 'Frozen detector changed in smoke'
    assert native_capacity(head) == capacity
    record.update(status='PASS_REAL_GPU_SMOKE', frozen_model_state_sha256=frozen,
                  reset_before_actual_fit=True, synthetic_performance_claim=False)
    dump(out / 'SMOKE.json', record)
    return frozen, {arm: value['initial_state_sha256'] for arm, value in record['arms'].items()}


def verify_inputs(inputs):
    assert all(Path(path).is_absolute() for path in inputs)
    for path, expected in inputs.items():
        assert sha(path) == expected, 'Bound input changed: ' + path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    protocol = json.loads((HERE / 'TRAIN_PROTOCOL.json').read_text())
    config = json.loads((HERE / 'CONFIG.json').read_text())
    validate_protocol(protocol, config)
    out = HERE / ('smoke' if args.smoke_only else 'training')
    assert not out.exists(), 'Preserve existing attempt: ' + str(out)
    out.mkdir(exist_ok=False)
    inputs, legacy_inputs = source_bindings()
    dataset = CachedGroups('fit', verify_archives=True)
    p29_groups = [g for g in json.loads((HERE / 'GROUPS.json').read_text()) if g['role'] == 'fit']
    assert dataset.groups == p29_groups
    assert len(dataset.groups) == 240 and {g['pair'] for g in dataset.groups} == set(config['fit_pairs'])
    p28_config = json.loads((P28 / 'CONFIG.json').read_text())
    assert config['fit_pairs'] == p28_config['fit_pairs']
    for entry in dataset.feature['entries']:
        inputs[str(Path(entry['archive_path']).resolve())] = entry['archive_sha256']
    for entry in dataset.flow['entries']:
        inputs[str(Path(entry['path']).resolve())] = entry['sha256']
    schedule = build_schedule(dataset.groups, protocol)
    previous_schedule_path = bind_path(inputs, P28 / 'training/SCHEDULE.json')
    assert schedule == json.loads(previous_schedule_path.read_text()), 'Schedule differs from P28 paired FIT control'
    dump(out / 'SCHEDULE.json', schedule)
    bind_path(inputs, out / 'SCHEDULE.json')
    dump(out / 'INPUTS_VERIFIED_BEFORE_LABELS.json', dict(status='PASS', feature_archives=960,
         flow_archives=240, full_archive_hash_verification=True, inputs=inputs, role='fit', calibration_labels_read=False))
    targets, gt_bindings = load_targets(dataset.groups, 'fit')
    for path, expected in gt_bindings.items():
        assert legacy_inputs[str(Path(path).resolve())] == expected, 'FIT label lineage changed'
        bind_path(inputs, path, expected)
    model, runtime = model_init()
    head = model.detector.bbox_head
    assert_frozen(model)
    dump(out / 'PRE_TRAIN.json', dict(protocol=protocol, inputs=inputs, runtime=runtime,
         current_group_count=len(targets), target_count=sum(map(len, targets.values())), labels_only='fit',
         calibration_labels_read=False, calibration_status=config['calibration_status'], amp=False,
         imported_helper_source=str(Path(_p28.__file__).resolve()), native_capacity_before_override=native_capacity(head)))
    frozen, smoke_initial = smoke(model, dataset, targets, protocol, out)
    print(json.dumps(dict(event='REAL_GPU_SMOKE_PASS', updates_per_arm=SMOKE_UPDATES,
                          native_capacity=native_capacity(head))), flush=True)
    if args.smoke_only:
        verify_inputs(inputs)
        dump(out / 'SMOKE_RECEIPT.json', dict(status='COMPLETE_P29_REAL_GPU_SMOKE_ONLY', inputs=inputs,
             smoke_sha256=sha(out / 'SMOKE.json'), frozen_model_state_sha256=frozen, runtime=runtime,
             calibration_labels_read=False, official_val_or_test_read=False))
        return
    checkpoints = {}
    start = time.monotonic()
    shared_initial = None
    for arm, mode in protocol['arms'].items():
        armout = out / arm
        armout.mkdir(exist_ok=False)
        adapter = make_adapter(mode, protocol['seed'])
        adapter.train()
        groups = parameter_contract(adapter, mode)
        initial = state_sha(adapter)
        assert initial == smoke_initial[arm], 'Fresh reset after smoke failed'
        if shared_initial is None:
            shared_initial = initial
        assert initial == shared_initial, 'Actual FIT initial state differs by arm'
        group_initial = group_snapshot(groups)
        opt = optimizer(adapter, protocol)
        t0 = time.monotonic()
        running = []
        torch.cuda.reset_peak_memory_stats()
        with (armout / 'LOSSES.jsonl').open('x') as log:
            for step, key in enumerate(schedule, 1):
                current, past, flow, meta, geometry, _ = dataset.load(key, past=(mode != 'current_only'))
                opt.zero_grad(set_to_none=True)
                outputs = adapter(current, past, flow, **geometry)
                total, losses = loss_forward(head, outputs, meta, targets[key])
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad],
                                                      protocol['gradient_clip_norm'])
                assert bool(torch.isfinite(norm))
                assert_frozen(model)
                opt.step()
                running.append(float(total.detach()))
                row = dict(step=step, key=key, loss=running[-1], losses=losses, preclip_gradient_norm=float(norm))
                log.write(json.dumps(row, allow_nan=False) + '\n')
                if step % 30 == 0 or step == 1:
                    log.flush()
                    progress = dict(arm=arm, mode=mode, step=step, total_steps=len(schedule),
                                    mean_recent_loss=float(np.mean(running[-30:])), seconds=time.monotonic() - t0,
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    dump(out / 'PROGRESS.json', progress)
                    print(json.dumps(progress), flush=True)
                del current, past, flow, outputs, total
        assert_frozen(model)
        assert state_sha(model) == frozen, 'Frozen detector changed'
        assert native_capacity(head) == protocol['native_detection_config']
        deltas = {name: group_delta(values, group_initial[name]) for name, values in groups.items()}
        assert all(delta > 0 for name, delta in deltas.items()
                   if not (mode == 'past_fixed_offsets' and name == 'sampling_offsets'))
        if mode == 'past_fixed_offsets':
            assert deltas['sampling_offsets'] == 0
        checkpoint = armout / 'final.pth'
        torch.save(dict(state_dict=adapter.state_dict(), arm=arm, mode=mode, steps=len(schedule),
                        protocol_sha256=sha(HERE / 'TRAIN_PROTOCOL.json'), module_sha256=sha(HERE / 'residual_deformable.py'),
                        trainer_sha256=sha(Path(__file__).resolve()), schedule_sha256=sha(out / 'SCHEDULE.json'),
                        native_detection_config=protocol['native_detection_config'], seed=protocol['seed']), checkpoint)
        checkpoints[arm] = dict(path=str(checkpoint.resolve()), sha256=sha(checkpoint), mode=mode,
            parameter_counts=adapter.parameter_counts(), initial_state_sha256=initial, fresh_reset_after_smoke=True,
            final_state_sha256=state_sha(adapter), group_parameter_delta_l2=deltas,
            mean_first30=float(np.mean(running[:30])), mean_last30=float(np.mean(running[-30:])),
            seconds=time.monotonic() - t0, losses_sha256=sha(armout / 'LOSSES.jsonl'))
        dump(armout / 'RECEIPT.json', dict(status='COMPLETE_FIXED_FINAL', **checkpoints[arm], steps=len(schedule),
             frozen_model_state_sha256=frozen, native_detection_config=protocol['native_detection_config']))
        del adapter, opt, groups, group_initial
    verify_inputs(inputs)
    dump(out / 'TRAIN_RECEIPT.json', dict(status='COMPLETE_THREE_ARM_FIT', schema='p29_fixed_final_training_v1',
         checkpoints=checkpoints, steps_per_arm=len(schedule), groups=240, seconds=time.monotonic() - start,
         inputs=inputs, protocol_sha256=sha(HERE / 'TRAIN_PROTOCOL.json'), module_sha256=sha(HERE / 'residual_deformable.py'),
         trainer_sha256=sha(Path(__file__).resolve()), smoke_sha256=sha(out / 'SMOKE.json'),
         schedule_sha256=sha(out / 'SCHEDULE.json'), frozen_model_state_sha256=frozen,
         native_detection_config=protocol['native_detection_config'], calibration_labels_read=False,
         calibration_status=config['calibration_status'], official_val_or_test_read=False, runtime=runtime))
    print(json.dumps(dict(event='P29_FIXED_FINAL_TRAIN_COMPLETE', arms=list(checkpoints), steps_per_arm=len(schedule))), flush=True)


if __name__ == '__main__':
    main()
