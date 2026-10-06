"""Frozen RAFT input gate for object residual fields on five MDMT fit pairs."""
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
OLD = Path('/home/chenhc/mdmot_research_20261002')
P21 = HERE.parent / 'p21_residual_motion'
GPU_UUID = 'GPU-1b297aba-ae7e-e326-5903-476f1bb683d7'
sys.path.insert(0, str(P21))
from probe import image_index
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_UUID
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.optical_flow import raft_large
sys.path.insert(0, str(OLD / 'p2/src'))
from data import PairData, split_assignments

PAIRS = ['23', '25', '29', '69', '78']
WEIGHT = HERE / 'assets/raft_large_C_T_SKHT_V2-ff5fadd5.pth'
CONFIG = {'pairs': PAIRS, 'anchor_fractions': [1/3, 2/3], 'time_offsets': [-2, -1, 0],
          'image_resolution': 'native, pad to multiple of 8, never rescale',
          'raft_updates': 12, 'patch_grid': 16, 'background_sample_stride': 16,
          'foreground_mask_expansion': 0.1, 'background_ransac_pixels': 3.,
          'camera_intervention': 'first anchor each pair, view1; old-image rotation 3deg, translation (12,-8) pixels',
          'train_model': False, 'crossview_homography_required': False, 'GT_in_features': False,
          'cal_dev_val_test_read': False, 'gpu_uuid': GPU_UUID}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def read_rgb(pair, camera, frame):
    path = image_index(pair, camera)[frame]
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('image decode failed')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), {'path': str(path), 'sha256': sha(path), 'stream_frame': frame}


def tensor(image):
    value = torch.from_numpy(image.copy()).permute(2, 0, 1).float().unsqueeze(0).cuda() / 127.5 - 1.
    height, width = image.shape[:2]
    return F.pad(value, (0, (-width) % 8, 0, (-height) % 8), mode='replicate')


@torch.inference_mode()
def flow(model, first, second):
    height, width = first.shape[:2]
    assert first.shape == second.shape
    output = model(tensor(first), tensor(second), num_flow_updates=CONFIG['raft_updates'])[-1]
    return output[0, :, :height, :width].permute(1, 2, 0).cpu().numpy()


def sample(field, points):
    shape = points.shape[:-1]
    x = points[..., 0].astype(np.float32)
    y = points[..., 1].astype(np.float32)
    if x.ndim == 1:
        x, y = x[:, None], y[:, None]
    result = cv2.remap(field.astype(np.float32), x, y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return result.reshape(shape + (() if field.ndim == 2 else (field.shape[2],)))


def project(matrix, points):
    points = np.asarray(points)
    homogeneous = np.concatenate([points, np.ones(points.shape[:-1] + (1,))], axis=-1) @ matrix.T
    denominator = homogeneous[..., 2:3]
    safe = np.abs(denominator) > 1e-8
    result = homogeneous[..., :2] / np.where(safe, denominator, 1)
    return result.astype(np.float32), safe[..., 0]


def grid(shape):
    y, x = np.indices(shape, dtype=np.float32)
    return np.stack([x, y], axis=-1)


def background_mask(shape, rows, transform=None):
    mask = np.ones(shape, np.uint8)
    for row in rows:
        x1, y1, x2, y2 = row['bbox']
        if x2 <= x1 or y2 <= y1:
            continue
        points = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
        if transform is not None:
            points, valid = project(transform, points)
            if not valid.all():
                continue
        low, high = points.min(0), points.max(0)
        extent = (high - low) * CONFIG['foreground_mask_expansion']
        low = np.maximum(np.floor(low - extent).astype(int), 0)
        high = np.minimum(np.ceil(high + extent).astype(int), np.array(shape[::-1]) - 1)
        cv2.rectangle(mask, tuple(low), tuple(high), 0, -1)
    return mask


def residual_field(backward, forward, current_rows, previous_rows, previous_transform=None):
    shape = backward.shape[:2]
    xy = grid(shape)
    destination = xy + backward
    inside = ((destination[..., 0] >= 0) & (destination[..., 0] < shape[1]-1)
              & (destination[..., 1] >= 0) & (destination[..., 1] < shape[0]-1))
    fb = np.linalg.norm(backward + sample(forward, destination), axis=-1)
    visible = inside & (fb <= 1. + .05 * np.linalg.norm(backward, axis=-1))
    current_bg = background_mask(shape, current_rows)
    previous_bg = background_mask(shape, previous_rows, previous_transform)
    bg = (current_bg > 0) & (sample(previous_bg, destination) > .99) & visible
    stride = CONFIG['background_sample_stride']
    points = xy[::stride, ::stride][bg[::stride, ::stride]]
    targets = destination[::stride, ::stride][bg[::stride, ::stride]]
    summary = {'background_sample_count': len(points), 'forward_backward_visible_fraction': float(visible.mean())}
    if len(points) < 30:
        return None, visible, summary | {'status': 'INSUFFICIENT_BACKGROUND'}, None
    # Checkerboard cells are held out from the homography fit.
    holdout = ((points[:, 0].astype(int) // 128 + points[:, 1].astype(int) // 128) % 4) == 0
    matrix, inliers = cv2.findHomography(points[~holdout], targets[~holdout], cv2.RANSAC,
                                        CONFIG['background_ransac_pixels'], maxIters=3000, confidence=.995)
    if matrix is None or inliers is None:
        return None, visible, summary | {'status': 'BACKGROUND_FIT_FAILED'}, None
    predicted, support = project(matrix, xy)
    residual = destination - predicted
    visible &= support
    held_prediction, held_valid = project(matrix, points[holdout])
    held_error = np.linalg.norm(held_prediction - targets[holdout], axis=-1)[held_valid]
    summary.update(status='BACKGROUND_MODEL_ESTIMATED', inliers=int(inliers.sum()),
                   holdout_samples=int(holdout.sum()),
                   holdout_error_median_pixels=float(np.median(held_error)) if len(held_error) else None,
                   holdout_error_p90_pixels=float(np.percentile(held_error, 90)) if len(held_error) else None,
                   fit_is_physical_camera_calibration=False)
    return residual, visible, summary, matrix


def roi_points(box):
    x1, y1, x2, y2 = box
    x = x1 + (np.arange(16, dtype=np.float32) + .5) / 16 * max(x2-x1, 0)
    y = y1 + (np.arange(16, dtype=np.float32) + .5) / 16 * max(y2-y1, 0)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy], axis=-1)


def extract_task(model, data, pair, camera, anchor, first_anchor):
    frames = [anchor-2, anchor-1, anchor]
    packets = {t: data.frame(t) for t in frames}
    rows = {t: [r for r in packets[t]['rows'] if r['camera'] == camera] for t in frames}
    images, refs = {}, []
    for t in frames:
        images[t], ref = read_rgb(pair, camera, t)
        refs.append(ref)
    residuals, visibility, backwards, diagnostics, matrices = {}, {}, {}, {}, {}
    for t in frames[1:]:
        back = flow(model, images[t], images[t-1])
        forward = flow(model, images[t-1], images[t])
        result = residual_field(back, forward, rows[t], rows[t-1])
        residuals[t], visibility[t], diagnostics[t], matrices[t] = result
        backwards[t] = back
    cam_field, camera_diagnostic = None, None
    if first_anchor and camera == '1' and residuals[anchor] is not None:
        height, width = images[anchor].shape[:2]
        affine = cv2.getRotationMatrix2D((width/2, height/2), 3., 1.)
        affine[:, 2] += [12., -8.]
        transform = np.vstack([affine, [0., 0., 1.]])
        altered = cv2.warpAffine(images[anchor-1], affine, (width, height), borderMode=cv2.BORDER_REFLECT)
        altered_back = flow(model, images[anchor], altered)
        altered_forward = flow(model, altered, images[anchor])
        altered_residual, altered_visible, altered_summary, _ = residual_field(
            altered_back, altered_forward, rows[anchor], rows[anchor-1], transform)
        if altered_residual is not None:
            cam_field = altered_residual @ np.linalg.inv(affine[:, :2]).T
            camera_diagnostic = {'summary': altered_summary, 'affine_old_to_altered': affine.tolist(),
                                 'visible': altered_visible}
    fields, masks, keys, metadata = [], [], [], []
    for row in rows[anchor]:
        points = roi_points(row['bbox'])
        prior_points = points + sample(backwards[anchor], points)
        values, valid = [], []
        for t, p in [(anchor-1, prior_points), (anchor, points)]:
            values.append(sample(residuals[t], p) if residuals[t] is not None else np.zeros((16,16,2), np.float32))
            valid.append(sample(visibility[t].astype(np.float32), p) > .99 if residuals[t] is not None else np.zeros((16,16), bool))
        values = np.stack(values)
        valid = np.stack(valid)
        current = values[1][valid[1]]
        info = {'key': row['key'], 'bbox': row['bbox'], 'class': row['class'],
                'native_min_side': float(min(row['bbox'][2]-row['bbox'][0], row['bbox'][3]-row['bbox'][1])),
                'visible_fraction': float(valid.mean()), 'current_visible_fraction': float(valid[1].mean()),
                'current_median_residual_pixels': float(np.median(np.linalg.norm(current, axis=-1))) if len(current) else None,
                'current_spatial_std_pixels': float(np.sqrt(np.var(current, axis=0).sum())) if len(current) else None,
                'background_holdout_p90_pixels': diagnostics[anchor].get('holdout_error_p90_pixels')}
        if cam_field is not None:
            altered_values = sample(cam_field, points)
            common = valid[1] & (sample(camera_diagnostic['visible'].astype(np.float32), points) > .99)
            info['camera_intervention_common_fraction'] = float(common.mean())
            info['camera_intervention_residual_delta_pixels'] = float(np.median(np.linalg.norm(altered_values[common] - values[1][common], axis=-1))) if common.any() else None
        fields.append(values.astype(np.float32)); masks.append(valid); keys.append(row['key']); metadata.append(info)
    name = '%s-%s_t%04d' % (pair, camera, anchor)
    np.savez_compressed(HERE / (name + '.npz'), residual_fields=np.stack(fields), valid=np.stack(masks), keys=np.array(keys))
    report = {'pair': pair, 'camera': camera, 'anchor': anchor, 'images': refs,
              'geometry': diagnostics, 'background_matrices': {str(t): m.tolist() if m is not None else None for t,m in matrices.items()},
              'camera_intervention': {k:v for k,v in camera_diagnostic.items() if k != 'visible'} if camera_diagnostic else None,
              'objects': metadata, 'PID_read': False, 'raw_XML_read': False}
    dump(HERE / (name + '.json'), report)
    print(json.dumps({'pair': pair, 'view': camera, 'anchor': anchor, 'objects': len(keys),
                      'visible_objects_half_grid': sum(x['visible_fraction'] >= .5 for x in metadata)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        assert set(PAIRS) <= set(split_assignments()['fit'])
        if (HERE / 'PROTOCOL.json').exists():
            raise ValueError('protocol exists')
        source = Path(importlib.util.find_spec('torchvision.models.optical_flow.raft').origin)
        protocol = dict(CONFIG, created_utc=datetime.now(timezone.utc).isoformat(),
                        fixed_anchors={p: [int(round(PairData(p).frames*f)) for f in CONFIG['anchor_fractions']] for p in PAIRS},
                        code_sha256=sha(Path(__file__)), weights_sha256=sha(WEIGHT), torchvision= torchvision.__version__,
                        raft_source={'path': str(source), 'sha256': sha(source)},
                        hypothesis='Dense per-object residual fields have usable spatial and temporal evidence beyond a single box-center vector.',
                        gate='At least 20 objects with >=.5 valid two-step grid support and residual spatial standard deviation above held-out background p90, in >=3 fixed fit pairs; background errors and camera intervention separately reported.',
                        failure_scope='Frozen RAFT plus background homography at native resolution; not a learned representation or universal motion impossibility.')
        dump(HERE / 'PROTOCOL.json', protocol)
        print('PROTOCOL_WRITTEN')
        return
    protocol = json.loads((HERE / 'PROTOCOL.json').read_text())
    assert sha(Path(__file__)) == protocol['code_sha256'] and sha(WEIGHT) == protocol['weights_sha256']
    cv2.setNumThreads(1); cv2.setRNGSeed(20261004); torch.set_num_threads(2)
    torch.manual_seed(20261004)
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_properties(0).total_memory > 39*1024**3
    gpu_snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used', '--format=csv,noheader'], text=True)
    dump(HERE / 'GPU_START.json', {'bound_uuid': GPU_UUID, 'snapshot': gpu_snapshot, 'torch_name': torch.cuda.get_device_name(0)})
    model = raft_large(weights=None, progress=False).eval()
    model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=True), strict=True)
    model.requires_grad_(False).cuda()
    started, reports = time.monotonic(), []
    for pair in PAIRS:
        data = PairData(pair, with_labels=False)
        for index, anchor in enumerate(protocol['fixed_anchors'][pair]):
            for camera in ('1', '2'):
                reports.append(extract_task(model, data, pair, camera, anchor, index == 0))
    per_pair = {}
    for pair in PAIRS:
        objects = [o for r in reports if r['pair'] == pair for o in r['objects']]
        supported = [o for o in objects if o['visible_fraction'] >= .5]
        varied = [o for o in supported if o['background_holdout_p90_pixels'] is not None
                  and o['current_spatial_std_pixels'] is not None and o['current_spatial_std_pixels'] > o['background_holdout_p90_pixels']]
        per_pair[pair] = {'objects': len(objects), 'visible_half_grid': len(supported), 'spatial_variation_above_background_error': len(varied)}
    receipt = {'status': 'DENSE_RESIDUAL_FIELDS_FROZEN_BEFORE_LABELS', 'per_pair': per_pair,
               'gate_pass': sum(r['spatial_variation_above_background_error'] >= 20 for r in per_pair.values()) >= 3,
               'elapsed_seconds': time.monotonic()-started, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
               'PID_read': False, 'model_trained': False, 'cal_dev_val_test_read': False,
               'protocol_sha256': sha(HERE / 'PROTOCOL.json'), 'gpu_uuid': GPU_UUID,
               'files': [{'path': str(p), 'sha256': sha(p)} for p in sorted(HERE.iterdir()) if p.is_file() and p.suffix in ['.json','.npz','.py']]}
    dump(HERE / 'FEATURE_RECEIPT.json', receipt)
    (HERE / 'runner.exit').write_text('0\n')
    print(json.dumps(receipt | {'files': len(receipt['files'])}), flush=True)


if __name__ == '__main__':
    main()
