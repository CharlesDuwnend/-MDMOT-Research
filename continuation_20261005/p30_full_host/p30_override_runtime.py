"""Compat runtime with an optional detection override.

Identical to mdmt_compat_runtime.init_model_mdmt except that, when the
env var DET_OVERRIDE_DIR is set, ByteTrack.simple_test skips the detector and
reads pre-computed fused detections from
    $DET_OVERRIDE_DIR/<sequence>/<frame_id:06d>.npy     ([N,5] x1y1x2y2score,
original-image coordinates, i.e. the same convention the official demo uses
when rescale=True). Everything else (frame-0 GT seeding via `bil`, tracker,
outputs) is byte-identical to the official path.
"""
import os
from pathlib import Path

import numpy as np
import torch
import mmcv
from mmcv.runner import load_checkpoint
from mmtrack.models import build_model
from mmtrack.models.mot.byte_track import ByteTrack
from mmtrack.core import outs2results, results2outs

from collections import Counter
OVERRIDE_CALLS = Counter()
DETECTOR_HASHES = []
_WARNED = {'once': False}


def _det_file(dirname, img_metas, frame_id):
    if not dirname:
        return None
    fn = None
    for k in ('filename', 'ori_filename'):
        if isinstance(img_metas[0], dict) and k in img_metas[0]:
            fn = img_metas[0][k]
            break
    if not fn:
        if not _WARNED['once']:
            print('[fuse] no filename in img_metas; keys=%s' % list(img_metas[0].keys()))
            _WARNED['once'] = True
        return None
    seq = Path(str(fn)).parent.name
    p = Path(dirname) / seq / f'{frame_id:06d}.npy'
    if not p.is_file():
        raise FileNotFoundError('missing frozen detector override: ' + str(p))
    return p


def _patch_fuse():
    if getattr(ByteTrack, '_fuse_patched', False):
        return
    orig = ByteTrack.simple_test

    def simple_test(self, img, img_metas, frame_id, rescale=False, **kwargs):
        fp = _det_file(os.environ.get('DET_OVERRIDE_DIR'), img_metas, frame_id)
        if fp is None:
            raise RuntimeError('strict P30 runtime requires filename and DET_OVERRIDE_DIR')
        if not rescale:
            raise RuntimeError('strict P30 runtime requires original-image rescale=True')
        if frame_id == 0:
            self.tracker.reset()
        dets = np.load(str(fp), allow_pickle=False).astype(np.float32)
        if dets.ndim != 2 or dets.shape[1] != 5 or not np.isfinite(dets).all():
            raise ValueError('malformed frozen override: ' + str(fp))
        if len(dets) > 300:
            raise ValueError('detector capacity exceeded: ' + str(fp))
        OVERRIDE_CALLS[str(fp)] += 1
        bbox_results = [dets]
        num_classes = len(bbox_results)
        outs_det = results2outs(bbox_results=bbox_results)
        det_bboxes = torch.from_numpy(outs_det['bboxes']).to(img)
        det_labels = torch.from_numpy(outs_det['labels']).to(img).long()
        det_labels = torch.zeros_like(det_labels)
        track_bboxes, track_labels, track_ids, max_id = self.tracker.track(
            img=img,
            img_metas=img_metas,
            model=self,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id,
            rescale=rescale,
            **kwargs)
        track_results = outs2results(
            bboxes=track_bboxes, labels=track_labels, ids=track_ids, num_classes=num_classes)
        det_results = outs2results(bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)
        return dict(
            det_bboxes=det_results['bbox_results'],
            track_bboxes=track_results['bbox_results']), max_id

    ByteTrack.simple_test = simple_test
    ByteTrack._fuse_patched = True


def init_model_mdmt(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    config = config.copy()
    detector = config.model.detector
    detector.pop('init_cfg', None)
    detector.get('backbone', {}).pop('init_cfg', None)
    detector.get('neck', {}).pop('init_cfg', None)
    detector.get('bbox_head', {}).pop('init_cfg', None)
    model = build_model(config.model)
    if checkpoint:
        ckpt = load_checkpoint(model.detector, checkpoint, map_location='cpu', strict=True)
        model.CLASSES = ckpt.get('meta', {}).get('CLASSES', ('pedestrian', 'bicycle', 'car'))
    model.cfg = config
    model.to(device).eval()
    from hashlib import sha256
    digest = sha256()
    for name, value in sorted(model.detector.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    DETECTOR_HASHES.append(digest.hexdigest())
    if config.model.detector.type != 'AutoAssign':
        raise ValueError('P30 host requires AutoAssign')
    if os.environ.get('DET_OVERRIDE_DIR'):
        _patch_fuse()
    return model
