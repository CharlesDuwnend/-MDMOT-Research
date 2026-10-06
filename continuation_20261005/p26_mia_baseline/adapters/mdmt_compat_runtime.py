"""Runtime adapter for the pinned MDMT snapshot's detector checkpoint.

The released checkpoint is detector-only (state keys start at ``backbone``),
while the demo wraps it in ByteTrack and leaves ``init_cfg`` in a config field
unsupported by the installed mmdet release. This adapter removes only those
construction-time fields and loads the checkpoint into ``model.detector``.
"""
import mmcv
from mmcv.runner import load_checkpoint
from mmtrack.models import build_model


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
        ckpt = load_checkpoint(model.detector, checkpoint, map_location='cpu')
        model.CLASSES = ckpt.get('meta', {}).get('CLASSES', ('pedestrian', 'bicycle', 'car'))
    model.cfg = config
    model.to(device).eval()
    return model
