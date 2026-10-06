# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_mot, inference_sot, inference_vid, init_model,inference_reid_mdmt,inference_reid_mdmt_com
# The pinned snapshot omits SOT dataset modules imported by the generic
# train/test API.  Stage 1 is inference-only, so keep the released inference
# symbols importable without inventing the missing training package.

__all__ = [
    'init_model', 'inference_mot', 'inference_sot', 'inference_vid',
    'inference_reid_mdmt', 'inference_reid_mdmt_com'
]
