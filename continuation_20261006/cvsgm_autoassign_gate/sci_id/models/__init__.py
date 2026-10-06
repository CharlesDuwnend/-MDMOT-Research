from .backbone import CaffeResNetC2C4
from .identity_pyramid import IdentityPyramid, extract_multiscale_roi_maps, roi_align_normalized
from .set_composition_adapter import SetCompositionAdapter
from .assignment_head import MultiScaleSCIIDHead, SCIIDAssignmentHead
from .full_model import SCIIDBackboneModel, build_arm_model
from .baselines import (
    B1UniversalMultiScaleHead,
    B2PairwiseSpatialHead,
    B3PooledSetRelativeHead,
    B4PooledExactCompositionHead,
)

__all__ = [
    "CaffeResNetC2C4",
    "IdentityPyramid",
    "extract_multiscale_roi_maps",
    "roi_align_normalized",
    "SetCompositionAdapter",
    "SCIIDAssignmentHead",
    "MultiScaleSCIIDHead",
    "SCIIDBackboneModel",
    "build_arm_model",
    "B1UniversalMultiScaleHead",
    "B2PairwiseSpatialHead",
    "B3PooledSetRelativeHead",
    "B4PooledExactCompositionHead",
]
