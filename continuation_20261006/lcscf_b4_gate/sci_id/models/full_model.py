"""End-to-end Gate-N1 wiring from real Caffe ResNet blocks to SCI-ID logits."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .assignment_head import AssignmentOutput, MultiScaleSCIIDHead
from .backbone import CaffeResNetC2C4
from .identity_pyramid import IdentityPyramid, extract_multiscale_roi_maps
from .cvsgm import CrossViewScaleGatedModulation
from .lcscf import LCSCFTransport


class SCIIDBackboneModel(nn.Module):
    """Shared-weight two-view image encoder and multi-scale association head."""

    def __init__(
        self,
        channels: int = 256,
        prototypes: int = 4,
        output_size: int = 14,
        adapter_enabled: bool = True,
        head: Optional[nn.Module] = None,
        backbone_init: Optional[Union[str, Path]] = None,
        cross_view_modulation: bool = False,
        lcscf_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = CaffeResNetC2C4(norm_eval=False)
        self.backbone_init_counts = None
        if backbone_init is not None:
            loaded, skipped = self.backbone.load_autoassign_backbone(backbone_init, strict=True)
            if skipped:
                raise RuntimeError(f"unexpected skipped backbone tensors: {skipped}")
            self.backbone_init_counts = {"loaded": loaded, "skipped": skipped}
        self.pyramid = IdentityPyramid(out_channels=channels)
        self.cross_view_modulation = (
            CrossViewScaleGatedModulation(channels) if cross_view_modulation else None
        )
        self.lcscf = LCSCFTransport(channels) if lcscf_enabled else None
        self.last_lcscf_stats = None
        self.head = head or MultiScaleSCIIDHead(
            channels=channels, prototypes=prototypes, adapter_enabled=adapter_enabled
        )
        self.output_size = output_size

    def forward(
        self,
        target_images: Tensor,
        candidate_images: Tensor,
        target_boxes: Tensor,
        candidate_boxes: Tensor,
        mask: Tensor,
    ) -> AssignmentOutput:
        if target_images.shape != candidate_images.shape:
            raise ValueError("paired target/candidate image batches must share shape")
        b = target_images.shape[0]
        frame_images = torch.cat((target_images, candidate_images), dim=0)
        target_frame_index = torch.arange(b, device=target_images.device)
        source_frame_index = torch.arange(b, 2 * b, device=target_images.device)
        return self.forward_packed(
            frame_images,
            target_frame_index,
            source_frame_index,
            target_boxes,
            candidate_boxes,
            mask,
        )

    def forward_packed(
        self,
        frame_images: Tensor,
        target_frame_index: Tensor,
        source_frame_index: Tensor,
        target_boxes: Tensor,
        candidate_boxes: Tensor,
        mask: Tensor,
    ) -> AssignmentOutput:
        """Run the image backbone once for the unique frames in a batch.

        ``target_frame_index`` and ``source_frame_index`` are episode-to-frame
        lookups produced by ``collate_packed_frame_episodes``. Repeated frames
        therefore share one feature-map computation while each episode keeps
        its own target/candidate RoIs and K mask.
        """
        if frame_images.ndim != 4 or frame_images.shape[1] != 3:
            raise ValueError("frame_images must be [F,3,H,W]")
        if target_frame_index.ndim != 1 or source_frame_index.shape != target_frame_index.shape:
            raise ValueError("frame indices must be matching [B] vectors")
        b = target_frame_index.shape[0]
        if target_boxes.shape != (b, 4) or candidate_boxes.shape[:2] != mask.shape:
            raise ValueError("packed boxes and mask have incompatible batch shapes")
        if target_frame_index.numel() and (
            int(target_frame_index.min()) < 0
            or int(source_frame_index.min()) < 0
            or int(target_frame_index.max()) >= frame_images.shape[0]
            or int(source_frame_index.max()) >= frame_images.shape[0]
        ):
            raise ValueError("packed frame index is out of range")
        target_maps, candidate_maps = self.encode_packed(
            frame_images,
            target_frame_index,
            source_frame_index,
            target_boxes,
            candidate_boxes,
            mask,
        )
        return self.forward_maps(target_maps, candidate_maps, mask)

    def encode_packed(
        self,
        frame_images: Tensor,
        target_frame_index: Tensor,
        source_frame_index: Tensor,
        target_boxes: Tensor,
        candidate_boxes: Tensor,
        mask: Tensor,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Encode unique frames once and return per-episode RoI maps.

        Multiple factual/intervention masks can reuse these maps in one step;
        this is important because the SCI-ID loss evaluates several controlled
        candidate compositions for the same paired frame.
        """
        if frame_images.ndim != 4 or frame_images.shape[1] != 3:
            raise ValueError("frame_images must be [F,3,H,W]")
        if target_frame_index.ndim != 1 or source_frame_index.shape != target_frame_index.shape:
            raise ValueError("frame indices must be matching [B] vectors")
        b = target_frame_index.shape[0]
        if target_boxes.shape != (b, 4) or candidate_boxes.shape[:2] != mask.shape:
            raise ValueError("packed boxes and mask have incompatible batch shapes")
        if target_frame_index.numel() and (
            int(target_frame_index.min()) < 0
            or int(source_frame_index.min()) < 0
            or int(target_frame_index.max()) >= frame_images.shape[0]
            or int(source_frame_index.max()) >= frame_images.shape[0]
        ):
            raise ValueError("packed frame index is out of range")
        features = self.backbone(frame_images)
        pyramid = self.pyramid(features["c2"], features["c3"], features["c4"])
        target_pyramid = {level: value[target_frame_index] for level, value in pyramid.items()}
        candidate_pyramid = {level: value[source_frame_index] for level, value in pyramid.items()}
        if self.cross_view_modulation is not None:
            target_pyramid, candidate_pyramid = self.cross_view_modulation(
                target_pyramid, candidate_pyramid
            )
        target_maps, candidate_maps = extract_multiscale_roi_maps(
            target_pyramid,
            candidate_pyramid,
            target_boxes,
            candidate_boxes,
            mask,
            output_size=self.output_size,
        )
        if self.lcscf is not None:
            target_maps, candidate_maps, self.last_lcscf_stats = self.lcscf(
                target_maps, candidate_maps, mask
            )
        return target_maps, candidate_maps

    def forward_maps(
        self,
        target_maps: Mapping[str, Tensor],
        candidate_maps: Mapping[str, Tensor],
        mask: Tensor,
    ) -> AssignmentOutput:
        """Apply a head to already encoded per-episode RoI maps."""
        return self.head(target_maps, candidate_maps, mask)


def build_arm_model(
    arm: str,
    channels: int = 256,
    prototypes: int = 4,
    output_size: int = 14,
    backbone_init: Optional[Union[str, Path]] = None,
) -> SCIIDBackboneModel:
    """Build B1--B4/P1 with an identical image backbone and ROI contract."""
    from .baselines import (
        B1UniversalMultiScaleHead,
        B2PairwiseSpatialHead,
        B3PooledSetRelativeHead,
        B4PooledExactCompositionHead,
    )

    builders = {
        "B1": lambda: B1UniversalMultiScaleHead(channels, prototypes),
        "B2": lambda: B2PairwiseSpatialHead(channels),
        "B3": lambda: B3PooledSetRelativeHead(channels),
        "B4": lambda: B4PooledExactCompositionHead(channels, prototypes),
        "P1": lambda: MultiScaleSCIIDHead(channels, prototypes),
        "CVSGM": lambda: B1UniversalMultiScaleHead(channels, prototypes),
        "CVSGM_B4": lambda: B4PooledExactCompositionHead(channels, prototypes),
        "LCSCF_B4": lambda: B4PooledExactCompositionHead(channels, prototypes),
    }
    if arm not in builders:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(builders)}")
    return SCIIDBackboneModel(
        channels=channels,
        prototypes=prototypes,
        output_size=output_size,
        head=builders[arm](),
        backbone_init=backbone_init,
        cross_view_modulation=arm in ("CVSGM", "CVSGM_B4"),
        lcscf_enabled=(arm == "LCSCF_B4"),
    )
