"""Trainable P2/P3/P4 identity pyramid used only by the SCI-ID prototype.

This module deliberately has no detector, tracker, global-ID, future-frame, or
candidate-index input. RoIs are expressed in normalized image coordinates and
are sampled by torchvision's aligned RoIAlign implementation.
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class _Project(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(32, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class IdentityPyramid(nn.Module):
    """C2/C3/C4 -> 256-channel identity P2/P3/P4 at strides 4/8/16.

    The module is separate from the frozen Stage-1 AutoAssign detector. Gate N1
    wires it to a trainable Caffe-style ResNet-50 C2--C4 encoder; sharing it
    with the protected detector remains a later controlled integration choice.
    """

    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        self.p2_id = _Project(256, out_channels)
        self.p3_id = _Project(512, out_channels)
        self.p4_id = _Project(1024, out_channels)
        self.refine_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.refine_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.refine_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, c2: Tensor, c3: Tensor, c4: Tensor) -> Dict[str, Tensor]:
        p4 = self.refine_p4(self.p4_id(c4))
        p3 = self.refine_p3(self.p3_id(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.refine_p2(self.p2_id(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False))
        return {"p2_id": p2, "p3_id": p3, "p4_id": p4}


def roi_align_normalized(features: Tensor, rois: Tensor, output_size: int = 14) -> Tensor:
    """Differentiable aligned RoIAlign from normalized boxes.

    Args:
        features: ``[B,C,H,W]`` feature map.
        rois: ``[R,5]`` ``(batch_index,x1,y1,x2,y2)`` with coordinates in
            ``[0,1]`` relative to the feature/image extent.
    Returns:
        ``[R,C,output_size,output_size]`` spatial identity maps before pooling.
    """
    if features.ndim != 4 or rois.ndim != 2 or rois.shape[-1] != 5:
        raise ValueError("features must be [B,C,H,W] and rois must be [R,5]")
    if rois.numel() == 0:
        return features.new_empty((0, features.shape[1], output_size, output_size))
    batch = rois[:, 0].long()
    if batch.min() < 0 or batch.max() >= features.shape[0]:
        raise ValueError("RoI batch index is out of range")
    boxes = rois[:, 1:].clamp(0, 1).clone()
    if torch.any(boxes[:, 2:] < boxes[:, :2]):
        raise ValueError("RoI has x2/y2 smaller than x1/y1")
    # Normalized xyxy boxes describe image extents, not the centers of the
    # first/last feature pixels. With pad-to-32, image_extent / stride equals
    # feature_extent. RoIAlign applies its own aligned half-pixel shift.
    boxes[:, [0, 2]] *= features.shape[-1]
    boxes[:, [1, 3]] *= features.shape[-2]
    absolute_rois = torch.cat((batch.to(boxes.dtype)[:, None], boxes), dim=1)
    return roi_align(
        features,
        absolute_rois,
        output_size=(output_size, output_size),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )


def extract_multiscale_roi_maps(
    target_pyramid: Mapping[str, Tensor],
    candidate_pyramid: Mapping[str, Tensor],
    target_boxes: Tensor,
    candidate_boxes: Tensor,
    mask: Tensor,
    output_size: int = 14,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Extract P2/P3/P4 target and candidate maps with a shared box contract.

    ``target_boxes`` is ``[B,4]`` and ``candidate_boxes`` is ``[B,K,4]`` in
    normalized xyxy coordinates. Padded candidate maps are explicitly zeroed.
    """
    levels = ("p2_id", "p3_id", "p4_id")
    if tuple(target_pyramid) != levels or tuple(candidate_pyramid) != levels:
        if set(target_pyramid) != set(levels) or set(candidate_pyramid) != set(levels):
            raise ValueError("both pyramids must contain p2_id, p3_id and p4_id")
    if target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
        raise ValueError("target_boxes must be [B,4]")
    if candidate_boxes.ndim != 3 or candidate_boxes.shape[-1] != 4:
        raise ValueError("candidate_boxes must be [B,K,4]")
    if mask.shape != candidate_boxes.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be bool [B,K]")
    b, k = mask.shape
    if target_boxes.shape[0] != b or not 1 <= k <= 64:
        raise ValueError("incompatible batch or candidate count")

    target_rois = torch.cat(
        (
            torch.arange(b, device=target_boxes.device, dtype=target_boxes.dtype)[:, None],
            target_boxes,
        ),
        dim=1,
    )
    flat_candidate_boxes = candidate_boxes.reshape(b * k, 4)
    candidate_batch = torch.arange(
        b, device=candidate_boxes.device, dtype=candidate_boxes.dtype
    )[:, None].expand(b, k).reshape(-1, 1)
    candidate_rois = torch.cat((candidate_batch, flat_candidate_boxes), dim=1)
    target_maps: Dict[str, Tensor] = {}
    candidate_maps: Dict[str, Tensor] = {}
    valid = mask[:, :, None, None, None]
    for level in levels:
        target_maps[level] = roi_align_normalized(
            target_pyramid[level], target_rois, output_size
        )
        c = target_maps[level].shape[1]
        maps = roi_align_normalized(
            candidate_pyramid[level], candidate_rois, output_size
        ).reshape(b, k, c, output_size, output_size)
        candidate_maps[level] = maps * valid.to(maps.dtype)
    return target_maps, candidate_maps
