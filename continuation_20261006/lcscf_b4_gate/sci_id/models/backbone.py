"""Real trainable Caffe-style ResNet C2-C4 path for the SCI-ID prototype."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import torch

from torch import Tensor, nn


class CaffeResNetC2C4(nn.Module):
    """MMDetection ResNet-50 through layer3, matching AutoAssign style.

    This Gate-N1 backbone is independent of the protected detector checkpoint.
    Sharing C2-C4 with AutoAssign remains a later experiment after the identity
    representation gate; here the purpose is to prove gradients reach genuine
    residual blocks rather than only an FPN projection neck.
    """

    def __init__(self, norm_eval: bool = False) -> None:
        super().__init__()
        try:
            from mmdet.models.backbones import ResNet
        except ImportError as exc:  # pragma: no cover - environment contract
            raise RuntimeError("mmdet 2.x is required for CaffeResNetC2C4") from exc
        self.body = ResNet(
            depth=50,
            num_stages=3,
            out_indices=(0, 1, 2),
            strides=(1, 2, 2),
            dilations=(1, 1, 1),
            frozen_stages=-1,
            norm_cfg=dict(type="BN", requires_grad=True),
            norm_eval=norm_eval,
            style="caffe",
            init_cfg=None,
        )

    def load_autoassign_backbone(self, checkpoint: Union[str, Path], strict: bool = True) -> Tuple[int, int]:
        """Load detector-compatible Caffe-R50 weights into C2-C4.

        The MDMT AutoAssign checkpoint stores keys as ``backbone.*`` while
        this module wraps the same MMDetection ResNet under ``body.*``. Only
        backbone keys are accepted; FPN/head/optimizer state is ignored.
        Returns ``(loaded, skipped)`` tensor-key counts and raises on shape
        mismatches when ``strict`` is true.
        """
        payload = torch.load(str(checkpoint), map_location="cpu")
        state = payload.get("state_dict", payload.get("model", payload))
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint does not contain a state dictionary")
        target = self.state_dict()
        mapped = {}
        mismatches = []
        for key, value in state.items():
            if not key.startswith("backbone."):
                continue
            body_key = "body." + key[len("backbone."):]
            if body_key not in target:
                continue
            if tuple(value.shape) != tuple(target[body_key].shape):
                mismatches.append((body_key, tuple(value.shape), tuple(target[body_key].shape)))
                continue
            mapped[body_key] = value
        missing = [key for key in target if key not in mapped]
        if strict and (mismatches or missing):
            raise ValueError(
                f"AutoAssign backbone incompatibility: missing={len(missing)} "
                f"mismatches={len(mismatches)} sample_missing={missing[:3]} "
                f"sample_mismatch={mismatches[:1]}"
            )
        self.load_state_dict(mapped, strict=False)
        return len(mapped), len(target) - len(mapped)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        c2, c3, c4 = self.body(images)
        return {"c2": c2, "c3": c3, "c4": c4}
