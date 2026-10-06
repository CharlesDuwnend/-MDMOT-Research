#!/usr/bin/env python3
"""Flow-centred residual deformable attention used by the P29 pilot.

This file deliberately wraps the installed MMCV operator instead of defining a
new attention implementation.  The current FPN feature is passed as the
operator identity; the learned branch can therefore start as an exact native
detector.  Past feature maps are the values and RAFT supplies only the
reference points and the conditioning warp.
"""

import math
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor, nn


STRIDES = (8, 16, 32, 64, 128)
MODES = ("current_only", "past_fixed_offsets", "past_learned_offsets")


def source_bindings():
    """Hook for the trainer's source receipt; runtime dependencies are bound there."""
    return {}


def _geometry(original_hw, detector_img_hw, pad_hw):
    vals = []
    for name, value in (("original_hw", original_hw), ("detector_img_hw", detector_img_hw),
                        ("pad_hw", pad_hw)):
        if value is None or len(value) != 2:
            raise ValueError(name + " must be a two-tuple")
        pair = tuple(int(x) for x in value)
        if any(x < 1 for x in pair):
            raise ValueError(name + " must be positive")
        vals.append(pair)
    if any(a > b for a, b in zip(vals[1], vals[2])):
        raise ValueError("detector image must fit in padded image")
    return tuple(vals)


def _valid_mask(height, width, stride, detector_hw, device):
    dh, dw = detector_hw
    yy = torch.arange(height, device=device)
    xx = torch.arange(width, device=device)
    return ((yy[:, None] * stride < dh) & (xx[None, :] * stride < dw))


def _validate(current, strides):
    if len(current) != len(strides) or not current:
        raise ValueError("current feature levels do not match strides")
    batch, channels = current[0].shape[:2]
    if batch != 1:
        raise ValueError("P29 uses a single detector image per group")
    for x, stride in zip(current, strides):
        if x.ndim != 4 or not x.is_floating_point() or x.shape[:2] != (batch, channels):
            raise ValueError("feature tensors must be [1,C,H,W] with shared C")
        if tuple(x.shape[-2:]) != tuple(int(math.ceil(v / float(stride)))
                                          for v in _CURRENT_PAD_HW):
            raise ValueError("feature lattice is not ceil(pad/stride)")
    return channels


# Set for the duration of one forward call.  Keeping the shape check local to
# the adapter avoids silently accepting a detector export with a different pad.
_CURRENT_PAD_HW = (1, 1)


def _flow_sample(flows, height, width, stride, original_hw, detector_hw, device):
    """Sample current-to-past flow at feature-node centres.

    Returns feature-index displacements [R,H,W,2] and a current-lattice mask.
    The resize chain matches P28's frozen warp helper.
    """
    oh, ow = original_hw
    dh, dw = detector_hw
    fh, fw = flows.shape[-2:]
    dtype = torch.float64 if flows.dtype == torch.float64 else torch.float32
    # Feature nodes live at detector-image coordinates j * stride (the
    # AutoAssign offset-zero contract).  The original P29 implementation
    # passed j directly here, which sampled the flow raster at feature-index
    # coordinates and therefore divided motion by `stride`.
    y = torch.arange(height, device=device, dtype=dtype) * float(stride)
    x = torch.arange(width, device=device, dtype=dtype) * float(stride)
    flow_x = ((x + .5) * (ow / float(dw))) * (fw / float(ow)) - .5
    flow_y = ((y + .5) * (oh / float(dh))) * (fh / float(oh)) - .5
    grid = torch.stack(((2 * (flow_x + .5) / fw - 1)[None, :].expand(height, width),
                        (2 * (flow_y + .5) / fh - 1)[:, None].expand(height, width)), -1)
    grid = grid[None].expand(flows.shape[0], height, width, 2)
    import torch.nn.functional as F
    sampled = F.grid_sample(flows.to(device=device, dtype=dtype), grid,
                            mode="bilinear", padding_mode="border", align_corners=False)
    sampled = torch.where(torch.isfinite(sampled), sampled, torch.zeros_like(sampled))
    dx = sampled[:, 0] * (dw / float(fw)) / float(stride)
    dy = sampled[:, 1] * (dh / float(fh)) / float(stride)
    return torch.stack((dx, dy), -1), _valid_mask(height, width, stride, detector_hw, device)


class ResidualDeformableAdapter(nn.Module):
    """Shared flow-centred MS deformable residual over five FPN levels."""

    def __init__(self, mode="past_learned_offsets", channels=256,
                 strides=STRIDES, heads=4, points=9):
        super().__init__()
        if mode not in MODES:
            raise ValueError("unknown mode: " + str(mode))
        if channels % heads:
            raise ValueError("channels must divide attention heads")
        self.mode = mode
        self.channels = int(channels)
        self.strides = tuple(int(x) for x in strides)
        self.num_heads, self.num_levels, self.num_points = heads, 3, points
        self.query_conditioner = nn.Sequential(
            nn.Conv2d(4 * channels, 128, 1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, channels, 3, padding=1, bias=True),
        )
        try:
            from mmcv.ops import MultiScaleDeformableAttention
        except Exception as exc:
            raise RuntimeError("installed MMCV MSDA is required for P29") from exc
        self.msda = MultiScaleDeformableAttention(
            embed_dims=channels, num_heads=heads, num_levels=3,
            num_points=points, dropout=0.0, batch_first=True)
        self._initialize()

    def _initialize(self):
        # Query projections start silent.  A centred 3x3 stencil is used for
        # all three temporal maps and all heads; this is an explicit port
        # choice, rather than the MMCV radial default.
        nn.init.constant_(self.msda.sampling_offsets.weight, 0.)
        nn.init.constant_(self.msda.sampling_offsets.bias, 0.)
        nn.init.constant_(self.msda.attention_weights.weight, 0.)
        nn.init.constant_(self.msda.attention_weights.bias, 0.)
        nn.init.constant_(self.msda.output_proj.weight, 0.)
        if self.msda.output_proj.bias is not None:
            nn.init.constant_(self.msda.output_proj.bias, 0.)
        nn.init.xavier_uniform_(self.msda.value_proj.weight)
        nn.init.constant_(self.msda.value_proj.bias, 0.)
        stencil = [(-1., -1.), (0., -1.), (1., -1.),
                   (-1., 0.), (0., 0.), (1., 0.),
                   (-1., 1.), (0., 1.), (1., 1.)]
        b = []
        for _head in range(self.num_heads):
            for _level in range(self.num_levels):
                for dx, dy in stencil:
                    b.extend((dx, dy))
        with torch.no_grad():
            self.msda.sampling_offsets.bias.copy_(torch.tensor(
                b, dtype=self.msda.sampling_offsets.bias.dtype,
                device=self.msda.sampling_offsets.bias.device))
        if self.mode == "past_fixed_offsets":
            self.msda.sampling_offsets.requires_grad_(False)

    def parameter_groups(self):
        return {
            "query_conditioner": tuple(self.query_conditioner.parameters()),
            "value_projection": tuple(self.msda.value_proj.parameters()),
            "output_projection": tuple(self.msda.output_proj.parameters()),
            "sampling_offsets": tuple(self.msda.sampling_offsets.parameters()),
            "attention_weights": tuple(self.msda.attention_weights.parameters()),
        }

    def parameter_counts(self):
        groups = self.parameter_groups()
        result = {name: sum(p.numel() for p in values) for name, values in groups.items()}
        result.update({
            "active": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in self.parameters())})
        return result

    def _one_level(self, current, raw_past, warped, valid, displacement,
                   detector_hw, pad_hw, stride, return_aux=False):
        _, _, h, w = current.shape
        # Condition on a current feature and flow-warped support, but pass raw
        # support values to MSDA so the flow reference is applied exactly once.
        # current is [1,C,H,W], while the three support maps are [3,C,H,W].
        # They form four channel groups for one query image.
        cond_input = torch.cat((current, warped), dim=0).reshape(
            1, 4 * self.channels, h, w)
        cond = self.query_conditioner(cond_input)
        query = cond.flatten(2).transpose(1, 2)
        values = raw_past.flatten(2).transpose(1, 2).transpose(0, 1)
        # raw_past is [R,C,H,W]; transpose above gives [R,N,C], then reshape.
        values = raw_past.permute(0, 2, 3, 1).reshape(1, -1, self.channels)
        spatial = torch.tensor([[h, w]] * self.num_levels,
                               dtype=torch.long, device=current.device)
        starts = torch.tensor([0, h * w, 2 * h * w],
                              dtype=torch.long, device=current.device)
        # [1,N,3,2], with the same per-query flow reference for each map.
        yy = torch.arange(h, device=current.device, dtype=current.dtype)
        xx = torch.arange(w, device=current.device, dtype=current.dtype)
        qx = xx[None, :].expand(h, w) + displacement[..., 0].mean(0)
        qy = yy[:, None].expand(h, w) + displacement[..., 1].mean(0)
        # A separate reference for every lag is needed.  Preserve map order.
        refs = []
        for r in range(self.num_levels):
            rx = xx[None, :].expand(h, w) + displacement[r, ..., 0]
            ry = yy[:, None].expand(h, w) + displacement[r, ..., 1]
            refs.append(torch.stack(((rx + .5) / w, (ry + .5) / h), -1).reshape(-1, 2))
        reference = torch.stack(refs, 1)[None]
        # The value mask lives on the raw past lattice.  `valid` instead
        # describes current-query warp endpoints and must not mask raw source
        # nodes: an invalid endpoint at query j does not make raw value j
        # invalid.  Only image padding is zeroed before native MSDA sampling.
        raw_valid = _valid_mask(h, w, stride, detector_hw, current.device)
        key_mask = (~raw_valid).expand(self.num_levels, h, w).reshape(-1)[None]
        out = self.msda(query=query, value=values, identity=current.flatten(2).transpose(1, 2),
                        reference_points=reference, spatial_shapes=spatial,
                        level_start_index=starts, key_padding_mask=key_mask)
        result = out.transpose(1, 2).reshape_as(current)
        if return_aux:
            # Reconstruct the attention weights for diagnostics without a
            # second expensive operator call.  These are query projection
            # weights before the CUDA kernel and match its softmax contract.
            logits = self.msda.attention_weights(query).view(
                1, h * w, self.num_heads, self.num_levels * self.num_points)
            weights = logits.softmax(-1).view(1, h * w, self.num_heads,
                                              self.num_levels, self.num_points)
            weights = weights.mean(2).sum(-1).transpose(1, 2).reshape(1, 3, h, w)
            return result, weights, valid
        return result

    def forward(self, current_features: Sequence[Tensor],
                past_features: Optional[Sequence[Tensor]] = None,
                flows: Optional[Tensor] = None, *, original_hw=None,
                detector_img_hw=None, pad_hw=None, return_aux=False):
        global _CURRENT_PAD_HW
        original_hw, detector_hw, pad_hw = _geometry(original_hw or (1, 1),
                                                     detector_img_hw or (1, 1),
                                                     pad_hw or (1, 1))
        _CURRENT_PAD_HW = pad_hw
        channels = _validate(current_features, self.strides)
        if channels != self.channels:
            raise ValueError("unexpected channel count")
        if self.mode == "current_only" or past_features is None:
            raw = [torch.cat((cur, cur, cur), 0) for cur in current_features]
            warped = [torch.cat((cur, cur, cur), 0) for cur in current_features]
            valid = [torch.ones((3, 1) + cur.shape[-2:], dtype=torch.bool,
                                device=cur.device) for cur in current_features]
            displacement = [torch.zeros((3,) + cur.shape[-2:] + (2,),
                                        dtype=cur.dtype, device=cur.device)
                            for cur in current_features]
        else:
            if len(past_features) != len(current_features) or flows is None:
                raise ValueError("past mode requires matching features and flows")
            try:
                from temporal_module import warp_past_features
            except ImportError:
                import importlib.util
                helper_path = Path(__file__).resolve().parents[1] / "p28_causal_fgfa" / "temporal_module.py"
                spec = importlib.util.spec_from_file_location("p29_temporal_helper", helper_path)
                helper = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(helper)
                warp_past_features = helper.warp_past_features
            raw = list(past_features)
            warped, valid = warp_past_features(
                past_features, flows, original_hw=original_hw,
                detector_img_hw=detector_hw, pad_hw=pad_hw, strides=self.strides)
            displacement = []
            for cur, stride in zip(current_features, self.strides):
                disp, _ = _flow_sample(flows, cur.shape[-2], cur.shape[-1], stride,
                                       original_hw, detector_hw, cur.device)
                displacement.append(disp.to(cur.dtype))
        outputs, diagnostics = [], []
        for cur, refs, warped_i, valid_i, disp, stride in zip(current_features, raw, warped,
                                                               valid, displacement, self.strides):
            if refs.shape[0] != 3:
                raise ValueError("P29 expects three support maps")
            if return_aux:
                out, weights, vm = self._one_level(cur, refs, warped_i, valid_i, disp,
                                                   detector_hw, pad_hw, stride, True)
                diagnostics.append((weights, vm, out, cur))
            else:
                out = self._one_level(cur, refs, warped_i, valid_i, disp,
                                      detector_hw, pad_hw, stride, False)
            outputs.append(out)
        if return_aux:
            return {"features": outputs,
                    "warped": warped,
                    "valid": [x[1] for x in diagnostics],
                    "weights": [x[0] for x in diagnostics]}
        return outputs
