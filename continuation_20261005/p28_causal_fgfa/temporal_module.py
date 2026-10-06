"""Causal FGFA components, independent of the frozen MMTracking source.

Coordinates follow the frozen AutoAssign head's MlvlPointGenerator offset=0
and the nominal Caffe ResNet feature origin. A feature node j at stride s
has detector-image pixel-centre index j * s, not (j + .5) * s - .5.
Flow rasters cover a resize of the complete, unpadded original image; flow
vectors are current-to-past, in that resized flow image's pixel units.

No model weights, images, annotations, mutable caches, or external modules
are loaded by this file. Compatible with PyTorch 1.10.
"""

import math
from typing import Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_STRIDES = (8, 16, 32, 64, 128)


def _hw(value, name: str) -> Tuple[int, int]:
    if value is None or len(value) != 2:
        raise ValueError("%s must be (height, width)" % name)
    result = tuple(int(v) for v in value)
    if any(isinstance(v, bool) or int(v) != v for v in value):
        raise ValueError("%s must contain integers" % name)
    if min(result) < 1:
        raise ValueError("%s must be positive" % name)
    return result


def _geometry(original_hw, detector_img_hw, pad_hw):
    original_hw = _hw(original_hw, "original_hw")
    detector_img_hw = _hw(detector_img_hw, "detector_img_hw")
    pad_hw = _hw(pad_hw, "pad_hw")
    if any(a > b for a, b in zip(detector_img_hw, pad_hw)):
        raise ValueError("detector_img_hw must fit inside pad_hw")
    return original_hw, detector_img_hw, pad_hw


def _validate_features(features: Sequence[Tensor], strides, pad_hw,
                       expected_batch: Optional[int] = None):
    if len(features) != len(strides) or len(features) == 0:
        raise ValueError("features and strides must have equal nonzero length")
    batch = None
    channels = None
    for level, stride in zip(features, strides):
        if isinstance(stride, bool) or int(stride) != stride or stride < 1:
            raise ValueError("strides must be positive integers")
        if level.ndim != 4 or not level.is_floating_point():
            raise ValueError("each feature must be floating N,C,H,W")
        if expected_batch is not None and level.shape[0] != expected_batch:
            raise ValueError("unexpected feature batch/reference count")
        if batch is None:
            batch, channels = level.shape[:2]
        if tuple(level.shape[:2]) != (batch, channels):
            raise ValueError("all FPN levels must share reference count/channels")
        expected_hw = tuple(int(math.ceil(d / float(stride))) for d in pad_hw)
        if tuple(level.shape[-2:]) != expected_hw:
            raise ValueError("FPN shape %s differs from ceil(pad/stride) %s" %
                             (tuple(level.shape[-2:]), expected_hw))
    return batch, channels


def _feature_domain(height, width, stride, detector_img_hw, device, dtype):
    """Return P26 offset-zero pixel-centre indices and valid feature nodes."""
    y = torch.arange(height, device=device, dtype=dtype) * stride
    x = torch.arange(width, device=device, dtype=dtype) * stride
    current_valid = ((y[:, None] < detector_img_hw[0]) &
                     (x[None, :] < detector_img_hw[1]))
    # At an exact image boundary the centre is outside the half-open domain.
    max_y = min(height - 1,
                int(math.ceil(detector_img_hw[0] / float(stride))) - 1)
    max_x = min(width - 1,
                int(math.ceil(detector_img_hw[1] / float(stride))) - 1)
    return y, x, current_valid, max_y, max_x


def warp_past_features(past_features: Sequence[Tensor], flows: Tensor, *,
                       original_hw, detector_img_hw, pad_hw,
                       strides=DEFAULT_STRIDES):
    """Warp raw past FPN features onto the current FPN lattice.

    Args:
        past_features: one [R,C,H_l,W_l] tensor per stride (usually R=3).
        flows: [R,2,H_f,W_f], current-to-past displacements in flow-image
            pixels. H_f/W_f is the complete resized image, without padding,
            cropping, or letterboxing. Original and past image dimensions
            must be identical within the sequence.
        original_hw: original/native image size, before either resize.
        detector_img_hw: detector image size before bottom/right padding.
        pad_hw: padded detector input size. Coarse FPN levels use ceil sizes.

    Returns:
        (warped_levels, valid_levels). Valid tensors are bool [R,1,H_l,W_l].
        Invalid warped values are zero. A valid point has a current centre
        within the image, finite sampled flow, a past endpoint within the
        image, and no nonzero bilinear contribution from padding/outside
        feature nodes. This does not remove padding from CNN receptive fields.

    Flow sampling uses border extension only at the flow raster edge (the
    corresponding image-domain mask remains explicit). Feature sampling
    uses zeros and a stricter support mask; border clamping never legitimizes
    an out-of-domain feature endpoint. Exactly zero sampled displacement is
    copied from the original feature to avoid grid normalization round-off.
    The zero-flow copy is intended for frozen flow, not flow fine-tuning.
    """
    original_hw, detector_img_hw, pad_hw = _geometry(
        original_hw, detector_img_hw, pad_hw)
    count, _ = _validate_features(past_features, strides, pad_hw)
    if (flows.ndim != 4 or flows.shape[0] != count or flows.shape[1] != 2
            or min(flows.shape[-2:]) < 1 or not flows.is_floating_point()):
        raise ValueError("flows must be floating [R,2,H_f,W_f]")
    fh, fw = flows.shape[-2:]
    oh, ow = original_hw
    dh, dw = detector_img_hw
    warped_levels, valid_levels = [], []

    for refs, stride in zip(past_features, strides):
        _, _, height, width = refs.shape
        calc_dtype = torch.float64 if refs.dtype == torch.float64 else torch.float32
        if count == 0:
            warped_levels.append(refs.clone())
            valid_levels.append(torch.zeros((0, 1, height, width),
                                             dtype=torch.bool, device=refs.device))
            continue
        y, x, current_valid, max_y, max_x = _feature_domain(
            height, width, stride, detector_img_hw, refs.device, calc_dtype)
        # Detector centre -> original edge -> flow pixel-centre index. Keep
        # both resize steps explicit so their provenance is reviewable.
        flow_x = ((x + .5) * (ow / float(dw))) * (fw / float(ow)) - .5
        flow_y = ((y + .5) * (oh / float(dh))) * (fh / float(oh)) - .5
        flow_grid = torch.stack((
            (2 * (flow_x + .5) / fw - 1)[None, :].expand(height, width),
            (2 * (flow_y + .5) / fh - 1)[:, None].expand(height, width)), dim=-1)
        flow_grid = flow_grid[None].expand(count, height, width, 2)
        sampled = F.grid_sample(flows.to(device=refs.device, dtype=calc_dtype),
                                flow_grid, mode="bilinear", padding_mode="border",
                                align_corners=False)
        finite = torch.isfinite(sampled).all(dim=1)
        sampled = torch.where(torch.isfinite(sampled), sampled,
                              torch.zeros_like(sampled))
        dx = sampled[:, 0] * (dw / float(fw)) / float(stride)
        dy = sampled[:, 1] * (dh / float(fh)) / float(stride)
        jj = torch.arange(width, device=refs.device, dtype=calc_dtype)[None, None, :]
        ii = torch.arange(height, device=refs.device, dtype=calc_dtype)[None, :, None]
        qx, qy = jj + dx, ii + dy
        endpoint_x = x[None, None, :] + dx * stride
        endpoint_y = y[None, :, None] + dy * stride
        # Floating arithmetic can put an exact integer endpoint a few ulps
        # beyond the last valid feature. Only that numerical neighborhood is
        # accepted, and coordinates are clamped back before interpolation.
        tol = 8 * torch.finfo(calc_dtype).eps * max(height, width, 1)
        image_tol = tol * stride
        valid = (current_valid[None] & finite &
                 (endpoint_x >= -image_tol) & (endpoint_x < dw + image_tol) &
                 (endpoint_y >= -image_tol) & (endpoint_y < dh + image_tol) &
                 (qx >= -tol) & (qx <= max_x + tol) &
                 (qy >= -tol) & (qy <= max_y + tol))
        if max_x < 0 or max_y < 0:
            valid = torch.zeros_like(valid)
        # Invalid endpoints get a harmless grid coordinate; they are masked
        # to zero after sampling. The clamp also snaps ulp-only boundary noise.
        safe_qx = torch.where(valid, qx.clamp(0, max(max_x, 0)), torch.zeros_like(qx))
        safe_qy = torch.where(valid, qy.clamp(0, max(max_y, 0)), torch.zeros_like(qy))
        feature_grid = torch.stack((2 * (safe_qx + .5) / width - 1,
                                    2 * (safe_qy + .5) / height - 1), dim=-1)
        sampled_refs = F.grid_sample(refs.to(calc_dtype), feature_grid,
                                     mode="bilinear", padding_mode="zeros",
                                     align_corners=False).to(refs.dtype)
        zero_displacement = ((dx == 0) & (dy == 0))[:, None]
        sampled_refs = torch.where(zero_displacement, refs, sampled_refs)
        valid = valid[:, None]
        warped_levels.append(torch.where(valid, sampled_refs,
                                         torch.zeros_like(sampled_refs)))
        valid_levels.append(valid)
    return warped_levels, valid_levels


def _unaligned_features(past_features, detector_img_hw, pad_hw, strides):
    count, _ = _validate_features(past_features, strides, pad_hw)
    warped, valid = [], []
    for refs, stride in zip(past_features, strides):
        _, _, h, w = refs.shape
        _, _, mask, _, _ = _feature_domain(h, w, stride, detector_img_hw,
                                          refs.device, torch.float32)
        mask = mask[None, None].expand(count, 1, h, w)
        warped.append(torch.where(mask, refs, torch.zeros_like(refs)))
        valid.append(mask)
    return warped, valid


class ResidualValueAdapter(nn.Module):
    """Shared pointwise value adaptation; zero residual means exact identity."""

    def __init__(self, channels=256):
        super().__init__()
        self.residual = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x):
        return x + self.residual(x)


class TemporalAdapter(nn.Module):
    """Learned-value control and causal FGFA with identical value adapter.

    Modes: single=current only; fgfa=flow warp + cosine weights;
    uniform=flow warp + uniform valid weights; unaligned=raw past + cosine.
    Embedding depth follows the FGFA paper's 1x1/3x3/1x1 layout, adapted to
    256-channel FPN features. Parameters are shared across levels and frames.
    The embedding is unused in single and uniform modes, disclosed by
    parameter_counts(); it is not used merely to equalize parameter totals.
    """

    def __init__(self, channels=256, mode="fgfa", strides=DEFAULT_STRIDES,
                 norm_eps=1e-6):
        super().__init__()
        if mode not in ("single", "fgfa", "uniform", "unaligned"):
            raise ValueError("unknown temporal mode: %s" % mode)
        if channels < 1 or norm_eps <= 0:
            raise ValueError("channels and norm_eps must be positive")
        self.channels, self.mode = channels, mode
        self.strides, self.norm_eps = tuple(strides), norm_eps
        self.value_adapter = ResidualValueAdapter(channels)
        self.embedding = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 1))

    def parameter_counts(self):
        value = sum(p.numel() for p in self.value_adapter.parameters())
        embedding = sum(p.numel() for p in self.embedding.parameters())
        used = self.mode in ("fgfa", "unaligned")
        return {"value_adapter": value, "embedding": embedding,
                "active": value + (embedding if used else 0),
                "unused_embedding": 0 if used else embedding,
                "total": value + embedding}

    def forward(self, current_features: Sequence[Tensor],
                past_features: Optional[Sequence[Tensor]] = None,
                flows: Optional[Tensor] = None, *, original_hw=None,
                detector_img_hw=None, pad_hw=None, return_aux=False):
        """Return a list of FPN tensors, or an auxiliary dictionary.

        return_aux=True returns {features, warped, valid, weights}; weights
        include current as item zero and have shape [R+1,1,H_l,W_l]. Current
        is always retained, even on padded output nodes, so those nodes fall
        back to the original detector feature instead of introducing NaNs.
        All current tensors remain in the autograd graph.
        """
        if len(current_features) != len(self.strides) or not current_features:
            raise ValueError("current_features must match the configured strides")
        for current in current_features:
            if current.ndim != 4 or tuple(current.shape[:2]) != (1, self.channels):
                raise ValueError("current features must be [1,channels,H,W]")
            if not current.is_floating_point():
                raise ValueError("current features must be floating point")

        if self.mode == "single" or past_features is None:
            outputs = [self.value_adapter(current) for current in current_features]
            warped = [current.new_empty((0, self.channels) + current.shape[-2:])
                      for current in current_features]
            valid = [torch.zeros((0, 1) + current.shape[-2:], dtype=torch.bool,
                                 device=current.device) for current in current_features]
            weights = [torch.ones((1, 1) + current.shape[-2:], dtype=current.dtype,
                                  device=current.device) for current in current_features]
        else:
            original_hw, detector_img_hw, pad_hw = _geometry(
                original_hw, detector_img_hw, pad_hw)
            _validate_features(current_features, self.strides, pad_hw, expected_batch=1)
            _validate_features(past_features, self.strides, pad_hw)
            for current, refs in zip(current_features, past_features):
                if refs.shape[1:] != current.shape[1:] or refs.device != current.device:
                    raise ValueError("past/current shapes and devices must match")
                if refs.dtype != current.dtype:
                    raise ValueError("past/current feature dtypes must match")
            if self.mode == "unaligned":
                warped, valid = _unaligned_features(past_features, detector_img_hw,
                                                    pad_hw, self.strides)
            else:
                if flows is None:
                    raise ValueError("flow-aligned modes require flows")
                warped, valid = warp_past_features(
                    past_features, flows, original_hw=original_hw,
                    detector_img_hw=detector_img_hw, pad_hw=pad_hw,
                    strides=self.strides)
            outputs, weights = [], []
            for current, refs, mask in zip(current_features, warped, valid):
                raw = torch.cat((current, refs), dim=0)
                available = torch.cat((torch.ones_like(mask[:1], dtype=torch.bool)
                                        if len(mask) else torch.ones(
                                            (1, 1) + current.shape[-2:],
                                            dtype=torch.bool, device=current.device), mask), dim=0)
                if self.mode == "uniform":
                    weight = available.to(current.dtype)
                    weight = weight / weight.sum(dim=0, keepdim=True)
                else:
                    embedded = F.normalize(self.embedding(raw), p=2, dim=1,
                                            eps=self.norm_eps)
                    cosine = (embedded * embedded[:1]).sum(dim=1, keepdim=True)
                    weight = cosine.masked_fill(~available, float("-inf")).softmax(dim=0)
                values = self.value_adapter(raw)
                outputs.append((weight * values).sum(dim=0, keepdim=True))
                weights.append(weight)
        if return_aux:
            return {"features": outputs, "warped": warped,
                    "valid": valid, "weights": weights}
        return outputs


def validate_temporal_group(*, view, pair, current_frame=None,
                            reference_records: Sequence[Mapping],
                            expected_lags=(1, 4, 8), currentframe=None):
    """Validate same-sequence past records ordered from recent to old.

    Records require pair, view and frame (frame_id is also accepted). The
    helper reads metadata only; callers must bind it to actual feature/flow
    manifest records. It does not infer continuity from filenames or IDs.
    Returns the tuple of lags. currentframe is a compatibility spelling;
    providing both spellings is rejected to avoid silent ambiguity.
    """
    if current_frame is not None and currentframe is not None:
        raise ValueError("supply only one current-frame argument")
    current_frame = current_frame if current_frame is not None else currentframe
    if (current_frame is None or isinstance(current_frame, bool) or
            int(current_frame) != current_frame or current_frame < 0):
        raise ValueError("current_frame must be a nonnegative integer")
    lags = []
    seen = set()
    for record in reference_records:
        if str(record.get("pair")) != str(pair) or str(record.get("view")) != str(view):
            raise ValueError("reference must have the same pair and view")
        frame = record.get("frame", record.get("frame_id"))
        if frame is None or isinstance(frame, bool) or int(frame) != frame or frame < 0:
            raise ValueError("reference frame must be a nonnegative integer")
        if frame >= current_frame:
            raise ValueError("reference must be strictly past, never current/future")
        if frame in seen:
            raise ValueError("duplicate reference frame")
        seen.add(frame)
        lags.append(current_frame - frame)
    if lags != sorted(lags):
        raise ValueError("reference order must be recent to old")
    if expected_lags is not None and tuple(lags) != tuple(expected_lags):
        raise ValueError("reference lags differ from the declared group contract")
    return tuple(lags)
