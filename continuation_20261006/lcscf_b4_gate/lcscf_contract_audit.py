#!/usr/bin/env python3
"""Contract audit for latent LCSCF transport before any long run."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import torch

from sci_id.models.full_model import build_arm_model
from sci_id.models.lcscf import LEVELS, LCSCFTransport, transport_contract_loss


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def boxes(batch: int, k: int, device: torch.device) -> torch.Tensor:
    x = torch.rand(batch, k, 4, device=device)
    lo = x[..., :2].clamp(0.03, 0.65)
    hi = (lo + x[..., 2:].clamp(0.05, 0.3)).clamp(max=0.97)
    return torch.cat((lo, hi), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    torch.manual_seed(501)
    b4 = build_arm_model("B4", channels=8, prototypes=2, output_size=3, backbone_init=args.checkpoint).to(device).eval()
    lcs = build_arm_model("LCSCF_B4", channels=8, prototypes=2, output_size=3, backbone_init=args.checkpoint).to(device).eval()
    common = {key: value for key, value in b4.state_dict().items() if key in lcs.state_dict()}
    lcs.load_state_dict(common, strict=False)
    images = torch.randn(2, 3, 64, 96, device=device)
    target_index = torch.tensor([0], device=device)
    source_index = torch.tensor([1], device=device)
    target_boxes = torch.tensor([[0.1, 0.1, 0.8, 0.9]], device=device)
    step_zero = {}
    for k, valid_k in ((1, 0), (1, 1), (64, 64)):
        candidate_boxes = boxes(1, k, device)
        mask = torch.full((1, k), valid_k > 0, dtype=torch.bool, device=device)
        with torch.no_grad():
            ref = b4.forward_packed(images, target_index, source_index, target_boxes, candidate_boxes, mask)
            got = lcs.forward_packed(images, target_index, source_index, target_boxes, candidate_boxes, mask)
        step_zero["k%d_valid%d_max_abs_error" % (k, valid_k)] = max(
            float((ref.candidate_logits - got.candidate_logits).abs().max().cpu()),
            float((ref.dustbin_logit - got.dustbin_logit).abs().max().cpu()),
        )

    # K=4 gradient and weak pair-label transport loss.
    lcs.train()
    k = 4
    candidate_boxes = boxes(1, k, device)
    mask = torch.ones(1, k, dtype=torch.bool, device=device)
    output = lcs.forward_packed(images, target_index, source_index, target_boxes, candidate_boxes, mask)
    batch = {
        "supervision_valid": torch.ones(1, dtype=torch.bool, device=device),
        "positive_mask": torch.tensor([[True, False, False, False]], device=device),
        "known_negative_mask": torch.tensor([[False, True, True, True]], device=device),
    }
    contract, contract_terms = transport_contract_loss(lcs.last_lcscf_stats, batch)
    loss = output.candidate_logits[:, 0].mean() + output.dustbin_logit.mean() + 0.1 * contract
    lcs.zero_grad(set_to_none=True)
    loss.backward()
    backbone_grad = sum(
        float(parameter.grad.detach().abs().sum().cpu())
        for name, parameter in lcs.named_parameters()
        if name.startswith("backbone.") and parameter.grad is not None
    )
    residual_grad = float(lcs.lcscf.residual_scale.grad.detach().abs().sum().cpu())
    finite = all(torch.isfinite(parameter).all().item() for parameter in lcs.parameters())
    gradient_finite = all(torch.isfinite(p.grad).all().item() for p in lcs.parameters() if p.grad is not None)
    stats_finite = all(
        torch.isfinite(value).all().item()
        for level in LEVELS
        for value in lcs.last_lcscf_stats[level].values()
    )
    checks = {
        "backbone_init_counts": lcs.backbone_init_counts,
        "step_zero": step_zero,
        "backbone_gradient_l1": backbone_grad,
        "residual_scale_gradient_l1": residual_grad,
        "transport_stats_finite": stats_finite,
        "parameters_finite": finite,
        "all_gradients_finite": gradient_finite,
        "contract_terms": contract_terms,
        "official_val_test_access": False,
    }
    if any(value > 1e-5 for value in step_zero.values()):
        raise AssertionError("LCSCF is not step-zero equivalent to B4")
    if backbone_grad <= 0 or residual_grad <= 0 or not stats_finite or not finite or not gradient_finite:
        raise AssertionError("LCSCF gradient/finite contract failed")
    payload = {
        "status": "PASS_LCSCF_MECHANISM_CONTRACT",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "device": str(device),
        "operator": "3x3 bidirectional latent transport with cycle/mass/entropy diagnostics",
        "placement": "paired identity RoI maps before B4 pooling",
        "learned_dense_correspondence": False,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
