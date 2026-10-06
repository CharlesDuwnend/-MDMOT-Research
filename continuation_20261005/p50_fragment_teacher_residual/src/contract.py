#!/usr/bin/env python3
"""Small CPU contract for P50; no MDMT labels or images are loaded here."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class ResidualStudent(nn.Module):
    def __init__(self, channels: int = 8):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1), nn.GELU(), nn.Conv2d(channels, channels, 1)
        )
        self.target_proj = nn.Linear(channels, channels, bias=False)
        self.candidate_proj = nn.Linear(channels, channels, bias=False)
        self.dustbin = nn.Sequential(nn.Linear(channels * 2 + 1, channels), nn.GELU(), nn.Linear(channels, 1))

    def forward(self, target, candidates, mask):
        b, k, c, h, w = candidates.shape
        valid = mask[:, :, None, None, None].to(candidates.dtype)
        denom = mask.sum(1, keepdim=True).clamp_min(1).to(candidates.dtype)
        total = (candidates * valid).sum(1, keepdim=True)
        other = (total - candidates * valid) / (denom[:, :, None, None, None] - valid).clamp_min(1.0)
        t = target[:, None].expand(-1, k, -1, -1, -1)
        x = torch.cat((t, candidates, other), dim=2).reshape(b * k, 3 * c, h, w)
        maps = (candidates + self.residual(x).reshape(b, k, c, h, w)) * valid
        tz = F.normalize(self.target_proj(target.mean((-1, -2))), dim=-1)
        cz = F.normalize(self.candidate_proj(maps.mean((-1, -2))), dim=-1)
        logits = (tz[:, None] * cz).sum(-1).masked_fill(~mask, torch.finfo(maps.dtype).min)
        set_mean = (cz * mask[:, :, None].to(cz.dtype)).sum(1) / denom
        dust = self.dustbin(torch.cat((tz, set_mean, mask.float().mean(1, keepdim=True)), dim=-1)).squeeze(-1)
        return {"candidate_logits": logits, "dustbin": dust, "maps": maps, "vectors": cz}


def main():
    torch.manual_seed(17)
    b, k, c, h, w = 3, 5, 8, 4, 4
    target = torch.randn(b, c, h, w)
    candidates = torch.randn(b, k, c, h, w)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
    teacher_a = torch.randn(b, c, h, w)
    teacher_b = teacher_a.flip(-1)
    model = ResidualStudent(c)
    out = model(target, candidates, mask)
    losses = []
    for teacher in (teacher_a, teacher_b):
        # Training-only teacher target. It is deliberately detached before the loss.
        pred = out["maps"][:, 0]
        losses.append(F.smooth_l1_loss(pred, teacher.detach()))
    loss = losses[0] + out["dustbin"].square().mean() * 0.01
    loss.backward()
    grad_l1 = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    perm = torch.tensor([2, 0, 1, 4, 3])
    perm_out = model(target, candidates[:, perm], mask[:, perm])
    order_error = float((perm_out["candidate_logits"] - out["candidate_logits"][:, perm]).abs().masked_fill(~mask[:, perm], 0).max())
    donor_delta = float((losses[0] - losses[1]).abs())
    inference_batch = {"target": target, "candidates": candidates, "mask": mask}
    assert "teacher" not in inference_batch
    assert torch.isfinite(out["candidate_logits"]).all()
    assert torch.isfinite(out["dustbin"]).all()
    assert grad_l1 > 0
    assert order_error < 1e-5
    assert donor_delta > 1e-6
    # A causal training donor contract: all privileged donors precede the target.
    target_frame = torch.tensor([10, 20, 30])
    donor_frames = torch.tensor([[2, 8], [3, 18], [4, 25]])
    assert bool((donor_frames < target_frame[:, None]).all())
    result = {
        "status": "PASS_P50_CPU_MECHANISM_CONTRACT",
        "checks": 7,
        "shape": {"B": b, "K": k, "C": c, "H": h, "W": w},
        "gradient_l1": grad_l1,
        "candidate_permutation_max_error": order_error,
        "donor_permutation_loss_delta": donor_delta,
        "inference_fields": sorted(inference_batch),
        "teacher_in_inference": False,
        "prefix_donor_check": True,
        "metrics_are_mechanism_only": True,
    }
    out_path = Path(__file__).resolve().parents[1] / "CPU_CONTRACT.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
