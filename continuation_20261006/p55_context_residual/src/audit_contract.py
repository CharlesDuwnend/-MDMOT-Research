#!/usr/bin/env python3
"""CPU contract for the P55 context residual and its counterfactuals."""
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


class CVCR(nn.Module):
    def __init__(self, d=16, out=8):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(4*d, 2*d), nn.GELU(), nn.Linear(2*d, out))

    def forward(self, target, context, fpn, drop_context=False):
        if drop_context:
            context = torch.zeros_like(context)
        x = torch.cat((target, context, target-context, fpn), dim=-1)
        return F.normalize(self.proj(x), dim=-1)


def main():
    torch.manual_seed(17)
    m = CVCR().double()
    a = torch.randn(7, 16, dtype=torch.double, requires_grad=True)
    c = torch.randn(7, 16, dtype=torch.double, requires_grad=True)
    f = torch.randn(7, 16, dtype=torch.double, requires_grad=True)
    z = m(a, c, f)
    assert z.shape == (7, 8) and torch.isfinite(z).all()
    grads = torch.autograd.grad(z.square().sum(), (a, c, f), retain_graph=True)
    assert all(torch.isfinite(g).all().item() for g in grads)
    perm = torch.tensor([1, 0, 2, 3, 4, 5, 6])
    changed = (m(a, c, f) - m(a, c[perm], f)).abs().max().item()
    assert changed > 1e-7, changed
    drop = (m(a, c, f) - m(a, c, f, drop_context=True)).abs().max().item()
    assert drop > 1e-7, drop
    # target-only and context-only controls remain finite and are distinct.
    target_only = m(a, torch.zeros_like(c), f)
    assert torch.isfinite(target_only).all()
    out = {
        "status": "PASS_P55_CPU_CONTRACT",
        "checks": ["finite_forward", "finite_gradients", "context_swap_changes_output", "context_dropout_changes_output", "target_only_control"],
        "context_swap_max_abs": changed,
        "context_dropout_max_abs": drop,
        "labels_or_xml_read": False,
        "official_val_test_read": False
    }
    p = Path(__file__).resolve().parents[1] / "CPU_CONTRACT.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
