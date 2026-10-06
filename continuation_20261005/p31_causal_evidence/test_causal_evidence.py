#!/usr/bin/env python3
"""CPU mechanism-contract gate; no MDA/IDF1/MOTA and no dataset reads."""
import json
import time
from pathlib import Path

import torch

from causal_evidence import CausalEvidenceIntervention, parameter_count, sinkhorn_with_dustbin


def main():
    torch.manual_seed(31)
    b, na, nb, d = 3, 5, 4, 12
    a = torch.randn(b, na, d, requires_grad=True)
    c = torch.randn(b, nb, d, requires_grad=True)
    aa = torch.tensor([[0., 1., 4., 8., 12.], [0., 2., 5., 0., 0.], [0., 3., 7., 10., 0.]])
    ca = torch.tensor([[0., 1., 4., 8.], [0., 2., 5., 0.], [0., 3., 7., 0.]])
    am = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    cm = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
    model = CausalEvidenceIntervention(d, hidden_dim=32, output_dim=24, max_age=16.)
    out = model(a, aa, am, c, ca, cm)
    assert out['match_logit'].shape == (b,)
    assert out['left']['influence'].shape == (b, na)
    assert out['right']['influence'].shape == (b, nb)
    assert all(torch.isfinite(x).all() for x in [out['match_logit'], out['null_left_logit'], out['null_right_logit']])

    # Set contract: permuting support tokens and ages together is invariant.
    perm = torch.tensor([4, 1, 3, 0, 2])
    p = model(a[:, perm], aa[:, perm], am[:, perm], c, ca, cm)
    assert torch.allclose(out['match_logit'], p['match_logit'], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out['left']['embedding'], p['left']['embedding'], atol=1e-6, rtol=1e-6)

    # Mask contract: masked garbage cannot change the representation.
    garbage = torch.full_like(a, 1e5)
    masked = ~am
    amod = torch.where(masked.unsqueeze(-1), garbage, a)
    q = model(amod, aa, am, c, ca, cm)
    assert torch.allclose(out['match_logit'], q['match_logit'], atol=1e-6, rtol=1e-6)

    # Empty support is legal at the API boundary and must stay finite; the
    # host policy will normally inject the current observation before calling.
    empty = torch.zeros_like(am)
    e = model(a, aa, empty, c, ca, cm)
    assert torch.isfinite(e['left']['embedding']).all() and torch.isfinite(e['left']['quality']).all()

    # No negative ages means the operator cannot encode future evidence.
    try:
        model(a, aa - 20., am, c, ca, cm)
    except ValueError:
        pass
    else:
        raise AssertionError('future-age contract was not enforced')

    loss = (out['match_logit'].square().mean() + out['null_left_logit'].square().mean()
            + out['null_right_logit'].square().mean())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)

    logits = torch.randn(2, 3, 4, requires_grad=True)
    assn = sinkhorn_with_dustbin(logits, torch.randn(2, 3), torch.randn(2, 4), iters=10)
    assert assn.shape == (2, 4, 5) and torch.isfinite(assn).all()
    assn.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    receipt = {
        'status': 'PASS_P31_CPU_MECHANISM_CONTRACT',
        'tests': ['finite_forward', 'permutation_invariance', 'masked_garbage_invariance',
                  'empty_support_finite', 'future_age_rejected', 'finite_backward',
                  'dustbin_assignment_finite'],
        'parameter_count': parameter_count(model),
        'torch': torch.__version__, 'seed': 31, 'elapsed_seconds': 0.0,
        'official_metric_authorized': False, 'dataset_or_labels_read': False,
    }
    Path(__file__).with_name('P31_CPU_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
