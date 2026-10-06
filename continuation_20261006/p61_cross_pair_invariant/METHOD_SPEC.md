# P61 Cross-Pair Invariant Identity Adapter (CPIA)

## Problem

The frozen P23 head has a large fit-to-calibration gap and P59 showed a valid
strict-past residual can overfit pair-specific appearance. The candidate treats
pair identity as an environment and learns an identity residual whose update is
useful on two different pair environments at once.

## Input -> operator -> objective -> output

`frozen P23 128D identity -> zero-initialized residual adapter -> normalized identity`

For each update, one legal labelled fit group from pair `p` and one group from a
different fit pair `q` are sampled. Each group uses the existing symmetric masked
cross-view identity CE over its own candidate set. The objective is

`mean(L_p,L_q) + lambda * (1 - cos(grad_theta L_p, grad_theta L_q))`,

with `lambda=0.05` and second-order gradients. No labels or rows are shared across
pair environments; pair-local pseudo IDs remain pair-local. Inference uses only the
normalized identity embedding and the existing P4 cosine scorer.

The module does not consume local IDs, time history, geometry, OT/assignment plans,
thresholds, or official-val/test information. A positive result would mean the
representation objective generalizes across held-out UAV pairs; it would not by
itself establish formal MOT improvement.
