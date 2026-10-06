# P62 COST

P62 follows the verified COD bottleneck: candidate recall is high but wrong accepted
edges persist and contaminate owner state. It tests a learned prefix-causal state
transition rather than another descriptor residual. The first deliverable is a
train-only data and mechanism contract; no official metric or novelty claim is
allowed until descendant state replay is independently verified.

The repaired contract currently covers 20 legal train pairs, 179,949 candidate
rows, and 2,552 prefix-positive rows. It passes finite gradients and the synthetic
OwnerState provisional/revoke/final transition. One PyTorch API typo was repaired
before accepting this contract; it produced no scientific result.
