# P50 — Cross-view Fragment Teacher / Candidate Residual Student

## Boundary

This is a train-only representation candidate for MDMT. A privileged teacher may use
same-identity fragments from the paired training views while constructing supervision.
The inference student receives only the current target map, the current K64 source
candidate maps, and the candidate mask. It never receives GT IDs, XML boxes, future
frames, a global ID, pose, GPS, or a teacher feature.

## Input → operator → objective → output

1. **Input:** target ROI map `S`, source candidate maps `{C_k}`, `K<=64`, and a mask.
2. **Operator:** a candidate-conditioned spatial residual `R_k = f(S,C_k,
   LOO({C_j:j!=k}))`. The residual is map-level and is computed before pooling. A
   shared student projection produces an identity vector `z_k` and a no-match logit.
3. **Training-only teacher:** for a supervised positive target, aggregate identity
   fragments from other train episodes into a detached map prototype `T_id`; the
   teacher is used only as a target for `R_k` and is absent from the inference graph.
4. **Objective:** assignment loss over the K candidates plus dustbin, teacher residual
   consistency, positive-removal dustbin loss, same-cardinality negative-removal
   control, hard-negative margin, and a donor-permutation consistency check.
5. **Output:** candidate logits, a dustbin/no-match logit, and optional normalized
   identity vectors. The host solver is unchanged until the representation gate passes.

## Falsifiable gate

On disjoint train-pair calibration, P50 must beat the frozen visual-feature control by
at least +3 pp Recall@1 on 3/5 pairs, preserve or improve dustbin Brier, and change
under donor-fragment permutation. No official-val/test access is allowed before this
gate. A CPU/synthetic pass is only a mechanism contract.

## Risks

The concept is adjacent to fragment-level multi-view distillation and to set-relative
ReID. The claim is therefore deliberately narrow: map-level candidate residual
supervision for the MDMT K64/no-match episode, with no privileged-view inference or
visibility-memory module. If the mechanism collapses to ordinary ReID distillation or
post-pooling ranking, stop.
