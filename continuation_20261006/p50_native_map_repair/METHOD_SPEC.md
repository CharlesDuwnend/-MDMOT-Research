# Native cross-view set-relative residual (candidate gate)

## Research boundary

The protected host is the P26 MIA-Net + AutoAssign R50-FPN + ByteTracker baseline. This gate does not retrain the detector and does not alter the host solver. It tests one trainable representation/assignment module against the frozen native stage-3 feature cosine.

## Input → operator → objective → output

- **Input:** target native 256-D ROI feature `a`, source-view candidate features `c_1...c_K` (`K<=64`), and the candidate mask. The features are read from the detector's native `(1333,800)` keep-ratio pipeline; no direct 256x448 image resize is introduced.
- **Operator:** for candidate `i`, compute a leave-one-out set context `u_i = mean({c_j:j != i})`, then form `[a,c_i,c_i-u_i,c_i-a]`. A shared residual MLP produces `r_i`; a gate controls `c_i + gate_i*r_i` before normalization. The score retains the frozen cosine at initialization and adds a zero-initialized learned correction. The no-match head uses target/set summary, maximum candidate similarity, and candidate count.
- **Training-only teacher:** when a strict-past source feature for the target's training identity exists, its mean is a fixed raw 256-D native feature. It is detached by construction: there is no teacher projection and no teacher in the forward deployment graph.
- **Objective:** multi-positive assignment likelihood over candidates plus a dustbin, with optional teacher alignment on positive residual vectors. Rows with zero candidates or no positive candidate remain in the loss as dustbin supervision under the existing train episode contract.
- **Output:** candidate logits and a dustbin/no-match logit. The module can replace only the visual assignment score; detector, host initialization, Kalman state, geometry, and ByteTracker are unchanged.

## Falsifiable gate

On held-out train calibration pairs (69/70/74/76/78), compare exact same events and seeds:

1. `rel_on_teacher_off` vs frozen native cosine tests the set-relative residual itself.
2. `rel_on_teacher_on` vs `rel_on_teacher_off` tests whether the privileged teacher contributes beyond the operator.
3. `pooled_teacher_off/on` has equal parameter shapes but no leave-one-out set context; it tests whether a candidate-set operator is needed.
4. Every arm uses two seeds, 2400 updates, same batch/order, and the fit pairs 23/25/27/28/29/30/32/39/42/44/45/50/51/53/54.

Primary calibration readout is pair-macro Recall@1 and MRR on events with at least one labelled positive; dustbin accuracy and label rate are reported on all valid events. Frame rows are not independent replicates. No arm is selected from official val/test.

A direction is **KEEP for deeper spatial-map implementation** only if the relational-off contrast is positive on at least 3/5 pairs in both seeds, teacher-on adds no less than 1 pp pair-macro Recall@1 without degrading dustbin accuracy by >2 pp, and implementation audit passes. Otherwise mark the direction HOLD/STOP with the failure reason; do not tune thresholds to rescue it.

## Novelty boundary

Training-only privileged multi-view distillation is already present in 2026 person-ReID work. The candidate is therefore held at a narrow MDMT assignment/operator claim until a direct same-task collision search and all controls pass. A diagnostic gain is not a paper claim.
