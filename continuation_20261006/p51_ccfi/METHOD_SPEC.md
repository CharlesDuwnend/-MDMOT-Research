# CCFI: Causal Cross-View Factorized Identity

CCFI is a train-only representation candidate for the frozen MDMT association
diagnostic. It targets the observed bottleneck in MDMT-COD: candidate coverage
is high, while the identity ranking/commit is poor.

## Input -> operator -> objective -> output

`current frozen P1 128D embedding + strictly-past same-view local-track history
mean/dispersion -> factorized identity/change/uncertainty adapter ->
cross-view masked identity matching + residual orthogonality + causal-change
prediction + uncertainty calibration -> normalized identity core and pair score`.

For a row at time `t`, the history contains at most four observations from the
same local track, all strictly earlier than `t` and no older than eight frames.
The numeric local-track ID is only a history pointer. It is never a feature and
never enters the loss. Unknown, ambiguous, and unsupported labels are masked.

The adapter emits:

* `z_id`: normalized identity core used for cross-view matching;
* `z_chg`: normalized temporal-change residual used only by the factorization
  objective;
* `u`: positive uncertainty predicted from current/history disagreement.

The pair score is `z_id_a^T z_id_b / (tau * (1 + u_a + u_b))`. The uncertainty
term is bounded and cannot produce a non-finite score. The residual is trained
to explain the causal change direction while an orthogonality term prevents it
from becoming an identity shortcut.

## Why this is a substantive test

P38 showed a consistent but below-gate gain from causal history. P39/P41 showed
that simply averaging or anchoring on history is insufficient. CCFI tests the
next mechanism-level hypothesis: temporal change and identity evidence should
be represented separately, and unstable observations should contribute less to
the cross-view decision. It does not change the detector, candidate generator,
local tracker, homography, or official solver.

This remains a restricted train-only research candidate. A passing retrieval
gate would authorize a separate P26 front-door test; it would not by itself
establish official MOT improvement or originality.
