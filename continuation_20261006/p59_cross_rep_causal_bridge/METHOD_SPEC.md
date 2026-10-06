# P59: Cross-Representation Causal Bridge (CRCB)

## Problem

The protected P26/P23 identity head already gives a strong current-observation
representation, but the train-only error decomposition identifies cross-view
ranking and no-link decisions as the bottleneck. P38 showed that a causal local
track history can carry transferable identity signal in the older P1 space, while
P23 has a stronger current representation. P59 tests whether the two signals can
be coupled without replacing the detector or MIA solver.

## Input -> operator -> objective -> output

`current frozen P23 head embedding z_t + strict-past same-view P1 history h_t ->
shared projected history q_t, gated residual r_t, and normalized z_t + m_t*g_t*r_t
-> bidirectional cross-view masked identity CE + causal history consistency ->
128D identity embedding sent to the unchanged P4/MIA affinity readout`.

The history uses at most four feature-present observations from the same local
track in the preceding eight frames. The numeric local ID is only a pointer into
the frozen tracker cache; it is never encoded and never used as an identity
label. Unknown labels are masked from the fit loss. No future frame, homography,
calibration label, or official validation/test stream enters the operator.

The current P23 embedding and all P1 history vectors are frozen. Only the small
bridge is trained. The final residual projection is zero initialized, giving an
exact current-P23 control at step zero. The first phase is a 1,200-update fit15
run followed by a frozen five-pair calibration readout. It cannot enter P26 until
it clears the pre-registered representation gate and a separate host audit.
