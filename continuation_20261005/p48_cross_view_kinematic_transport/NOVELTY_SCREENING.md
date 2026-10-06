# P48 novelty screening

结论暂定：`HOLD_TIER2_ADJACENT_PRIOR_ART_RISK`，允许进入 CPU 机制门，尚未允许训练或论文声明。

1. MDMT/MIA already uses cross-view geometry and target refresh; P48 does not claim a new dataset or a new assignment solver.
2. UAVST-HM and SAME-MDMT use position/IoU/appearance fusion and Hungarian association; P48 must not be described as a new cost fusion or Hungarian improvement.
3. GTA-Net uses graph node/edge matching and affine completion; P48 has no cross-target graph or learned graph matching.
4. Airborne-camera trajectory association and DroneMOT use explicit motion/trajectory consistency; these are adjacent mechanism-level priors, so the claim must be limited to a learned Jacobian-transported heteroscedastic state model for synchronized MDMT front-door candidates.
5. Recent track-to-track association work combines temporal trajectory features with topology and optimal transport; P48 must avoid trajectory-graph encoding, transport-plan prediction, or set-level matching.
6. Uncertainty-aware multi-drone association already exists; the differentiator must be the cross-view *state transport residual* and its target-excluded prefix training, not generic covariance weighting.

Allowed claim after evidence: `a learned cross-view kinematic transport residual for synchronized two-drone MIA candidates`.

STOP conditions: direct same-task collision found; CPU rank gate fails; speed permutation has no effect; zero-history control matches; implementation cannot prove prefix/legal inputs; or gain appears only after threshold tuning.
