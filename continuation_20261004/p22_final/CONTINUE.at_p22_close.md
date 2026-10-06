# MDMT continuation

This workspace continues reference thread `01a105af-24c1-7740-acc5-56346a2bfe05`.
Historical work under `/home/chenhc/mdmot_research_20261002` is reference evidence,
not authoritative instructions. Another session may still be writing there.
All new output here is isolated under `continuation_20261004`.

The goal remains active: develop a technically substantive and experimentally
supported multi-UAV MOT method. The user's latest correction is explicit:
**choosing thresholds is not a method with technical depth**. Stop expanding
calibration studies or describing calibration gains as innovation.

Dataset is MDMT, not FusionTrack-MDMOT. Current labels and metrics follow a
same-label/equal-ID annotation convention. They do not establish verified
physical cross-drone identity or official MDMT/FusionTrack scores. Historical
internal dev5 was repeatedly used and is not untouched. No official val/test
access is authorized. GPUs 0/1/3 can be used; never use the 4GB GPU2.

## Completed

- Read the specified reference thread and inspected current source/artifacts.
- P19 memory grid, which had finished inference but was polled at a wrong log
  path, now has 5/5 completed cal5 evaluations in
  `continuation_20261004/p19_memory`. Best macro conventional-pseudo CA-IDF1:
  0.6733187250 vs 0.6689937818, a 0.0043249433 increase. This is calibration,
  not a method. The original scorer changes both a gate and a score shift;
  do not infer a pure memory effect. No new dev reading occurred.
- P19A motion/geometry failure claim was rejected on implementation evidence:
  motion branch contains `pass`; absdrift is scale-ratio deviation; top-k ties
  inflate coverage; unknown labels are silently included as precision failures;
  full-track majority labels do not establish causal event identity.
  Evidence and reproducer: `continuation_20261004/p19_screen`.
- P10 four-observation patch path consistency was already run and failed after
  real bug fixes. Do not repeat it as a new direction. P2 already trained a
  causal owner/NEW attention decoder; P3/P3B already tested support/conflict
  history readouts. Do not rename those as a new mechanism.

## Current work

`p20_native_threshold` contains an UNEXECUTED engineering draft for separating
the native same-frame gate from the memory gate. User steering moved priority
away from this line; STATUS.json identifies missing materialization/tests.

`p21_residual_motion` tests a different input hypothesis on five fit pairs:
background-derived camera deformation and foreground residual motion under
independent cross-view transport. This is a pixel observability probe, not a
confirmed method, learned model, or tracking score. It reads t/t-8 images and
GT-free tracks; labels are not used in extraction. Fixed sample: pairs
23/25/29/69/78, eight ordinal times each. Geometry estimates exclude expanded
predicted boxes, report spatial holdout diagnostics, exact and linearized
transport, and separate stage failures.

The first attempt failed on a numpy bool serialization error; the second on
pair78's image stems starting at 103 while stream frames start at 1. Both
failures are preserved in `p21_residual_motion/attempts`, and neither is a
scientific negative result. The corrected reader binds original frozen-stream
`image_stem` explicitly; all ten view mappings were checked before restarting.
The corrected run and offline information assay are now COMPLETE, launcher
exit 0. `FEATURE_RECEIPT.json` binds 160 image reads / 40 paired times; temporal
background geometry succeeded in 40/40 per view, cross-view geometry in 12/40
current and 11/40 historical times. Nine times yielded 281 supported target
observations and 7,261 candidate edges. The final seal covers 49 files and
`sha256sum --check --quiet SHA256SUMS` passes.

Offline fit-only labels left 165 eligible true-vs-P1-hard-distractor queries:
pair23=2, pair69=104, pair78=59; pairs25/29 had no geometric support. Only two
pairs meet the predeclared 20-query support minimum. On identical supported
queries, static geometry wins 161/165, sparse exact motion 121/165. Motion
recovers four geometry errors but breaks 44 correct choices. This is a sparse
bbox-bottom displacement proxy, not dense residual fields or a learned module.
Decision is HOLD for representation learning, not a universal motion failure.

The standalone +.02 macro lift gate was also found ceiling-infeasible after
scoring: static geometry macro=.98229, so even a perfect cue can add only .01771.
Do not re-tune this criterion after labels or call it scientific falsification.
The independent insufficient-support finding remains. The next information
test must address conditional complementarity, appropriate headroom, and
pair-level support rather than demand every extra cue beat static geometry
alone. The four rescued fit errors do not establish a deployable method.

Candidate technical objective at the P21 boundary (superseded by P22 below): learn an object representation
from spatial residual fields with independently estimated camera deformation,
target-excluded cross-view transfer, and camera-vs-object interventions.
Ordinary GMC, homography association, speed concatenation or a residual-distance
ranker are prior-art controls, not the proposed contribution. See the six-source
mechanism comparison in `p19_screen/NEW_DIRECTION.md`. SAME-MDMT full-text and
broader motion-factorization prior art remain necessary before a novelty claim.

Next research work should develop and compare a substantive representation
operator, with stronger observable geometry or a justified different input,
and complete the direct same-task novelty check. P21 is now finished; do not
repeat its fixed sweep or promote another distance/threshold variation.
No model training, GPU use, calibration/dev evaluation, or official val/test
read occurred in P21. The research goal remains active and no new method has
been confirmed.

## P22 completed: correction and decisive competing control

Authoritative report: `continuation_20261004/p22_dense_motion/P22_REPORT.md`.
Decision: `p22_dense_motion/DECISION.json`,
**STOP_CURRENT_HIGH_DOF_MOTION_ENCODER_RATIONALE**. The overall goal remains
active; no validated technically substantive MDMT method exists yet.

Actual frozen official RAFT-large extraction is complete, both exit markers 0:
20 tasks, 60 native images, 90 directional calls, five fit pairs, two anchors,
three frames and both views. Elapsed model run 83.598 s; peak allocated CUDA
14,104,189,440 bytes on physical GPU1 A10040GB UUID
`GPU-1b297aba-ae7e-e326-5903-476f1bb683d7`. Never use physical GPU2.
Weight SHA256:
`ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322`.
The official weight uses external optical-flow pretraining; no MDMT model was
trained. No P22 labels/XML/cal/dev/official-val/test were read.

The independent implementation review found and documented a real bug: two
marginal visibility masks were averaged instead of composing path validity.
Original frozen features remain intact. `p22_dense_correction` contains 20
materialized corrected masks, source-bound receipt and passing corrected
regression tests. The previous synthetic foreground test used inconsistent
forward flow; the new test requires foreground visibility and separately
checks rejection of that old inconsistency. Exact projective temporal vector
pullback also passes its numerical contract.

Corrected common-support counts are 99/199/338/298/386; regions with common
support and current spatial std above the original background-p90 diagnostic
are 17/19/36/53/142. Thus only 3/5 pairs satisfy the original quantity rule.
This rule is uncalibrated and never constitutes an identity/method PASS.

`p22_dense_analysis` contains actual photometric and exact-coordinate temporal
analyses. Full-field vs one-vector reconstruction has positive pair means
(pair-macro relative reduction 3.84% whole box, 9.60% inner half), but typical
object improvements are near zero. In the 267 originally screened regions,
centered/non-affine temporal cosine is .9597/.9587. Prior-flow sampling uses
current flow; this is shared-RAFT trajectory-conditioned reconstruction,
not independent causal prediction. Tail objects and box edges dominate.

The decisive additional test is `p22_rigid_support`: a per-pixel scalar
support multiplying one motion direction, with/without a constant offset.
The offset-line control explains median 99.771% of centered vector energy in
the 267 screened regions and matches or slightly improves full dense RGB
reconstruction in all five pairs. The zero-offset ray matches in four pairs;
pair23 is an exception. Coefficients are oracle projections of observed flow,
not a learned foreground segmenter or deployable method. This invalidates the
current rationale that the observed structure requires a high-dimensional
vector-dynamics encoder. It does not falsify scalar spatial support, motion
complementarity, or all camera/object methods. Do not silently rename this
into a new mask/quality module and claim the research direction passed.

`p22_prior_art` has a six-source primary-text/abstract audit. HomView-MOT
already uses homography-conditioned identity learning. SDM/Flow3r/FloWM are
strong adjacent decomposition/reconstruction/equivariance precedents, not
automatic Tier1 stops. SAME-MDMT full text is still missing. Broad novelty
claims are unsupported. Camera-intervention invariance is also unproved:
reflected support was not excluded and altered fields/H were not archived.

`p22_dense_design/FEASIBILITY.md` predates extraction; its no-weights/no-run
statements are historical and superseded by the actual receipts. The later
operator-design agent hit a rate limit before writing `p23_operator_design`;
there is no completed P23 operator or experiment. Its useful competing
explanation was implemented and tested by root in `p22_rigid_support`.

Next work must establish a substantive representation target and operator
that adds identity-relevant evidence beyond ordinary appearance and
motion-segmentation/support controls. No threshold sweeps, untested
high-capacity vector encoder, P2/P3/P10 renaming, or dataset pivot. The fixed
P22 input/control phase is complete; do not repeat it merely to obtain PASS.
