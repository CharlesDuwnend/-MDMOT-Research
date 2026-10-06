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

## P23 visual encoder: matched head-only vs backbone fine-tune - COMPLETE (2026-10-04)

Authoritative receipt: `continuation_20261004/p23_visual/P23_VISUAL_RECEIPT.json`
(sha256 `80e7ee39daceb1cade7f07d9e7f7d8d4d2c83d8c872ce78af6453529a52bc425`,
`SHA256SUMS` 9/9 OK). This is a visual-adaptation BASELINE, not a novel method.

Two arms, identical init (sealed P14 `frozen_adapter` head), identical 1200-step
fit15 schedule, no augmentation, seed 42, on physical GPU1 A100-40GB (never GPU2):

| arm | trainable params | final loss | cal5 pair-macro r1 |
|---|---:|---:|---:|
| sealed P14 frozen_adapter (reference) | - | - | 0.475915 |
| head_only (continue adapter, frozen DINOv2) | 50,048 | 0.238 | 0.479026 |
| visual_ft (fine-tune DINOv2 backbone + head) | 22,106,240 | 0.216 | **0.502028** |

`visual_ft` lift over `head_only` **+0.0230, 5/5 pair wins** -> pre-registered
verdict `KEEP_STRONGER_VISUAL_BASELINE`. The sealed `frozen_adapter` reference
reproduces 0.4759154376000505 exactly, and a byte-consistency probe re-encoding
the cal5 crops with the pretrained backbone matches the frozen `unit_dino` table
at min cosine 0.99991-0.99999 (float32 batch numerics, same as the earlier probe;
full-population floor 0.99986 over 54,488 fit rows).

Look-backs this stage: the smoke `cosine > 0.999999` assert was a float32
numerics artifact (512/512 self-argmax, min margin 0.0196 => row<->vector pairing
correct); tolerance patched to 0.9998 with the full-population probe as evidence.

**Meaning for the direction.** This is the first *held-out, mechanism-level*
positive signal in this workspace: a learned cross-view appearance
representation transfers to held-out cal5 pairs (unlike the P6 FLD / P6 ReID-head
/ P11 TEC NULLs, which were re-scoring on frozen features). It is still a
retrieval readout, not the complete-output MOT metric. The next test must carry
the `visual_ft` representation into the coverage-matched deferred-merge host run
and read dev5 CA-IDF1 - not another threshold/calibration study. No novel method
is yet validated; the goal remains active.


## P24 image background bootstrap complete; P25 actual image-localization pilot in progress (2026-10-05)

P24 isolated path: `continuation_20261005/p24_background_bootstrap/REPORT.md`.
54-file SHA seal verified; manifest SHA `bfc651722cf615c09a21d4341af93af92f7ce4ec0562b59b924d1f04b8eb0813`.
All 15 fit first-frame pairs, 30 images, 75 SuperPoint extractions / 60 LightGlue
matches completed exit 0,13.393 s on physical GPU3 A10040GB,557851136 peak bytes.
14/15 fixed background support checks pass; angle selection and H fit use only
spatial support matches. Joint matcher still sees all keypoints, so holdout is
H-fit consistency only. Four-angle matching is directly STCA prior art, not novel.

After feature freeze, fit-only cache-label readout gives876 bidirectional positive
queries, center geometric NN pair-macro R@1 .914193 (raw coordinate .070063).
This is annotation-convention initial-frame localization, NOT MDA or physical-ID
verification. Fifty errors >2 bbox diagonals all lie outside support hull; pair44
fails background gate but target R@1 .9773, so that gate is not target quality.
Official reference reproduction also seeds GT boxes/IDs/confirmed Kalman tracks,
not merely H; this thread has not executed that GT-initialized host.

P25 now: `continuation_20261005/p25_pixel_localization`. Implementing actual
source-conditioned receiver-image localization with ordinary Siamese spatial
correlation vs receiver-only, identical input-derived windows. Fit15×16 times,
then calibration5×8 times; per-frame image H (not stale first-frame reuse).
Source/receiver RGB from current predictions; receiver boxes/GT never select
window. P23 DINO backbone initialization, last4 blocks trainable; same1200-step
pair-balanced schedule. This is a strong image-localization pilot; new-method
claim remains HOLD. Do not call ordinary H+attention or Siamese search novel.
No official-val/test/dev reads or GPU2. Root training not yet launched at this
entry; inspect actual P25 receipts and tmux state before continuing.

## USER PRIORITY CORRECTION — BASELINE FIRST (2026-10-05)

User: 至少有一个和别人差不多的方法，或直接搬别人的过来再改；差20倍时
不应继续在弱host上堆新模块。This supersedes the immediately preceding P25 next-step.
P25 is DEFERRED before data decoding/GT read/training; scripts are unverified drafts.
Do not resume P25 automatically. `p25_pixel_localization/STATUS.json` is authoritative.

Active work: `continuation_20261005/p26_mia_baseline`. Import exact mature MIA-Net
full pipeline from reference, independently rescore archived test14 outputs, bind
code/config/checkpoint/protocol, and perform full fit23 inference replay in isolated
source/run paths. Keep its official GT first-frame initialization explicit. Aim for
concrete runnable paper-comparable baseline; no threshold sweep or new method claim.
Reference reported41.73 MDA vs published41.72 is being independently checked.

## P26 mature MIA baseline — VERIFIED AND CURRENT (2026-10-05)

See CURRENT_BASELINE.md and continuation_20261005/p26_mia_baseline/README.md.
Independent full test14 archive rescore: MDA41.7308952911%,IDF168.3410237378%,
MOTA49.6971680799%; published comparison41.72/68.24/49.68. All14pair metrics
match reference summaries exactly. This is macro-level comparable, not per-pair
identity with paper. Original official evaluator retained; 28JSON/GT snapshots
copied and sealed; no new test14 inference this phase.

Actual imported original MIA+supplement fit23 replay completed exit0:700frames
per view,16642+17193predictionrows,121.462s on physicalGPU3 A10040GB. Source488
files copied/verified; wrapper reconstructed EXACT historical SHA96c91509...;
checkpoint8894ea5f.../originalsingle-scaleconfig retained. Official first-frame
GT boxes/IDs/confirmed tracks/Kalman/H explicitly preserved. Do not call GT-free.

P25 is deferred by user, never trained/decoded/rawGT-read. Do not automatically
resume its scripts or threshold/TTA variants. User priority: start from this
mature, comparable full method, then develop substantive structural changes.
No new method has been validated; broader research goal remains unfinished.

## P27 diagnostics and P28 structural detector pilot — ACTIVE (2026-10-05)

The user clarified the reference thread already meets a paper baseline standard;
P26 adopts and independently verifies that achievement. Do not imply the
reference lacked a strong baseline. Its newer completed variants are audited
read-only under continuation_20261005/reference_improvements_20261005.

P27: continuation_20261005/p27_strong_host_temporal. Fit15 same-view strict-past
visibility audit completed: 868562 supported GT observations,33665 missing in
historical >=.1 detector streams;23930 of misses have past8 support (71.083% of
misses,2.755% of all GT). This is oracle observability, not recovered performance.
Fresh native vs historical exact-order parity FAILED42/90; preserve original
failure. Hungarian high-IoU matches6051/6052old/6053new only explain scale of
difference, never override strict failure. P28 uses fresh float32 FPN/native
outputs rather than treating the historical stream as an exact baseline.

P28: continuation_20261005/p28_causal_fgfa. Established causal FGFA+frozen RAFT
FPN migration, not novel. FIT240groups/960images, lags0/1/4/8; fresh feature
export and full readback PASS, flow240groups/720calls complete. Calibration
plan40groups/160images; fit15/cal5 are held out only for NEW module, baseline
detector already trained on full MDMT train. No dev/official-val/test inference.

TRAIN_PROTOCOL.json adds pre-fit uniform control while preserving immutable
CONFIG/GROUPS/FRAMES/PLAN_INPUTS export files. Three trained arms: single_adapter,
uniform_fgfa,causal_fgfa; same1200steps/5epochs pair-balanced schedule, AdamW1e-4,
all original current-frame fit GT boxes, original AutoAssign loss. Backbone/FPN/
head/RAFT frozen; value adapter identity init, FGFA additionally learned embedding.
16CPU operator tests and real GPU smoke pass, including exact current-only
identity detections, finite actual loss, value/embedding gradients and unchanged
frozen detector state. Training runs in tmux mdmt_p28_train on physicalGPU3 A100
UUID GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a; NEVER GPU2. Check train.exit and
training/TRAIN_RECEIPT.json before treating training as complete.

After all fixed final checkpoints: run_calibration.sh exports fresh features/flow
and freezes four prediction arms before opening calibration XML; independent
score_detection.py then reads AP50/default native-output TP/FP/FN and missed
recovery. No thresholds or checkpoint selection. FGFA detector pilot metrics
are not MDA/MOT results. Broader goal remains active; P25 remains deferred.
