# P22: dense motion input and competing explanation

**Decision: stop the current high-dimensional motion-field encoder hypothesis as a justified next model. Preserve the dense input and spatial-support evidence; no novel or effective MDMT method has been established.** Threshold selection is not the research contribution, and no new threshold sweep was performed.

## What actually completed

Frozen official torchvision RAFT-large ran on native-resolution MDMT images from fit pairs 23/25/29/69/78, two anchors per pair, three adjacent frames and both views: 60 images, 20 extraction tasks, 80 base directional calls and 10 image-intervention calls. Both `runner.exit` and `launcher.exit` are 0. Recorded model time was 83.598 s and peak CUDA allocation was 14,104,189,440 bytes on A100 UUID `GPU-1b297aba-ae7e-e326-5903-476f1bb683d7`. GPU2 was not used. The official pretrained checkpoint includes external optical-flow training; this is not MDMT-only self-supervision.

There are 1,356 object-region observations. They are not independent identities or independent experimental replicates. Images were bound to frozen stream `image_stem`, including pair78's offset. No MDMT identity labels, XML, calibration/dev or official val/test were read in P22. These are input diagnostics, not CA-IDF1/MDA/MOTA results.

## Implementation look-back and correction

The independent review found a real mask bug: earlier-edge visibility was not intersected with current-edge visibility, and the average of both masks was reported as two-step support. Original frozen artifacts remain unchanged. `../p22_dense_correction/` now contains 20 materialized corrected support archives, an input-bound receipt, and passing regression checks. The tests also repair the original synthetic foreground test's inconsistent forward flow and check an exact projective temporal endpoint pullback.

| Pair | Regions | Correct joint support >= 0.5 | Joint-pixel variation above original background diagnostic |
|---|---:|---:|---:|
| 23 | 102 | 99 | 17 |
| 25 | 207 | 199 | 19 |
| 29 | 342 | 338 | 36 |
| 69 | 317 | 298 | 53 |
| 78 | 388 | 386 | 142 |

The original quantity criterion survives in 3/5 pairs, rather than the misleading 5/5 reading. ROI spatial standard deviation and held-out background p90 are different statistics; their comparison is an uncalibrated screening diagnostic, not a signal-to-noise test or evidence of identity information. All temporal analyses use the corrected joint mask and pull the previous residual from t-2 into t-1 coordinates before comparison.

## What spatial and temporal evidence survives

Observed-interval RGB reconstruction compares the same visible pixels under background homography, one robust vector, an affine field, full dense residual, and spatially shuffled residual. The full field reduces mean error over the robust vector in all five pairs: the mean of pairwise relative reductions is **3.84% over complete boxes and 9.60% in their inner half**. Median per-object gains are near zero; only 48–59% of full-box observations improve. RAFT sees both reconstruction images, so this is not held-out prediction or optical-flow ground truth.

The 267 regions satisfying the corrected original screening condition have repeatable spatial structure. Their median mean-removed temporal cosine is 0.9597 and affine-removed cosine is 0.9587. However, current flow supplies the backtrace, neighboring intervals share an image, and the strongest spatial holdout control fits an offset on part of the current field. These are trajectory-conditioned reconstruction diagnostics, not independent causal forecasting. The largest 10% of object improvements account for 94.07% of positive holdout improvement over the prior affine control. Bbox-edge mixtures dominate much of the signal.

## Decisive competing explanation: one direction plus spatial support

A translating vehicle mixed with static pavement can yield a complex spatial field even when its physical motion has one direction. Therefore a comparison only against a single vector or an affine function of pixel position is insufficient. `../p22_rigid_support/` tests two stronger, explicitly oracle controls:

- `r(x) = alpha(x) v`, with a principal direction, nonnegative pixel coefficients and zero background residual.
- `r(x) = b + alpha(x) v`, the least-squares affine line in **vector-value space**, with a free constant offset. This differs from an affine field in image coordinates. It preserves an arbitrary scalar spatial support pattern.

Pixel coefficients are projected from the observed RAFT field; neither control is a learned or deployable segmentation method. The free offset is not independently measured background. These controls test whether an additional vector degree of freedom is needed, not whether spatial support itself is uninformative.

In the 267 screened regions, the offset-line control explains a median **99.771%** of mean-removed vector-field energy; its 10th percentile is **98.735%**. Full-field structure is thus predominantly a scalar spatial pattern along one direction. In all 1,356 regions its median explained shape energy is 91.750%.

| Pair | One-vector RGB L1 | Full dense L1 | Zero-offset ray L1 | Offset-line L1 |
|---|---:|---:|---:|---:|
| 23 | 0.020799 | 0.019663 | 0.020683 | 0.019645 |
| 25 | 0.023036 | 0.022398 | 0.022299 | 0.022312 |
| 29 | 0.018153 | 0.017486 | 0.017483 | 0.017436 |
| 69 | 0.039695 | 0.038932 | 0.038830 | 0.038899 |
| 78 | 0.015675 | 0.014828 | 0.014792 | 0.014791 |

Values are object-mean RGB absolute error in [0,1], on identical support within this control experiment. The offset-line control matches or slightly improves the full field in every pair. The more physically restrictive zero-offset ray matches it in four pairs; **pair23 is an exception**, so zero-background rigid occupancy is not universally established. Pooled gain-retention percentages should not hide this exception. Projection can denoise flow, so a control retaining over 100% of the dense-vs-constant photometric gain is possible.

This result removes the present justification for a high-dimensional vector-dynamics encoder. It does **not** falsify motion information, spatial masks, temporal support learning, or all camera/object factorization methods. It also does not identify which mixture pixels belong to the target.

## Camera intervention and novelty remain limited

The old-image affine pullback formula is correct. Median intervention deltas are approximately 0.076–0.152 native pixels across the five sampled views, often as large as weak baseline residuals. Reflected pixels were not explicitly excluded, and altered fields/H/common masks were not archived. Thus no camera-invariance PASS is supported; the materialized mask correction does not fix this separate limitation.

`../p22_prior_art/AUDIT.md` documents six mechanism-level comparisons. SAME-MDMT's public abstract already overlaps generic camera compensation and spatial alignment; its full text remains unavailable. HomView-MOT covers homography-conditioned identity-feature learning. Structured Dynamics, Flow3r and Flow Equivariant World Models cover adjacent decomposition, reconstruction, factor swapping and equivariance. Those Tier-2 precedents narrow claims rather than alone falsifying an MDMT candidate. No novelty/first claim follows from the present audit.

## Progression decision

1. **STOP this model expansion:** a high-capacity dense-vector encoder justified solely by high spatial standard deviation, non-affine fields or dense-vs-one-vector reconstruction. The rigid-support control explains those observations sufficiently well to require a different rationale.
2. **KEEP the corrected input substrate and limited spatial-support evidence.** A future representation candidate must specify what identity-relevant observation it recovers, and beat a motion-segmentation/support control and an ordinary appearance representation. Merely training an alpha mask, adding a renderer or assigning a new method name does not supply novelty.
3. **HOLD method declaration and cross-view training on this premise.** Real cross-view support transfer and identity complementarity have not been measured. No failed input hypothesis authorizes threshold sweeps, a dataset pivot or official validation access.

The overall research goal remains active. This phase produces a corrected input experiment and a concrete falsification of its strongest proposed interpretation; it does not claim that a technically substantive method has already been found.
