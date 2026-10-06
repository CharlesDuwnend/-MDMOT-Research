# P22 independent implementation review

Reviewed the frozen P22 extractor, protocol, receipt, all 20 NPZ/JSON pairs, synthetic test source, and the 10 allowlisted fit-stream image mappings. No extraction, GPU job, training, label read, or frozen-file modification was performed by this review.

**Decision: keep the artifacts as a usable-input engineering observation; correct the temporal support semantics before a learning gate. The artifacts do not establish identity information, camera invariance, a learned module, novelty, or MOT improvement.** The user's rejection of threshold-only methods remains applicable: this gate is a diagnostic, not the proposed method.

## Verified facts

- `runner.exit` and `launcher.exit` are both 0. All 45 files bound by `FEATURE_RECEIPT.json` match their hashes. The RAFT weight and recorded torchvision RAFT source hashes also match.
- All 20 archives have the declared `[objects, 2, 16, 16, 2]` residual fields and matching boolean masks/observation keys; arrays are finite. There are 1,356 object observations, not 1,356 independent identities.
- RAFT is frozen, in evaluation/inference mode. P2 inputs are opened with `with_labels=False`; the features use only each anchor and its preceding two stream frames. The code makes no cross-view homography or physical synchronization assumption.
- Every saved image reference agrees with the original frozen fit-stream `image_stem`, including pair78. For example, pair78 stream frames 114/115/116 map to image stems 216/217/218. Direct frame-number filenames would be wrong; the present mapping is correct.

## Actual semantic bug: the two-step mask is not a path-valid mask

`extract_dense.py:198-208` transports the current ROI grid to the preceding image using current backward flow, then samples the earlier visibility mask. It never intersects that earlier mask with visibility of the current-to-previous edge. Thus an invalid current backtrace can still supply a purportedly valid historical sample. The receipt gate subsequently uses the **average** of the two masks (`valid.mean()`), which is not common two-step support.

The saved masks contain 8,004 historical-valid/current-invalid grid samples. No rerun is needed to demonstrate the counting error. The following is a deterministic artifact-only correction: use `valid[0] & valid[1]` for path support and compute the current spatial statistic on those same common pixels. It is not a new parameter search.

| Fit pair | Original average-support objects | Common-support objects | Original variation count | Common support, common-pixel variation count |
|---|---:|---:|---:|---:|
| 23 | 102 | 99 | 20 | 17 |
| 25 | 204 | 199 | 25 | 19 |
| 29 | 342 | 338 | 39 | 36 |
| 69 | 311 | 298 | 66 | 53 |
| 78 | 387 | 386 | 143 | 142 |

The original numeric requirement of at least 20 such observations in at least three pairs still holds for pairs 29/69/78 after this correction. The stronger reading that all five pairs pass with valid two-step spatial evidence does not survive.

Before downstream temporal modeling, preserve the two original marginal masks but supply an explicit composed-path mask. An earlier descriptor sampled through an invalid current flow edge must not be treated as valid temporal evidence.

## Spatial standard deviation versus background p90 is an uncalibrated diagnostic

The units are compatible: both describe pixel vectors in the preceding native-resolution image. The quantities nevertheless answer different questions. ROI standard deviation measures spatial dispersion after subtracting its own mean; held-out background p90 measures the norm of flow-minus-homography errors. The latter is not a matched null distribution for the former.

Large ROI dispersion can come from foreground/background mixtures, depth parallax, boundary interpolation, nonrigid motion, or RAFT errors. A coherent translating object's useful residual can have nearly zero spatial standard deviation. The current comparison therefore proves neither identity information nor information beyond a pooled vector. It should not be used to reject coherent motion fields either.

The minimum missing controls for that later claim are a matched background-ROI dispersion control, a within-ROI spatial shuffle that preserves the vector distribution, and a pooled-vector counterpart using identical observations and masks. None is present in this extraction, and this review does not add or authorize their execution.

## The affine camera pullback is mathematically correct, but invariance is unproven

`extract_dense.py:181-194` keeps the current image unchanged and transforms only the preceding image with `A(p)=Lp+b`. For an exact corresponding background map, the altered residual is `L r`; the row-vector expression `altered_residual @ inv(L).T` recovers `r` exactly. Translation cancels. No spatial rewarp of the current-indexed field is needed. This is an exact affine endpoint-vector pullback, not a mistaken Jacobian approximation.

However:

1. `BORDER_REFLECT` invents old-image content. The code checks only whether the altered destination is inside the altered image, not whether its inverse-affine location lies in real original-image support. Reflected pixels can enter the forward/backward and background-fit supports.
2. The altered homography is re-estimated and then discarded. The altered residual field and common mask are also not archived; only object-level median deltas survive. The exact real-image intervention cannot be independently reconstructed from the frozen artifacts without another flow run.
3. There is no matched interpolation/no-op-warp control, and the intervention changes fit support. Its delta includes RAFT response, interpolation, masking, and background-refit changes; it is not a pure measurement of camera nuisance removal.

Existing median/p90 object-level intervention deltas are 0.094/0.477 px (23), 0.076/0.226 (25), 0.152/0.426 (29), 0.089/0.286 (69), and 0.093/0.368 (78). For pair78, 80/91 objects with defined deltas exceed their reported baseline background p90, and the median baseline residual magnitude is only 0.018 px. These are descriptive conditional statistics, not a passed invariance gate.

## Further limits and test gap

- The two sampled residual vectors are expressed in different endpoint image coordinates: the current residual is in frame `t-1`, the earlier residual in frame `t-2`. Sampling the latter at the tracked location aligns the grid index, not its vector basis. This is acceptable as explicitly separate time channels. A claim of a common temporal vector field requires an explicit basis transport or a model that consumes the corresponding transforms; direct subtraction is not justified.
- `test_contract.py`'s foreground-displacement test changes backward flow by `[4,-3]` but leaves forward flow unchanged and discards the returned validity mask. Its foreground forward/backward error is 5 px against an allowed approximately 1.141 px, so the recovered residual is actually invalid under the pipeline's own visibility rule. The test verifies stored arithmetic, not visible foreground recovery. A consistent forward-flow counterpart and an assertion on foreground validity are needed for that claim.
- Background H is always returned after any successful OpenCV fit; held-out p90 is reported but does not enforce geometric adequacy. This is acceptable for an observation audit, not a validated physical camera model. Predicted-box masks also do not certify that all retained pixels are background.
- Imported `p21_residual_motion/probe.py` and `p2/src/data.py` are not directly bound by the P22 receipt; the recorded RAFT source hash is not checked in `--run`. Current source inspection found no label leakage, and the recorded RAFT source matches now. These dependency hashes should be included in a subsequent reproducibility contract.

## Claims that survive

Frozen RAFT and same-view background fitting can produce finite object-centered dense residual tensors on all five fixed MDMT fit pairs, without current identity labels or cross-view geometry. Common-path support and spatial variation under the existing diagnostic remain available on at least three pairs. This justifies preserving the input substrate and addressing its mask semantics. It does not select or validate a neural architecture, establish additional cross-view identity evidence, recover true object motion, or create a publishable method.
