**P28 FIT-only cached RAFT motion lookback — complete.**

The cached current-to-past RAFT fields contain useful geometric motion on the fixed FIT cohort: center prediction is better than zero flow at lags 1, 4 and 8. Source direction, per-axis resize, halfpixel coordinates, border sampling and saved-result checks pass. This supports further implementation investigation; it does not explain the failed FGFA pilot or validate a new method.

**Scope and cohort.** The diagnostic reads only FIT15 XML and the existing 240 FIT groups, each with cached fields at lags 1/4/8. No images, GPU, model loading, new inference or non-FIT GT. No P28 model, feature, flow, prediction, trainer, P26 or reference-workspace artifact is changed. Identity is `(pair, view, raw_local_id, class)`; the same supported class/local track must be annotated at both endpoints and every intermediate frame. This is a same-view annotation correspondence, not a cross-view/global identity assumption.

FIT pairs: 23, 25, 28, 29, 39, 44, 45, 51, 53, 63, 66, 69, 70, 74, 78. The 240 current frames contain 13,663 supported boxes; the three lags produce 40,420 comparisons. Occluded annotations are retained. The parser, manifest and FIT XML hashes match the pinned training inputs.

| Lag | Eligible comparisons | Past endpoint absent | Interior gap |
|---:|---:|---:|---:|
| 1 | 13,579 | 84 | 0 |
| 4 | 13,468 | 195 | 0 |
| 8 | 13,373 | 290 | 0 |

**Geometry contract.** Export calls `RAFT(current, past)`; torchvision correlates image1 against image2 and returns `coords1 - coords0`, in `(dx,dy)` order. Every flow archive is `[3,2,360,640]` for original images of `1080×1920`, with no padding. For an original continuous-edge box center `c`, the flow raster pixel index is `c*(Fw/W,Fh/H) - 0.5`. The predicted past center is `c + sampled_flow*(W/Fw,H/Fh)`, using bilinear sampling, border extension and `align_corners=False`.

The known-translation test supplies a constant synthetic flow field and verifies projection sign and resize scale; it does not run RAFT on synthetic images. Zero flow, border extension, spatial ramps, an independent CPU `grid_sample` comparison and the gap/class guard also pass. At all real GT sample locations, NumPy64 versus Torch32 sampling differs by at most 0.000837326 original pixels per component (fixed numerical gate: ≤0.01 px). GT continuous-edge coordinates must remain distinct from feature/pixel-center indices. These checks do not establish feature-content alignment.

**Observed center errors in original pixels.** Comparisons are annotation weighted. “Win” means strictly smaller error than zero flow for the same observation; it is not an AP/recall metric.

| Lag | n | RAFT median | Zero median | RAFT mean | Zero mean | RAFT p90 | Zero p90 | Win |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13,579 | 0.816 | 2.500 | 1.269 | 3.555 | 2.713 | 7.280 | 78.96% |
| 4 | 13,468 | 1.595 | 10.012 | 3.623 | 14.003 | 8.253 | 29.155 | 86.31% |
| 8 | 13,373 | 2.253 | 20.125 | 6.816 | 27.761 | 16.227 | 58.516 | 88.66% |

**Two separate size normalizations.** Each error is divided by the square root of current box area or past box area, respectively. The denominators are never pooled or substituted.

| Lag | Denominator | RAFT median | Zero median | RAFT p90 | Zero p90 |
|---:|---|---:|---:|---:|---:|
| 1 | current sqrt(area) | 0.0251 | 0.0774 | 0.1085 | 0.2588 |
| 1 | past sqrt(area) | 0.0251 | 0.0773 | 0.1081 | 0.2590 |
| 4 | current sqrt(area) | 0.0500 | 0.2887 | 0.3596 | 1.0253 |
| 4 | past sqrt(area) | 0.0500 | 0.2885 | 0.3562 | 1.0268 |
| 8 | current sqrt(area) | 0.0678 | 0.5742 | 0.7276 | 2.0335 |
| 8 | past sqrt(area) | 0.0682 | 0.5774 | 0.7203 | 2.0376 |

**Size strata, defined on the current box.** Min-side <16 and sqrt(area) <16 are different cohorts. Normalized columns below use current sqrt(area); the CSV also contains past normalization, mean errors and all complements/intersections.

| Criterion | Lag | n | RAFT median px | RAFT norm median | RAFT norm p90 | Zero norm median | Win |
|---|---:|---:|---:|---:|---:|---:|---:|
| min-side <16 | 1 | 3,661 | 0.730 | 0.0509 | 0.1407 | 0.1091 | 71.21% |
| min-side <16 | 4 | 3,607 | 1.506 | 0.1037 | 0.4843 | 0.4147 | 79.54% |
| min-side <16 | 8 | 3,578 | 2.259 | 0.1549 | 0.9393 | 0.8324 | 83.12% |
| min-side ≥16 | 1 | 9,918 | 0.850 | 0.0196 | 0.0908 | 0.0685 | 81.82% |
| min-side ≥16 | 4 | 9,861 | 1.634 | 0.0380 | 0.2868 | 0.2644 | 88.78% |
| min-side ≥16 | 8 | 9,795 | 2.252 | 0.0514 | 0.5918 | 0.5221 | 90.68% |
| sqrt(area) <16 | 1 | 2,207 | 0.687 | 0.0565 | 0.1416 | 0.1026 | 70.14% |
| sqrt(area) <16 | 4 | 2,170 | 1.392 | 0.1107 | 0.4934 | 0.3924 | 79.08% |
| sqrt(area) <16 | 8 | 2,155 | 2.131 | 0.1649 | 0.9397 | 0.7845 | 83.02% |
| sqrt(area) ≥16 | 1 | 11,372 | 0.844 | 0.0216 | 0.0987 | 0.0721 | 80.67% |
| sqrt(area) ≥16 | 4 | 11,298 | 1.645 | 0.0425 | 0.3236 | 0.2782 | 87.70% |
| sqrt(area) ≥16 | 8 | 11,218 | 2.275 | 0.0575 | 0.6638 | 0.5490 | 89.74% |

**Current occlusion strata.** The annotation flag is descriptive. The occluded and unoccluded cohorts differ in size, scene and motion; these aggregates do not establish an occlusion effect. Past-endpoint and any-path occlusion plus size×occlusion intersections are in `STRATA.csv`.

| Current occluded | Lag | n | RAFT median px | Zero median px | RAFT current-area norm p90 | Win |
|---|---:|---:|---:|---:|---:|---:|
| no | 1 | 10,764 | 0.846 | 2.828 | 0.1123 | 79.35% |
| no | 4 | 10,739 | 1.637 | 10.440 | 0.3792 | 86.67% |
| no | 8 | 10,704 | 2.310 | 21.095 | 0.7915 | 88.91% |
| yes | 1 | 2,815 | 0.735 | 2.062 | 0.0930 | 77.48% |
| yes | 4 | 2,729 | 1.488 | 8.062 | 0.2638 | 84.87% |
| yes | 8 | 2,669 | 2.052 | 16.378 | 0.4397 | 87.64% |

**Pair heterogeneity and domain flags.** All 15 FIT pairs have lower mean and median RAFT error than zero flow at each lag. This does not imply that every observation benefits: pair 78 at lag 1 has only 42.33% wins and a mean error reduction of 0.0348 px. At lags 4/8 all 15 pairs have win fraction >50%. Pair/lag and class tables are retained in the CSV.

Every current GT center is inside the image. One past GT center at lag 4 is outside; it is retained and flagged. RAFT endpoints outside the image number 0/22/54 at lags 1/4/8; they are also retained and flagged. No outlier or error-based exclusion is applied.

**Bounded interpretation.** There is no evidence here for a gross exporter direction, original-pixel scale or halfpixel sampling error. The useful FIT geometric signal does not support a blanket STOP for temporal motion. However, long-lag tails remain material: for both small-object definitions, lag-8 normalized p90 is about 0.94 current sqrt(area). A GT bounding-box center is not necessarily a tracked physical surface point under deformation or occlusion. The analysis does not verify FPN feature lattice/content, resampling blur, correspondence under occlusion, learned weighting or optimization, nor causally explain the failed calibration pilot. It provides no new detection/tracking performance, validation selection, threshold/window recommendation or authorization to alter the trained system.

**Verification and artifacts.** The producer completed with exit 0 in 23.754 s. Independent saved-row verification completed with exit 0 in 1.310 s: 40,420 rows, 240 groups and all 112 published strata agree; all 286 bound input hashes remain unchanged. Maximum independently recomputed pixel-error difference is 5.68e-14 px. This second check reconstructs result arithmetic and count conservation; annotation eligibility is enforced by the producer and its contract test, not re-parsed by the saved-row verifier.

- `PROTOCOL.md`, `audit_motion.py`: scope, eligibility and producer.
- `DIRECTION_AUDIT.json`, `TESTS.json`: source contract and synthetic/numerical evidence.
- `observations.npz`, `SUMMARY.json`, `STRATA.csv`: complete rows and all descriptive strata.
- `verify_saved.py`, `VERIFY.json`, `verify.log`: independent saved-row checks.
- `INPUTS.json`, `run.log`, `EXECUTION.json`, `RECEIPT.json`, `SHA256SUMS`: lineage and execution seal.
