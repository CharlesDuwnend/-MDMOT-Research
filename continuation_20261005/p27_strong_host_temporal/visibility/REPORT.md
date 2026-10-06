# P27 FIT temporal observation evidence

**PASS diagnostic execution; old-stream Oracle evidence only.**
15 FIT pairs, 30 sequences, 15,572 full stream frames; 868,562 supported annotated observations.
CURRENT detected 834,897 (96.124%); missing 33,665 (3.876%).

The detector streams apply score >= 0.1. This is not exact P26 detector evidence (native post-NMS default 0.05 reported separately). No inference, training, images, GPU, or non-FIT data used in the computation.

| Strictly prior horizon | Missing with same-GT past evidence | % of missing | % of all GT |
|---|---:|---:|---:|
| 1 | 10,616 | 31.534% | 1.222% |
| 4 | 20,183 | 59.952% | 2.324% |
| 8 | 23,930 | 71.083% | 2.755% |

These counts describe available historical observations with Oracle local GT correspondence. They do not establish recoverable detections, feature alignment quality, or expected FGFA/MDA gains. Annotation gaps reset every history.

| Miss category | Observations | % of missing |
|---|---:|---:|
| annotation_birth_no_prior_annotation | 824 | 2.448% |
| reappearance_after_annotation_gap | 48 | 0.143% |
| continuous_segment_never_detected_before | 3,816 | 11.335% |
| continuous_prior_detection_older_than_8 | 5,047 | 14.992% |
| continuous_prior_detection_within_8 | 23,930 | 71.083% |

| XML occlusion | GT observations | Missing | Missing with prior 8 |
|---|---:|---:|---:|
| 0 | 668,659 | 19,721 | 15,771 |
| 1 | 199,903 | 13,944 | 8,159 |

| Minimum GT side | GT observations | Missing | Missing with prior 8 |
|---|---:|---:|---:|
| <16 | 243,083 | 28,474 | 19,312 |
| 16 to <32 | 304,409 | 4,682 | 4,199 |
| >=32 | 321,070 | 509 | 419 |

| Pair | GT observations | Missing | Missing prior 1 | Missing prior 4 | Missing prior 8 |
|---|---:|---:|---:|---:|---:|
| 23 | 34,030 | 178 | 74 | 113 | 123 |
| 25 | 51,626 | 2,921 | 1,049 | 1,965 | 2,189 |
| 28 | 102,839 | 8,831 | 2,110 | 4,604 | 5,919 |
| 29 | 107,020 | 2,696 | 1,456 | 2,210 | 2,309 |
| 39 | 8,440 | 13 | 10 | 10 | 10 |
| 44 | 31,747 | 128 | 64 | 99 | 103 |
| 45 | 77,381 | 4,029 | 2,128 | 3,564 | 3,818 |
| 51 | 41,226 | 506 | 104 | 167 | 183 |
| 53 | 35,097 | 28 | 13 | 24 | 25 |
| 63 | 25,367 | 324 | 115 | 228 | 281 |
| 66 | 40,656 | 839 | 209 | 421 | 529 |
| 69 | 101,348 | 5,576 | 1,407 | 3,153 | 3,983 |
| 70 | 48,013 | 3,157 | 615 | 1,276 | 1,639 |
| 74 | 94,207 | 1,592 | 842 | 1,302 | 1,389 |
| 78 | 69,565 | 2,847 | 420 | 1,047 | 1,430 |

Cross-view appendix: 261,395 coannotated conventional matches; both-current-missing 271.
Equal raw ID and normalized class label define the cohort; physical identity and hardware synchronization are unverified.

| Prior horizon | Either-view evidence among both missing | Both-view evidence among both missing |
|---|---:|---:|
| 1 | 169 | 45 |
| 4 | 241 | 125 |
| 8 | 257 | 161 |

Retrospective miss runs: 11,488; longest 156 frames. Full lengths/endpoints are summary-only and never enter causal history flags.

| Full miss-run length | Runs | Missing observation mass |
|---|---:|---:|
| 1 | 6,144 | 6,144 |
| 2 to 4 | 3,768 | 9,753 |
| 5 to 8 | 928 | 5,605 |
| 9 to 16 | 421 | 4,818 |
| >=17 | 227 | 7,345 |

`rows/*.npz` retains every supported GT observation. `cases.csv` deterministically retains first misses per sequence/category/occlusion/size cell, not selected success imagery. `miss_runs/*.npz` and `cross_view/*.npz` are separate explicitly marked diagnostics.
`summary.json` retains class, occlusion × size, pair, sequence, and exact run-length counts. `VERIFY.json` records synthetic and independent saved-row verification; `RECEIPT.json` binds all inputs, code, and outputs.
