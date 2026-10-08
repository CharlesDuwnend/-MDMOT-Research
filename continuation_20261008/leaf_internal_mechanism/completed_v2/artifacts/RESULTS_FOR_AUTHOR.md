# Corrected online internal mechanism results

All nine cells use fresh corrected-metadata inference over the same 17 sequences / 6635 frames. Frozen source/model hashes, actual prediction and GT hashes, the scorer, ordered detector/ReID input ledgers and seam contracts were verified. The sealed identity gate is resolved. Historical prediction-file parity is not a completion gate.

| Arm | MOTA | IDF1 | FP | FN | IDs | FM |
|---|---:|---:|---:|---:|---:|---:|
| C11_full_reference | 58.0251 | 71.3737 | 26047 | 69497 | 791 | 2357 |
| C00_nativecost_reference | 57.8177 | 71.1857 | 26252 | 69759 | 800 | 2368 |
| C10_mask_only | 57.8800 | 71.2224 | 26190 | 69678 | 800 | 2358 |
| C01_penalty_only | 57.9802 | 71.3441 | 26060 | 69587 | 791 | 2363 |
| E10_fusion_no_selective | 57.8242 | 71.2605 | 23944 | 72198 | 654 | 2178 |
| H_paired_hierarchical | 57.9445 | 71.3416 | 26452 | 69258 | 810 | 2383 |
| H_paired_flat | 58.0177 | 71.3931 | 25963 | 69610 | 779 | 2361 |
| E00_fusion_off_no_selective | 57.6294 | 70.9271 | 21723 | 74935 | 585 | 2074 |
| E01_fusion_off_selective | 57.9553 | 71.3309 | 25865 | 69865 | 765 | 2344 |

| Conditional contrast | MOTA delta (pp) | IDF1 delta (pp) | FP delta | FN delta | IDs delta | FM delta |
|---|---:|---:|---:|---:|---:|---:|
| mask_without_penalty | +0.0623 | +0.0367 | -62 | -81 | +0 | -10 |
| penalty_without_mask | +0.1625 | +0.1584 | -192 | -172 | -9 | -5 |
| penalty_with_mask | +0.1451 | +0.1513 | -143 | -181 | -9 | -1 |
| mask_with_penalty | +0.0449 | +0.0296 | -13 | -90 | +0 | -6 |
| SRC_joint_cost_operators | +0.2074 | +0.1880 | -205 | -262 | -9 | -11 |
| selective_with_fusion | +0.2009 | +0.1133 | +2103 | -2701 | +137 | +179 |
| fusion_without_selective | +0.1948 | +0.3334 | +2221 | -2737 | +69 | +104 |
| fusion_with_selective | +0.0697 | +0.0428 | +182 | -368 | +26 | +13 |
| selective_without_fusion | +0.3259 | +0.4039 | +4142 | -5070 | +180 | +270 |
| paired_hierarchy_minus_flat | -0.0732 | -0.0515 | +489 | -352 | +31 | +22 |

Every corrected contrast uses the nine new cells only. Zero and negative values are retained. C11/C00 and the other fc_full cells verify exact association tuples and pending responsibilities against the frozen same-state corrected-seam oracle; C10/C01 verify actual native seams and candidate legality.

The hierarchy/flat contrast uses the declared paired fitting scaler and event weights, with frozen heads. It is separate from F00's original fitting recipe; shared C does not equate L2 parameter geometry. No refitting for the metadata repair or test-dev winner selection is implied. Activity counts are diagnostics, not tracking accuracy.

## SRC mask × learned penalty

Every cell uses corrected metadata and frozen models; C00 retains the system's initialization/update and EGIA while using native assignment cost.

| Cell | MOTA | IDF1 | FP | FN | IDs | FM |
|---|---:|---:|---:|---:|---:|---:|
| C00_nativecost_reference | 57.8177 | 71.1857 | 26252 | 69759 | 800 | 2368 |
| C10_mask_only | 57.8800 | 71.2224 | 26190 | 69678 | 800 | 2358 |
| C01_penalty_only | 57.9802 | 71.3441 | 26060 | 69587 | 791 | 2363 |
| C11_full_reference | 58.0251 | 71.3737 | 26047 | 69497 | 791 | 2357 |

## Fusion × selective policy

All four cells are newly inferred with corrected metadata. Coherence and all fitted weights are fixed; E00/E01 reuse frozen A04/A05 weights, not their historical scores.

| Cell | MOTA | IDF1 | FP | FN | IDs | FM |
|---|---:|---:|---:|---:|---:|---:|
| E00_fusion_off_no_selective | 57.6294 | 70.9271 | 21723 | 74935 | 585 | 2074 |
| E01_fusion_off_selective | 57.9553 | 71.3309 | 25865 | 69865 | 765 | 2344 |
| E10_fusion_no_selective | 57.8242 | 71.2605 | 23944 | 72198 | 654 | 2178 |
| C11_full_reference | 58.0251 | 71.3737 | 26047 | 69497 | 791 | 2357 |

## Historical archive — scientific_use=false

The twenty historical score rows are rehashed and retained in retained_historical_controls.csv as historical_metadata_contaminated. None enters the corrected tables or contrasts. Their hard-update/recurrence null observations remain recorded below and do not establish corrected null effects.

| Historical observation | MOTA delta (pp) | IDF1 delta (pp) | Identical files | Scientific use |
|---|---:|---:|---:|---|
| responsibility_soft_minus_hard | +0.0000 | +0.0000 | 17/17 | false |
| recurrence_with_SRC_cost | +0.0004 | +0.0002 | 10/17 | false |
| recurrence_without_SRC_cost | +0.0000 | +0.0000 | 17/17 | false |

Factorial interactions are descriptive differences of conditional effects, with no significance claim. The paired hierarchy/flat comparison retains its declared fitting recipe; canonical F00 is not an isolated hierarchy-versus-flat fitting control. The frozen-weight metadata repair leaves historical fitting lineage unchanged.
