# Frozen nine-arm contract audit

Read-only audit snapshot: 2026-10-08 CST (2026-10-07 17:36 UTC). Only this review document was written. No GPU was accessed and no production source, manifest, receipt or model was changed.

**Outcome:** the nine smoke cells now close their actual input/association contracts. One guaranteed final-analysis schema failure exists in the frozen scripts: E00/E01 omit the arm-level `bundle_sha256` field. Their model files are nevertheless sealed in `frozen_files_sha256`, and runner/sealer do not read the missing field. This affects postprocessing, not the live experiment. No other necessary schema failure was found after checking the fields and executing the read-only recipe/history validators. Full nine-arm MOT completion is not established; final receipt/launcher exit are absent at this snapshot, and C11 full inference has begun.

## Frozen artifacts and independently checked scope

Manifest: `071e1bfa2530ef9b7f79884c3264f5969b19db2f842903732e509d7fa81c965b`, 210 sealed files. Every sealed file exists and its current SHA-256 matches. All nine cells use the declared frozen detector/ReID source and 17-sequence/6635-frame full protocol; the smoke subset uses `uav0000073_00600_v`, frames 1–200.

The reference CPU receipt is `PASS_NINE_CORRECTED_REFERENCE_CPU_CONTRACTS`, nine actual checks in three rounds. Its sixteen absolute-path source/payload/test hashes were rehashed. Real F00/summary estimators and STrack objects were used without oracle, matching or estimator mocks. Two whole immutable associate implementations, tuple/pending/stage equality, empty inputs, class/score-distinct equal boxes, state isolation, committed semantic updates and injected mismatch rejection were checked. Shared v2 corrected `CaptureTracker.det_record` is declared; called feature functions are AST-equal to the original, while the feature file has only an unused `source_vectors` counter extension. The source head is not invoked by association under the sealed clutter/edge-off restrictions.

Smoke aggregate receipt: `7e6fff91b2c5383cabfdacde1db15451e842cfa06a7bae22179ecc45a4dd802c`. Actual smoke exit is `0`. Identity gate: `ca08836e20a00b88c3a19b5d1685d7a161a9c7cf626bd504c1d4aac328ec44f4`, status `RESOLVED_DETECTION_METADATA_IDENTITY`, sealed to this manifest; all 28 evidence hashes were independently rechecked. No final full-run PASS flag was inferred from these artifacts.

## Actual completed smoke coverage

| Cell | Frames | Executed seams | Actual native calls | Legal/native seam checks | Whole-source tuple/pending checks | Fusion / teacher |
|---|---:|---:|---:|---:|---:|---|
| C11 full | 200 | 600 | 600 | 600 | 600 | on / loaded |
| C00 native-cost | 200 | 600 | 600 | 600 | 600 | on / loaded |
| C10 mask-only | 200 | 600 | 600 | 600 | 0 (declared) | on / loaded |
| C01 penalty-only | 200 | 600 | 600 | 600 | 0 (declared) | on / loaded |
| E10 fusion-on/selective-off | 200 | 600 | 600 | 600 | 600 | on / loaded |
| Paired hierarchical | 200 | 600 | 600 | 600 | 600 | on / loaded |
| Paired flat | 200 | 600 | 600 | 600 | 600 | on / loaded |
| E00 fusion-off/selective-off | 200 | 600 | 600 | 600 | 600 | off / unloaded |
| E01 fusion-off/selective-on | 200 | 600 | 600 | 600 | 600 | off / unloaded |

All nine actual prediction files were rehashed against their receipts. All 200 input records per cell passed the frozen ledger validator and are equal across cells; shared stream hash is `80a872cdb7b8c103eea76df0d52276ded3ef54adaa081554fe9f3d1e7a61dd38`. Frame 186 exactly reproduces detector digest `15d69e475607041a541b2d1245bcc7ebb722993f2bf19dbadbeffa824dafa585`, the confirmed equal-box problem frame. Inline GT-guard evidence is PASS with zero blocked attempts in every completed smoke.

E00/E01 keep source/SRC inference active. E00 has 306 source-probability events and zero fusion/selective calls; E01 has 168 source events, zero fusion calls, 85 native keeps and 83 native rejections. These are execution observations, not MOT gains. Closed-loop source/association event counts may differ across policies despite identical detector/ReID inputs.

## Contract chain and comparison boundaries

1. `reference_oracle.py:48` runs after real binding and prior pending flush; it clones the whole `(trks,dets)` graph, preserving aliases, remaps current detection IDs, copies observation/responsibility estimators and mutable semantic state, and uses the original frozen receiver class. C11/other `fc_full` cells use the old main runtime; C00 uses old internal-path `path_update_only`, whose assignment is actual host `super().associate`. The old bbox binder is skipped by verified prebinding.
2. `reference_oracle.py:100` compares the full ordered returned tuple, pending count/track identity/observation/ratio/responsibility, and stage. Every successful check increments one seam counter. It is a same-state association reference: it does not independently replay all prior birth/memory history. CPU committed-update checks and byte-equal semantic-update source cover that separate scope; they are not an end-to-end corrected gold predictor.
3. `run_mechanism.py:105` checks actual flags, state updates, source activity, fusion/teacher routing, disabled-policy call counts, input ledger coverage and oracle coverage. Every arm rechecks the 210 frozen hashes, actual sequence/frame completion, physical GPU UUID, predictions, scoring and GT metadata. Old prediction comparisons are diagnostic only and cannot reject corrected outputs. Other full arms must match the new C11 input digests before scoring.
4. `seal_corrected_smoke.py:20` keeps dummy-head metadata tests separate from the real-head reference CPU tests. It rehashes CPU sources, nine real smoke predictions/reports/receipts, actual frame-186 detector input, ledger equality and exit 0, then seals an identity gate. Its native/legal count test is weaker than final analysis (`>0` rather than equality), but the actual nine smoke reports independently satisfy 600/600 equality.
5. `analyze_completed.py:160` requires all nine fresh arms, all 17 sequences/6635 frames, sealed identity/source/model/GT/scorer evidence, success exit, no stop marker, matching local/global receipts, ordered equal input ledgers and per-sequence native/legal counts equal to **all** executed seams. Seven `fc_full`/`fc_control` cells must verify whole-source pending/tuple parity on every seam; C10/C01 must claim zero oracle checks. Fusion-off cells must have no teacher/fusion, and selective-off cells must have zero selective calls.
6. `analyze_mechanisms.py:178` validates current completion, paired recipe and retained history before writing scientific tables. Its ten signed corrected contrasts and both interactions contain only new C/E/H cells. Gamma is held fixed: the policy factorial is fusion×selective. Twenty old rows remain `historical_metadata_contaminated`, `scientific_use=false`, outside corrected estimates. Actual read-only `verify_history` rehashed all twenty and preserved hard/soft and update-only exact nulls plus the very small cost-path difference (−1 FP), rather than rounding it into a null or dropping it.

The paired-recipe validator was executed read-only on the current copied models: common scaler payload hash `026e1e343cfd2e4164490acb9e4a9510ea8675083e1adc3d836df83b215fc4e0`, common event-weight hash `d6af4359f85caeb4ab704bd0b33e08f173c4d9fd1e1d2e25bc49d02dae482c56`, fit 8024 and foreground coverage 6093, fixed estimator recipe, no calibration/test-dev fitting. This supports the paired parameterization comparison; it does not prove equal capacity/L2 geometry or corrected historical fitting metadata.

## Guaranteed failure and compatible postprocessing correction

Frozen `analyze_completed.py:221` evaluates `arm['bundle_sha256']`; E00/E01 do not have that key. This will raise `KeyError` after their full cells complete. The corresponding exact sealed model entries exist and were independently matched to disk:

- E00 `models/FusionOff_NoSelective.pkl`: `bbd97ac63539083accaf655d93cd3dd3ca442330af89b411ecfc0162d0f8cd21`.
- E01 `models/FusionOff_Selective.pkl`: `4165da0896b5246d7a0c494b17c38625fbbb8e55fdee10c6fb0c848e3887ddd9`.

Root plans an immutable compatibility copy under `review/final_analysis`. Use arm-level hash when present and otherwise the exact absolute model entry in this frozen manifest, then require both arm receipt hashes and an actual rehash to equal that value. Set the copied modules' `ROOT` explicitly to this experimental root, preserving the old frozen analysis files and all experiment settings. Record original frozen analysis hashes and executed compatibility-module hashes distinctly; `analyze_mechanisms.py:211` currently hashes the root's original primary module, so an overlay must also identify the actually imported compatible primary module.

Schema review found no other missing required current manifest fields. All literal receipt/runtime fields requested by final validation exist in the nine actual smoke reports; future full records add the two metric fields via runner `record.update`. The frozen scorer produces the required sequence/OVERALL/GT/result schema. Read-only paired-recipe and historical validators completed successfully on actual artifacts, and the ten contrast names match the interaction definitions. No scientific completion report was produced during this audit.

## Remaining scope limits

Full completion and per-sequence activity gates are empirical and still pending; smoke cannot establish all other clips' valid execution or any tracking score. Runtime snapshots protect selected semantic fields; immutable flags and CPU tests additionally demonstrate isolation of exposed real track/model states, while unrelated removed history is intentionally not fully cloned. The final analysis records runtime-file hashes after reading but runner receipts do not seal producer runtime/guard file hashes separately; current checks close totals/input ledgers, not every untouched diagnostic byte. GT guarding has its stated Python-open/designated-GT-root scope. These limits should remain explicit rather than turn a smoke/CPU PASS into formal mechanism effectiveness.

Frozen analysis/source hashes at review: completed `cddb4d9f1fd9b4aa67ce2d1723fc0a312590f7f6b83787a67110f3dcf6b7ba06`; mechanisms `e6c1542ffcf2c27125037d46904676cc9dc92b9e7730997d88e9f6e8baf7540e`; runner `95d63a0a154382cc1d50f2e9c6211f237239b8c0da05bf1b0fc8eccde90739ac`; sealer `32005e8485db9e0911395ba9418cf64a058feacd6c268adea99fa6b6d47f8405`.
