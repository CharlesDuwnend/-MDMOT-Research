# Corrected metadata: independent online reference and analysis review

Status: design/read-only audit, 2026-10-08. No GPU run or inference source modification was performed for this review. The parent stopped v1 after detecting bbox-only binding to wrong class/score metadata. Corrected v2 results are not assumed complete, positive, or valid. Byte identity to old predictions cannot certify corrected semantics.

## 1. Reference methods that actually implement the intended controls

| Actual arm | Frozen reference | Required configuration | What the whole method checks |
|---|---|---|---|
| C11 | `/home/chenhc/leaf_v5_evidence_completion_20261006_v1/experiment/belief/runtime.py:494` `BeliefTracker.associate` | `use_src=True`, original full/both arm, soft responsibility, shadow off; same F00 payload and host flags | Original appearance-class mask, min fusion, learned penalty, complete returned assignment tuple, observation likelihood ratios and queued responsibility |
| C00 | `/home/chenhc/leaf_v5_evidence_completion_20261006_v1/experiment/internal_path_ablation_20261007_v1/belief/runtime.py:535` `BeliefTracker.associate` | `path_update_only`: cost off, update/state on, EGIA on, shadow off; same F00 payload and host flags | Actual frozen host assignment via `super().associate` at line 632, plus the original masked-base responsibility features and queued semantic update |

Frozen main runtime SHA-256: `731f6be2e71fa1ec8b537da7e8b13966d527492889fa474b7e67e2f3569ac200`.
Frozen internal-path runtime SHA-256: `619b4f354c475c117d09f769c0fa2968abd7bd05ed1b9a981831f352332c4d28`.
Frozen and current v2 host `host/yolox/tracker/u2mot_tracker.py` both hash to `c658a5e110e6b40c8b83ea8f17e2104dcfd5c5054535bc65a1172d0a5b9349c6` at this audit.

The main frozen runtime has no update-only arm. Its `src_shadow_only=True` chooses LAP on its wrapper base cost (lines 558–566), which already contains the coarse class mask; this cannot substitute for C00's unmasked host assignment. A composite host-assignment plus separately reconstructed update reference is possible, but the frozen internal-path whole method gives a simpler independent reference. Reusing this code does not rehabilitate its old prediction artifacts.

Both frozen methods contain the defective bbox-only binder. Prebind **every input detection on the clone** from independently verified correct ordinals before invoking the frozen method; their `id(det) in self.det_ordinals` branch then skips the defective binder. This is a reference for the original operator under corrected inputs, not an unchanged old end-to-end predictor. Never require corrected C11/C00 to reproduce the old 17 prediction files.

## 2. Correct binding and safe same-state cloning

Place the observer at the actual `associate` seam before `_flush_committed`, `stage += 1`, model calls, or assignment. Preserve identical ordered `trks`, `dets`, threshold, `fuse_score`, `iou_only`, current frame and stage. The observer must not execute the actual association twice.

Independently validate the chosen raw detector row against the new detection's immutable `_tlwh`, `cls`, `score`, and `semantic_score` (`row['class_confidence']`), including exact shapes, finite values and ordinal uniqueness. Use the original detector row/provenance, not the same v2 helper as the only checker. For duplicate boxes with differing metadata, all four fields identify the row. Completely identical metadata rows require deterministic ordinal reservation; validate embedding/source-row identity too if those rows have different appearance vectors. Reused detection objects retain their original ordinal across cascade stages. Bind all current clone detections before calling either frozen method; an unresolved row must raise, never take a fallback unused row.

Use one `copy.deepcopy((tracker_state, trks, dets), memo)` object graph rather than independently copying lists. Preserve aliasing between prior `pending` tracks, host pools, and current inputs. Create the receiver with `object.__new__(FrozenBeliefTracker)` and assign cloned state: calling a frozen unbound method on a v2-class instance can fail or dispatch incorrectly because zero-argument `super()` retains its defining class. Do not call tracker/STrack constructors: constructors can change the global track counter, open GMC files, and normalize supplied appearance arrays in place (`STrack.update_features`, host line 86).

State requiring independent mutable copies includes:

- `pending`, `beliefs`, `observed`, `frame_observations`, `det_ordinals`, `new_candidates`, high/low ordinal arrays and candidate/source dictionaries;
- all live/removed/unconfirmed track pools and seam inputs, with each track's `_tlwh`, last-observation arrays, Kalman mean/covariance, semantic arrays and `cls_hist`, appearance arrays/deque, motion arrays, lifecycle fields and counters;
- adapter/payload estimator arrays, prior and mutable adapter diagnostics such as `last_coherence`; preferably a frozen adapter-class receiver with copied adapter state, rather than sharing a v2 adapter;
- mutable counters, `module_seconds`, `association_stats`, diagnostics/ledgers/hashers and semantic-update accumulators.

Rebuild `det_ordinals` keys using the deepcopy memo because `id(original_det)` is not `id(clone_det)`. Preserve used-ordinal reservations for earlier detections where needed, without sharing original objects; reserved-only keys can be disjoint sentinel keys since the old binder uses their values. Compare canonical ordinal identities rather than raw object-id dictionaries. Assert no shared numpy memory for critical semantic/appearance/state arrays and no shared pending/list/dictionary state.

GMC file handles, output streams, locks and external detector/GPU objects should not be shared mutable handles or blindly deepcopied. Association does not need GMC file input; use a non-operational reference-side placeholder and assert no handle/cursor access. Check the actual file positions before/after the reference. Snapshot class/global state (`BaseTrack._count`, shared Kalman state, matching helpers) and require it unchanged. Frozen references must not allocate IDs, monkeypatch live matching, or change CUDA environment after detector initialization.

Import provenance is part of the reference. Loading the old file by `spec_from_file_location` alone does not ensure old dependencies: cached `capture_belief_evidence`, `belief.*` and `yolox.*` modules may resolve to v2. Record each actual provider's `__file__`, helper `__code__.co_filename`, class MRO and SHA-256 against the immutable counterpart. Load the frozen capture/runtime in separate namespaces, with dependency code hashes verified; do not silently inherit v2's binder, det-record implementation or operator methods. A byte-identical host provider is acceptable when its code and relevant flags are sealed. This bootstrap must be completed without touching live GPU visibility or active inference state.

## 3. Per-seam comparison and acceptance

Execute the actual method once and the selected frozen method once on the clone from the same pre-call state. Compare the full ordered `(matches, unmatched_tracks, unmatched_detections)` tuple, normalizing only representation/dtype; do not reduce it to a set of pairs. Include empty, high/low, tracked/lost, unconfirmed and reused-detection seams, with all argument variants actually encountered.

Compare post-flush `beliefs`/`observed` and the entire new `pending` queue: track identity/index, correct detection ordinal, full observation metadata including semantic evidence, observation ratio and soft responsibility. The latter two use the same frozen model and frozen responsibility features even when C00 disables cost. Check input state and semantic state hashes before/after reference execution to prove the observer did not change the real tracker. Timing and differently instrumented diagnostic counters need not be equal; algorithmic output and semantic state do.

Integer assignment and ordinal identity comparisons should be exact. Equal source data/payload and equal numeric operations should first be checked exactly; any necessary floating tolerance must be specified and justified independently before the run, not enlarged after an online mismatch. Save the maximum discrepancy and mismatch count separately for probabilities, responsibilities and beliefs.

At the next real host-commit boundary, the frozen `_flush_committed` on a corresponding clone can additionally verify the committed update. It requires copied pending track references to reflect the actual current `frame_id` (main runtime lines 311–327). Matching only immediate pending values cannot certify every subsequent commitment, admission or end-to-end prediction rule.

Record per-sequence seam counts, argument/stage coverage, identity checks, return/pending/state comparisons and a rolling digest; total checked seams must equal total executed C11/C00 seams, not just nonempty or eligible seams. Reject on the first mismatch with compact frame/stage/input/state/model provenance. Require zero mismatches and hash-sealed source/binding protocol at start/end. Synthetic matrices remain supplementary contracts; they cannot replace these whole-method checks on the actual online stream.

This oracle validates C11/C00 association and semantic responsibility/update under corrected inputs. It does not independently establish detector accuracy, GT correctness, EGIA birth policy correctness, prepared-training metadata correctness, formal MOT improvement, or absolute parity with an old buggy predictor. C10/C01 still require factorial-definition checks; H heads still require their existing paired-fit contract.

## 4. Analysis-script corrections before any result production

At audit, v1 and copied v2 `analyze_mechanisms.py` have the same SHA-256 `69627db06d3f95de6f937e41567e2d679ce3bce301632974c77f847321e36537`. The current code does not run inference or select settings, and retains signed zero/negative contrasts. Its completion gates are useful but insufficient for corrected v2.

| Priority | Current code | Required correction |
|---|---|---|
| P0 | lines 99–100, 131: old 17-file parity and A06 prediction equality | Replace with corrected-input whole-online reference coverage and zero mismatches; old hashes remain archival provenance only. Rewrite `analyze_completed.py`'s hardcoded parity sentence too. |
| P0 | lines 118–131, 152 onward: old A04/A05 mixed with fresh E10/C11 | Forbid cross-version fusion contrasts and the old/new fusion×selective interaction. Only corrected C11−E10 is presently preregistered for selective-at-fusion. A corrected 2×2 needs corrected-input A04/A05 runs. Gamma remains fixed: this is fusion×selective, not coherence×selective. |
| P0 | lines 74–77: resolved status plus iteration over an evidence dictionary | Require nonempty mandatory binding and independent-reference evidence, full coverage, zero mismatch, current workspace/input/code/model hashes and artifact integrity. An empty dictionary cannot prove resolution. |
| P0 | `analyze_completed.py:main` can run independently | Put the same metadata/reference/stop/frozen-source gates in this entry point or a common preflight. It must reject incomplete, stopped or unresolved runs before writing a scientific report. |
| P1 | line 125: full `OVERALL == a['metrics']` | Guaranteed failure: policy audit has 7 metric fields while the saved metric has 19. Verify each audited field against the full dict, as inventory code already does at line 140. All A04/A05/A06 direct comparisons were independently confirmed False. |
| P1 | lines 106–111: unsealed runtime reports and receipt totals | Rehash every runtime/GT-guard/reference evidence artifact against sealed receipt hashes. A recomputed total alone cannot detect altered reports retaining the same values. Require exact per-sequence input/coverage digests as well as aggregate input equality. |
| P1 | lines 87–89 and `verified_metric` | Seal the reference GT-contract artifact hash, scorer and dependency hashes/version, exact sequence list, class/filter/IoU conventions and result location. Independently compare current arms with this contract; GT file hashes plus the MOTA formula do not reproduce IDF1/scorer semantics. |
| P1 | lines 129, 144, 147: `reused_verified_online_control` | Old metadata-contaminated artifacts may be hash verified but must be labeled `historical_metadata_contaminated`, `scientific_use=false`, with the reason and original source hashes. Do not merge them into corrected estimates. |
| P2 | primary analysis at line 150 and append/write steps | Complete all semantic/schema/lineage/contrast checks before output; write a transactional output bundle to avoid a primary report surviving a later failure or duplicate append on rerun. No completed report should pre-exist as apparent evidence after a gate failure. |

Missing gate/receipt/result files currently raise before analysis. Keep this rejection behavior; a graceful machine-readable HOLD can live separately from final results. A status label cannot replace evidence. The GT guard establishes its documented Python-open/designated-root scope, not a universal no-GT theorem; preserve the scope explicitly and seal its own receipt. Analysis/scoring may read GT after inference, with its separate permitted stage.

## 5. Historical nulls and interpretation limits

Retain all old hard/soft, recurrence, source-pairing and birth-gate metrics and signed differences as archival observations; include exact counts and prediction-equality hashes where known. Keep zero and negative values in JSON/CSV/table rather than omit or relabel them as success. Show metadata contamination next to each historical contrast. Neither unchanged old predictions nor a historical nonzero difference certifies the corrected mechanism. Do not claim corrected recurrence/responsibility is ineffective from an invalid historical null, and do not hide the null.

For corrected SRC four-cell results, `(C11−C10)−(C01−C00)` is a descriptive difference of conditional effects at one frozen operating point. It is not a causal panel DiD estimate, a significance test, or a mechanism-necessity proof. Report MOTA/IDF1 in percentage points and FP/FN/IDs/FM with exact signed count differences; lower error counts and higher accuracy metrics have different favorable directions. Flight means are means of clip deltas, not pooled flight IDF1; frames are not independent statistical replicates.

Paired Hhier−Hflat compares the predeclared factorized hierarchy and flat softmax under common fit rows/scaler/event weights and fixed optimizer settings, within the corrected runtime. It does not isolate pure graph topology at equal regularization geometry/capacity, and F00−Hflat also changes the historical binary-specific recipe. Calibration NLL and online tracking results remain separate. A figure can show the two head factorizations and the corrected signed contrast with zero visible; no positive result is presupposed.

For the author figure, show corrected SRC 2×2 cells and the Hhier/Hflat comparison only after their gates pass; show C11−E10 as selective policy conditional on fusion. Put old fusion-policy/null controls in a clearly marked historical-invalid inset or audit table, never a connected corrected 2×2. Record that the metadata fix changes the evaluation lineage even when detector/ReID input bytes remain the same.
