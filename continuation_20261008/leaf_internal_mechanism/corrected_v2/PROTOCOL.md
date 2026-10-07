# Corrected internal-mechanism protocol amendment, 2026-10-08

This amendment supersedes the original reference-parity and historical-policy-reuse
requirements below. Runtime root: `/home/chenhc/leaf_internal_mechanism_20261008_v2_metadata`.
The v1 queue was stopped before any complete new scored arm after detector-byte
matching and frozen-code CPU replay confirmed bbox-only detection-ordinal
mismatches (9 frames, 18 detections in 3055 exact-input frames). This finding is
input correctness evidence, not a measured MOT-score difference. Original paper,
frozen predictions, checkpoints, heads and old receipts remain unchanged.

The sole shared execution repair verifies detection bbox, class, objectness and
class confidence, including cached identities. Equivalent complete-metadata
duplicates receive deterministic unused ordinals; no arbitrary fallback is allowed.
All nine new online cells use the same repair, detector/host/GMC/GT/scorer and
frozen weights. The paired hierarchical/flat heads are reused byte-for-byte from
the preregistered common-recipe fits; no response to observed test-dev scores is
used for refitting or parameter selection.

The new protocol contains the original seven cells plus corrected
`E00_fusion_off_no_selective` and `E01_fusion_off_selective`, using the original
A04/A05 models unchanged. These two controls are rerun because historical scores
cannot provide a corrected fusion × selective factorial. Coherence gamma remains
fixed in all policy cells. Only fc_full supports either declared fusion setting;
mask-only, penalty-only and off/off cost cells retain original fusion/selective.

For every actual full and off/off association seam, an independent reference
executes the immutable original full or update-only runtime method on cloned
same-state tracks/detections with verified metadata identities. Returned matches,
unmatched indices, pending observation, evidence ratio, responsibility and stage
must match exactly. The immutable host native seam and threshold-edge legality
are independently checked for every cost cell. Historical 17-file prediction
parity is recorded as a diagnostic, since retaining the known metadata defect is
not a correctness condition. Source/provider hashes identify the shared strict
metadata guard explicitly; original cost, responsibility and update operators
remain immutable.

Before full inference, three audit rounds with three implementation checks each
must cover real-input discovery, strict identity repair and independent corrected
reference contracts. A 200-frame smoke on `uav0000073_00600_v` includes the confirmed
problem at frame 186 and runs all nine cells. Verify real terminal exit, same-process
physical GPU UUID, every actual reference seam, full raw detector/ReID input ledger,
actual operator calls, metadata assertions and Python GT guard. Seal a nonempty
identity gate binding CPU and smoke receipts and current frozen source hashes.
Then evaluate all nine arms once over 17 sequences / 6635 frames using original GT.

Report every conditional cost/policy contrast, the paired common-recipe head
contrast and descriptive factorial interactions, including zero/negative results.
Historical controls remain visible with `scientific_use=false` for corrected
mechanism attribution until independently corrected and rerun; preserving their
exact scores does not prove unaffected metadata. No corrected-vs-historical
factorial subtraction is permitted. This author-review addendum does not overwrite
the canonical manuscript or change the primary method based on test-dev ranking.

---

## Original preregistration retained for lineage (superseded requirements above)

# Current LEAF internal mechanism addendum, 2026-10-08

The user requested an experiment-gap plan followed by actual scoring. This
document freezes the comparisons before new test-dev inference. Historical
PLAN ONLY and paused preparations are reference material; historical completed
outputs remain immutable. Runtime root:
`/home/chenhc/leaf_internal_mechanism_20261008_v1`.

## Existing evidence and the gaps

The current paper has benchmark results, ByteTrack/BoT-SORT adapted pairs,
configuration-level SRC/EGIA rows, source-pairing diagnostics and selected real
cost examples. Independently audited October 7 runs additionally cover frozen
SOE x SRC, source-pool choice, source-pair shuffle, hard updates, separate SRC
cost/update paths, and native N0--N5 birth-gate/SRC controls. These are completed
experiments, not remaining work. Source receipts and metrics are listed in
`EXISTING_EVIDENCE_INVENTORY.json`.

The SRC cost path contains both a coarse-class appearance mask and a learned
semantic penalty. Existing full-minus-noSRC differences do not isolate them.
Recursive-update and hard-update controls have essentially null closed-loop
effects and must remain in the report. The existing A04/A05/A06 policy results
are current F00 controls, not an older H2 benchmark; their model/source/input
lineage has been checked. They fill three cells of fusion x selective, leaving
fusion-on/selective-off missing. The hierarchical anchor lacks a matched flat
three-class fitted alternative.

## Phase 1: frozen-head online controls

All five runs use current F00 heads and the same birth initialization,
responsibility/update operator, EGIA source scorer, source pool and gate.
Two runs verify the new runner; three are genuinely new controls.

| Name | Appearance class mask | Learned semantic penalty | Fusion | Selective | Role |
|---|---:|---:|---:|---:|---|
| C11_full_reference | on | on | .40 | on, margin .20 | 17-file canonical F00 parity prerequisite |
| C00_nativecost_reference | off | off | .40 | on, margin .20 | 17-file update-only control parity prerequisite |
| C10_mask_only | on | off | .40 | on, margin .20 | isolate mask contribution |
| C01_penalty_only | off | on | .40 | on, margin .20 | isolate learned penalty contribution |
| E10_fusion_no_selective | on | on | .40 | off, argmax | complete policy factorial |

For the cost factorial, learned responsibility features always consume the
original masked base B_ref. Assignment-mask toggling must not silently change
the responsibility model's input definition. Off/off commits the actual native
assignment; on/on must reproduce F00. Existing gate .70 affects new-track
eligibility only and is held fixed. No parameter sweep, retraining, model
selection or winner selection is included in phase 1. All cells are reported.

Fusion/selective table uses the verified existing A04 (off/off), A05 (off/on),
F00 (on/on), and the single new E10 (on/off) result. Existing recurrence and
responsibility null controls remain visible and are not discarded in favor of
the new tests.

## Phase 2: hierarchical versus flat EGIA anchor

Fit paired hierarchical and direct three-class logistic anchors on the exact
original anchor fitting events, ten context features, labels and fit39/cal8
contract. Both use one common scaler fitted on all fitting rows and the same
three-class event weights from the original beta=.5 balancing family. The
coverage binary head uses the foreground subset of those weights without
renormalization. Hold regularization, solver and seed fixed. Hold the source
cue heads, coherence weight, summary model, fusion .40, selective margin .20,
SRC heads, detector and host fixed. The historical F00 is an external
reference: its separate binary balancing/scalers differ from the common
recipe, so F00-minus-flat is not a pure hierarchy contrast. Only the paired
refit contrast isolates factorization under the declared common recipe.
No new detection/appearance training or heldout selection is permitted.
Calibration diagnostics are reported separately from the complete online
tracking comparison. Each frozen head is evaluated once on all test-dev
sequences after the fitting receipt is sealed. Exact rows/weights/source
hashes and hyperparameters are fixed in PHASE2_HEAD_PREREGISTRATION.json.

If original fitting rows or weighting cannot be reconstructed exactly, record
the discrepancy, perform the required implementation audits, and do not call
an unrelated legacy summary head a flat control. Resolve the prerequisite
instead of quietly changing the scientific question.

## Common protocol and validity requirements

- Current F00 bundle SHA-256:
  `ee903030e48f73fe87411b6e9aa60d52bbf2ee2d895aebfa8085d260779702ac`.
- Frozen summary SHA-256:
  `aa48c972a3ddbe4c5e4b82c5f2a64f9032978d7c15a863a5a3caefe67ac04ce8`.
- Detector checkpoint `/home/chenhc/20250715/u2mot/visdrone.pth.tar`, SHA-256
  `8cb39ca273dbe620a8551a247b72865ad4f870225351d0937e306980c594c939`.
- VisDrone test-dev: all 17 sequences, 6,635 frames. Fixed detector confidence
  .09, NMS .70, size 896x1600, fp16/fuse, host high/low thresholds .50/.10,
  matching gate .80, track buffer 15, birth score .60, class gate .70, file GMC.
- Same frozen host, ReID, evaluator and original GT contract. Inference reads
  no GT. The evaluator alone reads annotations. Record prediction hashes,
  raw detector/embedding rolling hashes, runtime calls and same-process GPU
  UUID verification. Use physical GPU1 when available; never physical GPU2 or
  a 4GB device. Other workloads are preserved.
- CPU operator/model contracts precede an 80-frame engineering smoke.
  Full and native-cost reference prediction parity precede the new cells.
  Both references must complete all 17 sequences; a scalar metric match alone
  is insufficient. Actual execution and independent source audits, not saved
  PASS flags alone, establish validity.
- Exclusive output paths and persistent tmux launcher. Inspect live handles,
  completed sequences/frames, logs and terminal exit receipt before calling a
  run complete. Preserve partial/failed runs as such. Failures receive three
  audit rounds with three independent implementation checks per round.

## Measurements and paper use

Report MOTA, IDF1, FP, FN, IDs, fragmentation; per-sequence and per-flight paired
deltas; prediction/input hashes; actual mask/penalty calls and changed legal
edges; responsibility/committed update counts; source/fusion/selective events.
Frames and tracks are not independent experiment replicates; no frame-level
significance claim is planned. Closed-loop conditional contrasts need not add
up. A selected calibration cost matrix is local mechanism evidence, not an
aggregate causal estimate. UAVDT remains vehicle-only under original-GT
Protocol B; a human/vehicle SRC contrast is not portable to that benchmark.

Deliver a compact completed-result table and mechanism interpretation for
author review. Zero or negative results are retained. The existing main paper,
canonical method, frozen baselines and old results remain protected during
this experiment. No positive outcome is promised in advance.
