# Duplicate geometry ordinal audit

Scope: source inspection and synthetic CPU execution only. No frozen v1 files,
manifest, models, GPU jobs or prediction streams were modified. The parent
separately reported byte matching 3,055 real frames and 9 frames / 18 detections
with wrong metadata; that stream proof is owned by the parent and is not replaced
by this synthetic receipt.

## What the actual source does

- `tools/track_dynamic_online.py:109` sorts retained raw rows by x1. Host
  `u2mot_tracker.py:1641` uses raw objectness column 4 to partition high/low,
  and constructs high detections before low detections. Within each pool the
  boolean selection preserves the raw order.
- Current canonical `/home/chenhc/src_egia_belief_20260923/tools/capture_belief_evidence.py:91`
  uses a rounded geometry index and then `allclose`; it consumes the first
  unused matching ordinal. Its indexed optimization does not include class or
  scores. The older copied capture in v1 scans geometry and also has an unused-row
  fallback. v1 `belief/runtime.py:694` scans geometry without that fallback.
  All three have the same ambiguity for exact duplicate geometry.
- Capture `det_record` checks only exact bbox equality. v1
  `BeliefTracker.det_record` directly inherits it (`belief/runtime.py:187`).
  Equal geometry therefore passes despite swapped class, objectness and class
  confidence.
- A new STrack stores class column 6 as `cls`, raw objectness column 4 as
  `score`, and class confidence column 5 as `semantic_score` after clipping to
  [0,1]. There is no objectness times class confidence reduction in `score`.
  Compare new detections, rather than mutable class histories of existing tracks.
- Wrong metadata reaches observation probabilities (`belief/runtime.py:718`),
  pending committed observations (`:819`), EGIA candidate/source event (`:845`),
  and birth belief initialization (`belief/online.py:142`). Meanwhile `B_ref`
  class masking still reads actual `det.cls`, so its class can disagree with the
  learned observation on the same detection.

## Executed reproduction

`review/duplicate_ordinal_reproducer.py` uses actual frozen and canonical
associate/det_record methods and actual STrack objects, with synthetic FP16 raw
rows and no GPU. `review/DUPLICATE_ORDINAL_CPU_RECEIPT.json` contains the full
row records and source hashes. Nine cases executed successfully:

| Actual implementation | Same bbox, low before high | High before low | Same pool stable order |
| --- | --- | --- | --- |
| copied CaptureTracker | 2 silent metadata swaps | correct | correct |
| canonical indexed CaptureTracker | 2 silent metadata swaps | correct | correct |
| frozen factorial BeliefTracker | 2 silent metadata swaps | correct | correct |

Every corrupt row passed the actual exact bbox assertion. With the synthetic
observation model used by the existing CPU fixtures, a real high vehicle
detection acquired the low human row and changed posterior from [0.05,0.95] to
[0.95,0.05]. This demonstrates feature-path sensitivity; it is not a claim about
the numerical probabilities of the trained frozen head.

Receipt SHA256: `582d6a4f8e468d1acd8658b616eb1c826d281e081fd986d910add65f4ab707cc`.

## Isolated correction contract

Bind on `(bbox_tlwh, predicted_class, raw_objectness, class_confidence)` against
`(det._tlwh, det.cls, det.score, det.semantic_score)` using the unchanged
coordinate conversion and lossless scalar conversion. Require finite fields;
valid detector class confidence already lies in [0,1]. Do not compare a
combined score, a coarse human/vehicle class, or normalized appearance features.
STrack normalizes its feature in place; embeddings from duplicate centers can
also coincide, so features do not supply a reliable row identity.

Only unused ordinals are eligible. Zero candidates raises unresolved identity;
there is no geometry-only or arbitrary-row fallback. Fully identical metadata
duplicates may consume the first unused original ordinal deterministically, as
authorized by the parent. They are necessarily in the same high/low pool;
the ordinary host visits that pool in its preserved raw order. The synthetic
reference matcher shows six formerly ambiguous rows become uniquely matched
after the four metadata fields are included. Two fully identical metadata rows
remain equivalent candidates and require the specified deterministic policy.

An already cached object id must also validate all four fields before returning
its ordinal, including direct det_record calls. Do not silently trust a cached
geometry-only binding or remap it to a different row. Retain the mapping for
the same detection object when it reappears in a residual or unconfirmed seam;
reset mappings each frame. If future stages permit object destruction/reuse,
retain per-frame object references to prevent Python id reuse from impersonating
a cached object. Explicit raw row ordinals stamped at STrack construction are a
stronger future option, but are not required for this narrow correction.

## Independent three rounds, three checks per round

These are required validation cases for the isolated v2 correction, not PASS
claims for v1.

| Round | Check | Independent expected observation |
| --- | --- | --- |
| 1: direct identity | R1.1 FP16 exact bbox, different classes, low raw row before high; run actual high then low host seams | High binds its own class/score/confidence and low binds its own; repeat reversed raw order and swapped classes. Compare each field against raw original ordinal, not against helper output. |
| 1: direct identity | R1.2 Same bbox and same class with different objectness and/or class confidence, in same and different pools | Every detection keeps its original confidence tuple. Include exact objectness equal to host high threshold, which is low because host uses `high > threshold` and `second & ~high`. |
| 1: direct identity | R1.3 Fully identical metadata duplicates and zero-track birth-first frame | Consume original ordinals once in raw order, deterministic across repeats; no premature SRC/EGIA inference required for empty association. |
| 2: failure guards | R2.1 Manually poison an existing id mapping, then reenter associate and direct det_record | Wrong bbox/class/objectness/class confidence each raises before model/assignment or return; already correct id remains stable across reused seams. |
| 2: failure guards | R2.2 Correct geometry but no exact metadata match, including exhausted candidates | Raise unresolved identity; model/solver counters do not advance. A near-but-not-identical box cannot bypass the final exact geometry contract. |
| 2: failure guards | R2.3 Bijection and frame reset | No distinct live detections consume one ordinal; same object reappearing preserves ordinal; new frame starts fresh and accepts a new metadata tuple without stale ids. |
| 3: runtime integration | R3.1 Real online CPU host update with duplicate rows, mocked small heads, one birth then one matched update | Actual det_record, birth initialization, source candidate and postcommit memory contain the intended raw class/confidence; no memory update precedes host commit. Features may differ from v1 only where corrected bindings differ. |
| 3: runtime integration | R3.2 Four factorial cells and no-duplicate control | Input detector/embedding bytes, row order and rolling hashes unchanged; C00 still executes actual native assignment, B_ref still original masked reference, masks/penalty switches and thresholds unchanged. On duplicate-free inputs old/new assignments and output bytes agree. |
| 3: runtime integration | R3.3 Real captured stream identity independently checked against paired input artifacts | On the parent's byte-matched real frames, compare raw ordinal to actual new-det bbox/class/score/confidence at every actual high/low/reused seam; wrong metadata count must be zero. Match detector and embedding stream hashes before drawing runtime conclusions. |

Earlier F00 byte parity can reproduce a shared ordinal bug. It establishes
historical behavior compatibility, not correctness of metadata identity. Current
v1 results retain their corrupted implementation lineage; corrected v2 needs a
new receipt and cannot reuse v1 mechanism results as validated evidence.
