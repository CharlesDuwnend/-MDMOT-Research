# Full-SRC EGIA detailed ablation (2026-10-08)

Question: with every SRC component and operating setting held fixed, which EGIA internals change closed-loop tracking? Prior Table V already isolates fusion/selective and is retained; no claim that Table III's branch-refitted EGIA-only row is the same control.

Nine fresh corrected-metadata cells are declared before observing any new MOT score:

| Cell | Intervention | Fixed context |
| --- | --- | --- |
| D00_full_reference | None; exact current full system | Full SRC, F00 EGIA, fusion .40, selective .20 |
| D01_EGIA_bypass | Birth decision uses the existing native gate; no EGIA probability calls | Full SRC including birth initialization/memory; native gate .60/.70 |
| D02_no_foreground_evidence | Replace learned foreground posterior with its fit39 weighted prior | Other learned heads and policies frozen |
| D03_no_coverage_evidence | Replace learned conditional coverage posterior with its fit39 weighted prior | Other learned heads and policies frozen |
| D04_no_learned_context | Both binary contextual posteriors replaced by fit39 priors | Source co-reference and policies frozen |
| D05_no_source_coherence | Gamma = 0 | All fitted cue/context heads and policies frozen |
| D06_broken_source_pairing | Deterministically permute appearance/availability across source rows; marginals/context unchanged | Same full learned heads and policies |
| D07_no_geometry_cue | Neutral geometry likelihood ratio, log ratio zero | Original contextual heads, appearance and policies frozen |
| D08_no_appearance_cue | Neutral appearance likelihood ratio, log ratio zero | Original contextual heads, geometry and policies frozen |

Neutral foreground/coverage controls retain legal three-class distributions rather than setting probabilities to impossible all-foreground/all-covered values. Their weighted priors use the canonical fit39 rows, base flight/group weights and original binary square-root balancing. No refit, calibration selection or test-dev selection. These are frozen-operator evidence-availability interventions, not architecture retraining controls.

For source co-reference kappa = log(n) + logsumexp(a+b) - logsumexp(a) - logsumexp(b), a neutral cue makes kappa exactly zero. D07/D08 are therefore algebraically redundant with D05 at the decision level; retain this fact and verify expected identical predictions. They test degeneration of the two-cue operator, not three independent sources of tracking gain. Pairing permutation instead preserves both marginals while changing co-reference, and is the independent pairing mechanism test.

All cells use original 17-sequence/6635-frame VisDrone test-dev, identical detector/ReID ordered-array ledgers, fixed detector/checkpoint/GMC/scorer/GT/host and physical GPU1 A10040GB. Full SRC includes the coarse-class appearance mask, semantic cost, observation initialization and learned soft responsibility/memory updates. The original responsibility B_ref is preserved. Association tuples and pending responsibilities are checked against the immutable corrected-seam reference in every actual frame.

CPU semantic/identity/reference and new operator contracts precede 200-frame smoke on the previously confirmed equal-box frame. Freeze source/model/fit-data hashes before GPU. The new full reference must match the previous corrected C11 in all 17 actual prediction hashes and input ledgers; stop on any mismatch. Every source/anchor weight outside the named intervention is hash-compared to F00. The three prior substitutions have fit-only provenance; removed learned heads are not refit. Full runs remain GT-free, evaluator alone uses original GT. Report pooled MOTA/IDF1/FP/FN/IDs/FM plus admission/output counts, keeping reduced coverage separate from identity improvement. Flight/clip diagnostics are descriptive, not independent-frame significance.

Deliver tables only, no new graphs. Keep raw streams/checkpoints local and snapshot compact source/evidence with stage_snapshot.py at phase boundaries. Every complete, zero, negative or failed result is retained. Existing fusion/selective four-cell results may be combined only after the new full-reference/source/input parity gates close.
