# Mechanism figure options for author review

Status: design only, 2026-10-08. The nine-arm 200-frame engineering smoke was
live when this memo was requested. No unfinished MOT values, new result plots,
GPU work, frozen edits or manuscript edits were used. Recommendation: use
option A as the quantitative mechanism figure and option B as its error audit;
option C is conditional on an actual, sufficiently complete assignment trace.

## Current Figure 4: verified visual reference

The manuscript `main.tex:284` currently includes
`/home/chenhc/src_egia_icme_paper/figures/src_assignment_mechanism_v8/src_semantic_response_triptych.pdf`
through `\graphicspath{{figures/}}`. Its current caption and paragraph distinguish
five-class schematic evidence, fixed-belief analytic responses, and a closed
calibration 3×3 candidate component. They do not claim an online IDF1 gain.

I inspected its actual color preview. It is a clean white-background triptych,
with normal-weight `(a)/(b)/(c)` titles, DejaVu Sans/STIX math, minimal axes,
black numerical labels and thin assignment outlines. The renderer is
`/home/chenhc/claude_try_MDMOT/continuation_20261007/src_mechanism_v8/build_clean_layout.py`.
The export width is 7.16 in, height 2.40 in; base text 7.2 pt, ticks 6.8 pt,
titles 8.4 pt. Preserve this visual language, while allowing a taller quantitative
figure rather than shrinking labels.

- Existing scalar cost palette, from `scripts/build_semantic_analysis.py:34`:
  `#FBF8F4, #F2DED1, #DBDCEA, #B7C6D8, #90A9C2`.
- Text/muted colors: `#111111` / `#58636F`; H/V line colors:
  `#6C839D` / `#B88469`; compatible/conflicting edge outlines:
  `#506B87` / `#A86644`.
- Schematic Track/Detection accents: `#0072B2` / `#E69F00`.
- Cost panels use the same [0,1] scale, explicit numbers, solid/dashed boxes,
  and crosses for costs above the stage threshold. A new signed-effect panel
  needs a zero-centered diverging scale or signed points; the existing sequential
  cost palette must not be repurposed to imply positive/negative effects.

Paths for review: the same folder contains `src_semantic_response_triptych_preview.png`,
`src_semantic_response_triptych_grayscale.png`, PDF, SVG and PNG. Source caption:
`continuation_20261007/src_mechanism_v8/FIGURE_CAPTION.md`.

Verified reference hashes:

- Current Fig.4 PDF: `936f9eca7db0df235b867acf2c200d2001fd964a635104c91d9cbd28d922ab92`.
- v8 renderer: `b7487634bf95221ad4953b90065bc65a94b97ce766350df71763c5a590d886c6`.
- Current frozen nine-arm manifest: `071e1bfa2530ef9b7f79884c3264f5969b19db2f842903732e509d7fa81c965b`.

## The nine runs form three questions

These are nine unique runs, not ten: C11 is also the E11 corner.

| Question | Cells / runs | What remains fixed |
| --- | --- | --- |
| SRC assignment factors | C00 native cost, C10 mask only, C01 penalty only, C11 mask+penalty | Frozen F00 heads, SRC initialization/soft memory updates, original masked B_ref responsibility input, fusion on and selective on |
| EGIA deployment factors | E00 fusion off/selective off, E01 off/on, E10 on/off, E11=C11 on/on | SRC full assignment and memory, trained source heads, prior/scalers; fusion weight γ=0.4 when on and selective margin=0.2 when on |
| Source head parameterization | H_paired_hierarchical versus H_paired_flat | Predeclared common fit rows, scaler, event weights and optimizer recipe; full SRC, fusion on, selective on |

For the cost grid, “mask” is a coarse-class mask on the appearance channel
before min fusion with geometry. It is not a hard ban on every cross-class
candidate edge. C00 still contains learned SRC memory and EGIA; it is not a
standalone native tracker baseline. For the EGIA grid, the crossed factors are
fusion and selective policy, not coherence and selective policy. Hhier−Hflat
tests factorization/parameterization under the paired recipe, not equal-capacity
pure topology, and must not be replaced by F00−Hflat.

The algorithms/context are fixed within each contrast, but online beliefs,
source ledgers and track pools can diverge after earlier decisions. A closed-loop
contrast includes those trajectory consequences. Option C separately holds an
actual pre-assignment state fixed to explain the local operator.

## Option A — two conditional-effect grids and a paired-head panel (recommended)

**Question:** Does each operator help, harm or leave tracking unchanged under
the other operator's state? Are the conditional effects consistent or interacting?

Layout: a three-panel vector figure, approximately 7.16×3.2–3.6 in.

- **(a) Assignment factors:** a 2×2 grid, horizontal mask off→on, vertical penalty
  off→on, with C00/C10/C01/C11 explicitly labeled. Each cell prints its aggregate
  IDF1 and MOTA. Four thin edge arrows mean “turn this factor on”, and print the
  corresponding signed IDF1 contrast. Arrows are intervention directions, not
  evidence of improvement. A small line beneath reports the descriptive
  interaction, with the zero reference visible.
- **(b) Birth decision factors:** the same layout, horizontal fusion off→on,
  vertical selective off→on, E00/E10/E01/E11=C11 labels. Print all four absolute
  aggregate scores and all four conditional contrasts, including zeros and
  negatives. Label γ=0.4 and margin=0.2 as frozen operating settings.
- **(c) Paired source heads:** a compact signed point plot centered at zero for
  Hhier−Hflat, with separate IDF1 and MOTA rows in percentage points. Print both
  raw head scores nearby; a zero difference remains a visible point at zero.

For metric Y (IDF1 or MOTA, in percentage points), compute:

| SRC conditional effects | EGIA conditional effects |
| --- | --- |
| Mask with penalty off: Y(C10)−Y(C00) | Fusion with selective off: Y(E10)−Y(E00) |
| Mask with penalty on: Y(C11)−Y(C01) | Fusion with selective on: Y(E11)−Y(E01) |
| Penalty with mask off: Y(C01)−Y(C00) | Selective with fusion off: Y(E01)−Y(E00) |
| Penalty with mask on: Y(C11)−Y(C10) | Selective with fusion on: Y(E11)−Y(E10) |
| Interaction: (Y(C11)−Y(C10))−(Y(C01)−Y(C00)) | Interaction: (Y(E11)−Y(E10))−(Y(E01)−Y(E00)) |

A signed cell/background alternative is a pale blue–white–ochre diverging map
centered on zero, with explicit `+ / − / 0.00` labels and a symmetric scale.
Color denotes the signed effect, not H/V class. Missing or unresolved results
are hatched `pending/invalid`, never zero. Keep the ordering set by factors,
rather than sorting arms by performance. Do not use significance stars or
frame-wise error bars. With one frozen run per arm, report exact descriptive
contrasts; any uncertainty requires a separately defined sequence/flight
resampling estimand and pairing, not frames treated as independent replicates.
Retain unrounded values in the CSV/JSON. If a tiny nonzero effect rounds to
0.00, print its sign with `<0.01` or increase precision; reserve “exact zero”
for an exact verified metric difference, and reserve “unchanged predictions”
for prediction equality established separately by hashes.

**Mechanism interpretation:** a positive conditional contrast supports utility
of the switched operator at that fixed operating context; a negative contrast
shows a tradeoff or failure at that context; an exact zero means no change in
that metric and may reflect redundancy or cancellation among changed outcomes.
A nonzero interaction
shows context dependence. None proves SRC reliability weighting, recurrence,
EGIA necessity, universal benefit, significance or a causal panel DiD effect.
If IDF1 and MOTA disagree, both must stay visible.

**Evidence required:** completed nine-arm predictions, exact sequence/frame
coverage, common detector+embedding stream hashes, sealed corrected-identity
and reference gates, frozen model/policy/source hashes, scorer/GT contract,
and reverified aggregate counts. Use the completed v2 `artifacts/results.csv`,
`artifacts/receipt.json`, per-arm receipt and `tracking_metrics.json`. Smoke
completion alone supplies no such MOT evidence. Do not reuse old A04/A05 or
v1 polluted outputs as E00/E01 cells; the new corrected v2 runs supply them.

Working caption, to fill only after completion:

> Conditional tracking effects of assignment and birth operators under frozen
> models. The two grids retain all four cells and signed conditional differences;
> C11 is shared with the fusion-on/selective-on corner. The paired-head panel
> compares common-recipe hierarchical and flat source heads. Values are pooled
> metrics and percentage-point differences on the same complete sequences.

## Option B — signed FP/FN/ID-switch error decomposition

**Question:** Which tracking errors account for a gain or loss, and what cost
is paid elsewhere? This is more diagnostic than another ranking of nine scores.

Layout: three aligned blocks for SRC, EGIA and paired heads. Columns are FP,
FN and IDs (optionally a separate final ΔIDF1 column). Use signed points or
horizontal bars around a shared zero line, with exact count differences printed.
For the primary view use the same four conditional contrasts per 2×2 grid,
plus Hhier−Hflat: nine contrast rows. These are nine comparisons, not nine
independent runs. Place interaction rows in an appendix if the main panel gets
too dense. The accompanying CSV must still retain raw errors for all nine arms.

Define favorable MOTA contributions consistently:

`g_FP=100*(FP_reference−FP_variant)/N_GT`, and likewise for FN and IDs.

Positive means fewer errors; negative means more. With the same GT denominator,
`ΔMOTA=g_FP+g_FN+g_IDs` exactly. Show a small total marker to check this identity,
and print the signed raw counts so normalization cannot obscure an exact zero.
If raw `variant−reference` counts are used instead, explicitly mark “lower is
better” and do not silently reverse their sign relative to metric deltas.

IDF1 has its own identity-matching definition and cannot be decomposed as the
sum of FP/FN/IDs. Fragmentations may be an auxiliary row, but are not a fourth
term in the MOTA sum. Never combine calibration NLL with these errors on one
performance axis. Retain benefits, harms and zeros in every error category;
do not remove a row because its aggregate MOTA is negative or its IDF1 is zero.

**Mechanism interpretation:** an FP reduction with FN growth can expose a
suppression/admission tradeoff; an IDs reduction with changed FP/FN can expose
association/detection tradeoffs. The aggregate counts do not identify whether
births, matches or lost-track states caused them. Linking an FP to a rejected
birth or an ID switch to a particular cost edge needs matched scorer events and
a trace of that actual decision. Ordinary module-call/changed-edge counters
cannot supply that link.

**Evidence required:** all option-A gates plus per-sequence scorer counts,
identical `num_objects`, exact aggregation, and the scorer's ID-switch/ignore/
class conventions. For event-level attribution, add GT-based post hoc scorer
event matching with explicit unknown cases; GT remains outside inference.

## Option C — actual same-state assignment cost matrices (only with trace)

**Question:** What numerical decision does the mask or penalty change on an
actual frozen pre-assignment state? This option matches the existing Fig.4
language most closely, but the currently logged counters/hashes are insufficient.

Layout: four numeric matrices in a 2×2 factor arrangement, C00/C10/C01/C11,
with identical row/column order, beliefs, detector observations, embeddings,
prior, stage and threshold. Use the existing Fig.4 sequential cost palette and
shared [0,1] colorbar. Preserve unrounded matrix values in the artifact and
print three decimals only for readability. Gray/hatch nonfinite or unavailable
edges instead of treating them as cost 1. Put actual selected pairs in outlined
boxes; mark threshold-excluded cells separately. Ground-truth-compatible,
incompatible and unknown owner labels may be added only through an independent
post hoc owner audit. The threshold is the actual stage's threshold, e.g. 0.80
for the first high stage; do not substitute 0.80 for a low stage using 0.50.

The four matrices are counterfactual operator applications to one held state,
not four independently evolved arm trajectories at the same frame. C00 must be
the actual native assignment matrix; C10 is the original masked B_ref; C01
adds the unchanged semantic penalty to the unmasked assignment base; C11 adds
it to the masked assignment base. The original B_ref remains the responsibility
input in all cells. If the actual stage is IoU-only, the mask may have no effect;
show that null rather than selecting another stage to make it look active.

For a compact 3×3 component, prove closure against the full native legal-edge
graph: no legal outside edges for its tracks/detections at that threshold.
Otherwise label it a matrix crop and do not infer a global LAP optimum from it.
Include unmatched outcomes and tied optima when assessing local assignments.
Do not label the base as uniquely wrong when two optima tie. A cost change on
an already illegal edge is not evidence of an effective association change.

Minimum trace evidence for each illustrated state:

1. Correct raw row identities and detector/embedding input hashes; immutable
   track/detection metadata, pre-assignment beliefs, frozen heads and group prior.
2. Full geometry/appearance arrays, original B_ref, four assignment matrices,
   finite masks, legal-edge masks at the actual threshold, and the unchanged
   argument tuple (`fuse_score`, `iou_only`, stage).
3. Actual solver input/output tuples, ties/unmatched decisions and component
   boundary audit; if owner correctness is claimed, its separate post hoc labels
   and unknown count. A full-state clone must preserve object aliases and leave
   the live tracker unchanged.

Case selection should be reproducible and recorded before viewing tracking
gains: e.g. the first eligible closed component per declared sequence order.
For author review, retain the observed beneficial, adverse and unchanged local
cases with their eligibility denominator; absent categories remain zero, never
fabricated. A single selected favorable matrix is an illustration, not evidence
that the intervention helps across the dataset. Do not replace this with a
heatmap of module calls, nonzero penalties or changed numerical edges.

Current `track_dynamic_online.py` writes aggregate `runtime_report` and ordered
input hashes; `reference_oracle.py` verifies seams but does not persist the full
candidate matrices/snapshots. Hashes cannot reconstruct detector rows or
embeddings. Hence option C needs an existing complete trace or a separately
authorized, isolated diagnostic capture/replay after the current run; never
edit the frozen live source to add a plotting log. The older calibration
`SELECTED_CASE.json` remains an independently bounded historical same-state
example and is not a corrected v2 online trace.

## Author review delivery and stop rules

The quantitative figure title should name the question, such as “Conditional
effects of assignment and birth operators”, rather than claim improvement.
Use zero-centered effects, consistent signed labels, no broken effect axes,
and all cells/contrasts. Preserve the Fig.4 typography/spacing and gray-scale
readability. Export final plots as PDF/SVG with embedded fonts, and color/
gray previews at the final intended width; inspect them at print scale before
manuscript use. Use `(a)/(b)/(c)` panel labels to match the manuscript.

Keep calibrated head diagnostics, engineering activity, held-state numerical
operator examples and completed tracking outcomes distinguishable in captions.
A module executing or a penalty being nonzero proves activity, not a legal-edge
intervention, a correct assignment or a tracking gain. A failed identity,
reference, common-input, coverage or scorer check leaves the corresponding
quantitative figure on HOLD; it must not silently omit the failed cell. No
publication result plot should be generated from the current live smoke.

Sources inspected: current `manifest.json`, `belief/runtime.py`,
`belief/reference_oracle.py`, `belief/online.py`, `tools/track_dynamic_online.py`,
`tools/run_mechanism.py`, `tools/evaluate_belief_replay.py`, current manuscript
Fig.4/caption/preview and renderer. The separate author memo
`review/CORRECTED_REFERENCE_AND_ANALYSIS_REVIEW.md` contains historical analyzer
findings; its old seven-arm comments do not describe the current nine-arm
manifest as deficient. Figure selection/design followed the local
`scipilot-figure-skill` and `scientific-visualization` skills.
