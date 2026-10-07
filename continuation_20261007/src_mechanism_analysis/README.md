# SRC mechanism and statistical figure selection, 2026-10-07

The user asked for a new mechanism/statistical explanation, with positive value
for the paper. The recommended claim is **bounded semantic intervention with
a measured benefit beyond the fixed birth gate**. Current manuscript equations,
exact runtime operators and the completed N0-N5 control experiment are the
authority. No flowchart, training, new inference, threshold search or manuscript
insertion is performed.

## Recommended main-paper figure

`src_bounded_intervention.pdf` has two vertically aligned panels on a fixed
88.9 x 121.92 mm single-column canvas.

1. The analytic panel answers a precise question: how much semantic conflict
   can an already legal candidate tolerate? SRC computes
   `u = clip(1 - sum_g b(g) z(g)/pi(g), 0, 1)` from the stored belief and learned
   observation evidence. Its cost is `C + (1-C)u`. For the frozen first-stage
   threshold `tau=0.80`, eligibility is equivalent to
   `u <= (tau-C)/(1-C)`. Smaller base costs tolerate more conflict; candidates
   closer to the host threshold tolerate less. This is a response of the
   existing operator, not a new threshold policy or observed edge sample.
2. The measured panel shows full SRC (N4) versus gate-only control (N1). Both
   use birth gate 0.70, the same detector/host/scorer, and disabled EGIA, over
   17 VisDrone test-dev sequences and 6,635 frames. The differences are
   139 fewer FP, 185 fewer FN and 27 fewer ID switches, corresponding to
   +0.152937 MOTA and +0.150353 IDF1 percentage points.

This connects a concrete operator property with a genuine controlled benefit.
It complements the existing semantic response heatmaps, and supplies a reason
for the operation rather than another diagram of execution order. The preferred
placement is the SRC mechanism or analysis subsection, before discussing the
full LEAF outcome. A concise narrative is:

> SRC converts semantic incompatibility into a bounded increase of candidate
> cost. Strong base matches tolerate more conflicting evidence than candidates
> near the association threshold. Under a fixed birth gate and disabled EGIA,
> the complete SRC intervention reduces false positives, missed targets and
> identity switches in the completed controlled diagnostic.

## Statistical alternative

`src_paired_sequence_effects.pdf` uses a fixed 88.9 x 132.08 mm canvas and
retains every sequence as paired MOTA/IDF1 change points. IDF1 improves in
13/17 sequences and MOTA in 12/17. Negative effects are present in the plot;
no subset is selected. This is suitable as supplementary evidence or an
alternative analysis figure. All changes are in percentage points. Aggregate
improvements come from pooled evaluator metrics, not the mean of these points.

A further alternative is a per-track timeline of observed class evidence,
belief and association costs around a verified recovery event. Existing saved
prediction cases do not expose those internal edge states, so that alternative
is not fabricated or represented by the current statistics.

## Interpretation and independent checks

- The analytic `C` is the masked pre-SRC geometric/appearance base cost. The
  N1/native shadow comparison additionally contains the host's historical
  continuous semantic rule. Boundedness is claimed against base `C`, not
  every entry of that historical native adjusted matrix.
- Per-edge eligibility is a necessary condition for a global assignment,
  not a proof that the edge is selected or correct. Responsibility affects
  post-commit belief updates; it is not a multiplier of candidate cost.
- Full SRC changed 888/14,700 nonempty association calls (6.0408%) against
  the native shadow assignment on the same current state. Empty calls are
  excluded from this denominator. These are calls, not frames or corrected
  ground-truth associations. All calls including empty ones total 19,905.
- This native diagnostic provides a small positive aggregate effect beyond
  the gate. The gate's contribution is not credited to the learned SRC cost.
  The independent update effect is not established: update-only N3 is
  prediction-byte-identical to N1, and full N4 versus cost-only N2 has
  -0.000871 MOTA / -0.000337 IDF1 pp with identical switch totals. All six
  controls and these limitations are retained in `ANALYSIS_RECEIPT.json`.
- Prediction/model/metric/GT hashes were verified against frozen receipts.
  Runtime flags and counters were checked directly, as were scorer-error
  totals, MOTA reconstruction, matched GT contracts and the exact operator.
  The figure does not claim specific cross-human/vehicle errors were corrected.
- These are completed frozen diagnostics, not new formal benchmark results,
  a significance claim, or a new model selected on test-dev. UAVDT is not
  forced into a semantic mechanism plot because its vehicle-only contract
  makes the semantic association branch neutral.

Exports are in
`/home/chenhc/src_egia_icme_paper/figures/src_mechanism_analysis_v2/`:
independent PDF/SVG/600-DPI PNG versions, grayscale previews, captions and
SHA-256 manifests. The main paper remains unchanged. Preview rendering and
all-text bounding-box checks precede vector export; final PDF renderings and
fonts are verified separately.

```bash
python3 continuation_20261007/src_mechanism_analysis/build_src_analysis.py
python3 continuation_20261007/src_mechanism_analysis/build_src_analysis.py --export
```
