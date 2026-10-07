# SRC cost-refinement and assignment-disambiguation figures

The user requested matrix and assignment-based mechanism graphics, and supplied
a three-panel reference with semantic evidence, disagreement and cost refinement.
The new preferred figure follows that structure using a real closed candidate
component. A second, axis-free graphic shows the same assignment links.

## Recommended main-paper graphic

`src_cost_refinement_triptych.pdf` is a fixed 7.16 x 2.95 inch double-column
figure with:

1. The current stored beliefs and learned incoming evidence for three tracks
   and detections, represented by human/vehicle probability glyphs.
2. The exact semantic disagreement matrix
   `u_ij = clip(1 - b_i @ (z_j/pi), 0, 1)`.
3. The recorded masked base cost and refined cost side by side. Assignment
   outlines show the frozen host LAP choice; crosses mark costs over 0.80.

The supplied reference's reliability-times-disagreement description belongs to
a different mechanism. In the current source, responsibility `r` only changes
post-commit belief updates; it is not multiplied into current candidate costs.
The new figure therefore depicts the exact current semantic disagreement.
Class-level car/truck probability distributions are not invented: the current
head returns the actual coarse human/vehicle probabilities shown here.

## Concrete, verified case

The case is `uav0000222_03150_v`, calibration frame 177, first association stage.
The complete candidate pool has 41 tracks and 43 detections. Global track rows
25, 26 and 27 and detection columns 39, 40 and 41 form a **closed 3 x 3 legal
component**: no displayed node has a legal edge to an omitted node. The saved
costs and heads are unchanged. Beliefs are causally reconstructed on the frozen
H2 trajectory, exactly as in the preceding internal-mechanism diagnostic.

The base matrix has **two exactly tied optimal assignments**, each with cost
0.05967998337803937. They are `(T1,D1),(T2,D2),(T3,D3)` and
`(T1,D1),(T2,D3),(T3,D2)`. The host's deterministic LAP implementation selects
the crossed solution. Independent Hungarian solving is also permitted to select
the diagonal solution, because it is equally optimal. The figure explicitly
shows this ambiguity; it does not describe a uniquely wrong base optimum.

SRC increases `T2-D3` from 0.013473 to 0.988896 and `T3-D2` from 0.030749 to
0.991859. Both exceed the unchanged 0.80 threshold. The diagonal costs are
unchanged, and the owner-consistent diagonal becomes the **only** optimum of
this component. This directly illustrates the conditional operator claim:

> Semantic evidence resolves a base-cost ambiguity without changing the
> candidate set, matching threshold, or the costs of compatible pairs.

`src_assignment_links.pdf` is the same case in a fixed 3.5 x 4.75 inch
single-column layout. Track rectangles and detection circles are assignment
nodes, not workflow steps. Faint links show the alternative tied base pairs;
selected consistent and conflicting pairs have redundant color/line encodings.

All displayed values are rounded to three decimals. Exact values, original
IDs, geometry, observation provenance, beliefs and solver objectives remain
in `SELECTED_CASE.json`. Offline owner annotations are based on unique
class-agnostic geometry matches of the last committed observation and current
detection. Their GT classes are 1, 3 and 2 in the broader 10-class semantic
diagnostic contract. This is not a claim that the formal MOT class filter counts
three identity corrections.

## Selection, independent audit and limits

The search consumes all eight calibration sequences and all 11,274 association
records. It evaluates 3,750 nonempty first-stage calls. Base-versus-SRC
assignments change in 525 calls. Across 650 changed legal components, 105 have a
fully known favorable owner-score change, 153 have an adverse or mixed change,
and 392 have unresolved or unchanged owner scores. These descriptive counters
are retained in full; they do not establish an average tracking benefit.

The illustration is the first fully known favorable closed 3 x 3 component in
the deterministic calibration scan. It is selected to illustrate a mechanism,
not represented as a random or typical example, and no model or threshold is
selected using these results.

`audit_case.py` independently reconstructs all nine penalties, verifies the raw
full-matrix slice and its closure, reproduces the host LAP choice, enumerates
every legal partial matching and uses an independently augmented Hungarian
solver to check objective values. It confirms two base optima and one SRC
optimum. The display preserves the base tie rather than hiding it.

The reference is pre-semantic base `C`, not the historical host's separate
continuous-semantic native cost. These assignments are same-state operator
counterfactuals on the recorded H2 trajectory. No new online tracking run,
training, GPU operation, parameter tuning or formal MOT result is performed.
This case supports semantic disambiguation; it does not establish independent
effectiveness of responsibility-weighted temporal memory.

Outputs are in
`/home/chenhc/src_egia_icme_paper/figures/src_assignment_mechanism_v4/`:
independent PDF, SVG and 600-DPI PNG figures, color/grayscale previews,
standalone captions and artifact hashes. The existing manuscript and analytic
figure remain unchanged.

```bash
/home/chenhc/.conda/envs/u2mot/bin/python continuation_20261007/src_assignment_visuals/extract_cases.py
/home/chenhc/.conda/envs/u2mot/bin/python continuation_20261007/src_assignment_visuals/audit_case.py
python3 continuation_20261007/src_assignment_visuals/plot_case.py
python3 continuation_20261007/src_assignment_visuals/plot_case.py --export
```
