# VisDrone repeated-box motivation audit

The authoritative paper audit is `/home/chenhc/src_egia_icme_paper/scripts/build_duplicate_motivation_audit.py`. It verifies the Native and v5/LEAF prediction and annotation SHA-256 manifests before counting events.

The broad event used for the two bar charts is one valid GT target in one frame with at least two distinct prediction track IDs. Each prediction has IoU >= 0.5 with that GT and overlaps no other valid GT target at IoU >= 0.4. Predicted class is unrestricted, so class-confused repeated outputs remain visible. GT is read only after inference as a locator. Event keys are paired by `(sequence, frame, GT category, GT id)` before reporting resolved and residual counts.

The strict same-class audit is retained in `duplicate_motivation_audit.json` as the `primary` statistic. The paired result in `paired_resolution_audit.json` is the conclusion-figure source: broad Native 2,913 events, LEAF 415 events, 2,516 Native events absent under LEAF, 397 residual same-key events.

The paper directory contains the generated vector figures and candidate contact sheet. `analyze_and_plot.py` is an earlier exploratory implementation with a pairwise prediction-IoU variant; it is retained for provenance and is not the authoritative two-bar result.

## Four independent single-column figures completed on 2026-10-07

The final request in thread `01a11542-f887-7631-86c3-689689be96de` is implemented
by `build_four_figures.py`. Each dataset has one Native motivation figure and
one LEAF outcome figure. The four figures are separate PDF, SVG and 600-DPI PNG
files in `/home/chenhc/src_egia_icme_paper/figures/motivation/four_figures/`.
All use a fixed 88.9 x 91.44 mm canvas, absolute count labels, zero-based linear
axes, and consistent blue/teal/vermillion/gray colors. Hatching and direct labels
retain meaning in grayscale. The manuscript's `main.tex` is not modified.

| Figure | Content |
| --- | --- |
| `fig3_visdrone_motivation` | Five GT categories: 1,395 / 395 / 418 / 636 / 69; total 2,913 |
| `fig6_visdrone_resolution` | Native 2,913; duplicate event absent 2,516; paired residual 397 |
| `fig3_uavdt_motivation` | Five largest-count sequences: 833 / 123 / 94 / 88 / 75; other fifteen combined: 317 |
| `fig6_uavdt_resolution` | Native 1,530; single track 1,516; no related prediction 4; paired residual 10 |

The VisDrone residual 397 refers only to original Native event keys; LEAF's
total is 415, including 18 other events. Its 2,516 absent duplicate events have
not been classified as single-track outcomes. UAVDT outcomes are explicitly
partitioned so the four uncovered targets are not called single-track fixes.
UAVDT uses Protocol-B, original-GT test20, and frozen ignore-region preprocessing.
These are descriptive post-hoc counts, not new MOT scores or module attribution.

The builder rechecks all 115 input/receipt SHA-256 values and verifies event-key
partitions and category/sequence totals before drawing. Its receipt contains
export hashes, plotted data and automated text-overlap checks. The final PDFs
were independently checked for fixed page size and embedded TrueType fonts,
rendered with Poppler, and visually reviewed in color and grayscale.

Reproduce with:

```bash
python3 continuation_20261007/motivation_duplicate_analysis/build_four_figures.py
```

Compact evidence is in `FOUR_FIGURES_DATA.csv`, `FOUR_FIGURES_CAPTIONS.md`,
`FOUR_FIGURES_RECEIPT.json` and `FOUR_FIGURES_QA.json`. Standalone LaTeX
figure snippets are supplied as `FIGURE_CAPTIONS.tex` in the export folder;
their final numbering depends on insertion order.
