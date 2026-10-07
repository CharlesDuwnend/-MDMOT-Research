# SRC cost-refinement triptych with the manuscript Figure 4 palette

The user requested selective use of the colors in `/home/chenhc/src_egia_icme_paper/main.pdf`, Figure 4. The palette is taken directly from the Figure 4 plotting source and checked against its recorded design evidence: `#FBF8F4`, `#F2DED1`, `#DBDCEA`, `#B7C6D8`, `#90A9C2`.

This variant applies that palette to the semantic-disagreement matrix and the before/after cost matrices in the existing real-case triptych. Numeric scales stay at [0,1]. Dark labels remain readable on every cell; muted slate and copper accents encode the selected matching edges, with solid/dashed outlines retained. The companion assignment-link figure retains its previous palette.

Data and claims are inherited from `../src_assignment_visuals/SELECTED_CASE.json` and `INDEPENDENT_CASE_AUDIT.json`: a selected real, closed calibration component with two equal-cost base optima and one unique SRC optimum. This is a mechanism example, not a population-level effectiveness statistic or a new online tracker evaluation. All 27 displayed cells are checked against the original case. No method equation, cost, eligibility threshold or solver choice is changed.

Preview first:

```sh
python3 continuation_20261007/src_fig4_palette/build_palette_variant.py
```

Export after visual review:

```sh
python3 continuation_20261007/src_fig4_palette/build_palette_variant.py --export
```

Outputs: `/home/chenhc/src_egia_icme_paper/figures/src_assignment_mechanism_v5_fig4_palette/`. The paper PDF, TeX source, and original v4 outputs are protected by before/after hash checks. `PALETTE_QA.json` stores palette, source, data and output hashes, grayscale ordering, cell-label contrast, and layout checks. The existing triptych caption applies unchanged to this palette variant.
