# Compact SRC figure layout

The user explicitly requested removal of the overall title, five-class subtitle, panel-A commentary, assignment-optimum callout, evidence-type footer, panel-B formula and faint-curve note, and a cleaner treatment of the gradient legend and other annotations.

This revision keeps only panel titles, axis labels and values, short track/detection and H/V-track keys, matrix labels, a matching arrow and one shared cost colorbar. Main data areas share a common top and bottom. The repeated track row labels on the right cost matrix are removed. All interpretation of faint curves, neutral belief, circles, box styles, and the fixed eligibility threshold is in `FIGURE_CAPTION.md` and its LaTeX version.

The disagreement equation already appears in the paper (`eq:penalty`); repeating it within panel B is unnecessary. The LaTeX caption cites that equation rather than putting another equation in the graphic. Human prior π(H)=0.332 is documented in the caption; the plot retains its prior tick and reference line.

The five-class values remain explicitly illustrative in the external caption. The implemented SRC state is still human/vehicle. The response curves are analytic; their weaker-belief and neutral properties are inherited from the exact saved prior and equation. The cost matrices, selected assignments and eligibility crosses retain the independently audited real calibration case. This layout change creates no new experimental result.

```sh
python3 continuation_20261007/src_mechanism_v8/build_clean_layout.py
python3 continuation_20261007/src_mechanism_v8/build_clean_layout.py --export
```

Outputs: `/home/chenhc/src_egia_icme_paper/figures/src_assignment_mechanism_v8/`. Publication exports are PDF, SVG and 600 dpi PNG, with reviewed color and grayscale renders. The receipt includes source and artifact hashes, all ten bar heights, all eighteen displayed cost values, nine replayed penalties, neutral and weaker-belief checks, requested text removals and layout checks. Existing paper and v7 artifacts are protected by hashes.

The existing repository allowlist excludes standalone `.tex` files. The exact LaTeX insertion template is also preserved in `FIGURE_CAPTION.md`; the local `.tex` copy is hash-indexed with the exported artifacts. No ignore policy is changed.
