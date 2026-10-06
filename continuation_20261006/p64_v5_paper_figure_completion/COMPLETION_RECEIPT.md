# SRC-belief + EGIA-SOE v5 paper figure completion

Date: 2026-10-06 (Asia/Shanghai)

The paper source is external to this MDMT repository at `/home/chenhc/src_egia_icme_paper`.
No paper source or large image/PDF is copied into this repository. The paths and
hashes below are the final external artifacts verified after a clean two-pass
LaTeX build.

## Selected cases now embedded

- Fig. 1 motivation: pedestrian `uav0000297_02761_v`, frame 256, Native IDs
  167/232 versus LEAF 118; vehicle `uav0000355_00001_v`, frame 89, Native IDs
  50/58 versus LEAF 30. Continuous windows are 17 and 23 frames.
- Fig. 5 qualitative: vehicle `uav0000120_04775_v`, frames 694/707/720,
  Native 547/665 versus LEAF 236; pedestrian `uav0000073_00600_v`, frames
  154/163/173, Native 58 -> 58/98 -> 98 versus LEAF 76.
- Prediction panels contain evaluated prediction boxes only; GT boxes are used
  for after-the-fact verification and are not drawn.
- Fig. 3 uses the author-selected editable vector replacement
  `figures/method_overview/editable_figure.pdf`; the previous figure is retained
  in `.revision_history/before_fig3_temporary_overview_20261006.tar.gz`.

## Build and visual checks

- `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice: PASS.
- `pdfinfo main.pdf`: 10 pages, letter, unencrypted.
- `pdftoppm` rendered pages 2, 3, and 8; visual inspection: PASS. Fig. 1,
  Fig. 3, and Fig. 5 are embedded, legible, and unclipped.
- `pdftotext` caption checks: PASS for the selected IDs/sequences and the
  temporary Fig. 3 caption. No Overfull, undefined-reference, or rerun warning.
- No GPU inference or FPS experiment was started by this completion pass.

## External artifact SHA-256

- `main.tex`: `2d5adf93fd437a851b329a06e5f0b7ff68b8d5bb7c882f0ceaa5e18cbc754123`
- `main.pdf`: `d782f57e021b7130fdd02d85de0ef37e254e1defec9ba7d75c20d74bfcb00d62`
- `figures/motivation/motivation.pdf`: `c864e3acfbb11ff8e836ff8c241145b99b3c02be105e9dcebadfa3eda2ee9123`
- `figures/qualitative/qualitative_cases.pdf`: `718bb2a7db69d4e1726700463cd33f71e9534f8c0445a6d3c4b2094457ff44ec`
- `figures/method_overview/editable_figure.pdf`: `7b130f54a4db779c101ee81f7b1ebccd355ece964a53be3c7de5774a52b6db8f`

The independently audited five-arm evidence is retained at
`/home/chenhc/leaf_v5_evidence_completion_20261006_v1/reports/FIVE_ARM_FINAL_AUDIT.md`
with status `PASS_INDEPENDENT_FIVE_ARM_FINAL_AUDIT`; this receipt is a pointer,
not a reclassification of the test-dev evidence.
