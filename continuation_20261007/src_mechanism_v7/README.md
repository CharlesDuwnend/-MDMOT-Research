# SRC panel-A color revision

The user requested a better palette for the five-class panel (a), and an explanation of panel (b). This variant uses blue `#0072B2` for track evidence and golden orange `#E69F00` for detection evidence. Solid bars replace the previous copper hatching and dark outlines. The color pair has distinct grayscale brightness and follows the Okabe-Ito palette.

The five illustrative categories and probabilities are inherited exactly from v6. Panel (a) remains explicitly illustrative; it is not a measured five-class SRC state. Panels (b,c) keep their v6 colors and content, verified by pixel comparisons.

Panel (b) is an analytic response of the actual two-group disagreement operator. The horizontal axis is observation human probability; the vertical axis is semantic penalty. Blue and copper curves correspond to certain human and vehicle track beliefs, respectively. Faint curves move those beliefs halfway toward the saved prior and have half the disagreement response. A prior-like belief is neutral. Hollow circles mark the two conflicting real-case pairs used in panel (c). This explains selective, belief-conditioned penalties; it is not an empirical performance statistic or a test of the reliability head. Reliability is used in the post-commitment update, not as a multiplier on these curves.

```sh
python3 continuation_20261007/src_mechanism_v7/recolor_panel_a.py
python3 continuation_20261007/src_mechanism_v7/recolor_panel_a.py --export
```

Outputs are under `/home/chenhc/src_egia_icme_paper/figures/src_assignment_mechanism_v7/`. Source, data, previous outputs and final export hashes are recorded in `FIGURE_QA.json`. The existing v6 caption applies after replacing the reference to hatched detection bars with solid orange detection bars.
