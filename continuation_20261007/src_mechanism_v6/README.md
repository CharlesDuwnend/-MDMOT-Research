# SRC mechanism triptych: historical semantics and continuous response

User direction: keep panel (c), replace the nearly binary 3×3 penalty table in (b), and use the grouped-bar "Historical semantics" form from the supplied reference for (a).

The user subsequently requested five classes in panel (a). Panel (a) now uses five explicitly specified illustrative classes: car, van, truck, pedestrian and other. Track evidence [0.70,0.08,0.12,0.05,0.05] and detection evidence [0.08,0.04,0.08,0.74,0.06] each sum to one. These are conceptual inputs modeled on the user's illustrative reference, not extracted dataset probabilities, not a stored five-class SRC belief, and not the numerical inputs to panels (b,c). Solid blue track bars and hatched copper detection bars show the contrasting class preferences. The actual SRC state remains human/vehicle, which is stated within the figure. Historical fine-class probability vectors are not available from the current saved two-group state.

Panel (b) sweeps observation probability continuously through the current disagreement equation. H-track and V-track curves use the case's limiting stored beliefs; the neutral curve uses the saved prior. Faint curves interpolate each limiting belief halfway toward the prior, and shaded regions cover the resulting continuum of responses. Their penalties are independently verified to halve exactly up to numerical precision. This interpolation is a property of the belief input and is not an extra reliability multiplier. Both prior entries are retained exactly, including floating-point storage rounding. Hollow circles locate the two real conflicting base-assignment pairs. Curves are analytic responses, not dataset samples or empirical statistics. Reliability r is not an input to disagreement or candidate-cost refinement.

Panel (c) is retained from v5: the same base and refined cost matrices, assignments, eligibility marks, palette, colorbar and placement. A pixel comparison of its preview region verifies that its body remains identical. The two equal-cost base optima and unique SRC optimum were independently audited in `../src_assignment_visuals/INDEPENDENT_CASE_AUDIT.json`. This selected calibration example illustrates a mechanism; it does not establish an average online tracking gain.

Verification covers all ten specified bar heights, the normalization of both illustrative distributions, all nine independently replayed real-case penalties, 1,001 points per response curve, neutral evidence and neutral belief, directional monotonicity, text bounds and layout, and the unchanged panel-c region. The paper manuscript and previous figure artifacts are protected by hashes.

```sh
python3 continuation_20261007/src_mechanism_v6/build_semantic_response.py
python3 continuation_20261007/src_mechanism_v6/build_semantic_response.py --export
```

The first command renders previews for review. Final PDF, SVG and 600 dpi PNG are exported only after the required checks pass, to `/home/chenhc/src_egia_icme_paper/figures/src_assignment_mechanism_v6/`. The figure and caption distinguish the specified five-class illustration in (a), the analytic response in (b), and the real calibration case in (c).
