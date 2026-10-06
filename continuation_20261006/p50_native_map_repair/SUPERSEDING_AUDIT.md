# P50 superseding implementation audit

Prior P50 performance and STOP/PASS wording cannot authorize a scientific rejection or paper claim. Preserve all old files; rerun a native-space equal-budget teacher-on/off and spatial-vs-pooled factorial with explicit no-match rows.

- **IA-1 (critical)**: The old map comparison changes the image preprocessing from native detector resolution to a 256x448 direct resize; this is a backbone input change, not a module-only comparison.
- **IA-2 (critical)**: The teacher projection is trainable and used in the teacher loss, so the target representation can move with the student; this does not establish a fixed privileged-teacher causal effect.
- **IA-3 (critical)**: Training rows require an available teacher and at least one same-ID candidate. Events with no candidate or no prefix teacher are removed, so no-match/dustbin behavior is untested.
- **IA-4 (high)**: The old evaluation appends student and frozen-feature ranks through different continue paths, so equal matched denominators are not established.
- **IA-5 (high)**: The FPN arm copies selected lateral weights into a new pyramid, adds a normalization block, trains P2/refine parameters, and calls train(); it is not a frozen native FPN control.
- **IA-6 (high)**: The prior implementation audit checked checkpoint shape/finite values but did not test native-feature equivalence, no-match retention, teacher fixedness, BN statistics, equal denominators, or causal teacher-off controls.
