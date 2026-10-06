# P50 novelty screen (2026-10-06)

| Anchor | Overlap | Boundary decision |
|---|---|---|
| VLA-ReID | set-relative association features | Adjacent; P50 must remain a spatial-map residual and include no-match/positive-removal supervision |
| QAConv / SuperGlue-style correspondence | local pairwise map matching | Adjacent; P50 cannot be a pairwise matcher or a learned assignment solver |
| CAL / causal ReID | counterfactual feature intervention | Adjacent collision risk; do not claim causal ReID or causal discovery |
| TCFNet / TSMMT / UAVST-HM | cross-view temporal feature fusion | Adjacent MDMT/UAV collision; no temporal/global-ID memory in P50 |
| Learning from multi-view fragments (2026) | training-only privileged multi-view teacher for occluded identity | Tier-2 direct mechanism adjacency; the only defensible difference is map-level candidate residual + MDMT K64/no-match contract |
| DADA-Track / OASAMT | visibility/dual-memory association | Adjacent; P50 has no visibility classifier or inference memory |

**Status:** `N0_NARROW_HOLD`. No Tier-1 same-task/full-chain collision was found in
the initial screen, but the Tier-2 fragment-distillation overlap is material. P50 may
enter only a short mechanism/data gate; it is not a paper claim.

## Legal supervision

GT identity is read only while building training teacher prototypes and assignment
labels from the 25 configured train pairs. The candidate input cache remains GT-free.
The teacher is detached and excluded from the inference graph; no validation/test
fragment can be used to select a model or temperature.
