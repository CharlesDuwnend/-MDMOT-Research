# P56: UAVST-HM direct prior control

UAVST-HM is a published MDMT method that fuses appearance, local target
distribution, and projected geometry for cross-view identity matching, then
solves a bipartite assignment. Its MDMT result is a direct same-task prior and
must be treated as a comparator rather than a novelty claim:
<https://www.mdpi.com/2504-446X/8/12/704>.

This phase first reproduces the paper's local appearance aggregation contract on
the frozen P23 calibration rows. For a target row, the local descriptor is its
P23 head-only identity vector plus the mean of its two nearest same-frame
neighbours in the same view. The control compares target-only and aggregated
descriptors under the exact same-frame, same-class, positive-present readout.
No threshold is selected and no host IDs are changed. This is a diagnostic
control, not a formal MOT result or a faithful claim about the paper's unseen
FastReID/geometry implementation.

If the direct control is useful, the next gate will add the paper's image-based
homography/geometry and one-to-one assignment in a separate P26-compatible
replay. If it is not useful, the prior is retained as a negative comparator.

