# P30 full continuous host result

P30 sealed 3,900 frames across MDMT calibration pairs 27/32/42/64/65, generated
native FPN and 3-lag causal flow caches, and injected frozen detector arrays into
the official P26 MIA host. The two valid host runs used the AutoAssign config,
epoch-60 checkpoint, cap300 detector protocol, first-frame XML initialization,
ByteTrack/Kalman state, and MIA cross-view update code. They ran on the frozen
arrays through the strict P30 runtime; both exit markers are zero and neither log
contains a model-mismatch warning.

| arm | MDA (%) | IDF1 (%) | MOTA (%) |
|---|---:|---:|---:|
| native | 70.160338 | 90.967883 | 89.092826 |
| P29 past fixed offsets | 68.076413 | 90.671244 | 88.889807 |
| fixed - native (pp) | -2.083925 | -0.296639 | -0.203019 |

The fixed residual loses 2.083925 percentage points MDA overall. Pair 27 improves
by 2.680949 points, but pair 42 collapses by 11.009193 points; IDF1 and MOTA also
decline in the macro mean. This falsifies the P29 host candidate under the frozen
protocol. The earlier CARAFE-configured attempt is archived under
`host/attempts/carafe_config_v1/REJECTED.json` and is excluded from all scores.

The array-only operator diagnostic confirms that all first eight warm-up frames are
bit-identical to native. On pair 42, the later fixed arrays change the mean detection
count by +3.36/+1.73 per view, with mean matched IoU 0.959/0.952 and mean score
changes 0.027/0.032. These modest detector perturbations are enough to destabilize
the MIA cross-view association on that pair; the diagnostic reads no XML or labels.
It supports stopping the operator rather than attributing the failure to a missing
training seed.

This is calibration evidence only. It does not authorize official-val/test claims,
additional training, or a novelty claim. The detailed machine-readable decision is
`host/DECISION.json`; hashes and every valid JSON output are in `host/HOST_MANIFEST.json`.
