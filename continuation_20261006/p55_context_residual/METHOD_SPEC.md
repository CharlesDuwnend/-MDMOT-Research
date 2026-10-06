# P55: Cross-View Context Residual (CVCR)

## Status

Candidate mechanism under train-only screening. This is not a validated MOT
method and has no permission to read dev, official validation, or test data.

## Problem and boundary

The protected P26 MIA pipeline obtains a cross-view candidate set from a
homography and then uses a target appearance signal. The current association
can fail when the vehicle crop is too small or occluded, while the nearby road,
lane, and object layout remains informative. P55 changes the learned identity
representation; it does not change the detector, ByteTrack, first-frame GT
initialization, homography estimator, candidate gate, or output scorer.

For a predicted box `b`, the input is the existing target crop `x_b` and an
expanded crop `x_e` from the same frame. The target interior is replaced by a
constant in the expanded crop, so `x_e` is a local context observation rather
than a second copy of the target. A shared frozen/finetuned image encoder gives
`f_t` and `f_c`. The trainable representation is

```text
r = normalize( P([f_t, f_c, f_t - f_c, fpn]) )
```

where `P` is a small adapter. Context dropout and view-swap controls test that
the model does not simply memorize camera-specific background. At inference,
only current target/candidate images and the current detector feature are used.
No XML ID, future frame, timestamp, GPS, or cross-view label is an input.

## Objective and output

For each predeclared paired fit group, the loss is bidirectional masked
cross-view InfoNCE over all legal same-frame candidates, with a hard-negative
margin term on the closest wrong candidate. A context-consistency term compares
the target representation under the original and context-dropout views. Rows
with no legal positive remain in the no-match denominator and are not silently
discarded. The output is a normalized identity vector and a calibrated
candidate logit; host integration is a later, isolated step.

## Falsification

The candidate is stopped before training if any of these occur: context-only
features outperform target-only features equally on both views (background
shortcut), swapping the context between candidates leaves the score unchanged,
the train-free residual has no positive signal on at least 3/5 calibration
pairs, or the short grouped train gate fails. A passing vector gate authorizes
only a representation audit and matched P26 host smoke; it does not authorize
formal MOT claims.

## Data split

The existing whole-pair fit/calibration partition from P23 is reused unchanged:
15 fit pairs and 5 calibration pairs. Pair IDs are the grouping unit; no frame
from a calibration pair is used for fitting. The source rows and image paths are
the frozen P23 prediction-crop manifest. The context construction itself reads
only images and predicted boxes and has no label-dependent branch.

