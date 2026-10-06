# P38 causal track-conditioned memory adapter

P38 is a train-only research candidate, not a novelty claim. It tests whether a
causal track-conditioned representation can close the held-out-pair transfer gap
without changing the detector or using future frames.

Input -> operator -> objective -> output:

`frozen P1 128D object embedding + same-view past embeddings keyed only by the
local track and frame -> shared current/history residual adapter -> cross-view
masked identity CE + causal temporal consistency -> normalized 128D target
representation`.

The local track ID is used only as an inference-time history pointer; its numeric
value is never an input feature and it is never used as an identity label. History
is limited to four observations strictly before the current frame. Missing or
unknown identity labels are masked from the CE objective; temporal consistency is
computed only against the causal history vector and does not read labels.

The fixed short gate is calibration pair-macro R@1 >= 0.4641, at least 3/5
calibration-pair wins over the frozen P1 independent embedding, and no fit15 loss
greater than 5 percentage points. Fit and calibration predictions are frozen
before the score pass. Dev and official val/test are forbidden at this stage.
