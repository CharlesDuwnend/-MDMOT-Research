# P57: bidirectional partial patch correspondence

P26's cross-view identity readout uses a pooled target descriptor. P57 asks a
narrower question: can a small vehicle's identity evidence survive if matching
uses a symmetric partial correspondence between DINO patch tokens instead of a
single pooled vector?

For a target/candidate crop, let `Q,C` be normalized patch tokens. The score is
the mean of the top-k row maxima in `Q C^T` and `C Q^T`. It is computed only for
the existing same-frame, same-class candidate pool. No labels, future frames,
or homography are used by the operator. This first phase is a frozen signal
control; it does not claim a trainable method or MOT improvement.

The intended trainable follow-up, only if the signal gate passes, is a compact
shared patch scorer with a learned visibility weight and an explicit cycle loss.
That follow-up must be screened against dense correspondence and cross-view
ReID prior art before any training.

