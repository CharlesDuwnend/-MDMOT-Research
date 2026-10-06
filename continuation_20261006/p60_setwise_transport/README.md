# P60 Setwise Entropic Identity Transport

P60 is a narrow trainable candidate built on the frozen P23 head. It applies a
fixed-temperature balanced transport operator to each same-frame, same-class
opposite-view candidate set, then learns a zero-initialized residual adapter with
the same setwise positive-mass objective. The calibration signal passed before
training (`0.5158327` versus the P23 target-only `0.4791246`, 4/5 pairs), and the
CPU operator contract passed. This is not an official MOT result and makes no
novelty claim.

The first signal script was named `signal.py`, which shadowed Python's standard
library `signal` during PyTorch import; it was renamed to `run_p60_signal.py`.
The first run also had a stale continuation path for the P23 checkpoint; the path
was corrected before the accepted signal. These are recorded implementation
findings, not scientific outcomes.
