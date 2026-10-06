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

The accepted 1,200-update run used physical GPU3 (`NVIDIA A100-SXM4-40GB`, UUID
`GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a`). The independent 3-round, 3-check audit
passed. P23 target-only is `0.4791246`, fixed transport is `0.5158327`, and trained
SEIT is `0.5169382`; the learned residual contributes only `+0.0011055` beyond the
fixed transport operator. The branch therefore remains a validated engineering
control and is stopped for paper novelty and P26 host attachment.
