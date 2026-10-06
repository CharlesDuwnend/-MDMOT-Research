# P55 phase report

P55 tested a target-masked local context residual on top of the verified P23
head-only representation. The context vector was extracted from a 3x expanded
current-frame box after replacing the target interior with a local median. The
head-only adapter was frozen in the final v4 run; only the context residual and
dustbin were trainable. The actual GPU process was verified as GPU3,
A100-SXM4-40GB, UUID `GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a`.

The first internal fit holdout was positive, but it was not the preregistered
calibration gate. After fixing four implementation contracts (two train-time
and two candidate-pool evaluation defects), the five calibration pairs gave
target-only R@1 0.490514 and CVCR 0.438132, a -0.052383 delta and 0/5 pair
wins. The 3 rounds x 3 checks in `FAILURE_AUDIT_V4.json` independently confirm
the data split, frozen baseline/operator, and final metric replay. No dev,
official validation, or test data were read, and no P26 host replay was opened.

The candidate is stopped as a paper module and retained as a negative control.
The result does not invalidate context information in general; it rejects this
residual operator and training contract on the valid five-pair gate.
