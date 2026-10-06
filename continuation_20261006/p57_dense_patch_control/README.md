# P57 dense patch control

This phase measures a frozen DINO patch correspondence signal on the five P23
calibration pairs. It is intended to decide whether a trainable local scorer is
worth implementing. Read the specification and preregistration before using the
result.


## Outcome

The final normalized-token control (v3) completed on physical GPU3 (A100-SXM4-40GB). Target-only macro R@1 was 0.4905143803303213; symmetric top-16 patch correspondence was 0.2351223863536386 (delta -0.25539199397668266), with 0/5 pair wins. The v2 missing-token-normalization defect was fixed before this final gate. `FAILURE_AUDIT.json` records the three-round audit. Training, official validation, and test reads are closed.
