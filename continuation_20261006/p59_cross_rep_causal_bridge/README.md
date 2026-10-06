P59 tests a narrow cross-representation causal bridge on top of the protected
P23 current identity head. The P1 history cache is used only as a strict-past
pointer-based input. Run `src/run_crcb.py --mode support` before the GPU run;
only a passing support/CPU contract authorizes the 1,200-update fit15 training.

## Outcome

The corrected v2 run completed 1,200 fit15 updates on physical GPU3
(`GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a`, A100-SXM4-40GB). On frozen
calibration5 outputs, target-only macro R@1 was `0.47912456211781995` and CRCB
was `0.4645403896930788` (`-0.014584172424741138`, 0/5 pair wins). The gate is
closed and no P26 host attachment, official validation, or test read is allowed.
The startup invocation and duplicate preregistration-key defects were repaired
before this final result; `FAILURE_AUDIT.json` records three rounds with three
checks each.
