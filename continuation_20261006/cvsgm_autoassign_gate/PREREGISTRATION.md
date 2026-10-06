# CVSGM 短留出预注册

- run id: `short_gate_600_v2`
- GPU: physical `GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a` (A100-SXM4-40GB)
- initialization: AutoAssign epoch-60, SHA-256 `8894ea5ffe8309017d78e2dac359d405b38aa66725e1b465a8dce30beb12c901`
- train pairs: 23, 25, 27, 28
- eval pairs: 29, 30, 32, 39
- arms: B1, B4, CVSGM
- updates: 600 per arm; batch 4; channels 32; prototypes 2; ROI 7
- preprocessing: BGR; mean `[102.9801,115.9465,122.7717]`; std `[1,1,1]`; value scale 1
- formal scope: train-only representation holdout; no official-val/test or MOT metrics
- advancement: CVSGM must beat both AutoAssign-B1 and AutoAssign-B4 on at least 3/4 pairs by Recall@1, with no simultaneous macro MRR/margin decline
- failure: `HOLD_CVSGM_SHORT_GATE` or `STOP_CVSGM_NOT_INCREMENTAL`; no threshold tuning, solver change, selective pair removal, or post-hoc arm selection
