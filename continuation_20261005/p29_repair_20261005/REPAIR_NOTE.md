# P29 repair candidate

The sealed P29 run is retained unchanged. Static and synthetic review found a material coordinate defect in `residual_deformable.py::_flow_sample`: the flow raster was sampled at feature indices `0..W-1` and `0..H-1`, while the P26 offset-zero feature lattice is at detector-image coordinates `j*stride` and `i*stride`. The subsequent conversion divided by `stride`, so the reference displacement was wrong by a stride-dependent factor. P28's warp path used the correct `j*stride` domain; P29's query reference path did not.

This directory contains only the corrected operator source and the original protocol copy. It has no training, calibration, official-val, or MOT result. A rerun is required before drawing a method conclusion from P29/P30.

Repair cache root is `/raid/datasets/chc_data/claude_try_MDMOT_p29_repair_20261005`; it is regenerated independently because the original P28 archives were removed. GPU UUID for this run is physical GPU3 `GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a`; other A100 cards are also authorized by the user.

## Second implementation audit

The first coordinate-only rerun was stopped at step 600 after a CPU/MSDA counterexample showed that current-query warp validity had been reused as the raw past value token mask. The repaired source now separates the two domains: query endpoint validity remains diagnostic/warp-side, while raw value padding uses the source lattice image mask. See `../implementation_audit_20261005/P29_RAW_VALUE_MASK_COUNTEREXAMPLE.json`, `P29_TWO_DOMAIN_CONTRACT.json`, and the archived interrupted attempt.

The repaired three-arm FIT and calibration readout completed, but the preregistered learned-offset gate failed. See `REPAIR_DECISION.json`; P30 host integration was not started.
