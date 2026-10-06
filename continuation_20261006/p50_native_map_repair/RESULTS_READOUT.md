# P50 native gate readout (train-only)

The implementation audit passed three rounds with three checks each. All 20 train/calibration pairs were loaded without dropped events; 5,234 fit events and 1,845 held-out calibration events were evaluated on the same rows for every arm. The frozen native cosine had pair-macro Recall@1 0.11596 and MRR 0.24426.

`rel_on_teacher_off` reached pair-macro Recall@1 0.22329 (seed 7) and 0.21640 (seed 17), gains of 10.73 and 10.04 percentage points over the frozen cosine, with four of five pair wins in both seeds. That is a reproducible trained-adapter feasibility result.

The leave-one-out set contribution over the equal-shape pooled control was only +1.10 pp on seed 7 (2/5 pair wins) and +2.37 pp on seed 17 (4/5), failing the preregistered 3/5-per-seed gate. The privileged strict-past teacher changed Recall@1 by -0.18 pp and +0.06 pp in the relational arm, also failing the +1 pp additive gate. These results do not support a teacher-distillation or set-context novelty claim.

Decision: keep the native candidate residual as a strong adapter control, hold P50 as a paper method, and test the counterfactual positive-removal/dustbin objective in P52. No official-val/test or MOT score is authorized by this gate.
