# P54 novelty and collision screen (conservative)

No `first`, `novel`, or `SOTA` claim is authorized. The candidate is adjacent to
several established mechanisms and may be stopped even if its train-only signal
is positive.

| Prior or local control | Mechanism contract | P54 boundary / risk |
|---|---|---|
| Privileged Knowledge Distillation for Online Action Detection (PKD, 2020) | future video frames are training-only teacher input; causal student at inference | direct LUPI precedent; P54 is a cross-view identity candidate-set residual, not action recognition |
| Multi-View Consistency Distillation (MVCD, 2026) | training-only multi-view identity teacher with labels, single-view ReID student | strong Tier-2 adjacency; P54 must report candidate ranking and MDMT prefix legality, not claim generic multi-view distillation |
| Future Temporal Knowledge Distillation (FTKD, AAAI 2026) | future-frame teacher distills features/logits to online 3D detector | adjacent future distillation; P54 targets cross-view identity assignment and no-match outputs |
| P50 fragment teacher | strict-prefix source donor teacher and candidate-conditioned residual | local direct precursor; P54 changes supervision to strictly future source evidence and evaluates all positive-candidate events, not donor-eligible-only selection |
| P38/CCSI temporal memory | strict-past same-view history and temporal consistency | local temporal representation control; P54 has no future input at inference and uses a distinct privileged target |
| P52/SCI-ID | pre-pooling candidate composition, positive removal, cardinality controls, dustbin | direct collision boundary; P54 cannot make positive-removal/dustbin the claimed contribution |
| MIA/FusionTrack/ReST/graph association | cross-view appearance/geometry or graph assignment | task-level adjacent prior; P54 keeps the protected MIA host and changes only a trainable ranking representation during screening |

Direct literature pages consulted:

* PKD: https://arxiv.org/abs/2011.09158
* MVCD: https://www.sciencedirect.com/science/article/pii/S0925231226004121
* FTKD: https://ojs.aaai.org/index.php/AAAI/article/view/38343
* Multi-camera MOT review: https://doi.org/10.1016/J.NEUCOM.2023.126558

The screen is a KEEP/HOLD decision only. A positive CPU or train-only signal is
mechanism evidence and cannot support a paper effectiveness claim.
