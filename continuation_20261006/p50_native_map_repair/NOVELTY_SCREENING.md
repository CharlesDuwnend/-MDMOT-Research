# Native Set-Relative Residual: novelty screen (2026-10-06)

This is a provisional collision screen, not a novelty claim. The candidate takes a frozen native ROI feature for the target and a bounded cross-view candidate set, applies a leave-one-out set-relative residual before assignment, and uses a fixed strict-past multi-view feature only as a training target. At inference the teacher and labels are absent.

| Prior art | Mechanism overlap | Boundary / consequence |
|---|---|---|
| Dong et al., **Learning from multi-view fragments: An adaptive consistency distillation framework for occluded person re-identification**, *Neurocomputing* 676 (2026), DOI 10.1016/j.neucom.2026.133015, [primary page](https://www.sciencedirect.com/science/article/pii/S0925231226004121) | Training-only privileged multi-view teacher distilled to a single-view student | Directly adjacent LUPI/distillation. A teacher-on gain cannot be claimed as novel; any claim must depend on the MDMT candidate-set/no-match objective and causal set operator, and must survive a teacher-off control. |
| Liao & Shao, **Interpretable and Generalizable Person Re-Identification with Query-Adaptive Convolution and Temporal Lifting**, arXiv:1904.10424 / ECCV 2020, [arXiv](https://arxiv.org/abs/1904.10424) | Query-conditioned matching and feature-map-level matching | Adjacent matching operator; it raises reviewer-risk for any query-conditioned residual. The current native-vector gate is not a QAConv reproduction and cannot claim firstness. |
| Multi-scale QAConv, IEEE 2023, [primary record](https://ieeexplore.ieee.org/document/10219804/) | Query-adaptive local matching | Adjacent operator prior; a future spatial-map version must compare mechanism-level inputs, operator and output explicitly. |
| Set-relative / candidate-context ReID family | Candidate-set context can alter pair scores | Narrow claim only: a bounded MDMT K<=64/no-match assignment operator with explicit leave-one-out context. No broad “set-aware ReID” novelty claim. |

**Current decision:** HOLD novelty. Run the implementation-corrected causal gate first. A train-only diagnostic gain is not paper evidence; official-val/test remains locked until the gate, collision review, and baseline protocol audit pass.
