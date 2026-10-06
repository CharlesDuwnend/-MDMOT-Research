# P61 novelty screen

No first/novel/SOTA claim is made. The claim boundary is a narrow empirical one:
**pair-environment gradient alignment for a frozen MDMT cross-view identity head**.

| Prior | Mechanism | P61 risk and boundary |
|---|---|---|
| MLDG / cross-domain episodic ReID | meta-train/meta-test domains and gradient-based generalization | direct adjacent collision; P61 must not claim meta-learning or DG-ReID firstness |
| Meta Distribution Alignment, CVPR 2022 | meta-learning for generalizable ReID | same family; P61 is a restricted pair-local identity adapter with no test-time adaptation |
| IRM/BIRM | invariant risk / gradient penalty across environments | direct objective overlap; no IRM novelty claim |
| CCFI/P59 in this workspace | temporal/history residual and causal consistency | P61 removes history and changes target to cross-pair generalization |
| P60 / Sinkhorn and graph solvers | setwise assignment and transport | P61 does not alter candidate scoring with OT or solve assignment |
| MDMT-COD/P26 | cross-view candidate ranking and host association | P61 only changes the learned descriptor before the frozen diagnostic scorer |

This is a `KEEP_FOR_TRAIN_ONLY_RESEARCH` candidate only if the CPU contract passes and
calibration gain survives an independent audit. If it fails, it must be stopped as a
negative representation result rather than renamed.

Primary references:

- [Meta Distribution Alignment for Generalizable Person Re-Identification, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ni_Meta_Distribution_Alignment_for_Generalizable_Person_Re-Identification_CVPR_2022_paper.html)
- [Bayesian Invariant Risk Minimization, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Lin_Bayesian_Invariant_Risk_Minimization_CVPR_2022_paper.html)
- [Domain Generalized Person Re-Identification via Cross-Domain Episodic Learning](https://arxiv.org/abs/2010.09561)
