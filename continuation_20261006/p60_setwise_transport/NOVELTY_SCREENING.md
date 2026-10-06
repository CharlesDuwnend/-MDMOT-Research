# P60 novelty and collision screen

No novelty or first-use claim is made. Setwise optimal transport and assignment
normalization are established in multi-object tracking and matching. Relevant risks:

| Prior / boundary | Collision risk | P60 boundary |
|---|---|---|
| MIA/MDMT candidate association and P26 | direct same-task scoring/assignment overlap | P60 is only a frozen diagnostic operator screen |
| MCTR, arXiv:2408.13243, https://arxiv.org/abs/2408.13243 | multi-camera track embedding and probabilistic association | no claim of a new cross-camera representation |
| FusionTrack, arXiv:2505.18727, https://arxiv.org/abs/2505.18727 | multi-view temporal fusion and association | no temporal module or end-to-end system claim |
| Sinkhorn/OT matching literature | direct operator overlap | any later work must claim only a domain-specific training objective if independently supported |
| MOTIP, arXiv:2403.16848, https://arxiv.org/abs/2403.16848 | track identity prediction from candidate/history context | P60 has no trajectory decoder or history input |
| P55/P59 local residual and causal-history branches | internal collision with stopped branches | P60 changes the target operator to group-level transport |

Decision boundary: if fixed-temperature transport fails the frozen calibration signal,
stop before training. If it passes, audit legal set support, no-match handling, and
one-to-one assumptions before any learned adapter.
