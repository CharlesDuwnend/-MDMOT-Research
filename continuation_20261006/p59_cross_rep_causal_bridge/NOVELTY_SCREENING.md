# P59 mechanism-level prior-art screen

Status: `HOLD_NARROW_CANDIDATE_NO_NOVELTY_CLAIM`.

P59 is not a claim that temporal fusion, track memory, cross-view ReID, or gated
association is new. The direct and adjacent risks are explicit:

| Prior | Overlap | Narrow boundary retained for the experiment |
|---|---|---|
| MOTIP, CVPR 2025, https://arxiv.org/abs/2403.16848 | Track history is used to predict identity | P59 keeps the frozen MIA/P23 current embedding and learns only a causal cross-representation residual; it is not ID-token prediction or an end-to-end detector. |
| MCTR, 2024, https://arxiv.org/abs/2408.13243 | Track embeddings and probabilistic multi-camera association | P59 does not replace the solver or maintain global track embeddings; it tests a small edge-representation bridge before the unchanged readout. |
| FusionTrack, 2025, https://arxiv.org/abs/2505.18727 | Multi-view temporal feature fusion and ReID | P59 is a frozen-detector, two-view diagnostic candidate with P1-history/P23-current cross-representation supervision; no broad fusion or MDMOT novelty claim. |
| Gated Temporal Fusion Transformers, WACV 2026, https://openaccess.thecvf.com/content/WACV2026/papers/Kim_Gated_Temporal_Fusion_Transformers_for_Robust_Multi-Object_Tracking_WACV_2026_paper.pdf | Gated history-to-current temporal fusion | P59 is not an encoder memory or Transformer and does not claim a new gate; this prior makes reviewer risk high. |
| P38/P39/P40 in this workspace | Causal local-track history and residual/mean controls | P59 differs only by coupling P1 history to the stronger P23 current head; this is a local research hypothesis, not a novelty claim. |

The candidate is retained only as a technically testable representation transfer
experiment. A positive calibration gate would authorize a full prior-art review
and host-level causal audit, not a paper claim. A negative or unobservable gate
stops training/host expansion.
