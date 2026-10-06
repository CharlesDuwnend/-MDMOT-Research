# CCFI novelty and collision screen

The screen is conservative and makes no `first`, `novel`, or `SOTA` claim.

| Prior mechanism | Relation to CCFI | Tier |
|---|---|---|
| MeMOT / MeMOTR | learned spatio-temporal memory for MOT | adjacent/direct temporal-memory prior |
| ReST | spatial and temporal graph association for MC-MOT | adjacent multi-camera association prior |
| TransMOT | graph-transformer spatial-temporal association | adjacent association prior |
| Visio-Temporal Attention | temporal attention for multi-camera target association | adjacent cross-view temporal prior |
| camera/ReID uncertainty or score calibration | uncertainty-weighted identity matching | adjacent representation/calibration prior |
| P38 in this work | causal track-conditioned residual adapter | local baseline; CCFI adds explicit identity/change factorization and uncertainty objective |

The direct same-task/full-chain collision is not established by this screen, but
the surrounding mechanism family is crowded. The only defensible provisional
claim is a narrow, auditable hypothesis: **factorizing causal temporal change
from cross-view identity evidence and using the factorized uncertainty in the
pair score may repair the MDMT candidate-ranking bottleneck**. If the train-only
gate fails, the candidate is stopped rather than renamed.

Primary references consulted:

* [MeMOT, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Cai_MeMOT_Multi-Object_Tracking_With_Memory_CVPR_2022_paper.html)
* [ReST, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_ReST_A_Reconfigurable_Spatial-Temporal_Graph_Model_for_Multi-Camera_Multi-Object_Tracking_ICCV_2023_paper.pdf)
* [TransMOT, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Chu_TransMOT_Spatial-Temporal_Graph_Transformer_for_Multiple_Object_Tracking_WACV_2023_paper.html)
* [Visio-Temporal Attention, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Visio-Temporal_Attention_for_Multi-Camera_Multi-Target_Association_ICCV2021_paper.pdf)

Status before training: `HOLD_FOR_CPU_AND_TRAIN_FREE_GATES`.
