# P31 机制级先验审计

状态：`STOP_P31_DIRECT_COLLISION_WITH_P3`。以下审计只支持收窄 claim，不构成“首次/新颖”结论。

| 先验 | 输入 → 算子 → 目标 → 输出 | P31 边界与风险 |
|---|---|---|
| TCFNet（MDMT） | 多无人机时空特征 → temporal-guided cross-view fusion、object-aware/temporal-aware alignment → 检测/跟踪目标 → 融合特征与轨迹 | 对“时序+跨视图特征融合”是 Tier 1 同任务直接碰撞；CEIN 排除 broad fusion claim。 |
| MIA-Net（MDMT） | 检测框/特征 → local-global matching 与遮挡互补 → MDMT tracking → 匹配/轨迹 | 作为 P26 主机基线；几何 matching 不是 CEIN 的新颖点。 |
| P3 support/conflict evidence（本工作区既有 MDMT 实验） | 当前 local 的因果观测前缀 + 候选 owner/bag → 共享序列/关系编码，分离 support/conflict readout → 接入一致性与已知反例目标 → 双路证据与跨视图接入决定 | **Tier 1 同任务直接碰撞**。P3 已明确保留支持和冲突观测、mixed bag 监督及接入一致性目标；CEIN 的 leave-one-out 与 null/dustbin 不能把完整链重命名为新方法。P31 在真实特征 port 前停止。 |
| STCA/同类 MDMT 时序外观匹配 | 当前/历史外观与位置 → temporal aggregation/cross-drone matching → association loss → pair affinity | Tier 1/邻近风险；CEIN 不能声称“历史外观用于匹配”新颖。 |
| U2MOT | 观测与不确定性 → uncertainty-aware temporal embedding/tracklets → 无监督 tracking objective → tracklet/embedding | Tier 2；不能声称 uncertainty、tracklet 或 temporal embedding 本身新颖。 |
| MeMOT | 检测/轨迹 → memory encoding/decoding → tracking loss → memory-enhanced identity | Tier 2；CEIN 不使用“memory module”作为 claim。 |
| SCGTracker、GSM | 节点/边或轨迹 → spatiotemporal graph / graph similarity → matching/tracking objective → assignment | Tier 2；CEIN 不使用“evidence graph/graph matching”作为 claim。 |
| REGR（2026 预印本） | aerial detections → reciprocal evidence graph/agreement → MOT objective → graph-enhanced association | Tier 2，且是近期 reviewer 风险；“证据图/互证”不能单独申新。 |

## 原先候选边界（已被本地先例否定）

仅可进一步核验如下组合是否有直接同任务先例：

> 对每个严格过去的对象级支持集合，计算 leave-one-support-out 表征干预，显式输出每条支持的 influence 与集合质量，并将左右集合的干预证据和双侧 null/dustbin 联合用于跨视图配对。

该组合曾被作为 candidate composition，但 P3 的输入、support/conflict 干预目标和接入输出已经覆盖同一研究问题。因此 P31 立即 STOP；不得通过把 influence、quality、null 或 dustbin 加到已有链条上恢复 novelty。后续方向必须改变观测对象或学习目标，并重新做同任务审计。

## 来源

- [TCFNet official article](https://www.ejournal.org.cn/en/article/doi/10.12263/DZXB.20240727/)
- [MIA-Net/MDMT official repository](https://github.com/VisDrone/Multi-Drone-Multi-Object-Detection-and-Tracking)
- [U2MOT ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Uncertainty-aware_Unsupervised_Multi-Object_Tracking_ICCV2023_paper.pdf)
- [MeMOT CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Cai_MeMOT_Multi-Object_Tracking_With_Memory_CVPR2022_paper.pdf)
- [Online MOT cross-task synergy](https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_Online_Multiple_Object_Tracking_With_Cross-Task_Synergy_CVPR2021_paper.pdf)
- [SCGTracker](https://doi.org/10.1016/j.patcog.2023.110249), [GSM](https://www.ijcai.org/proceedings/2020/74)
- REGR（2026 aerial-MOT preprint，检索记录已保存在本轮审计日志中；其 reciprocal evidence graph 仅作为 reviewer-risk 先例，未将其作为可复现代码来源）
