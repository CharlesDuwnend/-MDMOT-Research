# P46 novelty screen

状态：`HOLD_PENDING_FULL_AUDIT`

| 先例 | 已知机制 | P46 风险与收窄 |
|---|---|---|
| MIA-Net/P26 | bbox 中心/角点 + homography + local-global matching + supplement | Tier 1 基线。P46 只能替换几何观测，不能把 homography 或跨机 ID 更新称新颖。 |
| HomView-MOT, [arXiv 2403.10830](https://arxiv.org/abs/2403.10830) | fast homography、view-centric ID learning、跨视角 homographic matching filter | Tier 1/2。P46 不声称 homography matching 新颖；候选边界缩到 bbox vertical support 的潜变量和双向不确定性。 |
| SAME-MDMT, [DOI 10.1109/CAC67268.2025.11486840](https://doi.org/10.1109/CAC67268.2025.11486840) | drone motion compensation + position/IoU fusion for MDMT | Tier 1 同任务风险。P46 不使用 DMC/PIF，也不把位置/IoU 加权作为贡献。 |
| UAV airborne trajectory association, [CMU page](https://publications.ri.cmu.edu/trajectory-association-across-multiple-airborne-cameras) | canonical trajectories and inter-camera projective transformation | Tier 2 几何轨迹先例。P46 不宣称轨迹/投影变换新颖。 |
| Multi-drone tracking with localization uncertainty, [Drones 2025](https://www.mdpi.com/2504-446X/9/12/867) | temporal/spatial evidence and uncertainty-aware multi-view association | Tier 2/邻近。P46 必须证明学习的是 bbox 支撑线观测，而非普通 uncertainty gate。 |
| View/ground-plane multi-camera tracking | bottom/feet point mapped through homography is established | 直接算子风险。固定 bottom-center 只能作为必要控制；若 P46 最终退化为 bottom point，应 STOP novelty。 |
| P37/P42/P43/P44 | appearance/mean/geometry host transfer and owner continuity repair | 本地 controls。P46 必须在真正 MIA 前门生效，不能用后处理 ID 改写制造收益。 |

暂定：`KEEP_FOR_OBSERVABILITY_DIAGNOSTIC`，不授权训练和论文 claim。只有诊断显示“中心无法区分、支撑线稳定区分”，且公开同任务审计未发现同样的 latent anchor likelihood，才进入实现门。
