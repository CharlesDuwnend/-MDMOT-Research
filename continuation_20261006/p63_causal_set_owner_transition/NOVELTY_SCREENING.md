# P63 机制级筛查

结论：`KEEP_AS_CANDIDATE; NO_NOVELTY_CLAIM`。

| 机制 | 已知边界 | P63 可保留的窄边界 | 决定 |
|---|---|---|---|
| RANSAC/加权 homography、DLT | 经典几何鲁棒估计和可微 DLT 已有充分先例 | 只研究 prefix support reliability 对 P26 owner transition 的作用 | 不能宣称几何估计新颖 |
| 位置代价/一对一 Hungarian | P47 已证明是强工程控制；多相机/MDMT 也有直接先例 | set assignment 只作约束，不能作为贡献 | 作为 fixed control |
| GNN/neural MOT solver | Brasó CVPR20、FAMNet、Deep Hungarian 等已覆盖图/分配学习 | P63 不宣称新 solver，而是 support confidence→projective state→owner 更新链 | 邻近风险高 |
| ReST/轨迹一致性/全局 flow | 已有空间-时间图、path consistency、全局 network-flow objective | P63 只保留严格 ready-frame causal support state | 需 pair-level 证据 |
| P47 geometry frontdoor | 同一 P26 host 上已有强几何前门 | P63 必须超过固定 residual/geometry control，不能重复命名 | attribution gate |
| P62 owner-state scorer | P62 训练后 calibration 几乎全拒绝 | P63 改为 support estimator + set transition，减少 no-link collapse | 独立候选 |

正式论文只能声称“在 MDMT P26 协议下的 prefix-causal support reliability and owner transition”，不能声称首个、通用新 homography、首个 GNN 或首个可微匹配。文献锚点包括 [Brasó CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Braso_Learning_a_Neural_Solver_for_Multiple_Object_Tracking_CVPR_2020_paper.html)、[FAMNet ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chu_FAMNet_Joint_Learning_of_Feature_Affinity_and_Multi-Dimensional_Assignment_for_ICCV_2019_paper.pdf)、[ReST](https://arxiv.org/abs/2308.13229) 和 [path consistency CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_Self-Supervised_Multi-Object_Tracking_with_Path_Consistency_CVPR_2024_paper.html)。
