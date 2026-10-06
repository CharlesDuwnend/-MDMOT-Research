# P45 mechanism-level novelty screen

状态：`HOLD_NOVELTY_UNRESOLVED`

本审查只判断是否值得做 CPU 合约和真实接入，不授权“创新/首次”表述。P45 的 claim 必须收窄为：**严格过去的同视角目标关系配置被编码为 permutation-equivariant evidence，并直接决定带未匹配容量的跨视角整体 transport，再由连续 owner publication 接入 MIA**。

| 先例 | 输入 → 算子 → 目标 → 输出 | P45 的碰撞/差异 |
|---|---|---|
| MIA-Net / P26 | 两视角检测框与轨迹 → 几何映射、顺序 refresh、补框和 ID 写回 → tracking 输出 → view-wise boxes/IDs | Tier 1 主机基线。P45 只能替换实际跨视角决策，不能把几何或 owner continuity 当新颖点。 |
| TSMMT, TCSVT 2025, [DOI](https://doi.org/10.1109/TCSVT.2024.3478758) | 多无人机 tracklet/时空特征 → temporal-spatial feedback、temporal localization、cross-drone target/background interaction → tracking losses → 跟踪/身份 | Tier 1 同任务碰撞风险。“跨机关系+历史”宽泛 claim 直接 STOP；P45 必须坚持 relation configuration 到 global partial plan 的窄边界并核对全文实现。 |
| SAME-MDMT, CAC 2025, [DOI](https://doi.org/10.1109/CAC67268.2025.11486840) | UAV motion/位置/IoU → DMC + PIF → cross-view association → identity-consistent tracking | Tier 1/2 邻近。P45 不能声称 position/IoU fusion 或 motion compensation 新颖；其差异在整批 learned transport 和 relation signature。 |
| View-Centric MOT with Homographic Matching in Moving UAV, [arXiv](https://arxiv.org/abs/2403.10830) | UAV tracks/frames → fast homography + view-centric ID learning → MOT objective → homography-aware IDs | Tier 2，homography-aware identity 不是新颖点。P45 复用 homography 只作固定几何输入。 |
| Multi-Object Tracking Meets Moving UAV, CVPR 2022, [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Multi-Object_Tracking_Meets_Moving_UAV_CVPR_2022_paper.pdf) | detected objects → graph nodes/edges and relation-aware tracking → MOT loss → trajectories | Tier 2/可能同任务邻近。节点/边关系和 graph matching 不能单独申新；需要确认其是否已具备跨视角 partial transport 和连续 owner 状态。 |
| Multi-camera MOT global graph, [arXiv](https://arxiv.org/abs/1709.07065) | multi-camera observations → global graph / maximum multi-clique → cross-camera trajectories → identity trajectories | Tier 2 经典直接风险。P45 不得声称“整体匹配/全局图”本身新；候选仅保留 MDMT strict-causal relation token 与 online owner publication 的组合。 |
| FAMNet, [arXiv](https://arxiv.org/abs/1904.04989) | detections → differentiable affinity and multi-dimensional assignment → assignment supervision → MOT association | Tier 2 直接 operator precedent。独立 edge scorer、Hungarian 或 assignment loss 不构成贡献，故 B2 是强制控制。 |
| P31 / 本工作区 | causal observation prefix + support/conflict relation evidence + null/dustbin + cross-view decision | Tier 1 本地碰撞。P45 不得把 support/conflict、leave-one-out、dustbin、evidence graph 改名复活；关系签名必须是同一时刻的多目标配置结构，且输出是整体 plan。 |
| P32 / P43 / P44 / P38–P41 | owner-state revision、continuity repair、tracklet evidence、temporal residual/mean adapter | Tier 1 本地 controls。P45 不能靠 window、均值、后处理 relabel 或 tracklet aggregation 宣称新颖；其接口必须在原 refresh 前输出整批 proposal。 |

## 暂定判定

`KEEP_FOR_CONTRACT_ONLY`。候选有结构性差异，但已有同任务时空交互、图关系、跨视角几何和可微 assignment 先例，不能从名称或组合直接推导 novelty。若全文审计发现 TSMMT 或其他 MDMT 工作已经实现相同的配置关系到整体 transport 和在线 owner 状态，则转为 `STOP_DIRECT_COLLISION`。如果文献风险可收窄，仍需在完整 MIA 上证明它改变了真正的 association front door，而不是生成一批随后被 MIA 忽略的分数。

## 证据边界

当前检索核到了公开摘要/论文页和本地代码；尚未取得所有先例的训练代码、损失细节和运行协议。本文不把搜索摘要中“first”或“improves”当作可迁移结论，也不把 P45 CPU PASS 当成效果证据。
