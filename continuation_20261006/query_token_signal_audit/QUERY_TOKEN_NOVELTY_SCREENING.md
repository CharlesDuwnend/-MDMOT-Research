# AutoAssign query-token / role-whitening candidate screen

日期：2026-10-06。状态：`STOP_BEFORE_TRAINING`。本文件是基本机制筛查和负方向记录，不是 novelty/first 结论。

候选边界是：从冻结 AutoAssign Stage-3 cache 读取当前 target ROI 的 256-D `visual_feature` 和 285-D `query_token`，学习或固定一个跨无人机可靠性混合；另一个候选是按 UAV role 估计无标签均值/方差后做 feature whitening。推理不读 GT、local ID、future frame、homography 或 solver output。

## 机制级碰撞

| 先例 | 输入→算子→输出 | 本候选边界与结论 |
|---|---|---|
| P26/MIA-Net | ROI/框→局部/全局外观与几何匹配→跨视角 owner | 只能作为输入和完整主机，不能把外观/几何融合称新颖。 |
| CRAFT, TPAMI 2017 | camera/view feature distribution→view-specific feature augmentation→cross-view ReID descriptor | role-conditioned feature transformation 是直接同族先例；不能声称 camera normalization 首创。见 [primary paper](https://xiatian-zhu.github.io/papers/TPAMI17/ChenEtAl_TPAMI2017.pdf)。 |
| Camera-based Batch Normalization, arXiv 2001.08680 | camera feature statistics→camera-wise normalization→ReID representation | role mean/std whitening 与其直接重合；候选不能以“跨无人机统计校准”作独立贡献。见 [paper](https://arxiv.org/abs/2001.08680)。 |
| Camera-aware Re-ID feature for MTMC tracking, Image and Vision Computing 2024 | camera statistics/position→equalization and camera position encoding→tracking ReID feature | 同一 tracking 目标和相机差异已被明确处理；只能作为工程对照。见 [publisher record](https://doi.org/10.1016/j.imavis.2023.104889)。 |
| Camera-aware similarity consistency, ICCV 2019 | intra/cross-camera pair similarities→camera-aware consistency loss→ReID embedding | query/visual reliability loss 若只约束跨相机相似性，会落入已有 camera-aware metric family。见 [CVF paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Wu_Unsupervised_Person_Re-Identification_by_Camera-Aware_Similarity_Consistency_Learning_ICCV_2019_paper.html)。 |
| TrackFormer/JointTrack 类 query/temporal interaction | detector/query tokens→cross-frame or cross-view interaction→association | 将 detector query token 直接加入身份分数属于常见 feature fusion；不能把 token 来源本身作为创新。 |

## 无训练信号门

在完全相同的 train-only episode 候选集和分母上，4 个 primary holdout pair（29/30/32/39）得到：

* visual cosine：宏 Recall@1 `0.2047869`，MRR `0.3515088`；
* query-token cosine：宏 Recall@1 `0.1941831`，MRR `0.3447575`，相对视觉 0/4 pair 胜；
* 等权 visual+query：宏 Recall@1 `0.1982479`，仅 1/4 pair 胜；
* fit-only 的 UAV-role z-score：宏 Recall@1 `0.2124508`，2/4 pair 胜，未达到预注册的 3/4 gate。

全部事件 0 dropped，候选顺序和分母完全相同，未更新参数，未读 val/test。独立 3 轮、每轮 3 项检查见 `QUERY_SIGNAL_FAILURE_AUDIT.json`。

## 决策

`STOP_QUERY_TOKEN_AND_ROLE_WHITENING_BEFORE_TRAINING`。query token 没有独立排序信号，role whitening 既未过信号门又直接落入 camera-aware normalization 先例。保留 JSON 作为负方向和后续协议论文的可复核诊断，不写成论文主方法，不接 P26 tracker，不访问 official val/test。
