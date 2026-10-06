# Cross-View Temporal Differential Co-Movement Field：创新筛查

日期：2026-10-06。状态：`STOP_CVTDCF_BEFORE_TRAINING_WEAK_DIFFERENTIAL_SIGNAL`。这不是 novel/first 结论，也没有授权训练。

## 候选边界

候选不是 aggregate-distance reranker、tracklet mean、普通 temporal memory、阈值或候选集改写。输入是当前 target/source ROI feature 与冻结 Stage15 support evidence 对应的 support-frame ROI feature。对每个 support offset 计算双视角一阶变化

`delta_v(tau) = norm(phi_v(tau) - phi_v(tau-1))`，

再构造跨视角 co-movement field `C(tau) = [delta_target * delta_source, abs(delta_target - delta_source)]`，通过 support-valid mask 和 elapsed-frame encoding 送入身份表征 head。训练目标仍是 pair-level same-ID / known-negative / no-match；推理输入不包含 GT ID、local track ID、future frame 或 solver output。缺失 support 必须显式 mask，不能补零、丢行或改变分母。

可检验的技术假设是：同一目标在两视角中的变化方向具有比静态 appearance mean 更稳定的跨视角协同性，而 hard negative 只在一侧变化时会产生不一致 field。必须用 static ROI、ordinary temporal mean、detached co-movement、random support order 和 zero-valid-support 对照拆开验证。

## 可观测性结果

`TEMPORAL_EVIDENCE_OBSERVABILITY_AUDIT.json`：25 个 train pair、8,311 events、226,550 candidates、1,056,682 support entries。通过 Stage1 stream 的 bbox/class 反查 feature row 后，2,011,092 / 2,113,364 support ROI rows 可取（95.1607%）；164,488 candidate entries 完整，62,062 个有至少一条缺失。block1 的 23/25/27/28→29/30/32/39 全部 support 行完整，因此可在 train-only primary block 做机制门；后续含 53/54/58 等 pair 必须先实现 missing mask 审计，不能直接扩大训练。

## 机制级碰撞风险

| 先验 | 风险 | 本候选允许的窄边界 |
|---|---|---|
| TCFNet（MDMT） | Tier 1：时序引导跨视角对齐/融合，不能声称 broad temporal cross-view fusion 新颖 | 只主张 support-frame **一阶变化 co-movement field**，不估计跨视角图像变换，不改 detector/geometry |
| MIA-Net / 当前 P26 | Tier 1 基线已有局部/全局匹配与几何更新 | 只作用于 identity representation，不改 solver、homography 或 tracker initialization |
| 本地 P38 temporal memory adapter | Tier 1/近邻：已有 temporal memory，且校准 gate 未过 | 不使用 memory pooling；学习 pairwise differential field 和 co-movement consistency |
| 本地 P44 tracklet owner evidence / P31 support-conflict | Tier 1：support evidence aggregation / influence 已碰撞或失败 | 不读取 owner/local ID，不做 leave-one-support-out influence，不输出接入决定 |
| MeMOT / MeMOTR | Tier 2：memory-augmented association | 不建 track memory，不把历史 embedding 作为 owner state；只读冻结 support ROI rows |
| ReST | Tier 2：spatio-temporal graph for multi-camera MOT | 不建 graph/trajectory solver；field 在 pair ROI feature 内计算 |
| GMT / cross-camera association | Tier 2：跨相机跨帧 feature fusion/association | 不生成 global trajectory、不使用 camera calibration 或 trajectory prediction |

TCFNet 的公开描述明确包含 object-aware alignment、temporal-aware alignment 和 cross-drone feature fusion；MeMOT 明确使用 large spatio-temporal memory 做 association；ReST 属于 multi-camera spatio-temporal graph。因而本候选只能作窄机制假设，不能使用“temporal memory / cross-view fusion / evidence graph”作为论文标题 claim。

## 训练前信号门

CPU 合同 `prototype/CPU_CONTRACT.json` 的三轮、每轮三项检查通过，但冻结特征的 train-only 信号审计没有通过：8 个 primary pair 上 static cosine 的正负差为 `+0.07743` 且 8/8 pair 为正；一阶 differential cosine 的差为 `-0.00157`，仅 3/8 pair 为正；差分 discrepancy 仅 5/8 pair 为正。独立重放见 `TEMPORAL_EVIDENCE_FAILURE_AUDIT.json`。因此不能把静态外观的可分性冒充 co-movement 的贡献，也不应为追求训练结果而改变阈值或放宽 pair 门。

## 进入机制门的条件

1. 先完成 CPU contract：offset 对齐、mask、support order invariance、zero-valid-support、无 GT/local-ID 输入。
2. 使用 block1 的 B4 对照，600 updates/arm；至少 3/4 pair strict Recall@1 wins，macro MRR/margin 不降。
3. 必须比较 static、temporal mean、differential field、detached field、random-order 和 missing-mask controls。
4. 只要碰撞审计发现同任务完整的 differential co-movement operator，立即 STOP；不得改名复活。
5. 由于训练前 differential 信号门失败，不授权真实 B4 adapter、短门或第二个 disjoint block；保留代码和审计作为负方向证据，不接 P26 tracker，不访问 official val/test。
