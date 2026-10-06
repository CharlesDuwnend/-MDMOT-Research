# P52 碰撞停止审计

P52/NCCA 的结果可以作为 native feature adapter 的训练诊断，但不能继续作为论文创新方法推进。SCI-ID 项目已经明确实现并审计了“pooling 前候选组成、正例移除、同基数负例控制、正例缺失 dustbin”的完整组合；这与 P52 的核心技术合同逐项重合。把输入从 identity map 改成 native ROI 向量，或者改名为 NCCA，不改变该碰撞。

更关键的是，SCI-ID 的预注册多对留出结果已经失败：pairs 29/30/32/39 的平均 Recall@1 delta 为 -0.0718，只有 1/4 对提升；MRR delta -0.0445、margin delta -0.1506、Brier delta +0.0132。该分支的停止码为 `STOP_SCI_ID_NOT_INCREMENTAL_OVER_ADJACENT_PRIORS` / `STOP_SCI_ID_BACKBONE_NOT_CAUSAL`。

因此决策边界是：

- `P52`：保留作 native adapter/compatibility 诊断，不接 official-val/test，不写成新的论文方法。
- 新方向：从 AutoAssign epoch-60 Caffe R50 初始化身份骨干，并在 C3/C4→identity pyramid 的 paired maps 上加入跨视角、尺度一致的 gated modulation；先做初始化与机制审计，再与 AutoAssign 初始化的 B1/B4 做相同预算的短留出训练。

证据：`/home/chenhc/mdmot_sci_id_20260906/N0_NOVELTY_REPORT_ZH.md`、`/home/chenhc/mdmot_sci_id_20260906/N4_MULTIPAIR_RESULTS_20260907.md`。
