# EGIA 内部机制结论（已完成真实推理）

所有新 EGIA 格关闭 SRC、类别置信度出生 gate 和 H2 fusion，保留同一 selective margin=0.2、检测分数出生条件 0.6、源分类头权重、detector/ReID/GMC 和原始 GT。四格均真实完成 17 序列、6635 帧，所有有序 detector/ReID 输入与历史基准相同；不加载旧 H2 分类器。

- pure v5 完整基准 57.1597 MOTA / 71.0347 IDF1；本轮 17 份预测与已完成 pure 基准逐字节一致。相同设置下带 fusion 的历史结果为 57.3196 / 71.0710，辅助 H2 头的增量仅 0.1599 / 0.0362 个百分点。
- 删除来源一致性分支（真实删除 geometry/appearance 两个模型）后，MOTA 降 0.4972、IDF1 降 0.5808；FP 增 3029、FN 减 2205、IDs 增 317。固定的前景/覆盖头仍存在，说明来源配对分支在这个 pure、无 SRC/无 gate 的运行点具有明确效果。它贡献的是误轨迹/身份控制，存在召回代价。
- 删除前景头后，MOTA 降 0.1076、IDF1 降 0.0683，FP/FN/IDs 分别增 133/103/11。前景头贡献较小，不应描写成主要增益来源。
- 删除覆盖判别后，MOTA 降 1.2322、IDF1 降 1.0986。更强的证据是其 17 份预测与 Native 基线全部逐字节一致，6595 次出生也一致，当前 selective 工作点没有任何不同于 Native 的出生动作。

最后一项不是“覆盖分类器参数独立贡献 1.23”的证明：c=0 同时取消 E 类动作，所以只修正 E logit 的来源一致性也失去决策作用。准确表述应为“覆盖判别功能及其关联的已覆盖目标抑制路径是 EGIA 当前收益的关键”。source cue 删除保留三类任务，因此来源分支的 0.50/0.58 消融更直接支持内部机制。

本轮没有公平比较二级层次分解与新训练单个三分类器，没有跨数据集验证，也未作统计显著性声明。沿用当前冻结 head 的 SRC-off 运行消融，与另行拟合的论文 Table III EGIA-only operating point 应保持区分。

论文当前仍含两条 lineage：Table III EGIA-only 无 summary fusion；joint row 和旧 Table V full 为带 summary fusion 的 F00。旧会话说“pure v5 不沿用 H2”对应 pure branch，不能自动推广到当前所有论文结果。论文的 full-SRC/no-fusion/selective-on 已完成分数 57.9553/71.3309，接近带 fusion 的 58.0251/71.3737；去掉 H2 头没有使 pure v5 崩掉。当前论文未修改，需统一最终配置口径后再替换其数字、正文及定性案例。
