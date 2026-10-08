# 现有 EGIA 减部件实跑

最新用户约束：baseline without SRC，gate 也不要。所有方案关闭 SRC 的代价修正、递归更新和状态初始化，类别置信度出生门控为 0；检测分数出生条件 0.6 保留。

统一保留现有 EGIA 的 fusion（0.4）、selective（margin 0.2）、权重、输入、来源池和跟踪配置。固定参数，未看新实验结果前声明以下四格：

|方案|实际减少的部件|保留的 EGIA 部件|
|---|---|---|
|R00 full EGIA|无；新实跑核对已有 SRC-off、gate-off full|全部|
|R01 去前景分类头|物理删除 foreground_model，f=1；source 后验只保留 U/E|覆盖分类头、来源一致性、fusion、selective|
|R02 去覆盖分类头|物理删除 coverage_model，c=0；source 后验只保留 U/C|前景分类头、fusion、selective；原 E 项一致性不再具备决策作用|
|R03 去来源一致性分支|删除 geometry/appearance 两个 cue 分类器及其一致性修正|前景、覆盖两个分类头、fusion、selective|

此处前景是目标/杂波类别，不是图像分割。分类头删除使用结构性退化 f=1 或 c=0，不是将输出替换为经验训练先验。U=未覆盖真实目标，E=已覆盖真实目标，C=杂波。删除覆盖判别同步取消 E 决策功能；不能把它解释为在同一三分类任务中仅减少模型参数。fusion 的 summary 三分类器仍保留，所以它仍提供其它类别的辅助证据；selective 仍可在不确定区保留原生出生。

每格真实运行 17 test-dev 序列、6635 帧，原始 GT 同一评分分母 229506。运行前 CPU 结构检查、真实 GPU smoke，随后 R00 先完成并要求 17 份预测与历史 full 完全一致，再允许 GPU0/1/3 并行运行减少部件的三格。全程不在推理中读 GT；同帧 detector/ReID 输入与已完成 baseline 比对；SRC-off 原生关联每个 seam 做完整方法对照。只产表格，保存所有零效应、退化和改善结果，不根据新结果调阈值或选择性隐去方案。

源权重和其它已有研究工作不改；本轮没有新训练，没有新三分类替代方案。旧 structural 草稿在用户澄清前未启动，记录为 SUPERSEDED_NOT_RUN。
