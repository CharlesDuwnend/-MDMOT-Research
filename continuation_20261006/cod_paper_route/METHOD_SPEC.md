# MDMT-COD：可审计的跨视角身份错误分解协议

当前可推进的贡献是一个**协议/诊断方法**，不是已验证的可训练模块。它回答的问题是：在固定检测器、局部 tracker 和跨视角 solver 下，错误究竟来自候选覆盖、给定候选排序、solver assignment、owner commit，还是由于标签/可见性不可判定。

## 输入

* 固定的两视角 MDMT train stream、检测/局部轨迹输出和候选集合；
* 事件级 source/target 配对与严格 prefix-causal 的 ready frame；
* XML 只在预测冻结后用于诊断标注，不能进入候选生成、排序或 commit。

## 算子

1. 先按 class、source prefix、visibility 和 track confirmation 对事件分层；
2. 单独记录 candidate recall；
3. 在给定候选子集内记录 Top-1/MRR，避免把“候选存在”误写成“身份解决”；
4. 再记录真实 solver edge、owner accepted edge、new/persistent false commit；
5. 以 causal lag-4 paired-link 指标检查跨视角链接，但把未来 co-observation 只作为结果诊断，不让它回流到因果状态。

## 输出

输出是事件轴、pair macro、错误提交持久性和 paired-link precision/recall/F1。它明确区分“可观测的错误比例”和“方法带来的因果性能增益”。

## 当前证据边界

25 个 train pair、8,311 个事件均为 train-only replay。candidate recall 宏平均为 0.9918146，给定候选 Top-1 为 0.0927080，给定候选 MRR 为 0.2149330；25/25 pair 存在 persistent false commit。没有训练、GPU、val/test、IDF1、AssA、HOTA 或 MOTA 证据。因此当前结论是“排序/commit 是主要可观测瓶颈”，不是“新算法提升了 MOT”。

## 论文使用方式

该协议可以作为问题定义、误差分解、审计协议和后续方法的统一 gate。后续任何新模块必须先在这套诊断上证明可解释的 causal repair，再进入正式 MOT 评估；当前不授权继续为 COD 训练一个被称为方法的模块。
