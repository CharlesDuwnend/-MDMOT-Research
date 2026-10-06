# 保守新颖性边界

当前不提交 “new/first/SOTA” 表述。候选机制是把跨视角多目标跟踪的错误拆成 candidate coverage、candidate ranking、solver assignment 和 owner commit，并使用严格 prefix-causal、class-aware、paired-link 诊断。

这条贡献与普通 ReID、阈值调节、固定延迟 reranker、全局 ID 后处理和新的 association head 有清晰边界；但它仍属于多摄像头/多无人机跟踪评估与错误分解的邻近区域，不能仅凭概念相似性宣称首创。论文必须引用相邻的跨视角跟踪、图匹配、ReID 和评估协议工作，并把主张限定为：

> 在 MDMT 的合法 prefix-causal replay 中，提供可复现的事件级错误分解协议，量化候选覆盖与给定候选排序/owner commit 的分离，并把未来观测污染显式隔离为结果诊断。

这是协议贡献的候选措辞，不是已经完成的投稿新颖性结论。正式投稿前仍需做完整文献审查、独立实现复核和至少一个合法的后续 repair 方法。
