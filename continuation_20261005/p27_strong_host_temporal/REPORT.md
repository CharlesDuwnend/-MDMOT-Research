# P27 强主机时序证据与迁移决策

日期：2026-10-05。状态：root 已复核；最终文件清单与验证见 SEAL_RECEIPT.json。

**结论：保留 P28 的成熟 causal FGFA 结构试验；历史检测流与新原生检测器的严格 parity 仍为 FAIL（48/90）。** P27 的时间支持诊断可以解释为什么值得试验，但不能替代新原生检测基线、预告恢复量或证明方法有效。P28 必须使用新导出的原生 FPN 和与这些 FPN 对应的检测结果建立对照。

迁移对象是 FGFA（ICCV 2017）的稠密光流对齐与 embedding 加权聚合：当前及过去图像 → 同视图原始 FPN → current-to-past flow 反向采样 → 按时间归一化权重聚合 → 原 AutoAssign 检测头与当前帧检测监督。原论文使用双向窗口；past-only、AutoAssign/FPN、冻结检测器与 flow 的轻量训练都属于已知算子的明确迁移差异。它是成熟方法的结构 pilot，不能称为完整端到端 FGFA 复现、原创方法或 MOT 效果证据。机制比较、文献来源与接口风险见 [OPERATOR_REVIEW.md](OPERATOR_REVIEW.md)。

P26 的冻结 source、checkpoint、结果和 seal 保持原样；后续完整 MIA 主机仍须披露官方首帧 GT 初始化。新检测分支不读 GT 的推理约束，不会把完整主机变成 GT-free。P25 保持暂缓（DEFERRED）。本报告只汇总已有证据，不启动模型、训练、推理或重读原始数据。

| 证据门 | 结果 | 可以支持的判断 |
|---|---|---|
| 成熟算子/接口审查 | KEEP_CAUSAL_FGFA_PORT_FEASIBILITY | 因果稠密聚合具有可实现的检测前接口；原创性未成立 |
| 旧流与新原生检测器采样 parity | FAIL_SAMPLED_PARITY，48/90 通过 | 不能把旧流诊断当作精确 P26 原生检测证据 |
| FIT15 历史流时间支持诊断 | PASS_FIT_ORACLE_OBSERVATION_DIAGNOSTIC | 已保存的严格过去观察统计可信，足以保留试验动机 |
| 独立保存行重算 | PASS，868,562 行；128 个输入 hash 重查 | 因果历史、annotation-gap reset 与一对一关系已核验 |
| 原生检测/训练/校准效果 | P27 未建立 | 留给独立 P28 receipt 和冻结预测后的评分 |

采样 parity 使用 FIT15 × 两视图 × 首/中/末帧，共 90 图。原生 score_thr=0.05、NMS=0.6、max_per_img=100 保持不变，仅取新输出 score≥0.1 子集与历史保留流比较；预先固定的行序/bbox/score 绝对误差容差为 0.001。通过 48 图、失败 42 图，结论保持 FAIL。这个容差门不能称为逐位相等门，不能事后放宽容差改成 PASS。

[PARITY_LOOKBACK.json](PARITY_LOOKBACK.json) 的 Hungarian 近匹配回看记录：旧 6,052 行、新 6,053 行，其中 6,051 对达到记录的 IoU≥0.99 匹配条件；旧未匹配 1 行、新未匹配 2 行。接近的框/分数及排序变化与数值或运行时差异一致，**具体原因未确定**。近匹配只说明两者高度相似，不推翻严格 parity 失败，也不证明所有历史帧与新检测结果可互换。

时间支持诊断只使用 FIT15 的 30 序列、15,572 个视图帧观察和 868,562 条支持类别 GT 观察；历史 detector 流保留阈值为 score≥0.1。当前匹配 834,897，缺失 33,665，诊断 recall 为 96.124%。每个视图/局部原始 ID/类别独立处理；历史仅来自当前帧之前，annotation gap 后清空连续段历史。GT 对应关系仅供离线 Oracle 统计，不能进入部署特征。

| 严格过去窗口 H | 当前缺失且过去 H 内检测到同一 GT | 占当前缺失 | 占全部支持 GT |
|---|---:|---:|---:|
| 1 | 10,616 | 31.534% | 1.222% |
| 4 | 20,183 | 59.952% | 2.324% |
| 8 | 23,930 | 71.083% | 2.755% |

这些是窗口内“存在过检测”的计数，并不等于固定 lag 1/4/8 参考恰好含有有效证据，也不等于 FGFA 可恢复的框数。P28 选择的固定 lags 必须独立写入其协议，不能把 23,930 直接当成该三参考集合的覆盖量。

| 当前缺失类别 | 观察数 |
|---|---:|
| 第一次 annotation，无过去 annotation | 824 |
| annotation gap 后重新出现 | 48 |
| 当前连续 annotation 段尚未检测到 | 3,816 |
| 同一连续段曾检测到，但最近一次早于 8 帧 | 5,047 |
| 同一连续段过去 8 帧内检测到 | 23,930 |

最短 GT 边长 <16 px 的观察为 243,083，缺失 28,474，其中过去 8 帧内检测到 19,312；小目标仍是明显的适用性风险，不能以时序存在性替代可辨识性。XML 遮挡观察为 199,903，缺失 13,944，过去 8 帧支持 8,159；非遮挡相应为 668,659 / 19,721 / 15,771。annotation birth 总数 3,932，其中流起点左截断 1,646；annotation reappearance 为 135，均不应改称已验证物理进入/离开事件。按 pair、类别、遮挡与尺度的完整分层保留在 [visibility/summary.json](visibility/summary.json)。

跨视图附录共有 261,395 条同期、同标签且相同 raw ID 的共同 annotation；双方当前缺失 271 条，其中过去 8 帧任一视图支持 257、双方支持 161。它只采用 annotation convention，不验证物理全局身份、硬件同步或标定，不能用于推断可部署的跨视图补检能力。

独立验证按 local ID/class 分组、分割 annotation gap，再对检测帧号执行 strict-left searchsorted，重算全部 868,562 条记录的时序列；一对一约束、完整 miss-run 质量/区间和跨视图保存行引用同时 PASS。原始支持诊断的 receipt 为 PASS，且明确限制于旧 score≥0.1 流。该 PASS 与 detector parity FAIL 是两个不同命题，均予保留。

当前推进判断为 **KEEP_P28_ESTABLISHED_CAUSAL_FGFA_PILOT**。P27 不再把原 OPERATOR_REVIEW 写作时的训练 HOLD 当作当前授权状态；root 已单独安排 P28 的机制测试和受控训练，其可训练参数、固定预算、单帧与均匀聚合控制、校准预测封存及效果门以 P28 预先固定协议为准。P27 本身不赋予任何 official-val/test 推理或选择权限，也没有新 MDA/IDF1/MOTA。未来是否扩展到完整检测器/主机证据，必须依据 P28 实际检测结果，而非本阶段 Oracle 支持率。

本报告与 DECISION.json 已经 root 复核；DECISION.json 绑定本文及 10 个原始证据文件的 SHA-256。既有原始证据文件未修改，最终封存见 SEAL_RECEIPT.json。
