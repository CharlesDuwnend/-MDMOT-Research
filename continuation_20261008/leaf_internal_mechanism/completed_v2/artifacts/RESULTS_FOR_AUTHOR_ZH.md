# SRC / EGIA 内部机制补实验：完整结果与论文使用建议

九个预先声明的条件均已完成真实在线推理和评分：VisDrone test-dev 的全部 17 段、6635 帧，每组同一原始 GT、检测器、ReID 输入流、宿主和评分器。总计 59715 个在线帧，153 个完整预测文件。九组同进程验证均为物理 GPU1 A100 40GB；未使用 GPU2。完整性复核见 FINAL_DELIVERY_AUDIT.json。

本轮完成三个问题：SRC appearance mask × semantic penalty 四格消融；EGIA fusion × selective 四格消融（完整配置与 SRC 四格共享一个条件）；同一预声明拟合配方的 hierarchical / flat source head 两组对照。因此总计九次完整运行，而不是十次。

## 可以支持的机制结论

1. **SRC 的两个代价算子都有正向条件作用，语义惩罚作用更大。** 相对保留 SRC 记忆和 EGIA、仅使用 native assignment cost 的 C00，mask 单独提升 MOTA/IDF1 +0.0623/+0.0367 pp；penalty 单独 +0.1625/+0.1584 pp；联合 +0.2074/+0.1880 pp。联合时 FP/FN/IDs 分别减少 205/262/9。mask 是几何与外观 min 融合之前的粗类别外观屏蔽，不能写成所有跨类边的硬约束；C00 不能标成独立原生 baseline。

2. **EGIA 的融合与选择性策略在各自两个条件下都有正向 MOTA/IDF1 作用，但有错误类型的权衡。** 开启 selective 后，fusion 额外 +0.0697/+0.0428 pp；开启 fusion 后，selective 额外 +0.2009/+0.1133 pp，FN 减少 2701，但 FP 增加 2103、IDs 增加 137。该选择性策略在不确定区间保留 native birth 决策；其改善主要体现为召回恢复与精度之间的平衡，不能说开启后所有错误都减少。这里消融的是 fusion × selective，coherence 固定。

3. **当前同条件配方不支持层级分解优于平坦头。** hierarchical−flat 的 MOTA/IDF1 为 −0.0732/−0.0515 pp，FP/IDs 增加 489/31、FN 减少 352。cal8 的源线索合入后加权 NLL 为 hierarchical 0.65547、flat 0.61981（越低越好）；校准诊断与 MOT 分开报告。不能用 canonical F00 与 flat 的差异替代该配对对照，也不能据此在 test-dev 改选模型。两个配对模型共享数据、scaler、事件权重和优化器设置；相同 C 不等于相同参数化/L2 几何。

两个四格的交互差均为负，表示条件收益有重叠/依赖；不要写成超加性协同或显著性结论。所有数字均为此冻结工作点的完整闭环比较，未进行 test-dev 调参、选优或模型再训练。

## 新图及各自要回答的问题

- `figures/completed_mechanisms/conditional_operator_grids.pdf`：四格+条件差，回答“改变内部算子，收益是否仍然存在？”左侧 SRC，右侧 EGIA；所有四格与方向保留。背景采用论文 Fig.4 的浅米色/灰蓝风格，没有伪装成 cost 数值。它适合作为正文的紧凑机制统计候选。
- `figures/completed_mechanisms/conditional_error_tradeoffs.pdf`：带符号的错误贡献矩阵，回答“改善由哪些错误项构成，又付出什么代价？”保留全部八个条件差和层级头的负结果。FP/FN/IDs 三项按共同 GT 分母换为 MOTA 贡献，三项之和等于 ΔMOTA；括号为减少的错误数，负数表示增加。IDF1 独立显示，不做错误项相加。

两图均有 PDF/SVG/PNG、灰度预览、图注、源代码与数据哈希，实际 PDF 排版已经检查。既有论文 main.tex/main.pdf/表格和冻结基线未改动；这里交付可审阅图与结果，不自动插入正文。

## 论文表述边界和仍未证明的内容

这些比较保持相同宿主与冻结模型，并且九组检测器/ReID 数组和顺序完全相同。因此结果能够支持内部开关本身的条件贡献，回应“效果只是因为 baseline 很强”的一部分疑问；它不能替代跨宿主迁移证据或证明普遍收益。

初始化、软/硬 responsibility、递归更新、源轨迹配对的独立有效性，尚不能从本轮四格推出。旧二十组结果保留真实评分及原始文件，但受历史 metadata binding 问题影响，标记 historical_metadata_contaminated / scientific_use=false，不加入此次机制差值；旧软/硬更新为零的观察也没有被删掉。若后续要强调这些机制，须另行做更正输入绑定后的独立对照。当前完整声明范围的九组补实验已完成，这些是额外的研究问题。

当前 runtime 日志只保存聚合计数与输入哈希，没有完整代价矩阵/同状态候选快照。因此不从日志编造真实 3×3 在线 cost 示例；旧校准图仍保持其局部示例性质。调用次数、非零 penalty、改变的 assignment 数也不等价于 GT 正确关联。

## 可直接改写进论文的结果段落

With all fitted models and host settings fixed, appearance masking and semantic cost modulation each provide positive conditional effects. Their joint use increases MOTA and IDF1 by 0.207 and 0.188 percentage points over the native-cost control, reducing false positives, false negatives, and identity switches by 205, 262, and 9, respectively. The native-cost control retains semantic memory and EGIA. Fusion and selective admission also improve the aggregate metrics within both policy contexts, while trading precision against recall. The paired hierarchical head does not outperform the flat head under the common fitting recipe; we therefore make no superiority claim for this factorization.

## 验证与复现

- 预先冻结：manifest.json / PROTOCOL_AMENDMENT.json；210 项生产源码、模型及依赖哈希逐项重验。metadata 修复沿用原冻结权重，没有重拟合。
- 47 项 CPU 合同、九组 200 帧工程检查、当前相同状态原实现 oracle/实际 LAP 与边合法性检查各自独立；它们不是跟踪提升数值。
- 九组完整预测与 GT SHA、逐帧输入 ledger、原始评分日志、全局与逐组回执、launcher/pipeline 实际退出 0 均已核验。GT guard 覆盖指定 annotation 根目录的 Python open/io.open/os.open，不把它夸大为全系统 I/O 审计。
- 两个关闭 fusion 条件缺少 arm-level 模型哈希字段，独立后处理副本从原 manifest 的 frozen_files_sha256 严格取值并验证实际模型与回执。原始生产文件、manifest 和实验设置均未改动。
- 完整数值、十个条件差、两个描述性交互、17 段/14 航次配对明细在 artifacts 下；航次行是片段差的描述性平均，不能叫 pooled-flight IDF1，不做帧独立显著性检验。
