**只读参考增量核查 — 2026-10-04**

核查目录：`/home/chenhc/mdmot_research_20261002`。本次只读取代码、既有报告、日志及文件元数据；未训练、未提取图像、未运行任何 cal/dev/val/test 评测，未改动参考目录。36 个关键文件的当前 SHA-256、18 项 receipt→文件哈希核对及汇总算术记录在 [EVIDENCE.json](./EVIDENCE.json)，18/18 匹配。这里核对的是既有产物与实现，不是重新运行模型。

**结论：存在可靠的 held-out-pair 表征落差，但尾部 P6 的全族否定已经被实施复查推翻。** 普通 CE/triplet、VLA common suppression、pair-disjoint MLDG、局部轨迹 TEC 都已有实际试验；不能把这些重新当成新方向。没有找到本地 DANN/GRL/domain-classifier 的执行试验，但这既不赋予泛化对抗训练新颖性，也不支持“所有 domain adversarial 已失败”。尚未证实可用的新方法。

1. **准确的落差与协议。** P4 的 `RESULTS.json` 保存了逐 pair 与 pooled-query 两套数据。从逐 pair 已有汇总重新做等权算术，结果为：

| 固定通道，P4 相同候选支持 | fit15 pair-macro R@1 | cal5 pair-macro R@1 | cal − fit |
|---|---:|---:|---:|
| raw FPN256 | 18.5927% | 17.5051% | −1.0876 pp |
| P1 independent128 | 84.7626% | 44.4068% | **−40.3558 pp** |
| frozen DINOv2 CLS384 | 24.8970% | 27.1906% | +2.2936 pp |

P1 的 pooled-query 数字才是 **84.6742% → 42.9218%**；fit/cal 分别有 31,398 / 8,180 个 positive-present 查询。不能把 fit macro 84.7626% 和 cal micro 42.9218% 拼成一个落差。P1 全部 fit pair 为 71.1777–95.2311%，全部 cal pair 为 32.0866–54.5726%，两组逐 pair 区间确实不重叠。移除 unknown 候选的诊断宏平均仍为 86.0208% → 45.5168%，因此 unknown 竞争本身不能解释约 40 pp 落差。

实现合同为：同一 pair、同一 frame、同一 detector class、相反 view；所有方法共用 P2 track row 与 `feature_present & crop_valid` 的交集。主池保留 unknown 候选；只有已知 query 且存在同 pseudo-PID 正例才计 R@1；float64 cosine 的 exact-maximum tie 按正例数量/并列数量计分。GT 标签只参与离线 eligible/credit，未进入排序。它是 **positive-present 条件检索诊断**，不测自然 no-match，不是官方 MDA、MOT IDF1 或已验证物理 global-ID 性能。代码证据：[compare_crop32.py:46](/home/chenhc/mdmot_research_20261002/p4_diagnosis/compare_crop32.py:46)、[CONFIG.json](/home/chenhc/mdmot_research_20261002/p4_diagnosis/crop32_comparison/CONFIG.json)、[既有 RESULTS.json](/home/chenhc/mdmot_research_20261002/p4_diagnosis/crop32_comparison/RESULTS.json)。

2. **监督与划分。** 冻结 [ASSOCIATION_DIAGNOSTIC_SPLIT.json](/home/chenhc/mdmot_research_20261002/p0/data/ASSOCIATION_DIAGNOSTIC_SPLIT.json) 的原子单位是整对双视角及其全部帧：

| role | pairs | 合同 |
|---|---|---|
| fit15 | 23,25,28,29,39,44,45,51,53,63,66,69,70,74,78 | 仅此集合可拟合新 association/identity 模块 |
| calibration5 | 27,32,42,64,65 | 不拟合 encoder/decoder 权重 |
| dev5 | 30,50,54,58,76 | 已用于历史 P1E/P2，后来又有 P14 association confirmation；不再是全局未读集合 |

标签明确为 `same_label_equal_id` annotation-convention pseudo-identities，`physical_global_id_verified=false`、`scene_independence_verified=false`；所有 25 对以前均分析过。冻结 detector 来自 full MDMT train，故这只是新关联模块的 pair holdout，不能称整个系统的 unseen-detector/unseen-scene 基准。没有跨 pair 的同一物理身份标签；训练中的正负关系必须留在合法 pair/class 命名空间内，不能将不同 pair 的数字相等当正例。

P1 输入只有冻结 detector FPN256，经 `Linear→LN→GELU`、残差 FFN、LN、L2 normalize 得 128 维。P1 **早已使用**双向跨视角 masked matching CE：同 frame/class 的合法同 PID 为正例；unknown query/candidate 均从训练 CE 分母与目标中屏蔽；没有 observed positive 的 query 跳过，绝不成为 no-match。见 [模型:145](/home/chenhc/mdmot_research_20261002/p1/src/train_association_probe.py:145)、[损失:174](/home/chenhc/mdmot_research_20261002/p1/src/train_association_probe.py:174)。

P7R/P8/P9/P11 读取完整 fit15 P2 track-row 来源：868,102 rows，其中 868,077 feature-present、779,918 known-PID；这些是可采样来源总量，并非每次训练遍历的 row 数。组为完整同 pair/frame/class 两视角；正例与负例都可观测才进入监督组；未知 row 保留于输入，但没有身份监督。P7R 从 **已训练 P1 seed42/step4000** 初始化，随后 1,200 个单组更新，不能称从零训练视觉 backbone。

3. **已经做过的表征控制及精确边界。**

| 既有阶段 | 实际算子/学习目标 | 结果与可支持判断 |
|---|---|---|
| P7R | 同一 99,328 参数 P1；CE vs CE + 双向 batch-hard Euclidean triplet；正例最远、负例最近，限定对侧 view 且 unknown 被屏蔽 | cal macro 45.3311% vs 45.3412%，增量 **+0.0101 pp**；普通 hard-triplet 无明显增益 |
| P8 VLA | joint two-view FCAE basis + CAS suppression；CE + 0.5 × 两个 within-view common regularizer 平均 | cal 45.3667%，比 CE **+0.0356 pp**；132,352 参数，非参数量匹配；属于 VLA 机制适配控制 |
| P9 MLDG | 真二阶 `θ′=θ−0.01∇L_inner`，目标 `0.5(L_inner+L_outer(θ′))`；pair-disjoint vs same-pair placebo | cal 45.3379% vs 45.3300%，**+0.00789 pp**；fit 92.4590% vs 92.4749%；该成熟 DG 控制未修复落差 |
| P11 normalization | 对所评 pair/view 无标签 centering/ZCA；无训练 | 最好 view-centering 相对 P1 +0.8649 pp，ZCA 为负；只支持“这些一二阶归一化没有恢复”，不能证明所有统计偏移机制不存在 |
| P11 TEC | GT-free 同 view local-track 在 t−8/t−4 的连续性正例 + 当前同类不同 local-track 排他负例；CE+TEC / TEC-only | cal 45.5098% / 44.7975%；CE+TEC 比 CE +0.1786 pp、2/5 pair 提升；现有短门拒绝 |
| P10 patch path | frozen DINO patch 的双路线 partial transport `T_A C_now` vs `C_old T_B`，真实 t−8 history | true-vs-hardest-distractor **二选一诊断** cal 57.39%，single-time local matching 66.83%；不能把它换算为 R@1；路径一致性解释已被该试验拒绝 |
| P14 residual | cached frozen DINO CLS 经 LN+Linear 残差进入 P1；frozen_adapter 冻结 P1、仅训新支路 | cal 47.5915%，比 matched CE +2.2604 pp、5/5；参考记录还有 dev +1.18 pp、4/5，但完整输出 P15/P16 未确认收益 |

直接产物：[P7R decision](/home/chenhc/mdmot_research_20261002/p7r/DECISION.json)、[P8 receipt](/home/chenhc/mdmot_research_20261002/p8_vla/P8_RECEIPT.json)、[P9 receipt](/home/chenhc/mdmot_research_20261002/p9_meta/P9_RECEIPT.json)、[P11 normalization](/home/chenhc/mdmot_research_20261002/p11_diagnosis/P11_DIAG_RECEIPT.json)、[TEC decision](/home/chenhc/mdmot_research_20261002/p11_tec/evaluation/DECISION.json)、[P10 decision](/home/chenhc/mdmot_research_20261002/p10_falsification/DECISION.json)、[P14 result](/home/chenhc/mdmot_research_20261002/p14_fusion/evaluation_v2/RESULTS.json)。

**P9 不是只写了方案。** 每次使用同 4 个 fit pairs、每 pair 2 组，共 8 组；pair-disjoint 的 inner/outer 各来自互不相交的 2 pairs，placebo 的 inner/outer 都含同 4 pairs，且两臂暴露完全相同 8 组。99,328 参数、同 P1 初始化、同 3,000 更新；AdamW lr/weight decay 均 1e−4。代码 `create_graph=True`、无 detach；两臂都保留真实二阶梯度。[实现:36](/home/chenhc/mdmot_research_20261002/p9_meta/train_meta.py:36)、[分组验证:71](/home/chenhc/mdmot_research_20261002/p9_meta/train_meta.py:71)。既有 CPU gate 有有限差分最大误差 4.21e−10、namespace/unknown 重命名损失与梯度误差 0、alpha=0 与同八组 CE 平均等价；两臂日志最后都是 step3000，训练/launcher/evaluator exit 均 0。P7R CE 只有 1,200 更新，P9 receipt 正确将其列作不同预算参考；主因果对照是 same-pair placebo。

**对 generic pair-domain adversarial 的回答。** 搜索了本地实验 Python/Markdown 与设计 JSON，剔除文献快照、数据/大结果载荷；没有找到 DANN/GRL、CORAL/MMD、GroupDRO 或 pair-domain classifier 的实际训练实现/执行 receipt。`sources/representation_20261004/DomainBed__domainbed__algorithms.py` 含这些成熟方法的快照，不能算本地实验。已执行的是上表的 MLDG 等泛化/表示约束。因此准确状态是“**泛化正则已做，特定 domain-adversarial 试验未找到执行证据**”，不是“空白创新点”。

4. **失败后的实现回看确实改变了哪些结论。**

| 旧记录 | 代码/既有复核证据 | 本次采用的结论 |
|---|---|---|
| P6 FLD 已证实无效 | `HATReID_FLD_V1.py:84` 两个 1D 向量 `@` 得标量后广播进 S_B，应为 outer product；全维投影也不能代表 HATReID 机制 | 撤回 HATReID/FLD 全族 falsification；未自动授权重跑 |
| P6 “TransReID/FusionTrack hard cross-view triplet” | 实际为两层 MLP；triplet 函数无 view 输入，并取 first positive/negative；既有 checkpoint 分数可复现 | 仅保留该 MLP 试验的窄负结果，不能否定真实 FusionTrack/TransReID 或全部 rescoring |
| P7 草稿可作为负结果 | diagonal proxy 几乎配到不同 PID；unknown triplet；固定 first512 混 frame/class；评测破损 | 无有效旧 P7 训练结论；修复后的 P7R 才可用 |
| P3/P3B 双证据无效 | P3 有 83.36% positive donor bags 混入部署时禁止的 query own-local history；P3B 修复为实际 host 合法候选后仍 reject-all | P3 原失败含 regime defect；P3B 是修复后的短门拒绝，不能重新命名其 donor/history readout |
| P15/P16 control 相当于 continued CE | dispatch 把 control_ce 当成 P1-init，加载了错误 checkpoint；后已重跑修复 control | 修复后完整输出仍 NOT_CONFIRMED；P14 检索收益不等于 MOT 收益 |

证据链：[REASSESSMENT.json](/home/chenhc/mdmot_research_20261002/p7r/review/REASSESSMENT.json)、[P6 最小实现核对](/home/chenhc/mdmot_research_20261002/p7r/review/P6_CPU_MINIMAL_CHECK.json)、[FLD:84](/home/chenhc/mdmot_research_20261002/p6/HATReID_FLD_V1.py:84)、[P15 control correction](/home/chenhc/mdmot_research_20261002/p15_output/P15_CONTROL_CORRECTION.json)。因此 `CONTINUE.md:802–819` 的“两个机制 confirmed NULL / 唯一剩余 raw crop 路线”等属于被开头更新覆盖的历史意见，不能当约束执行。相同道理，`CONTINUE.md:176` 的“embedding axis exhausted”超出了已有有限模型/损失的证据。

5. **对下一步真正有用的约束与机会。**

- **应保留的问题：** 在合法 pair-local 跨视角监督下，学习出的目标视觉表征为何不能迁移到 held-out pairs，以及如何在新输入/算子/学习目标上改变这种现象。P1 本就有跨视角 CE，不能把“补上 cross-view loss”作为新意；pair-local label 命名空间与落差相关，但没有因果证据证明它是唯一原因。
- **尚未被这些试验排除的范围：** 本批 P7R/P9/P11 主要训练 pooled frozen-FPN 上的小 MLP，P14 的 DINO 也是缓存的冻结 CLS；这些结果不等于“可学习的空间对象表征、可见部分建模或真正训练视觉 backbone 都已失败”。但尚未证明其中任何新目标有效，也未形成训练授权。
- **不可重复包装：** 普通 domain confusion/MLDG/common-component suppression、CE/triplet、local continuity/exclusivity、P2 pointer、P3 donor readout、P10 双路线 patch cycle、P14 frozen-DINO residual 已有相应先例或本地试验。新方向要写清独立的 input→operator→objective→output、可识别的监督、最低退化解及相应机制对照，不能只改阈值、分数、ranker 或 module 名字。
- **视觉粒度机会有限但真实：** frozen DINO 单独弱，却在 P14 提供互补信息，否定了“单独检索弱等于毫无补充信息”。P10 single-time local matching 的二选一结果可作为空间证据线索，但不能复活已拒绝的 two-route history consistency，也不能冒充全候选 retrieval 增益。
- **协议边界：** 先按当前根任务批准的 CPU/fit-only 可观测性与机制审查推进。不能依据历史 CONTINUE 中的自主训练、GPU、cal/dev 指令启动下一轮；cal5 已经历多轮方法开发，dev5 也有历史曝光。任何后续正式确认必须明确这些事实，继续禁止物理 global-ID/官方指标暗示。
- **根任务 P22 dense-motion 的结果单独保留：** “267 screened ROIs 中 support × single motion direction 解释 99.77% centered-field energy”来自当前根任务，不来自本参考目录；本报告不将其与参考目录自己的 `p22_second_split` 混同，也不据其另造高自由度 motion 贡献。

6. **并行目录的实时状态。** 在本次 20:41–20:42（Asia/Shanghai）只读观察中，`tmux p20fit` 的父 PID 879961、worker PID 1211887 正在执行 `p20_global_fit15.py --tag p22_fit15_sf0.70 --split fit15`；父命令显式 `CUDA_VISIBLE_DEVICES=""`。`p20_run.log` 当时仅有 `EXIT_p22_fit15_sealed=0`，第二臂仍在运行，属于另线程的 **fit15 deferred-merge 补充**，不是新表征训练。该状态会变化；未接管、打断或触发任何该目录任务。`CONTINUE.md` 读取时为 819 行，mtime 2026-10-04 19:50:29 +0800，不能把其尾部历史内容误认作正在运行的最新表征候选。

**本轮决定：`HOLD_NEXT_REPRESENTATION_UNVALIDATED`。** 可以继续构造具有新学习目标的表征候选并做当前授权范围内的可观测性/实现审查；证据不足以宣布新方法已找到，也不足以把旧负结果扩张为“所有跨视角表征均无路可走”。
