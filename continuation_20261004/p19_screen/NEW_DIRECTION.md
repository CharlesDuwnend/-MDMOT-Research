# P19 后续候选：相机变形与目标自身运动的结构化分解

状态：**值得做真实可观测性短门；尚未确认方法、创新性或有效性。** 本文没有启动模型、训练、阈值搜索或新评测。技术目标是学习可迁移的目标表示，阈值标定不能充当方法贡献。

先纠正此前建议：四观测 patch partial-transport / 两路径一致性门已经在 P10 实施，并修复共享数组 aliasing 和 P1 normalization。当前 `p10_falsification/DECISION.json` 的 cal5 discrimination 为单时刻 0.668275、时序路径 0.573878，差值 -0.094397，0/5 pair 改善。此前把它称为未实施是参考了过时的 `sources/spatial_route_20261004/ASSESSMENT.json`。**撤回该推荐，不再通过温度、dustbin、entropy、窗口或阈值变体复活 P10。** 数字来自本次读取的封存结果，未重新跑 P10。

## 先排除已经做过的机制

| 阶段及本次读取的原始实现 | 已经做了什么 | 本候选必须增加的内容 |
|---|---|---|
| `p2/src/temporal_model.py`；`p2/DESIGN.md`、`DECISION.md` | P1-128 visual token + view/age → 当前/历史 attention → 运行时 owner/NEW pointer 与 birth head；independent/joint 只改变可见性；已训练并回放完整输出 | 再提出 joint causal ID decoder 是重复。必须先改变观测与可学习表示，不能仅换记忆/attention/ID head |
| `p3/src/evidence_model.py`；`p3/IMPLEMENTATION_CONTRACT.json`、`DECISION.json` | 独立128D前缀 GRU → 关系 token → single/dual support-conflict readout；因训练/运行候选合同错配而停止 | 新候选不重新包装证据双分支或混合bag读出 |
| `p3b/CONTRACT.json`、`DECISION.json` | 修复合法候选后保留同一 GRU/readout；固定训练和preflight未支持效果，结论限于实现 | 需要当前视觉向量中不存在的原始运动观测 |
| `p4_diagnosis/compare_crop32.py`（由P10及P6调用合同核对） | 固定 predicted-box crop pool 上 P1/FPN/DINO 表征诊断 | 不再把更换 pooled vector 或扩大候选当新机制 |
| `p5/CONTRACT.json`、`DECISION.json` | 用离线正集限制候选，导致单类集合上的 AP/AUROC 空结论 | 禁止按GT正例净化推理候选、背景支撑或时序历史 |
| `p6/FUSIONTRACK_REID_HEAD.py`、`HATReID_FLD_V1.py` | 冻结128D向量后的MLP/ID+triplet或FLD再评分；未消费原始背景流或目标运动场 | 必须超出同一冻结向量的投影/重评分；P6代码缺陷需要单独审计，不能当完整强方法否定证据 |
| `p10_falsification/run_minimal_falsification_v1.py` | DINO patch相似度→带dustbin的行随机对应矩阵→两条时空路径重叠分数 | 不重复patch矩阵路径一致性；本候选采用真实像素位移、背景扰动与目标运动重建 |
| `p19_merge_geometry/build_tracklet_geometry.py` | 整段bbox中心差、尺度比、时长等统计；运动更新为`pass` | 原始流场、因果相机变形估计和目标残差尚未在这些已读实现中出现 |

这是对已读实现的有界判断，不能据关键词搜索断言整个历史工作区从未做过任何几何实验。

## 候选的输入 → 算子 → 学习目标 → 输出

**输入。** 同一合法检测前端的当前预测框，两视图截至当前时刻的短原始视频片段、逐帧像素位移场及局部背景对应。无需相机GPS、外参或物理3D真值。背景估计排除所有预测目标框；查询框仅界定需解释的目标区域。目标历史可以从当前框内像素沿已观测位移反向追溯，避免把既有tracker ID当身份真值；所有源帧仍必须 ≤ 当前时刻。遮挡、错误光流和漏检不是已解决条件。

**算子。** 建立有结构约束的双因素运动场：背景支路只从预测框外观测估计每个视图的空间变形 `W_bg`；目标支路编码框内实际位移减去该位置背景预测后的残差场 `u_obj = flow_obs - flow_bg`，同时保留残差的空间布局与噪声/可见支持。相邻静态背景的跨视图对应提供局部微分映射 `J_AB`，用于迁移位移向量，而不是将两个图像中心直接相减。比较对象在不同视图中的运动需要 `J_AB u_A` 与 `u_B`；绝不能因为已去相机运动就默认它们处于同一坐标系。对目标及其候选的对应关系必须从背景变换估计中留出，防止用待证明的身份关系“校准”自身。

**学习目标候选。** 在fit-only数据上联合约束：(1) 框外背景位移的留出重建，使相机因素解释共同变形；(2) 目标残差场的跨时间预测/重建，使目标支路不能以全零或完全忽略运动过关；(3) 在独立估计的局部微分映射下迁移目标残差，让同对象在两视图的动力学表达相容；(4) 保持目标自身残差、替换相机变形因素时，目标表示应稳定，而只替换目标残差时应有可检测变化；(5) 用合法的同类不同对象/同约定身份监督约束身份分辨力，未知标签mask。背景扰动合成只能提供机制合同监督，不能冒充真实跨机测试结果。普通ID CE、域对抗或几个loss相加不足以证明成功分解，需观测干预验证。

**输出。** 每个观测的目标表示、可解释的局部运动残差及其支持/不确定性。只有该表示通过真实数据门后，才接入固定关联/ID后端作完整输出验证；当前不提出新的候选ranker、阈值gate或永久ID合并策略。这里的技术核心应是受观测约束的相机/对象分解与跨视图可迁移性，不是将已有外观和速度向量送入普通temporal fusion。

## 与至少五个原始机制的对照

| 原始来源与本次证据级别 | 实際输入/算子/目标/输出 | 对本候选的约束及尚待证明的差异 |
|---|---|---|
| **BoT-SORT GMC**：本地upstream-derived实现 [`gmc.py:127`](/home/chenhc/botsort_for_u2mot/tracker/gmc.py:127)、`:221`、`:239`；未重新核验该副本commit | 排除检测框的背景关键点或稀疏光流→RANSAC `estimateAffinePartial2D`→单视图相邻帧2×3相机变换 | 背景估计相机运动与补偿已经有实现。本候选若只是调用GMC再拼速度即没有足够差异；需目标残差的可识别分解与跨视图局部迁移 |
| **STCA**：原文全文 [`direct_stca_pdf.txt:330`](/home/chenhc/mdmot_research_20261002/sources/direct_stca_pdf.txt:330)、`:375`、`:422`；[DOI](https://doi.org/10.3390/rs16203861) | T-LightGlue旋转角复用与RANSAC两图变换→共同视野polygon→映射后中心点local matching；跨机部分仅inference | 两视图配准、共同视野和几何近邻关联都已有。候选必须胜过正确实现的该类几何对照，不能拿P19原始坐标差当几何基线 |
| **GTA-Net**：原文全文 [`direct_gta_net_pdf.txt:329`](/home/chenhc/mdmot_research_20261002/sources/direct_gta_net_pdf.txt:329)、`:374`、`:438`；[DOI](https://doi.org/10.3390/drones9040300) | 关键目标节点与连线采样外观/背景/位置边特征→edge triplet→RRWM/Hungarian→由可靠对应迭代估计homography/affine→目标对应 | “集体几何/目标邻域图”并不新。候选不能仅换成图网络；其独立背景支撑、真实目标动力残差与未参与估计目标上的迁移需要单独证实 |
| **ReST**：原文全文 [`e2e_ReST_paper.md:71`](/home/chenhc/mdmot_research_20261002/sources/e2e_ReST_paper.md:71)、`:107`、`:765`；[原文](https://arxiv.org/abs/2308.13229) | 同步静态标定相机，框底点投影公共地面→位置/速度差+appearance→空间/时间GNN及图重构→global tracks | 世界坐标位置/速度关联成熟。候选差异仅在移动无标定条件下从图像估计可迁移残差及其可辨识条件，不能声称首次加入速度/图/跨时空一致性 |
| **HomView-MOT**：本次读原始arXiv摘要 [`direct_homview_arxiv.txt:129`](/home/chenhc/mdmot_research_20261002/sources/direct_homview_arxiv.txt:129)；[原文](https://arxiv.org/abs/2403.10830) | 单移动UAV，FHE相邻帧单应→VCIL view-centric身份学习；HMF将框映射共同平面算physical IoU | 相机运动/单应辅助身份特征已有。单相机是Tier 2强机制近邻，不能仅因此停止双无人机候选；全文学习细节仍待补，当前不宣称分解目标完全不同 |
| **SAME-MDMT**：归档出版商原始摘要 [`direct_archived_ieee_abstracts.md:38`](/home/chenhc/mdmot_research_20261002/sources/direct_archived_ieee_abstracts.md:38)；[DOI](https://doi.org/10.1109/CAC67268.2025.11486840) | Drone Motion Compensation作tracking correction与spatial alignment；Position-IoU Fusion作跨机匹配 | 直接同任务强近邻。“相机补偿+几何关联”宽泛贡献已不成立；因只有摘要，不能编造其flow分解/损失细节，也不能确认完整链碰撞。候选新颖性必须保持HOLD直到读到全文 |

还需在立项前补充相机/对象运动分解、scene flow、动态场景视觉里程计与等变表征的原始文献审计；上表只是本次有界对照，不是全领域新颖性清单。

## 最短可证伪门：先验证这个物理量是否真的存在

1. 固定现有fit/cal整pair边界和32 anchors；先选固定fit子集形成观测工件，只访问official-train。读取原始图像以取得此前未用的背景/目标位移；此备忘尚未执行。保留原始候选全集，所有支持筛选只用可见观测和时钟，不能按GT清洗历史或变换支撑。
2. 在背景支撑中做空间留出，分别估计普通全局相机补偿和局部变形/微分transfer。必须在未参与估计的背景点上报告残差与不确定性，再判断目标运动是否高于误差；报告每pair覆盖率，不能只展示少数可配准漂亮片段。若背景transfer不可观测或多数目标残差低于噪声，直接HOLD/STOP该信息假设，不进入模型训练。
3. 在相同真实可观察支持上比较：原始目标运动、常规GMC/全局单应补偿、独立背景局部transfer后的目标残差序列；当前appearance与静态几何对照固定。用当前同类难负和真实跨视图正关系检查新增运动证据，同时报告静止目标、同速车流、独立运动、强视差/遮挡等预先定义场景。该阶段计算表征可分性只是信息门，不是把一个距离ranker宣称成方法。
4. 以实际残差观测做两种有区别的干预：保留对象残差替换相机变形；保留相机背景替换为另一同类对象残差。它们应分别保持和破坏身份相关信息。加入zero-motion、背景支撑错配对照；单纯历史敏感性不能通过，因为P10已经示范“对错误历史敏感但身份分辨反而更差”。
5. 在任何cal pass前固定统计口径和预算；提出实用门为：相对**正确的常规补偿对照**，cal5 pair-macro可分性至少+0.02，至少3/5对改善，并且相机替换保持增量、目标残差替换消除增量。阈值是预声明的研究决策标准，不是推理调参或性能主张；报告pair级差异与区间，5对小样本不宣称统计稳健性。

若通过，只允许进入小规模表示学习验证：相同图像/候选/监督预算下对照普通GMC+同容量编码器、同容量未分解运动编码器和现有外观表征。主效益需要最后的完整输出验证；若仅动态对象子集有益，论文范围必须收窄。静止目标、共同匀速车流、目标遮挡、非平面视差及框内背景污染都可能使因素不可辨识，不能靠增加网络层数掩盖。

当前可决定的是：**停止阈值扩展；撤回重复的P10推荐；把“真实运动能否在独立背景变换下迁移”列为一个尚待证伪的输入与表示假设。** 没有确认新方法，没有改旧实验，没有启动训练或官方验证。
