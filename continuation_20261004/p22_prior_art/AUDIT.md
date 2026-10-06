# P22 有界原始文献机制审计

审计日期：2026-10-04。对象：P19“独立背景相机变形 + 目标局部残差场 + 跨视图迁移 + 因素干预”候选。仅做公开来源阅读；未读数据、未改旧工作区、未训练或调用GPU。本审计不能证明方法有效或“首次”。

## 决策

- **STOP_BROAD_DMC_GEOMETRY_MDMT_CLAIM**：SAME-MDMT 已在同任务提出 Drone Motion Compensation、tracking correction / spatial alignment、Position-IoU Fusion 跨视图关联。“针对多无人机引入相机补偿并做几何匹配”不能作为新贡献。
- **HOLD_FULL_CHAIN_NOVELTY_SAME_FULLTEXT_UNAVAILABLE**：SAME-MDMT 全文未取得，不能猜其光流、残差、网络或损失细节；当前完整链是否相同仍未知。不能把“同任务近邻 + 摘要”写成完整链碰撞已经证实。
- **CONTINUE_INPUT_OBSERVABILITY_GATE_ONLY**：本次可读全文未发现“独立背景支撑 → 留出目标的局部残差场迁移 → 面向跨无人机身份的表示学习”完整链直接同任务证据。该表述只支持继续核实输入，不等于 novelty PASS、方法确立或训练许可。
- 三个近期强邻域工作已覆盖：**顺序primary/residual补偿及motion swapping（SDM）**、**相机/几何latent通过非对称flow监督解耦（Flow3r）**、**自运动/对象运动结构化等变递归（FloWM）**。它们是 Tier 2；不能据此停止MDMT方向，也不能忽略其算子/学习目标先例。另用 GeoNet 作为残差光流的基础边界，不扩展综述。

## SAME-MDMT：已知与缺口

[DOI 10.1109/CAC67268.2025.11486840](https://doi.org/10.1109/CAC67268.2025.11486840)，题名 *Spatial Alignment and Matching Enhancement for Multi-Drone Multi-object Tracking*。本次 Crossref 确认 Tao Fang、Lianmeng Jiao、Quan Pan，2025 CAC，6296–6302 页。会议年是2025；历史出版商摘要记录 Xplore 加入日期2026-04-28。这里不混为2026新会议。

**可支持的机制边界**：两架/多架具有重叠视野的无人机视频；DMC用于tracking correction及spatial alignment；PIF融合position与IoU进行cross-view matching；在MDMT实验。来源为此前归档出版商摘要，已复制并标记到 [same_archived_abstract_excerpt.md](sources/same_archived_abstract_excerpt.md)，没有把它当本次新读到的全文。

**未核实**：DMC是affine还是homography、对应来自背景还是目标、是否dense flow、是否显式camera/object residual、是否局部微分迁移、是否训练目标或只推理算子、PIF具体公式、是否使用因果未来帧、是否做因素交换。不得由其引用BoT-SORT或homography文献反推实现。

**公开获取记录**：IEEE document及DOI落地当前返回202空内容；abstract地址418；Crossref返回元数据；OpenAlex `is_oa=false / oa_status=closed / any_repository_has_fulltext=false`；Semantic Scholar `openAccessPdf.url`为空；精确题名arXiv查询和SAME-MDMT GitHub仓库查询无匹配。Bing返回无关结果、Google跳转壳、DuckDuckGo挑战页，三者不作为“全网无全文”的证据。未使用付费绕过或需授权的staging全文通道。原始返回及状态保留在 `sources/fetch_batch*.json`。

## 机制对照矩阵

完整输入/算子/目标/输出/约束见 [MECHANISM_MATRIX.csv](MECHANISM_MATRIX.csv)。Tier 1表示MDMT同任务；只有完整链已证明相同才停止完整候选。Tier 2表示任务相邻但算子/目标强近邻，限制贡献范围。Tier 3仅是概念类比；本次没有把任何仅类比的工作升级为Tier 1。

| 来源、证据 | 原文实际机制 | 与候选的重叠 | 尚未覆盖的差异及判断 |
|---|---|---|---|
| SAME-MDMT；历史出版商摘要+当前元数据 | DMC修正跟踪和空间对齐；position+IoU跨机关联 | 相机补偿→跨机几何匹配 | **Tier 1同任务、完整链未知**。STOP宽泛claim；HOLD完整链新颖性 |
| HomView-MOT；arXiv2403.10830v3全文PDF+HTML | SuperPoint/SuperGlue估计场景H；FHE关键帧估计/插值；H进入slot attention→跨帧view slots相关→重构/更新ID；双向H映射框后平均IoU | H辅助身份学习、重构、视图稳定性、几何匹配 | **Tier 2单移动UAV MOT**。没有在已读方法中看到双无人机独立背景局部残差场迁移；该差异仍须实证 |
| SDM；arXiv2607.21576v1全文 | 冻结DINO特征；primary token先补偿源特征；residual token从剩余误差提取再补偿；弱静态场景/静态相机监督；特征MSE；跨clip motion-token swapping | 两因素顺序补偿、预测/重建监督、motion swapping | **Tier 2表示学习强近邻**。没有身份输出/独立框外背景支撑；primary不是严格camera因素。单纯把两阶段换成MOT backbone不够 |
| Flow3r；arXiv2602.20157v1全文 | 多视图transformer产camera/geometry tokens；只用源geometry+目标camera latent的非对称flow head；DPT解码；UFM伪flow+共可见mask+Charbonnier；仍联合3D/pose监督 | 以可观测flow及因素特定信息入口学习表示，跨view camera/geometry组合 | **Tier 2几何学习强近邻**。不是显式相机flow减目标flow，更不是MDMT ID。不能误写成完全无3D监督或camera/object解耦已有相同实现 |
| FloWM；arXiv2601.01075v1全文 | hidden state带速度通道；外部object flow与已知agent action的逆变换作用在递归记忆；预测未来图像MSE；2D/3D合成世界 | 自运动/对象运动分离、结构化迁移/等变、运动可组合表示 | **Tier 2等变算子强近邻**。依赖已知action及群作用；MDMT无标定图像流不能直接继承；3D编码器精确等变也没有被形式保证 |
| GeoNet；arXiv1803.02276全文PDF | depth+pose+K生成rigid flow；ResFlowNet估计残差；两者相加为full flow；分阶段视图重建、平滑及前后向一致性 | camera-induced / object residual分解、重建和consistency早已有 | **Tier 2基础算子**。无跨无人机身份表示。其残差也修正刚性估计/光照错误，提醒不能把非零残差全当目标自身运动 |

## 关键原文核实

**HomView-MOT。** 本次 `/pdf/2403.10830` 返回 **v3, 2026-02-08**，题名 *View-Centric Multi-Object Tracking with Homographic Matching in Moving UAV*；早先尝试的v1 HTML另存，分析依据v3，避免版本拼接。§III.E / Eq.12–13、Fig.6 给出HSA、slot相关、MLP解码及cross-attention更新；§III.F / Eq.14 为双向映射IoU均值；§III.G / Eq.15 与§IV.B明确slot重构CE、ID CE+triplet。证据：[PDF文本378行起](/home/chenhc/claude_try_MDMOT/continuation_20261004/p22_prior_art/sources/homview_pdf_layout.txt:378)、[原始v3 HTML](https://arxiv.org/html/2403.10830v3#S3.SS5)。已视觉检查PDF第5、6页的公式和图，不以摘要代替全文。

边界：论文Alg.1在时刻t调用`t+h`关键帧；FHE全文以邻接采样关键帧推导中间变换。若当前方案严格online causal，不能原样照搬其未来关键帧可见性后声称同一协议。可比较当前/历史可见帧的普通GMC/H，但需明确这是因果对照实现，不是完全复现HomView。

**SDM。** §2.1明确transition token读取目标帧`f_t`来重建该帧，称为“non-causal”；它不是仅凭过去外推的模型。§2.2为`p_t=phi_p(p_{t-1};f_{t-1},f_t)`、先预测`f_t^>`，再`r_t=phi_r(r_{t-1};f_t^>,f_t)`、预测`f_t^>>`。§2.3有静态场景时旁路residual、静态camera时约束primary不改源特征。§3.4表明primary会在静态camera场景编码object dominant motion，所以**不能把两token直接当可识别camera/object disentanglement**。§3.6还有跨clip latent motion swapping及短期外推；长时漂移。证据：[全文§2](https://arxiv.org/html/2607.21576v1#S2)、本地 `2607.21576.md:60–111,164–204`。当前候选若只用冻结feature加两个token、重建loss与swap，会非常接近该工作。

**Flow3r。** 核心是 `F_(i→j)=Phi_flow(g_i,c_j)`（§3.2, Eq.6），而非两个view全部patch feature任意匹配。作用是把flow监督约束到camera/geometry信息路径。§3.3还含标注3D/pose数据监督，未标注视频由UFM给pseudo correspondence；mask只支持共可见flow回归。作者认为dynamic motion被latent方式隐式容纳，没有提供显式`u_obj=u_obs-u_bg`保证。证据：[全文§3.2–3.3](https://arxiv.org/html/2602.20157v1#S3.SS2)、本地 `flow3r_md.txt:85–125`。该文限制“用flow监督因素特定表示”的宽泛claim，但其geometry目标不同于identity目标。

**FloWM。** §3.1 Eq.7把已知action的逆群变换与内部速度通道flow作用到memory。形式等变定理要求encoder/update满足等变且各速度通道输入为trivial lift；§3.2明确3D图像到top-down map的encoder并不通过结构确保精确等变，而是希望训练学出近似性质。MSE是未来图像目标，测试是合成world modeling。证据：[原文§3](https://arxiv.org/html/2601.01075v1#S3)、本地 `2601.01075.md:84–130,554–560`。在当前MDMT条件下，“背景估计的变换有噪声、对象/背景不共面”的问题仍需实测，不可沿用它的已知action前提。

**GeoNet。** §3.1–3.4直接把depth/pose刚性流与learned residual flow组合，§3.4又承认residual可修正非动态区域的初阶段错误；故`residual≠纯物体运动`是早已存在的可识别性问题。证据：[原文PDF](https://arxiv.org/pdf/1803.02276)、本地 `geonet_layout.txt:110–158,240–277`。

## 对候选最有约束力的技术边界

1. **真正待验证的是跨机传递的残差观测，不是减法。** 如果结果退化为GMC/H补偿后pool一个速度向量，再拼appearance进MLP/GRU，它只是成熟算子的组合，不能用“双因素”“物理一致”改名提升贡献。GMC、HSA、residual reconstruction、factor-conditioned flow和运动swap都已有来源。
2. **独立支撑是证据合同，不自动等于算法创新。** query、candidate及其身份关系不能用于估计评估它们的背景mapping；检测框外不保证全为静态背景，漏检动态物体和视差仍会污染。需要未参与估计的背景点验证迁移误差，并在所有预选pair报告可用支持，不能只保留漂亮片段。
3. **残差必须处于正确坐标及时间。** 令`W_A^t`为A中t→t+1的背景warp，`r_A=x_A(t+1)-W_A^t(x_A(t))`，它在A的t+1图像坐标。若`h_AB^(t+1)`是在t+1从A到B的有效局部背景映射，有限残差迁移应为

   `Delta_B = h_AB^(t+1)(W_A^t(x)+r_A) - h_AB^(t+1)(W_A^t(x))`。

   一阶形式才是`J_h(W_A^t(x)) r_A`。还需背景映射的时空闭合`h_AB^(t+1) o W_A^t ≈ W_B^t o h_AB^t`。这是链式法则/坐标变换要求，不应包装成新数学。跨机静态mapping适用于背景表面不意味着对高于路面的目标也有效；视差残差不能直接解释为身份动态。此处是定义推导，未做数据验证。
4. **干预要检查语义，不只看数值敏感。** 换camera时要同步坐标warp、support/mask和vector transport；点对点替换不同clip的原始flow不一定保留同一个object motion。换object residual后表示变化仅证明响应运动；还必须检查跨视图匹配中的运动增量是否被破坏，而非噪声扰动普遍降低分数。**换速度/残差不等于换身份**：不能把donor motion的ID当合成负标签，也不能强迫整个identity embedding随运动替换而翻转。应分别检查动态表示`z_dyn`及renderer响应，与身份表示`z_id`的同对象不变性。静止目标、同速车流不能被迫产生虚假动力学区分。
5. **若信息门通过，后续模型必须给出一个不可被同容量未分解模型替代的结构约束。** 可能的研究对象是“在独立background transport下保留空间布局的目标运动表示”，而不是generic two-token predictor。需预先定义可用支撑、对应的transport作用、camera不变量与motion等变量，以及同容量GMC+encoder、未分解dense-flow encoder、appearance+静态geometry对照。当前尚无证据决定哪种网络值得训练。

## 可供输入门通过后检验的窄候选，当前仍是HOLD

**HOLD_METHOD_LOCK_OBSERVABILITY_AND_NON_EQUIVALENCE_UNPROVEN**。当前不能锁定为“已成立的新方法”。父任务正在做的native-resolution frozen RAFT真实残差及图像级camera warp输入门，与此HOLD兼容；它回答观测能否支持学习，不会自动使模块创新性通过。

可保留的**条件式贡献假设**是：

> 面向重叠移动无人机，在相机因素仅由目标留出的背景独立约束、且局部跨视图映射误差可验证的条件下，学习保留空间布局的对象运动表示；其解码残差在跨视图迁移下相容，并为身份关联提供超过外观、静态几何和常规GMC的增量。

这句话目前必须用“待验证”，不能用“首次”“解决了相机/对象解耦”。它不是单纯`GMC + velocity + MLP`，但“不是该组合”也不是创新证明。真正值得检验的模型合同至少应具体到：

- **相机因素入口**：`B_theta`只见全部预测框外的像素/对应；用独立留出背景重建监督约束`W_bg`及跨view transfer。目标identity损失不能通过目标像素把“背景camera factor”变成identity旁路。是否冻结背景估计器或隔离梯度需明示，但隔离本身是必要控制，不是新算子。
- **对象空间因素入口**：`E_theta`接收每对象短序列残差场、位置和可见支持，产`z_dyn`或空间动态tokens；`R_phi`从该因素及明确的坐标变换解码对象残差/观测flow。保留二维布局须在实验中胜过同容量的pooled velocity与未分解dense-flow encoder，否则“field encoder/renderer”只是重新命名。
- **跨view作用与身份目标**：renderer的残差遵守上面的有限transfer或经过误差检验的Jacobian作用，形成motion-equivariance；独立`z_id`或与appearance联合的ID readout使用合法fit身份监督形成identity-invariance。相机扰动保持身份；同一物体改变运动也保持其身份。禁止用干预合成伪造identity。运动只应贡献同期跨视图对应/身份连续性的附加证据，不能替代身份定义。

| 对照 | 候选可能仍有的机制差异 | 不能作为差异的部分 |
|---|---|---|
| SDM | camera因素被框外独立观测锚定；每对象局部场而非scene-level dominant/residual token；跨无人机的明示field transfer和identity readout | 两个分支/两token、按顺序补偿、重建、预测、motion swapping |
| HomView-MOT | 观测flow中的对象局部空间残差作为被学习/被重建的对象；背景支撑留出后跨机transfer误差可核实 | H辅助ID学习、视图稳定性、slot/attention、重构identity feature、框映射IoU |
| Flow3r | 由目标留出背景建立camera因素，监督对象局部残差而非场景geometry；ID输出并单独检验motion增量 | camera/geometry因素特定入口、非对称跨view组合、flow伪监督、confidence/covisibility mask、DPT/decoder |
| GeoNet / FloWM | 移动无标定MDMT中的独立背景估计误差、对象支撑和局部非刚性场都在学习合同中被显式约束 | rigid+residual、forward/backward consistency、群作用/等变递归、camera/object变换可组合性 |

因此，**窄候选的可取之处是一个具体未验证的任务条件与可学习对象，尚不是已经证明的新算子**。若输入门显示去相机后的目标运动低于估计噪声、局部跨机transfer不成立，或增量只来自静态几何，应停止该信息假设；若通过，再用上述同容量对照和结构非等价验证决定是否进入方法训练。现阶段不要以Tier 2“不是同任务”来推导“所以novel”。

本审计推进决策：**保留原始运动输入的短门；不宣称已找到有技术深度的新方法。** 当前候选的技术深度将取决于可识别、可迁移且能带来identity增量的对象运动表示是否存在；增加loss名称或继续选阈值都不能补上这项证据。

## 工件与审计范围

- `MECHANISM_MATRIX.csv`：6个来源的机制与边界（SAME、HomView、3个近期紧邻、GeoNet基础边界）。
- `DECISION.json`：machine-readable claim分层与推进状态。
- `SOURCE_MANIFEST.json`：本次来源路径、SHA256、字节数及在线/历史来源状态。
- `SHA256SUMS`：报告、矩阵、决策和全部source工件校验；不包含自身。
- 未完整审计本领域所有论文或代码；未声称已通过全领域新颖性检索。搜索命中仅用于导航，结论基于所列原始正文/已标注历史摘要。
