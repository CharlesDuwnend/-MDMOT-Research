# P27：因果 FGFA 向成熟 MIA 主机的迁移审查

日期：2026-10-05。性质：成熟算子的基线迁移与代码可行性审查，**不是新方法、不是效果验证，也不授权本轮训练**。

**结论：选择 past-only FGFA 的稠密特征对齐聚合作为唯一迁移候选。** 它在 AutoAssign 产生框以前改变图像证据，能检验“当前帧响应暂时不足、过去帧仍有有效外观”这一问题；不依赖先检出当前目标，也不只是改分数、阈值或配对器。工程可行，但现有文件不能直接组成可运行 FGFA。先完成真实 fit 内时序可见性门与下列代码契约，才适合建立受控训练副本。

当前判定：`KEEP_CAUSAL_FGFA_PORT_FEASIBILITY`；效果与训练门：`HOLD_PENDING_FIT_TEMPORAL_SUPPORT_AND_PORT_CONTRACT`；原创性：`ESTABLISHED_OPERATOR_PORT_NO_NOVELTY_CLAIM`。

## 1. 已冻结的起点和本次证据边界

- `CURRENT_BASELINE.md` 和 `CONTINUE.md` 的最新 P26 记录优先：原版 AutoAssign R50-FPN + ByteTracker + MIA 几何 ID 更新/遮挡补充；保留官方首帧 GT 框、ID、confirmed/Kalman/H 初始化。
- P26 test14 **存档重评分** MDA 41.730895 / IDF1 68.341024 / MOTA 49.697168，宏平均接近论文 41.72 / 68.24 / 49.68；新的 fit23 两路完整 700 帧推理已完成。此审查未重跑或重评任何数据。
- P26 的 source、checkpoint、运行结果和 seals 都保持原样。任何迁移只能放在新的隔离副本。P25 被用户暂停，本审查不恢复它。
- `/home/chenhc/mdmot_p24_test_20261004/PROGRESS_20261004.md` 中“未覆盖目标边长中位数 15 px、60% 两视图都无检测”等是**既有 official-val5、特定 alloc/检测输出的历史诊断**。它提示检测覆盖问题，但不是本轮对 P26 supplement 输出重新得到的统计，更未证明过去两帧存在可恢复证据。不可据此直接预告 FGFA 增益。
- 本轮只读文档、源码及公开论文网页；未读原始 GT/XML、未解码视频帧、未加载模型、未使用 GPU、未训练、未进行 test/official-val 评估。唯一新增文件是本文。

## 2. 机制比较：为何选择稠密 FGFA，而非直接加时序匹配

证据级别：A = 原始全文/公开源码本轮直接核对；B = 本轮读到的本地出版商摘要存档，细节仍有限；R = 本地已验证运行与代码。跨数据集的 VID 论文仅证明算子来源，不提供 MDMT 成绩。

| 方法与来源 | 输入 → 核心算子 → 训练目标 → 输出 | 因果性、当前漏检和移植含义 |
|---|---|---|
| **FGFA**, ICCV 2017，[原文](https://arxiv.org/html/1703.10025)，§3.1–3.3、§4.1；A | 多帧 RGB → 每帧 backbone；当前到参考帧的光流；沿光流反向采样参考特征；embedding cosine + 按帧 softmax 加权 → 当前帧检测监督，原论文联合训练 feature/flow/embedding/detection → 当前帧框与类别。 | 原版用 `[t−K,t+K]`、默认 K=10，**并非零延迟因果算法**。改为 `s≤t` 是明确的部署变体。稠密聚合发生在 proposal/框以前，符合暂时漏检的接口需求。选它作为迁移基线。 |
| **DFF**, CVPR 2017，[原始摘要](https://arxiv.org/abs/1611.07715)，[官方 MMTracking 实现](https://github.com/open-mmlab/mmtracking/blob/v0.14.0/mmtrack/models/vid/dff.py)；A 摘要/代码 | 当前 RGB + 稀疏关键帧 RGB/深特征 → 光流传播关键帧特征，非关键帧跳过重 backbone → 检测/识别任务监督联合训练 → 当前预测。 | 已查实现从最近过去关键帧传播，可因果运行；主要目的为省计算。缺少每帧新特征与多次观测聚合，关键帧漏检证据可能持续传递，故不作为本次覆盖增强主算子。 |
| **SELSA**, ICCV 2019，[原文](https://arxiv.org/html/1907.06390)，§3.2、实验 sampling strategies；A | 视频各帧 RPN proposals/ROI 特征 → proposal 间语义相似度与 softmax attention 聚合 → 检测分类/框回归 → 增强后的当前 proposal 检测。 | 默认从完整视频无序采样，不能直接宣称因果；可裁成过去 proposal 库，但 AutoAssign 没有 RPN/ROI head。额外搭 proposal 层会改变主机，且仍要求当前候选具备空间支持。语义相近也不是物理同一 ID。此次不选。 |
| **STCA**, Remote Sensing 2024，[全文](https://doi.org/10.3390/rs16203861)，Fig.4、§3.2、§4.2、Eq.10；A | 同期两视图 → CenterNet/DLA34 检测；当前/前一帧 appearance+position 向量 Transformer 关联；T-LightGlue 几何跨机匹配 → `L_all=L_det+L_single` → 本机轨迹与跨机一致 ID。 | 相邻帧关联可因果；原文明确跨机模块只在 inference。其时序 Transformer 接收已检出的高置信目标，并不是把过去稠密图像证据送回 AutoAssign 检测层。是直接 MDMT 必须比较的成熟完整链，但换成整套 STCA 会同时换 detector/tracker/geometric matching。 |
| **TSMMT**, TCSVT 2025/online 2024，[DOI](https://doi.org/10.1109/TCSVT.2024.3478758)，本地 `direct_archived_ieee_abstracts.md`；B | 多机视频 + 既有 tracklets → temporal-spatial feedback；temporal-oriented localization 用历史轨迹增强困难目标响应；spatial identification 交互目标/背景/跨机轨迹特征 → 定位与身份学习（具体 loss 未取得）→ 跟踪输出。 | 直接 MDMT 已有“历史特征改善困难目标定位”。不能声称这项宽泛动机新颖；也不能从摘要补写精确对齐、缓存、因果时刻或损失。未核得作者完整实现，故不能把它描述为本轮已可直接复制的模型。 |
| **MIA-Net / P26**, [官方仓库](https://github.com/VisDrone/Multi-Drone-Multi-Object-Detection-and-Tracking)，本地冻结代码；R/A 代码 | 两路当前检测/轨迹 + 官方首帧 GT 初始化 → local/global 几何匹配、映射和遮挡补充、ID 更新 → 各视角框/ID及跨机关联；本次没有重新训练该跨机过程。 | 已有强主机。本次 FGFA 仅替换其检测输入形成过程，仍使用原三类检测头、阈值、ByteTracker、MIA 和初始化；任何 MDA 改善必须由完整输出确认。 |
| **CVRP**, EICARS 2025/IEEE 2026，[DOI](https://doi.org/10.1109/EICARS68214.2025.11320227)，本地出版商摘要；B | 本机 MOT 输出与两视图 → perspective matching；跨机位置校正、未匹配目标补检与 ID reallocation → 改善跟踪 → 修正框/ID；训练目标未核实。 | 直接说明“跨视图补检”不是新贡献。其已公开摘要支持的是输出侧补全，与本次检测前同视图时间聚合的接口不同；不据摘要判定细粒度完整链等同。 |

至少五项机制比较已完成。没有用一般视频检测的相邻领域先例来 STOP MDMT 研究；选择 FGFA 正是主动采用成熟先例。未来若要形成论文贡献，必须在这项强时序基线之上另建可验证的学习对象，而非把 past-only、FPN 适配或注意力重命名为创新。

## 3. 唯一候选的精确计算契约

推荐基线名称：**AutoAssign + causal FGFA**。保留 FGFA 的核心“光流对齐 + embedding 加权稠密特征聚合”，明确记录三处迁移差异：R-FCN 改为原 AutoAssign；单尺度深特征改为原 FPN 的 P3–P7；双向窗口改为仅当前/过去帧。

### 输入

对每个视图独立处理当前 `I_t` 与已到达的历史 `I_s, s<t`。试验契约可先固定三帧 `{t,t−1,t−2}`；这是供审查和算力估算的短窗口，**不是从 val/test 调出的窗口**。最终窗口须由 root 的 fit 内时间支持审查一次性定下，不开展窗口/阈值 sweep。

- 输入只含 RGB、图像 resize/pad 元数据、帧索引和合法的图像特征缓存。
- 不输入当前 GT 框/ID、未来帧、GT flow、tracker ID、跨视图正配对或 target-conditioned search window。
- 两视图各自保持缓存；不得将 view 1 的 `t` 当成 view 2 的历史。frame 0、序列切换、跳号或重启需显式重置/检查。开头用已有过去帧，不复制未来或以 GT 补齐。
- P26 官方首帧 GT 初始化仍留在后续主机，必须继续在协议中披露；“新检测算子无 GT 输入”不使完整系统变成 GT-free。

### 算子

令 `F_s^l = FPN_l(ResNet50(I_s))`，`l∈{P3,…,P7}`，每层 256 通道，stride 为 8/16/32/64/128。对过去帧估计当前到历史的采样场 `u_(t→s)`，并使用 backward sampling：

```text
G_(s→t)^l(p) = bilinear_sample(F_s^l, p + u_(t→s)^l(p))
G_(t→t)^l    = F_t^l
q_t^l       = normalize(E(F_t^l), eps)
k_s^l       = normalize(E(G_(s→t)^l), eps)
a_s^l(p)    = softmax over valid frames s of dot(q_t^l(p), k_s^l(p))
Fbar_t^l(p) = sum_s a_s^l(p) G_(s→t)^l(p)
```

参考集合始终包含当前特征；它保证无历史时回到原特征，不另加阈值门或身份筛选。无效 padding/越界参考只按几何有效域屏蔽，当前有效区域总能保留一个项，避免全 invalid 的 softmax。该屏蔽是坐标/数值契约，不作为“质量模块”贡献。

可以复用 `EmbedAggregator` 的 256 通道 embedding 结构（该本地实现默认一层 3×3），但必须称为 MMTracking 风格实现；原 FGFA 论文 §4.1 是 1×1 / 3×3 / 1×1 的三层 embedding，不能假装精确同构。初次移植不并入 ReID、跨视图 attention、mask 学习、RAFT 或新的自监督目标，避免同时改变多个因素。

### 目标

采用**当前帧 AutoAssign 原始检测目标**，让分类/定位误差直接反传至聚合器；当前帧 train bbox/class 即可监督，不要求跨帧 ID，也不要求 MDMT 光流 GT：

```text
L = AutoAssignHead.forward_train(Fbar_t, meta_t, gt_boxes_t, gt_classes_t)
  = loss_pos + loss_neg + loss_center
```

本环境 `mmdet 2.25.1` 的 `autoassign_head.py:390–435` 把 GIoU 回归成本用于正袋概率；对外不是简单独立的 `loss_cls + loss_bbox` 两项。保留该实现与原权重，不添加未核实的轨迹/全局 ID 损失。

### 输出及 AutoAssign hook

1. `SingleStageDetector.extract_feat`（已安装源码 `single_stage.py:41–46`）完成 backbone+FPN 后，得到五层原始特征。
2. 只在新副本的 detector wrapper 中生成 `Fbar_t`；训练调用原 `bbox_head.forward_train`，推理调用原 `bbox_head.simple_test(Fbar_t, img_metas, rescale=...)`。
3. 用原 `bbox2result` 生成三类 `[bbox, score]` 数组，保持 `detector.simple_test` 的返回结构。
4. 原 `mmtrack/models/mot/byte_track.py:198–209` 继续读取三类结果、拼接为 tracking 使用的单类，再走原 ByteTracker / MIA supplement。不得返回 SELSA/FGFA VID wrapper 的 `dict(det_bboxes=...)` 直接替换原 list-of-classes。

这不是 post-NMS 重排；新的稠密框可在当前原检测没有框的位置生成。相反，若对象在全部输入图像中都不可辨、或目标尺度低于原 FPN 表征能力，聚合无法保证创造可靠观测。

## 4. 是否必须训练检测器

**必须训练时序聚合模块，不能把随机 embedding 加零训练融合视为成熟方法。是否更新 AutoAssign 原参数，应分阶段回答。**

- 算子契约/小型机制试验可以冻结原 R50、FPN、AutoAssign head 以及外部 flow，只训练聚合 embedding。梯度仍可穿过冻结 head 到达聚合特征。此时是 `frozen-detector learned-fusion pilot`，不是原论文端到端 FGFA 的完整复现；`loss_center` 在相关 head 参数冻结时也不会提供新可训练梯度。
- 原 FGFA 论文明确联合训练 feature extraction、flow、embedding 和 detector。若目标是一个训练充分的强时序检测基线，**应在 P26 的隔离副本里训练 detector+fusion，并设置完全匹配的单帧 fine-tune 对照**，否则增益可能仅来自再次训练 AutoAssign。初始化同为 P26 权重，训练 split、step、图像规模、增强和 optimizer 一致；原冻结 P26 仍作为独立零更新基线。
- FlowNetSimple 代码已存在，**本轮未核定其可用预训练 checkpoint、权重 SHA、色彩规范和运行兼容性**。原 FGFA 使用 FlyingChairs 预训练的 FlowNet simple，属于外部光流预训练；不冒充 MDMT 学到的运动。若 flow 固定以降低试验成本，应明确偏离全网络联合训练；若再引入 RAFT，需列为另一移植差异，不能悄悄借用 P22 输出。
- 本轮没有启动上述任何训练，也没有给出训练收敛、40GB 能装下或 MDA 将提升的承诺。

## 5. 现有源码审查：存在可复用部件，但不是开箱 FGFA

P26 来源版本由其 `SOURCE_IMPORT.json`/README 固定。`source/mmtrack/models/vid` **实际不存在**；本次直接检查目录和文件列表所得，不是推测。只找到 `EmbedAggregator`、`SelsaAggregator`、`FlowNetSimple`、flow warp、SELSA ROI 和 ImageNet VID 采样配置等部件。

| 位置 | 已确认事实 | 迁移前必须完成的动作 |
|---|---|---|
| `models/aggregators/embed_aggregator.py:55–86` | `[1,C,H,W]` query 与 `[N,C,H,W]` refs，按帧 softmax 加权；只支持 key batch=1；两处除以 L2 norm 没有 eps。 | 支持零/无效特征的稳定 normalize；显式 current reference；单样本训练或实现真正 batched 版本，不把 batch 当时间混合。 |
| `core/motion/flow.py:18–21` | 按 feature 宽度比计算一个 `scale_factor`，同样缩放 H/W 与 flow 两分量。 | 改为显式目标特征格点/尺寸与相应 stride 单位。P6/P7 采用 ceil 下采样时不能仅靠宽比：例 800×1344 pad image 的 P6 可为 13×21，按宽比 1/64 缩流却得到高 12。这是静态尺寸反例，未运行模型。 |
| `core/motion/flow.py:34–41` | 归一化使用 `2*x/W−1`、`2*y/H−1`，但 `grid_sample(align_corners=True)`。 | 配对正确的坐标规范。当前零 flow 在一维实际采样 `x*(W−1)/W`，不是恒等。采用 `align_corners=True` 时应处理 W−1/H−1 及单像素边界，或完整采用 pixel-center/False 规范。冻结 P26 内此未启用部件不能原地修补。 |
| `models/motion/flownet_simple.py:149–192,235–242` | 从 detector norm 还原图像后改用 flow norm；有图像缩放、flow 倍率和 padding 操作。 | 验证 detector BGR (`to_rgb=False`) 与 flow checkpoint 期望的通道顺序；还原/归一化往返；流方向、图像/特征坐标及倍率。不能仅凭函数名判断方向正确。 |
| `configs/_base_/datasets/imagenet_vid_fgfa_style.py:41–45,63–78` | train 是 bilateral sampler，val/test 窗口 `[-15,15]`；归一化/scale 也不同于 P26。 | 禁止直接借此 config。用相同视频/视图的过去帧 loader，并共用 resize/flip 几何；保留 P26 单尺度 1333×800 和 BGR norm。 |
| `models/roi_heads/selsa_roi_head.py:80–108` | 依赖 proposal lists、ROI extractors、ROI bbox head。 | 这是 SELSA 不适合本次轻量接口迁移的源码依据；不硬塞到 AutoAssign。 |
| 上游 MMTracking `v0.14.0/mmtrack/models/vid/fgfa.py:343–354` | 分支检查 `self.detector.bbox_head`，调用却写成 `self.bbox_head`；类构造只设置 `self.detector`，未见相应别名。返回 VID dict 也不同于 ByteTrack detector 契约。 | 即使补回上游 VID 目录也不能宣称 single-stage 分支已可运行。直接按已安装 `SingleStageDetector` 的 head API 实现薄 wrapper，更易保持 P26 返回格式。 |
| 同一上游 `fgfa.py:151–157` 与 inference 分支 | training 仅聚合 warped refs；inference 会显式包含/恢复 current 特征。 | 本迁移训练与推理统一 current+past 集合；将差异写入 port mapping，不能无检查整段复制。 |

上述 flow 问题是**新时序分支需要处理的静态代码风险**，不表示 P26 当前主基线有这些误差；P26 常规 AutoAssign 推理未调用这些 warp。

进入图像 pilot 前的 CPU 契约应覆盖：零流恒等、已知平移/方向、非方形与奇数尺寸 P3–P7、像素到 stride 坐标、越界/全零特征无 NaN、current-only 与原 extractor 输出一致、未来帧不可访问、双视图/序列缓存隔离。当前仅列出契约，未运行它们。重复 current 图像 control 也应保持特征一致，排除归一化/缓存错误制造的假“时序收益”。

## 6. 必要对照和推进门

### 同一成熟主机上的控制矩阵

| 对照 | 改变项 | 要排除的替代解释 |
|---|---|---|
| A：冻结 P26 | 无参数更新，原检测+完整 MIA | 已验证论文水平起点；不要与历史弱 GT-free host 混成一条增益曲线。 |
| B：匹配单帧 fine-tune | 与时序训练相同的 detector 可训练部分和训练量，仅当前图像 | 增益来自额外 detector 训练或域适配。若阶段只训练 embedding，B 对应保持原 detector 的 current-only wrapper。 |
| C：过去帧、无对齐的同构聚合 | 与 FGFA 相同 embedding、训练预算和输入帧数，采样位移为零 | 多帧/参数本身是否足够；FGFA 原文也显示 naive fusion 可能损害快运动目标。 |
| D：flow 对齐 + 均匀平均 | 同一 current+past、同一 detector 训练预算，仅取消可学习 embedding 权重 | 收益是否只来自对齐；必要时区分额外 embedding 参数的作用。 |
| E：causal FGFA | flow 对齐 + learned embedding aggregation | 完整已知算子的实际价值。 |

“是否有对齐”的控制不能调用存在零流非恒等问题的原 warp；C 必须是真实恒等采样。所有臂保持原检测 `score_thr=0.05`、NMS 0.6、`max_per_img=100`、ByteTracker/MIA 参数、初始化、输入分辨率和单尺度；不同时加 TTA、增大 top-k 或寻找低置信阈值。

### 数据与观测门

root 正在独立检查 fit 内时间支持。需分别回答：

1. 当前漏检事件中，**过去**短窗口同一目标是否存在可用图像/检测证据；双向任意邻帧可见率不能替代 past-only 支持率。
2. 过去可见但当前漏检、持续从未检出、刚进入视野、长遮挡、离开视野分别占多少；不能把“最近见过”自动当“当前仍应输出”的证据。
3. 按小目标/类别/运动与 gap 长度分层；pair 作为汇总单位，不把高度相关的视频帧当独立样本。

历史/GT 仅可在合法 fit 诊断或训练监督中标记这些分层，不可进入部署特征和缓存。P26 checkpoint 已经使用哪些原始 MDMT train 数据，应在确定 held-out 小门时继续披露；不能把仅对新增 fusion 留出的 pair 称为整套 detector 从未见过。

如果过去根本没有证据或绝大部分事件是持续极小目标，判定 `STOP_TEMPORAL_COVERAGE_RATIONALE_FOR_THIS_HOST`；这不是否定所有时间建模，而是停止用它解释当前覆盖缺口。若时间支持成立，再进入 CPU 契约、可运行 wrapper、小型受控训练。机制 pilot 以检测召回/定位精度与 FP、短时漏检恢复为读数；通过后才看原完整 MIA 的 MDA/IDF1/MOTA 和 TA/FA/MA 转换。仅 recall 提高而误报扩张不足以过门。

本文不设从历史 val/test 读数拟合的数值门槛，也不授权 official-val/test 模型选择。下一阶段在合法内部 split 和资源确定后预先固定训练预算及升阶标准。

## 7. 计算与风险的可审查量级

假设 padded image 为 800×1344、五层为 100×168 / 50×84 / 25×42 / 13×21 / 7×11，一个 256 通道 float32 FPN 原始缓存约 **21.9 MiB/帧/视图**；两视图各两帧历史约 **87.5 MiB**。这是张量尺寸估算，未测量显存；不含当前特征、RGB、flow、embedding、detector 激活、optimizer 或双视图主机。

因果缓存下，每个新视图帧仍需一次 backbone/FPN，另对 K 个过去参考估计 flow 和执行聚合；K=2、两视图时每个同步时刻有四对 flow。其成本大于原逐帧 AutoAssign。训练时不能跨 optimizer 更新重用已过时的可训练 backbone 特征；若 backbone 解冻，需按 clip 重算。故不能用上述小缓存数字承诺训练能装入 40GB，更不能引用旧 K40/VID 吞吐推算 A100/MDMT 总时长。

主要科学风险：小目标 flow 易跟随背景；当前 query 已弱时 cosine 可能仍偏向背景；过去目标离开/完全遮挡时会产生拖影 FP；原 P3 stride=8 的分辨率上限没有被 FGFA 本身移除。报告任何增益时必须同时给误报、定位、耗时与完整跟踪结果。

## 8. 来源与可复核定位

本地直接读取：

- `/home/chenhc/claude_try_MDMOT/CURRENT_BASELINE.md`；`CONTINUE.md` 的最新 P26 段；`continuation_20261005/p26_mia_baseline/README.md` 和 `runs/fit23/ENVIRONMENT.txt`。
- `/home/chenhc/mdmot_research_20261002/notes/direct_literature.md`；`sources/direct_stca_pdf.txt` 的 §3.2、§4.2；`sources/direct_archived_ieee_abstracts.md` 中 TSMMT/CVRP 摘要。TCFNet 原文在这些材料中仍未核定，本审查未补写其机制。
- `/home/chenhc/mdmot_p24_test_20261004/PROGRESS_20261004.md`：仅作为历史失败分解来源，未重跑其脚本或使用其 test 成绩调参。
- P26 `source/mmtrack/models/aggregators/`、`models/motion/flownet_simple.py`、`core/motion/flow.py`、`models/mot/byte_track.py`、`models/roi_heads/selsa_roi_head.py`、AutoAssign/VID configs。
- 实际运行环境 `/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/lib/python3.8/site-packages/mmdet/models/`：`detectors/autoassign.py`、`detectors/single_stage.py`、`dense_heads/autoassign_head.py`。fit23 记录确认 mmdet 2.25.1 / mmcv 1.4.8 / torch 1.10.1+cu113。

2026-10-05 在线原始来源：FGFA `https://arxiv.org/html/1703.10025`、SELSA `https://arxiv.org/html/1907.06390`、DFF `https://arxiv.org/abs/1611.07715`，以及 MMTracking v0.14.0 的 `mmtrack/models/vid/{fgfa,dff}.py`。Hugging Face 三个 `.md` 入口均返回 404，按 huggingface-papers 技能回退原始 arXiv；未把 404 当论文不存在。为遵守仅写本文的范围，未另外保存网页/源码副本；正式移植时需固定上游 commit/下载文件。

关键已读源文件 SHA-256（只读计算）：

```text
651bfa8ba07258a7776f9a08829eaab74d1dd968e769601d5dcc7dc4a51358ce  P26/source/mmtrack/models/aggregators/embed_aggregator.py
9950d60bd8347014a778e222947aa1168e082482a530262970b7e3bd3f6df9ef  P26/source/mmtrack/models/aggregators/selsa_aggregator.py
b1bdeac6d5d5ad5862df859b60e3da1c8938cc9fc524e7087eefd397dc7ca527  P26/source/mmtrack/models/motion/flownet_simple.py
c6600937a454e9453f750757b18f8cbdd8999c3cbc3f001a4fe7dc290d876aeb  P26/source/mmtrack/core/motion/flow.py
07ac943d0fcafe470dccec5edd44f2872d39629e04b33d77398947287cc01792  P26/source/mmtrack/models/mot/byte_track.py
04fea9eda775ec3d0ad6ce129b16918b94ba8668913eb1045d222d0829fcf07f  P26/source/configs/_base_/models/autoassign_r50_fpn_8x2_1x.py
4bae1fc7ad385596a98da7f72b19b2b218f7ef1729d0e48e1b125f551563bf7d  P26/source/configs/_base_/datasets/imagenet_vid_fgfa_style.py
a13d599724625fa0e3b62760e5765b17fba395276f9be0abf36f52e09b5077c1  online mmtracking/v0.14.0/mmtrack/models/vid/fgfa.py
```

**本阶段交付的是可审查的成熟算子迁移方案与具体接口风险；尚无可运行时序模型、训练结果或新方法成立证据。**
