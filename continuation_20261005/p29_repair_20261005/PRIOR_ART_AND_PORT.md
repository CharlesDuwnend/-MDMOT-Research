# P29：成熟可学习对齐算子的来源与 AutoAssign 移植审查

2026-10-05；只读文献、源码和已有汇总/receipt。未读原始图像、GT/XML 或校准预测，未训练或运行模型。本文是已知算子的结构对照选择与实现契约，不是原创性结论或 MDMT 效果证据。

**建议实现保留原生 current 残差支路、以 RAFT 为 reference point 的标准 deformable-attention 控制。** 使用当前环境的 `mmcv.ops.MultiScaleDeformableAttention`；每个 FPN 层独立处理、层间共享参数，三个同尺度过去特征作为三个输入 map；显式 current/support query conditioner；用原 AutoAssign 检测损失学习偏移、采样权重和投影。它借用成熟算子，不能命名为新的 STSN、完整 Deformable DETR 复现或新型跨机方法。

这个选择保留了已验证有用的运动先验。P28 的固定训练预算后校准失败，不足以说明 RAFT 不可用：[FIT flow lookback](../p28_causal_fgfa/flow_lookback/REPORT.md) 在 240 组、40,420 条连续同视图 annotation 对应上得到 lag 1/4/8 的 RAFT 中心误差中位数 0.816/1.595/2.253 px，对应零位移为 2.500/10.012/20.125 px；RAFT 胜率 78.96%/86.31%/88.66%。这是离线 GT 中心诊断，既不证明 FPN 语义已对齐，也不识别 P28 的失败因果。报告中的小目标尾部和局部例外仍需保留。

P29 要区分三个可检验问题：原生 current 是否能作为稳定起点；在有用的 flow 位置附近，学习采样是否优于固定采样；增益是否真的来自过去图像而非当前图像上的额外空间变换。不能只把旧融合改名，也不能把任一跨领域先例视为停止 MDMT 研究的理由。

| 成熟方法 | 输入 → 算子 → 原始目标 → 输出 | 与本次问题的关系及来源 |
|---|---|---|
| FGFA，ICCV 2017 | 多帧 RGB → flow warp、embedding cosine/时间归一化加权 → 当前检测监督、联合训练 → 检测框 | 稠密检测前聚合；原版双向窗口。P28 是冻结检测器/RAFT、因果 FPN pilot，不能当作原文端到端结果。[原文 §3–4](https://arxiv.org/html/1703.10025)；[官方代码](https://github.com/msracver/Flow-Guided-Feature-Aggregation)。 |
| STSN，ECCV 2018 | reference/support 深特征拼接 → 迭代预测 deformable sampling offsets，从 support 取特征 → 检测监督 → 聚合特征和 R-FCN 检测 | **采样可由任务训练，但聚合仍沿用 FGFA cosine 归一化**；原始 current 也经 self-sampling。换采样器本身不会自动提供原生 current 直通。[原文 §4、§4.1](https://arxiv.org/html/1803.05549)。 |
| PSLA，ICCV 2019 | 两帧特征 → 渐稀疏的固定局部位置集合、learned embedding 内积/softmax → 检测监督 → 对齐传播特征；完整系统另有 RFU/DenseFT | 证明“在语义特征中找局部对应”已成熟；采样位置集合预定义，不能等同 learned continuous offsets。原测试把片段中间帧作为 keyframe，不能直接宣称在线因果。[原文 §3.2–3.5](https://arxiv.org/abs/1903.09126)。 |
| SELSA，ICCV 2019 | 多帧 proposals/ROI → proposal 间语义 attention → 分类/框回归 → 当前 proposal 检测 | 全序列语义聚合；依赖 RPN/ROI，与 AutoAssign 稠密特征入口不同，不能补上不存在的 proposal 支持后仍称同主机。[原文](https://arxiv.org/html/1907.06390)；[作者 MXNet 实现](https://github.com/happywu/Sequence-Level-Semantics-Aggregation)。 |
| Deformable DETR，ICLR 2021 | 图像特征与 query/reference point → query 预测多头偏移和权重、稀疏双线性取样、投影 → set-based 检测目标 → 框/类别 | 所选核心算子已有标准实现。原生 offsets/weights 由 query 线性投影得到，**不计算 sampled support 与 query 的 QK affinity**；把 support 放进 conditioner 是移植增加的输入结构。原完整系统不是视频模型。[原文 §4.1](https://arxiv.org/abs/2010.04159)；[官方算子代码](https://github.com/fundamentalvision/Deformable-DETR/blob/11169a60c33333af00a4849f1808023eba96a931/models/ops/modules/ms_deform_attn.py)。 |
| TDAN，CVPR 2020 | LR reference/support → 拼接特征预测 deformable offsets、重建对齐 LR → `L_align + L_SR`，均为图像 L1 → HR 图像 | 是可学习特征对齐的重要先例，但有图像重建监督。把它的 loss 直接换成 AutoAssign 后不能称原版复现；也不应无证据添加 SR 分支。[原文 §3.2–3.4](https://arxiv.org/html/1812.02898)；[作者代码](https://github.com/YapengTian/TDAN-VSR-CVPR-2020)。 |
| EDVR，CVPR 2019 Workshops | 多帧图像 → 金字塔/级联 modulated deformable alignment、TSA → 图像重建损失 → 当前恢复图像 | TSA 用逐帧 sigmoid correlation 后 concat/conv，非 FGFA 的跨帧 convex mean；最终有图像残差支路。其恢复网络残差不能冒称本次 FPN 残差的直接复现。[原文 §3.2–3.3](https://arxiv.org/html/1905.02716)；[官方 BasicSR 架构](https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/archs/edvr_arch.py)。 |

STSN 必须保留的原始细节是：reference 和 support 特征共同决定四层 deformable sampling；最后一层的 offsets 用于采样**原 support 特征**。原文用 deformable ResNet-101/R-FCN，训练时 reference 加一前一后，推理时前后各 13 帧；不是原生因果方法。其检测监督和可训练采样支持本次技术路径，但冻结 R50-FPN、past-only、MSDA、flow reference 和 current 残差都是需要明确记录的移植差异。

本轮在[第一作者论文页](https://www.gedasbertasius.com/publications)只核到 STSN 的 paper/results 链接，未核得作者维护的发布代码。公开 `lyj96/STSN` 自称论文复现，属于第三方 MXNet 实现；其 `stsn_rfcn/symbols/resnet_v1_101_stsn_rfcn.py` 的 `get_aligned_feat` 有四层采样，但训练聚合直接使用 `conv_feat[0]` 当前特征，与原文 current self-sampling 有差异。不可把这个代码未经验证地当作官方、完整或可直接接 AutoAssign 的版本。PSLA 的作者维护完整实现本轮也未核得；不据此断言其代码不存在。

本轮直接核对的本地算子为：

`/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/lib/python3.8/site-packages/mmcv/ops/multi_scale_deform_attn.py`

SHA-256：`b0a857df6c3be86ea409698911a7baa14b8a7114fd02f982f3626be9cbd87718`。使用现有 `mdmt_env/bin/python`；未安装、升级或替换环境。这里只证明源码接口存在，实际 CPU/GPU 执行能力由独立测试和 smoke receipt 建立。

| 本地 API/实现事实 | 移植要求 |
|---|---|
| `query/value` 支持 `batch_first=True`，C=256；默认 `identity=query` | **必须传入 `identity=flatten(F_current)`**，而不是 conditioner 的输出；否则所谓原生 current 直通不成立。 |
| `sampling_offsets: Linear(C,M*L*K*2)`；`attention_weights: Linear(C,M*L*K)`；softmax 在 `L*K` | M=4、L=3、K=9 时输出为 216/108。权重是 query-conditioned，不是 QK 匹配、无 null token。 |
| `value_proj` 后按 key padding mask 置零；`output_proj` 后返回 `dropout(output)+identity` | 将 `output_proj` 换成 bias=False 的 256→256 projection，并零初始化权重；dropout 显式设 0。不把 LayerNorm/完整 Transformer FFN 隐藏加入原生支路。 |
| 原生 offset 权重为零、bias 为按 head 放射状 stencil；logit 权重与 bias 均为零 | 改为含中心的 3×3 stencil，并按 `[head,map,point,xy]` 写入 bias；这是显式初始化变更，不是假称原生默认。各 head 可先共享这个位置集合。 |
| 三个输入 map 只要求各自 shape 与 flatten 索引一致，不要求物理上不同尺度 | 每个 FPN 层以三个 past frames 作为 L=3，`spatial_shapes=[[H,W]]*3`，`level_start_index=[0,HW,2HW]`。这是时间 map 适配，不能声称进行了跨 FPN 尺度 attention。 |
| CPU fallback 用 `grid_sample(2*location-1, align_corners=False, padding_mode=zeros)`；CUDA 走自定义 Function | CPU 数值测试可在当前环境运行；CUDA kernel/梯度及真实形状必须另做 smoke。import 本身仍依赖安装的 mmcv extension，不应宣称任意纯 CPU 环境无扩展也可导入。 |

建议的计算契约如下。每层 `F_t,F_(t-1),F_(t-4),F_(t-8)` 均为原生 float32 256-channel FPN。RAFT 只提供冻结几何先验，不用 flow GT、annotation 中心或对象 ID。初始 warp 用已审坐标路径得到三个 support 特征，仅作为 conditioner 输入：

```text
Q_t = Conditioner(concat(F_t, initial_warp(F_t-1),
                         initial_warp(F_t-4), initial_warp(F_t-8)))  # -> 256
Delta, logits = MSDA.query_projections(Q_t)
A = softmax(logits, over 3 maps x 9 points, separately per head)
Z_t = W_out [multihead sum A * W_value F_s(reference_flow + Delta)]
F_out = F_t + Z_t
loss = original_AutoAssign.forward_train(F_out, current_boxes, current_classes)
```

这个模块学习的是检测需要的 support 采样与残差特征，不是优化物理光流。它可以输出带正负通道投影的残差，故最终输出不受“原始四帧特征必须 convex 替换 current”的限制；采样内部权重仍归一化，不能因此称其没有 softmax。没有保证 residual 训练后一定无害，只有零初始化的准确起点。原生 MSDA offsets 无界，“围绕 flow reference 稀疏采样”不能写成存在硬局部半径约束；不自行增加 tanh 半径、QK、阈值门或 null token。

对 stride s 层的 query 节点 `(j,i)`，P26 物理 detector pixel-center index 为 `(j*s,i*s)`。先按既定半像素 resize 关系在 flow 图查询位移，再换算到该 FPN 的 feature-index 单位 `(dx,dy)`；每个 past map 的 reference point 为：

```text
reference_x = (j + dx + 0.5) / W_fpn
reference_y = (i + dy + 0.5) / H_fpn
final_location = reference + Delta / (W_fpn, H_fpn)
```

`W_fpn,H_fpn` 必须取当前层实际 ceil-shaped 特征尺寸，不能用 detector 尺寸/stride 的浮点近似。值输入保持原始 past map，**不能先 warp value 再把 flow 加进 reference 造成双重位移**。同视图/同 pair/严格过去的输入检查沿用已冻结元数据；无原始 GT 进入推理。

采用原生 MSDA padding 语义会改变 P28 的严格几何 mask：key padding mask 把 padded feature-node 的 projected value 置零；bilinear 边界的部分有效贡献仍可进入，softmax 不按有效 support 重归一化。需在 port mapping 和测试中说明，不能声称与 P28 strict-valid warp 完全等价。conditioner 的 initial warp 与最终 value sampling 是两个用途，若前者沿用 P28 strict mask，也须分别说明。所有采样无效时，bias-free W_out 对全零聚合给零残差；这不等于所有低质量历史都能自动被识别。

真实更新测试要按计算依赖检查：零 W_out 时首步只有 output projection 可有梯度；offset/logit/value projection 通常从第二步开始；由于原生 offset/logit 投影权重也初始化为零，conditioner 通常到第三步才得到梯度。至少做 3–4 个更新步再判断所有活跃分支，而不能要求首步 conditioner/offset 全部非零。未进入训练的初始化测试应验证 native feature 与 native head 输出相等；另外核对已知非整数位移、P6/P7 ceil 尺寸、边界部分支持、finite gradients、无 future/view 串入和不修改输入 tensor。

| 固定控制 | 输入与唯一关键差异 | 解释范围 |
|---|---|---|
| native | P26 current FPN 原样进入 head，不更新 | 冻结成熟起点 |
| repeated-current learned sampling | 3 个 value map 都复制 current、flow=0，conditioner 仍是 current+3 maps；同结构、同参数、同预算 | 排除新增当前帧空间变换/容量本身造成的收益；不是多帧方法 |
| past fixed stencil | 真 past+RAFT；offset weight 与 stencil bias 均冻结，仍训练 conditioner/logit/value/output | 测试残差与已对齐局部 evidence；固定 offsets 的活跃参数较少须披露 |
| past learned sampling | 真 past+RAFT；相同初值，offset weight/bias 可训练 | 与 fixed stencil 差值用于判断是否需要任务学习采样；与 repeated-current 差值用于判断真实时间信息 |

四者使用同一原生 detector、score_thr=0.05、NMS=0.6、max_per_img=100、输入分辨率、数据/损失与初始化；不得把 P28 不同协议下的数值差当作上述因素的单独因果效应。P26 与 P28 文件均保持冻结。若只改善 repeated-current，说明额外空间适配有作用，不建立时间利用成功；若 fixed 与 learned 相当，优先保留简单控制，不强行宣称 learned offset 有效。

FIT 内设计建议（执行前由独立 P29 protocol 固定，不是本文已经完成的实验）：以 pair 为分组单位，两个视图及所有 anchor 始终在同一 fold；不能按帧拆分。对已知 FIT15 升序列表做 `random.Random(42).shuffle`、每五对一组，建议 held-out folds 为：

- A：45、51、53、74、78。
- B：28、29、44、63、70。
- C：23、25、39、66、69。

每 fold 从 P26 和新 adapter 初值重新开始，训练其余 10 pairs，禁止加载 P28 已经看过全部 FIT15 的 adapter。若保持每序列 8 anchors，则每 fold 160 train groups、80 held-out groups；建议五次完整遍历，即每臂 800 steps，固定最终 checkpoint。三个 fold 产生 240 组 out-of-fold 预测，每个 FIT pair 只贡献其未参与该模型拟合的输出。原始 full-train detector 已见过这些 pair，这只是新模块的 grouped CV，不是全检测器的 unseen-data 性能。

同 fold 各臂共享 sample order、seed 和训练预算；交错/轮换臂顺序记录运行时漂移。所有臂的该 fold 预测先冻结，再打开 held-out 标签评分。主要比较预先固定的 pair-macro AP50，同时保留 native-output TP/FP/FN、recall 和 recovered/lost TP；报告全部 15 个 pair 的差值，不把视频帧当独立样本，也不把重叠训练 fold 当独立实验重复。建议只有 temporal 相对 native 和 repeated-current 的宏平均与各 fold 方向一致、native-output recall 不降低，并且 learned sampling 相对 fixed stencil 有可复现优势，才保留 learned 分支；若候选有希望，再以第二预固定 seed 重复同 folds 检查符号稳定性。具体 KEEP 数值门和第二 seed 应在执行前写入协议，不能根据 P28 calibration5 的失败形状拟合。

calibration5 已被 P28 效果分析使用，后续不能宣称它仍是盲终评集合；本轮方法选择完全留在 FIT 内，不用它调偏移半径、阈值、输出 cap 或训练步数。dev、official-val、test 不属于本报告的实现/选择范围。通过 FIT 控制以后是否扩展检测器联合训练、完整 MIA 主机及更高阶段由独立证据门决定，不能从结构契约 PASS 推出 MDA/IDF1/MOTA 改善。

计算上仅一个共享层，避免 STSN 原文的 1024-channel 四层 deformable stack 或全时空 dense attention。M=4、L=3、K=9 对每个 query 产生 108 个 head/map/point 权重和 216 个坐标数；显存/耗时仍需实测，尤其 PyTorch fallback 会显式物化采样值。若 conditioner 用单层 1×1 Conv(1024→256,bias=True)，整个共享模块约 476,996 参数，其中冻结 offsets 控制少 55,512 个活跃参数；实际实现不同则以 `parameter_counts()` 为准。不要据参数小就预告 40GB 容量或速度。

当前结论为 `KEEP_ESTABLISHED_RESIDUAL_DEFORMABLE_OPERATOR_CONTROL`，效果为 `HOLD_PENDING_FIT_GROUPED_CONTROLS`，原创性为 `NO_NOVELTY_CLAIM`。flow lookback 支持保留 RAFT，但没有把“剩余错位/融合形式/训练量”中的任何一个确认为 P28 唯一失败原因。

来源绑定（仅源文件与已有报告；未绑定或打开原始数据）：

- STSN PDF，arXiv 1803.05549：SHA-256 `f089d027e645726e86d05128a4051343ed67d335609236028b0ee13609e9080a`；PSLA PDF 1903.09126：`248d74bffeadf1cd6c83dd099e3328f359cc5876625ccc19af9bd5d88af67221`。
- TDAN PDF 1812.02898：`6d71cab6eb5f596cbdbd46999588340e10a80a0434251aafe64811fae5356921`；EDVR PDF 1905.02716：`1e6b4430edab4630cfc60f805dd5c6f9810f915b752ebe935af267e5524e6ee3`。上述 HF markdown 页本轮返回 404，已按技能回退到 arXiv 原文，不能写成 HF 全文已可用。
- Deformable DETR [HF markdown](https://huggingface.co/papers/2010.04159.md)：`43663cf0ce2b5a9de1c3c2f028b33f091d3e6d9099e2f9dccdb231af9fcf449e`；官方 `ms_deform_attn.py`：`1a093690d4fd1cb50aadba936f3450b54f7919dbed7dc68410a1ee622a40e408`；`deformable_transformer.py`：`bc33ad1e3149be1903588e246b02148019c8c1339ece70bfc24cbf4d0a143bfc`。
- BasicSR `edvr_arch.py`：`1c4fdeb081f20d41f58db555321a9241e230575453f869843a7ec15426657676`。第三方 STSN commit `2ae1bdf7476ecfb21d39c1a534098f429e5a0cbb` 的 symbol 文件：`45f0633ca91451361abb0d0ec1e32572195ae77315508fa7139c29f3d3c7e2d5`。
- 已封存 FIT flow lookback `SUMMARY.json`：`f0e99db9b6a2c5ff877d4138b726b845a937cc9d10b9a140336e365d8e647014`；`REPORT.md`：`ccb4d675820273afa77418d4eb2c0888f76ca2c9df98d72c4d44fff4e5a3e470`；`RECEIPT.json`：`1a29b17948e68d3fcaee87a9d8a1902a30a3570255c27490e7aaeb09b7d56996`。

本轮应用 [huggingface-papers 技能](/home/chenhc/.codex/skills/huggingface-papers/SKILL.md) 查阅原文，并用 [experimental-design 技能](/home/chenhc/.codex/skills/experimental-design/SKILL.md) 约束 pair 级分组、匹配控制和预先固定选择；没有用技能推定额外权限。
