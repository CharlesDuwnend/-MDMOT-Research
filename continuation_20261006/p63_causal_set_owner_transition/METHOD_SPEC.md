# P63 Causal Support-Reliability Projective Owner Transition (CSR-POT)

## 真实问题

在保护的 P26 MIA 主线中，跨视角新 ID 的几何刷新是实际错误入口。pair23 fit trace 的可判定事件中，A→B 几何 top1 约 68.8%，B→A 约 85.8%；用于估计当前单应矩阵的已知 support correspondence 中约 10.1% 是错误身份。错误 support 会把投影位置推向错误目标，随后污染 owner ID。P47 证明直接几何前门有收益，但它仍是已有位置/一对一机制，不能作为论文创新。

## 模块替换

`native support correspondences -> reliability-weighted differentiable DLT -> uncertainty-aware projective candidate set -> conflict-free owner transition`。

每个 support pair 的输入只包含当前 ready frame 以前可见的 source/target 中心、框尺度、当前 H 下的双向投影残差、support 位置和 track confidence；不使用 GT、future frame、local/global ID 数值或时间戳。一个小型 reliability head 输出 support weight，使用 weighted normalized DLT 重新估计 H；候选 owner 更新采用整帧 partial assignment 和 dustbin，避免逐边最近邻的顺序依赖。

训练目标分三层：support correctness BCE、真实跨视角边的对称投影 NLL、以及 assignment/owner conflict loss。后续 descendant consistency 只在 prefix replay 中作为状态监督，不把未来结果输入模型。

## 已完成的机制证据

严格按 frame 490/490 分割的 pair23 fit→held-out 诊断中，native H 的 new-ID candidate top1 为 68/99；固定 residual 加权 DLT 为 77/99；可训练 reliability-weighted DLT 为 78/99。结果是机制信号，不是正式 MOT 分数。三轮各三项实现审计通过：split/hash/future closure、weighted-DLT permutation/finite contracts、checkpoint/replay/attribution。固定算子已经贡献大部分收益，所以后续必须证明 learned head 在 pair-level held-out 上仍有额外贡献。

## 推进门

下一步只允许扩展到至少五个 fit pair 的真实 P26 trace；先冻结 detector、ByteTrack 和首帧 GT 初始化，比较 native、fixed-residual、learned-weight、descriptor-only 四臂。只有 learned head 在至少 4/5 pair 上超过 fixed control，且完整 host 输出不降低 MDA/IDF1，才允许 GPU 训练和正式 host 接入。
