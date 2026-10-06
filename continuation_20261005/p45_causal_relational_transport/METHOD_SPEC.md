# P45：Causal Relational Transport for MIA

状态：`CANDIDATE_MECHANISM_PENDING_CPU_GATE`

日期：2026-10-05。P45 是一个待审查、待实现的跨视角决策模块，不是结果、不是论文结论，也不授权 official-val/test。

## 要解决的具体问题

P43/P44 表明，把外观描述符在 MIA 之后写回 view-2 ID 会破坏 owner continuity，甚至在 continuity-safe application 下变成不活跃证据。P45 将改变插入位置：在原 MIA 的 `get_matched_ids` 产生 owner 更新之前，直接替换“逐目标几何最近邻 + 顺序写 ID”的跨视角决策，输出一个**整批的一对一边匹配和拒配状态**。原 ByteTrack、首帧 GT 初始化、检测器、Kalman、补框和输出协议保持不变。

## 核心边界

P45 不把“加历史均值、加阈值、换 Hungarian cost、加 dustbin”单独当作贡献。它学习的对象是当前时刻两个视角的**关系结构到整体匹配计划**，而不是独立 pair score：

1. 每个节点是一个当前 local track event，输入当前 P14 表征、严格过去的同视角 tracklet 状态、bbox/置信度和由当前 homography 归一化的中心/尺度。
2. 对每个节点，构造相对配置签名：它到同视角其它节点的归一化位移、尺度比、排序稳定性，以及映射到另一视角后的残差。签名是 permutation-equivariant 的，不能依赖 local ID 数值或列表顺序。
3. 两侧节点和跨侧候选边经过共享的 node/edge encoder，产生整批 cost logits；使用带未匹配容量的**部分传输**得到一对一计划。未匹配容量是 transport 约束的一部分，不单独作为质量阈值或 post-hoc dustbin 规则。
4. 只将确定的边作为 owner proposal 交给一个连续状态更新器：已有 confirmed owner 不能被当前帧直接重写；新边须在短暂 provisional 状态中累积，跨过固定的证据边界后才发布。这一层是为了修复 P43 暴露的写 ID 缺陷，不是调 window 的补救。

## 计算契约

对时间 `t`、视角 `v∈{A,B}` 的节点集合 `X_v^t`：

```text
z_i       = P14(current_i)                         # inference-time frozen input
h_i^t     = causal_state({z_i^s, box_i^s}: s<t)    # no future, no GT
r_i^t     = RelEncode({box_i^t - box_k^t, scale_i/scale_k}_{k!=i})
n_i       = NodeEncode([z_i, h_i^t, box_i^t, score_i, r_i^t])
e_ij      = EdgeEncode([n_i^A, n_j^B, geom_ij, r_i^A-r_j^B])
P_t       = PartialTransport({e_ij}, row/column capacity, unmatched mass)
proposal  = StableOwnerUpdate(P_t, previous_owner_state)
```

`geom_ij` 只包含 homography 后的中心/框残差及合法性标记。任何 XML、首帧以外的 GT ID、未来帧、人工 global namespace、GPS 或 timestamp assumption 都不能进入 `z/h/r/e`。P14 只作为冻结输入表征，P45 的 trainable parameters 是 relation encoder、edge scorer 和 transport temperature/normalization；不能把 P14 的 1200 steps 重新算成 P45 训练。

## 训练目标（候选，不是已运行）

在合法 train-only pair split 上，以整批 assignment label 监督：

```text
L = L_edge + 0.5 L_plan + 0.25 L_cycle + 0.1 L_state
```

- `L_edge`：真实跨视角 owner 边与 hard negative 的边损失；负例从同帧候选和合法未匹配节点产生。
- `L_plan`：计划矩阵对 GT partial permutation 的行/列约束及边交叉熵；它约束整体计划，不能由独立 edge ranker 替代。
- `L_cycle`：交换 A/B 输入后，计划转置与原计划在有效节点上的一致性；只用同一时刻的输入，不引入未来。
- `L_state`：owner proposal 的前缀因果状态损失，惩罚已经 confirmed 的 owner 被单帧反复改写。

`L_state` 不是拿 GT 去在线更新状态；GT 只在训练和离线诊断标注目标。首阶段冻结 P26/P14，训练 P45 适配器；若适配器通过 grouped fit gate，再做完全匹配的 current-only/set-relation controls。不得用 cal5 或 official-val 选择 transport 温度、未匹配质量或状态窗口。

## 输出与主机接入

P45 的运行时输出是 `(accepted_edges, provisional_edges, unmatched_A, unmatched_B, audit)`。它不直接改数组中所有 ID，也不对已经 confirmed 的 view-2 owner 做 post-hoc relabel。接入点放在 `supplement_mia.py` 的首次 `get_matched_ids` 之后、`A_same_target_refresh_same_ID`/`B_same_target_refresh_same_ID` 之前；需要同时绕过原顺序写 ID 的分支，并将确定边作为受 owner-state 约束的 proposal。若只能通过旧 refresh 函数逐框写回，P45 视为未接入，不能用该结果评价方法。

## 预注册控制

所有控制在同一 P26 detector/checkpoint、输入、首帧 GT、MIA 参数、score/NMS、GPU 类型和输出评分脚本下运行：

- `B0`：原 P26 MIA。
- `B1`：同样候选输入，原几何/顺序匹配。
- `B2`：独立 edge scorer + Hungarian，去掉关系签名和状态层；用于排除普通 ranker 增益。
- `B3`：关系签名 + partial transport，但不学习、只固定几何 cost；用于排除传输约束本身。
- `P45`：关系签名、整批 partial transport、cycle loss、owner-state 更新。

所有比较以 pair 为独立单位；不得按 frame 随机拆分。pair23 只用于第一次完整 host replay，不用于选择超参数。若 CPU 合约、fit grouped gate 或真 MIA 接入任一失败，停止，不训练更长、不打开 official-val。

## 通过条件

第一道门不是 MDA/IDF1，而是：输出计划行列容量满足、无重复 owner、confirmed owner continuity 违规为 0、严格过去依赖为 0、A/B 交换一致性通过。第二道门是 grouped fit 上相对 B1 的 pair-macro assignment quality 和 owner repair 同向改善，并且不被 B2 单独解释。只有两道门都通过，才允许对 pair23 做完整 P26 MIA replay；pair23 若仍低于 P26，只关闭 P45 host route，不把失败归因于 detector。

## 当前创新判断

当前只允许写成：`configuration-level relation-to-plan candidate`。多视角图匹配、轨迹级关系、部分匹配和 owner-state 都有先例；P45 的可能贡献只能来自“严格过去的目标关系配置 → 整体 partial transport → 连续 owner publication”这一**组合是否在 MDMT 同任务中尚未被直接实现且能通过完整主机验证**。在完成机制级文献审计前，不得使用“首次”“原创”“SOTA”。
