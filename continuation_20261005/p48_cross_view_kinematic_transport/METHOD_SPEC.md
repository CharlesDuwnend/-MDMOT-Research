# P48：跨视角运动状态输运（Cross-View Kinematic Transport）

状态：`CANDIDATE_CPU_GATE_PENDING`

## 研究问题

P47 的 geometry-only 前门说明位置证据比 P14 appearance 更可靠，但同一帧的单应投影仍会在近邻目标之间产生歧义。P48 不替换成熟的 ByteTrack、MIA 或一对一求解器，而是在候选代价之前增加一个可学习的**跨视角运动状态观测模型**。

## 输入 → 算子 → 目标 → 输出

* **输入**：当前源/目标 track 的中心、宽高、最近 `K` 个严格过去位置及其帧间差分；当前源到目标的单应矩阵 `H_t`；不读取时间戳、全局 ID 或未来帧。
* **算子**：用 `H_t` 在源中心处的 Jacobian 输送源速度/加速度，形成目标视角的预测运动状态；一个小型异方差残差头预测目标视角状态的均值和协方差。候选关系只在 MIA 已产生的 new/old-unmatched 集合内计算，不构造跨目标图。
* **训练目标**：对同一物理目标的目标视角状态最小化 Gaussian NLL，并用同帧异目标和 hard-neighbor 候选做排斥；按遮挡/短轨/长轨分层报告，防止只学到静态位置。
* **输出**：候选的运动创新似然与校准协方差，作为 P47 geometry front door 的附加证据；不做后处理 relabel，不修改检测框。

## 合法性与门

训练只使用 fit pairs 的 XML identity，calibration pairs 只在训练冻结后读取标签；官方 val/test 不打开。所有历史窗口为严格 prefix，第一帧首帧 XML 初始化沿用 P26。先过 CPU 合同和真实 pair23 诊断，再做小规模 fit/calibration 训练；未过表示/运动分离门不进入 GPU 大训练或官方评分。

## 关键反事实审计

必须验证：置换源速度会改变输出；把速度/加速度清零后性能退化；仅几何控制与运动输运控制可分；协方差为正且有限；无未来帧、标签或时间戳输入；近邻 hard-negative 的 rank/残差确实改善。
