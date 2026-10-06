# P46：Latent Anchor-Line Projective Matching

状态：`CANDIDATE_MECHANISM_DIAGNOSTIC_ONLY`

P46 是一个候选几何模块，目标是替换 MIA 中把 bbox 中心直接作为跨视角几何证据的做法。它不是 P14 外观适配器、不是图匹配器、不是阈值调节，也不是 post-hoc ID relabel。

## 假设

无人机视角下同一目标的框中心会同时受目标高度、俯视角和框截断影响；单一 homography 对图像平面点的映射不一定是最稳定的身份证据。目标与地面接触的垂直支撑位置应比框中心更接近跨视角可重复的几何量，但实际框底部可能被遮挡或检测截断，因此锚点位置应是不确定的潜变量，而不是固定写死在 `y2`。

## 输入 → 算子 → 目标 → 输出

- 输入：当前帧两视角 bbox、类别/置信度、MIA 当前 homography、可选冻结 P2 box feature；不输入 GT、未来帧、local ID 数值或外部相机标定。
- 算子：对 bbox 底部支撑线采样固定 `α∈{0,0.1,...,0.4}` 的候选锚点 `pα=(xc, y2-αh)`；使用 homography 计算 A→B 与 B→A 的对称投影残差、box-size/line-height 一致性，并由轻量 anchor uncertainty head 产生 `μ, logσ`。候选边代价是对锚点分布的稳定 soft-min/NLL，不是一个可调阈值。
- 训练目标：以 train-only 跨视角 GT owner 边作为结构化正负边，联合优化真实边的对称投影 NLL、同一目标两视角锚点分布一致性和 one-to-one assignment loss。`μ/σ` 只学习几何不确定性，不学习或修改检测分数。
- 输出：候选边 cost/uncertainty 和整体一对一 proposal，送入原 MIA owner 更新前门；confirmed owner 不由单帧后处理改写。

第一道门只测试固定几何观测是否真的提供信号：在冻结 P26 pair23 输出和首帧 H 上，对 GT 已匹配的检测行计算 center、bottom、lower-line soft-min 三种成本，比较真边 rank-1、真负边 margin 和有效覆盖。诊断读取 fit XML 仅用于标注，不产生正式 MOT 结果，也不授权训练。

## 训练和控制（只有诊断通过才允许）

`G0` 原中心/角点几何；`G1` 固定 bottom-center；`G2` 固定 lower-line soft-min；`P46` 学习 anchor μ/σ 和结构化 assignment。四臂共享 P26 detector、MIA、输入、H 初始化、候选上限和 pair-level split。先冻结 detector/P2，训练 P46 几何头；不能用 official-val/test 或 calibration5 调 anchor 范围、σ 下限、NLL 温度。

P46 只有在 `G1/G2` 相对 `G0` 的真实边 margin 在 fit 内有稳定改善，并且改善不是由减少候选数或阈值造成，才有理由训练 learned head。若固定几何都没有信号，则 `STOP_GEOMETRY_OBSERVABLE_NOT_SUPPORTIVE`；若固定几何有信号但 learned head 无法超过固定控制，则停止训练扩展。

## 论文边界

当前只允许称为“目标框支撑线的潜在几何观测候选”。Homography、底部接触点、轨迹几何和不确定性建模都有先例；是否存在 MDMT 同任务直接碰撞，必须在诊断后再次审计。即使通过，也不能声称首次或 SOTA，直到真实 MIA 完整输出通过。
