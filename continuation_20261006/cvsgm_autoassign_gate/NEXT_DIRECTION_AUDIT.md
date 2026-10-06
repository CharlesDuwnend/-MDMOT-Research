# CVSGM 之后的技术方向审查

CVSGM 已停止。下一候选不能继续增加 gate hidden、训练步数或调 dustbin 阈值；失败发生在第二个 disjoint block，并且接到 B4 后第一组也失败。

## 只保留一个可审计假设

**Local Cross-View Spatial Correspondence Field（LCSCF）**：在 C3/C4→P2/P3/P4 的 paired identity maps 上，预测 bounded 2D offset field，进行双向局部采样/残差滤波；输出再进入固定的 B4 association head。训练增加 local correspondence cycle/occlusion consistency，但不接 GPS、homography、轨迹 ID、future frame 或 tracker solver。

## 为什么先做 collision audit

- GMT 已把 cross-view feature consistency enhancement 作为多摄像头跟踪模块，不能把“跨视角特征一致性”作为独立新颖性。
- CVPR 2024 的 OAFA/ODAF 已将 offset modeling 与 deformable alignment 用于 UAV 多模态检测，不能把“预测 offset + deformable sampling”泛称为首创。
- 因此只有以下窄合同值得继续审计：MDMT 当前 paired identity maps、RoI 前局部 offset、同尺度双向 cycle/occlusion response、无几何/轨迹先验、与 B4 同预算对照。

在完成至少五篇机制级比较、确认没有同任务直接完整碰撞之前，不写代码、不接 official-val/test。若碰撞无法收窄，直接停止候选；若通过，再做 CPU/单 batch 合同和两个 disjoint block 的短门。
