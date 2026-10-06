# MDMOT 前几轮实现审查（2026-10-05）

本审查覆盖 P24–P30，并回看 P6/P7R/P10/P15/P16 中已经发生过的失败回查。审查先检查代码和收据，再做可重放或 CPU 反例；没有读取 official-val/test，也没有把校准结果改写成正式结果。

## 结论

发现一个会改变研究判断的 P29 实现错误。`p29_learned_alignment/residual_deformable.py` 的 `_flow_sample` 用 `0..W-1, 0..H-1` 作为光流栅格采样坐标，但 AutoAssign offset=0 的 FPN 节点在 detector-image 坐标中应为 `j*stride, i*stride`。随后代码再除以 `stride`，造成 deformable-attention reference displacement 的 stride-dependent 缩小。P28 的 `warp_past_features` 已使用 `j*stride/i*stride`，因此 P29 的 query reference 和 conditioning warp 实际上没有使用同一坐标合同。

最小 CPU 反例使用宽度方向线性光流 `flow_x[x]=x`、stride=8。原实现得到 `[0, .125, .25, .375]`，正确 feature displacement 应为 `[0, 1, 2, 3]`。修复副本通过同一合同测试。证据和修复副本在 `../p29_repair_20261005/`。尝试启动 real GPU smoke 时，程序在读取第一份 fit archive 前就停止：此前授权清理删除了 P28 的 feature/flow archive，receipt/index 仍在，但 `/raid/datasets/chc_data/claude_try_MDMOT_p28_causal_fgfa_20261005/fit/features/23-1-t000001.npz` 已不存在。这不是新的模型结果，状态记为 `CACHE_REGEN_REQUIRED`。

因此：

- P28 的 FGFA 负结果目前没有发现足以推翻它的实现错误；它的 native identity、梯度、冻结状态和检测读出链已通过检查。
- P30 的 host 注入链没有发现实现错误。原始 P26 host 对 pair27 的直接重放，与 P30 native override host 的两个 JSON 输出逐字节相同；SHA-256 分别为 `0766307f…` 和 `dad01aa…`。
- P29 的 calibration 检测结果和 P30 的 fixed-host 负结果只能描述**错误 reference 坐标下的 P29 实现**。它们不能作为修正后 P29 算子的论文级失败结论。P30 native 对照仍然有效，fixed 与 native 的差异不再能归因于一个已知正确的算子。
- P29 报告还有一个只影响文字的数值问题：learned-current 的精确差值为 `0.0884816377 pp`，DECISION 中为 `0.0884815377 pp`（舍入误差）；正文写成 `0.088467 pp`，属于文档错误，不是实现错误。

## 前几轮回查

| 阶段 | 回查结论 | 对结论的影响 |
|---|---|---|
| P24 | 只有 fit15 首帧几何诊断；没有正式 MOT 效果。背景门失败和尾部外推已保留，不能将几何检索写成身份真值。 | 结论保持 substrate，不是新方法 |
| P25 | 用户要求先复现成熟基线；没有启动训练、读图、读 GT 或生成数组。 | 没有失败结果需要解释 |
| P26 | 基线收据和独立打分保留；pair27 直接 P26 host 重放与 P30 native 逐字节一致。 | 基线/host 注入链得到额外支持 |
| P27 | 历史流与 fresh native detector 的 sampled parity 明确失败（42/90）；没有放宽门限。P28 使用 fresh native feature/detection control，因此没有把历史流替代为控制。 | 只保留为可见性支持，不污染 P28/P30 效果结论 |
| P28 | 代码坐标域和 past-only 合约与 P26 offset-zero 语义一致；冻结模型、identity native output、训练梯度和独立 detection readout 收据一致。未发现 material bug。 | P28 STOP 仍可作为该具体 FGFA port 的结果 |
| P29 | 发现上述 stride 坐标 bug；原 P29/P30 fixed 结果不能作最终方法否定。 | 必须修复后重新 smoke、训练、预测、host 评分 |
| P30 | native/fixed 均 3900 帧、容量和 JSON 结构完整；直接 P26 重放验证了 native host。JSON 中少量重复 track ID 是继承自 P26 host 的行为，两 arm 共享同一语义，不是 P30 注入新引入的差异。 | host 实现通过；fixed 科学结论暂缓 |
| 更早 P6/P7R/P10/P15/P16 | 既有回查已修复 P7 草稿的错误 triplet/采样/eval、P10 的数组别名和归一化问题、P15/P16 control_ce 错 checkpoint；修复后结论分别为 P7R 无实质 triplet 增益、P10 STOP、P15/P16 NOT_CONFIRMED。 | 这些失败可以继续引用其修复后的窄结论，不能引用旧错误版本 |

## 下一道门

1. 依据 P28 的冻结 `FRAMES/FEATURE_INDEX/FLOW_INDEX` 计划重新生成 feature/flow archive；先核对 RAID 可用空间和 GPU3 外部作业，再使用 `p29_repair_20261005/residual_deformable.py` 做 physical GPU3 real smoke：检查 P28 warp 与 P29 reference 的 stride/flow 一致性、cap300 identity、有限梯度、冻结 detector state hash。
2. 在同一 P29 protocol、seed、fit groups 和 last-step selection 下重训三臂；不读 official-val/test，不扫阈值或 checkpoint。
3. 冻结 calibration predictions，重新做独立 detection readout；只有通过预注册 gate 才重做完整 P30 host。
4. 在修复 run 完成前，不把 P29 fixed residual 写成论文方法，也不把其原始 STOP 写成对该算子的最终否定。

机器可读审计：`AUDIT.json`。当前状态：`P29_IMPLEMENTATION_DEFECT_FOUND_P29_P30_NEGATIVE_NOT_FINAL`。

## 修复版 P29 复查结果（第二道实现门）

第一次修复训练只跑到 current-only 第 600 步即主动停止，没有 checkpoint、预测或评分。随后 CPU 反例发现第二个问题：P29 将当前 query 的 warp-valid mask 直接展平成 raw past value token 的 `key_padding_mask`。当 flow 把 query 指向别处时，这会错误地屏蔽仍在图像域内的 raw source node。该中止尝试及日志归档在 `attempts/p29_stride_only/`；没有用于科学结论。

修复副本现在同时修正：

1. reference flow 采样使用 offset-zero detector-image domain `j*stride/i*stride`；
2. raw past value mask 单独使用 source lattice 的 image-padding mask，不复用 query endpoint mask。

双域 CPU/MSDA 合同通过正负位移、raw padding 和各向异性 resize 反例；修复源 SHA-256 为 `1c597803d5ba1e60b9276eab8b66404726b636909e8d488b1c9be14ed5e3cd80`。GPU3 smoke、三臂各 1200-step FIT、40 组 calibration prediction 和独立 detection readout 均完成，官方 val/test 未读。

修复版 calibration pair-macro AP50：native `0.9444869057`，current `0.9443215136`，fixed-offset `0.9455491781`，learned-offset `0.9444387540`。learned 相对 native 低 `0.0000481517`，相对 fixed 低 `0.0011104241`，且只在 2/5 pairs 胜 fixed；因此停止在 P30 host 之前。fixed 的小幅领先不能归因于 learned offset 方法，也不授权 host/MOT 结论。

机器可读结果：`p29_repair_20261005/REPAIR_DECISION.json`；状态：`STOP_REPAIRED_P29_LEARNED_GATE_NOT_PASSED`。
