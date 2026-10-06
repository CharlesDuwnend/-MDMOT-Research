# P29 learned alignment pilot

日期：2026-10-05。P29 在冻结的 AutoAssign R50-FPN 检测器上移植标准 MMCV
multi-scale deformable attention，使用三个严格过去帧和冻结 RAFT reference
point。当前 FPN 直接作为 residual identity；训练只优化 adapter，原检测器、
RAFT、输入分辨率和 AutoAssign loss 保持冻结。算子来自 Deformable DETR/STSN
一类成熟工作，本报告不提出原创性或完整 STSN 复现声明。

## 证据边界

真实 smoke 和三臂 FIT 训练在 physical GPU3 A100-SXM4-40GB 完成，GPU UUID
`GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a`。每臂 240 groups × 5 epochs =
1200 steps，seed 42，AdamW 1e-4，检测器状态哈希保持
`39e5ec506b13592708fc2fd129a68383c7b87411da2daacf1dc353f32bcaa737`。Smoke
验证了 cap100 native 缓存逐数组一致、cap300 identity 一致、有限 loss/梯度、
fixed offset 不更新以及 learned offset 更新。训练和预测没有读取 calibration
标签；独立 scorer 在冻结四臂 predictions 后才读取 calibration5 的 40 个
anchor。没有读取 official-val/test，也没有完整 MIA/MDA/IDF1/MOTA 结果。

权威工件：

- `training/TRAIN_RECEIPT.json`：三臂训练完成、输入哈希、checkpoint 和物理 GPU。
- `PREDICTION_RECEIPT.json`：四臂 cap300 predictions 已冻结，producer 声明未读标签。
- `detection_readout/RECEIPT.json`：独立 scorer 和 calibration 标签绑定。
- `detection_readout/SUMMARY.json`：AP50、TP/FP/FN、pair 级和 recovery 读数。

## Calibration5 读数

以下 AP50 是 pair-macro，范围只有 calibration5 的 40 个 anchor；它是开发证据，
不是正式验证集或 MOT 分数。

| arm | pair-macro AP50 (%) | pooled recall (%) | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| native cap300 | 94.4486906 | 93.5467734 | 1870 | 2230 | 129 |
| current residual | 94.3620944 | 93.5967984 | 1871 | 2332 | 128 |
| past fixed offsets | **94.5166937** | **93.6468234** | **1872** | 2274 | 127 |
| past learned offsets | 94.4505760 | 93.5967984 | 1871 | 2292 | 128 |

`past learned offsets` 相对 native 为 +0.001885 个百分点，相对 current-only
为 +0.088467 个百分点，并在 5 对中胜 current 4 对；recall 不低于 native。
但它低于 `past fixed offsets` 0.066118 个百分点，且 offset-learning 没有带来
相对 fixed control 的宏平均提升。fixed control 相对 native 为 +0.068003 个
百分点，5 对中胜 3 对；它恢复 native miss 4 个、丢失 native TP 2 个，属于
检测层小幅开发信号，不能转换成跟踪收益。

## 研究决定

`KEEP_P29_FIXED_FLOW_CENTERED_RESIDUAL_AS_HOST_CANDIDATE`：固定 flow-centered
3×3 sampling stencil，学习 query conditioner、value projection、attention
weights 和 zero-initialized output residual。它是一个有明确输入→算子→检测
目标→输出的结构性候选，值得接入完整 MIA host 做同协议验证。

`NO_OFFSET_LEARNING_CREDIT`：learned-offset arm 的结果不足以证明任务学习采样
有益；不把它写成方法贡献，也不根据 calibration5 选择 checkpoint、阈值或
偏移半径。

`HOLD_FULL_HOST_INTEGRATION`：下一步必须为 calibration5 的五个 pair 生成完整
连续序列 feature/flow，并在保留官方首帧 GT 初始化、ByteTracker、几何 ID
更新和 MIA 补全的隔离 host 中比较 native、P26 MIA 和 P29 fixed。当前 40-anchor
检测读数不足以宣称 MDA/IDF1/MOTA 改善。若 fixed 在 pair-macro host 指标和
native-output recall 上不能保持方向一致，停止该算子；若保持，再做第二 seed
和预注册的 full host 评测。仍不作 novelty claim。
