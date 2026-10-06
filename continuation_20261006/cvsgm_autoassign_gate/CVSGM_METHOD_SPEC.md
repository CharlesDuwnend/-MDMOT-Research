# CVSGM 方法合同（探索分支）

## 方法名

Cross-View Scale-Consistent Gated Modulation（CVSGM）。这是候选方法名，不是新颖性结论。

## 输入→算子→目标→输出

- 输入：成对 target/source RGB 帧；共享 Caffe-ResNet-50 C2/C3/C4；每个事件的 target/source frame index 和候选框只用于 RoIAlign。
- 算子：IdentityPyramid 产生 P2/P3/P4 后，按 episode 配对每一层的 target/source 空间图。共享 gate MLP 读取 `GAP(target) || GAP(source) || GAP(|target-source|)` 与尺度 embedding，得到通道门控；共享 3x3 residual 在门控下调制两侧空间图。之后才做候选 RoIAlign 和 B1 association head。
- 目标：监督 factual association、positive removal、same-cardinality known-negative control、hard-negative/irrelevant control，沿用 SCI-ID episode 合同；CVSGM 本身不接当前 tracker ID、GT 初始化、future frame、homography 或 solver 输出。
- 输出：候选 logits、K+1 dustbin logits；模块的空间输出是 paired P2/P3/P4 maps。

## 结构约束

- residual convolution 零初始化，step-0 与同 backbone/head 的 B1 最大绝对误差应 ≤1e-5。
- gate/residual 参数在 P2/P3/P4 之间共享，只保留 3 个 scale gain 和 scale embedding；不物化 K×K 空间张量。
- AutoAssign epoch-60 backbone 严格加载 258/258 个 C2-C4 state tensors；所有比较臂使用同一 checkpoint、同一 BGR mean/std/value-scale、同一 train/eval pair、同一步数和 batch size。
- `valid_K=0` 用 tensor_K=1 的全 false mask 表示；K 上限保持 64。

## 预注册探索门

1. 初始化/机制审计：258/258 strict load；K=1/64；zero-valid mask；step-0 identity；梯度进入 backbone 和 CVSGM residual；所有输出/参数 finite。
2. 短留出：AutoAssign-B1、AutoAssign-B4、CVSGM，train pairs `{23,25,27,28}`，eval pairs `{29,30,32,39}`，相同 600 updates/arm、batch 4、channels 32、ROI 7、无 official-val/test。
3. 继续门：CVSGM 相对 AutoAssign-B1 和 B4 至少在 3/4 eval pairs 中 Recall@1 提升，且 pair-macro Recall@1、MRR、margin 不同时下降；若只赢单一 pair 或只赢 dustbin/Brier，保持 HOLD。
4. 通过短门后，增加 seed 17 和第二组 disjoint train/eval pair；仍不允许接入 tracker。只有两组 seed/pair 都通过，才进入完整 grouped cross-fit。

## 统计单位

独立单位是 paired sequence/pair；同一 pair 内 episode 只用于估计该 pair 的宏平均，不当作独立重复。seed 和 scene/pair 作为 block。所有臂的 step、batch、输入尺寸、归一化、候选宽度上限、评估事件筛选完全一致。

## 禁止的解释

不能把 CVSGM 叫作首次跨视角融合、首次多尺度 ReID、首次 gated attention 或首次 null/dustbin。QAConv、Set Transformer、VLA-ReID、KeepTrack/event-aware learning、MVCD、TSMMT/TCFNet/UAVST-HM 等相邻工作已构成碰撞约束；最终只能根据直接同任务、同输入合同和本次对照结果收窄 claim。
