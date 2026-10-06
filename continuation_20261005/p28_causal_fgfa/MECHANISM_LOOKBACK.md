# P28 causal FGFA：冻结结果的机制回看

日期：2026-10-05。范围：只读已归档的机制统计、训练损失、final checkpoint 与检测读数；本报告不重新打开原始标注、不重新评分、不进行模型前向、训练或阈值搜索。P26 与 P28 已封存代码及结果保持不变。

**结论：学习后的 FGFA 明显减轻了 uniform temporal aggregation 的损失，但没有保持 native 检测性能。** 三个训练臂均有有效优化；FGFA embedding 确实发生了学习，且在全部 200 个 group×level 对比中，比 uniform 分配更多 current 权重、产生更小的全图相对特征变化。与此同时，当前 cosine-softmax 结构在三个过去帧均有效时不能只选择 current。该限制已由公式证明，但它是不是这次性能下降的主要原因，现有归档不能证明。不能据此断言 RAFT 不可用或“训练没有动”。

## 1. 比较协议与实际结果

本试验是 established-operator pilot，不是原始 FGFA 的完全复现：P26 detector backbone/FPN/head 与 RAFT 冻结，仅训练共享 fusion embedding 和 identity-initialized value residual adapter；原始 FGFA 包含联合训练的 feature/flow/detection 组件。各臂采用 seed 42、相同初始化、相同 240 fit groups 顺序，5 epochs/1200 steps，AdamW lr=1e-4、weight decay=1e-4、clip norm=10、batch=1，无 augmentation/AMP。仅使用固定 final checkpoint，没有 calibration checkpoint 选择。

calibration5 的 40 groups 只对新模块拟合留出；它们已被原 full-train detector 看过，不能称为对整个 detector 未见的测试集。本表来自已封存的 `detection_readout/SUMMARY.json`，共 1999 个 GT observations；AP50 是 pair-macro，TP/FP/FN 是既定 native 输出规则的匹配计数。

|臂|pair-macro AP50 (%)|TP|FP|FN|pooled recall (%)|恢复 native misses|丢失 native TP|
|---|---:|---:|---:|---:|---:|---:|---:|
|native|91.736520|1773|1384|226|88.694347|0|0|
|single_adapter|91.720288|1775|1440|224|88.794397|10|8|
|causal_fgfa|88.769168|1760|1608|239|88.044022|22|35|
|uniform_fgfa|83.669869|1731|1990|268|86.593297|25|67|

FGFA 相对 uniform：AP50 **+5.099299 pp**，5/5 pair 胜出，TP +29、FP −382。FGFA 相对 native：AP50 **−2.967352 pp**，TP −13、FP +224；相对 native 和 single 都是 **0/5 pair 胜出**。single 相对 native 的 AP50 为 −0.016232 pp。

|pair|native AP50|single AP50|FGFA AP50|uniform AP50|
|---|---:|---:|---:|---:|
|27|0.953340544|0.950297500|0.924488963|0.890688641|
|32|0.914965398|0.915461338|0.892388562|0.858456611|
|42|0.989992673|0.989889570|0.982933740|0.933653540|
|64|0.968052538|0.968554797|0.921549481|0.815469564|
|65|0.760474833|0.761811185|0.717097650|0.685225091|

原门槛要求 FGFA 的 pair-macro AP50 同时超过 native 和 trained single、相对 single 至少 3/5 pair 胜出、pooled recall 不低于 native；本次 **FAIL**。恢复 22 个 native misses 不能抵消丢失 35 个 native TP，也不能单独支持继续扩展该配置。

## 2. 实际权重：先说明统计口径

`predictions/MECHANISM.json` 有 120 条记录，即 40 groups × 3 trained arms，每条包含 5 个 FPN levels。导出字段为：

```python
mean_current_weight = w[0].mean()
min_current_weight = w[0].min()
max_current_weight = w[0].max()
mean_valid_past = v.float().mean()  # E[K] / 3，不是 P(K > 0)
relative_feature_l2 = (f - c).norm() / c.norm().clamp_min(1e-12)
```

这些均值和范数包含整个 feature grid，包括 padding/current-only 位置。下表是 **40 个 group 空间均值的等权平均**；p10/median/p90 也是这 40 个 group 均值的分位数，不是像素权重分布。没有归档像素 histogram、valid-only 实测 quantiles 或 foreground-only 统计。

|level|FGFA current 均值|group 均值 p10 / median / p90|uniform current 均值|mean_valid_past|
|---|---:|---|---:|---:|
|P3|0.338731|0.321624 / 0.336653 / 0.360128|0.282704|0.950383|
|P4|0.325215|0.314440 / 0.325147 / 0.337675|0.286779|0.945009|
|P5|0.319844|0.308273 / 0.320246 / 0.334161|0.289259|0.941592|
|P6|0.362413|0.340670 / 0.360908 / 0.394091|0.325711|0.891402|
|P7|0.424268|0.395189 / 0.415092 / 0.474976|0.395707|0.793687|

每一个 group×level 的 FGFA current 均值均高于 uniform；两臂 valid-past 均值完全一致，符合使用相同 geometry/masks。FGFA 各 level 所有位置的最小 current 权重依次为 0.250043631、0.250129879、0.250186503、0.250263006、0.250648141；最大值均为 1。uniform 每层 min/max 均为 0.25/1，single 为 1/1。

最大值 1 包含所有 past 都无效的位置，不与“三个 past 都有效时的上界”矛盾。P7 全图 current 均值较大，也不能直接解释成模型学到了更强的粗层 current 偏好，因为 P7 有更多 K=0 的位置。

## 3. cosine-softmax 的可证明限制

设一个位置有 K 个有效 past，current embedding 归一化后具有单位范数。current self logit 为 1，past cosine logits 属于 [−1,1]，没有 temperature、独立 current logit 或 residual bypass。于是：

\[
\frac{1}{K+1}\leq w_0\leq\frac{e^2}{e^2+K}.
\]

|K|单位范数前提下的下界|上界|
|---|---:|---:|
|0|1|1|
|1|0.5|0.880797078|
|2|0.333333333|0.786986042|
|3|0.25|0.711234594|

因此三个 past 有效时，past 总质量至少为 **0.288765406**，无法仅通过该有界 softmax 把所有 past 权重置为零。这是结构性质，不依赖本次训练是否收敛。它不意味着最终 output 永远不可能等于 current：值相同或学习到补偿时，输出仍可能相等。

需要保留 `F.normalize(..., eps=1e-6)` 的边界：若 current 归一化后的范数 r≤1，self logit 为 r²，past logit 属于 [−r,r]，则

\[
\frac{1}{1+K e^{r-r^2}}\leq w_0\leq\frac{1}{1+K e^{-r-r^2}}.
\]

通用上界仍为 e²/(e²+K)；保守通用下界是 1/[1+K exp(1/4)]，不能无条件声称 uniform 下界。r=0 恰为 uniform。当前 embedding norm 未归档，不能声明所有实际像素均达到单位范数。以上公式由协议审阅者独立复核。

### 利用已归档矩的 valid-only 可行区间

这些是代数推导的可行界，**不是重新测得的像素直方图**。对每个 group/level，令 u 为 uniform current 空间均值，μ=3×mean_valid_past，t=p3=P(K=3)。由概率和、E[K] 与 E[1/(K+1)] 可得：

```text
A = 1.5 μ − 3 + 3 u; B = 6 − 6 u − 2 μ; C = 3 u + 0.5 μ − 2
p2 = A − 2.25 t; p1 = B + 1.5 t; p0 = C − 0.25 t
L = max(0, −B/1.5); U = min(1, A/2.25, 4 C)
L ≤ p3 ≤ U; C − U/4 ≤ p0 ≤ C − L/4
E[w0 | K>0] = (a − p0)/(1 − p0), a = FGFA current 空间均值
```

最后一个式子随 p0 单调递减。逐 group 求界，再对 40 groups 的界等权平均，得到下表；浮点矩的约 2e-6 数值误差不应解释成统计精度。

|level|P(K=3) 界 (%)|P(K=0) 界 (%)|FGFA E[w0 \| K>0] 界 (%)|
|---|---:|---:|---:|
|P3|93.2347–94.4371|3.7594–4.0599|31.0768–31.2914|
|P4|92.7152–93.9056|4.3087–4.6063|29.2667–29.4861|
|P5|92.3404–93.5419|4.6310–4.9314|28.4625–28.6874|
|P6|86.8452–88.3444|9.3375–9.7123|29.3954–29.6847|
|P7|75.7576–78.1313|18.2323–18.8258|29.1184–29.6056|

K=0 还包含 current 合法但所有历史 warp 均被拒绝的位置，不等于 padding 比例。去掉 K=0 的代数界后，P7 的 current 平均权重约 29.1–29.6%，不能用全图的 42.4% 描述有效历史区域内的选择强度。

## 4. 输出特征确有变化，但 L2 不能直接解释 AP

下表同样对 40 个 group 统计量等权平均。FGFA 的全图输出相对 current 的 L2 在全部 200 个 group×level 中小于 uniform。

|level|single L2 均值|FGFA L2 均值|uniform L2 均值|FGFA group p10 / median / p90|
|---|---:|---:|---:|---|
|P3|0.081830|0.258264|0.298201|0.220556 / 0.260731 / 0.299238|
|P4|0.076100|0.209781|0.233650|0.176064 / 0.209478 / 0.251482|
|P5|0.064782|0.191678|0.208916|0.155970 / 0.189956 / 0.234994|
|P6|0.064562|0.196086|0.217193|0.154443 / 0.197350 / 0.238501|
|P7|0.062571|0.193049|0.208241|0.139963 / 0.192526 / 0.240606|

该 L2 同时混合了 warp、aggregation、已训练 value transform、背景和 padding；不是目标定位误差，也不是 flow 损害的直接证据。single 的特征变化仍有 6–8%，但 AP 接近 native，这本身说明“特征 L2 更大 → AP 必然更差”不能成立。

## 5. 训练和 checkpoint 没有显示“学习分支未工作”

三臂均有 1200 条有限 loss/gradient 记录，serialized group 顺序完全相同，每个 epoch 覆盖相同的 240 distinct groups 各一次，总计各 5 次。所有 gradient norms 非零且最大值小于 10，clip 未激活。各臂 checkpoint 的 seed、module/protocol/schedule SHA 与 final step=1200 元数据一致；frozen detector state SHA 一致。

|臂|epoch1|epoch2|epoch3|epoch4|epoch5|同 group 第1→5次 loss 下降|
|---|---:|---:|---:|---:|---:|---:|
|single|0.806856472|0.798570915|0.793039028|0.789440397|0.786714188|228/240|
|FGFA|0.897391933|0.869356143|0.854772852|0.843604768|0.836138262|240/240|
|uniform|0.935286242|0.913654570|0.900986368|0.891388391|0.884075992|238/240|

|臂|gradient norm min / median / max|value residual Frobenius / identity Frobenius|value bias L2|
|---|---|---:|---:|
|single|0.276114 / 0.623573 / 3.668195|0.064840600|0.051667515|
|FGFA|0.265535 / 0.683258 / 5.330771|0.079210989|0.069349192|
|uniform|0.292675 / 0.664472 / 4.967496|0.085560560|0.081739806|

这里 identity 的 Frobenius 范数为 16。single 与 uniform 的未使用 embedding bit-identical，并结合共同初始化 receipt 用作初始化参照。FGFA embedding 初始范数 16.017787064、变化范数 3.823389599、相对变化 **0.238696493**；三个 conv weight 相对变化分别为 0.14384216、0.28207946、0.26410106。因此学习 embedding 并非静止分支。

所有 arm/epoch 的 `loss_center` 均值为 0.0597406392218545，符合 center prior 冻结。receipt 中 first30→last30 的均值有所增加，但它们对应不同的图像 group 混合，不能当作发散证据；匹配的 epoch 和逐 group 对比均显示优化有效。另一方面，这不证明已经收敛或 1200 steps 足够。

## 6. 可以成立与尚不能成立的解释

成立的观察是：相同 flow/mask 下，learned FGFA 整臂比 uniform current 权重高、特征扰动小、AP/TP/FP 好；但相对 native 仍出现稳定的五对下降，损失 native TP 多于恢复 misses。成立的结构限制是：有界 cosine-softmax 在存在有效历史时没有 exact-current 的选择；当前初始化 identity 只适用于 value adapter，不等于 temporal aggregation 输出初始化就与 native 一致。

“学习后的选择减轻 uniform 混合损害”与数据相容，但 **FGFA 与 uniform 还分别学到了不同的 value adapters**。因此这只是 whole-arm 比较，不能把 +5.099299 pp 全部因果归到 softmax 权重。现有归档也没有只改最终权重、共享同一 value checkpoint 的干预。

尚不能证明的原因包括：softmax current 上限是不是主要瓶颈、哪些目标受损来自 flow 局部误差、冻住 head 带来的分布不适配有多大、额外训练是否可修复、某层/某个 lag 是否负责退化。没有做这些干预，不应将其写成已定位的根因。全图均值、全图 L2 与优化记录不足以替代目标级证据；另行 flow lookback 的结论也不能由此报告代替。

本回看支持停止扩大 **当前 P28 配置**，保留其冻结结果。若另开新算子，应作为有独立来源审查与独立契约的新试验；本报告不提供 novelty、正式 MOT 增益或新训练授权。

## 7. 可核验输入

机制统计、training receipt/module 与 canonical prediction receipt 的绑定，final checkpoints 与 training receipt 的绑定，以及 scorer/SUMMARY 与 readout receipt 的绑定均已核验。canonical prediction receipt 只规范化 Python 3.8 `__main__.__file__` 相对路径 bookkeeping，原文件保留，scorer 字节未改。loss 文件 SHA 在本次 lookback 核验，不能声称其逐文件 SHA 原本已被 receipt 封存。

|文件|SHA-256|
|---|---|
|temporal_module.py|71d3b86dc2a2dd7b738155a6be14221c7633762ff61131733ea8737b2fc54fd9|
|test_temporal_module.py|e65d2bc81ac7b6e246e7033ec58354bcc9cdcc570b14e450fadd712e183c0012|
|PORT_MAPPING.md|9dfda70225d394dd4eeeb1365ac75de436d6270eaac0424cb06bb1b9e4401387|
|score_detection.py|50bbbcd33b81c339dc2819f5730677dc1b9b9e4afa398981831c5bd7f8db3389|
|predictions/MECHANISM.json|ca6cc611fbddd937aebb8e0ae21b03f380aaf0cc2a1ad4134746a59c72ab0767|
|training/TRAIN_RECEIPT.json|15b7fadc79183132a8fc3386297e56902988258c29fb7cdfb04534131cc0bb1e|
|PREDICTION_RECEIPT_CANONICAL.json|7bd2da4b71dc378d951014e01edc9ddce4c4e9b65636b03fa35c7545444dbff9|
|detection_readout/RECEIPT.json|79491f82ac2d974c1f4e05618c61ec960f6a41f0607c54942190be755b142b73|
|detection_readout/SUMMARY.json|f0aa7a8dfa02c97f91516757ab66d4c34261e05584c792266df0c1dc81b01057|
|TRAIN_PROTOCOL.json|560825a72c49d6e48d788af660b63ad5d2f724c49f94f1895e2fba80ee2b8036|
|training/single_adapter/final.pth|cdb73ec9ec9483710ab44227f8fbe7e60cb29f79a58fcd492650b116eb13ccee|
|training/causal_fgfa/final.pth|1ccca1c811f100daaec33977aeea6f299cb064a4ead31314f1bc8e3f8ef907a9|
|training/uniform_fgfa/final.pth|eaf9e2521fb2afa8bbe4b2d53dafc1e583bb780d891d4698268d63802350dadc|
|training/single_adapter/LOSSES.jsonl|33d4575756be7cf062fc5961b23ac6fd16bf4f39b3836236bf906ed121bc1206|
|training/causal_fgfa/LOSSES.jsonl|d72985a737cbee57f7712906ebf3a8cecb152469f3c8382a77c2b523313434bd|
|training/uniform_fgfa/LOSSES.jsonl|71694d1720dd8eeca1e626bd7d1e5481de345c09b493956512b2e2feb3b5def7|

frozen model state SHA-256：`39e5ec506b13592708fc2fd129a68383c7b87411da2daacf1dc353f32bcaa737`。
