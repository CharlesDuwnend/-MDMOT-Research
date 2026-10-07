# EGIA 内部机制：作者解释与图设计审计

**当前可写的是受控比较的对象和已完成的训练/校准诊断；Hhier、Hflat、E10 的新线上分数尚未完成，不能预设正结果。** 本文只读核查模型、训练程序、冻结 manifest 与已有证据，没有修改推理代码或启动 GPU。

## 1. Hhier 与 Hflat 比较什么

线上名称为 `H_paired_hierarchical` 和 `H_paired_flat`；两者使用相同 `fc_full` 宿主路径，只替换 F00 的 `source_model.anchor`。输入均为同一组10维 detector/track-pool summary，训练使用原 VisDrone train 的8024行、39序列、32 flights；cal8 为8411行、8 flights，与 fit 无 flight 重叠。未知监督行沿用原 prepared mask。两个新头共同使用 fit-only float64 StandardScaler、三类 β=.5 事件权重；coverage 子集6093行保留原事件绝对权重，未再次归一化。C=1、L2、lbfgs、tol=1e-4、max_iter=2000、seed=20260923 相同。SRC、owner cue models、γ=.6331914、source pool=`legacy_all`、teacher fusion α=.4、selective margin=.2 均冻结。

令三类为 NEW、EXISTING、CLUTTER，`z` 为共同标准化输入：

- Flat：三个线性 logits 经一次 multinomial softmax 得到 `p_anchor`。
- Hierarchical：`f=P(foreground|z)`、`c=P(covered|foreground,z)`；输出 `p_anchor=[f(1-c), fc, 1-f]`。两分支均计算概率；这不是先以阈值删掉 clutter 再运行 coverage 的硬级联。

在共同事件权重下，hierarchical 的训练 NLL 等于 foreground 二分类 NLL 加上前景行的 coverage 二分类 NLL；后者没有引入额外监督样本。两头进入相同 downstream：只给 EXISTING log-probability 加 `γC`，三类共同归一化；再与 frozen teacher 作 log-probability fusion；最后执行 selective admission。**anchor 分解、owner co-reference、summary fusion、admission 是四个不同算子，图中不能合并归因。**

这一配对能支持“在共同输入变换、事件权重与固定优化配方下，不同概率参数化/因子分解对预测及闭环跟踪的作用”。它没有证明所有参数化的 L2 几何完全相同，也没有消除参数化本身造成的假设空间差异；同 C 不能写成“严格相同有效容量/正则作用”。完整闭环差异还包含第一次不同 admission 之后的轨迹和 source ledger 变化，不能称两次运行的每个事件值始终相同。

**F00 是外部 canonical reference，不是此次共同配方的 hierarchical 组。** 原 F00 对两个 binary tasks 分别做类别平衡，coverage scaler 只 fit 前景行。F00−Hflat 混入这些训练配方差异；因子分解的主对照必须是 Hhier−Hflat。γ 保留原 F00 的值，结果只说明这一冻结 downstream context 中的比较，不是各头各自调到最优后的比较。

证据：[预声明](../PHASE2_HEAD_PREREGISTRATION.json)、[训练实现](../tools/fit_paired_anchor.py)（inspect_data:109–134；fit:233–248）、[训练收据](../artifacts/PHASE2_HEAD_TRAIN_RECEIPT.json)、[独立读回审计](../artifacts/egia_paired_independent_audit.json)。12个 CPU tests 已通过，其中 paired contracts 为3轮×3检查；不是正式 MOT 改善证据。

## 2. 已完成校准诊断应如实报告

| 共同配方 head | cal8 anchor weighted NLL | 加冻结 γC 后 weighted NLL |
|---|---:|---:|
| Flat | 0.6521 | 0.6198 |
| Hierarchical | 0.7028 | 0.6555 |

Flat 在当前 cal8 的总体及三个类别 anchor NLL 均较低，因此现在不能写“hierarchy 已证实更适合稀有 covered 类”或“hierarchy 必然改善校准”。这些是原 train cal8 的固定轨迹输入诊断，尚未包含 teacher fusion/selective 的完整闭环，不能替代 test-dev MOTA/IDF1；cal8 无择优选模型或调参数。

数值细节：已存 `with_frozen_source_gamma` 诊断经 `predict_rows` 读取 prepared float32，flat 与 live float64 输入的最大 posterior 差约 `9.97e-9`；hierarchy 为0。该精度差不会改变冻结模型契约，但不宜宣称离线诊断 posterior 与线上逐位相同。NLL 使用4位小数足够。[诊断来源](../artifacts/phase2_head_calibration_metrics.json)

## 3. A04/A05/A06 与 E10 是 fusion × selective 2×2

**不是 coherence × selective。** 下表四格都固定 hierarchy、coherence γ=.6331914、SRC、birth eligibility 与同一 F00 estimators。

| Fusion α | Selective | 对照 | 已完成 MOTA / IDF1 (%) |
|---|---|---|---:|
| 0 | off，直接 argmax | A04_coherence | 57.6294 / 70.9271 |
| 0 | on，margin=.2 | A05_selective | 57.9553 / 71.3309 |
| .4 | off，直接 argmax | E10_fusion_no_selective | **待完整线上结果** |
| .4 | on，margin=.2 | A06_full = F00 | 58.0251 / 71.3737 |

旧三格已复核 source estimator/γ、对应源码、17序列/6635帧、预测 SHA 和相同 scorer/GT；它们是当前 F00 的模型控制，不是把旧 H2 分数当新方法。[复用鉴定](../artifacts/egia_existing_controls_audit.json) E10只关闭 selective flag；无其他候选 policy fallback。

新 C11 须先与当前 F00 全17预测 SHA 匹配，E10 须有17序列/6635帧、完整预测/评分/receipt，才补齐表格并作条件差值。以 `Y_{a,s}` 记 fusion/selective 四格，报告 `Y_{1,1}-Y_{1,0}` 与 `Y_{0,1}-Y_{0,0}` 的 selective 条件效应、对应 fusion 条件效应，以及描述性交互 `(Y_{1,1}-Y_{1,0})-(Y_{0,1}-Y_{0,0})`。按14 flights配对，不能把6635帧作为独立重复；此前 test-dev 已被观察，不能宣称确认性显著性或无偏参数选择。

若要研究 coherence 本身，应使用已完成 F00/F01 的 γ干预及 pair_shuffle，分别解释 log-evidence 的作用和 cue-owner 配对。它们与上述 fusion/selective 四格是不同问题。当前四格不能推出 coherence×selective 交互，也不能推出 flat×hierarchy 与各下游算子的交互。

## 4. 建议机制图：先解释算子，再呈现结果

建议两部分，避免画“每加模块必上涨”的累计柱状图。

**机制部分：** 共同10D输入 → 同一 scaler → 并排 Flat softmax / Hierarchical `[f(1-c),fc,1-f]` → 相同 `+γC` → 相同 teacher fusion → 相同 selective admission。owner geometry/appearance 的配对只连接 `C`，不要连接到两种 summary anchor；teacher 从独立 side branch 进入 fusion。F00原训练配方放旁边作 reference 标注，不放入两head配对框中。所有三个类别保持同色，线型同时编码，灰度可读。框内只写输入/操作/输出，不填人为概率、Oracle owner 或 GT ID。

**证据部分：**

- 现阶段可画 cal8 两head anchor NLL及三类 NLL并排点图，明示“固定轨迹校准诊断”；保留实际 Flat 优势，不用正向设计预设 hierarchy 胜出。
- 完整线上评分后，用 Hhier−Hflat 的 MOTA/IDF1 paired difference、每flight散点及 FP/FN/IDs 变化呈现 head 对照；原 F00单独标 reference。零/负差值原样展示。
- Policy 四格可用两条线：横轴 selective off/on，线型区分 fusion off/on；MOTA与IDF1分开。标注旧三格“复用且哈希核验”、E10“新推理”，不能把连线当时间上的累计提升。未完成 E10 时不画虚构点或0占位值。
- 若加一个同输入 posterior 小例子，先按与标签无关的固定规则选一个已封存 cal8 input row，再计算两head及共同downstream；注明固定输入诊断。闭环结果的各组事件总体不同，不可混成一个“同帧同状态”的概率对照。

当前 online seam 先执行宿主 birth eligibility，拒绝者未进入 EGIA（[runtime.py](../belief/runtime.py):667–682）。旧 F00 有6809个 EGIA事件、3575个 selective rejection、3234个 keep、0个 new-certificate。图中突出对合格 unmatched candidates 的“强 non-new 抑制 / 不确定时保留 incumbent”即可；不能画成已实证的双向低分补生模块。`coherence_nonzero_events` 记录的是未乘 γ的原始证据，即使 γ=0也可非零，不能用该计数证明 coherence 产生了有效干预。

可复用的作者表述：*We compare a flat three-class anchor with a foreground/coverage factorization under shared fit-only normalization and event weights, while freezing source coherence, summary fusion and admission. A separate fusion-by-selective factorial measures their conditional closed-loop effects. Calibration diagnostics and benchmark trajectories are reported separately.* “优于”“改善”“更稳健”等结果词留待完整评分核验。
