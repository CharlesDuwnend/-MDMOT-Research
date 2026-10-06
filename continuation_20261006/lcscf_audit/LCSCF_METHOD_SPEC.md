# LCSCF 方法合同

## 输入

- target/source RGB frames，经共享 AutoAssign-initialized Caffe R50 C2/C3/C4 与 IdentityPyramid 得到 P2/P3/P4。
- 每个候选的 target/source RoI maps `[B,K,C,H,W]`；tensor K∈[1,64]，valid K 可为 0。
- train-only pair-level same-ID、known-negative、no-match 标签。

## 算子

对每个 scale 和每个候选，计算归一化 target/source map 的 3×3 局部相关；分别得到 source→target 与 target→source 的 soft transport plan。用反向 transport 的 cycle agreement 产生可靠性，transport mass 产生可见性 gate；输出为原 map 加 gated transported residual。P2/P3/P4 共享局部 transport 参数，仅保留 scale embedding。B4 association head 保持不变。

## 目标

`L_assoc`：原有 B4 factual/multi-positive/known-negative/no-match 目标。
`L_cycle`：双向 transport 的 cycle agreement，只在 pair-level positive 行启用。
`L_mass`：positive 行保持非退化 transport mass，negative/no-match 行不鼓励伪对应。
`L_anti-collapse`：对 zero-field、uniform-plan、detached-transport 做显式对照；不使用像素 GT、homography 或 tracker ID。

正式训练前必须先验证 loss 的合法 mask、无未知负例误标、无 candidate-set 泄漏、K=0/1/64 和 finite gradient。

## 预注册短门

- AutoAssign epoch-60 backbone，SHA-256 `8894ea5ffe8309017d78e2dac359d405b38aa66725e1b465a8dce30beb12c901`。
- arms：B4、LCSCF-B4、zero-field、uniform-plan（至少 B4/LCSCF-B4 先跑）。
- train block 1：23/25/27/28；eval block 1：29/30/32/39。
- train block 2：42/44/45/50；eval block 2：51/53/54/58。
- 相同 BGR 预处理、600 updates/arm、batch 4、channels 32、ROI 7；GPU 只用 A100 40GB GPU3。
- advancement：两组 block 都满足至少 3/4 pair R@1、macro margin 不降、transport anti-collapse 三项通过，才允许扩大训练。
