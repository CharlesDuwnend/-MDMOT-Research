# LCSCF 创新与可行性筛查

审查日期：2026-10-06。状态：`KEEP_FOR_MECHANISM_GATE_ONLY`。这不是 novelty/first 结论。

## 窄技术合同

Local Cross-View Spatial Correspondence Field（LCSCF）只主张一个可检验的算子：对 paired target/source identity RoI maps，在 pooling 前用固定局部邻域的双向 transport plan 进行 feature transport；transport 由 source→target 与 target→source 的局部相关共同决定，加入 cycle agreement 和可见性/transport-mass gate，再把 transport residual 交给冻结结构的 B4 association head。算子不读取 candidate set、tracker ID、future frame、homography 或 solver 输出。

训练只使用合法 pair-level same-ID/known-negative/no-match 标签；没有像素对应标签。offset/transport 是 latent，必须报告 zero-field、uniform-plan 和 detached-transport 对照。

## 五项机制比较

| 先验 | 机制重合 | 本合同允许的差异 | 风险 |
|---|---|---|---|
| MIA-Net / MDMT baseline | homography/common-view、跨机局部/全局匹配、遮挡补充 | 不改 homography solver；只研究 RoI identity map 内 latent local transport | Tier 1/2 高 |
| TSMMT / TCFNet / UAVST-HM / GMT | 跨视角 feature interaction、tracklet/global consistency、cross-view feature consistency | 不用轨迹、反馈、几何、global trajectory；只在 pair map 内做 bidirectional local transport | Tier 2 高 |
| QAConv / TransMatcher / local correspondence ReID | query-gallery 局部相关、局部匹配/attention | transport 在 association head 前修改 spatial identity maps，并用双向 cycle/visibility response 作为训练合同 | Tier 2 高 |
| Locally Aligned Feature Transforms / correspondence-structure ReID | camera-view local alignment、可学习对应结构 | 不学习 camera-pair transform；只使用当前 paired RoI maps 和同一共享算子 | Tier 2 中高 |
| OAFA/ODAF、Multi-Frame Attention with feature-level warping | offset/deformable/feature warping，且含 UAV 场景 | 不使用多模态或连续视频 dense flow；使用 pair-label latent transport + K64 association | Tier 2 高 |
| ReST / graph matching / FG² | graph/correspondence/pose matching | 不建跨相机轨迹图，不估计 pose，仅检验 identity map transport 是否提升 B4 | Tier 2 中高 |

## 审查结论

没有检出同时满足“MDMT、paired identity RoI maps、pooling 前双向 latent transport、cycle/visibility anti-collapse、B4 增量”的直接同任务完整碰撞；但算子级碰撞明显，不能宣称首次。只有以下结果同时出现，才允许进入第二轮：

1. transport 梯度进入真实 C3/C4/identity pyramid；
2. zero-field 与 uniform-plan 对照被显著超过；
3. 正例 transport mass/cycle consistency 改善，而 known-negative 不出现相同改善；
4. B4 上至少 3/4 pair 的 Recall@1 和 margin 改善；
5. transport 不依赖阈值、solver 或 official-val 调参。

否则状态改为 `STOP_LCSCF_MECHANISM_INVALID` 或 `STOP_LCSCF_NOT_INCREMENTAL`。
