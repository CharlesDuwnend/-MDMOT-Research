# P47：P14 表征的原始 MIA 前门接入

状态：`ENGINEERING_CONTROL_COMPLETE; NO_PAPER_METHOD`

P47 不是新的表征训练，也不做论文创新声明。它验证 P14 已通过 held-out retrieval gate 的 DINO residual descriptor，是否能在**原始 MIA wrapper 的实际 refresh 调用链**中提供有效跨视角证据。P37/P42/P44 使用的是简化 host；P43 还发现了 owner continuity 缺陷，因此不能把那些结果直接当作 P14 在 MIA 中失效。

## 唯一改变

在 P26 的 `A_same_target_refresh_same_ID`、`B_same_target_refresh_same_ID` 和旧目标 refresh 的实际调用位置，加入 P14 descriptor 与原几何投影的联合候选代价；检测器、ByteTrack、首帧 XML 初始化、H 更新、补框、NMS、输出和评分协议保持 P26。候选代价沿用 P37 已冻结的比例：

```text
cost = 0.7 * normalized_geometry/2 + 0.3 * (1 - cosine)/2
```

`geometry <= 2` 是 P37 固定可行域，P47 不扫阈值。descriptor 由 P37 已封存的 P14 checkpoint 产生，不重新训练 P14。

## 接入边界

只对原 MIA 判为 new/old-unmatched 的对象提议匹配；原生同 ID confirmed 边保持不变。每帧先产生一对一候选，再由 collision-safe owner map 应用到当前 track arrays；不做输出生成后的二次 relabel。P47 同时记录 raw proposals、accepted proposals、collision skips、owner continuity splits 和 descriptor-row IoU 对齐，防止把无效表征或行错位误判为算法效果。

## 结果边界

首次只跑 fit pair23，使用 P26 同一 700 帧和首帧 GT 协议。只允许报告完整 JSON 输出的 MDA/IDF1/MOTA、switches 和工程审计；不打开 official-val/test，不据此宣称 novel。真实 replay 中 geometry-only 前门达到 `MDA=0.75110`，P14 联合代价仅为 `0.67475`；这说明增益主要来自一对一几何前门接入，而不是 P14 descriptor。该接入与已有 Hungarian/位置代价跨视角匹配属于同一常见机制，故 P47 作为论文方法停止，仅保留为后续新模块的强工程控制。
