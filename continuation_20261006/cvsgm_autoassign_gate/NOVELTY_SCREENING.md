# CVSGM 预筛查

| 邻近机制 | 已知重合 | CVSGM 可保留的差异 | 风险 |
|---|---|---|---|
| QAConv / TransMatcher | pair-conditioned local correspondence | CVSGM 是共享 backbone identity maps 的 symmetric spatial modulation，非局部匹配 head | 中高 |
| Set Transformer / DeepSets | set/context aggregation | CVSGM 不读取候选集合成员，不做 set pooling；条件来自 paired maps | 中 |
| VLA-ReID / gallery context | set-relative association | CVSGM 不使用 gallery/member composition；需要对照证明增益来自 map modulation | 中 |
| TSMMT / TCFNet / UAVST-HM | cross-view feature interaction/fusion | 作用位置收窄为 C3/C4→P2/P3/P4、共享尺度门控、无轨迹/几何先验 | 高 |
| MVCD | multi-view cross-domain/fragment association | 数据/任务合同不同；不能声称 multi-view fusion 首创 | 中 |
| KeepTrack / event-aware learning | target/candidate intervention or event supervision | CVSGM 的主张是 map-level paired modulation；intervention loss 是训练合同，不是独立贡献 | 中高 |

结论：`HOLD_NARROW_MECHANISM_CLAIM`。在短留出通过前，不写论文主方法、不接 official-val/test、不声称 novelty/first。
