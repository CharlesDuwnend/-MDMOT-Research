# P31 Causal Evidence Intervention Network (CEIN)

日期：2026-10-05。本文封存一个已停止的 MDMT/MDMOT 机制原型，不是效果结论，也不授权训练或 official-val/test 评估。

## 研究边界

P26 的 detector、box、几何候选和 host 保持冻结。CEIN 只接收已经抽取的对象级支持观测：

```text
strict-past target observations
  -> independent set encoder (token + normalized age, masked softmax)
  -> order-invariant support representation z
  -> leave-one-support-out influence and quality
  -> pair operator [zL,zR,|zL-zR|,zL*zR,quality,count]
  -> match logit + explicit left/right null logits
  -> later dustbin assignment
```

每个支持观测的 age 必须非负；未来观测、另一视图的未声明同步信息、相机位姿/标定、全局物理 ID 都不在输入契约内。集合顺序不能改变 `z`、质量或匹配输出；mask 外的数值不能影响输出。空支持集只表示 API 的有限值边界，真实 host 策略需要显式决定如何处理。

## 可验证的目标

训练阶段（尚未执行）只能使用 legal FIT/grouped split 中的身份/配对标签。候选损失应同时覆盖：正负配对、leave-one-out 稳定性/冲突一致性、显式 null 的校准；具体权重必须在真实特征实验前固定。`sinkhorn_with_dustbin` 只是可微分 assignment primitive，不是评测器，也不自动决定 ID。

候选的技术差异在于：对对象级支持集合做显式 leave-one-support-out 干预，并把支持影响/质量作为配对证据的一部分；它不声称“时序融合”“记忆”“图匹配”或“跨视图融合”本身新颖。

## 当前证据边界与停止原因

- CPU 合约已通过，见 `P31_CPU_RECEIPT.json`；没有读取数据、标签或官方指标。
- `ID_AUDIT.json` 只审计 train XML 的原始 ID 重叠，不能解释为物理跨视图身份。
- 真实冻结 detector/ROI 特征尚未接入；不能写 MDA、IDF1、MOTA 改善。
- 本地 P3 已覆盖因果历史、support/conflict 证据、mixed-bag 反例目标和接入输出；P31 属于 Tier 1 直接碰撞，状态为 `STOP_P31_DIRECT_COLLISION_WITH_P3`。
- 因此不做真实特征 port；新增 influence/null/dustbin 只作为代码诊断，不能写成方法创新。
