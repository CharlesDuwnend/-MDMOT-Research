# P31 阶段报告

## 结果

`causal_evidence.py` 在固定 mdmt_env（PyTorch 1.10.1+cu113）下通过 CPU mechanism gate，参数量 8,873。测试覆盖有限值、集合置换不变、mask 外垃圾不变、空支持、禁止未来 age、有限反向传播和 dustbin assignment；receipt 为 `P31_CPU_RECEIPT.json`。这只证明代码合约，不证明研究价值。

`audit_ids.py` 对 `/raid/datasets/chc_data/MDMT/new_xml` 的 19 个 train-only 序列对生成 `ID_AUDIT.json`。它只说明 XML 原始 ID 的重叠/共同声明帧，明确不把该重叠当作物理跨视图身份、标定或同步证据。

## 技术判断

P30 的 detector/FPN residual 路线已经停止。P31 原本把对象级支持集合的干预表示 `z(S)-z(S\\i)`、quality 和 null/dustbin 作为候选差异；但本地 P3 MDMT 实验已经使用因果历史、support/conflict 分支、mixed-bag 反例监督和接入一致性输出。它是同任务直接碰撞，所以 P31 在真实特征 port 前停止，不能进入论文候选池。

## 推进门

下一步必须换到不同的 observable 或 learning target；先完成同任务 source/code audit，再写新的 CPU 合约。没有 official val/test，不能把任何 FIT/calibration 诊断写成正式 MDA/IDF1/MOTA。
