# P31：对象级因果证据干预（CEIN，已停止）

P30 已证明把 learned residual 直接注入 detector/FPN 会改变候选数量和边界，令成熟 MIA host 的校准结果下降。因此 P31 把时序信息移到 detector 之后的对象级支持集合，保留 P26 detector、几何候选和 host 不变。

本目录目前只有机制原型和审计：`causal_evidence.py`、`test_causal_evidence.py`、`audit_ids.py`。CPU 合约虽然通过，但本地 P3 已经覆盖同一 support/conflict 因果证据链，因此 CEIN 在真实特征 port 前停止。不要将它们当作正式跟踪器或论文结果。

执行顺序：

1. CPU 合约（有限值、因果 age、集合置换不变、mask 不变、反向传播、dustbin）通过；
2. `PRIOR_ART_AUDIT.md` 补入本地 P3 后确认 Tier 1 直接碰撞；
3. 停止真实 ROI/运动/质量 token port，不启动训练；
4. 下一候选必须更换 observable 或 learning target，并重新完成 source/code 审计。无 official-val/test 访问时不产生正式指标。
