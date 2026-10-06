# MDMT research workspace

本仓库存档 `/home/chenhc/claude_try_MDMOT` 的 MDMT 研究工作。目标是基于成熟
MIA-Net 基线开发并验证有实质贡献的模块。Git 中保留代码、方案、预注册协议、
审计和紧凑结果；数据、完整推理输出、训练权重与大型缓存仍保留在原工作区。

入口：

* [冻结基线与协议](CURRENT_BASELINE.md)
* [最新阶段状态](continuation_20261006/CONTINUATION_STATUS_20261006.md)
* [Git 阶段归档规则](versioning/WORKFLOW.md)

首次建仓只是保存研究现状，不表示候选方法已经有效。历史诊断的 train/calibration
分数不能作为正式 MOT 成绩；先前审计文件的 PASS 也不能替代独立代码审查。
停止或失效的尝试保留其状态与证据，后续阶段用新提交记录修正。

## 每个大阶段结束

```bash
python3 scripts/stage_snapshot.py --stage STAGE_NAME --message "Concrete stage outcome"
```

该命令检查待提交文件、生成 checkpoint 路径/大小/哈希索引，提交后推送至
`origin`，再核对远端分支与本地 commit。首次远端未配置时可加 `--local-only`
先保存本地提交。它不会删除本地数据、重写历史或强制推送。

训练和外部依赖仍需原有运行环境与数据路径；单独克隆本仓库不能直接复现全部
训练。各阶段的 receipt/manifest 记录外部依赖和权重路径。
