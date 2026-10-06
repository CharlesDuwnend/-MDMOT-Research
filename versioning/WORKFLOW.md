# 阶段结束提交与推送

用户于 2026-10-06 授权为此研究工作区创建远端 Git 仓库，并在每个大阶段结束
push。该授权持续适用，不需要为常规 commit/push 重复确认。

采用独立私有仓库、main 分支。用户提供远端地址或完成建仓 API 登录后，直接
创建/配置并 push；SSH 的推送认证与 GitHub API 的建仓认证是两种能力。

阶段提交包含：源代码、方案、预注册、实现修复、失败/结果审计、摘要和权重索引。
数据集、raw prediction、checkpoint、大缓存和鉴权材料不加入 Git。原本的
本地文件不会因 Git 归档而被删除。超过 2 MiB 的文本也只在索引中列出。

归档命令：

```bash
python3 scripts/stage_snapshot.py --stage PHASE --message "Describe the verified outcome"
```

该脚本只暂存经过 .gitignore 和大小/凭据检查的文件；若已有其他暂存文件则退出，
避免混入未知修改。它保存本地路径与 checkpoint SHA-256，commit 后普通 push，
最后以 `git ls-remote` 检查远端 main 与本地 HEAD 相等。缺少远端时会明确失败，
不会把本地提交说成远端备份。

不采用自动训练末尾 hook：失败训练也会产生文件；应先确认该阶段的真实状态、
失效结果和待修复问题，再调用阶段归档命令。论文结论必须对应可复查的 commit。
