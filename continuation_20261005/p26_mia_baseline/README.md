# 已接入并验证的 MIA-Net 主基线

本目录接入原版 MIA-Net 完整方法，作为后续改进的起点。此前以约 0.8 的内部主机分数直接对照论文约 41.7 的比较混合了初始化、完整推理流程和评估协议，不能据此认定特征差了几十倍。P25 新模块在数据构建和训练前停止。

## 已完成的成绩与可运行性验证

| test14 宏平均，百分数 | MDA | IDF1 | MOTA |
|---|---:|---:|---:|
| 论文参照 MIA-Net 行（STCA 原文表格） | 41.72 | 68.24 | 49.68 |
| 本轮独立重评分的 MIA-Net 封存输出 | **41.730895** | **68.341024** | **49.697168** |

重评分重新读取完整 14 对、28 份预测 JSON 和已存在的 28 份评价 GT-TXT 快照，重新转换 MOT 文件、执行未修改的官方 `calAAS` 和 motmetrics。14 对所有指标与参考 summary 的最大差为 0；独立矢量化 MDA 复核与官方实现差异小于 9e-16。不是把 test8 与 test6 的两个均值直接平均，也不是将 pooled-count 比例冒充 MDA。

这是**接近论文宏平均的成熟基线**。各序列与论文并非逐项完全一致；例如 pair68 与 pair31 的差异部分抵消。详细比较、官方 evaluator 的保留行为、重复预测 ID 及空 GT 末帧检查见 [EVIDENCE_REVIEW.md](EVIDENCE_REVIEW.md) 和 [INDEPENDENT_SCORES.json](INDEPENDENT_SCORES.json)。未改变官方 scorer 来提高或修正成绩。

本轮还执行了新的完整 **fit23 推理**，采用隔离导入的代码、原配置和权重：两视角各 700 帧，分别输出 16,642 和 17,193 行，exit 0，包含准备与导入约 121.462 秒，物理 GPU3 A100 40GB。见 [运行回执](runs/fit23/RECEIPT.json)。这证明本地实际可执行；上表来自已封存 test14 输出的独立重评分，**本轮未重新推理 test14**。

## 接入的完整方法

AutoAssign R50-FPN 官方 epoch60 权重 → ByteTracker → 双向几何对应/ID 更新 → MIA 遮挡补充及反馈 → 官方 JSON 输出。

保留原方案的首帧 XML 框、ID、confirmed 轨迹和 Kalman 状态、初始 H。该基线属于**官方 GT 初始化协议**，不是 GT-free 系统。后续主实验应在相同协议和输入下与它比较；若另做无真值初始化，应单独成组。

本配置为单尺度 1333×800，不进行 TTA、融合检测覆盖或阈值扫描。参数值继承原方法，不作为方法贡献。

## 复现工件

- [source](source/)：从官方兼容快照逐文件复制并验证，共 488 个文件；commit `551f90d998087ea2d02df75700e3c7739c4ecbe1`。原算法本身未改，导入层兼容差异有记录。
- [wrappers/supplement_mia.py](wrappers/supplement_mia.py)：恢复到历史封存脚本的精确 SHA `96c91509b3cf517dcbcbad5a0a73f2d40e46eae57b0950300027c65d55a422d8`，避免使用参考线程后来改动的版本。
- [REPRODUCTION_CONTRACT.md](REPRODUCTION_CONTRACT.md)：权重、配置、解释器、输入目录、GT 初始化、兼容适配及启动命令。
- [SOURCE_IMPORT.json](SOURCE_IMPORT.json)、[WRAPPER_RECONSTRUCTION.json](WRAPPER_RECONSTRUCTION.json)、[INDEPENDENT_SCORE_INPUTS.json](INDEPENDENT_SCORE_INPUTS.json)：逐文件来源及哈希。
- [ENVIRONMENT_IMPORTED.json](ENVIRONMENT_IMPORTED.json) 与 [ENVIRONMENT_FREEZE.txt](ENVIRONMENT_FREEZE.txt)：实测导入位置及版本。NumPy/SciPy/Pandas 来自 Python 3.8 user-site，不能仅依据虚拟环境包目录重建。
- [run_replay.sh](run_replay.sh)：完成过的受控 fit23 重放；默认拒绝覆盖现有结果。使用合同中的原入口命令及新的输出目录可以再次运行。
- [score_verify.py](score_verify.py)：`--rescore` 使用本地固定输入重新评分，无需再读原图或原始 XML。

本阶段结论：**KEEP_MATURE_PAPER_COMPARABLE_MIA_BASELINE**。后续以这套方法为主基线开展结构性修改；不再从相差悬殊的弱主机上继续堆新模块。本阶段没有提出或验证新的跟踪方法。
