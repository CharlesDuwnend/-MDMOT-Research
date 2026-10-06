# MIA-Net 成熟基线：独立重评分核验

2026-10-05。**已独立重算并确认参考工件的完整 test14 宏平均成绩：MDA 41.730895%、IDF1 68.341024%、MOTA 49.697168%。** 四舍五入后即 41.73 / 68.34 / 49.70。本轮重评分从封存的预测 JSON 重新执行官方 `mango_eval.calAAS` 与 MOTMetrics，没有复用旧分数作为计算输入，也没有运行新 test 推理、训练、GPU 或参数选择。

判断：`PASS_MIA_REFERENCE_MACRO_REPRODUCTION`，适用 **MIA 官方 GT 首帧初始化协议**。它是一条可与已发表 MIA 行对照的成熟方法基线，不能称作 GT-free 初始化基线、本文新方法，或与论文逐序列完全一致的复现。

## 1. 成绩和真正的原始来源

| 指标，百分数 | 本轮独立重算 | STCA 原文 Table 1 的 MIA-Net 行 | 差值，百分点 |
|---|---:|---:|---:|
| MDA | 41.7308952911 | 41.72 | +0.0108952911 |
| IDF1 | 68.3410237378 | 68.24 | +0.1010237378 |
| MOTA | 49.6971680799 | 49.68 | +0.0171680799 |

数值基准的可核全文来源是 **STCA，Remote Sensing 2024，16(20):3861，DOI [10.3390/rs16203861](https://doi.org/10.3390/rs16203861)，Table 1 / Table 2**。其中 MDA 原单位是 0.4172，本报告乘 100 写作 41.72%。论文 §5.1 明确 test 14 sequences；Table 2 枚举下列同一批 14 对场景。

`source_compat/README.md` 可核实 MDMT、MIA-Net 名称、数据规模和作者代码链接，**但该 README 没有 41.72 / 68.24 成绩表**。因此不能把这些数值引到 README，也不能在未读取 MIA 原论文表格的情况下声称本轮核验了其 2023 原表。用于此比较的 STCA 全文副本已独立封存为 `reference_outputs/metadata/STCA_primary_text.txt`（原文件约 530–695 行覆盖实验划分、公式和表格）。

## 2. 完整工件和独立执行证据

两次历史运行组成同一 test14，不是两个可等权平均的试验：

- `official_suppl_test`：56、57、59、61、62、68、71、73，共 8 对。
- `official_suppl_test6`：26、31、34、48、52、55，共 6 对。

本轮检查两个目录精确含预期的 16 + 12 个 JSON，无重复 pair、无额外 JSON、无缺少视图。复制 **28 个 sealed 预测 JSON、28 个既有 GT-TXT 评分快照、28 个旧 JSON→MOT 转换文件**及评估器/原汇总/来源元数据到 `reference_outputs`，共 92 个哈希绑定的文件。预测 JSON 全部匹配历史 `SEAL_mia_net_full_20261005.json` 中的 SHA256；GT-TXT 未出现在该历史 seal 中，本轮复制前后逐字节绑定，并没有将新封存说成历史已封存。

结果完整性如下：

| 项目 | 独立检查结果 |
|---|---:|
| 场景对 / 单视图输出文件 | 14 / 28 |
| pair-frames / 单视图 frames | 5,883 / 11,766 |
| 预测框记录 | 474,862 |
| 既有 GT-TXT 框记录 | 601,270 |
| JSON frame keys | 28/28 按插入顺序严格为 frame=0…frame=N−1 |
| 同一 pair 双视图 frame keys | 14/14 完全相同 |
| JSON 重新转 MOT 与旧转换文件 | 28/28 字节完全相同 |
| 预测非有限数 / 非正面积框 | 0 / 0 |
| 同一帧内重复预测 ID 的额外记录 | 19 |
| GT frame+ID 重复记录 | 0 |

唯一无 GT 框记录的帧是 **55-1 的最后一帧**：预测与另一视图均覆盖 151 帧，而该 GT-TXT 的最大 frame 为 150。它不构成输出缺帧；本轮保留第 151 帧参与原协议评分，没有截短 JSON 或删掉该帧。不能仅凭无 annotation 推断实际图像必然没有目标。

执行命令：

```bash
/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python -B score_verify.py --rescore
```

本轮通过 tmux `p26_independent_score` 运行，现已结束，`independent_score.exit` 为 **0**。4 个 CPU 进程共约 **14.97 秒**；Python 3.8.20、NumPy 1.24.4、MOTMetrics 1.4.0。没有导入模型进行推理，没有打开原始 XML/图像，没有触碰参考运行任务。

第一版检查曾错误要求 GT 最大框帧等于预测帧数，因此在 55-1 停止。该版本、日志、exit=1 已保存在 `attempts/frame_extent_equality_v1`；修正为所有 GT 帧必须落在预测范围内，并显式统计无标注末帧。它是验证脚本的实现纠错，不是算法或阈值变化。

## 3. 评估定义：不能把不同平均或单位混在一起

**MDA 原样执行官方代码。** 每帧先算

`TA / (GA + FA + MA)`，其中 `FA = RA − TA`、`MA = GA − TA`；先在该 pair 的全部帧上取平均，再对 14 个 pair 等权平均。它不是将全部场景的 TA/GA/FA/MA 汇总后求比，也不是 `mean(test8_macro, test6_macro)`。等价的组汇总必须按 **8/14 与 6/14** 加权。

**MOT 指标按旧评分 wrapper 精确复算。** 从每个 JSON 重新写出 frame+1、ID、xywh 的 MOT 文本，用 `motmetrics.compare_to_groundtruth(...,'iou',distth=0.5)` 计算每视图指标；每个 pair 先均值两视图，再均值 14 pairs。IDF1 是 **28 个单视图 IDF1 的宏平均**，不是跨相机全局 IDF1，也不是 MOTMetrics `OVERALL` 汇总全部计数后计算的 IDF1。

与两个原 `summary.json` 的逐 pair MDA/IDF1/IDP/IDR/MOTA/MOTP/IDSW 对比，最大绝对差为 **0.0**。这证明参考汇总能从其实际结果文件重复得到，不只是引用一段日志。

另一个独立的 NumPy 矢量化实现，在每个 pair 上重新计 GA/RA/TA/FA/MA，保留官方的 ID 选择语义，再重算 MDA；它与未改动 `calAAS` 的最大绝对差为 **8.881784197001252e−16**。汇总计数仅作诊断：GA=188,500，RA=127,283，TA=105,748，FA=21,535，MA=82,752；本报告的主 MDA 没有用这些 pooled counts 代替逐帧平均。

## 4. GT 初始化和官方实现的具体边界

当前阶段使用的是**官方首帧 GT 初始化 + 每帧几何 ID 更新 + 遮挡补全**链。精确重建的 `wrappers/supplement_mia.py` 在 `i==0` 时调用 `read_xml_r` 得到两视图的 bbox / IDs / labels，再将其传入 `inference_mot`。这比“只给一个外部 H”更强，不能在对照表中标成 fully GT-free。

既有评分 GT-TXT 来自参考 `stage15_global_host.py::xml_to_mot`：原 XML frame 加 1；排除 `outside!=0` 的框，其余类别进入输出；raw ID 保持跨视图相等约定；写为 confidence/class/visibility=`1,1,1`。官方 MDA 评分时忽略类别，仅按 ID 相等及双视图 IoU 判断，MOTMetrics 也按该 class-agnostic 文本评分。它与本工作前期 fit/cal 的“同类同号且 conflict mask”检索合同不同。本轮读取了既有 TXT 和转换源码，**未重新生成 XML→TXT**，不将它描述成新的原始标注审计。

官方评估器本身存在需保留的实现细节：

- `read_result` 使用 JSON 的插入序号和双视图 zip 计帧，不解析 frame key 的数值；本轮逐文件检查连续、有序且两视图相等，因此本批结果不受错位影响。
- 同帧重复预测 ID 时，官方 `interIndex` 保留最后一个匹配对象；本批有 19 个额外重复 ID 记录。本轮未去重再评分。
- 源侧 GT 在另一视图没有同 ID 时，`gtA.interIndex` 为 −1，官方仍可能访问 `gtB[-1]`。本批有 1,825 个满足源侧 IoU 的此类候选，其中 **44 次被该路径计为 TA**（pair26=15、pair59=27、pair73=2）。没有负 FA 或负 MA 帧。本轮忠实复现该行为，没有私自修评估器后仍称原官方分数。

因此这条分数可用于**固定官方实现下的成熟基线复现比较**；不能将其解释为已通过严格物理身份标注审计、去重修正或全局匹配修正后的另一种指标。

## 5. 宏平均接近，不等于各序列与论文逐项相同

| pair | 本轮 MDA，% | STCA Table 2 MIA，% | 差值，百分点 |
|---|---:|---:|---:|
| 26 | 43.79018 | 43.55200 | +0.23818 |
| 31 | 32.97761 | 33.89600 | −0.91839 |
| 34 | 38.73012 | 38.70300 | +0.02712 |
| 48 | 58.71687 | 58.82200 | −0.10513 |
| 52 | 61.25809 | 61.23900 | +0.01909 |
| 55 | 34.66482 | 34.63800 | +0.02682 |
| 56 | 49.60393 | 49.61300 | −0.00907 |
| 57 | 83.07398 | 83.10700 | −0.03302 |
| 59 | 4.39890 | 4.49000 | −0.09110 |
| 61 | 34.10011 | 34.11800 | −0.01789 |
| 62 | 57.15906 | 57.18400 | −0.02494 |
| 68 | 20.99646 | 19.56200 | +1.43446 |
| 71 | 35.15706 | 35.23500 | −0.07794 |
| 73 | 29.60534 | 29.88900 | −0.28366 |

最大的正负差出现在 68 与 31；宏平均中存在抵消。正确表述是 **“复现达到已发表 MIA 基准行的宏平均水平”**，不应写成“与论文所有序列完全一致”或据 +0.0109pp 声称优于 MIA。

## 6. 哈希与移交

历史 seal 的 82 项中，81 项当前仍匹配；唯一漂移是原目录 `suppl_compat.py`，后续加入了 4 个 `SUPPL_DUMP` 序列化块。另一个实现审查已从当前文件移除这 12 行，生成 `wrappers/supplement_mia.py`，其 SHA256 **精确恢复历史 sealed runner**：

`96c91509b3cf517dcbcbad5a0a73f2d40e46eae57b0950300027c65d55a422d8`

本审查实际读取 `WRAPPER_RECONSTRUCTION.json` 并核对重建文件哈希；原参考目录未修改。预测 JSON 的历史 seal 一直匹配，评分来源没有跟随后续 wrapper 漂移。

主要工件：

- `INDEPENDENT_SCORES.json`：全部逐 pair / 逐 view 实际指标、结构计数、与旧分数的差及官方实现诊断。SHA256 `bd90eef527771738c7df54b6a534b6313f256118337b1cf938bbe1287fa5d093`。
- `INDEPENDENT_SCORE_INPUTS.json`：92 个评分输入副本的来源和哈希。SHA256 `ca31e2a90b3b8588a387af9dc900ab24436611de12a7affd42df45c911bfa1ba`。
- `score_verify.py`：实际运行的独立重评分脚本。SHA256 `4f57a90ff6ffd445f96202da22a3e8632ac8e20d74a9d4dbe2afc76e55e08212`。
- `REFERENCE_SEAL_CHECK.json`：历史 82 项当前哈希核对；`WRAPPER_RECONSTRUCTION.json` 给出唯一漂移项的精确重建证据。
- `independent_score/`：28 个重新生成的 MOT 文件与 14 个逐 pair 重评分 JSON。
- `independent_score.log` / `independent_score.exit`：本次真实评分过程与 exit=0。

本子任务已经完成 **旧工件的独立分数核验**。根任务正在用隔离重建代码做新的 fit 序列运行；该运行的完成状态与当前 test14 档案重评分是不同证据，本文件不将进行中的新运行说成已经复现完成。
