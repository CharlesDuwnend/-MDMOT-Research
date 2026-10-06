# P28：成熟 MIA 检测器上的因果 FGFA 真实训练试验

**结论：停止将这个冻结检测器、冻结 RAFT 的 FGFA 移植版本接入完整 MIA。** 三个模块完成同预算真实训练后，FGFA 在五个校准 pair 上均低于原始检测器和单帧适配，恢复量不足以抵消原有检测的损失。保留已验证的 P26 MIA 基线；本结果不否定所有时序表示，也没有产生新的 MDA/IDF1/MOTA。

用户要求从成熟、论文可比的方法出发，继续有技术深度的工作。P26 已采用参考会话可复核的 MIA 全链；本阶段进一步实际移植已发表的 FGFA 算子，训练像素/特征层模块，而不是调整检测阈值。FGFA 的迁移、past-only 窗口、RAFT 替换和 FPN 接口本身不构成新方法创新。对应参考改进的独立审计见 `../reference_improvements_20261005/REPORT.md`；其中 cap300+SUPPL 的容量控制没有混入本阶段。

## 实现及证据

输入为当前图像与同视图过去 1/4/8 帧的原生 AutoAssign R50-FPN 特征。冻结的 RAFT-large 估计 current-to-past 稠密位移，将过去 FPN 反向采样到当前网格；共享 embedding 产生逐位置 cosine-softmax 时间权重，再将聚合结果交给原始 AutoAssign 检测头。训练使用当前帧全部支持类别 XML 框的原始 `loss_pos + loss_neg + loss_center`。检测器与 RAFT 均冻结，只有共享值适配层及 FGFA embedding 更新。

实现保留真实 AutoAssign `prior_generator.offset=0`，分别处理 flow/image/FPN 坐标缩放、实际 ceil FPN 尺寸、边界与 padding 支持，限制参考严格来自过去。16 项 CPU 算子测试及独立复核通过；真实 GPU smoke 验证了初始单帧适配与原生检测结果逐数组完全相同、真实损失有限、值适配与 embedding 梯度非零、参数实际更新以及整个冻结检测器 state hash 不变。它们是实现正确性证据，不是效果证据。

训练前完成 FIT 960 张图像、240 组的 float32 FPN 导出及全量哈希/形状/有限值回读；RAFT 为 240 组、720 次有向估计。`TRAIN_PROTOCOL.json` 在新模块 fit 和 calibration 之前增加均匀融合对照，原始 `CONFIG/GROUPS/FRAMES/PLAN_INPUTS` 四个导出计划不变。三组均使用相同五轮、1,200 步 pair-balanced 完整组 schedule，AdamW lr=1e-4，固定最后一步 checkpoint，没有校准早停或 checkpoint 选择。

| 训练臂 | 活跃参数 | 五轮平均训练损失 |
|---|---:|---|
| single_adapter | 65,792 | .80686 / .79857 / .79304 / .78944 / .78671 |
| uniform_fgfa | 65,792 | .93529 / .91365 / .90099 / .89139 / .88408 |
| causal_fgfa | 787,456 | .89739 / .86936 / .85477 / .84361 / .83614 |

single/uniform 注册但未使用的 embedding 明确冻结，不计入活跃参数，不能声称三臂活跃容量相同。原始 center prior 冻结，因此 `loss_center` 对模块参数是常量；原始 pos/neg 损失仍正常回传。三臂训练总计约 447.95 秒，物理 GPU3 为 A100-SXM4-40GB，UUID `GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a`，train.exit=0。训练阶段峰值 CUDA allocated 约 1.76 GB；这不是完整方法部署内存/FPS。

## 冻结预测后的检测结果

所有固定最后一步 checkpoint 完成后，才导出 calibration 的 160 张图像/40 个当前组并冻结四臂预测。评分器核验组、预测、checkpoint、代码/协议哈希后才解析 calibration XML。全部使用原生 score_thr=.05、NMS=.6、max_per_img=100；原始 score_thr 在 score factors 相乘之前生效，最终分数略低于 .05 的框仍保留，未追加过滤。

| calibration40 / 1,999 GT | pair-macro AP50 % | TP | FP | FN | recall % |
|---|---:|---:|---:|---:|---:|
| native | 91.7365 | 1,773 | 1,384 | 226 | 88.6943 |
| single_adapter | 91.7203 | 1,775 | 1,440 | 224 | 88.7944 |
| uniform_fgfa | 83.6699 | 1,731 | 1,990 | 268 | 86.5933 |
| causal_fgfa | 88.7692 | 1,760 | 1,608 | 239 | 88.0440 |

FGFA 比均匀聚合高 5.0993 个 AP50 百分点，但比 native 低 2.9674、比单帧适配低 2.9511；相对单帧适配是 0/5 pair 胜出，默认输出 recall 也低于 native。预先登记的扩展条件全部未通过，见 `PILOT_DECISION.json`。

FGFA 的 GT 匹配集合中保留 1,738 个 native TP、恢复 22 个 native miss、丢失 35 个 native TP，净少 13 个 TP，同时多 224 个 FP。均匀聚合恢复 25、丢失 67，净少 42。这里的恢复/丢失是逐图逐类最大基数匹配得到的 GT 节点集合差；存在等价匹配时受 tie 影响，不能解释为物理身份或跨帧恢复指标。

小目标（原图 sqrt(area)<16）native 检出 170/263，FGFA 检出 164/263；较大目标相应为 1,603/1,736 与 1,596/1,736。遮挡目标相应为 450/528 与 449/528。损伤并非仅出现在最小尺度；不能由该表直接断言唯一原因是小目标光流错误。

AP50 按 score 降序、逐图逐类一对一贪心 IoU>=.5，再计算 101 点插值；先对每 pair 的有 GT 类别平均，再对五对平均。TP/FP/FN 使用最大匹配基数、再最大总 IoU，是不同于 AP 的明确诊断定义。独立评分器源码复核及 612 个合成二分图契约通过。完整逐类、逐 pair、逐图与匹配明细保留在 `detection_readout/`。

## 实现回查与限制

1. 原生数据到 head 的恒等对照、梯度、冻结参数、时序顺序、坐标测试均通过，没有发现足以推翻当前负结果的实现错误。P27 的历史流严格 parity 失败仍保留，P28 比较使用独立新原生特征/输出。
2. 第一次评分在标注读取前因 Python3.8 主脚本 `__file__` 使用相对路径被严格收据门拒绝。原始 `PREDICTION_RECEIPT.json` 和失败 `score.log` 保留。`canonicalize_prediction_receipt.py` 根据 launcher 的显式工作目录将唯一相对键改为绝对路径，逐字节核验相同代码 SHA，并绑定原收据、launcher 和转换脚本。`PREDICTION_RECEIPT_CANONICAL.json` 没有改变预测、checkpoint、指标或门限。
3. `MECHANISM_LOOKBACK.md` 显示 FGFA 在全部 200 个 group×level 上给 current 更高权重、产生更小特征扰动，但三个有效 past 下有界 cosine-softmax 无法仅选择 current；该结构限制不是失败主因的因果证明。`flow_lookback/REPORT.md` 完成 FIT-only 40,420 个连续同 local-ID/class 比较：lag1/4/8 的 RAFT 中位中心误差 .816/1.595/2.253 px，对照零位移为 2.500/10.012/20.125 px，胜率 78.96/86.31/88.66%。几何信号有效，未见方向/半像素/缩放错误；小目标长间隔尾部误差仍显著，不能因此证明 FPN 表示融合正确。
4. calibration5 只对新模块拟合留出；原始 AutoAssign 检测器已在整个 MDMT train 训练。40 个稀疏当前组不是完整序列评测，也不是全新未见数据的泛化声明。本阶段没有新 official-val/test 推理。
5. 原完整 MIA 的首帧 GT 框/ID/轨迹及几何初始化协议保持显式；检测模块推理不读标签不意味着完整 MIA 是 GT-free。

后续审查转向已发表的可学习特征对齐与当前帧残差分支，明确对照其采样位置、目标函数与表示更新；不通过阈值或 cosine temperature 扫描挽救本阶段。P29 仍需完成算子/源码与当前失败证据对照，再决定具体实现。整个研究目标仍在进行，尚无经过验证的新方法贡献。
