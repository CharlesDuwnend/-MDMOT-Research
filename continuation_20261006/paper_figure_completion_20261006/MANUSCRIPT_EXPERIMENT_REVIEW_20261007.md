# 论文实验审查汇总（供是否加入正文决定）

日期：2026-10-07  
稿件：`/home/chenhc/src_egia_icme_paper`  
当前稿件构建：9 页，5 幅图，4 张表；`main.pdf` SHA-256 为 `3e4d7f012f82d71c2592bb355400265203b191cb21c29db7838dad703ef27cd4`。

## 当前已经可以支撑的结果

现有 VisDrone2019 test-dev（17 sequences，6,635 frames）配置结果来自
`research/module_ablation_verified.json`，可以作为部署配置比较：

| 配置 | MOTA | IDF1 | FP | FN | IDs |
|---|---:|---:|---:|---:|---:|
| A00 Native | 55.93 | 69.94 | 33,216 | 66,645 | 1,288 |
| A02 SRC | 57.56 | 70.87 | 28,701 | 67,722 | 987 |
| EGIA-only（native-branch refit） | 57.53 | 71.07 | 25,849 | 70,865 | 765 |
| A06 full | 58.03 | 71.37 | 26,047 | 69,497 | 791 |

这四行不是严格的同一 frozen head 下的独立主效应或交互作用消融：EGIA-only
使用 native residual refit，full 行使用 frozen summary fusion；gate、拟合分支和
fusion 设置也不同。因而正文目前把它们称为 deployed operating points，这是合适的。
`module_ablation_source_audit.json` 还保留 runner manifest hash drift，不能把这四行
扩写成完整 A00--A06 factorial 结论。

另外已有一组独立完成、通过三轮检查的五臂 frozen audit（外部报告：
`/home/chenhc/leaf_v5_evidence_completion_20261006_v1/reports/FIVE_ARM_FINAL_AUDIT.md`）；
它比当前正文四行更适合回答“内部消融”问题，但尚未写入论文：

| arm | MOTA | IDF1 | FP | FN | IDs |
|---|---:|---:|---:|---:|---:|
| F00 full | 58.0251 | 71.3737 | 26,047 | 69,497 | 791 |
| F01 no-SOE | 58.0246 | 71.2955 | 27,210 | 68,266 | 860 |
| F02 no-SRC | 57.8177 | 71.1857 | 26,252 | 69,759 | 800 |
| F03 neither | 57.8516 | 71.1167 | 27,360 | 68,498 | 875 |
| F04 current-source-pool diagnostic | 58.0116 | 71.3707 | 26,072 | 69,499 | 795 |

对应 conditional effects 为 SRC-at-SOE-on/off = +0.2074/+0.1730 MOTA pp，
SOE-at-SRC-on/off = +0.0004/-0.0340 MOTA pp；factorial interaction 为
+0.0344 MOTA pp、+0.0091 IDF1 pp。F04 是 source-pool contract diagnostic，
不是新的模块主效应。该 audit 的 manifest SHA-256 为
`dfb8efb3458bc9d31daa063551c0864db23f2222f4464ffa6b7afd8772194cba`。

已经有的机制证据：

- source-pairing 固定开发事件：8,411 个事件，weighted NLL 从 0.680（保留来源配对）变为 1.062（打乱 appearance 来源），见 `research/source_pairing_verified.json`。这是 scorer 诊断，不是闭环 MOT 消融。
- policy 表的三种 admission/fusion operating points 已有完整 test-dev 评估，可说明 selective admission 和 summary fusion 的部署效果，但它们共享 semantic/source 机制，不能单独归因到某一个内部模块。
- Table II 的 ByteTrack/BoT-SORT 是完整 adapted configurations。UAVDT 使用 vehicle-only semantic contract，shared U2MOT detector cache；它们适合说明接口可适配，不是 held-out detector generalization。

## 审稿人最可能追问的内部实验

1. **SRC reliability update 是否真的必要？**  
   当前 SRC 行同时改变了 semantic memory/update 与 cost modulation。最有价值的闭环对照是固定同一 SRC 模型和 detector，比较 learned reliability update 与 hard update（(r=1)），并报告 MOTA/IDF1、FP/FN、IDs，以及按 association-stage 的错误变化。

2. **source pairing 是否只改善校准 NLL，还是改变 tracking 输出？**  
   当前 8,411-event permutation 不能回答后半句。应在同一 frozen EGIA scorer 和同一检测流上做闭环 `pair_shuffle`，保持 cue 数值不变、只打乱 cue ownership；如果支持，再做 paired-sequence evaluator。这个结果最能补足 mechanism-to-output 链条。

3. **四行配置是否受 gate/refit 选择影响？**  
   gate=0.70 的选择包含 test-dev 比较，EGIA-only 还使用 native-branch refit。若需要更强因果表述，应在 fit/calibration split 上预先固定 gate，并做 branch-matched frozen-head 对照。否则保留当前“configuration comparison”措辞。

4. **跨宿主是否是泛化？**  
   当前结果是完整 adapted ByteTrack/BoT-SORT operating points，且 UAVDT 共用 U2MOT cache。可以保留适配性结论，不能称为 zero-shot 或 held-out generalization。

5. **覆盖率代价是否稳定？**  
   VisDrone 的 FN 增加而 FP/IDs 下降。若篇幅允许，可放每序列 MOTA/IDF1 变化或 FN-stratified analysis；它用于说明 trade-off，不应只挑正向序列。

## 建议的补跑顺序（暂不写入论文）

- **优先级 A：** frozen-model `pair_shuffle` 闭环对照；通过条件是实现检查、source ownership 保持可追踪、结果文件和 evaluator receipt 完整。
- **优先级 A：** frozen-model `hard_update`（(r=1)）闭环对照；通过条件同上，并确认只改变 update operator。
- **优先级 B：** branch-matched gate control（固定 gate、同一训练/校准协议），只在 A 结果支持后进行。
- **优先级 C：** per-sequence/FN-stratified 汇总，作为 coverage trade-off 的补充材料候选。

本轮没有启动新的训练或推理，也没有把候选实验当成论文结果。若你决定补强内部消融，
优先把上述 F00--F03 五臂 audit 结果整理成新的配置表，再决定是否需要补跑
`pair_shuffle` 或 `hard_update`；它们回答的是 source ownership 和 reliability
operator，属于另一层机制问题。是否放入正文，等你根据上面的优先级决定。

## 论文当前检查结果

- `python3 scripts/build_tables.py --check` 通过；表格证据哈希和 MOTA identity 检查通过。
- `pdflatex`、BibTeX、再两次 `pdflatex` 通过，输出 9 页。
- 无 undefined/multiply-defined/citation/overfull 错误；仅有页 3、页 7 的 `Underfull \\vbox` 分页提示。
- Table II 数据已取消加粗；Table IV 宽度为 `0.70\\columnwidth`，表下注释间距为 8pt，FPS 为 27.4/25.2/23.8。
- 主文已无 Fig. 6 引用；README 和 evidence ledger 已同步说明 Fig. 6 已从正文移除。
- U2MOT setup 已压缩为数据协议、共享 detector 和 vehicle-only lifecycle 设定；具体 fit/calibration 流程仍由证据清单保留。
