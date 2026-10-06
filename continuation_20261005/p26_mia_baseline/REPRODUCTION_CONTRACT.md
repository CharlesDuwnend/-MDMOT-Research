# MIA-Net 成熟基线复现合同

2026-10-05。此审计只读取代码、安装元数据和图像/XML 的路径信息，没有运行 GPU、读取原始 GT 内容或改动参考工作区。P26 保留官方首帧 GT 框、ID、confirmed 轨迹/卡尔曼状态以及几何初始化协议；它不是 GT-free host。

## 可直接使用的封存 wrapper

[wrappers/supplement_mia.py](wrappers/supplement_mia.py) 的 SHA-256 已精确恢复到参考 `SEAL_mia_net_full_20261005.json`：

```text
96c91509b3cf517dcbcbad5a0a73f2d40e46eae57b0950300027c65d55a422d8
```

当前参考 `suppl_compat.py` 修改于 05:20:27，晚于 05:16:01 的封存，当前哈希是 `2f1b42eab1f3d6ab26230534c184edebf3e2f8227d1a1e902b9a98b170c0054f`。在内存中移除且仅移除四段共 12 行 `SUPPL_DUMP` 保存代码后，哈希精确等于封存值；已将该版本写入上述 P26 wrapper。证据是 [WRAPPER_RECONSTRUCTION.json](WRAPPER_RECONSTRUCTION.json) 和 [WRAPPER_RECONSTRUCTION.diff](WRAPPER_RECONSTRUCTION.diff)。

相对官方 `demo/supplement_MIA.py`（`e05653fdeb18bf413f3c26be9aa9b253315641faac4c48fb3e055a1d85deea36`），封存 wrapper 仅有兼容初始化导入和五处 `SUPPL_NO_VIS` 可视化/视频 guard。几何更新、ID 分配、遮挡补充及结果输出逻辑保留。

## 最小运行依赖与导入边界

| 用途 | P26 路径或外部依赖 |
|---|---|
| 运行入口 | `wrappers/supplement_mia.py` |
| 官方算法快照 | `source/`，来源 `/home/chenhc/mdmot_strong_host_stage1_20260905/source_compat`，commit `551f90d998087ea2d02df75700e3c7739c4ecbe1`；逐文件记录见 `SOURCE_IMPORT.json` |
| 兼容初始化 | `adapters/mdmt_compat_runtime.py`；SHA `8a62f13c2102b04dbb75a6b902775c02fdd4859d17ec907da497e29e2aa9ef1b` |
| 正式配置 | `source/configs/mot/bytetrack/bytetrack_autoassign_full_mdmt-private-half.py`；SHA `97a718d988553bd073f98d3b1f8f888c87c457d05e1e023107afd63f39b748b1` |
| 配置继承 | 同一 `source/configs/_base_/models/autoassign_r50_fpn_8x2_1x.py`、`datasets/full_mdmt_challenge.py`、`default_runtime.py`；保留相对目录结构 |
| 检测器权重 | `/raid/datasets/chc_data/MDMT/checkpoints/autoassign_r50_fpn_8x2_1x_full_mdmt/epoch_60.pth`；SHA `8894ea5ffe8309017d78e2dac359d405b38aa66725e1b465a8dce30beb12c901` |
| 已有解释器 | `/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python` |
| 必需搜索路径 | P26 的 `adapters:source:source/demo:source/demo/utils`，按此顺序放入 `PYTHONPATH` |

`source/demo` 让 `utils.*` 可导入；`source/demo/utils` 也必须保留，因为 `utils/common.py` 使用了裸导入 `from matching_pure import ...`。`source` 必须优先于环境中已安装的另一份 `mmtrack`。推理至少需 `mmcv`、`mmdet`、`torch`、`numpy`、`cv2`、`matplotlib` 及跟踪/注册模块的依赖。

适配器移除 detector/backbone/neck/head 的旧 `init_cfg` 构造字段，使用 `build_model(config.model)`，并把 checkpoint 加载到 **`model.detector`**。权重是 detector-only，不能将其不加适配地加载到整个 ByteTrack wrapper。当前路线不需要额外 ReID 模型权重。

`source_compat` 对受版本控制的 Python 文件只有三处导入层改动：`mmtrack/apis/__init__.py` 不再导入缺失 SOT 依赖牵涉的通用 train/test API；`mmtrack/models/__init__.py` 不再导入 SOT/VID/VIS；`trackers/__init__.py` 额外注册 `ByteTrackerGTF`。本配置选择的是原有 `ByteTracker`，其 `byte_tracker.py`、`byte_track.py` 和 MIA 主脚本未出现相对该 commit 的算法修改。保留当前兼容快照可以避免重引入缺失包，且不启用 GTF 路线。其余大量 git 差异为历史 `__pycache__`；P26 用 `PYTHONDONTWRITEBYTECODE=1`。

配置应显式传入，不能依赖脚本的默认 CARAFE 配置。冻结运行设置为 AutoAssign R50-FPN、三类检测后合并处理，输入 `(1333,800)`、`keep_ratio=True`，`flip=False`，检测 `score_thr=0.05`、NMS IoU `0.6`、`max_per_img=100`；ByteTracker high/low `0.6/0.1`、init `0.7`、保留 30 帧。`MultiScaleFlipAug` 的名称不代表本配置启用了多尺度/TTA：这里只配置单尺度且不翻转。

实际运行版本与包元数据合并记录为 Python 3.8.20、torch 1.10.1+cu113、torchvision 0.11.2+cu113、mmcv-full 1.4.8、mmdet 2.25.1、mmcls 0.25.0、numpy 1.24.4（实际从 Python 3.8 user-site 导入）、scipy 1.10.1、pandas 2.0.3、lap 0.5.13、motmetrics 1.4.0、opencv-python 5.0.0.93。旧 requirements 的部分版本上界与当前已运行环境不同；这次复现复用现有环境并记录实际 import 路径/版本，不重新解析 requirements 安装升级。

## DUMP 与 TTA/覆盖路线不属于默认复现

- 封存后的四段 `SUPPL_DUMP` 代码只保存局部 detector/track 输出以及 H；默认未设置时不执行，也不改变对应数组。它增加 I/O 和耗时，不能把其计时当作原封存速度。P26 已恢复不含这些代码的精确封存 wrapper。
- `run_variant.sh`、`run_hires.sh` 显式替换配置/可能替换权重；这些属于另一组实验，不能静默进入基线。
- `alloc_fuse_compat.py` 改为导入 `mdmt_fuse_runtime.py`；设置 `DET_OVERRIDE_DIR` 后，后者 monkey-patch `ByteTrack.simple_test`，有缓存时跳过 detector，读取外部融合检测。它还在缺少缓存文件时回退正常 detector。因此它既是单独入口，也可能形成混合检测来源，不能充当默认 MIA 复现。
- P26 使用普通 `mdmt_compat_runtime`，不加载融合 runtime；启动环境移除 `SUPPL_DUMP`、`DET_OVERRIDE_DIR`、`CKPT_OVERRIDE` 等扩展变量。固定原配置，无高分辨率、外部检测覆盖、额外候选或阈值调整。

## fit23 全序列输入与命令

已核对两视角各有 700 个图像文件，名称从 `00000001.jpg` 到 `00000700.jpg`。P26 的局部链接如下；本审计仅检查了链接和文件名：

```text
inputs/fit23/1/23-1 -> /raid/datasets/chc_data/MDMT/train/1/23-1
inputs/fit23/2/23-2 -> /raid/datasets/chc_data/MDMT/train/2/23-2
xml/23-1.xml -> /raid/datasets/chc_data/MDMT/new_xml/1/23-1.xml
xml/23-2.xml -> /raid/datasets/chc_data/MDMT/new_xml/2/23-2.xml
```

输入父目录中只放 pair23。`--input` 和 `--xml_dir` 均须带尾随 `/`：官方脚本用字符串拼接路径，并通过把 `.../1/.../23-1/...` 替换为 `.../2/.../23-2/...` 构造第二视图路径。推理 frame index 为 `0..699`，不等于图像文件名中的 `1..700`。

已提供的受控启动命令是 `bash run_replay.sh`，它调用 `replay_fit23.py`，校验导入源码、封存 wrapper、权重和输入，绑定已识别的 GPU3 UUID，然后执行以下等价的核心命令；不要在已存在的 `runs/fit23` 上重复启动：

```bash
P26=/home/chenhc/claude_try_MDMOT/continuation_20261005/p26_mia_baseline
MDMT_PY=/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python
env -u SUPPL_DUMP -u DET_OVERRIDE_DIR -u CKPT_OVERRIDE -u CFG_OVERRIDE \
  PYTHONDONTWRITEBYTECODE=1 SUPPL_NO_VIS=1 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a \
  PYTHONPATH="$P26/adapters:$P26/source:$P26/source/demo:$P26/source/demo/utils" \
  "$MDMT_PY" "$P26/wrappers/supplement_mia.py" \
  --config "$P26/source/configs/mot/bytetrack/bytetrack_autoassign_full_mdmt-private-half.py" \
  --checkpoint /raid/datasets/chc_data/MDMT/checkpoints/autoassign_r50_fpn_8x2_1x_full_mdmt/epoch_60.pth \
  --input "$P26/inputs/fit23/1/" --xml_dir "$P26/xml/" \
  --result_dir "$P26/runs/fit23/json" --method mia_baseline \
  --output "$P26/runs/fit23/vis1" --output2 "$P26/runs/fit23/vis2" --device cuda:0
```

预期正式结果是 `runs/fit23/json/mia_baseline/23-1.json`、`23-2.json` 及 `time.txt`。两个 JSON 均应包含有序 `frame=0` 到 `frame=699`，每帧记录 `[id,x1,y1,x2,y2]`；补充目标合并在最终输出中，独立 supplement JSON 在官方代码中是注释掉的。受控 replay 另产出 `PRE_RUN.json`、`ENVIRONMENT.txt`、`inference.log`、`launcher.exit` 和 `RECEIPT.json`。`SUPPL_NO_VIS=1` 时无需期待可视化图像或视频。

全 fit23 重放只建立这套成熟算法的本地可执行性与流程一致性，不生成新的 test14 成绩。历史封存记录的 MDA 41.73% 对论文 41.72% 属于另一条已有 test14 证据链；本审计没有重算该分数。论文分数核验应使用同一封存输出、相同官方 evaluator/GT 协议，由主流程另行确认。
