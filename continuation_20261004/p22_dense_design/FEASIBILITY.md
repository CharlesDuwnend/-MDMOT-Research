# P22 dense motion 可实施性审计与最短输入门

日期：2026-10-04。状态：**真实 dense-field 输入工程可行；表示学习、跨视图可迁移性及新颖性仍 HOLD。**

本次仅检查本地代码/包/已有报告及官方远程源码。没有读取新的 MDMT 图像或标签，没有运行光流、模型前向、训练、GPU、calibration/dev/official-val/test，也没有下载权重、安装包或修改旧文件。本文的路径分为“已存在”与“建议新建”，后者不能视为实现完成。

## 1. 这次要补的真实缺口

P21 已完成，不能再以更换距离或阈值重复它。`p21_residual_motion/probe.py` 实际使用同一 local track 的两个 bbox 底点估计位移；SIFT 背景单应来自图像，但目标运动不是像素流场。它只支持以下结论：

- 5 fit pairs 23/25/29/69/78、每 pair 8 个 t/t-8 端点、160 张图；时间单应 80/80 成功。
- 跨视图 current/history 单应分别仅 12/40、11/40 成功，最终 9 个时刻、281 个支持目标、165 个有已知正例的诊断 query。pairs25/29 没有跨视图几何支持。
- 同支持 query：静态几何 161/165，稀疏 exact motion 121/165；motion 救回 4 个、损坏 44 个。该结果不能否定所有 dense motion 或条件互补。
- static pair-macro=.98229 时，额外 standalone +.02 门在数学上不可达到。这一门不能作为方向否定证据，也不能观察标签后修改成一个更容易通过的“预声明”门。

因此 P22 的第一对象是**局部真实位移是否有可靠的空间信息、能否从背景变形中被观测性地分离**。它不是再训练一个接收几何/速度标量的 ranker，也不是新建 owner/NEW decoder、support/conflict readout 或 P10 patch 两路径分数。

## 2. 已存在的本地资产与缺口

本地解释器：`/home/chenhc/.conda/envs/remdet_paper/bin/python`。实测可导入 torch `2.2.0+cu121`、torchvision `0.17.0+cu121`、cv2 `4.9.0`；未触发 CUDA 计算。

| 组件 | 已验证本地路径/状态 | 直接使用条件 |
|---|---|---|
| torchvision 官方 RAFT | `/home/chenhc/.conda/envs/remdet_paper/lib/python3.11/site-packages/torchvision/models/optical_flow/raft.py`；源码和 weight enums 均可导入 | **最短推荐入口**。没有确认可加载的本地权重；仍需独立权重收据和一次有界 smoke |
| OpenCV DIS / Farneback | `/home/chenhc/.conda/envs/remdet_paper/lib/python3.11/site-packages/cv2/__init__.py`；`DISOpticalFlow_create`、`calcOpticalFlowFarneback` 都存在 | 无权重，可作为 CPU dense-field 工程对照；它失败不能单独否定真实运动输入 |
| 原作者 RAFT checkout | 在下述限定搜索范围没有发现 | 已有 torchvision 版本无需为了首门再迁移旧 PyTorch 环境 |
| GMFlow / SEA-RAFT | 没发现本地 checkout 或 checkpoint；`gmflow`/`raft` Python package 未找到 | 需要独立源码、权重及 import audit；不能称“本地已经可跑” |
| CoTracker | `cotracker` package 未找到；限定范围未发现 checkout 或 checkpoint | 官方主要输出 sparse/quasi-dense 点轨迹，不能把少量点插值成图便称真实每像素稠密观测 |

检查过的权重缓存：`/home/chenhc/.cache/torch/hub` 当前不存在；`torch.hub.get_dir()` 正好指向该路径。`/home/chenhc/.cache/huggingface/hub` 只有 `version.txt`。TORCH_HOME、XDG_CACHE_HOME、HF_HOME、HUGGINGFACE_HUB_CACHE 均未设置。

限定源码/文件名搜索覆盖 `/home/chenhc/cache`、`/home/chenhc/claude_try_MDMOT`、`/home/chenhc/mdmot_research_20261002`、`/home/chenhc/20260824_h2_flow_assets`、`/home/chenhc/mdmt_multi_uav_route_gate_20260909_continuation`；另检查了上述缓存和旧 `p4_diagnosis/assets`。该 assets 唯一发现的相关视觉权重是已知 DINOv2 `dinov2_vits14_pretrain.pth`，不是光流权重。**这是限定范围内未找到，不是宣称整台机器不存在。** 没有进行全盘扫描。

当前可定位依赖：numpy `1.26.4`、scipy `1.17.1`、timm `1.0.29`、huggingface-hub `1.30.0`、imageio `2.37.4`、scikit-image `0.26.0`。`einops`、`yacs`、`omegaconf`、`hydra` 缺失；这些包不是使用 torchvision RAFT 的前提，不能为了审计批量安装。

### 首选 RAFT 的精确接口

```python
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
# 注意实际 enum 是 Raft_Large_Weights，不是 RAFT_Large_Weights。
# 后续显式加载封存 checkpoint；不要隐式联网下载 DEFAULT。
```

一个可预先锁定的强提取器是 `Raft_Large_Weights.C_T_SKHT_V2`，官方 enum 给出：

`https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth`

它包含 Chairs/Things/Sintel/KITTI/HD1K 外部光流训练，不能标成只用 MDMT 或自监督。如果要预先锁定纯合成训练来源，可使用 `C_T_V2`：

`https://download.pytorch.org/models/raft_large_C_T_V2-1bb1363a.pth`

这两个是**协议选择**，不是在 MDMT 标签上择优的两个试验臂。本审计没有选择后下载其中任何一个，也没有完整 SHA-256；URL 文件名的 8 位 hash 不能替代下载后的完整收据。建议先在方案封存时选择一个 extractor，以官方强预训练 `C_T_SKHT_V2` 为首门默认提议。

模型有 5,257,536 参数。输入必须是相同大小 RGB，按官方 transforms 转为 float 并归一化到 [-1,1]，H/W 均 pad 到 8 的倍数且至少 128。wrapper 要自行断言两个维度都满足条件：本地 v0.17 源码的 divisibility 条件写法并不覆盖所有非法尺寸。输出为每轮 `[B,2,H,W]` 像素位移，最终轮用于固定提取；返回值不是置信度或协方差。若 resize，恢复 flow 必须分别乘回 x/y 缩放比例，不能只 resize tensor。

### 官方远程代码核验，尚未本地安装

2026-10-04 读取了以下官方仓库 HEAD 元数据、README 和列出的依赖/核心文件；均按 commit 固定，不是只依据项目名推断。

| 官方源码及 commit | 核验到的依赖/权重合同 | 对本任务的判断 |
|---|---|---|
| [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT/tree/2888e15a51fa41140771d3f498ed8023cff098d1) `2888e15a51fa41140771d3f498ed8023cff098d1` | README 测试环境 torch1.6/CUDA10.1；`download_models.sh` 下载，示例 `models/raft-things.pth`；alternate correlation 扩展可选 | 不是最短本地接口；不要替换当前 torch 环境来复制旧依赖 |
| [haofeixu/gmflow](https://github.com/haofeixu/gmflow/tree/b5123431164d01ec14526a1c3d22218aecb62024) `b5123431164d01ec14526a1c3d22218aecb62024` | README torch1.9/CUDA10.2/python3.8，称高版本应可用；`environment.yml` 包含 einops0.3.2；示例 `pretrained/gmflow_sintel-0c07dcb3.pth` | 可作另一官方 extractor，但兼容性未实测、einops 缺失。首门没有理由并行扩张多模型 |
| [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT/tree/9137517ba24e628442aec097d3afe71d03503b75) `9137517ba24e628442aec097d3afe71d03503b75` | README torch2.2/CUDA12.2/python3.10；requirements 含 einops；`core/raft.py` 使用 `PyTorchModelHubMixin`；官方示例 `custom.py --cfg config/eval/spring-M.json --url MemorySlices/Tartan-C-T-TSKH-spring540x960-M` | 与本机 torch 版本接近，带 uncertainty 输出，值得后续候选；当前无源码/权重/import 验证。其不确定性也不自动在 MDMT 上校准 |
| [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker/tree/82e02e8029753ad4ef13cf06be7f4fc5facdda4d) `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` | README 明确 quasi-dense；`setup.py` version3.0 且 `install_requires=[]`，不能据 pip install 成功判断依赖齐备；官方 `https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth` / `scaled_offline.pth` | 若后续问题是遮挡点轨迹与长期可见性，可考虑。不是本次相邻帧 dense-flow 的最短替代；online/offline API 都须另审因果截止时刻 |

GMFlow 官方 checkpoint 入口：[README 所链 Google Drive](https://drive.google.com/file/d/1d5C5cgHIxWGsFR1vYs5XrQbbUiZl9TX2/view)。SEA-RAFT 本地示例权重名是 `models/Tartan-C-T-TSKH-spring540x960-M.pth`，**该文件此处不存在**。所有上述远程路径是来源信息，不是已准备完成的本地工件。

## 3. 小目标与算力决定实现方式

已读旧报告 `p4_diagnosis/assets/fit_bbox_size_statistics.json`：15 个 fit pairs 的 868,102 个原始预测框中，27.03% 短边 <16 px，63.25% <32 px，宽/高中位数约 29.7/30.1 px。这是全 fit 既有统计，不是 P22 60 张图的实测分布，也不是 GT 尺度。旧 fit23 图像记录为 1920×1080。

RAFT 特征下采样 8 倍。1920 宽图缩到 960 后，一个 16–32 px 的目标仅约 1–2 个特征格；最终上采样成每像素输出不等于获得同等数量的独立目标运动信息。必须保留原始坐标尺寸、有效支持像素、原始特征覆盖格数与空间相关性，不能只报告“32×32 flow patch”。

首门优先在原分辨率固定位置 context tile 上提取局部光流，例如 512×512 输入、固定中心有效区、同一 tile 的两时刻使用相同原始坐标，不按历史 GT/track ID 移动 crop。边缘跨出或运动超出上下文的目标标为无支持，不能用 resize 弥补。背景全局变形可以在低分辨率上估计，再按严格坐标变换投回 native tile。tile 大小和取样位置是首门工程合同，不是动态分辨率方法贡献。

torchvision RAFT all-pairs correlation 的内存下界（float32，batch1，4 层理想尺寸；不含网络激活/输出/框架开销）为：

| 输入 H×W | 初始 correlation | correlation pyramid 约值 |
|---|---:|---:|
| 544×960 | 254.0 MiB | 337.3 MiB |
| 1080×1920 | 4,004.5 MiB | 5,318.5 MiB |
| 512×512 | 64.0 MiB | 85.0 MiB |

这解释了 native tile 的必要性，不能把这些解析估算写成实测峰值或速度。双向 flow 顺序执行；缓存 CPU 原始场和可靠性量，避免同时保存多个全图 correlation。未来只在已授权 GPU0/1/3 中锁定一张并记录 UUID/型号/显存，不能使用 GPU2。本次没有查询或使用 GPU。

## 4. 表示算子在哪些条件下有技术内容

令同视图的实测像素映射为 `Φ_t(p)=p+F_t(p)`，独立背景变形为 `C_t(p)`。对象残差定义为：

`R_t(p) = Φ_t(p) - C_t(p)`。

这两个端点都在目标帧坐标系中。相机与对象的物理 3D 运动并不能从这个减法唯一恢复：深度视差、目标高度、背景错配与漏检都会进入 R。此处只能先称“背景变形条件下的对象区域残差”，不能先命名成纯物体运动或世界速度。

可能有技术深度的最小表示合同是：

1. **相机因素有观测锚定。** 背景分支只能接收所有预测框扩张区之外的像素/对应；独立空间留出验证背景预测。不是把全图 token 和目标 token 分成两支后让网络任意决定何者叫 camera。最短实现先使用有约束的背景映射，是否需要学习局部变形由真实留出误差决定。
2. **对象分支接收 native residual field。** 输入至少保留 `[R_x,R_y,forward-backward error,in-bounds/visibility support,local background error,normalized spatial coordinates]` 的空间布局与两个相邻时间步，而非先平均成一个速度向量。当前框只定义观测区域，不是假定的前景 mask；历史像素由 backward flow 追溯，不依赖当前 tracker ID 连贯性。
3. **学习的是可解释映射与表示，不是身份分数。** 对象 encoder 输出局部支持/残差重建及向量表示。非零的真实运动重建、背景留出预测、相机扰动下正确等变与对象残差改变下的灵敏度共同约束它。ID/contrastive loss 若以后加入，只能使用合法 fit 同约定标签，并 mask 未知；不能让纯 RGB appearance 支路绕过运动重建任务。
4. **跨视图坐标关系必须可观测。** 比较两 view 的 R 时，要有从独立背景得到的局部 transfer，不能直接比较两个坐标系的向量。精确差应计算 `H(Φ(p))-H(C(p))`，小位移 Jacobian 只是近似。单应对离地目标/非平面视差的误差必须单列。未得到有效背景 transfer 时该损失 mask，不得依赖待匹配目标的正例身份估计变换。

三个主要退化必须显式排除：`camera=observed flow, object=0`；`camera=0, object=all flow`；embedding 完全忽略 motion 而靠 appearance 完成 ID loss。独立背景输入边界、真实 residual 重建和干预诊断分别对付这些退化，不能仅通过加几个 loss 名称宣称分解成功。

以下仍是普通控制：GMC/单应 + bbox 速度；mean residual + appearance concatenation；两个 MLP 后 attention fusion；把残差距离加入 Hungarian；预测一个质量 gate/阈值；把稀疏 CoTracker 点插值再池化。这些可能有工程价值，但不足以兑现本方向的方法深度。

## 5. 训练前最短新门：60 张图，先分清三个问题

以下是**未执行的建议**。首门固定 5 fit pairs 23/25/29/69/78，每 pair 取两个内部 anchor `floor(F/3)` 与 `floor(2F/3)`，每个 anchor 只读 `t-2,t-1,t`，两 view 共最多 60 张新图。按 frozen stream 的 `image_stem` 读取，不能用 runtime frame 拼文件名；pair78 的已知偏移必须由 stream 绑定。它不是 P21 的八个 t/t-8 端点或 bbox 中心 sweep。

每个 view/anchor 计算两个邻帧间隔的双向光流：共 80 个方向调用（tile 数另乘，需先封存 tile 清单）。同一批图像可算 DIS 对照，但不要同时展开四个深度 extractor。先封存 pair/anchor/tile/frame/path/source SHA 清单、提取器权重 SHA、pad/scale/crop/flow-direction 合同；特征冻结后才附加离线标签。

### G0：工程合同，不是科学结果

不使用 MDMT 的已知平移/仿射 checker 可以检查 flow 向量缩放、crop offset、前后向 composition、mask 和精确坐标恢复。真实图首个固定 tile 再验证 RGB、flow shape、有限值、原始像素单位、pad 去除和因果截止。任何失败是 `HOLD_DENSE_INPUT_IMPLEMENTATION`，不能记成 motion 科学失败。随机权重 smoke 只验证形状，不能进入信息结论。

### G1：真实 temporal dense-field 是否有可解释的额外信息

单位是 pair→anchor→目标区域；像素/候选边不是独立 replicate。对所有预定目标报告连续的 native 像素 FB discrepancy、有效支持比例、框内/邻近框外残差、局部背景空间留出误差和估计空间自由度。框内像素不能统称 foreground；静止目标、同速车流、遮挡、边界出界必须保留，不能只选视觉漂亮的独立运动对象。

固定比较三个重建输入：普通背景映射；背景映射 + 对象区域单个 robust residual vector；背景映射 + 真实空间 residual field。报告在可见支持上的图像 warp 重建与连续时间可重复性，并把 raw flow、DIS/RAFT 等算法误差和背景模型误差分开。图像重建不是 optical-flow GT，也不是身份识别分数；它的作用是判定保留空间场是否比重复一个均值多解释了真实观测。

如果真实 dense 输出在 native 支持上仍退化为 <数个独立格、邻步不稳定或误差不小于匹配背景噪声，只能 `HOLD_DENSE_FIELD_OBSERVABILITY`，并记录是哪类尺度/纹理/遮挡失败。不能提高插值分辨率、降低 quality 阈值后称通过。如果空间场与 robust scalar 几乎等价，则只能保留“更稳健的运动测量”工程结论，没有证据支撑高容量空间表示。

这个 10-anchor 工程门不足以证明5个pair上的统计稳健性，也没有预设一个根据未知误差标度臆造的身份 lift 门。通过只表示值得形成下一份训练前协议，不自动授权训练。

### G2：两类干预要区分真实观测检查与代数自证

对同一个真实输入按固定 seed 施加以下配对条件，按 pair 汇总，不把 60 张图或几万像素计作独立样本；处理顺序随机，原图与干预图共享完全相同有效域。这使用 experimental-design 的 blocking/repeated-measures 原则。

- **相机/坐标扰动。** 对各帧 RGB 施加已知小投影变换 `G_t`，重新运行 flow 与目标排除的背景估计，再按精确端点变换拉回原域。若原始映射为 Φ，正确变换是 `Φ^G=G_(t+1) ∘ Φ ∘ G_t^-1`；背景 C 同理。必须分别拉回 `Φ^G` 与 `C^G` 的端点后再取差，不能把所有透视流都当平移向量。这检验提取流程是否对已知相机坐标变化稳定。只在 tensor 上令 `flow'=flow+camera` 再减 camera 是恒等式，不能算成功证据。
- **对象残差干预。** 保留背景映射和支持，在同类、相近 native 尺度的观测之间替换 residual 的空间组织；同时比较空间打乱但保留均值、整场置零、真实对象 field。首门这是 field-space 机制检查，并非真实物理对象交换或真实视频性能。若后续表示对全部变化完全不敏感，则运动被忽略；若对坐标扰动与对象残差改变同样敏感，则分解目标尚未达到。

图像投影扰动也只是受控坐标/合成相机变化，不能证明真实跨无人机视点不变性。真实物理对象交换需要可靠 mask/渲染，本数据合同尚未提供，不能伪称完成。真正跨视图可迁移仍由 G3 决定。

### G3：cross-view 支持与条件互补独立审计

相邻时间光流成功不自动增加跨视图背景对应。不要把普通 RAFT/GMFlow 直接用于跨度很大的 cross-view 图对后默认其输出可信。可以另提一个预先锁定的独立背景匹配器/局部映射，但必须先在未参与拟合的背景上给出位置覆盖与留出误差；不能用同ID目标作为几何锚点。P21 原始结果只存了几何诊断，没有保存全部变换矩阵，不能假称已有 H cache 可直接提取。

若 G1/G2 成功而 G3 缺乏独立对应，结论是 **temporal dense input 可观测，但跨视图分解表示 HOLD**，不转成单视图方法、不切数据集。

若之后形成足够支持的 fit-only 标签信息门，应使用完整冻结合法候选与固定 appearance/static reference，考察**给定静态/外观不确定性时的额外信息、真实救错与破坏**；同时报告每 pair 所有目标的覆盖，不能只在 P21 的四个错误上设计分支。未知和正例不在候选的情况单列，拒绝 GT 净化候选。不能再要求额外 cue 的 standalone score 在 near-perfect static 上统一 +.02，也不能让一个新可学习 selector 成为输入证明本身。该阶段仍是 exploratory 信息证据，不是正式 MOT 或训练授权。

## 6. 最小代码路线，全部是待建模块

建议只在新目录开发，旧 P21/历史 p2 等只读：

| 建议新文件 | 最小职责 |
|---|---|
| `p22_dense_design/prepare_manifest.py` | 从 frozen stream 显式绑定 fit pairs、内部 anchors、image_stem、全图背景与 native tile；冻结清单；拒绝 cal/dev/val/test 路径 |
| `p22_dense_design/dense_flow.py` | 官方 torchvision RAFT wrapper；显式本地 checkpoint、RGB/pad/单位/方向；双向输出；可选固定 DIS 控制，不暴露 sweep 参数 |
| `p22_dense_design/background_field.py` | predicted-box exclusion、独立空间 holdout、全局映射及 native-domain 变换；不加载身份标签 |
| `p22_dense_design/object_field.py` | 从当前框 backward pixel trajectories 构建真实 residual tensor、支持与噪声诊断；不以 local tracker ID 选历史 |
| `p22_dense_design/intervene_and_audit.py` | G1/G2 所需原图/重新估计后的相机扰动、field-space object 控制、pair级报告；不输出 ranker score |
| `p22_dense_design/OPERATOR_CONTRACT.json` / `INPUT_RECEIPT.json` | 先封存预算/边界，后记源码/权重/图像/flow/输出 SHA、实际调用数、失败原因与退出码 |

先实现这些有界输入工件；只有可观测性与 direct same-task novelty 审查支持后，才值得新增 `residual_encoder.py` 和学习目标。不能以实现一个 encoder 就把状态改成方法确认。当前 SAME-MDMT 全文及广义 camera/object motion factorization 先例仍是必要新颖性缺口；本文件只做可实施性审计，不为 novelty 背书。

## 7. 当前决策与审计指纹

**KEEP：** 利用已有 torchvision RAFT 做真实、原分辨率局部 dense-field 输入工程。**HOLD：** 相机/对象表示学习、cross-view loss、训练、cal/dev/official-val/test、方法新颖性与正式跟踪分数。**排除：** 新一轮 bbox 速度/阈值/ranker 扫描及 P2/P3/P10 重包装。

本次检查时 SHA-256：

```text
16948b3d78acba0bb9473a7f0f95c2af4dfbb2705aa4c2747433ebc3d73708d9  torchvision/models/optical_flow/raft.py
f1ae236d1d0e1d2d759b0d3bd307ee4f4d492f31dda1d4a4602ac5e83dd9bcfa  /home/chenhc/claude_try_MDMOT/CONTINUE.md
247e5de3af8cd0d82759b50ac6ab950286c329b65a874fec848dc3fe231f7ae8  p21_residual_motion/probe.py
5f17ed4eb7a5873b88323c7b3bd06350bd88064339f7a29ce49f1e412e3c2472  p21_residual_motion/DECISION.json
```

以上源码/报告指纹不是新实验收据。P22 目前只有设计与可实施性结论，没有 dense-field 结果。
