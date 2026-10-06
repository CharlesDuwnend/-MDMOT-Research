# P28 causal FGFA：已实现的算子映射

日期：2026-10-05。范围：独立 PyTorch 模块及 CPU 契约验证。该组件采用已有 FGFA 算子；不构成新方法或 MDMT 效果证据。数据缓存、RAFT 抽取、训练与评估由 root 的其他阶段负责，本文不报告其完成状态。

## 已实现 API

```python
adapter = TemporalAdapter(
    channels=256, mode='fgfa', strides=(8, 16, 32, 64, 128), norm_eps=1e-6)

features = adapter(
    current_features,       # 每层 [1,256,H_l,W_l]
    past_features,          # 每层 [3,256,H_l,W_l]，依次 lag 1 / 4 / 8
    flows,                  # [3,2,H_flow,W_flow]，current -> past
    original_hw=(H0, W0),
    detector_img_hw=(Hd, Wd),
    pad_hw=(Hp, Wp))

aux = adapter(..., return_aux=True)
# dict: features / warped / valid / weights，每项都是按 FPN 层排列的 list
# valid: bool [3,1,H_l,W_l]
# weights: [4,1,H_l,W_l]，第0项始终是current
```

`warp_past_features` 单独导出相同的 warp 契约，返回 `(warped_levels, valid_levels)`。原始特征输入不被修改，当前特征不被 detach；模块没有持久化的帧缓存，也不加载数据、权重或 MMTracking 模块。

`validate_temporal_group(pair=..., view=..., current_frame=..., reference_records=...)` 验证 metadata 的同 pair/view、严格过去、无重复、由近到远及精确 lag `(1,4,8)`。record 字段为 `pair/view/frame`（兼容 `frame_id`）。返回 lag tuple。调用方必须把这些记录与实际 feature/flow 文件绑定；函数不会仅凭数组位置推断时间身份。序列内图像尺寸固定，flow 与 past feature 的顺序必须一致。

## 坐标契约与 P26 对齐

实际安装的 `mmdet/models/dense_heads/autoassign_head.py:165` 明确为：

```python
self.prior_generator = MlvlPointGenerator(self.strides, offset=0)
```

因此采用 **P26 offset=0 格点**，而非通用 half-stride anchor：

```text
x_det = j * stride                         # detector pixel-centre index
x_orig = (x_det + 0.5) * W_orig / W_det - 0.5
x_flow = (x_orig + 0.5) * W_flow / W_orig - 0.5
dx_feature = sampled_flow_x * W_det / W_flow / stride
x_past_feature = j + dx_feature
grid_x = 2 * (x_past_feature + 0.5) / W_level - 1
```

y 轴独立使用对应高度比例。光流栅格覆盖完整原图经 resize 得到的图像，没有 padding、crop 或 letterbox；root 的 flow 方案为 max-side 640、两轴各自 round 到 8 的倍数，故必须使用实际 H_flow/W_flow，而非只用一个宽度比例。所有 `grid_sample` 均 `align_corners=False`。

光流仅在其栅格边缘用 border 延拓，以避免图像首/末像素附近的常量位移被零 padding 衰减；有效图像范围另行检查。参考特征不靠 border 填充越界点：mask 同时要求当前格点与过去端点在未 pad 图像内、光流有限、双线性非零支持节点全部有效。超出有效特征节点的分数端点会被拒绝，即使它仍在原图范围内。该 mask 只排除无效 feature 节点，不声称卷积感受野完全没接触 padding。

P6/P7 显式使用 `ceil(pad_hw / stride)`，不会把 flow 按宽比 resize 成错误高度。offset=0 时粗层末节点可能仍有效，例如 pad 768×1344 / image 751×1333 的 P7 是 6×11，末列 x=1280 有效；half-stride 格点会错误将它视为 x=1344 后丢弃。

有效节点上零光流直接返回原参考值，避免 grid 归一化造成浮点非恒等。这一精确分支面向 **冻结 flow**；没有据此宣称 flow fine-tuning 的导数契约。非有限 flow 在采样后标记 invalid，并使用安全坐标，最终 warped invalid 值置零。

## 论文算子、适配和控制

| 项目 | 本实现 |
|---|---|
| FGFA 特征对齐 | 当前到过去光流的 backward sampling；从当前图像特征格点查询完整 flow 栅格。 |
| FGFA embedding | 共享 `1×1 → ReLU → 3×3 → ReLU → 1×1`，各层 256 通道；按 channel L2 normalize，eps=1e-6。这是原论文三卷积深度对 FPN256 的适配，不是原论文2048通道模型的精确复现。 |
| FGFA 权重 | current embedding 与对齐参考 embedding 的 cosine，按可用帧 softmax；current 总包含在集合中。 |
| 值适配 | 所有模式共享同构 `V(x)=x+Conv1×1(x)`；残差权重/bias初始化为零。A/B均可训练此同一结构。该适配用于本次冻结检测器的受控局部训练，非原 FGFA 额外创新。 |
| `single` | 仅 `V(current)`；不使用 flow/reference/embedding。初始化时逐元素精确恒等。 |
| `fgfa` | flow warp + embedding cosine 权重 + 加权 V(values)。 |
| `uniform` | 同一 flow warp 和 V；可用帧均匀平均，不使用 embedding。 |
| `unaligned` | 原位置 reference + embedding cosine + V；真正零位移，忽略 flow 值。 |

256 通道时 value adapter 为 **65,792** 参数，embedding 为 **721,664** 参数。`single/uniform` 活跃参数 65,792，另 721,664 embedding 参数未参与计算、梯度为 None；`fgfa/unaligned` 活跃参数 787,456。`parameter_counts()` 明示这一区别，不能称 A/B 活跃参数完全相等。两个模式都注册 embedding 仅便于 state_dict 对齐，不用于伪造控制条件。

没有历史或全部过去无效时，输出 `V(current)`；在初始化处精确等于原 current。padding 输出节点也保留原 current 路径，过去权重为零。目标仍由外部原 AutoAssign head 的 `loss_pos/loss_neg/loss_center` 监督；本文件未引入质量学习、阈值学习、ReID 或全局 ID 损失。

## CPU 验证

使用 P26 实际 `mdmt_env` 的 **PyTorch 1.10.1+cu113** 执行 CPU unittest；没有 CUDA 调用、模型权重或数据/GT读取：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python -m unittest test_temporal_module -v
```

首次完整执行 15/15 PASS。随后追加真实尺寸 float32 检查，发现精确负一格位移时少量 x=0 端点会因舍入误差被拒绝；已统一图像域和特征域的 ulp 容差，并将该例纳入第 16 项回归测试。该容差只处理数值边界，接受的采样坐标会夹回有效节点，未引入可调质量阈值。

修复后 **16/16 PASS**（0.247 s）。最终测试包含随机 feature 零流有效节点精确恒等；正/负位移与 x/y 分轴单位；由独立解析 ramp 检验的空间仿射 flow（可发现错误半 stride 原点）；padding 支持；P6/P7 ceil；flow 边缘常量不衰减；float32 非整数 resize 的精确边界；NaN/Inf/outside 全 invalid；三层 embedding 全零无 NaN；current-only/init 恒等；value/embedding/current 梯度；single 未使用 embedding 的披露；uniform/unaligned 控制语义；空参考；严格历史/同视图/顺序/重复拒绝。

这些测试证明本文件的数值与接口契约，不证明实际 RAFT 对 MDMT 小目标的对齐质量，也不证明训练收敛或 MDA 改善。P26 原文件没有被修改。
