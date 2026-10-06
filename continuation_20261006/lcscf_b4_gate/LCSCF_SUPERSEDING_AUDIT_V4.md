# LCSCF v3 结论撤回与 v4 修复审计

v3 的 600-step 负结果不能作为方法失败证据。复核发现四类实现/合同问题：

1. 3x3 反向 transport plan 直接在同一 query 位置相乘，没有在目的位置读取反向概率；移动后返回的最小反例旧值为 0，正确值为 1。
2. 跨视角一致性比较的 map reference 对调，最小方向一致反例旧值为 0，正确值为 1。
3. unknown candidate 虽然在部分 loss 中被 mask，评估排序和概率分母仍可能包含它；高分 unknown 的最小例子把已知正例从 rank 1 算成 rank 2。
4. 预处理、RoI extent 与 detector 的 keep-ratio/pad32 合同没有逐像素锁定；v3 使用 800x1333 zero canvas，而 native batch 实际为 768x1344 padded shape。

修复后由 `audit_lcscf_v4.py` 执行三轮、每轮三项检查：算子坐标与极端温度梯度；mmdetection 原生像素/box/RoI/标签边界；真实 AutoAssign 258 tensor、K=0/1/64、真实 episode backward/optimizer。全部通过，报告为 `LCSCF_IMPLEMENTATION_AUDIT_V4.json`。v3 run 保留为 invalid implementation attempt，禁止用于论文方法判停。
