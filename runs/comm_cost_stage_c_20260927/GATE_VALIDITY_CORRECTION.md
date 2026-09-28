# ROUND23 历史门禁有效性勘误

保留 `FIRST4_GATE_s11.json`、其他旧 gate 文件与 `STAGE_C_VERDICT.md` 原样。它们记录旧实现下运行完成并通过原门禁这一历史事实；这些 `passed` 不能证明登记的 N4 学习目标正确。

## 旧 gate 的盲点

- G3 检查器复制了训练端的负成本输入、全零非终止 bootstrap 和 `lambda*A_task - A_cost` 组合，因而“训练端与检查端重算相同”仍会认可双重取负。
- G1 把落盘 optimizer 参数形状与同一个不完整 `controller.parameters` 形状列表比较。旧优化器与旧 controller 同时漏了 cost_head，仍能相等；整体参数 hash 的变化不能证明 cost head 自身训练过。
- G3 的总和闭合不能识别时间错位。旧 `_bundle` 把 tick 9 的收益归到动作 10；同一组正负事件总和仍能闭合。

## 修订后的核验

- 检查器从 owning module 独立构造 trunk/action/task-value/cost-value 参数期望集，并要求模块参数对象无重复，controller 完整包含该集合。历史落盘记录只有 shapes，没有跨进程可比的对象 ID；因此形状门禁只作为历史筛查，真正的对象 ID / 缺项 / 外来参数 / 重复检查由同进程合成生命周期测试执行。
- 生命周期测试分别给四个模块非零合成 loss，核对每组权重与 optimizer state；再用实际 PPO update 检查 cost_head 的梯度、zero_grad、裁剪、权重更新与 checkpoint 恢复。
- 正确性预期取自独立的 SEND/HOLD 精确枚举与有限差分、双步非零 value GAE 闭式、以及事件顺序案例。时间重算采用决策区间 `[d_i,d_(i+1))`：前缀单列，区间内事件归给动作 `d_i`，最后区间延伸到 episode horizon。

具体失败/通过结果和计数位于 `CORRECTNESS_REGRESSION_EVIDENCE.json`；旧 checkpoint/tape 的只读事实见 `OPTIMIZER_PARAMETER_AUDIT.json` 和 `CORRECTED_TAPE_RECOMPUTE.json`。

## verdict 状态

旧执行的 677 次启动不撤销，原观察值不覆盖。登记目标的正确学习结论改为 `not_established_due_to_implementation`：

```text
execution_complete = true
registered_objective_valid = false
observed_comparisons = retained historical observations
correct_N4_learning_verdict = not_established_due_to_implementation
```

此勘误不把旧规则臂表现归因给学习缺陷，也不把原 task/byte 联合分类重述为 task-only 改善。本轮没有新环境启动或真实训练；修复通过也不自动触发重跑。
