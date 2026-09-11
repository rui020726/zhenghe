# 项目状态

更新时间：2026-09-11

## 已完成

- 阅读并确认 PILOC 搜索主流程与 task-03 的 `old_*` CTDE 主流程。
- 建立 `integrated_sar` 独立集成层，未侵入两个原项目。
- 定义发现、任务、协调建议、分配命令和进度反馈契约。
- 实现同步事件总线和幂等任务注册表。
- 实现无协调基线与规则残差协调器。
- 实现全局时钟和事件驱动联合调度器。
- 实现 PILOC 兼容适配器，可将当前“发现即删除”行为重建为事件。
- 实现 task-03 适配器，可注入外部多阶段任务、叠加协调 bias、执行并返回进度。
- 端到端演示能够完成 `发现 → 分配 → 执行 → 完成`。
- 统一场景生成器 `scenarios.py`：障碍图生成与加载（`.npy`/图片/文本）、最大连通域筛选、
  稳定 `target_id`、`CoordinateTransform`（`x = col/(width-1)`、`y = row/(height-1)`）。
- 仅推理联合入口 `joint_driver.py`：统一全局时钟、命令行参数、错误分类与 JSON 摘要。
- 搜索侧运行时：`search_runtime.py` 提供仅依赖 NumPy 的 `MockGridSearchEnv` 与前沿贪心策略，
  `piloc_runtime.py` 在有权重时按 strict 模式加载 PILOC 策略。
- 任务侧运行时：`task03_runtime.py` 按 strict 模式加载 CTDE 权重，并在加载期探针中
  拒绝产生 NaN logits 或违反环境硬 mask 的权重。
- 分配闸门 `assignment_gate.py`：TTL 过期与 `task_revision` 不一致的命令不进入执行，
  受影响 agent 转为重规划请求。
- 生命周期日志 `lifecycle_log.py`：JSONL 输出 12 个必需字段，可按 `target_id`/`source_event_id`
  追踪单个目标从发现到完成的完整轨迹。
- 无 checkpoint 的 mock 模式可在仅安装 NumPy 的机器上跑完整个 episode，且同一 seed 结果逐字节一致。
- 修复 `JointSAROrchestrator` 的注入缺陷：`registry or TaskRegistry()` 与 `clock or GlobalClock()`
  因 `__len__`/`__float__` 返回假值而静默丢弃调用方实例，已改为显式 `is None` 判断并补回归测试。
- 36 项集成测试通过。

## 尚未完成

按推荐优先级排列：

1. 在 PILOC 环境中原生发出 `DetectionEvent`，不再依赖适配器重建。
2. 让 PILOC 环境在发现后保留目标实体。当前 `timemap` 仍会删除已发现目标；
   集成层已通过注册表把任务生命周期保留到真正完成，但搜索侧的原生行为尚未修改。
3. 用共享障碍图的路径代价替换 task-03 欧氏距离与直线移动。
4. 修复 `old_network` 全 padding 邻居 attention 的 NaN 风险。目前只在加载期探针中拦截，
   `summary` 模式的权重会被拒绝加载，主线 `tokens` 模式不受影响，上游代码尚未修改。
5. 修复 `old_driver` critic padding mask 和 agent/时间步 GAE 对齐。
6. 将规则协调器替换为有界学习型残差网络，并加入训练 buffer。
7. 建立搜索、协调、任务分配联合奖励与延迟信用关联。
8. 增加端到端基准和消融：无协调、规则协调、学习协调、无反馈。
9. 用真实 checkpoint 复跑联合闭环。本仓库不含任何 `.pth`，当前真实权重路径只用
   架构匹配的合成 checkpoint 验证过（加载、strict 校验、探针、分配、执行、日志），
   真实训练权重的数值行为仍待在有权重的机器上确认。

## 推荐下一任务

在有权重的机器上跑一次真实闭环，并确认日志可追踪：

- 用团队共享存储中的 `shared_mappo_model_*.pth` 与 `ctde_final.pth`；
- 固定 60×60 障碍图与 seed，`--steps` 取足以完成若干目标的长度；
- 检查摘要中 `search_backend=piloc-trained`、`allocation_backend=ctde-network`、
  `rejected_assignments` 是否非零；
- 若 PILOC 权重与当前网络结构不匹配，驱动会以 `CheckpointIncompatibleError` 明确报错，
  此时应核对 `dim_info` 与 `num_agent` 而不是放宽 strict 加载；
- 之后再做第 1、2 项，让搜索侧原生发事件并保留目标。

验收标准是可以从日志中用 `target_id/source_event_id` 追踪每个目标的完整生命周期。

## 资产说明

本仓库不包含 checkpoint、DungeonMaps、RALTestSet、TensorBoard 日志、动画、论文和结果图。
这些文件应放在团队共享存储或 GitHub Release/LFS 中，不能直接提交普通 Git 历史。
