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
- 7 项集成测试通过。

## 尚未完成

按推荐优先级排列：

1. 在 PILOC 环境中原生发出 `DetectionEvent`，不再依赖适配器重建。
2. 创建统一场景生成器和稳定目标 ID；目标发现后保留到真正救援完成。
3. 编写 `joint_driver.py`，加载 PILOC 与 `old_network` checkpoint 并运行真实闭环。
4. 用共享障碍图的路径代价替换 task-03 欧氏距离与直线移动。
5. 修复 `old_network` 全 padding 邻居 attention 的 NaN 风险。
6. 修复 `old_driver` critic padding mask 和 agent/时间步 GAE 对齐。
7. 将规则协调器替换为有界学习型残差网络，并加入训练 buffer。
8. 建立搜索、协调、任务分配联合奖励与延迟信用关联。
9. 增加端到端基准和消融：无协调、规则协调、学习协调、无反馈。

## 推荐下一任务

实现一个仅推理的 `joint_driver.py`：

- 使用固定小地图和固定 seed；
- 分别加载已有搜索与任务分配 checkpoint；
- 使用 `HeuristicCoordinator`；
- 运行一个 episode；
- 输出每个目标的发现时间、首次分配、阶段进度、完成时间和重规划次数；
- 不在该阶段做联合训练。

验收标准是可以从日志中用 `target_id/source_event_id` 追踪每个目标的完整生命周期。

## 资产说明

本仓库不包含 checkpoint、DungeonMaps、RALTestSet、TensorBoard 日志、动画、论文和结果图。
这些文件应放在团队共享存储或 GitHub Release/LFS 中，不能直接提交普通 Git 历史。
