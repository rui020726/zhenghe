# Agent 工作说明

本文件作用于整个仓库。任何自动化编码 agent 在修改代码前都必须先读本文件，再读
`ARCHITECTURE.md` 和 `PROJECT_STATUS.md`。

## 项目任务

构建搜索、协调和任务分配一体化的多机器人搜救系统。PILOC 负责发现目标，
CTDE 模块负责异构救援机器人任务分配，中间协调模块利用搜索态势和执行反馈优化分配。

## 唯一主线

### 搜索模块

- 入口：`piloc/train.py`、`piloc/test.py`
- 环境：`piloc/src/Map/timemap.py`
- 策略：`piloc/src/DDQN/Agent.py`
- `piloc/src/CPP/` 和 `piloc/src/base/` 是旧代码，只用于参考。

### 任务分配模块

- 训练：`task_allocation/old_driver.py`
- 环境：`task_allocation/old_task_env.py`
- 网络：`task_allocation/old_network.py`
- 评估：`task_allocation/old_test.py`

不要误用以下分支：

- `driver.py + network.py + env/task_env.py`：简化但不完整。
- `runner.py + worker.py + attention.py + central_controller.py`：废弃的 Ray 架构。
- `Ab_*`、`*_linear.py`：消融实验，不是生产主线。

### 集成模块

- 数据契约：`integrated_sar/contracts.py`
- 事件总线：`integrated_sar/event_bus.py`
- 任务真值：`integrated_sar/task_registry.py`
- 协调器：`integrated_sar/coordinator.py`
- 总调度：`integrated_sar/orchestrator.py`
- 适配器：`integrated_sar/adapters/`

新增集成功能优先写在 `integrated_sar/`。只有适配器无法完成时，才最小化修改两个原模块。

## 架构不变量

1. PILOC 的“发现”不是任务“完成”。目标必须经历独立的任务生命周期。
2. 搜索模块只发布一次 `DetectionEvent`；事件总线负责分发，禁止分别修改多个消费者。
3. `TaskRegistry` 是任务状态的唯一可信来源，所有事件必须幂等并带稳定 ID/时间戳。
4. 栅格坐标 `(row, col)` 转换为连续坐标时必须使用
   `x = col/(width-1)`、`y = row/(height-1)`，禁止交换。
5. 协调器优化分配 logits，但不替代分配器：
   `final_logits = ctde_logits + alpha * coordinator_bias`。
6. 环境硬 mask 最后应用；协调器绝不能使非法动作重新合法。
7. 任务反馈必须包含阶段、剩余五维能力需求和进度，不得只返回布尔完成值。
8. episode 内策略只推理；权重更新放在 rollout/episode 边界，避免在线漂移。
9. actor/协调器只能使用通信可达的 belief；环境真值只供模拟器和训练 critic 使用。
10. 第一阶段默认搜索机器人与救援机器人是两组实体，共享场景和时钟但不共享位置状态。

## 已知问题，修改前先确认

- `piloc/src/Map/timemap.py` 构造函数在 `test=False` 时可能在设置 `self.map` 前访问它。
- PILOC 当前在发现目标后立即删除 `goals_pos`；适配器只是兼容方案，最终应原生发事件。
- `piloc/src/DDQN/Agent.py` 的 `normalization` 导入路径可能在仓库根运行时失败。
- PILOC 当前 critic 并非真正全局 centralized critic。
- `old_network.py` 在所有邻居槽都是 padding 时需要防止 attention 产生 NaN。
- `old_driver.py` 的 critic padding mask 和按 agent 决策排列的 GAE 仍需系统修正。
- task-03 使用欧氏直线移动，尚未接入 PILOC 障碍地图的最短路径代价。

不要在无测试的情况下顺手修复多个问题；每个问题使用独立提交和对应回归测试。

## 修改规则

- 保留用户已有改动，不回退与当前任务无关的文件。
- 接口优先：先修改 contracts/ports，再实现 adapter/caller，最后补测试和文档。
- 不硬编码本机绝对路径、GPU 编号或 checkpoint 位置。
- 随机过程必须可注入 seed；训练和评估配置必须可序列化。
- 不提交模型权重、日志、动画、论文、数据集、IDE 配置或密钥。
- 不新增第二套同义事件/状态结构；扩展现有 contracts。
- 改变 observation 维度时，同步更新网络、padding、checkpoint 兼容策略和测试。

## 验证要求

最低要求：

```bash
python -m unittest discover -s integrated_sar/tests -v
python -m integrated_sar
```

若修改 PILOC，还应验证其环境 reset/step 和目标发现逻辑；若修改任务分配模块，
还应验证 `old_task_env` 外部任务注入、mask、阶段推进和 `old_network` 前向传播。

提交说明中必须写清：修改范围、数据维度变化、运行的测试、未解决风险。

## 完成定义

一项集成任务只有在以下条件同时满足时才算完成：

- 从发现事件能够追踪到最终任务结果；
- 没有重复任务或陈旧命令覆盖新状态；
- 无协调器基线仍可运行；
- 新行为有自动化测试；
- README/ARCHITECTURE/PROJECT_STATUS 与实现一致。
