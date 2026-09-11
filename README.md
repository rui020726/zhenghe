# Zhenghe：搜索—协调—分配一体化搜救系统

本仓库整合两个已有研究模块：

- `piloc/`：多智能体区域搜索、时间戳地图、信息素和受限通信。
- `task_allocation/`：异构机器人、多阶段任务和 CTDE/MAPPO 任务分配。
- `integrated_sar/`：两者之间的事件、任务状态、协调优化与执行反馈框架。

当前目标是形成完整闭环：

```text
PILOC 搜索目标
  → DetectionEvent
  → TaskRegistry
  → Coordinator 提供残差优化
  → CTDE 分配机器人
  → 执行任务并产生 ProgressEvent
  → 注册表和协调器更新
  → 必要时重新分配
```

## 先读哪些文件

新加入的开发者或 agent 请按以下顺序阅读：

1. [`AGENTS.md`](AGENTS.md)：代码主线、约束和禁止事项。
2. [`ARCHITECTURE.md`](ARCHITECTURE.md)：模块边界、数据流和训练方案。
3. [`PROJECT_STATUS.md`](PROJECT_STATUS.md)：已经完成和下一步尚未完成的工作。
4. `integrated_sar/contracts.py`：模块间的数据契约。
5. `integrated_sar/orchestrator.py`：端到端运行顺序。
6. `integrated_sar/adapters/`：现有两个项目的适配方式。

## 当前代码主线

- 搜索训练：`piloc/train.py`
- 搜索环境：`piloc/src/Map/timemap.py`
- 搜索策略：`piloc/src/DDQN/Agent.py`
- 任务分配训练：`task_allocation/old_driver.py`
- 任务环境：`task_allocation/old_task_env.py`
- CTDE 网络：`task_allocation/old_network.py`
- 任务分配评估：`task_allocation/old_test.py`

`task_allocation/driver.py`、`network.py`、`env/task_env.py` 是未完成的简化分支；
`runner.py`、`worker.py` 等是旧架构。除非任务明确要求，否则不要基于它们继续开发。

## 快速验证

框架核心测试仅依赖 Python 标准库和 NumPy：

```bash
python -m unittest discover -s integrated_sar/tests -v
python -m integrated_sar
```

预期演示最后输出：

```text
final_status=completed
```

当前测试覆盖：事件扇出、目标去重、坐标转换、协调偏置、PILOC 发现适配、
外部任务注入 `old_task_env`，以及“发现—分配—执行—完成”端到端闭环。

## 环境

建议使用 Python 3.10 或 3.11：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

模型权重、DungeonMaps 和大规模测试集没有提交到 Git。需要运行完整训练或论文评估时，
请从团队共享存储取得这些资产，并放到代码期望的本地目录中。

## 协作方式

不要直接在 `main` 开发。每项工作建立独立分支，例如：

```bash
git switch -c feat/native-detection-event
git add -p
git commit -m "feat: emit detection events from PILOC"
git push -u origin feat/native-detection-event
```

通过 Pull Request 审核后再合并。详细规则见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
