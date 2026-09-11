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
7. `integrated_sar/joint_driver.py`：把上述部件接成可运行的推理闭环。

## 当前代码主线

- 搜索训练：`piloc/train.py`
- 搜索环境：`piloc/src/Map/timemap.py`
- 搜索策略：`piloc/src/DDQN/Agent.py`
- 任务分配训练：`task_allocation/old_driver.py`
- 任务环境：`task_allocation/old_task_env.py`
- CTDE 网络：`task_allocation/old_network.py`
- 任务分配评估：`task_allocation/old_test.py`
- 联合推理入口：`integrated_sar/joint_driver.py`
- 场景生成与坐标转换：`integrated_sar/scenarios.py`
- 搜索侧运行时：`integrated_sar/search_runtime.py`、`integrated_sar/piloc_runtime.py`
- 任务侧运行时：`integrated_sar/task03_runtime.py`

`task_allocation/driver.py`、`network.py`、`env/task_env.py` 是未完成的简化分支；
`runner.py`、`worker.py` 等是旧架构。除非任务明确要求，否则不要基于它们继续开发。

## 快速验证

框架核心测试仅依赖 Python 标准库和 NumPy：

```bash
python -m unittest discover -s integrated_sar/tests -v
python -m integrated_sar
python -m integrated_sar.joint_driver --mock --steps 20 --seed 42
```

预期演示最后输出：

```text
final_status=completed
```

预期 `--mock` 运行逐 tick 打印进度，最后输出一段 JSON 摘要（`targets_discovered`、
`status_counts`、`rejected_assignments` 等）并以退出码 0 结束；
checkpoint、地图或依赖缺失时打印一行 `[joint-driver] <ErrorType>: <原因>` 并以退出码 2 结束。

当前测试覆盖：事件扇出、目标去重、坐标转换、协调偏置、PILOC 发现适配、
外部任务注入 `old_task_env`、“发现—分配—执行—完成”端到端闭环、
mock 联合推理的确定性与生命周期日志字段、分配命令 TTL 与 revision 校验、
硬 mask 不可被残差绕过，以及 checkpoint/地图/依赖缺失时的清晰报错。

## 联合推理入口

`integrated_sar/joint_driver.py` 只做推理，不训练任何网络。

无 checkpoint 的 mock 模式（CI 与未配置权重的机器可用，只需 NumPy）：

```bash
python -m integrated_sar.joint_driver --mock --steps 20 --seed 42
```

加载真实权重的模式：

```bash
python -m integrated_sar.joint_driver \
  --piloc-checkpoint <shared_mappo_model_*.pth 或其所在目录> \
  --task-checkpoint <ctde_final.pth 或其所在目录> \
  --map <障碍图 .npy/.png/.txt，可省略以按 seed 生成> \
  --seed 42 --steps 200 --device cpu --output joint_lifecycle.jsonl
```

常用参数：`--search-agents`、`--num-goals`、`--grid-size`（使用 PILOC 权重时必须为 60）、
`--obstacle-density`、`--rescue-species`、`--rescue-per-species`、`--actor-kv-mode`、
`--residual-scale`（协调残差上界，0–1）、`--command-ttl`、`--coordination-interval`、
`--allocation-interval`、`--dt`、`--detection-radius`、`--no-coordinator`、`--quiet`。
`--mock` 与任一 checkpoint 参数互斥。

两侧后端互相独立：省略 `--piloc-checkpoint` 时搜索侧使用前沿贪心策略，
省略 `--task-checkpoint` 时分配侧使用启发式打分，摘要中的
`search_backend`/`allocation_backend` 会标明实际使用的后端。

`--output` 写出 JSONL 生命周期日志（`-` 表示标准输出），每条记录至少包含
`target_id`、`source_event_id`、`event_type`、`timestamp`、`discovered_by`、
`assigned_agents`、`task_revision`、`stage_index`、`progress`、
`remaining_requirements`、`status`、`failure_reason`，
可用 `target_id` 与 `source_event_id` 串起单个目标从发现到完成的完整轨迹。
日志属于运行产物，不要提交。

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
