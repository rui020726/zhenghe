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
硬 mask 不可被残差绕过、被拒命令在下一 tick 恰好重规划一次、
检测半径在环境/策略/适配器三处只有一个真值、通信范围外的目标不被任何机器人或协调器使用、
非整数 `dt` 下三个时钟仍然一致、场景载入 PILOC 后不残留随机初始状态、
复用适配器开新 episode 不会重复事件 id、mock 的双通道地图与 PILOC 逐格一致，
以及 checkpoint/地图/依赖缺失时的清晰报错。

第二轮审查后的补充覆盖：远距离检测经上行中转送达救援 belief 并完成闭环、
未收到消息的机器人看不见也选不到该任务、消息首次到达当 tick 立即协调并恰好分配一次、
gossip 每 tick 最多传播一跳、丢包时 belief 保持原值、通信随机过程按 seed 复现、
断联期间真值推进不改变 observation/mask 而新消息到达后才更新、
公开 reset 恢复三个时钟/两个模拟器/日志与计数器且同 seed 逐记录复现、
日志只记录 Registry 接受的事件（迟到/未知/终态后事件三者一致被拒）、
mock 前沿策略不读视野外的信息素，以及负的检测半径以退出码 2 拒绝。

第三轮审查后的补充覆盖：真实 PILOC 的 reset 会 reseed 并重建信息素/位置历史/通信统计/
延迟消息，适配器与协调器的每 episode 计数器一并复位；协调偏置与导航目标只来自
该机器人的 belief（只改注册表真值不改变二者）；belief 的位置/截止时间/未来需求/
复杂度/进度/成员/动态与已分配标志任一变化都被报告为调度信号，纯时间戳刷新不算；
被注册表拒绝的检测不入通信队列也不触发协调；通信拓扑可通过配置收窄并在集成测试中生效；
合法 re-detection 的日志按注册表快照写入并保留最初 detection 作为关联键。

第四轮审查后的补充覆盖：协调器完全不读注册表（只改注册表置信度、或把任务置为终态，
都不改变任何偏置，也不会让还没收到消息的机器人重规划）；一个机器人 belief 的变化不会经
池级 `task_bias` 影响另一个机器人；中转消息携带的 progress/confidence/revision 全部写进
belief，同时间戳但版本更高的合法更新不再被丢弃，更旧或同刻同版本的更新会被丢弃；
`ExecutionStepResult` 的字段顺序保证位置参数调用仍然是 metrics；真实 PILOC reset 用例
改用 60×60 场景，并用 stand-in 环境覆盖生产构造路径（60×60 守卫与派生状态重建）。

第五轮审查后的补充覆盖：机器人仍相信一个已被注册表关闭的任务时，命令带缓存的终态
revision 通过闸门，机器人前往现场核实并学到“已完成”，同场的活跃任务不会被饿死；
gossip 在时间戳相同时按 revision 取新，两台相邻机器人不会长期保留不同版本的同一任务；
`AgentTaskBelief` 与 `CoordinationAdvice` 的新字段为 keyword-only 且追加在原字段之后，
旧的位置参数调用不会被重新绑定；真实 PILOC 测试只在 gym/matplotlib/Pillow 确实缺失时跳过，
依赖齐全后的导入失败会让测试失败而不是伪装成 skip。

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
`--allocation-interval`、`--dt`、`--detection-radius`（不允许为负，0 合法但退化）、
`--comm-delivery-probability`、`--comm-delay-ticks`、
`--comm-subscribers`（`all` 默认覆盖全救援队，或 `first:N`，或机器人 id 列表）、
`--no-coordinator`、`--quiet`。
`--mock` 与任一 checkpoint 参数互斥。

使用 `--piloc-checkpoint` 时还有两个约束，启动即校验，不会静默放行：
`--detection-radius` 必须为 5（时间戳地图、信息素窗口和 `last_actions` 的形状都由
`obs_radius` 推出，半径不一致会让环境按一个半径删除目标、适配器按另一个半径发事件）；
`--dt` 必须为 1.0（`timemap.Environment` 每步固定把自己的时钟推进 1，无法配置）。

两侧后端互相独立：省略 `--piloc-checkpoint` 时搜索侧使用前沿贪心策略，
省略 `--task-checkpoint` 时分配侧使用启发式打分，摘要中的
`search_backend`/`allocation_backend` 会标明实际使用的后端。

目标如何传到救援队，决定了这一轮能完成多少。检测先只写入环境真值（`task_dic`），
`integrated_sar/comm.py` 的 `CommRelay` 再把它作为显式通信投递进机器人 belief：
只有注册表已接受的检测才会被转发，上行链路默认覆盖订阅它的整支救援队，可用
`--comm-subscribers` 收窄到 `first:N` 或指定 id；每包有可配置的延迟与丢包率，丢包
不重发、belief 保持原值，整个随机过程由 seed 控制；中转不读任何人的位置，因此救援队
是否与搜索起点重合不是闭环启动的必要条件。收窄拓扑只限制投递范围——机器人仍可能
经视距感知或 gossip 获知任务，那两条信道由 `old_task_env` 自己建模（gossip 每 tick
最多一跳）。分配器、协调器与执行模块都不能使用机器人尚未收到的信息：协调偏置只按
该机器人自己的 belief 打分，导航目标取 believed 位置，在途重规划也只看 belief，
断联期间真值推进不会改变 observation、mask 或行动目标（架构不变量 9）。
为此救援队的集结点默认仍放在搜索队的出发格上，两队共享同一个灾区；
`TaskEnv` 默认会把集结点随机撒在整个单位正方形。

`--output` 写出 JSONL 生命周期日志（`-` 表示标准输出），每条记录至少包含
`target_id`、`source_event_id`、`event_type`、`timestamp`、`discovered_by`、
`assigned_agents`、`task_revision`、`stage_index`、`progress`、
`remaining_requirements`、`status`、`failure_reason`，
可用 `target_id` 与 `source_event_id` 串起单个目标从发现到完成的完整轨迹。
日志属于运行产物，不要提交。

## 训练

不要把三个模块一起端到端训练。按以下顺序推进，每一步稳定后再进下一步：

```text
PILOC 单独训练 → 任务分配单独训练 → 冻结两端跑通联合推理 → 训练协调器 → 按 episode 交替微调
```

搜索模块：

```bash
cd piloc
python train.py --device auto --seed 0 --episodes 120000 \
  --output-dir ../runs/piloc_search --save-every 1000 --log-every 100
```

产出 `<output-dir>/checkpoints/shared_mappo_model_21_<episode>.pth`、`run_meta.json`
（git commit、seed、device、dim_info、全部超参）与 `episode_metrics.jsonl`
（发现率、首次发现步、覆盖率、移动距离、重复搜索比例）。
`--resume <checkpoint>` 以 strict 方式续训，不接受部分加载。

任务分配模块：

```bash
cd task_allocation
python old_driver.py --device auto --seed 0 --episodes 20000 --no-resume \
  --checkpoint-dir ../runs/ctde --actor-kv-mode tokens
```

产出 `ctde_best.pth`、`ctde_final.pth`、`ctde_ep_<episode>.pth` 与 `run_meta.json`。
每个 checkpoint 都记录 `actor_kv_mode`、网络维度、seed 与 commit；`summary` 与
`tokens` 的参数形状完全相同，加载端会拒绝模式不匹配的权重。新训练请加
`--no-resume`，否则会自动加载目录里最新的 checkpoint 并跳过 `global_episodes_done`，
而熵权重、学习率衰减和课程调度都由它驱动。

两端训好后冻结参数，用 `joint_driver` 验证完整数据链（见上一节）。

两点运行注意：

- 本机 `torch` 为 CPU 版，只适合冒烟验证；正式训练请放在有 GPU 的机器上。
- `--checkpoint-dir` / `--output-dir` 请用绝对路径且不含非 ASCII 字符：
  TensorFlow 的 gfile 在含中文的路径下使用相对目录会以
  `FailedPreconditionError` 失败。

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
