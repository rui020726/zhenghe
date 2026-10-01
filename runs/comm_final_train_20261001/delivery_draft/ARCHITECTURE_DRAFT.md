# 系统架构说明草稿（通信控制训练路径）

状态：草稿（ROUND36 并行文档任务 3a）。所有行号指向本地分支 `fix/comm-task-reward-event-timing`
上的实际文件；本文描述"修复版采集时序"生效后的现状。

## 1. 组件与所有权

| 组件 | 文件 | 拥有的东西 | 训练路径里谁调用它 |
|---|---|---|---|
| `RunConfiguration` / `build_joint_run` | `integrated_sar/joint_driver.py` | 一次运行的全部配置字段与 `run_signature["config_hash"]` | `v2_run._run_episode`（`v2_run.py:139-147`） |
| `JointSAROrchestrator` | `integrated_sar/orchestrator.py:30` | 逐步调度顺序、`GlobalClock`、事件总线、协调/分配触发 | `run.run()` 内部 |
| `Task03Adapter` | `integrated_sar/adapters/task03.py:164` | 执行环境（executor）、`comm_relay` 挂载点、`ProgressEvent` 构造 | `orchestrator.executor.step(now, dt)` |
| `CommRelay` | `integrated_sar/comm.py` | relay tick 计数、SEND/HOLD 决策点、预算/容量/合并、成本计数器、`transport_parameters()` | `Task03Adapter.step` 第一句（`task03.py:1193`） |
| `GlobalClock` | `integrated_sar/clock.py:6` | `now`（决策时间），只在事件发布之后才被推进 | `orchestrator.step` 末尾（`orchestrator.py:100`） |
| `InMemoryEventBus` | `integrated_sar/event_bus.py:13` | 同步分发顺序 = 订阅顺序，`history` 记录发布次序 | `orchestrator.event_bus` |
| `TaskRegistry` | `integrated_sar/task_registry.py` | 唯一把事件变成"事实"的地方：`was_accepted`、快照状态、deadline、`expired` | 总线第一个 `ProgressEvent` 订阅者（`orchestrator.py:73`） |
| `PublishedProgressRewardCollector` | `runs/comm_cost_v2_rerun_20260928/v2_reward_timing.py:14` | `task_stream[relay_tick]` 与逐事件账本 `records`；不发布、不改 registry、不动时钟、不抽随机数 | `v2_run._install_taps`（`v2_run.py:109-118`）注册在 registry 之后 |
| `StageBCollector` + `BudgetMaskedCollector` | `runs/comm_cost_stage_b_20260927/sb_collector.py`、`runs/comm_budget_action_mask_fix_20260922/budget_masked.py` | 逐决策 tape：`features / targets / actions / actionable / log_prob / value / cost_value / step` | `v2_run.run_training_episode`（`v2_run.py:308`） |
| Controller | `runs/comm_cost_v2_rerun_20260928/v2_controller.py` | trunk / action head / task value head / cost head 四个模块的参数 | `fresh_controller(seed)`、`load` 与 `_restore_controller` |
| Trainer | `runs/comm_cost_v2_rerun_20260928/v2_trainer.py` | 样本构造、GAE、PPO 更新、对偶更新、checkpoint 保存与恢复门 | `v2_train.train_seed` |
| 训练循环 | `runs/comm_cost_v2_rerun_20260928/v2_train.py:101` | 批结构（4 集/批、32 批/seed）、里程碑写一次、批记录落盘 | 本轮 `r36_worker.py` 直接调用 |

## 2. 一个 orchestrator step 内的事件顺序（时序契约的机器事实）

```
now = clock.now                       # 仍是"步前"时间（orchestrator.py:80）
decision_time = now + dt              # dt=1.0（本冻结配置）

1  search.step(now, dt)               # 探测；DetectionEvent 发布（orchestrator.py:87-89）
2  task_sink.sync_tasks(registry)     # 新发现立即对执行环境可见（:92）
3  executor.step(now, dt)             # → Task03Adapter.step（task03.py:1187）
     3a  comm_relay.tick()            # relay._tick 变为 t；策略在这一拍被采样（SEND/HOLD）
                                       # v2_run 的成本 tap 记录计数器增量 → cost_stream[t]
     3b  env.step(dt)                 # 物理时间推进到本拍末端
     3c  构造 ProgressEvent(observed_at = env.current_time = t)
4  for e in progress_events: bus.publish(e)          # orchestrator.py:95-96，同步分发
     订阅顺序：① registry.handle_progress（提交事实）
               ② PublishedProgressRewardCollector.handle_progress
        → 采集器看到的 registry 状态已是提交后的状态
        → 此刻 clock.now 仍是步前值（观察到的 clock_now = t - 1）
        → 合格完成事件当场写 task_stream[t] += 1，并追加一条账本记录
5  clock.advance_to(decision_time)     # orchestrator.py:100，采集器之后
6  下一次采样开始                        # 奖励已在本拍内记好，不需要等下一拍
```

关键点：**采集键 = `relay._tick`**，不是 `env.step` 之后的扫描时刻，也不是 `clock.now`。
这就是 ROUND34 修复的内容（旧实现包装 `env.step`，在下一拍才扫 registry，使 `task_stream`
整体晚一拍）。

## 3. 从事件到梯度的数据流

```
task_stream{tick→奖励}  cost_stream{tick→字节}
        │                       │
        └────── v2_trainer._bundle(stream, decision_steps, horizon) ──────┐
                 每笔 tick 收益 → "该 tick 当时或之前最近的一条 tape 决策行" │
                 tick 0 保留为 unassigned prefix（不塞给未来动作行）        ┘
   ↓                                   ↓
逐决策 bundled_task / bundled_cost   prefix_task / prefix_cost_bytes
   ↓ cm.gae(γ=1, λ_GAE=0.95, bootstrap=下一行 value, 末行终止=0)
advantages_task, returns_task   advantages_cost, returns_cost
   ↓ cm.actor_advantage(λ·A_task − A_cost)          # λ 整批固定
样本 {features, targets, actions, free, log_prob, advantage, return_task, return_cost}
   ↓ v2_trainer.likelihood_alignment               # 权重移动之前，逐样本重算 log_prob ≤ 1e-6
   ↓ v2_trainer.ppo_update                          # 4 epoch × ceil(N/256) minibatch
loss = −surrogate.mean() + 0.5·(MSE_task + MSE_cost) − 0.01·entropy   # ALL_ROWS 归约
   ↓ clip_grad_norm 0.5 → optimizer.step（每 minibatch 一次）
   ↓ v2_trainer.batch_g_hat(批内 4 集 vs 只读 immediate reference 标量)
   ↓ v2_trainer.dual_update：λ_next = max(0, λ + 0.05·g_hat)
   ↓ v2_train：批记录 JSON（前后参数哈希、四模块 digest、形状、冻结权重哈希）
   ↓ v2_trainer.save_checkpoint → rolling；episodes_done ∈ {0,64,128} 时写不可变里程碑
```

## 4. checkpoint 身份与恢复边界

| 入口 | 行为 | 需要的字段 |
|---|---|---|
| `v2_trainer.save_checkpoint` | 写 payload 与 `training_state` **两处**的版本字段 | `training_version=constrained-cost-v2/published-completion-event-at-relay-tick/2`、`reward_timing_version=published-completion-event-at-relay-tick/2`、loss/gae/optimizer/reward_timing schema、`freeze` 四字段、`rng` |
| `v2_trainer.load_checkpoint` | 先 `_validate_checkpoint_base`（schema/loss/gae/optimizer/reward_timing_schema/freeze），再版本门；**任一不符即在构造 controller、构造 Adam、恢复 RNG 之前抛出** | 上表全部 + 两个版本字段 |
| `v2_trainer.load_checkpoint_read_only` | 只装权重，返回 `load_mode=read_only`；缺 `reward_timing_version` 标为 `legacy_unversioned_capture_semantics`；`optimizer_restored=false`、`rng_restored=false` | 同一套 base 校验 |
| `v2_train.save_milestone` | 里程碑只写一次；若已存在必须逐字段相等才复用，否则抛错 | — |

旧轮（ROUND24/32）的 `.pt` 没有两个版本字段 ⇒ 在本代码下**不能被 `load_checkpoint` 恢复**，
只能只读观察。这就是"禁止热启动旧 checkpoint"的机器实现（ROUND35 的 R2/R3 已在真实 `.pt`
上验过 5/5 篡改拒绝与旧件拒恢复）。

## 5. 记账与产物位置（训练协议第 2 条的实现）

| 事实 | 由谁写 | 落在哪里 |
|---|---|---|
| 每个 episode 启动在构造 driver 之前记账 | `v2_common.LaunchLedger.reserve`（`v2_run.py:306`） | `seed<NN>/logs/START_LEDGER_p<pid>.json` 的 `entries`/`results` |
| 每集观测事实（成本/发射/任务/事件账本/哈希/耗时） | `v2_run._run_episode` 的闭合断言之后 | `seed<NN>/rows/v2_train_s<seed>_e<i>_s<scenario>_a<i>.json` |
| 逐决策 tape | `v2_run`（collector.records） | `seed<NN>/logs/..._tape.jsonl` |
| 每批更新与前后权重 | `v2_train.train_seed` | `seed<NN>/checkpoints/v2_batch_s<seed>_b<k>.json` |
| 权重 | `v2_trainer.save_checkpoint` | `seed<NN>/checkpoints/v2_train_s<seed>.pt`（rolling）与 `_ep{0,64,128}.pt`（里程碑） |
| 启动级记账与上限 | `r36_common.book_launch` | `TRAIN_LAUNCH_BOOK.json` + `seed<NN>/LAUNCH_RECORD.json` |
| 单进程 stdout（逐集/逐批进度） | `r36_launch.py` 重定向 | `seed<NN>/launch_console.log`（Git 外） |

报告里的所有计数都由这些账本重算（`r36_reconcile.py`），不是手打。

## 6. 本轮的干预边界（一句话）

干预只有一处：任务奖励的采集点从"`env.step` 之后扫 registry（晚一拍）"搬到"同一拍
`ProgressEvent` 同步分发时读已提交的 registry"，并把 tick 0 保留为未归属 prefix，外加两个
版本字段与恢复门。世界、搜索、分配、传输、成本、观测、mask、网络、目标、奖励数值、GAE/λ/
超参、场景块、行过滤、训练量全部按 ROUND32 冻结件不变（见 `FROZEN_CONFIG_MANIFEST.json`
与 `ROUND36_PREFLIGHT.json` 的 G4-G7）。
