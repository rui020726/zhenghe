# 系统架构与奖励事件顺序（终评视角）

状态：交付文档（ROUND37 §7 第 1 项）。行号指向本地分支 `eval/comm-final-fixed-evaluation` 上的实际
文件；训练视角的完整版在 `runs/comm_final_train_20261001/delivery_draft/ARCHITECTURE_DRAFT.md`，
本文件补充**评估路径**与训练路径的差异，并复述必须一致的那一段事件顺序。

## 1. 评估路径的组件与所有权

| 组件 | 位置 | 在终评里做什么 |
|---|---|---|
| `run_eval` | `runs/comm_cost_v2_rerun_20260928/v2_run.py:330` | 一个 (模型, 场景, 模式) 格：记账 → 构造 driver → 装 tap → 跑 1000 步 → 闭合断言 → 写行 |
| `BudgetMaskedCollector` | `runs/comm_budget_action_mask_fix_20260922/budget_masked.py:100` | 逐候选 SEND/HOLD 决策与逐决策 tape；`greedy=True` 用 `p[SEND] >= p[HOLD]`（`budget_masked.py:238-239`），`greedy=False` 用 `torch.multinomial` |
| `load_checkpoint_read_only` | `v2_trainer.py:409` | 只装权重：不建 optimizer、不动 RNG，返回 `load_mode=read_only` 与版本字段身份 |
| `run_rule` | `v2_run.py:274` | 规则臂：`immediate` 走原生策略；其余用 `v2_rules.build_rule` 注入 |
| `PublishedProgressRewardCollector` | `v2_reward_timing.py:14` | 与训练时同一个观察者；评估里它写入的 `task_stream` 不再进任何梯度，只用于闭合与账本 |
| `_install_taps` | `v2_run.py:92` | 成本 tap（relay.tick 计数器增量）+ 事件总线上的奖励观察者，训练/评估共用一份实现 |

**要点**：终评没有第二套采集代码。评估行里的 `reward_timing_version`、`task_stream`、
`task_reward_events` 与训练行由同一函数写出，因此"修复后的采集接线在评估里也生效"是可检查项
（门禁 T5），不是假设。

## 2. 一个 orchestrator step 内的事件顺序（不变的那一段）

```
now = clock.now                         # 步前时间（orchestrator.py:80）
1  search.step(now, dt)                 # DetectionEvent 发布（:87-89）
2  task_sink.sync_tasks(registry)       # 新发现立即对执行环境可见（:92）
3  executor.step(now, dt)               # → Task03Adapter.step（adapters/task03.py:1187）
     3a comm_relay.tick()               # relay._tick = t；策略在这一拍被采样（:1193）
                                         # 成本 tap 记计数器增量 → cost_stream[t]
     3b env.step(dt)                    # 物理时间推进到本拍末端（:1194）
     3c 构造 ProgressEvent(observed_at = env.current_time = t)
4  bus.publish(progress_events)         # orchestrator.py:95-96，同步、按订阅顺序
     ① registry.handle_progress         # 事实被提交
     ② PublishedProgressRewardCollector  # 合格完成事件当场写 task_stream[t]，并追加账本
5  clock.advance_to(now + dt)           # orchestrator.py:100，观察者之后
6  下一次采样                            # 奖励已在本拍记好
```

采集键是 `relay._tick`：这是 ROUND34 的修复本体，也是 ROUND35 逐事件实机核对（45 事件）与
ROUND36 全规模复核（2187 笔、384 集闭合）所依据的时序契约。

## 3. 评估与训练的三处差异

| 维度 | 训练（ROUND36） | 终评（本轮） |
|---|---|---|
| 权重 | 每批之后移动 | 固定 ep128，只读装载（`optimizer_restored=false`、`rng_restored=false`） |
| 决策规则 | masked 多项式采样 | 同一采样器（sampled）+ 掩码 argmax（greedy），两者分开报告，不合并 |
| 动作流种子 | `sha256("sc-action:<seed>:<episode>")` | `sha256("round24-eval:<seed>:<scenario>:<mode>")`（`v2_run.eval_action_seed`，标签未改） |
| 采集 tap | 有效（进 GAE/优势） | 仍执行，但只用于闭合与记录，不进任何梯度 |
| 场景 | 48 个训练场景 × 128 集的重排顺序 | 新登记块 1600000..1600063，每臂每场景一次 |

## 4. 一集评估的产物与闭合断言

| 产物 | 内容 | 闭合断言（写行之前） |
|---|---|---|
| `rows/v2_eval_s<seed>_final_<mode>_s<场景>_a0.json` | 观测事实 + `task_reward_events` 逐事件账本 + `checkpoint_load_identity` | `on_time == Σtask_stream == credited`；tap 字节 == 预算账本 `spent_bytes`；发射数 == transcript 的 send 条数；ACK 账本全零；步数 == 1000 |
| `logs/..._tape.jsonl` | 逐决策 features/targets/actions/actionable/log_prob/value/cost_value/step | 门禁 T4 用冻结权重重算联合对数似然，逐格 ≤ 1e-6 |
| `logs/..._rp.jsonl`、`..._ct.jsonl` | 生命周期与通信 transcript | 上面两条断言的数据源 |
| `logs/START_LEDGER_p<pid>.json` | 每 PID 的启动账本（含失败照计） | 交付对账四方闭合 |

## 5. 并行与不重复启动

- 作业清单由冻结件枚举：`index = 9 × 场景序 + 臂序`，共 576；分片规则 `index % shards == shard`。
- 每格先以 `O_CREAT|O_EXCL` 写 `claims/job_<index>.claim` 领取；领不到即跳过 ⇒ 4 进程并行、中断
  重跑、门禁格被后续分片再次遇到，都不会二次消耗。
- 真实记账仍由 `v2_common.LaunchLedger.reserve` 在构造 driver **之前**完成，硬上限 672 由它执行；
  本轮自记的 `EVALUATION_LAUNCH_BOOK.json` 与之对账（`EVALUATION_LEDGER_RECONCILIATION.json`）。
