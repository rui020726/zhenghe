# 通信控制范围界定（COMMUNICATION_CONTROL_SCOPE）

日期：2026-09-19　分支：`feat/communication-control-baselines`
起点：`fix/task-membership-consistency @ fd06d45abab83e595c907e3a5d6b0d3d28554117`（远端已核对），
生产冻结版本 `a26e5b54990e2d3b2ead8c542057d6cfa2b23922`（`freeze.check()` 返回 True，
`integrated_sar` 与 `task_allocation` 与该提交内容一致）。

原始目标是**学习型协调模型优化任务信息的传递频率与更新时机**。联盟预约、执行连续性、
成员一致性环境是这条链的基础设施。本文回答一个问题：**"任务传递频率"在这套架构里到底是哪条链路，
它的实际调用链、发送条件、接收条件和现有配置是什么。** 并且明确把三个容易混淆的频率分开。

## 一、四条链路的实际调用链

### A：搜索端 → 分配端的新检测（`CommRelay` 上行链路）

```
PilocSearchAdapter.step() 发布 DetectionEvent
  → InMemoryEventBus.publish                       （event_bus.py:31，订阅顺序：registry 先、relay 后）
  → TaskRegistry.handle_detection                   （task_registry.py:81，去重/迟到/终态判定）
  → CommRelay.handle_detection                      （comm.py:232，只登记"登记处接受了这条消息"）
  ── 同一联合步内 ──
  → JointSAROrchestrator.step: task_sink.sync_tasks （orchestrator.py:92）
  → Task03Adapter.step → CommRelay.tick              （task03.py:1193 → comm.py:268）
      → 发送策略 decide → 每个订阅机器人独立抽丢包 → 入队 due = tick + delay_ticks
      → 到期包 → Task03Adapter.deliver_detection      （task03.py:381）
      → env.learn_task_view                           （old_task_env.py:308）
```

* **发送条件（本轮之前，硬编码）**：登记处接受一个 `DetectionEvent` ⇒ 立刻为每个订阅者生成一个包。
  同一 `event_id` 只发一次（`comm.py:_seen_event_ids`），被登记处拒绝的事件不发
  （`accepted_of`，`comm.py:241`）。此外没有任何发送决策：`ProgressEvent` 不产生包。
* **接收条件**：任务必须已被 `sync_tasks` 注入（否则丢弃）；按 `(sent_at, revision)` 判新，
  不新则不覆盖（`task03.py:401-409`）。丢包不重传，belief 保持原值。
* **现有配置**：`--comm-delivery-probability 1.0`、`--comm-delay-ticks 0`、`--comm-subscribers all`。
  上一轮冻结的两臂与全部压力条件都用 `1.0 / 0 / all`（`ARMS_CONFIG_TRUTH.json`、
  `STRESS_PROTOCOL.md:31`）——**这条链路的发送时机与频率此前从未被任何实验改变过。**

### B：任务状态、阶段、剩余需求的更新

`ProgressEvent → TaskRegistry.handle_progress`（`revision += 1`、`updated_at` 前进）
→ `sync_tasks` 把真值写进 `env.task_dic`（`task03.py:302-312`，只写 deadline/location/confidence/
revision/finished）。**到这里就停了：`sync_tasks` 不写任何机器人的 belief。** 之后 belief 里任务状态
的变化只来自三个地方：视距刷新（A 之外）、一跳 gossip、到场核实。也就是说
**当前架构里 B 根本没有"消息"这个对象**：没有发送方决策、没有队列、没有投递时刻，
因此也没有可测量的"传递频率"。

### C：机器人之间的信息传播（环境自带）

`TaskEnv.step:1109` 每 `gossip_interval=2` 步执行一次 `propagate_task_completion`：
第一部分对 `comm_range=0.3`（单位方格，≈18 格）内的任务调用 `refresh_task_belief` 用现场真值刷新 belief
（`old_task_env.py:487-492`）；第二部分按 tick 初快照做一跳 `_share_task_belief`
（`:497-540`）。**同一个开关同时门控视距刷新与 gossip**，因为它两半都在
`propagate_task_completion` 里。这条链路在 `task_allocation/old_task_env.py` 内部，属于本轮冻结范围。

### D：触发分配器重新决策

`JointSAROrchestrator.step`：`advice_due = state_changed or replan_requested or step % 5 == 0
or advice.valid_until < now`；`allocation_due = state_changed or replan_requested
or step % 5 == 0 or advice.replan_agents`（`orchestrator.py:114-136`），其中
`state_changed = knowledge_grew or execution_result.belief_changed_agents`
（`:108-109`）——即"某条消息终于到达"会被当作一等调度信号。

## 二、三个频率必须分开（本轮实测的口径）

| 频率 | 由什么决定 | 现有默认 | 本轮如何计 |
| --- | --- | --- | --- |
| **消息发送频率** | `CommRelay` + 发送策略 | 每条被接受的检测发一次，延迟 0 | `transport.emissions` / `packets_sent` / `payload_bytes_total` |
| **协调器调用频率** | `coordination_interval=5` 与调度信号 | 5 | `decisions.coordinator_calls` |
| **分配重规划频率** | `allocation_interval=5`、`need_decision`、`replan_agents` | 5 | `decisions.allocation_calls`、`decisions.replan_decisions` |

三者互不等价：A 的一个包可以触发一次 D 的重新决策（`belief_changed_agents`），
但把 D 调密并不会让 A 多发；反过来本轮把 A 变密（周期 2→20）时
`coordinator_calls` 与 `allocation_calls` 由同一个 5 步周期与同样的信号决定，不随之变化。
本轮**不动** B/C/D 的任何默认与实现。

## 三、第一阶段选 A：理由

1. **原始文档已经把范围写明**：`ARCHITECTURE.md:57` 说 `CommRelay` 是"搜索发现与救援 belief 之间
   唯一的通道"，`:57-63` 明确它的投递受拓扑/延迟/丢包控制，而"机器人获知任务后的后续更新仍走
   TaskEnv 自身的视距感知与一跳 gossip"。按指示"若原始文档已明确范围，优先遵循"⇒ 受控对象是 A。
2. **A 是架构支持最清楚的一条**：它有消息契约（`DetectionMessage`）、队列（`_pending`）、
   逐接收者的投递决策、去重与新旧合并规则（`deliver_detection` 的 `(sent_at, revision)`），
   因此"何时发、发几次"可以做成决策而不新造结构。
3. **B 可以沿 A 的同一契约到达**：`DetectionMessage` 本来就携带阶段、剩余需求、进度、名册、
   revision、deadline（`comm.py:_message_from_snapshot`）——它是任务视图，不只是发现通知。
   本轮的周期/事件策略在同一条链路上发送"刷新视图"，从而在没有第二套事件结构
   （`AGENTS.md` 修改规则）的前提下覆盖 B 的语义。
4. **不选 C**：它在刚冻结的 `old_task_env.py` 内部；选它等于改环境实现。C 只通过构造参数
   （`gossip_interval`）在测量隔离条件中被关到实验之外，实现未改一行。
5. **不选 D**：那是控制器/分配器调用频率，不是信息传递频率。

其余链路保持现状：B 仍无消息对象（除经 A 的刷新视图外），C 默认 2 步一跳，D 默认 5 步。

## 四、A 上"信息量"与"批次"的区分（本轮的测量约束）

一个包的载荷在**发射时刻**由登记处最新快照构造（`comm.py:_message`），所以"晚发"意味着
"发更新的东西"，而不是"发更旧的东西"或"少发字段"。队列里同一目标只保留一条待发言论
（`_TargetLedger`），旧未发消息被新状态取代——合并、旧消息替换、终态不播报这三条规则
写在传输层，**对所有策略一致生效**，不是只约束学习策略。因此：

* `emissions`（批次）下降而 `payload_bytes_total`（信息量）不降 ⇒ 只是打包，不算通信量下降；
* 两者同时下降才是真的少说；
* 若少说之后接收端 belief 与任务结局不变 ⇒ 才是"这条消息可以省"。

## 五、下一步（本轮内）

范围定了之后，先解决"暂停发送时信息是否还会到达"（`INFORMATION_BOUNDARY.md`），
再谈基线（`COMMUNICATION_BASELINE_PROTOCOL.md` / `_RESULTS.md`），最后才判断是否值得训练
（`LEARNED_COMMUNICATION_CONTROLLER_DESIGN.md`）。
