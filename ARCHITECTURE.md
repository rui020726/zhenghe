# 系统架构

## 1. 总体结构

系统采用事件驱动的快通路和较慢的协调优化通路：

```mermaid
flowchart LR
    W[统一场景与全局时钟]
    S[PILOC 搜索]
    B[(事件总线)]
    R[任务注册表]
    M[通信中转 CommRelay]
    C[协调优化器]
    A[CTDE 分配器]
    X[导航与任务执行]

    W --> S
    W --> X
    S -->|DetectionEvent| B
    X -->|ProgressEvent| B
    B --> R
    B --> M
    M -->|投递 DetectionMessage| X
    R -->|TaskSnapshot| A
    X -->|AgentSnapshot + task belief| C
    C -->|task/pair bias + replan| A
    A -->|AssignmentCommand| X
    C -.后续扩展：搜索热点.-> S
```

快通路保证发现后立即进入任务池；慢通路计算优先级、匹配度和重规划建议，不能阻塞救援。
发现消息只有经通信中转写进机器人 belief 之后才可能被分配。协调优化器的输入是
`AgentSnapshot` 里的 belief，不是注册表：注册表只把 `TaskSnapshot` 供给分配器（用于闸门
校验与执行注入），不给协调器供数，避免在线决策绕过通信链读真值。

## 2. 模块边界

### PILOC 搜索模块

输入局部障碍/探索地图和信息素，输出搜索动作、覆盖态势和 `DetectionEvent`。
它不创建 task-03 内部字典，也不决定救援机器人分配。

### 任务注册表

注册表执行目标去重、版本控制和状态机：

```text
HIDDEN → DETECTED → VERIFIED → ASSIGNED → EN_ROUTE
       → IN_PROGRESS → PARTIAL → COMPLETED
```

异常分支包括 `REPLAN_REQUIRED`、`EXPIRED`、`LOST` 和 `FAILED`。

### 通信中转

`CommRelay` 是搜索发现与救援 belief 之间唯一的**消息**通道。事件总线把每个
`DetectionEvent` 扇出一次，中转只登记注册表已接受的事件（被拒绝的事件不是新闻），
随后**每一步由一个发送策略决定哪些任务视图上空气**（`--comm-policy`，默认 `immediate` =
每条被接受的检测发一次，即历史行为；另有 `periodic`/`event`/`event_maxwait`/`event_priority`
/`hold_to_bound`/`random_send`，以及权重另存的 `learned`），
再按显式拓扑（`--comm-subscribers`：默认全救援队，也可指定 `first:N` 或救援机器人
id 列表）、每包延迟和丢包率（由 seed 驱动）投递；包丢了就丢了，belief 保持原值。

与策略无关、对所有策略一致的传输规则：载荷在**发射时刻**按注册表最新快照构造，
因此被推迟的变化是被取代而不是被迟到地重发；每目标每步至多一次发射；发射预算超出
`--comm-sends-per-tick` 时按统一顺序（剩余时间、等待步数、目标 id）裁剪；注册表已终态的
任务不再播报；`--comm-max-wait-steps`（紧急时收紧到 `--comm-urgent-wait-steps`）规定一条变化
最长能沉默多久，到点由传输层代发、同样计费、付不起同样被拒——它是**下界**而不是某个策略的
私有规则，所以任何臂都不能靠"什么都不发"逃开评价。中转不读任何人的位置，也不读任何机器人的
belief 或投递结果（上行无 ACK），所以搜索队与救援队仍是两组独立实体，救援集结点是否与搜索
起点重合不影响闭环能否启动。

每条发射都带一个互斥主原因：`policy`（发送端自己请求）、`forced_maxwait`（上界到点，传输层
代发）、`other`（预留给未来的强制协议事件）。发射数、尝试数、字节数与被拒次数都按原因分列并
要求闭合。这不是记账洁癖，而是这条研究线能否被证伪的关键：一个只学会"闭嘴"的策略，其全部
上行都是上界替它发的，若不标注原因就是"省了 60% 通信的学习成果"，标注之后才知道它是
`hold_to_bound` 这条规则本身（`--comm-policy hold_to_bound`，与 `random_send` 一起构成本轮
新增的两个地板臂）。`--comm-budget-bytes` 是同一执行器给的每 episode 字节额度，规则臂与学习臂
共用，见 `integrated_sar/comm_budget.py`。

三种"同一件事做两次"的口径必须分开，否则账本会被读反：`coalesced_changes` 是**生产侧**的合并
（两条新闻装进一封信），`redundant_emissions` 是**发送侧**的重复（一封信的内容与上一封逐字段相同，
即 `SendItem.changed_view` 为假），`delivery_duplicate` 则是**接收侧**某个机器人拒收它已经相信的包。
一条被轮询发现的变化若撞上一件仍在排队的旧变化，现在同时计入 `generated_items` 与
`coalesced_changes`——它确实既是一条新闻、又是一次合并——并且与事件路径共用 `ready_view`，
所以同一条新闻被事件和轮询各看一次时只计一遍。恒等式因此是
`generated_items = (emissions - redundant_emissions) + coalesced_changes + 终态抑制 + 终局未说`，
四项各归其位、不重复计数。回归见 `integrated_sar/tests/test_comm_send_policy.py` 的
`GenerationAccountingTests`，机制实测见 `runs/comm_tight_budget_20260919/` 的
`ledger_gap_reproduction.py` 与 `ledger_fix_invariance.py`（后者核对固定 checkpoint 在记账修正前后
的发射、扣费与任务结局逐值不变）。

收窄拓扑或压低发送频率都只限制**这条消息通道**——机器人仍可能经视距感知或一跳 gossip
获知同一信息（那两条信道由 `old_task_env` 自己建模，实测份额见
`runs/comm_control_20260919/INFORMATION_BOUNDARY.md`：默认传感半径下 95.6% 的"首次知晓"
在没有上行时也成立）。因此评估发送策略必须在关掉这两条旁路的隔离配置下进行
（`--gossip-interval` 大于步数；该开关同时门控视距刷新与 gossip，因为两者都在
`propagate_task_completion` 里）。`--comm-transcript` 把接受/发射/投递逐条写盘，
只用于离线测量，不被任何决策回读。

### 协调与执行的 belief 边界

在线协调器只按每台机器人自己的 belief 打分：`AgentSnapshot.task_beliefs` 携带该机器人
收到的任务位置、剩余需求、截止时间、阶段、进度、复杂度、置信度与登记版本号，
`HeuristicCoordinator` 的输出全部由它推出，绝不读注册表快照——注册表只负责闸门与
生命周期管理，它关闭一个任务并不是任何机器人观察到的事实。任务优先级按机器人分开
发布（`agent_task_bias`）而不是放进 `task_bias`：后者是会被所有机器人共享的池级修正，
任何由单台机器人 belief 推出的数值放进去，都等于把该机器人的知识广播给它从未收到过
消息的机器人。执行侧同理：`apply_action` 用 believed 位置作为导航目标、believed 状态
作为预期状态，任务真值只在到场核实（`verify_task_on_arrival`）时使用；在途重规划也只看
belief，真值显示完成而消息未达时机器人继续前往。分配器只在给命令盖上注册表 revision
时读快照，因为 `AssignmentGate` 需要用它拒绝过期命令，选哪个任务仍来自该机器人自己的
belief。

### 协调优化器

输入搜索覆盖/陈旧度、任务阶段/截止时间、机器人能力/位置/工作状态和执行失败；输出：

- `task_bias[task]`：池级任务优先级修正，所有机器人共享，因此只允许放共同知识；
- `agent_task_bias[agent, task]`：按机器人分开的任务优先级修正，由该机器人的 belief 推出；
- `pair_bias[agent, task]`：机器人—任务匹配修正；
- `reservations[agent, task]`：短期团队预约（默认空），指明该机器人覆盖该任务的哪个
  能力槽位、按该机器人 belief 的任务 revision、有效期与有界分数。pair bias 只能表达
  “这台机器人与这个任务有多匹配”，同一槽位的两台机器人会得到相同分数；预约把槽位写进
  契约，占用因此互斥，计划选中的机器人就是实际被派出的机器人。预约与 pair bias 走同一条
  有界残差通道，因此不变量 5 的形式不变，硬 mask 仍在最后应用（不变量 6）。预约键是
  (agent, task)，只对持约的机器人可见，不会像池级 `task_bias` 那样把一台机器人的知识
  广播给没收到消息的机器人（不变量 9）；发约方必须把自己的 `valid_until` 压到最早预约
  到期之前，过期预约才不可能通过缓存 advice 生效；
- `replan_agents`：需要重新决策的机器人；
- 后续可选搜索热点。

规则协调器（`HeuristicCoordinator`）与联盟打分协调器（`CoalitionAwareCoordinator`）都
不产出预约，默认配置下 `reservations` 为空，残差与旧行为逐位一致。联盟规划协调器
（`CoalitionPlannerCoordinator`，`--coordinator coalition_planner`）在空闲机器人与任务
开放槽位之间做确定性的贪心 set-cover，考虑截止时间、行进代价、能力互补与冗余派遣，输出
本轮团队预约。`CoalitionReservation` 带有身份（`commitment_id`、`created_at`）：规划器每轮
重建计划，但对同一 (agent, task, slot) 的再次发布视为**同一承诺的续订**——身份保持不变，
只有有效期、belief revision 与分数随本轮更新；只有槽位改变（真正的重规划）才产生新 id。
学习型协调器必须保持相同接口，并对输出进行有界限制。

### 承诺的执行侧约束（`--sticky-commitments`，默认关闭）

`Task03Adapter` 在决策时刻对持约机器人做六项有效性检查（belief 中任务仍活动、未过期、
槽位仍需、hard mask 合法、不在 `replan_agents`、可继续执行），全部通过则该承诺对分配器
有约束力：机器人已在承诺任务上时不再重复派令；空闲时保持承诺任务；分配器自身的原始选择
与承诺不同时，以承诺为准并在 trace 中记录 `raw_choice` / `forced_choice` /
`post_mask_choice` / `actual_assignment`。任一条件失败则按首个失败原因释放并正常重规划。
硬 mask 仍然最后应用（不变量 6），协调器依旧只提供建议（不变量 5）。

### CTDE 分配器

当前以 `task_allocation/old_*` 为主线。Actor 使用本地 belief 和协调消息；Critic 在训练时
使用全局状态。最终动作 logits 加入协调残差后再应用硬 mask。

### 执行模块

接收 `AssignmentCommand`，执行移动、到达、多阶段处理，发布 `ProgressEvent`。
进度定义为：

```text
(已完成阶段数 + 当前阶段完成比例) / 总阶段数
```

同时必须返回当前阶段剩余能力向量。

## 3. 核心数据契约

| 契约 | 生产者 | 消费者 | 关键字段 |
| --- | --- | --- | --- |
| `DetectionEvent` | PILOC adapter | Registry/Coordinator | target ID、位置、时间、置信度、目标属性 |
| `DetectionMessage` | CommRelay | 机器人 belief（经 Task03 adapter） | 发送时快照：位置、阶段、需求、进度、置信度、revision、截止时间 |
| `TaskSnapshot` | Registry | Coordinator/Allocator/Executor | 阶段、剩余需求、截止时间、revision |
| `AgentSnapshot` / `AgentTaskBelief` | Task03 adapter | Coordinator/Allocator | 该机器人收到的任务视图（位置/需求/阶段/进度/置信度/revision） |
| `CoordinationAdvice` | Coordinator | Allocator | 池级 task bias、按机器人的 agent task bias、pair bias、团队预约 reservations、replan、有效期 |
| `AssignmentCommand` | Allocator | Executor | agent/task、task revision、TTL |
| `ProgressEvent` | Executor | Registry/Coordinator | 阶段、剩余需求、进度、失败原因 |

实现定义位于 `integrated_sar/contracts.py`。

## 4. 单步运行顺序

`JointSAROrchestrator.step()` 的顺序固定为：

1. 搜索环境推进并发布新发现。
2. 注册表幂等接收发现并创建/更新任务。
3. 将最新任务快照同步到 task-03 环境。
4. 救援执行环境推进：先让上行按发送策略发射、再投递到期的通信消息（belief 变化被报告为调度信号），
   再推进模拟器并发布进度。
5. 注册表接收进度，协调器按周期或事件（新发现、进度、belief 变化、重规划请求）
   重新计算 advice。
6. 分配器在新任务、进度变化、belief 变化、超时或周期触发时生成命令。
7. 执行器应用命令；机器人从下一物理步开始移动。

## 5. 世界与坐标

PILOC 使用二维栅格 `(row, col)`，task-03 使用归一化连续坐标 `(x, y)`：

```text
x = col / (width - 1)
y = row / (height - 1)
```

正式联合环境还需要统一 `Scenario`，同时产生障碍图、目标隐藏属性、搜索机器人初始状态和
救援机器人能力。任务分配中的距离最终应改为同一障碍图上的最短路径代价。

## 6. 训练阶段

1. **闭环验证**：预训练策略或规则策略 + `HeuristicCoordinator`。
2. **分配适配**：冻结 PILOC，用真实发现事件训练 CTDE 分配器。
3. **协调器训练**：冻结两端策略，训练有界残差协调器。
4. **交替微调**：按 rollout 交替更新搜索、协调、分配策略。

不建议从随机初始化同时联合训练三个网络，因为环境非平稳性和信用分配难度过高。

建议全局目标包含及时完成、阶段进度、有效发现、行进代价、超期、漏检、重复分配、
重规划和通信代价。使用 `source_event_id` 将搜索发现与最终救援结果关联，以提供延迟信用。

## 7. 部署与信息约束

环境维护 `truth state`，Actor/在线协调器维护 `belief state`。严格 CTDE 实验中，
在线模块只能使用经过通信链（上行中转、视距感知、gossip）到达的信息：
观测特征、硬 mask、协调器打分、导航目标与在途重规划全部读 belief 副本，
断联期间真值推进不改变机器人的决策输入与行动目标。机器人是否听说某任务决定它
能否选择该任务；belief 中与决策相关的任一字段变化都会立即成为调度信号。
全局真值只允许模拟器、训练 Critic 和统计指标使用。
