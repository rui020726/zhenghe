# 信息边界（INFORMATION_BOUNDARY）

日期：2026-09-19　测量：`probe_traffic.py`（12 个开发 seed × {发送/静默} × {部署/隔离} = 48 个 episode），
统计见 `TRAFFIC_PROBE.json`。

本文回答 §3 的六个问题，并给出"暂停发送时信息是否仍会到达"的实测答案。

## 一、控制模型部署在哪个节点

**发送侧：登记处所在的基地区域（ground station）。** `CommRelay` 与 `TaskRegistry` 同侧，
它合法持有任务真值（阶段、剩余需求、进度、名册、deadline、revision）。
控制模型读不到任何机器人的位置、belief、身份，也读不到投递结果——
**这条上行链路没有 ACK**（丢包只写进离线转录，不回流发送方）。
接收侧（机器人）与执行侧闸门都不属于控制模型的可见范围。

## 二、它能读取哪些本地状态（`comm_policy.SendItem`，逐字段）

`target_id`（外部 id 字符串；策略不得据其记忆种子，见设计文档）、
`ready_step` / `age_steps`（这条未说出口的变化已等待多久）、
`pending_event`（有被接受的检测尚未播报）、`changed_view`（当前视图与上次发射是否不同）、
`first_emission`、`revision` / `revision_delta`、`stage_index` / `stage_changed`、
`progress` / `progress_delta`、`requirements_changed`、`members_changed`、
`deadline` / `remaining_time`、`sent_count` / `last_sent_step` / `since_last_emission_time`、
`has_future_requirements`。

全部是**发送节点自有账本 + 它即将发送的那份载荷**的属性。
不含：机器人位置/状态/身份、谁已知道、丢包与否、执行侧闸门内部量、登记处之外的任何"接收端真值"。

## 三、消息何时入队、何时到达、接收方如何更新

* 入队（成为"可发"）：登记处接受该事件的当步（`ready_step = tick + 1`）；
  由 `ProgressEvent` 造成的视图变化在下一步 `tick` 开头被轮询发现并起算等待。
* 到达：`tick()` 内先发射选择、再投递 `due ≤ tick` 的包；`delay_ticks=0` 时**当步**到达，
  且发生在同一步的协调与分配之前（`task03.py:1193` → `orchestrator.py:94`）。
* 更新：`Task03Adapter.deliver_detection` 要求任务已注入，并按 `(sent_at, revision)` 严格判新；
  只整体替换、不做字段级合并；`confidence`/`revision` 只随消息写入（视距刷新不改它们，
  见 `old_task_env.py:300` 的注释）。

## 四、绕过消息、直接读取任务状态的路径（实测）

| 路径 | 代码 | 是否携带"未发出的信息" | 实测份额（12 seed） |
| --- | --- | --- | --- |
| 视距感知刷新 | `old_task_env.py:487-492`（`refresh_task_belief` 用现场真值） | **是** | belief 写 177 335 次 vs 上行 408 次（435:1） |
| 一跳 gossip | `:497-540`（`_share_task_belief` 直写邻居 knowledge） | 是（仅限邻居已知） | 527 次写，2 次首次知晓 |
| 到场核实 | `:968-972`（`verify_task_on_arrival` 刷新 belief） | 是（物理在场） | 288 次写（隔离设置 596 次） |
| 登记处→执行环境真值同步 | `task03.py:293-312` `sync_tasks` | 否：只写 `task_dic` 真值，不写 belief | — |
| 执行侧安全校验 | `assignment_gate.py` 读登记处 revision/stage/名册 | 否（不携带任务状态载荷） | 拒绝→`replan_agents`，见下 |
| `belief_changed_agents` 调度信号 | `orchestrator.py:108-109` | 否：它由投递本身触发，不是替代通道 | — |

**"暂停发送 ⇒ 信息不到达"在部署配置下不成立：**

| 设置 | 首次知晓来源 | 被告知的目标数 | 任务结局 |
| --- | --- | --- | --- |
| deployed + 正常发送 | 上行 408/408 | 68/68 | 完成 68 |
| deployed + **静默上行** | 视距 388 + gossip 2 = **390/408（95.6%）** | **65/68** | 完成 64、过期 1、未处理 3 |
| isolated + 正常发送 | 上行 408/408 | 68/68 | 完成 65、进行中 2、未处理 1 |
| isolated + **静默上行** | **无（belief 写 0 次）** | **0/68** | 完成 0（68 个目标停在 detected） |

结论：**基线比较只能在隔离设置（`gossip_interval=100000`）下做**，否则测的不是通信策略而是环境自带传感。
隔离设置的定义性事实由上表最后一行证明：不发就谁都不知道。
唯一保留的非消息通道是**到场核实**——它是物理在场的结果，不是信息传递，任何通信策略都无法绕过，
也不需要绕过（隔离设置下它贡献 596 次 belief 写，全部发生在机器人已经到达该任务之后）。

## 五、执行侧校验与"拒绝反馈是否构成第二条信息通道"

执行侧闸门按现有架构保留：它读登记处真值做命令合法性校验（TTL、目的任务 revision、源任务阶段与名册）。
它的反馈只有一条：`pending_replan → advice.replan_agents → 该机器人重新决策`。
这条反馈**不携带任务状态载荷**（选择哪个任务仍来自该机器人自己的 belief），
因此不构成被延迟信息的替代来源；实测：隔离静默设置下拒绝数为 0 且没有任何 belief 写。

但它确实是**第二条通道的一种弱形式**：它把"你的命令不被接受"这一事实送回决策层，
因而把通信质量转成了重规划流量。开发集实测（同 12 seed，隔离）：

| 条件（隔离，12 seed 合计） | 重规划决策数 | 上行包数 |
| --- | --- | --- |
| `isolated_immediate`（只发一次） | 140 | 408 |
| `isolated_periodic_5` | 10 | 2 340 |
| `isolated_event` | 0 | 2 940 |
| `isolated_periodic_2` | 0 | 5 466 |
| `deployed_immediate`（真实系统） | 0 | 408 |

也就是说：**信息陈旧不会立刻改变任务结局，但会转成拒绝—重规划开销**。
本轮因此把 `replan_decisions`、`allocation_calls`、每次调用墙钟与空闲机器人步
列为固定报告的次要指标（`COMMUNICATION_BASELINE_STATS.json`），
并要求学习模型的控制代价评估必须包含这一项，不能只看消息数。

**约束：执行侧真值不得回流为通信模型观测。** `AssignmentGate` 的
`release_roster_of` / revision 读取路径与 `CommRelay` 完全无交集；策略可见字段见第二节列表，
其中没有任何来自闸门、机器人状态或投递结果的项（无 ACK）。

## 六、评估真值与在线特征的隔离

信息年龄与 belief 误差使用登记处时间线（生命周期日志 `detection`/`progress` 记录）离线计算，
在 `comm_metrics.py` / `aggregate_baselines.py` 中完成；
`CommTranscript` 与 `belief_freshness` 只在测量路径上被调用
（`freshness_of` 仅在附带转录时接线，且发生在投递**之后**），
没有任何决策读取转录或年龄。测试 `test_a_transcript_changes_nothing_about_what_is_delivered`
钉住了这一点。

## 七、对下一节的直接影响

1. 策略比较只在 `isolated` 条件下解释为"通信策略的效应"；`deployed_immediate` 作为真实系统参照。
2. 由于视距刷新在部署配置下几乎完全替代了上行（95.6%），
   **"减少通信而任务效果不变"在部署配置下是一个平凡结论**，必须在隔离配置下检验；
   诊断组 `range_0p10_*` / `range_0p03_*` 进一步说明：把感知半径收窄并不会让刷新流量变得有用，
   因为工作中的机器人本来就在任务现场（`comm_range` 同时也定义邻居图，故这不是单因子消融）。
3. 若下一轮要研究"信息更新频率的真实价值"，需要的是**限制在场刷新**的场景设定
   （例如让机器人不总是留在现场、或让阶段要求在现场不可感知），这属于场景设计，本轮不做。
