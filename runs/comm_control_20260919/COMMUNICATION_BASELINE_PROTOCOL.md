# 通信基线实验协议（COMMUNICATION_BASELINE_PROTOCOL）

日期：2026-09-19　目录：`runs/comm_control_20260919/`
受控对象：**A 链路（搜索→救援 `CommRelay` 上行）的发送策略。**
分配侧冻结为上一轮的 C 臂（修正环境 + 绑定的联盟规划器，无释放机制），
所有条件共用同一份检测流、同一 CTDE 权重、同一场景参数。

## 一、被控通道与语义（代码级）

| 项 | 定义 | 出处 |
| --- | --- | --- |
| 消息 | `DetectionMessage`：任务位置、全部阶段需求、当前阶段、剩余需求、进度、名册、deadline、置信度、revision | `contracts.py:129` |
| 载荷构造时刻 | **发射时刻**读取登记处最新快照（不是发现时刻），因此"晚发"= 发更新的状态 | `comm.py:_message` |
| 入队（可发）时刻 | 登记处接受该事件的当步（`ready_step = tick + 1`）；由进度事件造成的视图变化在 `tick` 开头轮询发现 | `comm.py:handle_detection` / `_observation` |
| 真正到达 | `tick()` 内：先让策略选择发射，再投递 `due ≤ 当前 tick` 的包；`delay_ticks=0` ⇒ 当步生效，且早于本步的协调与分配 | `comm.py:tick`、`task03.py:1193` |
| 接收合并 | 任务必须已注入；`(sent_at, revision)` 不新于当前 belief 则整条丢弃，不做字段级合并 | `task03.py:401-409` |
| 队列合并 | 每目标每步至多一次发射；待发状态只有一份账本，旧未发内容被新快照取代 | `comm.py:_TargetLedger` |
| 终止消息 | 发射时登记处已终态 ⇒ 不发射，计入 `emissions_suppressed_terminal` | `comm.py:_observation/_emit` |
| 预算裁剪 | 超出 `sends_per_tick` 时按 `(剩余时间↑, 等待步数↓, target_id↑)` 截断，**对所有策略相同** | `comm.py:_emit_selected` |
| 丢包 | 每 (发射, 订阅机器人) 独立伯努利，seed 驱动，**不重传**；丢包判定发生在发射时刻 | `comm.py:_emit` |

## 二、策略集合与参数（运行前写定，未据结果增删）

| 条件 | 策略 | 参数 | 信息设置 | 角色 |
| --- | --- | --- | --- | --- |
| `deployed_immediate` | immediate | — | deployed（视距 0.3 + gossip 每 2 步） | 参照：上一轮冻结管线的原样 |
| `isolated_immediate` | immediate | — | isolated（`gossip_interval=100000`） | 隔离基线 |
| `isolated_periodic_2/5/20/100` | periodic | `interval=2/5/20/100` | isolated | 固定周期基线 |
| `isolated_event` | event | `progress_threshold=0.2, urgency_time=25.0` | isolated | 事件触发基线 |
| `isolated_event_maxwait` | event_maxwait | 同上 + `max_wait_steps=10, max_urgent_steps=3` | isolated | 有界静默的事件触发 |
| `stress_budget_1`、`stress_budget_1_periodic_5` | immediate / periodic(5) | `sends_per_tick=1` | isolated | 研究用压力（无真实带宽依据） |
| `stress_loss_0p7`、`stress_loss_0p7_periodic_5` | immediate / periodic(5) | `delivery_probability=0.7` | isolated | 研究用压力 |
| `stress_delay_5` | immediate | `delay_ticks=5` | isolated | 研究用压力 |
| `range_0p10_*`、`range_0p03_*` | immediate / event | `comm_range=0.10 / 0.03` | deployed | **诊断**：旁路敏感性，非主结论 |

阈值取值来源（在跑任何条件之前确定）：2 = 环境自身刷新周期；5 = `coordination/allocation_interval`；
20 ≈ `command_ttl` 的 2 倍；100 = 一个 stage 时长量级；`urgency_time=25` ≈ deadline 带宽
（`deadline_tightness` 0.5–2.0）的下段；`max_wait_steps=10` = 一个 `command_ttl`；
`max_urgent_steps=3` < 协调周期。**没有任何参数因结果而被改动或挑选。**

诚实披露：主表的 13 个条件与参数在启动前写入 `conditions.py`；随后的开发批运行之前只做过一次
**接线 smoke**（3 个 seed × 部分条件），它暴露的是测量定义问题（对已终态任务继续累计信息年龄，
导致所有条件年龄相同），修复只改 `comm_metrics.py` / 探针聚合口径，未改任何条件、参数或种子，
且修复后 12 seed × 13 条件全部重跑为同一版本。

## 三、信息与统计口径

* **种子**：`30000..30011`（12 个开发种子，检测流取自
  `runs/coalition_validation_20260918/streams/s<seed>.json`，即成员一致性轮已用作开发集的
  30000..30399 块的前 12 个；500000..500399 仍是上一轮登记的独立测试集，本轮不碰；
  600000+ 为在线系统种子）。
* **配对**：同一 seed 上所有条件重放**同一份原始 DetectionEvent**；各策略自行决定发送与到达时刻。
  禁止复用已定交付时间的消息序列——消息序列由策略产生，转录取自 `CommRelay` 自身。
* **主指标**：per-seed 按时完成目标数（`performance.on_time`，定义沿用上一轮：
  `final_progress ≥ 1 且 failure_reason ≠ deadline_expired`）与按时完成率。
* **通信指标**：`emissions`（批次）、`transmission_attempts`（=批次×订阅者，**丢包同样计入**）、`airtime_bytes`
  （规范化 JSON 载荷字节×包数，明确不含无线头部/帧开销）、批大小、队列等待与飞行步数、
  重复/过期/陈旧投递计数、刷新批次占比。
* **接收端指标**：信息年龄（`now − believed 载荷状态时刻`，仅对**仍活动**的任务计，
  终态后不计——否则所有条件同分）、阶段/剩余需求误差步数、"无人知晓的活动任务"步数、
  首次知晓时间。
* **决策代价**：分配/协调调用次数、每次调用墙钟毫秒、重规划决策数、空闲机器人步数。
* **推断**：seed 级配对差 + 百分位 bootstrap（B=10 000，RNG 20260919，先于看结果写定）。
  CI 跨过 0 ⇒ 证据不足，不等于等价。
* **零检测种子 / 技术失败**：预先声明为排除项并逐条列出（本轮 12 个种子全部检测到 ≥5 目标；
  `technical_failures = 0`；12 个种子检测到的目标数为 4–6 个，合计 68 个目标）。
* **新失败**：任何条件相对基线新增的过期/未解决目标都逐 seed 报告，不因"直接原因是环境超时"删除。

## 四、判定与停止规则（运行前写定）

1. 若某条件下"切断上行"仍能通过其他路径获得同一信息 ⇒ 该设置不可用于评估发送策略，
   只能在隔离设置下判断（`INFORMATION_BOUNDARY.md` 实测了这个前提，deployed 不满足、isolated 满足）。
2. 若把某类消息全部省略后接收 belief 与任务结局完全不变 ⇒ 判为"该维度无收益空间"，
   但必须先排除旁路与接线问题（已做：对照 408 次上行投递 vs 17.7 万次视距刷新）。
3. 若所有规则在每个 seed 上并列最优 ⇒ 判为"当前场景与当前分配策略下无可分辨差异"，
   如实报告并给出范围内原因，**不为制造优势削弱规则基线**。
4. 主表结论只用同一修正环境下的 C 臂；不与上一轮 400-seed 结果、不与旧环境结果混合。
5. 本轮在开发集基线 + 设计文档后停止，不开长训。

## 五、复跑

```bash
cd runs/comm_control_20260919
python probe_traffic.py --seeds 30000 ... 30011 --isolate --out TRAFFIC_PROBE.json
python comm_experiment.py --shard 0 --shards 4      # 断点续跑：全局 done 视图
python aggregate_baselines.py                       # COMMUNICATION_BASELINE_STATS.json
python identity_check.py                            # 与冻结提交逐记录比对（8 个 dev seed）
```

机器可读结果：`TRAFFIC_PROBE.json`、`COMMUNICATION_BASELINE_STATS.json`、`IDENTITY_CHECK.json`、
`rows/rows_shard*.jsonl`（每 (条件,seed) 一行，含全部指标与产物路径；原始 jsonl 不入库）。
