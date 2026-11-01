# ACK 反馈合同：真实、计费、可能延迟/丢失的累计确认

日期：2026-09-20。分支 `feat/comm-ack-feedback-gate`，起点 `feat/comm-policy-owned-timing@23feba8`
（只引入文档提交 `d4c73af`）。对应总纲 v1.7 §5.9/§9 与执行指令 `COMM_ACK_VALUE_NEXT_ROUND.md` §2。

本文在任何训练之前写定。它回答指令 §2 的七个问题，并用测试固定答案（§4 门禁 1–10，
`integrated_sar/tests/test_comm_ack_feedback.py`）。旧 no-ACK 路径逐字节不变：ACK 关闭时
不构造隐藏反馈状态、不改变候选顺序、不消耗任何额外 RNG。

## 〇、接缝审计（先于设计）

| 接缝 | 位置 | 现状 |
|---|---|---|
| 发送策略 | `comm.py:_emit_selected` → `policy.decide(SendObservation)` | 纯函数，只看发送端自己的账本与 `SendItem`；不看接收端 |
| 上行投递 | `comm.py:tick` → `_pending` → `_deliver` → `Task03Adapter.deliver_detection` | 每包按 (seed 驱动的) per-subscriber 独立丢包；无重传 |
| belief 接收 | `adapters/task03.py:381` `deliver_detection` | 返回 bool：belief 真的变了没有；按 `(sent_at, revision)` 拒绝更旧/同刻同版 |
| 消息身份 | `DetectionMessage.event_id` + `target_id` + `revision` + `sent_at` | 首次发射保留检测 event_id；刷新为 `refresh-<target>-<tick>` |
| 计费 | `comm_budget.py:CommBudget` + `comm.py:_emit` 的 `cost_bytes` | 载荷 × 订阅者 × 尝试（丢包照计）；额度硬顶 |
| 随机数 | `comm.py:232` `self._rng = np.random.default_rng(seed)` | 上行丢包单一流；ACK 反向信道用**同一**冻结参数独立播种 |

**结论：七项最小语义全部可实现，无需改任何一项的定义。** 唯一的实现新增在 `comm.py`
（ACK 队列、反向账本、`ack_*` 计数器、`transport_parameters()` 的 ACK 段）与一个新模块
`integrated_sar/comm_ack.py`（累计确认状态机 + 序列化 + 计数器键），加 `comm_learning.py`
的 `/2` 特征与 `joint_driver.py` 的显式开关。生产默认 `comm_ack=False`。

## 一、哪个事件产生 ACK

**接收端把一条上行包真正应用进 belief 之后**，由 `Task03Adapter` 在 `deliver_detection`
返回 True 的同一调用点产生一个累计 ACK 帧，交给本轮的 `AckChannel` 排队。仅此一个来源：

- 上行包丢失 ⇒ 接收端从未看到 ⇒ 不产生 ACK（门禁 1）；
- 送达但被拒收（重复/被取代/未知任务）⇒ `deliver_detection` 返回 False ⇒ 不产生 ACK；
  重发一个已被确认的视图不会产生新 ACK，但**可以**再次确认同一个最高视图（见三）；
- 发送端不能查询 registry truth、接收 belief 或离线 transcript 补 ACK：ACK 状态机的唯一
  写入方是"真实收到并处理"这一事件，读取方只有 `/2` 特征与规则（见七的信息边界）。

## 二、ACK 证明"包到达"还是"新视图已应用"

**新视图已应用。** ACK 载荷携带接收端已应用的最高 `(sent_at, revision,
content_fingerprint)`。`deliver_detection` 返回 False 的三种拒收（重复、被取代、未知任务）
都不改变已应用的最高视图，因此都不产生携带更高视图的 ACK。这不是"包到了"的回执，
而是"这个视图现在是这个接收端的知识"的证明——否则重发一个重复包就能伪造确认进度。

## 三、重复、乱序、合并消息如何累计确认

- **累计**：每个 (subscriber, target) 一份 `AckState`，记录最高已确认的
  `(sent_at, revision, content_fingerprint)` 三元组与确认步。新 ACK 只有在这三元组**严格
  更新**（字典序更大）时才推进 `highest`；否则记 `reaffirmed`，highest 不变。
- **重复包**：同一内容再次送达 ⇒ `deliver_detection` 拒收（重复）⇒ 不产生新 ACK；
  若协议配置为"重复也确认"，只发 reaffirm 同一最高视图，不推进状态。
- **乱序**：上行延迟为单一值（见五），同一目标的包按 `(sent_at, revision)` 排序投递；
  即使出现乱序，ACK 状态机只按三元组字典序取最大，**highest 永回退**（门禁 3）。
- **合并消息**：发送端把两条新闻装进一封信（`coalesced_changes`），接收端只应用一次、
  只确认一次；ACK 按消息身份计，不按被合并的新闻条数计。

## 四、ACK 何时进入发送端可见状态

ACK 经独立反向信道传输，到达步 = 接收端处理步 + 反向延迟。`CommRelay.tick()` 的顺序固定为：

1. 投递本步到期的上行包（`_deliver`，可能产生 ACK 帧入反向队列）；
2. 投递本步到期的 ACK 帧（推进 `AckState`，写 `ack_*` 计数器与 transcript）；
3. `_emit_selected()`：本步的发送决策——此时 `/2` 特征与规则看到的是**本步已到达的 ACK**，
   本步刚产生的 ACK 当步不可见（它还在反向信道上）。

即：发送决策永远只能引用已经完成反向传输的 ACK，在途 ACK 不可见（"包是否在途"被指令列为
禁止输入）。反向延迟与丢包与上行同参数（见五），所以 ACK 到达步严格晚于上行处理步。

## 五、ACK 的字节、延迟、丢包与预算如何计量

- **字节**：`estimate_ack_bytes()` 对 ACK 的序列化 JSON 计 UTF-8 长度，字段固定：
  `subscriber, target_id, sent_at, revision, content_fingerprint, confirmed_at, kind`。
  每个 ACK 发送**尝试**按该字节计费，丢失也计费。总成本恒等式：
  `total_bytes = uplink_attempt_bytes + ack_attempt_bytes`，三项分列闭合（门禁 6）。
- **延迟与丢包**：主设置采用与上行**同一冻结参数**：`delay_ticks`、`delivery_probability`
  各自从既有 CLI/配置读入，不做新的口头假定；反向信道用独立播种的 RNG
  （`comm_ack.py` 的 `reverse_rng(seed, episode)`，见六），因此跨策略是**同分布不同实现**，
  与上行的既有纪律一致，跨臂比较一律按场景配对解释。
- **预算**：ACK 走**独立的固定规则反向账本**（`AckLedger`：只计数、不设上限、不可为负），
  不消耗 29 300 B 任务上行额度。理由写在指令里：ACK 消耗反向额度与任务上行额度若混成一个
  动作约束，"省上行"就可以靠"随便确认"伪造。ACK 账本固定规则 = 每包按实际序列化字节全额计入，
  无容量限制、无拒付路径（反向信道不设硬顶，所以 ACK 永不因额度被丢弃；丢包只来自
  `delivery_probability`）。这样 ACK 成本可见、可复现，且不与上行策略形成隐性耦合。

## 六、不同策略如何共享可比的信道随机过程

主比较采用**事先定义的 counter-based key**，至少包含
`(episode_seed, direction, subscriber, target_id, attempt_ordinal)`：

- 上行沿用现有单一 `default_rng(seed)` 流（per-subscriber 独立抽取）；
- 反向信道 `np.random.default_rng([seed, 1, episode, subscriber, target, ordinal])` 风格的
  counter-based 派生：给定同一组 key，任意策略拿到的丢包/延迟抽取**逐位相同**，
  策略改变发送序列只改变"哪些 key 被抽取"，不改变每个 key 对应的随机数；
- 因此 ACK-on 各臂之间是**共同随机数配对**（counter-based key 一一对应）；
  与 no-ACK 历史臂比较时只能称同分布比较，不得称同一串事件。

## 七、episode reset、resume 与 checkpoint 如何恢复反馈状态

- `reset()` 清空 ACK 反向队列、`AckState` 表、反向账本与 `ack_*` 计数器，并按同一 seed
  重建反向 RNG（门禁 7 的 reset 半边）；
- checkpoint 保存/恢复：控制器权重、optimizer、数据 cursor、**策略 RNG、信道 RNG（上行与
  反向两个流）、ACK 反馈队列与确认状态**，重载后续跑与连续跑逐字节一致
  （`train_protocol` 的 `save`/`load` 增加 `ack` 段，`rng_state` 增加 reverse 流）；
- 运行签名新增：`ack_schema`、`ack_enabled`、载荷格式版本、反向信道参数（延迟/丢包/
  计数器键方案）、计费版本、observation schema（`comm-send-state/2`）。不匹配产物 fail closed。

## 八、观测扩展（`/2`），只增加发送端合法可推的量

在 19 维 `comm-send-state/1` 之上只增加四维，全部由**发送历史 + 已到达 ACK** 推出：

| 特征 | 定义 | 未知编码 |
|---|---|---|
| `confirmed_fraction` | 已配置接收端中确认当前视图的比例 | 无 ACK ⇒ `-1.0` |
| `max_ack_age_steps` | 各接收端确认状态中的最大 ACK age（步） | 从未确认 ⇒ `-1.0`（明确 unknown，不编码成"肯定没有"） |
| `max_view_gap` | 当前待发送视图与各接收端最高已确认视图的最大 revision/content gap | 无任何 ACK ⇒ `-1.0` |
| `unacked_broadcasts` | 当前视图产生以来已尝试但未全体确认的广播次数 | 无 ACK ⇒ `-1.0` |

确认状态**按 subscriber 分开维护**（一次广播有多个接收端），但策略动作仍是一次广播
SEND/HOLD，不扩展逐接收端路由。归一化常数训练前写定：`confirmed_fraction`∈[0,1] 原值；
`max_ack_age_steps` 除以 `ACK_AGE_SCALE_STEPS=100.0` 后截断到 [0,1]；`max_view_gap`
取 `revision` 差，除以 `VIEW_GAP_SCALE=5.0` 后截断到 [0,1]；`unacked_broadcasts`
除以 `UNACKED_SCALE=10.0` 后截断到 [0,1]。

**禁止输入（测试固定，不得出现在任何特征里）**：接收端实际 belief、包是否在途、未来是否
会到达、真实丢包结果、注册表状态、registry truth、未来检测/任务结局、离线 transcript。

## 九、ACK 规则对所有 ACK-on 臂完全相同

ACK 发送量依赖上行策略是**协议结果**：ACK 帧由"真实应用"事件驱动，因此发得多的臂确认也
多、ACK 字节也多——这必须如实报告（§7 的 total bytes 口径），不能通过免费或固定常数扣除
消掉。ACK 合并规则（每接收端/任务/tick 至多一个）与上行队列规则同源，对所有臂一致。

## 十、旧路径不变

`ack_enabled=False`（默认）时：不构造 `AckState` 表、不建反向 RNG、不写 ACK transcript、
`/1` 特征与旧版逐位相同、候选顺序与 RNG 消耗逐位相同。回归：
固定流上发射/投递/任务结局/配置哈希逐字节一致（门禁 5）。

## 十一、复跑

```bash
python -m unittest integrated_sar.tests.test_comm_ack_feedback -v   # 合同测试
python -m unittest discover -s integrated_sar/tests                  # 全套
python -m integrated_sar                                             # smoke
```
