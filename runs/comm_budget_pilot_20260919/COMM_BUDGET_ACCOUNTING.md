# 通信账本（COMM_BUDGET_ACCOUNTING）

日期：2026-09-19　实现：`integrated_sar/comm.py`（计数）+ `integrated_sar/comm_budget.py`（执行器）
测试：`integrated_sar/tests/test_comm_send_policy.py::BudgetLedgerTests`

## 一、七个量，各自独立记录

| 名称 | 含义 | 代码 |
| --- | --- | --- |
| `generated_items` | 发送端"知道有话说"的次数：登记处接受的检测 + 轮询发现的视图变化 | `handle_detection` / `_observation` |
| `coalesced_changes` | 同一目标已有未发变化时又来的新变化（被合并进同一封将要写的信） | `handle_detection` |
| `emissions` / `first_emissions` / `refresh_emissions` | 一次发射 = 一个任务视图上空气一次；首次与再供分开 | `_emit` |
| `transmission_attempts` | **成本口径**：每次发射对每个订阅机器人各算一次传输尝试，**丢包同样计数** | `_emit` |
| `packets_lost` / `packets_delivered` | 尝试中信道吞掉的部分 / 真正到达的部分；`attempts = delivered + lost` | `_emit` / `tick` |
| `deliveries_applied` | 到达且**改变了 belief** 的包（接收端合并规则接受） | `tick` |
| `delivery_duplicate` / `delivery_superseded` / `delivery_unknown_task` | 被拒投递的三类原因：内容与 belief 等价、比已知更旧、任务尚未注入 | `_deliver` |
| `backlog_steps` / `unsent_change_steps` / `max_backlog_age` | 已发包的排队等待 / 未说出口信息的积分 / 最老未发变化的年龄 | `_emit` / `_note_queue_depth` |

转录（`--comm-transcript`）逐条写 `event` / `send` / `refused` / `deliver` 四类记录，
每条带 `now`（发送节点时钟读数）与 `sent_step`，因此离线可以重建任意 (机器人, 任务) 的
信息年龄序列，而不需要在线读接收端状态。

## 二、广播怎么计费，与"每个接收端各算一次交付"的区别

一次发射把**同一份载荷**送给 N 个订阅机器人，占用 N 份下行时间，因此

```
airtime_bytes = payload_bytes(视图) × N        （成本，按尝试计）
packets_delivered = 其中真正到达的包数           （效果侧计数，≤ 尝试数）
deliveries_applied = 其中改变了 belief 的包数     （≤ 到达数）
applied_bytes = payload_bytes × applied 数        （效率读数）
```

若按"每次发射"计费，把订阅集合扩大就会让 arms 看起来更便宜——那是记账假象，不是省通信。
所以广播按接收端计，而 `emissions` 作为"批次数"单独报告：批次下降而空口字节不降，只是打包。
`applied_share_of_airtime = applied_bytes / airtime_bytes` 是"说的话有多少真的被人听见并用上"。

## 三、被修正的口径：408 → 288 不是省通信

上一轮 `stress_loss_0p7`（丢包 0.3、只发一次）报"288 包 / 118 855 字节"，
基线 `isolated_immediate` 报"408 包 / 168 498 字节"，于是看起来丢包**省了 30% 通信**。

核对结果：**那 120 个差额就是被丢掉的包**，发送尝试数与基线完全相同（408），
空口开销也完全相同（168 498 字节）。修正后：

| 条件 | 旧"包"（实为投递） | 发送尝试 | 旧字节 | 空口字节 | 配对差（尝试，vs 基线） |
| --- | --- | --- | --- | --- | --- |
| `stress_loss_0p7` | 288 | 408 | 118 855 | 168 498 | **0.0，CI [0, 0]** |
| `stress_loss_0p7_periodic_5` | 1 607 | 2 310 | 697 189 | 1 003 302 | **+158.5，CI [+116.5, +211.0]** |

结论：旧口径下"丢包看起来省钱"完全是记账产物；在正确口径下丢包**只让结果变差、不让成本变低**。
上一轮所有无损条件不受影响（尝试 = 投递），两条丢包行与相关配对差已在
`comm_control_20260919/COMMUNICATION_BASELINE_RESULTS.md` 末尾"口径修订"一节就地更正，
第一遍产物保留在 `comm_control_20260919/pass1_ledger_undercount/`。

## 四、ACK、重传、gossip 的计费

* **ACK：不存在。** 上行没有反向信道，发送端永远不知道某个机器人是否收到，
  因此"再发一次"只能是盲目重传——这也是这个学习问题难的真正原因。
* **重传：按普通发射计费**（`refresh_emissions` 计入同一 `airtime_bytes`）。
  中转自己不重传，只有策略再次选中该目标才会重发；没有免费重传通道。
* **gossip 与视距刷新：不计入本账本，也不能声称免费。** 它们不是发送端决策，
  而是 `old_task_env` 自带机制，因此：
  1. 受控通道成本只统计上行；
  2. 全系统通信量必须在**保留**这两条路径的设置下另测（`pilot_env.py` 的 `information="deployed"`），
     否则会把环境送的免费信息算成策略省下的开销；
  3. 上一轮已量化它们的份额：默认设置下视距刷新写 belief 177 335 次 vs 上行 408 次，
     静默上行后仍有 95.6% 的首次知晓发生。
  本报告的所有预算收益都限定在隔离设置，不外推为"整个系统的通信成本下降"。

## 五、字节估算与局限

`estimate_message_bytes` = 载荷字段的规范化 JSON（`sort_keys`、无空白）UTF-8 长度，**逐字段计入**。
已知局限，并且都是**保守且同向**的：

* 不含无线头、帧同步、ACK/重传协议开销、加密填充 → 绝对值偏小；
* 每个常数开销加在同一封信上，因此**同一档位下各臂的相对次序不变**，
  但"25% 预算"这类相对量会随真实头部占比变化 → 预算档只用于**同设置内比较**，不当作部署带宽结论；
* 载荷含 `event_id` 字符串（刷新信为合成 id），长度差异 ≤ 20 字节/封，远小于预算档位差；
* 若将来需要真实序列化，只需替换该函数并重跑标定；预算文件带 `reference_mean_airtime_bytes`
  作为可追溯锚点。

## 六、执行器语义（供 §4/§5 使用）

`CommBudget(bytes_per_episode=B, sends_per_step=C)`：

* 初始额度 = B，**不补充**；`sends_per_step` 是每步容量（默认 None = 不限）；
* 发射前定价，付不起就不发射：变化继续排队、继续变老，`budget_refused_sends` 计数（按步）；
* 额度**永不转负**，`charge()` 越界直接抛错（有测试）；
* 紧急任务没有特权：`max_wait` 上界到期但仍付不起的发送同样被拒，
  这类"因为算术而没能说出口的紧急信息"通过 `budget_refused_sends` 可见；
* 该执行器对所有策略同一个实例、同一套参数，学习策略没有例外通道。
