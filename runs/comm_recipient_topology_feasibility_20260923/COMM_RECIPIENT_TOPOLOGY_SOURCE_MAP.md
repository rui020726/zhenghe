# 接收者与真实拓扑：源码调用链

日期：2026-09-23。分支 `analysis/comm-recipient-topology-feasibility`，起点
`efa942857f5bce1af01b84c1739c8fc9583dcca7`。协议
[`COMM_RECIPIENT_TOPOLOGY_AUDIT_PROTOCOL.md`](COMM_RECIPIENT_TOPOLOGY_AUDIT_PROTOCOL.md) §2 九问。

**方法与可复核性**：每条结论给 `file:line`。行号不是手抄的——`topology_common.source_facts()`
在 15 个命名锚点上按**当前工作树文本**定位，`test_topology_harness.py` 里有一条测试会在任何锚点
移动时失败，所以这张图自带新鲜度检查。零 episode、零 production code 改动，由
`COMM_RECIPIENT_TOPOLOGY_VERIFICATION.json` 机械核实（618 个输入文件哈希不变、
`launches_added = 0`、tripwire 日志为空、12 个生产文件与起点无 diff）。

---

## 一问：subscriber 列表在哪里创建

`integrated_sar/joint_driver.py:951-963`，在一次 episode 的配置装配阶段：

```
951  requested = parse_comm_subscribers(configuration.comm_subscribers)
952  available = tuple(str(agent_id) for agent_id in task_adapter.env.agent_dic)
953  if requested is None:
954      subscribers = available
```

`parse_comm_subscribers` 在 `joint_driver.py:637-656`：`all`/空串 ⇒ `None`（即"全部救援机器人"）；
`first:N` ⇒ `("0",...,"N-1")`；逗号 id 列表 ⇒ 去重后的元组。生产默认值
`joint_driver.py:129` `comm_subscribers: str = "all"`（CLI 同名参数默认 `all`，`joint_driver.py:505`）。

⇒ 默认路径下 subscriber 集合**就是全体救援机器人**，由 `task_adapter.env.agent_dic` 的键决定，
即本轮的 6 台（搜索机器人不在这个空间里）。指定的 id 不存在时直接
`ConfigurationError`（`joint_driver.py:956-961`），不会静默缩小集合。

## 二问：谁向 CommRelay 注册 subscriber

只有构造时一次性注入：`joint_driver.py:964-965` `comm_relay = CommRelay(subscribers=subscribers, ...)`
⇒ 形参 `integrated_sar/comm.py:218` `subscribers,` ⇒ 赋值 `comm.py:255`：

```
255  self._subscribers = tuple(str(agent_id) for agent_id in subscribers)
```

**没有任何运行期注册/退订接口。** 全仓库 `grep "_subscribers"` 在 `comm.py` 里只出现 9 次，
其中**唯一一次赋值就是 255 行**；其余全是读取（`634, 654, 828, 859, 882, 895, 924, 978, 1058`）。
`handle_detection`（`comm.py:486` 起的总线订阅）注册的是**事件**，不是接收者。

## 三问：是否所有消息都广播给同一批机器人

**是。** 发射循环 `comm.py:924`：

```
924  for agent_id in self._subscribers:
```

一条消息的接收者 = `self._subscribers` 全体，与 target、消息类别、时刻无关。

## 四问：subscriber 数量是否在 episode 中固定

**固定。** `comm.py:255` 之后无任何重写（见二问）。已存数据侧的独立验证：
四层 576 格里全部 **8 904 条 `send` 记录**的 `subscribers` 字段
**distinct = {6}**，512 个有发射的 episode 中**变化数 = 0**
（`COMM_RECIPIENT_TOPOLOGY_STORED_DATA.json`）。

## 五问：是否按 target / coalition / 任务阶段 / 位置 / belief 变化

**全部不变。** 定价路径里没有任何条件分支：

| 依据 | 结论 |
|---|---|
| target | `comm.py:882` 与 `:895` 都对所有 target 用同一个 `len(self._subscribers)` |
| coalition | `members` 是**消息载荷里的一个字段**（`comm.py:119` `list(message.members)`），只被序列化进 payload；**不参与接收者选择** |
| 任务阶段 | 阶段只影响 payload 内容（`stages`/`current_stage`，`comm.py:112-113`），不影响 fan-out |
| 位置 | `joint_driver.py:945-947` 明写"Positions play no role in it (invariant 10)" |
| belief 状态 | relay 侧读的是 registry（`snapshot_of=registry.get`，`joint_driver.py:967`），不是接收者 belief；接收者 belief 不回馈到 fan-out |

⇒ **`members` 会变（见下），但它不进接收者集合。当前系统里唯一"像 recipient set"的运行期字段，
恰好是定价与路由都不用的那个。**

## 六问：sender 在动作前是否知道 subscriber 集合与报价

部分知道，且**结构上不可能按候选区分**：

- `integrated_sar/comm_policy.py:97-112` `SendObservation` 只有一个标量
  `subscribers: int = 1`（112 行），挂在**整个 observation** 上，不是 per-item；
- `comm.py:859` 把它填成 `subscribers=len(self._subscribers)`；
- 每个候选的成本在 `comm_policy.py:52-62` 的 `SendItem.attempt_charge_bytes /
  send_executable / send_block_reason` 里，由 `comm.py:878-883 _attempt_cost` **走与 `_emit`
  完全同一条代码路径**算出（注释原话："an approximation would let a controller believe a
  send was affordable"）；
- 但 `comm_policy.py:48-50` 明确："They are metadata for the executor and for the training tape,
  **never neural-network inputs**"。

⇒ 发送前可知道"本条候选现在多少钱、能不能付"；**不能**知道"这条消息会发给谁"，因为
observation 里根本没有 per-candidate 的接收者概念，只有一个全局标量。

## 七问：`attempt_charge_bytes` 是否已包含真实 fan-out

**已包含。** `comm_policy.py:52-55` 的定义就是"payload estimate ... multiplied by the subscriber
fan-out"，实现走 `comm.py:882`：

```
882  charge = int(estimate_message_bytes(message) * len(self._subscribers))
```

与发射端 `comm.py:895` `cost_bytes = payload_bytes * len(self._subscribers)` 同式，
`estimate_message_bytes`（`comm.py:97-123`）是载荷字段的规范化 JSON UTF-8 长度。

**恒等式实测闭合**：8 904 条 `send` 记录逐条核
`cost_bytes == payload_bytes × subscribers`，**违反数 0**；且
`cost_range == payload_range × 6` 逐层成立（如 g12：payload 极差 128 B × 6 = 768 B = 报价极差）。

## 八问：ACK、gossip、直接观测与通信订阅的关系

| 路径 | 是否写 `known_tasks` | 受哪个开关控制 |
|---|---|---|
| 上行通信（relay → `deliver_detection` → `learn_task_view`） | **是** | `--comm-subscribers` 决定谁能收到 |
| 视距刷新 `refresh_task_belief`（`old_task_env.py:481-492`） | 是 | `propagate_task_completion` |
| 一跳 gossip（`old_task_env.py:497-540` → `_share_task_belief:413`） | 是 | 同上 |
| ACK 反向信道 | **否**，只写 ACK 账本 | `--comm-ack`（本轮关闭） |
| 环境真值 `task_dic` | **否** | 见九问 |

**关键耦合**：视距刷新与一跳 gossip 在**同一个函数**里，被**同一个门**控制——
`old_task_env.py:1109` `if self._step_counter % self.gossip_interval == 0: self.propagate_task_completion()`。
本轮冻结的 isolated 配置 `gossip_interval > 1000 步` ⇒ **传感与旁路一起被关掉**，
于是运行期新发现**只剩"通信送达"这一条路**。（这两条路径不计入上行账本，
`COMM_BUDGET_ACCOUNTING.md` §四已声明，因此也不能声称"关掉后信息仍免费到达"。）

`task03.py:377-379` 把这件事写得很直白：

```
377  # Deliberately no notification here: detections reach robot beliefs only
378  # through the attached comm relay (topology, delay, loss, seed) and the
379  # environment's own sensing and gossip.
```

## 九问：分配器与 coalition coordinator 是否隐含依赖"所有人都收到所有目标"

**不依赖"所有人都收到"，但强依赖"要能行动就必须先收到"。** 这条区别决定了拓扑轴的性质。

1. **候选动作来自该机器人自己的 belief**，不是全局注册表：
   `old_task_env.py:902` `pending_tasks = knowledge["known_tasks"] - knowledge["known_finished"]`
   ⇒ `:931` `action_map.append(("task", int(tid)))`。
   硬掩码同源：`old_task_env.py:852-855`（注释："不变量 9：mask 只能来自该机器人自己的 belief"）。
2. **coalition 成员资格同样要求该机器人自己持有该 belief**：
   `integrated_sar/assembly.py:152` "a robot whose beliefs do not contain the task is not a
   candidate"，实现 `assembly.py:166-168`：
   ```
   166  own = next((b for b in agent.task_beliefs if b.task_id == belief.task_id), None)
   167  if own is None or own.finished:
   168      continue
   ```
   coordinator 侧同构：`integrated_sar/coordinator.py:672-681`
   `if known is not None and task_id not in known: continue`。
3. **reset 时刻是唯一的"人人全知"入口**：`old_task_env.py:283-286` `init_agent_memory` 把
   `known_tasks` 初始化成 `set(self.task_dic.keys())`。它在整个代码里只被调用三次，
   每次都对应一次 episode/环境重建：`task03.py:245`、`old_task_env.py:120`、
   以及 `old_task_env.py:447`（位于 `def reset()`（`:441`）体内）。
   运行期新目标**没有**这个待遇。
   coordinator 侧的同一门：`coordinator.py:124`、`:292`、`:679` 三处
   `if known is not None and task_id not in known: continue`。

### 因此：限制接收者 = 限制合法动作集合，不是限制信息速率

把三条合起来：

> **一条运行期新发现如果没有送达某台机器人 ⇒ 它不进该机的 `known_tasks` ⇒ 不进 `action_map`
> ⇒ 掩码里根本没有这一维 ⇒ 该机在整个 episode 内结构性地无法被派往该目标。**
> 而在 isolated 配置下（第八问），除通信外没有其他补救路径。

⇒ "发送给谁"在这个系统里**不是一个成本旋钮，而是一个可行性闸门**。
它改变的是任务分配问题的**动作空间本身**，而不是信息的时效或成本。
这正是本轮 R2 第 3 条判据（"不破坏任务发现与分配的基本语义"）失败的原因，
也是"不能把 recipient selection 与 send timing 混成一个动作空间"这条禁令的实证依据——
两者不是同一类东西：`send_once` 的 43–73 个未报出目标是**成本约束下的损失**，
而一个限制性的接收者规则会把它变成**定义上的不可达**。

---

## 附：五种候选接收者规则的源码依据（详表见 RESULTS §R2）

| 候选 | 能否从动作前状态机械构造 | 依据 |
|---|---|---|
| 1 全局广播 | 能（现状） | `comm.py:924`，fan-out 恒为 6 |
| 2 当前 coalition | **能**（`members` 在发送时刻已记录） | `comm.py:119`；实测 g12 首报 `members` **1107/1107 全为空** |
| 3 能力匹配且 deadline 可达 | **不能** | 需要他机位置/日程；`joint_driver.py:945-947` 位置被显式排除 |
| 4 邻域通信 | **不能** | 转录与 observation 均无 per-机器人位置/`comm_range` 视图 |
| 5 coordinator-only | **不能**（不是路由规则，是架构变更） | subscriber 空间 `{0..5}` 全是救援机器人，系统中不存在"协调器 agent id"这一接收者 |

候选 2 是唯一可以离线投影的真实规则，其代价与后果见
`COMM_RECIPIENT_TOPOLOGY_PROJECTION.json`（标签 `offline_recipient_cost_projection`）。
