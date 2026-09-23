# 接收者与真实拓扑可行性审计：实测与逐门判定

日期：2026-09-23。分支 `analysis/comm-recipient-topology-feasibility`，起点
`efa942857f5bce1af01b84c1739c8fc9583dcca7`（已在线核对为远端 HEAD）。协议
[`COMM_RECIPIENT_TOPOLOGY_AUDIT_PROTOCOL.md`](COMM_RECIPIENT_TOPOLOGY_AUDIT_PROTOCOL.md)，
调用链 [`COMM_RECIPIENT_TOPOLOGY_SOURCE_MAP.md`](COMM_RECIPIENT_TOPOLOGY_SOURCE_MAP.md)，
判定 [`COMM_RECIPIENT_TOPOLOGY_DECISION.md`](COMM_RECIPIENT_TOPOLOGY_DECISION.md)。总纲 v3.7。

**本轮 0 新 episode、0 新流、0 训练、0 `optimizer.step`、0 checkpoint 加载、0 启动；
未修改任何 production code。** 机械核实见 `COMM_RECIPIENT_TOPOLOGY_VERIFICATION.json`：
618 个输入文件 SHA256 全部不变、账本 `starts` 恒为 666、tripwire 调用日志为空、
12 个被引用的生产文件与起点无 diff、封存块 `800000..800399` 盘上 artefact 仍为 0。

## 一、一句话答案

> **`message_payload_bytes × subscriber_count` 里的 `subscriber_count` 在当前系统里是一个
> 逐 episode 恒为 6 的常数：它不随目标、机器人、消息类别、任务阶段或时刻变化，
> 也不在代码里有任何变化它的通道。** 因此"多订阅者 / 真实拓扑"在当前代码中**不是一条轴，
> 而是预算档位的另一个名字**；而唯一能从真实系统语义构造的接收者规则（发给当前 coalition）
> 会把**所有新目标的第一条报告**的接收者集合变成空集——它不是让报价变贵，而是让新闻永远发不出去。

## 二、门禁判定

| 门 | 判定 | 决定性证据 |
|---|---|---|
| **R0 语义可信** | **过** | 调用链 15 个锚点全部在当前代码定位成功（`test_topology_harness` 有一条测试专门盯这个）；定价恒等式 8 904/8 904 闭合；未读任何未来结局；0 episode / 0 启动 / 输入未变；production code 无 diff |
| **R1 当前拓扑已异质** | **不过** | 全部四层 distinct subscriber count = **{6}**；512 个有发射的 episode 中变化的 **0 个**；按消息类别（首报 / 刷新 / retry）分组，distinct 也都是 **{6}** ⇒ 结论名 `global_fanout_is_only_a_scalar` |
| **R2 存在合法 recipient set** | **不过** | 五种候选无一同时满足四条要求；唯一可构造的候选 2 破坏任务发现语义（§四） |
| **R3 发送前可观测** | **不过** | `SendObservation.subscribers` 是**一个全局标量**（`comm_policy.py:112`），per-candidate 接收者概念在契约里不存在；`members` 只在消息载荷里，`SendItem` 只有 `members_changed: bool`（`:88`），没有成员集合 |
| **R4 离线报价形成有意义选择** | **不过（且前提已失效）** | coalition 规则下 g12 的 174 个"同一步两个新目标首报"事件**全部被定价为 0**，`出现两个首报价格不同的流数 = 0/16`；广播规则下 16/16 流虽有非零差，但**中位差 = 6 B**（一个包的 0.23 %）⇒ 差异来自序列化字段长度，不是接收者语义 |

⇒ 落入协议 §6 **分支 1**：**R1 不过，且没有合法候选通过 R2 ⇒ 关闭多订阅者 / 拓扑轴。**

## 三、R1 的实测细节：fan-out 是标量，而且可证明只能是标量

### 3.1 已存数据侧

四层 576 格、**8 904 条 `send` 记录**：

| 层 | send 记录 | subscribers distinct | 有变化的 episode | 定价恒等式违反 |
|---|---:|---|---:|---:|
| g6 | 1 792 | {6} | 0 / 128 | 0 |
| g8 | 2 070 | {6} | 0 / 128 | 0 |
| g10 | 2 425 | {6} | 0 / 128 | 0 |
| g12 | 2 617 | {6} | 0 / 128 | 0 |

按消息类别分组同样恒定（每组的 distinct 都是 `{6}`）：`first_report`、`refresh`、
`retry(revision>1)`、`first_attempt(revision==1)`。
（retry 确实存在：各层 1 388–1 885 条 `revision > 1` 的发射。）

### 3.2 费用分解：报价的差异 100 % 来自 payload

`cost == payload × subscribers` 逐条闭合后，做两种口径的分解：

| 层 | payload 极差 | 报价极差 | 报价极差 / 6 | log 费用方差中 fan-out 占比 |
|---|---:|---:|---:|---:|
| g6 | 126 B | 756 B | 126 B ✓ | **0.0** |
| g8 | 112 B | 672 B | 112 B ✓ | **0.0** |
| g10 | 125 B | 750 B | 125 B ✓ | **0.0** |
| g12 | 128 B | 768 B | 128 B ✓ | **0.0** |

`cost_range / fanout == payload_range` **逐层精确成立**，`share_of_log_cost_variance_from_fanout`
= 0.0、`from_payload` = 1.0。上一轮报的"g12 发包口径极差 768 B"与本轮 payload 128 B × 6 完全对上。

### 3.3 标量不变性：不是"没测到变化"，而是"变了也不改变结构"

`COMM_RECIPIENT_TOPOLOGY_SCALAR_PROOF.json`，在 375 个存证的逐步候选视图上，
把 fan-out 乘上 s ∈ {1/6, 1/3, 1, 2, 3, 6}：

1. **窗口/包长比与尺度无关**：`window_as_fraction_of_one_packet` 在**全部六个尺度上都等于
   0.190648**（比值里不含 k）。⇒ 加机器人不会让"必须命中窗口"这件事变容易：
   窗口和跳步同时被放大。
2. **"加订阅者" ≡ "降预算档"**：对每一步检查
   `classify(charges×s, credit) == classify(charges, credit/s)`，
   **375 步 × 5 个非 1 尺度 = 1 875 次比较，不一致 0 次**。
   ⇒ 在预算不变的前提下增加 fan-out，与把预算档位调低是同一个实验——
   而预算档位本项目已经扫过（75 % / 25 %），25 % 档正是当前这四层用的档位。

这两条把 R1 的失败从"当前数据里没变化"加强成"**这个量在方程里只有一个自由度，
而那个自由度已经被预算档位占用**"。

### 3.4 窗口本来有多窄

逐步候选集（375 步）的**同一步内**报价极差：中位 510 B，折成 payload **中位 85 B**（min 34、max 118）。
一个包的 payload 中位约 434 B。⇒ 同一步里两个候选的价格差别通常是**一个包的百分之十几**，
而需要命中它的信用跳步是**整个一包**。

（与上一轮 5.8 % 的关系：那是**断供那一刻**那一步的带宽 144 B / 2 478 B——那里候选都是
同质刷新，所以窗口特别窄；本轮 19.1 % 是在**已经落进窗口**的步上量的。
两者是不同的样本框，不是矛盾，也不互相替换。）

## 四、五种候选接收者规则：逐条判定

判据（协议 §4 R2）：① 动作前可确定；② 不依赖未来结果；③ 不破坏任务发现与分配的基本语义；
④ 不让接收者选择依赖待学习策略自身的历史结果。

### 候选 1 全局广播（现状）
① 是 ② 是 ③ 是 ④ 是 —— **但它就是当前系统**，fan-out 恒为 6，
不产生任何跨消息的价格异质（§三）。作为"新轴"它**平凡失败**：它没有引入选择。

### 候选 2 只发给当前 coalition（唯一可从存证构造的规则）
接收者 = 消息自带的 `members`（发送时刻已记录在转录里，非事后字段）。

实测（g12，2 617 条发射）：

| 指标 | 广播 | coalition |
|---|---:|---:|
| fan-out 中位 / 最大 | 6 / 6 | 1 / 6 |
| **首报**的 fan-out distinct | {6} | **{0}** |
| **刷新**的 fan-out distinct | {6} | {0,1,2,3,4,5,6}，中位 2 |
| 接收者为空的消息占比 | 0 % | **44.5 %** |
| 同一步内价格的相对极差（中位） | 0.0713 | 0.7181 |

① 是（`members` 在发送时已确定）② 是（不是事后结局字段）
③ **否** ④ **否**。

**③ 为什么否（这是本轮最硬的一条结论）**：coalition 是对一个目标**已经承诺**的机器人集合，
而"谁已经承诺"这件事本身就是**先前通信送达的结果**（`assembly.py:166-168` 要求机器人自己持有该
belief 才能成为成员）。一个**刚被发现、还没人知道**的目标，其 coalition 按定义为空——
实测就是 1107/1107 条首报的 `members` 全是 `[]`。所以"只发给 coalition"对首报等于
"发给零个人"。再叠加 SOURCE_MAP 第九问那条链
（没收到 ⇒ 不进 `known_tasks` ⇒ 不进 `action_map` ⇒ 该机结构性不可派往该目标），
这条规则会把新目标**永久孤儿的化**：它不是提高成本，而是取消发现本身。

**④ 为什么否**：接收者集合由"谁已经被分配到这个目标"决定，而那又是策略先前发送的产物
⇒ 待学习策略的收件人由它自己的历史结果决定，是自证闭环。

**它产生的异质在错的那一侧**：刷新（对已知目标的续报）价格异质度提高 10 倍，
而首报（唯一携带新信息的一类）价格恒为 0、即恒不送达。
`first_report_choice`：174 个"同一步两个新目标"事件在 coalition 规则下**全部定价为 0**，
`出现两个首报价格不同的流数 = 0/16`。这正是 R4 第 4 条明令排除的"只有 refresh 之间的选择"。

### 候选 3 只发给能力匹配且能在 deadline 前到达的机器人
① **否**：需要他机的位置、日程与能力余量。`joint_driver.py:945-947` 显式声明
"Positions play no role in it (invariant 10)"，AGENTS.md 不变量 10 规定搜索机与救援机
"共享场景和时钟但不共享位置状态"；`SendItem`（`comm_policy.py:68-93`）没有任何他机位置字段。
要让它可确定，必须新增观测维度并改通信契约 —— 那是**实现新协议**，本轮明令禁止
（`不启用…；不改 production code`）。② 边界：deadline 可达性预测用到未来。

### 候选 4 邻域通信（只发给当前通信范围内的机器人）
① **否**，理由同候选 3，且更直接：转录与 observation 里**根本没有** per-机器人位置或
`comm_range` 视图。硬要做只能靠人工指定半径 ⇒ 属于"随机/人工制造拓扑"，协议 §5 禁止。

### 候选 5 只发给 coordinator
① **否**：subscriber 空间 `{0..5}` 全部是救援机器人（`joint_driver.py:952` 取
`env.agent_dic` 的键），系统里**不存在**"协调器 agent"这一可寻址接收者。
协调器（`coordinator.py`）是同进程内的集中式打分模块，不是网络端点。
要走这条路必须把架构从"分布式分配 + 协调器偏置"改成"集中式接收-再分发"，
那是**架构变更**而不是路由规则，且会推翻 AGENTS.md 主线第 5 条
（"协调器优化分配 logits，但不替代分配器"）。

**小结**：五种候选里，一种就是现状（无异质），一种在语义上自毁新信息通路，
三种在当前代码里不可确定、要实现必须先改协议与观测——而那已经不是本轮的问题了。

## 五、R3 的契约层面证据（即使换了规则也还差一层）

`SendObservation`（`comm_policy.py:97-112`）能表达的全部预算/接收者信息是：
`remaining_budget_bytes`、`budget_bytes_per_episode`、`capacity_remaining`、`backlog_targets`、
**`subscribers: int`**。`subscribers` 挂在 observation 上、**不是 per-item**，
所以"这一步这个候选会发给谁"在当前契约里**无法表达**，无论路由规则多聪明。
`SendItem` 侧只有 `attempt_charge_bytes / send_executable / send_block_reason`
（`comm_policy.py:52-62`），且同一段注释明写它们是
"metadata for the executor and for the training tape, **never neural-network inputs**"。

⇒ 结论：R3 不是"数据没记全"，是**契约里缺少这个概念**。
补它属于协议 §6 分支 3 说的"观测/契约修复"，而分支 3 的前提（R1 已过）不成立。

## 六、missing 登记（未用重放或新数据补）

| 被问到的量 | 状态 | 原因 |
|---|---|---|
| 拒付记录的 per-message subscriber count | **missing** | 671 604 条 `refused` 转录记录**不含** `subscribers` 字段（实测 0 条含）；只能按已证明的恒定 fan-out 反推 payload，本轮如此标注 |
| 候选 3 / 4 / 5 的接收者集合 | **不可构造** | 见 §四；不是数据缺口，是这些量在当前系统里不存在 |
| coalition 规则下的完整预算轨迹与逐步 payability | **未计算** | 需要在新报价下重跑额度累积（等于重放环境）；本轮 0 episode，且 R2∧R3 已不过，R4 不必出数 |
| `send` 记录里 `members` 的未来一致性 | **不适用** | 投影只用发送时刻字段；未读 `targets[].assigned_agents` 等事后/真值字段来做路由 |
| 真实无线电邻域拓扑 | **不存在于本仓库** | 系统是抽象订阅集合，没有链路层模型；本轮不引入 |

## 七、本轮**不**声称的事

1. 不声称"学习型协调器在任何拓扑下都无价值"——本轮只证明了**当前这套代码里
   subscriber count 没有自由度**。候选 3/4/5 需要真实的架构与观测变更，那是**另一个系统**。
2. 不声称 coalition 路由"效果会更差"——那是环境因果主张，本轮 0 episode，无权说。
   本轮说的是一件更弱但更硬的事：它在定义上取消新目标的首次送达，因此
   **不能当作"同一任务问题的一个成本变体"来评估**。
3. 不声称 §三 的标量不变性来自实测；它来自 `cost = k·payload` 这一形式的代数性质，
   实测（1 875 次比较 0 不一致）只是核对代码确实是这个形式。
4. 不为任何后续轮次预选 λ、不改 reward、不扩观测、不动 `send_once` 基线口径。
5. 上一轮的 `no_temporal_allocation_signal` 判定**不因本轮而放宽**：本轮找的是另一条轴，
   而那条轴在当前代码里同样不成立。

## 八、产物与复跑

| 文件 | 内容 |
|---|---|
| `COMM_RECIPIENT_TOPOLOGY_AUDIT_PROTOCOL.md` | 冻结协议（先于分析提交，commit `71cb144`） |
| `topology_common.py` | 只读工具、环境 tripwire、15 个源码锚点定位器、写盘沙箱 |
| `topology_inputs.py` → `..._INPUT_MANIFEST.json` | 618 文件 SHA256（163.4 MiB）、格覆盖、未读文件登记 |
| `COMM_RECIPIENT_TOPOLOGY_SOURCE_MAP.md` | §2 九问的 `file:line` 调用链 |
| `topology_stored_data.py` → `..._STORED_DATA.json` | subscriber count 与费用分解 |
| `topology_scalar_proof.py` → `..._SCALAR_PROOF.json` | 标量不变性与"fan-out ≡ 预算档"等价性 |
| `topology_projection.py` → `..._PROJECTION.json` | `offline_recipient_cost_projection` |
| `topology_verify.py` → `..._VERIFICATION.json` | 输入未变 / 0 启动 / production code 无 diff |
| `test_topology_harness.py` | 13 项 0 episode 测试（tripwire 有效性、恒等式、尺度不变、空集≠省钱） |
| `COMM_RECIPIENT_TOPOLOGY_RESULTS.json` | 本轮全部实测与逐门判定的机读版 |
| `COMM_RECIPIENT_TOPOLOGY_DECISION.md` | §6 分支判定与后续 |

```bash
cd runs/comm_recipient_topology_feasibility_20260923
python -m unittest test_topology_harness   # 13 项，0 episode
python topology_inputs.py
python topology_stored_data.py
python topology_scalar_proof.py
python topology_projection.py
python topology_verify.py                  # 必须 passed=true
cd ../.. && python -m unittest discover -s integrated_sar/tests && python -m integrated_sar
```
