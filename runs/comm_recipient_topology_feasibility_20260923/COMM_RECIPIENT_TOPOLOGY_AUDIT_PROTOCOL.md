# 接收者与真实拓扑可行性审计：冻结协议

日期：2026-09-23。分支 `analysis/comm-recipient-topology-feasibility`，起点
`analysis/comm-action-geometry-audit@efa942857f5bce1af01b84c1739c8fc9583dcca7`（该 SHA 已在线核对为
远端同名分支 HEAD：`git ls-remote` 与本地 `git rev-parse HEAD` 逐字符相同）。
上一轮：`runs/comm_action_geometry_audit_20260923/`，统一判定名 `no_temporal_allocation_signal`。
总纲 v3.7。

## 0. 本轮的性质：先问"这个动作是否合法"，再问"学不学得动"

上一轮证明：在**当前广播协议**下，额度失效的形式是"一次性花完然后冻结在一个残值上"，
同一步内可支付严格子集的窗口只有一个包的 5.8 % 宽，15/15 次被一步跨过，且窗口里的候选
100 % 是续报。因此"发送时机"这条轴上已经没有留给学习器的分配信号。

本轮**不**去找一个新的时机设定。本轮只回答一个更靠前的问题：

> **`message_payload_bytes × subscriber_count` 里的 `subscriber_count`，
> 在这套系统里是不是一个真实的、随目标/机器人/通信状态变化的量？
> 还是一个对所有消息都相同的常数？**

如果是常数，那么"多订阅者 / 真实拓扑"这条轴**在当前代码里不存在**，
增加机器人数或订阅者数只会把所有报价统一放大，不改变任何一步的选择结构。

**这一轮是可行性审计，不是为学习模型制造有利场景的尝试。** 判据的设计目标是**允许它失败**，
而且 R1 的失败结果（`global_fanout_is_only_a_scalar`）是本轮预期中最可能也最有价值的答案。
一条被人工放大或被随机化的 fan-out 不算发现——§5 的禁止事项把这写死。

**0 新 episode、0 新检测流、0 训练、0 `optimizer.step`、0 checkpoint 加载、0 启动。**
不改 production code、不改 reward、不选 λ、不改预算、不启用 ACK 或丢包、
不随机生成拓扑、不用 payload padding 扩大报价、不触碰 `800000..800399`。

## 1. 输入与冻结

只读两类东西：

1. **源码**（只读追踪，不修改）：`integrated_sar/comm.py`、`integrated_sar/joint_driver.py`、
   `integrated_sar/comm_budget.py`、`integrated_sar/adapters/task03.py`、
   `integrated_sar/coordinator.py`、`integrated_sar/assembly.py`、
   `task_allocation/old_task_env.py`、`task_allocation/old_driver.py`。
2. **已存产物**：上一轮冻结的 576 格行集与其通信转录（`runs/comm_task_pressure_learnability_20260922/`），
   以及四层结果、账本、provenance。输入 SHA256 见
   `COMM_RECIPIENT_TOPOLOGY_INPUT_MANIFEST.json`，沿用上一轮 tripwire 机制证明 0 episode。

**先于任何分析提交本文件与 `topology_common.py`（含 tripwire 与只读工具）。**

## 2. 源码调用链必须回答的九问（逐问给出 `file:line`）

产出 `COMM_RECIPIENT_TOPOLOGY_SOURCE_MAP.md`。九问：

1. subscriber 列表在哪里创建；
2. 谁向 `CommRelay` 注册 subscriber；
3. 是否所有消息都广播给同一批机器人；
4. subscriber 数量是否在一个 episode 内固定；
5. 是否按 target、coalition、任务阶段、机器人位置或 belief 状态变化；
6. sender 在采样动作**之前**是否知道 subscriber 集合（以及报价）；
7. `attempt_charge_bytes` 是否已经包含真实 fan-out；
8. ACK、gossip、直接观测与通信订阅之间是什么关系（哪些路径会写 belief、哪些受哪个开关控制）；
9. 分配器与 coalition coordinator 是否依赖"所有机器人收到所有目标"的隐含假设。

**方法约束**：每一问必须由当前代码的调用链回答（函数定义 + 调用点 + 赋值点），
不得只引用仓库内文档或注释的说法；文档与代码冲突时以代码为准并写出冲突。

## 3. 已存数据审计（不启动环境）

按消息统计 subscriber 数与费用分解：

- 每条消息的 subscriber count；全体 min / median / max；
- **每个 episode 内是否恒定**（逐 episode 检查取值集合大小）；
- 不同 target 之间是否不同；first report / refresh / retry 之间是否不同；
- **费用恒等式闭合检查**：`cost_bytes == payload_bytes × subscribers` 逐条核，报违反数；
- 报价差异中来自 payload 与来自 fan-out 的分解（方差分解 + 极差分解两种口径都给）；
- 当前所有候选是否只是"同一个固定 fan-out × 各自的 payload"。

若转录未存某字段：**从配置与注册调用链机械推导**并写明推导路径；推不出来就登记 missing。
**不允许用新 episode 补数据。**

## 4. 门禁 R0–R4（阈值先于数据冻结）

**R0 语义可信**（五条同时成立，否则本轮不出结论）
1. subscriber 构造调用链完整（§2 的 1–5 问全部有 `file:line` 答案）；
2. 当前费用分解闭合：`cost == payload × subscribers` 违反数 = 0；
3. 全程未读取任何未来结局来构造 recipient 集合（离线投影只允许使用发送前状态字段）；
4. 0 episode、环境入口调用日志为空、四层输入文件 SHA256 不变；
5. 本轮不修改任何 production code。

**R1 当前拓扑是否已经异质**
通过条件：**当前系统中不同消息的真实 subscriber count 会变化**
（即存在某条流、某个 target 或某类消息，其 count 不同于其他）。
不通过的结论名写死为 **`global_fanout_is_only_a_scalar`**，并附标量不变性证明
（`topology_scalar_proof.py`：把全体报价除以同一个 k 后，"带内宽度 / 单包大小"这一比值
是否为尺度不变量——若是，则任何全局缩放都不可能打开选择窗口）。

**R2 候选 recipient set 是否合法**
至少存在一种 recipient set，同时满足：
1. 可由**动作前**信息确定；
2. 不依赖未来结果；
3. 不破坏任务发现与分配的基本语义；
4. 不让接收者选择依赖待学习策略自身的历史结果。

**R3 发送前是否可观测**
sender 必须在采样 SEND/HOLD **之前**知道：recipient count、projected charge、
recipient-set reason。三者任一不可观测，则该候选**不能**作为学习控制器的合法输入。

**R4 离线报价是否形成有意义选择**（只有在 R2∧R3 通过后才有意义）
至少需要**同时**满足：
1. ≥ **8/16** 条 g12 流出现"不同新目标首报之间"的报价差异；
2. ≥ **8/16** 条流出现真实 strict-subset-payable；
3. strict-subset 不能主要由同一余额状态重复计数（须报每流带内步数的分布与
   "多少条流贡献了 90 % 的步数"这一集中度指标）；
4. 不能只有 refresh 之间的选择；
5. 报价差异来自真实 recipient semantics，而非人工缩放
   （须给出该集合的语义构造规则，并证明它不是 `count = f(任意)` 的反向拟合）。

**阈值不可事后调整。** 若某条不满足，如实报不满足。

## 5. 严格禁止

不训练；不改 production code；不改 reward；不选择 λ；不改预算；不启用 ACK 或丢包；
不运行新流或新 episode；**不随机生成拓扑**；**不用 payload padding 扩大报价**；
不触碰封存测试集；**不将"发送给谁"和"什么时候发送"合并成一个动作空间**；
**不削弱 `send_once` 基线**（不得为了新轴的成功而重述它的表现）。

补一条本轮特有的：**不得为了通过 R4 而放宽 recipient set 的语义要求**。
R4 的目的在于判断"真实接收者语义能否产生新结构"，不是在 R4 上拿到绿灯。

## 6. 结果分支（判完即按此走，不自行加码）

| 实测 | 结论与下一步 |
|---|---|
| R1 不过，且无合法候选通过 R2 | **关闭多订阅者/拓扑轴**。不再做通信学习实验，转回系统集成、任务分配或论文负结果整理。 |
| R1 不过，但 R2–R4 对某个真实候选通过 | 只输出下一轮"小型规则协议验证"的**设计**，不实现、不训练。 |
| R1 已过，但 R3 不过 | 下一轮只能设计观测/契约修复，不能训练。 |
| R1–R4 全过 | 说明真实 recipient-aware topology 可能形成新估计对象；下一轮**先比较规则**，不直接训练；规则确实存在状态依赖 Pareto 前沿后，才讨论学习。 |

四条共用：本轮**不自动执行**任何一条后续；判完即停止并把决定权交回使用者。

## 7. 交付物

`runs/comm_recipient_topology_feasibility_20260923/`：

| 文件 | 内容 |
|---|---|
| `COMM_RECIPIENT_TOPOLOGY_AUDIT_PROTOCOL.md` | 本文（先于分析提交） |
| `topology_common.py` | 只读工具 + 环境 tripwire + 写盘沙箱 |
| `topology_inputs.py` → `COMM_RECIPIENT_TOPOLOGY_INPUT_MANIFEST.json` | 输入冻结、规模、原判据 SHA、0-episode 证据 |
| `COMM_RECIPIENT_TOPOLOGY_SOURCE_MAP.md` | §2 九问的 `file:line` 调用链 |
| `topology_stored_data.py` → `..._STORED_DATA.json` | §3 已存数据的 subscriber count 与费用分解 |
| `topology_scalar_proof.py` → `..._SCALAR_PROOF.json` | fan-out 标量不变性的机械证明（§4.6） |
| `topology_projection.py` → `..._PROJECTION.json` | 离线投影（若合法可做）`offline_recipient_cost_projection` |
| `test_topology_harness.py` | 合成数据 0-episode 测试（含 tripwire 有效性、分解闭合、标量不变性） |
| `topology_verify.py` → `..._VERIFICATION.json` | 事后核验：输入未变、0 启动、未改原判据 |
| `COMM_RECIPIENT_TOPOLOGY_RESULTS.md` / `.json` | 实测与 R0–R4 逐门判定 |
| `COMM_RECIPIENT_TOPOLOGY_DECISION.md` | §6 分支判定、结论范围与不做的事 |

## 8. 复跑

```bash
cd runs/comm_recipient_topology_feasibility_20260923
python -m unittest test_topology_harness   # 0 episode
python topology_inputs.py
python topology_stored_data.py
python topology_scalar_proof.py
python topology_projection.py              # 仅在 R2∧R3 通过时有意义，否则只报不适用
python topology_verify.py
cd ../.. && python -m unittest discover -s integrated_sar/tests && python -m integrated_sar
```
