# Zhenghe：搜索—协调—分配一体化搜救系统

当前研究入口：[项目总纲 v4.1](PROJECT_CHARTER.md) 与
[搜索轨迹逐时刻证据 + 同一真实环境的搜索策略对照](runs/search_trace_and_policy_20260925/SEARCH_POLICY_COMPARISON_RESULTS.md)
（机读：`SEARCH_TRACE_IDENTITY_REPORT.json`、`SEARCH_MISS_TIMELINES.json`、
`SEARCH_DETECTION_CORRECTNESS_VERDICT.json`、`SEARCH_POLICY_COMPARISON_RESULTS.json`、
`RUN_LEDGER.json`、`ROUND_VERIFICATION.json` `passed=true`、`MANIFEST.json`）。
本轮 **0 训练、0 奖励改动、0 通信配置改动、0 机器人数增加、0 目标数增加、未用 mock 替代真实环境**；
协议与启动账本在任何 episode 之前单独提交（R1：B1 被动日志正确性是 S1 的**硬前提**）。

* **B1 已进生产，被动性是测出来的**：`integrated_sar/search_probe.py` +
  `timemap.search_trace_enabled`（默认 `False`）在探测那一刻记快照——目标稳定 ID 与存活状态、
  机器人移动前/后位置、环境自己判定发现并删除的 ID、由环境原语重数的观察者—目标对、
  适配器实际发出的事件、未发现目标移动后的新格、终止原因与末尾位置。
  8 格 × 1000 步**逐步**比较动作/奖励/`done`/两个位置数组/`_py_rng` 与 `_np_rng` 状态摘要
  ⇒ **32 项检查全真**；回归 `test_search_probe.py` 11 项，项目套件 `Ran 713 tests, OK (skipped=2)`。
* **探测链没有可复现缺陷**：14 个历史 seed（10 遗漏 + 4 登记控制）**14/14 复现**，
  检测序列与归档逐条相同；三视图在 **120 000 步上 0 分歧、0 条身份异常**；
  12 个遗漏目标的时间线里类别 1（判据成立而未登记）= **0**、类别 2（删除了但事件缺失/错 ID/重复）= **0**、
  类别 4（记录不足）= **0** ⇒ `no_defect_found`，**未走修复分支**。
  第十七轮那两个候选（`1100051`、`1100053`）就此关闭为"该次轨迹没有可见接触"（最近接近 Chebyshev 6 格外）。
* **策略对照已完成，方向与预期相反**：同一真实环境、同 seed、只换动作提供器，
  60 新 seed `1200000..1200059` × 2 臂 = **120/120 完整**。每 episode 未发现目标
  **checkpoint 0.150 vs frontier 1.150**（2.50 % vs 19.17 %，分母 360 放置），
  配对差 **−1.000 个/episode**（= −60 个/批），95 % CI [−1.333, −0.667]，逐 seed **35/20/5**。
  预登记的停 1、停 2 **两条都不成立**（两条符号都按"换策略能减少遗漏"一侧写）⇒ 按协议"区间内"
  报**未达判据、正常交付**，**不追加更强策略、不追加 seed、不训练**。
* **系统级读数只能这样写**：按时完成任务 5.833 vs 4.833 个/episode 是搜索策略引起的
  **总系统效应**（下游任务属性随发现位置/时刻生成，两臂任务集不是同一份），**不是**分配能力提升；
  frontier 的输入经核查不含隐藏目标位置、未来状态或额外全局信息（合法性用与训练侧同源的
  `env.map`，信息素先掩到自己视野窗口内）。
* **启动 158/160**：超正常计划的 8 次全部走储备（4 次记录器 `TypeError` 技术失败 + 1 接线 + 3 诊断），
  产物隔离未删除；`SEED_COLLISION_AUDIT.json` clean（未顺延改号）、控制 seed 规则先行登记。
* **对学习型通信研究**：通信侧结论不必再向搜索侧借解释；本轮是整体系统的**搜索侧诊断**，
  主线仍是学习型任务信息传递控制，**不**意味着 frontier 坏、**不**意味着分配器变差、
  **不**提供新的通信效果读数。第十七轮那两份冻结件（B1 方案、S1 协议）**原件一字未改**，
  修订全部写进本轮的独立补充件。

* **交付状态**：本轮全部提交在**本地分支** `analysis/search-trace-and-policy`；
  远端**未核验**——`git push` 4 次与 `git fetch` 1 次全部连接失败（`github.com:443` 不可达），
  最后一次实际在线核验到的远端状态仍是 `80af16b…`。**不伪称推送成功**；
  网络恢复后执行 `git push -u origin analysis/search-trace-and-policy` +
  `git ls-remote origin refs/heads/analysis/search-trace-and-policy` 即可（结论与哈希不受影响，
  细节见 `runs/search_trace_and_policy_20260925/SEARCH_POLICY_COMPARISON_RESULTS.md` §十三）。

上一轮（第十七轮）入口：[真实 PILOC 未发现目标的逐个归因](runs/real_search_undetected_attribution_20260923/SEARCH_UNDETECTED_ATTRIBUTION.md)
（机读：逐目标表 `UNDETECTED_GOAL_ATTRIBUTION.json`、认证几何 `SCENARIO_GEOMETRY.json`、
字段负证据 `FIELD_INVENTORY.json`、计数闭合 `BATCH_RECONCILIATION.json`、
核验 `ATTRIBUTION_VERIFICATION.json` `passed=true`）。
本轮 **0 episode、0 训练、0 生产代码改动、0 通信配置改动、未加目标难度**；
600 个输入文件重哈希 0 变 0 缺、`piloc`/`task_allocation`/`torch` 在分析进程里从未导入、
环境入口 tripwire 解析到 3 个入口且分析期调用 0 次。

* **两批真实搜索各自闭合、不相加**：D1 的搜索侧（seed `1100000..1100059`）
  `360 放置 / 352 发现 / 352 按时`；2026-09-18 systemloop（`31000..31059`）`360 / 356`。
* **四桶**：不可达或场景问题 **0/720**（判据是 8-连通引理，且 120/120 个 seed 的重建栅格
  `map_sha256` 与运行记录逐字符相同）；曾进入合法范围却未发事件 **0 判定 + 2 个候选**；
  可达但轨迹未覆盖 **10**（判读以探测器该格正常为前提，证据等级写明）；日志不足 **2**。
* **未发现者不是疑难格**：距最近搜索起点的测地距中位 10.5 vs 已发现者 9.0、
  合法可见格数 65.5 vs 65.0、**起点到其可见集只需 2 步 vs 3 步** ⇒ "改放置规则消掉未发现"这条路线被排除。
* **一条会改变读法的环境事实**：未被发现的目标**每步自己走一格**（`timemap.py:814-845`）。
  日志独立复现：708 条发现里只有 **35.9 %** 发生在放置格上，晚发现偏移中位 6 格、最大 45 格，
  而"偏移 > 已过去步数"违例 **0** 条 ⇒ `scenario.goal_positions` 与 B0 的 `placed` 都只是 t=0 放置格，
  **"某格子没被扫过"不再等价于"某目标没被发现"**；第十五/十六轮的计数与判定一个都不改，
  只登记这条限定语。
* **缺的是两条位置路径**：逐时刻搜索机器人位置、逐时刻目标位置，在两批 360 个生命周期文件里
  存在数 **0/360**（B0 补了放置、没补路径）。两份**已冻结未执行**的下一步件因此交付：
  [B1 最小被动记录方案与两个候选的最小复现](runs/real_search_undetected_attribution_20260923/MISSING_FIELDS_AND_PASSIVE_RECORD_PLAN.md)、
  [只改搜索侧因素的 S1 配对验证协议](runs/real_search_undetected_attribution_20260923/SEARCH_SIDE_PAIRED_VALIDATION_PROTOCOL.md)。
  两个候选（seed `1100051`、`1100053`：机器人起点格就在目标可见集内）在被复算证明之前
  **不得**写成"发现逻辑有缺陷"——观测判据在机器人移动之后才评估。

上一轮（第十六轮）入口：[D1 实测结果：部署路径的信息路径值多少按时任务](runs/comm_info_path_validation_20260923/D1_RESULTS.md)
（判据 `D1_PROTOCOL_REVISION.md` 在任何新 episode 之前冻结并单独提交；`B0_PAIRED_REPLAY.json`、
`IDENTITY_AUDIT_data.json`、`D1_VERIFICATION.json` 均 `passed=true`）。
B0 的四类被动日志已进生产（写侧文件、不加 lifecycle 事件类型、不进观测、不消耗随机数），
D1 四臂 × 60 配对 seed **240/240 完整、0 技术失败、启动 264/296**，三个同 seed 配对比较
（配对 bootstrap B=10 000）：

| 代号 | 干预 | 按时完成差（个/ep） | 95 % CI | 胜/平/负 |
|---|---|---:|---|---|
| C1 | 上行说话 vs 完全静默 | **+0.31667** | [+0.18333, +0.45] | 17/43/0 |
| C2 | 投递 1.0 → 0.7 | **0.00000** | [0.0, 0.0] | 0/60/0 |
| C3 | 自带感知开 vs 关（**同一批丢包**） | **+1.28333** | [+1.01667, +1.55] | 45/15/0 |

C3 的"同一批丢包"是逐包核过的（发射序列与送达结局全同、计费相同），所以"同样每集丢 10.25 个包、
部署路径 0 损失 vs 隔离路径 1.283 33 个/ep"是本轮最硬的一条：**部署路径对丢包的免疫来自环境传感旁路，
不来自通信策略**；C1 约 **46 021 B/个**救回的按时任务；冻结主判据因分母为空判 `undecidable`，
N6 两方向都不判；机制记录的三条零与 `S_all=0.883` 的校准限制见该轮结果件（`800000..800399` 继续封存、
那 240 格不是学习模型的独立测试集）。

上一轮（第十五轮）的**收束件**
（[通信研究收束：可引用的阶段结论](runs/comm_research_closure_20260923/COMMUNICATION_RESEARCH_CLOSURE.md)）
把十四条轮次的结果分成四条**必须分开引用**的结论：
① 规则与协议层收益**成立**（`send_once` 以 14 916 B 拿 5.7917 个/ep 支配 `immediate` 的
26 897 B / 5.2917，其中 49.9 % 字节是上界替它发的刷新；ACK 规则在隔离有损 +0.835 个/ep
但 total +26 354.1 B/ep，正常路径差 0.000 ⇒ 被支配、不提部署）；
② 学习型发送控制器**未建立附加价值**（四批独立训练、门槛全 0/3，但每一批的每个运行都显著优于自身初值）；
③ 时机轴 `no_temporal_allocation_signal`；④ 拓扑轴 `global_fanout_is_only_a_scalar`。
四条都**只**限定在当前任务分布 / 广播协议 / 硬预算 / 观测·动作·奖励之下，
**都不构成**"强化学习普遍无效"，也不构成"不存在更优策略"或"规则全局最优"。
生产默认值全部保持原样（`comm_policy=immediate`、无额度、`legacy_shared_bound`、`comm_ack=False`、
`subscribers=all`、`comm_range=0.3`、`gossip_interval=2`）：强规则基线来自**隔离**实验，
而隔离与部署的差别是环境自带传感旁路，不是策略变好——所以不因它改默认值。

**瓶颈盘点**（[JOINT_SYSTEM_BOTTLENECK_AUDIT.md](runs/comm_research_closure_20260923/JOINT_SYSTEM_BOTTLENECK_AUDIT.md)）第一次把同一个 `target_id` 串过八个阶段（放置→发现→交付→获知→分配→到场→开工→按时/过期），
四条归档路径分开报告，**40 项与已发布结果的核对全部一致**。两条结论被本轮（D1）实测接管：
部署路径上"信息未到达"当时写的是 `not attributable`（不是 0，因为归档的四条部署路径全是无损信道
且没有 per-recipient 记录），而"同一批丢包在两种架构下价格差约 30 倍"当时只能给跨路径比。

再上一轮入口（第十四轮）：[接收者与真实拓扑可行性审计](runs/comm_recipient_topology_feasibility_20260923/COMM_RECIPIENT_TOPOLOGY_RESULTS.md)
（判定 [DECISION](runs/comm_recipient_topology_feasibility_20260923/COMM_RECIPIENT_TOPOLOGY_DECISION.md)、
[源码调用链](runs/comm_recipient_topology_feasibility_20260923/COMM_RECIPIENT_TOPOLOGY_SOURCE_MAP.md)）：
`subscriber_count` 逐 episode 恒为 6（8 904 条发射 distinct = {6}），且它的任意全局缩放与调预算档
在 1 875 次分类比较中 0 不一致 ⇒ 拓扑轴关闭 `global_fanout_is_only_a_scalar`。
更早：[动作几何审计](runs/comm_action_geometry_audit_20260923/COMM_ACTION_GEOMETRY_AUDIT.md)
（`no_temporal_allocation_signal`）与
[任务压力第四层补跑实测](runs/comm_task_pressure_learnability_20260922/COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.md)。

本仓库整合两个已有研究模块：

- `piloc/`：多智能体区域搜索、时间戳地图、信息素和受限通信。
- `task_allocation/`：异构机器人、多阶段任务和 CTDE/MAPPO 任务分配。
- `integrated_sar/`：两者之间的事件、任务状态、协调优化与执行反馈框架。

当前目标是形成完整闭环：

```text
PILOC 搜索目标
  → DetectionEvent
  → TaskRegistry
  → Coordinator 提供残差优化
  → CTDE 分配机器人
  → 执行任务并产生 ProgressEvent
  → 注册表和协调器更新
  → 必要时重新分配
```

## 先读哪些文件

新加入的开发者或 agent 请按以下顺序阅读：

1. [`AGENTS.md`](AGENTS.md)：代码主线、约束和禁止事项。
2. [`ARCHITECTURE.md`](ARCHITECTURE.md)：模块边界、数据流和训练方案。
3. [`PROJECT_STATUS.md`](PROJECT_STATUS.md)：已经完成和下一步尚未完成的工作。
4. `integrated_sar/contracts.py`：模块间的数据契约。
5. `integrated_sar/orchestrator.py`：端到端运行顺序。
6. `integrated_sar/adapters/`：现有两个项目的适配方式。
7. `integrated_sar/joint_driver.py`：把上述部件接成可运行的推理闭环。

## 当前代码主线

- 搜索训练：`piloc/train.py`
- 搜索环境：`piloc/src/Map/timemap.py`
- 搜索策略：`piloc/src/DDQN/Agent.py`
- 任务分配训练：`task_allocation/old_driver.py`
- 任务环境：`task_allocation/old_task_env.py`
- CTDE 网络：`task_allocation/old_network.py`
- 任务分配评估：`task_allocation/old_test.py`
- 联合推理入口：`integrated_sar/joint_driver.py`
- 场景生成与坐标转换：`integrated_sar/scenarios.py`
- 搜索侧运行时：`integrated_sar/search_runtime.py`、`integrated_sar/piloc_runtime.py`
- 任务侧运行时：`integrated_sar/task03_runtime.py`

`task_allocation/driver.py`、`network.py`、`env/task_env.py` 是未完成的简化分支；
`runner.py`、`worker.py` 等是旧架构。除非任务明确要求，否则不要基于它们继续开发。

## 快速验证

框架核心测试仅依赖 Python 标准库和 NumPy：

```bash
python -m unittest discover -s integrated_sar/tests -v
python -m integrated_sar
python -m integrated_sar.joint_driver --mock --steps 20 --seed 42
```

预期演示最后输出：

```text
final_status=completed
```

预期 `--mock` 运行逐 tick 打印进度，最后输出一段 JSON 摘要（`targets_discovered`、
`status_counts`、`rejected_assignments` 等）并以退出码 0 结束；
checkpoint、地图或依赖缺失时打印一行 `[joint-driver] <ErrorType>: <原因>` 并以退出码 2 结束。

当前测试覆盖：事件扇出、目标去重、坐标转换、协调偏置、PILOC 发现适配、
外部任务注入 `old_task_env`、“发现—分配—执行—完成”端到端闭环、
mock 联合推理的确定性与生命周期日志字段、分配命令 TTL 与 revision 校验、
硬 mask 不可被残差绕过、被拒命令在下一 tick 恰好重规划一次、
检测半径在环境/策略/适配器三处只有一个真值、通信范围外的目标不被任何机器人或协调器使用、
非整数 `dt` 下三个时钟仍然一致、场景载入 PILOC 后不残留随机初始状态、
复用适配器开新 episode 不会重复事件 id、mock 的双通道地图与 PILOC 逐格一致，
以及 checkpoint/地图/依赖缺失时的清晰报错。

第二轮审查后的补充覆盖：远距离检测经上行中转送达救援 belief 并完成闭环、
未收到消息的机器人看不见也选不到该任务、消息首次到达当 tick 立即协调并恰好分配一次、
gossip 每 tick 最多传播一跳、丢包时 belief 保持原值、通信随机过程按 seed 复现、
断联期间真值推进不改变 observation/mask 而新消息到达后才更新、
公开 reset 恢复三个时钟/两个模拟器/日志与计数器且同 seed 逐记录复现、
日志只记录 Registry 接受的事件（迟到/未知/终态后事件三者一致被拒）、
mock 前沿策略不读视野外的信息素，以及负的检测半径以退出码 2 拒绝。

第三轮审查后的补充覆盖：真实 PILOC 的 reset 会 reseed 并重建信息素/位置历史/通信统计/
延迟消息，适配器与协调器的每 episode 计数器一并复位；协调偏置与导航目标只来自
该机器人的 belief（只改注册表真值不改变二者）；belief 的位置/截止时间/未来需求/
复杂度/进度/成员/动态与已分配标志任一变化都被报告为调度信号，纯时间戳刷新不算；
被注册表拒绝的检测不入通信队列也不触发协调；通信拓扑可通过配置收窄并在集成测试中生效；
合法 re-detection 的日志按注册表快照写入并保留最初 detection 作为关联键。

第四轮审查后的补充覆盖：协调器完全不读注册表（只改注册表置信度、或把任务置为终态，
都不改变任何偏置，也不会让还没收到消息的机器人重规划）；一个机器人 belief 的变化不会经
池级 `task_bias` 影响另一个机器人；中转消息携带的 progress/confidence/revision 全部写进
belief，同时间戳但版本更高的合法更新不再被丢弃，更旧或同刻同版本的更新会被丢弃；
`ExecutionStepResult` 的字段顺序保证位置参数调用仍然是 metrics；真实 PILOC reset 用例
改用 60×60 场景，并用 stand-in 环境覆盖生产构造路径（60×60 守卫与派生状态重建）。

第五轮审查后的补充覆盖：机器人仍相信一个已被注册表关闭的任务时，命令带缓存的终态
revision 通过闸门，机器人前往现场核实并学到“已完成”，同场的活跃任务不会被饿死；
gossip 在时间戳相同时按 revision 取新，两台相邻机器人不会长期保留不同版本的同一任务；
`AgentTaskBelief` 与 `CoordinationAdvice` 的新字段为 keyword-only 且追加在原字段之后，
旧的位置参数调用不会被重新绑定；真实 PILOC 测试只在 gym/matplotlib/Pillow 确实缺失时跳过，
依赖齐全后的导入失败会让测试失败而不是伪装成 skip。

## 联合推理入口

`integrated_sar/joint_driver.py` 只做推理，不训练任何网络。

无 checkpoint 的 mock 模式（CI 与未配置权重的机器可用，只需 NumPy）：

```bash
python -m integrated_sar.joint_driver --mock --steps 20 --seed 42
```

加载真实权重的模式：

```bash
python -m integrated_sar.joint_driver \
  --piloc-checkpoint <shared_mappo_model_*.pth 或其所在目录> \
  --task-checkpoint <ctde_final.pth 或其所在目录> \
  --map <障碍图 .npy/.png/.txt，可省略以按 seed 生成> \
  --seed 42 --steps 200 --device cpu --output joint_lifecycle.jsonl
```

常用参数：`--search-agents`、`--num-goals`、`--grid-size`（使用 PILOC 权重时必须为 60）、
`--obstacle-density`、`--rescue-species`、`--rescue-per-species`、`--actor-kv-mode`、
`--residual-scale`（协调残差上界，0–1）、`--command-ttl`、`--coordination-interval`、
`--allocation-interval`、`--dt`、`--detection-radius`（不允许为负，0 合法但退化）、
`--comm-delivery-probability`、`--comm-delay-ticks`、
`--comm-subscribers`（`all` 默认覆盖全救援队，或 `first:N`，或机器人 id 列表）、
`--comm-budget-bytes`（每 episode 的字节额度，对所有策略同一执行器；按**发送尝试**计费，
丢包照计，上界替它发的也算，付不起就不发并记一次拒绝）、
`--comm-max-wait-steps` 与 `--comm-urgent-wait-steps`、`--comm-urgency-time`
（共享静默上界：一条变化最长能沉默多少步、任务临近 deadline 时收紧到多少步、以及多近算临近；
它是任何臂都逃不掉的下界，默认两者为空即不启用）、
`--comm-timing-mode`（谁按这条上界行事：默认 `legacy_shared_bound` = 到点由传输层代发、照计费照拒付，
即全部历史实验；`policy_owned_timing` 是研究模式，传输层不再替任何臂开口，3/10 只能作为显式规则
被请求，硬额度/每步容量/内容合法性/执行安全一概保留；未识别的模式启动即报错，不静默退回默认）、
`--comm-policy`（上行发送策略：默认 `immediate` = 每条被接受的检测发一次且不刷新，即历史行为；
另有 `periodic`、`event`、`event_maxwait`、`event_priority`、`hold_to_bound`、`random_send`，
以及 `policy_owned_timing` 模式下表达同一时机的 `first_send_plus_bound_3_10`、`bound_only_3_10`、
`send_once`、`always_hold`）、`--comm-policy-params`（如 `interval=5`，
未知策略名或参数键直接报错而非退回默认）、`--comm-sends-per-tick`（每步发射预算，默认不限）、
`--comm-transcript`（把接受/发射/投递逐条写盘，仅测量、不被任何决策回读）、
`--comm-ack`（打开反向 ACK 信道：接收端把上行包真正应用进 belief 后才发累计确认，
经独立的有延迟/丢包的反向信道回传，每次发送尝试按序列化字节全额计费、丢失也计费，
走独立的反向账本而不消耗任务上行额度；发送端观测随之增加四维 `comm-send-state/2`
——已确认比例、最大 ACK age、与最高已确认视图的最大 gap、未全体确认的广播次数，
无任何 ACK 到达时是显式的 unknown 编码；默认关闭，关闭时不构造任何反馈状态、
不改变候选顺序与 RNG 消耗，历史路径逐字节不变）、
`--comm-ack-delay-ticks` 与 `--comm-ack-delivery-probability`（反向信道的延迟与丢包，
默认取上行自己的 `--comm-delay-ticks` / `--comm-delivery-probability`，不搞第二套更友好的参数）；
与 ACK 一同可用的四条规则：`retry_until_acked`、`event_retry_until_acked`、
`exponential_backoff`、`deadline_tightening_retry`（都只读发送端自己的发送历史与已到达 ACK）、
`--comm-range` 与 `--gossip-interval`（救援环境自带的视距刷新半径与"刷新+一跳 gossip"周期，
默认 0.3 与 2 就是 `TaskEnv` 自己的取值；把间隔调到大于步数即可把这两条旁路关掉），
`--coordinator`（`heuristic` 默认；`coalition` 按缺口而不是重叠打分；`coalition_planner`
在空闲机器人与任务开放槽位之间规划团队预约；`deadline_aware` 是可学习性门禁用的同信息
oracle，只退还 belief 预算内到不了的任务）、
`--no-coordinator`、`--quiet`。
`--info-probe <路径>` 打开 B0 的四类**被动**日志（逐行 JSON 写到给定路径）：放置目标、
逐接收者送达结局、逐 (目标, 机器人) 的知晓来源与时刻、真实到场步。它默认关闭，只读不写状态、
不进 actor 观测、不消耗随机数，成对回放已证明开启前后全部生命周期记录与扣费字节逐字段相同
（`runs/comm_info_path_validation_20260923/B0_PAIRED_REPLAY.json`）。
`--search-probe <路径>` 打开 B1 的搜索侧逐时刻旁路（同一纪律：默认关闭、写侧文件、
不加 lifecycle 事件类型、不进观测与奖励、不消耗随机数）：探测时刻的目标格与稳定 ID、
机器人移动前/后位置、环境自己判定发现并删除的 ID、观察者—目标对、适配器实际发出的事件、
未发现目标移动后的新格、终止原因与末尾位置，以及逐步三视图闭合旗标。它的被动性是**逐步实测**
过的（1000 步的位置数组与两个随机流状态摘要全同，见
`runs/search_trace_and_policy_20260925/SEARCH_TRACE_IDENTITY_REPORT.md`）。
`--search-action-policy {checkpoint,frontier}` 决定谁驱动搜索环境：`checkpoint` 用冻结权重，
`frontier` 用现成的 `FrontierSearchPolicy` 接**同一真实环境**（不是 mock）。
启动即校验，不静默放行：`--search-probe` 要求给 `--piloc-checkpoint`（mock 后端没有那一步可记），
且与 `--replay-stream` 互斥（重放流里没有搜索模块可记）；`--search-action-policy frontier`
也与 `--replay-stream` 互斥（重放流没有可驱动的搜索环境）。`--mock` 与任一 checkpoint 参数互斥。

使用 `--piloc-checkpoint` 时还有两个约束，启动即校验，不会静默放行：
`--detection-radius` 必须为 5（时间戳地图、信息素窗口和 `last_actions` 的形状都由
`obs_radius` 推出，半径不一致会让环境按一个半径删除目标、适配器按另一个半径发事件）；
`--dt` 必须为 1.0（`timemap.Environment` 每步固定把自己的时钟推进 1，无法配置）。

两侧后端互相独立：省略 `--piloc-checkpoint` 时搜索侧使用前沿贪心策略，
省略 `--task-checkpoint` 时分配侧使用启发式打分，摘要中的
`search_backend`/`allocation_backend` 会标明实际使用的后端。

目标如何传到救援队，决定了这一轮能完成多少。检测先只写入环境真值（`task_dic`），
`integrated_sar/comm.py` 的 `CommRelay` 再把它作为显式通信投递进机器人 belief：
只有注册表已接受的检测才会被记为"有待播报的新闻"，**是否当步上空气由发送策略决定**
（默认 `immediate`：每条被接受的检测发一次且不刷新，即历史行为）。载荷按**发射时刻**的
注册表快照构造，所以被推迟的变化是被取代、不是被迟到地重发；每目标每步至多一次发射，
终态任务不再播报，超出 `--comm-sends-per-tick` 的请求按统一顺序裁剪——这三条传输规则对
所有策略相同，不只约束学习策略。上行链路默认覆盖订阅它的整支救援队，可用
`--comm-subscribers` 收窄到 `first:N` 或指定 id；每包有可配置的延迟与丢包率，丢包
不重发（只有策略再发一次才可能补救）、belief 保持原值，整个随机过程由 seed 控制；
每条发射都带一个互斥主原因（`policy`：发送端自己要发；`forced_maxwait`：共享静默上界到点，
传输层替它发；`other`：预留给未来的强制协议事件），发射/尝试/字节/被拒四分列并按原因闭合，
被额度拒掉的强制发送单独计数，因此"只学会闭嘴、由规则说话"的策略不可能被当成省通信的学习成果。
三种"同一件事做两次"分别计数、不可混用：`coalesced_changes` 是生产侧合并（两条新闻装进一封信），
`redundant_emissions` 是发送侧重复（这封信的内容与上一封逐字段相同），`delivery_duplicate` 是某个
接收端拒收它已经相信的包。生成量恒等式因此写成
`generated_items = (emissions − redundant_emissions) + coalesced_changes + 终态抑制 + 终局未说`，
两侧同用"机器人会据以行动的量"这一个口径（`_View`，刻意排除 registry `revision`）。
规则菜单除 `immediate`/`periodic`/`event`/`event_maxwait`/`event_priority` 外还有
`hold_to_bound`（永不主动请求，只有上界发声）与 `random_send`（逐候选的可复现硬币），
二者与所有规则共用同一套合并、裁剪与上界。中转不读任何人的位置，也不读接收端状态（无 ACK），
因此救援队是否与搜索起点重合不是闭环启动的必要条件。压低频率或收窄拓扑都只限制这条消息通道
——机器人仍可能
经视距感知或 gossip 获知同一信息，那两条信道由 `old_task_env` 自己建模（gossip 每 tick
最多一跳；默认传感半径 0.3 下，实测 95.6% 的"首次知晓"在没有上行时依然成立），
所以比较发送策略必须用 `--gossip-interval` 大于步数的隔离配置。分配器、协调器与执行模块
都不能使用机器人尚未收到的信息：协调偏置只按该机器人自己的 belief 打分，导航目标取
believed 位置，在途重规划也只看 belief，断联期间真值推进不会改变 observation、mask
或行动目标（架构不变量 9）。
为此救援队的集结点默认仍放在搜索队的出发格上，两队共享同一个灾区；
`TaskEnv` 默认会把集结点随机撒在整个单位正方形。

`--output` 写出 JSONL 生命周期日志（`-` 表示标准输出），每条记录至少包含
`target_id`、`source_event_id`、`event_type`、`timestamp`、`discovered_by`、
`assigned_agents`、`task_revision`、`stage_index`、`progress`、
`remaining_requirements`、`status`、`failure_reason`，
可用 `target_id` 与 `source_event_id` 串起单个目标从发现到完成的完整轨迹。
日志属于运行产物，不要提交。

## 训练

不要把三个模块一起端到端训练。按以下顺序推进，每一步稳定后再进下一步：

```text
PILOC 单独训练 → 任务分配单独训练 → 冻结两端跑通联合推理 → 训练协调器 → 按 episode 交替微调
```

搜索模块：

```bash
cd piloc
python train.py --device auto --seed 0 --episodes 120000 \
  --output-dir ../runs/piloc_search --save-every 1000 --log-every 100
```

产出 `<output-dir>/checkpoints/shared_mappo_model_21_<episode>.pth`、`run_meta.json`
（git commit、seed、device、dim_info、全部超参）与 `episode_metrics.jsonl`
（发现率、首次发现步、覆盖率、移动距离、重复搜索比例）。
`--resume <checkpoint>` 以 strict 方式续训，不接受部分加载。

任务分配模块：

```bash
cd task_allocation
python old_driver.py --device auto --seed 0 --episodes 20000 --no-resume \
  --checkpoint-dir ../runs/ctde --actor-kv-mode tokens
```

产出 `ctde_best.pth`、`ctde_final.pth`、`ctde_ep_<episode>.pth` 与 `run_meta.json`。
每个 checkpoint 都记录 `actor_kv_mode`、网络维度、seed 与 commit；`summary` 与
`tokens` 的参数形状完全相同，加载端会拒绝模式不匹配的权重。新训练请加
`--no-resume`，否则会自动加载目录里最新的 checkpoint 并跳过 `global_episodes_done`，
而熵权重、学习率衰减和课程调度都由它驱动。

两端训好后冻结参数，用 `joint_driver` 验证完整数据链（见上一节）。

两点运行注意：

- 本机 `torch` 为 CPU 版，只适合冒烟验证；正式训练请放在有 GPU 的机器上。
- `--checkpoint-dir` / `--output-dir` 请用绝对路径且不含非 ASCII 字符：
  TensorFlow 的 gfile 在含中文的路径下使用相对目录会以
  `FailedPreconditionError` 失败。

## 环境

建议使用 Python 3.10 或 3.11：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

模型权重、DungeonMaps 和大规模测试集没有提交到 Git。需要运行完整训练或论文评估时，
请从团队共享存储取得这些资产，并放到代码期望的本地目录中。

## 通信控制研究轮次（每轮一个 run 目录，入口文档可点击）

研究问题、判据与结果按轮分开归档；每轮目录内的 `.md` 是协议/结果，`.json` 是机读摘要，
checkpoint、录制流、逐 episode 日志与转录**不入 Git**，只在 manifest 里记路径与 sha256。

| 轮次 | run 目录 | 关键文档 |
|---|---|---|
| 通信基线与规则 | `runs/comm_control_20260919/` | `INFORMATION_BOUNDARY.md`（旁路份额实测） |
| 预算标定与 pilot | `runs/comm_budget_pilot_20260919/` | `COMM_BUDGET_LOCKED.json`（三档锁定额度） |
| 归因与稳定性 | `runs/comm_stability_20260919/` | `COMM_TRAINING_STABILITY_PROTOCOL.md`、`COMM_TRAINING_STABILITY_RESULTS.md` |
| 概率正确性与紧预算 | `runs/comm_tight_budget_20260919/` | `COMM_PPO_PROBABILITY_AUDIT.md`、`COMM_TIGHT_BUDGET_PROTOCOL.md`、`COMM_TIGHT_BUDGET_RESULTS.md` |
| **发送时机协议消融** | `runs/comm_protocol_v2_20260920/` | `COMM_MASK_CAUSE_AUDIT.md`（掩码归因与勘误）、`COMM_PROTOCOL_V2_PROTOCOL.md`（判据先于结果）、`COMM_PROTOCOL_V2_RESULTS.md`（协议收益与学习收益分开）、`COMM_PROTOCOL_V2_MANIFEST.json` |
| **ACK 信息价值门禁** | `runs/comm_ack_value_20260920/` | `COMM_ACK_CONTRACT.md`（反向信道合同）、`COMM_ACK_VALUE_GATE.md`/`.json`（四项门禁：1/2/4 过、3 未过 ⇒ 未训练）、`COMM_ACK_MANIFEST.json` |
| **S3 收尾与规则独立确认** | `runs/comm_rule_confirm_20260921/` | `COMM_RULE_CONFIRMATION_PROTOCOL.md`（判据先于数据）、`COMM_RULE_BLOCK_AUDIT.json`（`900000..900199` 与全项目四源零重叠）、`COMM_RULE_CONFIRMATION_RESULTS.md`（3 000 episode：隔离有损 +0.835 个/ep 而 total +26 354 B、正常路径被支配）、`COMMUNICATION_LEARNING_VERDICT.md`（S3 学习结论冻结为范围准确的负结果，研究先导关闭）、`COMM_RULE_CONFIRMATION_MANIFEST.json` |
| **H=20 终态归因 + 动作路径审计** | `runs/comm_h20_terminal_attribution_20260922/` | `COMM_H20_TERMINAL_ATTRIBUTION.md/.json`（3 seed 成功 EARLY 后结局分歧；1 对 `boundary_refused`）、`COMM_POLICY_ACTION_MASK_AUDIT.md/.json`（预算不进 mask；非可执行 SEND 进训练；双动作自由训练侧 **missing**）、`analyze_h20_terminal.py`；**0 新 episode / 0 训练** |
| **任务压力 calibration：四层全套（最新，0 训练）** | `runs/comm_task_pressure_learnability_20260922/` | 前三层：`COMM_TASK_PRESSURE_LEARNABILITY_PROTOCOL.md`（九臂规则库、L0–L4 阈值、四源 seed 审计口径先于第一条新流冻结，§11/§12 记录开跑前的 harness 修正与归档一致性核验）、`SEED_COLLISION_AUDIT.json`、`STREAM_MANIFEST_g{6,8,10}.json`、`RULE_MATRIX*.json`、`PRESSURE_METRICS*.json`、`GATES_g{6,8,10}.json`、`ROW_PROVENANCE_g*.json`、`COMM_TASK_PRESSURE_LEARNABILITY_RESULTS.md/.json`（L0/L1 三层全过；L2 仅 10 目标过带 0.900；L3 三层 0/16；L4 CI 下界为 0）。**第四层 g12（追加授权 640→656 的独立扩展）**：`COMM_TASK_PRESSURE_G12_EXTENSION_PROTOCOL.md`（地位、授权范围与"不得改写原三层判定"先于第一条 episode 冻结）、`G12_SEED_COLLISION_AUDIT.json`（只审 `1000300..1000315`，四源命中全 0）、`G12_FREEZE.json` + `G12_START_LEDGER.json`、`STREAM_MANIFEST_g12*.json`、`RULE_MATRIX_g12.json`（含逐步候选几何）、`GATES_g12.json`、`ROW_PROVENANCE_g12.json`、`COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.md/.json`、`..._VERDICT.json`（L0/L1/L2 过，L3 3/16、L4 +0.0625 ⇒ 未过 ⇒ 关闭"只加目标数"这条轴）、`..._MANIFEST.json`、`LAUNCH_LEDGER*g12*.json`（数据 656/656、核验 10/12 未加、历史失败 20 全计入）；原判据文件字节不变，stream 与逐集日志不入库 |
| 修复后从初值重训 | `runs/comm_budget_masked_retrain_20260922/` | `COMM_BUDGET_MASK_PROVENANCE_CORRECTION.md/.json`（幽灵模块 `comm_transcript.py` 的成因与两个头的分离）、`COMM_BUDGET_MASKED_RETRAIN_PROTOCOL.md`（判据先于第一个训练 episode 冻结）、`CODE_FREEZE.json`（54 模块执行图，逐次启动重校验）、`PREFLIGHT.json`/`THROUGHPUT.json`/`RESUME_CHECK.json`（12 探针 + 8 集确定性恢复 + greedy 通路，0 optimizer.step）、`COMM_BUDGET_MASKED_RETRAIN_RESULTS.md/.json`（3 seed × 400k：正确性 0 违规、actionable 20,110/8,020/38,404，主效果 0/3 与非劣降本 0/3）、`RULE_REPLAY_AGREEMENT.json`（`send_once` 与归档逐包 0 差异）、`LAUNCH_LEDGER*.json`（1,489/1,500，逐进程账本）；checkpoint 与逐集日志不入库 |
| 预算动作 mask 修复 | `runs/comm_budget_action_mask_fix_20260922/` | `COMM_BUDGET_ACTION_MASK_PROTOCOL.md`（先于 episode 冻结）、`COMM_BUDGET_ACTION_MASK_RESULTS.md/.json`（M0–M3 全过：8 流 20 655 候选→真双动作 181、请求 86=执行 86、policy 拒付 0、log-prob 差 0.0）、`budget_masked.py`/`train_budget_masked.py`（采样前顺序 hard mask，tape actionable=free 故复用旧 PPO 循环）、`test_budget_action_mask.py`（16 项真实接缝）、`LAUNCH_LEDGER.json`（8/8 启动配额）；**其 manifest 的模块清单误列不存在的 `comm_transcript.py`，由下一轮更正件说明** |
| target-scoped 边界可行性 | `runs/comm_target_scoped_feasibility_20260922/` | `COMM_TARGET_SCOPED_PROTOCOL.md`、`TARGET_POINTS_INITIAL.json`、`COMM_TARGET_SCOPED_BOUNDARY_RESULTS.md/.json`、`COMM_TARGET_SCOPED_INTERFACE_CHECK.json`、`COMM_TARGET_SCOPED_MANIFEST.json`；24/28 启动；T0/T1 过；T2 仅 H=20 3/4 过 |
| 动作可行性审计 | `runs/comm_send_actionability_20260922/` | `COMM_SEND_ACTIONABILITY_PROTOCOL.md`、`COMM_SEND_ACTIONABILITY_ROWS.json`、`COMM_SEND_ACTIONABILITY_RESULTS.md/.json`、`COMM_SEND_ACTIONABILITY_POINTS.json`、`COMM_SEND_ACTIONABILITY_MANIFEST.json`；23 612 候选，both_actions_free 101/8 流，主阻塞预算；**A/B 过、C 不过** |
| 多步诊断 Stage1 | `runs/comm_multistep_stage1_20260922/` | `COMM_MULTISTEP_PROTOCOL.md`（修订协议）、`FROZEN_CONFIG.json`（`event_retry_until_acked_5`）、`POINTS_INITIAL.json`/`POINTS.json`、`COMM_MULTISTEP_RESULTS.md/.json`、`COMM_MULTISTEP_VERDICT.md`、`DIAGNOSTIC_TEXT_FIXES.md`、`COMM_MULTISTEP_MANIFEST.json`；40/40 启动（8 free + 28 成功分支 + 4 失败保留），7/8 点完整；所有 S 强发均因 insufficient_credit 被拒 ⇒ 无有效 S−H 对照 |
| 证据复核与训练诊断 | `runs/comm_evidence_review_20260921/` | `COMM_EVIDENCE_CORRECTIONS.md`/`.json`（十项核对：单位、成功率 vs 丢包率、字节方向、CI 含 0 ≠ 等效、条件区间、ACK 基线开关、因果范围、学习三事实、训练充分性、门禁样本性）、`COMM_RULE_CONFIRMATION_RESULTS_REVISED.md`/`.json`（同一统计量只抽一次样，主结果统一到 +0.835 个/ep [+0.685,+0.995] 与 +26 354.1 B/ep [+26 002,+26 688]；585 个均值字段核对未变）、`COMMUNICATION_LEARNING_VERDICT_REVISED.md`（结论范围重述，停止判定保留）、`COMM_LEARNING_DIAGNOSTIC.md`（四种解释与缺失项）、`COMM_MULTISTEP_DIAGNOSTIC_PROTOCOL.md`（≤96 分支的小实验设计，**尚未执行**） |
| 动作几何与时序价值审计 | `runs/comm_action_geometry_audit_20260923/` | `COMM_ACTION_GEOMETRY_AUDIT.md`/`.json`（A-D 四问、七套分母各算各的）、`RECOMPUTE_L0_L4.json`（四层独立重跑，L0-L4 布尔逐层一致）、`COMM_ACTION_GEOMETRY_AUDIT_VERDICT.json`（判定名 **no_temporal_allocation_signal** + 六条范围）、`AUDIT_VERIFICATION.json`；**0 episode / 0 训练** |
| 接收者与真实拓扑可行性 | `runs/comm_recipient_topology_feasibility_20260923/` | `COMM_RECIPIENT_TOPOLOGY_AUDIT_PROTOCOL.md`（R1 的失败名字先于数据写死）、`..._SOURCE_MAP.md`（九问调用链）、`..._RESULTS.md`/`.json`、`..._SCALAR_PROOF.json`（1 875 次分类比较 0 不一致）、`..._DECISION.md`（判定名 **global_fanout_is_only_a_scalar**）；**0 episode / 0 训练** |
| 通信研究收束与完整系统瓶颈盘点（0 训练） | `runs/comm_research_closure_20260923/` | `COMMUNICATION_RESEARCH_CLOSURE.md`（四条结论分开引用 + 强规则基线与适用条件 + 重开条件 N1-N6）、`JOINT_SYSTEM_BOTTLENECK_AUDIT.md`/`.json`（八阶段链、四条路径分桶、**40 项核对全一致**）、`CHAIN_RECONSTRUCTION.json`（P1 120 + P2 90 + P3 576 格逐目标链）、`STAGE_CAPABILITY.json`（哪几阶段可归因、哪几处只有代理量）、`MINIMAL_PAIRED_VALIDATION_PROTOCOL.md`（D1，**由下一轮修订后执行**）、`CLOSURE_INPUT_MANIFEST.json`、`CLOSURE_VERIFICATION.json`、`test_closure_harness.py`（21 项）；逐集日志与转录不入库 |
| 完整系统信息路径验证（B0 日志 + D1 四臂 × 60 配对 seed） | `runs/comm_info_path_validation_20260923/` | `D1_PROTOCOL_REVISION.md`（**判据在任何 episode 之前冻结并单独提交**：三个配对比较、三态机制谓词与显式分母、身份门禁 I1–I6、`information_bucket_share` 的分母定义与守卫、两条停止条件、296 次启动上限）、`B0_PAIRED_REPLAY.json`（16 格成对回放 + 4 格时机模式惰性，`passed=true` ⇒ 开日志前后检测/结局/发射/扣费全同）、`SEED_COLLISION_AUDIT.json`（`1100000..1100059` 四源命中 0）、`IDENTITY_AUDIT_smoke.json`/`IDENTITY_AUDIT_data.json`（I1–I5 逐 seed，`phrasing_licensed` 规定哪一句"同一批丢包"可说）、`D1_RESULTS.md`/`.json`（C1 +0.31667 [+0.18333,+0.45]、C2 0.00000 [0,0]、C3 +1.28333 [+1.01667,+1.55]；机制记录三条零；`declared=undecidable`）、`D1_VERIFICATION.json`（生产改动仅限 5 个 B0 文件、checkpoint 未动、封存块 artefact 0，`passed=true`）、`test_d1_harness.py`（14 项 0 episode）＋ 生产侧 `integrated_sar/info_probe.py` 与 `tests/test_info_probe.py`（12 项）；逐集日志、转录与探针文件不入库 |
| **搜索轨迹逐时刻证据 + 同一真实环境的搜索策略对照（最新：158/160 启动、0 训练）** | `runs/search_trace_and_policy_20260925/` | `SEARCH_TRACE_AND_POLICY_PROTOCOL.md`（**协议与启动账本在任何 episode 之前单独提交**；R1 把 B1 被动日志正确性写成 S1 的硬前提、R2 只做 S1b、R3 不增机器人数不用 mock、R4 效应量一律带单位、R5 判据不放宽且冲突时停在 B1）、`SEARCH_TRACE_IDENTITY_REPORT.md`/`.json`（B1 开/关 × 两 seed × 两次重复：1000 步逐步比较动作/奖励/`done`/两个位置数组/`_py_rng` 与 `_np_rng` 状态摘要 ⇒ **32 项检查全真**；14 个历史 seed **14/14 复现**，检测序列与归档逐条相同）、`SEARCH_MISS_TIMELINES.md`/`.json`（12 个遗漏目标逐个时间线：类别 1 = 0、类别 2 = 0、类别 4 = 0、类别 3 = 12；含三条口径限定）、`SEARCH_DETECTION_CORRECTNESS_VERDICT.md`/`.json`（`no_defect_found`、`s1_matrix_allowed=true`、三视图 120 000 步 0 分歧 ⇒ **未走修复分支**）、`SEARCH_POLICY_COMPARISON_RESULTS.md`/`.json`（60 seed `1200000..1200059` × 2 臂 = **120/120 完整**；未发现目标 0.150 vs 1.150 个/episode、配对差 **−1.000** 个/episode CI [−1.333,−0.667]、逐 seed 35/20/5；停 1 与停 2 **都不成立** ⇒ `neither_criterion_met`、按"区间内"如实报未达判据；系统按时完成任务 5.833 vs 4.833 个/episode 只读作总系统效应）、`RUN_LEDGER.json`（计划块冻结 + 活账 158/160）、`SEED_COLLISION_AUDIT.json`（clean、未顺延改号）、`CONTROL_SEEDS.json`（无遗漏控制 seed 规则先行登记）、`ROUND_VERIFICATION.json`（`passed=true`）、`MANIFEST.json`（原始日志/流/step 记录的位置+大小+SHA，不入库）＋ 生产侧 `integrated_sar/search_probe.py` 与 `tests/test_search_probe.py`（**11 项**，在真实 `timemap.Environment` 上跑）；项目套件 **Ran 713 tests, OK(skipped=2)** |
| 真实 PILOC 未发现目标归因（上一轮：0 episode、0 训练、只读） | `runs/real_search_undetected_attribution_20260923/` | `SEARCH_UNDETECTED_ATTRIBUTION.md`（两批各自闭合 360/352/352 与 360/356、四桶判定、**目标每步移动**这条会改读法的事实、`search_coverage` 不是空间覆盖）、`UNDETECTED_GOAL_ATTRIBUTION.json`（12 行逐目标表：连通域/距起点/可见格数/距实测到达格 + 每行 `grade` 与 `why`）、`SCENARIO_GEOMETRY.json`（**120/120 seed 重建栅格经 `map_sha256` 认证**、两套视线判据 72 457 格对 0 分歧、未发现 vs 已发现对照）、`FIELD_INVENTORY.json`（四条所需路径在 **360/360** 文件里不存在、PILOC 源码证据按起点版本引用、工作树漂移为空）、`BATCH_RECONCILIATION.json`（四臂发现集合逐 seed 相同、`detected+remaining==placed` 120/120、目标移动性）、`ATTRIBUTION_VERIFICATION.json`（`passed=true`、600 输入重哈希 0 变 0 缺、后端从未导入、tripwire 调用 0）、`MISSING_FIELDS_AND_PASSIVE_RECORD_PLAN.md`（B1 最小被动记录 + 2 个正确性候选的最小复现与修复计划，**未执行**）、`SEARCH_SIDE_PAIRED_VALIDATION_PROTOCOL.md`（只改搜索侧因素的配对验证，判据与停止条件先写死，**未执行**）、`test_rs_harness.py`（19 项 0 episode） |

传输层的两个协议由 `--comm-timing-mode` 选择：`legacy_shared_bound`（默认，全部历史实验，
上界到点由传输层代发、计费、可拒付）与 `policy_owned_timing`（研究模式，只有策略自己请求的
发送才上空气）。协议差异的逐包证据与训练判据见上表的协议消融行。

反向 ACK 信道由 `--comm-ack` 打开（默认关闭），合同见 `COMM_ACK_CONTRACT.md`：
ACK 只来自接收端真实应用的上行包，经有延迟/丢包的反向信道回传，每次尝试按序列化字节
全额计费并计入与上行并列的总成本。四项信息价值门禁全部通过才允许训练；
未通过即交付负结果并停止，不提高丢包、不收紧 deadline、不改场景制造信号。

## 协作方式

不要直接在 `main` 开发。每项工作建立独立分支，例如：

```bash
git switch -c feat/native-detection-event
git add -p
git commit -m "feat: emit detection events from PILOC"
git push -u origin feat/native-detection-event
```

通过 Pull Request 审核后再合并。详细规则见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
