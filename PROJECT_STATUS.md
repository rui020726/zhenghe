# 项目状态

更新时间：2026-09-19（同日第二轮：通信账本修正、字节预算执行器与统一静默上界、预算标定与三 seed 学习型发送 pilot）

## 已完成

- 阅读并确认 PILOC 搜索主流程与 task-03 的 `old_*` CTDE 主流程。
- 建立 `integrated_sar` 独立集成层，未侵入两个原项目。
- 定义发现、任务、协调建议、分配命令和进度反馈契约。
- 实现同步事件总线和幂等任务注册表。
- 实现无协调基线与规则残差协调器。
- 实现全局时钟和事件驱动联合调度器。
- 实现 PILOC 兼容适配器，可将当前“发现即删除”行为重建为事件。
- 实现 task-03 适配器，可注入外部多阶段任务、叠加协调 bias、执行并返回进度。
- 端到端演示能够完成 `发现 → 分配 → 执行 → 完成`。
- 统一场景生成器 `scenarios.py`：障碍图生成与加载（`.npy`/图片/文本）、最大连通域筛选、
  稳定 `target_id`、`CoordinateTransform`（`x = col/(width-1)`、`y = row/(height-1)`）。
- 仅推理联合入口 `joint_driver.py`：统一全局时钟、命令行参数、错误分类与 JSON 摘要。
- 搜索侧运行时：`search_runtime.py` 提供仅依赖 NumPy 的 `MockGridSearchEnv` 与前沿贪心策略，
  `piloc_runtime.py` 在有权重时按 strict 模式加载 PILOC 策略。
- 任务侧运行时：`task03_runtime.py` 按 strict 模式加载 CTDE 权重，并在加载期探针中
  拒绝产生 NaN logits 或违反环境硬 mask 的权重。
- 分配闸门 `assignment_gate.py`：TTL 过期与 `task_revision` 不一致的命令不进入执行，
  受影响 agent 转为重规划请求。
- 生命周期日志 `lifecycle_log.py`：JSONL 输出 12 个必需字段，可按 `target_id`/`source_event_id`
  追踪单个目标从发现到完成的完整轨迹。
- 无 checkpoint 的 mock 模式可在仅安装 NumPy 的机器上跑完整个 episode，且同一 seed 结果逐字节一致。
- 修复 `JointSAROrchestrator` 的注入缺陷：`registry or TaskRegistry()` 与 `clock or GlobalClock()`
  因 `__len__`/`__float__` 返回假值而静默丢弃调用方实例，已改为显式 `is None` 判断并补回归测试。
- 经一轮独立代码审查后修复的问题，每项都有对应回归测试：
  - 检测半径原本有三个真值来源（mock 环境默认 5、策略与适配器用配置值、真实 PILOC 用
    `obs_radius`），半径不等于 5 时既会漏任务也会给同一物理目标发多个 id；现在全链路共享一个值，
    真实模式下非 5 直接拒绝启动（不变量 2、3）。
  - 协调残差原本加在环境硬 mask 之后，只因内置 bias 恰好被 clamp 到 ±1 才没出事；
    现在残差写完再 `masked_fill` 一次（不变量 6）。
  - 被拒命令原本要等到协调周期才重规划，且消费后 `replan_agents` 留在缓存 advice 里导致每 tick
    重复下发；现在下一 tick 恰好重规划一次（新增 `ReplanSource` 端口）。
  - actor 与协调器原本直接读环境真值：新任务被广播进每台机器人的 knowledge，启发式分配读
    `task_dic`，前沿策略读全体真实位置。现在检测只进真值，启发式与协调器只用该机器人自己的
    belief，分离项改读 PILOC 本就共享的信息素（不变量 9）。该轮曾把“谁听说过任务”完全交给
    `comm_range` 视距感知加一跳 gossip，代价是远距离目标永远无法分配；第二轮已用显式通信桥
    （CommRelay）补上搜索侧到救援侧的消息通路。
  - 配套修复：`TaskEnv` 默认把救援队集结点随机撒在单位正方形，belief 收口后两队可能永远不在
    通信范围内导致闭环死锁，现在集结点落在搜索队出发格上，两队共享同一灾区。
  - 非整数 `dt` 时 mock 搜索环境仍按 1.0 推进，三个时钟分裂；现在统一按 `dt`，
    真实 PILOC 无法配置故要求 `--dt 1.0`。
  - 场景 `load()` 进 PILOC 后残留构造函数随机起点留下的信息素与位置历史，真实策略会绕着
    没人去过的格子走；现在按构造函数同样的单遍顺序重建。
  - 生命周期日志的 progress 记录原本写自己的 event_id，无法与 detection/assignment/final 串联；
    且日志消费者不幂等，同一事件重发会写两行。两者均已修复。
  - 复用适配器开新 episode 会重复 progress event id 并被注册表当重复丢弃；现在按 episode 命名空间隔离。
  - `ProgressEvent.failure_reason` 从未填写，任务过期或超时在日志里也是空白；现在会给出原因。
  - 注入任务的 `task_source` 由 `"piloc"` 改为 `"discovered"`：前者不在 `old_task_env` 的白名单里，
    会静默把 `dynamic_tasks` 观测特征喂错给网络；溯源信息保留在 `external_task_id` 与 `discovered_by`。
- 经第二轮独立代码审查后修复的问题，每项都有对应回归测试：
  - 新增 `integrated_sar/comm.py` 的 `CommRelay` 作为搜索发现到救援 belief 的显式通信桥：
    检测只进真值曾导致远距离目标永远无法分配（救援机器人必须亲自走到目标附近才能“感知”
    到一条 PILOC 已经发出的消息）。现在事件总线把检测扇出一次，中转按注册表已接受的快照
    生成消息，经显式拓扑、每包延迟与丢包率（seed 驱动）投递；不读任何人的位置，无条件广播
    不会回归。`old_task_env` 的 gossip 改为基于 tick 起点快照，一次最多传播一跳（此前原地
    修改集合会让 A 的新知识在同一 tick 经 B 直达 C）。
  - `get_local_obs`/`get_hard_mask` 曾继续读取 `task_dic` 真值（阶段索引、未来阶段需求、
    复杂度和若干 status/location/deadline 回退）：机器人 belief 时间戳不变、只改真值就会
    改变 actor 输入甚至翻转硬 mask。现在观测与 mask 全部来自该机器人自己的 belief 副本，
    belief 新增阶段索引、未来需求与复杂度三个字段并由所有 belief 路径统一维护；观测维度
    不变，checkpoint 仍按 strict 加载。
  - 通信消息到达或 belief 变化成为显式调度信号：`ExecutionStepResult.belief_changed_agents`
    由内容指纹（不含裸时间戳）计算，orchestrator 在同一 tick 立即重算 advice 并分配，
    不再等 `allocation_interval`；知识传播绝不伪装成 `ProgressEvent`。
  - 公开 reset 现在恢复完整 episode 状态：TaskEnv 按 seed 重置并重新落位集结点、
    mock 搜索环境恢复初始位置/隐藏目标/信息素/步数（真实 PILOC 路径重载场景而不是
    `Environment.reset` 的随机图）、中转清空 outbox、logger 清空去重状态、JointRun 计数器
    清零。此前 reset 后机器人带着旧 `target_id` 走到已删除的任务上会 `KeyError`；
    且 `DeterministicTargetProfileProvider` 曾按含 episode 号的 track id 哈希任务画像，
    reset 后同 seed 复现会因任务难度不同而分叉，现改为按物理格点哈希。
  - 注册表区分 seen/accepted/rejected：迟到、未知任务、终态后的非法事件不再计入
    `accepted_event_count`，logger 只记录注册表已接受的规范事件（审查复现的
    “Registry=assigned 但日志多一条 completed”不再发生），`source_event_id` 仍指向
    最初的 DetectionEvent。
  - mock 前沿策略的信息素惩罚裁剪窗以候选位置为中心，曾读到当前视野外一圈；现在先把
    信息素遮蔽到机器人自身视野窗口再打分（与真实 PILOC 策略读局部窗口的方式一致），
    视野外变化不再影响动作。
  - `--detection-radius` 为负会在 `RunConfiguration.validate()` 阶段以退出码 2、
    单行 `ConfigurationError` 拒绝（此前会带着 NumPy traceback 崩溃）；0 明确合法但退化。
- 经第三轮独立代码审查后修复的问题，每项都有对应回归测试：
  - 真实 PILOC 的 reset 只调用了 `Environment.load`，第二轮遗留的信息素/位置历史/
    时间戳地图/通信统计/延迟消息全部存活，且两个 RNG 都没有重新播种；现在
    `reset_piloc_environment` 统一完成 reseed + load + 派生状态重建，构造与 reset 走同一路径。
    适配器的 `_untracked_counter` 与协调器的 advice revision 也随 episode 复位。
  - “只读 belief”此前只覆盖了 observation 与 mask：协调器仍按注册表真值给 pair_bias
    打分（复现：belief 位置固定为 (0.9,0.9)、只改注册表到 (0.1,0.1)，pair_bias 从 0.335
    变成 0.615），`apply_action` 用真值位置导航（复现：机器人直奔从未收到的 (0.25,0.25)），
    在途重规划也用真值状态。现在 `AgentSnapshot.task_beliefs` 携带每台机器人收到的任务视图，
    协调偏置、导航目标与在途重规划全部只读它；真值只在到场核实时使用。
  - belief 变化指纹只比较状态/成员/阶段/完成标志，位置、截止时间、未来需求、复杂度、
    进度、动态与已分配标志的变化都不触发调度；现在指纹覆盖所有进入决策的字段，
    同时仍然排除裸时间戳（机器人站在闲置任务旁的刷新不算学到新东西）。
  - `CommRelay` 没有读取注册表判定，被拒绝的 late/terminal 检测仍会入队并写入机器人 belief
    （等于让被拒事件重发旧任务视图、把终态任务重新写成待处理）；现在中转只转发注册表
    已接受的事件，orchestrator 的知识调度信号也改为比较 `accepted_event_count`，
    被拒检测不再触发协调。
  - 生产 wiring 此前把所有救援机器人无条件放进上行链路，拓扑只存在于手工单元测试里；
    现在 `--comm-subscribers` 支持 `all`（默认全队）、`first:N` 与显式 id 列表，
    未知 id 与非法规格都以退出码 2 拒绝，并有集成测试覆盖收窄后的实际投递。
  - 合法 re-detection 的日志硬编码 stage 0/progress 0/status detected 并把 source_event_id
    换成当前检测 id，与注册表状态（如 assigned）矛盾且破坏同一目标的关联键；现在生命周期
    字段全部取自注册表快照，当前观测的事件 id 放进 `detail.detection_event_id`。
- 经第四轮独立代码审查后修复的问题，每项都有对应回归测试：
  - 协调器进入在线决策时仍读注册表：它用注册表的活动任务池决定哪些 belief 参与打分、
    是否触发 replan，并直接读注册表的 `confidence`。复现：belief 完全不变、只改注册表
    置信度，同一台机器人的偏置就从 0.785 变到 0.695；注册表把任务置为终态后，尚未收到
    消息的机器人也被判定需要重规划。现在 `advise` 只读 `AgentSnapshot.task_beliefs`：
    任务是否已结束由 `belief.finished` 决定，重规划只在“机器人自己的知识说承诺已结束、
    或它对所承诺的任务一无所知”时触发。
  - 任务优先级此前被 `max` 聚合进池级 `task_bias`，而这张表会发给所有机器人：复现中
    只把机器人 A 的 belief 截止时间从 1000 改成 10，机器人 B 的 belief 与 pair_bias
    都没变，总偏置却从 1.075 涨到 1.525。现在优先级按机器人分开发布
    （`CoordinationAdvice.agent_task_bias`，`bias_for` 累加池级/按机器人/配对三项），
    并新增 `AgentTaskBelief.confidence`，置信度只随消息到达写入 belief。
  - `DetectionMessage.progress` 在构造 belief 视图时被丢掉，`learn_task_view` 回退默认 0.0：
    注册表进度 0.833 的合法 re-detection 投递后机器人 belief 仍是 0.0，协调器继续按
    “未开始的任务”打分。现在 progress、confidence、revision 随消息一起落盘。
  - 同时间戳的注册表合法更新进不了 belief：注册表只拒绝严格更旧的事件，belief 却用 `<=`
    丢弃同刻更新。复现：第二个事件已被接受、revision 变成 2、位置变为 (0.2,0.2)，
    机器人仍保留 (0.9,0.9)。现在 `DetectionMessage` 与 belief 都携带 revision，
    投递按 `(sent_at, revision)` 共同排序：更新则接受，同刻且版本更高则接受，
    更旧或同刻同版本则丢弃。
  - `ExecutionStepResult` 新增字段插在 `metrics` 之前，破坏了仓库自己的位置参数调用：
    `demo.py` 把 metrics 字典当成了 `belief_changed_agents`，导致每个 tick 都被判定为
    belief 变化、协调器 revision 变成 1/2/3 而无视配置周期。现在新字段追加在 `metrics`
    之后，`demo.py` 改用关键字，并有契约测试锁定字段顺序。
  - 真实 PILOC reset 测试用默认 8×8 场景构造环境，而生产构造器明确拒绝非 60×60 场景，
    安装 gym/Pillow 后测试会在 reset 断言前抛 `CheckpointIncompatibleError`。现在真实
    用例改用 60×60 场景，skip 条件覆盖 timemap 的全部导入依赖（gym、matplotlib、Pillow）
    及实际导入失败，另增一组 stand-in 测试覆盖生产构造路径（60×60 守卫、构造、
    派生状态重建）与越界场景被拒绝。
- 经第五轮独立代码审查后修复的问题，每项都有对应回归测试：
  - 上一轮把“机器人仍相信一个已被注册表关闭的任务”整体判断为既有缺陷，这个判断不完整：
    单独只剩终态任务时的死循环确实早已存在，但协调器改为按 belief 打分后，会给这个陈旧的
    终态 belief 较高偏置，从而新增“抢占并饿死仍活跃任务”的行为。复现（一个已关闭任务 +
    一个仍活跃任务）：`02d29e9` 连续多轮都选关闭任务、`task_revision=None`、连续
    `missing_task_revision`，活跃任务始终拿不到执行；`1a41471` 选活跃任务因而正常。
    现在分配器用自己的缓存给命令盖上该任务的终态 revision（`sync_tasks` 每 tick 记录所有
    已注入任务的 revision，含终态），命令因此通过闸门：机器人按自身 belief 前往现场，
    由 `verify_task_on_arrival` 在地面了结，belief 随即跟上，活跃任务在同一个闭环里被选中
    并完成（新回归测试：第 1 轮接受 → 现场核实 → 第 2 轮改选活跃任务 → 完成）。协调器仍然
    不读注册表，闸门语义未变。
  - belief 已存 `task_revision`，但“双方都知道同一任务”时的 gossip 仍只比较时间戳：同一
    时刻、revision 2 的合法更新到不了 revision 1 的邻居（复现：两台相邻机器人时间戳相同，
    位置分别为 (0.8,0.9) 与 (0.9,0.9)，gossip 后接收方仍是旧位置与 revision 1）。现在按
    `(时间戳, revision)` 比较，只有严格更大才覆盖，双方一致时不搬迁。
  - 上一轮修 `ExecutionStepResult` 位置参数兼容时，又在 `AgentTaskBelief`（`confidence`/
    `revision` 插在 `future_requirements` 之前）与 `CoordinationAdvice`（`agent_task_bias`
    插在 `pair_bias` 之前）上犯了同一错误：旧的位置参数调用会把需求元组绑成 `confidence`
    （协调器随后对浮点数乘元组）、把 `replan_agents` 绑成 `pair_bias`（`bias_for` 对
    frozenset 调 `get`）。现在两个新字段都是 keyword-only 且追加在原字段之后，并有测试
    锁定旧的构造顺序、keyword-only 约束与三项 bias 求和。
  - 真实 PILOC 测试把 `import_timemap_module` 抛出的任意异常都转成 skip：依赖齐全时，
    语法错误、API 回归或初始化异常都会被报告成“缺依赖”。现在只在 gym / matplotlib / Pillow
    确实缺失时跳过，其余导入失败直接让测试失败。
  - `ARCHITECTURE.md` 架构图仍画着 `Registry → 全局任务态势 → Coordinator`，与正文“在线
    协调器只读 belief”矛盾，后续 agent 可能照图把真值泄漏引回来；已改为
    `AgentSnapshot + task belief → Coordinator`，并写明注册表只向分配器供数
    （闸门校验与执行注入）。
- mock 的双通道地图与 `timemap.Environment.two_channel_map` 逐格一致（含 `[0, 0.3]` 两端边界）。
- 训练入口工程化与训练正确性修复（步骤 0–2）：
  - `timemap.Environment` 的 `self.map` 只在 `test=True` 分支赋值，而 `map_partition(self.map)`
    无条件执行，默认参数下直接 `AttributeError` —— `piloc/train.py` 根本启动不了。
    现按 `reset()` 的同款逻辑在构造函数里生成障碍图；这同时让
    `build_piloc_environment` 里为绕过它而写的 `__new__` + 预置场景地图两步构造变得多余
    （构造函数本来就会覆盖那张图），现已改为直接构造。
  - `piloc/train.py`：device 可配置（原硬编码 `cuda`）、占位保存目录改为 `--output-dir`
    并自动创建、修掉 `episode_num % 1000 == 0` 在 episode 0 每步都存盘的缺陷、
    `--seed` 统一驱动 random/numpy/torch 与动作空间、`--episodes` 生效（原 `max_episodes`
    读了不用）、`--resume` 走 strict 加载而不经过硬编码 device 与文件名的 `MAPPO.load`、
    每回合重建环境时经 `build_search_environment` 统一补齐 `--num-goals`（原先只传
    `num_agents`，第二回合起静默退回 `configs.num_goals`，与 `run_meta.json` 记录的世界不一致）、
    每回合搜索指标写入 TensorBoard 与 JSONL、每次运行写 `run_meta.json`。
  - `task_allocation/old_driver.py`：`D:\tb_logs` 改为 `--log-dir`、`RESUME` 默认关闭、
    补 `--seed`、网络维度收敛到 `NETWORK_DIMS` 并随运行记录、三处 checkpoint 写入统一走
    `save_checkpoint` 并内嵌 `actor_kv_mode` 与 `run_meta`、`--episodes` 可界定训练长度；
    集成层加载端会拒绝 `actor_kv_mode` 与请求不符的权重。
  - CTDE 训练正确性三处：critic 从未收到 padding mask（rollout 与 update 的 pad 宽度不同，
    同一状态得到两个 value，critic 目标不可表示）；全程没有 `eval()`，dropout 使
    `old_logprobs` 来自未记录的掩码、行为策略不可复现；GAE 沿着 round-major/agent-minor
    的扁平 buffer 走，把同一回合的下一台机器人当成下一时间步，`V(s)-V(s)` 相消导致
    没有 bootstrap，并把别人的奖励折进自己的优势。
  - 两个入口都在 CPU 上冒烟跑通：PILOC 3 回合产出的权重经 strict 加载驱动联合闭环到
    `search_backend=piloc-trained` 并完成一个目标；CTDE 一轮 6 个 episode 产出
    `ctde_final.pth`，记录 `actor_kv_mode=tokens`、seed、commit 与网络维度，并通过加载期探针。
- `piloc/src/DDQN/Agent.py` 首行的 `import tensorflow as tf` 是死导入：整个 `src/DDQN`
  只有这一处出现 `tf`，真正使用它的 `ModelStats.py` 与 `base/BaseDisplay.py` 都不在
  `train.py` 的导入链上。它却让任何训练运行都必须装 tensorflow，也挡住了只有 CUDA torch
  的环境；已删除，`piloc_runtime.import_agent_module` 的报错文案与测试的依赖探测同步去掉
  tensorflow（`requirements.txt` 本就把 tensorflow 归在 legacy 组，无需改动）。
- 集成测试共 156 项：在 torch / gym / gymnasium / Pillow / matplotlib 齐备的环境里通过，
  仅 2 项“无 GPU 时回退 CPU”的用例因检测到 GPU 而按设计自我跳过。
- `piloc/train.py` 已在 CPU 与 CUDA 上各以 3 回合冒烟跑通，回合边界的环境重建路径实际执行，
  `run_meta.json` 记录了命令行传入的 num_agents / num_goals 与 device / cuda_available /
  torch_version。
- 奖励构成诊断（奖励重构第 1 步，行为不变）：`timemap.Environment` 新增 `track_reward_terms`
  开关与 `last_reward_terms`（默认关闭），`train.py` 加 `--track-reward-terms`，把每步奖励拆成
  9 个分量、按回合累加写进 `reward_term/{agent}/{term}`；新增 `test_piloc_reward_terms`，用
  “各分量之和恒等于 `step` 返回的奖励”锁住这个不变量（另有 2 项断言该开关确实传到环境）。
  实测（250 步回合、随机动作）：`reexploration_shaped` +231 与 `reexploration_baseline` −250
  几乎完全相消，`exploration_new` 甚至可为负（−16.5），另有 collision −27/−48、
  `teammate_proximity` −14、repeat −2.5/−4.1、pheromone ≈ −2；同一环境短回合（10 步）合计
  +78、长回合 −7/−102 —— **累计奖励随回合变长的塌陷是直接量出来的，不是学习退化**。
  另实证：前五个目标没有任何发现奖励——180 次 `update_timestamp_map` 调用中剩余目标出现在
  视野内 0 次（检测并删除目标的判据严格覆盖更新时的判据），所以 `find_target` 分支恒不触发。
- 奖励重构第 2–5 步（本轮完成，详见 `REWARD_AND_TRAINING.md`）：
  - **目标发现流程修复**：`step()` 改为先基于本步目标快照计算被任意机器人观测到的
    目标集合（同一目标两人同看只计一次），产生发现事件后再统一删除；中间发现立即计酬；
    `finish` 累加不覆盖；`done` 只剩“全部目标完成”一个语义。`nx != x and ny != y`
    修复为 `or`（只排除中心格），修复后的旧格统计只留在 legacy；
    `reExploration_reward_find_target` 跨 agent 泄漏与恒不触发的死分支已删除。
  - **Reward V2**（`piloc/src/Map/reward_modes.py`，`--reward-mode v2` 为默认）：
    `team = 1.0*N_t + 2.0*ΔC_union + 3.0*all_found − 0.5/T_max`，
    `agent = team − 0.2*collision_i`；删除隐式 −1、队友邻近 −v 惩罚与旧格正奖励；
    repeat/pheromone 默认权重 0，信息素有 epsilon 与裁剪保护；
    `potential` 模式 `F = β(γΦ'−Φ)`，真实终止 Φ_next 归零、截断保留 bootstrap。
    `legacy` 模式逐位保留旧行为作对照（含 finish 覆盖与发现奖励恒 0 两个已知缺陷）。
  - **课程门控**（`piloc/src/curriculum.py`）：删除“历史最高累计奖励 + 200 回合”规则；
    每 `--eval-every` 回合在固定验证 seed 上跑确定性（argmax）评估，
    连续 `--required-passes` 次任务指标达标才升级，永不因等待自动升级；
    门控输入白名单拒绝任何 reward/return 键；`--curriculum off` 用固定回合上限；
    最优 checkpoint 按验证任务指标存 `best/`。E 只控制回合步数上限，已在文档写明。
  - **γ 与可复现性**：`--gamma/--lmbda` 暴露为 CLI（V2 推荐 0.99，legacy 对照 0.95）；
    `seed_all` 统一 Python/NumPy/Torch CPU/CUDA/动作空间 seed；环境拥有独立 RNG
    （视野噪声、目标移动、通信噪声不再消耗全局流），`reset_piloc_environment` 经
    `reseed()` 重放；验证 seed 默认 `seed + 1_000_000_007`，与训练流隔离；
    gamma/lambda/reward_mode/全部权重写入 `run_meta.json` 与 checkpoint 内嵌 `run_meta`；
    CUDA cudnn.benchmark 非确定性已在元数据与文档记录。
  - **失效逻辑清理**：`reward_fn` 的 `move/stay_on_goal/stay_off_goal` 死键删除；
    `exp_xishu` 相同分支合并；`repeat_check` 从 train.py 移除并注明原因
    （rescue 覆盖动作会使执行动作与保存的 log-prob 不一致，评估路径 test.py 保留）；
    collision、短周期循环、重复覆盖分别统计（`last_step_stats` +
    `collision_rate/repeat_ratio/recoverage_rate`）。
  - 测试：新增 `test_piloc_reward_v2`（15 项：发现立即计酬、双机器人同看只计一次、
    未完成不终止、finish 累加、并集覆盖计酬、重复看格不重复计、纯重复无正奖励、
    分量之和恒等于 total、信息素无 NaN/Inf、发现奖励与 E 无关、potential 公式级、
    legacy 可运行、step 接口兼容）、`test_curriculum_gate`（13 项：拒绝 reward 输入、
    等待不升级、连续达标只升一次、窗口重置、关闭课程）、`test_piloc_seeding`
    （5 项：同 seed 逐位复现、全局随机消耗不影响、验证 seed 存活、seed_all 覆盖三流）。
    该阶段集成测试 189 项通过；现为 369 项，见下文联盟预约阶段。
    v2/legacy/potential 三模式与断点恢复各跑通 CPU smoke；
    smoke 中 episode 3 实测 discovered=1、return=+1.8，中间发现奖励首次真正生效。
- 联盟预约接口与规则规划协调器（`feat/coalition-reservation-coordinator`，逐项证据见
  `runs/coordinator_baselines_20260917/RESERVATION_VERDICT.md`，上一阶段的两个基线见同
  目录 `VERDICT.md`）。本阶段不训练任何网络，PILOC、CTDE 与全部 checkpoint 冻结：
  - **契约扩展**：`CoordinationAdvice` 新增 keyword-only 的
    `reservations: Mapping[(agent, task), CoalitionReservation]`，默认空。`CoalitionReservation`
    至少含 agent、task、capability slot（需求向量的扁平单元索引）、slot dimension、
    该机器人 belief 的 task revision、issued_at/valid_until 与有界 score，构造时校验槽位
    非负、有效期不早于签发、分数落在 [-1,1]。`Task03Adapter` 把持约折进与 `pair_bias`
    相同的有界残差通道，仍在环境 hard mask 之前（不变量 6），因此 trace 的
    `final = ctde + offset + bias` 恒等式不变；无预约时残差算术逐位不变，旧 checkpoint
    与全部既有测试不受影响。
  - **规则规划器**：`CoalitionPlannerCoordinator`（`--coordinator coalition_planner`）在
    空闲机器人与任务开放槽位之间做确定性贪心 set-cover，每轮对全部候选针对**实时**缺口
    重新打分，因此冗余派遣被排除而不是被惩罚；槽位占用由单元索引递增保证互斥。它只读每台
    机器人自己的 belief 与团队自己的派单计划（`AgentSnapshot.current_task`），完全不读
    `CoordinatorInput.tasks`；非空闲机器人整体退出 pair 通道（比 busy 惩罚更强）；belief
    已判定过期的任务两个通道都不发。advice 的 `valid_until` 被压到最早预约到期之前，规划器
    每轮报告预约的 expired / superseded / cancelled 与被拒的冗余候选。`HeuristicCoordinator`
    未改动，默认策略仍是 heuristic。新增 37 项测试。
  - **trace 与指标**：决策 trace 新增 advice revision、该 advice 的全部预约、本机器人持约
    （含是否被采纳）与规划器每轮计数；`allocation_trace` 据此给出预约 planned/honoured/broken、
    到期与作废、detection→预约/到场/首次推进三段时间、按能力维度的 missing-capability 向量
    与超额派遣，规划器计数按 advice revision 去重。`arm_name` 增加 `coalplan` 前缀，
    `dev_sweep.py`（`--planner`）与 `heldout.py` 接入新 arm，残差预算与两个 pair 策略相同。
    新增 15 项测试。
  - **结论**：100 个全新 held-out seed（20000–20099）上，相对 `rs0.25`，expired 53→16
    （−70%）、completion 0.910→0.973（+6.3pp，配对 +0.0623±0.0310）、
    `short_group_idle_steps` 合计 −47%、`members_waiting_steps` 合计 −64%（中位数 54→13）、
    机队利用率最低（0.066 对 0.089）、完成时间 29.0 对 37.4 步。700 个 (run, seed) 逐事件
    核对检测流 0 不一致；100 次运行 0 次非法命令；既有 arm 复现上一阶段数字
    （`coal0.25−rs0.25 = +0.0051±0.0073`、`cal0.25−rs0.25 = −0.0185±0.0273`），
    说明契约与指标改动零回归。集成测试共 369 项通过（2 项按设计跳过）。
- 联盟协调器的可学习性门禁（`feat/learned-coalition-coordinator`，证据见
  `runs/learnability_20260917/LEARNABILITY_VERDICT.md`）。**门禁未通过，未训练任何网络**：
  - 同信息 oracle `DeadlineAwareCoalitionCoordinator`（`--coordinator deadline_aware`）与规则
    规划器逐位一致——11 615 条决策记录 0 处不同，160 个 seed 上过期与完成的配对差全为 0，
    可修复比例 0%（门槛 30%）。它的第一版从零构建联盟并要求"当下空闲机器人即时全覆盖"，
    比规划器的渐进补位更脆（legacy block 上过期 16→30），已改为"从规则计划出发，只退还
    预留组在 belief 预算内无法到达的任务，并把释放的机器人重规划到可救的任务"。
  - 逐目标归因（六类，确定性顺序）：legacy 16 个过期目标中 12 个是"预约/执行中断"
    （计划已承诺、分配器却另作决定 11 例，或已完成工作被丢弃 2 例）、2 个是"belief 缺失或
    迟到"；candidate-ordering / task-contention / capacity-conflict 三类均为 0。
    事后上界：`infeasible_with_travel = 0`，即每个过期目标在物理上都可避免——
    若检测后 5–40 步内有机器人到场且工作不被打断，100% 可救。
  - 结论：残余失败不在协调器的排序里，训练网络没有可学信号。给出三项建议（按损失大小）：
    ① 让已成立的承诺对执行侧有约束力（持有有效预约的机器人不应在每个分配周期被重新决策）；
    ② 把阶段工作时长写进 `AgentTaskBelief`（`DetectionMessage` 已携带、belief 现在丢弃），
       否则"行程+工作 ≤ 截止"根本不可判定；③ 其余按机队构成/分批派遣处理，而非协调。
  - 数据：20000–20099 降级为 `analysis_legacy`（上一阶段曾作为 held-out 报告，且本门禁设计时
    已读它），新增 `analysis_fresh` 21000–21059（门禁指定后录制）；split 互斥、含 checkpoint
    摘要与录制协议。异常已记录：两 block 过期率差一个数量级（16/588 对 1/355），不可合并。
  - 集成测试共 385 项通过（2 项按设计跳过）。
- 承诺执行连续性与实验隔离（`fix/coalition-commitment-continuity`，证据见
  `runs/learnability_20260917/COMMITMENT_CONTINUITY.md` 与
  `commitment_continuity.json`；不训练模型，不实现 stage duration）：
  - **决策 trace 完整化（schema `allocation-decision-trace/2`）**：每条决策同时记录
    `raw_choice` / `forced_choice` / `post_mask_choice` / `actual_assignment` / `command_id`、
    预约身份（`reservation_id`）、六项有效性条件（belief 活动、未过期、槽位仍需、
    on-menu、mask 合法、可继续）与首个失败的释放原因；trace 与命令由同一组值生成，
    禁止事后重推。指标拆分为 `counterfactual_override_blocked`（原始选择不同但承诺被
    执行——修复在起作用）与 `executed_override_unjustified`（有效承诺却未被执行——
    唯一有验收要求的指标）。schema /1 的冻结产物仍可按原语义分析。
  - **承诺身份**：`CoalitionReservation` 增加 `commitment_id`/`created_at`；规划器每轮
    重建计划，但对同一 (agent, task, slot) 的再次发布是**同一承诺的续订**——身份不变，
    仅有效期/revision/分数推进；只有槽位改变才生成新 id。
  - **实验隔离**：每个 arm 写入独立目录（`runs/<...>/arms/<arm>/`），arm label、目录与
    文件名三者必须一致；`run_signature`（arm、sticky、code commit、checkpoint 摘要、
    seed、流指纹、config hash、trace schema）写入 lifecycle log 与 trace config 两条
    记录。resume 仅复用“签名逐字段一致且含完整 final record”的产物，否则先隔离到带
    时间戳的 quarantine 目录再重跑；聚合前 preflight 逐文件校验签名/完整性/sticky
    元数据/config 一致性/配对检测指纹，并要求仅差 sticky 标志的两个 arm config hash
    不同——任一失败禁止生成 summary。本轮即靠它抓出并修复了驱动脚本的两个真实缺陷：
    sticky 命令拼接位于 return 之后（死代码）导致的“贴牌 advisory 运行”，以及
    分片在 resume 扫描之后二次应用导致的 1/4 运行量；混入的 /1 schema 产物与竞态窗口
    产物全部隔离重跑，最终 legacy 100×3 arm、fresh 60×3 arm 全部通过 preflight。
  - **结果（干净复跑后）**：legacy（analysis_legacy 100 seeds）advisory→binding
    expired 16→9（配对 −0.0700±0.0750），completion 配对 +0.0118±0.0126；7 个
    “幸存者”在 advisory 下全部是 `reservation_or_execution_break`，binding 下该类
    降为 0（剩余 9 个：6 个排序、3 个 belief 迟到）。fresh（21000–21059）1→0。
    binding arm：承诺 honour 99.9%/99.7%，`executed_override_unjustified` 两块均为
    **0**（legacy 阻止了 151 次、fresh 98 次 counterfactual override），0 非法命令、
    0 拒绝分配、最长空闲持约 13/9 个决策周期（死锁阈 20）。
  - **split diagnostics**：两 block 通信配置相同（delay 0、delivery 1.0、interval 5）、
    checkpoint 与 commit 相同；差异在检测时刻（中位 4.5 对 2.0）、assignment latency
    （26 对 4）与 delivery budget（72.1 对 60.3）——fresh block 的目标被发现更早、
    被服务更快，解释 16/588 对 1/355 的过期率差。
  - 集成测试共 435 项通过（2 项按设计跳过）。
- stage-duration 相关性门禁（基线 `fix/coalition-commitment-continuity@868d68a`，证据见
  `runs/duration_relevance_20260917/DURATION_RELEVANCE.md` 与 `DURATION_GATE.json`；
  **门禁未通过，Phase B 未开展**：`AgentTaskBelief` 未改动、未建
  `feat/belief-stage-duration-slack` 分支、无 `coalition_binding_duration` arm、不训练模型）：
  - **门槛**：binding 后 legacy 残余的 6 个 `candidate_ordering_error` 中至少 3 个可被
    duration-aware 排序救回，且不依赖 truth。实测 **0/6**（prune 与 rank 两种模式均 0/6，
    rank 反而让原本 completed 的 `20083/piloc-e1-target-3` 转为 expired，净 −1）。
  - **信息合法性**：五个 seed 从各自 commit 的 `run_config` 复跑，5 645 条决策逐字段与已提交
    trace 比对 **0 差异**（挂载录制钩子后仍然成立）；时长只取该机器人自己 accept 过的
    `DetectionMessage`，并按 `first_received <= now` 门控；`CoordinatorInput.tasks`（注册表真值）
    从未被读取，钩子只记录 `typed_input.agents`。`CommRelay._build_message` 原样复制
    `snapshot.stages`、`_inject_task` 写入 `"time": float(spec.duration)`，因此消息里的时长与
    环境倒数的是同一个数——无需任何估计，也未创建第二套时长语义。
  - **为什么无信号（三条实测理由）**：① 六个目标在其 belief 预算内的最佳余量是 44–97 步，
    可行性判据永不触发，784 次合法出现中 0 次被提前；② 784 次里只有 31 次（4%）发生在目标
    仍处预算期内，此时 duration 项的唯一作用是“更早放弃”而非“优先做”，其中 14 次分配器本来就
    选了它（`20008/target-2` 的 6 次全部选中）；③ 五个目标死在不足一个 tick 上（还欠
    0.24–0.71 步覆盖工时，最后阶段完成度 91.9–97.7%），第六个从未获得覆盖。真正的分化变量是
    **首个覆盖 tick 相对检测的延迟**（4、9、42、66、157 步，预算仅 62.7–105.6 步）。
  - **结构性原因**：deadline 由场景生成器按 `travel + total_work + 30 + 0.3·total_work + U(0,50)`
    再乘 tightness 生成（`task_allocation/old_task_env.py:210-216`），即“行程+工作 ≤ 截止”本就是
    场景构造性质而非需要分配器发现的约束；这与 learnability 门禁中 `deadline_aware` oracle
    修复 0/16 是同一堵墙。
  - 集成测试仍为 435 项通过（2 项按设计跳过），本阶段无源码改动。
- 联盟成型与连续覆盖分析（基线 `fix/coalition-commitment-continuity@868d68a`，分支
  `analysis/coalition-formation-latency`，证据见
  `runs/coalition_formation_latency_20260918/COALITION_FORMATION_VERDICT.md`）：
  - **六个"ordering failure"的标签是错的**。五个目标实际交付了**全部**工时（whole-task
    progress 1.0），只是因为最后一 tick 落在截止线后被记为 expired——超时分别为 1.3、2.4、
    7.1、10.4 与 85.9 步（四个是险些错过，`20003/target-1` 因 157 步才凑齐覆盖而从未接近）；
    第六个（`20003/target-0`）从未获得完整覆盖。六个目标全部 `belief_spread = 0`（检测当
    tick 每个机器人都已收到消息）、0 个可用机器人被占用、0 次覆盖重置、0 工时被丢弃。
  - **primary cause**：`arrival_desynchronised` ×3、`stage_transition_handoff` ×2、
    `planner_dispatch_late` ×1。两个是阶段交接失败（`20008/target-2` 阶段 1 完成后等 85 步、
    `20070/target-2` 等 66 步）。
  - **反事实门槛（A/B/D 为解析可行性估计，非实测策略；不重放模拟器，对其他目标的全局
    影响未测量）**：A earliest-belief dispatch 与 B synchronised arrival 各 6/6
    potentially feasible，C stage-scoped lease 0/6（无覆盖可保护），E final-tick credit
    0/6（五个迟到者超时 1.3–85.9 步，无一落在单 tick 边界窗口内，从未完成者不适用）；
    D instant communication 与 A 同公式同派发假设，在 `belief_spread = 0` 下通信-only
    增量逐目标实测为 **0.0**，其 6/6 继承自 A 的派发时机假设，**不得记为通信收益**，
    且禁止进在线 actor。按序门槛选中 10B assembly scheduler。
  - **原型 `coalition_assembly_scheduler`（默认关闭）小门禁未通过**：5 个 seed 上 baseline 与
    candidate 完全一致（completed 21、expired 8、reset 1、lost work 1.2，0 override、0 拒绝、
    0 死锁）。同步项**确实生效**（seed 20003 的 1085 条决策中 13 条 bias 被抬高恰好 0.0375），
    但**未翻转任何选择**——分配器自身 logits 的间距远大于该有界项。这仅说明该原型未作用到
    控制路径，训练是否可行暂缺充分依据；按 Phase 12 第 7 步停止，未跑批量评估。
  - **deadline tick 语义**已用 18 项测试钉死（`test_deadline_tick_semantics.py`）：覆盖中断时
    `time_left` 重置为**完整** stage 时长、只有已到场成员计入覆盖、多余成员不提速、stage 时长
    固定、deadline 用严格 `>`。发现一处真实 off-by-one（`task_update` 先判过期后结算工时，
    导致"恰在跨线 tick 完成"的任务被报成 expired），**仅记录未修复**。
  - 集成测试 476 项通过（2 项按设计跳过）。
- 事件触发派发审计与冗余成员释放（基线同上，分支
  `fix/event-driven-coalition-dispatch`，证据见
  `runs/dispatch_trigger_audit_20260918/DISPATCH_TRIGGER_AUDIT.md`）：
  - **首个阻塞点（逐 tick 实测，instrumentation-only 探针复跑并与已提交 trace 逐位核对：
    5 seed 决策行与 final 全同）**：六个失败目标在缺口打开期间，能补齐且有 belief 的机器人
    全部处于 working 态 `need_decision=False`——planner 只规划空闲机器人
    （`coordinator.py` `_is_free`），allocator 跳过非 eligible（`task03.py:523`），唯一出口是
    U(50,120) 随机工作超时。六个目标的 pre-deadline 缺口 tick 中 408/412 由
    `capable_all_locked` 解释；belief/通信/能力/覆盖/承诺均被同一测量排除。
    14 个匹配对照呈同样的锁死画像（757 tick），失败只是 deadline 更紧。
  - **两个 stage handoff 案例（20008/t-2、20070/t-2）完整钉死**：阶段边界 keep 规则只要求
    "任一能力重叠"，把对下一阶段零贡献的成员留在 working 态；20008 出现跨任务互锁
    （target-2 需 dims 0/3 被 target-0 上的机器人占着，target-0 需 dim 4 被 target-2 上
    两台冗余 dim-4 机器人占着），靠随机超时 85 步后才解锁。五个 seed 共测得
    **2150 个零边际贡献工作 tick**（belief-only 可判定）。
  - **证据文档修正**：A/B/D 反事实改为 analytical feasibility estimates（potentially
    feasible / global_effect_unmeasured），D 与 A 同公式同派发假设、通信-only 增量逐目标
    实测为 0.0；按 completed on time / finished after deadline / never finished / 边界语义
    分列（旧文"超时 1.3–10.4 步"有误，实为 1.3–85.9）。
  - **修复（默认关闭，单一开关）**：`--coordinator coalition_redundant_release`——自身 belief
    判定对当前 stage 无可填单元的 working 成员可被重新规划，且仅当计划为其找到**不同**任务时
    才进 replan；allocator 侧默认关闭地抑制"重选自己正在执行任务"的重复命令（否则会重置工作
    时钟并解除超时武装，首跑实测 98 条 churn，抑制后为 0）。不触环境/通信/机队/checkpoint，
    不绕 mask，不扩可见信息。15 项新测试（默认关恒等、仅冗余且有去处才释放、被需要成员永不
    释放、重定向后承诺身份延续、确定性、mask/band 不变等），全套 **495 通过（2 跳过）**。
  - **闭环小门禁全部通过**（5 失败 seed × 2 臂，同录制流，`CLOSED_LOOP_GATE.json`）：
    completed 21→**30**（全部目标完成）、expired 8→**0**、reset 1→0、lost work 1.2→0、
    0 override、0 拒绝、0 churn、0 死锁；**6/6 获救**（5 个从"迟到完成"变按时，
    20003/target-0 从"从未覆盖"变按时完成），0 新增失败；baseline 逐目标复现 868d68a。
  - **配对扩展（§八条件满足后执行，冻结 commit `6915d63`，320 次运行、8 worker 分片、
    preflight 校验签名，`PAIRED_EXPANSION.md`）**：legacy 100 seed 上 baseline
    579/9 → candidate **588/0**（9 个过期目标全部按时完成，含 20003 上 5 个），
    reset 1→0、lost 1.2→0；fresh 60 seed 上两臂同为 355/0（无病无扰）。
    **两块 0 回归、0 新增失败**。两块均为开发已读集合，不作为 held-out 报告。
- 冗余释放的安全审查、机制拆分与独立测试（分支 `validation/coalition-redundant-release`，
  证据见 `runs/coalition_validation_20260918/`）：
  - **安全审查发现并修复 5 个真实缺陷**（`REDUNDANT_RELEASE_SAFETY_REVIEW.md`）：① 冗余判定
    各自独立后合并，两名互为冗余的成员可同时被释放而掏空任务覆盖 → 改为单趟确定性**联合核算**
    （每次释放先从能力池扣除）；② 填充池把 `moving`（未到场）机器人算作覆盖 → 池与候选均限
    `ONSITE_STATUSES={"working"}`（环境到场态），并明确不用 belief.members 作覆盖来源（其新鲜度
    受消息延迟限制）；③ `_build_coordinator` 中 planner 分支提前 return，使
    `release_redundant_members` 字段路径不可达（四臂结果完全相同即是症状）；④ arm 标签不编码两个
    新开关，四种臂共用 `coalplan0.25sticky` → 加 `+redrel`/`+noself`；⑤ 自有任务抑制复用了
    `commitment_suppressed_command` 使两种原因混淆 → 新增 `own_task_suppressed`，schema 升至
    `allocation-decision-trace/3`。另确认：一机器人一去处由 `_plan` 的 picked 集合保证（补断言）、
    被需要成员永不释放、被拒改派不虚报成功、不读注册表真值、不绕 hard mask。
  - **机制拆分消融（预注册 15 个开发 seed × 4 臂 = 60 局闭环，`MECHANISM_ABLATION.md`）**：收益
    100% 来自 planner 的冗余释放（81/8 → 90/0，9 个目标获救、0 回归，零贡献工作 tick
    5233 → 3213）；抑制开关单独存在时与 baseline **逐指标完全相同**（默认关恒等的端到端证据），
    其作用是把 C 臂 120 条"重选自己任务"的命令（会重置工作时钟、解除超时武装）变为 120 次静默
    decline；改派接受延迟均值 5.43 tick。
  - **独立测试（400 个全新 seed 30000-30399，协议与统计在看到任何结果前冻结，
    `HELDOUT_PROTOCOL.md` / `HELDOUT_VERDICT.md`）**：身份前置——两开关全关时在 5 个开发 seed 上
    逐字段复现 868d68a 时代轨迹（5645 条决策行）；800 次运行 preflight 全绿。主指标（逐 seed 按
    时完成率配对差 + seed 级 bootstrap 95% CI，B=10000、RNG 20260918）= **+0.008333，
    CI [+0.004167, +0.013750] → 显著提升**；计数闭合 2355 检出：baseline 2335 按时/20 过期 →
    candidate **2355 按时/0 过期**，两臂 active 0，**逐目标回归 0**；完成延迟均值 24.85 → 17.61；
    零检测 seed 为 0（预声明剔除规则未被使用）。**天花板效应如实报告**：baseline 已 99.15%，
    总余量仅 20 个目标，未补选 seed。正确性四项通过；candidate-only 拒绝 2 起（均
    `stale_task_revision`，gate 正常拒绝过期 revision 命令；baseline 同类事件本身有 11 起），未丢目标。
  - **完整系统闭环（60 个新 seed 31000-31059，真实 PILOC 在线搜索，不共用 stream，
    `SYSTEM_CLOSED_LOOP_VERDICT.md`）**：两臂搜索覆盖率完全相同（356/360），candidate 把 6 个过期
    清零、已发现目标完成率 98.31% → 100%；配对差 +0.0167、CI [0.0, 0.0389]（下界触 0 → 按预登记
    规则记为**改善趋势**，3 局更好 0 局更差）；0 拒绝、0 重置、0 重复命令、0 死锁；救援里程 +2.9%
    是机制的真实代价。60 局**发现序列逐局相同**——不变量 10 下搜索队不读救援态，分配无法反馈到
    发现；若未来合并两队或让搜索读任务状态，此阶段必须重跑。
  - **默认值与版本状态**：两个开关仍默认 False（`coalition_redundant_release` 仅为显式快捷策略）；
    稳定基线仍是 `fix/coalition-commitment-continuity@868d68a`，通过独立测试的候选是
    `validation/coalition-redundant-release@1f4a598`（其 `integrated_sar` 与 `304fc60`、`e96f34c`
    逐字节相同，`git diff` 为空）。**建议**：证据支持默认启用（独立块显著提升、四项正确性通过）；
    "620" 是该轮所有实验块去重后的 **seed 数**而非配对局数，且开发块彼此嵌套（gate 的 5 ⊂
    消融的 15 ⊂ legacy 的 100），只有 400-seed 独立块支持显著性表述——该计数已由
    `runs/release_validation_20260918/evidence_accounting.py` 从 manifest 重算并纠正。
    另：该轮结论所测的"冗余判定"在下一轮被查明读错了分母，其泛化声明只适用于那个旧判定。
    默认启用与合并由用户决定。
- 候选发布验收：信息来源、陈旧命令恢复与固定方案压力测试（分支
  `release/coalition-controller-validation`，起点 `validation/coalition-redundant-release@7e5ba3a`，
  证据见 `runs/release_validation_20260918/`）：
  - **版本与证据核对**：本地与远端 `7e5ba3a` 一致；测试执行点 `1f4a598` → 交付头 `7e5ba3a` 的
    差异 **不含任何 `integrated_sar` 生产文件**（20 个文件全为 .gitignore/PROJECT_STATUS/分析脚本与
    小汇总）。既有 400-seed 独立结果的协议、标签与统计口径未改。累计数字重算：全部块共
    **800 个去重 seed / 835 个配对场景 / 1700 次回放**，其中独立证据 460 个新 seed
    （400 回放 + 60 在线），开发块去重后仅 160 个 seed 且相互嵌套。
  - **信息来源审查发现一个真实缺陷并修复（`fb1c85b`）**：冗余判定用的是 belief 里的
    `remaining_requirements`，而该残差是环境**把这台机器人算进 members 之后**得到的——于是"独自
    撑住某需求"的机器人必然看到需求为 0，被判冗余。15 个开发 seed 实测：3556 次候选判定中
    3111 次残差全零；205 次真正执行的释放里有 **33 次会移走现场确实需要的能力**。旧结果之所以
    未出现覆盖回归，只因为环境对被改派的 working 机器人**仍保留在 members 名单里**（幽灵在场），
    这是环境侧既有行为，不能作为安全依据。改为按同一消息已送达的 **stage 总需求**比较"有它/没它"
    的覆盖，并加两条保守拒判（belief 无总需求、belief 名单不含本机），按原因计数进 metadata。
    全程未新增任何信息权限：所需字段本就随消息送达，只是从未被 surface 到 `AgentTaskBelief`。
  - **修复后的开发复测**：五 seed 闭环 baseline 21/8（1 重置、1.2 丢失工时）→ candidate
    30/0（0/0），9 个目标获救、0 回归；15 seed 消融仍为 81/8 → 90/0，但释放次数
    205 → 125、接受改派 86 → 51，即更少动作换到同样的收益。开关全关时仍逐字段复现 868d68a
    轨迹（5 seed 共 5645 条决策行 IDENTICAL）。
  - **陈旧命令拒绝审计（`STALE_REVISION_RECOVERY_AUDIT.md`）**：读完 800 份 held-out 日志，
    baseline 11 起 / candidate 13 起，**全部是同一形态**——任务在"写缓存 revision"与"闸门读
    registry"之间的同一 tick 内完成，命令带的 revision 恰好落后 1。四项指标分列：按原因
    拒绝（仅 `stale_task_revision`）、实际执行的非法命令 **0**、恢复延迟 **1 tick（24/24）**、
    重复陈旧命令尝试 **0**；拒绝不改机器人执行态、不丢既有工作、下一次合法更新即恢复，
    gate 与 hard mask 强度不变。结论为"行为正确，仅补证据与测试"，新增
    `test_a_task_completing_mid_tick_is_refused_once_and_then_travelled`。
  - **三条件压力测试（预注册后运行；`STRESS_PROTOCOL.md` / `STRESS_VERDICT.md`）**：A 更紧
    deadline（band 0.4-1.2）、B 更高负载（9 goals）、C 两者，各 60 个全新 seed（32000/33000/
    34000 段，排除全部历史 seed，块间互不重叠），共 360 次回放、preflight 0 问题。
    主指标为逐 seed 按完成率配对差，bootstrap B=10000，三条件同时给出 95% 与
    Bonferroni 校正 98.33% 区间：**B +0.0243，校正后 CI [+0.0019, +0.0579] → 显著**
    （12 迟到 + 1 未完成被救，expired 12→0，7 局更好 1 局更差）；A +0.0056
    CI [0, 0.0222]、C +0.0037 CI [0, 0.0148] **均触 0**，且两臂已 98.7-99.4%——按预登记
    规则记为"证据不足"，并如实报告 deadline 下限（0.95×估计）使该压力未能真正收紧，
    未补选 seed。正确性：0 非法执行、0 重复命令、0 无进展死锁，覆盖重置不高于 baseline
    （A/B 持平、C 1→0）。唯一新增失败 33011/target-7 经 trace 定位为两次**环境随机工作超时**
    掏空 dim-4 覆盖，该 episode 中"因释放而被唤醒且离开该目标"的决策为 0 条。
  - **释放安全性实测（180 局带记录器的复跑，复现存储结果 180/180）——未解决**：23311 次判定、
    1498 次实际执行，其中 **19 次（1.3%）实际移走了现场需要的能力**，全部属 belief 落后于
    stage 切换。**本轮不将其记为"残余风险"，而记为覆盖保护的失败**；且"这 18 个 seed 与出现
    覆盖重置/工时丢失的 seed 不相交"这一观察**不能为其开脱**：若环境仍把已改派机器人计入旧任务
    members，则这类释放本就不会产生可测的重置，指标在两种情形下都长成同一个样子。旧判定同口径为
    33/205（16%）且属系统性。
  - **成本**：里程 5.83→5.80 / 7.17→7.33 / 7.01→7.10（+2.3% 仅出现在取得显著提升的 B）；
    协调决策 p99 0.38→0.60 / 0.73→1.09 / 1.11→1.42 ms。
  - **配置防护**：`--release-redundant-members` 未配 `--suppress-recommand-own-task` 现在直接
    ConfigurationError，并给出 `--coordinator coalition_redundant_release`（成对快捷）或
    `--allow-release-without-suppression`（显式消融，保留 C 臂用途）两条出路，附测试。
    `deadline_tightness` 只是把 TaskEnv 既有参数接通过来（默认与其自身默认一致，有测试钉住），
    未改任何环境生成逻辑。
  - **发布判定（已更正）：发布阻塞——阶段信息陈旧导致错误释放，且环境旧任务成员残留可能掩盖
    覆盖损失。** 两个开关**仍默认 False**，本轮不建议默认启用、不作为默认版本合并；
    `RELEASE_CANDIDATE.md` 中原"全部正确性条件通过/推荐默认启用"的表述已撤销。
    稳定基线仍是 `868d68a`；`release/coalition-controller-validation`
    （`fb1c85b` + `fd0dd76` + `3fcef63`，交付头 `bed60f9`）保留为**待修正候选**。
    原 400-seed 独立块现标注为**旧候选版本证据**，其结论在成员一致性问题下是否成立尚待评估。
    上一轮的顺序已在 `fix/task-membership-consistency` 全部走完，见下条。
    `RELEASE_MANIFEST.json` 记录代码/模型/seed manifest/原始产物 fold hash；
    该轮全套 **513 通过（2 跳过）**，smoke `final_status=completed`。
- 成员一致性审计与修复（分支 `fix/task-membership-consistency`，证据见
  `runs/membership_consistency_20260918/`）：
  - **"幽灵贡献"确认成立**（`MEMBERSHIP_CONSISTENCY_AUDIT.md`）。区分五件事：在册、当前指派、
    物理在场、状态可服务、真被计入覆盖——环境算覆盖时**只看第一条**：`task_update` 用
    `max(requirements − get_abilities(members), 0)`，而 `get_abilities` 对名单里每个 id 求和能力，
    不看位置也不看状态；`apply_action` 改派时**从不**把机器人从旧任务 members 里摘掉
    （摘除只发生在超时/完成/阶段切换三处）。最小复现（两任务两机器人、无协调器）：改派后旧任务
    仍被视为已覆盖，机器人在场数为 0 的情况下工时继续推进 **20 tick**，同一机器人同时挂在两个
    任务名单上，旧任务在无人到场的情况下**完成**，而其完成时把当时正在新任务工作的机器人释放了。
  - 17 个含 19 次错误释放的压力 episode 带记录器复跑：232 次被执行的释放中 **185 次（80%）
    至少产生一次幽灵推进**，共 339 tick（单例最长 14），**87 次**存在跨任务重复计入，
    残留最长 34 tick；**177 个旧任务是在仍记名缺席机器人的情况下完成的**。全部结果与未插桩
    存储逐目标相同（180/180），故记录器仅作观测。
  - **成员一致性修复原型（`98a34b3`，默认关闭 `--task-membership-consistency`）**：被接受的跨任务
    /回场改派按确定顺序把机器人从所有仍列出它的名单上摘掉并重算这些 stage 缺口；重选当前任务
    成为真正的 no-op（不重置工时、不动名单）；未触碰 deadline、工作时长、随机超时与覆盖中断重置。
    10 项新测试断言的是**真实覆盖与工时推进**，不只是 members 列表内容。
  - **源任务阶段闸门（`7bbc3ac` + `d194c31`）**：advice 现按机器人发布 `ReleaseProvenance`
    （离开哪个任务、判定时该任务的 belief stage/revision），allocator 仅当去处不同于来源时随命令
    下发，既有校验边界比对来源 stage：不匹配即无副作用拒绝、保留原工作，**不静默升级到新 revision**。
    初版同时比较 revision，结果在开发集上误拒 **979 次**——注册表每 tick 都推进 revision，
    "消息更新"根本不是阶段一致性的证据；该比较已移除并补测试钉住。
  - **真正清零 19 次的是规划侧补的另一半**：冗余判定此前只看机器人所站 stage，现对 belief 里
    **下一阶段**同样比较（`future_requirements`，消息本就携带、协调器从未读取）。同一批 17 episode
    上，被执行的有害释放 **19/1498 → 0/118**，闸门自身只额外拒绝 1 次（开发集）。
  - **四格分离测量（`CORRECTED_ENV_DEVELOPMENT_RESULTS.md`）**：A 原环境+baseline、B 原环境+候选、
    C 修复环境+baseline、D 修复环境+安全候选。开发 15 集：A 81/90、8 过期、1 重置、1.20 丢失工时、
    **1** 幽灵 tick、**0** 重复计入；C 与 A **逐列相同**（环境修复对 baseline 零影响）；
    B 90/90 但 **156 幽灵 tick + 597 重复计入**；D 90/90、0 过期、0 重置、**0 幽灵、0 重复计入**。
    17 个压力 episode 同向（B 195/538，D 0/0）。因此：旧候选的收益**无法与幽灵能力区分**这一点
    已被排除——在诚实的环境下收益依旧存在，但只在 C vs D 的意义上成立，且样本仅 15+17 集，
    不作泛化。压力块唯一新增失败 33011/target-7 单独跑四格：B 8/9（1 重置、2.531 丢失，
    旧环境把它藏了起来），D 9/9；本轮把它**继续记在 candidate 名下**，未因环境修复而剔除。
  - **33011 首处分歧**：第 7 条决策行（candidate t=3 因释放被唤醒，baseline 直到 t=5 才决策），
    此后到 t=276 candidate 共发出 369 次指派 vs baseline 296 次——每次到场都要抽一次工作超时，
    抽样消费序列因此不同；直接致败的是 271/276 两次环境随机超时掏空 dim-4。本轮未改任何 RNG 机制。
  - **状态**：五个开关（policy、sticky、release、suppress、source-check、membership）**全部默认
    关闭**；不合并为默认版本，不撤销发布阻塞判定。下一步需要以修复环境为记录基线、在未读过
    seeds 上重做 C vs D 才能重新谈推荐。全套 **535 通过（2 跳过）**，smoke
    `final_status=completed`，开关全关时对 868d68a 轨迹仍逐字段复现（5 seed 5645 行 IDENTICAL）。

- 成员一致性收尾 + 修正环境独立验证（同分支，证据见 `runs/corrected_env_final_20260918/`）：
  - **C 臂剩下那 1 个幽灵 tick 归类为测量误判（分类 C）**，不是缺席贡献。定位到
    dev seed 20008 第 132 tick / task 3：工时更新时刻 roster `[2,4]` 两人都 `status=working`、
    `current_task=3`、到场距离 0.0，能力 `[1,1,1,2,0]` 覆盖需求 `[1,0,0,1,0]`，工时确实推进 1.0；
    这一步稍后 `_check_working_timeouts` 把 4 号超时释放，整步之后再读在场就只剩 2 号，缺口
    `[1,0,0,0,0]`。A/C 两臂逐位相同，本身就排除了释放机制与改派残留。
  - **口径修正保留双读**：`ghost_recheck.py` 把在场状态搬到 `task_update` 入口读，同时在同一次
    重放里保留旧口径以便对账。四格重测（15 dev + 17 stress）：C/D 的虚假能力工时、双名册
    robot-tick、"旧任务误释放他任务机器人"全为 **0**，旧环境 B 分别 141/183、597/538、85/82；
    A 与 C 依旧逐列相同。"执行后降低真实覆盖的释放"改用**边际贡献**判据（有他/没他在当前
    stage 是否严格少覆盖），seed 20083 tick 15 那次误报（3 号的 dim 0 被 5 号重复供给）随之消失，
    而 B 的 11/12 次真事件照旧计数。
  - **释放执行边界补上了"支援名单"这一半**：advice 的 `ReleaseProvenance` 现在带上判定所依赖的
    在场队友，命令随之携带，gate 在读得到注册表名册时按批逐条校验——队友已不在册
    （`stale_release_support`）、或机器人自己已不在被离开的任务上（`stale_release_agent_moved`）
    即无副作用拒绝；同批两条各自合法但合并抽空覆盖的，第二条被拒。六场景全部有测试钉住
    （`test_release_execution_boundary.py`），并显式记录边界读真值的权限：只用于拒绝、
    反馈仍走既有 `replan_agents`、不回灌规划特征、不做 revision 机械相等。
  - **三类"仍需要它"与"没有下一阶段信息"的释放终于分开计数**：stress D 为
    `stage_needs_this_robot 3689 / next_stage_needs_this_robot 490 / absent_from_believed_roster 181`，
    发布 118 而边界执行 44；执行后降低覆盖 0；`releases_without_future_information` 0。
    下一阶段保护未观察到长期占用（按时率与 C 相同、释放仍在发生），但空等时长这 32 集没测，
    留给独立块——独立块测到 **D 的空闲 robot-tick 多 1.4%**，这是该机制的真实代价。
  - **冻结两臂并预注册 400 seed 独立块**（先提交协议与清单，再跑）：seed 500000..500399，
    排除并集 833 个（扫 2 845 份 manifest + 全部产物文件名，摘要入册），两臂都启用修正环境。
    构造断言不看名字看对象（环境类与源文件、`membership_consistency`、gate 两个回调的接线、
    协调器/allocator 上的开关、arm 后缀、生产树与冻结提交的 diff）；arm 标签新增 `+memfix`
    使两个环境不再共用一个名字。
  - **独立块结果**：800 episode、400 对、0 技术失败、0 零检测对。目标闭合
    2346 = C 2303 按时 + 43 过期 = D 2341 按时 + 5 过期，未解决 0。
    **主指标配对差 +0.016125，95% CI [0.009250, 0.023959]（下界 > 0）**；
    373/400 集两臂都满，不匹配 26 集（D 好 25、D 差 1），平均完成时刻少 6.08 tick。
    正确性门槛（虚假能力、双名册、误释放、降低覆盖的释放、重复命令重置、对已结束任务写入）
    **两臂全 0**。唯一新增失败 500168（1.0 → 0.833）保留记录，未因"环境超时"剔除。
  - **一处口径自我纠错并如实记账**：预注册里写作"非法执行"的那项，第一遍给出 C 4 / D 6 判为不过；
    逐事件重放（`illegal_probe.py`，10 个事件全部列在裁决文档表格里）显示注册表当时已
    `completed/terminal` 而机器人 belief 尚未知道，命令于是把人朝刚结束的任务派出去，
    到场核实拒绝、**没有写入任何状态**。拆分后 `state_written_on_a_closed_task` 为 0，
    浪费行程 4/6 单独报告；是否要在边界直接挡掉 terminal 目的地**留给使用者决定**
    （挡掉会有把 belief 未更新的机器人钉在原任务上的风险）。第一遍汇总保留为
    `STATS_pass1_coarse_illegal_definition.json`。
  - **条件性在线块（60 个额外 seed，真实 PILOC 在线搜索）**：120 episode、60 对，
    **两臂逐 seed 的有序检测序列完全相同**（解耦不变量经验成立）。360 目标 / 发现 355 /
    按时 355 → 已发现完成率 1.0000、全部目标按时比例 0.9861，两臂**每一集都相同**：
    该难度下瓶颈是搜索覆盖而非分配，这块对性能差异**没有可分辨信号**，只支撑系统集成与
    真实搜索输入的检验，不支撑任何新反馈能力的说法。
  - **状态**：六个开关仍全部默认关闭，未合并、未重训、未删除旧产物；全套 **551 通过（2 跳过）**，
    smoke `final_status=completed`；`RELEASE_MANIFEST_V2.json` 记录冻结提交、checkpoint 摘要、
    配置真值表、seed 清单与全部原始产物的 fold hash。

- 2026-09-19 起：**研究方向转入通信控制（信息传递频率与更新时机）的定义、基线与可学习性验证**，
  新分支 `feat/communication-control-baselines`（起点 `fd06d45…`，生产代码内容仍等于冻结提交
  `a26e5b5…`）。产物在 `runs/comm_control_20260919/`。
  - **范围界定**（`COMMUNICATION_CONTROL_SCOPE.md`）：把"任务传递频率"落到链路 A——
    搜索端→救援 belief 的 `CommRelay` 上行；并逐条给出 A/B/C/D 四条链路的实际调用链、
    发送/接收条件与配置，明确**消息发送频率 ≠ 协调器调用频率 ≠ 分配重规划频率**。
    架构文档已把上行写为"唯一的消息通道"，故选 A：它有消息契约、队列、逐接收者投递与新旧合并规则；
    B（任务状态更新）今天根本没有消息对象，C/D 属于冻结范围之外不动。
  - **最小机制改造**：`comm.py` 拆成"记录被接受的新闻 + 每步由 `comm_policy` 决定发射哪些视图"，
    新增 `integrated_sar/comm_policy.py`（`immediate|periodic|event|event_maxwait`）；
    传输层对所有策略统一保证：载荷取发射时刻最新快照（晚发=更新，不是更薄）、每目标每步至多一次、
    终态任务不播报、超预算按统一顺序裁剪；新增 `--comm-policy[-params]`、`--comm-sends-per-tick`、
    `--comm-transcript`、`--comm-range`、`--gossip-interval`（后两个只是 `TaskEnv` 原有构造参数的
    透传）。**默认值全部等于历史行为**：`periodic/event` 才产生的刷新流量在默认下为 0。
    臂标签新增 `+comm<policy><params>` 后缀，`config_hash` 覆盖新字段，转录路径不入 hash。
  - **等价性被验证而非声明**：`identity_check.py` 用临时 worktree 把默认 C 臂在冻结提交与本工作树
    各跑 8 个开发 seed，逐记录比对生命周期日志（去掉运行标识后）**8/8 逐位相同**
    （每 seed 1 248–1 345 条记录）。
  - **信息边界实测**（`INFORMATION_BOUNDARY.md`、`probe_traffic.py`，12 seed × {发送/静默} ×
    {部署/隔离} = 48 episode）：默认配置下 belief 写入里视距刷新 177 335 次 vs 上行 408 次（435:1），
    **切断上行后 95.6% 的 (机器人,任务) 首次知晓仍由视距刷新建立**（65/68 目标仍被告知、64 个完成）
    ⇒ "暂停发送即信息不到达"在部署配置下**不成立**；隔离配置（`gossip_interval` > 步数，
    一个开关同时关掉视距刷新与一跳 gossip）下静默的 belief 写入为 **0**，才是可评估发送策略的设置。
    执行侧闸门保留且其真值不回流为策略观测；拒绝→重规划被确认为第二条（无载荷的）信息通道，
    实测陈旧 belief 会转成 140 次重规划决策而 fresh 策略为 0。
  - **开发集基线**（`COMMUNICATION_BASELINE_PROTOCOL.md` + `_RESULTS.md`，12 seed × 17 条件 =
    **204 episode，技术失败 0、零检测 seed 0**，条件与参数在跑之前写定）：
    只发一次 408 包 → 65/68 按时；周期 2 → 5 466 包 → 68；事件触发 → 2 940 包 → 68 且需求误差 0；
    周期 100 → **38/68（12/12 seed 全更差，CI [−2.83, −1.67]）并有 5 332 个"活动却无人知晓"任务步**；
    丢包 0.3 下只发一次塌到 51/68，周期重供恢复 66 但要 1 607 包；发射预算 1/步 与 5 步延迟
    在这批场景上是**零效应**（逐 seed 完全相同）。周期 2 有 **1 728 包（32%）被接收端判为重复**，
    事件规则用 54% 的包达到同样完成率——**"可省略流量"被量化了**。
  - **可学习性判定（诚实结论）**：主指标上规则基线已把开发集打满（68/68 与真实系统参照相同），
    `event` 与 `event_maxwait` 在本集几乎逐位相同（最长静默约束基本没生效），
    12 个 seed 的规则间差平均只有 0.09 目标/episode ⇒ **在无约束、无损、传感受限未启用的当前系统里
    没有可训练出来的完成率收益**。剩下的空间被明确定位为两条：同预算内更接近 5.67 上界
    （61 包/episode 时规则只有 5.58）与有损/受限信道下不失守。
    `LEARNED_COMMUNICATION_CONTROLLER_DESIGN.md` 给出动作/状态（只用发送端合法可见量、不喂 ID）、
    硬约束 `max_wait`（非紧急 10 步、紧急 3 步）、ε-约束目标与由实测标定的预算阶梯
    {34, 61, 128, 245, 455} 包/episode、λ 的对偶更新（不做未标定相加）、回退策略、
    数据划分（训练用新 700000 块，验证用 dev 块未用部分，500000..500399 仍不碰）与
    训练成本估算（1 个工作日 CPU 起），并写明**进入训练的三条门槛**（场景可分辨、信息确有后果、
    旁路已受控），当前只有第 2、3 条成立。
  - **状态**：默认配置未变（`comm_policy=immediate`、`comm_range=0.3`、`gossip_interval=2`、
    六个既有开关仍全关），未合并、未重训、未删除旧产物；全套 **582 通过（2 跳过）**，
    smoke `final_status=completed`；`COMM_ROUND_MANIFEST.json` 记录提交、checkpoint 摘要、
    条件表、seed 来源、门槛结果与产物清单。本轮**未开始长训**。

- 2026-09-19（同日第二轮）：**通信硬预算下的学习型发送控制 pilot**，
  分支 `feat/learned-communication-budget-pilot`（起点 `0794de9d…`），产物在
  `runs/comm_budget_pilot_20260919/`。
  - **先修通信账本**：核对 `408 → 288` 的疑问，确认上一轮把**投递数当成了成本**——丢包率 0.3 下
    发送尝试仍是 408 次、空口字节仍是 168 498 B。账本改为 generated / attempts（丢包照计）/
    delivered / applied / duplicate / superseded / queued / coalesced 分列，主成本 =
    载荷字节 × 订阅者 × 尝试；**上一轮开发基线 204 episode 全部重跑**，受影响两行就地更正，
    第一遍产物保留在 `comm_control_20260919/pass1_ledger_undercount/`。
  - **执行器与统一约束**：`comm_budget.py` 给出每 episode 字节额度（不补充、不可为负、越界抛错）
    与每步容量；`--comm-max-wait-steps/--comm-urgent-wait-steps` 把"最长静默"从某个策略的私有规则
    提升到传输层，规则臂与学习臂同一条；强制发送同样计费、付不起同样被拒。新增 `event_priority`
    规则臂（同候选集、不同排序），并验证无预算时它与 `event` 逐投递相同。默认全关，
    `identity_check.py` 重跑 **8/8 seed 逐记录相同**。
  - **标定（先固定种子与参数，再只改预算）**：本轮自录 72 条检测流（48 训练 / 24 验证），
    4 规则 × 4 档 × 48 seed = **768 episode，技术失败 0**。无预算 `event` 均值 117 272 B ⇒
    按预登记规则锁定 **75%=87 900 / 50%=58 600 / 25%=29 300 B** 并选中 75% 档训练。
    关键事实：`immediate` 三档都不触顶（只用 14 737 B），因而在所有档位上**胜过**被预算掐住的
    `event` 与 `periodic_5` ⇒ 便宜规则在有预算时并不弱，学习臂要打败的是"最省"而不只是"最准"。
  - **pilot（3 个训练 seed × 100 000 环境步 = 100 episode、25 次更新、5 635 参数）**：六项 smoke
    全过（预算不超限、观测白名单无泄漏、动作改变交付时刻、参数确实更新、逐步奖励和 == 按时数、
    关掉上界的全沉默得 0 分）。24 个验证 seed 公平比较（各臂自行决定发射与到达、同额度同上界）：
    **3 个 seed 里 1 个成功**——它以 39 360 B 达到与无预算天花板相同的 5.958 按时完成
    （比最强规则 `periodic_5` 省 **34 768 B，CI [−40 363, −28 787]**，完成率差 CI 含 0）；
    有损 0.3 下**同时**赢完成率（+0.333，CI [+0.042, +0.750]）与开销；另两个 seed 一个未收敛
    （3.50→5.25 仍在爬）、一个显著更差（−0.958，CI [−1.292, −0.667]）。
  - **换回现场感知与 gossip 的部署设置复核**：学习臂与 `periodic_5` 按时完成 **24/24 逐 seed 相同**
    （也与无预算参照相同），但空口字节少 **41 533 B（省 59%，CI [−45 780, −36 989]）**；
    同时它发出的尝试只有 71% 真正改变 belief（其余被视距/gossip 抢先）⇒ 本轮可主张的是
    **同完成率下的受控通道降成本**，不是完成率提升，也不是全系统通信量下降。
  - **判定**：不成立"稳定优于最强规则"；成立"在预登记完成率容忍内显著降低通信"。
    失败模式是"乱发到额度耗尽"而非缺特征（同一观测/算法下成功 seed 立即达到天花板），
    因此最有依据的下一步是**先扩大训练**（每更新 4→16 episode、100k→400k 环境步，判定线已写死），
    不扩动作空间、不动收件人选择；未达判定线前不预注册新的独立测试。`500000..500399` 未读未看。
  - **状态**：默认配置逐字段从 `RunConfiguration` 反读核对与历史一致
    （`configuration_defaults_match: true`），全套 **592 通过（2 跳过）**，
    `python -m integrated_sar` `final_status=completed`；录制流、逐 episode 日志、传输转录与
    torch checkpoint 均不入库，清单记录其数量、字节数与 sha256 折叠值。

## 尚未完成

按推荐优先级排列：

0. 奖励构成重构（第 1–5 步全部完成，见上）。剩余：跑正式规模的
   v2/legacy 配对 seed 对照（γ 0.99 vs 0.95），并记录 value loss /
   explained variance / entropy / KL / clip fraction；V2 权重调参。

1. 在 PILOC 环境中原生发出 `DetectionEvent`，不再依赖适配器重建。
2. 让 PILOC 环境在发现后保留目标实体。当前 `timemap` 仍会删除已发现目标；
   集成层已通过注册表把任务生命周期保留到真正完成，但搜索侧的原生行为尚未修改。
3. 用共享障碍图的路径代价替换 task-03 欧氏距离与直线移动。
4. 修复 `old_network` 全 padding 邻居 attention 的 NaN 风险。目前只在加载期探针中拦截，
   `summary` 模式的权重会被拒绝加载，主线 `tokens` 模式不受影响，上游代码尚未修改。
5. 固定训练/验证/测试地图与 split。PILOC 每回合重新随机生成地图，仓库又不含
   `DungeonMaps`，所以目前无法留出验证集，也就无法按“验证集最优”而不是“最后一个”
   来选模型。
6. 学习型协调器**暂缓**：可学习性门禁未通过——同信息 oracle 与规则规划器逐位一致
   （可修复比例 0%，门槛 30%），残余失败是执行中断而非排序错误。三项建议中
   “承诺对执行侧的约束力”已实现并验证（见上）；“belief 携带阶段工作时长”已作为
   stage-duration 相关性门禁单独测过并**被否**（0/6，见上），不再重复提议，除非出现新的
   证据。剩下的开放杠杆是**联盟成型延迟**：检测后多久真正凑齐覆盖所需的能力组合
   （六个残余目标该值为 4–157 步，而预算只有 62.7–105.6 步），它属于分批派遣与机队构成，
   不属于协调器的排序；只有该方向出现可排序类失败时，才重新录制 fresh block、
   用同一门禁复测，并训练 `LearnedCoalitionCoordinator`。
7. 建立搜索、协调、任务分配联合奖励与延迟信用关联。
8. 增加端到端基准和消融：无协调、规则协调、学习协调、无反馈。
9. 跑一次真正收敛的训练。两个训练入口现在都能在 CPU 上端到端跑完并产出带元数据的
   checkpoint，且都能被 `joint_driver` 以 strict 方式加载（PILOC 侧已实测跑到
   `search_backend=piloc-trained` 并完成一个目标，CTDE 侧已实测通过加载期探针），
   但本机 `torch` 是 CPU 版、没有 GPU，冒烟规模的结果不代表收敛后的策略质量。
10. 提高 mock 搜索侧的保真度。目前只验证了双通道地图与 PILOC 逐格一致；时间戳老化、
    智能体间地图融合没有实现（搜索到救援的上行链路已由 CommRelay 提供延迟与丢包模型，
    但搜索队内部与真实无线电环境仍未建模），所以 mock 的完成率与策略表现
    不能作为真实 PILOC 闭环有效性的证据，只能证明接口、日志与确定性。
11. 加强加载期探针。当前只喂全零输入、固定 4 个任务，且发生在协调残差之前；
    应补真实量级输入、不同任务数，以及“残差之后再校验 mask”的用例。
12. 可复现性的上限。`old_driver` 的 `--episodes` 粒度是 `N_ENVS`（请求 2 个会跑满一轮 6 个）；
    TF32 与 bfloat16 AMP、以及动作采样本身，使固定 seed 也只能保证统计可复现，
    不能保证逐位一致。

## 推荐下一任务

四项决定权仍在用户手上：**环境修复是否并入主线**、**协调器候选是否走向默认启用**、
**通信控制这条方向是否继续投入**，以及本轮新增的**是否批准把 pilot 扩大成正式训练**。
三者实现状态都未推进——默认配置逐字段核对与历史一致（新增的 `comm_policy=immediate`、
`comm_budget_bytes=None`、`comm_max_wait_steps=None`、`comm_range=0.3`、`gossip_interval=2`
全部保持原样），三条分支都仍是隔离分支。

通信这条支现在有了**正向但不可复现**的第一个结果，因此下一步是明确的取舍：

1. **扩大训练（我推荐的下一步，代价约 3 个工作日 CPU）**：判定线已写死在
   `COMM_POLICY_PILOT_RESULTS.md` §六——3/3 seed 对最强规则 `periodic_5` 的**字节差 CI 上界 < 0**
   且**完成率差 CI 下界 ≥ −0.05**，才升级为"降低通信"的方法主张并预注册新的独立测试集。
   规模、评估节奏、λ/熵退火都先登记，不边跑边改。
2. **先补场景可分辨性**：完成率维度上学习臂与规则臂在 24 个验证 seed 上几乎逐 seed 相同
   （部署设置 24/24 相同），说明**当前设定无法检验"同预算更高完成率"**；若要把这条也变成主张，
   需要先收紧 deadline 带或加密目标（只跑规则、预先登记），再谈训练。
3. **接受"降本不增收"作为可发表结论**：现有证据已经支持"同完成率、受控通道开销降 47–59%"，
   且在有损信道上同时赢完成率与开销；若不再扩大训练，就按这条写结论，不写完成率提升。
4. 仍未回答的设定问题：是否把"机器人留在现场即可免费刷新 belief"当作真实系统性质——
   若是，上行永远只能主张降本；若否，需要环境设定层面的改动，那属于场景设计决定，等使用者定调。

通信控制这一支的实测结论是**限制性的**，因此下一步选择应当由使用者做：

- **A. 就此如实收尾。** 开发集证据显示：真实系统里环境自带视距刷新替代了 95.6% 的上行信息，
  而四个固定基线在无损条件下已经把完成率打满（68/68），规则之间只差 0.09 目标/episode。
  如果研究对象必须是"真实部署中的通信调度收益"，那么当前设定下没有可训练的东西，
  应当把这条结论写进论文而不是一条新方法。
- **B. 转向成本轴（预算内保完成率）。** 预注册 `B ∈ {61, 128}` 包/episode 的硬预算实验：
  61 包时最好规则 5.58、上界 5.67，差距可达但有统计功效问题（需要更多 seed 或更紧场景）。
- **C. 转向鲁棒轴（有损/受限信道）。** 丢包 0.3 下只发一次跌 14 个目标、固定周期重供只恢复一半，
  这一条在开发集上已经可分辨，代价是主张范围必须限定在"该信道模型下"。
- **无论选哪一条，前置工作是同一件**：先做场景标定（收紧 deadline 带或增加目标数），
  让规则基线之间出现可分辨差；标定只跑规则，不碰模型，且必须在训练前预先登记。
- **一个尚未回答的设定问题**：是否把"机器人总是留在任务现场就能免费刷新 belief"视为真实系统性质。
  若视为不真实，需要的是环境设定层面的改动（现场可感知性、`comm_range` 语义），
  那属于场景设计决定，应当由使用者定调后我再动手——本轮只把它列为诊断（`range_*`）而非修改。

上一轮列的第 1 项（"以修复环境为记录基线、在未读过 seeds 上重做 C vs D"）已经完成：
400 个未使用 seed、两臂同一修正环境、正确性门槛全 0、主指标配对差 +0.0161
（95% CI [0.0093, 0.0240]）。因此现在*可以*在该测试分布上说 D 有改善；下面几项是仍然开着的：

1. **环境修复是否成为默认。** `--task-membership-consistency` 改变的是"谁在为某个任务的进度
   记账"，影响任何改派 working 机器人的路径，也改变"任务在无人到场时完成"的含义；
   它不应随协调器合并被悄悄带进主线。还需要决定：既有已发布结果是否要在修正环境下重新解释。
2. **执行边界对"目的地已结束"要不要挡。** 现状是放行、到场核实拒绝（独立块：C 4 / D 6 次浪费行程，
   对已结束任务的状态写入 0）。挡掉的代价是 belief 未更新的机器人可能在原任务上每周期被拒——
   需要同时回答"由谁、什么时候给新目的地"，所以本轮只做记录，不做修改。
3. **1.4% 空闲 robot-tick 的代价是否接受。** 这是候选在当前难度分布上的真实成本，
   与 +38 个按时目标同时成立；默认启用与否应当把它算进去。
4. **在线块需要更难的场景。** 60 seed 在线两臂逐集相同（瓶颈是搜索覆盖），当前无法用它
   分辨分配差异；若要用，必须先预先登记新的难度参数，不要事后调难度。
5. **保留 33011/target-7 为 candidate 名下失败**（旧环境四格里的结论，本轮未重跑该 seed 的
   RNG 路径）；本轮独立块的唯一新增失败 500168 同样保留记录。RNG 机制两轮都没动。

## 资产说明

本仓库不包含 checkpoint、DungeonMaps、RALTestSet、TensorBoard 日志、动画、论文和结果图。
这些文件应放在团队共享存储或 GitHub Release/LFS 中，不能直接提交普通 Git 历史。
