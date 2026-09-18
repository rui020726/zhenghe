# 项目状态

更新时间：2026-09-17（可学习性门禁）

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

在有权重的机器上跑一次真实闭环，并确认日志可追踪：

- 用团队共享存储中的 `shared_mappo_model_*.pth` 与 `ctde_final.pth`；
- 固定 60×60 障碍图与 seed，`--steps` 取足以完成若干目标的长度；
- 检查摘要中 `search_backend=piloc-trained`、`allocation_backend=ctde-network`、
  `rejected_assignments` 是否非零；
- 若 PILOC 权重与当前网络结构不匹配，驱动会以 `CheckpointIncompatibleError` 明确报错，
  此时应核对 `dim_info` 与 `num_agent` 而不是放宽 strict 加载；
- 之后再做第 1、2 项，让搜索侧原生发事件并保留目标。

验收标准是可以从日志中用 `target_id/source_event_id` 追踪每个目标的完整生命周期。

## 资产说明

本仓库不包含 checkpoint、DungeonMaps、RALTestSet、TensorBoard 日志、动画、论文和结果图。
这些文件应放在团队共享存储或 GitHub Release/LFS 中，不能直接提交普通 Git 历史。
