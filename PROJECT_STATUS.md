# 项目状态

更新时间：2026-09-12

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
    及实际导入失败，另增一组 stand-in 测试覆盖生产构造路径（60×60 守卫、两步构造、
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
    现按 `reset()` 的同款逻辑在构造函数里生成障碍图。
  - `piloc/train.py`：device 可配置（原硬编码 `cuda`）、占位保存目录改为 `--output-dir`
    并自动创建、修掉 `episode_num % 1000 == 0` 在 episode 0 每步都存盘的缺陷、
    `--seed` 统一驱动 random/numpy/torch 与动作空间、`--episodes` 生效（原 `max_episodes`
    读了不用）、`--resume` 走 strict 加载而不经过硬编码 device 与文件名的 `MAPPO.load`、
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
- 148 项集成测试通过（本机无 torch/gym 时其中若干项按依赖跳过）。

## 尚未完成

按推荐优先级排列：

1. 在 PILOC 环境中原生发出 `DetectionEvent`，不再依赖适配器重建。
2. 让 PILOC 环境在发现后保留目标实体。当前 `timemap` 仍会删除已发现目标；
   集成层已通过注册表把任务生命周期保留到真正完成，但搜索侧的原生行为尚未修改。
3. 用共享障碍图的路径代价替换 task-03 欧氏距离与直线移动。
4. 修复 `old_network` 全 padding 邻居 attention 的 NaN 风险。目前只在加载期探针中拦截，
   `summary` 模式的权重会被拒绝加载，主线 `tokens` 模式不受影响，上游代码尚未修改。
5. 固定训练/验证/测试地图与 split。PILOC 每回合重新随机生成地图，仓库又不含
   `DungeonMaps`，所以目前无法留出验证集，也就无法按“验证集最优”而不是“最后一个”
   来选模型。
6. 将规则协调器替换为有界学习型残差网络，并加入训练 buffer。
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
