执行 ROUND42-R 的固定计划续执行。你可能没有历史记忆，请按以下背景、输入与边界工作。

一、研究目标与当前状态

项目由冻结的 PILOC 搜索模型、冻结的 CTDE 任务分配模型和待训练的通信控制器组成。

当前主线是：
在尽量保持按时任务数的前提下，减少已计量的通信载荷尝试成本。

不能将全沉默造成的任务损失包装成降本成功；不能将规则可运行包装成学习模型达到目标。

ROUND42 研究的是一次性首报选择：
每个 episode、每个 target，在首次合法且按顺序预算可付的首报机会，选择 SEND_NOW 或 SKIP_EPISODE。
这是首报选择实验，不是完整的等待、刷新或通信频率优化方法。

ROUND42 首个参考任务 immediate_reference_scene1300000 在 JointRun 构造成功后，因向只读的 run 属性赋值而失败。没有调用 run()，没有仿真步，没有训练更新。

相关身份：
- 原 ROUND42 最终停止报告头：
  70faab3f451b3c17690c189216c315cccaf4a6d6
- 原 ROUND42 实际执行冻结头：
  3c0b61cbdb702c3e4d8b1a4668030bda2db102ab
- 原执行依据提交：
  ffa87ceb1a9d48abe27fc124ef1a963470878604
- ROUND40 交付头：
  8f8245a665b15960e70194fc6e245221b70db90e
- ROUND41 交付头：
  cc28a28fc5fb13e31ab9ed0fc3456797aadcc115

ROUND42-R 已报告完成运行器修复：
用 _make_counted_builder 和 _CountedJointRunProxy 替代对 JointRun.run 的实例赋值。
生产 JointRun 不修改，原对象不复制，调用参数、返回值和异常身份保持不变。

算法语义继续为：
one-shot-send-skip/1

训练版本继续为：
constrained-cost-v2/one-shot-send-skip/1/published-completion-event-at-relay-tick/2

单独的执行身份为：
counted-jointrun-proxy/1

当前说明尚未给出修复分支完整 HEAD、真实参考重试结果或最新启动总数。请从实际仓库与账本取得，不猜测。

本提示词授权：完成必要的零启动状态核验，执行真实参考重试，并在门禁通过后自动完成原定固定计划。
不需要修复验收后再次请示是否继续。

二、首先核验实际进度，禁止重复启动

定位仓库，读取适用 AGENTS.md、PROJECT_STATUS，以及：
- ROUND42 原执行依据与冻结件；
- ROUND42-R 修复说明、回归证据、冻结件；
- ROUND42R_CARRYOVER.json；
- 原失败 claim、续执行账本与 stop 记录；
- ROUND42R_REFERENCE_RETRY_GATE.json，如已存在。

核验并记录当前完整 HEAD、修复提交、预启动身份和工作区状态。

优先继续已有 ROUND42-R 分支，不为更换会话再复制一套运行器。
保留其他会话修改和未跟踪文件；不得 reset、覆盖或删除旧实验。

确认是否已有同一计划的运行进程：
- 已在运行的任务不得重复启动。
- 已完成且身份、final record、账本与门禁一致的任务直接复用。
- 不因报告未列出结果而假定它没有执行。
- 未完成或身份冲突的任务按原恢复纪律处理，不伪装成已完成。

回归证据在输入代码、测试和冻结身份未变时可以复用，不因新会话重跑全部历史审计。
若修复代码尚未提交或预启动冻结尚未完成，先完成这一必要步骤，再启动。
不另写新的研究 spec；将本提示词原文存档为续执行依据即可。

三、预算是原额度结转，不是重新获得 700 次

原计划：
672 个成功逻辑任务；
672 个 normal slot；
28 个 technical reserve；
累计启动硬上限 700。

原失败已经消耗：
normal slot 1；
1 次 builder invocation/JointRun 构造；
0 次 run 调用；
0 个成功 episode。

原失败原件与 claim 保持 failed，不能改成成功，不能退还额度。

若失败参考尚未重试：
- 用 reserve slot 1 执行其 attempt 2；
- 绑定原 logical job immediate_reference_scene1300000；
- 新 attempt、日志和证据独立命名；
- 只有真实重试门禁通过，logical job 才指向成功 attempt。

后续只剩 671 个新的 normal slot，不是 672。

没有新增失败时，最终累计必须为：
- normal claims：672，含原失败；
- reserve claims：1；
- 总启动/构造：673；
- 成功逻辑任务：672；
- 成功完整 episode：672；
- 累计技术失败：1，即原失败；
- reserve 剩余：27；
- 原硬上限仍为 700。

如果 ROUND42-R 已执行部分任务，从当前账本扣除已消耗额度和已完成任务，不重复领取 reserve slot 1。

累计启动台账、真实构造计数、run 调用和成功结果分别报告。
失败可能发生在构造前、构造中或 run 中，不能强行让所有计数相等。
每次正式尝试必须先 claim；技术失败、重试不能从计数中删除。

不得将剩余储备用于额外 smoke、调参、挑模型或确认实验。

四、执行器边界保持不变

不得：
- 再次给 JointRun 实例的 run 属性赋值；
- 修改生产 JointRun 的 slots；
- 全局替换生产类或方法；
- 修改搜索、分配、信道、预算或事件顺序；
- 在冻结运行期间热改会影响结果的 harness。

计数必须区分：
builder 调用、成功返回对象、代理 run 调用、底层 run 调用、正常返回/异常、实际完成步数。

仅构造对象或进入 run() 不代表仿真已经推进。
步数需要核对本次运行的起止状态及实际结果，不能把已有 step_index 当成本次完成步数。

代理只能透明转发同一个原对象；不复制环境、状态或 RNG，不额外调用 run，不改变异常和返回对象。

五、先通过一个真实参考重试门禁

若已有有效的真实门禁结果，核验身份后复用，不重跑。

否则用上述 reserve attempt 跑：
immediate_reference_scene1300000。

它是正式的参考 episode，不额外增加 smoke。

门禁至少核验：
1. 正确场景、immediate 策略、冻结世界配置；
2. 搜索和分配 checkpoint 文件哈希一致；
3. builder 成功返回一次；
4. 代理与底层 run 各调用一次；
5. 完成冻结的 episode 步数及正常 final record；
6. 实际任务结果、奖励和计费按既有口径闭合；
7. claim、attempt、normal/reserve 归属及执行身份一致；
8. 原失败原件哈希未变；
9. 尚无训练更新或 ROUND42 NEW 训练 checkpoint 写入。

门禁通过后自动继续剩余固定计划。
失败则保留证据，停止受影响运行，不在失败版本上热修续跑。

六、完成原定 672 个逻辑任务

计划分解保持不变：

A. immediate reference
48 个训练场景 + 16 个开发场景 = 64。
成功的 reserve 重试计入这 64 个逻辑任务，因此其后只补其余 63 个。
已完成的参考不重复生成。

B. NEW 从初值训练
3 个训练 seed：11、22、33；
每 seed 128 episode、32 个四集批；
合计 384 episode。

C. 模型评估
OLD/NEW × 3 seed × greedy/sampled × 16 开发场景；
合计 192 episode。

D. 规则评估
late_discovery_gate_147、always_hold × 16 开发场景；
合计 32 episode。

总计：
64 + 384 + 192 + 32 = 672。

训练场景：
1300000..1300047，按原冻结的 128 集顺序重放。

开发场景：
1301000..1301015。

这些都是已使用开发数据，不是独立确认集。
不生成新流，不启用新场景，不触碰封存块。

OLD 明确使用 ROUND40 的 NEW ep128 通信权重，不是 ROUND36 权重。
NEW 从 ROUND40 同 seed 的登记 ep0 初值开始，不能从 OLD 或其他已训练模型热启动。

OLD 与 NEW 都在相同 one-shot SEND/SKIP 协议下评估。
必须声明：OLD 是旧模型迁移到新动作运行语义，不是其历史运行行为的逐字复现。
不使用 ROUND41 hazard 解码器。

七、冻结数学、观测与 one-shot 语义

其余参数逐项继承原 ROUND42 机读冻结件，不根据本提示词摘要另造配置。

保持：
- 原 19 维发送端合法观测及原期限编码；
- 原候选顺序、真实报价、顺序 shadow budget 和执行器硬门；
- ACK 关闭，policy_owned_timing；
- 29,300 B 硬预算，原 horizon 与世界配置；
- ALL_ROWS 归约，不切换 ACTIVE_ROWS；
- 原 PPO、GAE、网络、双 critic、优化器、初始化和数据顺序；
- 原任务事件奖励时间契约；
- 原成本目标与对偶更新。

目标方向：
c = C_measured / B0
g_hat = mean(Y_immediate - Y_policy) - delta
A_actor = lambda_dual * A_task - A_positive_cost
lambda_next = max(0, lambda_dual + eta_lambda * g_hat)

成本是载荷尝试计费；丢包照付，拒付不算主动节省。
任务单位是按时任务数/episode，不是百分点。
实验 delta=0 不代表总体或部署零损失保证。

每 target 的一次性决策：
- 首次合法可付机会前为 UNDECIDED；
- SEND_NOW 与 SKIP_EPISODE 都是真实自由动作；
- 首次 SKIP 必须保留 log_prob、entropy 和 actor 权重；
- 已决定后的候选才固定 HOLD，似然/熵权重为 0；
- budget mask 不得消耗 target 的一次性决策；
- 新 revision 不重新开放已决定目标；
- 新 episode 清空局部决策状态；
- 不伪造 relay 的实际 first_emission 事实；
- 不刷新、不重试、不免费发送。

不得为了改变特征而删除已决定目标、缩短 tape 或更改 backlog 定义。

八、各 seed 的首四集训练门禁

每个 seed 的首四集计入其 128 集，不额外启动。

通过其真实首批门禁后才继续该 seed 的剩余 124 集。
可以按已有安全并行机制执行，但每 seed 输出、RNG、checkpoint 和账本必须隔离。

门禁穿过实际 collector → tape → batch_samples → actor/critics → optimizer：
- 初值身份正确；
- logged/recomputed likelihood 在原 1e-6 容差内；
- 首次 SKIP 是自由动作，后续固定 HOLD 不泄漏似然；
- positive cost 和组合优势方向正确；
- task reward 按实际发布 tick 归属，不能只证明总和；
- GAE 与终止/非终止 next-value 正确；
- actor、task critic、cost critic 都在登记优化器集合中；
- 对偶统计与更新可独立重算；
- 梯度、参数及 optimizer 状态有限；
- 旧语义 checkpoint 不得静默恢复。

不要求每个模块在每个 minibatch 梯度都非零。
不因中间任务或成本表现差而修改参数、换初值、追加训练。

恢复只使用身份匹配的新版本状态，包含模型、优化器、双 critic、lambda、RNG 与数据位置。
不得重放已经提交的优化器更新来“补日志”。
无法按原恢复契约恢复时停止，不擅自整 seed 重训。

九、最终分析与报告

固定使用各 seed 最终 ep128，不挑 checkpoint，不挑最佳 seed。
greedy 与 sampled 分表；训练 seed 不混为独立场景样本。

输出：
1. 每 seed、每模式的绝对按时任务数、计费字节、实际首报数；
2. NEW−OLD 配对差；
3. 各模型相对 immediate、late、always_hold 的任务和成本差；
4. 逐目标救回、丢失和目标集合交换；
5. SEND/SKIP 比例及预算屏蔽，区分主动 SKIP 与不可执行；
6. 训练曲线、约束统计、lambda 与 checkpoint 索引；
7. 执行完整性与科学结论分开。

沿用原冻结统计方案，不读过结果后新增显著性或非劣判据。
开发数据只能支持探索性判断。

必须如实识别：
- 全 SEND 近似 immediate：可能任务保持，但没有建立学习附加降本；
- 全 SKIP 丢任务：不算成功；
- 一个 seed 有收益：不能代表算法稳定；
- sampled 保任务：不能搬成 greedy 的保证；
- SEND/SKIP 没有等待动作：不能宣称学会延迟发送。

最小交付：
- 更新后的执行完成/停止报告；
- 累计 START_LEDGER 与 carryover/retry 对账；
- 真实参考重试门禁；
- 三 seed 首批门禁与训练/权重索引；
- 绝对表现、配对比较、逐目标差集；
- 原件路径/哈希清单；
- 当前完整 HEAD、提交链和复现命令。

既有证据有效则复用，不重复建立同义审计系统。
原日志、失败、权重与旧 verdict 保留；原始大文件继续在 Git 外。

十、停止与发布边界

本轮允许完成上述固定计划，不允许：
追加训练、调参、改动作、换解码器、扩大场景、
挑最佳候选、部署、合并、推送或额外确认实验。

不运行会在账本外构造环境的全量测试套件。
不因网络不可达阻止已授权的本地执行。

出现新的有效性缺陷时保留原件并停止受影响部分。
外部技术失败只按既有冻结重试纪律使用剩余 reserve，不能借重试改变执行代码或科学配置。

完成固定计划后，本地按问题提交并报告完整 SHA，然后停止。
最终回答先给：
“固定计划是否完成、累计消耗是否闭合、学习目标是否获得证据”，
不要只汇报修复通过，也不要将运行完成写成科学成功。