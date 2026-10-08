你是本项目的新执行 agent，不需要继承旧会话记忆。完整阅读本提示词，以它作为本轮执行依据。历史报告里的“下一步建议”不构成额外授权。

任务：ROUND56-SOFT-EARLY-REPORT-PRIOR

一次完成：最小实现、零环境正确性检查、固定训练、开发评估、统计与本地交付。正常情况下不要只做准备后再次等待授权；但出现影响有效性的错误时必须停止受影响运行，不得为了完成计划绕过门禁。

一、项目目标与当前事实

主线是在冻结搜索与任务分配的条件下，让学习型通信策略保留按时任务、减少已计量载荷尝试成本，不是单纯少发消息，也不是模仿某条规则本身。

当前策略是一次性首报 SEND/SKIP：SKIP 的永久性、候选时序及预算不足处理，均沿用 R53 合同。本轮不引入等待、刷新、ACK或新观测。

已完成的关键结果：

1. R53 使用 critic 梯度隔离：两个 critic head 正常训练，但 critic loss 不更新 actor 的共享主干。
2. R53 三个 seed 的 greedy 在既有16个开发场景上，按时目标 ID 集合均与 immediate 相同。
3. R53 greedy 平均计费分别为：
   s11：4995.375 B/集；
   s22：3923.25 B/集；
   s33：4851.0 B/集。
   late 基线为4838.25 B/集，因此原 development_go=false。
4. R54 将 s11/s33 剩余净成本缺口定位到首报集合差异，没有发现共同发送目标的计价差。
5. R55 查实 late 的实际规则是：
   SEND iff first_emission AND ready_step > 147.0。
   这是严格大于；ready_step 是发送端 relay 簿记时刻，不得直接混同 tape step。
6. R55 在现有48条候选 greedy tape、288个首报点上，使用19维向量恢复 late 动作集合，48/48场景一致：
   first_emission = (f[5] == 1)
   step = round(f[18] × 1000)
   ready_step = step − round(f[0])
   该恢复关系须绑定实际冻结源码和适用范围，不能仅凭这批面板宣称对任意状态都成立。
7. s11 greedy 相对 late 的成本差为+157.125 B/集，95% CI=[0,471.375]，不能写成“显著更贵”。
8. “BC在焦点状态发送是因为状态偏离训练分布”目前只是解释假设，没有被现有证据证明。

late 是用于提供归纳先验的强规则，不是任务安全 oracle。历史中存在提前首报改变失败结局的案例，因此本轮禁止硬性屏蔽早期发送。

二、起点、必读与复用

执行仓库预计为：
D:\code\整合\zhenghe-round42

Python预计为：
D:\project_env\python.exe

起点：
analysis/round55-late-gate-observability
完整 HEAD：
b3aff3d5f942096e961f273ee4f83020f94ea854

核验该提交，并确认以下两个交付头为祖先：
R54：d0b91b94ded47ddb6cc10ff8b950cb0c9b768549
R53：39b9cccf2a87862304ebde112c382bb8aa2b842c

建立独立分支：
exp/round56-soft-early-report-prior

同名分支或本轮产物若已存在，先核查状态；已完整完成则只核验交付，不重复执行。不得重置、覆盖或删除已有运行。

完整读取：

- 当前 AGENTS.md、PROJECT_STATUS及总纲的当前行动与授权部分；
- R55报告、恢复映射证据和STATEMENT_RECONCILIATION；
- R54的EXECUTION_VALIDITY、DECISION_POINTS、STATISTICS_RECHECK；
- R53实际冻结配置、初始化/恢复合同、critic隔离接缝、训练入口和评估入口。

路径由 Git 跟踪文件与清单解析，不猜文件名，不导入带顶层执行或覆盖旧报告副作用的历史分析脚本。

复用 R54 已核验的 R53 有效性证据，不重复528集、27项旧套件或整个历史审计。只检查本轮实际依赖的源码、模型、行和清单身份是否未变。若关键依赖不一致，停止并说明，不擅自改用另一套源文件。

保护两份现有脏文件，记录接手时 SHA256，结束时核验字节未变；不编辑、不暂存、不回退：

runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_CURRENT_TASK.json
runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_R2_V6_STOP_STATE.json

其它无关文件同样不动。

三、唯一方法变量：单侧软先验

以 R53 critic隔离版 PPO 为基线。保留其任务/成本目标、critic隔离、采样、mask、PPO数学与恢复合同，仅给 actor 的最小化 loss 增加以下项。

对第 i 个训练行、第 j 个候选：

q_ij = 1，当且仅当：
该候选是记录中的真实自由首报决策，
且由19维输入恢复的 ready_step ≤147。

其它情况 q_ij=0。

新增：
L_gate = (1/N_rows) × Σ_i Σ_j q_ij × [−log πθ(SKIP | x_ij)]

总 loss：
L_R56 = L_R53 + β × L_gate

本轮 β 固定为0.10，从首次启动前锁定到结束。
它是事前选择的实验系数，不是已知最优值；不得扫描、衰减、自适应调整或根据中间结果改变。

具体合同：

- 使用当前网络重算的、保留梯度的 log_softmax；不得使用落盘旧概率或脱离计算图的值。
- SEND/SKIP动作索引从实际接线核验，不凭名称猜索引。
- N_rows 为基线当前 minibatch 的 ALL_ROWS 行数；不改变原 PPO/entropy/critic 的归约分母。
- 无自由首报的行对 L_gate 贡献0；不另做有效项数量归一化。
- 同步多候选按真实记录的条件顺序 mask 判定自由性，不使用静态 send_executable 代替。
- masked/forced/非首报项不进入新增项。
- ready_step>147 时新增项为0：不奖励或强迫 SEND。
- 新增项不是硬 action mask，不覆盖 greedy/sample动作。
- 不将它写入真实成本、任务奖励、cost critic target或g_hat。载荷尝试成本仍按执行器原账本计量。
- actor仍仅输入原19维；不新增ready_step列，不使用场景ID、目标ID、未来结局、接收端状态或分析标签。
- 基线 critic detach保持；新增项只能通过actor路径作用，不直接更新两个critic head。

本轮属于“加入规则先验的学习方法实验”。即使成功，也不能说成模型完全自主发现了late门控，更不能把late的SKIP标签称为安全真值。

四、其它全部冻结

逐字段继承 R53 实际机读冻结件：

- 搜索/分配模型及世界参数；
- 原19维编码，不使用R30新期限编码；
- 一次性SEND/SKIP、ACK关闭、policy_owned_timing；
- 1000步、6目标、29300 B硬预算；
- 正成本及B0=10000；
- δ=0、λ初值1；对偶步长沿用R53冻结值，不拿早期轮次值替换；
- gamma、GAE、ALL_ROWS、clip、学习率、epoch/minibatch、熵项、价值系数、裁剪、CPU/线程设置；
- sampled训练、原greedy解码及平局规则；
- 三个训练seed：11/22/33；
- 48个训练场景1300000..1300047及每seed原128集顺序；
- 16个既有开发场景1301000..1301015；
- 原参考缓存及其键、统计方法。

从每个seed对应的真实BC_POST四模块初始化，复用R53登记的初始化/RNG合同，fresh Adam、fresh λ。核验新ep0参数与该seed的R53 ep0一致。

禁止从R53 ep128或旧PPO状态热启动。不得混入被排除的旧R43模型。新训练版本必须明确包含本轮目标版本；旧checkpoint不得静默按新目标resume。

允许模型只读装载用于参照核验。记录实际加载角色，区分模型初值、RNG来源与恢复状态。

五、新额度：480正常＋24技术储备＝504硬上限

这是本轮新额度，不继承旧轮余量。

计划：

- 3 seed ×128训练集＝384；
- 3 seed ×16场景 greedy评估＝48；
- 3 seed ×16场景 sampled评估＝48；
- 合计480正常任务。

每seed首4集共12集计入384，不是额外smoke。

参照复用已保存、经身份核验的：

- R53 critic隔离版greedy/sample；
- 真BC控制greedy/sample；
- 共享critic PPO；
- immediate与late；
- 训练块immediate reference。

不重跑上述基线，不生成新流或新场景，不打开封存块，不跑会暗含仿真构造的全量套件。

技术储备只用于同一逻辑任务的真实技术失败，失败不退额。若无法精确恢复已提交训练状态，不允许借储备重训整个seed或重复optimizer更新。

预检为纯计算/只读/autograd检查，本轮不需要合成optimizer.step。所有真实构造、失败、重试及成功episode均进入启动账本；另列PPO批、optimizer步、checkpoint写入。

六、首次启动前门禁

在实际 batch_samples、actor与loss组合路径上增加窄测试，不复制被测实现作为预期答案。至少覆盖：

1. β=0时，总loss及实际参数梯度与R53相同。
2. 单个自由首报、ready≤147、N_rows=1：
   设z=SEND logit−SKIP logit，独立推导新增项梯度为β·p(SEND)；
   p=0.5时为+0.05，梯度下降应降低SEND倾向。
3. ready>147、masked、forced、非首报分别贡献0。
4. ready=147必须属于惩罚区；ready=148不属于。
5. 两候选共享预算时，第二候选若因前一个已采样SEND变为masked，不能仍计入新增项。
6. 真实合法SendObservation经实际features后，恢复ready与原字段一致；覆盖age>0、预算/容量等待、阈值两侧及episode边界，不只测试age=0。
7. 验证新增项直接对两个critic head的梯度为0，critic→trunk仍隔离；actor路径保持可导。
8. cost/task奖励、g_hat及原计费闭合不含新增项。
9. 新训练版本的保存/恢复可识别目标与β；拒绝旧目标checkpoint的无声明续训。

检查的是当前实际冻结源码。若恢复式在合法、实际可达状态上不成立，停止并报告范围，不悄悄改为读取额外原始字段来通过测试。

先保存本提示词原文及身份，提交执行代码、配置差异和预检证据，再冻结启动身份。分析/报告程序与真实执行身份分开登记，不因后续写报告误称训练代码漂移。

七、连续执行与首批真实门禁

每seed完成首4集及第一个真实更新后，独立检查：

- 初值、fresh optimizer/λ、登记RNG与数据顺序；
- 新增项在真实free首报上作用，数值与独立重算一致；
- 更新前权重的likelihood对齐；
- critic隔离、optimizer参数集合与梯度有限；
- 正成本、同tick任务奖励、GAE、组合优势、g_hat、λ更新；
- 本轮新增项未污染真实账本和critic target；
- checkpoint包含完整可恢复状态与新版本。

通过则自动完成该seed剩余124集，随后完成固定96格评估。不要根据成本或任务中间结果调整系数、训练量、阈值或选择checkpoint。

表现差是有效结果，不是技术失败。不得挑最好seed或sampled/greedy模式。

若发生影响执行有效性的缺陷：保留现场、claim与原件，停止受影响运行；不冻结后热修继续，不重复已提交更新。仅报告渲染缺陷可在不改执行路径、不重跑episode的情况下修正，并留痕。

八、预定比较与判读

主比较：本轮对R53 critic隔离版，按同seed、同mode、同场景配对。

其它参照：真BC、共享PPO、immediate、late。不能把对immediate的收益全部说成本轮新增项的贡献。

分别报告三个seed、两个mode，不池化、不挑选：

- 按时任务绝对数、配对差；
- 精确按时目标ID集合、丢失/救回/交换；
- 已计量B/集、配对差；
- 首报数、拒付数；
- 原始状态ready区间与策略概率/实际动作，分析字段不得进入actor；
- λ/g_hat轨迹及任务约束是否在训练样本上被违反。

新旧策略经历的状态不同；同输入forward探针与闭环效果必须分开，不能将概率变化直接当作任务因果。

沿用现有场景配对bootstrap口径，统计值只生成一次供表格和文字复用；保留精确零端点。CI含0不写显著或等效，未观察到任务损失不写总体零损失。

原development_go不放宽：

1. 三个greedy seed各自在16/16场景的按时目标ID集合与immediate完全一致；
2. 三个seed各自平均已计量成本严格低于4838.25 B/集。

任一不满足即false。即使满足，也只表示这个已反复使用开发块上的筛选门槛通过，不授权独立确认、泛化、部署或交付收尾。

九、有限交付与停止

在一个独立R56目录中保存：

- 执行依据、冻结配置与唯一变量差异；
- 窄测试、首批真实门禁；
- 启动/逻辑任务账本；
- 新模型索引与版本；
- 分seed/mode配对结果、精确目标集合差、单一统计来源；
- 本轮报告及机器verdict；
- Git外权重/tape/日志的路径、大小、SHA256清单。

报告必须回答：

1. 新增先验是否通过实际actor训练路径作用？
2. 相对R53，三个seed的任务与成本分别怎样变化？
3. 是否纠正了部分early发送，同时保留已有late区间SKIP？
4. development_go是否满足？
5. 哪些结论仍只是开发性、规则先验辅助的结果？

全部完成后本地提交，报告完整HEAD、实际消费及剩余储备，然后停止。

不推送、不合并、不部署、不自动续训、不调β或147、不换候选、不追加场景、不启动确认实验，也不把未使用储备转成研究额度。