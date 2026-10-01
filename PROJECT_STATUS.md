# 项目状态

## 2026-10-01：ROUND35 修复版奖励接线的真实 smoke 验证与恢复边界实机核对——45 个完成事件逐条全对、8/8 集写入键恰为归档键减 1、A 级被动性逐字节复现归档、5/5 篡改副本在恢复前被拒、18/18 旧 checkpoint 内容层与 native likelihood 实做补齐；结论 go（15/16 构造、60/100 步；被测系统 0 项正确性失败，本轮自有 harness 2 处缺陷已登记并复现）

分支 `fix/comm-task-reward-event-timing`（开工头 `c43ec5f1a12ef18238d4eed3aa90e3681e9ae58e`，
基线 ROUND33 头 `c7d47d25…` 为其祖先，本地 `git merge-base` 核验）；产物
`runs/comm_reward_smoke_20260930/`。判定一律以 ROUND34 的 `TASK_REWARD_TIMING_CONTRACT.md` 为准。
**未使用**提示词里的 GitHub 快照（本地仓库与分支/头齐备），0 次网络尝试、未推送、该分支无 upstream。
本轮把 ROUND34 因"本机无 PyTorch"而豁免的四层全部实做（torch 2.9.0+cu128、numpy 2.2.6）。

```text
runner 进程 3（有效启动 2 次 + 首次尝试 4 秒失败）；episode 构造 15 / 16，剩 1
optimizer.step 60 / 100（启动内 36 + 验证重算 24）；单次最长 97.85 s / 2 h
新流 0；封存块 0；旧产物改写 0；网络 0；未热启动任何旧 checkpoint
```

- **唯一改动面**：harness 只把 `v2_common` 的五个输出路径与启动上限重绑定到本轮目录，参照用
  ROUND24 的 64 份 immediate reference **只读**；rollout/PPO/对偶/保存/恢复全走真实模块
  （`v2_run` / `v2_trainer` / `v2_train.train_seed`），未改生产与训练文件、超参、目标、奖励数值
  或成本路径。启动前 12 项门禁全清，其中 P5 证明 `fresh_controller(11)` 与归档 ROUND32 批 0
  更新前参数、git 里的启动前配对件、ROUND24 ep0 里程碑恢复结果**四源相同**
  （`7eff0f40203fdae6…`）——这是被动性能做逐字节比对的前提；P9 把 ROUND34 两套回归在有真实
  torch 的解释器下重跑通过（上一轮是 torch stub）。
- **任务 1/2/3（45 个真实完成事件逐条）**：8/8 集 `task_stream` 键恰为归档键 **减 1**
  （e0 `[9,26,28,34,48,194]` → `[8,25,27,33,47,193]`）；`relay_tick == int(observed_at)` 45/45；
  归属用**真实 `_bundle`** 重算，全部落在"当时或之前最近动作行"，0 意外 prefix；契约六类情形中
  本场景块真实可达的四类都出现（同 tick 45、下一拍有新动作行 30、末动作行完成 6、deadline 过期
  2 集），另两类（事件拍无动作行、首动作前 prefix）**在本块结构性不可达**（归档 4281 笔 0 命中），
  只有合成用例覆盖，未声称实机覆盖；资格五条 45/45 无违例，event_id 唯一，无重复入账，
  `on_time == 入账总数 == task_reward_total` 且成本 tap == 账本字节 8/8 集成立；本块的目标过期
  表现为**非完成**事件，故账本连记录都不产生（第一道状态闸挡住，deadline 守卫是第二层）。
- **任务 5 被动性分级**：A 级（直接逐字节）episodes 0..3 与归档修复前记录 17 个可观测字段 +
  逐决策 `features/actions/log_prob/actionable/targets/step` **全等**，`log_prob` 差恰为 **0.0**，
  `behaviour_sha256` 相同，唯一差异是奖励流键与新增账本；B 级（间接）e4..e7 权重已因修复后的
  更新移动（`log_prob` 差 0.0034–0.0080）而行为哈希仍相同；C 级为闭合与结构性质。
- **任务 4 恢复边界**：新 checkpoint 两版本字段齐全、真实恢复且二次加载幂等、RNG 与文件一致；
  5 种篡改副本 **5/5 在 controller/optimizer/RNG 恢复之前**被拒（计数桩观测 0/0/0，进程 RNG 未动）；
  真实旧 ROUND32 `.pt` 以其归档 manifest **重建** freeze 身份（9 字段，重建==文件自述）后被
  **版本门**拒绝，换身份则更早的 freeze 门先拒（两层分开登记），只读加载成功并给
  `legacy_unversioned_capture_semantics` 且 optimizer/RNG 不恢复；跨进程恢复后续跑的 e4、e5 两格
  对连续运行逐字节相同（20 字段 + 逐决策 tape），恢复态 λ=1.0125 确实进入采样；
  **未覆盖**：恢复之后的那一次更新（额度先拦住 rollout）、活的 collector 仪表未落盘。
- **任务 6 内容层**：18/18 旧 `.pt` 反序列化后 whole-controller 参数摘要与四模块 digest 与
  ROUND34 登记的"待复算声明"**全等**（shapes/5700 参数相符、digest 命名空间不互相替代）；
  native forward + likelihood 18/18 在 1e-6 内（最大重算差 4.0605664253234863e-07），且每点重算差
  与归档侧车记录之差 **0.0**；处理前后字节哈希相同 ⇒ 旧文件零改写；ROUND34 原结论不改写。
- **修复改变了什么（真实数据 + 真实 `_bundle`）**：**不变**的是 episode 总量、每批样本数
  （623/555）、minibatch 与 optimizer 步数、`g_hat`（两批都 +0.25）与 λ（1.0125→1.025）、
  未归属 prefix；**变**的是每笔入账的 tick 与归属行（8 集 45 笔中 30 笔换行）以及由它驱动的
  surrogate / 两个 critic loss / 梯度范数 / 更新后参数哈希。归档侧量化：768 格 4281 笔里
  2788 笔（65%）在修复前被下一条动作行抢走，750/768 格的归集奖励变化 ⇒ **约束统计量不因该修复
  改变，改的是逐决策信用分配**，因此不得把任何行为差异表述为"约束被满足得更好"。
- **登记的两处本轮缺陷**：启动②首次尝试因本轮仪表读了一个不存在的快照字段名（`task_revision`）
  在 4 秒处失败，消耗 1 次 episode 构造**不退还**，含最小复现与真实 traceback；
  `r35_launch.py` 绑定每目录上限时少算了一次失败起始，导致 launch2 实际只允许 6 次新构造
  （误差方向保守：提前停止，从未超预算）。
- **结论 go 的边界与残余风险**：规模外推未知（只有 8+6 构造、60 步，不作任何学习效果判断）；
  两类契约情形在本场景块不可达、换新场景块须重做可达性判断；幂等未经实机触发（换用
  `v2_recover` 之类重放路径前需重查）；旧运行仍是 legacy 采集语义基线，不能当修复版预测；
  额外进程的授权只用了用户批准的那一次，未扩边界。
- **停止**：本地交付后停止。不续训、不调参、不追加启动额度、不开启下一轮实验、不部署、不推送。

## 2026-09-30：ROUND33 任务奖励时间跨度与信用传递审计——发现 task reward tap 晚一 relay tick；2821/4363 奖励转给完成后的 tape 行；不调参、不修复、不训练

分支 `analysis/comm-temporal-credit-audit`，基于 ROUND32 交付头
`01e5a677cd18a7abfdb97585315ba05251d4e0de`。完整报告与机读件位于
`runs/comm_temporal_credit_audit_20260930/`（`ROUND33_REPORT.md`、`TEMPORAL_CREDIT_AUDIT.json`、
`GAE_COMPONENTS.json`、`CRITIC_DIAGNOSTIC.json`、`ROUND33_VERDICT.json`、
`SYNTHETIC_GAE_TESTS.json`、只读审计脚本与纯计算测试）。执行依据原文独立提交。

- **契约缺陷**：`v2_run._install_taps` 在原始 `env.step` 返回后、Task03 的新 `ProgressEvent` 返回并被 orchestrator 发布之前读取旧 registry。全量 768 集中 4,363 个完成事件均在同一时间戳后一 tick 才进入 task stream；其中 2,821 个奖励由完成后的新 tape 行接收。总量仍闭合，但时序信用不符合 ROUND23 契约。只登记、不改训练器。
- **完整度**：训练 tape 768/768，无缺失或解析错误；16/16 已用开发场景的 immediate 行均 6/6 按时且共用 29,300 B 硬预算；GAE 源码翻译的合成算例 4/4 通过。
- **固定批边界**：18/18 checkpoint 文件 SHA256 符合 index；当前 checkpoint 参数/lambda 与批记录身份仅 6/18 相符，另外 12 个 b15/b31 点与 `UPDATE_MECHANISM` 样本数也冲突。C/D/E 实际批次计算只报告六个 b0 点，完整 18 点诊断停止。
- 新环境启动 0、新流 0、真实训练更新 0、合成 optimizer 更新 0、checkpoint 写回 0；Python/PyTorch 不可用，checkpoint 未反序列化/forward replay。未推送、未进行网络尝试。
- **唯一后续动作**：修正 task reward tap 的同 tick 事件归属并添加纯计算回归；通过后再只读协调 12 点批次身份冲突，之后才判断是否需要参数实验。

## 2026-09-30/10-01：ROUND32 actor surrogate 归约分母消融（ALL_ROWS vs ACTIVE_ROWS）——1040/1040 零失败；6 格里 0 格变好、2 格变差；N/M 放大在 18/18 点成立但成本项幅度始终小于任务项；"零梯度行稀释"假设被排除

分支 `exp/comm-actor-reduction-ablation`（起点 ROUND31 交付头完整 SHA
`614df51709a8d82525405b46468a537f4ccd259e`，本地 `git merge-base --is-ancestor` 核验），
产物 `runs/comm_actor_reduction_ablation_20261001/`（`ROUND32_REPORT.md`、`ROUND32_VERDICT.json`、
`REDUCTION_MATH.md/.json`、`CONFIG_AND_MODEL_MANIFEST.json`、`RUN_PREFLIGHT.json`、
`FIRST4_GATE_*.json`、`START_LEDGER.json`、`TRAINING_CURVES.json`、`PAIRED_COMPARISON.json`、
`UPDATE_MECHANISM.json`、`INDEPENDENT_RECHECK.json`、`POST_FREEZE_CHANGES.json`、
`CHECKPOINT_INDEX.json`、`ARTEFACT_MANIFEST.json`、`EXECUTION_BASIS.md` sha256 `49d52f95…`、
`r32_*.py`）。生效冻结 `b5615c1bed8b5686…`（提交 `3bafb36`：首次启动之前修 CLI 入口缺陷
`main()` 未接 `sys.argv` 后重新冻结，该修正消耗 0 次启动、0 次合成更新；差异只有
`r32_blocks.py`/`r32_train.py` 各一行 `__main__`）。

```text
新环境启动 1040 / 计划 1040 / 硬上限 1080（技术失败 0、重试 0、储备 40 未动用）
真实训练 = 6 运行 × 32 批 = 192 批、2628 次 optimizer 步（逐运行 464/464/340 vs 464/432/464）
纯合成 optimizer 更新 = 授权 32 / 实执行 40（上限 64，与真实更新分账）
新流 0；只用既有 1300000..1300047 与 1301000..1301015；未触碰封存块；诊断阶段 0 次 optimizer.step、
0 个 .pt 写回；未推送、本轮 0 次网络尝试、该分支无 upstream
```

- **唯一主动变化**：`ppo_update` 里 actor surrogate 的归约集合与分母
  （`-surrogate.mean()` → `ACTIVE_ROWS: (surrogate*row_active).sum()/max(M,1)`）。相对冻结父件
  被删行 3 行（归约行 / `reduction` 形参 / 仅报告字段），逐块差异 unexplained = 0；
  `r32_run` 与 ROUND30 运行器 11 处接缝全属声明、`run_episode` 函数体逐行一致；
  生产/导入模块/搜索分配三段身份哈希 vs ROUND30 **0 差异**；`batch_samples`、`_bundle`、
  `likelihood_alignment`、`dual_update`、`batch_g_hat`、`cost_math.gae`、`cost_math.actor_advantage`
  与冻结件是**同一函数对象**。`ALL_ROWS` 与冻结 `v2_trainer.ppo_update` **逐 optimizer 步权重哈希
  相同**（T9），故两臂任何行为差都可归到那一行。代价如实登记：actor 相对 entropy 与两个 critic 的
  更新权重本轮**确实**改变，不宣称其他优化关系不受影响。`verified` 块 7 项全真、由机读件派生。
- **结果（对"稀释假设"是否定）**：Δ = ACTIVE_ROWS − ALL_ROWS，16 个已使用开发场景、ep128、
  greedy/sampled 分开、三 seed 不池化。
  ΔY = {s11g 0, s11s 0, s22g 0, s22s −0.0625, s33g **−0.5**, s33s 0}，
  ΔC = {−9.0, +77.625, 0, **+11064.0**, **+25709.25**, −62.625} B/ep；
  两个 CI 不含 0 的格都是**变差/变贵**。绝对配对任务损失（vs immediate）六格均值
  ALL_ROWS 0.333333 → ACTIVE_ROWS 0.427083（改善 0 格、持平 4 格、变差 2 格）；
  s33 greedy 从 ALL_ROWS 的"6 按时 / 2398.5 B / 损失 0"退化为"5.5 按时 / 28107.8 B / 损失 0.5"。
  规则参照 immediate 6.0/14939.25 B、late 6.0/4838.25 B。
- **机制（补齐 ROUND31 只存活批 0 的证据缺口）**：本轮为批 0/15/31 在**权重移动之前**落盘 18 份
  完整更新前 checkpoint（模型/优化器/λ/RNG/数据位置/编码与归约模式），身份与批记录
  `controller_parameters_before` 18/18 相符。同状态、同 tape 的算术比较：actor 项梯度精确放大
  `N/M`（18/18 点，最大相对残差 2.9e-07）；actor 在裁剪前总梯度里的占比
  0.95 %–20.4 % → 14.9 %–97.1 %；但 **13/18 点两臂同时触发 `MAX_GRAD_NORM=0.5`**，裁剪后同为 0.5
  ⇒ 变的是方向不是大小，因此不把 `N/M` 称为参数更新倍数。新增读数：
  `abs_mean(A_cost)/abs_mean(λ·A_task)` 在 18/18 点为 **0.18284–0.424743**（λ 1.0→1.475 仍 <0.5）
  ⇒ 被放大的那一整项始终以任务侧为主。行为侧：ACTIVE_ROWS 三格消息更多（+0.0625/+4.1875/
  +10.0625 条/集）且**正是**成本上升的三格，三格两臂消息数完全相同，`s22_greedy` 两臂 0 条/0 B。
- **约束与 λ**：192 批里 `g_hat` **仍无一次为负**；两臂 λ/g_hat 序列本轮真正分叉（首轮 λ 分叉批
  s11=10、s22=3、s33=14；行为分叉批 1/1/2），`ACTIVE_ROWS_s22` 违背批数 21 vs 13、λ 终值 1.4250 vs
  1.2375 ⇒ **加强 actor 权重没有让约束更接近满足**。λ 链、权重链、样本计数、优势组合、
  likelihood 对齐全闭、零非有限值。
- **独立复核**（0 启动）：从原始行重建 6 格 ΔY/ΔC/绝对值与交付件**逐格全等**；192/192 批 λ 公式
  独立重算闭合；bootstrap 用 10000×16 慢循环重算并与交付舍入一致（含独立手算反例）；
  账本、1040 行/960 tape/192 批记录/18 更新前 checkpoint 计数相符；报告表格 42 个数字串与机读件
  同源。冻结后 6 处工具级修正逐处 AST+difflib 登记（G8/曲线断言的 `epochs × 行数` 关系、
  相对 1e-5 容差、行文件通配符漏 `_s{scenario}` 段、bootstrap 舍入口径、新增消息数读数、
  判定件派生字段），**训练/采样/更新/恢复路径八件哈希未变、未重跑任何环境**。
- **独立复核抓到的一处来源缺陷（只登记不修补）**：`init_pairing_provenance` 逐格比对启动前配对件
  （提交 `df176bdf…`）与工作区文件 ⇒ 承重字段 **6/6 格 0 处变化**（两臂同初值的主张成立），但
  `r32_train.record_init_pairing` 在首次真初始化处**重写**了同一文件，工作区版本因此丢掉
  `written_before_the_first_environment_launch` 标记；`r32_freeze` 文档字符串所称"改为复核而非
  覆写"与交付代码不符。因二者均在训练路径清单内，冻结后不改，改为在报告 §1.1 如实登记，
  并把生效冻结提交里那六份带标记的原件另存为 `INIT_PAIRING_PRELAUNCH.json`
  （`all_load_bearing_fields_equal = true`，丢失的键恰只有那一个标记），
  下一轮应把运行期证据写入独立文件名。
- **结论边界**：已削弱/排除 = "成本或任务信号没进 actor"、"零梯度行占分母是主要瓶颈"。
  仍未建立 = 更多批数、不同 λ/η_λ、不同熵系数、成本项尺度取值的效应（本轮只读出比值、未干预）、
  "actor 占比上升 = 参数步更大"、独立泛化、全系统通信成本、δ=0 的部署级保证、三 seed 当总体。
  下一步唯一建议（本轮不执行、需批准）：**0 启动**地把六格剩余任务损失按场景归因
  （承重首报/严格冗余/搜索侧零接触，沿用十九轮分类），量出通信侧可达的任务上界；
  若上界接近 0，则控制器侧目标函数的进一步消融不值得再申请。
- **停止**：本地交付后停止。不续训、不调 λ、不改熵系数、不选中间 checkpoint、不加场景、
  不部署、不推送、不开启下一轮。

## 2026-09-30：ROUND31 通信控制器约束优化诊断——192/192 批 λ 公式与配对完全闭合、g_hat 从未为负；actor 确实收到任务与成本两项信号但 70 %–96 % 样本零 actor 梯度；未发现正确性错误（0 启动、0 optimizer.step）

分支 `analysis/comm-constraint-optimization-diagnosis`（起点 ROUND30 报告头完整 SHA
`915868297c08d4b01b7dfeff0ea810d5394057a3`，本地 `git rev-parse` 核验；该分支名此前不存在），
产物 `runs/comm_constraint_optimization_diagnosis_20260930/`（`ROUND31_REPORT.md`、
`ROUND31_VERDICT.json`、`ABSOLUTE_PERFORMANCE.json`、`DUAL_TRAJECTORIES.json`、
`UPDATE_SIGNAL_DIAGNOSIS.json`、`EXECUTION_BASIS.md` sha256 `e8e9029c…`、`r31_*.py`）。
纯读取 + 纯计算：新环境启动 0、新流 0、真实训练更新 0、合成 optimizer 更新 0、`.pt` 写入 0；
梯度诊断为 `autograd.grad` 于 ep0 milestone 权重的内存副本（6 运行 × 1 minibatch）。
发布状态：未推送、本轮 0 次网络尝试、该分支无 upstream（最近登记态
`not_pushed_network_unreachable`）。

- **身份**：1 040 行行内身份与文件名候选键逐项核对 **0 不一致**；6 × 11 项 milestone↔batch↔
  eval 行↔CHECKPOINT_INDEX 交叉核对全真；192 批的搜索/分配 `frozen_checkpoints` 哈希完全相同、
  optimizer 参数集合逐批等于 controller 集合、likelihood 192/192 passed；六格契约均为
  `constrained-cost-v2`/`constrained-lagrangian/2`。**未发现新的正确性缺陷**。
- **绝对表现（补齐 ROUND30 只报 Δ 的缺口）**：immediate 6.0/14 939.25 B、late 6.0/4 838.25 B。
  greedy 六格：s11 两臂 5.5/≈28 200 B（immediate 的 1.89 倍）、s22 两臂 5.5/**0 B**、
  s33 两臂 6.0/2 398.5 与 2 063.6 B。**配对任务损失 vs immediate：4/6 个 greedy 格仍为 0.5
  个/ep**，最大单场景损失 4 个按时任务（1301002）。sampled 六格全部既有损失（0.0625–0.4375）
  又普遍更贵。与 ROUND30 汇报数字 **92/92 相符**（含 `PAIRED_COMPARISON.json` 逐格）。
  登记式澄清：ROUND30 的"cheaper_but_lost_task = 0"范围是 NEW vs OLD；以 late 为参照时
  s22 greedy 两臂各有 5 个场景既更便宜又更少按时。
- **约束与 λ（6 运行 × 32 批全量重建）**：`λ_after = max(0, λ_before + 0.05·g_hat)` **192/192
  批成立**（差 < 1e-12）；`g_hat` 配对、λ 链、权重哈希链、ep128 `lambda_history` 四方逐批一致；
  `lambda_at_rollout == lambda_before` 192/192 成立 ⇒ **每批 actor 用更新前的 λ**。
  **192 批中 `g_hat` 无一次为负**（正/零 = 23/9 或 13/19），λ 只单调不降：1.0→1.5000（s11 两臂）、
  1.2375（s22 两臂）、1.4750 vs 1.4875（s33，自第 29 批分叉）。零非有限值、零版本混用、零配对错误。
  ⇒ "λ 按登记公式反应"已证，"约束被处理到满足"未证。
- **更新信号（固定批 0/15/31，18 个点）**：样本数、前缀未归属收益、`λ·A_task − A_cost`
  组合（最大误差 0.0）、批 0 似然重算（与记录差 < 1e-9）**全部与训练器记录一致**。
  只有批 0 的更新前权重落盘，批 15/31 记为 **missing**（未用更新后权重冒充、未重放 optimizer）。
  批 0 更新起点上**动作头只由 `λ·A_task`、`−A_cost`、熵驱动**（两个 critic 对动作头梯度恰为 0）：
  动作头 pre-clip L1 task 0.39–1.06 / cost 0.25–0.78 / 熵 0.0003–0.0057，两支 cos −0.73…−1.00。
  同时登记三项边界：① 全运行 **70.4 %–96.5 % 样本所在 tape 候选全被 mask**、actor 梯度恒 0
  却仍进 `surrogate.mean()` 分母（候选级可执行占比仅 5.4 %–7.1 %，s22 为 41 %–42 %）；
  ② 主干 L1 中任务 critic 88.5–107.0 远大于 policy 两项 1.4–7.2，总 pre-clip 范数与
  critic 之和之比 0.9999–1.0011；③ 每批都触发 `MAX_GRAD_NORM=0.5` 裁剪（缩放 0.186–0.210）。
  `clip_fraction ≤ 0.0115`、`|approx_kl| ≤ 1.3e-3`（训练器原字段，未重算未填零）。
- **对偶尺度（仅描述）**：greedy 开发块 `s22_greedy` 对 late 的经验交叉点 `λ_cross = 0.96765`，
  恰在训练走过的 [1.0, 1.5] 区间下沿之下；`s22_greedy` 对 immediate 为 2.98785；
  `s11_greedy` 对两条规则与 `s33_greedy` 的 `λ_cross` 均为负（在 λ ≥ 0 上被支配）。
  sampled 表内 λ ∈ [1, 1.5] 时 late 的 J 高于全部六个 learned 格。**不称最优 λ、不作充分条件、
  不作部署参数、未据此选 λ。**
- **五问**：① 绝对水平见上（"保任务"在 4/6 greedy 格未成立；"降成本"只在 s33 两臂相对两条规则
  同时成立，且仅覆盖受控上行通道）；② 约束违背与 λ 更新**正确闭合**，但违背从未反向；
  ③ actor **确实**收到登记的两项信号（同量级、近反向），范围与尺度限制已量化登记；
  ④ 材料**不支持**实现/身份问题，**支持**"有限训练内约束仍未满足"与"sampled/greedy 解码差异
  不可互换"，并给出更新尺度线索；**λ 太小、训练太少、熵太强均不足以判定**；
  ⑤ 唯一后续最小实验（本轮不执行、需批准）：**只改 surrogate 归约分母**（全部决策 tape →
  含可执行候选的 tape），用以区分"对偶上升在本登记尺度内太慢"与"成本信号被零梯度样本稀释"；
  λ/ηλ/熵/学习率/clip/批结构/预算/场景/seed/搜索与分配 checkpoint 全冻结，seed22 作内部对照，
  预计 960 次环境启动（64 个 reference 复用归档行，0 次启动）。
- **停止**：本地交付后停止，不续训、不调参、不追加评估、不部署、不自动推送。

## 2026-09-30：ROUND30 原/新期限编码从初值配对训练消融——1040/1040 零失败；greedy 任务 48/48 持平、成本 −9/0/−335 B/ep 方向不一；sampled 有零星任务增益（启动 1040/1080）

分支 `exp/comm-deadline-encoding-ablation`（起点 ROUND29 完整头
`99eb746baaa073550f9062fec5c87a4b4711827b`，继承 ROUND28 `21250b5`），产物
`runs/comm_deadline_encoding_ablation_20260930/`（`ROUND30_REPORT.md`、
`ROUND30_VERDICT.json`、`PAIRED_COMPARISON.json`、`DIAGNOSTICS.json`、
`CONFLICT_RECHECK_3_1.json`、`CLOCK_RECHECK_3_2.json`、`RUN_PREFLIGHT.json`、
`FIRST4_GATE_{old,new}_s11.json`、`INIT_PAIRING*.json`、`START_LEDGER.json`、
`CHECKPOINT_INDEX.json`、`TRAINING_CURVES.json`、`ARTEFACT_MANIFEST.json`、
`CODE_IDENTITY.json`、`CONFIG_AND_MODEL_MANIFEST.json`、`ROUND30_PROTOCOL.md`）。
执行依据 sha256 `d80a7055…`，冻结件 freeze_sha256 `b05f07bb…`；
`publication_status=PENDING_NETWORK`、本会话网络重试 0。

- **两处冲突零启动裁定**：① ROUND29 的"seed33 greedy 开发块 5/16 失败"实为 **seed11 行误挂
  seed33 标签**（96 行身份表 + checkpoint 参数哈希；seed33 final greedy 实为 16/16 全按时，
  ROUND24 判定成立；ROUND29 块归属核心不受影响）；② `clock.now = tape step − 1` 经源码调用
  顺序（relay.tick 先于 clock.advance_to）+ 81 tape 756 个未饱和行 0 失配 + 首候选步恒等式
  478/478 独立证实。
- **接线**：作用域 collector = 父类 `_choose_masked` 逐字复制 + difflib 登记单行差异
  （特征行构造）；OLD 与父类 tape 逐字段一致；同 seed 两臂参数/optimizer/λ 逐字节相等；
  64 个新跑 reference 与 ROUND24 归档行 behaviour_sha256 逐字节一致 64/64；
  configuration_hash `e9b2c067db5951f2` 与 ROUND24 相同；预检 132 项 0 失败（合成
  optimizer 步 44 ≤ 64）。
- **执行**：64 reference + 768 训练（6 cell × 128 集，两臂首批门禁 G0–G7 全过）+ 192 评估
  + 16 late 规则 = **1040/1040 正常、0 失败、储备 40 未动**。
- **结果（配对 Δ=NEW−OLD，16 场景描述性区间）**：greedy **ΔY 48/48 全零**、无目标集合交换、
  无"省字节丢任务"；ΔC −9.0 / 0.0 / −334.9 B/ep（s11/s22/s33，CI 分别 [−27,0]/[0,0]/
  [−847.5,0]）。sampled：ΔY 0 / +0.0625 / +0.1875（s22 场景 1301003 +1 目标且省 14 328 B；
  s33 场景 1301002 +3 目标且贵 552 B），ΔC −57.0 / −4197.8 / +58.5 B/ep（中间者 CI 不含 0）。
- **诊断**：OLD 臂 272 个不同原始期限（61.1–180.2 步）在训练内全部折叠为 1 个编码值；
  NEW 臂保留 272 个（0.061–0.180）——ROUND29 审计在训练内部复现。λ：s11/s22 两轨完全相同
  （1.5/1.2375），s33 微叉（1.4875 vs 1.4750）。
- **五问结论**：① 两臂仅有登记的单列编码差异（是）；② 任务 greedy 全零变化、成本小且方向
  不一致；③ 任务侧跨 seed 一致（全零），成本侧不一致；④ greedy 与 sampled 结论不同
  （任务增益只出现在 sampled 的 2/16 场景）；⑤ 学习/任务表现改善的证据**未建立**——
  无确认性显著性/非劣效/泛化主张，出现收益也只能归因于整项编码替换（信息保留与数值尺度
  同时改变）。
- **下一轮建议（仅建议、非授权）**：在编码替换内部解耦信息保留与数值尺度（如新列再做无
  截断尺度归一），或推进 ACK/接收端状态的观测设计讨论；均需另行批准。

## 2026-09-29：ROUND29 发送端期限编码审计与最小观测原型——期限列在全部 102 个首报点饱和为 1.0；单列新编码保留差异；学习收益未证（0 启动）

分支 `analysis/comm-deadline-encoding-audit`（起点 ROUND28 完整头
`21250b5eb093dd1a7f97cd06e9993ef1ac6abab0`），产物
`runs/comm_deadline_encoding_audit_20260929/`（`ROUND29_REPORT.md`、
`ROUND29_VERDICT.json`、`ENCODING_AUDIT.json`、`SCENE_BLOCK_CORRECTION.json`、
`r29_encoding_prototype.py`、`r29_proto_tests.py`、`r29_audit.py`、
`r29_proto_on_data.py`、`ENCODING_PROTO_ON_DATA.json`）。执行依据 sha256
`873df4f8…`，首启前独立提交；`publication_status=PENDING_NETWORK`、本会话网络重试 0；
**0 环境启动、0 新流、0 训练、0 optimizer.step、0 生产套件运行**。

- **场景归属更正**（`SCENE_BLOCK_CORRECTION.json`，旧报告保留）：1301000..1301015 是
  ROUND22/24 **开发**块（"仅作开发历史"，其 seed33 final greedy 5/16 场景有按时失败、
  行已在盘上），不是 ROUND25 块，也不是含 1400031 失败的块；1400031 在 ROUND25 评估块
  1400000..1400063。ROUND28 建议中的"ROUND25 开发块"称谓据此更正。
- **真实编码路径**（18 个生产/导入模块与 ROUND28 启动身份逐字节一致）：`remaining_time =
  registry 快照 deadline − clock.now`（步，`comm.py:819-820`，动作前可用、发送端合法可见）；
  第 12 列实际为 `clip(remaining/50, -1, 1)`（`comm_learning.py:113`，
  `URGENCY_SCALE_STEPS=50`），`urgency_now` 另用 25 步阈值（:114）；horizon=1000 是运行前
  常量且等于配置 steps（`comm_learning.py:71`、`v2_common.py:110`）；collector→网络无二次
  缩放/归一化（`budget_masked.py:177-179`、`train_pilot.py:54-75`）。性质判定：**命名不
  准确 + 声明式有损编码，非实现违反契约**。
- **信息损失量化**（三来源 102 个首报点，分来源不合并；原始期限由 registry deadline 与已
  实证时钟规则 `now = tape step − 1` 重建，197/197 行与 102/102 点交叉验证）：A 源 ROUND28
  96 点原始期限 62.9–177.6 步、95 个不同值，编码列仅 **1 个值（全 1.0）**；B 源 ROUND27
  target-0/5 两点、C 源 ROUND26 四改善场景同此（全部饱和）。102/102 记录无法从编码值恢复
  原始期限。15 个恒定列的恒定原因逐列分类（选择条件/编码截断/证据不足）。
- **原型与测试**：候选列 `remaining/1000` 不截断、只改这一列（其余 18 列由构造保证一致），
  登记语义版本 `comm-send-state/1r29-deadline-over-horizon` 与"旧 checkpoint 不得静默换用"
  的兼容规则；167 项纯计算检查 0 失败（从真实 `SendItem`/`SendObservation` 进入，预期值手算）；
  原型作用于 102 个真实原始值 **3/3 来源完整保留差异**（新列 range 0.063–0.178）。
- **三问结论**：(1) 是——编码压缩了合法可见的期限差异；(2) 是——新编码在现有数据中保留这些
  差异（仅就可证的信息保持而言）；(3) **本轮不能**证明这些差异能改善学习或任务表现。
- **下一步建议（仅建议、非授权）**：从初值配对训练消融"原编码 vs 单列新编码"，网络/目标/
  训练预算/其余超参一致，唯一差异是第 12 列定义，启动额度另行批准。

## 2026-09-29：ROUND28 训练场景首报反事实样本诊断——96/96 accepted、0 任务效应，首报决策点特征近乎退化（启动 113/120）

分支 `exp/comm-cost-firstreport-diagnosis`（起点 ROUND27 头 `5982677`），产物
`runs/comm_cost_firstreport_diagnosis_20260929/`（`ROUND28_REPORT.md`、
`ROUND28_VERDICT.json`、`PAIRED_SAMPLES.json`、`ANALYSIS_SUMMARY.json`、
`INTERVENTION_POINTS.json`、`FIRST_SCENE_GATE.json`、`RUN_PREFLIGHT.json`、
`CODE_IDENTITY.json`、`ARTEFACT_MANIFEST.json`、`ROUND27_SCOPE_REGISTRATION.json`）。
执行依据 sha256 `f9185812…`，首启前独立提交；`remote_verification=NOT_PERFORMED`、
`publication_status=PENDING_NETWORK`、本会话网络重试 0。

- **启动 113/113 正常**（16 baseline + 1 禁用门禁 + 96 干预），硬上限 120、储备 0 未动、
  0 失败、0 重试、0 训练、0 optimizer.step；单一启动身份 `d5b3cee7…` 且可从 harness 提交
  `0db4ef2` 复现（两个模块按 CRLF 工作树字节）。
- **§1 套件范围登记**：只读源码核查确认 ROUND27 那次生产套件包含 17 个 `build_joint_run`
  调用点、8–20 步长短仿真构造（从未进入 ROUND27 五格账本）；措辞更正为“套件通过但范围含
  账本外仿真构造”，本轮不再重跑套件、未凭通过推断零、未重新追账；模型/身份/结果污染检查无。
- **选择算法（冻结于干预启动前）**：首次合法可付机会且原 greedy 动作为 HOLD 才登记干预点；
  16 场景 96 个目标全部合格、0 排除（这批 greedy 在首报机会上全是 HOLD）。
- **结果**：ΔY 0 正 / 96 零 / 0 负（16 个 baseline 全部 6/6 按时 ⇒ 构造上没有救回信号，
  ROUND27 的结局改变效应只出现在有失败目标的开发场景 1400031）；ΔC 83 正 / 6 零 / 7 负
  （中位 +2484 B；最负 −5238 B 来自早发抑制了后续两次发送）。96/96 干预消息被 3–6 个接收方
  accepted 且 0 superseded，但 0 个改变任务结局、只有 60/96 触动分配/作业链。
- **特征观察**：首报决策点上 19 维发送端特征有 15 维恒定（age 0、sent_count 0、pending 1、
  first_emission 1、revision_delta 1、remaining 裁剪饱和、urgency 0 等），仅
  step_over_horizon / queue_backlog_targets / budget_remaining_fraction /
  has_future_requirements 变化；96 个目标期限余量全部 > 50 步（紧性信号不可见）。
- **下一步建议（仅建议、非授权）**：把同一冻结选择器+最小单发覆盖协议跑到含已观测失败的
  场景族（ROUND25 开发块 1301000..1301015），以制造标签方差；下一步方向指向观测设计分支，
  当前合法观测下首报决策“几乎无信息量”，动作价值研究暂无可估计信号。

## 2026-09-29：ROUND27 固定失败场景的首报提前干预——E5 在该场景该实现中改变失败结局；E0 accepted 仍不足（启动 5/8）

分支 `exp/comm-cost-firstreport-intervention`（起点 ROUND26 头 `e22db99`），产物
`runs/comm_cost_firstreport_intervention_20260929/`（`ROUND27_REPORT.md`、
`ROUND27_VERDICT.json/.md`、`ROUND27_PROTOCOL.md`、`FIVE_PATH_COMPARISON.json`、
`PER_TARGET_TIMELINES.json`、`FIRST2_GATE.json`、`RUN_PREFLIGHT.json`、`CODE_IDENTITY.json`、
`ALIAS_REGISTRATION.json`、`TARGET5_STEPS3_11_READBACK.json`、`ARTEFACT_MANIFEST.json`）。
执行依据 `EXECUTION_BASIS.md`（sha256 `eeb982de…`），首启前独立提交；`remote_verification=
NOT_PERFORMED`、`publication_status=PENDING_NETWORK`、本会话网络重试 0。

- **五路径（场景 1400031，各一次启动）**：L/L0/E0 均为 on_time 4、expired 2、10578 B、4 发射；
  **E5 = on_time 6、expired 0、7764 B、3 发射**（target-5 t=60、target-0 t=70 均按时）；
  **E05 = on_time 6、0、10578 B、4 发射**（两目标 t=60/66 按时）。0 失败、0 重试、
  储备 3 未动、0 训练、0 optimizer.step、0 阈值调整、生产树零 diff。
- **干预器**：`FirstReportAdvancer` 在 `BudgetMaskedCollector` 顺序预算路径内单点覆盖
  HOLD→SEND（同 shadow gate、同报价、同候选顺序、不绕 mask、不免费、不追加发射；每目标一次、
  首发射后恢复原 greedy）；零环境预检 16/16，生产套件 OK(skipped=2)。
- **门禁**：L 与 L0 行为字段零差异、tape 逐行相同；L 与 ROUND25 该场景行 27 个共同字段零差异、
  tape 逐行相同 ⇒ 干预器安装零扰动。
- **机制（描述性）**：E0 首报被 5/6 accepted 但 target-0 分配链与 L 逐事件相同 ⇒
  "accepted 不足以改变结局"的一例，不是"提前发送无效"的证据；E5 的 step-1 accepted 之后
  target-5 分配链自 t=1 展开并按时完成，但同路径里 target-0 的首报仍在 step 32 且全部
  superseded ⇒ 两个目标的变化不共享同一机制。
- **措辞更正四条已登记**（早期首报是关联非充分原因、ACK 盲区未证须加观测、superseded 不等于
  可安全删除、视距/gossip 是未计量旁路非免费）；immediate/send_once 行为等价（64/64 场景
  零刷新）⇒ 登记别名、不新增同义规则。

## 2026-09-29：ROUND26 ROUND25 统计复核与任务得失机制诊断——结论不变，失败近因在分配侧（0 新启动）

分支 `exp/round26-round25-recheck`（起点 ROUND25 完整头 `fdf97d1`），产物
`runs/round26_analysis_20260929/`（`ROUND26_ANALYSIS.md`、`STATISTICAL_RECHECK.json`、
`SCENE_1400031_DIAGNOSIS.json`、`IMPROVED_SCENES*.json`、`ROUND26_SUMMARY.json`）。

- **统计复核 22/22 通过**：手算反例（values=[0,10]、索引 [[0,0],[0,1],[1,1]] ⇒ [0,5,10]）；
  独立索引取值路径从 192 份 final record 的原始整数重算，与提交件逐项一致；
  三个判定分别保留：**主成本优势未建立**（对 late dC CI [−1849.2, +502.2] 含 0）、
  **对 immediate 观察到任务权衡**（1400031 净 −2）、**seed 稳定性与约束保证未建立**。
- **1400031 机制**：learned 与 late 对两个丢失目标的链**逐字节相同**；learned 两份迟发首报
  8/8 `delivery_superseded`（免费信道已送到更新版本）；压制期间始终合法可付（actionable、
  0 拒付）；发送精确发生在 p(SEND) 越过 0.5 时（压 12/33 步）。近因失败在分配侧：target-5
  空窗 44 步（11→55）与 29 步（66→95），immediate 则连续作业按时完成。分类 A+B+C，
  反事实（早发是否救回）证据不足（E）。
- **四个改善场景**（1400010/1400014/1400017/1400020）：gained 目标均发现于 147 之前，
  learned 首报被接受（3-6 条、0 superseded）且**首分配与 accepted 投递同刻**；late 从未或
  520 步才分配。64 场景描述统计：learned 投递 25.6% accepted / 74.4% superseded。
- **actor 信息边界**：19 维特征只有发送端紧迫性信号，ACK 结构性关闭、无接收端知识——
  决定首报是否有效的接收端状态不在合法观测内，刀锋 p≈0.5 是紧迫性先验而非"接收端需要"估计。
- **唯一推荐下一步（未执行，需批准）**：最小干预 `first_report_at_discovery` vs `always_hold`
  同批 64 场景、≤128 启动，检验首报时机的边际因果作用；不续训、不调阈值、不换候选、不部署。

## 2026-09-28：ROUND24 完成；ROUND25 固定候选复验执行完毕——降本证据未建立、观察到一次任务数退化（本地完成，远端发布待补）

**ROUND24（修复版从初值重跑，672/672 启动，0 技术失败）**：三个通信模型全部从
`Controller(seed)` 初值训练，不读 v1 checkpoint；训练/评估/参照/规则四类启动闭合，门禁全过。
科学结论：**学习收益未建立**——三个 seed 差异大（seed33 最终 greedy 在 16 个开发场景任务数与
immediate/late 全平且成本更低；seed22 greedy 0 B、seed11 与 sampled 接近 immediate 成本），
sampled 与 greedy 的模式差是结构性的（seed33 的 p(SEND)≈0.33–0.41）。最稳的规则臂仍是
`late_discovery_gate_147`（0 任务差、−10 101 B/ep、16/16 联合占优）。

**ROUND25（固定 seed33 ep128 greedy 候选 × 64 个新场景 1400000..1400063 × 三臂，192 启动、
0 失败、0 储备消耗）**：执行依据是本会话用户直接指令（`EXECUTION_BASIS.md`，hash 已登记），
旧 spec 的离线 hash 校验维持 MISMATCH、未声称已核验。

- 绝对：learned 2 779.7 B/场景、5.844 个/ep；immediate 14 672.8 B、5.875 个/ep；
  late 3 476.0 B、5.781 个/ep；拒付 0、预算使用率 learned 9.5% / late 11.9% / immediate 50.1%。
- 唯一主成本比较（vs late）：dC 均值 −696.3 B/场景，95% CI [−1 849.2, +502.2] **含 0**
  ⇒ `primary_cost_evidence = lower_cost_not_established`；dY 60 平 / 4 改善 / 0 退化。
- 对 immediate：dY 63 平 / 0 改善 / **1 退化**（1400031 净 −2，失去 target-0 与 target-5，
  该场景成本也更低）⇒ `observed_task_count_status = task_count_tradeoff_observed`；
  dC 64/64 更低（均值 −11 893 B，CI [−12 567, −11 492]）。
- 目标身份：除该退化场景外无身份替换，没有"净数打平但换了目标"的场景。
- 统计缺陷登记：首次 bootstrap 区间有重采样算术缺陷（`matrix @ values` 而非
  `values[matrix].mean(axis=1)`），已用更正算法重算；均值与计数不变、0 episode 受影响，
  作废的首次判定原样保存（`SUPERSEDED_FIRST_VERDICT.*`、`STATS_CORRECTION.json`）。
- 结论边界：候选是看过开发集后**事后选定**的固定模型；64 场景是探索性复验规模，不是功效样本；
  任务全打平不证明 delta=0 非劣；不把固定候选的结果写成学习方法普遍优于强规则。
- **无自动下一步授权**：不续训、不换候选、不追加确认、不部署。远端 `remote_verification =
  NOT_PERFORMED`、`publication_status = PENDING_NETWORK`（本会话按指令 0 次网络重试），
  网络恢复后普通推送并核对远端，不 force。

## 2026-09-26：返回学习型通信主线——成本第一次被写成显式目标；降本设计与 pilot 方案已交付待批准（第十九轮，0 新 episode、0 训练、0 生产代码改动）

产物目录 [runs/comm_cost_objective_design_20260926/](runs/comm_cost_objective_design_20260926/)：
范围与起点核验 [ROUND19_SCOPE.md](runs/comm_cost_objective_design_20260926/ROUND19_SCOPE.md)、
步骤① [ROUND18_ARCHIVE_CLOSURE.md](runs/comm_cost_objective_design_20260926/ROUND18_ARCHIVE_CLOSURE.md)
（机读 `ROUND18_ARCHIVE_CLOSURE.json`，含 120 格逐格明细）、
步骤② [COMM_COST_COVERAGE.md](runs/comm_cost_objective_design_20260926/COMM_COST_COVERAGE.md)
（机读 `COMM_COST_COVERAGE.json` + `LEARNABILITY_PROBE_OFFLINE.json`）、
步骤③ [CONSTRAINED_COMM_COST_DESIGN.md](runs/comm_cost_objective_design_20260926/CONSTRAINED_COMM_COST_DESIGN.md)、
步骤④ **待批准** [CONSTRAINED_COMM_COST_PILOT.md](runs/comm_cost_objective_design_20260926/CONSTRAINED_COMM_COST_PILOT.md)、
`ROUND19_VERIFICATION.json`、`MANIFEST.json`、`rejected/`（本轮内部三次改动的旧件，隔离不删除）。
分支 `docs/comm-cost-objective-design`，**起点 = 第十八轮头
`4c71fead0047298101869200a8518059afc42ccc`（本地逐字符核对）**，未切回旧代码执行。
**边界**：0 新 episode、0 新流、0 训练、0 生产代码改动、0 默认值改动、0 搜索/分配权重改动、
封存集 `800000..800399`/`900000..900199` 未开启、pilot 未启动、其他会话的未跟踪文件一个未动。

* **步骤①：第十八轮按现状归档，两个数用它自己的日志闭合**。120/120 格、
  24 项检查 × 120 = **2 880 次全部通过**：发现 + 未发现 = 放置 = 6/集（A 351+9、B 291+69）、
  未发现目标**从未出现**在任务生命周期日志里、每笔 `airtime_bytes` = transcript 逐包
  `payload × subscribers` 之和（0 例不符）、`subscribers` 恒 {6}。
  按时/完成/过期用生产函数 `lifecycle_metrics.window_metrics` 重算，与已交付文档逐位一致
  （A 5.833333、B 4.833333；文档写 5 位是四舍五入，容差已登记，旧件在 `rejected/`）。
  **新增的限定语**：120 集里"发现但未按时"合计只有 **2 个** ⇒ 以按时任务为约束在此无分母。
* **步骤②：只用磁盘上已有的部署数据把"通信成本"量化**。单价 **414 B 载荷 / 2 484 B 一份首报**；
  第十六轮四臂 + 第十八轮两臂的 `refresh_emissions` **合计 0**、`emissions == 发现数` ⇒
  **部署默认配置没有"省刷新"这件事**；240 个部署格里 `budget_refused_sends`/`capacity_refused_sends`
  恒 0 且 `budget_enforced=null` ⇒ **磁盘上没有"部署 + 真预算"的格子**，成本端点仅
  **0 B（`always_hold`）与 14 573.3 B/ep（`immediate`）两个**，上行值 **46 020.9 B / 新增 1 个按时任务**。
* **冗余被逐目标判定（与沉默臂同 60 seed 配对）**：352 份首报 = **19 份承重（5.46 %、795.1 B/ep）
  + 110 份严格冗余（31.24 %、4 552.7 B/ep）+ 223 份"买到时间"（63.30 %）**；
  宽松口径 94.5 % **明令不得当作可省估计**（沉默改变机器人轨迹，免费视距信道取决于位置）。
  机制量：该层任务完成时剩余中位 **96.5 秒**（p10 53.1）、发现到完成中位 20 拍、
  A1 **352/352 全按时**。
* **承重的那 5.5 % 在决策时刻可分**：报告步号 **AUC 0.9174**（承重中位 214 对 其余 4）；
  剩余时间只有 0.585 且**承重者剩余时间反而更多**、`severity` 0.4608 ⇒
  决定承重的是"发现晚不晚"，不是"任务紧不紧"。单阈值规则 `步号 > 147` 的反事实记账读数
  **3 212.2 B/ep（账单 22.0 %）保住全部 352 个按时**——**只能当上界与基线定义**（同批 seed 读出、
  混合静默运行不等于任何归档臂）。
* **公平强规则基线被换掉**：隔离路径的 `send_once`（27 134.6 B / 8.875 个/ep）在部署默认配置下与
  `immediate` **行为重合**（那里本无刷新可省）⇒ 部署降本基线是
  `always_hold` + `immediate` + 一条同预算规则臂 `late_only`（需一次最小生产实现，待批准）。
  `comm_policy.py:305-307` 那句"`event_priority` 是最强简单规则"与实测前沿矛盾，本轮按实测取值、**不改代码**。
* **步骤③：设计 = 成本进目标、效果作约束，但 reward 一字不改**。落地形式是
  **额度档扫描 + 同额度档配对比较**（报"效果守得住的最低档"，单位一律写明：B/episode 与个/episode），
  λ/Lagrangian 明确留作下一台阶，前置条件是先有 ≥3 个 binding 的部署内点——
  这是对 N4"字节不是 reward 项，所以省通信不能被假定会自然学出"与 §6.3:1189 的正面回答。
  **合格出口预固定两个**：P＝学习臂同额度支配同预算规则；N＝规则已吃满该档前沿（同样正常交付）。
  关键防自欺条款：**执行器拒付不算省**，主指标只取 policy 归因 attempts。
* **步骤④：pilot 方案已冻结待批准**，本轮不启动。运行点标定门槛 K1（"发现但未按时" ≥ 0.50 个/ep）、
  K2（`immediate − always_hold` ≥ 0.30 个/ep 且 CI 下界 > 0）、K3（额度真 binding）按阶梯顺序取第一个
  满足层，**与任何臂的胜负无关**（`总纲 §11:1457` 禁止用更紧 deadline 让学习好看）；
  额度档 {0.15, 0.30, 0.60} × B̄；δ = **0.10 个/episode**；启动上限 **1 148 正常 + 40 储备 = 1 188**
  （单格耗时按第十八轮实测 16.054 s/格估 ≈ 5.1 h 串行、6 路分片 ≈ 51 min）、训练 ≤ 3 seed × 400k；
  停止条件 S-1…S-7 全部预登记。**要批准的是**：`late_only` 的一次最小实现 + 这套矩阵；
  **明确不要**：加训练量、扩动作空间、改观测维度、开 ACK、动搜索/分配权重、开封存集。
* **治理件按登记式修订，原文不动**：`PROJECT_CHARTER.md` 升到 **v4.2**，§8.1 与 §6.3 各补一条
  "部署降本问题上基线/目标侧的替换"并保留原句；第十八轮那节改为"历史状态（已被上一节接替）"。
* **交付状态：本地已提交，远端未核验（不伪称交付成功）**。本会话**成功的远端接触数为 0**：
  要求核对的
  `docs/comm-cost-objective-plan@72c4c3f71e07339bf6533fbd771aef3b2682103c` **未能核对**——
  `git fetch` 1 次 + `git ls-remote` 3 次全部失败（`Recv failure: Connection was reset` /
  `Failed to connect to github.com port 443`），而 `pypi.org` 可达 ⇒ 只有 `github.com:443` 不通；
  仓库私有 ⇒ 未认证 `api.github.com` 返回 404（本轮**不注入凭据**）；`git push` 未能执行成功。
  指定要读的 `ROUND19_COMM_MAINLINE_RETURN.md`、`ROUND19_CONSTRAINED_COMM_CONTROLLER_DESIGN.md`
  **未读到**（本地对象库、reflog、`fsck` 游离对象、全库文件名搜索均无），**cherry-pick 未执行**。
  那两份文档到达后若已冻结基线/容忍度/资源上限，**本轮提案一律让位**（恢复命令在 pilot 件 §七）。

* **同日修订（2026-09-26 晚，独立提交，旧件一字未改）**：`COMM_COST_OBJECTIVE_DESIGN_REVISED.md`、
  `COMM_PILOT_STAGED_APPROVAL.md`、`DECISION_SUMMARY.md`、`DESIGN_SPEC_COMPLIANCE.md`（**挂起待补**）、
  `BUCKET_ARITHMETIC_RECHECK.json`（`c19_recheck.py`）。五处更正：①三桶改为**整策略对照下的描述性标签**
  （`outcome_differs_under_full_silence` 19 / `known_no_later_and_outcome_same…` 110 /
  `known_later_but_outcome_same…` 223），**从未做单消息干预**，故不是任何一条消息的因果贡献；
  ②本节与总纲/README 里写在目标数旁的 **5.46 %/31.24 %/63.30 % 是字节份额**（÷874 398 B），
  按目标数应为 **5.40 %/31.25 %/63.35 %**（÷352），两种分母不得混写（重算与旧件逐值相同、四项闭合全真）；
  ③AUC 0.9174 与阈值 ĉ=147 标为**开发集探索性结果**（无划分、无拟合、阈值与标签同批数据，
  `1100000..1100059` 不得再作未见测试集）；④`late_only` 明确为**"晚发现才发送"（门控）**、
  不是"延迟已有信息"（推迟），改名 `late_discovery_gate(ĉ)`；⑤费用一律称
  **载荷尝试成本（当前计费模型口径）**，不称真实空口或全系统通信成本，未计量项（视距刷新、gossip、
  ACK、轮询、事件总线、协调器 advice、被拒发送）继续单列。
* **推荐路线也随之改**：不再是"额度档扫描 + 同档配对比较"，而是
  **最小化期望已计量成本、约束相对固定参考策略（`immediate`）的按时损失 ≤ δ**，
  用**带对偶变量 λ 的约束式策略梯度**实现（新增 cost critic + λ 更新 + 配对约束估计；
  actor 观测/动作/mask/计费单点/协议/默认值全不动，**无 schema 变更**）；
  预算仍是硬执行栏杆、**不承担**成本目标，"≥3 个 binding 内点"的前置**已删除**
  （成本水平是训练输出而非输入）；额度扫描降为**备选**。授权改为 **A/B/C 三阶段**：
  A = 0 启动（纯合成正确性测试，生产改动目标 0、硬上限 2 个文件）、
  B ≤ 24 次启动（0 训练步，闭环接线 + "λ=0 且强制全发时逐字节复现 `immediate`"的强规则检查）、
  C ≤ 488 次启动（硬上限 520，3 seed × ≤400k 步，**另行批准**）。
  δ=0.10 个/episode、降本门槛、训练量、规则臂实现**全部仍是建议、待批准**；
  `late_discovery_gate` 只是新候选对照，**不代表学习可行性已建立**。
  运行点门槛 K1 也已更正：约束的信号是**配对损失**而非参考臂的失败数 ⇒
  **6 目标部署层可判**（全静默相对 `immediate` 的配对差实测 0.316667 个/ep），
  不需要收紧 deadline 或加目标数；但 δ=0.10 在 L≈0.05 时需 **n=120** 才可判定（n=60 时 CI 上界 0.105）。
* **仍未获得的两份原 spec**：`ROUND19_COMM_MAINLINE_RETURN.md`、`ROUND19_CONSTRAINED_COMM_CONTROLLER_DESIGN.md`
  （提交 `72c4c3f71e07…`）⇒ `DESIGN_SPEC_COMPLIANCE.md` 挂起待补，本轮推荐方案标注为
  "依据使用者 2026-09-26 指令，**未经原 spec 核对**"。已请求以粘贴内容 / 本机路径 / `git bundle`
  三种方式之一传送；在原件到达前不再凭摘要推定 spec 内容。

## 2026-09-25：搜索侧逐时刻证据补齐——探测链无缺陷、遗漏全是零接触；换策略后未发现目标反而增 1.0 个/episode（第十八轮，158/160 启动，0 训练）

产物目录 [runs/search_trace_and_policy_20260925/](runs/search_trace_and_policy_20260925/)：
协议 [SEARCH_TRACE_AND_POLICY_PROTOCOL.md](runs/search_trace_and_policy_20260925/SEARCH_TRACE_AND_POLICY_PROTOCOL.md)
（**在任何 episode 之前单独提交**，R1–R5 五条修订把"缺日志仍可先跑"的歧义消掉）、
[SEARCH_TRACE_IDENTITY_REPORT.md](runs/search_trace_and_policy_20260925/SEARCH_TRACE_IDENTITY_REPORT.md)、
[SEARCH_MISS_TIMELINES.md](runs/search_trace_and_policy_20260925/SEARCH_MISS_TIMELINES.md)、
[SEARCH_DETECTION_CORRECTNESS_VERDICT.md](runs/search_trace_and_policy_20260925/SEARCH_DETECTION_CORRECTNESS_VERDICT.md)、
[SEARCH_POLICY_COMPARISON_RESULTS.md](runs/search_trace_and_policy_20260925/SEARCH_POLICY_COMPARISON_RESULTS.md)、
`RUN_LEDGER.json`（计划块冻结 + 活账）、`MANIFEST.json`、`ROUND_VERIFICATION.json`（`passed=true`）。
分支 `analysis/search-trace-and-policy`，起点 `analysis/real-search-undetected-attribution@80af16b`。
**边界**：0 训练、0 奖励改动、0 通信协议/预算/ACK/丢包/拓扑/观测改动、0 机器人数增加、
0 目标数增加、0 deadline 收紧、`detection_radius` 仍为与 checkpoint 绑定的 5、步数上限仍 1000、
未用 mock 后端替代真实环境、未触碰封存测试集；第十七轮与第十六轮的协议/判据/结果一字未改。

生产改动只有一处能力变更（B1 被动旁路，默认关闭）：新增 `integrated_sar/search_probe.py`
与 `integrated_sar/tests/test_search_probe.py`，`piloc/src/Map/timemap.py` 加默认 `False` 的
`search_trace_enabled`，`adapters/piloc.py` 记三视图并集，`joint_driver.py` 加
`--search-probe` 与 `--search-action-policy {checkpoint,frontier}`，
`info_probe.IDENTITY_FIELDS` 扩到探针自述字段。

* **B1 的被动性是测出来的**：两个候选 seed × 日志开/关 × 两次重复（8 格 × 1000 步），
  逐步比较动作、奖励、`done`、`agents_pos`、`goals_pos`、`_py_rng` 与 `_np_rng` 状态摘要，
  再加原有生命周期/检测/发射/结局四类记录——**32 项检查全为真**，同条件两次重复亦相同。
* **历史 14 格全部复现**（D1 七个遗漏 seed + systemloop 三个 + 规则先行登记的 4 个无遗漏控制
  `1100000..1100003`）：配置行为字段、`map_sha256`、初始布局、两个 checkpoint 摘要、
  **完整检测序列**逐条与归档相同；0 格记为"未复现历史运行"。
* **正确性判定 = `no_defect_found`**：第十七轮留下的 2 个候选（`1100051`、`1100053`）
  被逐时刻证据判为类别 3——机器人在任何探测时刻都距目标 Chebyshev 6 格以外；
  12 个遗漏目标的时间线里类别 1 = 0、类别 2 = 0、类别 4 = 0。
  三视图闭合是逐步强制的：120 000 个（格 × 步）检查里 **0 步不一致、0 条身份异常**
  ⇒ **不进修复分支**，因为无可修缺陷。
* **正式矩阵 120/120 完整、缺格 0**：同一真实环境、同 seed、只换动作提供器。
  每 episode 未发现目标 **A 0.150 vs B 1.150**（分母 360 放置：2.50 % vs 19.17 %），
  配对差 **A − B = −1.000 个/episode**（= −60 个/批），95 % CI
  [−1.333333, −0.666667]（按 seed 配对百分位 bootstrap，B=10 000、RNG 20260925）；
  逐 seed **35 个 A 更少 / 20 打平 / 5 个 B 更少**。⇒ 现成的 frontier 不是更好的搜索侧替换。
* **预登记判据按原文跑完，两条都不成立**：停 1 需两臂 < 0.5 % 且 CI 含 0（B 为 19.17 %、
  CI 不含 0 ⇒ 不成立）；停 2 需 CI 下界 > 0 且 ≥ 15/60 seed 非零（非零 40 个满足、
  方向相反不满足 ⇒ 不成立）；守卫 G 未触发（类别 4 占比 0.000）。
  两条的符号都按"换策略能减少遗漏"一侧写，实测反号 ⇒ 属协议 §七 的"区间内"，
  如实报 **未达判据、正常交付**，**不追加更强策略、不追加 seed、不训练**；
  这也不是 R5 所说的判据冲突（无需两个互斥解释各说一遍）。
* **系统级读数只能这样写**：按时完成任务 A 5.833 vs B 4.833 个/episode（+1.000，个/批 +60）。
  下游任务的属性随发现位置/时刻生成 ⇒ 这是搜索策略引起的**总系统效应**，
  **不是**"对完全相同任务集的分配能力提升"。成本顺带读数：墙钟 16.05 s vs 12.66 s/episode，
  上行字节 14 529 vs 12 007 B/episode（B 更省是**因为它发现得更少**，通信配置零改动）。
* **登记在案的随机性**：同 seed 同初始场景不保证两臂后续目标轨迹逐位相同（策略改发现时机
  就改随机数消耗次数），所以只主张"共享初始场景与 seed 的配对策略比较"。
* **额度**：158/160。超正常计划的 8 次全部走储备（4 次记录器 `TypeError` 技术失败 +
  1 次接线格 + 3 次诊断格），产物隔离在 `rows/rejected/` 与 `rejected/`，未删除；
  储备余 2，未用于挑选案例或提高显著性。
* **测试（本轮实测）**：新增回归 `integrated_sar/tests/test_search_probe.py` **11 项全绿**
  （在真实 `timemap.Environment` 上跑，覆盖多机器人同看一目标、多目标同格、删除后下标左移、
  移动前后可见性翻转、初始可见但移动后不可见、空目标集合正常终止、探针卸载/异常恢复）；
  项目套件 **Ran 713 tests，OK (skipped=2)**，其中 `test_info_probe` 的 12 项仍绿。
* **对学习型通信研究**：搜索骨干在此场景分布下把 351/360 放置转成发现事件、
  剩余 2.5 % 是零接触而非事件链丢失 ⇒ 通信侧结论不必再向搜索侧借解释；
  第十六轮"部署路径靠环境自带传感旁路"的诊断不因本轮而改，本轮也不为该诊断提供新证据。
  **主线仍是学习型通信控制**，本轮是整体系统的搜索侧诊断；搜索质量被证明是一个真实杠杆，
  但拉动它的代价（训练/奖励/场景改动）全在本轮禁止清单内。
* **交付状态：本地已提交，远端未核验**（不伪称推送成功）。开跑前 `git ls-remote` 1 次失败，
  交付阶段 `git push` 6 次 + `git fetch` 1 次 + `git ls-remote` 2 次全部失败（`github.com:443` 连不上 /
  `Recv failure: Connection was reset`）⇒ `analysis/search-trace-and-policy` 的远端 SHA **未知**，
  最后一次**实际在线**核验到的远端状态仍是第十七轮回末 `80af16ba4b90…`
  （`refs/remotes/origin/analysis/real-search-undetected-attribution`）。
  网络恢复后只需 `git push -u origin analysis/search-trace-and-policy` 再 `git ls-remote` 核对；
  本轮全部产物与账本已在本地历史里，推送不改变任何实验结论
  （细节：`SEARCH_POLICY_COMPARISON_RESULTS.md` §十三）。

上一轮入口：[真实 PILOC 未发现目标的逐个归因](runs/real_search_undetected_attribution_20260923/SEARCH_UNDETECTED_ATTRIBUTION.md)。


## 2026-09-24：真实 PILOC 未发现目标逐个归因完成——不是死角、不是疑难格，缺的是两条位置路径（第十七轮，0 episode、0 训练）

产物 [runs/real_search_undetected_attribution_20260923/SEARCH_UNDETECTED_ATTRIBUTION.md](runs/real_search_undetected_attribution_20260923/SEARCH_UNDETECTED_ATTRIBUTION.md)、
逐目标表 [UNDETECTED_GOAL_ATTRIBUTION.json](runs/real_search_undetected_attribution_20260923/UNDETECTED_GOAL_ATTRIBUTION.json)、
认证几何 `SCENARIO_GEOMETRY.json`、字段负证据 `FIELD_INVENTORY.json`、
计数闭合 `BATCH_RECONCILIATION.json`、核验 `ATTRIBUTION_VERIFICATION.json`（`passed=true`），
以及两份**已冻结、本轮不执行**的下一步件：
`MISSING_FIELDS_AND_PASSIVE_RECORD_PLAN.md`、`SEARCH_SIDE_PAIRED_VALIDATION_PROTOCOL.md`。
分支 `analysis/real-search-undetected-attribution`（起点 `exp/comm-info-path-validation@037fde424b3c…`，
即上一轮在线核对过的头；本轮开跑时 `github.com:443` 不可达，改以本地远程跟踪 ref 核到同一 SHA），
总纲 v4.0。**边界**：0 episode、0 训练、0 生产代码改动、0 通信配置改动、未加目标难度、
未触碰 `800000..800399`、未读取 `900000..900199`、D1 的协议/判据/结果一字未改；
测试 `test_rs_harness` 19 项（0 episode）+ 项目套件 702 项 OK(skipped=2)。

### 一、两批真实搜索各自闭合（绝不合并成一个配对实验）

| 批次 | seed 块 | 放置 | 发现 | 未发现 | 按时 | 有未发现的集数 |
|---|---|---:|---:|---:|---:|---:|
| B1 = D1 四臂的搜索侧 | `1100000..1100059` | 360 | 352 | **8** | 352（A1/A3） | 7/60 |
| B2 = systemloop（09-18） | `31000..31059` | 360 | 356 | **4** | 350 / 356（两臂） | 3/60 |

四臂的发现集合逐 seed 相同（场景、起点、栅格哈希、检测序列漂移 0）⇒ 搜索侧读一个臂即可；
`detected + remaining_search_targets == placed` 在 **120/120 集**成立、每目标至多一条事件、
发现步全在 `[1,1000]` 内；按时完成用项目自己的 `window_metrics`。

### 二、栅格是"重建后认证"的，不是假设的

日志只存栅格哈希不存栅格，所以用同一个纯函数 `scenarios.build_scenario(同 seed 同参数)` 重建：
**120/120 个 seed 的 `map_sha256` 与运行当时记录的值逐字符相同**，放置格、起点格、障碍格数也逐项一致；
未通过认证的 seed 不参与任何几何计算。两套视线判据（本仓库的 `has_line_of_sight` 与
按 PILOC 规则独立复写的一份）在 **72 457 个格对上 0 分歧**。

### 三、四桶判定（每行都带证据等级，12 行全表在结果件里）

| 桶 | B1 | B2 | 依据 |
|---|---:|---:|---|
| 不可达或场景问题 | **0** | **0** | 无一落障碍格；无一与两起点不同 8-连通域（引理见下） |
| 曾进入合法范围却未发事件 | 0 判定（**2 候选**） | 0 判定（0 候选） | 需要逐时刻位置 ⇒ 现在不可判 |
| 可达但轨迹未覆盖 | **6** | **4** | 可达实测 + 无实测位置在其可见集内；**以探测器该格正常为前提**（grade 标 conditional） |
| 现有日志不足以判定 | **2** | **0** | 机器人起点格已在目标可见集内，但判据在移动之后才评估 |

引理（有单测在真实地形上钉住）：合法观测要求一条**内部全是空格**的离散直线，而直线每步至多
对角相邻 ⇒ 观测者与目标必在同一 **8-连通**自由域；机器人与目标都只做 4-邻移动 ⇒ 各留自己那一域。
所以"不同 8-域"就是永不可见——这一桶是**可判**的，且答案是 0。

对照（放置格口径，中位/均值；n=12 太小，**不做**分布检验）：距最近起点测地距 **10.5/10.75** vs
已发现者 9.0/9.89；合法可见格数 **65.5/60.4** vs 65.0/63.6；
**起点到可见集的测地距 2.0/2.75 vs 3.0/3.3** ⇒ 未发现者不但不是死角，反而"更近"。
这直接排除"目标被生成在扫不到的地方"以及"改放置规则就能消掉未发现"这条路线。

### 四、本轮真正改变读法的一条环境事实：目标每步自己走一格

`piloc/src/Map/timemap.py:814-845`：未被发现的目标每步在 `{停, ±1 行, ±1 列}` 里选一个合法空格移动。
日志侧独立复现（两批 708 条发现事件）：只有 **254/708 = 35.9 %** 的发现发生在放置格上；
发现步 ≤10 的偏移中位 0（最大 4），步 >10 的偏移中位 **6**、最大 **45**；
**"偏移 > 已过去步数"的违例 0 条** ⇒ 与"每步至多一格"的随机游走完全一致。

后果与不改的判：

* `run_config` 的 `scenario.goal_positions` 与 B0 探针的 `placed` 都是 **t=0 放置格**（B0 命名准确），
  **不是**目标当时所在格；
* "某格子有没有被扫过"不再等价于"某目标有没有被发现"，静态覆盖模型在本环境里没有对象；
* 第十五/十六轮已发表的计数（`4/360 = 1.1 %`、`9/180 = 5 %`、以及通信侧的 C1/C2/C3 与
  `undecidable`）**一个都不改**，本轮复现了它们；要改的只是引用时的限定语。

### 五、缺的是两条位置路径（负证据是搜出来的）

对两批全部 **360 个**生命周期文件逐记录扫描（跳过 `run_config` 那次性快照）：
逐时刻搜索机器人位置、逐时刻目标位置、截止位置、逐时刻覆盖事实——**各 0/360 文件有**。
B0 补上了"放置"，没有补上"路径"。另外两处易误读：B2 摘要里的 `search_coverage`
定义是 `len(discovered)/NUM_GOALS`（发现比例，不是空间覆盖率）；
PILOC 内存里其实**有** `agent_positions_history`（只留最近 10 步、从不落盘）⇒ 补记录的成本很低。

### 六、两个正确性候选：是靶，不是结论

B1 的 seed `1100051`（放置 `[18,44]`）与 `1100053`（放置 `[22,58]`）：某搜索机器人的**起点格本身**
就在该目标的合法可见集合内（起点→可见集测地距 0、切比雪夫 ≤5、视线通），
而该目标整场 1000 步没有任何发现事件。**不能据此判缺陷**：`Environment.step` 在 682 行先移动机器人、
778 行才评估观测判据，"起点看得见"≠"第 1 步看得见"，之后 999 步无记录。
最小复现（单集、有 B1 之后一条命令）与修复顺序（先写失败测试：
适配器事件集合 == 环境该步删除集合；再把判据收敛到单一持有者）写在
`MISSING_FIELDS_AND_PASSIVE_RECORD_PLAN.md` §三。

### 七、下一步由使用者定（本轮跑完即停）

(a) 批 **B1 被动记录**（生产改动，需单独授权；验收比 B0 更严：必须逐步比较两个随机流状态与
每步后的位置数组，因为目标移动由环境 `_py_rng` 驱动）；(b) 批 **S1 只改搜索侧因素的配对验证**
（前置 B1，主结果 = 同 seed 未发现目标数差，上限 128 次启动，停 1/停 2 与守卫已先写死）；
(c) 按现状交付"未发现率 1.1 %–2.2 %，非场景非几何，机制待有记录"这一口径。
**不重训 PILOC、不改通信、不加难度**这三条在两份文档里都写成了禁止项。

## 2026-09-23（状态已由上一节接替）：完整系统信息路径验证已执行——B0 日志验收通过、D1 四臂 × 60 配对 seed 跑完，冻结主判据判 `undecidable`（第十六轮，0 训练）

产物 [runs/comm_info_path_validation_20260923/D1_PROTOCOL_REVISION.md](runs/comm_info_path_validation_20260923/D1_PROTOCOL_REVISION.md)（判据，
在任何新 episode 之前冻结并单独提交）、
[D1_RESULTS.md](runs/comm_info_path_validation_20260923/D1_RESULTS.md) / `.json`、
门禁件 [B0_PAIRED_REPLAY.json](runs/comm_info_path_validation_20260923/B0_PAIRED_REPLAY.json)、
[IDENTITY_AUDIT_data.json](runs/comm_info_path_validation_20260923/IDENTITY_AUDIT_data.json)、
[SEED_COLLISION_AUDIT.json](runs/comm_info_path_validation_20260923/SEED_COLLISION_AUDIT.json)、
边界核验 [D1_VERIFICATION.json](runs/comm_info_path_validation_20260923/D1_VERIFICATION.json)。
分支 `exp/comm-info-path-validation`（起点 `analysis/comm-research-closure@b73876013e76…`，
建分支前用 `git ls-remote` 取回远端完整 SHA 与本地逐字符核对），总纲 v3.9。
第十五轮留下的两件事（B0 补日志、D1 最小配对验证）**本轮都执行完了**，顺序按授权门槛串行：
**判据先冻结 → 日志先验收 → 身份门先过 → 才读数据**；上一轮的
`MINIMAL_PAIRED_VALIDATION_PROTOCOL.md` 原件一字未动，修订件另立。

### 一、B0：四类被动日志进入生产，验收方式是"关掉它之后逐字段相同"

`integrated_sar/info_probe.py` + `comm.py`/`adapters/piloc.py`/`joint_driver.py` 的挂钩点，记四类事实：
**放置**（PILOC 在 `reset()` 里就会删掉 `goals_pos`，故在删除前快照）、**逐接收者送达结局**
（每个 (发射, 订阅者) 抽样一行：到了没、第几修订号、计费字节、包序）、
**逐 (目标, 机器人) 知晓来源与时刻 + belief age**、**真实到场时刻**（沿用项目自己的 `on_site = task["members"]` 定义）。
纪律是硬的：**写侧文件、不加 lifecycle 事件类型**（归档的"这些字段不存在"断言因此仍然成立）、
用稳定外部 ID、不进 actor 观测、不消耗随机数（比特发生器状态逐位相同）、默认关闭（`--info-probe` 不给就不写）。

**成对回放 16 格（2 seed × 2 臂 × 日志前/后 × 重复）⇒ `passed=true`**：开/关探针后
全部生命周期记录逐字段相同（比较时剥离身份字段与传输层复述自己的那一块）、检测序列与发射序列 SHA 相同、
`airtime_bytes` 相同、`(on_time, expired, discovered)` 相同、重复格自洽。
另有 **I6 时机模式惰性 4 格**：未配置 `max_wait` 时 `legacy_shared_bound` 与 `policy_owned_timing`
两条臂的记录全同 ⇒ A1 与 A2 可以只差在 `comm_policy` 一个字段上。
按协议的停止规则，任一行为差异未解决就不许跑数据臂（未触发）。

### 二、身份门禁决定了哪一句"同一批丢包"可以说

前向信道是**单一顺序流**、每个 (emission, subscriber) 只抽一次数 ⇒ **发射序列相同就推出逐包结局相同**。
60 个 seed 上 I1（检测流）/I2（发射序列）/I3（逐包结局）/I4（阈值嵌套）/I5（配置单字段差）全过
（`gates_passed {C1,C2,C3} = true`），四源 seed 冲突审计 clean（`1100000..1100059` 命中全 0）。
逐对比允许的措辞已机读登记在 `phrasing_licensed`：

* **C3（A3 vs A4）可以说"同一批丢包"**——发射序列与逐包送达结局逐条相同、计费字节相同。
* **C2（A1 vs A3）只能说"同一批包、阈值不同（嵌套）"**——投递概率本身是干预，逐包结局不可能相等。
* **C1（A1 vs A2）不得使用"同一批丢包"**——一臂从不上空气，没有共享包序列可言。

### 三、三个配对比较（本轮唯一的因果陈述）

同 seed 按时完成目标数差，配对百分位 bootstrap（B=10 000、RNG 20260923）。四臂 × 60 seed
**240/240 格全完整、0 技术失败、0 探针异常**，启动账本 264/296（技术失败计入上限）。

| 代号 | 干预 | 差（个/ep） | 95 % CI | 胜/平/负 | 字节差/ep |
|---|---|---:|---|---|---:|
| C1 | 上行说话 vs 完全静默 | **+0.31667** | [+0.18333, +0.45] | 17/43/0 | +14 573.3 |
| C2 | 投递 1.0 → 0.7 | **0.00000** | [0.0, 0.0] | 0/60/0 | 0.0 |
| C3 | 自带感知（部署）vs 关掉（隔离），同一批丢包 | **+1.28333** | [+1.01667, +1.55] | 45/15/0 | 0.0 |

* **C3 是本轮最硬的一条**：同样每集 10.25 个丢失的包，在部署路径上造成 **0** 个任务损失，
  在隔离路径上造成 **1.283 33** 个/ep ⇒ **部署路径对丢包的免疫来自环境自带感知的旁路，不来自通信策略**。
  这把第十五轮"同一批丢包价格差约 30 倍"的跨路径比，第一次变成了配对身份成立之下的两个可引用数字。
* **C2 的精确零**必须限定读：它等于"视距刷新与 gossip 开着时，丢 30 % 的包不改变任务结局"，
  不等于"丢包无害"；且两臂计费字节相同（按尝试计费，丢掉的钱照付）。
* **C1 不为零** ⇒ 完全静默付 0.31667 个/ep、省 14 573.3 B/ep ⇒ 约 **46 021 B/个**被上行救回的按时任务；
  与机制记录自洽：A2 比 A1 多出的"整集无人知晓"目标（27 − 8 = 19 个/360）
  与按时完成差（352 − 333 = 19 个/360）是同一个 19/60，两条独立口径撞到了一起。

### 四、逐目标机制记录：三条零比任何份额都有信息量

分母显式（`D_place=360 / D_det=352 / D_fail`），谓词三态、允许多原因并存与"未能判定"。
A3（冻结主口径）：`D_fail = 0`。A4：77 个未按时目标，其中
"上行从未送达" **0**、"整集无人知晓" **0**、"知晓已在 deadline 之后" **0**、
只带 M3（决策时刻 belief 与真值不符）**68**、一条旗标都不带（纯执行侧）**9**、
未能判定 **0** ⇒ `S_all = S_det = 0.883`、`S_unres = 0.0`。

**M3 的校准必须先读**：两个臂上**按时完成的目标有 100 % 带 M3**（belief 只在写入步更新、
真值每步推进，"轮询式落后至少一步"是常态）⇒ **`S_all=0.883` 不得读成"88 % 的失败由信息造成"**，
校准值已与份额并排发表。有判别力的是三条零 + 程度量：隔离条件下 `immediate` 造成的不是"消息没到"，
而是**到达过一次的那条信息后来一直没人更新**（与收束件里 `send_once`"只被告知一次然后没做完"同一件事，
这次有逐 (目标, 机器人) 记录支撑）；A4 失败目标的 belief age 中位 **823 步**、
belief 内容与真值不一致占检查数的逐目标中位 **0.985**。
一处本轮自纠：机制谓词最初把"查过、都不成立"与"缺记录、判不了"合并返回，会把 9 个普通执行失败
误标为"未能判定"；已分成 `(evaluable, flags)` + `M0_no_mechanism_flag_applies`。
逐目标链与项目原生 `window_metrics` 在 **240/240 格 0 冲突**。

### 五、冻结判据机械适用的结果：`declared = undecidable`

主口径固定为 A3，而 A3 在预登记难度（6 目标 / 1000 步 / 检测半径 5）下**按时失败数为 0**
⇒ `information_bucket_share` **分母为空** ⇒ 守卫按原文生效 ⇒
**本轮既不判 N6 满足，也不判 N6 不满足**。这不是实验失败：部署路径在该难度下没有可归因的按时失败，
所以"信息侧占失败的份额"在 A3 上无定义。写成"信息侧不是瓶颈"是超出证据（用 0 个失败除出来的），
写成"实验没结论"也不对（三个干预差都是有效估计，C1/C3 的 CI 下界都 > 0）。
要让 N6 可判，唯一诚实的做法是**另写一次预登记实验换难度**；本轮禁止并明确未做
（加目标数、收紧 deadline 都在修订件 §七里排除）。

### 六、本轮的边界

* **0 训练、0 `optimizer.step`、未加载或改动任何 checkpoint、未改 reward、未选 λ、
  未改通信拓扑、未扩观测、未改任何生产默认策略**
  （`comm_policy=immediate`、`comm_delivery_probability=1.0`、`gossip_interval=2`、
  `comm_budget_bytes=None`、`comm_ack=False`，`joint_driver.py:126-171`）。
  生产改动只有 B0 的被动日志五个文件（`D1_VERIFICATION.json` 核对未越界、`passed=true`），
  两个 checkpoint 的 SHA256 与修改时间未被触碰。
* **这 240 格不是学习模型的独立测试集**：没有任何学习策略跑在其上，它是九臂规则矩阵中四个臂的机制验证；
  `1100000..1100059` 不得如此引用。`800000..800399` 封存档上 artefact 仍为 0，
  `900000..900199` 仍是规则工程块且本轮未读取。
* 机制旗标是**共现描述不是归因**；只有 C1/C2/C3 可以作因果陈述，且各自受 §二 的允许措辞约束。
* 逐集日志、转录与探针文件不入 Git（`logs/` 下 792 个文件实测约 148 MB：生命周期 142.3 MB、
  探针 5.0 MB、通信转录 1.9 MB；另 240 个行摘要约 1.2 MB），只登记在启动账本与结果件里。
* 测试：本轮 `test_info_probe` **12 项** + `test_d1_harness` **14 项（两者 0 episode）** +
  项目套件 **702 项 OK (skipped=2)**（刚重跑核实）。本轮**没有**再运行 `python -m integrated_sar`
  的独立 smoke：296 次启动上限之内不再追加 episode，闭环可用性由上述 702 项里的
  `joint_driver` 测试覆盖。

### 七、下一步由使用者定（本轮跑完即停）

(a) 若仍要回答"部署路径上信息侧损失占多大比例"：批准一次**新的预登记实验换难度**（本轮不得修补判据）；
(b) 把 C1/C2/C3 作为"**架构旁路**"结论交付——通信协议的鲁棒性收益与环境自带感知的旁路收益必须分开引用；
(c) 按第十五轮瓶颈归因转系统集成/任务分配（搜索覆盖率与执行连续性仍有可归因损失）。

## 2026-09-23（状态已由上一节接替）：通信研究收束 + 完整系统瓶颈盘点已交付（第十五轮，0 训练、0 新 episode）

产物 [runs/comm_research_closure_20260923/COMMUNICATION_RESEARCH_CLOSURE.md](runs/comm_research_closure_20260923/COMMUNICATION_RESEARCH_CLOSURE.md)、
[JOINT_SYSTEM_BOTTLENECK_AUDIT.md](runs/comm_research_closure_20260923/JOINT_SYSTEM_BOTTLENECK_AUDIT.md) /
`.json`、未执行的 [MINIMAL_PAIRED_VALIDATION_PROTOCOL.md](runs/comm_research_closure_20260923/MINIMAL_PAIRED_VALIDATION_PROTOCOL.md)。
分支 `analysis/comm-research-closure`（起点 `analysis/comm-recipient-topology-feasibility@9db4cdf073d2…`，
建分支前用 `git ls-remote` 取回远端完整 SHA 与本地 HEAD 逐字符核对），总纲 v3.8。

**0 新 episode、0 新流、0 `optimizer.step`、0 checkpoint 加载、0 启动；未修改任何 production code、
reward、通信拓扑或模型权重。** 机械核实（`CLOSURE_VERIFICATION.json`，`passed=true`）：
41 个只读输入 SHA256 不变、8 个日志族文件数与总字节数不变、生产目录与起点 `git diff` 为空、
本轮改动只落在四份治理文档上（`git diff --name-only` 机械核对）、
账本 `starts` 恒 666、封存块 `800000..800399` 盘上 artefact 仍为 0、五个环境入口 tripwire 日志为空。
测试：本轮 `test_closure_harness` **21 项 0 episode OK**、项目套件 **690 项 OK (skipped=2)**、
`python -m integrated_sar` **final_status=completed**、上一轮 `test_topology_harness` 13 项仍绿。

### 一、收束件：四条结论必须分开引用

| # | 结论 | 状态 | 决定性数字与出处 |
|---|---|---|---|
| 1 | 规则与协议层收益 | **成立** | `send_once` 14 916 B / 5.7917 个/ep 支配 `immediate` 26 897 B / 5.2917（后者 49.9 % 字节是上界代发）；`hold_to_bound` 单独 −37 967 B；`ack_retry_2` 隔离有损 +0.835 [+0.685,+0.995] 但 total +26 354.1 B、正常路径 0.000 ⇒ 不提部署 |
| 2 | 学习型发送控制器 | **附加价值未建立**（四批独立训练、门槛全 0/3） | 第三/四/五/十轮；每批**全部**运行显著优于自身初值（+0.542～+2.333、+1.04～+4.21、+2.125/+0.833/+2.708）；两条臂与规则**逐字相同**且自主字节 0 ⇒ 不是 ε-greedy 假象；第十轮 actionable 20.1–96.0 个/ep 否证"样本太少" |
| 3 | 时机轴 `no_temporal_allocation_signal` | **机制层面关闭** | 4 852 个多候选步里 63.1 % 全拒付、仅 4.1 % 严格子集；窗口 144 B = 单包 5.8 %、15/15 一步跨过；采样严格子集步上首报候选 min/median/max 全为 0；576 格无一用到 100 % |
| 4 | 拓扑轴 `global_fanout_is_only_a_scalar` | **契约层面关闭** | 8 904 条发射 subscriber distinct = {6}；缩放与预算档等价（1 875 次比较 0 不一致）；coalition 规则下 1 107/1 107 首报收件人为空；`SendObservation.subscribers` 只有一个全局标量 |

* 四条都**只**限定在当前任务分布 / 当前广播协议 / 当前硬预算 / 当前观测·动作·奖励之下，
  **都不构成**"强化学习普遍无效"、"不存在更优策略"或"规则全局最优"。
* 强规则基线按四层重算并与归档**逐位一致**（g6 `periodic_20` 5.375 @25 416.0 B、
  g8/g10/g12 `send_once` 7.1875 / 8.750 / 8.875 个/ep；`best_on_time_rate_of_detected`
  0.927 08 / 0.937 5 / 0.9 / 0.781 72 = **逐流比率的均值**，不是总和之比）。
* **生产默认值一条都不改**（`comm_policy=immediate`、`comm_budget_bytes=None`、
  `legacy_shared_bound`、`comm_ack=False`、`subscribers=all`、`comm_range=0.3`、`gossip_interval=2`，
  `joint_driver.py:126-170`）：基线全部来自隔离实验，而隔离与部署的差是**环境自带传感旁路**，
  不是策略变好；把 `send_once` 设默认会在部署路径上引入"不再刷新"这一行为改变，
  而该改变在部署配置下的后果**从未被测量过**。
* 重开条件写成 N1–N6 六条（每条带"当前状态"与"最低验收"），并明确列出**已被实测排除**的
  七种"重开理由"（加步数、再减预算、扩动作空间、ε-greedy 掩盖、样本稀缺、欠训练、把 3/10 还给策略）。

### 二、瓶颈盘点：八阶段第一次被同一个 target_id 串起来

四条归档路径**分开报告、绝不合并**：P1 部署+真实 PILOC 搜索（120 集）、P2 部署+回放检测（90 集）、
P3 隔离+硬额度（576 格）、P4 有损确认块（3 设置 × 5 臂 × 200 流，读归档聚合 + 60 格转录抽查）。
**40 项与已发布结果的核对全部一致**（`reconciliation_summary`）；核对过程中抓到并修掉的是
**本轮自己的两处错**（把 16 条流的比率之和当成比率、把归档的 +26 354.1 抄成 26 354.0），
而不是去改期望值。

1. **八阶段里只有 3 个是日志直接可归因的**（发现、分配、结局）；获知/到场/开工是代理量；
   **放置与"逐接收者送达"根本没有记录**（生命周期 `event_type` 全集恰为 6 种，
   脚本用"字面量不存在"反过来证明这条断言）。代理量的代价看得见：
   g12 `send_once` 的 `dispatched→working` 均值是 **−18.0 步**（n=46），
   因为"指纹不变就不写行"，`en_route` 可能晚于 `in_progress` 被观测到。
2. **部署路径**：可归因损失 = 搜索未发现（P1 4/360 = 1.1 %、P2 9/180 = 5 %）+ 分配/执行
   （P2 无协调器 35/171，其中"从未派出"18 与归档 `never_assigned=18` **逐类相等**；
   有协调器降到 19/171）。**信息侧那一桶写成 `not attributable`，不是 0。**
   P1 candidate 臂（356/356 按时）唯一的损失就是搜索的 4 个 ⇒ 独立复现"瓶颈是搜索覆盖率"。
3. **隔离路径**：信息侧才是主损失（`always_hold` 555/555 从未送达、`event` 216、`periodic_20` 95、
   `send_once` 8，与第十三轮独立登记的数字一致）；但 `send_once` 另有 **68 个目标是
   "只被告知一次然后没做完"**（这些目标刷新数恰好为 0）⇒ 额度的伤害主要通过**信息陈旧**而非沉默起作用；
   而在同一额度里改买刷新（`periodic_20`、`event`）实测**更差**（403、311 < 478）。
   天花板配对差（精确配对）：`event_unbudgeted − send_once` = −12/−7/−12/−40，
   g12 合 **197 517 B/ep 换 2.5 个/ep = 79 007 B/个**，是第七轮 ACK 路线 break-even 的 2.5 倍；
   **混淆项已标明**（天花板臂同时改了额度与刷新策略）。
4. **同一批丢包的价格差 30 倍**（本轮新量，此前无人发表这个配对）：三条臂在 `normal-lossy` 与
   `isolated-lossy` 之间**前向信道账本逐字段相同**（尝试 35.16 / 到达 24.605 / 丢失 10.555 /
   发射与首报数全等）⇒ 差只可能来自自带传感：on-time **5.810 0 vs 4.480 0** 个/ep、
   过期 **0.045 vs 1.370**。与从 48 行 `TRAFFIC_PROBE.json` 重算的归属一致：
   部署静默时 **390/408 = 95.6 %** 首次知晓仍发生（视距 388 + gossip 2），隔离静默时 **0 次**、
   belief 写 0 次。配对分级逐条写明：**跨路径的比一律不算配对**。
5. **576 格里"检测→首次发射"与"首次发射→首次被知晓"全是 0 步**（无损、零延迟）：
   隔离设置下"何时说"的全部自由度就是"说不说"——这是 `no_temporal_allocation_signal` 的机制面重述。

### 三、因此本轮交付的判定

* 通信研究线**当前不需要新的训练实验**，需要的是一次**日志能力变更**（B0：per-recipient 送达结局、
  per-(目标,机器人) 知晓时刻与 belief age、真实到场步、放置记录）+ 一次
  **D1 最小配对验证**（296 次启动上限，四臂 × 60 seed；停 1 < 5 % ⇒ 关闭该路线、
  停 2 ≥ 20 % 且 ≥ 15/60 流 ⇒ 才允许一次无模型 calibration、区间内不选边）。
  协议已写、**本轮不执行**，且 B0 属生产改动、需单独授权。
* **不做的六件事**：未训练、未改 reward/预算/ACK/丢包/拓扑/生产默认值/观测维度、
  未运行任何 episode 或新流、未触碰 `800000..800399`（盘上 artefact 仍 0）、
  未修改任何一轮的原判据与结果文件（写盘沙箱 + 一条测试证明越界即拒）、
  未把"当前量不出来"写成"信息侧没有损失"。
* **下一步由使用者定（本轮跑完即停）**：(a) 批准 B0 + D1；(b) 按瓶颈盘点转系统集成/任务分配
  （本轮给出该方向可直接使用的三条损失归因）；(c) 论文负结果整理
  （两条命名结论 + "同一丢包在两种架构下价格差 30 倍"的配对，都是可引用件）。

## 2026-09-23（状态已由上一节接替）：接收者与真实拓扑可行性审计完成——fan-out 是标量，该轴关闭（0 训练、0 新 episode）

产物 [runs/comm_recipient_topology_feasibility_20260923/COMM_RECIPIENT_TOPOLOGY_RESULTS.md](runs/comm_recipient_topology_feasibility_20260923/COMM_RECIPIENT_TOPOLOGY_RESULTS.md)、
判定 [COMM_RECIPIENT_TOPOLOGY_DECISION.md](runs/comm_recipient_topology_feasibility_20260923/COMM_RECIPIENT_TOPOLOGY_DECISION.md)、
调用链 [COMM_RECIPIENT_TOPOLOGY_SOURCE_MAP.md](runs/comm_recipient_topology_feasibility_20260923/COMM_RECIPIENT_TOPOLOGY_SOURCE_MAP.md)。
分支 `analysis/comm-recipient-topology-feasibility`（起点 `analysis/comm-action-geometry-audit@efa942857f5b…`，
即上一轮 `no_temporal_allocation_signal` 的**已推送**头，`git ls-remote` 逐字符核对过），总纲 v3.7。

**0 新 episode、0 新流、0 训练、0 `optimizer.step`、0 checkpoint、0 启动；未修改任何 production code。**
机械核实（`COMM_RECIPIENT_TOPOLOGY_VERIFICATION.json`，`passed=true`）：618 个输入文件 SHA256 不变、
账本 `starts` 恒为 666、五个环境入口 tripwire 日志为空、12 个被引用生产文件与起点无 diff、
封存块 `800000..800399` 盘上 artefact 仍为 0。测试：本轮 `test_topology_harness` **13 项 0 episode OK**、
项目套件 **690 项 OK (skipped=2)**、`python -m integrated_sar` **final_status=completed**。

| 门 | 判定 | 决定性证据 |
|---|---|---|
| R0 语义可信 | **过** | 源码 15 个命名锚点全部在当前工作树定位成功（有一条测试专门盯锚点漂移）；定价恒等式 `cost == payload × subscribers` **8 904/8 904** 闭合；未读未来结局 |
| R1 拓扑已异质 | **不过** | 四层全部发射的 subscriber count **distinct = {6}**；512 个有发射的 episode 中变化数 **0**；按首报/刷新/retry 分组同样是 {6} ⇒ 结论名 **`global_fanout_is_only_a_scalar`** |
| R2 存在合法 recipient set | **不过** | 五种候选无一同时满足四条要求 |
| R3 发送前可观测 | **不过** | `SendObservation.subscribers` 是**一个全局标量**（`comm_policy.py:112`）；`SendItem` 只有 `members_changed: bool`，没有成员集合 ⇒ 契约里缺这个概念，不是数据没记全 |
| R4 报价形成真实选择 | **不过**（前提已失效，仍作诊断出数） | coalition 下 174 个"同一步两个新目标首报"**全部定价为 0**、出现价格差异的流 **0/16**；广播下 16/16 但中位差仅 **6 B**（一个包的 0.23 %），差异来自序列化字段长度而非接收者语义 |

- **两条机械证明把 R1 从"没测到变化"加强成"变了也没有自由度"**：
  ① 把 fan-out 乘 s ∈ {1/6, 1/3, 1, 2, 3, 6}，"**同一步内价格窗口 / 单包**"比值在六个尺度上恒等于
  **0.190648**（比值里不含 k）；② `classify(charges×s, credit) ≡ classify(charges, credit/s)`
  在 **1 875 次比较中不一致 0 次** ⇒ **加订阅者就是调预算档**，而档位本项目已扫过（75 % / 25 %，
  当前四层用的正是 25 % 档）。费用分解同样闭合：报价极差 = payload 极差 × 6 逐层精确成立
  （g12：128 B × 6 = 768 B），`log` 费用方差中 fan-out 占比 = **0.0**。
- **唯一能从真实语义构造的接收者规则会取消"发现"本身**：`members`（coalition）在 g12 的
  **1107/1107 条首报上全为空**。叠加调用链硬依赖
  （未收到 ⇒ 不进 `known_tasks` ⇒ 不进 `action_map` ⇒ 该机整场不可派往该目标，
  `old_task_env.py:320 / 855 / 902 / 931`、`assembly.py:166-168`、`coordinator.py:124/292/679`），
  且 isolated 配置下视距刷新与一跳 gossip 被 `gossip_interval`（`old_task_env.py:1109`）
  **同一个门一起关闭**、`task03.py:377-379` 明写运行期发现"只经通信与其自身传感进入 belief"
  ⇒ 这条规则把新目标**永久孤儿化**。它买到的价格异质（同步内相对极差 0.071 → 0.718）
  **全部落在续报那一侧**，携带新信息的那一类被定价为"永远不发"。
- **一条对该方向以外也有用的结构性事实**：**通信接收者集合 = 动作可达性集合**。
  今后任何改动多机信息共享的方案，都必须先回答"未收到消息的目标如何进入合法动作掩码"。
  这也是"不得把 recipient selection 与 send timing 合并成一个动作空间"这条禁令的实证依据：
  `send_once` 那 43–73 个未报出目标是**成本约束下的损失**，而限制性接收者规则会把它变成
  **定义上的不可达**——两类不是同一种东西。
- **不做的三件事**：未训练、未改 reward/预算/ACK/丢包/生产代码/观测维度、未随机生成拓扑或用
  payload padding 扩报价；四层原判据与结果文件一个字节都没写（写盘沙箱 + 一条测试证明它会拒绝越界路径）。
- **两级可引用的负结果，且互相支撑**：① `no_temporal_allocation_signal`（时机轴：额度冻在残值上、
  窗口只有一包的 5.8 %、15/15 一步跨带、命中的候选 100 % 是续报）；
  ② `global_fanout_is_only_a_scalar`（拓扑轴：fan-out 无自由度、唯一合法规则取消发现、
  契约里没有 per-candidate 接收者概念）。两者都**只**限定在当前任务分布 / 当前广播协议 /
  当前硬预算 / 当前观测·动作·奖励之下，**都不构成**"强化学习在其他拓扑或任务分布中无效"。
- **本轮不是为学习模型制造有利场景的尝试**：判据在设计时允许失败，R1 的失败名字是**先写进协议**的；
  协议 §5 明令禁止放宽 recipient set 的语义要求去凑 R4 的绿灯。

**下一步由使用者定（本轮跑完即停，不自动展开）**：按协议 §6 分支 1，多订阅者/拓扑轴关闭，
剩下三条是系统集成或任务分配、论文负结果整理（现有两级结论可直接引用）、
或**先做一次显式的架构与契约变更**（位置感知的通信模型 + per-candidate 接收者视图 +
"无人知晓的新目标如何被发现"的替代机制）再谈拓扑——在那样的系统存在之前，本轮不为它设计判据。

## 2026-09-23（状态已由上一节接替）：通信动作几何与时序价值审计完成，L3 那 198 步被证明是残值窗口而非选择空间（0 训练、0 新 episode）

产物 [runs/comm_action_geometry_audit_20260923/COMM_ACTION_GEOMETRY_AUDIT.md](runs/comm_action_geometry_audit_20260923/COMM_ACTION_GEOMETRY_AUDIT.md)，
分支 `analysis/comm-action-geometry-audit`（起点 `analysis/comm-task-pressure-g12-extension@8e35a24e60a1…`，建分支前先 `git rev-parse HEAD` 逐字符核对），总纲 v3.6。
执行入口是使用者本轮交接指令；**本轮是回顾性机制审计，没有预登记判据、没有门禁、没有通过/不通过**，
不包装成性能实验。

**0 新 episode、0 新检测流、0 `optimizer.step`、0 checkpoint 加载、0 启动**。
"没跑环境"不是叙述：`run_episode / run_arm / record / matrix.run / agreement_with_archived_rows`
五个入口被 tripwire 包成"调用即抛"，调用日志为空；1 917 个输入文件（1.75 GiB）分析前后 SHA256 逐个相同；
账本 `starts` 分析前后同为 666。封存块 `800000..800399` 盘上 artefact 仍为 0。

| 核验 | 结果 |
|---|---|
| 独立重跑 `judge_layers.judge()` 比对四层已发布 `GATES_g*.json` | L0–L4 布尔**逐层完全一致**；g12 **0 差异**，g6/g8/g10 各 15 项差异但**无一取值变化**（10 项"判定那一刻的活账本读数" + 5 项 g12 轮新增的只读诊断键） |
| 本轮自行推导 `best_fixed` vs 已发布 | 四层逐层相符（g6 `periodic_20`，g8/g10/g12 `send_once`） |
| 行集发射序列 vs 通信转录 | 576/576 格逐条相等；671 604 条拒付记录行/转录一致；`first` 标志与历史推断 0 处不符 |
| 测试 | 本轮 `test_audit_harness` 21 项 0 episode OK；项目 690 项 OK (skipped=2)；`python -m integrated_sar` completed；被审计轮 `test_pressure_harness` 71 项 OK（未改） |

- **交接文档里那条"g12 缺 16 次启动未闭合"是过期陈述，实测已闭合**：追加授权使上限 640→656，g12 实到
  160 次（16 录制 + 144 矩阵）、0 失败、144/144 格；总数据启动 656 = 上限 656。之所以看着对不上：
  账本 `entries` 670 行 vs `starts` 666 次，差的 4 行是 g6 的"重登记而未跑 episode"审计痕迹
  （GridPosition 崩在登记步、episode 已跑完、就地重登记不重复扣启动）。**数行数会得到错误结论。**
- **198 步严格子集的机制已量清（这是本轮的主要交付）**：g12 最佳规则把 11 条首发在前 ~20 步发完
  （50 % 额度中位在第 **5** 步）后就再也不发，**额度冻结在一个残值上**；残值落在自家包价带之上/之内/
  之下分别是 1/6/9 条流，而"带内"只是必要条件——最终 3/16 条流有多候选步落在带内。
  决定性一测：15 条走到"看得见却一个都付不起"的流，**把额度往回倒一个包，15/15 都在带之上**——
  **一步跨过整条带，15 次里 15 次**。原因是带宽 vs 步长：**144 B vs 2 478 B = 5.8 %**。
  所以 147 步集中在一条流不是它面临 147 次抉择，是**同一个算术巧合被重复计了 147 步**。
- **那些步上选的是"要不要再报一遍"，不是"要不要第一次说"**：g12 最佳规则 52 个采样严格子集步上
  `first_report_candidates` 的 min/median/max **全是 0**，可见候选数 == 续报候选数。
  ⇒ 即便 L3 被这些步跨过，它们也不携带"新信息之间取舍"的价值内容。
- **可支付与顺序无关，被谁送出才与顺序有关**：四层九臂 375 个采样步 **375/375** 满足
  `payable == #{charge ≤ remaining}`。所以"transport order 改变可支付集合"这一问的答案是**不改**；
  改的是同一个严格子集内批次实际送出谁。
- **一处已发布说法需要收紧（本轮按臂逐条重核的结果，不是引用汇总句）**：
  "首次断供那一步候选全是刷新"对 `send_once`(15/15) 与 `first_send_plus_bound_3_10`(16/16) 成立，
  但 `bound_only_3_10` 13/16、`periodic_20` 12/16、`periodic_5` 10/16、`event_priority` 10/16、
  `event` 9/16 —— 其余流在那一步**仍有一个从未报出的新目标**。反向也成立：
  **没有任何臂、任何一条流**的首次断供是"全是新目标"。今后引用必须带臂名。
- **一笔被算清的账**：**576 格里没有任何一格把预算用到 100 %**（全矩阵最高 0.999 93，差 2 B；
  四层最佳规则分别只到 0.999 11 / 0.726 55 / 0.883 62 / 0.964 91）。"额度用满"只能按 ≥90 % 口径说；
  本轮确认 L3 的 ≥0.90 门槛没有被当成 =1.0 使用。另外 g8/g10 的最佳规则连 75 % / 90 % 都没到，
  也没有任何拒付或全不可支付步——L3 在那两层不是"没过"，是**门槛对应的事件根本没发生**。
- **D 节（ledger-only 单包节省）诚实结果**：**`send_once` 在四层一律 0 条续报**，所以对被判据读取的那条臂
  这个检验**没有对象**（16/16 条流属"节省后无消息可解锁"）。在会发续报的臂上机会确实普遍
  （`event` 四层全 16/16 条流，单包中位解锁 2–5 个**新目标首次报告**、延迟 3–5 步、
  deadline 余量中位 100–140）。但这条账本机会的全部内容就是"别把额度花在续报上"——
  **而那正是 `send_once` 已经在做、且它在四层都是赢家**。加回来的那一包还被假设永久可用（真实会再被续报
  扣掉），所以是上界；且 HOLD 续报会让 belief 变旧，本轮不建模接收端 ⇒
  **全部数字带 `ledger_only_opportunity` 标签，不声称任务效果改善。**
- **断供的真实信息损失按臂闭合**（g12，183 个被检测目标）：`send_once` 从未报出 **8** 个、
  `periodic_20` 43、`bound_only_3_10` 48、`periodic_5` 66、`event` **73**；
  "检测数 − 首发数"与它逐臂精确相等。拒付尝试数（`event` 70 657 次）除以从未报出目标数 ≈ **968 次尝试/目标**
  ⇒ **拒付次数绝不是损失的消息数**。
- **missing 已登记，未用重放补**：前三层没有 ≥2 候选步的精确直方图（只能给界）、
  没有 `all_payable`/`none_payable(≥2)` 逐步分布、没有断供那一步的候选构成
  （只能报"首次被拒付的那条消息"的类别，**分母不同、不互换**）；
  g12 那 198 步里只有 52 步（26 %）存到逐候选（`COMPETITION_SAMPLE_LIMIT = 20`），
  故 1000301(147) 与 1000311(39) 的"首次严格子集步"在下两层是**下界**，已在每格标注。
- **不做的三件事**：未训练、未改 reward/预算/ACK/丢包/拓扑/规则臂/生产默认值、未跑 g14/g16 或恢复 H=20；
  四层结果文件一个字节都没写（`write_json` 沙箱只允许写本轮目录，并有测试证明它会拒绝越界路径）。

- **统一判定名 `no_temporal_allocation_signal`**（写进本轮 `..._VERDICT.json` 的 `decision.name`，
  它**不是第五道门禁**，只是本轮新增机制证据的引用名；四层 L0–L4 原判据与取值未改写）。
  引用时必须带六条范围：当前任务分布（`num_goals` 6/8/10/12、64 条流）、当前广播协议
  （`policy_owned_timing` + 全订阅者广播按 fan-out 乘计费、isolated-lossless、无 ACK）、
  当前硬预算（29 300 B/episode，且实测无一格用到 100 %）、当前观测/动作/奖励
  （19 维、SEND/HOLD、reward 只奖励按时完成而字节仅为约束）、
  **`send_once` 已实现"把预算保留给未报告目标"这一有效行为**（四层一律零续报，
  正是假设中学习器该学到的策略，故时机轴的净收益空间已被固定规则占据）、
  以及**没有证明强化学习在其他通信拓扑或任务分布中无效**。
- **下一步由使用者定（本轮不自动展开）**：本轮能支持的读法是——若还要测"逐步预算选择"，
必须先证明存在一种候选报价分布，使残值命中的是**多个新目标之间**的取舍而不是续报之间的取舍；
这在已存 576 格上问不出来，需要新的实验设计与新的授权。其余方向见总纲 §9。

## 2026-09-23（状态已由上一节接替）：预登记的 12 目标层已按追加授权补跑完，任务压力这条轴四层全未通过（0 训练）

产物 [runs/comm_task_pressure_learnability_20260922/COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.md](runs/comm_task_pressure_learnability_20260922/COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.md)，
分支 `analysis/comm-task-pressure-g12-extension`（起点 `analysis/comm-task-pressure-learnability@6b4452f5de88…`），总纲 v3.5。
执行入口是使用者本轮指令；协议 [COMM_TASK_PRESSURE_G12_EXTENSION_PROTOCOL.md](runs/comm_task_pressure_learnability_20260922/COMM_TASK_PRESSURE_G12_EXTENSION_PROTOCOL.md)
在第一条 g12 episode 之前单独提交。

**0 训练、0 `optimizer.step`、0 checkpoint、0 模型加载、0 新增验证 episode。** 唯一自变量 `num_goals = 12`；
29 300 B、`policy_owned_timing`、isolated-lossless、ACK 关闭、九臂规则库与 bootstrap 口径逐项沿用前三层。

| 压力层 | L0 | L1 | L2 | L3 | L4 | 该层结论（判据一字未改） |
|---:|---|---|---|---|---|---|
| 6 / 8 / 10 目标（已交付） | 过 | 过 | 仅 10 过（0.900） | 不过（0/16） | 不过 | 未通过——原判据，本轮未改写 |
| **12 目标（本轮）** | 过 | 过（0.95313） | **过**（0.78172 在带内；+8.875 CI [+7.625, +10.000]） | **不过**（3/16 流有争用点，门槛 8/16） | **不过**（envelope +0.0625，CI [0, 0.1875]） | 未通过 |

- **地位是这一条最容易被读错，先钉住**：g12 是原协议预登记的第四层，因上一轮技术失败计入 640 上限而差 16 次没跑；
  使用者只追加授权这 16 次（640→656）。它是**容量扩展 / 探索性闭环**，不是对原三层门禁的事后改写；
  原三层的 `GATES_g{6,8,10}.json`、结果、manifest、freeze、归档核验件全部字节不变
  （`common.write_json` 对已交付件"内容不同即拒绝"，`report_g12.py` 只能写自己的文件名）。
- **12 目标确实把额度逼紧了，但没有变成 L3 要的那种选择**：最佳规则 `send_once` 用满 ≥90 % 额度的流数
  从前三层的 0/16 跳到 **15/16**，并且第一次出现"同一步只付得起严格子集"（3/16 流、合计 198 步）。
  门槛是 8/16，所以 L3 仍不过。
- **机制第一次被量清（不参与任何判据）**：16 流 × 1 000 步里 4 852 步有 ≥2 个可见候选（30.3 %），
  其中 **63.1 % 是"一个都付不起"、只有 4.1 % 是严格子集**；候选报价极差 762 B = 额度的 2.6 %；
  15 条流首次断供那一步上所有候选都是**已经发过消息的刷新**。⇒ 失效形式仍是成片跨进全拒付区。
- **不做的三件事**：未算 λ（均值 Pareto 只有 1 点，且门槛未过）、未改 reward、未新增规则/seed/验证 episode；
  `800000..800399` 盘上 artefact 仍为 0，生产默认值未改。
- **上限诚实性**：数据启动 656/656（= 前三层 496 + 本层 16 录制 + 144 矩阵）、核验 10/12 未加、
  历史技术失败 20 次照旧计入；本层 144 格 0 缺格 0 失败。
- **两处披露（写在结果的"披露"节，不藏在脚注）**：
  (1) `episode_code_identity` 的按-commit 那条路径随 locale 变（同批 commit：cp936 → `c922ca100665609c`，
  UTF-8 → `23390f55a6b90afa`）；跨层"同一 episode 代码状态"改在**给行打戳**那条路径上核实
  （g8/g10/g12 的 432 格同为 `ff0b4139b1fff700`；g6 的 144 格本就没有戳）。本轮没有改这条哈希路径。
  (2) `GATES_g12.json` 的 `stop_reason` 仍写"进入下一层"——那句话写于 12 还不是末层时，判定器自第一次数据启动后
  禁止修改，故未改文案；本轮真正的下一步是扩展协议 §8 的分支，且不存在第五层。
  (3) 本轮自己的"追加授权恰好 16 次"这条不变量原先对着活账本算，跑完就不成立；改为对着开跑前的
  `G12_START_LEDGER.json` 快照核（496 已花、656 上限、需要 160、空出 160）。只动自检与一条测试，不动 episode。

**下一步由使用者定（本轮跑完即停）**：(a) 接受这条轴的四层负结果并按总纲 §9 停止场景扩展；
(b) 做只读 **action-geometry audit**（机制诊断指出的方向，0 训练）；(c) 换轴（非平稳信道 / 多订阅者与真实拓扑 /
把成本显式写进目标，最后一项要先解决 reward 结构）。

## 2026-09-22（报告完成于 09-23，状态已由上一节接替）：任务压力 calibration 已执行三层，没有任何层通过 L0–L4（0 训练，第四层未跑）

执行入口 [COMM_TASK_PRESSURE_LEARNABILITY_NEXT_ROUND.md](COMM_TASK_PRESSURE_LEARNABILITY_NEXT_ROUND.md) **已执行完毕**；
产物 [runs/comm_task_pressure_learnability_20260922/](runs/comm_task_pressure_learnability_20260922/COMM_TASK_PRESSURE_LEARNABILITY_RESULTS.md)，
分支 `analysis/comm-task-pressure-learnability`（起点 `docs/communication-learnability-calibration@ddee56db`，
其父为已核对的 `exp/comm-budget-masked-retrain@7e0a423`），总纲 v3.4。

**0 训练、0 `optimizer.step`、0 checkpoint、0 模型加载。** 唯一自变量 `num_goals`；29 300 B、`policy_owned_timing`、
isolated-lossless、ACK 关闭、checkpoint/任务/奖励/消息/规则参数、19 维观测与生产默认值全部冻结。
四源 seed 冲突审计 clean（唯一命中是执行指令自己的预登记），48 条全新流（`1000000..1000015`、
`1000100..1000115`、`1000200..1000215`），九臂规则库在第一条新流之前冻结，432 个矩阵格全部完整。

| 压力层 | L0 | L1 | L2 | L3 | L4 | 下一步（协议规定） |
|---:|---|---|---|---|---|---|
| 6 目标 | 过 | 过（0.96875） | **不过**（最佳 `periodic_20` 完成比 0.9271 > 0.90） | **不过**（0/16 流争用） | **不过**（envelope +0.3125，CI [0, 0.875]） | 进入下一层 |
| 8 目标 | 过 | 过（0.96094） | **不过**（`send_once` 0.9375 > 0.90） | **不过**（0/16） | **不过**（+0.000；均值 Pareto 只剩 1 点） | 进入下一层 |
| 10 目标 | 过 | 过（0.97500） | **过**（0.900 在带内；+8.750 CI [+7.8125, +9.500]） | **不过**（0/16） | **不过**（+0.250，CI [0, 0.6875]） | 需要第 12 目标层，但额度不足 |

- **机制**：每包 2 190–2 952 B、极差只占额度的 2.3 %–2.6 %，中位包价占 9 %，额度总共只够约 11 个包 ⇒
  额度生效的形式是**同一时刻全部付不起**（每层 82–96 个臂×流出现"看得见却没额度"），而不是"只付得起严格子集"。
  这是 L3 在三层都停着的直接原因，也是**没有跑到 12 目标前不能改判据**的原因。
- **不作为结论的三件事**：不得写"学习无效/规则全局最优/不存在更优策略"（只测了一条轴，且没跑到 12）；
  L2 在 10 目标过带不等于收益空间已打开；本轮**未计算 λ、未改 reward**（用未通过门槛的层定成本系数正是禁止的暗加惩罚）。
- **上限诚实性**：数据启动 496/640、核验启动 10/12、账本技术失败 20 次全部计入。第 12 目标层需要 160 次而只剩
  144 次可用——差额正是本轮自己两处缺陷（探针把无限额度转整数、录制登记展开 `GridPosition`）吃掉的 16+4 次。
  继续须由使用者明确追加授权；本轮不自动越线，也不删臂/不换 seed。
- **可核对的身份**：5 次 code-freeze 的 **episode 代码身份全是 `c922ca100665609c`**（37 个生产模块 blob +
  `run_episode` 源码哈希），所以修探针/修登记没换过世界；`ROW_PROVENANCE_g*.json` 与协议 §11 写明这一点。
- **反向发现**：第五轮 `ct_first_send_plus_bound_3_10_s700048.jsonl` 与它自己的行不自洽（28 632 vs 28 704 B，
  且 28 632 恰等于同目录 normal 诊断趟）⇒ 不同并行维度共用 scratch 目录被覆写；本轮按层/设置分目录，
  该臂归档转录不作为证据引用。
- 卫生：封存块 `800000..800399` 盘上 artefact **0**；生产默认值未改；本轮合成测试 50 项 + `integrated_sar` 套件
  + `python -m integrated_sar` 全绿；stream、逐 episode 日志、checkpoint 一律不入库。

**下一件事由使用者决定**：(a) 追加授权跑完 12 目标层；(b) 接受"仅提高目标数未建立逐 step 额度选择"的负结果并按 §9 停止；
(c) 换轴（非平稳信道 / 多订阅者与真实拓扑 / 把成本显式写进目标，最后一项要先解决 reward 结构）。

## 2026-09-22（历史，计划已由上一节执行）：任务压力 learnability calibration 当时的规划口径

当前入口 [COMM_TASK_PRESSURE_LEARNABILITY_NEXT_ROUND.md](COMM_TASK_PRESSURE_LEARNABILITY_NEXT_ROUND.md)，
总纲 v3.3，起点 `exp/comm-budget-masked-retrain@7e0a423`。

六目标下 `send_once` 已完成5.7917/6，修复后学习三臂仍两门槛0/3；继续追加同设置训练不再授权。
下一轮只改变 `num_goals={6,8,10,12}`，每层16条新开发流，冻结九臂规则矩阵、29,300 B和其余系统，
0训练、最多640次启动。用 L0–L4 寻找首个同时具备搜索有效、非天花板、预算严格竞争、固定规则情境差和
多点成本—效果 Pareto 的压力层。

当前 reward 只有按时完成奖励，字节只是硬约束和事后指标。本轮不改 reward，只计算规则前沿的经验交换率。
通过只允许另写成本感知目标/训练协议；不通过则停止该场景扩展。最终封存块、H=20、ACK和丢包继续冻结。

## 2026-09-22：修复动作空间后的从初值重训已执行，两条预登记门槛均未通过（3 seed × 400k）

执行入口 [COMM_BUDGET_MASKED_RETRAIN_NEXT_ROUND.md](COMM_BUDGET_MASKED_RETRAIN_NEXT_ROUND.md) **已执行完毕**；
分支 `exp/comm-budget-masked-retrain`（起点文档提交 `23160f3`），产物在
`runs/comm_budget_masked_retrain_20260922/`（provenance 更正件、协议、PREFLIGHT/THROUGHPUT/RESUME_CHECK、
CODE_FREEZE、逐臂账本、结果与 manifest）。

做对的部分：**先闭合身份再花 checkpoint**——上轮 manifest 把从不存在的 `integrated_sar/comm_transcript.py`
列为执行模块并显示"脏"，成因是 `git rev-parse HEAD:<缺失路径>` 回显参数而非报错；builder 改为由 import
派生、缺失先用 `git cat-file -e` 判定，证据头 `7f9929c` 与只追加只读测试的交付头 `e692607` 分开表述。
执行代码冻结为 54 模块的 SHA，每次启动重新派生图并检查 HEAD/工作区/未跟踪/新增模块四件事；
报告自己发现"延迟 import 的统计模块不在冻结图里"时如实报 clean=False，修的是图不是规则。

训练事实：seed 1/2/3 各从 `Controller(seed)` 随机初值起 400k 环境步（总 1.2M），4 episode/update，
0/100k/200k/300k/400k checkpoint，400k 末轮是唯一主判据，旧 checkpoint 零加载。正确性 0 违规：
请求发射 = 实际执行（4,434 / 4,448 / 4,222）、policy 端拒付 0、代发 0、NaN/Inf 0、
logged/recomputed log-prob 记录差 ≤1e−6、每集 1,000 步、deterministic save/resume 逐项完全相等、
生产代码自 `e692607` 起零改动。可学习样本按绝对量报：actionable/actor 项 20,110 / 8,020 / 38,404，
即 20.1–96.0 个/episode、零可执行 episode 为 0——"0.88 % 太少所以学不动"这一读法被实测否证。

结果：24 条开发流上 400k 对 `send_once` 的完成差 −0.7083 [−0.9583, −0.4583]、−2.0000
[−2.5000, −1.5000]、−0.1250 [−0.3750, +0.0000] 个/episode，字节差 +13,318 / +13,368 / +3.8 B/episode；
**主效果 0/3、非劣降本 0/3**。三臂都显著优于自身 0 步初值（+2.125 / +0.833 / +2.708），只作机制证据。
`send_once` 重放与归档在任务结果、字节、发射数与**发射序列**上逐流 0 差异（`config_hash` 的 24 处差异
已按机制解释：该哈希仍计入 `comm_transcript` 路径，换目录必然不同）。

本轮不追加：加训练量、改特征/观测、换预算、加 ACK、恢复 H=20 均未授权也未执行；
`800000..800399` 盘上 artefact 仍为 0，`900000..900199` 未进入本轮。启动 1,489/1,500，
其中门禁类 43 次是三遍 preflight 的累计（第一遍崩在本轮自己的 JSON 读取上、第二遍因改树而作废，
两遍都如实判不过），并行训练曾发生逐进程账本的 Windows 文件锁竞争并据此改为串行。
下一步只能是新假设（非平稳信道或任务压力 + 无模型 learnability calibration），由使用者决定。

## 2026-09-22：预算 mask 修复后从初值重训已规划（**已执行，见上一条**）

当前入口 [COMM_BUDGET_MASKED_RETRAIN_NEXT_ROUND.md](COMM_BUDGET_MASKED_RETRAIN_NEXT_ROUND.md)，
总纲 v3.1，起点 `fix/comm-budget-action-mask@e692607`。

下一轮先更正上轮 manifest 对不存在/未导入 `comm_transcript.py` 的误列并冻结真实执行代码 SHA；随后在唯一主设置
29,300 B、`policy_owned_timing`、isolated-lossless、无 ACK、19 维旧观测下，从随机初值训练 seed 1/2/3，
各 400k 步，总上限 1.2M。旧 checkpoint 不加载、不 warm start、不恢复 optimizer；主对照固定 `send_once`，
400k 末轮是唯一主判据。

0.88% 只作候选占比，必须同时报告绝对 actionable、每局/时间/消息类型分布和 actor/critic 优化诊断。
最多 12 episode 训练前探针（0 optimizer.step），全轮启动上限 1,500。`800000..800399`、H=20、第二预算、
ACK、observation 扩维继续冻结；即使开发门槛通过也不自动开放最终测试。

## 2026-09-22：预算动作 mask 修复已交付，M0–M3 全通过（0 训练、8 集 smoke、optimizer.step=0）

执行入口 [COMM_BUDGET_ACTION_MASK_FIX_NEXT_ROUND.md](COMM_BUDGET_ACTION_MASK_FIX_NEXT_ROUND.md) **已执行完毕**；
分支 `fix/comm-budget-action-mask`（起点文档提交 `31da5f7`），产物在 `runs/comm_budget_action_mask_fix_20260922/`。
总纲 v2.9 → **v3.0**。协议 `COMM_BUDGET_ACTION_MASK_PROTOCOL.md` 先于任何 episode 提交。

**修的是训练正确性**：policy-owned timing 下预算不足的 SEND 过去可被采样、写进 joint log-prob、进 PPO surrogate，
直到 `CommRelay._emit` 才被拒。现在改为**采样前、按 relay 实际执行顺序的 sequential hard mask**：
候选携带与 `_emit` 同源计算的 `attempt_charge_bytes`；collector 的 shadow credit 从**观测里的剩余额度**起算
（不是每步重新满额——这个错误被本轮测试抓到并修掉）；付不起的候选唯一合法动作是 HOLD 且
likelihood/entropy 权重 0；采到 SEND 立即扣 shadow；全 mask step 仍保留 critic 样本；relay 硬门保留。
**没有**采用"拒绝后事后删样本"：那会按已实现动作选择性改变联合概率。

| 事实 | 数值 |
|---|---|
| 生产改动 | 仅 `integrated_sar/comm.py`、`comm_policy.py` 与新增测试；其余生产路径 diff 为空 |
| 8 流 smoke | 20 655 候选 / 动作前可付 197 / **顺序后真双动作 181**（顺序自身再消 16）/ masked 20 474 |
| 动作—执行闭合 | 请求 86 = 实际发射 86，执行端 budget+capacity 拒付 **0**（同接缝修复前留 133 次拒付） |
| PPO 概率一致 | logged vs recomputed 最大差 **0.0** ≤ 1e−6；翻转 actionable 时检查确实失败 |
| 训练量 | 训练 0 次、optimizer.step **0**、新流 0、H=20 spillover 0；启动配额 8/8 用满（技术失败计入） |
| 隔离 | `budget_mask=False` 委托回旧 collector（含 tape schema）；/1=19、/2=23 维不变；默认值未动 |
| 身份 | trainer tag `onpolicy-v3-budget-mask` + mask/tape schema 为 resume 前置；旧 checkpoint 只可前向读 |

**测试**：本轮新增 30 项（生产侧契约 14 + collector/接缝/PPO 16）；`integrated_sar` 全量 **690 通过（2 跳过）**；
`python -m integrated_sar` 冒烟 `final_status=completed`。本轮实测 = 8 episode（8 000 环境步），其余为引用历史。

**含义与下一步**：可学习样本极少（actionable 占比 0.88 %）这一事实在修复前被"候选数"掩盖；旧的非 mask 训练结果
仍属已知缺陷产物，不能当修复后基线。若要谈学习效果，只能另立"从初值重开"的协议，并先重新定义真实双动作样本数、
训练上限与强规则对照 —— **本轮不启动**，等使用者决定。`800000..800399` 仍封存（盘上 artefact 0）；
`900000..900199` 只作既有证据。非本轮创建的 `architecture/`、`dev_summary.json` 与另两轮未跟踪文件未改、未删、未提交。

## 2026-09-22：（历史，计划已被下一节执行）预算动作 mask 正确性修复曾待执行

当前入口 [COMM_BUDGET_ACTION_MASK_FIX_NEXT_ROUND.md](COMM_BUDGET_ACTION_MASK_FIX_NEXT_ROUND.md)，
总纲 v2.9，起点 `analysis/comm-h20-terminal-attribution@b2061ae`。

下一轮在 relay observation 中提供精确 attempt charge/动作前可执行性，并在 learned collector 采样前按实际
顺序维护 shadow budget/capacity。不可执行 SEND 固定为 HOLD，不进 actor likelihood/entropy；critic step 保留。
禁止 relay 拒绝后事后删样本。feature 维度与默认规则行为不变。

本轮只实现、测试和最多 8 条既有流 smoke，optimizer.step=0；不训练、不跑 H=20 spillover。全部 M0–M3
通过后再单独决定是否从初值重新训练，旧 checkpoint 不作修复后续训起点。

## 2026-09-22：H=20 终态归因 + 预算动作路径审计已交付（0 新 episode）

分支 `analysis/comm-h20-terminal-attribution`（基 `analysis/comm-target-scoped-feasibility@9001a85`），
产物 `runs/comm_h20_terminal_attribution_20260922/`。**0 新 episode、0 训练、0 新流、0 参数修改、0 训练器补丁**。
只使用既有 target-scoped 24 条边界分支、actionability free 基线、训练源码与已有训练记录。

### H=20 终态归因（`COMM_H20_TERMINAL_ATTRIBUTION.md/.json`）

| seed | 分类 | EARLY/DEFER 终态时刻 | 成功 EARLY 投递/应用 |
|---:|---|---|---|
| 700010 | `both_terminal_same_time_outcome_diff` | 23 / 23 | ok / rev3 |
| 700018 | `both_terminal_same_time_outcome_diff` | 23 / 23 | ok / rev3 |
| 700020 | `boundary_refused` | — / — | ok / rev3（边界 flush 拒付） |
| 700022 | `both_terminal_same_time_outcome_diff` | 23 / 23 | ok / rev3 |

- `both_terminal_identical` **0**；`terminal_time_diff` **0**；`one_terminal_only` **0**。
- 成功 EARLY 投递伴随 assignment / task_revision / progress / remaining 分歧的 seed：
  **700010、700018、700022（3 个）**。DEFER 侧常见 `stale_task_revision` 残留指派。
- **不把 `terminal_before_boundary` 解释成有效 flush**；700020 是预算拒付，不计入终态信号。
- 与 free 对照均有差异（窗口截断 + 干预），不作因果外推。

⇒ 满足“≥2 个不同 seed 成功 EARLY + 分歧”⇒ **可再讨论**小型完整 H=20 spillover（**未运行**）。

### 预算动作路径审计（`COMM_POLICY_ACTION_MASK_AUDIT.md/.json`）

- 预算**不是** action mask；唯一 mask 是静默上界 `forced()`（age/MAX_WAIT）。
- 预算不足 SEND **能被采样且进入** tape → flatten → PPO loss（非 forced 项）。
- rollout→optimizer **无**预算过滤；relay `refused_credit` **无**样本级训练信号。
- 训练记录**不能**恢复真正双动作自由样本数 → **`training_dual_action_free_samples: missing`**。
- 离线审计可恢复：`both_actions_free=101` 点 / 8 流（主阻塞预算 23 511）。
- **不是**已修的 PPO 概率不一致；是动作空间/rollout 过滤/可观测性缺口。
  **定性为训练正确性风险**（信用分配到不可执行动作）。

### 判定与下一步（未执行、未授权）

- H=20 分歧信号：**有**（3 seed）→ 可讨论 spillover。
- 训练正确性风险：**有** → 按预注册应 **先单独修复训练路径**（mask/样本过滤 + 补
  `send_executable`/`attempt_charge_bytes` 等字段），**不继续通信效果实验**。
- 本轮**不修**训练器。H=20 spillover 须待修复完成后或用户明确覆盖该顺序。

## 2026-09-22：target-scoped 边界可行性已交付（T0/T1 过；T2 仅 H=20 过）

执行入口 [COMM_TARGET_SCOPED_FEASIBILITY_NEXT_ROUND.md](COMM_TARGET_SCOPED_FEASIBILITY_NEXT_ROUND.md) **已执行**（指令已标历史）。
分支 `analysis/comm-target-scoped-feasibility`（起点 `dfe120a`），产物
`runs/comm_target_scoped_feasibility_20260922/`。**未训练、未生成新流、未改预算/信道/默认值**；
启动 **24/28**；每条只跑到窗口边界，不跑 episode 终局。

**固定点**：700010 / 700018 / 700020 / 700022，各取最早 post-first `both_actions_free` **update**（均为 t*=3）。
分支 EARLY_FLUSH / DEFER_FLUSH，H∈{1,5,20}。边界最新视图走真实 `_message`/`_emit` 与真实计费，不预留字节。

**门禁**：
- **T0 实现可信：通过**（7 项合成测试 + 两点 H=1 冒烟）
- **T1 起点可执行：通过**（4/4 点 EARLY t* 真实发射并扣费）
- **T2 边界可解析（按 H，阈值 3/4）**：
  - H=1：**2/4 未过**（700010、700022 双分支 `flushed_latest`；700018 双分支 `refused_credit`；700020 EARLY 拒付）
  - H=5：**0/4 未过**（四对均 `refused_credit`）
  - **H=20：3/4 通过**（700010/700018/700022 双分支 `terminal_before_boundary`；700020 `refused_credit`）

**判读**：仅 **H=20** 达到资源门槛，**可建议**以后跑完整 target-scoped 结局实验（**本轮不执行**）。
H=20 可解析性主要来自 target **提前终态（吸收）**，不等于边界同步在预算上可支付；H=1/5 失败原因是
`refused_credit`，**不得**免费 flush、提高预算或改成窗口末不发送。T2 通过**不**证明任务收益、
观测可预测优劣或学习优于规则。

**下一步（未执行、未授权）**：使用者决定是否立项 H=20 完整 episode；或单独讨论 credit reservation。

## 2026-09-22：SEND/HOLD 动作可行性审计已交付（A/B 过、C 不过）

执行入口 [COMM_SEND_ACTIONABILITY_AUDIT_NEXT_ROUND.md](COMM_SEND_ACTIONABILITY_AUDIT_NEXT_ROUND.md) **已执行**（指令已标历史）。
分支 `analysis/comm-send-actionability-audit`（基 `5b10626` + cherry-pick `fc34545`），产物
`runs/comm_send_actionability_20260922/`。**训练 0、新流 0、无新 S/H 结局分支、生产默认值未改**；
启动 **8/8**（8 条 Stage1 free 重放；短探针 0）。`800000..800399`、`900000..900199` 未触碰。

**六个问题**：① 候选合计 **23 612**（逐流 1 158–4 956）；② **both_actions_free 101 点 / 8 流**；
③ SEND 阻塞几乎全是**预算** 23 511（容量/终态/重试/协议强制均为 0）；④ 自由点：首发 30、更新 37、
同视图重试 34；⑤ post-first 双动作可行 **71 点 / 8 流**（≥4，**B 过**）；⑥ 20 步内同视图重试且不被
revision 提前截断：**0 流**（**C 不过**）——15 点有 20 步内重试请求，但几乎全部 `steps_to_new_revision=1`。

**门禁**：A 通过（恒等式闭合、无 outcome 字段、8 流完整）；B 通过；C 未过 ⇒ 按指令应讨论
**target-scoped**“延迟更新、窗口末发最新视图”这一**新估计对象**，**不**暗改 Stage1 view-scoped 协议。
本轮不设计、不执行该新对象。

**方法说明**：首遍重放曾把终局预算/终态写回决策前状态；已用 Stage1 free 转录重建预算时间线并离线
重分类（**0 新启动**），见 `repair_budget_timeline.py`。

**下一项（未执行、未授权）**：由使用者决定是否立项 target-scoped 干预；在此之前不跑 S/H、不训练、
不改预算/协议。

## 2026-09-22：多步诊断第一阶段已有限交付（未训练、未改默认值）

执行入口 [COMM_MULTISTEP_STAGE1_NEXT_ROUND.md](COMM_MULTISTEP_STAGE1_NEXT_ROUND.md) **已执行**（该指令已标为历史）。
分支 `analysis/comm-multistep-stage1`（起点 `docs/communication-multistep-start-plan@3cca880`），
交付头 `e42057d`，产物 `runs/comm_multistep_stage1_20260922/`。

**边界（强制）**：未训练、未生成新流、未改生产默认值；`800000..800399` 未读；`900000..900199` 未用于选点。

**规模（含失败计入启动上限）**：

- 总仿真启动 **40 / 40**
- **8** 条 free 基线
- **28** 条成功干预分支
- **4** 条失败分支保留（`points/*/H*/failed_attempt1/`，视图锁定缺陷，修复后重跑并仍计上限）
- **7 / 8** 点四分支完整（s700033 因启动上限缺额，不补抽、不换 seed）

**结果（逐点描述，不作方法主张）**：

- 所有 **S 强发**均因 **`insufficient_credit`（预算拒绝）** 未成功 ⇒ **没有形成有效 S−H 对照**
- **H=20** 同视图窗口被 **新 revision 大量截断**，有效检验时间很短
- 已跑点上 S/H/free 的任务结果（on_time）相同 ⇒ **不能解释为发送时机无跨步影响**
- 同视图补发多为 never；替代发射来自新 revision

协议修订、接口门禁、12 项合成机制测试与文案更正见
`runs/comm_multistep_stage1_20260922/COMM_MULTISTEP_VERDICT.md`、`COMM_MULTISTEP_PROTOCOL.md`、
`DIAGNOSTIC_TEXT_FIXES.md`。

**下一项最小验证（未执行、未授权）**：在预算有余量的更早决策点确保 S 成功强发一次，或先做“强制发射可行性”门禁。

## 2026-09-21：证据复核与训练诊断已交付（第八轮，未跑任何 episode）

执行入口 [COMM_RESEARCH_REVIEW_NEXT_ROUND.md](COMM_RESEARCH_REVIEW_NEXT_ROUND.md) **已执行完毕**。
分支 `analysis/comm-evidence-review`（起点文档提交 `6fcac4f`），产物在 `runs/comm_evidence_review_20260921/`。
总纲已更新到 v2.2。**本轮 0 新 episode、0 新流、0 训练、生产默认值未改**（`comm_ack=False`、
`comm_timing_mode=legacy_shared_bound`）；归档报告、归档 verdict 与其停止判定原样保留、可追溯。

**唯一的数值改动是"统一到已有的一套"**：归档 `confirm_matrix.judge()` 用两臂行集重算配对差，
`report()` 用五臂行集，而 `evaluate_pilot.paired` 每次重新播种但按臂序逐块消耗抽样 ⇒ 九处 CI 边界出现两套值。
修订版每 (设置, 轴, 基线) 只抽一次样、判读段与 verdict 只切片：

| 设置 | `ack_retry_2 − noack_send_once` | 统一后 CI | total bytes | 统一后 CI |
|---|---:|---|---:|---|
| isolated-lossless（负控） | 0.000 个/ep | [0.000, 0.000] | +10 364 B | [+10 240, +10 471] |
| isolated-lossy | **+0.835 个/ep** | **[+0.685, +0.995]** | **+26 354.1 B/ep** | **[+26 002, +26 688]** |
| normal-lossy | 0.000 个/ep | [0.000, 0.000] | +26 354.1 B/ep | [+26 002, +26 688] |

均值与 3 000 行输入摘要逐项核对**完全相同**（585 个字段）；旧 verdict 的 [+0.685,+0.990] 与
[26 018, 26 687] 作废并登记在 `old_output_diff`。

**十项更正**（`COMM_EVIDENCE_CORRECTIONS.md/.json`，每项含原表述/证据字段/更正/是否改数值）：
`delivery_probability` 是**成功率**（1.0/0.7 ⇒ 丢包率 0.0/0.3）；字节差只写"更高/相等/更低"；
CI 含 0 只写"未检出差异"（isolated-lossy 的 periodic −0.005 [−0.170,+0.160] **未通过** −0.05 非劣界）；
"前沿"拆成样本均值 Pareto／逐流支配／有统计支持的支配三个定义；丢掉非正分母的区间标 **conditional**
（4 116/10 000、1 222/10 000），主比值用独立种子复核 [26 503,38 459] vs [26 522,38 508] ⇒ 稳健；
第六轮基线 `send_once` 的 **ACK 是开的、策略不消费**（total 含 7 374 B ACK 字节）⇒
+19 562 / +19 100.7 / +26 354 是三个不同比较；重试差不单独证明反馈净价值；
学习结论拆三条事实（no-ACK 未建立优势 / ACK 从未训练 / 本轮无学习臂）；删除"扩训已排除欠训练"；
门禁 3 的 93→425 扩样顺序如实披露，不写"96.5 % 的所有决策没有学习空间"。

**训练诊断**（`COMM_LEARNING_DIAGNOSTIC.md/.json`，只用已存记录）：第五轮 6 个运行的曲线、
四个计数分开（环境步 400k / episode 400 / 决策样本 16.5–35.1 万 / optimizer.step 2 792–5 676）、
KL·clip·熵·EV 只作诊断、逐消息序列一致率（learned2 15/24 全同 + 9/24 同时刻不同）、
观测合法性表（紧急度/预算/反馈不确定性如何合法推得）与四种解释表（欠训练、观测不足、
时序信用分配、强规则接近可达效果，允许并存）。**缺失项如实登记**：value loss 与贪心评估熵未持久化；
"一次 HOLD 多久被补发"无法从旧 trace 归因（每 seed 只留 1 份 transcript、转录不标强制步）⇒ 列为待测。

**下一轮设计（未执行）**：`COMM_MULTISTEP_DIAGNOSTIC_PROTOCOL.md` —— 单步 SEND/HOLD 之外做
H=1/5/20 的消息级干预，≤16 点 × 3 个 H × 2 分支 = 96 条闭环分支上限 + ≤16 次复现检查，
开发集探索性诊断、无伪理论门槛；**启动需使用者另行下令**。`800000..800399` 继续封存。

**卫生**：生产代码 `git diff` 对 `integrated_sar/`、`piloc/`、`task_allocation/`、AGENTS、ARCHITECTURE 为空；
676 集成测试通过（2 跳过）+ 本轮新增 24 项合成回归；`python -m integrated_sar` `final_status=completed`；
checkpoint/原始流/逐集日志不入 Git；非本轮创建的 `architecture/` 与 `dev_summary.json` 未修改、未删除、未提交。

## 2026-09-21：（历史，计划已被上一节执行）下一步曾为证据复核与模型诊断

## 2026-09-21：下一步为证据复核与模型诊断（计划，尚未执行）

当前执行入口：[COMM_RESEARCH_REVIEW_NEXT_ROUND.md](COMM_RESEARCH_REVIEW_NEXT_ROUND.md)，总纲 v2.1。
基于独立规则确认交付 `b5fb34e`。已有 3000 集保留，不重跑；本轮仅修分析口径、核对已有训练记录、
设计小规模多步发送时机诊断。新训练、新流、新评估 episode 均为 0，生产默认值不变。

下文是历史状态，引用时须采用总纲 v2.1 更正：31,562 B 为每新增任务成本；ACK 模型未训练；
单步反事实打平和有限扩训都不足以证明学习空间不存在或排除欠训练。
现有工程结果支持当前不默认启用 ACK，不代表取消学习型任务消息控制的最终目标。

## 2026-09-21：S3 学习先导已关闭（第七轮：强规则独立工程确认，未训练任何模型）

当前执行入口 [COMMUNICATION_S3_CLOSURE_NEXT_ROUND.md](COMMUNICATION_S3_CLOSURE_NEXT_ROUND.md) 已执行完毕。
分支 `analysis/comm-rule-independent-confirmation`（起点 `feat/comm-ack-feedback-gate@fbf96d8`，只引入文档提交
`2e341b1`），产物在 `runs/comm_rule_confirm_20260921/`，协议冻结提交 `5ddd7f5`、脚本与流登记提交 `fbeb0b8`。
总纲已更新到 v2.0；**S3 标记为范围准确的负结果，研究先导关闭**。

**本轮不训练**：0 个 seed、0 环境步、未延长旧 checkpoint、未确认单个最好 seed、未改奖励/网络/任务分布/
deadline/生产默认值。`800000..800399` 继续封存（未生成、未读取、盘上无 artefact）。

**数据与规模。** `900000..900199`（200 seed）与全项目**四源零重叠**后才登记录制：盘上 episode artefact
（1 371 seed / 15 201 次运行）、已录制流（1 245 seed）、他轮脚本声明的 seed 区间、manifest 中的同段 seed。
该块只用于规则工程确认，不得改称学习未见测试集。规模 5 臂 × 3 设置 × 200 seed = **3 000 episode**
（每格 200/200，无缺格、无剔除）；吞吐实测 2.53 s/ep、2.79 MB/ep（75 集探测，5 个非新块 seed）。

**结果（200 新流上的同 seed 配对差，B = 10 000，RNG 20260919）：**

| 设置 | `ack_retry_2 − noack_send_once` | total bytes |
|---|---|---|
| isolated-lossless（负控） | 完成数 **0.000**，CI [0.000, 0.000]（0/200/0） | +10 364 B |
| isolated-lossy | 完成数 **+0.835 个/ep**，CI [+0.685, +0.995]（118/72/10） | **+26 354 B** |
| normal-lossy | 完成数 **0.000**，CI [0.000, 0.000]（0/200/0） | +26 354 B |

- **有损隔离确认成立**：第六轮 +0.875 个/ep 在全新数据上复现（+0.835），代价是 total bytes 14 605 →
  40 959 B/ep。**每多一个按时任务约 31 562 B total**（上行 15 218 + ACK 16 343，CI [26 503, 38 459]），
  同一交换新增 23 个逐目标失败、救回 190 个。
- **无损负控通过**：同规则完成数与 `noack_send_once` 逐字相等，只贵 10 364 B ⇒ 增益**来自丢包**。
- **正常信息路径不提部署**：完成数差 0.000 而 total bytes +26 354 B ⇒ 按预注册判读为**工程上被支配**；
  视距与 gossip 旁路使 ACK 增益不可分辨。
- **规则前沿不稳定（如实报告，不重选主规则）**：`ack_periodic_20` 在 isolated-lossy 上完成数 −0.005
  （CI [−0.17, 0.16]）与锁定规则打平却省 3 569 B；normal-lossy 上 +0.03（CI [−0.015, 0.08]）且省 18 148 B。

**冻结的学习结论**（逐字）：在已经测试的任务分布、两类发送协议、no-ACK/ACK 反馈、观测、动作、奖励和
有界训练资源下，学习型发送控制器没有建立相对强规则的附加价值；简单规则形成了当前可复现的成本—效果前沿。
明确**不**写成"强化学习永远无效""规则全局最优""没有任何可学习空间"。四类结论（学习负结论 / 协议与规则
收益 / ACK 任务收益与总成本 / 正常路径工程适用性）在 `COMMUNICATION_LEARNING_VERDICT.md` 分开写死。

**Break-even 由使用者决定**：只有当一个按时任务的价值高于约 31 562 B total 时，ACK 重试才在该效用定义下
合算；本轮不预设该权重，不自动宣布 ACK 值得部署。生产默认值未改动（`comm_ack=False`）。

**卫生**：生产代码零改动（`git diff` 对 `integrated_sar/`、`piloc/`、`task_allocation/` 为空）；
676 测试通过（2 跳过）；`python -m integrated_sar` `final_status=completed`；12 项 preflight 全过；
manifest `dirty_modules` 为空、被执行模块 blob 全部匹配 HEAD；checkpoint、原始流、逐 episode 日志不入 Git；
非本轮创建的未跟踪 `architecture/` 与 `dev_summary.json` 未修改、未删除、未提交。

## 下一轮计划（2026-09-20，ACK 信息价值门禁已执行，停在未通过分支）

当前执行入口 [COMM_ACK_VALUE_NEXT_ROUND.md](COMM_ACK_VALUE_NEXT_ROUND.md) 的 §5 门禁**已跑完**：
分支 `feat/comm-ack-feedback-gate`（起点 `feat/comm-policy-owned-timing@23feba8`，只引入文档提交
`d4c73af`），产物 `runs/comm_ack_value_20260920/`。本轮实现真实、计费、有延迟/丢包的累计 ACK
反向信道（`--comm-ack`，默认关闭），并跑 48 流 × 22 臂规则矩阵（2 112 episode）与 425 点受控反事实。

**四项门禁：1、2、4 通过，第 3 项未过 ⇒ 未训练任何新 seed。**

1. **信息存在：通过（48/48 流）**。每条训练流都出现"上行已尝试、接收未确认、随后 ACK 状态变化"的
   合法决策链——ACK 确实提供了发送端原本没有、真实可获得的信息。
2. **信息有回报：通过，但只在完成数轴**。训练集锁定的最强 ACK 规则 `retry_until_acked_2` 对最强
   no-ACK 规则 `send_once`：**+0.875 个/ep，95% CI [+0.5625, +1.1875]**（26 流更好 / 2 更差 /
   20 相等）；**total bytes（上行+ACK）却贵 +19 562 B，CI [+19 104, +20 044]，48/48 流全贵**。
   无损负控上同一规则与 `send_once` 在 48/48 流上完成数逐字相等（差 0.000，CI [0,0]）且贵
   19 980 B ⇒ 增益来自丢包，不是"花钱买分"。
3. **形成真实取舍：未过**。425 个受控反事实点中 **SEND 更优 3 个（3 条流）、HOLD 更优 12 个
   （8 条流）、完成数打平 410 个（96.5 %）**；门槛为各 ≥20 个且在 ≥8 条流。字节轴同样以打平为主
   （363/425），`send − hold` 的 total-byte 差中位数为 0。一致性检查 425/425 通过
   （与规则自身选择一致的分支逐字节复现未强制运行）。
4. **无简单规则占优：通过**。22 条 ACK 规则 × 48 流，两个信道上都没有逐流双轴占优的臂。

因此本轮**不训练**，不产生 `COMM_ACK_POLICY_PROTOCOL.md`，**不**提高丢包、**不**收紧 deadline、
**不**改场景/奖励/网络，**不**追加训练。最终测试继续封存，默认 no-ACK 路径不变；
以下第五轮及历史记录保持原含义。

## 已完成（最近一轮摘要在下，历史记录保持原含义）

> **2026-09-20 第六轮：ACK 信息价值门禁已执行，门禁 3 未过，未训练任何新 seed。**
> 分支 `feat/comm-ack-feedback-gate`（起点 `feat/comm-policy-owned-timing@23feba8`，只引入文档提交 `d4c73af`），
> 产物在 `runs/comm_ack_value_20260920/`。`--comm-ack` 打开一条独立的**反向 ACK 信道**：
> ACK 只由接收端把上行包真正应用进 belief 产生，载荷携带已应用的最高
> `(sent_at, revision, content_fingerprint)`，累计且单调（重复/乱序不回退），
> 经有延迟/丢包的反向信道回传（参数默认与上行相同，counter-based 共同随机数），
> 每次发送尝试按序列化字节全额计费、丢失也计费，走**独立反向账本**而不消耗 29 300 B 任务额度。
> 发送端观测增加四维 `comm-send-state/2`（已确认比例、最大 ACK age、最大视图 gap、
> 未全体确认广播次数），全部由自身发送历史与已到达 ACK 推出，无 ACK 时为显式 unknown 编码；
> 默认关闭时**不构造任何反馈状态**，旧 no-ACK 路径在发射/投递/任务结局/预算上逐字节一致。
> 同时新增四条只有反向信道存在才可能存在的规则（`retry_until_acked`、`event_retry_until_acked`、
> `exponential_backoff`、`deadline_tightening_retry`），与学习发送端读同一份信息。
>
> **2 112 episode 规则矩阵（22 臂 × 48 训练流 × 两信道）+ 425 点受控反事实** ⇒ 四项门禁
> **1、2、4 通过，第 3 项未过**：
> ① 信息存在 **48/48 流**通过；② 最强 ACK 规则 `retry_until_acked_2` 比最强 no-ACK 规则
> `send_once` **+0.875 个/ep，CI [+0.5625, +1.1875]**（26/2/20 流），但 **total bytes
> （上行+ACK）48/48 流更贵，+19 562 B，CI [+19 104, +20 044]**（贵 88.5 %，ACK 自身约占三分之一）；
> 无损负控上同规则与 `send_once` 完成数逐字相等（0.000，CI [0,0]）⇒ 增益来自丢包而非花钱；
> ③ **SEND 更优仅 3 个决策点（3 条流）、HOLD 更优 12 个（8 条流）、完成数打平 410 个（96.5 %）**，
> 门槛为各 ≥20 且在 ≥8 流 ⇒ **未过**；④ 22 条规则中没有任何一条逐流双轴占优。
> 反事实一致性检查 **425/425**（与规则自身选择一致的分支逐字节复现未强制运行）。
>
> 按指令**停止**：不训练三个新 seed、不产生 `COMM_ACK_POLICY_PROTOCOL.md`、不提高丢包、
> 不收紧 deadline、不改场景/奖励/网络、不追加训练。成立的主张只有三条且必须一起引用：
> **ACK 信息真实且有后果**（一条简单规则拿到全部 +0.875 个/ep）；**ACK 不是免费的且净增成本**；
> **ACK 状态没有形成可学习的逐决策取舍**。**不主张**"ACK 没有可学习空间"（门禁 3 只否定这
> 425 个点），**不主张**"ACK 值得部署"（那是成本—效果决策）。
> 全套 **676 测试通过（2 跳过）**；`python -m integrated_sar` `final_status=completed`；
> 12 项 preflight 全过；`800000..800399` 仍封存、未生成未查看。
>
> 开发期一致性检查抓到一个真缺陷并已修：反事实探针把 `orders_selection` 规则的有序选择
> 转成 `set` 再转回 tuple，破坏了优先级顺序——在额度不足时这会改变谁被付费，
> 使探针 episode 不再是分支所对比的那个 episode。修复后一致性由 92/93 变为 425/425。

> **2026-09-20 第五轮：发送时机协议消融已完成，并停在执行指令写明的停止条件上。**
> 分支 `feat/comm-policy-owned-timing`（起点 `fix/comm-ppo-probability-budget@edba1e3`，只引入文档提交 `185c2a2`），
> 产物在 `runs/comm_protocol_v2_20260920/`。`--comm-timing-mode` 双协议：默认 `legacy_shared_bound`
> 与全部历史实验逐字节一致；`policy_owned_timing` 只取消"上界到点由传输层代发"这一件事
> （硬额度、每步容量、内容合法性、终态抑制、候选生成、消息合并、丢包计费全部不变）。
> 3/10 时机写成显式规则（`first_send_plus_bound_3_10`/`bound_only_3_10`/`send_once`/`always_hold`），
> 48 个 episode 证明前两条与 `immediate`/`hold_to_bound` **逐包、逐时刻、逐顺序、逐字节等价**；
> 1 152 个 episode 规则矩阵在训练集锁定主比较对象：29 300 B → `send_once`、87 900 B → `first_send_plus_bound_3_10`。
> 两档 × 三 seed 各 400k 步有界训练：**主门槛 0/3、降本门槛 0/3**。协议收益成立且在规则侧——
> `send_once` 以 14 916 B 拿 5.7917 个/ep，支配 `immediate` 同行为（26 897 B、5.2917 个/ep，
> 后者 49.9 % 字节是上界代发的刷新）；学习附加价值未建立（25 % 最好 +0.125 [+0.000, +0.333] 且多花
> 3 196 B；learned2 与 `send_once` 逐字相同；75 % 最好 −0.042 [−0.125, 0] 且省 4 529 B 但低于 −0.05 非劣界），
> 六个运行全部显著优于各自初值（+1.04～+4.21 个/ep）。按指令**停止**：不追加训练、不转 ACK/新观测。
> 全套 **639 测试通过（2 跳过）**；`python -m integrated_sar` `final_status=completed`；
> 11 项训练门禁全过；`800000..800399` 仍封存、未生成未查看。

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

- 2026-09-19（同日第三轮）：**通信学习收益归因与训练稳定性**，
  分支 `feat/communication-training-stability`（起点 `ab5728be…`），产物在
  `runs/comm_stability_20260919/`；总纲 `PROJECT_CHARTER.md` 与执行指令
  `COMMUNICATION_NEXT_ROUND.md` 从文档分支 `b5821b6` **只引入这两个文档文件**，未回退其旧生产代码。
  - **归因结论先行**：24 条验证流 × 14 臂 = 336 episode 的对照表显示，**首轮那条省通信收益
    主要来自共享静默上界，不是学到的发送时机**。把同一条上界写成规则 `hold_to_bound`
    即复现整个结论形状：对 `periodic_5` +0.083 [−0.083, +0.25] 个/episode、
    **−37 967 B [−42 471, −33 470]**，比学习模型还省 3 199 B（CI [+1 296, +5 750]）。
    首轮的 learned seed 3 在 24 条流上与该机器的完成数**逐 seed 全等（差恒为 0）**。
    补齐成本—效果表后，`immediate`（6.000 个/ep、35 084 B）同时支配所有学习臂与其余规则，
    `send_once_no_bound`（真实历史"只发一次"）只要 14 916 B。
  - **发送归属账本**：`CommRelay` 给每条发射标注互斥主原因（`policy` / `forced_maxwait` /
    `other`），发射/尝试/字节/被拒四分列并按原因闭合，transcript 逐条带 `send_reason`；
    废弃原来"计费前就记为强制发送"的 `forced_emissions`。新增规则臂 `hold_to_bound`、
    `random_send`（可复现硬币）与不带共享上界的 `send_once_no_bound` 参照。
    同时修学习型策略把掩码项混入自身优先序列的问题（规则臂的强制发送一律排队尾），
    72 对 episode A/B 实测**不改变任何字节或完成数**，只改正归属标签。
    336 条 episode 的 7 条成本/结局恒等式**残差全 0**，离线 transcript 与在线计数在 8 个共享量上
    逐集一致；生成量平衡式有一处已知缺口（−40 单位 / 21 集，`coalesced_changes` 不计轮询侧合并），
    如实登记并留作单独修复。
  - **单位与口径核对**：首轮 `+0.083`/`+0.333` 与 `−0.05` 判据的单位经查为**个任务/episode**
    （不是完成率、不是百分点），原文保留并登记纠错；每臂同时给出个/ep、逐局率 [0,1]、
    Σ汇总率、B/批次、B/包、自主字节占比。另纠正首轮推理开销数字：记录的 200–255 ms/决策
    是三进程 × 全核 torch 线程超订的产物，单线程复测 **0.19–0.24 ms**，首轮数字作废。
  - **训练量核查**：实测 100 000 环境步 = 100 episode × 1 000 联合步（无提前终止，逐集断言），
    但只有 25 个更新轮、**1 168 / 1 212 / 1 072 次 optimizer.step** 与约 7×10⁴ 条决策样本
    ——用环境步表征学习量高估近一个数量级；48 条训练流被重放 2–3 次（本轮 6–7 次），
    **不是 100/400 个不同场景**。更关键：候选决策里 **66–86 % 在到达模型前就被安全掩码定死**。
    ε-greedy 把行为 log-prob 记成 π 的偏差、以及"训练 rollout 恒打满额度（0.96–0.997）而
    greedy 评估只用 0.40"的训练/评估分布错配，均已量化登记但**本轮不修**（只改时长一个主变量）。
  - **有界续训（rollout 保持 4 episode，100k→200k→400k）**：权重与 Adam 动量精确恢复、
    首轮未存 RNG ⇒ 如实标为 **warm start**；里程碑落在 152k/200k/300k/400k（更新边界 4 000 步）。
    实测吞吐 3.05 s/episode、12.2 s/更新轮，**100k→400k 三 seed 并行 < 20 分钟**，
    首轮"三个工作日"估计作废。结果：三 seed 全部到达平台，完成率极差 1.833 → 0.042、
    字节 70 738 → 40 487（−42.8 %）——**"2/3 seed 失败"主要是欠训练**；
    但预注册的四条判定线 **0/3 通过**：seed 3 在 400k 与 `hold_to_bound` **逐值相同
    （36 161.5 B、自主字节 0 B）**，seed 2 打平 `immediate` 完成率（24/24 Δ=0）却贵 4 028 B，
    seed 1 贵 11 102 B。丢包 0.3 诊断下 seed 2 把接收端信息年龄从 42 步压到 **10.3 步**，
    完成率仍不足越线；正常路径（视距 + gossip 全开）下只有 learned3 比 `immediate` 省
    （−8 043 B），但其自主字节仅 3 %，且受控通道投递率仅 78.5 %——不主张全系统通信下降。
  - **判定与下一轮**：成立的是"**规则在本场景已足够**"（三分诊断逐项给证：欠训练已被扩训排除、
    缺 ACK 是真实上限、额度对最优三条臂只用 0.40–0.45 因而从未强制取舍），
    **不成立**"没有学习空间"这一泛化结论。唯一有依据的下一步是**只改一个变量：把预算降到
    25 % 档（29 300 B < immediate 的 35 084 B）**，让额度真正 binding 后重跑归因矩阵与三 seed；
    不扩动作空间、不改奖励、不收紧 deadline、不新增数据。
  - **数据划分**：审计 `runs/**` 全部 artefact 后确认，首轮登记的"独立测试集"
    `500000..500399` **400 个 seeds 全部执行过**（修正环境 C/D 对比，每 seed 2 集，
    commit `f7fea38`，配置与流内容哈希逐条登记），不再可作未见测试；1 245 个有录制流的 seed
    **零个**存在多内容，故编号重叠即数据重叠。新预留 `800000..800399`（仅登记，未生成未查看），
    全项目累计 1 368 个 seed 执行过 episode。本轮训练/验证块用途不变。
  - **状态**：成员一致性开启与 C 臂配置从**构建出的 `RunConfiguration`** 反读核对
    （`mismatches: {}`，不再拿 dataclass 默认值冒充实验配置）；默认配置不变；
    12 项 smoke **`failures: []`**；全套 **598 通过（2 跳过）**；`python -m integrated_sar`
    `final_status=completed`；`.pt` checkpoint、录制流、逐 episode 日志与 transcript 均不入库。
    交付头 `545e526fd140887a39ea924f501b76054f885615`（episode 跑在该树内容上；
    `COMM_STABILITY_ROUND_MANIFEST.json` 于提交后重生成，故其 `dirty_modules` 为空、
    被执行模块哈希与 HEAD 一致）。

- 2026-09-19（同日第四轮）：**PPO 概率一致性、账本口径与紧预算对照**，
  分支 `fix/comm-ppo-probability-budget`（起点 `7834034b58fcb71dda2643f8216cbc8639189a69`），
  产物在 `runs/comm_tight_budget_20260919/`；总纲 v1.3 与执行指令
  `COMM_PPO_BUDGET_NEXT_ROUND.md` 来自文档提交 `4bd44da`，**只引入文档**
  （三个文件按 staged blob 逐字节核对：`4fed2d69` / `ae4c1059` / `7c0f5c4c`）。
  - **概率正确性修复（单独提交 `f6827c7`）**：pilot 采集器用
    `mu = 0.9·pi + 0.1·Uniform` 抽动作却记 `log pi`，更新也用纯 π 的 ratio。改为直接从有掩码的
    π 抽样、采样侧与更新侧共用一份 `joint_log_prob` / `action_entropy`、被上界接管的候选不进
    likelihood 与 entropy、并记录当步可行动作集合。常规体检看不见这个缺陷（权重未动时 ratio 仍为 1），
    所以审计改测"抽到的频率 vs 记录的概率"：初值上差 1.15 SE、逐决策偏差 ≈0（π 接近均匀，这就是
    它三轮混过去的原因）；在第三轮 400k checkpoint 上旧采样器实际发送率 **0.1778 vs 记录的 0.1261**
    （高估 41 %，4.32 SE），逐决策 `log mu − log pi` 平均 0.034 nat、平均绝对 0.150 nat、最坏 1.394 nat。
    修正后两条主体都回到噪声内（−0.65 / −1.29 SE）。新增 12 项回归（含"同样的检查必须能失败"的
    混合采样复现、被强制行梯度恰好为 0、optimizer 动量与场景 cursor 随 checkpoint 往返）。
    **训练器拒收错配带**：翻转一条带的动作后训练抛错（实测偏差 0.531 nat）；六次运行全程
    重算偏差 ≤ 1e−6 nat。
  - **账本口径修复（单独提交 `237126d`）**：第三轮把 −40 单位的生成量残差归给"轮询侧 coalescing
    漏计"，本轮分别复现两个候选机制后**更正归因**——轮询侧合并确实漏计，但它两边同减、残差恒为 0，
    产生不了负号；负号来自恒等式两边用了两套"同一件事"（`generated_items` 按 `_View` 计，
    transcript 的 `redundant` 按 registry `revision` 计）。给 `_TargetLedger` 加 `ready_view`
    使轮询侧合并同时计为一条生成与一次合并（与事件路径同规则、不重复计同一新闻），
    并新增发送侧 `redundant_emissions`。**624 格（全规则臂 × 两档 × 48 流）修正前后逐格核对
    发射、字节、自主/强制字节、完成数、检测数、超时、信息年龄与拒付明细全部相同**，
    两个计数各 +44 124（相等），旧读法残差变化 0；剥掉产物路径后配置 56 格一致。
  - **29 300 B 是否真约束强规则（§4 闸门，先测规则）**：`event`/`event_priority`/`periodic_5`
    **48/48 集**出现拒付、`immediate` 与 `hold_to_bound` 34/48 集，首次拒付中位步 11–20（强规则）
    / 64–97（稀疏规则），且被拒的大多是臂**自己请求**的发送（`immediate` 16 743 次 policy 拒付）
    ⇒ 闸门通过，进入双预算训练；同时如实记下"额度触顶是逐集属性"：`immediate` 在 25 % 档仍有
    14/48 集完全没触顶，75 % 档平均只用 0.40 额度、0 集拒付（与第三轮一致）。
  - **判据先于结果冻结**（`COMM_TIGHT_BUDGET_PROTOCOL.md`）：主指标个/episode，非劣界保持
    **−0.05 个/episode** 不换单位；比较对象只能由训练集选定 ⇒ 两档都选 `immediate`
    （75 % 5.8958、25 % 5.3542），`immediate` 与 `hold_to_bound` 始终完整报告；
    `send_once_no_bound`（5.5833 @14 737 B，比任何上界内规则又好又省）按协议成本诊断保留呈现，
    明确排除在任何门槛之外。
  - **两档 × 三 seed 配对训练**：全部**从随机初始化重训**（ε-greedy 时代的 checkpoint 被新 trainer
    拒绝加载），每档 400k 步、100 更新轮、100k/200k/300k/400k 里程碑精确落点，
    3 164–6 068 次 optimizer.step、18.8–37.7 万条决策样本。吞吐实测 24.0 s/更新轮、
    167 环境步/秒（六进程并行），旧"三个工作日"说法彻底作废。
  - **主判据结果：0/3**（隔离无损，逐流配对 vs `immediate`）。25 % 档 seed1 −1.125 [−1.542,−0.708]、
    seed2 −1.792 [−2.333,−1.292]，seed3 **+0.000 [0,0]，24/24 条流与 `immediate` 完成数逐字相等**、
    字节 +100 [−12,+309]；75 % 档 seed1 −1.000、seed2 −0.042、seed3 −0.083，字节均略高。
    `learned2_at400000` 与 `hold_to_bound` 在 24/24 流上（完成数, 字节, 发射数）**逐项相同、自主字节 0 B**
    ⇒ 第三轮的"学成等待规则"现象**不是 ε-greedy 假象**。卫生项干净：无一分突破额度、恒等式残差全 0。
  - **正向那一半：修好采样器后学习稳定发生**。与**各自未训练初值**逐流配对，六个运行**全部**为正且
    CI 下界 > 0：75 % 档 +1.208 / **+2.167** / +1.917 个/episode（后两者同时省 50.8k / 50.5k B），
    25 % 档 +1.208 / +0.542 / **+2.333**（seed 3 还省 1 333 B [−2 485,−365]）。
    因此失败定位不再是"学不动"，而是"**学到的极限就是规则**"。
  - **附加诊断（不改训练目标、不作主判据）**：丢包 0.3 下仍 0/3 越线（25 % 档 seed 3 与 `immediate`
    打成 4.833 平手）。正常信息路径 + 75 % 档出现两条**过"同效果省通信"门槛**的臂
    （seed2@300k：完成数 24/24 与 `immediate` 相同、**−2 074 B [−3 361,−759]**；
    seed3@400k：0/24 全等、−2 719 B [−3 971,−1 451]）——但该通道上 `hold_to_bound` 只用
    26 734 B 就拿 5.9583，比两者更省，且它只是探索性诊断，不构成方法主张。
  - **结论范围**：只主张"修正后的这套训练方法，在这两档预算与当前协议上未优于强规则"，
    **不**写成"已证明当前观测/动作没有更优策略"；平台不证明优化充分。局限逐项列出：
    25 % 档仍有 14/48 集不触顶、75 % 档几乎不 binding、学习臂可选候选仅占 3.7–9.1 %、
    上行无 ACK、决策样本量仍偏小、48 条流被重放 8–9 次而非 400 个不同场景。
  - **状态**：全套 **614 通过（2 跳过）**（598 + 12 概率回归 + 4 记账回归）；本轮 12 项 smoke
    `failures: []`（含续训等价：连跑两轮与"跑一轮、存盘、重载、再跑一轮"末次报告与权重指纹逐值相同）；
    `python -m integrated_sar` `final_status=completed`；配置真值 `mismatches: {}`；
    `800000..800399` 仍封存、盘上无 artefact；两档额度均等于锁定标定值；
    checkpoint / 流 / transcript / 逐集日志不入库，`COMM_TIGHT_BUDGET_MANIFEST.json` 以
    **git blob 身份**（而非裸字节，避开 checkout 的 CRLF 假不匹配）核对被执行模块与 HEAD 一致。
    逐问题提交：文档引入 `e9dc21b`、概率修复 `f6827c7`、记账修复 `237126d`、
    紧预算闸门 `0464f3b`、判据冻结 `a4cfb4f`、结果与交付 `157898e`；
    manifest 在 `157898e` 之后重生成（`executed_code.head` 即该值、`dirty_modules: []`、
    全部被执行模块 matches_head），随后以一次 chore 提交登记交付 SHA。
    旧的二、三轮预注册判定与产物原样保留，未重跑、未改写。

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
13. ~~通信账本有一处已知缺口（本轮登记，未中途修）。~~ **已于第四轮修掉并复验**（提交 `237126d`）：
    给 `_TargetLedger` 加 `ready_view`、发送侧加 `redundant_emissions`，两侧同用 `_View` 口径；
    624 格实测修正前后发射/字节/扣费/任务结局逐格不变，两个计数各 +44 124。原文保留备查：
    `CommRelay.counters["coalesced_changes"]` 只在事件路径（`handle_detection`）上计数；
    若一条变化已排队、而更新是**轮询注册表**时发现的，它既不计入 `generated_items` 也不计入
    `coalesced_changes`，于是生成量平衡式在 21/336 条 episode 上短 40 单位
    （learned_seed1 −29、init_seed3 −5、learned_seed2 −3、init_seed1/2 与 random_send 各 −1）。
    **所有成本与结局恒等式残差为 0**。另：第四轮 `probe_after/` 的转录文件名只含臂与流、
    两档互相覆盖，75 % 档的事后复核路径已不可用（其**发表的**按档聚合 JSON 不受影响）；
    本轮及以后的产物一律带档前缀。

## 推荐下一任务

**通信学习这条先导已按预注册停止条件关闭**（S3 七轮，v2.0）：不再有"继续训练"或"挑单个最好 seed
做独立确认"的授权。当前没有通过门槛的学习候选，`800000..800399` 未打开、未生成、未查看，
也没有独立学习性能主张。若使用者未来仍要求学习贡献，必须另立新假设与新阶段（例如有现实依据的
非平稳信道或任务压力），并先做无模型 learnability calibration；不得以提高丢包率、收紧 deadline
或改变场景制造信号。

剩下的决定权分两类，都在使用者手上：

1. **工程类（本轮已给出证据，等你决定）**：`ack_retry_2` 在隔离有损信道以约 +31 562 B total/ep
   换 +0.835 个按时任务/ep（200 条全新流，CI [+0.685, +0.995]），在正常信息路径上无可分辨增益
   且更贵 ⇒ 不提部署。只有当"一个按时任务的价值高于约 31 562 B"在你的效用定义下成立时才合算。
   生产默认值未改动（`comm_ack=False`）。
2. **基础设施类（与通信研究无关，一直开着）**：环境修复是否并入主线、协调器候选是否走向默认启用、
   执行边界对"目的地已结束"要不要挡、1.4 % 空闲 robot-tick 的代价是否接受。这些改变的是任务分配
   与执行侧，**不是**通信学习结论的证据，也不因通信先导关闭而自动推进。

已排除（总纲 §11 与各轮停止条件）：单纯加步数、再减预算、扩动作空间、改奖励加信息年龄项、
收紧 deadline 制造难度、rollout 16 episode、以及"提高丢包/改场景制造学习信号"。

以下第五、六轮与历史记录保持原含义。


**ACK 信息价值门禁（第三轮列的第 3 条"改观测/反馈"）已经执行完毕，答案分两半**：
`feat/comm-ack-feedback-gate` 实现了一条真实、计费、有延迟/丢包的累计 ACK 反向信道
（`--comm-ack`，默认关闭；旧 no-ACK 路径逐字节不变），并用 48 流 × 22 臂规则矩阵与
425 点受控反事实检验它是否创造可学习取舍：

- **ACK 信息是真实的、可获得的、有后果的**：门禁 1 在 **48/48 条训练流**上通过；
  训练集锁定的最强 ACK 规则 `retry_until_acked_2` 在主设置有损信道上比最强 no-ACK 规则
  `send_once` 多 **+0.875 个/ep，95% CI [+0.5625, +1.1875]**（26 流更好、2 更差、20 相等）。
  无损负控上同一规则与 `send_once` 在 48/48 流上完成数逐字相等（0.000，CI [0,0]）
  ⇒ 增益来自丢包，不是"多花字节买分数"。
- **ACK 不是免费的，而且是净增成本**：`total bytes（上行+ACK）` **48/48 条流更贵**，
  +19 562 B，CI [+19 104, +20 044]（22 110 → 41 672 B，贵 88.5 %）；ACK 自身占总成本约三分之一，
  连 `send_once` 这种不发刷新的臂也要为每次成功应用付 ACK 字节。**只省上行、把成本移到反向通道
  不算降本**——本轮实测反向通道也变贵。
- **ACK 状态没有形成可学习的逐决策取舍**：425 个受控反事实点中 SEND 更优 3 个、HOLD 更优 12 个、
  完成数打平 410 个（96.5 %），门槛各 ≥20 且在 ≥8 流 ⇒ **门禁 3 未过**。一致性检查 425/425。
- 门禁 4 通过：22 条 ACK 规则中没有一条逐流双轴占优，所以不能说"某条简单规则吃光了全部收益"。
- 因此本轮落在执行指令写明的第三种情形：**ACK 没有创造可学习的取舍**，
  **不训练**、不产生 `COMM_ACK_POLICY_PROTOCOL.md`、不提高丢包/收紧 deadline/改场景制造信号。

通信这一支现在剩下的可选方向（决定权都在使用者手上）：

1. **如实收尾为范围准确的否定结论**：六轮合起来的交付物已经完整——闭合到残差 0 的发送归属账本、
   624 格行为不变的记账修正、一份能失败的概率一致性审计、两档 × 三 seed 的预注册负结果、
   第五轮"协议收益与学习收益分开"的双层结论、以及本轮"ACK 信息真实但未形成可学习取舍"的反馈层结论。
   代价：不再新增训练。
2. **另立新的反馈/观测定义**：本轮证明的是"这条 ACK 协议 + 这 425 个决策点没有可学取舍"，
   **不是**"ACK 没有可学习空间"。若要继续，需要一种让确认状态更频繁变化、且变化与发送时机
   真正相关的反馈定义（例如把重发与 urgency 收紧耦合起来），并单独预注册。
3. **把 `retry_until_acked_2` 当工程方案评估**：它在有损信道上用 +88.5 % 的总通信换 +0.875 个/ep
   的按时完成。这是成本—效果决策，不是学习主张，也不在本轮授权范围内。
4. **为 75 % learned3 另立独立确认协议**（第五轮留下的唯一"方向对"的臂，−0.042 个/ep、−4 529 B，
   卡在 CI 宽度上）。
5. **已排除**：单纯加步数、再减预算、扩动作空间、改奖励加信息年龄项、收紧 deadline 制造难度、
   rollout 16 episode、以及本轮之后的"提高丢包/改场景制造学习信号"。

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
