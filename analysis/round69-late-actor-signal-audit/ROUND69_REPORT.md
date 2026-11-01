# ROUND69 报告 — R68 late actor 更新信号只读审计

任务：`ROUND69-LATE-ACTOR-SIGNAL-READONLY-AUDIT`
分支：`analysis/round69-late-actor-signal-audit`（建于已核验 HEAD `420fa9af18932f735efdf6528ab68174c561ec52`）
被审计轮次：`ROUND68-LATE-INITIAL-CONFIDENCE-ABLATION`（运行目录
`runs/comm_bc_initialized_constrained_ppo_20261007_r2/round68_late_initial_confidence_ablation/`）
机器结果：`analysis/round69-late-actor-signal-audit/ROUND69_LATE_ACTOR_SIGNAL_AUDIT.json`
审计脚本：`analysis/round69-late-actor-signal-audit/r69_late_actor_signal_audit.py`
执行依据：`ROUND69_EXECUTION_BASIS.md`（授权提示词原文存档）

**零副作用。** 本轮不构造环境、不新增 episode/claim、不做真实或合成 optimizer 更新、不重放 PPO、
不做模型 forward/autograd、不恢复 RNG、不写 checkpoint；只读 R68 既有 384 个训练 row/tape、96 个
batch 提交记录、门禁、冻结统计与链资格文件，并在内存中做纯算术回归。

## 1. 接手核验

HEAD 核验为 `420fa9a`（R68 收口提交），工作区仅有既定未跟踪的 r59 日志，无同目标任务在运行
（raw 最后写入时间早于接手时刻）。R68 收口件（最终账本 775/800、三 seed 链资格 verified、
机器 verdict）均在位且哈希相符，直接在其上做只读审计，未改动任何 R68 原件。

## 2. 方法：沿真实源码确认字段时点与单位

- 采样粒度：`v2_trainer.batch_samples` 对每个 episode 的 tape（`onpolicy-tape/2`）**一行一个联合
  决策步**，联合 log_prob、`value`/`cost_value`（rollout 前、即更新前的行为 critic 值）都按决策步
  保存；多候选联合决策在训练里是**一行**（行权重 1），其目标不是独立样本。
- 奖励归集：`v2_trainer._bundle` 把每个 tick 的事后奖励记到**不晚于该 tick 的最近决策步**；首个
  决策之前的前缀单独报告（本报告用其做对账）。成本奖励 = 归集 airtime_bytes / B0（B0=10000），
  取正成本约定；任务奖励 = 归集任务事件数。
- GAE：`cost_math.gae`，`A_t = delta_t + gamma*lambda*(1-done)*A_{t+1}`，`returns = A + V`；
  末决策 `terminal=True` 且 bootstrap=0，非末决策 bootstrap 取**下一决策行的保存值**。
- 组合优势：`cost_math.actor_advantage` = `lambda*A_task - A_cost`；因 `A = R - V`，它精确等于
  `(lambda*R_task - R_cost) - (lambda*V_task - V_cost)`，本审计即按此分解。
- 活体常量（非导入默认）：从 trainer 实际导入的 `v2_common` 取 GAMMA=1.0、B0=10000、
  MINIBATCH_STEPS=256、EPOCHS_PER_BATCH=4、DATA_ORDER_SEED=20260927；`r68_execute` 在活体
  common 上覆盖 GAE_LAMBDA=1.0、ETA_LAMBDA=0.5（导入默认 0.95/0.05 未使用），batch 记录
  `dual.eta=0.5` 与门禁 `common_gae_lambda_1=true` 运行时佐证。
- 路由：`r66_frozen.route_logits`，`ready_step = round(f[18]*1000) - round(f[0])`，<=147 早头、
  >147 晚头，按目标逐行路由；采样、greedy、似然重算、评估同一路由。
- **MC 边界实测**：gamma=1、gae_lambda=1 且末决策真终止时，首决策 return 伸缩为归集流的完整
  Monte-Carlo 和。对全部 384 个 episode 实测：`return_task[0] + 前缀 == row.task_reward_total`
  （最大绝对差 0.0），`return_cost[0]*B0 + 前缀 == row.airtime_tap_total`（最大 1.8e-12）。前缀
  之后无截断丢失，边界不是"预设"而是逐 episode 验证。
- **手算例**：rewards [1,2,3]、values [0.5,0.7,0.9]、bootstrap [0.7,0.9,0] 得
  A=[5.5,4.3,2.1]、return0=6.0=MC 和，与独立实现一致（脚本内置断言，通过）。
- **精确 minibatch 重建**：用冻结数据序 RNG（`sha256(DATA_ORDER_SEED:update_index)`、每 epoch
  原地 shuffle、256 切块）在内存重建全部 676 个 minibatch 成员，与 96 个 batch 记录里 676 条
  `gate_records` 的 `n_rows/early_free/late_free` **逐条一致（0 不符）**；同时 96/96 batch 的
  samples、early/late free（x4 epoch）、n_rows 加总与记录一致；384 个 row 的 `lambda` 与所属
  batch `lambda_at_rollout` 零 mismatch。样本顺序、路由分类与数据序种子由此得到实证。

## 3. 交付①：计数、ALL_ROWS 分母与 late 占比

96 批（3 seed x 32 批 x 4 episode）合计 **32,046 个决策步行**：

| 类别 | 行数 | 占全部行 |
| --- | --- | --- |
| 全 mask 行（无自由候选） | 30,156 | 94.11% |
| 含早期自由候选的行 | 1,377 | 4.30% |
| 含晚期自由候选的行 | 513 | 1.60% |
| 自由行合计 | 1,890 | 5.90% |

自由候选按目标计 2,286 个：早期 1,767（77.3%）、**晚期 519（22.7%）**；晚期目标实际动作
SEND 423（81.5%）、SKIP 96（18.5%）。含晚期自由候选的行占自由行 27.1%。

ALL_ROWS 归约分母：每个 minibatch 的 surrogate/entropy/value 均值分母是**该块全部样本行**
（含全 mask 行），全轮共 **128,184 个行实例**（32,046 x 4 epoch），分布在 **676 个 minibatch**；
其中 550 个 minibatch 含至少一个晚期自由行，晚期行实例共 2,052 个，即平均每个 minibatch
约 3.04 行、约 1.2% 的行实例带有晚期 actor 信号。按 seed 与 batch 的完整分列见机器 JSON
`deliverable1_counts.per_batch`（96 行，含每批 samples/all_mask/late_rows/targets 及与记录的
一致性核对）。lambda_at_rollout 从 1.0 单调升至 1.625 / 1.375 / 1.500（按 seed）。

## 4. 交付②：晚期自由行上的组合优势分解

对 513 个含合法晚期自由候选的决策步行（行权重 513，多候选联合行不拆成独立目标样本）：

| 量 | 均值 | 中位 | p10 | p90 | 最小 | 最大 | 正例比例 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 组合优势 lambda*A_task - A_cost | +1.633 | +1.469 | +0.314 | +3.306 | -1.444 | +4.880 | 92.2% |
| lambda*A_task | +1.941 | +1.848 | +0.412 | +3.704 | -1.228 | +5.911 | 93.8% |
| A_cost | +0.308 | +0.257 | +0.025 | +0.576 | -0.253 | +1.295 | 92.4% |
| return 部分 lambda*R_task - R_cost | +1.975 | +1.731 | +0.879 | +3.519 | 0.000 | +5.002 | 98.8% 非负 |
| value 部分 lambda*V_task - V_cost | +0.342 | +0.253 | -0.271 | +1.062 | -1.554 | +2.568 | 75.4% |
| critic 残差 R_task - V_task | +1.523 | +1.407 | +0.299 | +2.956 | -0.893 | +4.855 | 93.8% |
| critic 残差 R_cost - V_cost | +0.308 | +0.257 | +0.025 | +0.576 | -0.253 | +1.295 | 92.4% |

分量尺度：`R_task` 均值 1.786（中位 2）、`R_cost` 0.347、`V_task` 0.262、`V_cost` 0.039；
`lambda*A_task` 均值约为 `A_cost` 的 6.3 倍——在 lambda>=1 下任务项主导量纲。

动作分布（按目标）：晚期自由目标 SEND 423 / SKIP 96。按行分：任一晚期目标取 SEND 的行 417
个（组合优势均值 +1.545，91.6% 正）；全部晚期目标取 SKIP 的行 96 个（均值 +2.014，94.8% 正）。
晚期目标的行为发送概率均值 0.810。**两类行的优势都以正为主且量级相近**：组合优势度量的是
"实际联合动作相对于 critic 基线"，不构成 SEND 与 SKIP 的因果价值对比。

分桶对照（masked/forced/冻结早期角色分开）：仅含早期自由候选的 1,377 行组合优势均值 +7.09、
100% 为正（早期路径在 R68 已冻结，该信号无处可去）；全 mask 的 30,156 行均值 +2.66、83.1% 为正，
但它们不产生任何 actor 梯度，只稀释 ALL_ROWS 分母。按 seed：晚期行各 171 行，组合优势均值
s11 +2.050（99.4% 正）、s22 +1.559（91.2%）、s33 +1.290（86.0%）。

趋势（按 8 批一段）：s22 的组合优势均值 2.13 -> 1.61 -> 1.41 -> 0.73、正例比例 100% -> 71.9%；
s33 为 1.37 -> 1.28 -> 1.20 -> 0.95、97.9% -> 70.8%；s11 维持 1.67 -> 2.16 -> 2.26 -> 1.76、
基本 100% 正。已记录的 late_head 每 minibatch 截断前梯度 L1：96 批均值 0.269（0.011..2.018），
同期 task_value_head 均值 78.07（约 290 倍）、cost_head 3.73（约 13.9 倍）。beta=1.00 门禁先验
对可训练路径的梯度在三 seed 门禁记录中均为 0（先验候选全部路由到已冻结的早期头），96 批
gate_loss 约为 0——晚期头只收到 surrogate 信号。

## 5. 交付③：哪些解释得到支持、哪些不成立、哪些不可判

**"自由晚期动作几乎没有可用更新信号"（强形式）不被支持。** 晚期行携带结构化、以正为主的组合
优势（92.2% 正，return 部分 98.8% 非负）；late_head 的实测梯度在全部 96 批非零且在案；链资格
证明晚头参数确实移动（trunk 与 early_head 到 ep128 逐位未动）。

**"有信号，但其方向/幅度未形成稳定保任务降本"与数据相符。** 信号稀疏（1.60% 的行；每 minibatch
约 3 行；94.11% 的行是无动作全 mask 行）；方向不区分 SEND/SKIP（两类行都约 92-95% 正优势，
优势主体是"实际结局高于 critic 基线"而非两种动作的对比）；幅度相对 critic 小 1-2 个数量级且随
训练衰减（s22/s33 末段均值与正例比例双降）。这与 R68 观测一致：greedy 三 seed 任务/成本零变化
（CI95 [0,0]）；sampled 仅 s11 获 4 目标（+551.8 B/scene，CI [+161.8,+1017.4]）、s33 成本
+84.2 B/scene（CI [0,+252.6]）、s22 不变；无跨 seed 一致成本下降；无任何丢目标。

**不可判 / 缺证据（登记，不补造）：**

- 联合行内**每目标**优势未保存，无法给出单个晚期目标的 SEND/SKIP 优势差。
- on-policy chosen-action 优势不等于 SEND/SKIP 因果价值；反事实动作价值需要模型前评或重放，本轮禁止，故"晚发是否更值"无法从现存数据得出。
- critic 是否"错"、错是否可修：value_loss 是回归损失尺度（batch0 约 15.4），不是错误率，不能
  据此判定 critic 失修；需要额外诊断，超出本轮只读范围。
- 精确 minibatch 与随机排列**不属于缺失**：已由冻结数据序 RNG 完整重建并与 gate_records 逐条
  核对一致。
- 晚头初始 0.10 缩放只保 argmax、不创造新信号方向；本审计不复核该构造（R68 已核验）。

## 6. 结论边界

这是对单一冻结策略族、单一已用开发块（1800000..1800031）、on-policy 数据的事后描述：能陈述
该实现中晚期自由动作的更新信号存在且以正为主、稀疏、不区分 SEND/SKIP、幅度衰减；不能证明消息
普遍必要或可安全跳过，不能给出总体非劣或泛化，不能据此删消息、换阈值或认定某一 loss 根因。
greedy 零变化不是晚头饱和的证明；sampled 的局部移动不是跨 seed 稳定收益。性能差不是技术失败。

## 7. 零副作用计数与 HEAD

环境构造 0；episode 0；claim 0；真实/合成 optimizer 更新 0；PPO 重放 0；模型 forward 0；
autograd 0；RNG 恢复 0；checkpoint 写入 0。读取：384 row、384 tape、96 batch 记录、6 门禁/
统计/链资格文件；输出：本报告、机器 JSON、执行依据存档、测试与运行日志。
HEAD：`420fa9af18932f735efdf6528ab68174c561ec52`（本轮提交后 advancing，见提交信息）。

## 8. 更正与措辞收窄（追加登记，不改写原文）

本节为追加，第 1-7 节原文一字未改。以下各项在机器 JSON 的 `corrections`（C1-C6）与
`withdrawn_inferences` 中逐条落库。

**C1 minibatch 总数更正。** 第 3 节的 "676 个 minibatch" 是对的；早期草稿曾给出
`minibatches_total = 501`，算法为 `128184 // 256 + 1`，隐含"每个 minibatch 都是满 256 行"，
但每个 epoch 的末块是残缺块，该式因此少算。正确值 = 逐批 `4 x ceil(samples / 256)` 之和 = 676，
与 96 个 batch 提交记录里 676 条 `gate_records`、`update.minibatches` 加总 676、
`update.optimizer_steps` 加总 676 完全一致（96/96 批三者相等，无例外）。501 仅作为
`superseded_erroneous_minibatches_total` 的更正证据保留，不参与任何结论。理论四 epoch 计划与
实际执行因此不是两套日程，而是同一套。

**C2 minibatch 成员证明强度下调。** 冻结数据序 shuffle 可由源码导出，逐 minibatch 的
`n_rows` / `early_free` / `late_free` 与记录零不符；但每个 minibatch 的**逐条成员行下标从未
落盘**。所以这是一致性交叉核验，不是"精确成员已被独立证实"。

**C3 撤回"反事实动作价值需要模型前评或重放"。** 第 5 节该条暗示补一次前向或 PPO 重放即可回答
"晚发是否更值"，这不成立：on-policy chosen-action 优势在任何允许的重算下都不是因果动作价值。
该因果问题在已存 on-policy 数据上**不可识别**，本轮允许的任何计算都不能改变这一点。

**C4 撤回"方向不区分 SEND/SKIP"作为推断。** 两类动作行的 chosen-action 优势都为正，推不出
期望策略梯度无法区分 SEND 与 SKIP——chosen-action 优势是"实际联合动作 vs critic 基线"之差，
从来不是两个动作之间的差。SEND/SKIP 分布（目标 423/96、行 417/96）继续作为**纯描述**保留，
不作为可分性证据。

**C5 撤回由梯度幅度差推出的裁剪/抑制读法。** `late_head` L1 均值 0.269 与 `task_value_head`
78.07、`cost_head` 3.73 的差距不能支撑任何裁剪或抑制推断：三个头参数量不同，原始 L1 幅度不可
跨头比较。仍然成立的只有：信号稀疏（1.60% 行、每 minibatch 约 3 行）、残差正偏（92.2% 正）、
96 批 recorded late_head 梯度均非零。这些事实**不被指定**为 R68 greedy 零变化的唯一根因。

**C6 暴露数与唯一数的口径（委托提醒，已满足并显式化）。** `update` / `gate_diag` 的
`early_free` / `late_free` 是 4 个 PPO epoch 的**累计暴露数**，不能除以 tape 行数当作行占比。
本审计全部行/目标数均由 tape 重建为**每批唯一**联合行或唯一自由候选，记录暴露仅以
`唯一 x 4 == 记录` 交叉核验（96/96 批一致；全轮 1767x4=7068、519x4=2076）。混合 early+late 的
同一行在行分母中只计一次（`rows_with_late_free` 与 `rows_early_only` 不相交，其和恰为
`free_rows` 1890）。首批抽查：s22 记录 64/28 对应唯一 16/7；s33 同法重建亦为 16/7，委托提醒中
"s33 为 16/6" 在盘上未复现，据实登记为 16/7；此差异不影响任何审计总数（总数取全 96 批重建）。

**一处读数澄清（非数字错误）。** 第 3 节 "平均每个 minibatch 约 3.04 行" 指全部 676 个
minibatch 的平均（2052/676 = 3.04）；若只数含晚期行的 550 个 minibatch，则为 2052/550 = 3.73。

第 6 节结论边界不变：这是单一冻结策略族、单一已用开发块（1800000..1800031）、on-policy 数据的
事后描述。greedy 零变化不是晚头饱和的证明；sampled 的局部移动不是跨 seed 稳定收益；性能差不是
技术失败。本节不新增、不改动任何科学读数，也不重跑任何门禁。

**本轮 HEAD**：本次更正pass提交于已核验的 R69 首提 `5d7319713d6c7eb69200df400d0c27838eaad2eb`
之上（该提交本身建于 R68 收口头 `420fa9af18932f735efdf6528ab68174c561ec52`）。更正产生的新
commit SHA 以提交信息与最终答复为准，本文件不预写哈希。
