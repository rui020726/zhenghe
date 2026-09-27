# Stage B 协议：有界真实接线验证（不训练）

日期：2026-09-27。执行依据：`specs/ROUND21_COMM_COST_STAGE_B_INTEGRATION.md`（第二十一轮指令）。
分支 `feat/comm-cost-stage-b`，起点 `5d0d71f4f779fb36982c468f34e7d801f1bc5de8`（= Stage A1 交付头，
本地逐字符核验），Stage B 工作在其上，**不回退任何历史**。

## 0. 本轮验证什么、不验证什么

验证"真实任务轨迹 → 实际费用 → rollout → 双 critic/组合优势/对偶统计"能否**一致接通**。
**不检验学习效果，不调策略，不训练**：0 optimizer.step、不加载旧通信模型恢复训练、
不改生产默认值、不打开封存集、不实施新的规则搜索。

B0_test / lambda_test / delta_test / eta_test **仅是按 spec 作的可重算接线算例**，
不批准它们为训练超参数或任务损失容忍度。不得从本轮结果声称学习收益或任务非劣成立。

## 1. 启动预算与记账

| 项 | 值 |
|---|---|
| 正常启动 | 18（六路径 × 三 seed） |
| 储备 | 6（仅技术失败后的复跑） |
| **硬上限** | **24**（所有环境启动、reset、技术失败、参考臂、smoke 全计数） |

记账实现：每进程一份 `START_LEDGER_<pid>.json`（`sb_common.LaunchLedger`），全局求和判上限；
达上限即拒绝启动并抛错。**纯数学单测不计启动**（本协议 §7 的合成测试只用合成
`SendObservation`/stub controller，不导入 driver、不构造 `JointRun`）。
失败产物保留不删除；复跑消耗储备；**绝不因任务表现差复跑**。

执行顺序：先 seed `1100000` 的六条路径，全部接线门禁（§5）通过后再跑 `1100001`/`1100002`。

## 2. 固定配置（单一集成测试配置，不扫描预算）

三个既有开发 seed：`1100000, 1100001, 1100002`（来自第十六轮 D1 已claim块 `1100000..1100059`，
本轮是**开发接线复跑，不是新独立测试集**）。

每个 seed 的配置**逐字段继承自该 seed 自己的 D1 A3 归档 run_config**
（`runs/comm_info_path_validation_20260923/logs/data_A3_s<seed>_rp.jsonl`，即 D1 normal-lossy
原配置：成功率 0.7、延迟 0、ACK 关、gossip 开、comm_range 0.3、num_goals 6、1000 步、
deadline_tightness (0.5, 2.0)、两个冻结 checkpoint、search_action_policy=checkpoint），
继承方式与第十八轮相同（`configuration_from_archive`，替换列表写入行文件）。

Stage B 覆盖（六路径一致，运行覆盖不改变生产默认值）：

| 键 | 值 | 说明 |
|---|---|---|
| `comm_timing_mode` | `policy_owned_timing` | 显式运行配置；传输层不代发 |
| `comm_max_wait_steps` / `comm_urgent_wait_steps` | `None` / `None` | 无公共强制发送兜底 |
| `comm_budget_bytes` | `29300` | 所有臂同一硬预算（task-pressure 轮锁定的 25% 档执行器口径） |
| `comm_sends_per_tick` | `None` | 与 A3 归档一致；本轮真实 mask 是**字节预算 mask**，容量 mask 无触发点（随容量口径留待 C） |
| `comm_ack` | `False` | 结构性零，缺字段不得填 0 |
| `comm_subscribers` | `all` | 全订阅者（6 台救援机器人） |
| `comm_delivery_probability` / `comm_delay_ticks` | `0.7` / `0` | D1 normal-lossy 原值 |
| `comm_range` / `gossip_interval` | `0.3` / `2` | 本地感知/gossip 开启（部署路径） |
| `steps` / `num_goals` | `1000` / `6` | 与归档一致 |
| `comm_policy` | 按路径（§3） | 学习路径标签为 `learned`，构造后注入真实策略 |
| `comm_transcript` / `output` | 本轮流式路径 | 元数据差异是 B1 预先列明的唯一允许差异 |

其他字段（device=cpu、quiet=True、mock=False、两个 checkpoint 路径、场景相关字段）
逐字段继承归档，**不凭记忆补齐**；继承替换列表与归档 run_config 原样落入每行的
`substituted_fields` / `archived_run_config`。seed 变化只改 `seed` 一个字段。

## 3. 六条运行路径（每条三个 seed）

| 路径 | 构造 | 用途 |
|---|---|---|
| `R_send` | 原生 `immediate`（`orders_selection=False`，传输层按 `transport_trim_key` 修剪顺序） | 参考规则 + 固定任务效果参考（g_hat 的 Y_ref） |
| `I_send` | `LearnedSendPolicy(decide_fn)`，`decide_fn` = 该配置 immediate 的**实际请求谓词**（`item.pending_event`）∩ 按 `transport_trim_key` 排序 | 新路径精确复现 R_send |
| `R_hold` | 原生 `always_hold` | 原生静默 |
| `I_hold` | `LearnedSendPolicy(decide_fn=∅)` | 新路径精确复现 R_hold |
| `P_init` | `StageBCollector`（`BudgetMaskedCollector` 子类）+ 固定初值 `StageBController` | 真实采样/mask/tape/费用任务 return |
| `P_repeat` | 与 P_init 同初值、同采样 seed、同配置 | 可复现性 |

关键纪律（spec §4）：
- **I_send 不是对所有候选盲目 SEND**：只复现 immediate 的 `pending_event` 谓词，
  顺序 = 传输层自己的 `transport_trim_key`（从 `comm_policy` 导入，不复制公式）。
- I_send/I_hold **只注入确定动作**，不做任何参数更新来逼近参考；它们是接线测试，不是学出来的策略。
- P_init 可以表现差或静默；**不能据此筛掉 seed 或作优劣结论**。
- 学习路径在 `build_joint_run` 后经 `relay.attach_policy(...)` 注入（公开接缝，
  参考 `comm_budget_action_mask_fix_20260922` 轮的注入方式）；注入前断言 relay 尚未 tick。

## 4. 接线常数与网络构造（全部仅是算例，非训练参数）

| 名 | 值 | 说明 |
|---|---|---|
| `B0_test` | 10 000 B | 归一化常数；逐集同时保存原始字节/任务数，换归一化无需重跑环境 |
| `lambda_test` | 1 | 组合优势 `λ·A_task − A_cost` 的 λ |
| `delta_test` | 0 | 约束统计 `g = Y_ref − Y_π − δ` 的 δ |
| `eta_test` | 1 | 对偶步长（量纲 归一化成本/任务数²），只算 hypothetical `λ_next` |
| `gamma` | 1.0 | 目标是**无折扣**窗口总量（A1 已登记 γ<1 ≠ 无折扣总量） |
| `gae_lambda` | 0.95 | GAE 自己的 λ，与对偶 λ 分名 |
| `ACTOR_INIT_SEED` | 20260927 | 预登记固定初始化 seed，不根据结果挑选 |
| `SAMPLING_SEED` | 20260928 | `build_joint_run` 之后、`run()` 之前重设全局 torch seed，使采样流可复现 |

网络构造（复用生产相关实现，不写第二套）：
- actor + **task critic** = pilot 轮的 `Controller(ACTOR_INIT_SEED)`
  （`runs/comm_budget_pilot_20260919/train_pilot.py`：19→64→64 Tanh 主干 + 动作头 + 价值头），
  task critic 取值 = 价值头在候选上的均值（pilot 口径）；
- **cost critic** = 同主干上的新增 `Linear(64,1)` 头，在 `Controller` 构造后按固定顺序创建
  （Stage A 登记："新增 cost critic 改变训练状态"，故参数哈希覆盖 trunk/action/value_task/value_cost 四项）；
- 采样器 = `BudgetMaskedCollector` 子类（`runs/comm_budget_action_mask_fix_20260922/budget_masked.py`：
  shadow gate 就是生产 `CommBudget`，masked 项零似然零熵，tape 记 features/actions/charges/
  shadow/log_prob/value）；`forced` 恒 False（policy_owned_timing 且无 bound ⇒ 无强制项）。
- **只推理，不更新权重**：无 optimizer 对象；逐臂运行前后参数摘要必须相同。

逐决策步奖励（接线定义，非因果归因）：
- `r_cost_t = −(该决策步捆绑区间内 relay 空气账本增量) / B0_test`；
- `r_task_t = 窗口内按时完成数，按完成发生步计入`（与 pilot 轮/压力轮同一 tap 口径：
  registry 快照 COMPLETED 且未 expired，预订到 `orchestrator.step_index + 1`）；
- 决策步 = relay 给出 ≥1 候选的步；相邻决策步之间的奖励**捆绑到下一个决策步**，
  保证 Σr_cost = −episode 总空气字节/B0_test、Σr_task = 窗口按时完成数（闭合由门禁验证）；
- 末个决策步记 `terminal=True`（1000 步配置视界即 episode 边界，**不悄悄额外运行**到未来 deadline）。
  `gamma=1` 保证逐决策步奖励流是**无折扣**的 episode 总量口径；`gae_lambda=0.95` 只加权 advantage
  估计（GAE 的 λ，与对偶 λ 分名）——**episode 闭合校验的是奖励和与执行器账本，不是 `returns[0]`**
  （`returns[0]` 在 `gae_lambda<1` 时是 GAE 混合量，仅 `gae_lambda=1` 时才等于无折扣总和，
  该区别有合成测试锁定）。

## 5. 门禁（逐条 passed/failed/incomplete）

**B1 原生/注入行为等价**（每 seed 两对）：R_send↔I_send、R_hold↔I_hold。
按实际消息/时刻/顺序/载荷/尝试计费/信道结果/任务结局核对：emission 序列
（step,target,revision,cost_bytes,subscribers,first）、refusal 序列、
deliveries_applied、spent_bytes、airtime_bytes、relay 全 counters、on_time/expired、
lifecycle 行为哈希。**允许差异只有预先列明的元数据**（artefact 路径、run_id/arm 标签）；
不得在读差异后扩大白名单。两条 hold 路径受控上行尝试与费用均为 0；ACK 关闭是结构性零
（`ack_attempts` 等必须为 0 且 `ack_enabled=false`，不是缺字段填 0）。

**B2 新 actor 的真实动作与 mask**（P_init）：
- logged `log_prob` 与相同参数从 tape features 重算的差 ≤ 1e-6（既有门槛，
  `runs/comm_budget_masked_retrain_20260922/train_retrain.py:57`）；
- 逐步闭合：requested SEND ⊆ (emitted ∪ refused)，顺序一致；masked 项 likelihood/entropy 零权重；
  forced 集为空；charge 等于 relay 自己的 `attempt_charge_bytes`；shadow gate 就是生产 `CommBudget`；
- 策略不应请求不可付的发送：若出现拒付，定位顺序/报价差异，**不得从账本删除**；
- 参考注入路径的规则 mask 与学习路径的环境可执行 mask 不混淆（I_* 无 mask 概念，P_* 才有）。

**B3 真实费用与任务 return**（每 seed 学习路径）：
- 每 episode：Σ 逐步成本 == `spent_bytes` == 转录 Σ cost_bytes == `airtime_bytes` == Σ 发射计费；
  归一化和 == 累计字节/B0_test；丢包照付、拒付不收费、费用只结算一次；
- Σ `r_task` == 同窗口生命周期按时完成数（`window_metrics(...).on_time`）；
- task/cost return、critic prediction、GAE、`λ_test·A_task − A_cost` 与**独立离线重算**
  （`sb_check.py` 只读落盘 artefact）一致；
- `g_hat = Y_ref − Y_π − delta_test`，Y_ref 取同 seed R_send（记录完整配置键，不跨配置复用）；
- 对偶更新只算 hypothetical `λ_next`，**不写回训练状态、不影响后续决策**；
  不宣称短轨迹已满足期望非劣约束。

**B4 重复性与权重冻结**：P_init/P_repeat 同 seed 参数配置 RNG 可复现——费用、tape
（features/actions/log_probs/逐步成本流）、任务结局一致；不同策略臂**不预先声称**共享未来
信道/目标随机实现，只对确定性等价路径要求逐位复现；输入观测 19 维且不含结局/接收端真值；
双 critic 与 actor 运行前后参数哈希不变；optimizer.step 计数 0。

**停止条件**：全部门禁通过、任一非技术门禁失败、或启动额度耗尽 ⇒ 停止并报告。
门禁失败不补跑"更好看"的配置。

## 6. 产物与命名

每 cell 产物含 seed/路径/attempt；签名含代码内容身份（harness 与依赖源文件 SHA-256）、
checkpoint 哈希、配置与 schema。`logs/`、`rows/`、模型权重**不入 Git**（.gitignore allow-list）。
本轮交付：`STAGE_B_PROTOCOL.md`（本件）、`START_LEDGER.json`、`CONFIG_AND_MODEL_MANIFEST.json`、
逐 seed 等价检查（`EQUIVALENCE_s<seed>.json`）、真实 tape/费用/任务闭合摘要
（`CLOSURE_SUMMARY.json`）、重复性结果（`REPEATABILITY.json`）、`STAGE_B_VERDICT.md`。
冻结运行依赖源文件后**不并发改代码**；纯报告修改不伪装成相同代码身份。

## 7. 零启动合成测试（启动前必跑）

`test_stage_b_math.py`：合成 `SendObservation` + stub controller 上验证
collector 的 log_prob/重算一致、mask 分类互斥闭合、shadow gate 与生产 `CommBudget` 同构、
GAE/捆绑奖励闭合、B0 归一化换算；**不导入 driver、不构造 `JointRun`、0 环境启动**。
Stage A 的 30 项合成测试同样重跑通过（只改过文案，无数值变化）。

## 8. 未执行 / 未授权

- 未训练、未更新任何权重、未加载旧通信 checkpoint 恢复训练；
- 未跑 `integrated_sar` 全量测试套件与 AGENTS.md 的 full smoke（46 项中 18 项导入 driver，
  无法预先确认为零 episode 启动；本轮只跑 §7 的自有零启动单测）——如实登记为未执行；
- C 阶段的 delta、rho、归一化选择、训练量与 700/1600 上限**仍未批准**；
- 封存集未读取；生产默认值未改；未实施新规则搜索。
