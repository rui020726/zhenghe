# Stage A 落实记录：成本目标、任务约束、对偶更新与双 critic

状态：**Stage A 已完成并通过**（0 环境启动、0 新流、0 optimizer.step、0 生产代码改动）。
分支 `feat/comm-cost-stage-a`，本轮起点 `efca2a50c8d06f6fa7062c204d94c98b137206a4`，
其上先离线导入了原 spec 文档提交 `72c4c3f71e07339bf6533fbd771aef3b2682103c`（新增两文件）。

## 一、授权边界的实际执行情况

| 项 | 结果 | 证据 |
|---|---|---|
| 环境启动 | **0** | 轮次目录内无任何 `.jsonl` / `.pth` / `.pt`；从未调用 driver |
| 新流 | **0** | 未生成任何检测流 |
| `optimizer.step` | **0** | 测试只用前向/反向与有限差分，无优化器 |
| 真实 checkpoint 加载 | **无** | 状态测试用合成随机权重（`{"w": [1.0]}` 这类 stub） |
| 封存集 | **未读取** | `800000..800399` / `900000..900199` 盘上 artefact 仍 0，未触碰 |
| 生产树 diff | **空** | `git diff efca2a5 -- integrated_sar task_allocation piloc` 无输出 |
| 默认值/生产路径 | **未改** | 同上；本轮零生产代码改动 |
| 仿真器导入 | **0** | 子进程导入审计：`cost_math`/`test_cost_math` 只带入 torch/numpy 与标准库 |

以上由 `c20_audit.py` 落盘为 `STAGE_A_TEST_RESULTS.json`，**不是**靠项目套件退出码推断。

## 二、§3.1 目标、符号与单位（已固定为测试基准）

```text
c = 实际已计量尝试字节 / B0        B0 为预锁定常数（建议 30 000 B，待批准）
g = E[Y_ref] - E[Y_policy] - delta Y 单位：按时任务/episode；>0 = 约束被违背
L_max = -E[c] - lambda * g,  lambda >= 0
lambda_next = project_nonnegative(lambda + eta * g_hat)
```

三条符号纪律，各有对应用例：
- **任务效果相同而字节更多 ⇒ 优化方向必须更差**（`test_same_effect_more_bytes_is_worse`）；
- **任务损失超过 delta ⇒ lambda 必须上升**（`test_dual_update_direction_projection_and_zero`）；
- **reference 是外部固定比较对象**，不参与 actor 求导；`Y_ref` 常数与 delta 只影响约束估计/
  乘子更新，不通过伪梯度改变 actor（`test_reference_has_no_gradient_and_policy_has_correct_sign`：
  断言 `y_reference.grad is None`）。

**两个 critic 各估什么、组合优势的符号**：
- task critic `V_r(s)`：任务回报的价值（按时任务/episode 口径，折扣与否显式登记）；
- cost critic `V_c(s)`：**已计量成本 to-go** 的价值（`c = C_measured/B0` 口径）；
- 组合优势 `A = λ · A_task − A_cost`：由 `L = E[c] + λg` 对策略参数取上升方向得到
  （`−dE[c]/dθ + λ·dE[Y_policy]/dθ`），逐决策奖励 `r = −c + λ·y`，优势算子线性 ⇒
  **λ 挂在任务项上、成本以负号进目标**；λ=0 时只剩成本偏好（`−A_cost`）。
  （`test_combined_advantage_is_lambda_task_minus_cost`）
  > **A1 修订（2026-09-27，见 §十）**：原写 `A = A_task − λ · A_cost`、
  > "λ=0 时恰好退回纯任务优势"是**实现错误**，已按推导更正；旧公式是对偶问题
  > （最大化任务 s.t. 成本预算）的接线，在 λ=0 与 λ=10 都给反方向。
- λ 的量纲 = **归一化成本 / 任务数**（Stage B spec §2 更正：不是倒数），使 `λ·A_task`
  与 `A_cost` 同量纲；本版**不做逐 batch 分量标准化**（标准化会等效改写
  λ 的单位，引入它必须作为独立变更重新登记）。若 Y 也归一化，则 delta 与 λ
  **必须同步换单位**——本实现不归一化 Y，只归一化 C。

**折扣与终止的区别**（`cost_math.gae` 的 docstring + 三个用例）：
- 真正终止 ⇒ 续接项为 0；时间截断 ⇒ 用 bootstrap 值继续；
- `γ=1` 且价值为 0 时 `returns[0]` 等于后续无折扣奖励和（有测试锁定）；
- `γ<1` **不等于** episode 无折扣总量（有测试锁定该差异），因此是否折扣必须显式登记，
  不许把折扣量悄悄当作目标里的"episode 总成本"。

## 三、§3.2 观测不变 ≠ checkpoint 架构不变

actor 仍是 `comm-send-state/1` 的 19 维、消息契约不变；但**新增 cost critic 改变了训练状态**。
因此在 `cost_math` 里显式登记：

```text
TRAINER_TAG    = "constrained-cost-v1"
LOSS_VERSION   = "constrained-lagrangian/1"
STATE_SCHEMA   = "constrained-cost-state/1"
NORM_SCHEMA    = "cost-normalization/1"
```

`new_cost_state()` 落 λ、λ 轨迹、双 critic、归一化常数、delta、γ；
`load_cost_state()` 对 schema/loss version **fail closed**：旧版不匹配直接抛错，
绝不静默 resume（`test_state_roundtrip_and_fail_closed_on_old_version`）。
旧 checkpoint 不得静默 resume 这条要求也由此落到代码里。

## 四、§3.3 不把强制全发误当作 λ=0 的策略性质

λ=0 的目标只惩罚成本，**不必然选择全发**。两个独立测试：

1. **传输接线（合成 stub）**：`StubExecutor` 注入强制 SEND，走同一报价与计费路径，
   计费等于同配置 `immediate` 的定义（每个发现一份首报，载荷 × 订阅者）
   ——`test_forced_send_billing_matches_immediate_on_stub`。
   **这只是接线测试**，Stage A 只在合成替身上做，不涉及真实传输。
2. **λ=0 的方向**（`test_lambda_zero_is_cost_minimising_not_all_send`）：
   只含成本项时，**上升方向 = 少花字节**；成本恒为 0 时梯度为 0 ⇒ **没有方向**，
   不被迫全发。测试**不断言** λ=0 的网络复现 `immediate`。

## 五、§3.4 拒付与成本账本分开

六个互斥桶：`hold_active` / `send_executed` / `send_dropped` /
`send_refused_budget` / `send_refused_capacity` / `send_suppressed_terminal`。
只有真正发射的两桶计费；拒付/抑制/主动 HOLD 计 0，`report_metric_row` 里
`counts_as_saving` 恒为 `False` ⇒ **不给拒付加虚假费用，也不把它记成省下的成本**。
ACK 两件事分开记：`ack_off_structural_zero`（未开启 ⇒ 结构性零）与
`ack_expected_but_missing`（应有而日志缺失），二者不得混记。
对应测试：`test_dropped_send_billed_refused_not_billed`、
`test_episode_cost_increments_sum_to_total`、
`test_ack_structural_zero_distinguished_from_missing`。

## 六、§4 合成测试清单的覆盖情况（实测 30 项，全部通过；A1 修订后计数，原交付为 27 项）

| 清单条目 | 用例 |
|---|---|
| 成本同/效果异、效果同/成本异，目标与约束符号正确 | `test_same_effect_more_bytes_is_worse`、`test_better_task_effect_beats_cheaper_cost_when_constraint_binds` |
| g>0/g<0/g=0、λ 非负投影、更新输入不误用参考臂失败数 | `test_constraint_statistic_signs`、`test_dual_update_direction_projection_and_zero`、`test_dual_update_input_is_paired_difference_not_reference_failures` |
| 配对 reference 无梯度、policy 的成本与任务优势有正确梯度、有限差分核对 | `test_reference_has_no_gradient_and_policy_has_correct_sign`、`test_finite_difference_matches_autograd` |
| actor masked likelihood/entropy 为零，全 mask 无 NaN、无伪更新 | `test_masked_items_carry_zero_likelihood_and_entropy_weight`、`test_all_mask_batch_has_no_actor_sample_and_no_nan`、`test_critic_samples_survive_masking` |
| dropped 收费、refused 不收费、ACK 结构性零与缺日志区分 | `test_dropped_send_billed_refused_not_billed`、`test_billed_buckets_are_only_real_attempts`、`test_ack_structural_zero_distinguished_from_missing` |
| episode 费用增量等于累计；终局任务计数闭合；bootstrap/termination | `test_episode_cost_increments_sum_to_total`、`test_terminal_task_counts_close`、`test_termination_truncation_and_bootstrap` |
| 固定归一化可重算、B0 单位转换、不含结局派生 actor 特征 | `test_fixed_normalisation_recomputable_and_b0_conversion`、`test_b0_must_be_positive_and_finite`、`test_actor_features_reject_outcome_derived_fields`、`test_actor_features_require_all_nineteen_fields` |
| 新训练状态序列化保存/恢复 λ、双 critic、版本；旧版 fail-closed | `test_state_roundtrip_and_fail_closed_on_old_version` |
| λ=0 成本方向正确、不强制复现 immediate | `test_lambda_zero_is_cost_minimising_not_all_send` |
| 强制 SEND 走同执行器/报价路径 ⇒ 同配置 immediate（stub） | `test_forced_send_billing_matches_immediate_on_stub` |
| 双 critic 各自的估计对象与组合优势符号 | `test_combined_advantage_is_lambda_task_minus_cost`（A1 重写）、`CounterexampleWiring` 三项反例（A1 新增） |
| γ<1 不等于无折扣总量 | `test_gamma_one_reproduces_undiscounted_total`、`test_gamma_below_one_is_not_the_undiscounted_total` |

`c20_audit.py` 另外核对：30 项由 unittest 自身计数（不是引用旧项目套件数字）；
未跑全量 smoke（本阶段未授权，已列入未执行清单）。

## 七、模块与真实训练的接缝（先写清楚，Stage B 才接线）

| 接缝 | 现在（Stage A） | Stage B 要做的 |
|---|---|---|
| 成本报价 | `cost_math.billed_bytes()` 是纯函数副本 | 由 relay 自己报价；纯函数只做单测，**不得成为第二套价格定义** |
| 逐决策归因 | `DecisionOutcome` 结构 + 桶分类 | collector 按既有 tape 元数据填同一结构，断言 Σ 决策成本 == `airtime_bytes` |
| `g_hat` | 由调用方喂入 `dual_step` | 训练循环里按 seed 配对计算，禁止按消息池化 |
| mask | `masked_log_likelihood()` | 复用既有顺序 shadow-budget mask；全 mask 步不产 actor 样本 |
| 状态 | `new_cost_state` / `load_cost_state` | 写盘/读盘与 trainer tag 一起进入运行身份 |

## 八、未执行 / 未授权（不伪称）

- 未做真实 rollout、未接线真实 relay、未加载 PILOC/CTDE checkpoint；
- 未跑 `integrated_sar` 全量测试套件与 AGENTS.md 要求的 full smoke（会启动 episode）；
- 未实现 `late_discovery_gate`（未获批准，且它是候选对照而非学习可行性证明）；
- 未启动 B/C，未申请 δ / 降本门槛 / 训练预算的批准（仍是建议值）。

## 九、结论

 Stage A 的四个设计问题都以**可测**的形式落到代码里并通过合成测试：符号与单位固定、
双 critic 与 λ 登记进训练状态、λ=0 不被误读为"全发"、拒付与账本分开。
实现是纯函数集合，**没有第二套环境/训练器**，也没有触碰生产路径。

## 十、A1 修订记录（2026-09-27，依据使用者 Stage A1 指令）

上一交付 `6812a66` 后，使用者要求查处 `L_max = −E[c] − λg` 与
`A = A_task − λ·A_cost` 的矛盾、补经实际 actor surrogate 的反例测试、
核对 GAE/成本归一化/分量标准化/对偶更新的一致性、并为 B 臂 291/69/290/0 补互斥结局字段。
完整记录见 **`STAGE_A1_ADVANTAGE_WIRING_FIX.md`**，要点：

1. **判定为实现错误（actor 接线）**，不是文字误写：模块的拉格朗日量、约束统计量、
   对偶更新与 N4 主线及原 spec §2/§4 全部一致，唯独 `actor_advantage` 实现了对偶问题
   （最大化任务 s.t. 成本预算）的优势公式；同文件两条用例对 λ=0 的语义互相否定。
2. **正确组合优势 `A = λ·A_task − A_cost`**（推导：`r = −c + λ·y`，优势算子线性）。
   反例（SEND 任务 1/成本 0.2，HOLD 均 0）经 `masked_log_likelihood` +
   `actor_advantage` 的真实路径验证：正确梯度 `(λ−0.2)p(1−p)`，λ=0 = −0.05（压 SEND）、
   λ=10 = +2.45（抬 SEND）；旧接线两点皆反向。测试 27 → 30 项。
3. **四项一致性核对通过**；分量标准化原为未登记缺口，现登记决策：本版不做，
   λ 固定量纲为 归一化成本/任务数，引入分量标准化须作独立变更。
4. **互斥结局字段**：placed 360 = on_time + completed_late + expired + unassigned
   + active + never_discovered（A：350/0/1/0/0/9；B：290/0/0/0/1/69）；
   两个"发现但未按时"目标分别标注 `discovered_expired`（A）与 `discovered_active`（B），
   并更正旧 notes 的"一个 completed_late"误写（实际无任何 completed_late 目标）。

边界不变：0 环境启动、0 新流、0 optimizer.step、0 生产代码改动；δ、ρ、B0、
训练量、700/1600 预算仍待批准；未启动 B/C。
