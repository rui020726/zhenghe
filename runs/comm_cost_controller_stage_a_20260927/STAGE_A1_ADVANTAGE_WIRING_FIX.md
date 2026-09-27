# Stage A1 修订记录：组合优势接线错误 + B/A 臂互斥结局字段

日期：2026-09-27。分支 `feat/comm-cost-stage-a`，起点 `efca2a50c8d06f6fa7062c204d94c98b137206a4`，
上一交付 `6812a6641e045f19f114c986e51108d19d0e6b5e`（工作区保留后续改动，未回退历史）。
边界不变：0 环境启动、0 新流、0 optimizer.step、不加载真实 checkpoint、不改生产路径/默认值、
不读取封存集；只改纯计算模块、合成测试与本目录文档。δ、降本门槛、B0、训练量、
700/1600 启动预算仍待批准。

## 一、矛盾的性质：**实现错误（actor 接线），不是文字误写**

指令要求判定 `L_max = −E[c] − λ·g` 与 `A = A_task − λ·A_cost` 的矛盾属于报告误写还是实现错误。
结论是**实现错误**，四条证据：

1. **同一测试文件自相矛盾**：`test_lambda_zero_is_cost_minimising_not_all_send`
   （λ=0 时目标只惩罚成本）与 `test_actor_advantage_sign_and_lambda_units`
   （λ=0 时组合优势恰好等于纯任务优势）对 λ=0 的语义互相否定，不可能同时正确。
   这不需要外部证据就能判定接线有问题。
2. **模块其余部分全部一致地实现 N4 主线**：`dual_objective`（`−E[c] − λg`）、
   `constraint_statistic`（`g = E[Y_ref] − E[Y_policy] − δ`）、`dual_step`
   （`max(0, λ + η·ĝ)`）、`test_reference_has_no_gradient_and_policy_has_correct_sign`
   里的 `loss = −c − λ·g`、以及模块顶部登记的 `L_max`，全部对应
   "最小化期望已计量成本、约束相对固定参考策略的按时任务损失 ≤ δ"。
   唯独 `actor_advantage` 实现了**对偶问题**（最大化任务 s.t. 成本 ≤ 预算）的优势公式。
3. **docstring 的理由是被移植的**："成本要变小，所以成本优势以负号进入"是
   max-task-s.t.-budget 形式的推理；在 N4 形式里进目标的是成本、经约束进目标的是任务，
   权重因此必须挂在任务项上。
4. **原 spec 支持登记的拉格朗日量**：离线导入并逐字符核对的
   `ROUND19_CONSTRAINED_COMM_CONTROLLER_DESIGN.md` §2/§4 写的是
   `minimize E[C_measured]` s.t. `E[Y(policy) − Y(reference)] ≥ −δ`、成本为目标、
   任务效果约束用独立统计量。与登记一致、被偏离的是优势接线。

## 二、按真实 advantage 定义的推导（非机械替换）

    L = E[c] + λ·g,   g = E[Y_ref] − E[Y_policy] − δ

最大化形式与其对策略参数的梯度（`Y_ref` 与 `delta` 是常数，不进 actor 求导）：

    L_max = −E[c] − λ·g
    grad L_max = −grad E[c] + λ·grad E[Y_policy]     ← 训练器应上升的方向
    loss_to_minimize = −L_max = E[c] + λ·g           ← 训练器的下降形式

**注意**：`grad L_max` 不能写成 `−dL_max/dθ`——那是 `L_max` 自身的下降方向，符号会反。
逐决策奖励即 `r = −c + λ·y`。优势算子对奖励线性、双 critic 各估一项
（GAE 同样线性：`values` 可加、递推系数固定），所以

    A = λ·A_task − A_cost

**λ 挂在任务项上**：任务经约束进目标（λ=0 表示当前约束松），成本直接以负号进目标。
λ 的量纲是 **归一化成本 / 任务数**（Stage B spec §2 固定，不是倒数），
`dual_step` 步长 `eta` 的量纲随之是 归一化成本 / 任务数²。

## 三、反例测试：经实际 actor surrogate，λ=0 与 λ=10 均验证方向

反例（单决策点，SEND/HOLD）：SEND 得任务 1、成本 0.2（归一化）；HOLD 均为 0。
精确 surrogate `J = Σ_a π(a)·r(a)`，`r = λ·y − c`：

    ∂J/∂z_SEND = (λ − 0.2)·p·(1−p)        ← 指令给出的正确梯度形式

| λ | 正确接线 (λ−0.2)p(1−p) | 方向 | 旧接线 (1−0.2λ)p(1−p) | 方向 |
|---|---|---|---|---|
| 0 | −0.05（p=σ(1.4)） | 压 SEND，纯成本目标 | +0.25 | **反向**（抬 SEND） |
| 10 | +2.45 | 抬 SEND，任务压过成本 | −0.25 | **反向**（压 SEND） |

旧接线在 λ=0 与 λ=10 两点给出的方向都与正确方向相反，与指令预告一致。

三个新用例（`test_cost_math.py`，`CounterexampleWiring` 类）：

| 用例 | 验什么 |
|---|---|
| `test_exact_gradient_matches_closed_form` | λ ∈ {0, 0.5, 1.5, 10} 四档，autograd 精确梯度 == (λ−0.2)p(1−p) |
| `test_directions_and_legacy_wiring_is_opposite` | λ=0 为负、λ=10 为正；旧接线两点皆反号 |
| `test_gradient_through_masked_actor_surrogate` | 同一反例穿过 `masked_log_likelihood` + `actor_advantage` 的真实 actor 路径：单动作样本损失梯度、对采样分布取期望后等于精确梯度；(λ−0.2)p(1−p) 复算一致 |

`test_actor_advantage_sign_and_lambda_units` 已按正确语义重写为
`test_combined_advantage_is_lambda_task_minus_cost`（λ=0 ⇒ 只剩成本偏好 −A_cost）。
测试计数 **27 → 30**（unittest 自身计数，子进程复核）。

## 四、四项一致性核对（GAE / 成本归一化 / 分量标准化 / 对偶更新）

| 项 | 结论 | 依据 |
|---|---|---|
| GAE | **一致** | 递推 `A_t = δ_t + γ·λ_GAE·(1−done)·A_{t+1}`；真正终止不 bootstrap、时间截断用 bootstrap；γ=1 复现无折扣总量、γ<1 不等于无折扣总量均有测试锁定；是否折扣由调用方显式登记 |
| 成本归一化 | **一致** | 固定**预锁定**常数 B0（运行身份，非每策略分母）；`normalise`/`to_bytes` 互逆且测试锁定；登记进训练状态（`cost-normalization/1`）；符合原 spec §2"登记固定归一化常数、评估同时报告原单位、不用每臂各自成功数作分母" |
| 分量标准化 | **原为未登记缺口，现已登记决策** | 代码从未实现、文档从未声称，修订前无自相矛盾；但两项单位不同（任务数 vs 归一化成本），若 Stage B 改为逐 batch 标准化两个分量，λ 的有效量纲被等效改写。**本版决策：不做分量标准化**，λ 以 归一化成本/任务数 为固定量纲进入组合；凡引入分量标准化必须作为独立变更重新登记（已写入 `actor_advantage` docstring） |
| 对偶更新 | **一致** | `λ_next = max(0, λ + η·ĝ)`，ĝ 是配对点估计（不是参考臂失败数、不是按消息池化）；方向与 `L = E[c] + λg` 的极小化匹配；`g ≤ 0` 与 spec 约束 `E[Y(policy)−Y(ref)] ≥ −δ` 等价 |

修正后的闭环：`L_max = −E[c] − λg`（最大化形式）→ `r = −c + λy` →
`A = λ·A_task − A_cost` → `λ_next = max(0, λ + η·ĝ)`，四处的符号现在同向。

## 五、B/A 臂互斥结局字段（原"291/69/290/0"无标签）

原 `SPEC_ITEM_VERIFICATION.json` 把 `on_time`（`completed` 的子集）与 `expired`
（划分成员）并列摆出，无标签、易重复计数。改为与 `lifecycle_metrics.py:125-139`
同口径的**互斥**划分，`SPEC_ITEM_VERIFICATION.json` 新增
`outcome_partition_of_placed` 与 `partition_sums_to_placed`：

    completed = on_time + completed_late
    discovered = completed + expired + unassigned + active
    placed     = discovered + never_discovered

| 档 | A 臂 | B 臂 |
|---|---|---|
| discovered_on_time | 350 | 290 |
| discovered_completed_late | 0 | 0 |
| discovered_expired | 1 | 0 |
| discovered_unassigned | 0 | 0 |
| discovered_active | 0 | 1 |
| never_discovered | 9 | 69 |
| 合计 = placed | 360 | 360 |

`detected − on_time = 1` 的两个目标现带 `partition_label`：

- A：`1200025/piloc-e1-target-5`，deadline 360.68 ≤ 窗口 1000 ⇒ **discovered_expired**；
- B：`1200010/piloc-e1-target-3`，deadline 1132.50 > 1000、窗口内已被分配、
  窗口末仍在进行 ⇒ **discovered_active**。

**同时更正旧 notes 的一处误写**：原写"一个 completed_late、一个 not_completed"与数据不符——
两臂 `completed − on_time` 均为 0，不存在任何 completed_late 目标；
两条记录 `completed_at` 都是 null（从未完成）。已在新 JSON 的 `laggard_notes` 登记更正。

## 六、本轮（A1）改动清单与未执行项

改动（仅纯计算模块、合成测试、文档）：

- `cost_math.py`：`actor_advantage` 接线与推导、模块符号表补一行；
  `test_cost_math.py`：重写 1 项、新增 3 项反例；
- `c20_spec_items.py`：互斥结局字段 + laggard `partition_label` + notes 更正；
- `c20_manifest.py`：交付件清单补两项、cherry-pick SHA 改为显式常量（不再用 `HEAD^`）；
- 本文档 + `STAGE_A_IMPLEMENTATION.md` §十修订记录 + `DECISION_SUMMARY.md` 公式更正；
- 重新生成 `STAGE_A_TEST_RESULTS.json`、`MANIFEST.json`（旧件隔离到 `rejected/`）。

未执行/未授权（不伪称）：未接线真实 relay/trainer、未加载任何 checkpoint、未跑全量 smoke、
未启动 B/C；δ、降本门槛 ρ、B0、训练量、启动上限 700/1600 仍是**待批准**建议值。
生产三棵树（`integrated_sar`/`task_allocation`/`piloc`）相对起点 `efca2a5` 仍**零 diff**。

## 七、Stage B 启动前复核（2026-09-27，0 环境启动）

`ROUND21_COMM_COST_STAGE_B_INTEGRATION.md` §2 要求先更正 λ 量纲与 L_max/loss 梯度符号文案
再进真实接线。本轮只改**文字**，无数值/接线变化，按 spec"若只是文字，登记更正，不另开全面审计"：

- **λ 量纲**：A1 时误写为"任务数 / 归一化成本"（与被否决的旧公式自洽）。对 `A = λ·A_task − A_cost`
  与 `L = E[c] + λg`，λ 的量纲必须是 **归一化成本 / 任务数**。已更正 `cost_math.py`
  （模块符号表 + `actor_advantage` docstring）、本文档 §二/§四、`STAGE_A_IMPLEMENTATION.md` §二。
- **梯度符号文案**：固定为 `grad L_max = −grad E[c] + λ·grad E[Y_policy]`、
  `loss_to_minimize = −L_max = E[c] + λ·g`，并显式警告**不要**写成 `−dL_max/dθ`
  （那是 `L_max` 的下降方向）。`test_cost_math.py` 里以 `loss = −(c)−λg` 命名最大化形式的
  混淆已改为 `l_max`。
- **步长/初始化数值**：`dual_step` 步长 `eta` 的量纲登记为 归一化成本/任务数²；
  Stage A 中出现的 `eta_lambda=0.01` 只是测试桩，无生产数值继承；
  Stage B 的 B0_test/lambda_test/delta_test 仅作接线算例（见 `STAGE_B_PROTOCOL.md`）。
- **A1 反例复跑**：0 环境启动下 30 项合成测试重跑全绿（含 `CounterexampleWiring` 三项），
  actor 接线无实质矛盾 ⇒ 不触发"停止本轮"条件。
