# 任务压力 learnability calibration：冻结协议

日期：2026-09-22。分支 `analysis/comm-task-pressure-learnability`，起点 `docs/communication-learnability-calibration@ddee56db2d7ff0de991520fae3bc7100319d10f8`（其父为 `exp/comm-budget-masked-retrain@7e0a4233127a19fa9bf49672bbcf37218b587fff`，两个远端 SHA 已在开跑前核对）。总纲 v3.3，执行指令 `COMM_TASK_PRESSURE_LEARNABILITY_NEXT_ROUND.md`。

**本文在任何一条新 stream 生成之前提交。本文之后，判据、规则库、seed 块、预算与信息路径均不再修改；若必须修改，停止并把旧产物隔离成新一轮，而不是在同一个聚合里混入新口径。**

## 0. 本轮要回答的唯一问题

在只提高目标数量、其余一切保持不变时，29 300 B 的额度是否第一次变成**固定规则无法统一覆盖**的约束：

1. 通信仍有价值，但离效果天花板还有距离；
2. 额度在同一步上迫使发送端从多个可见候选里**只挑得起一个严格子集**；
3. 不同固定规则在不同流上互有胜负，形成至少两个真实的成本—效果折中点。

这是**必要条件标定**。通过只允许设计下一轮的成本感知目标与训练协议；本轮 **0 训练、0 `optimizer.step`、0 checkpoint、0 模型加载**。

本轮不建立任何学习模型结论。历史上"学习附加价值未建立"的口径（总纲 §9）不因本轮结果而改变。

## 1. 唯一自变量与冻结表

| 项 | 值 | 来源 |
|---|---|---|
| **唯一自变量** | `num_goals ∈ {6, 8, 10, 12}` | 执行指令 §3 |
| 字节额度 | 29 300 B/episode（锁定的 25 % 档） | `protocol_common.tier_bytes("25pct")`，从标定文件读出 |
| 发送协议 | `policy_owned_timing` | 第五轮，逐字节沿用 |
| 信息路径 | isolated（`gossip_interval` 大于步数，视距刷新关闭） | 第一轮信息边界 |
| 信道 | 投递概率 1.0（无损），延迟 0 tick | 冻结 |
| ACK | 关闭 | 第六轮，本轮不打开 |
| 搜索/分配权重 | 原 PILOC `cur_v2rep05_s*` 与 CTDE `ctde_final_7_1.pth`，不加载任何通信学习 checkpoint | 冻结块 |
| 机器人数 | 搜索 2，救援 3 种 × 2 = 6 | 冻结块 |
| 地图/障碍/栅格 | 60×60，密度 0.15，检测半径 5，`cluster_radius = max(4, size//6)` | 冻结块 |
| deadline/工作量生成 | `deadline_tightness = (0.5, 2.0)`，`dt = 1.0`，1 000 步 horizon | 冻结块 |
| 环境修复 | 成员一致性 `True`、binding、释放源阶段检查沿用 C 臂 | 冻结块 |
| 观测/动作/奖励 | 不触碰（19 维 `comm-send-state/1` 只在旧轮次使用，本轮无模型） | 冻结 |
| 生产默认值 | `comm_ack=False`、`comm_timing_mode=legacy_shared_bound`、`num_goals=6` 不改 | `pressure_common.production_defaults_ok()` 逐字段核 |
| 规则参数 | 九臂参数全部从第五轮 `protocol_common` 的臂表读出，不重新调参 | `arm_spec()` |

`pressure_common.assert_only_num_goals_changed()` 对每一层机械核对：录制配置相对冻结录制配置**只有 `num_goals` 一个键可能不同**，且 `num_goals` 在录制侧与回放侧同时改为同一个值（否则 `assert_bundle_matches_configuration` 会拒绝）。分配侧只在录制中不使用的键（`use_coordinator` 等）单列为 `DECLARED_REPLAY_ONLY_KEYS` 声明，不假装相等——沿用第二至十轮的 C 臂口径。

**明确禁止**：H=20 spillover、增加丢包、修改 reward、切换预算档、扩维、打开 `800000..800399`、把 `900000..900199` 或 `700000..700071` 引入本轮矩阵。

## 2. 数据块与四源冲突审计

候选块（写死，见 `seed_audit.py` 的 `BLOCKS`）：

| num_goals | episode seeds | 数量 |
|---:|---|---:|
| 6 | `1000000..1000015` | 16 |
| 8 | `1000100..1000115` | 16 |
| 10 | `1000200..1000215` | 16 |
| 12 | `1000300..1000315` | 16 |

审计四源（artefact 优先于声明，因为已执行的 episode 无法"未看见"）：

1. **Git**：`git ls-files` 路径名、`git grep` 跟踪内容、`git log --all` 提交说明；
2. **manifest/登记**：`runs/**/*.json` 全部整数 token（跳过 >16 MB 者并如实登记跳过数）；
3. **盘上 artefact**：`runs/**` 全部文件名的 seed 模式（`s<seed>.json`、`*_s<seed>.jsonl`）；47 GB 的逐 episode JSONL **不打开**，因为它的名字就是它的 seed；同名不同字节算命名冲突；
4. **文档声明**：仓库内全部 `*.md`，既匹配裸数字也匹配 `A..B` / `range(A, B)` 区间（跨度 >500 000 者按字节/日期处理，不当 seed）；并与项目已承诺的块（`3000..30011`、`500000..500399`、`600000..600059`、`700000..700071`、`800000..800399`、`900000..900199`）做算术求交。

分类规则只有一处例外且必须显式：`COMM_TASK_PRESSURE_LEARNABILITY_NEXT_ROUND.md`（执行指令本身）列出这四个块，记为 `own_pre_registration`，不算冲突；该白名单是常量、写进输出、不因任何命中而扩充。

**审计结果（在录制任何 stream 之前）**：`SEED_COLLISION_AUDIT.json`，四源命中 `git=0, manifests=0, artefacts=0, documentation=0`，`clean=true`；封存块 `800000..800399` 盘上 artefact 为 0；与项目已拥有块求交为空。因此允许登记并录制，**不自动换号**；若任一源命中，本轮停止并提交协议修订。

每条 stream 只录制一次搜索事件，之后由九臂重放；录制失败、目标放置不足或未检测目标如实保留。录制器幂等（已存在则只报告不重写），仅 `process_killed` / `disk_full` / `checkpoint_load_failure` 三种预登记原因允许在同一 seed 上重录，旧文件移入 `stream_reruns/` 隔离而非删除。

## 3. 冻结规则库（九臂）

全部预算内臂共用同一个 29 300 B 执行器与同一逐 step 容量口径；参数不重新选择。

| # | 臂 | 策略名 | 参数 | 角色 |
|---:|---|---|---|---|
| 1 | `send_once` | `send_once` | 无 | competitor（第四/五轮锁定的主对照） |
| 2 | `event` | `event` | 无 | competitor |
| 3 | `event_priority` | `event_priority` | 无 | competitor（`orders_selection=True`） |
| 4 | `periodic_5` | `periodic` | `interval=5` | competitor |
| 5 | `periodic_20` | `periodic` | `interval=20` | competitor |
| 6 | `first_send_plus_bound_3_10` | 同名 | `max_wait_steps=10,urgent_wait_steps=3,urgency_time=25.0` | competitor |
| 7 | `bound_only_3_10` | 同名 | 同上 | competitor |
| 8 | `always_hold` | `always_hold` | 无 | control（通信地板，不参与"最佳规则"选择与 envelope） |
| 9 | `event_unbudgeted` | `event` | 无额度（`budget_bytes=None`） | ceiling_reference（不参与预算内竞争，不得称 oracle） |

`random_send` 不进入矩阵。看到结果后不得新增间隔、优先级权重或规则。

臂 6/7 沿用第五轮口径：relay 仍被声明 `max_wait_steps=10 / urgent=3`，但 `policy_owned_timing` 下传输层不代发（`transport_parameters()["bound_enforced_by_transport"]` 必须为 `False`，否则该 episode 记为技术失败）。

## 4. 执行方式与启动上限

逐层执行 `6 → 8 → 10 → 12`：每层 16 次录制 + 16 流 × 9 臂 = **160 次启动**；该层判完 L0–L4 后，**通过即停止，不生成更高层**；L2/L3/L4 未过才进入下一层；L0 或 L1 未过则整轮停止（不向更高压力偷偷前进）。四层全执行的矩阵启动为 **640**。

账本口径（写死，避免事后解释）：

- `LAUNCH_LEDGER*.json` 逐进程一份（并行写同一文件会互相覆盖计数），全局求和后判上限；
- kind ∈ {`record`, `matrix`} 计入 640 的数据启动上限，达上限即拒绝启动；
- kind ∈ {`gate`, `selftest`} 是 harness 自身的核验启动，另设 **12** 的上限并全部登记、在结果里报出实到数——640 是执行指令对矩阵的算术，把核验成本藏进别处比超出上限更糟。

任何失败（进程被杀、录制器无输出、断言拒绝）都计入实到启动，不补 seed、不换 seed、不重跑更"好看"的流。

## 5. 必须测量的量与定义

逐 episode 记录（每臂每流一行），单位在字段名里写明。

### 5.1 搜索与任务压力（流属性，与臂无关的部分单列）

- `placed`：scenario `num_goals_placed`；**合法性**要求 `placed == 该层 num_goals` 且各目标格互异、为自由格、不与搜索机器人起点重合。
- `detected`：录制流中出现过 ≥1 次检测事件的去重目标数（流属性，九臂共享同一次录制）。
- `registered`：回放中进入任务生命周期的目标数（timeline 条数）。
- `on_time` / `expired` / `unresolved`：沿用冻结定义 `final_progress ≥ 1.0 且 final_failure_reason != "deadline_expired"`；三者与 `registered` 必须逐臂闭合。
- `detection_time`：每目标 `detected_at` 的均值/中位/最大。
- `assignment_latency`：`detected_at` 到首次分配的均值（无分配者计入 `never_assigned`）。
- `deadline_slack`：`deadline - first_work`（未开工者记 `None` 并单列计数，不填 0）。
- `concurrent_open`：每步"已检测未终结"任务数的均值与 p95（并发开放任务压力）。
- `robot_utilisation`：每步被占用救援机器人数 / 6 的均值（从 timeline 的在途成员区间重建）。
- `capacity_infeasible`：`work_needed` 之和与 `6 × 1000` 执行容量之比，以及 `never_assigned` + `assigned_but_no_work` 数——用于把"分配/执行容量瓶颈"与"通信瓶颈"分开，不得用通信解释前者。

### 5.2 通信与预算

来自 relay 自身账本与转录，不重新推导：`transmission_attempts`、`spent_bytes`（**发送尝试计费字节**，非投递数）、`emissions`、`first_emissions`/`refresh_emissions`（closure：`first + refresh == emissions`）、`redundant_emissions`、`generated_items`、`coalesced_changes`、`deliveries_applied`、`delivery_duplicate`、`delivery_superseded`、`budget_refused_sends`、`capacity_refused_sends`、`emissions_suppressed_terminal`、belief age。

- `budget_used_fraction = spent_bytes / 29300`。
- `first_refusal_step`：转录里第一条 **预算** 拒付的步（`tight_common.refusal_detail` 口径）；`first_exhaustion_step`：探针里第一次"有可见候选但一个都付不起"的步。两者分列，不互相替代。
- 发送类型与时间分布：first/update/retry 计数与各臂发送步的分位数。

### 5.3 逐 step 候选探针与 `budget_competition_points`

探针是**规则的策略包装器**，只读 `SendObservation`（relay 自己算好的 `attempt_charge_bytes` / `send_executable` / `send_block_reason`，见 `comm.py:742`），把规则的返回值**原样返回**，不额外枚举、不改顺序。测点：

- `visible`：该步可见候选数；
- `payable`：该步开始时剩余额度下**单独**付得起的候选数（`send_executable` 为真）；
- `all_jointly_payable`：`sum(charge) ≤ remaining`。

定义（主口径）：

> **`budget_competition_points`** = 满足 `visible ≥ 2` 且 `1 ≤ payable < visible` 的步数——同一步、同一剩余额度下，只能支付可见候选的一个非空严格子集。

同一测点另外两列一并报告，但不参与判据：

- `joint_only_points`：`visible ≥ 2` 且 `payable == visible` 且 `sum(charge) > remaining`（每个都付得起、合计付不起 ⇒ 也必须选）；
- `payable_candidate_points`：`Σ payable`（每步付得起的候选数，第五/九轮"双动作可行点"的量级参照；与第九轮顺序 walk 的 181 口径不同，因为规则不必然发送它付得起的一切）。

`always_hold` 与 `event_unbudgeted` 的探针点只作参照：ceiling 臂无额度，其 `payable` 恒等于 `visible`，L3 不使用它。

**探针透明性（第六轮教训 6 的硬要求）**：必须有一条断言证明"包上探针的 episode 与不包探针的同臂 episode 逐转录、逐字节、逐任务结果相同"，并且有一条**会失败**的测试证明该断言有判别力（故意让探针改顺序/改返回类型时它必须报错）。

### 5.4 规则比较（每个目标数分别算）

- 每臂 `on_time/episode`、逐局率（`on_time/detected` 的流内均值）、`bytes/episode`，以及按 stream seed 配对的 10 000 次 percentile bootstrap CI；
- 三种东西**分开报**，不合并成一张"谁最好"的表：样本均值 Pareto、逐流支配计数、带统计支持的比较；
- **最佳预算内固定规则 `best_fixed`**：七个 competitor 中 `on_time/episode` 均值最高者；并列取字节更少；再并列按 §3 表中的行序（预登记字典序，不允许临时裁定）；
- **逐流 hindsight envelope**：每条流在七个 competitor 里取 `on_time` 最高、并列取字节更少的结果。它是**不具部署性的经验上界**，不是策略 oracle，也不是任何可执行策略；
- `envelope − best_fixed` 的逐流配对差；
- 每臂成为逐流**唯一赢家**/并列赢家的次数；至少一对规则是否双向胜负。

## 6. 统计口径

- 配对 bootstrap：`evaluate_pilot.paired`（复用，不重写），`B = 10 000`，`RNG = 20260922`，逐流配对、按 16 条流为抽样单位；
- **一个统计量只允许一次抽样**：每（层, 轴, 基线）只调用一次 `paired_differences`，下游一律切片；`(label -> {seed: value})` 映射里额外放 `best_fixed` 与 `envelope` 两个合成分组，标签遍历顺序固定为 `BOOTSTRAP_LABEL_ORDER`；结果 JSON 记 `bootstrap_calls`（每次调用的层/轴/基线/标签序），出现同一键二次抽样即视为缺陷；
- 均值相同而 CI 不同的历史事故（第八轮教训 15）由上面两条防住；聚合摘要另用一次"重算不变性"核对未漂移；
- 比值型口径（λ、每任务成本）在分母 ≤ 0 时记 `undefined` 并保留分子，不删样本（第八轮教训 18）；
- 跨压力层**不作差**：各层 seed 块不重叠，层间只作趋势描述；一切因果式表述只允许发生在层内配对上。

## 7. 压力层门禁 L0–L4（先于数据冻结）

每层按下列顺序机械判定；`通过` 仅指该层同时通过 L0–L4。

**L0 数据与接口可信**
16/16 流有完整 final record；`placed` 合法（§5.1）；九臂除 `comm_policy`/`comm_policy_params`/`comm_budget_bytes` 与臂标签外配置逐字段一致；`requested == executed`、policy 端预算/容量拒付为 0（ceiling 臂除外，它本就不该被拒付）；账本闭合（`first + refresh == emissions`，扣费 ≤ 额度，且 `spent ≤ 29300`）；探针透明性通过；转录/行集与 code-freeze 一致、无混合 commit 聚合；技术失败为 0。
**未过：整轮停止**，不向更高压力层前进。

**L1 不是搜索缺失造成的假压力**
所有 placed 合法，且 `mean_s(detected_s / placed_s) ≥ 0.75`（逐流比值后取均值），并报告最差流。
**未过：更高目标数不再运行**——压力已落在搜索覆盖而非通信调度。

**L2 通信确实有价值且未到天花板**
`best_fixed − always_hold` 的按时完成差 95 % CI 下界 > 0；**且** `best_fixed` 的逐流 `on_time/detected` 均值落在 **[0.40, 0.90]**（闭区间）。低于 0.40 判为系统容量/搜索地板，高于 0.90 判为效果天花板。

**L3 预算产生真实选择**
`best_fixed` 上 `budget_competition_points > 0` 的流数 **≥ 8/16**；**且** `best_fixed` 预算使用率 ≥ 0.90 的流数 **≥ 4/16**。仅有大量候选或大量拒付不算通过（对应 §5.3 的两列另报，不参与判据）。

**L4 固定规则之间存在可利用的情境差**（四条同时成立）
1. `envelope − best_fixed` 的 `on_time/episode` 差 ≥ **0.25** 且 95 % CI 下界 > 0；
2. 至少两条 competitor 各自在 **≥ 3/16** 条流上是**唯一**赢家；
3. 至少一对在层均值上互不支配（完成与字节两轴不同时劣）的 competitor 出现双向胜负：A 在 ≥3 流严格完成更多，B 也在 ≥3 流严格完成更多；
4. 样本均值 Pareto 至少两点，两点完成差 ≥ **0.25 个/episode** 且字节差 ≥ **2 930 B**（额度的 10 %）。

L4 只证明"固定规则不能统一覆盖所测情境"的必要信号，不证明神经策略能学到 envelope。

## 8. 通过 / 不通过之后

- **无任何层通过**：停止这条路线的现有场景扩展。结论口径限定为"**仅提高目标数未建立规则之外的可利用通信调度空间**"；不训练、不改 reward、不再试别的 goal 数、不打开封存块。若 `event_unbudgeted` 也差，优先归因到搜索/分配/执行容量，不包装成通信学习机会。
- **某层通过**：交付该层的规则前沿、经验 envelope、预算竞争测点，并按 §9 计算经验交换率；**只**允许据此设计下一轮成本感知目标与训练协议。仍不训练、不开放 `800000..800399`、不加 ACK/丢包/H=20。

## 9. 经验交换率（仅在通过层计算，不选 λ）

对该层预算内 Pareto 的相邻点 `(b, c)`（b 更省、c 更贵）：

```
lambda_break_even(b→c) = Δ on_time 个/episode / ( Δ bytes / 29 300 )
```

只报斜率与逐流 bootstrap 支持区间；分母 ≤ 0 记 `undefined`。本轮**不**据斜率挑一个"最有利于学习模型"的 λ，不修改 reward。下一轮协议最多预登记三个 λ（含 λ=0 的纯完成对照）或选择约束式目标。

同时做 sender-visible 特征审计（仅描述，不拟合分类器、不使用接收端 belief/真值）：按逐流赢家分组比较 deadline slack、backlog、阶段/进度变化数、剩余额度比例是否有支持差异。

## 10. 产物、身份与卫生

目录 `runs/comm_task_pressure_learnability_20260922/`：

| 文件 | 内容 |
|---|---|
| `pressure_common.py` | fixtures：层/块/九臂/额度/探针口径/bootstrap 常量/账本 |
| `seed_audit.py` → `SEED_COLLISION_AUDIT.json` | 四源冲突审计 |
| `pressure_env.py` | 本轮 episode runner + 候选探针 + 与归档 runner 的一致性证明 |
| `record_layer.py` | 逐层录制 + `STREAM_MANIFEST.json`（sha256、placed、detections） |
| `pressure_matrix.py` | 9 臂 × 16 流执行器（分片、幂等、逐片账本；原名 `rule_matrix.py`，因与第五轮同名模块冲突而改名，见 §11） |
| `pressure_metrics.py` | §5 全部量的聚合与逐流表 |
| `judge_layers.py` | L0–L4 机械判定 + bootstrap + Pareto/envelope/λ |
| `freeze_pressure.py` | code-freeze 登记与校验（执行图来自 `sys.modules`） |
| `report_pressure.py` | `RULE_MATRIX.json`、`PRESSURE_METRICS.json`、`COMM_TASK_PRESSURE_LEARNABILITY_RESULTS.md/.json`、`..._MANIFEST.json` |
| `test_pressure_harness.py` | 合成数据单元测试：判定边界、计数闭合、审计 fail-closed、探针透明 |

- 提交顺序按执行指令 §10：①冲突审计与协议；②harness/合成测试；③逐层 stream manifest；④矩阵与结果；⑤总纲/状态。
- **不提交** stream、逐 episode 原始日志、checkpoint、临时产物，也不触碰其他会话的未跟踪文件。
- 执行中若必须改 harness：停止、隔离旧产物（移入 `quarantine_<reason>/`，不删除）、登记新 code-freeze，重跑该层并在结果里写明旧产物为何作废；不得把两个 code-freeze 的产物混进同一个聚合。
- 交付前必须通过：`python -m unittest discover -s integrated_sar/tests`、`python -m integrated_sar`、本轮 `test_pressure_harness.py`。
## 11. 开跑前的协议修订（在任何一条新 stream 之前提交，逐条列明改了什么与为什么）

§0 声明本文冻结后不再修改，指的是 **判据、阈值、seed 块、规则库、口径**；下列六项是 harness 与测量仪器的修正，
其中没有任何一项放宽判据或改变任何一条阈值。

1. **`rule_matrix.py` 改名 `pressure_matrix.py`。** `pressure_common` 在导入各轮 fixtures 时，那些 fixture 会把自己
   的目录重新插到 `sys.path` 最前，于是 `import rule_matrix` 会静默解析到 **第五轮同名模块**——两套 harness 混进
   同一个聚合，而且不报错。这条由单元测试当场抓到（`AttributeError: module 'rule_matrix' has no attribute 'cells'`）。
   修正有两层：`pressure_common` 在所有继承导入之后把本轮目录重新插回首位，并把模块改名彻底消除同名。
2. **`budget_competition_points` 增加"原因分列"子计数。** 主口径与 §5.3 完全一致（`visible ≥ 2` 且 `1 ≤ payable < visible`），
   另外记录 `competition_points_credit_cause` / `competition_points_capacity_only`。理由：`send_executable` 由
   `CommBudget.why_blocked` 决定，它对"额度不够"和"本步容量已满"给同一个 False。本轮冻结的 C 臂配置
   **没有**逐 step 容量（`comm_sends_per_tick` 未设），因此两者在本轮恒等——但要证明恒等，不能靠假设。
   判据仍只读主口径；子计数不参与任何门槛。
3. **探针透明性改为"一次运行同时证两件事"。** 原写"包探针 vs 不包探针"；实际做法是：带探针的本轮 runner 必须
   逐字段复现第五轮**无探针**归档行。两条主张（runner 非漂移、探针无副作用）由同一批 episode 承担，
   因此不必再单独跑一遍"不包探针"的对照。核验启动实到 **10 / 12**（明细见 §12 末条），矩阵启动 0。
4. **启动账本的口径写明。** 640 = 4 × (16 录制 + 16 × 9 臂)，是执行指令对 **矩阵** 的算术；harness 自身的核验
   启动另记 `gate` 类并自设上限 12，全部登记、在结果里报出实到数。把核验成本藏进矩阵数、或干脆不记，都是错的。
5. **审计重跑一次，判读不变。** 单元测试把冲突检查喂到仓库外的临时路径上，暴露出渲染相对路径的
   `relative_to(ROOT)` 会抛 `ValueError`——一个"因为崩溃而没报出冲突"的检查等于没有检查。修好后重跑
   `seed_audit.py`：四源命中仍为 `git=0, manifests=0, artefacts=0, documentation=0`，`clean=true`，
   封存块仍空。候选块、白名单与阈值一字未改。

**§12 harness 核验结果（先于任何新 stream，写在下面，含一处对归档件的反向发现）**

`AGREEMENT_GATE.json`：本轮 runner（探针开启）在归档流 `s700048`（`num_goals=6`、isolated-lossless、29 300 B）上
对四臂逐字段复现第五轮归档行——`send_once` 6 个/15 054 B/6 发、`event_priority` 2 个/28 824 B/11 发/2 932 次预算拒付、
`first_send_plus_bound_3_10` 5 个/28 704 B/11 发/826 次拒付、`always_hold` 0 个/0 B/0 发。

- **皮对皮**：`send_once`、`event_priority`、`always_hold` 三支归档转录与本轮转录逐包
  （step, target, revision, cost_bytes, subscribers, first）完全相等，且覆盖到 `orders_selection=True` 的规则
  ——第六轮教训 6 的那一类缺陷正是被这条抓住的；探针因此被证明无副作用。`first_send_plus_bound_3_10`
  只在**行级**一致，其归档转录不可作为证据引用，原因见下条。
- `always_hold` 的 `belief_age`：本轮记 `None`（没有任何 robot-target 对被更新过），第五轮的渲染器写
  `... or 0.0`。同一空集，两种渲染；该情形被显式分类为 `explained_differences`，且要求两侧发射数都为 0，
  所以"说过话却没有 belief"仍然是不通过。
- **反向发现（不改本轮任何数字，只更正可引用性）**：`first_send_plus_bound_3_10` 的归档 *转录文件* 与归档 *行* 自不相洽——
  转录发送字节合计 **28 632 B**，行记 **28 704 B**；而 28 632/11 发/5 个恰好等于同目录另一趟
  **normal 信息路径** 诊断行（`rows_25pct_normal_1.0_diag_s*.jsonl`）的值。也就是说第五轮把不同并行维度
  （isolated / normal）写进了同一个 scratch 目录，转录按 `arm + seed` 命名 ⇒ 被后跑的那趟覆写。
  这正是第八轮教训 5 的复发，本轮按 §5.3/§10 用 `g<goals>` 分目录、行/转录/账本逐层隔离来避免。
  本轮的对照结论只依据**行级**一致性（四臂全等）与三支自洽转录的皮对皮一致性；该 bound 臂的归档转录不作为证据引用。
- gate 启动实到 **10 / 12**（`LAUNCH_LEDGER_agreement.json`）：四个核验格跑了多遍，因为本轮自己的比较器先
  抛 `KeyError`（转录拒付键名）、后又遇到 `belief` 渲染口径未定——每次中止时 episode 都已跑完，所以账本记
  `starts=10, failures=0`：没有一次以环境异常结束，但确有 6 次是同一格的重复成本（`send_once` ×4、
  `event_priority` ×2、`first_send_plus_bound_3_10` ×2、`always_hold` ×2）。如实计入，不当作"没花"。
  矩阵启动 **0**；封存块 `800000..800399` 盘上 artefact 仍为 **0**。

## 13. 复跑

cd runs/comm_task_pressure_learnability_20260922
python pressure_common.py                      # fixtures 自检：冻结项仍然冻结
python seed_audit.py                           # 四源冲突审计（clean 才允许录制）
python pressure_env.py                         # 归档一致性 + 探针透明（gate 启动）
python pressure_env.py explain                 # 发射序列差异的字段级解释（不花启动数）
python freeze_pressure.py --register
python -m unittest test_pressure_harness       # 50 项合成测试，0 episode
python record_layer.py --goals 6 --shards 4 --shard 0   # 2/3 同理
python record_layer.py --goals 6 --mode merge
python pressure_matrix.py --goals 6 --shards 4 --shard 0
python pressure_matrix.py --goals 6 --mode coverage
python pressure_metrics.py --goals 6
python judge_layers.py --goals 6
python report_pressure.py
cd ../.. && python -m unittest discover -s integrated_sar/tests && python -m integrated_sar
```

`python -m unittest test_pressure_harness`（50 项合成测试，0 episode）、归档一致性核验与套件实跑结果写在
`COMM_TASK_PRESSURE_LEARNABILITY_RESULTS.md`，不在本文预先填写。推送后核对远端 SHA，然后停止等待使用者。
