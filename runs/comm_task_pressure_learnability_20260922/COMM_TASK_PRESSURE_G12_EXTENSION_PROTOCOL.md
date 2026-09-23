# 任务压力 g12 独立扩展闭环：冻结协议

日期：2026-09-23。分支 `analysis/comm-task-pressure-g12-extension`，起点
`analysis/comm-task-pressure-learnability@6b4452f5de88f7cfd0c17ce475de269eb13dd6b7`（该 HEAD 与本地
`refs/remotes/origin/analysis/comm-task-pressure-learnability` 记录的推送结果一致；开跑时的实际网络核对见 §1.3）。
被扩展的协议：`COMM_TASK_PRESSURE_LEARNABILITY_PROTOCOL.md`（先于该轮第一条 stream 提交，总纲 v3.3→v3.4）。

**本文在任何一条 g12 episode 运行之前提交并单独成 commit。本文之后，判据、阈值、九臂、seed 块、预算、
信息路径与统计口径不再修改；若必须修改，按 §7 终止本轮，而不是在同一个聚合里混入新口径。**

## 0. 本轮的性质：唯一问题的最后一层

原协议 §0 的问题是：在只提高目标数量、其余一切不变时，29 300 B 额度是否第一次变成**固定规则无法统一覆盖**
的约束（L2 非天花板、L3 同一步只能支付严格候选子集、L4 规则间存在可利用情境差）。原协议按
`6 → 8 → 10 → 12` 逐层执行，跑完了前三层：L0/L1 三层全过，L2 只在 10 目标进带，L3 三层都是 0/16 流，
L4 三层 CI 下界都是 0 ⇒ **无层通过 L0–L4**（`COMM_TASK_PRESSURE_LEARNABILITY_RESULTS.md` §一）。

第四层 12 目标**从未运行**，原因写在该结果 §六：该轮自己的两处缺陷（探针把无限额度转整数、录制登记展开
`GridPosition`）吃掉 20 次启动，按执行指令"失败计入上限"，640 次里只剩 144 次，而该层需要 160 次。

本轮唯一动作：**把那 16 次启动的缺口补上，把预登记的第四层跑完并判完**。本轮不新增任何自由参数，
不新增规则，不新增判据，不训练。

## 1. 起点核对与前置复算（已在写本文之前完成，逐条列出结果）

### 1.1 起点

| 项 | 要求 | 实测 |
|---|---|---|
| `analysis/comm-task-pressure-learnability` 远端 HEAD | `6b4452f5de88f7cfd0c17ce475de269eb13dd6b7` | 本地分支与 `origin/…` 远程跟踪 ref 均为该 SHA；远程跟踪 ref 的 reflog 为 `update by push`（上一轮推送的记录） |
| 新分支 | 从该提交创建 | `analysis/comm-task-pressure-g12-extension`，`git rev-parse HEAD = 6b4452f5de88…` |
| 必读 | 总纲 v3.4、PROJECT_STATUS、README、NEXT_ROUND、三层协议/结果/判定/账本/manifest/provenance | 已读，见 §1.2 的复算 |

**未做到的那一项必须如实写出来**：开跑时 `github.com:443` 不可达（`git fetch`/`git ls-remote` 连续失败：
`Failed to connect to github.com port 443`），因此**没有在线核对远端 HEAD**，只核对了本地对该分支的远程跟踪
ref（由上一次成功推送写入）。本文不把它说成"远端已核对"；交付时的推送与 SHA 核对见 §9。

### 1.2 前置复算（全部通过，才允许进入 §2）

| 复算项 | 协议口径 | 实测 |
|---|---|---|
| 每层规模 | 16 流 × 9 臂 = 144 格 | `rows_g{6,8,10}`：144 格 / 16 seed / 9 臂，seed 连续且等于预登记块 |
| 聚合未漂移 | 重算 == 盘上矩阵 | `pressure_metrics.layer_report(g)` 与 `RULE_MATRIX_g{6,8,10}.json` **逐字段全等**（三层各比一次） |
| L0 | 三层全过 | `GATES_g{6,8,10}.json` 的 `L0.passed = true`（144/144 格、0 缺格、0 超额、账本闭合、单一 episode 代码状态） |
| L1 | 三层全过 | `mean detected/placed = 0.96875 / 0.96094 / 0.975`，阈值 0.75 |
| L2 @10 | 0.900 落在闭带 [0.40, 0.90] | 见 §1.4 |
| L3 | 三层 0/16 | `best_fixed` 的 `competition_streams = 0`（6/8/10 三层），逐流争用点全 0 |
| 启动账本 | 只剩 144，g12 需 160 | `ledger_totals()`：`matrix_starts 496 / cap 640`、`gate_starts 10 / 12`、`failures 20` ⇒ 余 144，`16 + 16×9 = 160`，缺 16 |

### 1.3 六目标层的 `placed` 合法性与流身份

三层 `STREAM_MANIFEST_g*.json` 的 `illegal_streams` 均为 `{}`；48 条流的 `map_sha256` 逐流按 seed 复现；
`ROW_PROVENANCE_g*.json` 的 `single_episode_code_state = true`，episode 代码身份
**`c922ca100665609c`**（37 个 `integrated_sar/**` 生产模块 blob + `run_episode` 源码文本的哈希）。
本轮 §6 要求 g12 的冻结必须落在**同一个 episode 代码身份**上，否则 g12 与前三层不同源，比较即失效。

### 1.4 L2 边界读法（本轮必须预先说清的一处测量细节，不改任何阈值）

g10 最佳规则 `send_once` 的 `on_time/detected` 逐流均值：

- 16 个逐流比值 `(7,8,7,…)` 的**精确有理数均值 = 9/10**，即真实值恰为 0.90，不多不少；
- 朴素 `sum(values)/len(values)` 的浮点和是 `0.9000000000000001`（**求和顺序的浮点残差**，不是"值真的超出 0.90"）；
- 精确求和 `math.fsum(...)/16` = `0.9`；harness 的 `_mean()` 先 `round(…, 5)` 再返回，同样得 `0.9`；
- `judge_layers.l2()` 用**闭区间** `0.40 <= rate <= 0.90` 判带，因此 **L2@10 通过**。

由此固定两条，供 g12 沿用且预先登记：

1. 闭区间、原值判定，不因"接近边界"而放宽或收紧任何阈值；
2. 若 g12 的该比值落在任一边界 ±0.005 之内，结果必须同时给出**三种读法**（朴素和、`fsum`、精确有理数），
   并说明判定依据的是 harness 的那一个值；这样"卡在边界上"这件事是可核对的，而不是事后叙述。
   这条只增加报告内容，不改变 `L2_ON_TIME_BAND`、不改变比较式、不改变 `_mean` 的实现。

## 2. 协议地位：追加授权的范围，以及它不能改写什么

- **g12、seed 块 `1000300..1000315`、九臂规则库全部是原协议 §2/§3 的预登记内容**（原文写死这四层与这四块
  seed；§4 的 640 = 4 × (16 录制 + 16 × 9) 本来就把 12 目标层算在内）。本轮不是新假设、不是新轴、不是新参数。
- 原协议未跑 g12 的**唯一**原因是技术失败计入 640 上限后额度不足（原结果 §六）。
- 使用者本轮明确追加授权：**只增加 16 次数据启动，使矩阵上限 640 → 656**。
  实现方式：`pressure_common.MATRIX_LAUNCH_CAP` 保持 **640 不变**（原协议那个数仍是事实），新增
  `G12_EXTENSION_GRANT = 16` 与 `MATRIX_LAUNCH_CAP_WITH_GRANT = 656`，`LaunchLedger.reserve()` 只对
  `record`/`matrix` 两类使用 656，并在拒绝信息里同时报出两个数。496 次历史启动照旧计入，不重开、不豁免。
- **核验（gate）上限不提高**：仍为 12，已用 10。本轮**不跑任何新的核验 episode**（§5 明令"不允许增加验证
  episode"），归档一致性/探针透明性沿用已提交的 `AGREEMENT_GATE.json`（`agree = true`），并在结果里写明
  它是**继承的证据**（在前三层的冻结下产生），本轮的探针扩展只用零 episode 测试证明其透明性（§6.2）。
- **本轮结果属于"容量扩展 / 探索性闭环"**：不得倒推修改原三层的 confirmatory 判定。原三层的
  `GATES_g{6,8,10}.json`、`RULE_MATRIX_g*.json`、`PRESSURE_METRICS_g*.json`、`STREAM_MANIFEST_g*.json`、
  `ROW_PROVENANCE_g*.json`、`CODE_FREEZE.json`、`AGREEMENT_GATE.json`、
  `COMM_TASK_PRESSURE_LEARNABILITY_RESULTS.md/.json`、`COMM_TASK_PRESSURE_LEARNABILITY_MANIFEST.json`
  **一律不覆盖、不重写**；由 §6 的代码守卫保证（第四层存在时 `report_pressure.py` 拒绝重写；
  `report_g12.py` 只允许写 `COMM_TASK_PRESSURE_G12_EXTENSION_*` 与 `G12_*` 命名）。
- **无论结果如何，本轮结束后停止**：不追加 14/16 目标，不追加规则，不追加 seed，不训练。下一轮做什么由
  §8 的分支判定交回使用者。

## 3. 固定实验配置

只允许改变 **`num_goals = 12`**。其余逐项沿用原协议 §1 的冻结表，且由同一段代码核对：

| 项 | 值 | 核对方式 |
|---|---|---|
| 唯一自变量 | `num_goals = 12` | `assert_only_num_goals_changed(12)`（录制侧与回放侧同时改，其余键一个都不许动） |
| seed 块 | `1000300..1000315`（16 条全新流） | `pressure_common.LAYERS[12]`，与 `G12_SEED_COLLISION_AUDIT.json` 双向核对 |
| 字节额度 | 29 300 B/episode（锁定的 25 % 档） | `protocol_common.tier_bytes("25pct")` 现读，非硬编码 |
| 发送协议 / 信息路径 / 信道 / ACK | `policy_owned_timing` / isolated（gossip > horizon）/ 投递 1.0 无损、0 延迟 / ACK 关闭 | `run_episode()` 在第一步之前逐项读 relay 自身的声明并拒绝偏离 |
| 权重 | 原 PILOC `cur_v2rep05_s*` 与 CTDE `ctde_final_7_1.pth`；不加载任何通信学习 checkpoint | 录制配置常量 |
| 机器人数 / 地图 / horizon | 搜索 2、救援 3×2；60×60、密度 0.15、半径 5；1 000 步 | 冻结常量 + `steps == 1000` 断言 |
| deadline/工作量生成 | `deadline_tightness = (0.5, 2.0)`，`dt = 1.0` | 冻结常量 |
| 规则库 | 九臂（表见原协议 §3），参数从第五轮 `protocol_common` 臂表读出 | `arm_spec()`；`ARM_ORDER` 常量 |
| 统计 | 逐 seed 配对 percentile bootstrap，`B = 10 000`，`RNG = 20260922`，固定标签序，一个 (层,轴,基线) 只抽一次 | `paired_differences()` + `bootstrap_audit` |
| 生产默认值 | `comm_ack=False`、`comm_timing_mode=legacy_shared_bound`、`num_goals=6` | `production_defaults_ok()` |

**规模与启动预算**：16 次录制 + 16 流 × 9 臂 = **160 次数据启动 = 144 格**。
不允许增加验证 episode、不允许替换 seed、不允许删臂、不允许追加重跑。
若出现技术失败：**计入 160**，该格如实缺失（`coverage.missing_cells`），本轮停止并把缺格报告出来，
不补 seed、不重跑"更好看"的流。144 格不满 144 时，聚合按 `pressure_metrics.load()` 的规则
**拒绝出数**（部分矩阵不出聚合）。

**运行前重新审计这 16 个 seed**：`python seed_audit.py --goals 12 --out G12_SEED_COLLISION_AUDIT.json`，
四源（Git 路径/跟踪内容/`git log --all`；`runs/**/*.json` 整数 token；盘上 artefact 文件名与内容碰撞；
全部 `*.md` 的裸数与区间声明 + 与项目已承诺块的算术求交）。
**任何真实碰撞 ⇒ 停止，不自行换 seed**。审计的读法预先写明：本轮只审 12 目标块这 16 个 seed，
前三层已生成的 `1000000..1000215` 不在被审范围内（那是本项目自己已登记的流，改判它们不会让 g12 变干净），
因此本轮**不需要也不使用任何白名单**；对 `1000300..1000315` 而言，任何一次命中都是冲突。

## 4. 判定：完全沿用原 L0–L4，不新增、不放宽

本轮只判 12 目标这一层。门槛常量与原文一字不改，全部集中在 `judge_layers.py` 顶部：
`L1_MIN_DETECTED_RATIO = 0.75`、`L2_ON_TIME_BAND = (0.40, 0.90)`（闭区间）、
`L3 = 8/16 流争用点 且 4/16 流额度使用 ≥ 0.90`、`L4 = 0.25 个/episode + CI 下界 > 0 / ≥2 条规则各在 ≥3 流唯一获胜 /
≥1 对双向胜负 / ≥2 个 Pareto 点且效应差 ≥0.25 且字节差 ≥2 930 B`。

- **L3 的定义不动**：`budget_competition_points` 仍是"同一步 `visible ≥ 2` 且 `1 ≤ payable < visible`"，
  即**只有严格候选子集可支付**。"任何一次预算拒付"不是争用点，`joint_only_points`（每个都付得起、合计付不起）
  也不参加判据——把它们改称 budget competition 正是原协议 §7 明令排除的读法。
- L0 仍要求：144/144 格、九臂除三个允许键外逐字段一致、`placed` 合法、账本闭合、扣费 ≤ 额度、
  转录与 relay 对拒付次数一致、单一 episode 代码状态、无未恢复的技术失败、`steps == 1000`、全格带探针。
- L2/L3/L4 只读 `best_fixed`（七个 competitor 里 `on_time/ep` 均值最高，并列取字节更少，再并列按预登记行序）
  与逐流 hindsight envelope（七个 competitor 的逐流最大值，经验上界，不是可部署策略）。

## 5. 机制诊断（预先登记，只作解释，不参加任何门槛）

执行指令 §5 要求把"L3 为什么是/不是 0"拆成可核对的量。本轮为 g12 的每条格（144 格）在探针里加计以下量，
**全部是只读计数**，不进任何判据：

| 量 | 定义 | 用途 |
|---|---|---|
| `candidate_histogram` | 每 episode 里可见候选数为 **0 / 1 / ≥2** 的步数 | 候选并发是否足够 |
| `multi_price_min/max/range/mean_bytes` | **多候选步**上各候选 `attempt_charge_bytes` 的包价分布 | 包价是否过于接近 |
| `multi_all_payable_steps` | `visible ≥ 2` 且 `payable == visible`（且 `Σ charge ≤ remaining`） | 额度不 binding |
| `multi_strict_subset_steps` | `visible ≥ 2` 且 `1 ≤ payable < visible`（= L3 主口径） | 真正的逐步选择 |
| `multi_none_payable_steps` | `visible ≥ 2` 且 `payable == 0` | 直接跨入全拒付区 |
| `multi_joint_over_credit_steps` | `visible ≥ 2` 且 `payable == visible` 且 `Σ charge > remaining` | 跨步选择（不是逐步选择） |
| `first_credit_exhaustion_facts` | 第一次"有可见候选而一个都付不起"那一步的步号、候选数、各候选包价、其中首发/刷新各几个 | 首次额度耗尽发生在哪类消息 |

三条实现依据（写死，供复核）：`CommRelay.tick()` 每步无条件调用 `_emit_selected()`（`comm.py:443-446`），
后者在候选列表为空时**先返回、不调用 `decide`**（`comm.py:700-701`），因此
`0 候选步 = 1000 − probe.steps_offered`（`steps == 1000` 由 runner 断言）。
包价、可支付位与阻断原因全部取自 relay 自己为每个候选算好的
`attempt_charge_bytes / send_executable / send_block_reason`（`comm.py:742` 一侧），
探针不另算一套、不改候选顺序、不改规则返回值。

**L3 为 0 时的成因归类**（描述量，不设阈值、不参与判定）：并报三个占比 ——
`多候选步 / 总步`（并发不足）、`range / 29 300`（包价过近）、`multi_none_payable / 多候选步`（直接跨入全拒付区）；
其中数值上最占解释地位的那一个由这三个数自己指出，不由叙述指出。

**与前三层的可比性预先限定**：`candidate_histogram` 与 `multi_*` 四项需要逐候选的每步视图，
前三层的行里**没有**这些键（当时只存了 `steps_offered / visible_total / payable_total / max_visible /
competition_points / joint_only_points / credit_blocked_steps`）。因此：
0 候选步数与平均可见候选数**三层可比**（可从已存字段重算），
"全可支付 / 严格子集 / 全不可支付"的逐步分布**只有 g12 有**；
报告必须把这一条不对称写出来，不得用 g12 的分布去改写前三层已经判完的 L3。

## 6. 代码冻结与"改代码就终止"

### 6.1 第一条数据启动之前必须完成

1. 全部零 episode 测试：`python -m unittest test_pressure_harness`（含本轮新增的边界/诊断测试）、
   `python -m unittest discover -s integrated_sar/tests`、`python -m integrated_sar`；
2. `python pressure_common.py`（冻结自检）、`python seed_audit.py --goals 12 …`（§3）；
3. harness 改动全部提交，工作树对 HEAD 干净；
4. `python freeze_pressure.py --mode register --goals 12`（写 `G12_FREEZE.json`）与
   `G12_START_LEDGER.json`，固化**生产代码 blob、runner 源码哈希、配置哈希、执行图模块集**；并按 §6.3
   的**两条口径分别**核对 episode 代码身份与前三层相同——任一条不同 ⇒ 世界被换过，g12 不得与前三层
   同表比较，本轮停止；
5. 新建 `G12_START_LEDGER.json` 快照：登记本轮开始时的 496/640、10/12、20 次失败与 656 的有效上限。

### 6.2 第一条数据启动之后禁止

禁止修改：episode 世界代码（`integrated_sar/**`）、`pressure_env.py` 的 runner、`CandidateProbe`、
`judge_layers.py` 的判据、`pressure_metrics.py` 的口径。若发现必须改：
**立即终止本轮**，把已产出的 g12  artefact 原位隔离（不删除、不与修复后的产物合并聚合），
如实报告改了什么、废掉多少启动。理由与第十一轮 §10 相同：两个代码状态混进一个聚合，
就再没有人能说出哪个数是哪套代码生的。

探针的扩展（§5 的只读计数）在冻结前完成，并且必须满足：`decide()` 仍**原样返回**被包裹规则的返回对象
（`orders_selection` 规则给出的有序元组一旦被重排，被测的就是另一个 episode）；
新增测试要同时证明"加了计数器"与"没有改变 episode"两件事，并保留一条**会失败**的元测试
（故意让探针改返回类型/顺序时断言必须报错）。

### 6.3 登记冻结前发现的一处度量学问题（在任何一条 g12 episode 之前提交；不改任何判据）

§6.1 第 4 条要求核对 episode 代码身份。核对时发现 `pressure_matrix.episode_code_identity()`
有**两条口径不同、数值不同**的算法，二者都自洽但不可互换：

| 读法 | 用什么文本 | 前三层的值 |
|---|---|---|
| 按 commit（`freeze_history` / 报告里的 freeze 时间线） | `git show <commit>:runs/.../pressure_env.py` 里从 `def run_episode(` 切到下一个 `\ndef `（**含函数后面的注释块**） | `c922ca100665609c`（`run_episode` 源哈希 `bdaf5cf95aac7c95`） |
| 按工作树（`registered_freeze` 写进行戳的那个） | `inspect.getsource(run_episode)`，只到函数自身结束 | `ff0b4139b1fff700`（源哈希 `a6a00ec98030dbf1`） |

因此 §6.1 第 4 条的核对按**同一条口径**执行，两条都要核，且核的是"本轮与前三层相同"而不是某个具体字面值：

1. 工作树口径：本轮探针扩展之后必须仍等于前三层**行戳里实际记录**的 `ff0b4139b1fff700`
   （已核：本轮改动后工作树值 = `ff0b4139b1fff700`，与 `rows_g*_shard*.jsonl` 的 `code_freeze` 全等）；
2. 按 commit 口径：新冻结头算出的值必须仍等于前三层的 `c922ca100665609c`。

任一条不等 ⇒ 世界被换过，g12 不得与前三层同表比较，本轮停止。
**这条披露不改任何判据、任何阈值、任何一次启动**；它只防止一个真实的误读：把两个不同的哈希当成
"代码变了"。改的只有 §5 的只读计数器与 `__pycache__` 之外的 harness 文件，生产模块 blob 一个都没动。


## 7. 严格禁止

- 不加载、不续训任何旧模型；不执行 `optimizer.step`；不写 checkpoint。
- 不改 reward、不改预算档、不改 ACK、不改丢包率/投递概率、不改拓扑或订阅者数量、不恢复 H=20 spillover。
- 不触碰 `800000..800399`（生成与读取都不许；盘上 artefact 数必须仍为 **0**）；
  `900000..900199`、`700000..700071` 不进入本轮矩阵。
- 不修改生产默认值；不扩观测维度。
- 不提交 stream、逐 episode 原始日志、checkpoint；不触碰其他会话的未跟踪文件
  （`architecture/`、`runs/coordinator_baselines_20260917/dev_summary.json`、
  `runs/comm_multistep_stage1_20260922/_DOC_PATCH_NOTES.md`、`_fix_diag_json.py`、
  `runs/comm_send_actionability_20260922/rows_s*.json`）。
- `git add` 必须逐个点名，不使用 `-A` / `.`。
- 不让任何其他并行分支改动本轮的 runner、协议、总纲或状态文档。
- 不放宽定义、不排除样本、不换 seed 以制造结果。

## 8. 结果分支（判完即按此走，不自行加码）

| 实测 | 结论与下一步（本轮仍不训练） |
|---|---|
| **L0 或 L1 未过** | 该层数据无效：只报告实验无效与失效原因，停止。不解释成任何通信结论。 |
| **L2 过、L3 未过** | 与前三层同向：**正式关闭"仅增加任务数量即可建立逐步预算选择"这条路线**，不再往 14/16 目标加。下一轮建议只做**只读 action-geometry audit**（不训练）。 |
| **L3 过、L4 未过** | 真实选择空间出现了，但固定规则之间还没有稳定的状态依赖优势。下一轮建议做**发送动作价值归因**（不训练）。 |
| **L0–L4 全过** | 才允许提出**成本感知目标（cost-aware objective）协议设计**；本轮仍不训练。按原协议 §9 给出规则 Pareto 相邻点的 `lambda_break_even` 区间与逐流 bootstrap 支持，**但不据此挑 λ、不改 reward**。 |

四条共用：本轮**不自动执行**任何一条后续；无论落在哪一支，跑完即停止并等使用者。

## 9. 交付物

目录 `runs/comm_task_pressure_learnability_20260922/`（与前三层同目录、同 harness、独立文件名前缀）：

| 文件 | 内容 |
|---|---|
| `COMM_TASK_PRESSURE_G12_EXTENSION_PROTOCOL.md` | 本文（先于任何 episode 单独提交） |
| `G12_SEED_COLLISION_AUDIT.json` | 16 个 seed 的四源冲突审计（`clean` 才允许录制） |
| `G12_FREEZE.json` | 本轮执行图冻结：生产 blob、模块数、episode 代码身份 |
| `G12_START_LEDGER.json` | 起始账本快照 + 656 上限的授权来源 |
| `STREAM_MANIFEST_g12*.json` | 16 条新流的逐流 sha256 / placed / detected / 合法性 + fold |
| `rows/g12/rows_g12_shard*.jsonl` | 144 格行集（**不入库**，盘上可核对） |
| `RULE_MATRIX_g12.json`、`PRESSURE_METRICS_g12.json` | 九臂聚合与搜索侧压力（含 §5 诊断量） |
| `GATES_g12.json` | L0–L4 逐门判定与证据 |
| `ROW_PROVENANCE_g12.json` | 哪套代码状态生了哪些格 |
| `COMM_TASK_PRESSURE_G12_EXTENSION_RESULTS.md/.json` | 本轮实测、机制诊断、上限诚实性、结论范围 |
| `COMM_TASK_PRESSURE_G12_EXTENSION_VERDICT.json` | §8 分支判定与下一轮建议（含"仍不训练"） |
| `COMM_TASK_PRESSURE_G12_EXTENSION_MANIFEST.json` | 身份、账本、卫生、测试、不入库清单 |
| `LAUNCH_LEDGER_g12_*.json` / `LAUNCH_LEDGER_record_g12_*.json` | 逐进程启动账本（含失败） |

总纲、PROJECT_STATUS、README 在交付时更新，但必须同时保留：原三层结论、"本轮是独立扩展而非对原门禁的
事后改写"、以及 §8 的下一步分支判定。

交付顺序：①本文；②harness 改动与测试；③冻结登记；④录制与矩阵；⑤结果/manifest；⑥总纲与状态。
完整测试与 `python -m integrated_sar` 通过，提交、推送并核对远端 SHA 后停止。
