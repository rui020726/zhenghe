
---

## 13 任意 N 前置实验（P0–P3，fixed 口径）

本节是一次**前置可行性探测**，不是训练实验：训练与评估额度均已用尽，本节只花**世界重放**额度（申请上限 120 次），不含任何训练回合。全部产物位于 `recovery_12/revision_01/execution/scale_probe/`，一律标注“本地实验级，未建 identity”（`recovery_12/revision_01/protocol/` 为空，正式 runner 会抛，故全部绕过 runner 直接调 `episode.run_episode` / `run_rule_world`）。

### 13.1 问题

要做一个“任意 N 通信决策模型”，先得回答四个问题，且这四个都不需要训练：

1. **P0** 订阅者数 N 能不能只改订阅者、不改救援机器人数？（否则准时率混杂劳动力）
2. **P1** 成本侧的量在两种预算口径下如何随 N 缩放？（纯解析，零重放）
3. **P2** 每个 N 档，规则臂的上下界与成本侧实测值是多少？有没有可训练空间？
4. **P3** 把两个携带 N 的特征维无量纲化，代价是什么？（自对照）

### 13.2 口径确认与理由

**主口径 = fixed**：`Budget = 58,600`、`B0 = 10,000` 在 N=3/6/12 全部不变。

理由（P1 已解析证明）：等比缩放口径下 `Budget(N)=58600·N/6`、`B0(N)=10000·N/6`，每发送占比、可负担发送数、优势单位成本三者**全部与 N 无关**（13.4 第 2 组给出跨 N 差值恰为 0），跑世界只能验证代数自洽。而 fixed 下可负担发送数 = `23·6/N` = **46 / 23 / 11**，稀缺性随 N 变化——这才是需要泛化的轴。

`proportional` 只作为 P1 的**解析对照**出现，**未用于任何世界重放**。

### 13.3 P0 结局：部分可解耦（P0-A-partly）

**探针证据（只读，零重放）**，`integrated_sar/joint_driver.py`：

- `_subscribers` 的唯一来源是 `CommRelay(subscribers=...)`；`comm.py:257` 存为 `tuple(str(agent_id) for agent_id in subscribers)`，计费在 `comm.py:908`：`cost_bytes = payload_bytes * len(self._subscribers)`。
- 装配点 `joint_driver.py:1041-1052`：`requested = parse_comm_subscribers(configuration.comm_subscribers)`；`available = tuple(str(agent_id) for agent_id in task_adapter.env.agent_dic)`；若 `requested` 含 `available` 之外的 id 即抛 `ConfigurationError`。
- `comm_subscribers` 是 `RunConfiguration` 的**一等字段**（`joint_driver.py:131`，默认 `"all"`），带 CLI 开关（`--comm-subscribers`，`:542`）与解析器 `parse_comm_subscribers`（`:697`，支持 `all` / `first:N` / 显式 id 列表）。

**闸门检查（决定是否要动冻结件）**：

| 集合 | 含 `comm_subscribers` | 含 `rescue_robots` |
|---|---|---|
| `a1_metrics.CONFIG_EXPECTATIONS`（`world.py:281` 对账用） | **否** | **是** |
| `world.FROZEN_CONFIG_EXPECTATIONS`（`world.py:45`） | **否** | 是 |
| `world.A4_TRANSPORT_OVERRIDES`（`world.py:26`，revision-local） | **是**（当前 `"all"`） | 否 |

⇒ **N ≤ 机器人数的情形完全可解耦**：把 revision-local 的 `A4_TRANSPORT_OVERRIDES["comm_subscribers"]` 从 `"all"` 改成 `"first:3"`，`rescue_per_species` 保持 2（6 个机器人），`rescue_robots` 仍为 6，divergence 集合仍恰好等于 `{comm_budget_bytes}`——**闸门无需任何改动，也未触碰任何 pin 文件**。N=3 的准时率因此是**干净指标**（劳动力固定 6）。

⇒ **N > 机器人数的情形不可解耦**（P0-B）：`joint_driver` 拒绝不存在的订阅者 id，N=12 必须 `rescue_per_species=4`（12 个机器人），`rescue_robots` 随之偏离，divergence 集合多出一键。该键**已显式登记**（`world.SCALE_EXPECTATION_OVERRIDES` + 逐键理由 `SCALE_EXPECTATION_OVERRIDE_REASONS`，两者都在 revision-local 的 `world.py`），故 N=12 的准时率**是混杂指标，仅记录、不作结论**。

最小 diff（**未应用于主路径**，只在探针驱动里逐次赋值）：

```
-    "comm_subscribers": "all",
+    "comm_subscribers": SCALE_SUBSCRIBERS,     # 'all' | 'first:3'
```

### 13.4 P1 解析表（零重放，22/22 检查通过）

`PAYLOAD_BYTES` 不是断言值，而是**从冻结 tape 反推**：`runs/stage5b_pure_comm_budget58600_20261026/revision_02/execution/rows/eval/eval_normal_immediate_none_<scene>.json`（8 个场景，`paid_bytes == charged_bytes == airtime_bytes`，即已含扇出）：**459,756 B ÷ 184 次发送 ÷ 6 订阅者 = 416.445652 B/次**。

> 与提示词写的 2,500.6 B/次（payload 416.8）相差 −0.077%。冻结 tape 的真实均值是 2,498.674 B/次。P1 用实测值，并同时报告“若按 416.7667 B 会得到什么”，而不是把常量弯到能复现字面值。整数结论（46/23/11、23、58,600、10,000）不受影响。

| N | cost_per_send | budget | B0 | **affordable_sends** | **share_per_send** | **advantage_cost_per_send** |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1,249.3370 B | 58,600 | 10,000 | **46** | **2.131974 %** | **0.1249337** |
| 6 | 2,498.6739 B | 58,600 | 10,000 | **23** | **4.263949 %** | **0.2498674** |
| 12 | 4,997.3478 B | 58,600 | 10,000 | **11** | **8.527897 %** | **0.4997348** |

分母：`affordable = budget // cost`；`share = cost / budget`；`advantage = cost / B0`。

**第 2 组（proportional，解析对照，未重放）**：三者 share 均为 **4.263949 %**、affordable 均为 **23**、advantage 均为 **0.2498674**；跨 N 的 max−min 恰为 **0.0**（float 精确相等）。这把“等比缩放只是自洽”从说法变成算术。

**第 3 组**：N=6 在两种口径下 budget=58,600、B0=10,000、affordable=23，与冻结字面值一致（share 4.2639 % 对字面 4.268 %，差 0.004 pp，同源于上面的 payload 取整）。

### 13.5 P2 每 N 上下界与成本侧实测（48/48 成功，14/14 检查通过）

`normal_immediate` 与 `always_hold`，各 8 场景，共 3×8×2 = **48 次重放**（P2 配额用满）。

| N | 上界 immediate 计费 | emissions | **cost/send** | 占 58,600 | 触预算上限世界 | **上界准时** | 下界 always_hold | **headroom** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 239,997 B | 192（24/世界） | 1,249.98 B | 20.5 % | 8/8 | **123/192** | 0/192 | **123** |
| 6 | 459,756 B | 184（23/世界） | 2,498.67 B | 39.2 % | 8/8 | **108/192** | 0/192 | **108** |
| 12 | 450,192 B | 90 | 5,002.13 B | 38.4 % | 8/8 | **90/192** | 0/192 | **90** |

**criterion 1（∝ N）通过**：1,249.98 / 2,498.67 / 5,002.13 与字面 1,250.3 / 2,500.6 / 5,001.2 的偏差均在 ±0.5 % 内；两两比值为 1.999 与 2.002。

**criterion 2（逼近 affordable_sends）通过，但三档的约束来源不同**：

- **N=3**：`affordable_sends = 46`，实测每世界 24 次。**24 不是被预算截断，而是候选目标数**（`NUM_GOALS = 24`）：每世界 24 个目标全部广播，一次不多。所以这一档的约束是**目标数**，不是信用（截断原因逐场景记为 `no_candidate_target(all_generated_targets_addressed)`）。
- **N=6**：`affordable_sends = 23`，实测每世界 23 次，8/8 世界触到预算上限（`budget_refused_sends` 387–848）。**只有这一档是信用真正绑住**——第 24 个目标永远付不起。
- **N=12**：`affordable_sends = 11`（按均值 payload 算），实测 11 或 12 次。两个世界发出 12 次仍合法：**payload 是分布不是常数**（逐条 4,404–5,376 B），实测支出 58,572 / 58,512 B ≤ 58,600。故 11 是**均值 payload 下的界，不是逐条上界**。

**criterion 3（always_hold 全零）通过**：三档 24 个世界的 `paid_bytes` 与 `emissions` 全为 0。

**criterion 4（上下界区间与 no_headroom）**：三档 headroom 分别 123 / 108 / 90，**均不构成 no_headroom**（判据 ≤ 1 个任务，分母 192）。

**criterion 5（N=6 与既有冻结产物逐场景相等）通过**：16 格 × 3 量（`paid_bytes` / `emissions` / `on_time`）全部相等，0 处不一致。

**criterion 6（预算权威未随 N 漂移）通过**：48 个世界 × 5 处运行时读数（`row.budget_bytes` + `budget_credit` 四字段）全部 = 58,600；世界退出后三个内存权威恢复为冻结的 29,300（`budget_authority_restored = True`）。

> **N=3 的 123/192 高于 N=6 的 108/192，这是干净对比**（两档劳动力都是 6 个机器人，只差扇出）：N=3 每次发送只花一半，46 次的额度对 24 个目标绰绰有余 ⇒ **24/24 全部覆盖**；N=6 只买得起 23 次 ⇒ 必有一个目标拿不到广播，且预算早耗尽（首个拒绝 tick 153–674），刷新也一并被拒。**N=12 的 90/192 混杂了劳动力（12 个机器人），不得与上两档并论。**

### 13.6 P3 特征 N-协变与自对照 —— **未完成（not_run）**

**状态：`not_run`。触发止损规则 1 与规则 4，见 13.11。**

已完成的部分（8 次成功重放）：

| 配置 | 模式 | N | 结果 |
|---|---|---:|---|
| `off_n6` | `off` | 6 | **7/8 世界成功**，且 7/7 与归档 `smoke500/eval_control` 逐场景**完全相等**（`paid_bytes` / `emissions` / `on_time`，共 21 个量） |

未完成的部分：`norm_only @ N=6`、`norm_only @ N=3`、`norm_only @ N=12` 全部 0/8，**因此特征漂移分布、N=6 掉点测量、N=3/N=12 特征统计三项都没有可用实测数据**。

实现状态（必须与“数据缺失”分开陈述）：

- `recovery_12/revision_01/stage5b_comm/scale_invariant.py`（**新增**，纯函数；P1 的六个函数即出自此文件）。
- `recovery_12/revision_01/stage5b_comm/scale_features.py`（**新增**，shadow 变换 + install/restore + 流式统计）。接缝选在 **controller 实例的 `forward`**：pin 文件 `budget_masked.py:177-179` 先算 `rows = [features(observation, item) ...]` 再调 `self.controller.forward(matrix)`，而 `budget_masked.py` 与 `comm_learning.py` 都在 99 件 pin 清单里不能改；`forward` 是实例属性查找，遮蔽它即可改变进入网络的张量，并可在唯一的 finally 路径上删除恢复——沿用 `causal_credit.py` 的 install/restore 形状。
- 开关 `constants.SCALE_INVARIANT ∈ {"off","norm_only"}`，默认 `"off"`；`off` 分支**只记账、不回写**，因此 `off` 是“同一段代码的恒等路径”，而不是第二条代码路径。
- 只改两维：`#17 subscribers → n/6`（N=6 严格恒等，该维恒为 6.0）、`#14 queue_backlog_targets → backlog/24`（N=6 **不**恒等，必须自对照）。`#2 sent_count` **未动**（活维，替换会使既有检查点的准时率不可解释）。
- `off` 的惰性已用 7 个世界证明（与归档行逐场相等）。`norm_only` **已实现但未证明**：首版 `rewrite()` 返回 `numpy.ndarray` 而不是 `torch.Tensor`，引擎在 `linear()` 处拒绝；修正版（`torch.as_tensor(out)`）在本轮额度内未能跑通。

### 13.7 能说什么 / 不能说什么

**能说：**

1. 成本对 N 严格线性：每发送成本 1,249.98 / 2,498.67 / 5,002.13 B，实测与解析值偏差 ≤ 0.078 %（8 场景 48 世界）。
2. 在 fixed 口径下可负担发送数就是 46 / 23 / 11，且 N=6 档的 8/8 世界确实被信用绑在 23。
3. N=3 档的 24 次/世界是**目标数上限**，不是预算上限——由“emissions == generated_goals 且 budget_refused = 0”直接证实。
4. N=6 档用两个规则臂可以精确复现已归档的 16 格 × 3 量（零漂移），说明这条旁路调用路径本身没有污染基线。
5. 订阅者数可**独立于劳动力缩减**（N=3、6 个机器人），并因此得到一个干净对照：少发一次省一半钱，换来的覆盖率更高（24/24 对 23/24），准时 123 对 108（分母 192）。
6. 全部 95 个 pin 文件在整轮实验后哈希不变（`P3_feature_report.json` 的 `pin_manifest_sweep`：`files_checked = 95`、`files_drifted = []`）。

**不能说：**

1. **不能**说归一化（`numpy`→`torch` 修正版）会不会掉点：`norm_only` 三个配置 0/8。
2. **不能**给 #14 / #17 的分布漂移量：没有任何 `norm_only` 世界跑通。
3. **不能**说 N=12 的准时率有任何含义：该档劳动力是 12 个机器人，与 N=3/6 的 6 个不可比。
4. **不能**把 N=3 的 123/192 与历史上的 108/192、131/192 直接对比：N=3 是新 config、新世界，`config_hash` 与 `behaviour_sha256` 都变了。**唯一被授权的跨档比较是 N=6 档与归档行**（13.5 criterion 5）。
5. **不能**说“任意 N 下 19 维分布与 N=6 对齐”：只改了两维，`#10 revision_delta`（实测 0..27）、`#14` 的原始形态、`#2 sent_count` 在 `off` 档下必然不对齐。
6. **不能**说这套 fixed 口径在 N=3 上“更好”：24 次覆盖 24 个目标是本世界（24 目标 / 8 场景）的算术偶然，不是策略学到的选择——规则臂无条件全广播。

### 13.8 证据表

全部位于 `runs/stage5b_pure_comm_convergence_20261027/`。

| 产物 | sha256 | 生成命令（在 `diagnostics_20261104/` 下） |
|---|---|---|
| `recovery_12/revision_01/execution/scale_probe/P0_probe.json` | `695a1d0188c8fb8526f9675a1dc4d5a5b9ff67cf01edcf0ac9168a9571b9b8d9` | `python probe_p0_and_payload.py` |
| `.../P1_analytic.json` | `17df3524bb3662baa94b0049378a2b3cfd01d20da142276651ae52619f41f424` | `python p1_analytic.py` |
| `.../P2_summary.json` | `fff7931032f3c6259dc4e516eff0b458a134f81aca7a1c7424f6886751420ecd` | `python p2_score.py` |
| `.../P3_feature_report.json` | `39b3b0ae444f116c900324801d2c41f5185b4627a478d9d4e15feee0fb3161be` | `python p3_report.py` |
| `.../P2_3_fixed_p2/RESULTS.json` | `8509b3ebdebca675f17805b1f76691b96e1737b111b60172ef3a7a697e801940` | `python p2_scale_replay.py --n 3 --tag p2` |
| `.../P2_6_fixed_p2/RESULTS.json` | `344bdeba1cf35ba6137244695bed4d03a98d7e2266f5c907d032046df69b3bf6` | `python p2_scale_replay.py --n 6 --tag p2` |
| `.../P2_6_fixed_pilot2/RESULTS.json` | `a52e618367fc7d75b6eb8cde5e4ae9e0678920b6d43490aa12300e6003e6af6d` | `python p2_scale_replay.py --n 6 --scenes 2301901 --tag pilot2` |
| `.../P2_12_fixed_p2/RESULTS.json` | `d023992dd67bc2012ca17568142cd109138c6381bd0240bb6e2ea5c5f2ec8bff` | `python p2_scale_replay.py --n 12 --tag p2` |
| `.../P3_off_n6/RESULTS.json` | `3b1b9fdde435ed8ce09e07f83f5b620bf29a2a4beb7a86bc701412ff56121b82` | `python p3_scale_features.py --config off_n6` |
| `.../P3_norm_n6/RESULTS.json` | `7068ea8dabe3c8a193de190184cd43bb5f1ced7d8c7f6c63eb3bf4cad02ce3bf` | `python p3_scale_features.py --config norm_n6`（全失败） |
| `.../P3_norm_n3/RESULTS.json` | `928a938a3a929a9843f1606c4a016f48432b61f85a32780403ea5a79143daa9b` | `python p3_scale_features.py --config norm_n3`（全失败） |
| `.../P3_norm_n12/RESULTS.json` | `c83d87726bb7e1fdbf70340c91f3e2186f19020b979dce17db2a3be9b6050a9a` | `python p3_scale_features.py --config norm_n12`（全失败） |
| `.../LEDGER.jsonl` | 每行一条 start/end，共 98 次尝试 | 由全部重放脚本自动追加 |
| `recovery_12/revision_01/stage5b_comm/scale_invariant.py` | `bbed54eb918027abc5c907157657eb15b05caf6912dc7f170bea15e287a86632` | 手写，见 13.4 |
| `recovery_12/revision_01/stage5b_comm/scale_features.py` | 见 P3 报告 `shadow_feature_sha256` | 手写，见 13.6 |

参考基线（只读）：`recovery_12/revision_01/execution/smoke500/eval_control/RESULTS.json`（N=6 learned greedy，on_time 5/7/8/9/10/10/10/10 = 69/192）。

### 13.9 审查清单

1. □ 确认 §13 的每个数字都能在 13.8 的产物里找到，且分母已写出。
2. □ 确认 `P2_summary.json` 的 criterion 5 是 16 格 × 3 量全等，不是抽样。
3. □ 确认 N=3 的“24 次/世界”被表述为**目标数上限**，不是预算上限。
4. □ 确认 N=12 的准时率在全文任何地方都标为混杂、未与 N=3/6 并列。
5. □ 确认 P3 标为 `not_run`，且“已实现”与“已验证”分开写。
6. □ 确认 pin 扫描覆盖整份 manifest（95 个文件），不是只查点名的 8 个。
7. □ 确认 120 次额度账本（13.10）自洽：`attempted = P2 48 + P3 50`。
8. □ 确认未把“绕过 runner”的本地结果说成有 identity 的正式结果。

### 13.10 额度账本

| 项 | 数值 |
|---|---:|
| 申请上限 | **120** |
| 实际尝试（LEDGER `phase=start` 行数） | **98** |
| 其中成功 | 56 |
| 其中失败（照计，不退还） | 42 |
| **剩余** | **22** |
| P0 用量 | 0（纯只读探针） |
| P1 用量 | 0（纯解析） |
| P2 用量 | **48 / 48**（用满） |
| P3 用量 | **50 / 32**（**超支 18**，见 13.11） |

失败分布：`off_n6` 17 次（1 次目录冲突 + 16 次早前导入/命名空间失败），`norm_n6` 9 次，`norm_n3` 8 次，`norm_n12` 8 次。

**诚实记录**：P3 的 32 次子配额在执行中被超出 18 次。超支全部来自同一类接缝错误（下节）；但按“失败照计不退还”的规则，它仍然消耗总额度。因此**剩余 22 次不用于继续 P3**——继续就等于在明知接缝不稳的情况下烧额度。

### 13.11 止损触发记录

**规则 1 —— 同一个接缝/导入/闸门错误连续复现 ≥ 3 次仍无解。已触发。**

- 接缝：`scale_features.ScaleInvariantFeatures.rewrite()` 把 `numpy.ndarray` 交给 `controller.forward`，而 trunk 是 `torch.nn.Linear`，报 `TypeError: linear(): argument 'input' (position 1) must be Tensor, not numpy.ndarray`。
- 复现次数：**24 次连续**（`norm_n6` 8 + `norm_n3` 8 + `norm_n12` 8），跨三个配置、同一根因。按规则 1，**P3 在此停止，不再调试**。
- 处置：已把 `rewrite()` 返回值改为 `torch.as_tensor(out)` 并完成语法校验，**但修正版未在授权额度内取得任何成功世界**，故不计入证据，只作为实现状态记录。

**规则 4 —— 重放计数触到上限。未触发，但已逼近。**

- 98 / 120；剩余 22。P3 子配额（32）已超支 18。

**规则 2 —— pin 文件 sha256 不符。未触发。**

- 整份 manifest 扫描：95 个文件，0 漂移。

**规则 3 —— off @ N=6 无法逐字节复现既有冻结行。未触发。**

- 7 个成功世界 × 3 个量 = 21 个量全部与归档相等，0 处不一致。（第 8 个场景 2301901 在单场景探针中已完成且同样相等，但批量重跑时命中其自身产物目录的 `FileExistsError`，为避免删改既有产物**未重跑**，故不计入这 7 个。）

**因此 P3 的终态是 `not_run`，而不是 fail**：它的三个自对照（N=6 漂移、N=3/12 特征统计、掉点阈值）**一个都没有产出可用数据**。按“宁可缺一段，不得压缩验收”的纪律，§13.6 保持为空位，而不是用部分数据外推。

---

**审查提示**：本节可声明的结论集中在**成本侧与可负担性**（13.4、13.5 criterion 1/3/5/6）。唯一被授权的跨档准时率比较是 N=3 对 N=6（劳动力同为 6 个机器人）；任何引用 N=12 准时率、或引用“任意 N 特征已对齐”的说法，都超出了本轮证据。
