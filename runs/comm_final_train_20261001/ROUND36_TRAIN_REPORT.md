# ROUND36 报告：修复版冻结训练（最后一次有界训练）与交付文档并行准备

分支 `fix/comm-task-reward-event-timing`；开工头 `577344aab8a5c9bfcf75e7ffbf6f77e503dbc200`
（= ROUND35 交付头，逐字相符；ROUND33 基线 `c7d47d25…` 为其祖先，本地 `git merge-base` 核验）；
本轮 harness 在三次启动之前已落为一枚提交 `93b27059c914…`，训练路径文件在两枚头之下逐字节相同
（`ARTEFACT_HASH_REGISTRY.json` 的 `training_path_files_are_byte_identical_at_both_heads: true`）。
产物 `runs/comm_final_train_20261001/`；提示词原文与开工核验在 `EXECUTION_BASIS.md`。
**未使用** GitHub 快照，0 次网络尝试，未推送。判定一律以 ROUND34
`TASK_REWARD_TIMING_CONTRACT.md` 为准；本报告不宣称任何学习效果，所有效果性数字留空待第 4 天。

## 0. 计数摘要（全部由账本重算，见 `LEDGER_RECONCILIATION.json`）

```text
训练启动进程 3 / 授权 3；失败储备使用 0 / 授权 1（未触发）
episode 构造 384 / 授权 384（每 seed 128；ledger 起始计数 128/128/128，失败 0，不退还）
批更新 32 / seed（共 96 批）；优化器步 1284（464 + 356 + 464）；评估启动 0；sanity 前向 3
wall-clock 1606.26 / 1572.48 / 1581.44 s，上限 2270.28 / 2395.66 / 2281.95 s（占 70.8% / 65.6% / 69.3%）
版本字段齐全率 checkpoint 12/12、row 384/384；事件账本 2187 笔逐集闭合
旧产物改写 0；新数据流 0；超参/目标/成本路径改动 0；热启动旧 checkpoint 0；ACTIVE_ROWS 未启用
网络尝试 0、未推送、本分支无 upstream
环境：torch 2.9.0+cu128、numpy 2.2.6、Python 3.10.19（D:\project_env\python.exe），未建 venv、未装依赖
```

失败计数分两栏：**被测系统（真实训练路径与世界模型）正确性失败 0 处**；**本轮自有 harness 缺陷
2 处**，均已登记、定位并复现说明（`ROUND36_HARNESS_DEFECTS.json`：sanity 脚本误把一维特征向量
当作 tape 形状，在写产物之前抛错；V5 把 ROUND35 以平台分隔符登记的仓库相对路径拼错目录，造成
10 项"假警报"而非"假通过"）。两处都不影响任何训练产物或证据。

## 1. 任务 1：冻结核验（0 启动，`FROZEN_CONFIG_MANIFEST.json` + `ROUND36_PREFLIGHT.json`）

12 项门禁 G1-G12 全清。逐项与 ROUND32 冻结 Stage C 的比对结果：

| 项 | 本轮实际 | 与 ROUND32 归档 | 判定 |
|---|---|---|---|
| PPO 超参（clip / entropy / epochs / lr / grad-norm / minibatch / value-coef） | 0.2 / 0.01 / 4 / 3e-4 / 0.5 / 256 步 / 0.5 | manifest `ppo_hyperparameters` | 逐字段相等 |
| 目标与对偶（B0 / δ / λ₀ / η / γ / λ_GAE / 对齐门） | 10000 / 0 / 1 / 0.05 / 1 / 0.95 / 1e-6 | manifest `objective` 与 `v2_common` 引用 | 同一常量对象 |
| 训练量 | 128 集/seed、4 集/批、32 批/seed、1000 步/集、里程碑 0/64/128 | manifest `episodes_per_seed=128`、归档批记录 32 | 相等 |
| seed 集合 | (11, 22, 33)，启动前写入执行依据 | `v2_common.TRAIN_SEEDS` 与 manifest | 相等，未挑选 |
| 场景块与 128 集顺序 | 48 训练 + 16 开发；顺序 sha256 | manifest `data_order.order_digest_sha256` | 相同；开发块 16 个逐元素相同 |
| 世界与传输配置 | `stage_b_inherited_config` 逐字段 | manifest `inherited_behaviour_fields` | 3 个场景 × 40 个行为字段 0 差异 |
| 行过滤 | ALL_ROWS（`-surrogate.mean()` 原表达式），不引用 ACTIVE_ROWS 任何入口 | manifest `arms.ALL_ROWS` 与 `reduction_definition` | 相同，未启用另一臂 |
| 生产件与导入件身份 | 8 个 `integrated_sar/*` + 13 个跨轮导入件 | ROUND32 manifest `code_identity` | 逐字节相同；`integrated_sar/` 自基线 0 次提交 |
| 冻结搜索/分配权重 | 两个 `.pth` | ROUND32 manifest `checkpoints` | 哈希相同 |
| 训练版本字段 | `constrained-cost-v2/published-completion-event-at-relay-tick/2` + `published-completion-event-at-relay-tick/2` | ROUND34 `TRAINING_VERSION_MANIFEST.json` | 相符；payload 与 `training_state` 双写 |

**唯一干预的函数级证明（G7）**：与 ROUND32 冻结头 `df176bd…` 相比，训练路径里
`ppo_update`、`batch_samples`、`likelihood_alignment`、`dual_update`、`batch_g_hat`、
`_unassigned_prefix_totals`、`run_training_episode`、`action_seed`、`_assert_transport`、
`_booked`、`eval_action_seed` **逐字节相同**；发生变化的只有 `_bundle`（tick 0 保留为未归属
prefix）、`save_checkpoint` / `load_checkpoint` / `fresh_training_state`（两个版本字段与恢复门）
与新增的 `load_checkpoint_read_only` / `_validate_checkpoint_base` / `_restore_controller`，
运行侧只有 `_install_taps` / `_run_episode` / `run_eval`（采集器搬到 `ProgressEvent` 总线、
行内写版本字段与事件账本）。⇒ 不存在第二处改动。

**wall-clock 上限的构造（G11）**：ROUND32 未记录 per-launch 耗时，只逐集记录 `seconds`，
故上限 = 该 seed 归档 128 集 `seconds` 之和 × 1.5。批更新耗时不在该基线内，实测三次启动的
wall-clock 与逐集 rollout 之和之差为 83.71 s（s11）、74.40 s（s22）、81.67 s（s33）（各 32 批），
完全落在 1.5 倍余量（≥ 756 s）内。上限由父进程的
子进程超时**强制执行**（`r36_launch.py` 的 `timeout=ceiling`），不是事后声明。

**初值四源（G9）**：三个 seed 的 `fresh_controller(seed)` == ROUND32 归档批 0 更新前参数 ==
git 里 ROUND32 启动前配对件 == ROUND24 同名 ep0 里程碑经真实 loader 恢复，四源全等
（s11 `7eff0f40203fdae6…`、s22 `b74c10c85c148a4b…`、s33 `0ddebe7dc04019eb…`）；每次启动前还
断言 seed 目录内**没有任何 `.pt`**，所以"从初值训练、不热启动"是机器条件而非承诺。

**参照缓存（G8）**：ROUND24 的 64 份 immediate reference 只读覆盖顺序中全部 48 个唯一场景，
缺 0；本轮 reference 启动 0 次。

## 2. 协议条款的执行位置（ROUND35 残余风险转写）

| 条款 | 本轮怎么做 | 证据 |
|---|---|---|
| ① 场景块与 smoke 相同，变化即停 | 顺序 sha256 与归档比对，不同即门禁失败 | G4 |
| ② 每进程独立记账，报告从账本对账 | 每 seed 自己的 `START_LEDGER_p<pid>.json`（128 起始 / 128 结果，含失败照计）+ `LAUNCH_RECORD.json` + 批记录；报告数字全部由 `r36_reconcile.py` 重算 | `LEDGER_RECONCILIATION.json` |
| ③ 每个 milestone 与 freeze manifest 都带两个版本字段 | 真实 `save_checkpoint` 双写；逐件读回断言 12/12；每次启动写 per-launch `FREEZE_MANIFEST.json`（两个字段 + 继承 token + 当时代码身份） | `V1`、`seed*/SANITY.json`、`seed*/FREEZE_MANIFEST.json` |
| ④ 崩溃保留现场 + 失败分类后才可用储备 | 父进程把子进程 stdout/traceback 整段落盘、ledger 与最后 checkpoint 原地保留；`--reserve` 写独立目录且要求先有登记原因（`record_failure` 强制三选一分类） | 本轮未触发；机制在 `r36_launch.py` / `r36_common.py` |
| ⑤ 正确性缺陷即停 | V1-V6 任一 `passed=false` 即退出码 1 并停止；本轮 V5 首跑确实因 harness 缺陷失败，**当时未继续任何启动**（三次启动已全部完成，未新增） | `ROUND36_HARNESS_DEFECTS.json` |
| ⑥ 不得把行为差异说成约束满足或学习效果 | §6 只用"改变了什么/没改变什么"的描述口径；结果表留空；批 0 之外不与归档做优劣比较 | §6、`RESULTS_TABLE_TEMPLATE.md` 禁则 |
| 禁用 v2_recover 一类重放 | 本轮不 import、不调用 `v2_recover`；恢复只发生在 `train_seed` 自身的 rolling 恢复路径（同一次启动内，本轮未触发中断） | G10、`v2_recover_used: false` |

## 3. 任务 2：三次从初值训练

流程每次都相同：绑定本 seed 目录 → 断言目录内无任何 `.pt` → 取继承 freeze token 与只读参照缓存 →
调用真实 `v2_train.train_seed(seed, episodes=128, ...)` → 结束后 1 次最小 sanity 前向。

### 3.1 seed 11（启动 1）

```text
构造 128（ledger 128 起始，失败 0）；批 32；更新 32；优化器步 464
wall-clock 1606.26 s / 上限 2270.28 s；逐集 rollout 合计 1522.55 s（归档 1513.52 s，比值 1.006）
likelihood 对齐 32/32 通过；每批更新前后参数摘要都变化（32/32）
初值 7eff0f40203fdae6… → ep64 2ea3fa8e9487…（240 步 / λ 1.2625）→ 末批 553d02dd1de285a7…
λ 1.0 → 1.4999999999999993；g_hat 32 项中 0 项为负；样本合计 26164（归档 26156）
```

| 文件 | SHA256（截 16） | 状态 |
|---|---|---|
| `v2_train_s11_ep0.pt` | `6855823aaf185bc2` | 0 集 / 0 更新 / 0 步 / λ 1.0；参数摘要 = 登记初值 |
| `v2_train_s11_ep64.pt` | `69822c41df7018b1` | 64 集 / 16 更新 / 240 步 / λ 1.2625 |
| `v2_train_s11_ep128.pt` | `032c2b77f60f1519` | 128 集 / 32 更新 / 464 步 / λ 1.5 |
| `v2_train_s11.pt`（rolling） | `032c2b77f60f1519` | 与 ep128 字节相同（末批之后再无写入） |

### 3.2 seed 22（启动 2）

```text
构造 128（失败 0）；批 32；更新 32；优化器步 356
wall-clock 1572.48 s / 上限 2395.66 s；逐集 rollout 1498.08 s（归档 1597.11 s，比值 0.938）
初值 b74c10c85c148a4b… → 末批 6a84ab76e206a511…；λ 1.0 → 1.2625；g_hat 0 项为负
样本合计 18880（归档 17751）；步数 356（归档 340）
```

| 文件 | SHA256（截 16） | 状态 |
|---|---|---|
| `v2_train_s22_ep0.pt` | `42923125bc46db91` | 0 集 / 0 步 / λ 1.0 |
| `v2_train_s22_ep64.pt` | `37a9acd7631b8799` | 64 集 / 16 更新 / 224 步 |
| `v2_train_s22_ep128.pt` | `de47f3c90789a8bd` | 128 集 / 32 更新 / 356 步 / λ 1.2625 |
| `v2_train_s22.pt`（rolling） | `de47f3c90789a8bd` | 与 ep128 字节相同 |

### 3.3 seed 33（启动 3）

```text
构造 128（失败 0）；批 32；更新 32；优化器步 464
wall-clock 1581.44 s / 上限 2281.95 s；逐集 rollout 1499.77 s（归档 1521.30 s，比值 0.986）
初值 0ddebe7dc04019eb… → ep64（240 步 / λ 1.25）→ 末批 cd0498693f7636ae…
λ 1.0 → 1.475；g_hat 0 项为负；样本合计 26032（归档 26010）
```

| 文件 | SHA256（截 16） | 状态 |
|---|---|---|
| `v2_train_s33_ep0.pt` | `4333c494ae5d2445` | 0 集 / 0 步 / λ 1.0 |
| `v2_train_s33_ep64.pt` | `6e5f39086a6dd8e8` | 64 集 / 16 更新 / 240 步 / λ 1.25 |
| `v2_train_s33_ep128.pt` | `98bb6e74e25fa7de` | 128 集 / 32 更新 / 464 步 / λ 1.475 |
| `v2_train_s33.pt`（rolling） | `98bb6e74e25fa7de` | 与 ep128 字节相同 |

### 3.4 每次训练后的最小 sanity（3 次，0 构造、0 步）

| seed | ep128 经真实版本门恢复 | 两个版本字段齐全 | 一次前向 | 训练量核对 |
|---|---|---|---|---|
| 11 | 成功 | checkpoint 4/4、row 128/128 | logits `[4, 2]` / values `[4]`、19 列特征、全部有限 | 32 批 / 128 集 / 464 步；批覆盖恰为 0..127 |
| 22 | 成功 | 4/4、128/128 | 同上形态 | 32 批 / 128 集 / 356 步 |
| 33 | 成功 | 4/4、128/128 | 同上形态 | 32 批 / 128 集 / 464 步 |

sanity 不读任何结果数字，也不做决策；它只证明"固定 checkpoint 能加载、版本字段齐全、可前向"。

## 4. 任务 3：并行文档（不占启动额度，在训练等待期间完成）

| 文件 | 内容 | 状态 |
|---|---|---|
| `delivery_draft/ARCHITECTURE_DRAFT.md` | 组件所有权表；一个 orchestrator step 内 6 步事件顺序（含 `task03.py:1187-1194`、`orchestrator.py:87-100` 行号）；从事件到梯度的数据流；checkpoint 四条入口对照；记账与产物位置表 | 完成 |
| `delivery_draft/REPRODUCTION_DRAFT.md` | 环境与依赖版本、身份核验命令、冻结配置读取位置、三次启动与 sanity/复核/对账命令、Git 内外产物表、换机复现 5 个检查点 | 完成 |
| `delivery_draft/RESULTS_TABLE_TEMPLATE.md` | 按时任务数与计量字节（greedy / sampled 分列；开发块 16 场景已标注"已被历轮使用，不是未见测试集"；独立测试块连块本身都留空并注明必须先登记）；派生量表；冻结 bootstrap 方案；7 条填写禁则；A/B/C/D 证据等级 | 完成，**数据单元格全部留空** |
| `delivery_draft/DELIVERY_SKELETON.md` | 交付目录树、代码身份 7 块、模型清单字段定义、Git 外原件登记表结构、与 ROUND24/32/33/34/35 的关系表 | 完成 |
| `delivery_draft/ARTEFACT_HASH_REGISTRY.json` | 代码身份 + 模型清单（含只读恢复后的 whole-controller 参数摘要与四模块 digest）+ 每 seed 全部原件逐文件 SHA256（554 件/seed，约 150 MB/seed） | 完成 |

## 5. 任务 4：交付对账（`LEDGER_RECONCILIATION.json`）

| seed | 启动进程 | 构造（ledger，含失败照计） | row 文件 | 批 | 优化器步 | 样本 | wall-clock | 上限 | 占上限 | 版本字段 checkpoint | 版本字段 row |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 1 | 128 | 128 | 32 | 464 | 26164 | 1606.26 s | 2270.28 s | 70.8% | 4/4 | 128/128 |
| 22 | 1 | 128 | 128 | 32 | 356 | 18880 | 1572.48 s | 2395.66 s | 65.6% | 4/4 | 128/128 |
| 33 | 1 | 128 | 128 | 32 | 464 | 26032 | 1581.44 s | 2281.95 s | 69.3% | 4/4 | 128/128 |
| **合计** | **3 / 3** | **384 / 384** | 384 | **96** | **1284** | 71076 | — | — | 全部 ≤ 上限 | **12/12** | **384/384** |

与 ROUND32 同口径训练量的一致性：

| 量 | 本轮 | ROUND32 归档 | 判定 |
|---|---|---|---|
| 每 seed episodes / batches / episodes_per_update | 128 / 32 / 4 | 128 / 32 / 4 | **相等（配置项）** |
| epochs / minibatch 步数规则 | 4 / 256 | 4 / 256 | 相等（`v2_common` 引用同一对象） |
| 批 0 样本数 | 623 / 626 / 616 | 623 / 626 / 616 | 逐 seed 相等 |
| 优化器步总数 | 464 / 356 / 464 | 464 / 340 / 464 | s11、s33 相等；s22 +16 |
| 样本总数 | 26164 / 18880 / 26032 | 26156 / 17751 / 26010 | 分别 +8 / +1129 / +22 |

**步数与样本数差的口径说明（不许含糊）**：优化器步数不是配置里的显式数字，而是
`4 epoch × ceil(每批决策行数 / 256)` 的结果；决策行数由 rollout 实际记录的行数决定。
本轮没有任何一批被增减步数、跳过或追加——批数、集数、epoch 数、minibatch 上限四项与冻结
配置逐字节相同（G4），差异全部来自"策略在不同权重下采样出不同数量的决策行"这一后果，
而这正是 §四.6 预期的干预效应。s22 的 +1129 样本对应 V6 里它最晚分叉（e68）之后长尾批次的
决策行数增加，属同口径的观测差，不是训练量被改动。

## 6. 修复在 384 集规模上改变了什么、没改变什么（`ROUND36_VERIFICATION.json`）

| 没改变 | 改变 |
|---|---|
| 每集 episode 级观测量：批 0 的 12 格（3 seed × e0..e3）与归档**逐字节相同**（20 项可观测字段 + 逐决策 `features/actions/targets/actionable/log_prob/step`，`log_prob` 差 **0.0**，`behaviour_sha256` 相同）；批 0 样本数逐 seed 与归档相同；批 0 的 `g_hat` 相同；`unassigned prefix` 三 seed 均 0.0 | 每笔入账的 tick 与**归属到哪一条动作行**：批 0 的 66 笔完成（3 seed × 22 笔）中 **42 笔**的归属行动作行被改变（每 seed e0..e3 为 4/6、3/5、4/5、3/6）；批 0 的 policy surrogate 变化（s11 0.461280 vs 归档 0.468154、s22 0.397680 vs 0.406187、s33 0.457250 vs 0.463832），**更新后参数摘要随之不同**（s11 批 0 后 48c05f14795cdd6a vs 归档 a7c1f4e3b3c48166） |
| 任务写入键的时序性质：384 集全部 `relay_tick == round(event_time)`；2187 笔入账中 **0 笔**落在无动作行的 tick、**0 笔**为 prefix、**0 次**重复 event_id、**0 次**同目标重复入账 | 之后的轨迹：与归档的行为分叉点分别为 s33 e17、s11 e26、s22 e68；`spent_bytes` 相同的集数为 78 / 82 / 80（分母 128）；参数轨迹、λ 轨迹、样本数与 ROUND32 不同 |

**约束统计量的读数（不是效果结论）**：三个 seed 的 `g_hat` 32 批全部 **≥ 0**（s11/s22/s33 均为
0 批为负），λ 单调不降地走到 1.5 / 1.2625 / 1.475。这与 ROUND31/32 记录的"约束被响应但未被
满足"现象同形；由于本轮没有评估、也没有对照干预，**不能**把它读成"约束满足得更差/更好"，
只能读成"对偶变量按公式上行，训练量按冻结配置用完"。

**deadline 与尾段的规模证据**：`on_time == sum(task_stream) == 账本 credited == task_reward_total`
在 384/384 集成立；有目标过期的集数为 34 / 19 / 34（分母 128），这些目标在账本里 0 记录
（collector 第一道状态闸就挡住，过期表现为非完成事件），且 `completed_event_recorded_after_its_deadline`
= 0；落在最后一个动作行上的入账 303 笔，说明尾段完成不需要再走一步即可入账。

**证据等级**：A 级（直接逐字节）= 批 0 的 12 格（V3，三 seed 全过）；C 级（结构 + 计数）=
逐事件账本闭合、event_id 唯一、prefix 为 0（V2，384 集规模，但账本由被测 collector 自己写入，
故只算结构与计数级，不等同于独立仪表）；B 级（同场景配对）= §5/§6 中与归档的观测差；
**本轮不产出 D 级以外的任何效果判断**。

## 7. go / no-go

**go：可以进入第 4 天固定评估。** 判据是：三个 seed 都从登记初值跑满冻结训练量（128 集 / 32 批，
批覆盖恰为 0..127，失败 0，储备未动用）；12 个 checkpoint 与 384 行全部携带两个版本字段且经
真实版本门可恢复；事件账本在 384 集规模上逐集闭合、无重复入账、无 prefix；wall-clock 三次都在
由归档实测构造的上限之内且上限被超时机制强制执行；旧轮产物 0 字节改写（18 个 legacy `.pt` 与
10 件 ROUND34 交付件对账 ROUND35 登记的哈希全部相等）。**没有**任何判据依赖本轮的 loss、回报、
字节或按时任务数水平——这些数字要么只作过程记录，要么留空待第 4 天。

残余风险（不阻塞 go，但第 4 天必须带着走）：

1. **效果仍为空白**：本轮 0 评估启动，三个模型（ep128）与初值（ep0）的表现未测。第 4 天的
   固定评估需要新的额度授权（16 个开发场景 × {init-greedy, final-greedy, final-sampled} × 3 seed
   = 144 次，若加规则臂与独立测试块另计），并且必须先登记独立测试块本身。
2. **开发块不是未见测试集**：1301000..1301015 已被 ROUND22/24/25/28-32/35 使用过；任何"泛化"
   表述都不成立，只能作为开发集上的观测。
3. **`g_hat` 仍从未为负、λ 一路上行**：与三十一/三十二轮的诊断同形（成本侧被响应、任务侧未
   被约束满足）。修复只改信用分配，不改这一形态；若第 4 天评估显示成本/任务组合仍不理想，
   那是设计层面的问题，不能归因于本次时序修复。
4. **两类契约情形在本场景块仍不可达**（事件拍无动作行、首动作前 prefix）：本轮 2187 笔入账
   中 0 笔属于这两类，与 ROUND35 的结构性结论一致；它们仍只有 ROUND34 的合成用例覆盖。
   若第 4 天换场景块，需重做可达性判断。
5. **幂等仍未经实机触发**：384 集里没有重复发布，因此"重复分发不重复入账"仍是合成 + 结构证据。
   本轮未启用 `v2_recover` 一类重放路径；若将来启用，必须先单独核查幂等。
6. **恢复边界本轮未重跑**：ROUND35 已在真实 `.pt` 上验过 5/5 篡改拒绝与旧件拒续训；本轮只新增
   了"新 checkpoint 可经版本门恢复"（V1/§3.4）与旧件字节未被改写（V5），没有重复篡改测试。
   第 4 天若要用只读加载旧件，应引用 ROUND35 的 `RESTORE_BOUNDARY.json` 并注明其口径。
7. **wall-clock 基线的口径局限**：ROUND32 未记 per-launch 耗时，上限由逐集 `seconds` 之和构造，
   更新开销不在基线内（本轮实测 74–95 s/seed，被余量覆盖）。这是"同口径近似"，不是同一量。
8. **本轮 harness 的两处缺陷**（`ROUND36_HARNESS_DEFECTS.json`）需带走：tape 特征形状假设错误、
   跨轮路径登记格式假设错误。两者都朝"假警报"方向失败而非假通过，但说明"用另一轮的登记件当
   键名格式"是重复出现的坑，后续轮次应改为按仓库相对 POSIX 路径登记。

## 8. 停止与交付

```text
训练启动 3 / 3；失败储备 0 / 1（未动用，无需分类）；episode 构造 384 / 384，剩 0
优化器步 1284（全部在启动内；复核与登记脚本 0 步）；评估启动 0；sanity 前向 3
新数据流 0；未调参、未换块、未启用 ACTIVE_ROWS、未热启动任何旧 checkpoint、未用 v2_recover
旧轮产物改写 0（18 个 legacy .pt + 10 件 ROUND34 交付件字节哈希对账相符）
网络尝试 0、未推送、本分支无 upstream
```

交付件（Git 内，按主题提交）：`EXECUTION_BASIS.md`、`FROZEN_CONFIG_MANIFEST.json`、
`ROUND36_PREFLIGHT.json`、`TRAIN_LAUNCH_BOOK.json`、`ROUND36_VERIFICATION.json`、
`ROUND36_HARNESS_DEFECTS.json`、`LEDGER_RECONCILIATION.json`、`ROUND36_TRAIN_REPORT.md`、
`r36_*.py`（7 个脚本）、`seed{11,22,33}/*.json`（每 seed 6 份记录：freeze manifest、launch
record、training record、sanity、checkpoint 哈希清单、λ 轨迹）、`delivery_draft/`（4 份草稿 +
哈希登记表）。Git 外原件：每 seed 554 件（rows / tapes / lifecycles / transcripts / checkpoints /
批记录，约 150 MB/seed），逐件路径与 SHA256 登记在 `delivery_draft/ARTEFACT_HASH_REGISTRY.json`。

最终头哈希在交付提交完成后由 `git log --oneline -1` 给出，本文件不预填（避免猜）。
本轮到此停止：不追加额度、不调参、不续训、不做评估、不开启下一轮、不部署、不推送。
