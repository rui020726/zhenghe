# 任务压力 g12 独立扩展闭环：第四层实测与分支判定

日期：2026-09-23。分支 `analysis/comm-task-pressure-g12-extension`，起点 `analysis/comm-task-pressure-learnability@6b4452f5de88f7cfd0c17ce475de269eb13dd6b7`。协议 `COMM_TASK_PRESSURE_G12_EXTENSION_PROTOCOL.md` 先于任何一条 g12 episode 单独提交。

**本轮 0 训练、0 `optimizer.step`、0 checkpoint、0 模型加载、0 新增验证 episode。** 唯一自变量 `num_goals = 12`；额度 29 300 B、`policy_owned_timing`、isolated-lossless、ACK 关闭、九臂规则库、bootstrap 口径与前三层逐项相同。

**地位**：这是对原协议预登记却没跑完的第四层的容量扩展（探索性闭环）。它不倒推修改原三层的 confirmatory 判定；原判据文件一字未动。

## 一、判定

| 压力层 | 流 | L0 | L1 | L2 | L3 | L4 | 该层结论 |
|---|---:|---|---|---|---|---|---|
| goals=6（已交付） | 16 | 过 | 过 | **不过** | **不过** | **不过** | 未通过（原判据，本轮不改） |
| goals=8（已交付） | 16 | 过 | 过 | **不过** | **不过** | **不过** | 未通过（原判据，本轮不改） |
| goals=10（已交付） | 16 | 过 | 过 | 过 | **不过** | **不过** | 未通过（原判据，本轮不改） |
| **goals=12（本轮）** | 16 | 过 | 过 | 过 | **不过** | **不过** | L2 过、L3 未过 |

**下一步（协议 §8 规定的分支）**：正式关闭“仅增加任务数量即可建立逐步预算选择”这条路线；不再往 14/16 个目标加。下一轮只做只读 action-geometry audit，不训练

本轮不训练：`training_authorized = false`。

## 二、12 目标的九臂实测

最佳预算内固定规则 `send_once`；逐流 hindsight envelope 均值 8.9375 个/episode；mean detected/placed 0.95313（最差流 0.75，seed 1000305）。

| 臂 | 角色 | 按时完成/ep | 字节/ep | on-time/detected | 发射/ep | 额度≥90 % 流 | 争用点流 | 争用点合计 | 仅合并超支 | 拒付/ep | 多候选步 | 其中全不可付 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `send_once` **(最佳)** | competitor | 8.875 | 27134.6 | 0.7817 | 10.94 | 15 | 3 | 198 | 981 | 355.2 | 4852 | 3061 |
| `event` | competitor | 6.062 | 28367.2 | 0.5331 | 11.25 | 16 | 5 | 29 | 49 | 4419.4 | 15742 | 15605 |
| `event_priority` | competitor | 6.000 | 28316.2 | 0.5275 | 11.25 | 16 | 5 | 29 | 49 | 4369.7 | 15742 | 15605 |
| `periodic_5` | competitor | 5.938 | 28101.4 | 0.5235 | 11.31 | 16 | 0 | 0 | 77 | 855.0 | 14567 | 14387 |
| `periodic_20` | competitor | 7.000 | 28244.6 | 0.6154 | 11.31 | 16 | 2 | 40 | 258 | 130.9 | 11087 | 10415 |
| `first_send_plus_bound_3_10` | competitor | 6.938 | 28429.9 | 0.6119 | 11.31 | 16 | 4 | 21 | 225 | 2792.9 | 12078 | 11720 |
| `bound_only_3_10` | competitor | 7.062 | 28352.2 | 0.6228 | 11.25 | 16 | 4 | 20 | 261 | 2751.6 | 12129 | 11608 |
| `always_hold` | control | 0.000 | 0.0 | 0.0000 | 0.00 | 0 | 0 | 0 | 5683 | 0.0 | 15995 | 0 |
| `event_unbudgeted` | ceiling_reference | 11.375 | 224651.6 | 0.9948 | 84.94 | 0 | 0 | 0 | 0 | 0.0 | 1397 | 0 |

| 臂 | 相对 `always_hold` 完成差 | 95 % CI | 相对最佳规则完成差 | 95 % CI | 相对最佳规则字节差 | 95 % CI |
|---|---:|---|---:|---|---:|---|
| `send_once` | +8.8750 | [+7.6250, +10.0000] | +0.0000 | [+0.0000, +0.0000] | +0.0000 | [+0.0000, +0.0000] |
| `event` | +6.0625 | [+5.2500, +6.8750] | -2.8125 | [-3.8125, -1.8750] | +1232.6250 | [+610.0969, +1917.0187] |
| `event_priority` | +6.0000 | [+5.1250, +6.8750] | -2.8750 | [-3.7500, -2.0000] | +1181.6250 | [+559.1250, +1878.0094] |
| `periodic_5` | +5.9375 | [+4.7500, +7.1250] | -2.9375 | [-3.6875, -2.3125] | +966.7500 | [+377.6250, +1683.7500] |
| `periodic_20` | +7.0000 | [+5.7500, +8.1875] | -1.8750 | [-2.4375, -1.3125] | +1110.0000 | [+493.4906, +1841.2875] |
| `first_send_plus_bound_3_10` | +6.9375 | [+5.7500, +8.0625] | -1.9375 | [-2.5000, -1.3750] | +1295.2500 | [+716.9906, +1984.8750] |
| `bound_only_3_10` | +7.0625 | [+5.9375, +8.1875] | -1.8125 | [-2.4375, -1.2500] | +1217.6250 | [+666.3469, +1906.1344] |
| `always_hold` | +0.0000 | [+0.0000, +0.0000] | -8.8750 | [-10.0000, -7.6250] | -27134.6250 | [-27664.1250, -26422.8469] |
| `event_unbudgeted` | +11.3750 | [+10.9375, +11.6875] | +2.5000 | [+1.3125, +3.8750] | +197517.0000 | [+181898.0812, +213652.3312] |

- 样本均值 Pareto：[{'arm': 'send_once', 'bytes_per_episode': 27134.625, 'on_time_per_episode': 8.875}]
- 逐流唯一赢家计数：{'send_once': 15}；互不支配且双向胜负的规则对：['bound_only_3_10 vs periodic_20', 'event vs first_send_plus_bound_3_10', 'event vs periodic_5', 'event_priority vs periodic_5', 'periodic_20 vs periodic_5']
- envelope − 最佳固定规则：**0.0625 个/episode**，95 % CI [0.0, 0.1875]
- 相邻点 lambda break-even（只在 L0–L4 全过时才是可引用口径）：[]

## 三、L2 的带边界读法

- `best_on_time_rate_of_detected = 0.78172`，闭带 `[0.4, 0.9]`，`band_holds = True`
- 该比值不在任何边界的 ±0.005 之内，因此不需要 §1.4 的三读并报。

## 四、机制诊断：额度以什么形式 binding（不参加任何判据）

- 几何量已记录：`True`
- 每步候选数分布（16 流 × 1 000 步合计）：`{'one': 6336, 'two_or_more': 4852, 'zero': 4812}`
- 多候选步的结局划分：`{'all_payable_within_credit': 612, 'closes_on_multi_candidate_steps': True, 'every_candidate_payable_but_not_jointly': 981, 'none_payable': 3061, 'strict_subset_payable': 198, 'unclassifiable': 0}`
- 多候选步的包价：`{'candidates_priced': 18295, 'max': 2940, 'mean': 2628.1, 'min': 2178, 'spread': 762, 'spread_as_fraction_of_credit': 0.02601}`
- 三个成因指标：`{'share_of_steps_with_two_or_more_candidates': 0.30325, 'price_spread_as_fraction_of_credit': 0.02601, 'share_of_multi_steps_with_nothing_payable': 0.63087, 'share_of_multi_steps_where_any_subset_was_payable': 0.04081}`

- 发包口径（**已发出的包**，与前三层同算法、来自已存转录，不花 episode）：2617 包，最小 2178 B、中位 2604 B、最大 2946 B、极差 768 B = 额度的 2.62%；额度总共约 11.25 个包。上面那行是**候选报价**口径（多候选步上所有候选的价格，含没发出去的），两个数不同是量纲不同，不是矛盾。
- 首次额度耗尽发生在哪类消息：最佳规则 15 条流走到过"看得见候选但一个都付不起"，其中 **15 条**的第一个 such 步上所有候选都是**刷新**（`first_emissions = 0`）——额度是在已经发过消息的目标续报时耗尽的，不是在新目标的第一条消息上。逐条明细（步号、候选数、各候选包价、首发/刷新计数）在 `GATES_g12.json` 的 `gates.L3.evidence.step_geometry_of_the_best_rule.first_credit_exhaustion`。

**L3 为什么仍未过（成因归类，按协议 §5 的三个指标，不设阈值）**：候选并发**不是**限制——16 流合计 4852 个多候选步（占全部步的 30.3%）；包价极差 2.60% 额度，『剩余额度恰好落在两个候选报价之间』这条缝本来就窄；而多候选步里 63.1% 直接是一个都付不起，只有 4.1% 出现真正的严格子集。**三种成因里 binding 的是第三种：额度成片跨进全拒付区**，与前三层的读法同向，只是 12 目标把它推到了能量化的程度（严格子集步从三层的 0 变成 198 步、集中在 3 条流）。

| 压力层 | 几何量已记录 | 多候选步 | 严格子集步 | 全不可付步 | 极差/额度 |
|---|---|---:|---:|---:|---:|
| 6 | False | — | — | — | — |
| 8 | False | — | — | — | — |
| 10 | False | — | — | — | — |
| 12 | True | 4852 | 198 | 3061 | 0.02601 |

- **这张表的不对称是预先登记的**（协议 §5）：逐 step 的候选几何只有 g12 记了，因为前三层的行里没有这些键。因此不得用 g12 的分布回头改写前三层已判完的 L3；能跨层比的仍是`competition_points`、包价极差与 `credit_exhausted_with_candidates_streams`。

## 五、压力与容量

- placed 12（`illegal_streams = {}`）、detected 均值 11.438、registered 均值 11.438、并发开放任务均值 1.025、p95 4.812、机器人利用率 0.1426、工作量/执行容量 0.028、从未被分配 0.5
- 无预算参考的 `event_unbudgeted` 完成 11.375 个/ep：若它也不高，未完成任务里有一部分属于搜索/执行容量，不能记在通信账上（原协议 §8）。

## 六、启动账本与上限诚实性

- 原协议上限 **640**，本轮追加授权 **16** 次 ⇒ 有效上限 **656**；实到数据启动 **656**（含前三层的 496 次与本层的 16 录制 + 144 矩阵）。
- 核验启动 **10 / 12**：本轮没有新增核验 episode（执行指令 §3 不允许增加验证 episode），探针扩展只用零 episode 测试证明其透明。
- 账本记录的技术失败 **20** 次（含前三层的 20 次），全部计入，不抹账。
- 格覆盖：144/144，缺格 0

## 七、身份与卫生

- 冻结文件 `G12_FREEZE.json`，登记头 `098ae183e`，模块 56 个。**给行打戳的那条身份路径**（`inspect.getsource`，与环境无关）给出 `ff0b4139b1fff700`，`run_episode` 源哈希 `a6a00ec98030dbf1`——g8 与 g10 的全部 288 格行戳是同一个值，本轮 144 格也是，所以这三层加本层同属一个 episode 代码状态（由哈希核对）。g6 的 144 格**没有行戳**，因为戳是在那层跑完之后才加的——这一点已交付轮的 `ROW_PROVENANCE_g6.json` 就写明了，本层不替它补叙述性归因。
- 另一条按 commit 算的路径（报告里 freeze 时间线用的那条，`git show` 按当前 locale 解码）在 cp936 下给出 `c922ca100665609c`、在 UTF-8 下给出 `23390f55a6b90afa`，**同一批 commit 也是这两个值**——它是环境相关的，本文全程用与已交付报告相同的 cp936 环境生成，两个数都写在这里以免被当成"代码换了"。
- 行戳：{'098ae183e': 144}，未戳 0，单一 episode 代码状态 = True
- 四源冲突审计（[12] 块）：命中 {'artefacts': 0, 'documentation': 0, 'git': 0, 'manifests': 0}，`clean = True`
- 封存块 `800000..800399` 盘上 artefact **0**；生产默认值 `comm_ack=False`、`comm_timing_mode=legacy_shared_bound`、`num_goals=6`
- 归档一致性（继承自原轮）：`agree = True`；本轮探针扩展的透明性由 `test_pressure_harness` 的零 episode 测试证明
- 测试：本轮合成套件 通过：OK；项目套件 通过：OK (skipped=2)；集成 smoke 通过：final_status=completed
- 起点核对的补记：协议 §1.1 如实写过"开跑时 github.com:443 不可达，只核对了本地远程跟踪 ref"；交付时再核一次：在线核对通过：`git ls-remote` 报起点分支 `analysis/comm-task-pressure-learnability` 的远端 HEAD = `6b4452f5de88f7cfd0c17ce475de269eb13dd6b7`，与执行指令要求的 SHA 一致。本分支自身的远端 HEAD 无法写进本报告（报告所在的提交就是被核对的对象），它随最后一次推送单独报出。

## 八、本轮欠读者的披露（不是判据，逐条写明动了什么）

- **the round's own grant invariant could only pass before the layer spent anything**
  - found: after the matrix ran, when `python -m unittest test_pressure_harness` failed under the UTF-8 environment the report stage runs in
  - symptom: grant == 160 - (640 - matrix_starts) reads 176 once matrix_starts is 656: a live ledger cannot restate what was left at authorisation time
  - fixed: pressure_common.launch_grant_audit() answers the question against G12_START_LEDGER.json (matrix_starts = 496 before the layer), and the self-check and the test read that instead
  - touched: a self-check and one test; no episode, no row, no aggregate, no gate
- **episode_code_identity has two routes and only one of them is stable**
  - detail: the route that stamps rows (inspect.getsource, locale-independent) gives ff0b4139b1fff700 for the 288 stamped cells of the delivered round (g8, g10) and for all 144 g12 cells; g6's 144 cells carry no stamp at all, as its delivered provenance file already says. The route the freeze timeline uses (git show, decoded in the ambient locale) gives c922ca100665609c under cp936 and 23390f55a6b90afa under UTF-8 for the *same* commits
  - consequence: the single-code-state claim is checked on the stamping route, which agrees across all four layers; a timeline hash is only comparable inside one pinned environment, and this report is generated in the one the delivered round used
  - not_fixed: the hashing route is left as the delivered round froze it: changing it after the launch would move an identity number without moving an episode
- **GATES_g12.json's stop_reason says 'proceed to the next layer'**
  - detail: that sentence was written when layer 12 was not the last layer; the judge is frozen from the first data launch, so the wording was not edited
  - consequence: the next step the round actually prescribes is the §8 branch recorded in this report's section 1, and there is no fifth layer

- 追加授权的算术（对着开跑前的快照核）：授权前已花 496 / 640，有效上限 656 ⇒ 空出 160 次，第四层需要 160 次，`exact = True`；本轮实到 656 次，未越线。

## 九、结论范围与下一步

- **地位**：g12 是原协议 §2 预登记的第四层，本轮按使用者追加授权（640 → 656）补跑；它是容量扩展 / 探索性闭环，不是对原三层门禁的事后改写。原判据与结果文件未被覆盖、未被重写（`report_g12.py` 与 `common.write_json` 双重守卫）。
- 12 目标层：最佳固定规则 `send_once`，按时完成 8.875 个/episode，bytes 27134.625，on-time/detected 0.78172，L0 True / L1 True / L2 True / L3 False / L4 False。
- L3 实测：最佳规则争用点流 3/16，额度 ≥90 % 流 15/16。
- L4 实测：envelope − 最佳固定规则 +0.0625 个/episode，95 % CI [0.0, 0.1875]。
- 分支判定：**l2_pass_l3_fail** ⇒ 正式关闭“仅增加任务数量即可建立逐步预算选择”这条路线；不再往 14/16 个目标加。下一轮只做只读 action-geometry audit，不训练
- 不得据本轮写“学习无效”“规则全局最优”“不存在更优策略”：本轮仍只测了目标数这一条轴，且 L2/L3/L4 是必要条件门禁，不是策略优劣的判决。
- 本轮未训练、未改 reward、未新增规则/seed/验证 episode、未打开 `800000..800399`、未修改生产默认值；λ 只在 L0–L4 全过时才是可引用口径。
- 跑完即停止：下一轮方向由使用者按 §8 的分支决定，本轮不自动展开。

## 十、复跑

```bash
cd runs/comm_task_pressure_learnability_20260922
python pressure_common.py
python seed_audit.py --goals 12 --out G12_SEED_COLLISION_AUDIT.json
python -m unittest test_pressure_harness
python freeze_pressure.py --mode register --goals 12
python record_layer.py --goals 12 --shards 4 --shard 0   # 1..3 同理
python record_layer.py --goals 12 --mode merge
python pressure_matrix.py --goals 12 --shards 4 --shard 0   # 1..3 同理
python pressure_matrix.py --goals 12 --mode coverage
python pressure_matrix.py --goals 12 --mode provenance
python pressure_metrics.py --goals 12
python judge_layers.py --goals 12
python report_g12.py
cd ../.. && python -m unittest discover -s integrated_sar/tests && python -m integrated_sar
```
