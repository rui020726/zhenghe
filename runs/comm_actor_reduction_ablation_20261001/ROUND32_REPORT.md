# ROUND32 报告：actor surrogate 归约分母消融（ALL_ROWS vs ACTIVE_ROWS）

分支 `exp/comm-actor-reduction-ablation`，起点 ROUND31 交付头完整 SHA
`614df51709a8d82525405b46468a537f4ccd259e`（本地核验：`git merge-base --is-ancestor
915868297c08d4b01b7dfeff0ea810d5394057a3 HEAD` 为真；`git log --ancestry-path
9158682..HEAD` 只含两枚 round-31 提交加本轮四枚提交，无其他会话的提交混入）。
产物 `runs/comm_actor_reduction_ablation_20261001/`，
执行依据 sha256 `49d52f95…`，生效冻结件 freeze_sha256 `b5615c1bed8b5686…`
（提交 `3bafb36`：首次环境启动之前因 CLI 入口缺陷 `main()` 形参未接 `sys.argv` 而重新冻结，
该次修正消耗环境启动 **0** 次、消耗合成 optimizer 更新 **0** 次，重冻结前的 `fa5e7638…`
未用于任何一次启动，两份哈希的差异只在 `r32_blocks.py` 与 `r32_train.py` 的 `__main__` 行）。

本轮问题：**仅**把 PPO actor surrogate 从"所有记录行平均"改为"有自由决策的记录行平均"，
是否改善有限训练内的任务—成本表现？这是优化设计消融，不宣称修 bug。

```text
环境启动 1040 / 1040 正常、0 失败、0 重试、储备 40 未动、硬上限 1080
真实训练 = 六条运行 × 32 批 = 192 批、2628 次 optimizer 步
           （逐运行 464/464/340 vs 464/432/464，见 §2 表下的说明）
纯合成 optimizer 更新 = 授权 32 / 实执行 40（含开发期 8 次重复，单独分账，上限 64）
```

---

## 1. 唯一主动变化是否确实只是归约？——是（工具级修正已登记）

| 核验 | 结果 |
|---|---|
| 生产/导入模块/搜索分配权重三段哈希 vs ROUND30 冻结件 | **0 处差异** |
| `configuration_hash` | `e9b2c067db5951f2`，与 ROUND24/30 逐字符相同 |
| 128 集数据顺序 | 与 ROUND30 记录逐元素相同（摘要 `346f2cdde8d658f9`） |
| 编码器 | 两臂同为 ROUND30 的 old 臂 `comm-send-state/1`（生产 `features`），新期限编码未参与 |
| `r32_trainer.ppo_update` 逐块差异 | research_variable 8 块 + passive_read 4 块 + **unexplained 0 块**；被删父件行仅 3 行（归约行 / `reduction` 形参 / 仅报告字段） |
| `r32_run` 与 ROUND30 运行器 | 11 处差异块全属声明接缝；`run_episode`/`run_reference`/`run_rule` **函数体逐行一致** |
| 未触碰的数学 | `batch_samples`、`_bundle`、`likelihood_alignment`、`dual_update`、`batch_g_hat`、`cost_math.gae`、`cost_math.actor_advantage` 与冻结件是**同一函数对象**（断言全真） |
| 初始化配对 | 三 seed 两臂 parameter_sha256、四模块摘要、optimizer 状态哈希、λ、fresh_state **全等**（首次环境启动之前生成并随冻结件提交）；**发现一处来源缺陷见下** |
| 归约数学 | 11 项检查全过；`ALL_ROWS` 与冻结 `v2_trainer.ppo_update` **逐 optimizer 步权重哈希相同** |
| `ROUND32_VERDICT.json` 的 `verified` 块 | **8 项全真**，且每项都由本目录已交付机读件派生（唯一变化 / 归约确实改变了更新 / 首批门禁 / 干预抵达 λ 与 g_hat 序列 / 场景块未被重排 / 18 份更新前 checkpoint 身份相符 / 配对件承重字段跨重写全等 / 独立复核 clean） |

**冻结后的六处工具级修正**（`POST_FREEZE_CHANGES.json`：AST 函数级定位 + difflib 行级差异，
逐处证明只落在只读验证/事后分析函数内，且训练/采样/更新/恢复路径八件的哈希**逐字节未变**）：

① `r32_check.py::first4_gate` 的 G8 把"逐 minibatch 求和"当成"逐 epoch 一次"来比，
正确关系是 `recorded = epochs × tape 行数`（实测 2492 = 4×623、120 = 4×30，计数本身没错）；
② `r32_curves.py` 同一 epochs 关系在建曲线时也被写成对原始行数比较（因此每格虚报 64 项
"问题"），改为按 epoch 断言，并补上本文件自己的描述性辅助函数；
③ `r32_mechanism.py` 的放大比值容差由绝对 1e-5 改为相对 1e-5（比值本身达 51，绝对带宽只在
校验 float32 位数），同时修正行文件通配符漏掉的 `_s{scenario}` 段；
④ `r32_verify.py` 的手算 bootstrap 把未舍入区间与机读件自身的舍入（ΔY 6 位、ΔC 3 位）比较，
以及训练行通配符同一缺陷；
⑤ `r32_stats.py::absolute_block` **新增**两列消息数读数（`emissions_total_16_scenes`、
`mean_emissions_per_episode`），使"谁发得更多"与判据（字节、按时数）出自同一件产物而不是报告
手抄；既有字段、估计量、重采样索引与舍入规则一字未动，`r32_verify.py` 的重建表仍独立地从原始
行重新导出 Δ 字段（重建 vs 交付 6/6 格全等）；
⑥ `r32_verdict.py` 增加七问的结论字段、`verified` 块与消息读数——全部由本目录已交付机读件派生。

六处均未重跑任何环境、未动账本、未新增启动；登记件里逐文件列出
`functions_touched / lines_changed / module_level_changed_lines`，并断言
`training_path_unchanged = true`。

### 1.1 独立复核发现的一处来源缺陷（如实登记，不事后改训练路径文件）

`INDEPENDENT_RECHECK.json` 的 `init_pairing_provenance` 把每格的"启动前配对件"（任务 C+D 冻结
提交 `df176bdf8c3c265d8109494446073b23aee384a6` 与生效冻结提交 `3bafb36` 里的版本，二者内容相同）
与工作区里的当前文件逐字段比对，结论：

- **承重字段 6/6 格 0 处变化**（`parameter_sha256`、四模块摘要、optimizer 状态哈希、λ、
  `feature_layout`、`fresh_state` 全等），所以"两臂同初值"的配对主张仍然成立；
- 但 `r32_train.record_init_pairing` 在每条运行的首次真初始化处**重写**了同一个文件，
  因此工作区版本**丢掉了 `written_before_the_first_environment_launch` 这个来源标记**
  （`pre_launch_marker_still_in_the_working_file = false`）；启动前那份带标记的原件只存在于
  已提交的冻结版本里；本轮另存了一份，见下一条；
- 由此 `r32_freeze.init_pairing_evidence` 文档字符串所称"r32_train 改为复核而非覆写"
  **与实际交付代码不符**。本轮不修改 `r32_freeze.py` / `r32_train.py`（二者都在
  `POST_FREEZE_CHANGES.json` 的训练/采样/更新/恢复路径清单里，冻结后改动必须停止并报告），
  所以这一点**只登记不修补**；下一轮若沿用该 harness，应把运行期证据写到独立文件名。
- **补救（0 启动、不改训练路径）**：`r32_postfreeze.py::preserve_prelaunch_pairing` 把生效
  冻结提交 `3bafb36`（同一内容的更早一份是 `df176bdf…`）里那六份带标记的原件逐字段另存为
  **`INIT_PAIRING_PRELAUNCH.json`**，并附"启动前 vs 运行期"的逐格比对：
  `all_load_bearing_fields_equal = true`，六格丢失的键恰好只有
  `written_before_the_first_environment_launch` 一个。交付因此不再依赖"记得去 `git show`"。
- 影响范围评估：该缺陷不影响任何测量结果（配对、更新、评估、统计都不读这个标记），
  影响的是"证据自述能力"，属于我在报告里差点写错、被独立复核抓出来的那一类。

## 2. 实际 actor 信号与总更新发生了什么

任务 D 的"更新前完整 checkpoint"（批 0/15/31 × 六条 = 18 份）全部落盘，身份与批记录的
`controller_parameters_before` **逐格相符** —— ROUND31"只有批 0 存活"的限制已消除。
18/18 点均可分析，样本计数与优势组合逐点闭合。

**同状态、同 tape 的算术比较**（该批 actor 项 L2 之比 = 观测放大 vs 理论 `N/M`）：

- 观测放大在 **18/18 点**都等于 `N/M`，最大相对残差 **2.9e-07**（最大绝对残差 1.5e-05 出现在
  N/M=51 处，属 float32 位数，不是偏差）；
- actor 项在**裁剪前总梯度**里的 L2 占比：`ALL_ROWS` **0.95 %–20.4 %** → `ACTIVE_ROWS`
  **14.9 %–97.1 %**；
- 但 **13/18 点两臂同时触发 `MAX_GRAD_NORM=0.5` 全局裁剪**，裁剪后总范数两臂都恰好是 0.5 ——
  改变的是**方向（actor 占比）**，不是被裁剪向量的大小；
- 例（`ALL_ROWS_s11/b0`，N=256、M=13）：actor 项 L2 0.0552 → 1.0868（×19.69 = N/M），
  总 pre-clip 2.386 → 2.615（+9.6 %），actor 占比 2.31 % → 41.56 %，critic 占比 99.99 % → 91.25 %，
  两臂裁剪后同为 0.5。

因此**不能**把 `N/M` 说成参数更新倍数：Adam 对梯度非线性，entropy 与两个 critic 仍在同一向量里。

**被放大的那一项里装的是什么**（`UPDATE_MECHANISM.json` 各点的 `advantages` 读数，18/18 点）：
组合优势 `A_actor = λ·A_task − A_cost` 的两项幅度比 `abs_mean(A_cost) / abs_mean(λ·A_task)` 在
**0.18284 – 0.424743** 之间（`ROUND32_VERDICT.json` `answers.2...advantage_scale_read.min/max`；
分析点上的批 λ 从 1.0 涨到 1.475，运行 λ 终值最高 1.5000，也没把这个比推过 0.5）
——也就是说 `ACTIVE_ROWS` 乘上
`N/M` 的那一整项里，任务侧幅度始终是成本侧的 2.4–5.5 倍。这与观测方向一致：稀释去掉后
先变强的是"多报告以保任务"的一边，于是 `s33_greedy` 从 0.9375 条/集跳到 11.0 条/集、成本
+25709.25 B 而任务不涨（本来已在那一格的参考值上）。**这是与数据一致的解释，不是本轮单独
识别出的因果**：要判定尺度本身，得做只改尺度、不改归约的另一次消融（本轮无授权）。

**训练轨迹侧（六条 × 32 批）**：两臂的 `g_hat` 与 λ 序列本轮**确实分叉**（ROUND30 编码消融里
两臂逐批相同）——首轮分叉批 s11=10、s22=3、s33=14，行为分叉批 s11=1、s22=1、s33=2。

| 运行 | λ 0→终 | `g_hat` 正/零/负 | 全 128 集配对任务损失 | 平均 c=C/B0 | active 行占比 | 裁剪 minibatch 占比 |
|---|---|---|---|---|---|---|
| ALL_ROWS s11 | 1.0000→1.5000 | 23 / 9 / **0** | 0.3125 | 2.8278 | 3.69 % | 58.8 % |
| ACTIVE_ROWS s11 | 1.0000→1.4875 | 23 / 9 / **0** | 0.3047 | 2.8268 | 3.16 % | 94.4 % |
| ALL_ROWS s22 | 1.0000→1.2375 | 13 / 19 / **0** | 0.1484 | 2.4988 | 28.70 % | 73.5 % |
| ACTIVE_ROWS s22 | 1.0000→1.4250 | 21 / 11 / **0** | 0.2656 | 2.8186 | 10.83 % | 95.1 % |
| ALL_ROWS s33 | 1.0000→1.4750 | 23 / 9 / **0** | 0.2969 | 2.8317 | 4.32 % | 58.6 % |
| ACTIVE_ROWS s33 | 1.0000→1.4875 | 23 / 9 / **0** | 0.3047 | 2.8289 | 3.81 % | 95.7 % |

**六条运行、192 批里 `g_hat` 仍无一次为负**；`ACTIVE_ROWS_s22` 的违背反而更多（21 批为正 vs
13 批），λ 终值 1.4250 vs 1.2375。零非有限值、λ 链/权重链/配对全闭（`problems = 0`）。

上表的 minibatch / optimizer 步数在两臂之间**并不相等**（`ALL_ROWS_s22` 只有 340 个 minibatch，
`ACTIVE_ROWS_s22` 有 432 个）：每批的样本数取决于该臂当时的行为（决策步数不同），
`MINIBATCH_STEPS=256` 的切分因此给出不同块数。这是**行为分叉的结果**，不是第二次干预——
每批的数据顺序、episode 集合与更新触发条件两臂完全相同（`the_scene_block_was_not_re_shuffled`
= 真：三对 seed 的场景序列与 `Y_ref` 序列逐项相同）。

## 3. greedy 与 sampled 的任务与成本结果（16 个已使用开发场景，ep128）

规则参照：immediate **6.0000 / 14 939.25 B**、`late_discovery_gate_147` **6.0000 / 4 838.25 B**
（与 ROUND30/31 的重建完全一致）。Δ = ACTIVE_ROWS − ALL_ROWS。

| 格 | ALL_ROWS Y / B / 配对损失 | ACTIVE_ROWS Y / B / 配对损失 | ΔY（CI95） | ΔC B/ep（CI95） |
|---|---|---|---|---|
| s11 greedy | 5.5 / 28207.9 / 0.5 | 5.5 / 28198.9 / 0.5 | +0.0（[0, 0]） | -9.0（[-27.0, 0]） |
| s11 sampled | 5.5625 / 28128.8 / 0.4375 | 5.5625 / 28206.4 / 0.4375 | +0.0（[0, 0]） | +77.625（[-82.125, +303.375]） |
| s22 greedy | 5.5 / 0 / 0.5 | 5.5 / 0 / 0.5 | +0.0（[0, 0]） | +0.0（[0, 0]） |
| s22 sampled | 5.875 / 15835.5 / 0.125 | 5.8125 / **26899.5** / **0.1875** | -0.0625（[-0.25, +0.125]） | **+11064.0**（[+7339.05, +14652.009]） |
| s33 greedy | 6 / **2398.5** / **0** | 5.5 / **28107.8** / **0.5** | **-0.5**（[-1.0625, -0.125]） | **+25709.25**（[+24153.741, +27076.509]） |
| s33 sampled | 5.5625 / 28199.2 / 0.4375 | 5.5625 / 28136.6 / 0.4375 | +0.0（[0, 0]） | -62.625（[-376.134, +295.5]） |

表内数字按 `PAIRED_COMPARISON.json` 自身的舍入逐字誊写（B 为 6 位有效数字，字节均值不加千位空格），
`INDEPENDENT_RECHECK.json` 的 `report_same_source` 就是把这 42 个字符串（6 格 × 7 量）逐项在报告正文里
查找；未舍入的均值留在机读件里，报告不另起一套数。规则参照的精确值：immediate
6.0000 / 14939.25 B、`late_discovery_gate_147` 6.0000 / 4838.25 B。

逐场景计数与绝对位置（每臂）：

| 格 | 臂 | 有损失场景 | 最大单场景损失 | vs late ΔY / ΔC | 比 late 便宜又更少按时 |
|---|---|---|---|---|---|
| s11 greedy | 两臂同 | 5/16 | 4 | −0.5 / +23 369.6、+23 360.6 | 0 |
| s22 greedy | 两臂同 | 5/16 | 4 | −0.5 / −4 838.25 | **各 5 个场景** |
| s33 greedy | ALL_ROWS | **0/16** | 0 | 0.0 / −2 439.75 | 0 |
| s33 greedy | ACTIVE_ROWS | **5/16** | 4 | −0.5 / +23 269.5 | 0 |
| s22 sampled | ALL_ROWS → ACTIVE_ROWS | 2 → 3 | 1 → 1 | −0.125 → −0.188 / +10 997.3 → +22 061.3 | 0 |
| s11/s33 sampled | 两臂 | 4 → 4 | 4 | −0.438 / ≈+23 3xx | 0 |

- 目标集合交换只在 `s22_sampled` 出现（1 个场景净 ΔY −1 但伴随 2 个场景变化，`swap=True`）；
  其余格 `gained/lost` 皆空。
- 行为逐字节相同的场景数：s11 greedy 16/16、s22 greedy 16/16（两臂都不发）、s33 greedy 仅 2/16
  —— **改动最大的那一格正是变差的那一格**。
- 没有任何一格出现"比 ALL_ROWS 更省字节又丢任务"（`cheaper_but_lost_task_vs_base = 0`，6/6 格）。

**发送次数读数（同份 `PAIRED_COMPARISON.json` 的 `emissions_total_16_scenes`，不是第二套数）**：

| 格 | ALL_ROWS 消息/16 集（每集均值） | ACTIVE_ROWS 消息/16 集（每集均值） | Δ 消息/集 |
|---|---|---|---|
| s11 greedy | 177（11.0625） | 177（11.0625） | +0.0 |
| s11 sampled | 176（11.0） | 177（11.0625） | +0.0625 |
| s22 greedy | 0（0.0） | 0（0.0） | +0.0 |
| s22 sampled | 95（5.9375） | 162（10.125） | **+4.1875** |
| s33 greedy | 15（0.9375） | 176（11.0） | **+10.0625** |
| s33 sampled | 176（11.0） | 176（11.0） | +0.0 |

成本上升的三格（s11 sampled +77.625、s22 sampled +11064.0、s33 greedy +25709.25）**正是**消息数
上升的三格（+0.0625、+4.1875、+10.0625 条/集），其余三格两臂消息数完全相等；而 `s22_greedy`
两臂都是 0 条 / 0 B —— ACTIVE_ROWS 并没有把"HOLD"换成"发一条小的"，它是把整集的沉默换成了
整集的持续上报（`cells_where_test_arm_sends_more = 3`、`cells_where_test_arm_sends_less = 0`）。

## 4. 三个训练 seed 是否一致

不一致，且不一致的方向不同：

- s11：两臂行为完全重合（greedy 16/16 场景 `behaviour_sha256` 相同；sampled 8/16），差别只有
  -9.0 B（greedy）与 +77.625 B（sampled），两者 CI 都含 0；
- s22：greedy 两臂仍是"整集不发 + 0.5 任务损失"（0 条消息 / 0 B），sampled 下 ACTIVE_ROWS 明显
  更贵（+11064.0 B，CI [+7339.05, +14652.009] 不含 0；消息 5.9375 → 10.125 条/集）且多损 0.0625
  任务；
- s33：**方向相反**——ALL_ROWS 落在"6 按时 / 2398.5 B / 0.9375 条消息"（配对损失 0），
  ACTIVE_ROWS 退回 "5.5 按时 / 28107.8 B / 11.0 条消息"（ΔY -0.5、ΔC +25709.25，两个 CI 都不含 0）。

三 seed 分别展示、未池化、未挑最佳模式。

## 5. 约束违背是否减少？

**没有。** 判据是与配对的 immediate reference 比较（δ=0），不是消息数：

- 训练侧：`g_hat` 在 192 批里仍无一次为负；ACTIVE_ROWS 三条运行的正批数
  23/21/23 vs ALL_ROWS 的 23/13/23 —— **s22 明显更差**；
- 评估侧：greedy 的配对任务损失 ALL_ROWS `{0.5, 0.5, 0.0}`、ACTIVE_ROWS `{0.5, 0.5, 0.5}` ——
  ACTIVE_ROWS **没有一格比 ALL_ROWS 更接近约束满足**，且在 s33 从 0.0 退化到 0.5；
  六格合起来的均值绝对损失 ALL_ROWS `0.333333` → ACTIVE_ROWS `0.427083`（改善 0 格、持平 4 格、
  变差 2 格，见 `ROUND32_VERDICT.json` `answers.5...read.violation_shrank`）；
- 成本侧：ACTIVE_ROWS 未降低已计量成本；两格里成本上升的 CI 不含 0（s22 sampled +11064.0、
  s33 greedy +25709.25），其余四格的差都在 +77.625 B 以内（相对 ≈28 000 B 的量级 <0.3 %）。
  六格按符号：3 格更贵、2 格更便宜（-9.0、-62.625）、1 格恰好相同（s22 greedy 两臂都 0 B）；
  发送次数同为 3 格更多、0 格更少，且**成本变高的两格正是消息变多的两格**。

因此本轮**排除了**"相对改善但绝对仍违背"被误读的可能：这里连相对改善都没有。

## 6. 尚未建立 / 已被本轮削弱的结论

已被本轮证据**削弱或排除**的：

1. ~~"成本/任务信号没有真正进入 actor 更新"~~ —— 信号进入且被放大到
   `N/M` 倍（18/18 点，相对残差 2.9e-07），actor 在裁剪前梯度里的占比从 <20 % 升到
   最高 97 %；
2. ~~"零梯度行占据 mean 分母是当前唯一/主要瓶颈"~~ —— 把它去掉后**没有任何一格**任务—成本
   变好，一格明显变差、一格（s22 sampled）也更贵更差。**方向性反证**：加大 actor 权重在本
   设置下不产生"更好的取舍"，反而更容易停在持续发送的高成本区。

仍未建立（本轮不做干预，因此不能判定）：

1. 更多批数、或不同 λ/η_λ、或不同熵系数会怎样 —— 全部冻结未动；成本项尺度同理：本轮**读出**了
   它与任务项的幅度比（0.18284–0.424743，18/18 点），但没**干预**过它，因此不能由该读数判定改成多少能闭合约束；
2. "actor 占比上升"是否等价于"参数步更大" —— Adam 非线性 + 13/18 点两臂都被裁剪到同范数；
3. 独立泛化 —— 训练块与开发块都被多轮反复使用；无确认性显著性、无 δ=0 保证；
4. 全系统通信成本 —— 本轮计量只覆盖受控上行通道的已计量尝试字节；
5. sampled/greedy 不可互换：s33 的 sampled 两臂都 0.4375 损失，greedy 一侧 0.0、另一侧 0.5；
6. 以上六条的完整八条清单见 `ROUND32_VERDICT.json` 的 `not_confirmed`（含"把 3 个 seed 当总体"、
   "δ=0 的部署级保证"、"把 sampled 轨迹当成 greedy 的约束轨迹"）。

## 7. 七问简答与下一步

1. **是**：唯一主动变化就是 actor surrogate 的归约集合与分母（三段身份哈希 0 差异、
   `ALL_ROWS` 与冻结训练器逐步权重相同、seam/逐块差异全归类、`verified` 8 项全真）。
   独立复核另抓到一处**不影响任何测量**的来源缺陷（配对件被重写时丢了"启动前"标记），
   已按 §1.1 登记且写入 `ROUND32_VERDICT.json` 的 `registered_defects`。
2. actor 项梯度精确放大 `N/M`（18/18），actor 占比 0.95–20.4 % → 14.9–97.1 %，
   但 13/18 点两臂裁剪后同为 0.5，变的只有方向。
3. greedy：ΔY ∈ {0, 0, −0.5}、ΔC ∈ {−9, 0, +25 709}；sampled：ΔY ∈ {0, −0.0625, 0}、
   ΔC ∈ {+77.6, +11 064, −62.6}。绝对配对任务损失 ALL {0.5, 0.5, 0.0}、ACTIVE {0.5, 0.5, 0.5}。
4. **不一致**：s11 两臂几乎重合、s22 sampled 更贵更差、s33 greedy 明显退化。
5. **约束违背没有减少**；s22 的正 `g_hat` 批数从 13 增到 21，成本上升的两格 CI 不含 0。
6. 见上文"仍未建立"。
7. 唯一后续建议（仅建议、本轮不执行、需另行批准）：见 `ROUND32_VERDICT.json`
   `answers.7_next_single_experiment`。本轮**已经**读出组合优势里 λ·A_task 与 A_cost 的幅度比
   （0.18284–0.424743，18/18 点），所以下一步不再是尺度读数本身，而是**剩余任务损失的可回收上界**：
   把六格里每一格仍丢的任务按场景归因（承重首报 / 严格冗余 / 搜索侧零接触），沿用十九轮已有的
   分类方法作用在本轮 192 份评估记录上，回答"任何通信策略最多还能拿回多少任务"。
   **0 次环境启动、0 次 optimizer.step**，只用已落盘的 lifecycle 与 tape。
   该上界若接近 0，则控制器侧的目标函数改动（含尺度）都不值得再做；若不接近 0，才值得另案申请
   一次只改尺度的训练消融。任何 λ/熵/训练量调整都不在本轮授权范围内。

---

## 8. 产物

`EXECUTION_BASIS.md`、`ROUND32_PROTOCOL.md`、`CONFIG_AND_MODEL_MANIFEST.json`、
`RUN_PREFLIGHT.json`、`REDUCTION_MATH.md` / `REDUCTION_MATH.json`、`INIT_PAIRING*.json`
（含 `INIT_PAIRING_PRELAUNCH.json`：生效冻结提交里那六份带"启动前"标记的原件快照）、
`SYNTHETIC_UPDATES.json`（合成更新分账件）、
`POST_FREEZE_CHANGES.json`、`FIRST4_GATE_{ALL,ACTIVE}_ROWS_s11.json`、`FIRST4_CROSS_s11.json`、
`START_LEDGER.json`、`CHECKPOINT_INDEX.json`、`ARTEFACT_MANIFEST.json`、
`TRAINING_CURVES.json`、`PAIRED_COMPARISON.json`、`UPDATE_MECHANISM.json`、
`INDEPENDENT_RECHECK.json`、`ROUND32_REPORT.md`、`ROUND32_VERDICT.json`、`r32_*.py`。
原始 logs/tape/rows/权重留在 Git 外，`ARTEFACT_MANIFEST.json` 对 **4329** 份产物记
路径+大小+sha256（合计 1 215 719 316 B；分类计数：lifecycle 1040、transcript 1040、
tape 960、checkpoint 44、其它 1245 = 1040 行 + 192 批记录 + 索引/配对等）。
`CHECKPOINT_INDEX.json` 逐批登记权重哈希、优化器/λ/RNG/数据位置与
`data_order_position_matches_batch_record`，并登记 18 份更新前 checkpoint 的身份相符。

## 9. 停止

```text
新环境启动 = 1040（计划 1040，硬上限 1080，技术失败 0，储备 40 未动用）
新流 = 0（未生成任何新场景；只用既有 1300000..1300047 / 1301000..1301015）
真实训练 = 六条运行 × 32 批 = 192 批、2628 次 optimizer 步（逐批登记，机读计数见
           ROUND32_VERDICT.json counters）
合成 optimizer 更新 = 授权 32 / 实执行 40（上限 64，与真实更新分账）
纯计算梯度诊断 = 是（18 个更新前 checkpoint 的内存副本上 forward + autograd.grad；
                    诊断阶段 0 次 optimizer.step、0 个 .pt 写回）
远端发布状态 = 未推送；本轮 0 次网络尝试；该分支无 upstream
```

本地交付后停止：不续训、不调 λ、不改熵系数、不选中间 checkpoint、不加场景、不部署、不推送。

## 10. 交接摘要（下一轮起点）

- **交付头**：本轮最后一枚本地提交的完整 SHA 以 `git log --oneline -1` 为准（本文件不预填，
  避免写出猜的哈希）；分支 `exp/comm-actor-reduction-ablation`，**未推送、无 upstream**。
- **已被本轮否定的假设**（不要重复做）：① "actor 梯度被零梯度行稀释是主要瓶颈" —— 去掉稀释后
  0/6 格变好、2/6 格变差；② "成本/任务信号没进 actor 更新" —— `N/M` 放大在 18/18 点成立。
- **仍然开着的**：批数、λ/η_λ、熵系数、成本项**尺度取值**（本轮只读出比值 0.18284–0.424743，
  未做任何干预）、以及"剩余任务损失里有多少根本不在通信侧"。
- **建议的唯一下一步（0 启动）**：按场景归因六格剩余任务损失（承重首报 / 严格冗余 / 搜索侧零接触），
  量出通信侧可达的任务上界；它决定后续任何目标函数消融是否值得申请。方法与分类沿用十九轮。
- **可直接复用的资产**：18 份更新前 checkpoint（批 0/15/31 × 六条，含优化器/λ/RNG/数据位置）、
  192 份批记录、1040 行 + 960 tape + 1040 lifecycle、`ARTEFACT_MANIFEST.json` 的 4329 条路径+哈希、
  `REDUCTION_MATH.json` 的 11 项零启动核验、冻结 bootstrap 常量（PCG64 `20260930`、10000×16 索引、
  linear 分位）。
- **下一轮若继续，必须保持不变**：`v2_trainer` / `cost_math` / `v2_run` / 编码器 / 环境 /
  搜索与分配 checkpoint / `configuration_hash = e9b2c067db5951f2` / 数据顺序摘要
  `346f2cdde8d658f9`；封存块 `800000..800399`、`900000..900199` 仍不得触碰；
  启动账本按新一轮重新立账（本轮 1040/1080 已用尽正常额度与全部计划，**不得复用本轮账本**）。

