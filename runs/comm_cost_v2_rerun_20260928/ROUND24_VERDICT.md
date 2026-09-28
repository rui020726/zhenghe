# ROUND24 判定：修复版 N4 从初值重跑（有界开发 pilot）

日期：2026-09-28。执行依据 `ROUND24_COMM_COST_V2_RERUN.md`（`docs/comm-cost-v2-rerun-plan@86cff022`，
blob `5d8ae2a5`）。分支 `exp/comm-cost-v2-rerun`；起点 `b7217d1`（本地修复 tip，其完整 tree 与
规划端核验的远端修复头 `64cddcffc6f8796f132df8b8e003f005d0bdd749` 逐条目一致，31/31 顶层
blob/tree SHA 相同，证据 `START_EQUIVALENCE.json`）。本文件把执行状态与科学观察分开，不把
"跑完"当作科学验收通过。

## 1. 执行状态：execution_complete = true

| 项 | 实际 |
|---|---:|
| 启动账本 | **672 / 672**（正常计划全额用完） |
| 失败启动 | 0（技术储备 28 次未动用，硬上限 700 未触及） |
| immediate reference 重跑 | 64（48 训练 + 16 开发场景，每场景一次） |
| 从初值训练 | 3 seed × 128 集 = 384（seed 11/22/33，各 32 批） |
| 学习模型开发评估 | 3 seed × 3 块 × 16 场景 = 144 |
| 五条规则重跑 | 5 × 16 = 80 |
| 合成 optimizer.step | 28（Adam 12 + SGD 16，门禁用，与真实训练步分列） |

- immediate reference（本机重跑，同场景同配置）：**on_time 6.000/场景、14 939.2 B/场景**（16 开发场景均值）。
- 旧 Stage C 的 677 次启动与全部产物只读、未动用、未退还；本轮账本独立。
- 每行落盘件闭合核验：成本账本（tap == ledger）、任务 tap（== `window_metrics.on_time`）、
  ACK 结构关闭，672/672 行零违例；每场景每模式恰一次评估；三块评估命名含
  run_id/训练种子/场景种子/timepoint/eval_mode/attempt，greedy 与 sampled 未混表。

## 2. 正确性门禁：全部通过

| 门禁 | 结果 |
|---|---|
| 零环境合成门禁（ROUND23 反例套件在本轮代码身份下重跑） | **20/20 通过**，0 失败 0 错误；合成 optimizer.step 28；静态路径审计通过；门禁身份与冻结身份一致（`9cfc0650…`） |
| 启动前冻结 | 生产三棵树零 diff；场景块碰撞审计通过并把同块登记为 intentional_development_reuse（上轮 1682 个产物使用过同块）；与旧 Stage C 机读设置逐项 diff **仅剩已登记差异**；`configuration_hash e9b2c067db5951f2` 与旧冻结**一致** |
| 首4集门禁（seed 11，计入配额） | **G1–G7 全过**：四模块参数集合与 optimizer 一致且冻结权重哈希不变、通信权重确已更新；初值权重下 log_prob 重算最大差 4.8e-07（≤1e-6）；从落盘件独立重算奖励/GAE/组合优势/ĝ 与总账闭合；对偶公式方向正确；critic 有限且全 mask 无伪样本；纯计算进程恢复前向一致；样本计数闭合 |
| 逐批审计（3 seed × 32 批） | seed 22、33 直接通过；seed 11 报 1 条**记账字段**问题（见下），其余全过 |
| 从初值开始 | 三个 `ep0` 里程碑与新鲜 `Controller(种子)` **逐参数字节相等**；v1 checkpoint 恢复被 trainer 拒绝（合成测试锁定） |
| 每批证据链（只读复核） | 逐批参数链 31/31 一致；cost_head 权重 32/32 批均变化且预裁剪梯度每批 > 0（min 0.067，max 5.11）；likelihood 每批 ≤ 4.8e-07；冻结 checkpoint 哈希全程不变 |

**登记的账本病态（不影响任何训练量）**：冻结版 `v2_train.py` 的 `module_digests_before`
按训练进程起点计算一次，续跑进程写出的批记录里该字段对 batch 1..N-1 是进程起点值。
`ppo_update` 不读它；逐批完整的 `controller_parameters_before/after`、模块梯度、λ/g、
likelihood 均为逐批正确值。冻结身份锁定训练所用代码，运行中不改 harness，因此用独立的
只读复核（`batch_chain_review.py` → `BATCH_CHAIN_REVIEW_s*.json`）闭合该缺口。
旧 `sc_recover.py` 的旧提交/旧 manifest 硬编码恢复入口按 spec §2.5 禁用；本轮恢复入口
（`v2_recover.py`）只读本轮落盘件，本轮未发生需要它的故障（0 次调用）。

## 3. 约束诊断（仅来自训练采样策略，不作 greedy 保证）

| seed | g_hat > 0 的批数 | 平均 g_hat | λ 轨迹（每批 rollout 时） |
|---|---:|---:|---|
| 11 | 23/32 | +0.3125 | 1.0000 → 1.4750 |
| 22 | 13/32 | +0.1484 | 1.0000 → 1.2375 |
| 33 | 23/32 | +0.2969 | 1.0000 → 1.4500 |

δ=0 的实验目标在训练中基本持续被违反（ĝ 多批为正），对偶 λ 相应上升；这是诊断读数，
不构成对 greedy 策略的约束保证，也不证明任务约束成立。训练期采样策略成本长期贴近
29300 B 预算（seed 11/33 ≈ 28.2–28.4k，emissions ≈ 11/集；seed 22 后半降至 19.4k、
emissions 7.25），即采样策略本身并未稳定降本。

## 4. Sampled 主表（观察训练策略本身；与训练同一 masked 随机策略）

| 块 | on_time Δ（95% 描述性 CI） | bytes Δ/场景（95% CI） | 任务 改善/平/退化 | 字节 改善/平/退化 | 联合 dominance |
|---|---|---|---|---|---|
| s11 final_sampled | −0.4375 [−1.000, −0.063] | +13 189.5 [+12 879, +13 485] | 0/12/4 | 0/0/16 | 0/16 |
| s22 final_sampled | −0.1250 [−0.313, 0.000] | +896.2 [−2 556, +4 506] | 0/14/2 | 9/0/7 | 9/16 |
| s33 final_sampled | −0.4375 [−1.000, −0.063] | +13 260.0 [+12 933, +13 589] | 0/12/4 | 0/0/16 | 0/16 |

CI 为 16 场景（聚类单位）10 000 次重采样的百分位区间，种子 20260928，仅作描述。
**没有任何 sampled 块在任一场景改善任务**；三个 seed 中两个支出显著高于 immediate。

## 5. Greedy 另表（部署规则诊断；与 sampled 不同表、不混统计）

| 块 | on_time Δ | bytes Δ/场景（95% CI） | 任务 改善/平/退化 | 字节 改善/平/退化 | 联合 dominance |
|---|---|---|---|---|---|
| s11 final_greedy | −0.5000 [−1.063, −0.125] | +13 268.6 [+12 947, +13 584] | 0/11/5 | 16/0/0 | 0/16 |
| s22 final_greedy | −0.5000 [−1.063, −0.125] | −14 939.2 [−15 155, −14 741] | 0/11/5 | 0/0/16 | 11/16 |
| s33 final_greedy | **+0.0000 [0, 0]** | **−12 540.8 [−13 841, −11 123]** | 0/16/0 | 0/0/16 | 16/16 |
| s11 init_greedy | −0.5000 | +13 259.6 | 0/11/5 | 16/0/0 | 0/16 |
| s22 init_greedy | −0.2500 | −9 963.8 | 0/12/4 | 0/0/16 | 12/16 |
| s33 init_greedy | −0.5000 | +13 250.2 | 0/11/5 | 16/0/0 | 0/16 |

规则对照（同 16 场景，同表不混入 learned 行）：always_hold −0.500/−14 939.2；
event −0.4375/+13 123.1；**late_discovery_gate +0.000/−10 101.0（16/16 联合 dominance，
仍是最强规则）**；periodic_20 +0.000/−356.2；random_send −0.2500/+13 105.5。
（improved/tied/degraded 是任务与字节的**联合**分类，不等于"任务改善数"；上表已分列。）

**事后观察（不构成选择）**：s33 final_greedy 与 late_discovery_gate 在全部 16 场景任务差
均等（0），字节均差 −2 439.8/场景 [−4 134, −380]，但 2 个场景更差（1301003 +5 148、
1301004 +7 770）。这是对单个 seed 单个块的事后观察，不挑选 seed/模式，不据此宣称总体最优。
s22 final_greedy 在这些场景上零支出（行为上等价 always_hold）。greedy 与 sampled 的模式差
距很大（s22：0 vs 15 835.5；s33：2 398.5 vs 28 199.2），说明对偶约束估计描述的是采样策略，
不能自动搬到 greedy 规则上。

## 6. 科学结论与边界

1. 执行完成、正确性门禁全过；**本轮资源内未建立学习收益**：没有任何 learned 块在任一开发
   场景改善任务，三个 sampled 块中两个支出显著高于 immediate。
2. 唯一零任务损失的学习块是 s33 的最终 greedy（16/16 平任务、−12 540.8 B/场景），它相对
   最强规则 late_discovery_gate 平均更省但 2/16 场景更差；以 exploratory 观察登记。
3. 16 个开发场景、每场景每模式一次 sampled 评估只作探索；训练 seed 不是新独立场景，
   3×16 ≠ 48 个独立场景；零差、CI 含 0、退化 bootstrap 区间均不构成 δ=0 的总体非劣证明。
4. 旧 v1 结果只在独立历史表说明修复前后（`runs/comm_cost_stage_c_20260927/
   STAGE_C_VERDICT_ERRATUM.json`），不并入本轮平均数，也不把两轮差值归因到四个修复中的
   某一个。
5. 未调 λ/η/δ/超参，未开封存块，未扩样，未挑选最佳 seed/checkpoint/模式，无训练中途
    harness 改动。结果不理想不等于应回到搜索优化主线。

## 7. 交付物索引

机读：`CONFIG_AND_MODEL_MANIFEST.json`、`START_LEDGER.json`、`SCENE_REGISTRY.json`、
`SYNTHETIC_GATE.json`、`START_EQUIVALENCE.json`、`FIRST4_GATE_s11.json`、
`BATCH_AUDIT_s*.json`、`BATCH_CHAIN_REVIEW_s*.json`、`TRAINING_CURVES.json`、
`CHECKPOINT_INDEX.json`、`COMPARISON.json`、`UNCERTAINTY.json`、`ROUND24_VERDICT.json`。
原始权重/日志/tape/rows 不入 Git，manifest 记录路径与哈希（`CHECKPOINT_INDEX.json` 含
ep0/64/128 里程碑与 rolling 的 sha256）。
