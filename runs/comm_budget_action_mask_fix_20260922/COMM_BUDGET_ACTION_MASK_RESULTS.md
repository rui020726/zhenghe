# 预算可执行动作的采样前硬 mask：修复结果（M0–M3）

日期：2026-09-22。分支 `fix/comm-budget-action-mask`，交付头 `7f9929c83`，总纲 v2.9，执行指令 `COMM_BUDGET_ACTION_MASK_FIX_NEXT_ROUND.md`，协议 `COMM_BUDGET_ACTION_MASK_PROTOCOL.md`（先于任何 episode 提交）。

**本轮实测**：episode 8（8 条既有流各 1 集，8,000 环境步）、新流 0、训练 0 次、optimizer.step **0**、H=20 spillover 0。trainer = `onpolicy-v3-budget-mask`，mask schema = `budget-sequential-mask/1`，tape schema = `onpolicy-tape/2`。

## 一、修的是什么

policy-owned timing 的 collector 把所有非静默上界候选当作二元自由动作；预算不足的 SEND 会被采样、写进 joint log-prob、进 rollout 与 PPO surrogate，直到 `CommRelay._emit` 才被拒。**事后删除被拒样本不成立**：能否被拒取决于同一步内先前采到的 SEND，按已实现动作选择性删项会改变联合概率。修法是把约束挪到抽样之前，并按 relay 实际执行顺序走一条 shadow credit。

## 二、8 条既有流的行为闭合（固定初值，0 optimizer.step）

| 流 | tape | 候选 | 动作前可执行 | 顺序后真双动作 | 被额度 mask | 采到 SEND | 请求 | 实际发射 | policy 拒付 | 花掉字节 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `700010` | 1000 | 2132 | 17 | 14 | 2118 | 10 | 10 | 10 | 0 | 26,928 |
| `700018` | 1000 | 1136 | 19 | 18 | 1118 | 11 | 11 | 11 | 0 | 27,114 |
| `700020` | 1000 | 3052 | 29 | 25 | 3027 | 11 | 11 | 11 | 0 | 27,780 |
| `700022` | 1000 | 2674 | 26 | 23 | 2651 | 11 | 11 | 11 | 0 | 28,080 |
| `700031` | 1000 | 2166 | 22 | 20 | 2146 | 11 | 11 | 11 | 0 | 28,608 |
| `700033` | 1000 | 3686 | 33 | 32 | 3654 | 10 | 10 | 10 | 0 | 27,192 |
| `700042` | 1000 | 3726 | 17 | 17 | 3709 | 11 | 11 | 11 | 0 | 29,280 |
| `700046` | 1000 | 2083 | 34 | 32 | 2051 | 11 | 11 | 11 | 0 | 29,220 |
| **合计** | 8,000 | **20,655** | **197** | **181** | **20,474** | **86** | **86** | **86** | **0** | — |

- 请求发射 86 == 实际 policy 发射 86，执行端 budget/capacity 拒付 0：被采到的 SEND 都是发得出去的。
- 顺序效应：单看"下一次发送"有 197 个候选可付，按真实执行顺序走过额度后只剩 181 个真双动作点，差 16 个只能由顺序推导出来——事后过滤拿不到这一层。
- 修复前同一批流在 60 步小实验里请求 180 次并发射 47 次、留下 133 次 budget 拒付；修复后同类拒付恒为 0（`test_budget_action_mask.RelaySeamTests`）。

## 三、门禁判定

| 门禁 | 判定 | 关键证据 |
|---|---|---|
| M0_production_and_history_isolation | 通过 | 生产改动仅 ['integrated_sar/comm.py', 'integrated_sar/comm_policy.py', 'integrated_sar/tests/test_comm_budget_action_mask.py']；默认值 comm_ack=False、timing=legacy_shared_bound；完整集成套件 690 项通过 |
| M1_action_execution_agreement | 通过 | collector 套件 17 项通过；requested=executed=86；policy 拒付 0 |
| M2_ppo_probability_agreement | 通过 | logged/recomputed 最大差 0.0 ≤ 1e-06；masked 项 likelihood/entropy=0；critic 样本保留；未做事后过滤 |
| M3_smoke_closure | 通过 | episodes 8/8 启动、技术失败 0、optimizer.step 0、封存块 clean=True |

**全部通过：是**

## 四、测试

| 套件 | 数量 | 结果 |
|---|---:|---|
| integrated_sar：charge 契约与 gate 同源（本轮新增） | 14 | 通过 |
| integrated_sar 全量（AGENTS 要求的最低验证） | 690 | 通过（跳过 2） |
| 本轮 collector/接缝/PPO 套件 | 17 | 通过 |

## 五、身份与卫生

- trainer tag `onpolicy-v3-budget-mask`；候选顺序：relay._observation order (sorted target ids), which is also the order the transport emits the policy's selection in。
- feature layout 不变（True）；`attempt_charge_bytes`/`send_executable` 只进契约与训练 tape，不进网络输入。
- resume 需要 trainer tag + mask schema + tape schema 全同；旧 checkpoint 只可前向读，**不得恢复 optimizer 续训**（其权重受"给发不出的动作算梯度"塑造）。
- 未提交 checkpoint / 录制流 / 逐 episode 日志；他人未跟踪文件（`architecture/`、`runs/coordinator_baselines_20260917/dev_summary.json`、另两轮 run 目录里的 rows/_ 前缀文件）未修改、未删除、未提交。
- 封存块 `800000..800399`：盘上 artefact 0 个（应为 0）；`900000..900199` 已消费，只作既有证据。

## 六、读法限制（不能从本轮得到的结论）

- smoke runs fixed initial weights for one episode per stream: it measures the action set, not task performance, and does not rank policies
- the actionable counts are not comparable to the send-actionability audit's 101 both-actions-free points: that audit replayed a rule policy and counted every offered row, while this smoke runs a fixed initial controller that requests nearly every affordable send, so the credit empties sooner
- no trained checkpoint was resumed, and none was created
- fixing the action space does not by itself say learning will beat the rules; the next question is a fresh run from initial weights, which needs its own authorisation

## 七、下一件事（不自动执行）

动作空间修正后，"当前真双动作自由样本"这一量第一次可测（本轮 8 流 20 655 候选里 181 个，占 0.88%）。它同时给出一个约束：可学习样本极少，任何"从初值重训"的协议都必须先重新定义真实双动作样本数、训练上限与强规则对照，而不是沿用旧的环境步口径。本轮不训练、不恢复 H=20，是否提出该协议由使用者决定。

## 八、复跑

```bash
cd runs/comm_budget_action_mask_fix_20260922
python -m unittest discover -s . -p 'test_*.py' -v   # collector / 接缝 / PPO / 身份
python smoke_budget_mask.py --mode check            # 只读计划，不启动
python smoke_budget_mask.py --mode run              # 8 流 × 1 episode，0 optimizer.step
python mask_manifest.py
python report_mask.py
cd ../.. && python -m unittest discover -s integrated_sar/tests && python -m integrated_sar
```

