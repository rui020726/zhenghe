# 预算可执行动作的采样前硬 mask：修复协议（先于任何 episode 写定）

日期：2026-09-22。分支 `fix/comm-budget-action-mask`，起点文档提交
`31da5f7c36ed95b95d6752624bcbebd9f16221e6`（`docs/communication-budget-mask-fix-plan`，
后继 `analysis/comm-h20-terminal-attribution@b2061ae`）。对应总纲 v2.9 与执行指令
`COMM_BUDGET_ACTION_MASK_FIX_NEXT_ROUND.md`。产物目录 `runs/comm_budget_action_mask_fix_20260922/`。

**状态：协议冻结。本文在任何 smoke episode 之前提交；结果文档只能报告本文写明的量。**

## 1. 缺陷与本轮唯一的修复对象

`runs/comm_h20_terminal_attribution_20260922/COMM_POLICY_ACTION_MASK_AUDIT.md` 的 file:line 路径确认：
policy-owned timing 的 PPO collector 把**所有非静默上界候选**当作二元自由动作（`onpolicy.py:145-147`
的 `must` 是唯一 mask），预算不足的 SEND 仍可被 `torch.multinomial` 抽中（`onpolicy.py:175`）、
写入 `joint_log_prob`（`onpolicy.py:46-56`）、进 tape → `flatten` → PPO surrogate
（`train_onpolicy.py:113-216`），直到 `CommRelay._emit` 才被 `CommBudget.can_send` 拒绝
（`comm.py:862-886`、`comm_budget.py:68-81`）。执行端拒绝只进 episode 级计数，
样本级训练不知情。因此优化的是"想不想发"，不是"在当前预算下发不发得出"。

这与已修的 μ≠π 概率错配（`COMM_PPO_PROBABILITY_AUDIT.md`）**不是同一问题**，本轮只修前者。

## 2. 禁止的修法与规定的修法

**禁止**：episode 结束后把被 relay 拒绝的 SEND 从 likelihood 里删掉。拒绝与否取决于同一步内
已采样的 SEND；按已采样结果事后选择性删项会改变联合动作概率，不能保证 logged behaviour
probability 与更新时的 policy probability 描述同一个分布。

**规定**：采样前、按 relay 实际执行顺序做 sequential hard mask。

| # | 步骤 | 实现位置 |
|---|---|---|
| 1 | 每个候选带真实 attempt charge | 生产 `CommRelay._observation` 写入 `SendItem.attempt_charge_bytes` |
| 2 | 用动作前剩余额度/容量建 shadow budget | fixed collector `choose()` 内 |
| 3 | 按 learned policy 交给 relay、relay 实际执行的同一顺序处理候选 | `sorted(self._ledger)` 的 items 序，回归测试锁死 |
| 4 | 当前候选付不起 ⇒ 唯一合法动作 HOLD，记为非 actor 决策，likelihood/entropy 权重 0 | `usable=0`，不进 `actionable` |
| 5 | 可执行才采样 SEND/HOLD；采到 SEND 立即扣 shadow credit/capacity | 同上循环内 |
| 6 | relay 保留最终硬门，作执行安全边界与一致性检查 | `_emit` 不改语义 |

## 3. 契约字段（生产代码，向后兼容）

`integrated_sar/comm_policy.py:SendItem` 追加三个带默认值的字段（追加在末尾，不动既有位置参数，
遵守本项目"新字段只能追加或改 keyword-only"的既有教训）：

- `attempt_charge_bytes: int | None = None` —— 当前 sender-visible 消息的
  `estimate_message_bytes(_message(...)) × len(subscribers)`；
- `send_executable: bool | None = None` —— 该 observation 形成时、**本步尚未执行任何 policy send**
  的前提下，此候选作为"下一次发送"是否通过真实 budget/capacity 门；
- `send_block_reason: str | None = None` —— `allowed` / `budget_exhausted` / `capacity_step`，
  或 `charge_unavailable`（`_emit` 同样无法定价的情形，见 §3.1）。

`_observation()` 必须调用与 `_emit()` **同一个** `self._message(target_id, ledger, snapshot)`、
同一个 `estimate_message_bytes`、同一个 `self._subscribers` fan-out、同一个
`CommBudget.can_send/why_blocked`；不复制任何近似公式。observation 与本步 `_emit` 之间没有环境推进，
因此费用必须与实际扣费逐字一致——回归测试直接对比，不一致即 fail closed。

### 3.1 无法定价的候选

`_message` 在无 registry lookup 且 ledger 无 `pending_message` 时会抛错（`_emit` 同样会抛）。
此时契约字段写 `None/None/charge_unavailable`，**fixed collector 遇 `None` 必须报错**，
不得当作 0 B 或默认可发送。手工构造的旧 item 用 `None` 保持兼容，且 `budget_mask=False` 路径不受影响。

## 4. tape / 候选闭合

fixed collector 的 tape（schema `onpolicy-tape/2`）至少持久化：

`targets`、`attempt_charge_bytes`（逐 target）、`send_executable_initial`、`send_block_reason_initial`、
`actionable`（顺序 mask 后真正的双动作自由集合）、`forced`、`masked`（预算/容量 mask 掉的 target 及原因）、
`shadow_budget_before/after`、`shadow_capacity_before/after`（步级与逐候选判断前后）、
`actions`、`requested`、`log_prob`、`value`、`features`。

兼容字段 `free` 在 fixed collector 下**必须与 `actionable` 同义**（PPO 的 `usable` 取自它）。
候选闭合恒等式：`len(actionable) + len(forced) + len(masked) == len(targets)`，三类互斥。

`budget_mask=False` 时 tape 与修复前**逐键完全相同**（不加新键），使 M0 可以按 tape 直接比对。

PPO 重算（`drawn_versus_logged`、`update` 的 `usable`）改用 tape 里的 `actionable`。全候选被 mask 的
step：actor `log_prob=0`、`entropy=0`、actor surrogate 无梯度，但 value target/critic 样本保留，
**不得删除整个 step**。greedy 评价与 stochastic 训练共用同一 mask；规则臂不经该 collector 的新 mask。

## 5. 新入口与身份（不改历史 trainer）

历史 `runs/comm_tight_budget_20260919/{onpolicy,train_onpolicy}.py` 保持原样，不原地改名成"已正确"。
本目录提供 `budget_masked.py`（collector + tape）与 `train_budget_masked.py`（入口/适配层）：

- `TRAINER_TAG = "onpolicy-v3-budget-mask"`；`ACTION_MASK_SCHEMA = "budget-sequential-mask/1"`；
- 复用旧网络结构、GAE、PPO likelihood helpers 与超参（import，不复制）；
- resume 必须校验 trainer tag + action-mask schema + tape schema 三者匹配，否则拒绝；
- 旧 v1/v2 checkpoint 只做**前向只读** smoke 与行为兼容检查，**不得恢复 optimizer 后续训**；
- run signature/manifest 记录 mask schema、候选顺序、feature layout、trainer tag 与代码 blob 身份。

feature 维度不动：`comm-send-state/1` 19 维、`/2` 23 维；`attempt_charge_bytes`/`send_executable`
本轮**不进网络输入**（未来加 `attempt_charge_fraction` 属于新的表征实验）。

## 6. 回归测试清单（必须先红后绿；§6 逐条对应）

生产侧 `integrated_sar/tests/test_comm_budget_action_mask.py`：

1. observation 的 charge 与随后真实 `_emit` 扣费完全一致（不同 subscriber 数、revision、payload）；
2. 剩余额度恰好等于 charge ⇒ 可 SEND；少 1 B ⇒ `budget_exhausted`；`capacity_remaining=0` ⇒ `capacity_step`；
3. 默认路径（不启用 fixed collector）发送结果、规则臂、relay 行为逐项不变。

collector 侧 `test_budget_action_mask.py`（覆盖 relay observation → collector → `_emit` 真实接缝）：
4. 单候选预算不足：固定 HOLD、未请求发送、log-prob/entropy=0、relay policy budget refusal=0；
5. 两候选共享额度：第一个采 SEND 后第二个被 mask；第一个 HOLD 时第二个仍可采样；
6. `capacity=1` 的同型顺序约束；
7. 多候选 requested 的 charge 之和 ≤ 动作前额度，执行端 policy budget/capacity refusal=0；
8. stochastic 与 greedy 使用同一 actionable set；
9. sampled joint log-prob 与更新重算误差 ≤1e-6；翻转 tape 的 actionable 时该检查必须失败；
10. all-masked step：actor likelihood/entropy=0，critic/value 样本仍在；
11. `budget_mask=False` 与修复前固定输入的 action/requested/log-prob/tape 逐项相同；
12. 缺 charge 元数据时 fixed collector fail closed；
13. 候选闭合恒等式与三类互斥定义。

## 7. smoke（≤8 条既有流，0 optimizer.step）

单元/合成测试全绿后才跑，且**只跑这一批**：

- 流：动作审计同一 8 条既有训练流 `700010, 700018, 700020, 700022, 700031, 700033, 700042, 700046`
  （`runs/comm_send_actionability_20260922/`），每流 1 episode，共 8 次启动上限；
- 权重与 RNG 固定（同一初始化方案），`policy_owned_timing`、29 300 B、原成功率与 ACK 配置不变；
- 不更新参数，`optimizer_steps = 0` 显式断言；
- 报告 candidates、initial executable、effective actionable、masked by credit/capacity、
  sampled SEND、实际 emission、policy requested、refused/forced、log-prob alignment；
- 闭合判据：`requested_by_policy = executed_policy_emissions + policy_budget_or_capacity_refusals`
  （本轮要求后者为 0）；终态抑制等非预算原因另列；
- 技术失败计入 8 次上限，不补 seed；原始逐步日志不入 Git，只归档位置与摘要；
- **不以任务完成数作性能结论，不与旧模型排名。**

## 8. 验收门禁

| 门禁 | 判据 | 失败时 |
|---|---|---|
| M0 生产/历史隔离 | 默认 relay、规则臂、`budget_mask=False` 行为不变；feature 维度与 checkpoint 输入不变；生产改动只增可执行性元数据 | 停止并只提交隔离性修复 |
| M1 动作—执行一致 | §6 合成测试全过；fixed learned path 中预算/容量不可执行 SEND 从未被采样或请求；执行端同类 policy refusal=0；顺序 shadow 与实际 charge/顺序逐项一致 | 不进入 smoke 结论 |
| M2 PPO 概率一致 | actionable set 可从 tape 审计；sampled/logged/recomputed 最大差 ≤1e-6；masked 项 likelihood/entropy=0；critic step 保留；无事后样本过滤 | 修复不完整，登记失败归因 |
| M3 smoke 闭合 | 8 条流按实际完成数报告；候选分类恒等式闭合；无混合 run signature；`optimizer.step=0`；技术失败如实计数 | 只提交失败归因，不预登记训练 |

任一门禁失败：只提交失败归因/修复，不预登记训练。全过后才可**建议**下一轮"修复后从初值重训"的协议
（须重新定义真实双动作样本数、训练上限与强规则对照），本轮不自动启动。

## 9. 边界（不做什么）

不启动正式训练、`optimizer.step` 恒 0、不跑 H=20 spillover（保持冻结，本修复不改变其历史数据、
也不因 M0–M3 通过自动恢复）、不生成新流、不改 29 300/87 900 B 预算、成功率、ACK、奖励、网络结构、
feature 维度、场景与生产默认值；不打开 `800000..800399`，不把 `900000..900199` 用作测试；
不改 `integrated_sar` 之外的预算/信道语义；不提交 checkpoint、stream、原始日志或他人未跟踪文件。

## 10. 交付与提交拆分

`COMM_BUDGET_ACTION_MASK_PROTOCOL.md`（本文）→ ① 契约+测试 → ② 顺序 mask/tape/PPO+测试 →
③ 新入口与 smoke → ④ `COMM_BUDGET_ACTION_MASK_RESULTS.md/.json`、
`COMM_BUDGET_ACTION_MASK_MANIFEST.json`、总纲/PROJECT_STATUS/README 与把本轮指令标历史。
每步独立提交、显式暂存；末次运行受影响测试 + 完整 `integrated_sar` 测试 + `python -m integrated_sar`
冒烟，推送并核对远端 SHA 后停止。

```bash
cd runs/comm_budget_action_mask_fix_20260922
python -m unittest discover -s . -p 'test_*.py' -v     # collector / 接缝 / PPO / tape
python -m unittest discover -s ../../integrated_sar/tests -p 'test_comm_budget_action_mask*.py' -v
python smoke_budget_mask.py --mode check               # 先只读检查与配额账本
python smoke_budget_mask.py --mode run                 # 8 流 × 1 episode，optimizer.step=0
python report_mask.py                                  # M0–M3 结果
python mask_manifest.py                                # 代码/配置/流/checkpoint 摘要
```
