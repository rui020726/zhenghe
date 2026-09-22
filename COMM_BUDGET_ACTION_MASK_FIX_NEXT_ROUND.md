# 下一轮：预算可执行动作的采样前硬 mask 修复

日期：2026-09-22。对应总纲 v2.9 → v3.0。状态：**已执行完毕（历史指令，勿重复执行）**。

> 交付在 `runs/comm_budget_action_mask_fix_20260922/`：契约三字段与 `_emit` 同源、采样前顺序 hard mask、
> tape `actionable=free`（故复用第五轮 PPO 循环）、trainer tag/schema 身份隔离、30 项回归测试与 8 流 smoke。
> M0–M3 全过；0 训练、0 新流、optimizer.step=0、H=20 仍冻结。下一件事（是否从初值重开训练）由使用者决定。

## 1. 起点、目标与停止边界

从 `analysis/comm-h20-terminal-attribution@b2061ae7089b0bb5eeeaa8d3df2b34eabf40f89c`
创建 `fix/comm-budget-action-mask`。先核对远端 SHA、工作区和未跟踪文件，阅读 AGENTS、ARCHITECTURE、
PROJECT_STATUS、PROJECT_CHARTER v2.9，以及 `runs/comm_h20_terminal_attribution_20260922/` 的两份审计。

已确认的缺陷：policy-owned timing 的 PPO collector 把所有非静默上界候选当作二元自由动作；预算不足的
SEND 仍可被采样、写入 joint log-prob、rollout 与 PPO surrogate，直到 relay `_emit` 才被拒绝。
训练器无法逐样本区分“选择发送”与“实际不可发送”。

本轮只修这个正确性问题并验证，不训练模型、不继续 H=20 spillover、不生成新流、不改 29,300/87,900 B
预算、成功率、奖励、网络结构、feature 维度、场景或生产默认值。允许最多 8 条既有训练流 smoke，
每流 1 episode、optimizer.step=0；不得打开 `800000..800399` 或重新使用 `900000..900199` 作测试。

## 2. 为什么不能只在 PPO 更新时事后删拒绝样本

禁止把 relay 拒绝的 SEND 在 episode 结束后简单从 likelihood 中移除。拒绝与否取决于同一步前面已经采样的
SEND，事后按已采样动作删项会选择性改变联合动作概率，不能保证 logged behaviour probability 与更新时的
policy probability 仍描述同一个分布。

正确语义是**采样前、按实际执行顺序顺序 mask**：

1. 每个候选先得到真实 attempt charge；
2. 用动作前剩余额度与容量建立 shadow budget；
3. 按 learned policy 实际返回/执行的同一顺序处理候选；
4. 当前候选付不起时，唯一合法动作是 HOLD，记录为非 actor 决策，likelihood/entropy 权重为 0；
5. 当前候选可付时才采样 SEND/HOLD；若采样 SEND，立即从 shadow budget/capacity 扣除，再处理下一个候选；
6. relay 仍保留最终硬门，作为执行安全边界和一致性检查。

这样学到的是“在当前可执行集合中何时发”。预算耗尽后的 forced HOLD 不产生 actor 梯度，但 earlier SEND/HOLD
仍通过后续 reward/GAE 承担预算消耗的长期后果；critic 仍可用整个状态训练。

## 3. 通信契约：精确费用与动作前可执行性

在 `SendItem` 增加向后兼容的元数据，名称和语义写入 docstring：

- `attempt_charge_bytes: int | None`：当前 sender-visible message 的 `estimate_message_bytes(message) × subscribers`；
- `send_executable: bool | None`：在该 observation 形成时、尚未执行本步任何 policy send 的情况下，
  此候选作为“下一次发送”是否通过真实 budget/capacity 门；
- `send_block_reason: str | None`：`allowed / budget_exhausted / capacity_step` 或明确的不可用原因。

`CommRelay._observation()` 必须使用与 `_emit()` 同一 `_message`、`estimate_message_bytes`、subscriber fan-out 和
`CommBudget.can_send/why_blocked` 计算，不能复制近似公式。observation 形成与本步发送之间没有环境推进，
所以费用必须与随后 `_emit` 的实际扣费逐字一致；不一致 fail closed。

这些字段先作为执行约束和审计元数据，**本轮不加入神经网络特征**：

- `comm-send-state/1` 19 维、`/2` ACK 23 维保持不变；
- 旧 checkpoint 仍可用于只读 smoke/行为兼容检查；
- 未来是否增加 `attempt_charge_fraction` 属于新的 observation/表征实验，需新 layout 和重新训练，不能混入本修复。

手工构造的旧测试 item 可用 `None` 保持兼容；但启用预算 mask 的 collector 遇到费用元数据缺失必须报错，
不能把 `None` 当 0 B 或默认为可发送。

## 4. collector：顺序预算 mask 与可复算 tape

扩展修正后的 `OnPolicyCollector`，新增显式 `budget_mask` 开关；默认 `False`，保证历史脚本、规则臂与旧行为
不会因导入新代码静默改变。新修复训练/评价入口必须显式设 `budget_mask=True`。

顺序规则：

- 候选顺序必须与 learned policy 交给 relay、relay 实际执行的顺序完全相同；用回归测试锁住。
- 静默上界 forced 项仍不是 actor 动作，保持旧语义；policy-owned timing 下该集合为空。
- 对每个候选保存 shadow 判断前后的 budget/capacity、精确 charge、initial executable、effective actionable、
  block reason、采样动作和最终 requested 状态。
- 预算/容量 mask 的候选动作记录为 HOLD，`usable/actionable=0`，不进入 joint log-prob 或 entropy。
- 可执行候选采样 SEND/HOLD，`usable=1`；只有采样 SEND 才消耗 shadow credit。
- greedy 评价和 stochastic 训练共用同一 mask；规则 arms 不使用该 collector 的新 mask。

tape/flatten 至少持久化：

- `attempt_charge_bytes`（逐 target）；
- `send_executable_initial` 与 `send_block_reason_initial`；
- `actionable`（顺序 mask 后真正双动作自由集合，兼容字段 `free` 可保留但必须同义）；
- `shadow_budget_before/after`、`shadow_capacity_before/after`；
- `actions`、`requested`、`log_prob`、`forced`。

PPO 重算使用 rollout 时保存的 `actionable`。对同一已采样动作序列，顺序 mask 是其条件概率的一部分；
旧 likelihood alignment 检查继续成立。全候选都被 mask 的 step：actor log-prob=0、entropy=0、actor surrogate
无梯度；value target/critic 样本仍保留。不得删除整个 step。

## 5. 新的训练入口与 checkpoint 身份

不要把历史结果目录的旧 trainer 原地改名成“已经正确”。在新 run 目录提供明确入口/适配层，版本例如
`onpolicy-v3-budget-mask`，训练和 greedy evaluation 都显式构建 `budget_mask=True` collector。

- 新 trainer 可复用旧网络结构、GAE、PPO likelihood helpers 和超参数；
- resume 必须要求新 trainer tag、action-mask schema 与 tape schema 匹配；
- 旧 v1/v2 checkpoint 可用于前向 smoke，但**不得恢复 optimizer 后继续训练**；未来正式训练必须从同一初始权重
  方案重新开始，不能把已受错误动作空间训练的 400k checkpoint 当作修复后续训起点；
- run signature/manifest 写入 mask schema、候选顺序、feature layout、trainer tag 和代码身份。

本轮不启动正式训练，也不创建“最好 checkpoint”。

## 6. 回归测试（必须先红后绿）

至少覆盖：

1. relay observation 的 charge 与随后真实 `_emit` 扣费完全一致（不同 subscriber、revision、payload）；
2. 剩余字节恰好等于 charge 可 SEND，少 1 B 被 mask；
3. 单候选预算不足：固定 HOLD、未请求发送、log-prob/entropy 为 0、relay budget refusal 为 0；
4. 两候选共享额度：第一个采样 SEND 后第二个被 mask；第一个 HOLD 时第二个仍可采样；
5. capacity=1 的同型顺序约束；
6. 多候选全部 requested charge 之和不超过动作前额度，执行端无 policy budget/capacity refusal；
7. stochastic 与 greedy 路径使用同一 actionable set；
8. sampled joint log-prob 与更新重算误差 ≤1e-6，翻转 tape/actionable 时检查能失败；
9. all-masked step actor likelihood/entropy 为 0，但 value/critic 样本仍保留；
10. `budget_mask=False` 与修复前固定输入的 action/request/log-prob 逐项相同；规则策略、默认 relay 不变；
11. 缺少 attempt charge 时 fixed collector fail closed；
12. tape 八项字段和候选闭合：actionable + forced + budget/capacity masked = candidates（互斥定义写清）。

测试不能只调用复制的审计函数，必须覆盖 relay observation → collector → relay `_emit` 的真实接缝。

## 7. 最多 8 episode 的行为 smoke（0 optimizer.step）

单元/合成测试通过后，使用动作审计同一 8 条训练流做 fixed collector smoke，每流最多 1 episode：

- 固定初始化权重与 RNG，policy-owned timing、29,300 B、原成功率/ACK 配置；
- 不更新参数，明确 `optimizer_steps=0`；
- 报 candidates、initial executable、effective actionable、masked credit/capacity、sampled SEND、实际 emission、
  policy refused、forced/refused、log-prob alignment；
- `requested_by_policy = executed_policy_emissions` 在预算/容量维度闭合；终态抑制等非预算原因另列；
- 不把任务完成数作性能结论，不与旧模型排名。

启动上限 8，技术失败也计入；不补 seed。原始逐步日志不入 Git，只归档位置与摘要。

## 8. 验收门禁与后续分支

### M0 生产/历史隔离

默认 relay、规则臂、`budget_mask=False` 行为不变；feature 维度与 checkpoint 输入不变；生产代码变更只增加
可执行性元数据，不改变未启用 fixed collector 的发送结果。

### M1 动作—执行一致

合成测试全部通过；fixed learned path 中预算/容量不可执行 SEND 从未被采样或请求；执行器同类 policy refusal=0；
顺序 shadow 与实际 charge/顺序逐项一致。

### M2 PPO 概率一致

actionable set 可从 tape 审计；sampled/logged/recomputed 最大差 ≤1e-6；masked 项 likelihood/entropy=0；
critic step 保留。禁止以事后样本过滤代替。

### M3 smoke 闭合

8 条流按实际完成数报告，候选分类恒等式闭合、无混合 run signature、optimizer.step=0、技术失败如实计数。

任一门禁失败：只提交失败归因/修复，不预登记训练。全部通过后才可建议下一轮“修复后从初值重新训练”的
协议；该协议至少重新定义真实双动作样本数、训练上限和强规则对照，但本轮不自动启动。

H=20 spillover 保持冻结：本修复不改变其历史数据，也不因 M0–M3 通过自动恢复实验。

## 9. 交付与提交纪律

目录建议 `runs/comm_budget_action_mask_fix_20260922/`：

- `COMM_BUDGET_ACTION_MASK_PROTOCOL.md`（先于 smoke 提交）；
- `COMM_BUDGET_ACTION_MASK_RESULTS.md/.json`（M0–M3、8 流 smoke、0 optimizer.step）；
- `COMM_BUDGET_ACTION_MASK_MANIFEST.json`（源码/config/stream/checkpoint 摘要、默认值、生产 diff）；
- 新 trainer/eval 入口、tape schema 和相关测试。

提交拆分：① observation/charge 契约 + 测试；② 顺序 mask/tape/PPO 对齐 + 测试；③ 新入口与 smoke；
④ 结果/总纲/状态。每个问题独立提交，显式暂存文件，不提交 checkpoint、stream、原始日志或用户未跟踪文件。

运行受影响测试、完整 `integrated_sar` 测试和 `python -m integrated_sar` 冒烟；报告本轮实测数。
推送并核对远端 SHA 后停止。**不训练、不跑 H=20 完整 episode、不追加 observation feature。**
