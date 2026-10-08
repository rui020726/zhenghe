# ROUND56 报告 — soft early-report prior（单侧软先验，β=0.10）

分支 `exp/round56-soft-early-report-prior`。唯一方法变量：在 R53 critic 隔离 PPO 的 actor loss 上增加
`L_gate = (1/N_rows)·Σ q_ij·[−log πθ(SKIP|x_ij)]`（q=1 仅当真实自由首报且恢复 ready_step≤147），总 loss = L_R53 + 0.10·L_gate，β 事前锁定。

## 执行完整性

- 480/480 正常任务 complete（384 训练 + 96 候选评估；BC sampled 48 格复用 R53 交付）；技术储备 0/24；总启动 480 ≤ 504。
- 环境 480、episode 480、步数 480,000、PPO 批 96、optimizer 步 604、checkpoint 写入若干；无失败重试。
- 首 12 集+3 次真实更新后动态门通过；ep0 参数身份与 R53 gate receipts 一致（逐 seed 交叉核验）。
- 恢复/中断留痕：`LAUNCH_FAILURE_SANDBOX_RESOLVE_REGISTRY.json`（沙箱路径解析虚假拒绝，12 集后停止）、
  `RECONSTRUCTION_LOG.json`（从 96 条权威 batch 记录重建派生调用日志，标注 reconstructed）、
  `raw_launch_failure_dirty_worktree_20261008T045000Z`、`raw_launch_failure_sandbox_path_resolve_20261008T061500Z`（现场副本，SHA 校验）。

## 启动前窄测试（r56_gate_tests.py，9/9 通过）

β=0 时参数梯度与 R53 完全一致；门控梯度=β·p(SEND)（p=0.5 → 0.05）；ready>147/masked/非首报贡献 0；ready=147 惩罚、148 不惩罚；步内被 mask 的第二候选不计入；真实特征恢复 ready 与字段一致；门控梯度对两个 critic head 为 0 且 actor 路径可导；g_hat/计费/奖励账目不含新增项；新版本 checkpoint 识别 R56 目标并拒绝旧目标。

## 结果（16 开发场景；所有臂全场景 on-time 6/6，ID 集合与 immediate 16/16 一致）

| seed | R53 greedy | R56 greedy | Δ(R56−R53) | 95% CI | 低于 4838.25 |
|---|---|---|---|---|---|
| 11 | 4995.375 | 5476.875 | +481.50 | [0.00, 1255.50] | ✗ |
| 22 | 3923.250 | 4091.625 | +168.38 | [0.00, 505.12] | ✓ |
| 33 | 4851.000 | 4405.500 | −445.50 | [−904.50, 0.00] | ✓ |

sampled 与 greedy 同 seed 均值几乎相同（同 R53 行为）。**development_go = false**（s11 未达标；三 seed 方向不一致）。

## 判读

1. 先验通过了实际 actor 训练路径（同一 minibatch loss 内、保留梯度的 log_softmax、actor-only 梯度；对 critic head 梯度为 0）。
2. 相对 R53：s33 成本显著改善方向（−445.5，进入门槛内）；s11 恶化（+481.5）；s22 轻微恶化（+168.4）。三 seed 方向不一致，均未达 95% 显著。
3. 是否纠正了 early 发送：部分——s33 的改善来自少发；但 s11 反而多发。未观察到对 late 区间 SKIP 的系统性保留。
4. development_go=false；本轮仅为已使用开发块上的筛选性结果，不构成确认、泛化或部署证据。
5. β=0.10 是事前实验值：s33 改善与 s11 恶化并存说明该先验的作用依赖 seed 特定的策略状态，不能据此调整 β 或 147。

## Git 外原件

raw/ 下权重（ep0/ep64/ep128/rolling）、tape、rows、日志均为 Git 外原件；路径、大小、SHA256 见 Git 外清单（本目录 JSON 与 R53/R55 同口径登记）。
