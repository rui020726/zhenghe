# ROUND57 报告 — 先验作用归因（只读）

分支 `analysis/round57-prior-effect-attribution`，HEAD 见提交。零操作：环境/仿真/新流/optimizer.step/checkpoint 写入/新 episode 行均为 0；CPU forward 与 autograd.grad 诊断次数见 MACHINE_VERDICT 与 ZERO_OPS（forward 1152×2+诊断；grad 诊断 3 次，均内存副本）。

## 任务A 输入核验
96+96 行、版本、ep128 检查点身份、任务集合与成本整数全部核验一致（INPUT_IDENTITY.json）。R53/R56 的 sampled 与 greedy 已分别核对动作、发送集合与概率——sampled 非常接近 greedy 是策略近似确定性所致，不是行标签错误。

## 任务B 成本分解（COST_DECOMPOSITION.json）
R56−R53 全场景闭合（ΔC=A−B+C），**C=0 全部重新确认**（共同目标计价逐字节一致）：
- s11 greedy：**+7704**，来自两个场景的额外 early 首报（1301000 t4+t5=5022；1301005 t5=2682）；
- s22 greedy：**+2694**，一个额外 late 首报（1301006 t4）；
- s33 greedy：**−7128**，三个省略首报（1301005 t1、1301007 t3、1301014 t5）。
R56−late 同口径分解已登记；未发现计价差。

## 任务C 同输入策略探针（SAME_INPUT_POLICY_PROBE.json）
1152 个记录决策点（192 tape），自重算与原记录最大误差 3.4e-07；去重后分 early/late 桶：
- **s11：early 均值 p 0.045→0.117（反方向 +0.072），3 例 SKIP→SEND，无 SEND→SKIP**——与 +7704 成本恶化一致；
- **s22：early 不变（0→0），late 1 例 SKIP→SEND**——与 +2694 一致；
- **s33：early p 0.024→0.012（1 SEND→SKIP），late 0.967→0.905（2 SEND→SKIP）**——与 −7128 改善一致。
同输入 forward 只描述策略映射，不构成闭环反事实；去重向量与重复出现已分开统计。

## 任务D 先验暴露（PRIOR_EXPOSURE_AND_SIGNAL.json）
- 实际运行时 gate_loss 已由 loss 接缝写入 batch 记录（真实日志，非重算）：96 批中 32 批非零，**最大仅 0.0151**——eligible 格上策略本来就以高概率 SKIP，先验信号天然微弱。
- ep0 处先验梯度 L1（对 trunk+action_head，纯计算、无 step）：s11 0.0081 / s22 0.0504 / s33 0.0315。
- 基础 PPO 梯度与先验梯度的逐 minibatch 分离：**不可判**（minibatch 组成/advantages 未持久化；按约不复制 PPO 重放）。
- loss 接缝核验：源码补丁 16 行与白名单一致、β=0.10 恒定、critic 隔离保持、账目无污染（g_hat/计费/critic target 不含新增项）。

## 判读（允许多解释并存）
- s11 成本增加来自 **early 桶 3 例 SKIP→SEND 的闭环漂移**（先验梯度极小不足以约束，且其 eligible 格本已高 π(SKIP)）；
- s33 的改善与先验方向一致（early+late 均有 SEND→SKIP）；
- s22 增加来自一个 late 桶新增首报；
- 证据**不支持** β 太小论、实现错误论或可观测性缺口论；也**不能**把 s33 改善归因于先验单独起效（闭环漂移与先验混同）。哪种机制主导：**尚不可区分**。
- 无实现问题证据；不提出调 β/147 的建议，本轮不给出新方法变量。

## 结论
development_go=false 维持；三个成本区间均含 0，不宣称显著改善或恶化；任务全保留仅适用于已使用开发块。保护文件字节复验未变（CF3AD198…/93A2F69F…）。
