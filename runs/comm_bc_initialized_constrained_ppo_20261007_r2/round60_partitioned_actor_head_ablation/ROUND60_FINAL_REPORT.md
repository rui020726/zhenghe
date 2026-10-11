# ROUND60 最终交付报告：early/late 分区 actor 输出头消融（固定计划已完成）

本报告取代本目录早前“固定计划未执行”的 `ROUND60_REPORT.md`（该文件保留为历史）。分支 `exp/round60-partitioned-actor-head-ablation`。机器结论 `ROUND60_MACHINE_VERDICT.json`，配对统计 `raw/ROUND60_FINAL_STATISTICS.json` 与 `raw/ROUND60_STATISTICS.json`，账目 `raw/runtime/ROUND60_LEDGER.json`，probe 偏差登记 `ROUND60_PROBE_RECONCILIATION.json`。

## 结构干预与配置（唯一改动）

共享 trunk 与双 critic 不变；唯一结构改动是把 R58 的单一共享 `action_head` 换成两个各自独立的 `early_head`/`late_head`，均初始化为对应 seed 的 R58 ep0 action_head 的精确拷贝（独立参数对象、不抽 RNG）。路由只用合法 19 维输入与冻结常量 `ready_step=round(f[18]*1000)−round(f[0])`，`≤147` 走 early、`>147` 走 late；采样、greedy、似然重算、只读评估共用同一 `PartitionedController.forward`。β=1.00/ready_cut=147 gate prior 保留，经路由后的 early 头作用。活体配置（gae_lambda=1.0、eta_lambda=0.5、gamma=1.0、delta=0.0、data_order_seed=20260927、b0=10000、lr=3e-4、clip=0.2、entropy=0.01、value_coeff=0.5、max_grad_norm=0.5、epochs=4、minibatch=256、episodes_per_update=4）从 R58 ep128 checkpoint 的 training_state 逐字段核验，非模块默认。critic→trunk 梯度隔离用 R58 同一 `critic_input_detached` 上下文包住每次更新。

## 固定计划执行（576/576 完成）

独立 R60 接线：`r60_execute.py` 复用冻结 R46/R42 仿真与 r40_common 原子 claim/计数 builder，但绑定到独立 R60 raw 根与独立身份（task/branch/training_version/freeze、claim/slot 根、quota 576 正常 + 23 储备 + 硬顶 600 + 历史 probe 1）；不复用 R58 分支/版本 claim 校验器，不把 R58 claim 混入 R60 计数。零环境 CLI 正向验收通过后才运行。

- 训练：3 seed × 128 episode = **384**，每 seed 首 4 集计入 128，首批更新后立即跑真实门禁（通过才继续），只评估固定 ep128。三个门禁全 passed：双头路由 active（early/late 自由样本均 >0）、初值 == R58 ep0、optimizer 覆盖 12 参数、五模块、likelihood 对齐、value loss 有限、对偶 λ 更新、critic→trunk 隔离、actor 通路连通、checkpoint 身份。
- 评估：NEW × 3 seed × {greedy,sampled} × 32 场景 = **192**。
- 合计 **576/576 正常启动完成，0 技术失败，0 储备消耗**。真实 optimizer step 合计 **556**（seed 11=184、22=180、33=192，与 R58 的 556 一致）。λ：seed 11=1.0、22=1.0、33=1.125（batch 22 起，与 R59 审计一致）。

## 五点收口

1. **576 是否全部完成**：是。384 训练 + 192 评估全部 complete，0 failed_technical，0 reserve。
2. **消费**：历史非计划 probe=1（`ROUND60_PROBE_RECONCILIATION.json`，一次真实 1000 步 episode，非 claim/计划任务/失败/样本，不复用）；正常任务=576；技术失败=0；合成 optimizer 更新：上一会话验收报告 1 次，本轮真实训练用真实更新、合成更新 0；累计尝试含 probe=577，硬顶 600，余量 23。
3. **初始策略等价与结构差异**：初值前向在真实 576 行输入面板上与 OLD 逐比特相等（logits、value、greedy/sampled 动作），early=late=OLD action_head；init-equivalence 证明路由更新在 gate 关闭时复现冻结 ppo_update 的 loss（`ROUND60_INIT_EQUIVALENCE.log`）。训练后三个 seed 的 early_head 与 late_head 参数均已分化（early≠late），结构干预真实生效。
4. **三 seed 保任务/降本（NEW−R58，greedy 主读）**：
   - seed 11：任务 190→190，费用 Δ0 [0,0]，与 R58 **逐比特相同**（含 emission 序列，32 场景 0 差异）——分区未改变 seed 11 的开发块行为。
   - seed 22：任务 191→191（保持），费用 Δ−72 B/场景 [−216, 0]。
   - seed 33：任务 191→190（**丢失 1**），费用 Δ−390.56 B/场景 [−871.125, 0]。
   - sampled（次读，单列）：11 相同、22 Δ0、33 Δ−300.56 [−673.6875, 0]（任务保持）。
   - 所有费用 CI 均含 0（不宣称显著）；seed 11 费用为 0。故 **未建立跨 seed 一致的费用下降**（seed 11 持平，22/33 下降但不显著）。
5. **early 关键目标变化**：唯一发生变化的目标是 scene `1700027` / `piloc-e1-target-5`（ready_step=134，early，即 R59 标记的关键早目标）。
   - 相对 R58：仅 seed 33 greedy 丢失它——NEW 在该场景 SKIP 了 target-5 的 early 首报（step 134，R58 在此 SEND），target-5 随即 deadline_expired；其余 seed/模式保持。
   - 相对 immediate：seed 11、33 greedy 丢失（immediate 完成它），seed 22、33 sampled 保持（与 R58“seed 11 相对 immediate 丢 target-5”的模式一致）。
   - 相对 late：seed 22、33 sampled **救回**它（late 规则丢失、NEW 完成），seed 11、33 greedy 与 late 持平。
   - 即：分区 **未统一保住 early 关键目标**——seed 33 greedy 以少发（SKIP early 首报）换来了该早目标的任务损失与费用下降。

## 边界

评估块 1700000..1700031 为已使用的开发块，不是独立确认集。256 格只读复用（R58 同 seed/模式 192 + immediate/late 64）未计为新启动。CI 含 0 不称等效也不称显著。本结果只支持“整体分区 actor 结构在该开发块上的表现”：未证明参数共享干扰是唯一根因、不构成泛化/正式非劣/部署依据。early 目标仍有价值，不能把 late 标签当安全删除真值。性能差不是技术失败，未剔除、未重跑、未调参。

## 保护与提交

两个 ROUND46 脏文件 SHA256 维持 `cf3ad198…`、`93a2f69f…`，未修改、未暂存。早期非合规尝试（5 个 batch 曾绕过 critic→trunk 隔离）已隔离保留于 `_quarantine_attempt1_criticdetect_missing/`，未删除，合规重跑覆盖。本地显式点名提交代码、窄测试、统计脚本与冻结/账目/verdict/report；不提交权重、tape、原始仿真日志；不推送、不合并、不部署。完成后停止：不调参、不扩训、不挑 seed、不自动开启下一实验。
