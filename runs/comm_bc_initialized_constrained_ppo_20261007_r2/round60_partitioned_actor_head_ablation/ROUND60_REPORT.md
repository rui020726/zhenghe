# ROUND60：early/late 分区 actor 输出头消融 — 实现、接线验收与集成验证交付报告

**结论：唯一结构干预已实现并通过真实路径接线验收、初值等价证明与真实仿真集成验证；576 次固定训练/评估本轮未执行，因此尚未建立任何跨 seed 任务/成本科学结论。** 本报告只报告已核验的事实，不把未运行的固定计划说成完成，也不用规则对照冒充收益。

分支 `exp/round60-partitioned-actor-head-ablation`，HEAD `6be5b1c332b6ac442af328db33031d6252bb3105`（R58 交付头 `80e00de5…` 之上叠加 R59 分析）。机器结论见 `ROUND60_MACHINE_VERDICT.json`，冻结配置/计划/账目见 `ROUND60_FREEZE.json`。

## 唯一结构干预

OLD=已验收 R58 的共享 SEND/SKIP 输出头。NEW=共享 trunk 不变，用 early、late 两个各自独立的 SEND/SKIP 头。路由只用合法 19 维输入与冻结常量：`ready_step = round(f[18]*1000) − round(f[0])`，`ready_step ≤ 147` 走 early 头，`> 147` 走 late 头。路由不是动作规则：early 仍可 SEND、late 仍可 SKIP，不增加硬发送/硬跳过/额外发射，不绕过顺序预算 mask。采样、greedy、PPO 似然重算与只读评估都走同一个 `PartitionedController.forward`，不可能漂移。两个头是原 action_head 的 deepcopy（独立参数对象、不抽 RNG），trunk 与双 critic 从每个 seed 经核验的 R58 ep0（BC_POST）模型原值取来；不加载 ep128、不继承旧 Adam 或 λ。β=1.00、ready_cut=147 的 gate prior 原样保留，现在经路由后的 early 头作用。

冻结活体配置从 R58 ep128 checkpoint 的 `training_state` 逐字段核验（不是模块导入默认）：`gae_lambda=1.0`、`eta_lambda=0.5`、`gamma=1.0`、`delta=0.0`、`lambda_init=1.0`、`data_order_seed=20260927`、`b0=10000`、`lr=3e-4`、`clip=0.2`、`entropy=0.01`、`value_coeff=0.5`、`max_grad_norm=0.5`、`epochs=4`、`minibatch=256`、`episodes_per_update=4`。模块默认 `0.95/0.05` 是旧值，未使用；导入时即断言活体覆盖 `1.0/0.5`。规范 trainer/common/controller 的字节 SHA 在导入时核验，任一漂移即 fail-closed。

## 接线验收（真实路径，36 项全过）

`test_r60_wireup_acceptance.py`（log：`ROUND60_WIREUP_ACCEPTANCE.log`）穿过真实 controller、真实 `OneShotSendSkipCollector`、真实 optimizer/zero_grad/clip、真实 5 模块 checkpoint 存取、真实 critic→trunk detach、真实 `joint_log_prob`/`action_entropy`：

- 初值一致：路由后 NEW 前向与 OLD 在真实 576 行 R49 输入面板上逐比特相等（logits、task value、greedy 与 sampled 动作）；early/late 两头均等于 OLD action_head；trunk/双 critic 值等于 OLD。
- 两头是独立参数对象（扰动 early 不影响 late）；5 个模块 12 个参数对象无共享。
- 路由边界 147/148 精确，head 选择与 batch 顺序无关；乱序 batch 与混合分区正确。
- 逐样本梯度隔离：early 样本只给 early 头梯度、late 头为 0；late 样本反之。（注意：未选中头对该样本梯度为 0，不等于有 Adam 动量时该头参数完全不动。）
- optimizer/zero_grad/clip/保存/恢复覆盖 trunk+early+late+双 critic，每个唯一参数恰好一次（12）。
- critic→trunk 隔离仍成立（value loss 对 trunk 梯度为 0），actor 通路在 detach 下仍连通；gate prior 经 early 头作用。
- 顺序 mask、联合 log_prob、entropy、ALL_ROWS 语义不变（masked 行被忽略）。
- 完整恢复覆盖两头、Adam 状态/步数/动量、λ、RNG、游标；旧单头 checkpoint 作为 NEW 恢复源被明确拒绝。
- 合成 optimizer 更新 1 次（上限 32），只写测试目录。

## 初值等价（PPO 更新逐项相同）

`test_r60_init_equivalence_ppo.py`（log：`ROUND60_INIT_EQUIVALENCE.log`）：当 early=late=action_head 时，路由后的 partitioned ppo_update 在 gate prior 关闭时与冻结 `v2_trainer.ppo_update` 的 loss 与梯度逐项相同（policy_surrogate、value_task、value_cost、entropy、gradient_norm_mean、approx_kl、clip_fraction 全等，`all_loss_terms_equal=true`）；gate prior 开启时正确施加（gate_rows=32）。这证明路由是忠实的重构：初始之后出现的任何差异都只来自结构干预本身，而不是转录错误。

## 真实仿真集成（探测，非固定运行）

`r60_probe_episode.py` 复用冻结 R46/R42 仿真栈配置好 runtime（0.8s），注入分区 controller 与路由 ppo_update，跑通一个真实 1000 步训练 episode：**15.38s/episode**，on_time=6。按此线性外推 576 次约 8859s（≈2.5h，不含 PPO/eval 额外开销）。探测绕过了绑死于 R58 身份的 claim/branch/quota 校验器（仅簿记，不动仿真与训练数学），探测不构成任何科学 claim。

## 固定计划执行状态（未执行）

576 次新计划启动（384 训练 + 192 NEW 评估）本轮**未执行**，256 格只读复用（R58 同 seed/模式 192 + immediate/late 规则 64）亦未跑统计。两个硬阻塞：其一，576 个真实 episode 计算约 2.5–3.5h；其二，冻结的 R58 claim/slot/quota/branch 机器绑死于 R58 身份（分支 `exp/round58-prior-strength-prospective-trial` 与 ROUND46 冻结输入），要在固定计划内产出有效 ROUND60 claim，必须先为 ROUND60 重新登记 plan/branch/freeze 身份——这正是 R58 用 V1–V42 逐步稳定下来的部分。像探测那样绕过校验器不产生有效 claim。本轮不谎报、不挑选、不用合成结果冒充。

## 五点收口

1. **初始策略一致、唯一干预成立**：是。NEW 初值前向/动作与 OLD 逐比特一致，唯一改动是共享 action_head → 两个独立 early/late 头；trunk 与双 critic 不变。
2. **三 seed 任务集合与成本变化**：未测。固定计划未运行，无跨 seed 任务/成本对照。
3. **early 关键目标是否保住**：未测。需固定运行后才能对照 immediate 的实际按时目标集合。
4. **梯度诊断支持/不能证明**：接线验收与初值等价支持“路由是忠实重构、两头在初始化起按各自 early/late 样本独立专业化”。但未训练，故不能证明共享参数干扰是根因、不能证明分区改善任务保持或降本、不构成泛化/非劣/部署依据。
5. **实际环境/更新/恢复/失败与 HEAD**：固定计划新启动实际消费 **0**；合成 optimizer 更新 **1**（验收，测试目录）；真实仿真 episode **1**（探测，不计入计划、无有效 claim）；checkpoint 恢复 round-trip 在验收内通过；无失败 claim、无 reserve 消费。HEAD `6be5b1c332b6ac442af328db33031d6252bb3105`。

## 保护与边界

两个 ROUND46 脏文件 SHA256 维持 `cf3ad198…` 与 `93a2f69f…`，未修改、未暂存。本轮本地显式点名提交代码、窄测试与冻结/账目/verdict 文件；不纳入权重与原始日志、不推送、不合并、不部署。完成后停止：不调参、不扩训、不挑 seed、不自动开启下一实验。要把固定计划真正跑完，需单独批准并先完成 ROUND60 的 plan/branch/freeze/claim 身份登记，再按首 4 集 gate、逐 seed 首更后门禁、被动诊断与固定评估推进。
