# 第二十三轮：Stage C 训练目标接线修复与独立反例（0 新环境启动）

日期：2026-09-28。状态：待执行。规划端已核验远端 `feat/comm-cost-stage-c` = `cac8c60051f7cb89379104941113060758c36ab1`，读取该固定提交源码。不是仅根据聊天推断缺陷。

## 1. 核心决定

不批准续训、调 lambda/eta、改变 delta、强规则独立确认或新矩阵。先修复下列已经在固定源码中定位的正确性问题。

当前 Stage C 的677次启动与观测结果保留；“计划已跑完”仍是运行事实，但不能把它作为正确 N4 成本目标的有效学习检验，也不能据此判断成本学习无效、强规则不可超越或欠训练已排除。新增勘误与证据状态，不覆盖旧 verdict。

本轮允许：实验训练器/检查器最小修复、合成 tensor 的 optimizer.step、只读重算已有 tape/checkpoint、测试与文档、普通提交推送。禁止新环境 episode、真实 rollout、更新已有通信模型权重、开启封存块、生产三棵树改动。合成更新与真实训练计数分列。

## 2. 已定位的三个实质问题

路径均相对 `runs/comm_cost_stage_c_20260927/`，行号针对 cac8c60，修复后以新源码映射为准。

### F1. 成本在进入 GAE 前取负，组合时又减一次

- `sc_trainer.py:61`：`rewards_cost = -bundled_cost / B0`。
- `sc_trainer.py:77` 调用 A1 的 `actor_advantage`，定义为 `lambda*A_task - A_cost`。
- `sc_run.py` 的 cost_stream 是 relay airtime 计数器的 after-before，正常为非负尝试费用。

规范中的 A_cost 估计**正成本**优势。当前实际送入的是负成本回报优势，因此在零 critic 的简化路径上，组合相当于 `lambda*A_task + A_positive_cost`，不是所登记的成本最小化方向。

修复只选择一套贯穿账本→cost return→cost critic→actor的定义：本轮继续采用正归一化成本、`lambda*A_task-A_cost`，不要通过改目标函数或倒置 lambda 来包容旧代码。独立核对 Stage B、A1及其他调用者的影响，只新增限定语，不重写归档结果。

### F2. 新 cost_head 没有进入 optimizer

- `sc_controller.py:25-26` 的 parameters 仅返回 `inner.parameters`。
- `sb_controller.py` 将 cost_head 单独构造，未加入 inner.parameters。
- `sc_train.py:54` 及恢复路径 `sc_trainer.py:287` 用上述不完整列表构造 Adam；gradient clipping 也使用同一列表。

必须让共享 trunk、action_head、task value_head、cost_head 的参数恰好各出现一次，同时不包含搜索/分配模块。核对 fresh 和 resume 两条路径。optimizer.zero_grad/梯度裁剪都覆盖全部预期参数，不能让遗漏 head 的 grad 跨 minibatch 静默累积。

已有 checkpoint 的 cost head 输出可能随共享 trunk 改变，不能用输出变化证明 head 自身训练过。应按模块权重、参数对象身份、optimizer状态和受控梯度验证。

### F3. 非终止转移的 next-value 全被置零

- `sc_trainer.py:70` 对每个 decision step 构造全零 bootstrap。
- `cost_math.gae` 使用 `delta_t = r_t + gamma*bootstrap_t*(1-terminal_t)-V_t`。

全零仅适用于真正终止的续接值；非终止步应使用下一 decision state 的对应 critic 值。调用端并未在 gae 内自动移位补齐 next V，因此不能以“末步为 terminal”解释整个向量都为零。

明确 bootstrap 参数是逐转移 next-value 还是只表示末尾截断值；选择一个契约并校验所有相关调用者，不混用两种解释。冻结 rollout 的旧 value，计算 target/GAE 不引入反向梯度。

## 3. 为什么旧门禁没有抓到

`sc_check.py:86/94/101` 复制了同样的负成本输入、全零 bootstrap 和同一个组合函数。它证明的是“两个实现重算一致”，并非数学目标正确。

G1（约135–155行）把 optimizer 参数形状与同一错误的 controller.parameters 形状比较，两边同时漏掉 cost_head 也会通过；整体模型哈希变化又可由其他模块造成。

保留旧 passed 原件，新增 gate_validity_correction。以后 checker 的规范预期必须来自独立数学例子与按模块建立的完整参数集合，而非拿被测函数自己的输出当真值。不能通过修改测试预期迎合错误实现。

## 4. 必须先失败、后通过的真实路径测试

### T1 成本方向：经过 batch_samples 和 PPO surrogate

构造非负原始 cost_stream，SEND: Y=1,C/B0=0.2；HOLD: Y=0,C=0。p=sigmoid(theta)。独立闭式期望：

    grad L_max = (lambda-0.2)*p*(1-p)

p=0.5 时 lambda=0/0.1/1/10，对应 -0.05/-0.025/0.20/2.45。确保测试经过真实 `batch_samples` 的费用构造、GAE和actor组合入口，不能只调用已正确的 A1 actor_advantage。

隔离本测试的critic和entropy梯度，在PPO旧策略点的未clip区域核对最大化目标与最小化loss的反号。通过精确动作枚举/autograd/有限差分交叉验证；旧负费用实现必须失败。

### T2 optimizer 参数完整性

在同一进程按模块 named_parameters 建完整期望集合，比较对象ID集合，检查无缺项、无外来参数、无重复。每个模块构造可证明非零梯度的合成loss，执行一次optimizer.step，分别验证权重变化与state登记；不能要求任意真实batch中每个参数都必然变化。

测试故意漏 cost_head、重复 trunk、混入无关参数时必须失败；保存/恢复后重复同样测试。旧缺头列表必须失败。全mask时只要求actor直接梯度正确为0，不要求共享trunk在critic学习时不变。

### T3 非零 value 的双步 GAE

独立解析例：r=[0,1]，V=[0.4,0.6]，terminal=[false,true]，gamma=1，GAE_lambda=0.95，next_V=[0.6,0]。

    delta=[0.2,0.4]
    advantage=[0.58,0.4]
    return=[0.98,1.0]

旧全零 next-value 得 A0=-0.02，必须被捕获。另测 GAE_lambda=1 时完整终止序列的return与Monte Carlo总回报相符、非终止截断的末尾bootstrap、多个episode绝不互串。不能只用全零value测试。

### T4 奖励归属时间：契约核验项，不预先认定所有汇总错误

`_bundle` 当前把 reward step 9 分配给 decision step 10（决策步7/10），即归给后一个决策。若value和action定义在该步动作前，那么第9步已经发生的奖励不能充当第10步动作的后果。

画清 relay.tick、动作采样、env.step、reward产生与tape的先后关系；给出同tick、两决策之间、首决策之前、末决策之后四类案例。只把动作发生后到下一动作前的奖励计入对应转移，无法归给任何动作的前缀收益单列；不能仅为总和闭合把过去奖励塞给未来动作。

若decision-step GAE使用压缩时间，明确GAE_lambda按决策还是物理tick生效。gamma=1只消除折扣差异，不自动消除GAE时基差异。以现有协议为准说明，不在本轮顺便调参。

T4 如发现真正错误，独立修复/提交并补测试；如只是日志标签偏移，给出证明和准确命名。总量账本正确与时序信用正确是两件事。

## 5. 只读评估影响，不重新训练

在执行端现有初值/中期/最终checkpoint上检查cost_head权重摘要是否完全不变、optimizer是否缺少其state；不加载外部未知checkpoint，不更新真实权重。

对已有少量tape固定抽取规则、从原始cost/task/value重算正确target和advantage，记录与旧结果差异。此重算只证明影响，不产生修复后策略的反事实成绩。无法恢复的字段标missing。

保留16开发场景的规则结果为原配置的探索性观察，不能把学习侧缺陷自动归咎于规则臂；规则身份和执行路径独立核验后才能引用。late_discovery_gate 的零观测任务损失仍不证明总体非劣。

原对比表 improved/tied/degraded 是任务与字节的联合分类，不能简写成任务改善数；分别列 task_sign、byte_sign、联合dominance，避免误读always_hold的11个“改善”为任务改善。不要借机扩大统计分析范围。

## 6. 版本与证据地位

新增训练版本与loss/GAE/optimizer schema身份；旧训练状态不能静默resume进修复版。未来若获准重跑必须从初值开始，不能把这轮旧checkpoint当正确N4热启动。

旧677次启动全部保留，不退还预算，不拿剩余23次启动自动跑新实验。本轮0新增环境启动，修复通过也不自动启动三seed训练。

给旧 verdict 加独立勘误引用或状态索引：execution_complete=true；registered_objective_valid=false（以复现确认后状态为准）；旧 observed comparisons 保留，correct_N4_learning_verdict=not_established_due_to_implementation。不能直接覆盖原pilot_completed判定或删除旧passed文件。

## 7. 交付、发布与停止

新修复分支建议 `fix/comm-cost-stage-c-training-contracts`，从核验的Stage C头加本文档提交建立；若已有后续改动则保留并说明。原阶段实验依赖代码与manifest保留，通过新版本/新目录表达修复，保证旧数据仍有可定位源码。

交付：TRAINER_CORRECTNESS_FINDINGS.md、独立反例与old_fail/new_pass证据、OPTIMIZER_PARAMETER_AUDIT.json、GAE_TIME_CONTRACT.md、CORRECTED_TAPE_RECOMPUTE.json、GATE_VALIDITY_CORRECTION.md、修复版版本manifest。

每个问题独立提交。报告真实执行的合成测试与合成optimizer.step次数，环境启动/真实训练步为0；不执行会启动仿真的全套smoke。生产三棵树相对Stage C头零diff。

普通push与远端SHA核验获准，网络失败有限重试；不force、不触碰其他工作。最终交付后停止，等待用户决定修复版小规模接线/重跑额度。
