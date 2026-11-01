# ROUND72 收口报告：ROUND71 原件验收与 greedy 行为未变的机制诊断

分支 analysis/round72-calibration-behaviour-diagnostic；证据绑定 HEAD fe53d476efb56baaae194d68d7f749a546b14389（ROUND71 交付）。
本轮为零环境诊断：只读解析已落盘的 JSON/JSONL，不加载模型、不前向、不采 RNG、不训练、不重放 PPO、不写 checkpoint、不发新 claim。

## 一、机制证据（核心结论）

在已使用的开发块 1800000..1800031 上，对 R68 trained_alpha 与 R71 new_ep128 两个臂、192 格、共 1098 个匹配的自由首报决策逐条比较落盘 tape：

- early 区 804 个决策的最大概率差为 0.0，逐比特相同。这是结构性结果：R66 干预冻结了共享 trunk 与 early_head（requires_grad=False）， critic 校准只改两个价值头，够不到 early 前向路径。
- late 区 294 个决策全部发生变化（最大差 0.1143），但 0 次 argmax 翻转。R71 侧最小 |p-0.5| 边界余量为 0.0401，单决策最大只走到边界的 0.4865，从未跨过 p=0.5。
- greedy：1098 个决策 0 次翻转，每个 seed 32/32 格在步序、特征、动作上完全一致。
- sampled：14 次动作翻转，全部落在 late 区，且都发生在概率已变化的决策上，与概率位移方向一致，不是边界穿越。
- beta=1.00 的 early 门控先验在所有 gate_diag 更新上 prior_trainable_grad_l1 = 0.0：先验只作用于路由到冻结 early_head 的 early 首发行，对可训练参数（late_head + 双 critic）没有梯度通路。
- 校准把 critic 梯度 L1 均值压低约一个量级、late_head L1 约四成，但不足以把任何 late 决策推过边界。

## 二、greedy 为何完全不变

early 路径被冻结 => early 概率逐比特不变；R71 的 critic 头校准只能通过 late_head 梯度间接影响 actor，而 late 头虽在 294/294 个已访问 late 自由态上移动，却都因边界余量过大而未跨域。因此三个 seed 的 greedy 首报动作、按时目标集合与计费成本在 192 格上与 R68 完全一致（任务差 0、成本差 0、CI 均 [0,0]、无目标丢失）。

与 R67 的对照：R66 曾在 p 约 0.999 饱和；R68 的 late_init_alpha=0.10 把 late 概率移到 0.54-0.99，因此本轮的机制是“边界余量”，不是“饱和”。

## 三、五项只读核验（据现有源码与产物）

1. 预算：comm_budget.py 是唯一的硬执行器，所有臂（学习臂与规则臂）共用同一 byte 额度与容量；计费口径为 载荷 x 订阅数，丢包也全额计费，额度不被击穿，拒付计入 budget_refused_sends / budget_refused_bytes / capacity_refused_sends。29300B 为配置注入值，非源码字面量。开发评估块上 immediate 均值 14356.5B/场景（最大约 15876）、学习 greedy 4066.5B/场景，均远低于 29300，故该块预算不绑定；但训练用的是另一组场景 1300000..1300047，开发块 immediate 低于 29300 不能推出训练预算从未绑定，训练块绑定性缺少每集 tape 拒付计数，标为未核验，不断言。
2. 优势契约（活体值自 checkpoint index 核验，非 0.95 旧默认）：gamma=1.0、GAE lambda=1.0、eta_lambda=0.5、lambda_init=1.0；双 critic 回归 GAE 回报（returns=A+V，gamma=1 即无折扣未来和）并以价值为基线；组合优势 = lambda_dual*A_task - A_cost；无任何 advantage 标准化；成本回报 = billed_bytes/B0；对偶更新 lambda_next=max(0, lambda+eta*g_hat)。标准反例成立：成本基线含预期成本并不抵消成本策略梯度，故不得宣称基线中和了成本压力。只有落盘的 late_actor_grad_l1 / critic_grad_l1 可量化 actor 贡献，RMSE~1.01 是价值拟合、不是梯度噪声标准差。
3. effective logit：落盘 send_probabilities 是路由后 head logits 的 softmax；路由 ready_step=round(f[18]*1000)-round(f[0]) 仅用合法 19 维；真实顺序预算 mask 在选择前把不可付候选强制 HOLD（不花 RNG）；beta=1.00 先验只是训练期辅助损失，不进入推理期 logits。头缩放 0.10 只说明初始网络项缩放，不证明后续不能跨零；未缩放先验也不能套用完整边界符号不变。0 次翻转仅对这套固定方案描述，不称“结构锁死”。概率变化、训练差异、sampled 评估随机性分列陈述。
4. 信念来源：19 维 comm-send-state/1 的每个特征都是发送方所持消息、发送方自身账本/空口、或当前预算/容量的属性，明确禁止接收端信念、机器人位置、场景 ID、目标 ID 进入。always-hold 完成 170/183 是受控首报消融结果（on_time 170、spent 0），不是 92.9% 信息路径占比。能拟合 V(s) 不证明观测足以区分 SEND 与 SKIP 的动作价值。
5. 数值与来源校正：greedy 对 late 规则的 168.5625B/场景 = 5394/32，全部来自 scene 1800000 单场景，其余 31 场景相同，CI [0, 505.6875]，不是理论可优化空间；late 保住开发块 183 目标不推出所有 late 消息无价值或帕累托最优；训练教师块 1300000..1300047、开发评估块 1800000..1800031 分开；R71 计费 70693 行是离线拟合的 trunk 前向账本（上限 72000），不是 teacher 数据本身，teacher 为 R68 的 384 个训练 tape；sampled 动作 RNG 键为 runner.eval_action_seed(seed, scene, mode)，跨臂与 ROUND66 同键配对。

## 四、sampled 与跨 seed 方向

sampled 才出现行为差异：s11 成本 -155.8B/场景（CI 含 0）、s22 +331.1B/场景（CI 不含 0）、s33 -642.2B/场景但丢失 1 个目标。三 seed 成本方向不一致，且与落盘的逐决策 late 概率位移方向相符，故不把差异全部归因于采样方差。

## 五、不能支持的结论

仅支持“整体分区 actor 结构 + R71 critic 初始校准在该已使用开发块上”的行为描述。不支持泛化、正式非劣、部署；不证明参数共享干扰是唯一根因；不宣称晚期消息普遍无用或可安全删除；CI 含 0 不等价于等效；early 目标仍有价值。

## 六、正确性缺陷与未解释簿记

未发现影响已报结果的正确性缺陷。一项簿记现象登记待独立只读跟进：update.likelihood_alignment.max_abs_gap 在两轮全部 96 个 batch 记录上逐比特相同，部分可由 float32/float64 求和残差解释（单自由候选步恰为 0），但对 6 个 batch（每 seed 的 b11、b13，两候选均为 late）未完全解释；该检查容差 1e-6 两轮均通过，不用作任何科学论断的证据。

## 七、实际消费与 HEAD

本轮实际消费全为 0：0 环境构造、0 模型加载、0 前向、0 RNG、0 PPO/optimizer、0 checkpoint 写入、0 新评估、0 新 claim、0 历史非计划 probe。R60 的 608/600 超限与 R67 的 >=9419/5000 前向超限为历史偏差，原样保留，不作为本轮余量继承。576 项 R60 有效结果与 R68/71 统计为历史登记，仅只读复用，不计入本轮新启动。证据 HEAD fe53d476efb56baaae194d68d7f749a546b14389。

机器可解析验收包见 ROUND72_MACHINE_VERDICT.json 与 ROUND72_RESOURCE_LEDGER.json；逐决策证据见 ROUND72_BOUNDARY_AUDIT.json、ROUND72_VALIDATION.json、ROUND72_TRAINING_SIGNAL_AUDIT.json。