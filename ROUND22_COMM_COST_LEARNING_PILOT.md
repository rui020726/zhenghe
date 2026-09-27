# 第二十二轮建议执行 Spec：首轮有界成本学习 pilot（Stage C）

日期：2026-09-27。状态：供用户转交执行；不是训练已完成或正式非劣验收协议。

依据：用户报告 Stage B 本地头 `6ebefbe4e7a81182281807333d64d60cadf03198`，18/24 启动、接线检查通过、0 optimizer.step。规划端未独立核验该头源码。执行端必须核验本地完整提交与 Stage B 产物，不只引用聊天数字。

## 0. 远程交接与已建分支

本文件通过独立文档分支 `docs/comm-cost-learning-pilot` 发布。此分支基于旧文档提交 `72c4c3f71e07339bf6533fbd771aef3b2682103c`，**不是 Stage C 的执行代码起点**，不得 checkout 此分支后用旧代码训练。

执行端已经从已核验 Stage B 头建立 `feat/comm-cost-stage-c` 时，保留并复用该分支，无需重建或改名。fetch 文档分支后核对交接消息给出的完整 SHA，仅 cherry-pick 本次新增 ROUND22 文件的单个提交到执行分支；它不得覆盖 A/A1/B 的代码和结果。Git HTTPS 不通而 GitHub MCP 可用时，通过 MCP 按固定提交读取本文件完整内容并落盘，登记来源提交与内容摘要，不得只读截断输出或自行补写缺失段落。

原 spec 下文建议的 `exp/comm-cost-learning-pilot` 分支名不要求覆盖已经建立的 `feat/comm-cost-stage-c`；研究内容、672+28 启动额度与停止条件不变。本发布仅补文档交付，没有执行任何 episode 或训练。

## 1. 主线与授权范围

这是“冻结 PILOC 搜索与 CTDE 分配，学习任务信息发送时机以降低已计量成本”的首轮新目标训练。不是重开搜索研究，不是重复旧的只奖完成数训练。

收到用户转交本 spec 的执行指令后，可在独立实验分支实施最小训练 harness、合成 optimizer 测试、本文列明的真实运行与通信模型更新。只训练新通信 actor、task critic、cost critic；搜索/分配权重冻结且逐阶段核对 SHA256。不更改生产默认值，不引入 ACK/新拓扑/新 actor 特征，不关闭本地感知或 gossip。

在 `6ebefbe...` 或经核实的其后继建立 `exp/comm-cost-learning-pilot`，保护其他会话文件；若后继改动影响相关模块，先列清差异，不能默认为已验收行为。

允许普通提交与推送已核验的 A/A1/B 与本轮分支；禁止 force push。远端分叉或认证边界问题不覆盖他人，有限重试后本地交付。网络不是本地训练的科学门禁。

## 2. 先冻结目标、参数和实现身份

本轮实验设置（不作部署安全承诺）：

    c = C_measured / B0
    B0 = 10000 B
    delta = 0 个按时任务/episode
    g_hat = mean(Y_ref - Y_policy) - delta
    L_max = -E[c] - lambda * (E[Y_ref]-E[Y_policy]-delta)
    actor_advantage = lambda * A_task - A_cost

参考是同场景、同配置的固定 immediate，使用配对 episode reference。delta=0 是保守的**训练目标**，有限样本与对偶法都不保证真实期望或每局不退化。不得自动继承旧提案 delta=0.10。

训练初值 lambda=1；对偶步长 eta=0.05；每4个训练 episode 完成一次 rollout 更新后计算 `lambda_next=max(0,lambda+eta*g_hat)`。同一批 PPO 更新期间 lambda 固定，随后再更新它。验证集不更新 lambda，不用验证结果校准 eta。任何上界裁剪/平滑/学习率调度若拟采用，必须在启动前明确修订，不在运行中暗加；默认不加。

gamma=1，GAE lambda=0.95（变量名与对偶 lambda 区分）。固定成本单位；不分别对 task/cost advantage 做批内方差标准化。本版组合后也不做数据依赖的 advantage 标准化；记录此设置与旧 trainer 的差异。GAE bootstrapped return 不要求等于 episode 总量，账本闭合用原始 reward 和实际计费增量。

网络、PPO clip、optimizer/LR、epochs、minibatch、critic loss 权重、entropy 系数、gradient clipping 必须在任何真实 episode 前冻结成机读配置，并引用复用的现有版本。明确 entropy 是额外正则项，不隐去其对纯成本目标的影响。不允许只写“沿用默认值”；若复用版本间有歧义，先列一组选定值和依据，不启动后再选择。

所有训练状态保存 actor/双critic/optimizer/lambda/更新计数/数据顺序/全部相关 RNG/版本和归一化。旧通信 checkpoint 不加载恢复。本轮只从初值训练，不迁移任何旧最佳 seed。

## 3. 固定真实运行配置

继承 Stage B 的真实 PILOC normal-lossy 设置：6目标、1000步窗口、冻结搜索/分配 checkpoint、成功率0.7、ACK关、policy_owned_timing、无公共强制发送兜底、29300 B 上行硬预算、全部订阅者、正常本地感知/gossip。其余配置逐字段从 Stage B 冻结配置读取。

所有学习臂与规则共享真实报价/预算/capacity/合并/终态处理。费用口径为模型中的载荷尝试成本，不是真实无线能耗或全系统通信量。丢包收费，拒付不收费；策略主动少发与执行器限制带来的降费分列，不篡改账本。

不更改 horizon/deadline/场景难度来制造信号。窗口末 active 是本窗口未完成，不宣称未来永远失败。参考与训练采用同一终止/截断口径。

## 4. 数据划分与初始化

建议注册两个新的开发用途场景块：

- training：1300000..1300047（48个场景）；
- development：1301000..1301015（16个场景）。

先查全项目登记表、已有流/日志/manifest与已声明区间。任一区间碰撞则在0新启动下停止并请求新编号，不擅自顺延。封存块保持封存，不生成、不读取其内容。这里的 development 不是未来独立确认集。

通信训练随机种子固定11、22、33。三 seed 使用相同48训练场景及相同训练场景顺序：固定数据顺序 RNG=20260927，逐轮重排48场景，依次取满128集（48+48+32）。各通信策略的动作 RNG 独立；搜索/环境/信道 RNG 不受网络初始化或动作采样全局播种污染。

每个场景的 immediate reference 只运行一次，总48+16=64次，按模型/场景初态/配置/代码身份键缓存。环境随机性可复现是缓存前提。先用已有 Stage B 证据检查 RNG 接缝，不额外偷跑 reference。

跨策略只声称同初始场景/seed，不默认未来所有随机实现一致。搜索若按架构应与通信解耦，仍需记录其初态/发现序列身份；有异常就定位，不把不可比结果继续池化。

## 5. 启动预算：正常672，储备28，硬上限700

| 用途 | 计算 | 启动数 |
|---|---|---:|
| 3个通信训练 seed | 3×128 episode | 384 |
| immediate reference | 48 training + 16 development | 64 |
| 学习模型开发评估 | 3 seed × 3时点(0/64/128集) × 16场景 | 144 |
| 其他5条固定规则开发评估 | 5×16场景 | 80 |
| 正常合计 | | 672 |
| 技术错误/有限重测储备 | 不用于扩样或调参 | 28 |
| 总硬上限 | | 700 |

每 seed 最多128000环境步，三 seed 最多384000；如果合法提前终止则实际步数更少，不补跑episode凑步数。每4集一批，每 seed 最多32批；PPO minibatch/epoch 决定实际 optimizer.step，不能把批数当 optimizer.step 次数。

所有环境构造后进入episode的启动、失败、评估、基线、smoke、崩溃复跑均计数。无额外吞吐探测和全套 smoke。纯合成测试可做真实 optimizer.step，但独立标记 synthetic，不能混入环境学习的更新量。

训练参考缓存失败的重跑也计数。故障恢复只用本轮合法完整checkpoint，恢复 RNG 与顺序；不能挑性能更好的恢复点。中途失败的训练消耗不能从台账扣回。

## 6. 强规则：不再把别名算成竞争者

固定参考 immediate 加以下五条规则构成六条规则对照：

1. always_hold；
2. late_discovery_gate，阈值固定147，不搜索新阈值；
3. periodic，interval=20；
4. random_send，发送概率0.5，固定每场景规则 RNG；
5. event，progress_threshold=0.2、urgency_time=25.0。

对所有参数读取现有规则实际字段，未知字段必须报错，不能静默默认。late_discovery_gate 是新提出的强对照，不是学习成果：只能依据发送端可见的首次发现步号，对晚发现目标首报一次，早发现目标不发；不是把早发现延迟到147，也不允许读取终局或接收端知识。明确 tick/物理time转换与阈值比较运算符，启动前锁定纯合成边界测试。已有开发阈值147不能再在本轮改选。

send_once 若在本配置与 immediate 等价，作为别名说明，不另耗运行；若源码核验无法证明语义一致，不宣称逐包相同，只保留 immediate 参考。规则集合若与更高层总纲必需项冲突，在启动前解决并保持总账闭合，不能中途追加。

## 7. 同一轮内先过小规模训练正确性检查，再完成余量

先运行 seed11 的首4个训练episode，计入384，不另增C0配额。检查：

- optimizer 确实更新且只更新通信模块；搜索/分配参数哈希不变；
- rollout 时的 log_prob 在任何更新前可重算，与 Stage B 门槛一致；PPO更新后参数变化不能用来误判旧 log_prob；
- 实际成本/任务reward/GAE/组合优势/对偶g从落盘数据可重算；lambda更新方向正确且reference无梯度；
- cost/task critic 的target、预测、loss、梯度均有限；全 mask 情形不产生actor伪样本；
- 保存并在纯计算进程恢复本轮checkpoint，状态与同一观测前向一致，不额外启动episode；
- 丢包/拒付/主动HOLD/可执行样本计数闭合。

通过后按冻结计划继续三seed。任一数学/信息泄漏/计费/训练状态正确性门禁失败则暂停学习，隔离证据、最小修复并登记新代码身份；储备不足或修复改变研究协议时停止，不把不同训练版本池化。

不以首4集是否省字节或任务是否改善作为启动余量的门禁。之后不按中间显著性、单个seed最好表现或验证分数改变训练长度/超参数。

## 8. 评估与结果解释

固定评估时点：初值、64集、128集；最终比较固定128集checkpoint，不择优。每次评估冻结lambda和权重；任务/字节全由环境真实输出。合法提前停止/故障导致未到128集的seed标incomplete，不拿另一个点冒充。

主描述比较是每个训练seed最终模型对 immediate 的按时任务数差与实际字节差。还必须完整报告其他五条规则，特别是 late_discovery_gate；优于 immediate 不等于优于强规则。

逐scene配对，以episode为聚类单位报告均值、区间和改善/打平/退化数，三训练seed分别列出并报告差异。不把消息或目标当独立样本，不把3×16当48个独立场景；不选最佳seed作主结果。所有对照与阶段均完整发布。

16个开发场景只支持探索性pilot。零观测损失、CI含0或朴素bootstrap退化到[0,0]都不能证明总体非劣/等效或delta=0约束成立。不在本轮作正式学习方法主张，不开独立确认块，不因为区间宽自动扩样。

本轮不设已获批准的rho。报告实际降本比例/绝对B及其不确定性，候选是否有后续价值综合任务退化、费用、强规则和seed稳定性判断；候选标签不是正式验收。128k未收敛不能宣称排除欠训练，但不能因此自动续训。

保存训练诊断：原始reward/费用、有效actor样本量、entropy、task/cost value loss、PPO KL/clipfraction、梯度范数、lambda/g、预算拒付与参数哈希。lambda持续增长或在0边界只是诊断；不得在运行中靠放宽delta让图好看。

## 9. 冻结、交付、停止

启动前提交协议、精确配置、scene注册表、启动计划、代码内容哈希；计费/actor/trainer代码冻结，报告脚本独立。禁止矩阵运行同时改训练路径。

交付：训练功能检查、672格/启动台账实际完成情况、3seed曲线、全部规则与学习比较、逐场景差异、优化目标与账本闭合、checkpoint索引/哈希、方法限制和下一步建议。原始日志/模型不入Git，manifest记录存储位置与摘要。

正式结论只允许：pilot_completed / technical_failure / incomplete，再分别描述效果。不得把“代码能训练”写成“学习优于规则”。成功、有退化、无优势或资源耗尽均停止，追加训练/确认实验/部署均另行批准。
