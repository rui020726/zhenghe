执行 ROUND43：一次性首报选择的完整 GAE trace 消融。

本轮要求一轮闭环完成必要检查、实现、三 seed 训练、固定评估和结果判定。正确性门禁通过后自动继续，不要仅提交设计或诊断后再次等待授权。

一、背景与目标

项目冻结 PILOC 搜索与 CTDE 任务分配，只研究通信控制器。

主线目标：
尽量保持按时任务数，同时降低已计量的载荷尝试成本。

当前动作协议为 one-shot-send-skip/1：
每个 episode、每个 target，在首次合法且预算可付的首报机会，选择 SEND_NOW 或 SKIP_EPISODE。
首次 SKIP 是真实自由动作；决定后的候选固定 HOLD；预算 mask 不消耗决策资格。
不刷新、不重试、不使用 hazard decoder。

ROUND42-R 已完成：
672 个成功逻辑任务，累计673次启动；
三 seed 各128集、共96批；
恢复和路径门禁已修复。

但未达到研究目标：
- 三个 NEW greedy 都是88个按时任务、0字节；
- NEW sampled s11/s22/s33 为94/93/92个任务；
- immediate 为96个任务、239028字节；
- late_discovery_gate 为96个任务、77412字节。
以上均为同一16场景开发块的批次汇总。

不能把全 SKIP 丢任务解释成成功降本。
报告正文“两个 greedy NEW seed 全SKIP”与表格不符，应登记更正为三个，不改原始数据。

二、本轮唯一研究变量

检验假设：
首报动作发生较早，任务反馈发生较晚；有限 GAE trace 与不精确 critic 可能让任务信号难以传回早期自由动作。
这只是待检验解释，不是已确认根因。

唯一方法变化：
共享的 gae_lambda：0.95 → 1.0。
task 与 positive-cost 两条优势路径使用同一个新值。

明确区分：
- gae_lambda：优势估计的 trace 系数；
- lambda_dual：任务约束的对偶变量。

禁止把 lambda_dual 也改成1或改变其更新方式。

gamma 保持原值，不同时调整。
从真实配置和调用路径确认旧 gae_lambda 确实为0.95，以及 gamma 的实际值，不能只读配置声明。

对于完整真终止轨迹，验证：
A_t = Σ gamma^(k-t) r_k − V_t。
存在截断 bootstrap 时，保留原边界与尾值，不能将其冒称纯 Monte Carlo 回报。

GAE=1是方法消融，不是“旧GAE数学错误”的修复。
它可能增大方差，本轮不能通过额外技巧补偿方差。

三、起点与可复用材料

定位仓库，读取适用 AGENTS.md、PROJECT_STATUS 和 ROUND42-R 完成材料。

已知执行源码头：
0369408dec2cfabffb623e0445174725f9dde0d5

这不是最终结果提交头。
从实际包含完成报告、结果索引和账本的最新交付头建分支：
exp/comm-one-shot-full-trace-ablation

核验实际完整 SHA，并确认其继承上述执行源码头。
不要回退到只含执行代码的旧提交。

优先读取：
- ROUND42R_COMPLETION_AUDIT.json；
- ROUND42_RESUME_FINAL_RESULT.json；
- ROUND42_PAIRED_RESULTS.json；
- ROUND42_CHECKPOINT_INDEX.json；
- ROUND42_RAW_ARTIFACT_MANIFEST.json；
- 实际冻结配置和训练、采样、GAE调用源码。

复用已有效的 runner、路径门禁和恢复机制，不再手写另一套代理或运行器。
不重复扫描全部历史实验，不重跑无关套件。

输入未变的正确性证据可复用。
本轮重点核验唯一变化、训练初值、数据顺序、GAE数学与实际接线。

将本提示词原文存档为 EXECUTION_BASIS，不另起新的 spec 文档。

四、OLD、NEW及数据定义

本轮 OLD：
ROUND42 的 NEW ep128 模型。
不是 ROUND42 表中的 OLD，也不是 ROUND40模型。

本轮 NEW：
与 ROUND42 同 seed、同登记初值，从初值训练的新模型；
仅 gae_lambda=1.0。

训练 seed：11、22、33。
每 seed128集、32个四集批。

训练场景：
1300000..1300047；
逐 seed 使用 ROUND42 原冻结的128集顺序。

开发场景：
1301000..1301015。

以上均已用于开发，不是独立确认集。
不增加场景、不生成新流，不触碰封存块。

同 seed 的初值参数、初始optimizer状态和lambda_dual必须与ROUND42一致。
使用登记训练起点RNG；明确核验实际使用的Python/NumPy/Torch及其他generator。

允许只读使用未训练ep0状态核验初始化，不得加载ep64/128进行NEW热启动。
若训练随机起点不能严格复现，明确登记，不声称逐位随机配对。

五、其余全部冻结

保持ROUND42实际科研配置，包括：
- 搜索和分配checkpoint；
- 世界、成功率、ACK关闭、预算、horizon；
- 原19维合法发送端观测及原期限编码；
- 候选顺序、报价、顺序mask和真实执行器；
- one-shot SEND/SKIP协议；
- ALL_ROWS归约；
- 网络、双critic、optimizer、PPO参数、熵、裁剪；
- 奖励发布时序及其归属；
- 原gamma、终止/截断/next-value契约；
- B0、delta、lambda_dual初值和对偶步长；
- 不做分量优势标准化；
- greedy原argmax/tie规则、sampled原采样方式。

目标与方向保持：
c = C_measured / B0
g_hat = mean(Y_immediate − Y_policy) − delta
A_actor = lambda_dual*A_task − A_positive_cost
lambda_next = max(0, lambda_dual + eta_lambda*g_hat)

成本为载荷尝试计费，不是全系统通信或能耗。
丢包照付，拒付不能算主动节省。

若GAE变化同时改变 A+V 所形成的critic训练目标，这是该参数变化的直接后果，应登记；不得额外改critic loss或训练权重。

登记新训练版本，拒绝旧训练状态静默续训。
旧模型只作为只读比较输入。

六、启动前检查与独立反例：同轮完成

本阶段0环境、0 optimizer更新、0训练checkpoint写入。

通过真实 batch_samples/GAE/actor 接线检查，不复制被测实现作为预期答案。

至少覆盖：

1. 延迟奖励
构造早期自由动作、后续仅固定HOLD、较晚任务奖励的极小轨迹。
分别独立计算0.95和1.0的优势。
两种配置都必须符合各自公式，不能把旧配置的正常衰减判为bug。

2. 已知值函数
使用非零value验证GAE=1的望远镜求和；
覆盖真终止和带bootstrap的数据段末边界。

3. 双分量
task和positive-cost均走新trace系数；
组合仍为lambda_dual*A_task−A_cost，成本负号不翻转。

4. actor梯度
在gamma=1、value=0的受控例中：
SEND在首步付归一化成本0.2、末段得任务1；
SKIP不付成本、不获该任务。
核对真实actor路径的闭式方向。
按实际ALL_ROWS分母处理梯度尺度，不能偷偷改归约。

5. mask与一次性动作
首次SKIP仍有likelihood/entropy；
后续关闭行actor权重0，但仍保留原critic/tape时间结构。

6. 版本和恢复
新版本可精确恢复自己的状态；
旧已训练版本不得作为NEW恢复源。

在内存中只读重算ROUND42各seed的batch0：
固定原critic输出，比较0.95/1.0下首报自由行的任务/成本优势。
可以做奖励微扰来核验传递系数，但不能把它称为某条消息的因果价值。

上述诊断与实现一起完成，不另开一个“诊断完成待授权”的轮次。

检查通过后，在首次环境启动前提交代码并冻结：
实际源码身份、配置差异、初值、数据顺序、版本和启动计划。

七、额度：独立新授权，不使用ROUND42剩余额度

正常核心计划：
- NEW训练：3×128=384；
- NEW最终评估：3 seed×2模式×16场景=96。
合计480次正常启动。

默认复用：
- ROUND42的64个immediate参考；
- late和always_hold共32个规则结果；
- ROUND42 NEW最终模型的96个评估结果，作为本轮OLD。

先核验复用材料的实际模型、场景、世界配置、动作协议、模式和原件身份。
不要要求旧行的执行SHA等于新执行SHA；它们应分别符合各自冻结身份。

如参考或规则材料确实缺失、无效或无法核验：
可在首次启动前登记需要补跑的固定清单，
最多补齐64个reference+32个rule，共96次。
不能因为结果不理想而补跑，不能事后挑更有利的一次。

因此本轮：
- 核心正常480；
- 必要基线补齐0..96；
- 技术失败储备24；
- 环境启动硬上限600。

基线可完整复用时，预计只使用480正常启动。
未使用的补齐额度不得挪为加训、smoke或新场景。

失败不退额；所有实际构造、重试、短运行均入账。
不额外做账本外的环境测试。

八、直接完成训练和固定评估

各seed首四集计入128集。
通过实际首批门禁后继续剩余124集，不再次请示。

首批门禁：
- 登记初值和优化器集合；
- 真实顺序mask及likelihood重算；
- task与cost实际使用gae_lambda=1；
- 奖励事件时间、GAE与尾段；
- 组合优势、lambda_dual统计和更新；
- 参数与optimizer状态有限；
- checkpoint版本及精确恢复状态。

不要求每个模块每一批都有非零梯度。
科学表现差不是技术失败，不能剔除或换seed。

可以按已有安全机制并行训练三seed；
隔离输出与RNG，账本原子claim。
不在运行中热改影响结果的代码。

保存ep0/64/128及逐批完整恢复状态，含：
模型、双critic、optimizer、lambda_dual、RNG、数据位置和参数链。

固定使用最终ep128完成greedy/sample两模式评估。
不挑checkpoint，不扫阈值，不启用hazard、强制首报或定时兜底。

九、统计与明确判读

沿用原冻结配对统计；结果只作开发集探索性判断。
greedy主表、sampled辅表，三训练seed分开。

同时输出：
1. 每seed/mode绝对任务、字节、首报、SEND/SKIP；
2. NEW−OLD任务、成本及逐目标差集；
3. 相对immediate、late、always_hold；
4. 约束统计、lambda_dual、优势与梯度诊断；
5. 训练和执行完整性。

严格区分以下结果：

A. 全SKIP或减少发送但丢任务：
未达目标。

B. 恢复immediate任务，但成本不优于late：
任务侧改善，不建立相对强规则降本价值。

C. 仅某个seed或sampled有改善：
局部观察，不建立稳定性。

D. 三个seed的greedy在本开发块均保持immediate按时目标集合，
且平均计费低于late：
可以写“开发块观察到联合改善”；
仍不能写总体零损失、泛化或正式非劣。

GAE变化可能同时影响actor信号和critic拟合；
即使改善，也不能宣称已证明此前失败唯一由信用衰减导致。

不按中间效果扩训。
不自动进入新的独立确认实验。

十、最终交付与停止

一次完成：
检查→最小实现→冻结→训练→评估→对账→判定。

不要只提交preflight或训练报告后停下。

最小交付：
- ROUND43_REPORT.md；
- MACHINE_VERDICT.json；
- CONFIG_DIFF与复用清单；
- 首批门禁、训练曲线、checkpoint索引；
- 绝对表现、配对结果、逐目标差集；
- 累计启动账本；
- 原件路径/哈希清单；
- 完整交付HEAD与复现命令。

更新当前状态，旧报告和原件不覆盖。
原始权重/tape/日志继续留在Git外。

只在真实正确性阻断、不可恢复状态或硬上限触发时停止；
不要因为网络或未推送而阻止本地固定计划。

完成后本地提交并停止：
不推送、不部署、不调参、不追加实验。
最终先回答：
“是否完成、任务保持是否改善、是否优于强规则、跨seed是否一致”。

本提示词授权的是一次有界完整实验，不是无限试错。