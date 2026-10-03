执行 ROUND42：一次性 SEND/SKIP 首报选择消融。

本消息是完整执行依据。保存为 EXECUTION_BASIS.md，在首次环境启动前独立提交。完整阅读后按顺序执行。

一、研究目标和本轮问题

系统由冻结的PILOC搜索、CTDE分配和学习型通信控制组成。
当前目标是保持任务表现，同时降低已计量通信成本。打包和部署仍暂停。

此前：
ROUND39禁止刷新后，部分模型恢复了首报预算及任务。
ROUND40从初值训练first-report-only模型，但sampled与greedy行为差异很大。
ROUND41固定累计风险解码恢复了三个seed的96/96任务，成本239040/239040/239028 B，接近immediate的239028 B；late147只用77412 B。

这说明恢复发送并不等于学会节流。
重复HOLD只是等待，后续仍能不断尝试发送；累计风险解码也没有明确的“本集不发送”选择。

本轮检验一个新动作语义：
每个目标在首次合法可付机会，只做一次 SEND_NOW 或 SKIP_EPISODE 决策。
模型能否学会跳过不必要的首报，同时保持任务表现？

这是有意收窄的首报选择实验。
不同时研究延迟发送、刷新、ACK或新观测。
本轮不保证该语义优于原方法。

二、起点、分支和材料

仓库：
https://github.com/rui020726/zhenghe

从已完成ROUND41的本地分支：
exp/comm-temporal-decoder-diagnostic
建立：
exp/comm-one-shot-report-selection

ROUND41报告没有给最终交付HEAD。
必须从实际分支、报告和清单核验完整HEAD并登记。
报告中的4e01d5dc6fd5280f806f05e707987ba60e21d3b6是实验代码头，不能直接冒称最终交付头。

ROUND40核验后的完整头：
8f8245a665b15960e70194fc6e245221b70db90e

先读取适用AGENTS.md，保留其他会话文件，不覆盖或reset同名分支。
需要隔离时使用独立工作树。

必读：
- PROJECT_STATUS及总纲当前目标和边界；
- ROUND41报告、冻结配置、配对结果、解码器和执行清单；
- ROUND40报告、冻结配置、训练入口、NEW初值与ep128索引；
- 实际collector、batch_samples、PPO、双critic、奖励时序和checkpoint恢复路径；
- relay首发账本、报价、预算和真实_emit路径。

已有材料在身份一致时复用，不重做无关全仓审计。
缺少必要权重或依赖时列出缺项，不替换模型。

网络最多有限重试2次，本地材料完整时可离线执行，远端状态单列。

三、OLD与NEW

本轮OLD明确指：
ROUND40的NEW ep128三个模型，seed11/22/33。
不是ROUND36模型，也不是ROUND40旧对照。
只读加载，不续训或写回。

本轮NEW：
同三个seed，从ROUND40相同seed的登记初值开始。
只改变目标级动作资格和HOLD语义，训练量及其余设置继承ROUND40。

评估时OLD与NEW都使用本轮同一one-shot SEND/SKIP运行协议：
- greedy；
- sampled。

OLD的0动作在本轮被解释为SKIP，而非原先的重复等待。
这是明确的协议适配，不称它与历史greedy/sample行为等价。
主比较用于判断：按新语义训练是否优于直接适配旧权重。

ROUND41 hazard结果保留为历史参照，不接入本轮NEW训练或评估。

四、动作契约：必须精确实现

每个episode、每个target维护本轮局部决策状态。
它不修改环境任务状态、registry事实或relay真实首发历史。

UNDЕCIDED：
尚未作出自由策略选择。
预算或容量不可执行时仍保持此状态，不把mask当成SKIP。

首次顺序预算/容量门通过的合法首报机会：
模型在两个动作中选择一次：
1 = SEND_NOW
0 = SKIP_EPISODE

SEND_NOW：
通过原真实_emit路径请求立即首发。
只按真实发射和尝试计费。
丢包仍收费，且不重试或刷新。

SKIP_EPISODE：
本episode后续该target候选都固定HOLD，不再采样或重新决策。
同target新revision不重新打开选择。
新episode清空状态。

必须区分：
- 首次自由选择SKIP：真实策略动作；
- 已经SKIP后的固定HOLD：外部资格限制；
- 原预算/容量mask：不可执行机会。

首次SKIP必须进入free/actionable集合：
保存log P(SKIP)，参与PPO likelihood、entropy和actor梯度。
绝不能把它标成forced/masked而屏蔽学习信号。
只有后续已关闭候选才对likelihood/entropy贡献0。

SEND选择也是一次真实策略动作。
若在已通过报价和执行门后真实_emit发生意外拒绝，保留选择和拒绝证据，停止受影响运行检查；不得重新采样掩盖它。
丢包不是执行拒绝或技术失败。

其他要求：
- 每目标每episode最多一次自由选择；
- 保持原候选顺序和19维观测；
- 不删观测中的已关闭目标来偷偷改变backlog或特征；
- 不按分析标签、未来结局或接收端知识替模型决定SKIP；
- 不强制首报，不预留预算，不免费发送；
- 不把局部SKIP状态写成“已首发”；
- 无refresh，实际发送次数每目标最多一次。

本轮SKIP是episode级承诺，不是证明该目标在所有情形都不需要通信。

五、保持不变的训练部分

逐字段从ROUND40冻结件读取并继承：
- 网络、原19维特征和期限编码；
- ALL_ROWS归约；
- PPO学习率、clip、epoch、minibatch、熵系数；
- 双critic、价值系数及全局梯度裁剪；
- gamma、GAE、next-value与终止语义；
- 修复后的真实任务事件采集；
- 世界、信道、ACK、1000步horizon；
- 29300 B硬预算；
- 搜索和分配权重；
- 128集场景顺序和更新shuffle规则。

不同时调整熵系数、lambda步长、优势归一化、ACTIVE_ROWS或奖励。

原目标保持：
c = C_measured / B0
g_hat = mean(Y_immediate - Y_policy) - delta
A_actor = lambda_dual*A_task - A_positive_cost
lambda_next = max(0, lambda_dual + eta_lambda*g_hat)

B0=10000 B
delta=0
lambda初值=1
eta_lambda=0.05

delta=0是实验约束，不是部署保证。

保留原critic样本、ALL_ROWS和奖励时间基。
后续关闭候选不得为了“放大信号”而从训练行中删除。
自由动作减少导致的自然样本数、optimizer.step变化如实报告。

为新动作语义登记版本，例如：
comm_action_semantics = one-shot-send-skip/1

写入freeze、tape、checkpoint payload及training_state。
旧权重可只读评估，旧checkpoint禁止静默恢复进NEW训练。

六、数据与额度

沿用已使用数据：
训练48场景：1300000..1300047。
开发16场景：1301000..1301015。
使用ROUND40登记的128集训练顺序。

不生成新确认场景，不接触封存块，不把本轮称为独立测试。

正常启动：
1. immediate参考：48训练+16开发，各一次，共64。
2. NEW训练：3seed×128集，共384。
3. OLD/NEW×3seed×greedy/sample×16开发，共192。
4. late147、always_hold×16开发，共32。

合计672正常启动。
技术失败储备28。
总环境启动硬上限700。

首4集门禁计入每seed的128集，不另加smoke。
immediate开发行同时作评估参照，不重复运行。
不进行独立流录制或额外吞吐探针。
所有短运行、环境构造和重试都入账。

七、首次启动前的针对性测试

纯计算测试穿过实际collector、batch_samples、likelihood和actor路径。
0环境构造、0合成optimizer.step；允许forward/autograd。

至少覆盖：
1. 未决定目标通过合法门后只有一次自由选择。
2. SKIP首次选择有真实log_prob、entropy和actor贡献。
3. SKIP后的候选固定HOLD、无再次采样或扣费。
4. SEND真实计费、丢包仍关闭后续发送。
5. mask不消耗自由选择资格，revision不重开，episode正确重置。
6. 多目标状态隔离及顺序预算反例。
7. 联合概率按实际动作前缀和自由集合独立重算。
8. 全限制行无伪actor梯度，critic按原契约保留。
9. 旧checkpoint训练恢复拒绝，OLD只读加载合法。
10. NEW初值与相同seed的ROUND40 ep0逐参数一致。

再用两类可手算问题验证SKIP学习方向：
A. SEND和SKIP任务收益相同，SEND成本0.2、SKIP成本0：
   纯成本差应推动SKIP。
B. SEND任务1/成本0.2，SKIP任务0/成本0：
   p=0.5时，最大化目标期望梯度为
   (lambda-0.2)*p*(1-p)；
   lambda=0时为-0.05，lambda=1时为+0.20。

预期来自独立推导，不复制被测实现。
不将合成方向测试当作真实学习收益证明。

门禁通过后提交执行依据、配置差异清单、版本、初值/模型/场景顺序身份和672+28账本计划。

八、执行顺序

1. 执行64个immediate参考，核验任务与成本闭合。
2. seed11首4集训练、一个真实PPO批。
3. 通过首批门禁后完成其余124集。
4. seed22/33各自首4集也做同样门禁，通过后完成训练。
5. 三seed完成后固定ep128，执行192模型格和32规则格。

首批门禁：
- 只有通信参数进入optimizer，包含两个critic；
- 更新前logged/recomputed log_prob差<=1e-6；
- SEND和首次SKIP均被正确计入；
- 后续关闭候选、预算mask无似然/熵泄漏；
- 每目标自由选择至多一次；
- 成本、真实任务事件tick、GAE及组合优势独立重算；
- g_hat来自同场景参考，lambda方向和更新时间正确；
- 梯度和两个critic有限；
- checkpoint版本和恢复状态完整；
- 搜索/分配哈希不变。

不要求每模块每批都必须非零梯度；结合实际自由选择和受控反例判断。

每seed128集、32批，从初值训练。
保存ep0/64/128及每批可恢复状态和参数前后链。
不根据曲线挑checkpoint或seed。

训练sampled；评估greedy/sample分开。
不使用hazard累计、最迟发送兜底或其他阈值。

九、故障与冻结纪律

构造环境前原子claim，分片格互斥，使用已验证账本实现。

失败启动不退额。
效果差、全SEND、全SKIP、任务损失均不是技术失败。
技术恢复只用本轮完整同版本状态，已提交更新不能重做。
恢复重放也计入700上限，不能从旧tape静默重建训练。

无法证明恢复身份或无法在剩余额度内完成时停止。
不整条128集无限重训。

冻结后不在运行中更改动作语义、数学或研究配置。
出现影响有效性的错误时保留原件和版本，停止受影响运行。
分析器/簿记修正单独登记，不混合不同策略结果。

十、预定结果与诊断

主比较：
每seed、每mode的NEW-OLD。
两者使用同一one-shot协议。

次比较：
各候选相对immediate、late147、always_hold的绝对任务与成本。

分别报告：
- 按时任务和按时目标ID；
- 救回、新丢、目标交换及最大单场景损失；
- 首报数量、字节和实际接受记录；
- 自由选择SEND/SKIP比例；
- 预算mask、执行拒付与策略SKIP分开；
- 三seed、greedy/sample的差异；
- 每批g_hat、lambda、likelihood、模块梯度与裁剪读数。

记录目标首次选择时的合法actor特征，和分析结局分开保存。
不能用结局标签反向改变策略或拟合阈值。

本轮已用开发块：
采用逐场景配对差、总数、均值和改善/持平/退化计数。
不挑最佳seed、不池化成独立样本，不作总体非劣、显著性或泛化声明。

十一、固定判读与停止

- NEW保持任务、减少首报和成本：
  支持一次性选择值得继续，尚需另行确认。
- NEW全SEND，达到immediate：
  学会了规则水平，节流价值仍未建立。
- NEW全SKIP或省钱丢任务：
  任务约束/学习仍未解决，不能称达标。
- 只有一个seed成功：
  不建立训练稳定性。
- OLD同样好：
  需要区分运行协议本身的贡献，不能全部归因重新训练。
- greedy/sample结论不同：
  保留分表，不将采样约束保证搬到greedy。

不得因负结果调参数、扩样、加入WAIT或改变SKIP持续时间。
本轮研究首报选择，不能宣称已解决完整通信频率控制。

十二、交付

目录可按实际日期命名：
runs/comm_one_shot_report_selection_YYYYMMDD/

保留：
- 执行依据及冻结配置/唯一差异；
- 代码、初值、模型、场景顺序和版本身份；
- 纯计算与三seed首批门禁；
- 启动账本、失败/恢复记录；
- 训练曲线、checkpoint索引和参数链；
- 逐seed/mode配对与规则绝对表现；
- ROUND42_REPORT.md及机读verdict；
- Git外原件路径、大小和hash清单。

更新PROJECT_STATUS当前入口，旧结果保留。
原始日志/tape/权重不入Git。
显式暂存本轮文件，不碰其他会话文件。

本轮本地提交后停止。
不自动推送、合并、部署、续训或开启确认实验。

最终报告完整HEAD、实际启动/更新/储备、
OLD权重未变、NEW从初值的证据，
以及模型是否真的学会减少首报，而非仅增加等待时间。

现在从ROUND41交付头核验、材料读取和执行依据存档开始。