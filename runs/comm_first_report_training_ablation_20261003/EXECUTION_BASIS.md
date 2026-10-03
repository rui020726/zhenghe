执行 ROUND40：first-report-only 通信策略训练消融。

本消息是本轮完整执行依据。将原文保存为 EXECUTION_BASIS.md，并在首次环境启动前独立提交。完整读完后，按顺序完成实现、门禁、固定训练和评估。

一、项目目标与本轮问题

项目是多机器人搜索—任务分配—通信控制系统。
搜索模型 PILOC 和任务分配模型 CTDE 已冻结，当前研发主线是学习型通信控制：保持任务表现，同时降低已计量通信成本。

用户已暂停打包和部署，要求方法完成后再交付。

ROUND39 已完成：
- seed11/33 的原 greedy 模型大量刷新，挤占后续首报预算。
- 禁止刷新后，失败组首报可付机会均从25/60恢复到60/60。
- seed11救回7/11旧缺失目标，seed33救回11/11，没有新丢目标。
- seed11剩余4个失败目标有预算但一直HOLD。
- seed22一直全HOLD，刷新抑制对它没有作用。
- seed33 refresh-held 达到 immediate 的任务数和成本，尚未建立学习优于强规则的收益。

本轮检验：
训练阶段也禁止已首发目标的后续刷新，能否让模型更好地学习首报的 SEND/HOLD 取舍，改善有预算却漏报及跨seed行为不稳定的问题？

本轮是开发消融，唯一方法变化是训练中的候选动作资格。
不同时引入新观测、ACK、刷新价值头、奖励塑形或对偶调参。

二、起点、分支与必读材料

仓库：
https://github.com/rui020726/zhenghe

起点分支：
exp/comm-refresh-held-mechanism

指定完整HEAD：
4e989a9088111ce6d9171cb1a21629a4b2b1e861

先定位本机仓库，读取适用AGENTS.md，核验HEAD、工作区和其他会话文件。不要假定盘符。

从指定头建立独立分支：
exp/comm-first-report-training-ablation

同名分支已存在时先确认归属，不覆盖或重置。不干扰其他会话工作树；有冲突可使用隔离工作树。

必读：
1. PROJECT_STATUS.md及项目总纲的当前目标、配置和边界部分。
2. runs/comm_refresh_held_mechanism_20261002/
   ROUND39_REPORT.md
   PAIRED_ANALYSIS.json
   MECHANISM_CHAINS.json
   CHECKPOINT_IDENTITY.json
   POSTRUN_VALIDATION.json
   r39_collector.py
3. ROUND36：
   runs/comm_final_train_20261001/
   ROUND36_TRAIN_REPORT.md
   r36_worker.py、r36_common.py
   三seed的FREEZE_MANIFEST、训练记录、初值和最终checkpoint索引。
4. ROUND37：
   runs/comm_final_evaluation_20261002/
   冻结配置、运行器、模型身份、greedy/sample评估及计费门禁。
5. 实际导入的：
   budget_masked.py
   v2_controller.py
   v2_train.py
   v2_trainer.py
   v2_reward_timing.py
   v2_run.py
   relay的首发账本和真实_emit路径。

复用已经成立且输入身份未变的证明，不重做无关审计。
缺少Git外权重或依赖时列出具体缺项，不换模型或历史版本。

网络最多有限重试2次。本地拥有指定提交和完整材料时可离线执行，远端核验状态单独登记。

三、OLD与NEW的定义

OLD：
复用ROUND36三个seed（11、22、33）的最终ep128权重。
这些模型是在允许首报及刷新的动作空间中训练得到。
只读加载，不恢复OLD optimizer或RNG，不续训。

NEW：
同三个seed，从与ROUND36逐参数相同的登记初值开始。
训练时采用first-report-only动作资格：
- 尚未真实首发的目标：模型决定SEND/HOLD。
- 已真实首发的目标：后续候选固定HOLD。

NEW每seed128集、32批，每批4集。
使用ROUND36登记的128集场景顺序；不重新抽样或更换顺序。

OLD与NEW评估时都采用同一个first-report-only运行路径。
所以主比较研究的是训练动作空间变化，不混入不同的评估规则。

OLD本轮已增加外部运行限制，不能将它的结果直接写成ROUND37原greedy结果。

四、数据块、额度与执行范围

训练场景：
1300000..1300047，共48个，沿用登记的128集重放顺序。

开发场景：
1301000..1301015，共16个。

这两个块已使用，本轮没有独立测试或泛化主张。
ROUND39的1600000块不加入本轮训练，也不作为新确认集。
不读取或生成任何封存块。

本轮正常启动计划：

A. immediate参考：
48训练场景 + 16开发场景，各一次 = 64。

B. NEW训练：
3 seed × 128集 = 384。

C. 模型评估：
OLD/NEW × 3 seed × greedy/sampled × 16开发场景 = 192。

D. 规则参照：
late_discovery_gate_147、always_hold × 16开发场景 = 32。

正常启动合计：
64 + 384 + 192 + 32 = 672。

技术失败储备：
最多28次环境启动。

总硬上限：
700次环境启动，失败启动不退还。

immediate开发行同时作为评估参照，不重复启动。
每个seed首4集训练门禁计入其128集，不另加smoke。
不进行独立流录制或额外环境探针。

所有环境构造、短运行、重试均计入额度。
不得运行会在账本之外构造环境的全量生产套件。

本轮授权NEW真实训练，以及正常的checkpoint保存。
纯计算预检允许forward和autograd，合成optimizer.step为0。
OLD权重文件不得写回。
不部署、不修改生产默认值、不推送或合并。

五、冻结训练配置

从ROUND36真实冻结件逐字段继承：
- 网络结构、19维actor特征；
- 原期限编码clip(remaining/50,-1,1)，不切换ROUND30新编码；
- ALL_ROWS surrogate归约，不切换ACTIVE_ROWS；
- PPO学习率、clip、epoch、minibatch、熵系数；
- 两个critic的价值系数、全局梯度裁剪；
- gamma、GAE系数、终止及next-value语义；
- 场景世界配置、1000步horizon、信道、预算和ACK状态；
- 冻结搜索与分配权重；
- 数据顺序与更新shuffle规则；
- 修复后的真实任务事件采集时序。

必须从原件读出具体超参，列成机读配置，不凭摘要猜补。

以下目标保持不变：
c = C_measured / B0
g_hat = mean(Y_immediate - Y_policy) - delta
L_max = -E[c] - lambda_dual * g
A_actor = lambda_dual * A_task - A_positive_cost
lambda_next = max(0, lambda_dual + eta_lambda * g_hat)

沿用：
B0 = 10000 B
delta = 0
lambda初值 = 1
eta_lambda = 0.05

delta=0是实验期望约束，不是部署零退化保证。
不得改变任务/成本优势符号、分别标准化两个分量或新增惩罚项。

先登记与ROUND36的配置差异清单。
允许差异是动作资格、对应训练版本及输出/身份字段；意外的研究配置差异须解决后才能启动。

六、first-report-only真实接线

训练不能直接照搬只支持greedy推理的ROUND39 collector。
实现可供训练采样和评估解码共同使用的动作资格逻辑。

语义：
1. 使用relay权威first_emission/发送账本。
2. 执行器真实发射即算首发，丢包或未accepted不撤销资格。
3. 仅策略请求、shadow预扣或执行器拒付不算真实首发。
4. 同目标新revision不重置首报资格。
5. 新episode不继承上一集发送状态。
6. 未首发目标继续使用原顺序预算/容量mask。
7. 已首发目标在采样前固定HOLD，不扣shadow预算、不占发送容量。
8. 不强制首报、不预留预算、不免费发射、不添加ACK或重传。
9. 保留候选顺序及原观测，不删除刷新候选以暗改backlog或特征。
10. 实际发射仍走原relay硬门与计费路径。

训练记录中区分：
- 自由首报动作；
- 因原预算/容量门不可发送的候选；
- 因本轮资格限制固定HOLD的refresh候选。

固定HOLD refresh不进入自由动作集合，对联合likelihood和entropy贡献为0。
PPO重算须沿保存的实际动作前缀和自由集合，不重新采样。

保留原ALL_ROWS、critic样本和决策行时间基。
不得为了让信号更强，额外删除全mask行、改变actor归约分母或GAE间隔。
因此自由样本数量及optimizer.step数量可以随轨迹自然改变，不强求等于OLD。

记录真实计费正成本，费用不从被拒请求推算。
保留修复后的任务事件采集入口，不能仅以奖励总和闭合证明时序正确。

给NEW增加动作资格版本字段，例如：
comm_action_eligibility = first-report-only/1

在freeze、checkpoint payload、training_state和评估行中登记。
OLD允许只读评估，禁止恢复进NEW训练。
NEW只能从同一本轮版本和冻结配置恢复。

七、首次启动前的纯计算门禁

测试穿过实际collector、batch_samples、likelihood和actor路径，不复制实现作为预期答案。

至少验证：

1. 相同合法首报输入下：
   新collector与原collector的特征、报价、概率及预算门一致。
2. 已首发refresh固定HOLD，既不扣shadow预算，也不进入likelihood或entropy。
3. 余额6：
   前一refresh报价4，后一首报报价4。
   新路径抑制前者后，后者仍可付；是否SEND由模型决定。
4. 首发丢包、拒付、新revision、跨episode重置语义正确。
5. 混合首报/refresh行的联合概率可以独立重算。
6. 全限制行没有伪actor梯度，critic样本仍按原契约保留。
7. 正成本进入lambda*A_task-A_cost：
   手算二动作反例在lambda=0和较大lambda时梯度方向正确。
8. optimizer参数集合包含trunk、action head、task critic、cost critic；
   不包含搜索或分配参数。
9. 任务事件同tick采集、prefix和尾段分桶保持修复版契约。
10. NEW初值：
    三seed参数逐字节等于ROUND36对应ep0；
    optimizer空状态、lambda=1，无OLD权重热启动。
11. 旧checkpoint恢复到NEW训练被明确拒绝；只读评估合法。
12. OLD/NEW共用同一评估资格函数，greedy与sampled只改变解码。

没有授权合成optimizer更新，不调用含optimizer.step的旧合成套件。
已有不变数学路径的证明可以复用，新增接缝用针对性测试覆盖。

预检通过后提交：
执行依据、场景顺序hash、OLD/NEW初值和权重身份、
完整配置、代码身份、额度计划及预检结果。

八、运行顺序与继续条件

1. 先执行64个immediate参考。
   核验矩阵、成本/任务闭合和归档共同配置；
   训练只使用训练场景的配对参考，不读取开发结局来调训练。

2. 执行seed11首4集训练及一次真实PPO批更新。
   必须保存更新前权重、tape、事件流、样本与更新记录。

首4集门禁：
- 只有登记通信参数参与优化；
- 更新前logged/recomputed log_prob差<=1e-6；
- refresh限制项likelihood/entropy为0；
- 成本账本、奖励事件tick、GAE、正成本组合优势独立重算；
- g_hat来自同场景immediate配对；
- 同批PPO使用更新前lambda，批后按原公式更新；
- 两个critic和梯度有限；
- 有自由动作时真实actor信号可检查，全限制时不伪造；
- NEW checkpoint版本与恢复身份完整；
- search/allocation文件哈希不变。

不要求每个模块每一批都必须有非零梯度；结合实际自由样本和受控测试判断。

门禁通过后：
完成seed11剩余124集。
seed22/33各自首4集也做同样检查，均计入128集，通过后完成剩余训练。
可合理并行，不因中间效果改变配置或额度。

每seed固定128集、32批，从初值训练。
固定保存ep0、ep64、ep128；
每个已完成批保存可恢复状态，包括optimizer、lambda、RNG、数据位置及版本，
并登记参数身份和更新前后链，避免后续缺失更新前权重。

3. 三seed训练完成后，只评估固定ep128。
   不根据训练曲线选择checkpoint或seed。

4. 执行192个模型评估和32个规则格。
   OLD/NEW均走相同first-report-only接缝。
   greedy按argmax，平局沿用原规则；
   sampled沿用既有评估随机种子规则并冻结，不按结果换随机种子。

同scene/model seed/mode的OLD/NEW使用对应评估随机起点，
但不能声称不同动作轨迹共享逐消息同一批丢包。

九、故障、账本与冻结纪律

构造环境前原子claim，分片任务互斥。
不要让并行进程覆盖共享JSON账本。

每次失败启动留痕并消耗额度。
技术失败重试最多28次，效果差、全HOLD、任务损失不是技术失败。

训练技术恢复只能使用本轮相同版本的完整状态；
已经提交的PPO更新不得重复。
恢复所需的重放也计入启动额度，不能从旧tape静默重建新训练。
无法证明恢复身份或无法在剩余额度内完成时，停止报告，不整条128集无限重训。

冻结后运行中不改数学或候选资格实现。
发现影响有效性的错误时停止受影响运行，保留原件和错误版本；
不得混合修复前后结果。分析器或纯簿记修正须说明范围。

本轮参考和学习行均写入真实配置、版本、seed、mode、checkpoint身份。
不要将输出路径哈希差异当行为差异，也不要删除真正的行为差异。

十、预定统计与诊断

主比较：
同训练seed、同开发场景、同解码模式：
NEW - OLD。
两者评估时都使用first-report-only。

次比较：
OLD/NEW各自相对immediate、late147、always_hold的绝对任务和成本。

greedy/sample、三个seed分别报告，不挑最优、不池化成独立样本。

任务指标：
- 按时任务/集；
- 按时目标ID集合；
- 新增救回与新丢失；
- 净数相同但目标交换；
- 逐场景任务差和最大损失。

成本指标：
- 已计量载荷尝试字节/集；
- 首报发射、字节、接受记录；
- 预算mask与执行器拒付分开；
- refresh发射应为0。

训练诊断：
- 每批g_hat和lambda轨迹；
- 首报自由动作数量、SEND/HOLD比例；
- 全限制行比例；
- likelihood误差；
- actor/task critic/cost critic梯度和裁剪读数；
- 实际optimizer.step；
- 首报概率与最终greedy动作。
这些只作解释，不据此调参。

同时看：
- 是否全HOLD；
- 是否所有目标立即首报，等价退回immediate；
- 是否延迟首报但保持任务；
- 是否出现可付首报长期HOLD造成任务损失。

本轮主表采用逐场景配对差、总数、均值及改善/持平/退化计数。
开发集已使用，不作确认性显著性、总体非劣或泛化声明。
不用本轮结果拟合新的发送阈值。

十一、判读与完成后的停止

必须分别回答：

1. 训练中限制refresh是否正确生效？
2. NEW相对OLD是否改善首报决策，尤其全HOLD问题？
3. greedy和sampled是否给出不同结论？
4. 三个seed是否一致，而非只有一个成功？
5. 同等任务表现下，NEW成本是否低于immediate？
6. 相对late147，新增任务与额外成本的取舍是什么？

以下读法固定：

- NEW仅收敛到“每目标立即首报一次”：
  可以说明学会规则行为，学习附加价值仍未建立。
- NEW仍全HOLD：
  本轮动作空间变化没有解决约束/探索问题。
- 只有一seed改善：
  不建立训练稳定性。
- 省成本但丢任务：
  报任务权衡，不称完成目标。
- 任务与成本同时改善：
  仅支持本轮开发块上的候选方法，需要另行确认。
- 没有收益：
  如实交付，不自动调lambda、换编码、续训或扩样。

本轮first-report-only是有意收窄的候选方法，
不能将“刷新永远无用”写成普遍结论。
后续是否重新引入选择性刷新由新证据决定。

十二、交付与本轮边界

建议目录：
runs/comm_first_report_training_ablation_20261003/
可按实际日期命名，统一登记。

必要交付：
- EXECUTION_BASIS.md
- 冻结配置、唯一差异清单、场景顺序、代码/模型/初值身份
- 纯计算门禁及三seed首批门禁
- 启动账本、技术失败与恢复记录
- 训练曲线、版本化checkpoint索引及参数链
- OLD/NEW×seed×mode的配对结果和规则绝对值表
- ROUND40_REPORT.md与机读verdict
- Git外原件路径、大小、hash清单

代码、启动前冻结件、结果/报告按主题本地提交。
原始日志、tape和权重不入Git。
更新PROJECT_STATUS当前入口，保留旧报告。
不碰其他会话文件，不用git add -A，不删除旧产物。

登记一处ROUND39文案勘误：
“总字节减少恰等于原刷新字节”不准确。
正确为：
净节省 = 删除刷新字节 - 新增首报字节。
s11失败组：218502 - 77154 = 141348 B。
s33失败组：218838 - 87342 = 131496 B。
修正文案记录即可，旧整数表和实验不重跑。

完成后报告：
完整HEAD、实际672计划完成情况、700上限使用、
技术失败、真实训练/optimizer计数、OLD权重未变、
研究结果及下一步唯一建议。

本轮本地提交后停止。
不自动推送、部署、追加训练、调参或开启独立确认实验。

现在从起点核验、必读材料和保存执行依据开始。