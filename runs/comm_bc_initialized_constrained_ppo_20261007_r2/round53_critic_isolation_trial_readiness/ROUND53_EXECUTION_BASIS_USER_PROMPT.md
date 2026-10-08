你是本项目的执行 agent，不需要继承旧会话记忆。请完整阅读本提示词，以它作为当前执行依据。历史报告里的“建议”不构成新增任务。

任务：ROUND53 — critic 梯度隔离的固定训练与开发评估。

本提示词由项目负责人转交并要求执行时，授权本轮固定计划：528 次正常启动＋24 次全新技术失败储备，总启动硬上限552。不要挪用历史储备。正确性门禁通过后连续完成全部计划，不在每个工作块之间再次等待批准；出现影响有效性的错误时按下述停止纪律处理。

一、目标与已有结论

研究主线是：冻结搜索与任务分配，学习通信策略，在保持按时完成目标的前提下降低已计量通信成本。不是优化搜索策略，也不是准备部署或最终交付。

当前已有结果：

- BC 初值已经提供大部分相对 immediate 的降本。
- 原共享主干 PPO 尚未建立跨 seed 一致的附加收益。
- ROUND48 “BC 初值近似掷硬币”的说法已撤回：当时探针误用了旧模型初值。真实 BC 身份已在后续核验中纠正，不得重新引用该错误解释。
- 源绑定梯度诊断发现 critic 对共享主干的梯度明显强于 actor。这是待检验的机制线索，不是已经证明的性能根因，也不是原训练器必然存在正确性缺陷。

本轮唯一方法变量：

让 task critic 和 cost critic 的输入共享表示 stop-gradient，阻断 critic→shared trunk 的梯度；保留 actor/entropy→shared trunk 和 critic head 自身的训练。

全局梯度范数及裁剪公共缩放可能因此改变，这是该方法的间接效应，不能宣称实际 actor 更新强度完全不变。

二、接手起点与保护范围

仓库：
D:\code\整合\zhenghe-round42

Python：
D:\project_env\python.exe

准备分支：
feat/round53-critic-isolation-trial-readiness

已核验交付头：
c13df2f42f3277df5d5fdaf74bce898807dc04ae

任务目录：
runs/comm_bc_initialized_constrained_ppo_20261007_r2/round53_critic_isolation_trial_readiness

先核验本地实际 HEAD、上述提交的祖先关系、工作区和本任务运行状态。不得猜 SHA。

- 若本任务已经完成，只核验交付，不重复执行。
- 若存在同一任务的活跃进程，读取其状态并等待，不启动第二份。
- 若存在未完成产物，只能使用已核验的同版本精确恢复入口；不能删除输出后从头重跑，不能重复已提交的 PPO 更新。
- 若起点、源码或必要输入缺失，具体报告缺项，不凭本提示词重造历史文件。
- 不重置、不回退、不覆盖旧实验。

两份既有受保护脏文件保持接手时字节，既不修改也不暂存：

runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_CURRENT_TASK.json
runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_R2_V6_STOP_STATE.json

其他会话文件、旧 checkpoint、tape、账本、journal、失败原件和报告同样保留。

三、先读当前真实实现，再登记授权与冻结

完整读取任务目录中的：

- READY_CHECK.json
- TRIAL_PLAN.json
- BASELINE_REUSE_CHECK.json
- SOURCE_BINDING.json
- ZERO_LAUNCH_TEST_EVIDENCE.json
- ROUND53_FINAL_CONNECTION_REVIEW.json
- ROUND53_READINESS_REPORT_ADDENDUM.md
- r53_execute.py、r53_contracts.py、round53_trial_adapter.py
- 对应窄范围测试及实际初始化、首批门禁、保存和恢复接缝

当前预检状态为：

preflight_ready_for_authorized_first_four=true
first_four_dynamic_gate=PENDING_NOT_EXECUTED

此前22项窄测试通过，但真实首四集尚未运行。预检通过不等于动态门通过，也不等于科学成功。

新增保存本次人类转交的执行依据和授权回执，不覆盖准备阶段旧依据。按照当前 authorize_receipt 的实际字段登记任务、分支、528/24/552、真实计划摘要、批准来源和时间；approval_reference 指向本提示词的存档与摘要。授权来自项目负责人实际发送本提示词，不能拿旧 READY 文件或旧 goal 冒充批准。

计划摘要必须由当前 plan_from_frozen_sources/canonical_sha 的真实路径计算，不能沿用过期版本的摘要。授权和文档提交不应改变科学配置或训练代码身份。

首次启动前冻结当前执行源码、输入、计划、模型角色、数据顺序、输出根和统计约定。已有证据在输入身份不变时复用，不重新扫描全部历史账本。

允许必要的启动前最小接线修正和窄测试，但不改变研究设计。不要再另建一轮“准备方案”。

四、必须保持的科学与初始化合同

科研配置逐字段继承真实 ROUND46 冻结件，不根据聊天摘要猜未列出的参数：

- 训练 seed：11、22、33。
- 每 seed128集，1000步/集，4集/批、32批。
- 训练场景1300000..1300047，使用原冻结的128集顺序。
- 开发评估场景1301000..1301015；这是已使用开发块。
- 6目标，29300 B硬预算。
- B0=10000，δ=0，λ初值1，ηλ=0.5。
- gamma=1、GAE=1、ALL_ROWS。
- 19维原观测，one-shot SEND/SKIP；SKIP的原一次性语义不变。
- ACK关闭，policy_owned_timing。
- greedy使用原argmax与平局归SEND规则；sampled使用原登记采样合同。
- 搜索/分配 checkpoint、40项世界字段、PPO超参、线程/设备配置全部保持原冻结值。

目标与接线：

min E[c]，约束 g=E[Y_ref]−E[Y_pi]−δ≤0；
c是正的已计量成本/B0；
actor优势是 λ_before*A_task−A_positive_cost。

不得更改奖励、GAE、mask、归约分母、优势标准化、学习率、熵系数、clip阈值、优化器参数集合、对偶规则、预算、观测或通信协议。不得把 critic 隔离与其他修改混成一个实验。

真实 canonical trainer 原始字节 SHA256：

0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841

使用已核验适配层包住实际调用的同一个 trainer 对象，不复制重写 PPO 数学。临时 hook/wrapper 必须 finally 恢复。

初始化必须保持已修正的原合同：

1. 同 seed R45 BC_POST 四模块模型初始化。
2. 经真实 R42 路径，在模型初始化之后恢复原登记的 R44 ep0 RNG-only 状态。
3. fresh Adam、λ初值1，按冻结数据顺序开始训练。

严禁重新使用已撤回的 hash-derived seeding 路径。

真实 BC 初值参数摘要应与登记件一致：

s11：37303c494cd997894fba1441cf9d2cde86cf2135e71f443a4ed1bcb9ec7f0330
s22：e8531e0aff95ed207bc70026247d45869656d266aa974838a7ec2cfe352479b0
s33：10bae22f3896adc507f05c5d868aad43b0441d19f754f2ef4dc373fc964e57a5

R42旧ep0文件仅按已登记合同提供RNG，不提供本轮模型或optimizer初值。不得从旧PPO ep128续训。

五、一次性执行固定计划

正常启动严格分解：

1. 三 seed候选训练：3×128＝384。
2. 候选ep128 greedy评估：3×16＝48。
3. 候选ep128 sampled评估：3×16＝48。
4. 真BC_POST sampled评估：3×16＝48。

合计528正常启动。

技术储备最多24，总硬上限552。所有失败、重试和实际构造必须留痕，不退额。

复用已核验的：

- 48个训练场景 immediate参考缓存；
- BC greedy48行；
- 原共享PPO greedy/sample共96行；
- immediate、late开发基线各16行。

BC sampled48行是本轮要生成的输出，不是准备阶段缺失输入。不得额外重跑参考或规则，不增加smoke，不生成新流，不开启封存块800000..800399、900000..900199。

先执行每 seed首四集，合计12集，包含在384集内。通过真实首批动态门后继续余下训练，不重复首批更新。实际门禁至少覆盖：

- 初值、参数集合、fresh optimizer/dual和登记RNG身份；
- 真实 trainer 与 critic隔离适配层的活体绑定；
- critic→trunk切断、critic head训练路径保留；
- mask、同权重likelihood重算、正成本和任务奖励时序；
- GAE、组合优势、g_hat、λ更新；
- checkpoint版本和首四集后游标/恢复状态。

实际首四集门失败就停止受影响运行，不把已运行但未验证的批次写成有效训练结果。全部通过后，连续完成剩余训练、144格新增评估、最终对账和分析，不再逐块索要批准。

使用已有实际入口：
D:\project_env\python.exe -u runs\comm_bc_initialized_constrained_ppo_20261007_r2\round53_critic_isolation_trial_readiness\r53_execute.py

执行前以本地源码确认授权回执路径及入口契约。不得绕开授权检查直接调用旧 ROUND46 整体execute。

六、运行纪律

使用已实现的私有输出根、slots兼容计数代理和活体绑定表。核实 r44_runtime 等实际读取版本/路径的模块，不能只修改runtime字典。

- 核心claim与派生快照分离。
- 运行中允许真实的部分完成快照；最终门禁必须满额闭合。
- 派生快照失败不能翻转已持久化成功的核心claim。
- 重复、越界、错误身份、核心损坏必须停止。
- reserve仅用于计划内真实技术失败的合法重试，必须引用已持久化失败claim；不能靠自报original_status申请。
- 性能差、任务损失和约束未满足不是技术失败，不能重跑或剔除。
- 冻结后遇影响有效性的代码问题，保留失败和现场，停止受影响运行；不热修后悄悄继续。
- 非科学性的事后统计或渲染错误可在0新增episode下修正，保留旧版本并登记更正，不能借此换判据。
- 不跑会额外构造环境的全项目测试或smoke。
- 不选择最佳seed、checkpoint、模式或阈值；不追加训练，不扩样。
- 不推送、合并、部署。

七、比较与判读

首次启动前登记统计方法；使用场景作为配对单位，greedy与sampled、三个训练seed分别报告，不池化成一个“最佳学习臂”。

若没有已冻结且可直接复用的统计约定，本轮使用：

- 16场景按seed排序；
- PCG64随机种子20261008；
- 10000次场景有放回抽样；
- 同一索引矩阵供各比较复用；
- linear分位的95%描述性区间；
- 差值均为“隔离候选−对照”；
- 重采样均值使用values[indices].mean(axis=1)，独立手算小例核验，不能把索引矩阵当抽样次数矩阵。

逐seed、逐模式比较：

- 隔离PPO vs原共享PPO；
- 隔离PPO vs真实BC同模式；
- 隔离PPO vs immediate和late，分别报告。

必须报绝对任务数、逐目标按时ID集合、丢失/救回目标、已计量字节、首报数，以及任务与成本的联合变化。净任务数相同不能掩盖目标集合交换。

保留已有开发判据，不临时降低门槛：

三个seed的greedy臂都必须在16/16场景与immediate具有相同按时目标ID集合，并且每个seed的平均计费都低于4838.25 B/集，才可将本轮development_go记为true。

即使通过，也只是已使用开发块上的候选结果，不构成独立确认、正式非劣、总体零损失、泛化或部署证据。CI含0不写成等效。

区分：

- 技术执行是否完整；
- critic隔离是否按合同生效；
- 相对BC和共享PPO是否有附加收益；
- 三seed是否一致；
- 开发判据是否通过。

没有改善就如实报告，不开启新方案。

八、交付并停止

在本轮私有目录保存：

- 执行依据、授权回执与prelaunch冻结身份；
- 三seed首批真实门禁；
- 528正常任务及技术失败/储备的累计账本；
- ep0/64/128 checkpoint索引与每批真实参数/λ链；
- 逐场景、逐目标配对结果；
- 同源统计JSON、机器verdict和简洁报告；
- Git外原件路径、大小、SHA256清单；
- 本轮缺陷/更正登记。

最终严格核验528个正常逻辑任务、144个新增评估格及所有实际claim。未用reserve不需要填满。先确认输入完整和正确性，再生成科学比较。

报告实际环境构造、成功episode、失败/重试、训练批、optimizer步、checkpoint写入和储备使用，不用“行数”代替“启动数”。

显式点名提交本轮代码、报告和轻量索引；不提交权重、原始tape/logs或两份受保护脏文件。给完整HEAD、实际测试数量与退出码、运行完整性、逐seed结果及未建立的主张。

本地完成后停止，不自动推送或启动下一轮。