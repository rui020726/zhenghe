执行 ROUND45：规则标签初始化＋约束式成本学习，一轮闭环完成；不是交付打包，也不是继续扫描对偶步长。

你是执行 agent。此提示词是完整执行依据，可存档 EXECUTION_BASIS.md，但不需要另外索取 spec。按门禁自动继续完成下面的有上限任务，不逐阶段等待我再次批准。遇到实际有效性阻断按停止/版本化纪律处理；不以“所有检查通过”替代科学成功。

一、已知事实、目标与起点

目标仍是：冻结 PILOC 搜索与 CTDE 分配，学习合法通信首报选择，在逐场景保持 immediate 按时目标身份集合的同时，比强规则 late_discovery_gate_147 减少“载荷×订阅者×尝试”计量成本。全 SKIP 丢任务、全 SEND 等于 immediate、模仿强规则、选中一个 seed、跑通代码都不算实现学习增益。

实际工作树最近确认是 D:\code\整合\zhenghe-round42；先定位当前实际仓库，不盲切到旧 zhenghe。ROUND44 已闭合：
- 分支 exp/comm-dual-response-ablation，最终本地 HEAD cbd437afad6ddc3a7a68c8b7e10c0cd572a778dd；
- runs/comm_dual_response_ablation_20261004/ROUND44_REPORT.md，SHA256 2c078d055c73e7f270f74015be991491e0691d7593af32870e6eee895ab9912a；
- 96 唯一有效评估格；retry (11,greedy,1301000) 只纳入 a2，失败 a1 保留；
- immediate 开发16场景=96任务/239028 B，late=96任务/77412 B，always_hold=88任务/0 B；
- greedy s11=88/0 B，s22=96/239028 B，s33=88/0 B；加快 ηλ 后仍未建立联合改进。
旧不可变清单 CRLF/LF 问题已解决：09d758c 身份字段、原始清单及最终 Git blob 同为 b3bb6f7c2488b2acb03f467c799b6eb74554795cf9041eb30529031774a3ebeb；6114f4ed…只是未提交的临时 LF 视图，不重新调查、不重写旧身份。

核对完整 HEAD、tracked 状态、现有运行进程/锁、上述报告及关键机读索引。旧输入 hash 没漂移时复用已通过的正确性证据，不重复 R44 评估、训练、旧回归或全量套件。其它会话未跟踪文件不改、不暂存、不终止其进程。从该完整 HEAD 新建 exp/comm-rule-warmstart-cost-learning；同名存在先核实归属，不能重置/覆盖。输出独立 runs/comm_rule_warmstart_cost_learning_20261006/。

本轮覆盖的仅是 R44 完成后的“无后续授权停止”状态，不覆盖或改写任何旧停止证据、额度、失败、报告或 freeze。本地执行和提交授权；不 push、不 merge、不部署、不生成新流、不访问封存块。

二、唯一研究改动及其意义

R44 快对偶实验仍在 greedy 下分化成全发/全不发。本轮测试一个不同、尚未证明的假设：在现有合法发送端观测下，以规则动作监督得到任务可行、成本较低的神经起点，当前约束式 RL 能否保持它或取得附加价值。

唯一方法因素是“先做固定离线行为克隆 BC 初始化，再使用原约束式 PPO”。不改：
- one-shot-send-skip/1（每目标首次合法可付机会 SEND_NOW / SKIP_EPISODE，SKIP 真实自由动作）；
- 原19维 actor、原 deadline 裁剪编码、网络容量、双 critic、真实顺序预算 mask；
- positive cost，A=λ_before*A_task−A_positive_cost；
- R44 的 gamma=1、GAE trace=1、ALL_ROWS、δ=0、B0=10000 B、λ初值1、ηλ=0.5；
- 其它 PPO/价值/熵/剪裁/学习率/更新次数超参逐字段继承真实 R44 freeze，不从摘要猜；
- world/search/allocation/channel/ACK-off/29300 B/policy_owned_timing/1000步、原128集顺序及其登记 RNG。
不重新调 ηλ、actor 分母、熵、观测、阈值、BC超参或 teacher。

三、0环境的数据合法性、teacher 与监督学习

1. Teacher 固定为已注册且真正用于既有规则基线的 late_discovery_gate_147。先追真实源码调用及合法输入，核实“晚发现才发送”与“把已有信息等到147再发”不同。只可消费动作前发送端合法字段；不能读取目标结局、accepted/superseded、接收端知识、deadline truth 之外的隐藏字段、全局最优标签或事后 metadata。
2. 输入只从 R44 已有训练块 1300000..1300047 的训练 tape 合法特征及其动作前快照恢复；三 seed 共384训练集。每 episode/target 只取首次真实自由 one-shot机会，后续 masked/已决定行不当作 BC动作样本，不把首个被预算拒的时刻当作自由决策。若需要合法动作前字段而 tape 未保存，如实 missing 并在0环境阶段停止，不新增 rollout补数据。
3. 用同一条真实 rule 决策路径产生 SEND/SKIP 标签。不能凭 step>147猜规则实现；若真实 teacher 在一项 HOLD 后必须等后续机会、依赖合法观测外字段，或无法与当前一次性动作语义兼容，报告这个不兼容并停止，不换 teacher、不把等待静默改成 SKIP。
4. 必须独立检查：每标签来源、动作前时间、候选报价/顺序 mask，以及实际网络 float32 19维输入；同一 float32特征若有矛盾 teacher标签，报告不可由当前表示精确表达的证据并停止，不偷偷加列/用场景号。
5. 固定按场景分割：1300000..1300031拟合，1300032..1300047监督验证。验证也属已使用训练块，不叫独立测试；1301000..1301015仅后续开发评估，绝不进 BC拟合/标签调参。先提交分割算法和样本选择规则，再读取/生成标签统计。训练/验证类数量、唯一特征、来源重复均列清；重复合法 on-policy访问可保留，但不把同场景多次重放当独立样本。
6. BC三 seed11/22/33从对应 R44登记 fresh Controller初值开始，逐参数对齐 fresh构造与登记 ep0，不从任何已训 checkpoint热启动。
固定 BC：每seed 1000次 full-batch Adam update，lr=0.001，fit类频倒数权重的二类交叉熵，类权重只由拟合集算一次；global grad clip=0.5；无额外 entropy、weight decay=0；只取第1000次结果，无选最佳/早停/网格或扩训。Adam其余数值参数在首次真实BC前显式冻结，使用本机已安装 torch 默认 Adam 的真实参数值。
- BC optimizer只含真实 shared trunk＋action head参数，各参数只一次；task/cost head自身参数不更新。共享主干变化会改变 critic输出，是初始化因素的一部分，不能写成 critic函数不变。
- 通过独立枚举实际 module参数/ID对照 optimizer/zero_grad/clip集合，不按猜的名字或“模型可训练参数”泛化。
- 此1000×3是实际离线监督训练，应计真实 BC optimizer更新，不叫“纯合成/0真实训练”。合成回归更新另单列，本轮最多64次合成 optimizer更新，合成测试不得构造环境。
- 冻结 BC版本，记 preBC/postBC参数摘要、每步有限loss/梯度、fit/validation confusion、agreement、p分布，task/cost head参数自身前后不变。训练类缺失、梯度非有限、实际路径未更新等是真阻断；不按教师一致率高低再调参数。
- 学习模型实际推理必须由 controller forward→原 greedy argmax给动作；不加teacher fallback，不在推理时查询rule，不手工写成等价规则参数。teacher代码只能在离线构造标签/分析端使用。

先做小型0环境合成接缝回归：真实teacher输入和实际BC更新路径，独立算出标签/CE方向，证明错标签或错误参数集合会失败；禁止复制被测函数作答案。旧 task timing/GAE/positive cost/顺序 mask gates在输入代码哈希不变时复用，新增部分必须新增反例。

四、PPO初始化状态及版本

BC结束后，保留其 controller参数；新建原 R44完整 PPO Adam，必须包含shared trunk、actor、task_head、cost_head，空 moments、step0。不得携带BC optimizer状态，不得恢复R44 ep128/旧Adam。λ=1、episode0、update0，原登记 data order不变。

BC应使用私有/确定性计算RNG；正式rollout起点显式复位至同seed R44登记 fresh ep0的Python/NumPy/Torch（以及真实使用的其它RNG）状态，核验下一输出和参数/空optimizer/游标。不能根据当前ambient状态声称自动相同。保存“R44基初值→BC模型→RL初值”三段身份，BC-RL新训练版本。旧已训 checkpoint不得静默恢复到本轮；当前本轮中断恢复只接受同版本完整controller/双critic/Adam/dual/history/游标/RNG及hash链的已提交状态。

五、固定启动预算和自动继续门禁

本轮全新、不可挪用旧reserve：
- BC初始化模型 greedy开发控制：3seed×16既有开发场景 = 48正常；
- 约束PPO训练：3seed×128集 = 384正常；
- 最终ep128 greedy/sample：3seed×2mode×16 = 96正常；
合计528正常，技术失败储备最多24，总硬上限552。每次环境构造/claim消耗不可退还，技术失败不进科学行；储备只用于技术失败，不用于效果差重训/补样。无额外smoke/full suite/new stream/reference rollout；纯计算forward不占环境额度，但如实计。

旧参考、规则和R44配对结果：只读复用已登记64 immediate、32 rule和96 R44模型eval，逐文件hash/key/config/model绑定核验。不能补跑它们或重新挑 late。旧参考若不可复用则停，不能消耗本轮新额度偷补。所有场景已用，不称独立测试。

首次环境前分别冻结：BC源码和训练计划；postBC三个模型；528逻辑key/claim预算；研究配置差异；各推理/训练入口版本与输入hash。新生成机读件保持确定UTF-8/LF；旧输入hash精确字节保留。不因只写报告而要求当前HEAD严格等于执行HEAD；用冻结内容＋合法祖先/允许路径范围核验，避免再造自阻断。

顺序：
A. 跑48个 BC-only greedy 控制（只读模型、不恢复optimizer，不额外复制sanity场景）。首场景每seed一格属于48，验证模型/动作/计费/奖励时钟/实际teacher不在推理路径/参数加载前后不变，再自动跑全48。
B. “可行低成本初始化”开发门禁：每个seed各自、每个场景保持 immediate按时目标集合（不是只任务总数），且该seed 16场景均值成本≤late_discovery_gate_147的固定均值。三seed均满足才自动继续。此为事前固定的开发继续条件，不是统计确认或δ=0总体保证。
- 若未过，停止本轮：48或更少成功控制之外0PPO/0追加训练；如实报告BC失配/成本/任务集合，不再调BC、不强制rule动作，不用表现较好的单seed晋级。
- teacheragreement只是诊断，不能用100%agreement替代任务/成本门禁；也不要求事后全包轨迹与teacher一致。
C. 通过B后，从各postBC模型各128集/32批约束式训练，首4集真实更新门禁计入384。核验目标符号、costhead真实参数/gradient/clip、next-value/时间归属、顺序likelihood、λ_before使用及λ更新、权重链/optimizer/RNG恢复。这些检查必须穿真实batch/actor/optimizer，不复制实现当预期答案。合格就自动跑剩余，不按中间结果改参数。
D. 全部训练完成后只评估ep128固定模型96格，不选ep64/best checkpoint/最好seed，不改变greedy tie-break或sampled随机规则。评估每逻辑key固定RNG，样本一次，不把不确定性说成完整消除。训练checkpoint/计数日志评估前manifest，评估后逐件hash/大小与日志hash/行数不变。

六、执行与机械失败纪律

复用已经验证的 counted JointRun proxy、严格行/claim/receipt消费、单一游标推进和固定恢复接缝，不复制另一个“差不多”的runner；不要对slotted run属性赋值。补0环境连续至少3key消费回归及失败claim/已完成key不重跑反例。

在不改变科研/模型/监督标签/学习数学/超参的范围内，授权机械实现故障的一次最小版本化修复与技术reserve重试（每失败logical key最多一次retry，仍总552内），保留原异常/claim/冻结头/progress/原件。构造前纯计数缺陷单列，不冒充已跑场景。
- 已完成的episode、监督update、PPO update、评估key不得重放。
- 若中断后完整模型/Adam/dual/游标/RNG状态不可证恢复，停止而非重新训练已提交批。
- 需改teacher/观测/算法/预算/超参/世界，或超过机械授权/额度，则报告并停止。
在最后结果收尾时可0环境修分析/序列化/统计消费者与其独立反例；不重新跑整入口，不靠改index掩盖真实科学行缺失。原始失败不得改写complete；只把合法retry纳入严格索引。

七、分析与判读——必须区分模仿和RL附加价值

每seed、greedy与sampled分开，输出绝对按时目标数/计量字节/首报/拒付/双动作数及原始整数，按场景目标ID给丢失/救回/替换；开发规则同表。
比较：
1) BC-only greedy vs immediate/late（48控制）；
2) 最终greedy vs自己的BC-only greedy（保留还是破坏可行起点、成本是否进一步降低）；
3) 最终ep128两模式 vs R44 random-init同seed同mode与规则；
4) 三seed各自稳定性，不能池化掩盖失败。
没有BC-only sampled控制，不能声称sampled相对BC-only的RL附加值。

主开发成功要求：三个最终greedy均逐场景保持immediate按时任务身份集合，且每个seed均值字节低于late；同时独立报告RL是否超越自己的BC初始化。只有BC模仿成功、RL原地不变，写“规则初始化可行，RL附加价值未建立”，不冒称方法胜规则；即使某seed更好也不晋级独立确认。目标数已满6，不要求“提升到>6”。
描述性配对bootstrap以场景聚类、固定一次共享索引，均值/CI/判读同一机读源；按场景target不是独立样本。声明已使用开发数据、阶段门禁选择、单次sampled、BC从rule知识获益的限定。CI含0、16场景全等不是正式等效/总体δ=0保证。

八、一次完成的本地收尾

输出：执行依据/配置差异与版本/BC数据来源与label合法性/监督fit和validation指标/模型与optimizer/RNG身份/48控制及继续gate/实际启动台账/若继续则96最终完整矩阵与逐批训练记录/目标集合配对结果/报告/verdict/原件外部hash清单/状态入口。
各真实计数单列：BC supervised optimizer、合成optimizer、PPO/Adam、环境构造/完成步、失败/retry/checkpoint写入，不能把384集算成384训练进程。
源码和首次启动前freeze本地提交；按问题显式add，本地提交最终报告，tracked干净，旧weights/logs/streams不入Git，其它未跟踪件不动。本轮不新建spec，不推送、不merge、不部署、不另开确认实验、不启动ROUND46。最终回复给完整源码/执行/交付HEAD、真实状态及关键绝对结果，缺项missing，按以上门禁自动完成后停止。