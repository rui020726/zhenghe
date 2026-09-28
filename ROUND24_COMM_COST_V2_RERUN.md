# ROUND24：修复版 N4 从初值重跑（有界开发 pilot）

日期：2026-09-28。待执行；此文发布本身不表示已经启动训练。

## 0. 唯一执行入口、起点与权限

规划端已核验远端修复分支 `fix/comm-cost-stage-c-training-contracts` 的 tip：
`64cddcffc6f8796f132df8b8e003f005d0bdd749`。
`8970ce964ca820e85bce19537f35b98dff08a682` 是其祖先；四个修复和后续证据提交均在此链上。

接口发布导致本地/远端commit元数据不同本身不是代码错误。执行端先比较本地修复tip与上述远端tip的完整tree与相关文件blob；相同时选择已核验远端头作为统一新起点；不相同时停止并列diff，不强制覆盖或重复cherry-pick同一修复链。

从该头导入本次单文档提交，建立独立 `exp/comm-cost-v2-rerun` 分支，保护他人改动。读取 AGENTS、总纲与当前状态，以及 ROUND22、ROUND23、修复版四项证据和版本manifest。本文件是ROUND22的重跑补充：冲突项以此文为准；ROUND23原有禁止重跑的阶段边界由本次明确授权替代。

用户转交本执行指令后，允许本次固定预算内的新环境运行与通信模型训练、必要实验harness接线、合成测试、报告、普通提交/推送。不允许搜索/CTDE训练、改生产默认值、解封确认集、调超参或自动追加训练。

## 1. 为什么重跑，以及哪些证据保留

旧Stage C实际运行完毕，但错误的成本符号、optimizer参数集合、GAE next-value与奖励归属使其不能检验正确N4学习效果。ROUND23源码修复与独立反例是重跑依据，不是学习收益证据。

旧677次启动、权重、tape、rows、verdict和勘误全部保留；旧checkpoint只作只读审计，不作warm-start。旧规则比较只能作开发观察。

这轮是对已知开发场景的修复版复验，不是新独立测试。不是调lambda、提高预算或增强模型，也不是恢复旧训练。即使修复后仍未胜规则，也只说明本次资源内未建立收益。

## 2. 启动前零环境门禁

1. 重新运行ROUND23的独立反例合成套件，记录实际数量、合成optimizer.step数、代码身份；不能沿用历史20/20字符串。
2. 核对修复版实际batch_samples/PPO路径使用正cost/B0、`lambda*A_task-A_cost`，非终止next-value正确，奖励归给最近已采样动作，prefix单列。
3. 参数完整集合从四个模块独立构造，fresh/resume/zero_grad/clip均覆盖cost_head，无重复参数或外来参数；v1恢复必须拒绝。
4. 冻结全部已登记超参：精确抄录旧Stage C机读设置，与新运行字段逐项diff。只允许已登记的四项正确性修复、输出隔离和本文件的评估安排差异；不得趁重跑调LR、epochs、entropy、clip、网络宽度或特征。
5. 逐项检查写路径和读路径：rows/logs/checkpoints/registry/manifest/ledger/报告/恢复脚本全部指向本次run_id，旧目录保持只读。特别检查原sc_recover中旧提交/旧manifest的硬编码；未证明适用于v2则禁用该恢复入口，不从旧tape重建新训练。
6. 下一节数据块本来就被ROUND22使用：登记为intentional_development_reuse，仍审计其他用途，不通过把历史目录排除掉来宣称全新未见。封存块不读、不生成。

若以上出现新数学/训练正确性问题，0新环境启动下停止并报告；普通输出接线问题可在冻结前最小解决。禁止运行途中修改训练路径。

## 3. 保持研究配置不变

- 训练场景1300000..1300047（48），开发场景1301000..1301015（16）。
- 训练初始化seed 11/22/33；场景顺序RNG=20260927，与旧计划同序，128集=48+48+32。
- 每seed最多128集/128000环境步、每4集一批；合法提前结束不补episode凑步数。
- B0=10000 B，delta=0，lambda初值1，对偶eta=0.05；每批PPO期间lambda固定，批后按配对任务损失更新。
- gamma=1、GAE lambda=0.95按决策行递推；不做分量或组合advantage标准化。
- 真实PILOC与冻结CTDE，6目标/1000步，normal-lossy成功率0.7、29300 B硬预算、policy_owned_timing、ACK关、无公共强制发射、保留本地感知/gossip。完整设置继承ROUND22冻结配置。
- 费用仅指已计量载荷尝试成本，不称全系统能耗。拒付实际计0，丢包照付，主动HOLD/执行器限制分别记录。

delta=0是训练目标，不是总体或逐局安全保证。reference是同场景同配置immediate；prefix任务奖励保留在episode任务数和约束估计中，但不能塞入尚未发生的动作的actor return。

## 4. 新一轮资源账本：正常672，储备28，硬上限700

这是独立新预算，不退还旧677次消耗，也不动旧账本。

| 用途 | 数量 | 启动 |
|---|---|---:|
| 3 seed从初值训练 | 3×128 | 384 |
| immediate参考重跑 | 48训练+16开发 | 64 |
| 学习策略开发评估 | 3 seed×3种评估块×16 | 144 |
| 其他五条规则重跑 | 5×16 | 80 |
| 正常合计 | | 672 |
| 技术故障储备 | 不用于调参/扩样 | 28 |
| 硬上限 | | 700 |

所有环境启动包括失败、reference、规则、评估和smoke都计数。合成optimizer更新另计，不混入真实训练步。无免费吞吐探测、额外“确认一下”或整套仿真smoke。

reference和规则本轮重新运行，避免把输出路由/身份不兼容当成隐形缓存。旧原件不覆盖；同环境配置的规则复现差异只供技术核验，不选择更好的一批。

## 5. 明确训练策略与评估策略；评估总数不增加

旧计划在0/64/128集各做一次贪心评估。本轮**在首次启动前**改为以下三个固定块，仍是每seed48次开发评估：

1. 初值，greedy模式，16场景；
2. 128集最终checkpoint，greedy模式，16场景；
3. 同一个128集最终checkpoint，sampled模式，16场景。

不跑64集开发评估，以腾出相同48次总额度检查最终采样策略。训练仍保存64集checkpoint与批次曲线，但不得事后拿64集替换最终模型或额外评估它。

sampled模式必须与训练的masked随机策略一致，greedy只是另一个明确命名的决策规则。对偶约束估计来自训练采样策略，不能把它自动当成对greedy策略的保证。

评估动作RNG单独固定，例如稳定hash("round24-eval",training_seed,scenario_seed,mode)，完整算法与种子表在协议冻结；不能使用Python进程随机hash，不污染环境/信道RNG。每个场景每模式一次，不采样到得到好结果为止。

checkpoint、rows和tape命名都含run_id/training_seed/scenario_seed/timepoint/eval_mode/attempt。相同最终权重的greedy与sampled不可互相覆盖或混入同一统计表。sampled主表用于观察训练策略，greedy另表用于部署规则诊断；不在结果出来后择优挑模式。

16场景和每场景一次sampled评估只作探索；训练seed不是新的独立场景，3×16不能写成48个独立场景。零差、CI含0、退化bootstrap区间均不构成delta=0总体非劣证明。

## 6. 六条固定规则不变

immediate是reference；其余为always_hold、late_discovery_gate(147)、periodic(interval20)、random_send(p=.5)、event(progress_threshold=.2,urgency_time25)。使用旧轮已冻结的实际字段名、门控边界和RNG登记；不重新挑late阈值，不加入接收端真值。

send_once若仅为同配置immediate别名，说明语义，不另加运行。旧报告“强规则”是开发观察，不预先宣布任何规则总体最优。

## 7. 首4集内嵌 gate：通过后同轮继续，失败则停止真实训练

先执行seed11首4集，计入384配额，不额外开一轮试验。

验收至少包含：

- 用实际正费用、原始tape和冻结value进行独立区间/GAE重算；不只再次调用被测函数作为预期。
- 同tick、中间奖励、prefix、末决策tail归属正确；assigned+prefix与账本/同窗口任务数闭合。
- optimizer参数对象集合在运行进程与四模块独立集合一致；记录模块名/形状/权重摘要/optimizer state，确认实际cost_head被纳入梯度清零/裁剪/更新。若某个实际梯度恰为0，不以随意改测试预期绕过，使用已通过的受控合成测试解释。
- rollout更新前log_prob可重算，初始/更新后对齐不可混用；NaN、非有限loss/grad或参数异常即失败。
- lambda根据同batch的完整配对任务数更新，方向及单位正确；reference无梯度。
- 搜索/CTDE运行参数与文件哈希不变；通信权重确实更新，旧v1checkpoint没有被读取用于训练。
- 完整保存v2 checkpoint，纯计算恢复后下一次合成前向/状态一致；不额外启动真实episode测试恢复。

通过后按固定计划完成三seed；不以首4集是否胜规则作为正确性门禁。技术失败消耗储备；涉及目标/特征/时序语义的新修复则停止该批并报告，不能把不同训练版本拼起来。

首4集之后每批保留关键账本、参数集合、cost_head权重/梯度、lambda/g、likelihood与冻结权重核对，防止“首批绿，后续换路径”。

## 8. 原始证据必须可复核

使用独立输出目录，例如 `runs/comm_cost_v2_rerun_20260928/`。启动前提交协议与机读配置、数据复用登记、代码依赖内容哈希、训练/评估计划。

保存不可变的每seed初值/64集/128集checkpoint；另设rolling恢复checkpoint，不能拿它覆盖里程碑。保存训练状态的actor/双critic/optimizer/lambda/数据顺序/全部相关RNG/schema、参数模块清单及哈希。

保存每批费用/任务/critic loss/entropy/KL/clipfraction/gradnorm/可执行actor样本数；保存必要的逐决策positive cost、task reward、next value、advantage/return或可无歧义重算的字段。prefix、终止/截断标志与eval_mode要显式存在，不能事后靠猜补齐。

原始权重/log/tape不入Git，manifest记录路径/大小/哈希。合成测试实际数字与运行命令落盘。训练中的故障恢复必须只用本轮合法v2状态，验证RNG与数据位置；失败历史不删，恢复不能挑更好结果。

## 9. 判定、发布与停止

交付执行状态与科学观察分开：execution_complete、correctness_gates、sampled_comparison、greedy_comparison、constraint_diagnostics。不把运行完成写成科学验收通过。

分别报告每seed最终sampled/greedy对immediate和全部规则的任务差、字节差、不确定性与逐场景方向。任务改善数、字节改善数、联合dominance分列。不得挑最好seed/模式、池化消息、以训练reward上升证明任务约束成立。

旧v1结果只能在独立历史表说明修复前后，不混入新平均数，也不能从两轮差值单独归因到四个修复中的某一个。

允许普通提交/推送与远端核验；本地/远端因接口提交元数据不同则登记各SHA和tree等价证据，不声称普通push成功。必要时使用已授权GitHub通道，不force、不读取或暴露凭据。

完整执行、正确性失败或700次额度耗尽后都停止。没有自动第三轮训练、没有独立确认/部署、没有调lambda/eta/delta或扩样权限。结果不理想不等于应回到搜索优化主线。
