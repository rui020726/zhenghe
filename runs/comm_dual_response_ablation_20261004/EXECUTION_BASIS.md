# ROUND44 执行依据：对偶响应速度的单变量消融

本文件归档本轮授权要点。目标仍是学习型通信在保住按时任务的同时，比强规则更低的已计量成本；不是交付全静默或规则替代品。科研执行从 ROUND43 完整交付头 9f2b6b6411c50849ce1a9eedbaedffaf0aededa5 开始，执行源码参考冻结 7ed1a03c445fa31fe90fd70a224a97b8366fa182。报告代码提交 8730418882bf6f6d814debce249329665bcb057e。原 ROUND43 树、权重、tape、日志和失败记录只读保留；新工作在独立分支 exp/comm-dual-response-ablation，不覆盖用户文件、不推送。

## 科学假设与唯一自变量

检验更快对偶响应是否改善任务保持及其成本代价。唯一研究变量为每个 batch 的 eta_lambda：0.05 改为 0.5（十倍）。这是优化方法消融，不是修复旧公式，不改 GAE 系数或对偶初值；不做网格搜索、不依据中间结果换步长。

保留 ROUND43 共享 GAE_LAMBDA=1、gamma=1、lambda_dual 初值 1、delta=0、B0=10000 B。成本 c=C_measured/B0；g_hat=mean(Y_immediate-Y_policy)-delta；Lmax=-E[c]-lambda_dual*g；actor=lambda_dual*A_task-A_positive_cost；每 batch 一次 dual_next=max(0, dual_before+eta_lambda*g_hat)，actor 使用本批更新前 dual。

其他训练和执行契约全部沿用 R43：19维合法发送端观测、期限编码、one-shot-send-skip/1、顺序预算 mask、ALL_ROWS、正成本输入和双 critic、同 tick 奖励、GAE next-value/尾段、optimizer 集合、PPO 超参、data order、终止条件及 greedy/sample 解码。PILOC/CTDE 权重、1000步/6目标、29300 B、ACK关闭、policy_owned_timing、信道与旁路字段均冻结。按载荷×订阅者×发送尝试计费，丢包收费；拒付不算节省，也不将该口径称为全系统或真实空口成本。

训练新版本必须反映 eta 变化；不从 R43 ep128 热启动。R43 NEW ep128 仅作只读 OLD。科研配置 diff 仅允许 eta_lambda 改变；路径、臂名、版本、报告和本依据授权的机械索引修订另列。保持 CPU、原入口、线程与并行设置，不自行增加线程或并行训练。

## 执行器机械性能修订

R43 每个评估格会在含384条训练行的目录上重复扫描，progress 又重复查询96个 key，造成约100秒/格的非科学开销。本轮冻结前允许只在 R44 执行器做机械索引修订，不改 R43：

- 对已存在 row/claim 只读建立一次有效 key 索引；同时核对 kind、arm、seed、mode、scene、attempt、claim complete、版本、checkpoint 身份及产物完整性。仅唯一完整成功 key 可复用，不能以文件存在推断成功。
- 新格只有在成功落盘并验证后才能增量登记 key/progress；最终对整个矩阵独立复核。重复成功、缺产物、不匹配、外来标签和空目录均 fail closed；不移动/删除训练行，不改账本以提速，不挑选重复 attempt。
- 通过真实旧查询和新索引入口的合成 JSON/claim fixture 对照 key 集、拒绝结果与调度顺序，覆盖训练/foreign 行、重复成功、失败 claim 加合法重试、缺产物、错版本/hash、空目录。该门禁构造/环境/optimizer/权重写入必须为0。
- 机械执行器版本与 eta 科研版本分开登记。复用已核验 trainer、R40 resume adapter、代理与计数 hooks，不复制 PPO/GAE 实现或另造整仓审计框架。

## 计划、额度与复用材料

- NEW 三 seed 11/22/33，各128集、32批，共384训练 episode。首4集 gate 计入128集，不额外 smoke。训练场景1300000–1300047；沿用 RNG=20260927 和顺序 SHA256 346f2cdde8d658f927ea282f83e143d54a600f4f5c729e4521fe657421afe840。
- 只评 NEW ep128：三 seed × greedy/sample × 16开发场景1301000–1301015，共96格。
- 本轮新额度为480正常逻辑任务加24技术失败储备，504总 claim/构造硬上限；不继承 R43 的600、R42余额或未用 baseline 额度。失败 claim 不退还；reserve 仅真实技术失败按冻结纪律使用，效果差不是技术失败。
- 只读复用 R43 已核验的64 immediate、32 late/hold规则、96个 R43 NEW ep128评估作为 OLD；不得误用 R42/40 模型。启动前复核模型、配置、场景、哈希、解码、RNG和成本语义；缺失/不匹配即停，不补跑基线。
- 不新增场景或流，不触碰800000–800399、900000–900199或确认块。所有结果仅属开发用途。

## 启动门禁与停止条件

在首次环境启动前，提交依据、最小实现、配置 diff、初值/RNG 对比、复用登记、额度计划和零环境纯计算门禁。不得执行含真实环境构造的全量 suite；合成 optimizer 更新为0。

真实 dual_update 接线测试使用独立预期：lambda=1 时 g=+0.25 得1.125，g=-0.25 得0.875，g=0保持不变，负值投影至零。验证 eta 真实进入训练 batch 及 checkpoint/恢复版本，而不只是写在配置中。

每个 NEW seed 从 R43 对应 ep0 参数、空 Adam、lambda=1、已登记 Python/NumPy/Torch RNG 与 data order 开始；首次 rollout 前显式设置实际消费的 RNG，不能只依赖 fresh_controller(seed)。恢复入口拒绝 R43 训练后状态。

seed首4集 gate 必须通过真实 tape→batch/GAE→actor/双 critic→optimizer→保存/恢复路径。与 R43 同 seed、同初值和 RNG 独立对比首批行为、特征、动作、奖励、likelihood 和参数；batch0 actor 仍用 lambda=1，eta 差异只影响批末 dual 更新。核验 mask likelihood=0但 critic 保留、同权重概率误差≤1e-6、奖励/成本总量及同 tick 时序、配对 g、参数/梯度覆盖、版本边界。三个 seed 的首批 gate 都是正式训练集的组成部分。

首批通过后按固定顺序完成384集、96评估、分析和报告，不逐块等待。更新次数据实际样本计，不强求等于R43；不得增加128集/seed或32批/seed。真实运行失败或有效性问题须保留原件/claim并停止受影响工作，不默默重放更新、不热修冻结代码。零启动报告/路径/簿记故障只有在科研字节不变、完整状态/RNG/游标精确恢复且额度闭合时才可用独立机械版本续行；不得仅凭进度/lock/快照推断进程状态。

## 固定判读与交付

主判据：三个 greedy seed 各自按场景完整保持 immediate 的按时任务身份集合，且各自成本低于 late。仅报告开发块联合观察，不称总体 delta=0、正式非劣或泛化。sampled 独立报告；不池化训练 seed、不挑 seed、checkpoint 或 decoder。每 seed×模式报告 NEW/OLD 的 Y/C/首报/SEND-SKIP、相对 OLD/immediate/late/hold 的配对均值、预登记 CI 和任务身份变化。均值精度保留1/16=0.0625，表格与 verdict 使用同一来源。逐批报告 dual_before/after、g、任务/成本、裁剪和似然摘要。更快 dual 压力上升不等于约束达成；全 SEND 只可称恢复任务但未优于 late 降本，全 SKIP 丢任务不算成功，单 seed 好不代表跨 seed 稳定。

有限交付：本执行依据、冻结/config diff、复用及初值门禁、480任务账本和失败账本、三个首批 gate、checkpoint 索引及 Git 外原件 hash 清单、paired/curves/verdict、单一完整报告和状态。raw logs、streams、weights 不入 Git。报告分别登记科学执行 HEAD 与报告 HEAD。先跑零环境报告纯测试；矩阵完成后仅基于已存结果 report-only 收尾，不因报告错误重跑科研。本轮完成后本地提交并停止，不推送、合并、部署、确认、续训或试第二个步长；证据不足须如实报告。
