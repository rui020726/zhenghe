执行 ROUND42-R 的门禁修复与精确恢复，随后完成原固定计划。不要新增研究实验。

一、目标与已知状态

研究主线：冻结搜索与任务分配模型，训练一次性 SEND_NOW/SKIP_EPISODE 首报选择策略，研究保任务降已计量通信成本。

当前分支：
fix/comm-one-shot-runner-counting

已知复核 HEAD：
2825069eaf9acf0fff87a9a02e69ae2f8e88c785

原预启动 HEAD：
f46fe4703aa03ce6400f63c601c3203ef741dd9d

当前累计：
- normal claims 68，包含最初的构造失败；
- reserve claims 1；
- 总 claims/构造 69；
- 完成 episode 68：64 个 immediate 参考 + seed11 首四集；
- seed11 已完成一次 PPO 更新、12 次 Adam step；
- 剩余正常逻辑任务 604；
- 剩余 reserve 27；
- 原累计启动硬上限 700，不增加。

seed11 首批 gate 停在：
RuntimeError: seed 11: frozen PILOC/CTDE checkpoint path set changed

现有证据指出：批记录使用反斜杠，gate 使用正斜杠；规范化后路径集合一致，两份模型文件的批记录、冻结件和当前文件 SHA256 均一致。

这是待正式复核的门禁问题，不是 gate 已通过，也不是新的策略效果结果。

原执行入口还拒绝已有 progress，并要求 HEAD 等于预启动 HEAD，因此不能直接重启旧入口。

二、本次授权

允许：
1. 最小修复 checkpoint 路径身份比较；
2. 增加显式、版本化的恢复入口及内容身份核验；
3. 零环境、零 optimizer 更新的回归和恢复检查；
4. 对 seed11 原首批材料完整重跑只读 gate；
5. 门禁与恢复验证通过后，从 episode4 自动继续原固定计划。

不需要上述验证通过后再次请示是否继续。

禁止：
- 重跑已完成的64个参考；
- 重跑 seed11 episode0–3；
- 重复 seed11 batch0 的 PPO/Adam 更新；
- 改数学、策略、网络、观测、预算、奖励、GAE或超参数；
- 整个seed从头重训；
- 追加训练、场景、调参、部署或推送。

三、起点与原件保护

先读取适用 AGENTS.md、当前状态、冻结配置、原执行依据、停止报告、账本及 checkpoint 索引。

从实际最新交付头建立独立分支：
fix/comm-one-shot-gate-resume

要求已知复核头为其祖先。
如已有后续报告提交，核对差异并登记实际完整 SHA，不猜测或回退到旧执行头。

保持旧：
- prelaunch identity；
- gate 失败记录；
- stop state；
- claims、账本；
- batch0、tape、日志及 checkpoint
全部原样保留。

旧 gate 的失败不能改写成成功。
新结果使用独立的 gate 复核文件与恢复清单，引用原件路径和哈希。

其他会话文件不动。
先检查已有进程及最新进度，避免与同一任务并发重复执行。

四、最小修复路径身份比较

定位真实首批 gate 的失败代码，修复实际调用路径，不只在报告脚本里替换字符串。

路径规范化必须：
- 使用明确的冻结根目录；
- 保留 drive、目录层级及文件身份；
- 只接受等价路径表示；
- 显式拒绝规范化后键冲突；
- 对正确路径对应的每个文件重新核验冻结 SHA256。

不能仅按 basename 匹配，不能因哈希相同就接受任意其他位置的文件。
不能删除路径断言或降级为“只要两个文件存在”。

先增加能让旧实现失败的合成回归：
1. 同一冻结路径的正斜杠/反斜杠表示应等价；
2. 不同目录中的同名文件不能冒充冻结输入；
3. 内容哈希不同必须失败；
4. 文件缺失必须失败；
5. 两个键规范化后碰撞必须失败。

测试必须穿过真实 gate 的路径比较函数。
本阶段不构造环境、不执行 optimizer、不写训练 checkpoint。

五、新恢复身份，不能伪装原冻结身份

保持算法与训练语义：
one-shot-send-skip/1

保持原训练版本：
constrained-cost-v2/one-shot-send-skip/1/published-completion-event-at-relay-tick/2

保留透明代理：
counted-jointrun-proxy/1

另外登记 gate/resume 执行版本及新代码哈希。
原预启动 HEAD、原配置哈希不修改。

显式替代旧入口的“当前 HEAD 必须等于 prelaunch HEAD”条件：
- 原执行头必须是声明起点的祖先；
- 生产依赖、科学配置、模型与训练数学内容身份不变；
- 只允许本次登记的门禁和恢复代码差异；
- 未登记的执行代码差异必须失败。

不能简单删除 HEAD 检查，或把当前 HEAD 写进旧 prelaunch 文件。

原运行阶段与恢复阶段可能有不同执行身份，必须在清单中分别登记；不得把所有旧行重标为新版本。

六、完整重跑 seed11 首批只读 gate

输入必须是原 episode0–3、batch0、初值和已保存更新证据。
核验原件哈希后复用，不重新采样或更新。

必须运行完整 gate，包括原失败断言之后未执行的逐文件核验。
不能根据 traceback 推断前面通过，就手工拼一个 passed。

likelihood 使用 batch0 更新前的登记权重重算，不能拿 after_b0 权重代替。

按原契约核验：
- 首次 SKIP 为真实自由动作；
- 后续关闭候选不泄漏 likelihood/entropy；
- 顺序预算 mask；
- likelihood 容差；
- 奖励实际事件时间归属；
- 正成本与组合优势；
- GAE；
- 对偶统计与更新；
- 参数、优化器集合及链；
- 冻结搜索/分配路径和文件内容；
- 实际计数及版本。

新 gate 通过只表示原首批经修订门禁复核有效，不代表策略达标。
任何非路径格式问题均单独报告，不能为继续运行放宽断言。

七、零更新验证精确恢复

批准的恢复源：

v2_train_s11_after_b0.pt
或与其字节完全相同的 rolling checkpoint。

预期 SHA256：
a85722e0864456ea3eb735195c7ffcde4be9d42301bdf2bc0781db6747a851ef

预期游标：
- next episode = 4；
- completed PPO updates = 1；
- optimizer steps = 12；
- lambda = 1.0125。

先只读验证，再通过真实恢复入口装载到内存。

核验：
- actor、共享网络、task critic、cost critic；
- optimizer 参数对应关系及全部 Adam 状态；
- lambda、历史、数据顺序和批游标；
- Python、NumPy、Torch RNG；
- 实际代码使用的其他 RNG/generator，如存在；
- 训练版本、动作语义和源冻结身份。

不能仅检查 forward 相同。
独立核对加载后的参数、Adam step/exp_avg/exp_avg_sq 等状态与源 checkpoint 相符；验证 RNG 状态及下一次随机输出。

模型构造、只读前向或诊断可能消耗 RNG，因此正式恢复时，必须在这些操作之后、实际继续运行之前恢复登记 RNG。

若实际使用的必要状态缺失，不能声称精确恢复，更不能用默认 seed 补齐。
此时保留证据并停止，不从初值重训掩盖缺口。

上述检查：
0 环境启动；
0 optimizer.step；
0 训练权重写回。

八、显式 resume，不删除旧产物绕过拒绝

为已有 progress/产物增加明确的 resume 模式。
不能删除它们、清空目录或强制旧入口当成全新训练。

resume 必须：
- 仅接受本次批准、身份匹配的完整状态；
- 已完成任务按账本及 final record 复用；
- 验证 checkpoint 游标与已提交批记录一致；
- 不覆盖旧滚动 checkpoint 和 batch0；
- 将续执行输出写入隔离目录；
- 对重复提交、未知产物、状态冲突 fail closed；
- 保留旧停止状态，以新的恢复记录显式接续。

新阶段必须保存可恢复模型、optimizer、lambda、RNG和游标。
不能只保存权重再声称可以精确续训。

九、通过后自动完成剩余604个逻辑任务

原计划总数仍为672，不另开一轮672。

剩余：
1. seed11：episode4–127，共124；
2. seed22、seed33：各从登记初值训练128集，共256；
3. OLD/NEW ×3 seed×greedy/sampled×16开发场景，共192；
4. late_discovery_gate_147、always_hold×16开发场景，共32。

合计：
124+256+192+32=604。

64个 immediate 参考直接复用，不再运行。

seed11 下一批从 episode4 开始，不能再次执行 batch0。
seed22/33 仍从登记初值开始，各自首四集门禁计入128集；门禁通过后继续剩余124集。

OLD 使用 ROUND40 NEW ep128 模型。
OLD/NEW 均使用同一 one-shot SEND/SKIP 协议，greedy与sampled分开。
不启用 hazard decoder。

训练块：
1300000..1300047，原冻结128集顺序。
开发块：
1301000..1301015。
均为已使用开发数据，不是独立确认集。

按原冻结配置完成，不根据中间结果修改任何参数。

十、累计额度闭合

当前已用：
normal68 + reserve1 =69。

后续604个正常任务使用剩余604个 normal slot。
如无新增失败：
- normal累计672；
- reserve累计1；
- 总启动累计673；
- 成功逻辑任务672；
- reserve剩27。

700是整个ROUND42及恢复阶段的累计硬上限，不是本阶段新增额度。
全部阶段账本必须合并核验，但原账本不改写。

首批 gate 失败是一次验证阻断，不再额外算成一次环境启动；也不能因此返还已完成的四集。
原构造失败仍为技术失败，不进入科学结果。

储备仅按原技术重试纪律使用；不得用于额外smoke或研究扩样。
恢复代码若出现新的有效性缺陷，停止受影响运行，不能热修后静默继续。

十一、最终分析和提交

固定使用三个seed最终ep128，不挑最佳checkpoint或seed。

沿用原比较和统计：
- 每seed、每模式绝对按时任务数、计费字节、首报数；
- NEW−OLD；
- 对immediate、late、always_hold；
- 逐目标丢失、救回和集合交换；
- 主动SKIP与预算mask区分；
- sampled/greedy分表；
- 训练约束与部署保证区分。

先回答：
固定计划是否完成；
累计额度是否闭合；
是否取得保任务降本证据。

全SEND等于immediate不代表学到附加降本；
全SKIP丢任务不算成功；
开发块结果不建立泛化或总体零损失保证。

最小交付：
- 路径门禁回归与新首批复核结果；
- 精确恢复证据和执行版本清单；
- 累计账本；
- 固定计划结果；
- checkpoint/原件索引；
- 完整HEAD与复现命令。

门禁修复、恢复入口、结果按问题独立本地提交。
不重跑无关历史审计，不运行账本外环境测试套件。
不推送、不部署、不追加实验。

本提示词授权验证通过后直接续执行并完成固定计划，而非只提交一个恢复设计再次停下。