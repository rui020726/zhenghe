# ROUND50 执行依据

任务：`ROUND50-SHARED-TRUNK-GRADIENT-MECHANISM`  
起点：`analysis/round49-bc-input-policy-closure`，完整 HEAD `137f006da31da09535ad9245da12c2fc5f6d1a38`。  
目标分支：`analysis/round50-shared-trunk-gradient-mechanism`。

## 目的和背景

研究冻结搜索/任务分配模型下的学习通信保任务降本。ROUND49已更正ROUND48把R43旧ep0用于概率前向的加载接线错误；ROUND46实际训练初值是同seed R45 BC_POST。ROUND49首批观察：三个seed在BC实际轨迹上各96个决策，均33 SEND/63 SKIP；同输入初值→ep128转换s11=33/0/4/59、s22=30/3/0/63、s33=33/0/3/60。它们说明策略映射变化，不说明更新原因。

现有开发块表现：BC 5152.125 B/集，PPO ep128 s11/s22/s33 为5778/4694.25/5621.625 B/集，late 4838.25 B/集；16个已多轮使用开发场景中，贪心模型保留immediate按时目标集合。成本相对BC的冻结CI含0，`development_go=false`。

本轮只分析真实BC初始化训练的共享主干梯度耦合：actor目标、entropy、task critic、cost critic分别怎样影响共享主干和动作logit。共享critic影响策略可属正常架构设计，不能仅凭结果差判实现错误。

## 真实输入、代码和模型绑定

- 按ROUND46冻结配置、V6训练清单、首批门禁、训练行/批记录及tape精确索引，读取每seed batch0的4个已存episode和更新前BC_POST初值。验证来源、时序、old log_prob容差及批内lambda/advantage/return接线。
- 初值角色和真实加载接缝沿ROUND49复用；不得使用R43旧ep0或其他轮次同名权重。记录冻结训练代码/损失表达式身份、路径、函数/源位置和内容摘要。
- 只分析batch0完整批。实际随机minibatch索引未存时必须标missing；不把完整批梯度说成当时某个minibatch的实际更新。
- 若真实批/代码/模型身份、时间或likelihood关键门禁冲突，停止受影响的数值项，报告具体缺口；不得回放训练来补证据。

## 受控梯度分析

只读CPU autograd.grad。尽量用真实纯loss接口；若loss只存在于含ppo_update入口，只可对冻结源的纯tensor/loss表达式做可审计源绑定提取，不调用更新入口、不更改生产代码。

按真实实现拆 actor surrogate（必要时task与正cost局部项）、entropy、task value、cost value、真实总loss。保留真实系数、归约、ALL_ROWS/顺序mask、advantage归一化与B0/lambda时点。独立从named_parameters区分共享主干、动作头、task head、cost head及参数对象，不依赖parameters()汇总。记录每项组梯度范数、分量夹角/投影、加权分量和与总梯度闭合。全局裁剪仅计算预期范数与公共缩放，绝不更新参数。

在ROUND49完整真实输入支持上，按实际动作映射计算每seed初值的 `J_logit · (-g_component)`；logit为SEND-vs-SKIP log-odds。相同float32输入可缓存Jacobian，但保留逐目标记录。它是完整batch0下负梯度方向的一阶SGD敏感性，不是实际Adam/PPO更新，不是ep128最终漂移的因果分解；不能把分量直接加和解释Adam归一化后的多步更新。历史成本类别仅元数据，不进actor、不挑输入。

## 边界

环境构造0、仿真0、新流0、真实训练更新0、合成optimizer.step0、checkpoint写回0。允许已有文件读取、CPU纯autograd.grad、只读模型加载、小型纯计算回归、报告/脚本和本地提交。

禁止训练/评估运行器、optimizer.step、参数更新、恢复训练optimizer/RNG、写入.pt、修改训练器/架构/loss/归约/超参、创建新场景/冒充真实观测、全量测试套件、安装、联网、推送、合并、部署。不得重扫历史账本或晚期checkpoint；受保护的两份ROUND46脏文件和其他会话文件不改、不暂存。

## 回归、结论和交付

独立纯计算回归用可手算共享线性主干验证：critic到action head直接梯度可为0，但经共享主干对动作logit的一阶作用可非0；闭式导数是预期。不将结构现象称为bug。测试源绑定、正确BC角色/错误R43拒绝、分量加和、变量归约、模型文件/参数不变及七项零操作。

交付：`SOURCE_AND_INPUT_IDENTITY.json`、`GRADIENT_COMPONENTS.json`、`LOGIT_DIRECTIONAL_SENSITIVITY.json`、`PURE_TEST_EVIDENCE.json`、`ROUND50_GRADIENT_MECHANISM_REPORT.md`及必要脚本/纯测试。回答当前首批点actor方向、critic经共享主干影响策略方向、是否足以提出“仅隔离critic→actor主干梯度”的单变量消融。区分共享训练正确性和方法收益；不声称最终漂移根因、loss实现错误、detach保任务降本、欠训练已排除或development_go。

若提出下一步，仅一个有限候选，写明假设、唯一变量、原方案对照、可证伪结果与新增额度；本轮不执行。

显式提交ROUND50新文件。本地提交、不推送。最终报告完整HEAD、身份/缺失、真实计算/测试覆盖和七项零操作计数；`development_go=false`保持。
