# ROUND52：critic-to-shared-trunk stop-gradient 原型

**结论：隔离原型通过纯计算门禁，尚无训练/效果实验授权。** `development_go=false` 保持。本轮只在新目录使用临时 hook 和只读 BC 初值；未改生产 controller、trainer、runner、旧报告或 checkpoint。

已有开发结果仅作背景、不是本轮新实验：immediate 14939.25 B/集，late 4838.25 B/集；BC三seed各5152.125 B/集；当前R46 greedy PPO s11/s22/s33为5778/4694.25/5621.625 B/集。按时任务均为6/集；三个greedy PPO seed在16/16场景保持immediate精确目标集合。该块已重复用于开发选择，development_go仍false；本原型不改变其判定，也不把局部梯度称为成本收益。

## 实际源与接线

精确冻结 trainer SHA256：`0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841`。直接从冻结 `ppo_update` AST 提取相同的 loss-producing statements，删去后续 zero_grad/backward/clip/optimizer 调用；没有导入运行 ROUND51 analyzer 的顶层分析副作用。SEND=1、SKIP=0来自ROUND49输入身份记录。输入为每seed真实BC_POST初值和ROUND46 batch0 4集/265 decision；row/tape哈希及Round51基线梯度身份核验。

唯一方法变量：在task-value head及cost-value head入口，将共享trunk表示detach；actor和entropy仍接原共享trunk。使用临时`forward_pre_hook`，只改变 autograd 路由，值不变；finally反向卸载并断言原hook集合恢复。模型对象和state_dict不变；critic head继续获得正确梯度。

## 原/新真实批次对照

三seed的logits、task/cost预测、全部loss项精确相同；actor/entropy梯度和critic-head梯度与旧路由相同；两个critic到trunk梯度从非零变为unused(None)/精确0。全部265个decision、free/masked集合、ALL_ROWS池化与归约分母均由同一冻结AST代码产生。总梯度按独立named_parameters和对象ID枚举，分项求和闭合；参数、文件SHA不变，`.grad`为空。

| seed | 原总梯度L2 | 隔离后L2 | 源clip norm | 原缩放 | 隔离后缩放 | task trunk L2 原→新 | cost trunk L2 原→新 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 10.586299 | 6.777566 | 0.5 | 0.047231 | 0.073773 | 8.154600 → 0 | 0.455942 → 0 |
| 22 | 25.464657 | 23.040213 | 0.5 | 0.019635 | 0.021701 | 11.005554 → 0 | 0.272716 → 0 |
| 33 | 18.869883 | 17.467926 | 0.5 | 0.026497 | 0.028624 | 6.780646 → 0 | 1.362190 → 0 |

clip norm 0.5是从冻结trainer读取的`MAX_GRAD_NORM`，不是猜测；只是计算同一公共全局clip的预期缩放，没有实际执行裁剪。detach改变总范数及缩放，因此actor分量经公共缩放后的强度也间接改变；不能声称总更新幅度除此之外完全相同。

## 真实输入逐点方向

ROUND49两条轨迹完整面板576行/seed均逐条比较；三seed共1728个`J_logit·(-g_total)`记录。输入前向logits/task/cost值逐条完全相同；相同向量仅缓存Jacobian，原行仍保留。如下为输入集合上方向绝对值均值，列出未裁剪与按每个路由各自全批clip比例后的预计方向绝对值：

| seed | 未裁剪 old→isolated | 预计裁剪 old→isolated | 原/新未裁剪方向符号翻转 |
|---:|---:|---:|---:|
| 11 | 55.44096 → 0.05814 | 2.61852 → 0.00429 | 0/576 |
| 22 | 1319.48820 → 0.00948 | 25.90823 → 0.00021 | 309/576 |
| 33 | 445.03966 → 0.04114 | 11.79233 → 0.00118 | 0/576 |

方向并非只按范数等比例缩小；s22超过半数输入的方向符号变化。以上均为固定BC初值、固定首批full-batch梯度的局部一阶SGD敏感性，不是Adam轨迹/参数更新，不推断真实训练中的后续critic拟合、任务表现、通信成本或长期收益。

## 原型测试与零操作

独立标量解析反例穿过同一hook seam：原critic trunk梯度450，隔离后None；critic head梯度180保持；actor logit对共享权重导数12保持。也检查critic值/前向相同、异常退出hook卸载、重复调用不累积hook、同seed BC角色门禁；ROUND49已记录R43旧ep0的模型角色/参数不匹配，本轮不加载该旧角色。

纯计算单测：见`PURE_TEST_EVIDENCE.json`（以最终测试回执为准）。环境构造0、仿真0、新场景/新流0、新评估0、真实训练更新0、合成optimizer.step0、checkpoint写回0。没有运行可能构造环境的套件。

## 唯一后续比较方案（当前未授权）

只比较critic表示梯度回流这个变量；从同seed真实BC_POST初值分别训练共享回流对照/stop-gradient候选，不从ep128续训。seed 11/22/33各128训练集上限；ROUND46其余 actor/双critic、数据顺序、GAE、loss、entropy、dual、预算/环境与全部科学配置冻结；不调参、不挑seed/checkpoint。greedy和sampled分列。每seed报告精确按时目标ID集合、绝对按时任务和计费字节、候选相对BC的附加变化以及相对当前共享PPO和late规则的差；不池化。不能只因比immediate省钱就称成功。

### 封闭资源草案——全部 `proposed_not_authorized`

| 启动类别 | 数量 | 依据/限制 |
|---|---:|---|
| 新候选训练 | 384 | 3 seed × 128；BC_POST重新初始化 |
| 新候选greedy评估 | 48 | 3 seed × 16既用场景 |
| 新候选sampled评估 | 48 | 同一场景，分开报告 |
| BC sampled对照缺口 | 48 | 当前无同口径记录；同BC_POST、同场景新增 |
| 正常合计 | **528** | 384 + 96 + 48 |
| 新技术失败储备 | **24** | 仅技术失败重试；不得补科学表现；独立新台账 |
| 新实验硬上限 | **552** | 528 + 24；不借历史reserve |

已有对照只有在逐行哈希、模型/规则身份、协议和评估配置匹配后才可复用：BC greedy 48行索引`ROUND46_R2_V6_INPUT_IDENTITY.json`（SHA256 `0859d53d16d7dd949efc8bd70fabc9ce8337e08e830f268022627583faa5d1b0`）；当前共享PPO 96格为V10分析输入manifest（SHA256 `634e398016f305a0b3170000160ec779c3faf7731e7426c74fa39690b596bbc1`）及对应FINAL_RESULTS（SHA256 `25b8f8dd4b62a184ffde63e545be7fc74819f257f9718bae4205816b402b9801`）；immediate及late规则按`ROUND44_BASELINE_REUSE.json`（SHA256 `6decb2e53693694b75bca045534c7be559eeeda4faed6181cec1665d99e816e5`）中场景1301000..1301015的逐行SHA。BC sampled缺失，故计入48次新对照评估，不以BC greedy冒充。ROUND46原独立预算为480正常+24技术储备=504硬上限，不可结转；ROUND52建议552必须是新授权/新台账，旧余量全部不可用。开发块已重复使用，不是独立确认集；不打开封存块。此资源草案新台账、无claim、无harness构造，本轮消耗0。

### 可直接转交的审批提示词草案

> 仅在明确批准后实施上述ROUND52单变量比较。建立全新隔离账本；唯一差别是task/cost critic是否向共享trunk回流梯度。对照保持现行回流，候选在critic head入口detach共享表示，actor/entropy路径保持原样。每个seed从对应ROUND46核验过的BC_POST四模块初值重新开始，11/22/33各最多128训练集；不得从ep128续训。ROUND46其余配置全部冻结，不调学习率、entropy、dual、训练量或阈值。正常上限528次（384训练、96候选greedy/sampled、48 BC sampled缺失对照），独立技术失败储备24次，硬cap552；禁止挪用历史reserve。只复用本报告列出的ROUND46 BC greedy、V10当前PPO、ROUND44 immediate/late索引且逐行复验hash、模型/配置/协议身份。仅在1301000..1301015开发块探索，绝不打开封存测试块。逐seed、greedy/sample分列精确目标ID集合、按时任务和绝对计费字节，比较候选相对BC、现行PPO与late；不池化/选seed。若旧对照无法身份闭合则停止，不补造；development_go仍false，结果不声称独立确认或部署。

`PROTOTYPE_CONTRACT.json`、`FORWARD_AND_GRADIENT_COMPARISON.json`和纯计算测试记录包含完整机器数值/身份。完成后等待决策端是否批准真实实验。
