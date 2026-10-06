# ROUND45 执行依据

本地执行依据：用户 2026-10-06 提供的“ROUND45：规则标签初始化＋约束式成本学习，一轮闭环完成”完整指令。研究变量只有固定离线行为克隆初始化后再进行原约束式 PPO；不重新扫描对偶步长。允许按冻结门禁自动继续，本轮本地提交；不推送、合并、部署或启动 ROUND46。

## 起点

- 仓库：`D:\code\整合\zhenghe-round42`
- 分支：`exp/comm-rule-warmstart-cost-learning`
- 父冻结 HEAD：`cbd437afad6ddc3a7a68c8b7e10c0cd572a778dd`
- 原结果目录：`runs/comm_dual_response_ablation_20261004/`
- 本轮独立目录：`runs/comm_rule_warmstart_cost_learning_20261006/`

## 科学与额度边界

- 规则 teacher 固定为实际已注册 `late_discovery_gate_147`；任何标签只能依照发送端合法动作前输入及实际规则路径生成。
- 唯一监督源为 R44 训练块场景 `1300000..1300047` 的三 seed、384 episode 已有 tape；fit 场景 `1300000..1300031`，监督验证场景 `1300032..1300047`。开发评估块 `1301000..1301015` 不得用于标签或 BC 调参。
- BC 每 seed 1000 次固定 full-batch Adam 更新；只取第1000次结果。BC-only greedy 控制最多48个正常 episode；若三 seed 均通过身份集合与成本门禁，才自动继续384集约束 PPO 与96格固定评估。
- 正常额度528，技术失败储备最多24，总硬上限552；BC 更新单列为3000次真实监督 optimizer update；合成 optimizer update最多64次；新流、额外 smoke 和新增参考均为0。
- 继承实际 R44 冻结的19维 actor、one-shot 语义、成本奖励、预算 mask、GAE/PPO/双 critic、RNG顺序及除 BC 以外的全部超参。不得把教师写入在线推理，不得恢复旧训练模型/Adam。
- 若 tape 缺少合法动作前字段、实际 teacher 与 one-shot 语义不兼容、表示无法表达教师标签，或任何恢复/正确性门禁不能证明，保留证据并停止受影响工作；不得改 teacher、观测、数学、超参或额度。

此摘要保存执行边界；完整授权文本仍以本轮用户消息为准。
