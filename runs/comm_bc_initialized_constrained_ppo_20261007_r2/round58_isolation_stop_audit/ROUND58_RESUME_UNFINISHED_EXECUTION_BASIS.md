# ROUND58-RESUME-UNFINISHED-EXECUTION — 当前执行依据

- 本任务标识：`ROUND58-RESUME-UNFINISHED-EXECUTION`；来源：项目负责人本轮明确派发。批准原文：**“核验通过后批准完成剩余固定计划”**。该批准覆盖精确恢复核验通过后的固定剩余计划，不因旧只读报告的历史停止范围而撤销。
- 当前接手点：分支 `exp/round58-prior-strength-prospective-trial`，起始 HEAD `42faf9fcdb6ccf46b7c136f8ce9e6d21addb5a6c`。本文件与 JSON 状态只记录当前任务，不代替冻结配置或授权凭据。
- 当前未完成：迁移窄测试未纳入完整runner；历史旧main回归路径误指对象；计划身份迁移证明缺失；新版冻结/授权回执缺失；完整 `__main__` 恢复前置验收未执行；剩余固定计划尚未执行。
- 科学配置不变：β=1.00、R53 critic隔离、既有正成本/约束优势、19维观测、一次性SEND/SKIP、原mask/reward/GAE/PPO/dual/训练顺序和模型身份均冻结。仅允许登记的私有恢复执行元数据映射。
- 禁止重跑首12集/原3个batch、旧tape PPO重算、环境构造用于前置验收、调参、新场景、超额消费、push/merge/deploy。前置阶段硬计数均须为0：环境/episode/claim、optimizer.step/PPO重算、RNG恢复、磁盘权重/checkpoint写入、评估格。
- 前置路径须穿过实际 `__main__`、真实receipt校验、legacy历史计划精确断言、configure/bind、三seed来源选择、账本并在episode4前硬停止。当前原件和失败证据只读保留。
- 前置通过并提交冻结身份、proof及新版receipt后，按授权续行372集（seed11/22/33各从episode4续至127）及640评估格（576模型+64规则），遵守累计环境上限1064/技术reserve 24。任何实际运行核心身份/数学/提交/计数错误则按冻结停止并保留现场。
