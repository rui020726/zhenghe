# ROUND69 执行依据（授权提示词原文存档）

本文件是本轮 `ROUND69-LATE-ACTOR-SIGNAL-READONLY-AUDIT` 的授权原文，按约定直接存档，不另写 spec。
以下为委托输入原文：

---

下一项：ROUND69-LATE-ACTOR-SIGNAL-READONLY-AUDIT。按既有持续科研授权执行一次有限只读分析，不新训、不调参。先核验当前完整HEAD420fa9af18932f735efdf6528ab68174c561ec52及R68收口；若正在执行同任务不重复。保留当前分支，或在核验头建独立analysis分支（不得覆盖），原所有消费/违规/报告不改。

问题：R68三seed的late参数已改变而greedy任务/成本未变；现有训练数据能否区分"自由late动作几乎没有可用更新信号"和"有信号但其方向/幅度未形成稳定保任务降本"？不能预设critic错误、饱和、欠训练或信息不足。

输入仅R68既有384训练row/tape、96batch/commit、已核验配置/source与初值/ep128身份索引、已有R68评估统计。复用原身份/账本/门禁，不重跑。沿真实batch_samples/GAE/actor surrogate源码确认保存字段的时点和单位；从保存的rollout前critic值、奖励流、done、λ_before独立重算GAE与组合优势，先用极小手算例核对；不得用更新后值或推测minibatch代替。γ=1、GAEλ=1是否使当前终止完整流return成为MC总和需按真实边界验证，而非预设。

一次完成三项：①96batch的总行/全mask/early-free/late-free计数，实际ALL_ROWS分母与late有效样本占比，按seed和batch分列；②在合法late-free真实决策上分解组合优势=λ·任务return−正成本return−(λ·任务value−成本value)，报告分项尺度、符号、SEND/SKIP分布和critic残差。多候选联合决策只按真实行权重统计，不能当独立任务样本；masked/forced/冻结early角色分开。③结合已有INIT→TRAINED的greedy零变化、sampled局部变化，给出哪些解释得到/未得到支持；若缺少值/精确minibatch或随机排列则登记不可判，不补forward、重放PPO或伪造信息。on-policy chosen-action优势不等于SEND/SKIP因果价值，不能据此删消息、换阈值或认定某loss根因。

硬边界：新增环境/episode/claim/optimizer（真或合成）/PPO重放/模型forward/autograd/RNG恢复/checkpoint写入全部0；允许读文件、纯算术回归与内存统计。只新增小型分析脚本、机器JSON、中文报告和执行依据（直接存本提示词，不另写spec）；本地显式提交，原始tape/模型/log不入Git，保护文件哈希不变，无推送合并部署。最终只回完整HEAD、覆盖/缺失、3个关键结论和零副作用计数，然后停止。不展开另一项实验、不刷全部测试、无变化不反复回报。

---

## 执行对照

- HEAD 核验：`420fa9af18932f735efdf6528ab68174c561ec52`（R68 收口提交）；R68 收口件在位；无同任务在跑。
- 分支：`analysis/round69-late-actor-signal-audit`（自核验头新建，未覆盖任何分支）。
- 三项交付：见 `ROUND69_REPORT.md` 第 3/4/5 节；机器证据 `ROUND69_LATE_ACTOR_SIGNAL_AUDIT.json`。
- 零副作用：见报告第 7 节；保护文件哈希未变（提交前核验）。