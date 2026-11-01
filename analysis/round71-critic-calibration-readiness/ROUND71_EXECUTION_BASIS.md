# ROUND71 执行依据（授权提示词原文存档）

本文件是本轮 `ROUND71-CRITIC-CALIBRATION-ZERO-USE-READINESS` 的委托输入原文存档，按约定直接存档，
不另写 spec。本轮为**零消费准备**，不含任何实验授权。

---

任务：ROUND71-CRITIC-CALIBRATION-ZERO-USE-READINESS。用户对新增实验尚未批准；本次仅授权在既有科研推进
范围内完成私有代码准备与零消费验收，绝不启动科研运行。以已核验 `92571c62b27b0c38d755b501a6d0153b630a08ed`
为祖先，核对 HEAD/AGENTS/现有任务。已完成仅验收，运行中不另开。新建/复用不覆盖的
`analysis/round71-critic-calibration-readiness` 及独立 analysis 目录。保留所有旧失败、R70 资源违规和两个
保护文件，不推送/合并/部署。

研究候选（不是本次执行许可）：唯一干预是 R68 各 seed 模型初值的双 critic 仿射头校准初始化。原 trunk、
early/late actor 头（late 初值 x0.10）及其初始策略不变；优化器/lambda/RNG 从 R68 合格 ep0 合法新训练状态
开始，不用已训练 ep128 初始化/恢复。用同 seed R68 已有 128 集训练 tape 及冻结 trunk 的真实
mean-over-targets 表示、MC 正成本/任务回报，固定 ridge=1e-3、全训练 teacher 数据内中心/尺度，常数列
scale 1、截距不惩罚，双输出拟合一次/seed。不得使用 180 评估数据、拟合阈值/选择 seed/把 late 标签当安全
真值。此方法是旧训练数据提供的 critic 初始化，有额外离线学习成本及政策分布变化风险；不称同训练预算竞赛、
不预称 baseline 能提高策略。

候选后续计划（明确 APPROVAL_PENDING）：384 从初值训练 + 192 固定 ep128 评估（3seed x 2 模式 x 32 旧开发
场景 1800000..1800031）+ 6 格 1700027 冻结焦点 greedy（INIT/TRAINED x 3seed）= 582 正常，18 技术储备，
总 600；原 R68 同 seed/mode/场景对照与规则只读复用，需来源/配置/行为策略身份资格，不借旧余量。离线校准
及其验收的实际 19-D trunk 输入向量上限 72000、每批 1024；这只约束离线阶段，未来有许可的训练/评估消费另外
完整计数。候选纯合成 optimizer 上限 16，但本次 0。科研配置逐字段继承 R68 真实活体配置
(gamma=1, GAE lambda=1, eta_lambda=0.5, beta=1, ready147, ALL_ROWS, 正成本 lambda*A_task - A_cost,
critic->trunk 隔离、顺序 mask、同 tick 奖励)，不使用导入默认值。候选计划只作登记，不建立真实运行授权回执。

本次一次完成：

1. 私有校准初始化接缝：沿已验证真实 critic 输入/仿射头路径设计，在未来拟合得到系数后将标准化表示上的 W,b
   精确转换到原始 64 维 embedding 仿射头；只允许替换双 critic 头，不改 actor/trunk/数学。使用依赖注入/
   内存合成系数测试；本次不打开真实模型、不对真实数据拟合、不生成模型权重。
2. 资源/入口守卫：实际 19 维矩阵行先持久原子 debit 再 forward，包含 target 展开、失败、重复；计费单位不能
   改成 joint 行/调用数。分批以真实输入行 <= 1024；现存产物拒绝隐式重复运行。真实许可缺失时必须在任何数据
   拟合、模型前向、claim/checkpoint 写入、episode 调用前拒绝。不要新增"回执必须仓库外"或恰好若干 claim
   文件等条件。
3. 只读计数核对候选离线输入量并留源 hash/联合行与 target 输入行分列；不重新提取 embedding。复用 R70 均值
   聚合/仿射接线和 R68 初值/冻结模块证据，变更身份留新版本。teacher 数据输入在本次可读取/计数/hash，不加载
   .pt 或恢复 RNG。
4. 窄测试只用纯算术/静态/spy：系数逆变换独立手算反例；校准头改变 value 但不触及 actor/trunk 对象；19-D
   展开计费/真实 batch 1024 边界及超额前拒绝；假调用失败仍消费；旧 checkpoint 非法恢复拒绝；真实许可缺失
   fail closed。不要执行任何真实或合成神经 forward、optimizer、autograd、拟合或 save checkpoint。spy/no-op
   不是实际更新，也不证明物理断电原子性。
5. 未来 launcher 尽量复用已合格 R68 真实 train/claim/mask/gate 接缝，避免整份复制冻结训练器。本次实现可
   注入入口并验证许可缺失拒绝；尚需授权后的真实初始化等价/梯度/恢复门禁如实列尚未执行，不能冒充验收已覆盖。
   若需 canonical 修改则停止说明，不顺便改生产/旧训练器。

本次硬边界：环境/episode/claim、真实或合成 optimizer、PPO 重放、神经 forward（含 NumPy 等价）/autograd、
真实数据统计拟合、模型加载、训练 RNG 恢复、checkpoint 写入全部 0；允许现有 JSON/tape 只读、纯算术矩阵
系数转换、内存 spies、测试用私有临时计数器。保护文件 `cf3ad198...` / `93a2f69f...` 及无关 r59 日志不改不
暂存。不要另写 spec 或宣称新额度已批准。

交付一次：必要私有代码、窄测试、APPROVAL_PENDING 候选/输入计数/当前准备覆盖与未执行真实门禁、prompt 依据，
显式本地提交；不给出科学效果数值。报告 <=250 字阶段状态，最终完整 HEAD + 实际零消费 + 测试数/exit + 路径。
未收到人类新增实验批准，测试全绿也必须停在零消费准备阶段。