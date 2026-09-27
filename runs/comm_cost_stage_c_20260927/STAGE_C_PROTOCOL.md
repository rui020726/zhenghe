# Stage C 协议：首轮有界成本学习 pilot（不部署、非正式非劣验收）

日期：2026-09-27。执行依据 `ROUND22_COMM_COST_LEARNING_PILOT.md`（远端
`docs/comm-cost-learning-pilot@f73ebdf8cc3ab60a6a866fdbe0e299b122245cf7`，字节级一致，
cherry-pick 为 `39a3da5`）。分支 `feat/comm-cost-stage-c`，起点
`6ebefbe4e7a81182281807333d64d60cadf03198`（已核验的 Stage B 头；Stage B 产物已逐项落盘核验：
18/18 rows、0 optimizer.step、门禁 15/15、冻结 checkpoint 哈希一致）。历史未回退。

## 0. 本轮是什么

冻结 PILOC 搜索与 CTDE 分配，学习任务信息发送时机以降低**已计量通信成本**的首轮新目标训练。
**不是**部署承诺、**不是**正式非劣验收、**不是**重开搜索研究、**不是**旧"只奖完成数"训练。
正式结论只允许 pilot_completed / technical_failure / incomplete。

## 1. 冻结配置（`CONFIG_AND_MODEL_MANIFEST.json`，启动前机读锁定）

- **世界**：逐字段继承 D1 A3 normal-lossy（6 目标、1000 步、deadline_tightness (0.5,2.0)、
  两个冻结 checkpoint、success 0.7、delay 0、ACK 关、全订阅者、comm_range 0.3、
  gossip_interval 2、policy_owned_timing、无强制发送兜底、**29300 B 上行硬预算**）。
  学习臂与规则只差 `comm_policy` 一个键，其余逐字段相同。
- **目标与对偶**：`c = C_measured/B0`，B0=10 000 B；δ=0（本轮**实验训练目标**，不代表部署
  零退化保证）；λ 初值=1；η=0.05；每 4 集一批，批后
  `λ_next = max(0, λ + η·ĝ)`，ĝ = mean(Y_ref − Y_π) − δ（配对 immediate reference 标量，
  无梯度）；同一批 PPO 期间 λ 固定；验证集不更新 λ。
- **网络与初值**：actor + task critic = pilot 轮 `Controller(训练种子)`（19→64→64 Tanh +
  动作头 2 路 + 价值头）；cost critic = Stage B 登记的新增 `Linear(64,1)` 头。三个训练种子
  11/22/33 只从初值训练，不加载、不迁移任何旧权重。只训通信模块：optimizer 只含
  controller.parameters；搜索/分配权重来自两个冻结文件，逐批核对 SHA256。
- **PPO**（全部引用 pilot 轮既有值并逐项列明，不写"沿用默认值"）：clip=0.2、
  VALUE_COEFFICIENT=0.5 **每 critic 头**、ENTROPY_BONUS=0.01（额外正则项，其对纯成本目标的
  影响在此登记、不隐去）、Adam lr=3e-4、MAX_GRAD_NORM=0.5、EPOCHS_PER_BATCH=4、
  MINIBATCH_STEPS=256、EPISODES_PER_UPDATE=4。minibatch shuffle 用每批独立
  `random.Random(派生种子)`，不碰全局 RNG。
- **与旧 trainer 的登记差异**（spec §2 要求）：**不做任何 advantage 标准化**（旧 flatten 逐
  episode 除 std，本版禁止；分量也不分别标准化）；γ=1（旧 0.99）、GAE λ=0.95（与对偶 λ 分名）、
  末决策步 terminal=True；双 critic；组合优势 `λ·A_task − A_cost`；bootstrapped return 不要求
  等于 episode 总量，账本闭合用原始 reward 和与实际计费增量。
- **奖励口径**（接线约定）：`r_cost_t = −(捆绑区间内 relay 空气账本增量)/B0`、
  `r_task_t = 窗口内按时完成数按完成步计入`；相邻决策步的奖励捆绑到下一决策步，Σ 闭合计入
  episode 总量；tap 与 `window_metrics.on_time` 断言相等。
- **动作 RNG**：每训练 episode 由 `(训练种子, episode 序号)` 派生 torch 种子，episode 间独立、
  可复现，不触碰环境/信道 RNG（后者来自场景种子）。

## 2. 数据划分与场景块（`SCENE_REGISTRY.json`）

- training 场景块 **1300000..1300047**（48）；development 场景块 **1301000..1301015**（16）。
  均为本轮新注册开发用途块；development **不是**未来独立确认集；封存块保持封存。
- **碰撞审计已通过**（0 启动）：与 sealed 800000-800399 / 900000-900199、systemloop
  31000-31059、task-pressure 1000000-1000315、D1 1100000-1100059、round-18 1200000-1200059
  及全部既有 artefact 无交集。第十九轮设计件里的条件提案号（1300000..1300001 /
  1300100..1300219）从未被任何运行使用，由本轮注册取代——已登记，不静默忽略。
- 数据顺序：`random.Random(20260927)` 逐轮重排 48 场景，依次取 48+48+32 填满 128 集；
  三个训练 seed 共用同一顺序。
- immediate reference：每场景只跑一次（48+16=64），按（模型、场景初态、配置、代码身份）键缓存；
  可复现性是缓存前提（Stage B 已证 R_send≡I_send、P_init≡P_repeat 逐位一致）。

## 3. 启动预算：正常 672 + 储备 28 = 硬上限 700（`START_LEDGER.json`）

| 用途 | 启动数 |
|---|---:|
| 3 通信训练 seed × 128 集 | 384 |
| immediate reference（48 训练 + 16 开发场景） | 64 |
| 学习模型开发评估（3 seed × 3 时点 × 16 场景） | 144 |
| 五条规则开发评估（5 × 16 场景） | 80 |
| **正常合计** | **672** |
| 技术错误/有限重测储备（不用于扩样或调参） | 28 |

每 seed ≤128 000 环境步；每 4 集一批、每 seed ≤32 批；optimizer.step 数由 minibatch/epoch 决定，
不等于批数。所有环境启动（失败、评估、基线、复跑）均记账；纯合成测试不计。评估时点：
初值、64 集、128 集（冻结权重与 λ，贪心前向，任务/字节全由环境真实输出）。
执行顺序：64 references → seed 11 首 4 集 + 正确性门禁 → seed 11 余量 → seed 22 → seed 33 →
144 学习评估 → 80 规则评估。

## 4. 六条规则对照（spec §6）

固定参考 `immediate` + 五条规则：`always_hold`、`late_discovery_gate`（cut=147）、
`periodic(interval=20)`、`random_send(p=0.5, per-scenario seed)`、
`event(progress_threshold=0.2, urgency_time=25.0)`。`send_once` 与 immediate 的 decide
函数体逐字相同（同 `pending_event` 谓词、同非 orders_selection）⇒ 源码核验语义一致，
**按别名登记、不另耗运行**。

`late_discovery_gate` 是本轮提出的强对照，**不是学习成果**：只依据发送端可见的首个未发送
变更簿记步 `ready_step`，**严格大于 147** 才首报一次；早发现不发、不把早发现延迟到阈值、
不读终局或接收端知识；拒付后下步重试。`ready_step` 与首报步号的一致性已用 Stage B 归档件
核验（18/18 首报 `step == ready_step`），与第十九轮开发分析 `step_reported_at`（cut=147、
report_when_above）同口径；dt=1.0 ⇒ tick 与物理时间同刻度。合成边界测试（146/147/148）锁定。

## 5. 首 4 集训练正确性门禁（计入 384，无额外配额）

seed 11 首 4 集后逐条检查（`sc_check.py first4`，只读落盘件）：
G1 optimizer 只更新通信模块（参数恒等 + 冻结文件哈希不变 + 控制器哈希确实变化）；
G2 rollout log_prob 在任何更新前用初值权重可重算（≤1e-6）；
G3 成本/任务 reward、GAE、组合优势、对偶 g 从落盘数据独立重算且闭合（Σr 与账本相等）；
G4 λ 更新方向与公式一致、reference 为标量无梯度；G5 双 critic target/预测/loss/梯度有限、
全 mask 无 actor 伪样本；G6 checkpoint 在**纯计算进程**恢复后同一观测前向一致（0 额外启动）；
G7 丢包/拒付/主动HOLD/可执行样本计数闭合。
**门禁判据不含是否省字节或任务是否改善**。任一数学/信息泄漏/计费/训练状态失败 ⇒ 暂停学习、
隔离证据、最小修复并登记新代码身份；储备不足或修复改变协议 ⇒ 停止，不池化不同训练版本。

## 6. 评估与解释纪律（spec §8）

主描述比较：每个训练 seed 的**固定 128 集 checkpoint**（不择优）对 immediate 的按时任务数差
与实际字节差；同时完整报告五条规则（尤以 late_discovery_gate）。**优于 immediate ≠ 优于强规则**。
逐 scene 配对、episode 为聚类单位报均值/区间/改善-打平-退化数，三 seed 分列。不把消息或目标
当独立样本、不把 3×16 当 48 个独立场景、不选最佳 seed。16 个开发场景只支持探索性 pilot：
零观测损失、CI 含 0 或朴素 bootstrap 退化到 [0,0] 都**不能**证明总体非劣/等效或 δ=0 成立。
本轮无已批准 ρ：报告实际降本比例/绝对 B 及不确定性。128k 未收敛不能宣称排除欠训练，
不得因此自动续训。诊断全留：原始 reward/费用、有效 actor 样本、entropy、双 value loss、
PPO KL/clipfraction、梯度范数、λ/g、预算拒付、参数哈希。λ 持续增长或在 0 边界只是诊断。

## 7. 未执行 / 未授权

未训练权重之外不改任何生产路径（生产三棵树相对分支起点零 diff）；不开封存集；不跑会启动
episode 的全套 smoke；不改 horizon/deadline/场景难度；不新增 ACK/拓扑/actor 特征；不关闭
本地感知或 gossip；不按中间效果调参、扩样或续训；C 之后的任何追加训练/确认实验/部署另行批准。
原始日志/模型权重/`checkpoints/` 不入 Git（manifest 记录位置与摘要）。
