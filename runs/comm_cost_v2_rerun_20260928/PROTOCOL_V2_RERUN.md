# ROUND24 协议：修复版 N4 从初值重跑（执行前冻结）

日期：2026-09-28。执行依据 `ROUND24_COMM_COST_V2_RERUN.md`（远端
`docs/comm-cost-v2-rerun-plan@86cff0226e5552c66c799cc7d210fa6006d058e0`，blob
`5d8ae2a556636c747043dca648a19d2efe3c82bc`、sha256 `112b04c7…f0ad`，git 端口本会话不通，
经已授权 GitHub MCP 按固定提交读取全文后逐字节保存；单文档提交 `5be3f1e`）。
分支 `exp/comm-cost-v2-rerun`，起点 `b7217d1`（本地修复 tip；其完整 tree 与规划端核验的
远端修复头 `64cddcffc6f8796f132df8b8e003f005d0bdd749` 逐条目一致——31/31 顶层 blob/tree
SHA 相同，内容寻址保证整树相等，仅提交元数据不同，证据 `START_EQUIVALENCE.json`）。
历史未回退；旧 Stage C 677 次启动与全部产物只读。

## 0. 本轮是什么

对 ROUND23 修复版训练目标（正成本、cost_head 进 optimizer、GAE next-value、奖励归最近
已采样动作）的**有界开发复验**：三个通信模型全部从登记初值 `Controller(训练种子)` 开始训练，
不用任何旧 v1 权重热启动（v1 checkpoint 恢复被 trainer 拒绝，合成测试
`v2_test_cost_head_lifecycle` 锁定）。不是新独立测试、不是调参、不是部署承诺、不是正式
非劣验收。旧轮比较只作开发观察。结论只允许 execution_complete / correctness_failure /
incomplete + 分开的科学观察。

## 1. 冻结配置（`CONFIG_AND_MODEL_MANIFEST.json`，启动前机读锁定）

- **世界**：逐字段继承 Stage B 冻结覆盖 + D1 A3 normal-lossy（6 目标、1000 步、
  policy_owned_timing、ACK 关、全订阅者、delivery 0.7、comm_range 0.3、gossip_2、
  **29300 B 上行硬预算**、无强制发送兜底）。学习臂与规则只差 `comm_policy` 一个键。
- **目标与对偶**：正归一化成本 `c = C_measured/B0`，B0=10000 B；δ=0（本轮实验训练目标）；
  λ 初值=1；η=0.05；每 4 集一批，批后 `λ_next = max(0, λ+η·ĝ)`，ĝ = mean(Y_ref−Y_π)−δ
  （配对 immediate reference 标量、无梯度）；批内 λ 固定。
- **网络与初值**：actor+task critic = pilot `Controller(训练种子)`；cost critic = Stage B 登记
  `Linear(64,1)`；optimizer 只含 trunk/action_head/task_value_head/cost_head 四模块（各一次）。
- **PPO**（与旧 Stage C 逐项相同）：clip=0.2、每 critic 头 VALUE_COEFFICIENT=0.5、
  ENTROPY_BONUS=0.01、Adam lr=3e-4、MAX_GRAD_NORM=0.5、EPOCHS_PER_BATCH=4、
  MINIBATCH_STEPS=256、EPISODES_PER_UPDATE=4；γ=1、GAE λ=0.95 按决策行递推；**不做任何
  advantage 标准化**；组合优势 `λ·A_task−A_cost`。
- **奖励口径**（修复版）：tick 的成本/任务收益归该 tick 发生后**最近一次已采样动作**；首决策
  前的前缀单独记账（`unassigned_prefix_*`，不进 actor/critic target）；末决策后的尾部归最后
  一个动作；总量账本闭合单独核对。
- **数据**：training 1300000..1300047（48）、development 1301000..1301015（16），
  **intentional_development_reuse**（ROUND22 注册、Stage C 使用过；不宣称全新未见；其他
  用途已审计；封存块不读不生成）；场景顺序 `random.Random(20260927)`，128 集=48+48+32。
- **训练动作 RNG**：每 episode `sha256("sc-action:{训练种子}:{episode序号}")`（沿用 Stage C
  登记派生），与环境/信道 RNG（来自场景种子）隔离。

## 2. 三个固定评估块（ROUND24 §5，启动前锁定）

| 块 | 权重 | 模式 | 场景 | 每 seed |
|---|---|---|---|---:|
| init_greedy | 登记初值 `Controller(种子)` | greedy | 16 开发场景 | 16 |
| final_greedy | ep128 不可变里程碑 | greedy | 16 开发场景 | 16 |
| final_sampled | 同一 ep128 里程碑 | sampled（与训练同一 masked 随机策略） | 16 开发场景 | 16 |

- 不跑 64 集开发评估；ep64 里程碑只保存供审计，不得替换或评估最终模型。
- 评估动作 RNG：`sha256("round24-eval:{训练种子}:{场景种子}:{mode}")`，稳定、跨进程可复现，
  不碰 Python 进程随机 hash、不污染环境/信道 RNG；每场景每模式一次。
- greedy 与 sampled 分表：sampled 主表观察训练策略，greedy 另表作部署规则诊断；不事后择优。
- 对偶约束估计来自训练采样策略，不自动视为对 greedy 的保证。

## 3. 启动账本：正常 672 + 储备 28 = 硬上限 700（`START_LEDGER.json`）

| 用途 | 启动数 |
|---|---:|
| 3 seed × 128 集从初值训练 | 384 |
| immediate reference 重跑（48 训练 + 16 开发） | 64 |
| 学习模型开发评估（3 seed × 3 块 × 16） | 144 |
| 五条规则重跑（5 × 16） | 80 |
| **正常合计** | **672** |
| 技术故障储备（不用于调参/扩样） | 28 |
| 硬上限 | 700 |

所有启动（失败、reference、规则、评估、smoke）计数；合成 optimizer 更新另计。旧 677 次不
退还、不动。执行顺序：64 references → seed11 首 4 集 + 首4集门禁（计入配额）→ seed11 余 124
集 → seed22 128 集 → seed33 128 集 → 三块评估 → 五规则。`v2_train.py` 每批保存批记录、
rolling checkpoint 与 ep0/64/128 里程碑（只写一次）。

## 4. 启动前已完成的门禁（0 环境启动）

- `START_EQUIVALENCE.json`：本地修复 tip 与远端修复头逐条目一致（verified=true）。
- `SYNTHETIC_GATE.json`：ROUND23 独立反例套件在本轮代码身份下重跑，20 项测试全过、
  0 失败 0 错误、合成 optimizer.step 28（Adam 12 + SGD 16，与真实训练步分列）；
  静态路径审计通过（旧目录只读引用已登记、无旧提交恢复入口、路径常量单点声明）。
- 冻结脚本 `v2_freeze.py` 还强制：生产三棵树零 diff、碰撞审计（旧块登记为复用）、
  与旧 Stage C 机读设置逐项 diff 仅剩已登记差异（四项修复、输出隔离、评估安排）、
  合成门禁身份与待冻结代码一致、台账为 0、无已持久化 v2 checkpoint。

## 5. 首4集门禁与后续（ROUND24 §7）

seed11 首 4 集计入 384 配额。G1-G7（`v2_check.first4_gate`）：四模块完整参数集合与
optimizer 一致、冻结 checkpoint 哈希不变、通信权重确已更新；初值权重下 log_prob 重算
≤1e-6；从落盘件独立重算奖励/GAE/组合优势/ĝ 与总账；对偶公式方向；critic 有限且全 mask 无
伪样本；纯计算进程恢复前向一致；样本计数闭合。通过后按固定计划完成三 seed；失败则停止真实
训练并报告。之后每批过 `v2_check.batch_audit`（参数集合、cost_head 权重摘要与预裁剪梯度、
λ/g、likelihood、冻结权重），防"首批绿、后续换路径"。

## 6. 停止条件

完整执行、正确性门禁失败、700 额度耗尽即停。故障恢复只用本轮合法 v2 状态
（`v2_recover.py`：只读本轮 rows/tapes/rolling checkpoint，拒绝旧 tape、旧提交、旧
manifest；更新是落盘件与 rollout 权重的确定函数，重算不重跑）。涉及目标/特征/时序语义的
新修复停止该批并报告，不拼合不同训练版本。
