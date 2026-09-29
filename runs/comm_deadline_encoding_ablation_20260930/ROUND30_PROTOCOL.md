# ROUND30 协议：原期限编码 vs 新期限编码的从初值配对训练消融

分支 `exp/comm-deadline-encoding-ablation`（起点 ROUND29 完整头
`99eb746baaa073550f9062fec5c87a4b4711827b`，继承 ROUND28 `21250b5`）。执行依据
`EXECUTION_BASIS.md`（sha256 `d80a7055…`）。本文件是启动前冻结协议；冻结件为
`CONFIG_AND_MODEL_MANIFEST.json`。

## 1. 唯一研究问题

将期限列从截断编码改为保留差异的编码，在其他训练条件相同的情况下，是否改善学习表现？

- OLD：`clip(remaining_time / 50, -1, 1)`（生产编码 `comm-send-state/1` 原样）。
- NEW：`remaining_time / configured_episode_horizon(1000)`，不截断
  （`comm-send-state/1r29-deadline-over-horizon`，ROUND29 登记原型）。

唯一主动改变的是第 12 列（0-based 11）的定义；其余 18 列、顺序、`urgency_now`、19 维形状、
网络、训练目标、PPO 参数、预算、ACK（结构性关闭）、mask、候选顺序、评估解码均不变。

## 2. 数据与模型

- 训练场景 `1300000..1300047`（48，已使用数据）；开发评估 `1301000..1301015`（16，已使用）。
  本轮是开发性消融，不称未见测试或独立确认；不使用 ROUND25 评估块 `1400000..1400063`
  做参数/checkpoint/停止决策；不触碰封存块。
- 搜索与分配：ROUND24 登记的冻结 checkpoint（哈希见冻结件 `checkpoints` 段）。
- 训练 seed：11 / 22 / 33；每 seed 两条臂（old/new）配对。

## 3. 训练契约（全部引用 ROUND24 冻结件）

- 每臂每 seed：128 集 × 1000 步；每 4 集一批共 32 批；`ppo_update`、`batch_g_hat`（配对
  immediate reference 标量，无梯度）、`dual_update` 复用 `v2_trainer`。
- `B0=10000 B`、`δ=0`、`λ_initial=1`、`η_λ=0.05`、正成本输入、`A_actor = λ*A_task − A_cost`；
  其余 PPO/GAE/奖励归属逐项抄录在冻结件（不凭摘要补值）。
- 场景访问顺序：ROUND24 冻结算法 `random.Random(20260927)`，128 集清单展开在冻结件
  (`data_order.order_sha256`)；同 seed 两臂共用。
- 每 episode 动作 RNG：`sha256("sc-action:<seed>:<episode>")`；评估 RNG：
  `sha256("round24-eval:<seed>:<scenario>:<mode>")`（沿用已核验派生）。
- **同 seed 不代表干预后每条消息遭遇相同丢包**（信道 RNG 来自场景 seed，发送序列分叉后
  消耗不同）；不得作此声明。

## 4. 接线与身份

- 作用域 collector：父类 `_choose_masked` 逐字复制 + 单行登记差异（difflib 程序化登记在
  冻结件 `scoped_collector_source_diff`）；OLD 臂在预检中与父类 tape 逐字段一致。
- 初始化配对：每 seed 两臂参数/逐模块摘要/optimizer 状态/λ 逐字节相等
  （`INIT_PAIRING.json` + `INIT_PAIRING_<arm>_s<seed>.json`）。
- checkpoint 保存/恢复比较整个 freeze（含 arm 与 feature_layout）⇒ 跨臂、跨轮恢复显式拒绝；
  生产/导入模块身份与 ROUND24 逐字节相同是冻结的停止条件。
- 训练、采样、PPO 重算、评估使用同一臂编码；tape 存 float64 编码值，网络输入与 PPO 样本
  用其 float32（与父类同行为，预检按该等价关系断言）。

## 5. 启动额度与账本

| 块 | 次数 |
|---|---:|
| immediate reference（48 训练 + 16 开发场景） | 64 |
| OLD：3 seed × 128 集 | 384 |
| NEW：3 seed × 128 集 | 384 |
| 最终模型评估：2 编码 × 3 seed × 2 模式 × 16 场景 | 192 |
| late_discovery_gate_147：16 开发场景 | 16 |
| **正常合计** | **1040** |
| 技术失败储备 | 40 |
| **硬上限** | **1080** |

- 每个 episode 启动先记账（锁文件保护临界区）再构造 driver；失败、重跑、重复启动均计数，
  失败不返还；性能差不计为技术失败；储备不得用于增加场景/延长训练/增加采样/换候选。
- 纯合成 optimizer 更新上限 64 次（与真实训练更新分账，不构造环境）；预检实际消耗 44。
- 6 条训练按 (arm, seed) 分进程并行；各进程独立 RNG，账本全局一致。

## 6. 执行顺序

1. 跑 64 个 immediate reference；与 ROUND24 归档 reference 比对行为字段（on_time/spent/
   emissions/expired/completed/discovered/ack/steps），发现分叉即停止报告。
2. seed11 的 OLD 与 NEW 各先跑 4 集（计入各自 128 集），分别独立重算首批门禁
   （`r30_check.py gate <arm> 11`：G0 编码 / G1 账本闭合 / G2 数学 / G3 log_prob 1e-6 /
   G4 optimizer 参数覆盖 / G5 对偶公式方向 / G6 有限 / G7 恢复契约）。两臂均通过才继续。
3. 完成六条 128 集训练（不依据首批或中间效果改计划；首批"没有学好"不是正确性失败）。
4. 固定使用各自 ep128 里程碑 checkpoint 完成 192 次评估（greedy/sampled 分开，每格一次）
   与 16 次 late 规则评估。
5. 汇总、统计、交付、本地提交后停止。

正式运行后不得静默修改冻结的行为代码；若错误影响采样/编码/优化/环境行为，停止并报告；
纯报告字段错误可独立修正并披露。

## 7. 统计（预先冻结）

- 每 (seed, mode) 单独：ΔY = NEW − OLD（按时任务数/episode）、ΔC = NEW − OLD
  （已计量载荷尝试字节/episode）；逐场景改善/持平/退化；按时目标集合交换（lifecycle final
  记录取目标级结局）；相对 immediate 与 late 的任务/成本差。
- 描述性区间：场景升序、16 场景为配对抽样单位、PCG64 seed=20260930、10000 次重采样、
  linear 分位、所有比较复用同一份 10000×16 索引、`values[indices].mean(axis=1)`，独立手算
  反例在 `PAIRED_COMPARISON.json` 的 `hand_check`。区间不覆盖训练 seed 不确定性；
  sampled 每场景仅一次不估计其完整采样波动；三 seed 不池化、不挑最佳。
- 开发块已反复使用：无确认性显著性、无 δ=0 保证、无独立泛化主张。

## 8. 停止条件

- reference 与归档行为字段分叉；首批门禁任一失败；冻结清单与工作树不一致；账本超过硬上限；
  发现未解决的采样/奖励/训练时间契约缺陷（停止并不顺带改训练数学）；其余按执行依据 §7。

## 9. 发布

`PENDING_NETWORK`；本轮不主动重试网络，不以发布成功作为实验前提。
