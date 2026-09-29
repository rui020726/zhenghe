# ROUND32 协议：actor surrogate 归约分母的配对训练消融（启动前冻结）

分支 `exp/comm-actor-reduction-ablation`（起点 ROUND31 交付头完整 SHA
`614df51709a8d82525405b46468a537f4ccd259e`，继承 ROUND30 `915868297c08d4b01b7dfeff0ea810d5394057a3`，
已用 `git merge-base --is-ancestor` 与 `git log --ancestry-path` 核验）。执行依据
`EXECUTION_BASIS.md`（sha256 `49d52f95…`）。冻结件 `CONFIG_AND_MODEL_MANIFEST.json`。

## 1. 唯一研究问题

只把 PPO actor surrogate 从"所有记录行平均"改为"有自由决策的记录行平均"，在其他一切条件相同的
情况下，是否改善有限训练内的任务—成本表现？

一个 minibatch 里参与 surrogate 归约的记录行数 `N`；其中 `actionable`/`free` 非空（即保存的真实
顺序 mask 路径下至少有一个带策略 likelihood 的自由候选）的行数 `M`：

```text
ALL_ROWS     actor 项 = -(1/N) · Σ_{i<N} s_i          （冻结表达式逐字保留）
ACTIVE_ROWS  actor 项 = -(1/max(M,1)) · Σ_{i<N} s_i · 1[i active]
```

这是**优化设计消融**，不预先认定为 bug 修复。两臂都用 ROUND30 的 old 编码器臂
（`comm-send-state/1`，`clip(remaining_time/50,-1,1)`），不混入 ROUND29/30 新期限编码。

任务 B 已在 0 启动下证明（`REDUCTION_MATH.md` / `REDUCTION_MATH.json`，11 项全过）：

- 归约前提与冻结实现一致；`ALL_ROWS` 与 `v2_trainer.ppo_update` **逐 optimizer 步权重哈希相同**；
- `grad(ACTIVE) = (N/M)·grad(ALL)` 对 actor surrogate **梯度**成立（实测残差 1.9e-08，float32）；
  对 loss **标量**不成立——无自由候选的行其 `s_i` 值非零但 actor 梯度恒为 0（在任何权重下）；
- entropy、两个 critic loss、`approx_kl`、`clip_fraction`、优势构造与 minibatch 顺序两臂逐位相同；
- 全 mask minibatch 时 actor 梯度为 0 且可安全反向，critic 样本全部保留；
- 归约模式写入 checkpoint，跨归约与跨轮恢复均被拒绝。

**作用范围**：本轮改变 actor 相对 entropy 与两个 critic 的更新权重——这是消融本身，不是副作用；
不宣称其它优化关系不受影响；不把 `N/M` 称作参数更新倍数。

## 2. 配对与训练契约

- 训练 seed 11/22/33；每 seed 两臂各一条从初值起的 128 集、每 4 集一批共 32 批；
  episode 配置 1000 步不变。
- 同 seed 两臂初始 actor、主干、两个 critic、optimizer 与 λ **逐字节相同**
  （`INIT_PAIRING_<reduction>_s<seed>.json` + `INIT_PAIRING.json` 汇总）。
- 训练数学全部引用 ROUND24/30 冻结件：`v2_trainer.batch_samples`（GAE、奖励归属、B0 归一化、
  不做 advantage 标准化）、`cost_math.gae/actor_advantage/dual_step/constraint_statistic`、
  `likelihood_alignment`（1e-6）、`dual_update`、批内 λ 冻结、Adam(lr=3e-4) 四模块参数集合。
- `B0=10000 B`、`δ=0`、`λ_initial=1`、`η_λ=0.05`；`A_actor = λ·A_task − A_cost`。
- 场景访问顺序：`v2_common.scenario_order()`（`random.Random(20260927)`），同 seed 两臂共用。
- 每 episode 动作 RNG：`v2_run.action_seed(seed, episode)`；评估 RNG：
  `v2_run.eval_action_seed(seed, scenario, mode)`。同 seed **不**意味着策略分叉后逐消息遭遇相同丢包。
- 两臂各由独立进程运行，进程内独立 RNG；账本用锁文件保护临界区。

## 3. 数据

训练 `1300000..1300047`（48）、开发 `1301000..1301015`（16），两块都已反复使用——本轮是开发性
消融，不称独立确认。不用 `1400000..1400063` 选参数或 checkpoint；不触碰封存块
`800000..800399`、`900000..900199`。搜索与分配 checkpoint 哈希保持不变（每批记录里
`frozen_checkpoints` 逐字节登记）。

## 4. 启动额度（本轮新授权，不继承历史余量）

| 用途 | 次数 |
|---|---:|
| immediate reference（48 训练 + 16 开发） | 64 |
| ALL_ROWS：3 seed × 128 集 | 384 |
| ACTIVE_ROWS：3 seed × 128 集 | 384 |
| 最终评估：2 臂 × 3 seed × 2 模式 × 16 场景 | 192 |
| `late_discovery_gate_147`：16 开发场景 | 16 |
| **正常合计** | **1040** |
| 技术失败储备 | 40 |
| **硬上限** | **1080** |

- 每个 episode 启动先记账再构造 driver；失败、重试、短探针一律计数且不返还；表现差不是技术失败。
- 首批门禁只用正式训练的集（seed11 两臂各首 4 集），不加 smoke。
- 纯合成 optimizer 更新单独分账，上限 64（任务 B 已用 24）。
- 并行度 ≤ 4 个仿真进程；分片按 (reduction, seed) 6 格预先固定、互不重叠，原子 claim 后启动。

## 5. 任务 D：本轮必须新增的证据

逐 minibatch 落盘：`N`、`M`、`M/N`、actor 归约值与"臂无关的全行均值"、entropy 项、两个 critic
损失、`approx_kl`、`clip_fraction`、`ratio` 极值、实际 λ、裁剪前/后总梯度范数与是否触发及缩放、
模块级裁剪前 L1 与 L2、非有限标记。逐批落盘：权重前后哈希、模块摘要、optimizer/controller 参数
形状集合、冻结 checkpoint 哈希、`g_hat` 与配对、对偶记录、归约模式、以上读数的批级汇总。

**批 0、15、31 在 `ppo_update` 之前写一次"更新前完整 checkpoint"**（模型、optimizer、λ 与历史、
RNG、数据位置、freeze 含编码器与归约模式），写一次不覆盖。ROUND31 只能分析批 0 正是因为
只有 ep0 milestone 存活——本轮从源头消除该限制。读数只做 `backward()` 之后、`clip_grad_norm_()`
之前对 `.grad` 的读取，不触碰 RNG、动作与参数更新（由 `REDUCTION_MATH.json` T9 的权重哈希等价证明）。

## 6. 执行顺序

1. 64 个 immediate reference，并与 ROUND24、ROUND30 两套归档行的行为字段
   （`on_time/spent_bytes/emissions/expired/completed/discovered/ack_ledger/steps`）比对；分叉即停。
2. seed11 两臂各跑首 4 集（计入各自 128 集），独立重算门禁 G0–G9（含新增的归约生效与
   更新前 checkpoint 契约）+ 两臂交叉算术比较 G-cross。两臂全过才继续。
3. 六条 128 集训练跑完；不依首批或中间效果改计划，表现差不重跑。
4. 固定各格 ep128 里程碑完成 192 次评估（greedy/sampled 分开，每格一次）与 16 次 late 规则。
5. 统计、机制诊断、独立复核、交付、本地提交后停止。

冻结后若发现影响采样或更新的行为错误：停止受影响实验并报告，不拼接修复前后的训练链；
纯报告字段问题可独立更正并保留原件、不重跑环境。

## 7. 统计（预先冻结，沿用 ROUND30 方案）

每 (seed, mode) 单独：`ΔY = ACTIVE_ROWS − ALL_ROWS`（按时任务数/ep）、`ΔC = ACTIVE_ROWS −
ALL_ROWS`（已计量字节/ep）、逐场景改善/持平/退化、按时目标集合是否交换；**同时**列每臂绝对
任务数与字节、相对 immediate 的配对任务损失、相对 late 的任务差与成本差——不得只报"两臂相等"
而隐藏双方都比 immediate 差。

描述性区间：场景升序、16 场景为配对抽样单位、`PCG64 seed=20260930`、10000 次、linear 分位、
所有比较复用同一份 10000×16 索引、估计量 `values[indices].mean(axis=1)`、独立手算反例。
区间不覆盖训练 seed 不确定性；sampled 每场景一次；三 seed 分别展示、不池化、不挑最佳模式。
开发块已反复使用：无确认性显著性、无 δ=0 保证、无独立泛化主张。

## 8. 停止条件

reference 与任一归档行为字段分叉；任一门禁失败；冻结清单与工作树不一致；出现未解释的行为身份
冲突；账本超过硬上限 1080；发现未解决的成本符号/GAE/奖励时序/恢复缺陷（停止且不顺带改训练数学）；
其余按执行依据 §六/§十。表现不好不是停止理由。

## 9. 发布

`PENDING_NETWORK`；本轮 0 次网络尝试、不推送、不部署，发布状态单独登记，不阻断本地分析。
