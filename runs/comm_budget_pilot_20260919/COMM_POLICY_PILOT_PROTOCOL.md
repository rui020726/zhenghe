# 学习型发送控制 pilot 协议（COMM_POLICY_PILOT_PROTOCOL）

日期：2026-09-19　分支：`feat/learned-communication-budget-pilot`（基线 `0794de9d…`）
运行清单：`COMM_ROUND_MANIFEST.json`　账本定义：`COMM_BUDGET_ACCOUNTING.md`

方向：**通信硬预算下的学习型任务消息发送控制**。目标不是"完成率超过无预算规则基线"，
而是在**同一固定预算内**取得更高按时完成率（或在给定完成率下用更少通信）。

## 一、被控对象与不控项

| 项 | 内容 |
| --- | --- |
| 通道 | 链路 A：`CommRelay` 搜索端→救援 belief。同一 `DetectionMessage` 契约既承载首次检测也承载状态刷新，故本轮控制"检测与更新何时上空气"，**不**新增第二套消息结构 |
| 决策 | 每个联合步对每个"有未说出口变化"的目标选 `send` / `hold`；顺序与每步候选集合由传输层固定，规则臂与学习臂共用 |
| 载荷 | 发射时刻登记处最新快照（所有臂一致）；不学"少发字段"，因为那是削信息不是削通信 |
| 不控 | 接收端合并规则、订阅集合、机器人位置/belief、分配器、协调器、PILOC、CTDE 权重、机队、任务生成、deadline |

## 二、运行配置核对（`pilot_preflight.py`，`PILOT_PREFLIGHT.json`）

- 训练环境**确实启用**成员一致性：`env.membership_consistency = True`，臂标签含 `+memfix`；
  默认关闭的兼容测试不能替代这一项，因此逐 episode 断言而非引用默认值。
- 分配侧 = 冻结 C 臂：`coalition_planner` + 绑定承诺，`release_redundant_members=False`、
  `suppress_recommand_own_task=False`、`check_release_source_stage=False`、
  `residual_scale=0.25`、`coordination/allocation_interval=5`、`command_ttl=10`、
  6 机器人（3 物种 × 2）、`deadline_tightness=(0.5,2.0)`、60×60 栅格、6 目标、检测半径 5；
  CTDE/PILOC 权重 sha256 与冻结清单一致（导入即校验，漂移直接拒绝启动）。
- `mismatches: []`（零不符项）；`config_hash`、`stream_sha256` 记入清单。

## 三、信息边界（训练可见 vs 评估专用）

策略可见：`SendItem`/`SendObservation` 的字段白名单（`train_pilot.SENDER_SIDE_FIELDS`，
smoke 会检查类型未越界）——变化年龄、距上次发射时间、已发次数、各类"变了没有"标志、
剩余时间、队列积压、**剩余预算比例**、本步容量、订阅者数、episode 进度。

策略不可见：任何机器人位置/状态/身份、谁已经知道（无 ACK）、丢包结果、执行侧闸门读到的真值、
未来事件。载荷内容按"发射时刻可观察"打时间戳（`sent_at=registry.updated_at`，转录记 `now`），
因此不存在把未来状态提前用于重要性判断的路径。

评估专用（离线，绝不进特征）：信息年龄、需求/阶段误差步数、"活动却无人知晓"任务步——
它们从转录 + 生命周期日志的登记处时间线重建。

两条信息设置并行保留、分别报告：
1. **隔离设置**（`gossip_interval=100000`）：关掉环境自带视距刷新与一跳 gossip，用于机制验证与训练；
2. **原始设置**（默认 `comm_range=0.3`、`gossip_interval=2`）：用于检查真实可部署价值。
本轮不修改现场感知来制造学习优势；隔离设置里的结论不写成原始架构的收益。

## 四、预算标定（先固定种子与参数，再只改预算）

* 种子：训练/校准 `700000..700047`（48 个本轮自录 stream），验证 `700048..700071`（24 个）；
  `500000..500399` 仍属上一轮登记的独立测试集，本轮不读不看；`600000+` 为在线系统块。
* 参考臂 = 无预算 `event` 在 48 个训练 seed 上的平均空口字节 **117 272 B**；
  档位 75% / 50% / 25% → **锁定绝对额度 87 900 / 58 600 / 29 300 B**（`COMM_BUDGET_LOCKED.json`）。
  之后所有臂读同一文件；**禁止**用某集自身未来流量给该集定预算。
* 执行器 `CommBudget`：初始额度 = 档位、不补充、按"载荷字节 × 订阅者数"扣费、**丢包照扣**；
  付不起就不发射（额度永不转负），被拒次数与应计字节计数。所有臂同一实例同一参数。
* 统一静默上界（传输层，不再是某个策略的私有规则）：非紧急 `max_wait_steps=10`、
  紧急（剩余 ≤ 25 步）`urgent_wait_steps=3`；被强制的发射同样计费，付不起同样被拒。
* 预算耗尽后的行为（明确）：不发、继续排队、年龄继续增长、`budget_refused_sends` 增长；
  不存在"免费突破预算"的通道，紧急也不例外——代价是可能永远无人知晓，因此
  "活动却无人知晓任务步"作为固定上报的诊断量。

### 标定实测（48 训练 seed 均值，`COMM_BUDGET_CALIBRATION.json`）

| 臂 | 额度 | 按时完成 | 过期 | 空口字节 | 额度占用 | 被拒发射（seed 数） | 接收信息年龄(步) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `event`（无预算参照） | — | **5.833** | 0.08 | 117 272 | — | 0 (0/48) | 4.4 |
| `immediate` | 全部档位 | 5.583 | 0.33 | 14 737 | 12–17% | 0 (0/48) | 105.3 |
| `periodic_5` | 75% | 5.438 | 0.48 | 76 345 | 87% | 22/48 | 40.5 |
| `periodic_5` | 50% | 4.938 | 0.98 | 55 757 | 95% | 39/48 | 78.7 |
| `periodic_5` | 25% | 4.000 | 1.92 | 28 246 | 96% | 48/48 | 112.6 |
| `event` | 75% | 5.250 | 0.67 | 86 146 | 98% | 45/48 | 18.3 |
| `event` | 50% | 4.396 | 1.52 | 57 241 | 98% | 48/48 | 69.2 |
| `event` | 25% | 3.521 | 2.40 | 28 280 | 97% | 48/48 | 132.5 |
| `event_priority` | 75/50/25% | 5.250 / 4.375 / 3.583 | — | 同 event | — | 45/48、48/48、48/48 | 18.2 / 69.2 / 122.6 |

读法：预算一收紧，"说得多"的规则立刻比"说得少"的规则差——`immediate` 因为只花 12–17% 额度而在
50%/25% 档上**胜过** event 与 periodic_5。所以本 pilot 要学的不是"多刷"，
而是**在有限额度里挑该说的刷新**：上界是 5.833（无预算 event），
必须打败的同档规则是 5.25（75% 档 event）与 5.583（不受约束的 immediate）。

**档位选择规则（写于跑档之前）**：取满足"event 臂在 ≥50% seed 上有被拒发射"且"各臂最佳均值仍低于无预算参照"
的最大档位。按此规则选中 **75pct = 87 900 B**。更紧档位只作为已训控制器的敏感性诊断报告，
不据结果改选训练档。

## 五、学习模型（第一轮只控一个维度）

* 状态：19 维命名特征（`comm_learning.FEATURE_LAYOUT = comm-send-state/1`），逐候选消息一行，
  共享 MLP 19→64→64→(2 logits, 1 value)，候选间 mean-pool 成步级 value。无 ID 输入。
* 动作：每候选 `send|hold`；采样独立 Bernoulli，选中的按 p(send) 降序交给统一执行器；
  被静默上界强制的候选从策略梯度中剔除（不算它的学习成果，也不给它 credit）。
* 决策步长：每联合步一次决策，与"发了几条"解耦；`decisions` 与 `transmission_attempts` 同时上报，
  避免把"少调用模型"混成"少发消息"。
* 主目标：episode 内按时完成**增量**（与评分定义同一来源，逐 tick 记账并与生命周期日志核对）；
  预算由执行器硬保证，奖励里**没有**消息惩罚项；信息年龄等只作诊断。
* 回退：策略异常/缺失输出立即回退到 `event` 规则并计数；本轮网络不接管收件人选择、路由与分配。
* 训练：PPO（clip 0.2、GAE λ=0.95、γ=0.99、value 0.5、entropy 0.01、Adam 3e-4、grad clip 0.5、
  4 epochs、minibatch 256 步、每更新 4 episode、ε=0.10），CPU；PILOC/CTDE/联盟策略全程冻结。

## 六、pilot 规模与预登记口径

* **先测吞吐**：单进程 1 episode（1000 步）≈ 2.7–3.2 s，即约 320 环境步/秒；一次更新（4 episode）
  ≈ 12 s。故 100 000 环境步 ≈ **100 episode ≈ 5–6 分钟/seed**（3 seed 并行）。
  这明确是**先导预算**，不代表充分收敛，不因结果不理想自动延长。
* 每 20 000 步在 8 个验证 seed 上贪心评估一次；checkpoint 可续训（含 optimizer 与游标），不入库。
* 公平验证（§7）：同一验证流上比较 `periodic_5 / event / event_priority / learned`，
  同额度同静默上界；各臂自行决定发射与到达；无损为主比较，丢包率 0.3（`delivery_probability=0.7`）只作附加诊断。
* 信道随机过程：每 seed 一条 `default_rng(seed)` 均匀序列，按**尝试序号**消费——第 k 次尝试用 u_k。
  这样"少发"不会把整条序列往后挪而换个更宽松的随机尾部；不同臂的尝试次数不同，
  因此**不宣称逐包配对**，只在 seed 层配对，并同时报告 delivered/applied/lost 三列，
  使"少收到"不被读成"少发送"。
* 报告量：按时完成数与率、空口字节与尝试数、delivered/applied/duplicate/lost、
  发射等待与接收信息年龄、额度占用率、被拒发射、强制发射、与规则臂的 seed 级配对差 + bootstrap CI
  （B=10 000，RNG 20260919）、三个训练 seed 各自结果（不只报最好的一次）、推理开销 ms/决策。
* 停止与判定：
  - 学习策略在同预算下**稳定优于**最强规则 → 建议扩大训练并预注册新的独立测试；
  - 接近规则 → 报曲线、方差与收敛状态，提出一项最有依据的改进，不自动扩动作空间；
  - 规则明显占优 → 分析是训练不足/观测不足/规则已足够，不写成"没有学习空间"；
  - 预算实现或信息来源不成立 → 立即停止训练先修正确性（本轮 smoke 即为此设的闸门）。

## 七、脚本清单

```bash
python record_streams.py                 # 本轮自录 72 条检测流（冻结 PILOC）
python pilot_preflight.py                # §1/§2 配置与门槛核对
python calibrate_budgets.py --phase reference|lock|tiered|aggregate
python train_pilot.py --mode smoke --tier 75pct
python train_pilot.py --mode train --seed 1 --tier 75pct --directory train
python evaluate_pilot.py --tier 75pct --checkpoints train/seed1/pilot_seed1.pt ... \
    --include-lossy --include-deployed
python round_report.py                   # 清单与门槛汇总
```
