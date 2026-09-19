# 通信学习训练稳定性协议（S3 第二轮）

日期：2026-09-19（同日第三轮）
对应总纲：`PROJECT_CHARTER.md` v1.1 §9 阶段 S3；执行指令 `COMMUNICATION_NEXT_ROUND.md` §3–§6。
代码基础：`feat/learned-communication-budget-pilot@ab5728beb124d7d0edd710457b2f84cdd6a5d244`
本轮分支：`feat/communication-training-stability`
工作目录：`runs/comm_stability_20260919/`

本文在读取任何 200k 结果**之前**写定判定线（§7）。§2、§3 的核对与 §6 的续训配置在写定前已完成，
其结论只影响"是否值得续训"，不影响判定线的内容。

## 1. 本轮阶段与主问题

阶段 S3 第二轮。研究问题不是"能不能再涨一点"，而是首轮那条收益到底属于谁：

1. 属于模型的发送时机决策，还是属于所有策略共享的 3/10 步强制静默上界？
2. 属于学习，还是属于"比初始化更沉默"这一随机网络也能做到的事？
3. 同样的训练方法能否在 3 个训练 seeds 上稳定产生该效果？

PILOC、CTDE、修正环境（成员一致性开启）、分配侧 C 臂、预算 87 900 B、通信内容、静默上界
（`max_wait_steps=10`、紧急时 3 步、`urgency_time=25`）、19 维特征布局 `comm-send-state/1`
与网络结构全部沿用首轮。本轮不改 deadline、目标数、奖励项或动作空间。

## 2. 协议修正（先于对照运行）

### 2.1 发送归属修正：学习型策略的强制发送曾混入自身决策序列

首轮 `train_pilot.Collector` 把动作掩码判定的 must-send 直接并入 `decide()` 返回的选择列表。
两个后果：

- 归属失真：这些发送在传输层被算作"策略请求"，无法区分"模型想发"和"上界逼它发"；
- 与规则臂不对等：学习型策略自带优先级顺序（`orders_selection=True`），掩码项因此排在被
  传输层追加到队尾的强制发送**之前**，在额度耗尽时优先获得计费。规则臂不享有这一位置。

本轮改为：`Collector` 返回的列表中剔除掩码项，强制发送完全由传输层统一追加
（`stability_common.transport_bounded_collector`）。发送集合不变，变的是信用不足时的裁剪顺序
与归属标签。修改的实测代价见 §2.4。

### 2.2 每条发送尝试标注互斥主原因

`CommRelay._emit` 现在按 `policy` / `forced_maxwait` / `other` 三类记录
`emissions_*`、`attempts_*`、`bytes_*`、`refused_*`，并把 `send_reason` 写入 transcript；
投递、应用、重复按 `(sent_step, target)` 关联回产生它的那条发送，因此每一类原因都有
"花了多少"和"换回多少"两个读数。`other` 当前恒为 0——本协议不存在其它强制协议事件
（无 ACK、无重传，gossip 属于环境侧、在正常路径下单独计量）。

原先的 `forced_emissions` 计数发生在计费之前，一条因信用不足被拒的强制发送也会被记为
已发送；该口径已废除，改由 `emissions_forced_maxwait`（真实发射）与
`refused_forced_maxwait`（被拒）分别记录。

### 2.3 新增三个对照臂

`integrated_sar/comm_policy.py` 增加规则臂，与所有规则共用同一传输、预算与上界：

- `hold_to_bound`：永不主动请求，只有共享静默上界替它说话。这是"只学会闭嘴"的策略的形式化；
- `random_send(send_probability, seed)`：逐候选抛硬币，随机种子可复现、逐 episode 重启；
- `send_once_no_bound`：`immediate` 且**取消**共享上界，即历史"只发一次"本身。

第三个臂是必要的：在共享上界下 `immediate` 已经不是"只发一次"。24 条验证流实测
`immediate` 的 144 次首发之外还有 176 次由上界补发（35 084 B），而真正的只发一次是
14 916 B。首轮文档引用的 14 737 B 来自标定条件，与预算对照表不是同一个协议。

### 2.4 修正的实测代价

`attribution.py --mode collector-delta` 用同一批 100k checkpoint、同一 24 条验证流分别跑
首轮收集器与本轮收集器：

3 个 100k checkpoint × 24 条验证流 = 72 对 episode，两种收集器逐对比较
（`COMM_COLLECTOR_CORRECTION.json`）：

| checkpoint | 逐对同字节 | 逐对同完成数 | 平均字节（两式） | 平均完成数（两式） |
| --- | --- | --- | --- | --- |
| learned seed 1 | 24 / 24 | 24 / 24 | 86 169.75 | 5.333 |
| learned seed 2 | 24 / 24 | 24 / 24 | 86 685.00 | 4.125 |
| learned seed 3 | 24 / 24 | 24 / 24 | 39 360.25 | 5.958 |

即：修正不改变任何一集 episode 的发送集合、字节或任务结局（本场景裁剪从未因该顺序差别而落到
不同集合），只把归属标签摆正。首轮数字因此仍与本轮直接可比，本轮表中的 learned 100k 行与首轮
表格逐值一致（seed 3：+0.08333、−34 768.25 B）。修正的意义在于 §2.2 之后"自主发送占比"这一
列第一次成为可信量。

## 3. 指标单位口径

同一份结果按三种口径分别报告，字段名自带单位：

| 字段 | 单位 | 定义 |
| --- | --- | --- |
| `on_time_targets_per_episode_mean` | 个/episode | 每局按时完成任务数，再对 episode 取均值 |
| `on_time_rate_per_episode_mean` | [0,1] | 每局 `on_time / 该局检测目标数`，再取均值 |
| `pooled_on_time_over_detected_targets` | [0,1] | `Σon_time / Σ检测目标`，与上一行**不相等**（各局分母不同） |
| `airtime_bytes_mean` | B/episode | 发送尝试字节（含丢包） |
| `bytes_per_emission_batch` | B/批次 | 空口字节 ÷ 发射次数 |
| `bytes_per_receiver_packet` | B/包 | 空口字节 ÷ 传输尝试数（一发 × 每收件人计费） |
| `forced_share_of_bytes` / `by_reason.*` | 比例、计数 | 见 §2.2 |

### 3.1 首轮数字的单位核对

原文（`COMM_POLICY_PILOT_RESULTS.md` §三）：

> | learned 3 − `periodic_5` | +0.083 | [−0.083, +0.250] | 3 / 1 / 20 | **−34 768** | **[−40 363, −28 787]** |

同表"按时完成"列取值 5.958 / 5.875，即个/episode。因此：

- `+0.083` 的单位是 **个任务/episode**（= 6.0 个目标/episode 的 1.4 %），不是完成率，也不是百分点；
- 丢包 0.3 条件下的 `+0.333` 同样是 **个任务/episode**；
- 首轮 §六 判定线中的 `≥ −0.05` 沿用同列单位，即 **−0.05 个任务/episode**（约合每局 0.83 个
  百分点的逐局完成率），**不是** 5 个百分点。本轮不改写该判定线的历史表述，仅在此登记其单位。

本轮 24 条验证流复算得到 learned 3 − `periodic_5` = +0.08333 [−0.08333, +0.25]，
字节 −34 768.25 [−40 313, −28 776]，与首轮一致（首轮四舍五入到整数 B）。

### 3.2 推理开销纠错

首轮 checkpoint 内记录的 `decide_milliseconds_per_call` 为 200.2 / 206.8 / 254.7 ms。
单线程复测同一权重、同一验证流为 **0.19–0.24 ms/决策**。差约 1000 倍的成因：
`train_pilot.main` 未调用 `torch.set_num_threads`，三个训练进程各自按默认全核开线程池并互相
超订；`evaluate_pilot.main` 才设 `--threads 1`。首轮文档中的推理成本数字作废，本轮起训练与
评估脚本统一在入口设定线程数并记录。

### 3.3 未修复但已量化的训练口径问题

首轮探索用 ε-greedy（ε=0.10），而记录的 `log_prob` 是模型自身 π 的联合对数概率，不是行为策略
`(1−ε)π + ε·Uniform` 的。因此约 5 % 的候选动作来自硬币，而更新把它们当作 π 采出的样本。
一个决策步平均含 ~6 个候选，按步级联合 log-prob 计，一条样本被至少一次硬币翻转污染的概率
约 1 − 0.95⁶ ≈ 26 %。

配套后果：三 seed 训练 rollout 的额度使用率恒为 0.96–0.997，而 greedy 评估下 seed 3 只用 0.40
——模型是在"额度已被探索噪声打满"的世界里学习，而它的部署评价在另一个世界里。

本轮**不修**：指令 §5 要求"只改变一个主要变量"（时长）。这里登记为下一轮唯一值得先改的项，
并把 `voluntary_sends_share`、`emissions_by_policy` 计入每轮更新，使该效应可被后续实验量化。

## 4. 首轮训练量核对（执行值，不是注释值）

| 项 | 首轮实际值 | 依据 |
| --- | --- | --- |
| 100 000 环境步 | = 100 episode × 1 000 联合步 | `orchestrator.run` 为 `for _ in range(steps)`，无提前终止；lifecycle 末条 `final.timestamp = 1000.0` |
| 每更新轮 episode | 4（固定） | `EPISODES_PER_UPDATE` |
| rollout 更新轮数 | 25 | `state["updates"]` 长度 |
| 决策样本数（4 episode 一批） | 由 minibatch 数反推区间，见下 | `update()` 每 256 条决策切一个 minibatch |
| optimizer.step 次数 | 1 168 / 1 212 / 1 072（seed 1/2/3） | Σ minibatches |
| 决策样本总数 | seed1 ∈ (68 352, 74 752]；seed2 ∈ (71 168, 77 568]；seed3 ∈ (62 464, 68 608] | 4 epochs × k minibatch，k=⌈N/256⌉ |
| 每 episode 决策数 | 106–1 000（上界即步数） | 单线程复测 |
| PPO epoch / minibatch | 4 / 256 条决策 | 常量 |
| approx_kl | −7e-4 … +2e-3 | 逐更新记录 |
| value_loss | 0.017 … 0.153 | 逐更新记录 |
| entropy（步级求和） | 0.06 … 0.41 | 逐更新记录 |
| clip fraction、梯度范数、价值解释方差 | **首轮未记录** | 本轮起记录 |
| 48 条训练流抽样频次 | 100 episode = 48+48+4 → 前 4 条各 3 次，后 44 条各 2 次；**没有 100 个不同场景** | `cursor` 循环 |

结论：首轮的"100k 环境步"名义上是模拟器步数，学习量是 ~7×10⁴ 条决策样本与 ~1.2×10³ 次梯度步。
用环境步数代表训练充分程度会高估近一个数量级，本轮之后以 optimizer.step 与决策样本数并列报告。

## 5. 数据划分与最终测试封存

详见 `COMM_DATA_SPLIT_AUDIT.md` 与 `COMM_DATA_SPLIT_AUDIT.json`。要点：

- 48 条训练流（700000..700047）与 24 条验证流（700048..700071）保持首轮登记的用途；验证流可以被
  反复诊断，但不得称为独立测试。
- 首轮登记的"独立测试集" 500000..500399 **不是全项目未使用**：遍历 `runs/**` 后实测该块
  **400 个 seeds 全部有已执行 episode**（`corrected_env_final_20260918` 的 400-seed C/D 对比），
  不只 500168。该块作为"未见测试集"的资格作废，历史结论仍按其原口径保留。
- 重新预留 `800000..800399`（400 个 seeds）：全项目既无 artefact 也无脚本声明；本轮只登记不生成、
  不查看。选块规则写在登记之前（已用最大 seed 700071 之上、100 000 对齐的最低连续 400 块），不按结果挑选。
- 全项目累计有 1 368 个不同 scenario seed 执行过 episode。

## 6. 有界续训配置

| 项 | 值 |
| --- | --- |
| 起点 | 首轮 `pilot_seed{1,2,3}.pt`（100 000 环境步） |
| 第一段终点 | 200 000 环境步（总计），新增 25 个更新轮 |
| rollout | **保持 4 episode/轮**（不改成 16） |
| 预算 / 奖励 / 动作空间 / 特征 / 网络 | 与首轮逐字相同 |
| 验证间隔 | 每 20 000 环境步，前 8 条验证流，greedy（首轮既有协议） |
| checkpoint | 150k / 200k /（条件性）300k / 400k 里程碑，实际落在首个 ≥ 里程碑的更新边界 |
| 更新边界跨度 | 4 000 环境步 ⇒ 150k 里程碑落在 152k |
| 并行 | 3 进程（每 seed 一个），`torch.set_num_threads(1)`，独立 rollout/eval 目录 |
| 停止 | NaN/Inf、预算越界、每 episode 实测步数 ≠ 1000 ⇒ 立即抛错保存状态 |

**恢复语义（warm start）**：actor、critic、Adam 动量、`cursor`、`environment_steps`、历史更新与
评估记录全部从首轮 checkpoint 精确恢复；首轮未保存 Python/NumPy/Torch 的 RNG 状态，因此三条随机
流从 seed 重新初始化。本轮 checkpoint 起已保存 RNG 状态，本轮内部中断可精确恢复。
100k→200k 这一段因此标记为 **warm start**，不称为无缝续训。

**吞吐实测**（相同线程配置，`train_stability.py --mode throughput --threads 1`）：
3.05 s/episode，327.9 环境步/秒，12.2 s/更新轮（4 episode）。由此 100k→200k 每 seed ≈ 305 s
rollout + 5 次验证 ≈ 8 min，三 seed 并行 < 10 min；扩到 400k 每 seed 约 25 min。
首轮"三个工作日"的估计作废。

预算是否触发：本轮实测 `immediate`、`hold_to_bound`、learned seed 3 的 87 900 B 额度使用率分别为
0.40、0.41、0.45——预算对这三条不binding；对 `event`/`event_priority`/`periodic_5` 与 seed 1/2
（0.98–0.99，单次评估被拒 463–32 426 次）才 binding。预算计费与实现核对无误（`refused_bytes`
与剩余额度闭合，见 §8 残差表），因此这属于场景成本结构，不需要收紧 deadline。

## 7. 冻结判定线（在任何 200k 结果之前写定）

比较设置：隔离 + 无损 + 87 900 B + 24 条验证流；统计为 seed 级配对 percentile bootstrap，
B=10 000，RNG 20260919（沿用首轮，便于对齐）。

主比较对象不再是 `periodic_5`，而是归因表里实际站在前沿的两条规则：

- `immediate`（表内唯一同时取得最高完成率与最低字节的上界内规则）；
- `hold_to_bound`（只靠共享上界的规则）。

对每个训练 seed 独立判定，四条同时满足才算"续训产生了规则之外的价值"：

1. **完成率非劣**（单位：个任务/episode）：`learned − immediate` 的配对差 CI 下界 ≥ −0.05；
2. **字节严格优**（单位：B/episode）：`learned − immediate` 的配对差 CI 上界 < 0；
3. **自主发送非空**：`by_reason.policy.bytes > 0` 且其配对差相对 `hold_to_bound` 的字节 CI 上界 ≤ 0
   （即不是靠上界替它说话、也不比纯上界更贵）；
4. 三个 seeds 中**至少两个**满足 1–3。

规则先于结果写定的理由：`immediate` 是表内最强的规则，因此这条线不是为最容易通过而挑的。

分支（对应指令 §6）：

- 满足 1–4 → 提议扩训到 400k 的判断成立，并在下一轮预注册独立确认（`800000..` 块）；
- 满足但幅度小 / 只有 1 个 seed → 报告学习曲线、方差与是否收敛，提出**一个**最有依据的改进，
  不自动扩大动作空间；
- 不满足且训练仍在改善 → 区分欠训练 / 欠观测 / 规则已足够，各自给出证据；
- 不满足且 200k 时训练与验证都不再改善 → 停止加步。

`400k` 是本轮每 seed 的硬上限，不因结果不好追加预算、改奖励或扩动作空间。

## 8. 总账闭合与验收

`attribution.py` 对 336 条 episode 逐条计算并输出以下残差，全部为 0 才算通过
（结果见 `COMM_POLICY_ATTRIBUTION.json` → `ledger`）：

| 恒等式 | 含义 |
| --- | --- |
| `emissions_split` | Σ各原因发射 = 总发射 |
| `attempts_split` | Σ各原因尝试 = 总尝试 |
| `bytes_split` | Σ各原因字节 = 空口字节 |
| `airtime_is_the_bill` | 空口字节 = 预算执行器扣费 |
| `attempts_are_delivered_plus_lost` | 尝试 = 交付 + 丢包（丢包照样计费） |
| `delivered_split` | 交付 = 应用 + 重复 + 被取代 + 未知任务 + 其它拒绝 |
| `each_emission_reaches_every_receiver` | 尝试 = 发射 × 订阅者数 |

另有一条独立实现交叉核对：transcript 离线读数与 relay 在线计数在 8 个共享量上逐 episode 求差
（`implementations_agree`），必须全 0。

`generated_balance` 报告 `generated_items = emissions + coalesced + suppressed_terminal +
unspoken_at_end` 的逐臂均值，其中 `unspoken_at_end` 不为 0 也如实列出——episode 结束前未发送的
消息不能靠"继续等待"逃出评价。

## 9. 复跑命令

```bash
cd runs/comm_stability_20260919
PY=/d/project_env/python.exe

# §3 数据划分审计（只读盘，不跑 episode）
$PY data_split_audit.py --with-configurations

# §2 归因矩阵：14 臂 × 24 验证流；三进程按 seed 取模分片
$PY attribution.py --shard s0 --seed-modulus 3 --seed-remainder 0
$PY attribution.py --shard s1 --seed-modulus 3 --seed-remainder 1
$PY attribution.py --shard s2 --seed-modulus 3 --seed-remainder 2
$PY attribution.py --mode aggregate
$PY attribution.py --mode collector-delta

# §4 吞吐实测
$PY train_stability.py --mode throughput --seed 1 --episodes 8 --record

# §5/§6 三 seed 有界续训（100k → 200k，可中断后重跑同一命令精确恢复）
for s in 1 2 3; do $PY train_stability.py --seed $s --steps 200000 --threads 1 & done

# §6/§7 公平验证（三设置、逐 seed、按规则选点）与稳定性判定
$PY evaluate_stability.py --include-train-set
$PY stability_preflight.py            # 配置真值表 + smoke
$PY stability_manifest.py             # 运行 manifest
```
