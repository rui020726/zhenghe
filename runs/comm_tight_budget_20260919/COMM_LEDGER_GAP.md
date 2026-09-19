# 账本剩余缺口：机制、修正与不变性

日期：2026-09-19。分支 `fix/comm-ppo-probability-budget`。对应执行文档
`COMM_PPO_BUDGET_NEXT_ROUND.md` §3，总纲 v1.3 §5.4。与概率修复（`COMM_PPO_PROBABILITY_AUDIT.md`）
分开提交、分开解释。

## 一、上一轮留下的读数

第三轮的 `attribution_metrics._closure` 报告 `unspoken_or_uncounted_at_end` 在部分 episode 为**负**
（全矩阵 −40 单位 / 21 集，`random_send` 每集 −6.5）。当时把它写成"轮询侧 coalescing 漏计"，
并留下一条判断：负值"只是 `coalesced_changes` 的盲区，成本、投递与结局恒等式都不受影响"。
本轮按 §3 要求给它做最小复现，结果是：**盲区是真的，但它不是负值的来源。**

## 二、两个候选机制，分别实测

脚本：`ledger_gap_reproduction.py`；输出：`COMM_LEDGER_GAP_REPRODUCTION.json`。恒等式写作

```text
generated_items = (emissions - redundant) + coalesced_changes + suppressed + 终局未说
```

`redundant` 有三种读法：注册表 `revision`、中继自己按内容判定的 `redundant_emissions`、
以及离线从 transcript 重建的内容。三者的差别就是这个缺口的全部。

**A 轮询侧 coalescing（3 条变化 → 1 封信）。** 检测事件入队后，两条注册表变化先后被轮询发现，
都被同一封信带走。修正前 `generated=1 / coalesced=0`，修正后 `generated=3 / coalesced=2`
（本轮改动），**两种情况下的三种恒等式读数都等于 0**：这条合并少计一个 generated 的同时也少计
一个 coalesced，两边同时消失，残差不动。所以这个机制在结构上不可能产生负值——上一轮的诊断
把"看不见"与"造成偏差"混为一谈了。

**B revision 动了、可执行内容没动（2 封信 → 1 条变化）。** 一次注册表更新把 `revision` 推进了 1，
但 `stage / progress / remaining / members / deadline / position / confidence` 全部不变——
`_View` 因此相同，`generated_items` 不增加。而离线 `redundant` 若按 `revision` 判，会把第二封信
当作"答了一条新新闻"。实测：`generated=1 / emissions=2 / redundant_by_revision=0 → 残差 −1`；
按内容（离线重建与中继的 `redundant_emissions` 都给 1）→ 残差 0。
**这才是上一轮那个负号**：恒等式两边用了两套"同一件事"的定义。

`_View` 明确排除 `revision`（注释原话："它说明注册表动了，不说明任何机器人会据此行动的东西动了"）。
因此 generated 计数的是可执行视图的变化，而 `revision` 计数的是注册表写入次数——它们本来就不该
互为补数。§3 要的"明确状态变化项与重复发送尝试不是同一计数对象"，落点正在这里。

## 三、改了什么（生产侧，纯记账）

`integrated_sar/comm.py`：

1. `_TargetLedger` 新增 `ready_view`：发送端已经为其计入一条新闻（事件入队或轮询发现）的那个视图。
   与 `last_sent_view`（说过什么）分开，专门用来判断"轮询看到的这条变化是否已经记过账"。
2. `handle_detection` 在记完一条事件新闻后把 `ready_view` 置空——事件按定义就是一条新新闻，
   内容留待下一次轮询登记，避免同一条新闻被事件和轮询各计一遍。
3. `_observation` 的轮询分支从"只在队列为空时计 generated"扩成三分支：队列为空 → 新新闻
   （`generated_items += 1`，登记 `ready_view`）；队列非空但 `ready_view` 为空 → 这是事件那条新闻的内容，
   只登记不计数；队列非空且视图又前进了 → **一次轮询侧合并**（`generated_items += 1` 且
   `coalesced_changes += 1`，与事件路径对同一种情形的处理一致）。
4. 新增 `redundant_emissions` 计数：发射时 `SendItem.changed_view` 为假即记一次发送侧重复。
5. 成功发射与终态抑制都清空 `ready_view`；预算拒付不计数、不改队列、不让时钟倒退（原语义不变）。

**没改的东西**：载荷构造、发射时刻、候选裁剪顺序、预算扣费、上界强制、投递与丢包、以及任何
影响任务结局的路径。§3 明确禁止"为使恒等式成立而反推填充计数"——上面的计数都由"是否有一条可执行
变化 / 是否重发了同一视图"这两个独立事实驱动，不是拿恒等式反解出来的。

## 四、不变性验证（§3 的验收）

`ledger_fix_invariance.py`：同一套规则臂 × 两档额度 × 48 条训练流 = **624 格**，在修正前后各跑一遍
（`probe/` 与 `probe_after/`，同一冻结代码除 `comm.py` 记账外）。

| 检查 | 结果 |
|---|---|
| 行为字段（emissions / forced / airtime / autonomous_bytes / forced_bytes / on_time / detected / expired / belief_age / unknown_to_everyone）逐格相同 | 624/624，**0 处差异** |
| 拒付明细（首次拒付步、原因、消息类型、余额）逐格相同 | 624/624 |
| 配置同一性（剥掉 4 个产物路径字段后重建配置比对） | 56 格检查，0 处不一致 |
| `generated_items` 与 `coalesced_changes` 的变化量 | 各自 +44 124，逐格相等 |
| 旧读法下的 `unspoken_or_uncounted_at_end` | 总变化 0（预测成立：两边同增即抵消） |

`config_hash` 不列入行为字段：它把 transcript 路径折进了哈希，两份快照故意写到不同目录，
所以它必然不同——这一点单独由"配置同一性"检查覆盖，而不是被当成差异吞掉。

## 五、哪些产物需要更新，哪些保留

* **保留原样**：第三轮的 `COMM_POLICY_ATTRIBUTION.json/.md` 及其 §六判定线、首轮 pilot 结果、
  第二轮稳定性判据（0/3）。它们是历史事实，本轮不回头改写。
* **本轮重新生成**：本轮自己的规则/学习对照一律走 `probe_after/`（带 `redundant_emissions` 与修正后
  的 coalesced 计数）。第三轮的 transcript 已不在场，其负残差不重算，只在文档层面更正归因。
* **对本轮结论的影响**：负残差是记账口径问题，不涉及发射、扣费或任务结局，因此第三轮关于
  "收益来自共享静默上界"的归因不受影响；受影响的是"生成量"这一诊断量的绝对值，本轮已在
  `probe_after/` 用一致口径重测。

## 六、复跑

```bash
# 两个候选机制的恒等式三读法
python runs/comm_tight_budget_20260919/ledger_gap_reproduction.py \
  --seeds 700000 700001 700002 700003

# 修正前后 624 格不变性（需先跑 probe 与 probe_after 两个快照）
python runs/comm_tight_budget_20260919/ledger_fix_invariance.py

# 记账回归
python -m unittest integrated_sar.tests.test_comm_send_policy.GenerationAccountingTests -v
```
