# ROUND36 执行依据

本轮提示词原文存档如下（逐字保留，作为授权边界的唯一依据）。

---

# ROUND36 执行提示词：修复版冻结训练（最后一次有界训练）与交付文档并行准备

你是一名没有历史记忆的新执行 agent。本提示词自带全部必要背景。严格遵守授权边界，任何超出边界的动作先停止并报告。

## 一、背景与身份锚点

项目研究通信 SEND/HOLD 策略的保任务降成本训练。历史缺陷与修复链：
- ROUND32：旧版任务奖励采集晚一个 relay tick（legacy 采集语义），其结果只能作 legacy 基线，不得外推。
- ROUND34：确认时间错位并修复为事件总线上的 `PublishedProgressRewardCollector`，同 tick 写入 `task_stream[t]`；新训练版本 `constrained-cost-v2/published-completion-event-at-relay-tick/2`，奖励采集版本 `published-completion-event-at-relay-tick/2`。
- ROUND35：小规模实机 smoke 全部通过（45 个真实完成事件逐条核对 0 违例；恢复边界 5/5 篡改拒绝、跨进程恢复逐字节一致；被动性 A 级；旧 18 个 .pt 内容层 18/18 重算全等），结论 go，可进入冻结训练。

- 执行仓库：`D:\code\整合\zhenghe`（Windows）。分支 `fix/comm-task-reward-event-timing`，本地头 `577344aab8a5c9bfcf75e7ffbf6f77e503dbc200`（ROUND35 交付头）。
- 若本地路径不存在，停止并报告阻断；不得改用 GitHub 快照（快照不含完整历史，训练必须基于本地完整分支）。
- 开工前核验：仓库路径、分支、头哈希；ROUND34 目录 `runs/comm_task_reward_event_timing_20260930/` 与 ROUND35 交付目录（含 ROUND35_SMOKE_REPORT.md 与账本 JSON）存在；`import torch` 可用并记录 torch/numpy 版本。核验结果记入 EXECUTION_BASIS 类文件。

## 二、必读文件（开工前通读）

1. `runs/comm_task_reward_event_timing_20260930/TASK_REWARD_TIMING_CONTRACT.md` 与 `TRAINING_VERSION_MANIFEST.json`。
2. ROUND35 交付目录下的 `ROUND35_SMOKE_REPORT.md` 及账本/核对 JSON——其中的残余风险条款本轮必须执行（见第四节"训练协议"）。
3. `runs/comm_cost_v2_rerun_20260928/` 下 `v2_train.py`、`v2_trainer.py`、`v2_run.py`、`v2_reward_timing.py`、`v2_freeze.py`、`v2_common.py`、`v2_controller.py`。
4. ROUND32 归档的运行配置与每 seed 实际训练量/耗时记录（本轮训练量与 wall-clock 上限的基准来源）。

## 三、授权边界（硬性）

- 训练启动：最多 3 次训练启动，每次 1 个 seed，三个 seed 均从初值训练。另设失败储备：全部 seed 合计最多 1 次重跑，仅在已记录失败原因后使用。
- 训练量：每 seed 的批次数/episodes 数/优化器步数与 ROUND32 冻结的 Stage C 配置完全一致（沿用归档配置，不得增减）。
- wall-clock：单次启动不超过 ROUND32 同 seed 归档实际耗时的 1.5 倍；从归档日志读取基准并写入执行依据。
- 评估启动：本轮不做正式评估。仅允许每次训练结束后 1 次最小 sanity 前向（固定 checkpoint 加载成功、版本字段齐全、可前向），不计入评估额度、不产出结果数字。
- 冻结项（一律不得改）：搜索/分配（PILOC 与 CTDE 冻结现状）、网络结构、目标函数、奖励数值、成本定义与采集路径、观测与 mask、GAE/γ/λ/对偶预算/超参数、场景块（与 ROUND32/ROUND35 同一块，不换块）、行过滤策略（ALL_ROWS 原编码，不启用 ACTIVE_ROWS）、训练版本字段。
- 三个 seed 取自冻结 Stage C 配置中已声明的 seed 集合，启动前写入执行依据；不得根据中间结果更换、增删或挑选 seed。
- 禁止：热启动或续训任何旧 checkpoint；网格搜索与任何调参；启用 ACTIVE_ROWS；启用 v2_recover 一类重放路径（ROUND35 残余风险条款：启用前需另查幂等，本轮不启用）；改写 ROUND32/33/34/35 旧产物；推送 GitHub 与任何网络发布；把本轮任何中间数字对外表述为学习效果结论。
- 禁止新建虚拟环境或安装依赖。

## 四、训练协议（ROUND35 残余风险转写，必须执行）

1. 场景块与 smoke 相同，无需重做可达性判断；若任何配置差异导致场景块变化，立即停止并报告。
2. 每个训练进程独立记账：进程启动次数、episode 构造数、optimizer.step 数、wall-clock、事件账本完整性；账本随产物归档，报告从账本对账，不允许手打数字。
3. 每个 milestone checkpoint 必须携带 training_version 与 reward_timing_version 两字段；freeze manifest 同样记录两字段。
4. 进程崩溃或中断：保留现场（日志、tape、账本、最后一个 checkpoint），记录最小复现信息；该 seed 计为失败，使用失败储备重跑前必须写明原因分类（被测系统缺陷 / 执行方操作 / 环境）。
5. 发现影响正确性的被测系统缺陷：立即停止所有受影响运行并保留证据，不用失败储备绕过，直接转入退出条件第三支。
6. 修复改变了奖励入账 tick 与归属行（ROUND35 量化：归档同口径 65% 入账曾被下一条动作行抢走），因此 loss/梯度/参数轨迹与 ROUND32 不同是预期现象；不得在报告中把任何行为差异表述为"约束被满足得更好"或学习效果证据。

## 五、任务清单

1. **冻结核验**：对比本轮实际配置与 ROUND32 冻结 Stage C 配置，逐项列出相同项；任何 drift 立即报告。输出冻结配置清单 JSON（含 seed、训练量、超参数、场景块、版本字段）。
2. **三次从初值训练**：按协议执行 3 次启动。每次结束做最小 sanity（版本字段、可加载、可前向）并归档：checkpoint 文件字节 SHA256、milestone 清单、tape、事件账本、训练日志、λ 轨迹、freeze manifest。
3. **并行文档**（不占启动额度，训练等待期间完成）：
   a. 系统架构说明草稿：adapter/orchestrator/registry/event bus/collector/trainer 的事件顺序图与数据流；
   b. 复现命令草稿：环境、依赖版本、启动命令、配置路径、seed；
   c. 结果表模板：按时任务数、计量字节，greedy/sample 分列，开发/独立测试分列，留空待第 4 天评估填入；
   d. 交付目录骨架：代码身份、模型清单、Git 外产物位置/哈希登记表。
4. **交付对账**：汇总三个 seed 的账本对账表（启动数、构造数、optimizer.step 数、wall-clock、版本字段齐全率、checkpoint 哈希清单），标注与 ROUND32 同口径训练量的一致性。

## 六、交付物

- `runs/comm_final_train_2026XXXX/ROUND36_TRAIN_REPORT.md`：执行依据、冻结配置清单、每 seed 训练记录与 sanity 结果、账本对账表、失败与储备使用记录（如有）、明确的 go/no-go 结论（可否进入第 4 天固定评估）。
- 每 seed 一个产物子目录：checkpoints、tape、账本、日志、freeze manifest、哈希清单 JSON。
- 文档草稿与交付目录骨架放在 `runs/comm_final_train_2026XXXX/delivery_draft/`。
- 按主题分提交到本地分支 `fix/comm-task-reward-event-timing`，报告最终头哈希。不推送远端。

## 七、退出条件

- 3 个 seed 全部完成且账本/版本字段/checkpoint 齐备 → 报告"可进入第 4 天固定评估"，列出残余风险。
- 单 seed 失败且储备可用 → 记录原因后重跑该 seed 一次；储备耗尽仍有 seed 失败 → 停止，如实报告完成度，不追加额度。
- 出现被测系统正确性缺陷 → 立即停止全部运行，给最小复现、涉及文件与行号、证据位置；不带病训练，不承诺赶出新模型。
- 环境/配置阻断 → 报告阻断项与已尝试路径，不交付半成品结论。

---

## 八、开工核验结果（2026-10-01）

| 核验项 | 结果 |
|---|---|
| 仓库路径 | `D:\code\整合\zhenghe` 存在，本地完整分支可用（未使用 GitHub 快照） |
| 分支 | `fix/comm-task-reward-event-timing` |
| 开工本地头 | `577344aab8a5c9bfcf75e7ffbf6f77e503dbc200`（= ROUND35 交付头，逐字相符） |
| ROUND33 基线 `c7d47d25…` 是否为祖先 | 是（`git merge-base --is-ancestor` 返回 0） |
| ROUND34 目录 | `runs/comm_task_reward_event_timing_20260930/` 存在，含 `TASK_REWARD_TIMING_CONTRACT.md`、`TRAINING_VERSION_MANIFEST.json` |
| ROUND35 交付目录 | `runs/comm_reward_smoke_20260930/` 存在，含 `ROUND35_SMOKE_REPORT.md`、`SMOKE_LAUNCH_BOOK.json`、`LAUNCH_ACCOUNTING.json` |
| ROUND32 归档 | `runs/comm_actor_reduction_ablation_20261001/` 存在，含 `CONFIG_AND_MODEL_MANIFEST.json`、`START_LEDGER.json`、128 行/seed 的训练行与 32 份批记录/seed |
| 真实训练代码 | `runs/comm_cost_v2_rerun_20260928/`（ROUND24 目录，含 ROUND34 就地修复）：`v2_train/v2_trainer/v2_run/v2_reward_timing/v2_common/v2_controller/v2_freeze` 全部通读 |
| 解释器 | `D:\project_env\python.exe`，Python 3.10.19 |
| torch / numpy | `2.9.0+cu128` / `2.2.6`，`import torch` 成功，`cuda.is_available()=True`，逻辑 CPU 20 |
| 新建 venv / 安装依赖 | 0 项，未执行任何安装 |
| 冻结搜索/分配权重 | `runs/cur_v2rep05_s2/best/shared_mappo_model_21_3100.pth`、`runs/joint_eval_20260916/ctde_final_7_1.pth` 均在位，哈希与 ROUND32 manifest 相同（G6） |

### 8.1 训练量与 wall-clock 上限的归档基准（从 ROUND32 归档读，只读）

ROUND32 没有记录 per-launch wall-clock，只逐集记录了 `seconds`。因此每 seed 的基准取该 seed
归档 128 集 `seconds` 之和（`ALL_ROWS` 臂，本轮所用行过滤口径），上限 = 该值 × 1.5。
批更新耗时不在此基线内，本轮另行实测登记（见报告），且 1.5 倍余量 ≥ 756 s，远大于实测更新开销。

| seed | 归档 episode 数 | 归档 rollout 秒数和 | 均值/集 | 归档批数 | 归档优化器步数 | 归档样本数 | wall-clock 上限 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | 128 | 1513.52 | 11.8244 | 32 | 464 | 26156 | 2270.28 s（0.631 h） |
| 22 | 128 | 1597.11 | 12.4774 | 32 | 340 | 17751 | 2395.66 s（0.665 h） |
| 33 | 128 | 1521.30 | 11.8852 | 32 | 464 | 26010 | 2281.95 s（0.634 h） |

ROUND24 原始 v2 运行（同一训练器、同一顺序）作交叉核对：s11 1533.4 s、s22 1532.5 s、
s33 1537.5 s，与上表同一量级，说明该基准不是离群值。

### 8.2 三个 seed（启动前固定，取自冻结 Stage C）

`v2_common.TRAIN_SEEDS = (11, 22, 33)`，与 ROUND32 冻结 manifest 的 `train_seeds` 相同；
本轮三次启动按 11 → 22 → 33 顺序各一次，不据中间结果更换、增删或挑选。

### 8.3 授权额度与本轮实际用量（对账见 `LEDGER_RECONCILIATION.json`）

| 额度 | 授权上限 | 实际 |
|---|---|---|
| 训练启动进程 | 3（每次 1 个 seed） | 待填 |
| 失败储备重跑 | 全部 seed 合计 1 | 待填（未使用则记 0） |
| episode 构造 | 384 = 3 × 128 | 待填 |
| 评估启动 | 0（本轮不做正式评估） | 0 |
| sanity 前向 | 每次训练后 1 次 | 待填 |
| 网络 / 推送 | 0 | 0 |
| 旧轮产物改写 | 0 | 0 |

### 8.4 唯一干预的登记（G7 门禁的函数级证据）

生产代码 `integrated_sar/`（世界、PILOC 搜索、CTDE 分配、通信传输、任务登记、事件总线、
生命周期指标）自 ROUND33 基线以来 **0 次提交、0 字节变化**（G6）。训练路径相对 ROUND32 冻结
头 `df176bd…` 的函数级源码比对显示：`ppo_update`、`batch_samples`、`likelihood_alignment`、
`dual_update`、`batch_g_hat`、`_unassigned_prefix_totals`、`run_training_episode`、
`action_seed`、`_assert_transport`、`_booked`、`eval_action_seed` 逐字节相同；发生变化的只有
`_bundle`（tick 0 改为 unassigned prefix）、`save_checkpoint` / `load_checkpoint` /
`fresh_training_state`（两个版本字段与恢复门）以及新增的 `load_checkpoint_read_only` /
`_validate_checkpoint_base` / `_restore_controller`，运行侧只有 `_install_taps` /
`_run_episode` / `run_eval`（采集器搬到事件总线、行内写两个版本字段与事件账本）。因此本轮训练
与 ROUND32 的差异 = ROUND34 的奖励时序修复本身，没有第二处改动。
