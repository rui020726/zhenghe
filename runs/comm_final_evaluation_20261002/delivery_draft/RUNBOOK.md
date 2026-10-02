# 运行手册（环境、依赖、命令、可复现性）

状态：交付文档（ROUND37 §7 第 2 项）。命令均为本机实际用过的写法（Windows Git Bash）。

## 1. 环境

| 项 | 值 |
|---|---|
| 解释器 | `D:\project_env\python.exe`（Git Bash 内 `/d/project_env/python.exe`） |
| Python / torch / numpy | 3.10.19 / 2.9.0+cu128 / 2.2.6 |
| 并行度 | 4 个仿真进程（授权上限）；实测单格 12.7–13.3 s，576 格约 30 分钟 |
| 线程设置 | 未改：与训练轮保持同一 MKL/OMP 默认，避免浮点归约顺序差异影响可比性 |
| 新建 venv / 安装依赖 | 0（`import torch` 开工即成功） |
| 网络 | 0 次尝试；不推送 |

仓库与身份：

```bash
cd /d/code/整合/zhenghe
git rev-parse --abbrev-ref HEAD   # eval/comm-final-fixed-evaluation
git rev-parse HEAD                # 起点为 e2a272c0303f580d957e778f7e222a56d265372f（ROUND36 交付头）
git log --oneline -1 fix/comm-task-reward-event-timing
```

## 2. 被冻结的输入（改任何一项都不是"复现本轮"）

| 输入 | 位置 | 说明 |
|---|---|---|
| 三个模型 | `runs/comm_final_train_20261001/seed{11,22,33}/checkpoints/v2_train_s*_ep128.pt` | 只评 ep128；哈希与参数身份登记在 `FROZEN_EVALUATION_PLAN.json` |
| 场景块 | 1600000..1600063 | 占用审计 `SCENE_USAGE_AUDIT.json`；只用占用选块，未读结局 |
| 世界/通信配置 | `v2_common.stage_b_inherited_config()` + `STAGE_B_INHERITED_OVERRIDES` | 预算 29300 B、投递 0.7、ACK 结构关闭、range 0.3、gossip 2、steps 1000、num_goals 6、device cpu |
| 搜索/分配权重 | `runs/cur_v2rep05_s2/best/shared_mappo_model_21_3100.pth`、`runs/joint_eval_20260916/ctde_final_7_1.pth` | 哈希在冻结件里 |
| 解码语义 | `budget_masked.py:238-241` | greedy = 自由候选上 `p[SEND] >= p[HOLD]`；sampled = masked multinomial |
| 动作 RNG | `v2_run.eval_action_seed`，标签 `round24-eval` | 每 (seed, 场景, 模式) 一个派生种子 |
| 统计方案 | `r37_common.STATISTICS` | 场景升序、PCG64(20261002)、10000 × 64 共享索引、linear 分位 |

## 3. 执行顺序

```bash
cd /d/code/整合/zhenghe/runs/comm_final_evaluation_20261002

# 0) 占用审计（0 启动）——必须先于冻结件
/d/project_env/python.exe r37_scene_audit.py

# 1) 冻结评估对象（0 启动）——必须先于任何环境启动
/d/project_env/python.exe r37_freeze.py

# 2) 技术门禁：首个场景九臂（9 次，计入 576），随后逐项核对（0 启动）
/d/project_env/python.exe r37_run.py --shard 0 --shards 1 --scenario 1600000
/d/project_env/python.exe r37_gate.py

# 3) 其余矩阵：4 分片并行，原子 claim 保证不重复
for i in 0 1 2 3; do
  /d/project_env/python.exe r37_run.py --shard $i --shards 4 &
done; wait

# 4) 统计与表格（0 启动）
/d/project_env/python.exe r37_stats.py
/d/project_env/python.exe r37_tables.py

# 5) 账本对账与交付登记（0 启动）
/d/project_env/python.exe r37_reconcile.py
/d/project_env/python.exe r37_delivery.py
```

中断后重跑同一分片是安全的：已 claim 的格会被跳过，已写行的格即使被再次领取也只登记
`reused_rows` 而不启动环境。

## 4. 技术失败的正规处理（本轮如发生）

1. 失败格由 `run_job` 捕获并写入 `EVALUATION_LAUNCH_BOOK.json`（`technical_failures` 计数、
   异常类型与消息），真实 ledger 也留下一条 `ok=false`；**失败照计，不退还额度**。
2. 重跑必须显式使用 `--retry-failed`（领取 `job_<index>.retry.claim`），不得静默重跑混入结果。
3. 性能差（低完成率、零检测）不是技术失败，不重跑、不剔除。
4. 若异常指向被测代码影响行为：立即停止全部分片、保留现场、按退出条件第三支报告，
   不改代码后混结果。

## 5. 常见故障与判读

| 现象 | 先查什么 | 说明 |
|---|---|---|
| `freeze identity mismatch; refusing to resume` | 是否用了 ROUND24 manifest 的 token | 评估只读装载要交回 checkpoint **自己携带**的 freeze 块（`r37.stored_freeze_token`）；ROUND36 写盘时带 7 个字段，硬拼 4 字段会被拒 |
| `launch cap ... refusing a start` | 是否超过 672 | 由真实 ledger 判定，不是自记件 |
| 某臂某场景缺行 | `EVALUATION_LEDGER_RECONCILIATION.json` 的 `missing_scenarios_per_arm` | 缺格时统计脚本会直接退出而不是静默少算 |
| 行里 `reward_timing_version` 为 null | 是否误用旧轮 harness | 本轮走的是 ROUND24+ROUND34 的 `v2_run`，行内必写两个版本字段 |
| 区间宽到包含 0 | 无 | 这是描述性重抽的合法结果，**不得**据此说"等效"或"无差异" |
| 想改阈值/换里程碑 | 不允许 | 授权边界 §2/§5；任何改动都属于新一轮 |

## 6. Git 内 / Git 外

Git 内：脚本（`r37_*.py`）、冻结件、门禁件、统计件、表格件、对账件、报告、`delivery_draft/*.md`。
Git 外（逐件哈希登记在 `delivery_draft/ARTEFACT_HASH_REGISTRY.json`）：`rows/*.json`（576）、
`logs/*.jsonl`（每格 3 份：tape / lifecycle / transcript）、`claims/*.claim`、`shards/*.json`、
`logs/START_LEDGER_p*.json`。`.gitignore` 的白名单只放行脚本与 JSON 交付件。
