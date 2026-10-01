# 复现命令草稿

状态：草稿（ROUND36 并行文档任务 3b）。命令均为本机实际用过的写法；Windows Git Bash 环境。

## 1. 环境与依赖（本轮未新建 venv、未安装任何依赖）

| 项 | 值 |
|---|---|
| 解释器 | `D:\project_env\python.exe`（在 Git Bash 里写作 `/d/project_env/python.exe`） |
| Python | 3.10.19 |
| torch | `2.9.0+cu128`（`cuda.is_available() = True`，但训练本身 `device="cpu"`） |
| numpy | `2.2.6` |
| 逻辑 CPU | 20 |
| 备选解释器 | `D:\env`（CPU 全依赖），本轮未使用 |
| 依赖安装 | 0；`import torch` 开工即成功（见 `EXECUTION_BASIS.md` §8） |

## 2. 仓库与身份

```bash
cd /d/code/整合/zhenghe
git rev-parse --abbrev-ref HEAD   # fix/comm-task-reward-event-timing
git rev-parse HEAD                # 开工为 577344aab8a5c9bfcf75e7ffbf6f77e503dbc200（ROUND35 交付头）
git merge-base --is-ancestor c7d47d25e638cbe3728cf36e3a22b9cfc563a353 HEAD && echo ROUND33-baseline-ok
```

训练代码所在目录（ROUND24 目录，含 ROUND34 就地修复；本轮只读）：
`runs/comm_cost_v2_rerun_20260928/` → `v2_common.py` `v2_run.py` `v2_trainer.py`
`v2_reward_timing.py` `v2_train.py` `v2_controller.py` `v2_check.py`。

冻结权重（搜索/分配，必须原样在位）：
`runs/cur_v2rep05_s2/best/shared_mappo_model_21_3100.pth`、
`runs/joint_eval_20260916/ctde_final_7_1.pth`。

## 3. 冻结配置在哪读

| 内容 | 位置 |
|---|---|
| 训练常量（seed、128 集、4 集/批、1000 步、B0=10000、δ=0、λ₀=1、η=0.05、γ=1、λ_GAE=0.95、CLIP 0.2、VALUE 0.5、熵 0.01、LR 3e-4、grad-norm 0.5、epoch 4、minibatch 256 步、1e-6 对齐门、里程碑 0/64/128） | `v2_common.py:41-96` |
| 场景块与 128 集顺序 | `v2_common.TRAINING_SCENARIOS` / `scenario_order()`（`DATA_ORDER_SEED=20260927`） |
| 世界与传输字段（预算 29300 B、投递 0.7、ACK 结构关闭、range 0.3、gossip 2、steps 1000、num_goals 6、device cpu） | `v2_common.STAGE_B_INHERITED_OVERRIDES` + `stage_b_inherited_config()` |
| 本轮冻结对照清单（与 ROUND32 逐项比对结果） | `runs/comm_final_train_20261001/FROZEN_CONFIG_MANIFEST.json` |
| 门禁表 | `runs/comm_final_train_20261001/ROUND36_PREFLIGHT.json` |
| 版本字段定义 | `v2_reward_timing.py:10-11` |

## 4. 启动前的核验（0 环境启动）

```bash
cd /d/code/整合/zhenghe/runs/comm_final_train_20261001
/d/project_env/python.exe r36_preflight.py
# 期望输出：12 项 gates 全 true；G10 要求账本为空且 seed 目录内没有 .pt
```

## 5. 三次训练启动（每次 1 个 seed，均从登记初值）

```bash
cd /d/code/整合/zhenghe/runs/comm_final_train_20261001
/d/project_env/python.exe r36_launch.py --seed 11     # 上限 2270.28 s
/d/project_env/python.exe r36_launch.py --seed 22     # 上限 2395.66 s
/d/project_env/python.exe r36_launch.py --seed 33     # 上限 2281.95 s
```

`r36_launch.py` 是受保护的父进程：核对额度与起点目录 → 按 ROUND32 归档同 seed 实际耗时的 1.5
倍设 wall-clock 上限并被子进程超时强制执行 → 把 stdout 整段落盘 → 按真实 ledger 记账。
真正的训练在 `r36_worker.py` 里，它只做路径重绑定并调用真实 `v2_train.train_seed(seed,
episodes=128, ...)`；起点目录里若已有 `.pt` 会直接拒绝启动（防热启动）。

seed 集合固定为 `(11, 22, 33)`（`v2_common.TRAIN_SEEDS`，与 ROUND32 manifest 相同），
不据中间结果更换、增删或挑选。

失败储备（本轮唯一一次，且必须先有登记原因）：

```bash
/d/project_env/python.exe r36_launch.py --seed 22 --reserve   # 写入 seed22_reserve/，仍从初值开始
```

## 6. 每次训练后的一次最小 sanity 前向（0 启动、0 optimizer.step、不出结果数字）

```bash
/d/project_env/python.exe r36_sanity.py --seed 11
```

它用真实 `v2_trainer.load_checkpoint` 恢复 ep128（两个版本字段不齐即被恢复门拒绝）、读回
ep0/ep64/ep128/rolling 的版本字段与字节哈希、对本轮自己的 e0 tape 做一次前向（只断言有限与
形状）、从批记录核对训练量，并写 `SANITY.json`、`CHECKPOINT_HASHES.json`、
`LAMBDA_TRAJECTORY.json`。

## 7. 对账与复核（0 启动）

```bash
/d/project_env/python.exe r36_verify.py        # 逐集事件账本、版本字段、批 0 与归档逐字节比对
/d/project_env/python.exe r36_reconcile.py     # 从 ledger/批记录重算三个 seed 的账本对账表
```

## 8. Git 内外产物

| 内容 | 位置 | 是否进版本库 |
|---|---|---|
| 本轮脚本、门禁、对账 JSON、报告 | `runs/comm_final_train_20261001/*.{py,md,json}` + `delivery_draft/*.md` + `seed*/*.json` | 是 |
| 原始 lifecycle / transcript / tape | `seed<NN>/logs/*.jsonl` | 否（`.gitignore` 白名单之外） |
| 每集 row JSON（含逐事件账本） | `seed<NN>/rows/*.json` | 否 |
| 权重与批记录 | `seed<NN>/checkpoints/*.pt`、`v2_batch_*.json` | 否 |
| 训练 stdout 日志 | `seed<NN>/launch_console.log` | 否 |

它们的**名字与 SHA256** 登记在 `seed<NN>/CHECKPOINT_HASHES.json` 与
`delivery_draft/ARTEFACT_HASH_REGISTRY.json`，所以 Git 外的原件可被逐字节校验。

## 9. 复现一台新机器时的检查点

1. 本地完整分支必须在（GitHub 快照不含完整历史，训练不接受快照）；
2. `import torch` 必须先成功且记录版本，失败即停止报告，不装依赖；
3. 两个冻结 `.pth` 必须在位且哈希相符（G2/G6）；
4. `r36_preflight.py` 的 G6（生产件与导入件哈希 vs ROUND32）与 G7（函数级源码比对）必须全清——
   这两项就是在机器上验证"只有奖励时序被改过"的那道闸；
5. 参照缓存（ROUND24 的 64 份 immediate reference 行）必须覆盖 48 个训练场景，缺任何一个都
   不能开训（G8）。
