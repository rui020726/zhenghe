# PILOC 奖励与训练工作流

本文档描述搜索模块（`piloc/`）的三种奖励模式、课程门控、随机性设置，
以及如何运行测试、正式训练、验证与断点恢复。

## 1. 奖励模式

由 `--reward-mode` 选择，实现位于
[`timemap.py`](file:///D:/code/整合/zhenghe/piloc/src/Map/timemap.py) 与
[`reward_modes.py`](file:///D:/code/整合/zhenghe/piloc/src/Map/reward_modes.py)。
所有权重集中在 [`configs.py`](file:///D:/code/整合/zhenghe/piloc/configs.py)
（`REWARD_V2`、`T_MAX`、`POTENTIAL_BETA`），不再散落在环境代码里。

### 1.1 `legacy` —— 旧实验对照（默认关闭）

逐位保留旧公式，用于与历史 checkpoint（model_21 系列）对照：

- 每步每 agent 隐式 `-1`（reExploration 基线）；
- 已探索区域计数 ×0.01 的正再探索奖励（含 `nx != x and ny != y` 修复后的行/列统计，
  该修复只影响 legacy 的旧格统计，绝不进入 V2）；
- 队友邻近 `-v` 惩罚；
- 碰撞 `-3`、finish `+3`（finish 用 `update()` **覆盖**本步其他奖励——已知缺陷，
  为对照保真而保留）；
- 信息素 `0.1*(v_-v)/v_ + 0.1*(v_-v)`（无保护，可能爆炸）；
- 中间目标发现奖励恒为 0（先删目标再算奖励——已知缺陷，为对照保真而保留）。

### 1.2 `v2` —— 新的默认训练奖励

```text
C_t          = 团队并集覆盖率（见 §1.4）
N_t          = 本步首次被任意机器人观测到的唯一目标数（两人同看一个目标只计一次）
T_max        = 250

team_reward  = 1.0 * N_t
             + 10.0 * (C_{t+1} - C_t)
             + 3.0 * all_targets_found
             - 1.5 / T_max
             + 0.3 * pheromone_term   # 走低信息素新区域为正，clip ±1；见下

agent_reward = team_reward - 0.4 * collision_i
```

默认权重（`configs.REWARD_V2`）：`target=1.0, coverage=10.0, finish=3.0,
time_cost=1.5, collision=0.4, repeat=0.0, pheromone=0.3`。
（2026-09-13 起 collision 由 0.2 调至 1.0：0.2 过弱，E=10 随机探索期间碰撞率
平均 0.31 且 5000 回合不降，给不了策略足够的避碰梯度。
2026-09-14 起 coverage 由 2.0 调至 10.0 并开启 pheromone 0.3：60×60 地图每步
覆盖增量仅 0.01-0.02，×2.0 后每步 0.02-0.04，信号淹没在 PPO critic 噪声里，
策略在 E=30 学会原地打转（验证 travel=58/60 但 recoverage=0.96）；放大覆盖
信号并恢复 PILOC 的**信息素探索引导**——`0.1*(v_-v)/max(v_,eps) + 0.1*(v_-v)`
走低信息素新区域为正、重复踩旧区域为负，weight=0.3 时单步 ±0.3，量级足以
驱动策略继续向外探索（信息素带 epsilon 保护与 clip，`pheromone_epsilon=1e-6`、
`pheromone_clip=1.0`）。
同轮再调：collision 1.0→0.4、time_cost 0.5→1.5。best checkpoint 实测轨迹
（40 步只踩 5 个不同格、发现 0 目标）证明策略收敛到“出生点微振荡”：
密度 0.15~0.3（三角分布，本例 0.248）下走出去的期望 `-1.0×P(撞墙)+探索收益`
净负，而“不动”每步仅 -0.002 近免费，PPO 理性地选择不动。0.4 使探索期望转正，
1.5/250=-0.006/步给“不动”征税。）

与 legacy 的本质区别：

- **删除**每步隐式 `-1`：回合变长不再线性塌陷累计奖励；
- **删除**队友邻近 `-v` 惩罚（靠近队友同时是地图融合条件，惩罚与机制自相矛盾）；
- **不再**给"看到大量旧格子"正奖励——只有并集覆盖**增量**计酬，
  两个机器人重复看同一格不会重复计团队奖励；
- 中间目标发现**立即**获得 `target` 奖励（修复先删目标再算奖励的死路径），
  全部完成时 `finish` 是**累加**而不是覆盖；
- `done` 只有一个语义：全部目标完成。单个机器人看到目标不再置 per-agent done；
- 时间成本按 `1/T_max` 归一：单个发现事件的奖励幅度与当前课程阶段 E 无关；
- `repeat` 默认权重 0（保留参数但默认关闭）；
- `pheromone` 默认权重 0；启用时环境侧分母有 epsilon 保护（`1e-6`），
  奖励侧做有限值检查并裁剪到 ±1。

团队搜索收益（发现、覆盖、finish、时间成本）共享给所有 agent；
碰撞成本按 agent 单独计算。

### 1.3 `potential` —— 势函数覆盖塑形对照

在 V2 基础上，把覆盖增量项替换为势函数塑形：

```text
Phi   = 团队并集覆盖率 C_t
F     = beta * (gamma * Phi_{t+1} - Phi_t)     （beta = configs.POTENTIAL_BETA = 1.0）
```

- 使用与 MAPPO 相同的 `--gamma`（由训练循环传入环境）；
- **真实终止**（全部目标找到）时 `Phi_next := 0`；
- **时间上限截断**不是终止：critic bootstrap 保留（`done_bool` 只对真实终止为真），
  势函数不强制归零；
- 公式级单元测试在 `test_piloc_reward_v2.PotentialModeTests`；
- 注意：塑形使回报对回合长度的依赖**减弱**但不完全无关，不要按"与回合长度无关"解读。

### 1.4 coverage 的准确口径

`team_union_coverage`（`timemap.Environment`）与 `train.SearchEpisodeMetrics`
完全同口径：**任一**机器人时间戳观测 ≠ 0.5（未知）的自由格数 ÷ 全部自由格数
（障碍 = `map == 1` 之外的格）。V2 的覆盖增量、potential 的 Phi、课程门控的
`coverage` 全部使用这一个定义，不存在第二套覆盖率。

`coverage_auc` = 回合内每步覆盖率的平均值（越早覆盖得分越高）。

## 2. 课程门控（`src/curriculum.py`）

旧规则"历史最高累计奖励 + 200 回合未刷新即升级"已删除。新规则：

- 阶段阶梯：回合步数上限 E 从 10 到 250（每级 +10），`--curriculum gate`（默认）；
- 每 `--eval-every`（默认 50）个训练回合，在**固定验证 seed**
  （默认 `seed + 1_000_000_007`，与训练流隔离且永不被训练消耗）上跑
  `--validation-episodes`（默认 3）个**确定性动作（argmax）**回合；
- 门控输入**只允许**任务指标：`coverage, coverage_auc, discovery_rate,
  collision_rate, repeat_rate`。传入 reward/return 直接 `ValueError`；
- 默认阈值：`coverage ≥ 0.5, discovery_rate ≥ 4/6` 描述**最终阶段**（250 步）
  的能力，并按各阶段预算占最终预算的比例线性缩放——阶段 k 的门槛是
  `门槛 × E_k / 250`。stage 0（E=10）只要求 `coverage ≥ 0.02`、
  `discovery_rate ≥ 4/150 ≈ 0.027`（验证时至少发现 1 个目标）；
  `collision_rate ≤ 0.5, repeat_rate ≤ 0.9` 是每步比率，与回合长度无关，
  不缩放。collision 门槛特意设得宽松（0.5）：早期随机策略撞墙/出界是正常
  现象（实测平均碰撞率 0.31），避碰是奖励层（V2 的 collision 权重）该管的
  事；repeat 门槛放宽到 0.9 同理——E=10 短回合下 argmax 验证 repeat 恒
  0.7（起点附近无探索空间，策略打转，几何上降不到 0.5 以下），且 V2 的
  repeat 权重本就是 0（不惩罚重复）。课程只管 E 阶梯。背景：首轮 v2 训练
  5000 回合验证 coverage 平台仅 0.06、discovery 1/6，旧的同阈值设置在 E=10
  几何上不可达，课程永远卡在 stage 0；将 coverage/discovery 按预算缩放后，
  剩余卡点只剩 collision≤0.05 与 repeat≤0.5 这两个奖励层指标，故一并放宽；
- 连续 `--required-passes`（默认 3）次全部达标才升一级；任何一次失败清零计数；
- **长期不达标永不自动升级**；升级后窗口重置；
- `--curriculum off` 关闭课程，使用 `--fixed-episode-limit`（默认 250）；
- 最优 checkpoint 按验证任务指标
  （`validation_score = coverage_auc + discovery_rate - collision_rate - 0.5*recoverage_rate`）
  保存到 `<output-dir>/best/`，不按训练累计回报。

说明：旧代码中的 E 本来就只控制回合步数上限，与地图/目标数无关；
新实现保持这一点并明确记录在 `run_meta.json`。

## 3. γ 与随机性

- `--gamma` 暴露为 CLI（默认 `configs.GAMMA=0.95`）。**V2 推荐 0.99**：
  回合上限 250 步、γ=0.95 时有效视野约 1/(1-γ)=20 步，远小于任务时间尺度；
  γ=0.99 的约 100 步仍偏短，但配合逐步的稀疏发现奖励已可传播。
  legacy 对照实验可继续用 0.95。
- `seed_all()` 统一设置 Python `random`、NumPy、Torch CPU、Torch CUDA
  与每个 agent 的 `action_space(agent).seed(...)`。
- 环境拥有**独立 RNG**（`_py_rng`/`_np_rng`），步内随机（视野噪声、目标移动、
  通信噪声）不再与训练循环交织；`reset_piloc_environment` 经 `reseed()` 重放。
- 限制：策略模块设置了 `torch.backends.cudnn.benchmark = True`，
  CUDA 上卷积核选择可能非确定——CPU 上同 seed 可逐位复现（有回归测试），
  GPU 上只保证统计可复现。`run_meta.json` 记录 `cuda_deterministic: false` 与设备。

## 4. 日志与 checkpoint

输出目录 `--output-dir`（默认 `runs/piloc_search`）：

```text
<output-dir>/
├── run_meta.json          # 全部配置：gamma、lmbda、reward_mode、权重、门控、seed、设备
├── episode_metrics.jsonl  # 每 log-every 回合：return、mean reward/step、length、
│                          # coverage、coverage AUC、发现数、collision/repeat/recoverage 率、
│                          # （--track-reward-terms 时）逐分量奖励分解
├── checkpoints/           # 周期性 checkpoint（--save-every）
├── best/                  # 验证任务指标最优 checkpoint
└── tb_logs/               # TensorBoard：search/*、validation/*、reward_term/*
```

每个 checkpoint 内嵌 `run_meta`（gamma、lmbda、reward_mode、全部权重、seed、device）
与 `episode`，加载端 `resume_policy` 会读取 `episode`。

## 5. 如何运行

```bash
# 单元测试（仓库根目录）
python -m unittest discover -s integrated_sar/tests -v

# CPU 短程 smoke（几分钟内完成，验证闭环）
cd piloc
python train.py --device cpu --reward-mode v2 --gamma 0.99 --seed 0 \
    --episodes 3 --eval-every 2 --required-passes 1 --validation-episodes 1 \
    --log-every 1 --no-tensorboard --output-dir ../runs/smoke_v2

# 正式训练（V2 推荐命令）
python train.py --device auto --reward-mode v2 --gamma 0.99 --seed 0 \
    --episodes 5000 --save-every 500 --log-every 50 --eval-every 50 \
    --required-passes 3 --validation-episodes 3 \
    --track-reward-terms --output-dir runs/piloc_v2_s0

# legacy 对照（配对 seed）
python train.py --device auto --reward-mode legacy --gamma 0.95 --seed 0 \
    --episodes 5000 --save-every 500 --log-every 50 --eval-every 50 \
    --output-dir runs/piloc_legacy_s0

# 断点恢复
python train.py --device auto --reward-mode v2 --gamma 0.99 --seed 0 \
    --episodes 5000 --resume runs/piloc_v2_s0/checkpoints/shared_mappo_model_21_2000.pth \
    --output-dir runs/piloc_v2_s0

# 用产出权重做联合推理验证（仓库根目录）
python -m integrated_sar.joint_driver --piloc-checkpoint runs/piloc_v2_s0/best \
    --steps 200 --seed 7
```

## 6. 已知限制

- 每回合训练地图仍随机生成（`DungeonMaps` 资产不在仓库内），
  验证集是"固定 seed 的固定地图"而不是独立留出的真实地图集；
- V2 权重是初始默认值，未经大规模配对实验调参；
- `repeat`/`pheromone` 保留为默认关闭的实验项；
- rescue 启发式**不进入训练**（会破坏动作与 log-prob 的一致性），
  仅在 `piloc/test.py` 的评估路径保留。
