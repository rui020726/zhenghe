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

### 1.2 `v2` —— 新的默认训练奖励（baseline 标定版）

```text
U_t          = 本步团队并集新覆盖的整数自由格数
M_i          = 本步 agent i 的边际新格数（观察者分摊：同一新格被 n 个
               机器人同时看到，每个获得 1/n，既不重复计数又保留个体信用）
N_t          = 本步首次被任意机器人观测到的唯一目标数（两人同看一个目标只计一次）

team_reward  = 5.0 * N_t
             + 0.05 * U_t
             + 20.0 * all_targets_found
             - 0.05 * I(U_t == 0)          # 本步团队零新增：推动离开已探索区

agent_reward = team_reward
             + 0.20 * M_i                   # 个体边际覆盖信用
             - 1.0  * collision_i
```

默认权重（`configs.REWARD_V2`）：`target=5.0, marginal_coverage=0.2,
union_coverage=0.05, finish=20.0, collision=1.0, no_new_cell=0.05,
time_cost=0.0, repeat=0.0, pheromone=0.0`。

（2026-09-14 诊断：旧 v2 把覆盖增量按比例归一（coverage=10 × ΔC），60×60
地图上每步团队新增仅 ~4 格，ΔC≈0.0013，奖励 +0.013——比 legacy 的探索信号
小约两个数量级，且被信息素负偏置（环境先衰减再 +1，实测均值 −0.0128）与
碰撞期望（−0.054）抵消，随机 rollout 每 agent-step 总奖励 −0.04，策略学到
"原地撞墙"比探索更优。故改回**整数格子计数**，并设 `pheromone=0,
time_cost=0, repeat=0`——一次只验证核心覆盖闭环。`time_cost` 在固定长度
回合里对四个动作完全相同，不产生动作区分；pheromone 的更新顺序使其系统
性偏负，详见 git 历史。新权重下随机 rollout 每 agent-step 总奖励 +0.51
（marginal 覆盖 +0.36、碰撞 −0.14），探索期望显著为正。）

与 legacy 的本质区别：

- **删除**每步隐式 `-1`：回合变长不再线性塌陷累计奖励；
- **删除**队友邻近 `-v` 惩罚（靠近队友同时是地图融合条件，惩罚与机制自相矛盾）；
- 覆盖只计**整数新格增量**：两个机器人重复看同一格不重复计团队奖励，
  同时按观察者分摊给个体边际信用 `M_i`（解决团队共享奖励的信用分配问题）；
- 中间目标发现**立即**获得 `target` 奖励（修复先删目标再算奖励的死路径），
  全部完成时 `finish` 是**累加**而不是覆盖；
- `done` 只有一个语义：全部目标完成。单个机器人看到目标不再置 per-agent done；
- `repeat` / `pheromone` / `time_cost` 默认权重 0（保留参数但默认关闭，
  标定通过后再逐项开启）；
- 启用 pheromone 时环境侧分母有 epsilon 保护（`1e-6`），
  奖励侧做有限值检查并裁剪（注意：裁剪在乘权重**之后**进行，
  实际最大幅度由 `pheromone_clip` 控制）。

团队搜索收益（发现、并集覆盖、finish、零新增惩罚）共享给所有 agent；
边际覆盖与碰撞成本按 agent 单独计算。

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
- 注意：塑形使回报对回合长度的依赖**减弱**但不完全无关，不要按"与回合长度无关"解读；
- **暂缓消融**：count reward 是 O(1) 整数量级，potential 用覆盖比例通常只有
  O(10⁻³–10⁻²)，`beta=1` 会让 potential 重新落入弱密集信号问题。比较前须先把
  势函数换算成格子数单位（或校准 beta ≈ 10³），否则结果不可比。

### 1.4 coverage 的准确口径

`team_union_coverage`（`timemap.Environment`）与 `train.SearchEpisodeMetrics`
完全同口径：**任一**机器人时间戳观测 ≠ 0.5（未知）的自由格数 ÷ 全部自由格数
（障碍 = `map == 1` 之外的格）。V2 的整数新格数 `U_t`、边际信用 `M_i`、
potential 的 Phi、课程门控的 `coverage_gain` 全部从同一套布尔掩码算出，
不存在第二套覆盖率。

- `U_t`：本步 `after_seen & ~before_seen` 的整数计数（一步只算一次掩码，
  覆盖率、新格数、边际信用同口径）；
- `M_i`：每个新格被 n 个机器人同时看到时，每个观察者获得 `1/n`。
  归因基于**通信合并前**的本地 FOV 快照（`_local_obs_snapshot`）：只在
  合并后的地图上看到某格（收到队友广播）而本机从未实际观测的 agent
  不分摊信用；
- `coverage_auc` = 回合内每步覆盖率的平均值（越早覆盖得分越高）；
- `coverage_gain` = `C_T - C_0`（回合末覆盖率减回合初覆盖率）：
  初始视野覆盖率实测 ~0.044，不计入"新增"，避免"出生位置好"被误认为会探索。

## 2. 课程门控（`src/curriculum.py`）

旧规则"历史最高累计奖励 + 200 回合未刷新即升级"已删除。新规则：

- 阶段阶梯：回合步数上限 E 从 **50** 到 250（每级 +25），`--curriculum gate`（默认）。
  起点 E=50：E=10 时回合太短，指标主要由初始视野决定（随机策略 coverage 已达
  0.068），无法区分策略好坏；E=50 给随机策略的 coverage_gain 约 0.1，
  给可学习策略留出明确上升空间；
- 每 `--eval-every`（默认 50）个训练回合，在**固定验证 seed**
  （默认 `seed + 1_000_000_007`，与训练流隔离且永不被训练消耗）上跑
  `--validation-episodes`（默认 **20**）个**确定性动作（argmax）**回合，
  取平均任务指标。20 个固定 seed 而非 3 张图：验证方差大时 3 个 seed 的
  均值会把运气误当能力；
- 门控输入**只允许**任务指标：`coverage, coverage_gain, coverage_auc,
  discovery_rate, collision_rate, repeat_rate`。传入 reward/return 直接
  `ValueError`（门控不读累计回报是接口约束，不是约定）；
- 门控阈值：`coverage_gain` 按各阶段预算占最终预算的比例线性缩放——
  阶段 k 的门槛是 `0.5 × E_k / 250`。E=50 只要求 coverage_gain ≥ 0.1；
  `repeat_rate ≤ 0.9` 是每步比率，与回合长度无关，不缩放。
  **不门控 collision_rate**：合法动作 mask 上线后碰撞率恒为 0，它已不再
  衡量"学会避障"（`metrics_pass` 的下限优机制仍支持显式配置该指标，仅供
  部署一致性检查使用）。**不门控 discovery_rate**：E 较短时它主要由目标
  出生位置运气决定，且 discovery 已由验证分数与最优 checkpoint 选择覆盖。
  背景：首轮 v2 训练 5000 回合验证 coverage 平台仅 0.06，旧同阈值设置在
  E=10 几何上不可达，课程永远卡在 stage 0；E=10 的原始随机策略
  coverage=0.068、collision=0.375 已优于旧门槛，门槛只是暴露了策略坍塌，
  而非主因；
- 连续 `--required-passes`（默认 3）次全部达标才升一级；任何一次失败清零计数；
- **长期不达标永不自动升级**；升级后窗口重置；
- `--curriculum off` 关闭课程，使用 `--fixed-episode-limit`（默认 250）。
  **课程关闭时评估节奏仍然有效**：仍按 `eval_every` 跑确定性验证与随机基线
  早停，只是不升级——`--curriculum off` 的对照实验同样需要验证与早停；
- 最优 checkpoint 按验证任务指标
  （`validation_score = coverage_auc + discovery_rate - collision_rate - 0.5*recoverage_rate`）
  保存到 `<output-dir>/best/`，不按训练累计回报。

### 2.1 合法动作 mask（五条路径共用同一实现）

边界与相邻障碍始终处于局部视野内，掩码不引入真值泄漏。掩码（1=合法，
0=越界/障碍；四向全不合法时返回全 1，由碰撞惩罚处理）是
`timemap.legal_action_mask` 这**一个**函数，被五条路径共用——动作分布
在任何路径上都必须一致，否则非法方向（训练时几乎无梯度）可能在部署时
成为最大 logit 重新撞墙：

- 训练动作采样（`train.select_action`：概率过掩码归一后再 `Categorical.sample`）；
- 确定性验证（`train.select_action_deterministic`：argmax 只在合法动作上取，
  内部只调 `get_probs`，不触碰 torch 采样 RNG）；
- PPO 更新重算 log-prob（buffer 逐条保存掩码，新旧策略分布一致，ratio 无偏）；
- 评估脚本 `piloc/test.py`（`select_action`：采样前过同一掩码）；
- 集成 SAR runtime（`integrated_sar.piloc_runtime.TrainedPilocPolicy`：
  确定性部署推理，`argmax` 只在掩码后的概率上取）。

回归测试 `test_piloc_mask_and_resume.py` 锁住：即使非法方向持有最高原始
概率，上述每条路径最终也必须选出合法动作（含负对照：掩码失效时 argmax
确实会撞墙，证明测试有区分力）。

### 2.2 随机基线与早停

每个验证点在同批固定验证 seed 上跑两个均匀随机 rollout 基线（独立
`np.random.RandomState`，绝不触碰训练 RNG）：

- **raw-random**：4 个动作均匀采样（含碰撞，下界参照）；
- **legal-random**：只在合法动作上均匀采样（无碰撞上界参照）。

早停（默认开启，`--no-early-stop` 关闭）：episode ≥ `--early-stop-window`
（默认 500）后，若确定性验证 coverage < legal-random − `--early-stop-margin`
（默认 0.05），计一次失败；**连续 `--early-stop-patience`（默认 3）次失败**
才判定策略已坍塌并停止（单次验证的方差足以误杀仍在线性改善的策略）。
只与 legal-random 比较：raw-random 含非法动作，与带掩码的策略不可比。
触发后落盘 trainer state（含 `early_stop_failures`），不再跑满 5000 回合
浪费算力。

### 2.3 断点恢复（checkpoint 合一）

**权威副本内嵌在模型 checkpoint 本体**：`MAPPO.save` 把
`trainer_state`（episode、global step、最优验证分、早停标记与失败计数、
课程门控状态、Python/NumPy/Torch CPU/Torch CUDA 的 RNG 状态、**PPO buffer
全量数组**）与权重、optimizer、`run_meta` 写进同一个 `.pth`。
`--resume` 走 `resume_policy`：严格加载权重与 optimizer，取出内嵌状态，
并做 **episode 一致性校验**——权重轮次与内嵌 trainer state 的
`episode_num` 不一致直接拒绝（旧的旁路 pkl 就是这么错位的）。

细节约束：

- **PPO buffer 必须全量存取**：`learn()` 通过 `all()` 消费全部 capacity 行，
  连续训练中"未填充"行保留上一窗口的残留数据并参与梯度更新（learn 后的
  `clear` 只重置索引，不清数组内容）。只存前 `_size` 行会把残留行清零，
  恢复后的梯度更新与连续训练分岔（N+M ≠ N+resume M）。
- **恢复后按回合边界规则重建环境**：训练循环在每个回合边界用
  `seed + episode_num` 重建环境并重播动作空间种子，resume 的第一个回合
  也必须遵守同一规则，否则恢复后的第一张地图与连续训练不一致。
- 旧格式回退：checkpoint 没有内嵌 trainer state 时才读旁路
  `trainer_state.pkl`（两者分开覆盖、无法校验同源——可用但不保证同轮）。
- 回归测试 `test_piloc_mask_and_resume.ResumeEquivalenceTests` 锁住
  **N+M ≡ N+resume M**：连续训练 N+M 轮与"训练 N 轮后恢复再训练 M 轮"，
  learn 之后的网络参数逐位一致（同 HORIZON=256 约束，含 Adam 状态、
  buffer 残留窗口与 RNG 同源）。

## 3. γ 与随机性

- `--gamma` 暴露为 CLI（默认 `configs.GAMMA=0.95`）。**V2 推荐 0.99**：
  回合上限 250 步、γ=0.95 时有效视野约 1/(1-γ)=20 步，远小于任务时间尺度；
  γ=0.99 的约 100 步仍偏短，但配合逐步的稀疏发现奖励已可传播。
  legacy 对照实验可继续用 0.95。
- `TRICK['adv_norm'] = True`：v2 baseline 开启 advantage 归一，避免训练对
  奖励绝对尺度过度敏感。
- PPO 信用分配：一个 agent 的 ratio 只与它自己的 advantage 相乘（旧实现把
  (mb,1) 的 ratio 与 (mb,num_agents) 的 adv 广播相乘，个体碰撞信号被队友
  advantage 稀释，碰撞学不到）。
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
├── trainer_state.pkl      # 旁路副本（旧格式回退与人工检查用；权威副本在 checkpoint 内）
├── episode_metrics.jsonl  # 每 log-every 回合：return、mean reward/step、length、
│                          # coverage、coverage AUC、发现数、collision/repeat/recoverage 率、
│                          # （--track-reward-terms 时）逐分量奖励分解
├── checkpoints/           # 周期性 checkpoint（--save-every）
├── best/                  # 验证任务指标最优 checkpoint
└── tb_logs/               # TensorBoard：search/*、validation/*、reward_term/*、baseline/*
```

每个 checkpoint 内嵌 `run_meta`（gamma、lmbda、reward_mode、全部权重、seed、
device）、`episode` 与 `trainer_state`（课程/RNG/global step/最优分/早停计数/
PPO buffer 全量数组）。buffer 全量数组约 15MB/agent——这是"恢复后梯度更新
与连续训练逐位一致"的必要代价。加载端 `resume_policy` 读取 `episode` 并校验
与内嵌 trainer state 的轮次一致。

## 5. 如何运行

```bash
# 单元测试（仓库根目录）
python -m unittest discover -s integrated_sar/tests -v

# CPU 短程 smoke（几分钟内完成，验证闭环）
cd piloc
python train.py --device cpu --reward-mode v2 --gamma 0.99 --seed 0 \
    --episodes 3 --eval-every 2 --required-passes 1 --validation-episodes 1 \
    --log-every 1 --no-tensorboard --output-dir ../runs/smoke_v2

# 正式训练（V2 推荐命令：课程 gate，E 从 50 起步）
python train.py --device auto --reward-mode v2 --gamma 0.99 --seed 0 \
    --episodes 5000 --save-every 500 --log-every 50 --eval-every 50 \
    --required-passes 3 --validation-episodes 20 \
    --track-reward-terms --output-dir runs/piloc_v2_s0

# 第三步对照实验（绕开课程验证奖励本身；每组 ≥3 seed × 300 回合先跑）
# 判定规则：legacy 能学、新 v2 不能学 → 查 PPO；两者都不能学 → 不能再归咎奖励
python train.py --device auto --reward-mode legacy --gamma 0.95 --seed 0 \
    --curriculum off --fixed-episode-limit 50 \
    --episodes 300 --save-every 300 --log-every 50 --eval-every 50 \
    --validation-episodes 20 --output-dir runs/abl_legacy_s0_e50
python train.py --device auto --reward-mode v2 --gamma 0.99 --seed 0 \
    --curriculum off --fixed-episode-limit 50 \
    --episodes 300 --save-every 300 --log-every 50 --eval-every 50 \
    --validation-episodes 20 --track-reward-terms \
    --output-dir runs/abl_v2_s0_e50

# legacy 课程对照（配对 seed）
python train.py --device auto --reward-mode legacy --gamma 0.95 --seed 0 \
    --episodes 5000 --save-every 500 --log-every 50 --eval-every 50 \
    --validation-episodes 20 --output-dir runs/piloc_legacy_s0

# 断点恢复（网络 + trainer_state.pkl 全量状态）
python train.py --device auto --reward-mode v2 --gamma 0.99 --seed 0 \
    --episodes 5000 --resume runs/piloc_v2_s0/checkpoints/shared_mappo_model_21_2000.pth \
    --output-dir runs/piloc_v2_s0

# 用产出权重做联合推理验证（仓库根目录）
python -m integrated_sar.joint_driver --piloc-checkpoint runs/piloc_v2_s0/best \
    --steps 200 --seed 7
```

## 6. E=50 对照实验结果（2026-09-15）

第三步消融：`--curriculum off`、固定 E=50、每 50 回合 20 个固定验证 seed
确定性验证 + raw/legal-random 基线、300 回合 × 3 seed。运行目录
`runs/abl_{legacy,v2}_s{0,1,2}_e50`，每组约 20 分钟（CPU 单线程）。

**300 回合最终验证 coverage（验证 seed 均值）**：

| 组别 | seed 0 | seed 1 | seed 2 | 均值 | raw-random | legal-random |
|---|---:|---:|---:|---:|---:|---:|
| legacy (γ=0.95) | 0.214 | 0.217 | 0.243 | **0.225** | 0.110 | 0.120 |
| v2 (γ=0.99) | 0.174 | 0.201 | 0.207 | **0.194** | 0.110 | 0.119 |

coverage_gain（C_T−C_0）均值：legacy 0.183，v2 0.153。discovery_rate 均值：
legacy 0.20，v2 0.17（6 个目标）。无任何早停触发；全部 6 组全程在随机基线之上。

判定：

- **两组都能学**：300 回合内验证 coverage 单调爬升，最终高于 raw-random
  约 75–120%、高于 legal-random 约 50–95%。旧 v2（归一化 coverage +
  pheromone 负偏置）"学得比随机差"的坍塌模式没有复现——修复生效。
- **legacy 当前略优于 v2**（0.225 vs 0.194）。注意两组 γ 不同（0.95 vs 0.99，
  v2 按 §3 推荐），且 300 回合偏短，差距的归因需 γ 配对实验确认；
  v2 seed 0 在 ep 150 后回退（0.200→0.174），稳定性弱于 legacy。
- 旧 v2（当前 v2 对照组）未重跑：用户诊断已给出其坍塌证据
  （训练 2250+ 回合后 coverage 0.055 < raw-random 0.0685、collision 0.867），
  与本仓库随机 rollout 标定（每 agent-step 奖励 −0.0401，legacy +0.87）一致。
- v2 奖励分解（`--track-reward-terms`）符合设计：边际信用按个体分化
  （末回合 agent_0 23.8 vs agent_1 61.0），碰撞项全程为 0（合法 mask
  生效），pheromone/time_cost/repeat 恒 0（标定期关闭）。

### 6.1 γ 配对复测（2026-09-15，四路径一致性修复后）

为消除 0.225 vs 0.194 的 γ 混杂，legacy 与 v2 都用 γ=0.99、同 seed
（0/1/2）、同 20 个验证 seed 重跑 E=50（`runs/abl_{legacy,v2}_g99_s{0,1,2}_e50`）。
本轮代码含边际信用本地 FOV 归因修正，故 v2 数字与 §6 不逐位相同，但均值不变。

**300 回合最终验证 coverage**：

| 组别 | seed 0 | seed 1 | seed 2 | 均值 | legal-random | repeat 率 |
|---|---:|---:|---:|---:|---:|---:|
| legacy (γ=0.99) | 0.251 | 0.170 | 0.247 | **0.223** | 0.119 | 0.03 |
| v2 (γ=0.99) | 0.240 | 0.167 | 0.175 | **0.194** | 0.119 | 0.27 |

coverage_gain：legacy 0.191，v2 0.162。discovery_rate 两组相同（0.111）。

判定：

- **γ 混杂排除**：legacy 对 γ 不敏感（0.95→0.99 仅 0.225→0.223），v2 落后
  ~13% 的差距在配对后依旧存在。当前结论仍是"v2 已恢复学习"（高于
  legal-random 约 63%），不能说"v2 优于 legacy"。
- **v2 的新线索：重复访问率高一个数量级**（0.27 vs 0.03）。legacy 的隐式
  每步成本 + 再探索正奖励天然抑制重访；v2 标定期 `repeat=0`，没有对应
  项。下一步若开课程长训，优先考虑给 v2 开一个小幅 repeat 惩罚再验证，
  而不是直接上 5000 回合。

### 6.2 repeat 惩罚生效（2026-09-15，v2 + repeat=0.5）

按 §6.1 线索把 `REWARD_V2['repeat']` 从 0 开到 **0.5**（短周期打转检测：
10 步内同位置出现 3 次的 agent 该步 -0.5，与单步探索收益 ~+0.5 同量级），
同 γ=0.99、同 seed、同验证集重跑 E=50
（`runs/abl_v2rep05_g99_s{0,1,2}_e50`）：

| 组别 | coverage 均值 | coverage_gain | repeat 率 |
|---|---:|---:|---:|
| v2 (repeat=0) | 0.194 | 0.162 | 0.27 |
| **v2 (repeat=0.5)** | **0.216** | **0.184** | **0.07** |
| legacy (参照) | 0.223 | 0.191 | 0.03 |

判定：

- **假设成立**：repeat 率 0.27→0.07（-74%），coverage +11%、
  coverage_gain +14%；与 legacy 的差距从 13% 缩到 3%（在 seed 方差内）。
  v2 覆盖落后的主因确认是重复访问。
- **无副作用**：全部 6 组跑满 300 回合无早停；discovery 不受影响
  （s0 两轮都是 0.000，由目标出生位置运气决定，见 §2 门控注释）。
- repeat=0.5 成为 v2 的新默认权重（commit `4568281`）。
  v2 与 legacy 在 E=50 上统计打平，且 v2 没有 legacy 的已知缺陷
  （finish 覆盖、中间发现恒 0、信息素无保护）——进入课程长训。

后续（外部评审建议的执行顺序，2026-09-15 更新进度）：

1. ✅ 四条路径一致性（mask 五路共用 + checkpoint 合一 + N+M ≡ N+resume M
   回归测试，commit `314ebbb`）；
2. ✅ γ 配对 E=50 复测（见 §6.1，γ 混杂排除，差距真实）；
3. ✅ 单种子课程 smoke test（E 连续晋级 50→75→stage 2，509s）；
4. ⬜ 3–5 种子完整课程实验（建议先给 v2 开小幅 repeat 惩罚再跑）；
5. ⬜ centralized critic、potential reward 等结构性改进
   （potential 需先按 §1.3 校准 beta）。

## 7. 已知限制

- 每回合训练地图仍随机生成（`DungeonMaps` 资产不在仓库内），
  验证集是"固定 seed 的固定地图"而不是独立留出的真实地图集；
- V2 baseline 权重是首轮标定起点，不是最终最优参数；pheromone/time_cost/
  repeat 归零，标定通过后再逐项开启；
- `repeat`/`pheromone` 保留为默认关闭的实验项；
- rescue 启发式**不进入训练**（会破坏动作与 log-prob 的一致性），
  仅在 `piloc/test.py` 的评估路径保留。
