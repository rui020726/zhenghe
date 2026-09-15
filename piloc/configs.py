# dqn network setting
hidden_dim = 256


action_dim = 4 # 上下左右四个动作

# 优化器
actor_lr = 1e-5
critic_lr = 1e-5

# curriculum learning
init_env_settings = (1, 60)
max_num_agents = 5
max_map_lenght = 60
pass_rate = 0.8

# 时间戳地图大小
map_size = 60

# 智能体个数
num_agents = 2

# 超参数
MAX_EPISODES = 120000
GAMMA = 0.95
TAU = 0.01
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
HORIZON = 256
CLIP_PARAM = 0.2
K_EPOCHS = 5
ENTROPY_COEFFICIENT = 0.01
MINIBATCH_SIZE = 256
LMBDA = 0.95
HUBER_DELTA = 10.0
DEVICE = 'cuda'  # 或者 'cpu'
ENV_NAME = 'simple_spread_v3'  # 或者您自己定义的环境
TRICK = {
    'adv_norm': True,   # v2 baseline 开启 advantage 归一，避免训练对奖励绝对尺度过度敏感
    'ObsNorm': False,
    'reward_norm': False,
    'reward_scaling': False,
    'orthogonal_init': False,
    'adam_eps': False,
    'lr_decay': False,
    'ValueClip': False,
    'huber_loss': False,
    'LayerNorm': False,
    'feature_norm': False
}

# 旧版（legacy）奖励中实际被读取的键。move / stay_on_goal / stay_off_goal
# 从未被任何代码读取，属于历史遗留，已删除；需要旧值请查 git 历史。
reward_fn = dict(collision= -3 ,
                finish= 3 )

# === 奖励模式（见 REWARD_AND_TRAINING.md）===
# legacy: 旧公式对照；v2: 新的默认训练奖励；potential: 势函数覆盖塑形对照。
reward_mode = "v2"

# 回合长度上限：时间成本按 1/T_max 归一，使单步奖励幅度与课程阶段无关。
T_MAX = 250

# V2 baseline 权重（整数格子计数；见 REWARD_AND_TRAINING.md 与 reward_modes.py）。
# 旧版按比例归一的 coverage=10 在 60x60 上每步只有 ~0.013 的绝对信号，比 legacy
# 小约两个数量级，且被信息素负偏置（~ -0.0128）抵消——改回整数计数。
# 这是首轮标定起点，不是最终最优参数：pheromone/time_cost/repeat 全部归零，
# 一次只验证核心覆盖闭环。
REWARD_V2 = dict(
    target=5.0,           # 团队共享：发现一个目标
    marginal_coverage=0.2,# 观察者分摊的整数边际新格（个体信用）
    union_coverage=0.05,  # 团队并集整数新格（共享，不重复计数）
    finish=20.0,          # 团队共享：全部目标完成
    collision=1.0,        # 个体碰撞成本
    no_new_cell=0.05,     # 本步团队零新增时的惩罚，推动离开已探索区
    time_cost=0.0,        # 默认关闭：四个动作全合法时它不产生动作区分
    repeat=0.5,           # 短周期打转惩罚（10 步内同位置出现 3 次的 agent 该步 -0.5）：
                          # γ 配对复测显示 v2 的 repeat 率是 legacy 的 9 倍（0.27 vs 0.03），
                          # 且与覆盖落后 ~13% 同时出现；0.5 与单步探索收益（~+0.5）同量级但不压倒
    pheromone=0.0,        # 默认关闭：环境先衰减再 +1 的更新顺序使其系统性偏负
)

# potential 模式的势函数系数：F = beta * (gamma * Phi_next - Phi_current)，
# Phi = 团队并集覆盖率。gamma 必须与 MAPPO 使用的折扣一致（训练时由 --gamma 传入）。
POTENTIAL_BETA = 1.0
# 视野
obs_radius = 5

# 目标个数
num_goals = 6

# 通信模块
num_comm_layers = 2
num_comm_heads = 2
max_comm_agents = 3
hidden_dim = 256

# 信息素衰减
pheromone_decay = 0.95