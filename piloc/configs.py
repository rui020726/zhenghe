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
    'adv_norm': False,
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

# V2 权重。team_reward = target*N_t + coverage*(C_t+1 - C_t)
#                      + finish*all_found - time_cost/T_MAX
# agent_reward = team_reward - collision*collision_i
REWARD_V2 = dict(
    target=1.0,
    coverage=10.0,        # 2.0 过弱：60x60 每步覆盖增量 0.01-0.02，信号淹没在 critic 噪声里，策略学会原地打转
    finish=3.0,
    time_cost=0.5,
    collision=1.0,        # 每 agent 单独计；0.2 过弱，早期策略撞墙/出界率 31% 不降
    repeat=0.0,           # 默认关闭：V2 不用短周期惩罚
    pheromone=0.3,        # 恢复 PILOC 探索引导：走低信息素新区域得正奖励，推动策略走出起点
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

TRICK = {'adv_norm':False,'ObsNorm':False,'reward_norm':False,'reward_scaling':False,    # or
         'orthogonal_init':False,'adam_eps':False,'lr_decay':False, # 原代码中设置为False
          # 以上均在PPO_with_tricks.py中实现过
          'ValueClip':False,'huber_loss':False,
          'LayerNorm':False,'feature_norm':False,
          }

# 信息素衰减
pheromone_decay = 0.95