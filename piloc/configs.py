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

reward_fn = dict(move= -0.75 ,
                stay_on_goal=0,
                stay_off_goal=-0.075,
                collision= -3 ,
                finish= 3 )
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