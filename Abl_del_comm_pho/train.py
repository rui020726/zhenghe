import time
import numpy as np
import torch
from Agent import MAPPO
import configs
from timemap import Environment
from torch.distributions import Categorical
import random
from torch.utils.tensorboard import SummaryWriter

def repeat_check(repeat_mark,env,obs,rescue_policy):
    all_paths = {}
    if len(repeat_mark) != 0:
        for agent_id in repeat_mark:
            agent_index = int(agent_id.split('_')[1])  # 提取数字部分并转换为整数
            path = rescue_policy.rescue_action(agent_index,env)
            # 将路径保存到字典中，键是智能体的ID，值是智能体的路径
            all_paths[agent_id] = path
    return all_paths

def select_action(two_channelmap, policy, env,env_agents, device):
    """选择动作，使用Softmax"""
    actions = {}
    action_log_pis = {}

    for agent_id in env_agents:
        # 获取agent的状态，假设状态是60x60的二维数组
        obstacle_map = two_channelmap[agent_id]['obstacle_map']  # 获取障碍物地图
        explore_map = two_channelmap[agent_id]['explore_map']  # 获取探索地图
        agent_intid = int(agent_id.split("_")[-1])
        x,y = env.agents_pos[agent_intid]
        obs = np.stack([obstacle_map, explore_map], axis=0)  # 生成一个3通道的状态
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # 转换为tensor，并将其移动到指定的device
        # .unsqueeze(0) 这将状态从 [60, 60] 变为 [1, 1, 60, 60]，即批次大小为1，通道数为1

        # 获取动作概率
        action_probs = policy.agents.get_action(obs_tensor,batch_size=1)
        action_probs = action_probs[1]  # 提取第二个张量
        # 使用Softmax选择动作
        dist = Categorical(probs=action_probs)
        action = dist.sample()  # 使用sample()进行采样，Softmax已隐含

        # 计算对应的log概率
        action_log_pi = dist.log_prob(action)

        actions[agent_id] = action.item()  # 获取动作的实际值
        action_log_pis[agent_id] = action_log_pi.item()  # 获取log概率

    return actions, action_log_pis

def initialize_obs(env,num_agents = configs.num_agents, map_size=(60, 60)):
    """
    初始化每个智能体的观测为 60x60 的地图，初始值为 0.5

    num_agents: 智能体数量
    map_size: 地图的尺寸，默认60x60

    return: 一个字典，其中每个智能体有一个60x60的obs地图，值为0.5
    """
    obs = {}

    for i in range(num_agents):

        # 初始化每个智能体的obs地图，尺寸为60x60，所有元素初始为0.5
        agent_obs = np.full(map_size, 0.5, dtype=np.float32)
        x,y = env.agents_pos[i] # 当前智能体的初始位置
        agent_obs[x,y] = -4.5
        for dx in range(-5, 6):  # x轴方向从-1到1
            for dy in range(-5, 6):  # y轴方向从-1到1
                nx, ny = x + dx, y + dy  # 计算邻域坐标
                # 判断是否越界，确保新的坐标在地图范围内
                if 0 <= nx < env.map_size[0] and 0 <= ny < env.map_size[1]:
                    # 如果该位置是障碍物，则将其标记为1
                    if env.map[nx, ny] == 1:
                        agent_obs[nx, ny] = 1  # 障碍物位置标记为1
                    # 更新非障碍物的时间戳，排除已被标记的智能体位置
                    elif [nx, ny] != [x, y] and env.map[nx, ny] != 1 and agent_obs[nx, ny] not in [-4.5, -5.5,
                                                                                                   -6.5]:
                        agent_obs[nx, ny] = 0  # 不是障碍物的位置标记为0

        # 将初始化的obs地图赋值给对应的智能体
        obs[f'agent_{i}'] = agent_obs

    return obs

def main():
    # 使用configs中的超参数
    env_name = configs.ENV_NAME
    max_episodes = configs.MAX_EPISODES
    gamma = configs.GAMMA
    tau = configs.TAU
    actor_lr = configs.ACTOR_LR
    critic_lr = configs.CRITIC_LR
    horizon = configs.HORIZON
    clip_param = configs.CLIP_PARAM
    k_epochs = configs.K_EPOCHS
    minibatch_size = configs.MINIBATCH_SIZE
    lmbda = configs.LMBDA
    huber_delta = configs.HUBER_DELTA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trick = configs.TRICK
    writer = SummaryWriter('runs/experiment')
    # 初始化环境，这里假设你使用自己的环境（替换成你自己的环境代码）
    env = Environment()  # 请替换为您的自定义环境类
    dim_info = {f'agent_{i}': [121, 4] for i in range(configs.num_agents)}  # 这里面包含了初始化几个智能体，现在修改为所有智能体使用相同的参数，但是经验存储是分开的

    # 初始化算法
    policy = MAPPO(dim_info, actor_lr=actor_lr, critic_lr=critic_lr,device=device,horizon=horizon,trick=trick)


    # 训练配置
    model_dir1= r'E:/Desktop/代码/修改6--在修改5的基础上，所有网络使用相同的参数/path_to_save_model_Abl'  # 保存模型的路径
    writer = SummaryWriter(log_dir='runs/experiment_1')

    episode_num = 0
    step = 0 # 记录总步数
    step_t = 0 # 记录当前回合的步数
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}
    obs = initialize_obs(env)
    two_channelmap = {}
    two_channelmap = env.two_channel_map(obs,two_channelmap)
    {agent: env.action_space(agent).seed(seed=100) for agent in env_agents}  # 给每个智能体的动作空间设置种子
    rescue_actions = {}
    reward_max_M = -float('inf')  # 初始化为很小的负数，便于更新第一回合的奖励
    t = 10
    Curriculum_Training = 0 # 用于记录有多少个回合智能体的奖励函数没有超过R_max
    E = 10 # 用于随机生成环境步数,从40步开始训练
    step_reward_t = 0
    # 初始化rescue算法用于训练
    #rescue_policy = RescueAlgorithm(env.map_size, env.agents_pos, env,obs)
    episode_reward_agent = 0 # 回合奖励初始化为0
    obs1 = {}
    obs2 = {}
    next_obs1 = {}
    next_obs2 = {}


    # 训练循环
    while E <= 250 :
        step += 1
        step_t +=1
        step_reward_t += 1
        # 获取动作
        action, action_log_pi = select_action(two_channelmap, policy, env,env_agents, device)
        for agentid,position in enumerate(env.agents_pos):
            x,y = position
            agent_id = f"agent_{agentid}"

        # 环境更新
        next_obs, reward, terminated, repeat_Mark,next_two_channelmap= env.step(action,obs,rescue_actions,two_channelmap,E)
        for agentid,position in enumerate(env.agents_pos):
            x,y = position
            agent_id = f"agent_{agentid}"

        #for agent_id in env_agents:
            #writer.add_scalar(f'step_reward_{agent_id}', reward[agent_id],step_reward_t)
        # 更新obs
        done = {agent_id: (terminated[agent_id] or step_t >= t) for agent_id in env_agents}
        done_bool = {agent_id: terminated[agent_id] for agent_id in env_agents}

        for agentid in two_channelmap.keys():
            obs1[agentid] = two_channelmap[agentid]['obstacle_map']
            obs2[agentid] = two_channelmap[agentid]['explore_map']
        for agentid in next_two_channelmap.keys():
            next_obs1[agentid] = next_two_channelmap[agentid]['obstacle_map']
            next_obs2[agentid] = next_two_channelmap[agentid]['explore_map']

        # 保存数据
        policy.add(obs1,obs2, action, reward, next_obs1,next_obs2, done_bool, action_log_pi, done)

        episode_reward = {agent_id: episode_reward[agent_id] + reward[agent_id] for agent_id in env_agents}
        obs = next_obs
        two_channelmap = next_two_channelmap
        # 完成目标 或者 达到最大环境步 ，这一个回合结束
        if any(done.values()) or step_t >= t:
            # 显示信息
            if (episode_num + 1) % 100 == 0:
                print(f"Episode {episode_num + 1} | Reward: {episode_reward}")
                for agent_id in env_agents:
                    train_return[agent_id].append(episode_reward[agent_id])
            for agent_id, reward in episode_reward.items():
                # 为每个智能体生成唯一的标签
                writer.add_scalar(f'train_episode_reward/{agent_id}', reward, episode_num)
            print(E)
            print(Curriculum_Training)
            episode_num += 1
            env = Environment() # 重置环境
            obs = initialize_obs(env)  # 重置时间戳地图

            # 用于计算该回合奖励
            for agent_reward in episode_reward.values() :
                episode_reward_agent += agent_reward
            # 如果r_max更新，重新计算有多少轮没有更新
            if episode_reward_agent > reward_max_M:
                reward_max_M = episode_reward_agent
                Curriculum_Training = 0 # r_max更新时将其置0

            episode_reward = {agent_id: 0 for agent_id in env_agents}
            episode_reward_agent = 0
            step_t = 0
            t = random.randint(E-5, min(E+5, 250))
            Curriculum_Training += 1 # 用于记录r_max有没有变化
            # 如果200个回合没有r_max更新，则增加E，增加环境时间步
            if Curriculum_Training >= 150:
                E += 10
                reward_max_M = -float('inf')
                Curriculum_Training = 0 #

        # 每隔一定步数更新网络
        if step % horizon == 0:
            policy.learn(minibatch_size, gamma, lmbda, clip_param, k_epochs, entropy_coefficient=0.01, huber_delta=huber_delta)

        # 保存模型
        if episode_num % 1000 == 0:
            policy.save(model_dir1,episode_num)

    print(f"Training completed in {time.time() - start_time} seconds")
    policy.save(model_dir1,episode_num)

    writer.close()
    # 保存训练结果
    train_return_ = np.array([train_return[agent_id] for agent_id in env.agents])
    np.save(f"{model_dir1}/train_return.npy", train_return_)

if __name__ == '__main__':
    start_time = time.time()
    main()


