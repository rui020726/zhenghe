import random
from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces

plt.ion()
from matplotlib import colors
import configs
from collections import Counter
import os
import random
from PIL import Image
import numpy as np
from collections import Counter
import time

from src.Map.reward_modes import (
    RewardWeights,
    StepEvents,
    compute_potential_rewards,
    compute_v2_rewards,
)

action_list = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)

color_map = np.array([[255, 255, 255],  # white
                      [190, 190, 190],  # gray
                      [0, 191, 255],  # blue
                      [255, 165, 0],  # orange
                      [0, 250, 154]])  # green


def map_partition(map):
    '''
    将地图划分为相互独立的分区（寻找连通的空白区域）
    '''
    # 获取所有空白位置
    empty_pos = np.argwhere(map == 0).astype(np.int32).tolist()

    empty_pos = [tuple(pos) for pos in empty_pos]

    if not empty_pos:
        raise RuntimeError('no empty position')

    partition_list = list()  # 用来存储所有独立的区域
    while empty_pos:

        start_pos = empty_pos.pop()

        open_list = list()  # 用于寻找连通区域
        open_list.append(start_pos)
        close_list = list()
        # 深度优先搜索
        while open_list:
            x, y = open_list.pop(0)

            up = x - 1, y
            if up[0] >= 0 and map[up] == 0 and up in empty_pos:
                empty_pos.remove(up)
                open_list.append(up)

            down = x + 1, y
            if down[0] < map.shape[0] and map[down] == 0 and down in empty_pos:
                empty_pos.remove(down)
                open_list.append(down)

            left = x, y - 1
            if left[1] >= 0 and map[left] == 0 and left in empty_pos:
                empty_pos.remove(left)
                open_list.append(left)

            right = x, y + 1
            if right[1] < map.shape[1] and map[right] == 0 and right in empty_pos:
                empty_pos.remove(right)
                open_list.append(right)

            close_list.append((x, y))

        partition_list.append(close_list)  # 表示已经访问过这个区域

    return partition_list


def _add_reward_term(terms, agent_id, name, value):
    """Record one reward term for diagnostics; no-op when tracking is off.

    ``terms`` is None unless ``track_reward_terms`` is enabled, so the call sites in
    ``step`` stay one line each. The recorded terms must add up to the reward that
    step returns; ``test_piloc_reward_terms`` is what holds that invariant.
    """
    if terms is None:
        return
    bucket = terms[agent_id]
    bucket[name] = bucket.get(name, 0.0) + float(value)


class Environment:
    def __init__(self, num_agents: int = configs.num_agents, map_length: int = configs.init_env_settings[1],
                 obs_radius: int = configs.obs_radius, reward_fn: dict = configs.reward_fn, fix_density=None,
                 curriculum=False, init_env_settings_set=configs.init_env_settings, num_goals=configs.num_goals,
                 pheromone_decay=configs.pheromone_decay, test=False,
                 # === 新增：通信条件参数 ===
                 comm_failure_prob=0.05, comm_partial_loss_prob=0.10, comm_delay_prob=0.05, enable_comm_noise=True,
                 map_filename=None, track_reward_terms: bool = False,
                 reward_mode: str = configs.reward_mode, reward_weights: dict = None,
                 gamma: float = configs.GAMMA, t_max: int = configs.T_MAX,
                 potential_beta: float = configs.POTENTIAL_BETA, seed: int = None):
        """
        新增参数:
        comm_failure_prob: 通信完全失败概率
        comm_partial_loss_prob: 通信部分丢失概率
        comm_delay_prob: 通信延迟概率
        enable_comm_noise: 是否启用通信噪声
        reward_mode: legacy（旧公式对照）| v2（新的默认训练奖励）| potential（覆盖势函数塑形）
        reward_weights: 覆盖 configs.REWARD_V2 的权重字典（None 用默认值）
        gamma: 与 MAPPO 相同的折扣，potential 模式塑形项使用
        t_max: 回合长度上限，V2 时间成本按 1/t_max 归一
        potential_beta: 势函数系数
        """
        if reward_mode not in ("legacy", "v2", "potential"):
            raise ValueError(f"unknown reward_mode {reward_mode!r}; expected legacy, v2 or potential")
        self.reward_mode = reward_mode
        self._reward_weights = RewardWeights.from_config(
            reward_weights if reward_weights is not None else configs.REWARD_V2,
            t_max=t_max,
            gamma=gamma,
            potential_beta=potential_beta,
        )
        # 独立环境 RNG：seed 给定时完全由 seed 决定；未给定时从当前全局状态
        # 派生一个初始状态（保持“先播种全局再构造”的旧调用仍然可复现）。
        # 环境的步内随机（视野噪声、目标移动、通信噪声）不再与训练循环的
        # 全局抽取交织，固定验证 seed 也不会被训练过程消耗。
        if seed is None:
            seed = random.randrange(2 ** 63)
            np_seed = int(np.random.randint(2 ** 31 - 1))
        else:
            np_seed = int(seed) % (2 ** 31 - 1)
        self._py_rng = random.Random(seed)
        self._np_rng = np.random.RandomState(np_seed)
        # 最近一步的结构化事件（碰撞、发现、覆盖前后值），供训练循环分别统计
        # collision / 短周期循环 / 重复覆盖，而不是全部混进 repeat。
        self.last_step_stats = {}
        self.num_goals = num_goals  # 目标个数
        self.timestamp_map = np.full((60, 60), 0.5, dtype=np.float32)  # 初始时间戳地图
        self.agents = [f'agent_{i}' for i in range(num_agents)]  # 智能体列表
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.agents}  # 每个智能体有4个离散动作
        self.curriculum = curriculum
        self.check_repeat_pos = []
        # 逐项奖励分解日志：默认关闭。开启后 step() 把每步奖励拆成各项存进
        # last_reward_terms，各分量之和恒等于返回给训练循环的奖励值。
        self.track_reward_terms = track_reward_terms
        self.last_reward_terms = {}
        self._reexploration_parts = {}

        self.global_pheromone = np.full((60, 60), 0, dtype=np.float32)  # 初始化全局的信息素地图
        self.pheromone_decay = pheromone_decay  # 信息素挥发率
        self.local_pheromone = {}
        self.noise_sigma = 0.5
        self.size = 60

        # === 新增：通信条件配置 ===
        self.comm_failure_prob = comm_failure_prob
        self.comm_partial_loss_prob = comm_partial_loss_prob
        self.comm_delay_prob = comm_delay_prob
        self.enable_comm_noise = enable_comm_noise

        # 通信统计
        self.comm_stats = {'success': 0, 'failed': 0, 'partial': 0, 'delayed': 0, 'total': 0}
        self.delayed_map_buffer = []

        if curriculum:
            self.env_set = [init_env_settings_set]
            self.num_agents = num_agents
            self.map_size = (init_env_settings_set[1], init_env_settings_set[1])
        else:
            self.num_agents = num_agents
            self.map_size = (map_length, map_length)

        # set as same as in PRIMAL:是否随机生成一个障碍物密度
        if fix_density is None:
            self.fix_density = False
            self.obstacle_density = self._np_rng.triangular(0, 0.15, 0.3)
        else:
            self.fix_density = True
            self.obstacle_density = fix_density

        # 原逻辑保持不变...
        # self.map 必须在这里就有值：下面 map_partition(self.map) 无条件读取它，而 test 分支
        # 只在 test=True 时赋值，默认 test=False 会直接 AttributeError。生成方式与 reset() 一致，
        # test=True 时随后会被图片覆盖。
        self.map = self._np_rng.choice(
            2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]
        ).astype(np.int32)

        if test:
            map_dir = 'DungeonMaps/test'
            if map_filename is not None:
                image_path = os.path.join(map_dir, map_filename)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Specified test map not found: {image_path}")
                image = Image.open(image_path).convert('L')
                image_resized = image.resize((60, 60))
                map_data = np.array(image_resized)
                map_data = (map_data < 128).astype(np.int32)
                self.map = map_data
            else:
                # 保留原随机选图逻辑
                image_files = [f'img_{i}.png' for i in range(6000, 10000)]
                while True:
                    random_image = self._py_rng.choice(image_files)
                    image_path = os.path.join(map_dir, random_image)
                    if os.path.exists(image_path):
                        image = Image.open(image_path).convert('L')
                        image_resized = image.resize((60, 60))
                        map_data = np.array(image_resized)
                        map_data = (map_data < 128).astype(np.int32)
                        self.map = map_data
                        break

        partition_list = map_partition(self.map)
        partition_list = [partition for partition in partition_list if len(partition) >= 2]  # 过滤掉空白位置少于2的区域
        # 没有符合条件则重新生成
        while len(partition_list) == 0:
            self.map = self._np_rng.choice(2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]).astype(
                np.int32)
            partition_list = map_partition(self.map)
            partition_list = [partition for partition in partition_list if len(partition) >= 2]

        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int32)
        self.goals_pos = np.empty((self.num_goals, 2), dtype=np.int32)
        self.goals_pos.fill(-1)  # 用-1表示空位置

        max_list = max(partition_list, key=len)  # 获取面积最大的连通区域

        # 循环为每个智能体分配其初始位置
        for i in range(self.num_agents):
            pos = self._py_rng.choice(max_list)
            max_list.remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=np.int32)

        # 循环设置目标位置
        for i in range(self.num_goals):
            pos = self._py_rng.choice(max_list)
            max_list.remove(pos)
            # 找到goals_pos中的空位置，并将选中的pos添加到该位置
            empty_index = np.where(self.goals_pos == -1)  # 查找goals_pos中为-1的空位置
            self.goals_pos[empty_index[0][0]] = pos  # 更新空位置为选中的目标位置

        self.obs_radius = obs_radius
        self.global_timestamp_maps = np.full((60, 60), 0.5, dtype=np.float32)  # 初始化一个全局的时间戳地图，用于计算奖励
        for i, (x, y) in enumerate(self.agents_pos):
            self.global_pheromone[x, y] = 1  # 将智能体初始位置的信息素标1
            self.local_pheromone[f"agent_{i}"] = self.calculate_pheromone(x, y)
        self.reward_fn = reward_fn
        self.steps = 0

        # 记录最近10次的动作轨迹
        self.agent_positions_history = {f"agent_{i}": self.agents_pos[i] for i in range(num_agents)}  # 用来记录智能体过去的10个位置
        self.last_actions = np.zeros((self.num_agents, 10), dtype=np.bool_)

    # === 新增：设置通信条件的方法 ===
    def reseed(self, seed: int) -> None:
        """重置环境自有 RNG 流。

        集成层每个 episode 会重新装载场景；步内随机（目标移动、视野噪声、
        通信噪声）现在走环境私有流，所以必须在这里同步重播，
        否则同一 seed 的第二个 episode 与第一个不再一致。
        """
        self._py_rng = random.Random(seed)
        self._np_rng = np.random.RandomState(int(seed) % (2 ** 31 - 1))

    def set_communication_conditions(self, comm_failure_prob=0.05, comm_partial_loss_prob=0.10,
                                     comm_delay_prob=0.05, enable_comm_noise=True):
        """动态设置通信条件"""
        self.comm_failure_prob = comm_failure_prob
        self.comm_partial_loss_prob = comm_partial_loss_prob
        self.comm_delay_prob = comm_delay_prob
        self.enable_comm_noise = enable_comm_noise
        # 重置通信统计
        self.comm_stats = {'success': 0, 'failed': 0, 'partial': 0, 'delayed': 0, 'total': 0}
        self.delayed_map_buffer = []

    # === 新增：获取通信统计的方法 ===
    def get_communication_stats(self):
        """返回当前的通信统计信息"""
        total = self.comm_stats.get('total', 0)
        if total == 0:
            return {
                'success_rate': 0.0,
                'failure_rate': 0.0,
                'partial_rate': 0.0,
                'delay_rate': 0.0,
                'total_attempts': 0
            }

        return {
            'success_rate': self.comm_stats.get('success', 0) / total,
            'failure_rate': self.comm_stats.get('failed', 0) / total,
            'partial_rate': self.comm_stats.get('partial', 0) / total,
            'delay_rate': self.comm_stats.get('delayed', 0) / total,
            'total_attempts': total
        }

    # 将地图转换为双通道地图
    def two_channel_map(self, agent_timestamp_maps: dict, two_channelmap: dict):
        for agent_id, timemap in agent_timestamp_maps.items():
            Obstacle_map = np.full((60, 60), -1, dtype=np.float32)  # 初始化障碍物地图
            explore_map = np.full((60, 60), -0.5, dtype=np.float32)  # 初始化障碍物地图
            for i in range(timemap.shape[0]):
                for j in range(timemap.shape[0]):
                    v = timemap[i, j]
                    if v == 1:
                        Obstacle_map[i, j] = 1
                        explore_map[i, j] = 0
                    elif v == 0.5:
                        explore_map[i, j] = 0.5
                    elif 0 <= v <= 0.3:
                        Obstacle_map[i, j] = 0
                        explore_map[i, j] = v
                    else:
                        Obstacle_map[i, j] = v
                        explore_map[i, j] = v
                    # 将生成的障碍物地图和探索地图存入字典
            two_channelmap[agent_id] = {'obstacle_map': Obstacle_map, 'explore_map': explore_map}
        return two_channelmap

    def update_env_settings_set(self, new_env_settings_set):
        self.env_set = new_env_settings_set

    def action_space(self, agent):
        """
        返回指定智能体的动作空间
        """
        return self.action_spaces.get(agent)

    def reset(self, num_agents=None, map_length=None):

        if self.curriculum:
            rand = self._py_rng.choice(self.env_set)
            self.num_agents = rand[0]
            self.map_size = (rand[1], rand[1])

        elif num_agents is not None and map_length is not None:
            self.num_agents = num_agents
            self.map_size = (map_length, map_length)

        if not self.fix_density:
            self.obstacle_density = self._np_rng.triangular(0, 0.33, 0.5)

        self.map = self._np_rng.choice(2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]).astype(
            np.float32)

        partition_list = map_partition(self.map)
        partition_list = [partition for partition in partition_list if len(partition) >= 2]

        while len(partition_list) == 0:
            self.map = self._np_rng.choice(2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]).astype(
                np.float32)
            partition_list = map_partition(self.map)
            partition_list = [partition for partition in partition_list if len(partition) >= 2]

        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int32)
        self.goals_pos = np.empty((self.num_goals, 2), dtype=np.int32)
        self.goals_pos.fill(-1)  # 用-1表示空位置

        max_list = max(partition_list, key=len)  # 获取面积最大的连通区域

        # 循环为每个智能体分配其初始位置
        for i in range(self.num_agents):
            pos = self._py_rng.choice(max_list)
            max_list.remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=np.int32)

        # 循环设置目标位置
        for i in range(self.num_goals):
            pos = self._py_rng.choice(max_list)
            max_list.remove(pos)
            # 找到goals_pos中的空位置，并将选中的pos添加到该位置
            empty_index = np.where(self.goals_pos == -1)  # 查找goals_pos中为-1的空位置
            self.goals_pos[empty_index[0][0]] = pos  # 更新空位置为选中的目标位置

        self.steps = 0
        # 初始化每个智能体的位置历史
        self.agent_positions_history = {
            f"agent_{i}": np.empty((0, 2), dtype=np.int32)  # 初始化为空数组，形状为 (0, 2)，表示没有历史位置
            for i in range(self.num_agents)
        }
        self.agent_positions_history = {f"agent_{i}": self.agents_pos[i] for i in range(num_agents)}  # 用来记录智能体过去的10个位置
        self.last_actions = np.zeros((self.num_agents, 10), dtype=np.bool_)

        # === 重置通信统计 ===
        self.comm_stats = {'success': 0, 'failed': 0, 'partial': 0, 'delayed': 0, 'total': 0}
        self.delayed_map_buffer = []

        return self.map

    def load(self, map: np.ndarray, agents_pos: np.ndarray, goals_pos: np.ndarray):

        self.map = np.copy(map)
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)

        self.num_agents = agents_pos.shape[0]
        self.map_size = (self.map.shape[0], self.map.shape[1])

        self.steps = 0

        self.imgs = []

        self.last_actions = np.zeros((self.num_agents, 5, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
                                     dtype=np.bool_)

    def save_data(self, agent_id, agent_map, global_map, save_dir):
        """保存智能体的时间戳地图和全局地图"""

        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = time.strftime("%Y%m%d-%H%M%S")  # 使用时间戳作为文件名的一部分
        # 保存智能体的时间戳地图
        np.save(os.path.join(save_dir, f"agent_{agent_id}_timestamp_map_{timestamp}.npy"), agent_map)

        # 保存全局地图
        np.save(os.path.join(save_dir, f"global_map_{timestamp}.npy"), global_map)

    def update_timestamp_map(self, agent_timestamp_maps: dict, two_channelmap: dict):
        """根据环境中的智能体位置更新时间戳地图"""
        agent_invision = []
        agents_positions = self.get_agents_positions()  # 获取所有智能体的位置
        reExploration_reward_find_target = 0  # 当前视野范围内存在的目标数量
        env_map = self.map  # 环境中的地图，包含障碍物信息
        updated_to_zero_count_dict = {}
        reExploration_reward_dict = {}
        rewards = {}
        # 更新每个智能体的时间戳地图
        for agent_id, agent_map in agent_timestamp_maps.items():
            # 获取该智能体的位置
            agent_pos = agents_positions[int(agent_id.split('_')[1])]
            x, y = agent_pos
            reExploration_reward = 0
            updated_to_zero_count_1 = 0  # 用于统计值从0.5变为0的次数
            updated_to_zero_count_2 = 0  # 用于统计值从0.5变为0的次数

            vision_radius = 5 if self._py_rng.random() > 0.05 else 4  # 引入干扰，智能体获得观测半径有5%的概率变小为4

            # 先将探索过的时间戳地图更新
            for i in range(agent_map.shape[0]):  # 遍历行
                for j in range(agent_map.shape[1]):  # 遍历列
                    value = agent_map[i, j]
                    if value == 5.5:  # 上一时刻的其他智能体位置修改为0.01
                        agent_map[i, j] = 0.003
                    elif value == -4.5:  # 上一时刻当前智能体的位置
                        agent_map[i, j] = -5.5
                    elif value == -5.5:  # 前两个时刻当前智能体的位置
                        agent_map[i, j] = -6.5
                    elif value == -6.5:  # 最多记录三个位置
                        agent_map[i, j] = 0.03
                    elif 0 <= value < 0.3:  # 更新旧的时间戳地图
                        agent_map[i, j] += 0.003
                        updated_to_zero_count_1 += 1
                    elif value == 0.3:
                        updated_to_zero_count_1 += 1  # 统计探索了多少

            # 获得再次探索的奖励
            for dx in range(-vision_radius, vision_radius + 1):  # x轴方向从-1到1
                for dy in range(-vision_radius, vision_radius + 1):  # y轴方向从-1到1
                    nx, ny = x + dx, y + dy  # 计算邻域坐标
                    # 判断是否越界，确保新的坐标在地图范围内
                    if 0 <= nx < self.map_size[0] and 0 <= ny < self.map_size[1]:
                        # 修复：原条件 `nx != x and ny != y` 把中心所在的整行整列
                        # 都排除了；意图只是排除中心格本身。
                        # 注意：修复后的旧格统计只服务 legacy 对照，绝不进入
                        # V2/potential 的正奖励（那两个模式完全不读这里）。
                        if (nx != x or ny != y) and env_map[nx, ny] != 1 and agent_map[nx, ny] not in [-4.5,
                                                                                                      -5.5,
                                                                                                      -6.5]:
                            v = agent_map[nx, ny]
                            if 0 <= v <= 0.3:
                                reExploration_reward += 1 + v / 0.3  # 获得再次探索的奖励
            # 分解日志：把“已探索格子计数×0.01”与隐藏的无条件 -1 基线分开记录，
            # 这是定位“累计奖励随回合变长而塌陷”的关键（每步每 agent 恒扣 1）。
            if self.track_reward_terms:
                self._reexploration_parts[agent_id] = (reExploration_reward * 0.01, -1.0)
            reExploration_reward = reExploration_reward * 0.01 - 1
            reExploration_reward_dict[agent_id] = reExploration_reward
            # 目标发现奖励已上移到 step()：legacy 在删除目标前无法在这里看到任何目标
            # （删除判据严格覆盖这里的判据），旧代码中此分支恒不触发且
            # reExploration_reward_find_target 跨 agent 泄漏，属于死代码，已移除。
            # 更新当前智能体
            for dx in range(-vision_radius, vision_radius + 1):  # x轴方向从-1到1
                for dy in range(-vision_radius, vision_radius + 1):  # y轴方向从-1到1
                    nx, ny = x + dx, y + dy  # 计算邻域坐标
                    # 判断是否越界，确保新的坐标在地图范围内
                    if 0 <= nx < self.map_size[0] and 0 <= ny < self.map_size[1]:
                        if self._is_line_of_sight_blocked(x, y, nx, ny, env_map):
                            # 如果该位置是障碍物，则将其标记为1
                            if env_map[nx, ny] == 1:
                                agent_map[nx, ny] = 1  # 障碍物位置标记为1
                            # 更新非障碍物的时间戳，排除已被标记的智能体位置
                            elif [nx, ny] != [x, y] and env_map[nx, ny] != 1 and agent_map[nx, ny] not in [-4.5, -5.5,
                                                                                                           -6.5]:
                                agent_map[nx, ny] = 0  # 不是障碍物的位置标记为0
            agent_map[x, y] = -4.5

            # 统计还有多少未探索（用于计算当前智能体的探索奖励）
            for i in range(agent_map.shape[0]):  # 遍历行
                for j in range(agent_map.shape[1]):  # 遍历列
                    value = agent_map[i, j]
                    if 0 <= value <= 0.3:
                        updated_to_zero_count_2 += 1  # 统计探索了多少

            updated_to_zero_count_dict[agent_id] = updated_to_zero_count_2 - updated_to_zero_count_1

            # 更新回字典中
            agent_timestamp_maps[agent_id] = agent_map  # 更新该智能体的时间戳地图
        agent_invision = self.find_agents_in_vision(agents_positions)  # 得到智能体视野内有哪些智能体

        for agentid, agent_in_vision in enumerate(agent_invision):
            agentid = f"agent_{agentid}"
            v = len(agent_in_vision)
            rewards[agentid] = -v  # 有多少智能体受到多少惩罚
            if len(agent_in_vision) != 0:  # 如果视野内存在智能体，则更新时间戳地图
                map1 = agent_timestamp_maps[agentid]
                for i in agent_in_vision:
                    x, y = agents_positions[i]  # 获得其他智能体的坐标
                    i = f"agent_{i}"
                    map2 = agent_timestamp_maps[i]
                    map1 = self.merge_timestamp_maps(map1, map2)
                    map1[x, y] = 5.5
                agent_timestamp_maps[agentid] = map1
        two_channelmap = self.two_channel_map(agent_timestamp_maps, two_channelmap)  # 更新双通道地图
        return agent_timestamp_maps, updated_to_zero_count_dict, reExploration_reward_dict, two_channelmap, rewards

    # 计算智能体当前视野内的信息素
    def calculate_pheromone(self, x, y):
        value = 0
        for dx in range(- self.obs_radius, self.obs_radius + 1):
            for dy in range(- self.obs_radius, self.obs_radius + 1):
                if 0 <= x + dx <= 59 and 0 <= y + dy <= 59:
                    v = self.global_pheromone[x + dx, y + dy]
                    value += v
        return value

    def _observed_goal_indices(self, positions):
        """给定一组机器人位置，返回本快照中被任意机器人看到的目标下标集合。

        同一目标被两个机器人同时看到只算一次（集合语义），这是“团队发现
        不重复计数”的实现点。判据与旧 step 一致：切比雪夫距离 <= 5 且视线无遮挡。
        """
        found = set()
        for goal_index, goal_pos in enumerate(self.goals_pos):
            gx, gy = int(goal_pos[0]), int(goal_pos[1])
            for agent_pos in positions:
                x, y = int(agent_pos[0]), int(agent_pos[1])
                if abs(x - gx) <= 5 and abs(y - gy) <= 5 and self._is_line_of_sight_blocked(
                    x, y, gx, gy, self.map
                ):
                    found.add(goal_index)
                    break
        return found

    def _union_seen_mask(self, obs_maps) -> np.ndarray:
        """团队并集“已观测”布尔阵（自由格中被任一机器人观测到的）。"""
        obstacles = np.asarray(self.map)
        free = obstacles != 1
        seen = np.zeros(obstacles.shape, dtype=bool)
        for agent_obs in obs_maps.values():
            seen |= (np.asarray(agent_obs) != 0.5) & free
        return seen

    def team_union_coverage(self, obs_maps) -> float:
        """团队并集覆盖率：任一机器人观测到（时间戳 != 0.5）的自由格 / 全部自由格。

        与 ``train.SearchEpisodeMetrics.as_dict`` 的 coverage 完全同口径，
        避免定义第二套覆盖率。
        """
        free_count = int(np.count_nonzero(np.asarray(self.map) != 1))
        if free_count == 0:
            return 0.0
        return float(np.count_nonzero(self._union_seen_mask(obs_maps)) / free_count)

    def step(self, actions: dict, obs: dict, rescue_actions: dict, two_channelmap: dict, E):
        move_options = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
        new_goals_pos = []
        legacy = self.reward_mode == "legacy"

        rewards = {agent_id: 0 for agent_id in actions.keys()}
        terms = {agent_id: {} for agent_id in actions.keys()} if self.track_reward_terms else None
        if self.track_reward_terms:
            self._reexploration_parts = {}
        checking_list = [i for i in range(self.num_agents)]

        # 覆盖率口径：C_t 取本步观测更新前的团队并集，C_{t+1} 取更新后。
        coverage_before = self.team_union_coverage(obs)

        next_pos = np.copy(self.agents_pos)
        self.global_pheromone = self.global_pheromone * self.pheromone_decay  # 每一步信息素都在衰减
        # Step 1: Remove unmoving agent and calculate next positions
        for agent_id, action in actions.items():
            agent_index = int(agent_id.split('_')[1])  # 提取数字部分并转换为整数
            next_pos[agent_index] += action_list[action]

        # Step 2: Boundary check and obstacle collision
        collisions = {agent_id: 0 for agent_id in actions.keys()}
        for agent_id in checking_list.copy():
            agent_key = f"agent_{agent_id}"  # 获取字典中的键，如 'agent_0', 'agent_1' 等
            if np.any(next_pos[agent_id] < 0) or np.any(next_pos[agent_id] >= self.map_size[0]):
                if legacy:
                    rewards[agent_key] += self.reward_fn['collision']
                    _add_reward_term(terms, agent_key, 'collision', self.reward_fn['collision'])
                collisions[agent_key] += 1
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)
            elif self.map[tuple(next_pos[agent_id])] == 1:  # Check for obstacle
                if legacy:
                    rewards[agent_key] += self.reward_fn['collision']
                    _add_reward_term(terms, agent_key, 'collision', self.reward_fn['collision'])
                collisions[agent_key] += 1
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

        # Step 5: Update positions and check if all agents reach the goal
        self.agents_pos = np.copy(next_pos)
        for x, y in self.agents_pos:
            self.global_pheromone[x, y] += 1  # 新位置的信息素加1
            if self.global_pheromone[x, y] > 10:  # 设置一个上限
                self.global_pheromone[x, y] = 10
        pheromone_raw = {}
        for i, (x, y) in enumerate(self.agents_pos):
            v = self.calculate_pheromone(x, y)
            v_ = self.local_pheromone[f"agent_{i}"]  # 获取上一时刻的信息素浓度
            if legacy:
                rewards[f"agent_{i}"] += 0.1 * (v_ - v) / v_ + 0.1 * (v_ - v)  # 奖励设置选用相对变化和绝对变化
                _add_reward_term(terms, f"agent_{i}", 'pheromone_rel', 0.1 * (v_ - v) / v_)
                _add_reward_term(terms, f"agent_{i}", 'pheromone_abs', 0.1 * (v_ - v))
            else:
                # v2/potential：原始差值只记录，权重为 0 时不进入奖励；
                # 分母用 epsilon 保护，避免 v_ 极小时爆炸。
                eps = self._reward_weights.pheromone_epsilon
                pheromone_raw[f"agent_{i}"] = 0.1 * (v_ - v) / max(v_, eps) + 0.1 * (v_ - v)
            self.local_pheromone[f"agent_{i}"] = v  # 更新信息素浓度

        self.steps += 1
        self.record_position()  # 更新位置记录
        self.check_repeat_pos = self.check_all_agents_repeated_positions()  # 检查过去10个位置是否有三个重复的，如果有则返回智能体编号
        repeat_mark = self.check_repeat_pos
        # 处于营救策略中的agent不会再被repeat_mark
        for agent_id, _ in rescue_actions.items():
            if agent_id in repeat_mark:
                repeat_mark.remove(agent_id)
        # 营救策略触发时，给予一个惩罚（仅 legacy；V2 默认 repeat 权重为 0）
        if legacy and repeat_mark:  # 检查 repeat_mark 列表是否为空
            for agent_id in repeat_mark:  # 遍历 repeat_mark 中的每个智能体编号
                if agent_id in rewards:  # 确保 rewards 中有这个智能体的编号
                    rewards[agent_id] -= 0.1  # 将对应智能体的奖励减去 0.1
                    _add_reward_term(terms, agent_id, 'repeat', -0.1)

        # Step 6: Update timestamp map after all actions using the provided `obs`
        obs, Exploration_reward, reExploration_reward, two_channelmap, reward_comm = self.update_timestamp_map(obs,
                                                                                                               two_channelmap)  # Directly use `obs` to update the timestamp map
        coverage_after = self.team_union_coverage(obs)

        if legacy:
            # ---- legacy 对照路径：逐位保留旧行为，包括 finish 覆盖、per-agent done
            #      语义与“先删目标再算 find-target 奖励”（后者使中间发现恒为 0）。----
            done = {agent_id: False for agent_id in actions.keys()}
            for agent_id in actions.keys():
                agent_index = int(agent_id.split('_')[1])
                agent_pos = self.agents_pos[agent_index]
                indices_to_delete = []
                for goal_index, goal_pos in enumerate(self.goals_pos):
                    x = agent_pos[0]
                    y = agent_pos[1]
                    nx = goal_pos[0]
                    ny = goal_pos[1]
                    dx = abs(agent_pos[0] - goal_pos[0])
                    dy = abs(agent_pos[1] - goal_pos[1])
                    if dx <= 5 and dy <= 5 and self._is_line_of_sight_blocked(x, y, nx, ny, self.map):
                        done[agent_id] = True
                        indices_to_delete.append(goal_index)
                self.goals_pos = np.delete(self.goals_pos, indices_to_delete, axis=0)

            all_found = len(self.goals_pos) == 0
            if all_found:
                if terms is not None:
                    for agent_id in actions.keys():
                        _add_reward_term(terms, agent_id, 'finish_dropped_prior', -rewards[agent_id])
                        _add_reward_term(terms, agent_id, 'finish', self.reward_fn['finish'])
                rewards.update({agent_id: self.reward_fn['finish'] for agent_id in actions.keys()})
                done = {agent_id: True for agent_id in actions.keys()}
            else:
                done = {agent_id: False for agent_id in actions.keys()}

            # exp_xishu 的两个分支原本都是 0.5（死分支），合并为一个常量。
            exp_xishu = 0.5
            for agent_id in rewards.keys():
                exploration_reward = Exploration_reward.get(agent_id, 0)
                reexploration_reward = reExploration_reward.get(agent_id, 0)
                rewards[agent_id] += exploration_reward * exp_xishu + reexploration_reward
                _add_reward_term(terms, agent_id, 'exploration_new', exploration_reward * exp_xishu)
                if terms is not None:
                    shaped, baseline = self._reexploration_parts.get(
                        agent_id, (float(reexploration_reward), 0.0)
                    )
                    _add_reward_term(terms, agent_id, 'reexploration_shaped', shaped)
                    _add_reward_term(terms, agent_id, 'reexploration_baseline', baseline)

            new_target_count = sum(1 for v in done.values() if v)  # legacy 不单独计发现事件
        else:
            # ---- 修复后的目标发现流程：基于本步目标快照找出所有被任意机器人观测到的
            #      目标（同一目标被两个机器人同时看到只计一次），先产生发现事件，再统一删除。
            #      旧流程先删目标再算 find-target 奖励，导致中间发现奖励恒为 0。----
            observed_indices = self._observed_goal_indices(self.agents_pos)
            new_target_count = len(observed_indices)
            if observed_indices:
                self.goals_pos = np.delete(self.goals_pos, sorted(observed_indices), axis=0)
            all_found = len(self.goals_pos) == 0

            # done 只有一个语义：全部目标完成。单个机器人“看到目标”不再设置 per-agent done。
            done = {agent_id: all_found for agent_id in actions.keys()}

            events = StepEvents(
                coverage_delta=coverage_after - coverage_before,
                new_targets=new_target_count,
                all_found=all_found,
                collisions=collisions,
                repeat_flags={agent_id: agent_id in repeat_mark for agent_id in actions.keys()},
                pheromone=pheromone_raw,
                potential_current=coverage_before,
                # 真实终止时势函数归零；时间上限截断由训练循环处理（保留 bootstrap）。
                potential_next=0.0 if all_found else coverage_after,
            )
            if self.reward_mode == "potential":
                mode_terms = compute_potential_rewards(actions.keys(), events, self._reward_weights)
            else:
                mode_terms = compute_v2_rewards(actions.keys(), events, self._reward_weights)
            for agent_id, agent_terms in mode_terms.items():
                # V2/potential 的奖励完全由分量构成：先累加进 rewards，再记日志，
                # 保证 step 返回的奖励恒等于各分量之和（回归测试锁定）。
                rewards[agent_id] += sum(agent_terms.values())
                for name, value in agent_terms.items():
                    _add_reward_term(terms, agent_id, name, value)

        # Return updated observations, rewards, done flags, and info
        updated_obs = obs  # `obs` is now updated with the latest timestamp maps
        next_two_channelmap = two_channelmap

        # 目标也开始移动
        for goal in self.goals_pos:
            valid_position_found = False
            while not valid_position_found:
                # 随机选择移动方式
                move = self._py_rng.choice(move_options)

                # 计算新位置
                new_x = goal[0] + move[0]
                new_y = goal[1] + move[1]

                new_pos = [new_x, new_y]

                # 检查新位置是否越界
                if 0 <= new_pos[0] < 60 and 0 <= new_pos[1] < 60:
                    # 检查新位置是否碰到障碍物
                    if self.map[new_pos[0], new_pos[1]] == 0:  # 0 表示空地
                        # 新位置有效，退出循环
                        valid_position_found = True
                    else:
                        # 如果碰到障碍物，继续选择其他位置
                        continue
                else:
                    # 如果新位置越界，继续选择其他位置
                    continue

            # 将有效的新位置添加到列表中
            new_goals_pos.append(new_pos)

        # 更新目标位置
        self.goals_pos = np.array(new_goals_pos) if new_goals_pos else np.zeros((0, 2), dtype=np.int32)
        result = {}
        for key in rewards:
            if legacy:
                # 队友邻近惩罚只属于 legacy；它和“靠近队友才能融合地图”互相矛盾，
                # V2/potential 不再施加。
                result[key] = rewards[key] + reward_comm[key]
                _add_reward_term(terms, key, 'teammate_proximity', reward_comm[key])
            else:
                result[key] = rewards[key]
        # 分别统计碰撞、短周期循环与真正的重复覆盖，不再混为 repeat：
        # - collisions：每 agent 本步碰撞次数
        # - short_cycle：触发 10 步内位置重复 3 次的 agent（旧 repeat_mark）
        # - new_cells：本步团队并集新增覆盖格数（0 表示这一步纯重复旧区域）
        free_count = int(np.count_nonzero(np.asarray(self.map) != 1)) or 1
        self.last_step_stats = {
            "collisions": dict(collisions),
            "short_cycle": list(repeat_mark),
            "new_cells": int(round((coverage_after - coverage_before) * free_count)),
            "new_targets": new_target_count,
            "all_found": all_found,
            "coverage_before": coverage_before,
            "coverage_after": coverage_after,
        }
        self.last_reward_terms = terms if terms is not None else {}

        return updated_obs, result, done, repeat_mark, next_two_channelmap

    def get_agents_positions(self):
        """返回环境中所有智能体的位置"""
        return self.agents_pos

    def _is_line_of_sight_blocked(self, x1, y1, x2, y2, env_map):
        """检查两点之间的视线是否被障碍物阻挡"""
        # 使用Bresenham算法生成两点之间的直线
        points = self._bresenham_line(x1, y1, x2, y2)

        for px, py in points:
            # 如果遇到障碍物（但不包括目标点本身）
            if env_map[px, py] == 1 and (px != x2 or py != y2):
                return False
        return True

    def _bresenham_line(self, x1, y1, x2, y2):
        """Bresenham算法：生成两点之间的直线上的所有点"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        return points

    def record_position(self):
        """
        记录每个智能体的位置，最多保存最近10个时间步的位置。
        使用 `self.agents_pos` 来更新所有智能体的位置历史。
        """
        for agent_id in self.agent_positions_history.keys():
            # 获取当前智能体的索引
            agent_index = int(agent_id.split('_')[1])

            # 获取当前智能体的位置
            current_pos = self.agents_pos[agent_index]

            # 获取该智能体当前的历史位置数组
            position_history = self.agent_positions_history[agent_id]

            # 先确保 position_history 是二维数组
            if position_history.ndim == 1:
                position_history = position_history.reshape(-1, 2)  # 转换为二维数组，假设每个位置是2维坐标

            # 将 current_pos 变为二维数组 (1, 2)
            position_history = np.append(position_history, current_pos.reshape(1, -1), axis=0)  # 添加新位置

            # 如果位置记录超过10个时间步，则删除最早的记录
            if position_history.shape[0] > 10:
                position_history = np.delete(position_history, 0, axis=0)  # 删除最早的记录

            # 更新位置记录到字典中
            self.agent_positions_history[agent_id] = position_history

    def get_agent_history(self, agent_id):
        # 获取某个智能体的历史位置记录（最近10个时间步）
        return self.agent_positions_history[agent_id]

    # 用来检查
    def check_all_agents_repeated_positions(self):
        """
        检查所有智能体的历史位置是否存在最后一个位置重复三次或更多次。
        如果某个智能体的最后一个位置重复3次或更多，返回这些智能体的ID列表，若没有返回空列表。
        """
        repeated_agents = []  # 用来存储满足条件的智能体ID

        for agent_id, position_history in self.agent_positions_history.items():
            # 获取最后一个加入的位置
            last_position = tuple(position_history[-1])  # 假设位置是一个元组（x, y）

            # 统计位置出现的次数
            position_counter = Counter(tuple(pos) for pos in position_history)

            # 检查最后一个位置是否重复出现3次或更多次
            if position_counter[last_position] >= 3:
                repeated_agents.append(agent_id)  # 如果有位置重复3次或更多次，加入repeated_agents列表

        return repeated_agents  # 返回所有重复位置的智能体ID列表

    def render(self):
        if not hasattr(self, 'fig'):
            self.fig = plt.figure()

        map = np.copy(self.map)
        for agent_id in range(self.num_agents):
            if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                map[tuple(self.agents_pos[agent_id])] = 4
            else:
                map[tuple(self.agents_pos[agent_id])] = 2
                map[tuple(self.goals_pos[agent_id])] = 3

        map = map.astype(np.uint8)

        self.imgs.append([])
        if hasattr(self, 'texts'):
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(self.agents_pos, self.goals_pos)):
                self.texts[i].set_position((agent_y, agent_x))
                self.texts[i].set_text(i)
        else:
            self.texts = []
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(self.agents_pos, self.goals_pos)):
                text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
                plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
                self.texts.append(text)

        plt.imshow(color_map[map], animated=True)

        plt.show()
        plt.pause(0.5)

    def close(self, save=False):
        plt.close()
        del self.fig

    def is_within_vision(self, agent_position, other_agent_position, vision_size):
        # 判断另一个智能体是否在视野范围内，视野是一个11x11的矩阵
        return abs(agent_position[0] - other_agent_position[0]) <= vision_size // 2 and \
            abs(agent_position[1] - other_agent_position[1]) <= vision_size // 2

    def find_agents_in_vision(self, agents_positions, vision_size=13):
        agents_in_vision = []

        for i, agent in enumerate(agents_positions):
            in_vision = []  # 用于存储当前智能体视野范围内的智能体

            for j, other_agent in enumerate(agents_positions):
                if i != j:  # 不考虑自己
                    if self.is_within_vision(agent, other_agent, vision_size):
                        in_vision.append(j)  # 将其他智能体的索引添加到in_vision

            agents_in_vision.append(in_vision)

        return agents_in_vision

    def merge_timestamp_maps(self, map1, map2):
        """
        合并两个时间戳地图，并模拟通信过程中可能遇到的问题
        使用环境的通信参数配置

        Args:
            map1: 接收方的地图
            map2: 发送方的地图

        Returns:
            merged_map: 合并后的地图
        """
        # 如果启用通信噪声，进行噪声模拟
        if self.enable_comm_noise:
            rand_val = self._py_rng.random()

            # 1. 通信完全失败 - 直接返回原地图，不进行任何合并
            if rand_val < self.comm_failure_prob:
                self.comm_stats['failed'] = self.comm_stats.get('failed', 0) + 1
                self.comm_stats['total'] = self.comm_stats.get('total', 0) + 1
                return map1.copy()

            # 2. 通信延迟 - 将信息存入缓冲区，稍后处理
            elif rand_val < self.comm_failure_prob + self.comm_delay_prob:
                self.comm_stats['delayed'] = self.comm_stats.get('delayed', 0) + 1
                self.comm_stats['total'] = self.comm_stats.get('total', 0) + 1

                # 将延迟的地图信息存入缓冲区
                delay_steps = self._py_rng.randint(1, 3)
                self.delayed_map_buffer.append({
                    'map': map2.copy(),
                    'remaining_steps': delay_steps,
                    'target_map_id': id(map1)  # 记录目标地图的ID
                })

                # 暂时不合并，返回原地图
                return map1.copy()

            # 3. 部分信息丢失 - 对map2进行降质处理后再合并
            elif rand_val < self.comm_failure_prob + self.comm_delay_prob + self.comm_partial_loss_prob:
                self.comm_stats['partial'] = self.comm_stats.get('partial', 0) + 1
                self.comm_stats['total'] = self.comm_stats.get('total', 0) + 1

                # 对map2进行降质处理
                map2 = self._apply_communication_degradation(map2)

            # 4. 通信成功 - 正常合并
            else:
                self.comm_stats['success'] = self.comm_stats.get('success', 0) + 1
                self.comm_stats['total'] = self.comm_stats.get('total', 0) + 1

        # ===== 正常的地图合并逻辑 =====
        merged_map = map1.copy()
        rows, cols = map1.shape

        for i in range(rows):
            for j in range(cols):
                # 对于探索过的区域（值在0 - 0.3之间），以及未探索的区域，取两个地图中较小的值
                if 0 <= map1[i, j] <= 0.5 and 0 <= map2[i, j] <= 0.3:
                    merged_map[i, j] = min(map1[i, j], map2[i, j])
                # 将地图2中的障碍物（值为1）标记到地图1中
                elif map2[i, j] == 1:
                    merged_map[i, j] = 1

        # ===== 处理延迟到达的地图信息 =====
        if self.enable_comm_noise and len(self.delayed_map_buffer) > 0:
            updated_buffer = []
            for delayed_item in self.delayed_map_buffer:
                delayed_item['remaining_steps'] -= 1

                # 如果延迟时间到了，合并延迟的地图
                if delayed_item['remaining_steps'] <= 0:
                    delayed_map = delayed_item['map']
                    # 合并延迟到达的地图数据
                    for i in range(rows):
                        for j in range(cols):
                            if 0 <= merged_map[i, j] <= 0.5 and 0 <= delayed_map[i, j] <= 0.3:
                                merged_map[i, j] = min(merged_map[i, j], delayed_map[i, j])
                            elif delayed_map[i, j] == 1:
                                merged_map[i, j] = 1
                else:
                    # 继续等待
                    updated_buffer.append(delayed_item)

            self.delayed_map_buffer = updated_buffer

        return merged_map

    def _apply_communication_degradation(self, map_data):
        """
        对地图数据应用通信降质效果
        模拟部分信息丢失的情况

        Args:
            map_data: 原始地图数据

        Returns:
            降质后的地图数据
        """
        degraded_map = map_data.copy()
        rows, cols = degraded_map.shape

        # 随机选择一种降质方式
        degradation_type = self._py_rng.choice(['random_loss', 'block_loss', 'quality_loss'])

        if degradation_type == 'random_loss':
            # 方式1：随机丢失20%-40%的探索信息
            loss_rate = self._py_rng.uniform(0.2, 0.4)
            for i in range(rows):
                for j in range(cols):
                    # 只对探索信息（0到0.3之间的值）进行丢失
                    if 0 <= degraded_map[i, j] <= 0.3:
                        if self._py_rng.random() < loss_rate:
                            degraded_map[i, j] = 0.5  # 丢失的部分标记为未探索

        elif degradation_type == 'block_loss':
            # 方式2：块状信息丢失（模拟区域性干扰）
            block_size = self._py_rng.randint(3, 5)
            num_blocks = self._py_rng.randint(2, 4)

            for _ in range(num_blocks):
                if rows > block_size and cols > block_size:
                    start_i = self._py_rng.randint(0, rows - block_size)
                    start_j = self._py_rng.randint(0, cols - block_size)

                    for i in range(start_i, min(start_i + block_size, rows)):
                        for j in range(start_j, min(start_j + block_size, cols)):
                            # 只影响探索信息
                            if 0 <= degraded_map[i, j] <= 0.3:
                                degraded_map[i, j] = 0.5

        elif degradation_type == 'quality_loss':
            # 方式3：信息质量下降（时间戳增加，表示信息变旧）
            for i in range(rows):
                for j in range(cols):
                    # 只对探索信息进行质量降低
                    if 0 <= degraded_map[i, j] < 0.3:
                        # 增加时间戳，但不超过0.3
                        degraded_map[i, j] = min(degraded_map[i, j] + self._py_rng.uniform(0.05, 0.15), 0.3)

        return degraded_map