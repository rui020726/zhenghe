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


class Environment:
    def __init__(self, num_agents: int = configs.num_agents, map_length: int = configs.init_env_settings[1],
                 obs_radius: int = configs.obs_radius, reward_fn: dict = configs.reward_fn, fix_density=None,
                 curriculum=False, init_env_settings_set=configs.init_env_settings,num_goals = configs.num_goals,pheromone_decay = configs.pheromone_decay,test = False):
        """
        num_agents: 环境中智能体的数量，默认为configs.init_env_settings[0]。
        map_length: 地图的大小（即地图的边长），默认为configs.init_env_settings[1]。
        obs_radius: 每个智能体的观察半径，决定了它能看到多少周围的环境。
        reward_fn: 奖励函数，默认值为configs.reward_fn。
        fix_density: 如果不为None，则指定地图的障碍物密度；否则，障碍物密度是随机的。
        curriculum: 如果为True，则启用“渐进式难度”（curriculum learning）。
        init_env_settings_set: 用于初始化环境设置，默认为configs.init_env_settings。
        """
        self.num_goals = num_goals # 目标个数
        self.timestamp_map = np.full((60, 60), 0.5, dtype=np.float32)  # 初始时间戳地图
        self.agents = [f'agent_{i}' for i in range(num_agents)]  # 智能体列表
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.agents}  # 每个智能体有4个离散动作
        self.curriculum = curriculum
        self.check_repeat_pos = []

        self.global_pheromone = np.full((60, 60), 0, dtype=np.float32)  # 初始化全局的信息素地图
        self.pheromone_decay = pheromone_decay  # 信息素挥发率
        self.local_pheromone = {}
        self.noise_sigma = 0.5
        self.size = 60

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
            self.obstacle_density = np.random.triangular(0, 0.15, 0.3)
        else:
            self.fix_density = True
            self.obstacle_density = fix_density


        if test:
            map_dir = r'E:/Desktop/代码/修改6--在修改5的基础上，所有网络使用相同的参数/DungeonMaps/test'

            # 生成图片文件名列表 (img_1.png 到 img_5001.png)
            #image_files = [f'img_{i}.png' for i in range(6000, 10000)]
            #image_files = [f'img_6167.png']
            #image_files = [f'img_7923.png']
            #image_files = [f'img_10818.png']
            #image_files = [f'img_7846.png']
            image_files = [f'img_6086.png']
            # 随机选择一张图片并确保文件存在
            while True:
                # 随机选择一张图片
                random_image = random.choice(image_files)

                # 拼接路径
                image_path = os.path.join(map_dir, random_image)

                # 打印选择的图片路径
                print(f"Selected image: {image_path}")

                # 检查文件是否存在
                if os.path.exists(image_path):
                    # 如果文件存在，加载图像并转换为灰度图
                    image = Image.open(image_path).convert('L')  # 转为灰度图，'L'模式是8位像素，黑白图

                    # 调整图像大小为 60x60
                    image_resized = image.resize((60, 60))  # 调整大小

                    # 转换为 NumPy 数组
                    map_data = np.array(image_resized)

                    # 将图像数据归一化为0和1（假设地图上0代表空白，1代表障碍物）
                    map_data = (map_data < 128).astype(np.int32)  # 阈值处理，128以下为障碍物（1），其余为空白（0）

                    # 将地图数据赋值给实例的map
                    self.map = map_data

                    # 结束循环，文件已成功加载
                    break
                else:
                    print(f"File {image_path} not found. Retrying...")
        else:
            map_dir =  r'E:/Desktop/代码/修改6--在修改5的基础上，所有网络使用相同的参数/DungeonMaps/train'

            # 生成图片文件名列表 (img_1.png 到 img_5001.png)
            image_files = [f'img_{i}.png' for i in range(1, 5002)]

            # 随机选择一张图片并确保文件存在
            while True:
                # 随机选择一张图片
                random_image = random.choice(image_files)

                # 拼接路径
                image_path = os.path.join(map_dir, random_image)

                # 打印选择的图片路径
                print(f"Selected image: {image_path}")

                # 检查文件是否存在
                if os.path.exists(image_path):
                    # 如果文件存在，加载图像并转换为灰度图
                    image = Image.open(image_path).convert('L')  # 转为灰度图，'L'模式是8位像素，黑白图

                    # 调整图像大小为 60x60
                    image_resized = image.resize((60, 60))  # 调整大小

                    # 转换为 NumPy 数组
                    map_data = np.array(image_resized)

                    # 将图像数据归一化为0和1（假设地图上0代表空白，1代表障碍物）
                    map_data = (map_data < 128).astype(np.int32)  # 阈值处理，128以下为障碍物（1），其余为空白（0）

                    # 将地图数据赋值给实例的map
                    self.map = map_data

                    # 结束循环，文件已成功加载
                    break
                else:
                    print(f"File {image_path} not found. Retrying...")
        partition_list = map_partition(self.map)
        partition_list = [partition for partition in partition_list if len(partition) >= 2]  # 过滤掉空白位置少于2的区域
        # 没有符合条件则重新生成
        while len(partition_list) == 0:
            self.map = np.random.choice(2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]).astype(
                np.int32)
            partition_list = map_partition(self.map)
            partition_list = [partition for partition in partition_list if len(partition) >= 2]

        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int32)
        self.goals_pos = np.empty((self.num_goals, 2), dtype=np.int32)
        self.goals_pos.fill(-1)  # 用-1表示空位置

        max_list = max(partition_list, key=len) # 获取面积最大的连通区域

        # 循环为每个智能体分配其初始位置
        for i in range(self.num_agents):

            pos = random.choice(max_list)
            max_list.remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=np.int32)

        # 循环设置目标位置
        for i in range(self.num_goals):
            pos = random.choice(max_list)
            max_list.remove(pos)
            # 找到goals_pos中的空位置，并将选中的pos添加到该位置
            empty_index = np.where(self.goals_pos == -1)  # 查找goals_pos中为-1的空位置
            self.goals_pos[empty_index[0][0]] = pos  # 更新空位置为选中的目标位置

        self.obs_radius = obs_radius
        self.global_timestamp_maps = np.full((60, 60), 0.5, dtype=np.float32) # 初始化一个全局的时间戳地图，用于计算奖励
        for i,(x,y) in enumerate(self.agents_pos):
            self.global_pheromone[x,y] = 1 #  将智能体初始位置的信息素标1
            self.local_pheromone[f"agent_{i}"] = self.calculate_pheromone(x,y)
        self.reward_fn = reward_fn
        self.steps = 0


        # 记录最近10次的动作轨迹
        self.agent_positions_history = {f"agent_{i}": self.agents_pos[i] for i in range(num_agents)}  # 用来记录智能体过去的10个位置
        self.last_actions = np.zeros((self.num_agents, 10), dtype=np.bool_)

    # 将地图转换为双通道地图
    def two_channel_map(self, agent_timestamp_maps: dict,two_channelmap:dict):
        for agent_id , timemap in agent_timestamp_maps.items():
            Obstacle_map = np.full((60, 60), -1, dtype=np.float32)  # 初始化障碍物地图
            explore_map = np.full((60, 60), -0.5, dtype=np.float32)  # 初始化障碍物地图
            for i in range(timemap.shape[0]):
                for j in range(timemap.shape[0]):
                    v = timemap[i,j]
                    if v == 1:
                        Obstacle_map[i,j] = 1
                        explore_map[i, j] = 0
                    elif v == 0.5:
                        explore_map[i,j] = 0.5
                    elif 0<= v <= 0.3 :
                        Obstacle_map[i,j] = 0
                        explore_map[i,j] = v
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
            rand = random.choice(self.env_set)
            self.num_agents = rand[0]
            self.map_size = (rand[1], rand[1])

        elif num_agents is not None and map_length is not None:
            self.num_agents = num_agents
            self.map_size = (map_length, map_length)

        if not self.fix_density:
            self.obstacle_density = np.random.triangular(0, 0.33, 0.5)

        self.map = np.random.choice(2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]).astype(
            np.float32)

        partition_list = map_partition(self.map)
        partition_list = [partition for partition in partition_list if len(partition) >= 2]

        while len(partition_list) == 0:
            self.map = np.random.choice(2, self.map_size, p=[1 - self.obstacle_density, self.obstacle_density]).astype(
                np.float32)
            partition_list = map_partition(self.map)
            partition_list = [partition for partition in partition_list if len(partition) >= 2]

        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int32)
        self.goals_pos = np.empty((self.num_goals, 2), dtype=np.int32)
        self.goals_pos.fill(-1)  # 用-1表示空位置

        max_list = max(partition_list, key=len)  # 获取面积最大的连通区域

        # 循环为每个智能体分配其初始位置
        for i in range(self.num_agents):
            pos = random.choice(max_list)
            max_list.remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=np.int32)

        # 循环设置目标位置
        for i in range(self.num_goals):
            pos = random.choice(max_list)
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

    def save_data(self,agent_id, agent_map, global_map, save_dir):
        """保存智能体的时间戳地图和全局地图"""

        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = time.strftime("%Y%m%d-%H%M%S")  # 使用时间戳作为文件名的一部分
        # 保存智能体的时间戳地图
        np.save(os.path.join(save_dir, f"agent_{agent_id}_timestamp_map_{timestamp}.npy"), agent_map)

        # 保存全局地图
        np.save(os.path.join(save_dir, f"global_map_{timestamp}.npy"), global_map)

    def update_timestamp_map(self, agent_timestamp_maps: dict,two_channelmap:dict):
        """根据环境中的智能体位置更新时间戳地图"""
        agent_invision = []
        agents_positions = self.get_agents_positions()  # 获取所有智能体的位置
        reExploration_reward_find_target = 0  # 当前视野范围内存在的目标数量
        env_map = self.map  # 环境中的地图，包含障碍物信息
        updated_to_zero_count_dict = {}
        reExploration_reward_dict = {}
        # 更新每个智能体的时间戳地图
        for agent_id, agent_map in agent_timestamp_maps.items():
            # 获取该智能体的位置
            agent_pos = agents_positions[int(agent_id.split('_')[1])]
            x, y = agent_pos
            reExploration_reward = 0
            updated_to_zero_count_1 = 0  # 用于统计值从0.5变为0的次数
            updated_to_zero_count_2 = 0  # 用于统计值从0.5变为0的次数
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
            # 更新当前智能体
            # 获得再次探索的奖励
            for dx in range(-5, 6):  # x轴方向从-1到1
                for dy in range(-5, 6):  # y轴方向从-1到1
                    nx, ny = x + dx, y + dy  # 计算邻域坐标
                    # 判断是否越界，确保新的坐标在地图范围内
                    if 0 <= nx < self.map_size[0] and 0 <= ny < self.map_size[1]:
                        if nx != x and ny != y and env_map[nx, ny] != 1 and agent_map[nx, ny] not in [-4.5,
                                                                                                                  -5.5,
                                                                                                                  -6.5]:
                            v = agent_map[nx, ny]
                            if 0 <= v <= 0.3:
                                reExploration_reward += 1 + v / 0.3  # 获得再次探索的奖励
            reExploration_reward = reExploration_reward * 0.01 - 1
            reExploration_reward_dict[agent_id] = reExploration_reward
            for goal_index, goal_pos in enumerate(self.goals_pos):
                # 计算横纵坐标差的绝对值
                dx = abs(x - goal_pos[0])  # 横坐标差
                dy = abs(y - goal_pos[1])  # 纵坐标差

                # 判断目标是否在视野内，视野范围为3*3
                if dx <= 5 and dy <= 5:
                    reExploration_reward_find_target += 1
            if reExploration_reward_find_target != 0:
                reExploration_reward_dict[agent_id] = reExploration_reward_find_target*0.1

            for dx in range(-5, 6):  # x轴方向从-1到1
                for dy in range(-5, 6):  # y轴方向从-1到1
                    nx, ny = x + dx, y + dy  # 计算邻域坐标
                    # 判断是否越界，确保新的坐标在地图范围内
                    if 0 <= nx < self.map_size[0] and 0 <= ny < self.map_size[1]:
                        # 如果该位置是障碍物，则将其标记为1
                        if env_map[nx, ny] == 1:
                            agent_map[nx, ny] = 1  # 障碍物位置标记为1
                        # 更新非障碍物的时间戳，排除已被标记的智能体位置
                        elif [nx,ny] != [x,y] and env_map[nx, ny] != 1 and agent_map[nx, ny] not in [-4.5, -5.5,
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

        two_channelmap = self.two_channel_map(agent_timestamp_maps,two_channelmap) # 更新双通道地图
        return agent_timestamp_maps,updated_to_zero_count_dict,reExploration_reward_dict,two_channelmap

    """
    def update_global_timestamp_maps(self):
        agents_positions = self.get_agents_positions()  # 获取所有智能体的位置
        env_map = self.map  # 环境中的地图，包含障碍物信息
        updated_to_zero_count_dict = {}
        reExploration_reward_dict = {}
        # 更新每个智能体的时间戳地图

        # 获取该智能体的位置
        updated_to_zero_count_1 = 0  # 用于统计值从0.5变为0的次数
        updated_to_zero_count_2 = 0  # 用于统计值从0.5变为0的次数
        reExploration_reward = 0
        reExploration_reward_find_target = 0  # 当前视野范围内存在的目标数量

        # 先将探索过的时间戳地图更新
        for i in range(self.global_timestamp_maps.shape[0]):  # 遍历行
            for j in range(self.global_timestamp_maps.shape[0]):  # 遍历列
                value = self.global_timestamp_maps[i, j]
                if value == 5.5:  # 上一时刻的其他智能体位置修改为0.01
                    self.global_timestamp_maps[i, j] = 0.01
                elif value == -4.5:  # 上一时刻当前智能体的位置
                    self.global_timestamp_maps[i, j] = -5.5
                elif value == -5.5:  # 前两个时刻当前智能体的位置
                    self.global_timestamp_maps[i, j] = -6.5
                elif value == -6.5:  # 最多记录三个位置
                    self.global_timestamp_maps[i, j] = 0.03
                elif 0 <= value < 0.3:  # 更新旧的时间戳地图
                    self.global_timestamp_maps[i, j] += 0.01
                    updated_to_zero_count_1 += 1
                elif value == 0.3:
                    updated_to_zero_count_1 += 1  # 统计探索了多少

        for agent_id, (x, y) in enumerate(agents_positions):
            if isinstance(agent_id, int):
                agent_id = f"agent_{agent_id}"
            # 获得再次探索的奖励
            for dx in range(-5, 6):  # x轴方向从-1到1
                for dy in range(-5, 6):  # y轴方向从-1到1
                    nx, ny = x + dx, y + dy  # 计算邻域坐标
                    # 判断是否越界，确保新的坐标在地图范围内
                    if 0 <= nx < self.global_timestamp_maps.shape[0] and 0 <= ny < self.global_timestamp_maps.shape[1]:
                        if nx != x and ny != y and env_map[nx, ny] != 1 and self.global_timestamp_maps[nx, ny] not in [-4.5, -5.5,
                                                                                                      -6.5]:
                            v = self.global_timestamp_maps[nx, ny]
                            if 0 <= v <= 0.3:
                                reExploration_reward += 1 - v / 0.3  # 获得再次探索的奖励
            reExploration_reward = reExploration_reward * 0.03 - 1
            reExploration_reward_dict[agent_id] = reExploration_reward
            for goal_index, goal_pos in enumerate(self.goals_pos):
                # 计算横纵坐标差的绝对值
                dx = abs(x - goal_pos[0])  # 横坐标差
                dy = abs(y - goal_pos[1])  # 纵坐标差

                # 判断目标是否在视野内，视野范围为3*3
                if dx <= 5 and dy <= 5:
                    reExploration_reward_find_target += 1
            if reExploration_reward_find_target != 0:
                reExploration_reward_dict[agent_id] = reExploration_reward_find_target * 0.1

                # 更新当前智能体
            for dx in range(-5, 6):  # x轴方向从-1到1
                for dy in range(-5, 6):  # y轴方向从-1到1
                    nx, ny = x + dx, y + dy  # 计算邻域坐标
                    # 判断是否越界，确保新的坐标在地图范围内
                    if 0 <= nx < self.global_timestamp_maps.shape[0] and 0 <= ny < self.global_timestamp_maps.shape[1]:
                        # 如果该位置是障碍物，则将其标记为1
                        if env_map[nx, ny] == 1:
                            self.global_timestamp_maps[nx, ny] = 1  # 障碍物位置标记为1
                        # 更新非障碍物的时间戳，排除已被标记的智能体位置
                        elif [nx, ny] != [x, y] and env_map[nx, ny] != 1 and self.global_timestamp_maps[nx, ny] not in [-4.5, -5.5,
                                                                                                           -6.5]:
                            self.global_timestamp_maps[nx, ny] = 0  # 不是障碍物的位置标记为0
                self.global_timestamp_maps[x, y] = -4.5
                # 统计还有多少未探索（用于计算当前智能体的探索奖励）
                for i in range(self.global_timestamp_maps.shape[0]):  # 遍历行
                    for j in range(self.global_timestamp_maps.shape[1]):  # 遍历列
                        value = self.global_timestamp_maps[i, j]
                        if 0 <= value <= 0.3:
                            updated_to_zero_count_2 += 1  # 统计探索了多少
            updated_to_zero_count_dict[agent_id] = updated_to_zero_count_2 - updated_to_zero_count_1

        return  updated_to_zero_count_dict, reExploration_reward_dict
    """ # 全局时间戳地图

    # 计算智能体当前视野内的信息素
    def calculate_pheromone(self,x,y):
        value = 0
        for dx in range(- self.obs_radius, self.obs_radius+1):
            for dy in range(- self.obs_radius, self.obs_radius + 1):
                if 0 <= x+dx <= 59 and 0 <= y+dy <= 59:
                    v = self.global_pheromone[x+dx,y+dy]
                    value += v
        return value

    def _is_line_of_sight_blocked(self,x1, y1, x2, y2, env_map):
        """检查两点之间的视线是否被障碍物阻挡"""
        # 使用Bresenham算法生成两点之间的直线
        points = self._bresenham_line(x1, y1, x2, y2)

        for px, py in points:
            # 如果遇到障碍物（但不包括目标点本身）
            if env_map[px, py] == 1 and (px != x2 or py != y2):
                return False
        return True

    def _bresenham_line(self,x1, y1, x2, y2):
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

    def step(self, actions: dict, obs: dict,rescue_actions : dict,two_channelmap:dict,E):
        move_options = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
        new_goals_pos = []
        '''
        actions:
            dictionary where keys are agent IDs and values are action indices
            Example: {'agent_0': 1, 'agent_1': 2, 'agent_2': 3}

        obs:
            dictionary where keys are agent IDs and values are the previous timestamp maps
            Example: {'agent_0': timestamp_map_0, 'agent_1': timestamp_map_1, 'agent_2': timestamp_map_2}
        '''
        rewards = {agent_id: 0 for agent_id in actions.keys()}
        checking_list = [i for i in range(self.num_agents)]

        next_pos = np.copy(self.agents_pos)
        self.global_pheromone = self.global_pheromone*self.pheromone_decay  # 每一步信息素都在衰减
        # Step 1: Remove unmoving agent and calculate next positions
        for agent_id, action in actions.items():
            agent_index = int(agent_id.split('_')[1])  # 提取数字部分并转换为整数
            next_pos[agent_index] += action_list[action]

        # Step 2: Boundary check and obstacle collision
        for agent_id in checking_list.copy():
            agent_key = f"agent_{agent_id}"  # 获取字典中的键，如 'agent_0', 'agent_1' 等
            if np.any(next_pos[agent_id] < 0) or np.any(next_pos[agent_id] >= self.map_size[0]):
                rewards[agent_key] += self.reward_fn['collision']
                #if E<= 50:
                    #rewards[agent_key] += 2
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)
            elif self.map[tuple(next_pos[agent_id])] == 1:  # Check for obstacle
                rewards[agent_key] += self.reward_fn['collision']
                #if E<= 50:
                    #rewards[agent_key] += 2
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)
        """
        # Step 3: Agent collision detection
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:
                agent_key = f"agent_{agent_id}"  # 获取字典中的键，如 'agent_0', 'agent_1' 等
                target_agent_id = np.where(np.all(next_pos[agent_id] == self.agents_pos, axis=1))[0]
                if target_agent_id.size > 0:
                    target_agent_id = target_agent_id.item()
                    if target_agent_id != agent_id:
                        next_pos[agent_id] = self.agents_pos[agent_id]
                        rewards[agent_key] = self.reward_fn['collision']
                        next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                        rewards[agent_key] = self.reward_fn['collision']
                        checking_list.remove(agent_id)
                        checking_list.remove(target_agent_id)
                        no_conflict = False
                        break

        # Step 4: Check for agent-agent collision
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:
                collide_agent_id = np.where(np.all(next_pos == next_pos[agent_id], axis=1))[0].tolist()
                if len(collide_agent_id) > 1:
                    all_in_checking = all([id in checking_list for id in collide_agent_id])
                    if all_in_checking:
                        collide_agent_pos = next_pos[collide_agent_id].tolist()
                        collide_agent_pos.sort(key=lambda x: x[0] * self.map_size[0] + x[1])
                        collide_agent_id.remove(collide_agent_pos[0][2])
                        next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                        for id in collide_agent_id:
                            agent_key = f"agent_{id}"  # 获取字典中的键，如 'agent_0', 'agent_1' 等
                            rewards[agent_key] = self.reward_fn['collision']
                        for id in collide_agent_id:
                            checking_list.remove(id)
                        no_conflict = False
                        break
        """
        # Step 5: Update positions and check if all agents reach the goal
        self.agents_pos = np.copy(next_pos)
        for i,(x,y) in enumerate(self.agents_pos):
            v = self.calculate_pheromone(x,y)
            v_ = self.local_pheromone[f"agent_{i}"]

            #if v_ - v > 0.5:
                #v_reward = v_ - v
                #rewards[f"agent_{i}"] += v_reward # 如果下一时刻的信息素浓度小于当前时刻，则获得少量奖励
            self.local_pheromone[f"agent_{i}"] = v  # 更新信息素浓度
        for x,y in self.agents_pos:
            self.global_pheromone[x,y] += 1  # 新位置的信息素加1
            if self.global_pheromone[x,y] > 10:  # 设置一个上限
                self.global_pheromone[x, y] = 10

        self.steps += 1
        self.record_position() # 更新位置记录
        self.check_repeat_pos = self.check_all_agents_repeated_positions()  # 检查过去10个位置是否有三个重复的，如果有则返回智能体编号
        repeat_mark = self.check_repeat_pos
        # 处于营救策略中的agent不会再被repeat_mark
        for agent_id, _ in rescue_actions.items():
            if agent_id in repeat_mark:
                repeat_mark.remove(agent_id)
        # 营救策略触发时，给予一个惩罚
        if repeat_mark:  # 检查 repeat_mark 列表是否为空
            for agent_id in repeat_mark:  # 遍历 repeat_mark 中的每个智能体编号
                if agent_id in rewards:  # 确保 rewards 中有这个智能体的编号
                    rewards[agent_id] -= 0.1  # 将对应智能体的奖励减去 0.1

        # 检查是否有智能体找到目标
        done = {agent_id: False for agent_id in actions.keys()}
        for agent_id in actions.keys():
            agent_index = int(agent_id.split('_')[1])  # 提取数字部分并转换为整数
            agent_pos = self.agents_pos[agent_index]  # 获取当前智能体的位置

            # 记录需要删除的目标索引
            indices_to_delete = []

            # 遍历目标位置并检查每个目标是否在智能体的视野内
            for goal_index, goal_pos in enumerate(self.goals_pos):
                # 计算横纵坐标差的绝对值
                x = agent_pos[0]
                y = agent_pos[1]
                nx = goal_pos[0]
                ny = goal_pos[1]
                dx = abs(agent_pos[0] - goal_pos[0])  # 横坐标差
                dy = abs(agent_pos[1] - goal_pos[1])  # 纵坐标差

                # 判断目标是否在视野内，视野范围为3*3
                if dx <= 5 and dy <= 5 and self._is_line_of_sight_blocked(x, y, nx, ny, self.map):
                    done[agent_id] = True
                    indices_to_delete.append(goal_index)  # 记录需要删除的目标索引

            # 删除目标
            self.goals_pos = np.delete(self.goals_pos, indices_to_delete, axis=0)

        # 检测所有目标是否均被观测到
        if len(self.goals_pos) == 0:
            rewards.update({agent_id: self.reward_fn['finish'] for agent_id in actions.keys()})
            done = {agent_id: True for agent_id in actions.keys()}
        else:
            done = {agent_id: False for agent_id in actions.keys()}


        """
        # Ensure no agents overlap
        if np.unique(self.agents_pos, axis=0).shape[0] < self.num_agents:
            raise RuntimeError('Logic error: Agents overlap detected.')
        """

        # Step 6: Update timestamp map after all actions using the provided `obs`
        obs,Exploration_reward, reExploration_reward,two_channelmap  = self.update_timestamp_map(obs,two_channelmap) # Directly use `obs` to update the timestamp map
        if E <= 50:
            exp_xishu = 0.5
        else:
            exp_xishu = 0.5
        for agent_id in rewards.keys():
            # Assuming Exploration_reward is a dictionary storing the exploration rewards for each agent
            exploration_reward = Exploration_reward.get(agent_id, 0)  # Default to 0 if not present
            reexploration_reward = reExploration_reward.get(agent_id, 0)
            rewards[agent_id] += exploration_reward * exp_xishu + reexploration_reward # Add 0.4 times the exploration reward

        # Return updated observations, rewards, done flags, and info
        updated_obs = obs  # `obs` is now updated with the latest timestamp maps
        next_two_channelmap = two_channelmap

        # 目标也开始移动
        for goal in self.goals_pos:
            valid_position_found = False
            while not valid_position_found:
                # 随机选择移动方式
                move = random.choice(move_options)

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
        self.goals_pos = np.array(new_goals_pos)

        return updated_obs, rewards, done, repeat_mark,next_two_channelmap

    def get_agents_positions(self):
        """返回环境中所有智能体的位置"""
        return self.agents_pos

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
        # plt.xlabel('step: {}'.format(self.steps))

        # add text in plot
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
        # plt.ion()
        plt.pause(0.5)

    def close(self, save=False):
        plt.close()
        del self.fig

    def is_within_vision(self,agent_position, other_agent_position, vision_size):
        # 判断另一个智能体是否在视野范围内，视野是一个11x11的矩阵
        return abs(agent_position[0] - other_agent_position[0]) <= vision_size // 2 and \
               abs(agent_position[1] - other_agent_position[1]) <= vision_size // 2

    def find_agents_in_vision(self,agents_positions, vision_size=11):
        agents_in_vision = []

        for i, agent in enumerate(agents_positions):
            in_vision = []  # 用于存储当前智能体视野范围内的智能体

            for j, other_agent in enumerate(agents_positions):
                if i != j:  # 不考虑自己
                    if self.is_within_vision(agent, other_agent, vision_size):
                        in_vision.append(j)  # 将其他智能体的索引添加到in_vision

            agents_in_vision.append(in_vision)

        return agents_in_vision

    def merge_timestamp_maps(self,map1, map2):
        # 初始化合并后的地图，初始值为map1
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
        return merged_map