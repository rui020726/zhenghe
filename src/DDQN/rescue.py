import numpy as np
import heapq  # 用于A*算法的优先队列
from collections import deque

class RescueAlgorithm:
    def __init__(self, map_size,env,map_data : dict):
        self.map_size = map_size
        self.agents_pos = env.agents_pos
        self.map_data = map_data  # 地图数据，包含障碍物信息和其他智能体的位置

    def a_star(self, start, goal, agent_id, env):
        """
        A* 算法：计算从start到goal的路径
        """
        open_list = []
        start_tuple = tuple(start)  # 将 numpy.ndarray 转换为元组
        heapq.heappush(open_list, (0, start_tuple))
        g_cost = {start_tuple: 0}  # 使用元组作为字典的键，存储从起点到当前位置的代价
        h_cost = {start_tuple: self.heuristic(start_tuple, goal)}  # 启发式代价（曼哈顿距离或欧几里得距离）
        came_from = {}  # 记录每个节点的父节点

        while open_list:
            current_cost, current = heapq.heappop(open_list)
            if current == goal:
                # 回溯路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_tuple)  # 将起点加入路径
                return path[::-1]  # 逆序返回路径

            for neighbor in self.get_neighbors(current, agent_id, env):
                new_cost = g_cost[current] + 1  # 假设每步的代价是1
                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    h_cost[neighbor] = self.heuristic(neighbor, goal)
                    f_cost = new_cost + h_cost[neighbor]  # 计算总代价 f = g + h
                    heapq.heappush(open_list, (f_cost, neighbor))
                    came_from[neighbor] = current

        return []  # 如果没有找到路径，返回空路径

    def get_neighbors(self, pos, agent_id, env):
        """
        获取当前位置的邻居格子，只考虑上下左右四个方向
        """
        x, y = pos
        if isinstance(agent_id, int):
            agent_id = f"agent_{agent_id}"  # 将整数类型的 agent_id 转换为 "agent_X" 格式
        map_data = self.map_data[agent_id]  # 获取对应智能体的时间戳地图

        # 只考虑上下左右四个方向
        neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

        # 筛选有效的邻居（检查是否在地图范围内并且不是障碍物）
        valid_neighbors = [
            n for n in neighbors
            if 0 <= n[0] < self.map_size[0] and 0 <= n[1] < self.map_size[1] and env.map[n[0], n[1]] != 1
        ]

        return valid_neighbors

    def heuristic(self, current, goal):
        """
        计算启发式代价，使用曼哈顿距离作为启发式函数
        """
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def get_nearest_undiscovered_grid(self, agent_id, obs, env):
        """
        从智能体位置开始，广度优先搜索寻找最近的未探索格子
        如果没有找到值为0.5的格子，则逐渐寻找值为0.3、0.29等的格子
        """
        closest_grid = None
        closest_dist = float('inf')  # 初始化为无穷大，表示还未找到最短距离
        apos = agent_id
        # 如果 agent_id 是整数，则将其转换为 "agent_X" 格式
        if isinstance(agent_id, int):
            agent_id = f"agent_{agent_id}"

        # 获取对应智能体的时间戳地图
        map_data = obs[agent_id]

        # 获取智能体的位置
        agent_pos = env.agents_pos[apos]  # 使用正确的智能体位置

        # 将智能体位置转换为元组，确保它是可哈希的
        agent_pos_tuple = tuple(agent_pos)

        # 使用队列进行广度优先搜索（BFS）
        queue = deque([agent_pos_tuple])  # 初始化队列，从智能体位置开始扩散
        visited = set()  # 用于记录已经访问过的格子
        visited.add(agent_pos_tuple)  # 将起始位置加入已访问的集合
        tolerance = 0.01  # 允许的小范围误差，避免浮动值问题
        # 仅考虑上下左右四个方向
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # 尝试查找不同类型的未探索区域，从值为 0.5 开始，逐渐减少阈值
        exploration_values = [0.5, 0.3, 0.29,0.28,0.27,0.26,0.25,0.24,0.23,0.22,0.21,
                               0.20,0.19,0.18,0.17,0.16,0.15,0.14,0.13,0.12,0.11,0.10,
                              0.09,0.08,0.07,0.06,0.05]  # 逐渐寻找未探索区域，优先找值为 0.5 的格子

        for value in exploration_values:
            while queue:
                x, y = queue.popleft()

                # 检查当前格子是否为目标值（未探索区域）
                if abs(map_data[x, y] - value) < tolerance and env.map[x, y] != 1:
                    closest_grid = (x, y)
                    closest_dist = np.abs(agent_pos[0] - x) + np.abs(agent_pos[1] - y)
                    break  # 找到第一个符合条件的格子后，退出循环

                # 扩展到相邻的格子
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    # 确保邻居在地图范围内且没有访问过，且不是障碍物（值为 1）
                    if 0 <= nx < self.map_size[0] and 0 <= ny < self.map_size[1]:
                        if (nx, ny) not in visited and map_data[nx, ny] != 1:  # 排除障碍物
                            visited.add((nx, ny))
                            queue.append((nx, ny))

            # 如果找到了目标值的格子，直接跳出循环
            if closest_grid is not None:
                break

        # 如果没有找到目标格子，打印信息
        if closest_grid is None:
            print("没有找到符合条件的未探索的格子")

        return closest_grid

    def convert_path_to_actions(self, path):
        """
        将A*算法计算的路径转换为一系列动作
        """
        actions = []
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]

            dx = next_pos[0] - current[0]
            dy = next_pos[1] - current[1]

            # 判断相对位置来决定动作
            if dx == 1 and dy == 0:
                actions.append(1)  # 向下
            elif dx == -1 and dy == 0:
                actions.append(0)  # 向上
            elif dx == 0 and dy == 1:
                actions.append(3)  # 向右
            elif dx == 0 and dy == -1:
                actions.append(2)  # 向左
        return actions

    def rescue_action(self, agent_id,env,obs):
        """
        当智能体的当前位置重复三次时触发救援算法
        """
        # 获取当前智能体位置
        agent_pos = env.agents_pos[agent_id]

        # 获取最近未被探索的格子
        goal = self.get_nearest_undiscovered_grid(agent_id,obs,env)
        max_v = -1
        max_coordinates = []
        if isinstance(agent_id, int):
            agentid = f"agent_{agent_id}"

        if goal is None:
            tiem_map = obs[agentid]
            valid_values = tiem_map[(0 <= tiem_map) & (tiem_map <= 0.3)]
            if valid_values.size == 0:
                max_v = -1
                max_coordinates = []
            else:
                max_v = valid_values.max()
                # 找出所有等于最大值的坐标
                rows, columns = np.where(tiem_map == max_v)
                max_coordinates = [(rows[i], columns[i]) for i in range(len(rows))]

            if max_coordinates:
                # 计算每个坐标与 agent_pos 的启发式距离
                distances = [self.heuristic(coord, agent_pos) for coord in max_coordinates]
                # 找出最小距离的索引
                min_index = np.argmin(distances)
                # 选出离 agent_pos 最近的点
                closest_point = max_coordinates[min_index]
                goal = closest_point
        # 使用A*算法计算路径
        path = self.a_star(agent_pos, goal,agent_id,env)

        # 将路径转换为动作
        actions = self.convert_path_to_actions(path)

        return actions  # 返回计算的动作列表