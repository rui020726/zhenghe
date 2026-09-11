import numpy as np
import random
import matplotlib.pyplot as plt


class iAA_Algorithm:
    """
    反向蚁群算法 (Inverted Ant Algorithm)
    信息素具有排斥性，鼓励蚂蚁探索未探索区域
    """

    def __init__(self, num_ants, num_iterations, alpha=1.0, evaporation_rate=0.1,
                 initial_pheromone=1.0, grid_size=(50, 50)):
        """
        初始化 iAA 算法

        Args:
            num_ants: 蚂蚁数量
            num_iterations: 迭代次数
            alpha: 信息素影响因子（α）
            evaporation_rate: 信息素挥发率
            initial_pheromone: 初始信息素浓度
            grid_size: 网格大小 (rows, cols)
        """
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.evaporation_rate = evaporation_rate
        self.initial_pheromone = initial_pheromone
        self.grid_size = grid_size

        # 初始化信息素矩阵（c 代表常数，n_i(t) 代表位置 i 的信息素）
        self.pheromone = np.full(grid_size, initial_pheromone, dtype=float)

        # 记录探索历史
        self.exploration_history = []

        # 统计每个位置被访问的次数
        self.visit_count = np.zeros(grid_size, dtype=int)

    def calculate_probabilities(self, current_pos, neighbor_positions, c=1.0):
        """
        根据 iAA 的概率密度函数计算移动概率

        公式: P_i(t+1) = (c + n_i(t))^(-α) / Σ_j (c + n_j(t))^(-α)

        Args:
            current_pos: 当前位置 (row, col)
            neighbor_positions: 邻居位置列表 [(row1, col1), (row2, col2), ...]
            c: 常数项，防止分母为零

        Returns:
            probabilities: 每个邻居位置的选择概率
        """
        probabilities = []

        for pos in neighbor_positions:
            row, col = pos
            # 计算 (c + n_i(t))^(-α)
            pheromone_value = c + self.pheromone[row, col]
            prob_numerator = np.power(pheromone_value, -self.alpha)
            probabilities.append(prob_numerator)

        # 归一化概率
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            # 如果所有概率为0，均匀分布
            probabilities = [1.0 / len(neighbor_positions)] * len(neighbor_positions)

        return probabilities

    def get_neighbors(self, pos):
        """
        获取当前位置的有效邻居位置（上下左右四个方向）

        Args:
            pos: 当前位置 (row, col)

        Returns:
            neighbors: 有效邻居位置列表
        """
        row, col = pos
        neighbors = []

        # 四个方向：上、下、左、右
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            # 检查边界
            if 0 <= new_row < self.grid_size[0] and 0 <= new_col < self.grid_size[1]:
                neighbors.append((new_row, new_col))

        return neighbors

    def move_ant(self, current_pos):
        """
        根据 iAA 概率密度函数移动蚂蚁

        Args:
            current_pos: 当前位置

        Returns:
            next_pos: 下一个位置
        """
        neighbors = self.get_neighbors(current_pos)

        if not neighbors:
            return current_pos

        # 计算每个邻居的选择概率
        probabilities = self.calculate_probabilities(current_pos, neighbors)

        # 根据概率选择下一个位置
        next_pos = random.choices(neighbors, weights=probabilities, k=1)[0]

        return next_pos

    def update_pheromone(self, visited_positions, deposit_amount=1.0):
        """
        更新信息素

        Args:
            visited_positions: 本次迭代中访问过的位置列表
            deposit_amount: 每次访问增加的信息素量
        """
        # 信息素挥发
        self.pheromone *= (1 - self.evaporation_rate)

        # 在访问过的位置增加信息素（使其更具排斥性）
        for pos in visited_positions:
            row, col = pos
            self.pheromone[row, col] += deposit_amount

    def run(self, start_positions=None):
        """
        运行 iAA 算法

        Args:
            start_positions: 蚂蚁初始位置列表，如果为 None 则随机初始化

        Returns:
            exploration_coverage: 探索覆盖率
        """
        # 初始化蚂蚁位置
        if start_positions is None:
            ant_positions = []
            for _ in range(self.num_ants):
                row = random.randint(0, self.grid_size[0] - 1)
                col = random.randint(0, self.grid_size[1] - 1)
                ant_positions.append((row, col))
        else:
            ant_positions = start_positions

        # 主循环
        for iteration in range(self.num_iterations):
            visited_positions = []

            # 移动每只蚂蚁
            for i in range(self.num_ants):
                current_pos = ant_positions[i]
                next_pos = self.move_ant(current_pos)

                # 更新蚂蚁位置
                ant_positions[i] = next_pos
                visited_positions.append(next_pos)

                # 记录访问次数
                row, col = next_pos
                self.visit_count[row, col] += 1

            # 更新信息素
            self.update_pheromone(visited_positions)

            # 记录探索历史
            coverage = np.count_nonzero(self.visit_count) / (self.grid_size[0] * self.grid_size[1])
            self.exploration_history.append(coverage)

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.num_iterations}, "
                      f"Coverage: {coverage * 100:.2f}%")

        final_coverage = self.exploration_history[-1]
        print(f"\n最终探索覆盖率: {final_coverage * 100:.2f}%")

        return final_coverage

    def visualize_results(self):
        """
        可视化结果
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. 访问热图
        im1 = axes[0].imshow(self.visit_count, cmap='hot', interpolation='nearest')
        axes[0].set_title('访问热图 (Visit Count Heatmap)', fontsize=14)
        axes[0].set_xlabel('列 (Column)')
        axes[0].set_ylabel('行 (Row)')
        plt.colorbar(im1, ax=axes[0], label='访问次数')

        # 2. 信息素分布
        im2 = axes[1].imshow(self.pheromone, cmap='viridis', interpolation='nearest')
        axes[1].set_title('信息素分布 (Pheromone Distribution)', fontsize=14)
        axes[1].set_xlabel('列 (Column)')
        axes[1].set_ylabel('行 (Row)')
        plt.colorbar(im2, ax=axes[1], label='信息素浓度')

        # 3. 探索覆盖率曲线
        axes[2].plot(range(1, len(self.exploration_history) + 1),
                     [c * 100 for c in self.exploration_history],
                     linewidth=2)
        axes[2].set_title('探索覆盖率变化 (Exploration Coverage)', fontsize=14)
        axes[2].set_xlabel('迭代次数 (Iteration)')
        axes[2].set_ylabel('覆盖率 (%)')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建 iAA 算法实例
    iaa = iAA_Algorithm(
        num_ants=10,  # 10只蚂蚁
        num_iterations=100,  # 迭代100次
        alpha=2.0,  # α = 2.0，控制信息素的排斥强度
        evaporation_rate=0.05,  # 信息素挥发率 5%
        initial_pheromone=0.1,  # 初始信息素浓度
        grid_size=(30, 30)  # 30x30 的网格
    )

    # 运行算法
    print("开始运行 iAA 算法...")
    coverage = iaa.run()

    # 可视化结果
    iaa.visualize_results()