import time
import numpy as np
import torch
from Agent import MAPPO
import configs
from timemap import Environment
from torch.distributions import Categorical
from src.DDQN.rescue import RescueAlgorithm
import os
import imageio
import matplotlib

matplotlib.use('Agg')  # 使用非交互模式
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import imageio.v2 as imageio
import os
import re
import pandas as pd

# === 新增：复访率与覆盖指标采集 ===
from src.metrics.revisit_metrics import RevisitCoverageMeter


# === 修改：覆盖率历史记录包装类 ===
class CoverageTracker:
    """包装RevisitCoverageMeter以追踪历史数据"""

    def __init__(self, meter):
        self.meter = meter
        self.coverage_history = []
        self.revisit_history = []
        self.energy_history = []

    def update(self, agents_pos, env):
        """更新覆盖率、复访率及其他指标"""
        self.meter.update(agents_pos, env)

        # 总自由单元格数量
        free_cells = np.sum(self.meter.free_mask)

        # 已观测单元格数（已被智能体访问过）
        observed_cells = np.sum(self.meter.visited) if hasattr(self.meter, 'visited') else 0

        # 覆盖率计算
        if free_cells > 0:
            coverage_rate = observed_cells / free_cells
        else:
            coverage_rate = 0.0
        self.coverage_history.append(coverage_rate)

        # 计算复访率
        if observed_cells > 0:
            if hasattr(self.meter, 'visits'):
                visits = np.array(self.meter.visits)  # 确保转换为 NumPy 数组
                revisit_cells = np.sum(visits > 1)
            else:
                revisit_cells = 0
            revisit_rate = revisit_cells / observed_cells
        else:
            revisit_rate = 0.0
        self.revisit_history.append(revisit_rate)

        # 能量消耗记录
        self.energy_history.append(self.meter.energy if hasattr(self.meter, 'energy') else 0.0)

    def save_plots(self, *args, **kwargs):
        """代理保存图表方法"""
        return self.meter.save_plots(*args, **kwargs)

    def save_summary_csv(self, *args, **kwargs):
        """代理保存CSV方法"""
        return self.meter.save_summary_csv(*args, **kwargs)


def _is_line_of_sight_blocked(x1, y1, x2, y2, env_map):
    """检查两点之间的视线是否被障碍物阻挡"""
    # 使用Bresenham算法生成两点之间的直线
    points = _bresenham_line(x1, y1, x2, y2)

    for px, py in points:
        # 如果遇到障碍物（但不包括目标点本身）
        if env_map[px, py] == 1 and (px != x2 or py != y2):
            return False
    return True


def get_local_vision(x, y, view_radius=5, grid_size=(60, 60)):
    """
    获取智能体视野范围（11x11区域），返回所有视野范围内的坐标
    :param x: 智能体的x坐标
    :param y: 智能体的y坐标
    :param view_radius: 视野半径（默认5，表示一个11x11的区域）
    :param grid_size: 网格大小（默认60x60）
    :return: 视野内所有坐标 [(x1, y1), (x2, y2), ...]
    """
    vision = []

    # 计算视野范围（11x11区域）
    for dx in range(-view_radius, view_radius + 1):
        for dy in range(-view_radius, view_radius + 1):
            nx, ny = x + dx, y + dy
            # 确保坐标在网格范围内
            if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]:
                vision.append((nx, ny))

    return vision


def plot_heatmap(agent_0_positions, agent_1_positions, agent_0_vision, agent_1_vision, env,
                 frame_save_path='heatmap_two_agents_with_vision.png', grid_size=(60, 60)):
    """
    绘制两个智能体的热力图，基于它们的位置数据和视野范围生成热力图
    :param agent_0_positions: 智能体0的位置列表 [(x1, y1), (x2, y2), ...]
    :param agent_1_positions: 智能体1的位置列表 [(x1, y1), (x2, y2), ...]
    :param agent_0_vision: 智能体0的视野范围 [(x1, y1), (x2, y2), ...]
    :param agent_1_vision: 智能体1的视野范围 [(x1, y1), (x2, y2), ...]
    :param frame_save_path: 保存图像的路径
    :param grid_size: 热力图的网格大小 (默认60x60)
    """
    # 创建一个与网格大小相同的零矩阵作为热力图
    heatmap_0 = np.zeros(grid_size)
    heatmap_1 = np.zeros(grid_size)

    # 更新热力图：增加智能体位置的计数
    for (x, y) in agent_0_positions:
        if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
            heatmap_0[x, y] += 0.5  # 累计智能体0的轨迹
            for dx in range(-5, 6):  # x轴方向从-1到1
                for dy in range(-5, 6):  # y轴方向从-1到1
                    nx, ny = x + dx, y + dy  # 计算邻域坐标
                    # 判断是否越界，确保新的坐标在地图范围内
                    if _is_line_of_sight_blocked(x, y, nx, ny, env.map):
                        if 0 <= nx < env.map_size[0] and 0 <= ny < env.map_size[1] and env.map[nx, ny] != 1:
                            heatmap_0[nx, ny] += 0.5

    for (x, y) in agent_1_positions:
        if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
            heatmap_1[x, y] += 0.5  # 累计智能体1的轨迹
            for dx in range(-5, 6):  # x轴方向从-1到1
                for dy in range(-5, 6):  # y轴方向从-1到1
                    nx, ny = x + dx, y + dy  # 计算邻域坐标
                    # 判断是否越界，确保新的坐标在地图范围内
                    if _is_line_of_sight_blocked(x, y, nx, ny, env.map):
                        if 0 <= nx < env.map_size[0] and 0 <= ny < env.map_size[1] and env.map[nx, ny] != 1:
                            heatmap_1[nx, ny] += 0.5

    # 合并热力图：可以将两个热力图的数值叠加显示
    combined_heatmap = heatmap_0 / 250 + heatmap_1 / 250

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(combined_heatmap, cmap='hot', origin='upper', interpolation='nearest', vmin=0, vmax=25)

    # 设置标题和标签
    ax.set_title("PILOC-ph Heatmap")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Visit Frequency')

    # 保存当前热力图为图片
    plt.savefig(frame_save_path)
    plt.close(fig)
    print('热力图已保存')


def _bresenham_line(x1, y1, x2, y2):
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


def plot_agent_timestamp_map(agent_0_timestamp_map, frame_save_path='agent_0_timestamp_map.png'):
    """
    绘制 agent_0 的时间戳地图，按照指定规则绘制不同的区域
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # 创建一个 60x60 的空白图像
    image = np.zeros(agent_0_timestamp_map.shape)

    # 白色区域 (0到0.3之间)
    image[(agent_0_timestamp_map >= 0) & (agent_0_timestamp_map <= 0.3)] = 4  # 白色

    # 灰色区域 (0.5)
    image[agent_0_timestamp_map == 0.5] = 0.5  # 灰色

    # 黑色区域 (1)
    image[agent_0_timestamp_map == 1] = 1  # 黑色

    # 红色区域 (-4.5)
    image[agent_0_timestamp_map == -4.5] = 2  # 红色

    # 蓝色区域 (5.5)
    image[agent_0_timestamp_map == 5.5] = 3  # 蓝色

    # 绘制地图
    ax.imshow(image, cmap='gray', origin='upper', vmin=0, vmax=4)  # 设置范围，确保颜色映射正确

    # 设置标题和标签
    ax.set_title("Agent 0 Timestamp Map")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # 保存当前帧为图片
    plt.savefig(frame_save_path)
    plt.close(fig)


def repeat_check(repeat_mark, env, obs, rescue_policy):
    all_paths = {}
    if len(repeat_mark) != 0:
        for agent_id in repeat_mark:
            agent_index = int(agent_id.split('_')[1])  # 提取数字部分并转换为整数
            path = rescue_policy.rescue_action(agent_index, env, obs)
            # 将路径保存到字典中，键是智能体的ID，值是智能体的路径
            all_paths[agent_id] = path
    return all_paths


def plot_environment(env_map, episode_num, step_t, frame_save_path):
    fig, ax = plt.subplots(figsize=(6, 6))

    # 创建一个 60x60 的空白图像
    image = np.zeros(env_map.map.shape)

    # 黑色区域 (1)
    image[env_map.map == 1] = 1  # 黑色

    # 绘制地图
    ax.imshow(image, cmap='gray', origin='upper', vmin=0, vmax=3)  # 设置范围，确保颜色映射正确

    # 绘制 agents 的位置
    agents_pos = np.array(env_map.agents_pos)  # 获取代理的位置
    ax.scatter(agents_pos[:, 1], agents_pos[:, 0], c='red', label='Agents', marker='o', s=100)  # 红色圆点表示代理

    # 绘制 goals 的位置
    if len(env_map.goals_pos) != 0:
        goals_pos = np.array(env_map.goals_pos)  # 获取目标的位置
        ax.scatter(goals_pos[:, 1], goals_pos[:, 0], c='blue', label='Goals', marker='x', s=100)  # 蓝色叉号表示目标

    # 设置标题和标签
    ax.set_title("goalb Map")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # 保存当前帧为图片
    plt.savefig(frame_save_path)
    plt.close(fig)


def create_gif_from_images(image_folder, gif_filename='timestamp_map_simulation.gif', duration=0.1):
    images = []

    # 获取所有保存的图片文件路径，并确保按文件名顺序加载
    def natural_sort_key(filename):
        """通过提取文件名中的数字进行自然排序"""
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', filename)]

    # 获取图片路径并按数字顺序排序
    for filename in sorted(os.listdir(image_folder), key=natural_sort_key):
        if filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            images.append(imageio.imread(image_path))  # 使用 imageio.v2.imread

    # 如果没有图像，则跳过创建 GIF
    if len(images) == 0:
        print("No images found in the folder, skipping GIF creation.")
        return

    # 使用 imageio 将图像列表保存为 gif 文件
    imageio.mimsave(gif_filename, images, duration=duration)
    print(f"GIF saved to: {gif_filename}")


def select_action(two_channelmap, policy, env, env_agents, device):
    """选择动作，使用Softmax"""
    actions = {}
    action_log_pis = {}

    for agent_id in env_agents:
        # 获取agent的状态，假设状态是60x60的二维数组
        obstacle_map = two_channelmap[agent_id]['obstacle_map']  # 获取障碍物地图
        explore_map = two_channelmap[agent_id]['explore_map']  # 获取探索地图
        agent_intid = int(agent_id.split("_")[-1])
        x, y = env.agents_pos[agent_intid]
        obs = np.stack([obstacle_map, explore_map], axis=0)  # 生成一个3通道的状态
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # 转换为tensor，并将其移动到指定的device
        # .unsqueeze(0) 这将状态从 [60, 60] 变为 [1, 1, 60, 60]，即批次大小为1，通道数为1

        # 获取动作概率
        action_probs = policy.agents.get_action(obs_tensor, batch_size=1)
        action_probs = action_probs[1]  # 提取第二个张量
        # 使用Softmax选择动作
        dist = Categorical(probs=action_probs)
        action = dist.sample()  # 使用sample()进行采样，Softmax已隐含

        # 计算对应的log概率
        action_log_pi = dist.log_prob(action)

        actions[agent_id] = action.item()  # 获取动作的实际值
        action_log_pis[agent_id] = action_log_pi.item()  # 获取log概率

    return actions, action_log_pis


def initialize_obs(env, num_agents=configs.num_agents, map_size=(60, 60), ):
    """
    初始化每个智能体的观测为 60x60 的地图，初始值为 0.5

    num_agents: 智能体数量
    map_size: 地图的尺寸，默认60x60

    return: 一个字典，其中每个智能体有一个60x60的obs地图，值为0.5
    """
    obs = {}

    for i in range(num_agents):
        other_agents_positions = [pos for idx, pos in enumerate(env.get_agents_positions()) if
                                  idx != i]
        # 初始化每个智能体的obs地图，尺寸为60x60，所有元素初始为0.5
        agent_obs = np.full(map_size, 0.5, dtype=np.float32)
        x, y = env.agents_pos[i]  # 当前智能体的初始位置
        agent_obs[x, y] = -4.5
        for dx in range(-2, 3):  # x轴方向从-1到1
            for dy in range(-2, 3):  # y轴方向从-1到1
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


def plot_average_coverage_curve(all_episode_data, metrics_root, max_steps=250):
    """
    绘制所有回合的平均覆盖率曲线
    :param all_episode_data: 列表，每个元素是一个回合的覆盖率数据字典
    :param metrics_root: 保存图像的根目录
    :param max_steps: 最大步数（用于对齐）
    """
    # 收集所有回合的覆盖率数据
    all_coverage_curves = []

    for ep_data in all_episode_data:
        coverage_curve = ep_data['coverage_curve']
        # 如果某个回合的步数少于max_steps，则用最后一个值填充
        if len(coverage_curve) < max_steps:
            last_value = coverage_curve[-1] if len(coverage_curve) > 0 else 0
            coverage_curve = coverage_curve + [last_value] * (max_steps - len(coverage_curve))
        elif len(coverage_curve) > max_steps:
            coverage_curve = coverage_curve[:max_steps]
        all_coverage_curves.append(coverage_curve)

    # 转换为numpy数组以便计算
    all_coverage_curves = np.array(all_coverage_curves)

    # 计算平均值和标准差
    mean_coverage = np.mean(all_coverage_curves, axis=0)
    std_coverage = np.std(all_coverage_curves, axis=0)

    # 绘制平均覆盖率曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = np.arange(1, max_steps + 1)

    # 绘制均值曲线
    ax.plot(steps, mean_coverage, 'b-', linewidth=2, label='Mean Coverage')

    # 绘制标准差阴影区域
    ax.fill_between(steps,
                    mean_coverage - std_coverage,
                    mean_coverage + std_coverage,
                    alpha=0.3,
                    color='blue',
                    label='±1 Std Dev')

    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Coverage Rate', fontsize=12)
    ax.set_title('Average Coverage Rate over All Episodes (PILOC-ph)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_steps)
    ax.set_ylim(0, 1.0)

    # 保存图像
    save_path = os.path.join(metrics_root, 'average_coverage_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'平均覆盖率曲线已保存到: {save_path}')

    # 同时保存数据到CSV
    df = pd.DataFrame({
        'step': steps,
        'mean_coverage': mean_coverage,
        'std_coverage': std_coverage
    })
    csv_path = os.path.join(metrics_root, 'average_coverage_data.csv')
    df.to_csv(csv_path, index=False)
    print(f'平均覆盖率数据已保存到: {csv_path}')


def test():
    # 使用configs中的超参数

    device = torch.device(configs.DEVICE) if torch.cuda.is_available() else torch.device('cpu')
    trick = configs.TRICK
    # 初始化环境，这里假设你使用自己的环境（替换成你自己的环境代码）
    env = Environment(test=True)  # 请替换为您的自定义环境类
    dim_info = {f'agent_{i}': [9, 5] for i in range(configs.num_agents)}  # 这里是获取env的dim_info，如果是自己定义的环境，可能需要从环境中获取维度信息

    # 训练配置
    model_dir = r'E:/Desktop/代码/修改6--在修改5的基础上，所有网络使用相同的参数/path_to_save_model_Abl'  # 保存模型的路径
    success = 0
    episode_num = 0
    step = 0  # 记录总步数
    step_t = 0  # 记录当前回合的步数
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}
    obs = initialize_obs(env)
    two_channelmap = {}
    two_channelmap = env.two_channel_map(obs, two_channelmap)
    {agent: env.action_space(agent).seed(seed=100) for agent in env_agents}  # 给每个智能体的动作空间设置种子
    rescue_actions = {}
    t = 250  # 允许走250步
    E = 0  # 用于随机生成环境步数
    # 初始化轨迹记录
    agent_0_positions = []
    agent_1_positions = []

    # === 新增：指标输出目录 ===
    metrics_root = "metrics_out_ph"
    os.makedirs(metrics_root, exist_ok=True)
    per_ep_summary_csv = os.path.join(metrics_root, "summary.csv")

    # === 新增：用于存储所有回合的覆盖率数据 ===
    all_episode_data = []

    # 导入模型
    policy = MAPPO.load(dim_info, False, model_dir, trick=trick)
    rescue_policy = RescueAlgorithm(env.map_size, env, obs)
    # 创建文件夹存放帧图像
    temp_image_folder = 'framess'
    temp_image_folder_2 = 'frame'
    os.makedirs(temp_image_folder, exist_ok=True)
    os.makedirs(temp_image_folder_2, exist_ok=True)

    episode_step = 0
    res_t = 0
    res_step = 0

    # === 新增：统计变量 ===
    success_steps = []  # 存储每个成功案例的步数
    get_goals = []

    # === 修改：使用包装类追踪覆盖率历史 ===
    base_meter = RevisitCoverageMeter(free_mask=(env.map == 0), radius=5, window=100, use_los=True)
    tracker = CoverageTracker(base_meter)

    # 测试250个回合
    while E < 250:
        step += 1
        step_t += 1

        # 获取动作
        action, action_log_pi = select_action(two_channelmap, policy, env, env_agents, device)

        if len(rescue_actions) > 0:
            # 营救策略会不使用神经网络(将神经网络的输出覆盖掉)
            for agent_id, rescue_action_sequence in list(rescue_actions.items()):  # 使用 list() 创建副本
                # 将rescue_actions中的动作覆盖到actions字典中
                if rescue_action_sequence:
                    action[agent_id] = rescue_action_sequence[0]  # 替换原来的动作
                    rescue_action_sequence.pop(0)
                    res_step += 1
                    # 对应智能体使用救援策略时，action_log_pi可以设置为0，表示确定性动作
                    action_log_pi[agent_id] = 0
                if len(rescue_action_sequence) == 0:
                    del rescue_actions[agent_id]
        # 环境更新
        next_obs, reward, terminated, repeat_Mark, next_two_channelmap = env.step(action, obs, rescue_actions,
                                                                                  two_channelmap, E)

        # 获取当前智能体的位置
        agent_0_pos = env.agents_pos[0]  # 获取智能体0的位置
        agent_1_pos = env.agents_pos[1]  # 获取智能体1的位置
        agent_0_vision = []
        agent_1_vision = []

        # 保存位置
        agent_0_positions.append(agent_0_pos)
        agent_1_positions.append(agent_1_pos)
        # 获取智能体的视野范围（11x11区域）
        agent_0_x, agent_0_y = agent_0_pos
        agent_1_x, agent_1_y = agent_1_pos
        agent_0_vision_ = get_local_vision(agent_0_x, agent_0_y, view_radius=5)
        agent_1_vision_ = get_local_vision(agent_1_x, agent_1_y, view_radius=5)
        # 保存视野范围
        agent_0_vision += agent_0_vision_
        agent_1_vision += agent_1_vision_

        # === 修改：使用tracker更新并记录历史 ===
        tracker.update(env.agents_pos, env)

        # 更新obs
        done = {agent_id: terminated[agent_id] for agent_id in env_agents}
        done_bool = {agent_id: terminated[agent_id] for agent_id in env_agents}

        episode_reward = {agent_id: episode_reward[agent_id] + reward[agent_id] for agent_id in env_agents}
        obs = next_obs
        agent_0_timestamp_map = obs['agent_0']

        if len(repeat_Mark) != 0:
            rescue_actions_fu = repeat_check(repeat_Mark, env, obs, rescue_policy)
            rescue_actions.update(rescue_actions_fu)
            res_t += 1
            repeat_Mark = []  # 生成动作序列后，将其初始化
        # 完成目标 或者 达到最大环境步 ，这一个回合结束
        if any(done.values()) or step_t >= t:
            # 显示信息
            if any(done.values()):
                success += 1
                success_steps.append(step_t)
                get_goals.append(6)
            else:
                num_gaols = len(env.goals_pos)
                get_goals.append(6 - num_gaols)

            # === 修改：保存本回合指标与图表 ===
            ep_dir = os.path.join(metrics_root, f"ep_{episode_num:04d}")
            os.makedirs(ep_dir, exist_ok=True)
            tracker.save_plots(ep_dir, prefix=f"ep{episode_num:04d}")
            tracker.save_summary_csv(os.path.join(metrics_root, "summary.csv"), ep_idx=episode_num)

            # === 修改：从tracker获取覆盖率历史 ===
            episode_data = {
                'episode': episode_num,
                'coverage_curve': tracker.coverage_history.copy(),
                'final_coverage': tracker.coverage_history[-1] if len(tracker.coverage_history) > 0 else 0,
                'steps': step_t
            }
            all_episode_data.append(episode_data)

            E += 1
            print(
                f"回合 {episode_num}: 成功率={success / E:.3f}, 覆盖率={episode_data['final_coverage']:.3f}, 步数={step_t}")
            episode_num += 1
            env = Environment(test=True)  # 重置环境
            obs = initialize_obs(env)  # 重置时间戳地图
            two_channelmap = {}
            two_channelmap = env.two_channel_map(obs, two_channelmap)

            # === 修改：为新回合重建tracker ===
            base_meter = RevisitCoverageMeter(free_mask=(env.map == 0), radius=5, window=100, use_los=True)
            tracker = CoverageTracker(base_meter)

            episode_step += step_t
            step_t = 0

    # === 测试结束后，生成平均覆盖率曲线 ===
    print("\n正在生成平均覆盖率曲线...")
    if len(all_episode_data) > 0 and len(all_episode_data[0]['coverage_curve']) > 0:
        plot_average_coverage_curve(all_episode_data, metrics_root, max_steps=t)
    else:
        print("警告：没有足够的覆盖率数据来生成平均曲线")

    frame_save_path = os.path.join(temp_image_folder, f"heatmap_L2.png")
    plot_heatmap(agent_0_positions, agent_1_positions, agent_0_vision, agent_1_vision, env,
                 frame_save_path=frame_save_path)  # 生成热力图

    # === 新增：输出统计信息 ===
    print("\n" + "=" * 50)
    print("测试统计结果 (PILOC-ph):")
    print("=" * 50)
    print(f"总成功次数: {success}")
    print(f"成功率: {success / 250:.3f}")
    print(f"总步数: {episode_step}")
    print(f"平均每回合步数: {episode_step / 250:.2f}")
    print(f"救援触发次数: {res_t}")
    print(f"救援步数: {res_step}")

    if len(success_steps) > 0:
        steps_variance = np.var(success_steps)
        steps_mean = np.mean(success_steps)
        print(f"\n成功案例数量: {len(success_steps)}")
        print(f"平均步数: {steps_mean:.2f}")
        print(f"步数标准差: {np.sqrt(steps_variance):.2f}")
        print(f"步数方差: {steps_variance:.2f}")

    gaols_mean = np.mean(get_goals)
    print(f"平均捕获目标: {gaols_mean:.2f}")

    # 计算平均最终覆盖率
    final_coverages = [ep_data['final_coverage'] for ep_data in all_episode_data]
    mean_final_coverage = np.mean(final_coverages)
    std_final_coverage = np.std(final_coverages)
    print(f"\n平均最终覆盖率: {mean_final_coverage:.3f} ± {std_final_coverage:.3f}")
    print("=" * 50)

    # 创建 GIF - 时间戳地图的 GIF（可选）
    # create_gif_from_images(temp_image_folder_2, gif_filename='timestamp_map_simulation.gif')
    # create_gif_from_images(temp_image_folder, gif_filename='global_map_simulation.gif')
    # 清理临时文件夹（可选）
    # import shutil
    # shutil.rmtree(temp_image_folder)
    # shutil.rmtree(temp_image_folder_2)


if __name__ == '__main__':
    start_time = time.time()
    test()
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")