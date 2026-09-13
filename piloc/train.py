import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# Agent.py does `from normalization import ...`, which only resolves with src/DDQN on
# the path. Running `python train.py` from this directory puts only piloc/ there, so
# add it explicitly instead of requiring the operator to set PYTHONPATH.
_SRC_DDQN = str(Path(__file__).resolve().parent / "src" / "DDQN")
if _SRC_DDQN not in sys.path:
    sys.path.append(_SRC_DDQN)

import configs
from src.DDQN.Agent import MAPPO
from src.Map.timemap import Environment

SHARED_MODEL_PATTERN = re.compile(r"^shared_mappo_model_(\d+)_(\d+)\.pth$")

def repeat_check(repeat_mark,env,obs,rescue_policy):
    all_paths = {}
    if len(repeat_mark) != 0:
        for agent_id in repeat_mark:
            agent_index = int(agent_id.split('_')[1])  # 提取数字部分并转换为整数
            path = rescue_policy.rescue_action(agent_index,env)
            # 将路径保存到字典中，键是智能体的ID，值是智能体的路径
            all_paths[agent_id] = path
    return all_paths

def get_local_ph(global_ph, x, y, view_radius=5):
    """ 根据智能体当前位置裁剪局部信息素视野 """
    h, w = global_ph.shape

    # 计算裁剪边界
    x_min = x - view_radius
    x_max = x + view_radius + 1
    y_min = y - view_radius
    y_max = y + view_radius + 1

    # 检查是否超出边界
    x_min_outside = x_min < 0
    x_max_outside = x_max > 59
    y_min_outside = y_min < 0
    y_max_outside = y_max > 59

    # 修正边界值
    x_min = max(0, x_min)
    x_max = min(60, x_max)
    y_min = max(0, y_min)
    y_max = min(60, y_max)

    local_ph = global_ph[x_min:x_max, y_min:y_max]

    # 计算需要填充的数量
    target_size = 2 * view_radius + 1
    pad_top = target_size - (x_max - x_min) if x_min_outside else 0
    pad_bottom = target_size - (x_max - x_min) if x_max_outside else 0
    pad_left = target_size - (y_max - y_min) if y_min_outside else 0
    pad_right = target_size - (y_max - y_min) if y_max_outside else 0

    # 进行填充
    return np.pad(local_ph, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

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
        pheromone_map = get_local_ph(env.global_pheromone,x,y)
        pheromone_map_tensor = torch.tensor(pheromone_map, dtype=torch.float32).unsqueeze(0).to(device)  # 转换为tensor，并将其移动到指定的device
        obs = np.stack([obstacle_map, explore_map], axis=0)  # 生成一个3通道的状态
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # 转换为tensor，并将其移动到指定的device
        # .unsqueeze(0) 这将状态从 [60, 60] 变为 [1, 1, 60, 60]，即批次大小为1，通道数为1

        # 获取动作概率
        action_probs = policy.agents.get_action(obs_tensor, pheromone_map_tensor,batch_size=1)
        action_probs = action_probs[1]  # 提取第二个张量
        # 使用Softmax选择动作
        dist = Categorical(probs=action_probs)
        action = dist.sample()  # 使用sample()进行采样，Softmax已隐含

        # 计算对应的log概率
        action_log_pi = dist.log_prob(action)

        actions[agent_id] = action.item()  # 获取动作的实际值
        action_log_pis[agent_id] = action_log_pi.item()  # 获取log概率

    return actions, action_log_pis
def _is_line_of_sight_blocked( x1, y1, x2, y2, env_map):
    """检查两点之间的视线是否被障碍物阻挡"""
    # 使用Bresenham算法生成两点之间的直线
    points = _bresenham_line(x1, y1, x2, y2)

    for px, py in points:
        # 如果遇到障碍物（但不包括目标点本身）
        if env_map[px, py] == 1 and (px != x2 or py != y2):
            return False
    return True


def _bresenham_line( x1, y1, x2, y2):
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
                    if _is_line_of_sight_blocked(x,y,nx,ny,env.map):
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

def resolve_device(name: str = "auto") -> torch.device:
    """Resolve --device without assuming a GPU exists.

    ``auto`` picks cuda when it is really available. Asking for cuda on a machine
    without it is an error rather than a silent fallback, because a run that quietly
    trains on CPU is many times slower and the operator should know which happened.
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"--device {name} was requested but CUDA is not available")
    if name not in {"cpu"} and not name.startswith("cuda"):
        raise ValueError(f"--device must be auto, cpu or cuda[:N], got {name!r}")
    return torch.device(name)


def prepare_output_dir(root: str | os.PathLike[str]) -> dict[str, Path]:
    """Create one experiment directory holding checkpoints, logs and metrics."""
    base = Path(os.fspath(root))
    directories = {
        "root": base,
        "checkpoints": base / "checkpoints",
        "tb_logs": base / "tb_logs",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def should_save_checkpoint(episode_num: int, every: int) -> bool:
    """True only on a real episode boundary that is a multiple of ``every``.

    The previous check was ``episode_num % 1000 == 0`` evaluated on every step of the
    loop. Since ``episode_num`` starts at 0 and ``0 % 1000 == 0``, it wrote a
    checkpoint on every single step of the first episode, and again throughout every
    1000th episode.
    """
    if every < 1:
        raise ValueError(f"the checkpoint interval must be at least 1, got {every}")
    return episode_num > 0 and episode_num % every == 0


def git_commit() -> str | None:
    """The commit this run was trained from, or None outside a git checkout."""
    repository = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def write_run_meta(
    directory: str | os.PathLike[str],
    *,
    settings: dict,
    seed: int,
    device: torch.device,
    dim_info: dict,
) -> Path:
    """Record everything needed to identify or reproduce a run.

    Checkpoints alone cannot answer which commit, seed, device or observation
    dimensions produced them, which is exactly what makes a training result
    uninterpretable later.
    """
    meta = {
        "git_commit": git_commit(),
        "seed": int(seed),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "dim_info": {key: list(value) for key, value in dim_info.items()},
        "settings": settings,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = Path(os.fspath(directory)) / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def episode_from_checkpoint_name(path: str | os.PathLike[str]) -> int:
    """Read the episode number back out of a ``shared_mappo_model_*_*.pth`` name."""
    match = SHARED_MODEL_PATTERN.match(Path(os.fspath(path)).name)
    return int(match.group(2)) if match else 0


def resume_policy(policy: MAPPO, checkpoint: str | os.PathLike[str], device: torch.device) -> int:
    """Load weights strictly and return the episode the run resumes from.

    ``MAPPO.load`` is deliberately not used: it hardcodes ``device='cuda'`` and the
    filename ``shared_mappo_model_21_9232.pth``, so it cannot resume an arbitrary run.
    A partial load is refused rather than accepted, because silently missing layers
    produce a policy that runs and decides with the wrong weights.
    """
    path = Path(os.fspath(checkpoint))
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)

    agents = policy.agents
    targets = {"full_model": agents.full_model, "actor": agents.actor, "critic": agents.critic}
    missing = [name for name in targets if name not in state]
    if missing:
        raise KeyError(f"{path} does not contain {', '.join(missing)}; cannot resume strictly")

    for name, module in targets.items():
        module.load_state_dict(state[name], strict=True)
        module.eval()
    if "optimizer" in state:
        agents.optimizer.load_state_dict(state["optimizer"])
    return int(state.get("episode", episode_from_checkpoint_name(path)))


class SearchEpisodeMetrics:
    """The search-side indicators for one episode: what was found, how fast, how far.

    Coverage is measured from the per-agent observation maps because
    ``env.global_timestamp_maps`` is initialised once in the constructor and never
    updated, so it stays uniformly unknown forever.
    """

    def __init__(self) -> None:
        self.reset(0)

    def reset(self, num_goals: int) -> None:
        self.initial_goals = max(0, int(num_goals))
        self.first_discovery_step: int | None = None
        self.steps = 0
        self.moves = 0
        self.repeat_steps = 0
        self._previous_positions: list[tuple[int, int]] | None = None

    def observe(self, env, repeat_marks) -> None:
        self.steps += 1
        positions = [(int(p[0]), int(p[1])) for p in env.agents_pos]
        if self._previous_positions is not None:
            self.moves += sum(
                1 for before, after in zip(self._previous_positions, positions) if before != after
            )
        self._previous_positions = positions
        if repeat_marks:
            self.repeat_steps += 1
        if self.first_discovery_step is None and len(env.goals_pos) < self.initial_goals:
            self.first_discovery_step = self.steps

    def as_dict(self, env, obs) -> dict:
        remaining = len(env.goals_pos)
        discovered = max(0, self.initial_goals - remaining)
        obstacles = np.asarray(env.map)
        free = int(np.count_nonzero(obstacles != 1))
        unseen = np.ones(obstacles.shape, dtype=bool)
        for agent_obs in obs.values():
            unseen &= np.asarray(agent_obs) == 0.5
        return {
            "targets_discovered": discovered,
            "discovery_rate": discovered / self.initial_goals if self.initial_goals else 0.0,
            "first_discovery_step": self.first_discovery_step,
            "coverage": float(np.count_nonzero(~unseen & (obstacles != 1)) / free) if free else 0.0,
            "travel": self.moves,
            "repeat_ratio": self.repeat_steps / self.steps if self.steps else 0.0,
            "episode_steps": self.steps,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="piloc.train",
        description="Train the PILOC multi-agent area-search policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("runs", "piloc_search"),
        help="experiment directory for checkpoints, TensorBoard logs and metrics",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the env, the RNGs and action sampling")
    parser.add_argument("--num-agents", type=int, default=configs.num_agents)
    parser.add_argument("--num-goals", type=int, default=configs.num_goals)
    parser.add_argument("--episodes", type=int, default=configs.MAX_EPISODES,
                        help="stop after this many episodes")
    parser.add_argument("--save-every", type=int, default=1000, help="checkpoint interval in episodes")
    parser.add_argument("--log-every", type=int, default=100, help="how often to record episode metrics")
    parser.add_argument("--resume", default=None, help="a shared_mappo_model_*.pth to continue from")
    parser.add_argument("--no-tensorboard", action="store_true", help="skip TensorBoard logging")
    parser.add_argument(
        "--track-reward-terms",
        action="store_true",
        help="record the per-step reward decomposition in the episode metrics",
    )
    return parser


def build_search_environment(args: argparse.Namespace) -> Environment:
    """Build the search environment from the run's own command-line settings.

    ``main`` builds it once before training and rebuilds it at every episode
    boundary. The rebuild used to pass ``num_agents`` alone, so a non-default
    ``--num-goals`` was silently replaced by ``configs.num_goals`` from the second
    episode on: the curriculum then trained against a different world than the one
    ``write_run_meta`` recorded. Both sites go through here so the two cannot drift
    apart again.
    """
    return Environment(
        num_agents=args.num_agents,
        num_goals=args.num_goals,
        track_reward_terms=args.track_reward_terms,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # 使用configs中的超参数
    gamma = configs.GAMMA
    actor_lr = configs.ACTOR_LR
    critic_lr = configs.CRITIC_LR
    horizon = configs.HORIZON
    clip_param = configs.CLIP_PARAM
    k_epochs = configs.K_EPOCHS
    minibatch_size = configs.MINIBATCH_SIZE
    lmbda = configs.LMBDA
    huber_delta = configs.HUBER_DELTA
    device = resolve_device(args.device)
    trick = dict(configs.TRICK)

    directories = prepare_output_dir(args.output_dir)
    model_dir = str(directories["checkpoints"])
    model_dir1 = model_dir

    # One seed drives every random source the episode depends on, so a run is reproducible.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start_time = time.time()
    writer = None if args.no_tensorboard else SummaryWriter(log_dir=str(directories["tb_logs"]))

    env = build_search_environment(args)
    # 121 是 obs_radius=5 的信息素窗口 (2*5+1)^2，4 是动作数；网络形状在 Actor/Critic 内部固定，
    # dim_info 只决定 buffer 的观测宽度。
    dim_info = {f'agent_{i}': [121, 4] for i in range(args.num_agents)}

    # 初始化算法
    policy = MAPPO(
        dim_info,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        device=device,
        horizon=horizon,
        trick=trick,
        num_agent=args.num_agents,
    )

    write_run_meta(
        directories["root"],
        settings={
            "num_agents": args.num_agents,
            "num_goals": args.num_goals,
            "episodes": args.episodes,
            "save_every": args.save_every,
            "log_every": args.log_every,
            "resume": args.resume,
            "track_reward_terms": args.track_reward_terms,
            "gamma": gamma,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "horizon": horizon,
            "clip_param": clip_param,
            "k_epochs": k_epochs,
            "minibatch_size": minibatch_size,
            "lmbda": lmbda,
            "huber_delta": huber_delta,
            "trick": trick,
            "reward_fn": dict(configs.reward_fn),
            "obs_radius": configs.obs_radius,
            "pheromone_decay": configs.pheromone_decay,
        },
        seed=args.seed,
        device=device,
        dim_info=dim_info,
    )

    episode_num = 0
    if args.resume is not None:
        episode_num = resume_policy(policy, args.resume, device)
        print(f"resumed from {args.resume} at episode {episode_num}")

    step = 0 # 记录总步数
    step_t = 0 # 记录当前回合的步数
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}
    reward_term_sums = {agent_id: {} for agent_id in env_agents}
    obs = initialize_obs(env)
    two_channelmap = {}
    two_channelmap = env.two_channel_map(obs,two_channelmap)
    for agent in env_agents:  # 给每个智能体的动作空间设置种子
        env.action_space(agent).seed(seed=args.seed)
    rescue_actions = {}
    reward_max_M = -float('inf')  # 初始化为很小的负数，便于更新第一回合的奖励
    t = 10
    Curriculum_Training = 0 # 用于记录有多少个回合智能体的奖励函数没有超过R_max
    E = 10 # 用于随机生成环境步数,从40步开始训练
    step_reward_t = 0
    episode_reward_agent = 0 # 回合奖励初始化为0
    obs1 = {}
    obs2 = {}
    next_obs1 = {}
    next_obs2 = {}
    obs_pheromone = {}
    next_obs_pheromone = {}
    metrics = SearchEpisodeMetrics()
    metrics.reset(len(env.goals_pos))
    metrics_path = directories["root"] / "episode_metrics.jsonl"

    # 训练循环
    while E <= 250 and episode_num < args.episodes:
        step += 1
        step_t +=1
        step_reward_t += 1
        # 获取动作
        action, action_log_pi = select_action(two_channelmap, policy, env,env_agents, device)
        for agentid,position in enumerate(env.agents_pos):
            x,y = position
            agent_id = f"agent_{agentid}"
            obs_pheromone[agent_id] = get_local_ph(env.global_pheromone,x,y)

        # 环境更新
        next_obs, reward, terminated, repeat_Mark,next_two_channelmap= env.step(action,obs,rescue_actions,two_channelmap,E)
        metrics.observe(env, repeat_Mark)
        if args.track_reward_terms:
            for agent_id, terms in env.last_reward_terms.items():
                bucket = reward_term_sums[agent_id]
                for name, value in terms.items():
                    bucket[name] = bucket.get(name, 0.0) + value
        for agentid,position in enumerate(env.agents_pos):
            x,y = position
            agent_id = f"agent_{agentid}"
            next_obs_pheromone[agent_id] = get_local_ph(env.global_pheromone,x,y)

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
        policy.add(obs1,obs2,obs_pheromone, action, reward, next_obs1,next_obs2, next_obs_pheromone,done_bool, action_log_pi, done)

        episode_reward = {agent_id: episode_reward[agent_id] + reward[agent_id] for agent_id in env_agents}
        obs = next_obs
        two_channelmap = next_two_channelmap
        # 完成目标 或者 达到最大环境步 ，这一个回合结束
        if any(done.values()) or step_t >= t:
            # 先用回合结束时的环境与观测统计搜索指标，环境重建之后就拿不到了
            episode_metrics = metrics.as_dict(env, obs)
            episode_metrics.update(
                {
                    "episode": episode_num + 1,
                    "env_steps": E,
                    "curriculum_stall": Curriculum_Training,
                }
            )
            episode_metrics.update(
                {f"reward/{agent_id}": float(episode_reward[agent_id]) for agent_id in env_agents}
            )
            if args.track_reward_terms:
                for agent_id, terms in reward_term_sums.items():
                    for name, value in terms.items():
                        episode_metrics[f"reward_term/{agent_id}/{name}"] = float(value)

            if (episode_num + 1) % args.log_every == 0:
                print(
                    f"Episode {episode_num + 1} | E={E} stall={Curriculum_Training} | "
                    f"discovered={episode_metrics['targets_discovered']} "
                    f"coverage={episode_metrics['coverage']:.3f} "
                    f"first_find={episode_metrics['first_discovery_step']} | "
                    f"Reward: {episode_reward}"
                )
                for agent_id in env_agents:
                    train_return[agent_id].append(episode_reward[agent_id])
                if writer is not None:
                    for key, value in episode_metrics.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(f"search/{key}", value, episode_num + 1)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(episode_metrics, ensure_ascii=False) + "\n")

            episode_num += 1
            env = build_search_environment(args) # 重置环境
            obs = initialize_obs(env)  # 重置时间戳地图
            metrics.reset(len(env.goals_pos))

            # 保存模型：只在回合边界，且跳过 episode 0（0 % every == 0 会让第一回合每步都存盘）
            if should_save_checkpoint(episode_num, args.save_every):
                policy.save(model_dir, episode_num)

            # 用于计算该回合奖励
            for agent_reward in episode_reward.values() :
                episode_reward_agent += agent_reward
            # 如果r_max更新，重新计算有多少轮没有更新
            if episode_reward_agent > reward_max_M:
                reward_max_M = episode_reward_agent
                Curriculum_Training = 0 # r_max更新时将其置0

            episode_reward = {agent_id: 0 for agent_id in env_agents}
            reward_term_sums = {agent_id: {} for agent_id in env_agents}
            episode_reward_agent = 0
            step_t = 0
            t = random.randint(E-5, min(E+5, 250))
            Curriculum_Training += 1 # 用于记录r_max有没有变化
            # 如果200个回合没有r_max更新，则增加E，增加环境时间步
            if Curriculum_Training >= 200:
                E += 10
                reward_max_M = -float('inf')
                Curriculum_Training = 0 #

        # 每隔一定步数更新网络
        if step % horizon == 0:
            policy.learn(minibatch_size, gamma, lmbda, clip_param, k_epochs, entropy_coefficient=0.01, huber_delta=huber_delta)

    print(f"Training completed in {time.time() - start_time} seconds")
    policy.save(model_dir1,episode_num)

    # 保存训练结果
    train_return_ = np.array([train_return[agent_id] for agent_id in env.agents])
    np.save(os.path.join(model_dir, "train_return.npy"), train_return_)
    if writer is not None:
        writer.close()
    print(f"checkpoints, TensorBoard logs and metrics are in {directories['root']}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())


