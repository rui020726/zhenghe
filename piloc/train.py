import argparse
import json
import os
import pickle
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
from src.Map.timemap import Environment, action_list as ACTION_LIST
from src.curriculum import CurriculumGate, default_stages

SHARED_MODEL_PATTERN = re.compile(r"^shared_mappo_model_(\d+)_(\d+)\.pth$")

# repeat_check / rescue actions are deliberately NOT wired into training: a rescue
# action would overwrite the PPO-sampled action while the buffer still stores the
# policy's log-prob of the sampled one, making the recorded transition inconsistent
# with the executed action. piloc/test.py keeps the rescue path for evaluation only.

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

def legal_action_mask(env, agent_index):
    """每 agent 的合法动作掩码（1=合法，0=越界或目标格为障碍）。

    边界与相邻障碍始终处于局部视野内，掩码不引入真值泄漏。掩码必须同时
    用于训练采样、argmax 验证与 PPO 更新重算 log-prob（buffer 逐条保存）。
    四向全不合法（被障碍围住）时返回全 1：碰撞只会让 agent 停在原地，
    由碰撞惩罚处理，比给 softmax 一个全 0 掩码安全。
    """
    x, y = (int(v) for v in env.agents_pos[agent_index])
    mask = np.zeros(ACTION_LIST.shape[0], dtype=np.float32)
    for action in range(ACTION_LIST.shape[0]):
        dx, dy = (int(v) for v in ACTION_LIST[action])
        nx, ny = x + dx, y + dy
        if 0 <= nx < env.map_size[0] and 0 <= ny < env.map_size[1] and env.map[nx, ny] != 1:
            mask[action] = 1.0
    if mask.sum() == 0.0:
        mask[:] = 1.0
    return mask


def apply_action_mask(action_probs, mask, device):
    """掩码 + 归一化：非法动作概率清零后重新归一，供 Categorical 使用。"""
    mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=device)
    masked = action_probs * mask_tensor
    return masked / masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def select_action(two_channelmap, policy, env,env_agents, device):
    """选择动作：策略概率过合法动作掩码后再采样（训练用）。"""
    actions = {}
    action_log_pis = {}
    legal_masks = {}

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

        # 获取动作概率（不采样，不消耗 torch 采样 RNG）
        action_probs = policy.agents.get_probs(obs_tensor, pheromone_map_tensor, batch_size=1)
        mask = legal_action_mask(env, agent_intid)
        masked_probs = apply_action_mask(action_probs, mask, device)
        # 使用Softmax选择动作
        dist = Categorical(probs=masked_probs)
        action = dist.sample()  # 使用sample()进行采样，Softmax已隐含

        # 计算对应的log概率
        action_log_pi = dist.log_prob(action)

        actions[agent_id] = action.item()  # 获取动作的实际值
        action_log_pis[agent_id] = action_log_pi.item()  # 获取log概率
        legal_masks[agent_id] = mask

    return actions, action_log_pis, legal_masks
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


def save_trainer_state(path, *, episode_num, global_step, best_validation_score, gate, early_stop_triggered=False):
    """把断点恢复需要的全部训练状态落盘（pickle，含 RNG 状态）。

    旧 checkpoint 只存网络权重与 episode 号：课程阶段、连续通过计数、
    全局步数、最优验证分、Python/NumPy/Torch 的 RNG 状态全部丢失，任何
    resume 都事实上从 stage 0 重来且后续采样序列偏移。这里一次性补齐。
    """
    state = {
        "episode_num": int(episode_num),
        "global_step": int(global_step),
        "best_validation_score": float(best_validation_score),
        "early_stop_triggered": bool(early_stop_triggered),
        "gate": gate.state_dict(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    with open(path, "wb") as handle:
        pickle.dump(state, handle)


def load_trainer_state(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def restore_trainer_state(state, gate):
    """把 ``load_trainer_state`` 读回的状态应用到训练循环与门控上。"""
    gate.load_state_dict(state["gate"])
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch"])
    if state.get("rng_torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng_torch_cuda"])


class SearchEpisodeMetrics:
    """The search-side indicators for one episode: what was found, how fast, how far.

    Coverage is measured from the per-agent observation maps because
    ``env.global_timestamp_maps`` is initialised once in the constructor and never
    updated, so it stays uniformly unknown forever.

    Collision, short-cycle and pure-re-coverage are tracked separately (from
    ``env.last_step_stats``) so a rising "repeat" signal can be attributed to the
    right cause instead of all three being conflated.
    """

    def __init__(self) -> None:
        self.reset(0)

    def reset(self, num_goals: int) -> None:
        self.initial_goals = max(0, int(num_goals))
        self.first_discovery_step: int | None = None
        self.steps = 0
        self.moves = 0
        self.repeat_steps = 0
        self.collision_steps = 0
        self.no_new_cell_steps = 0
        self.coverage_sum = 0.0
        self._previous_positions: list[tuple[int, int]] | None = None
        # 回合起始覆盖率：第一个 observe 时从 env.last_step_stats 的
        # coverage_before 取（该值就是本步观测更新前的并集覆盖率）。课程门控
        # 用 coverage_gain = C_T - C_0，不受初始视野覆盖率（实测 ~0.044）影响。
        self._initial_coverage: float | None = None

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
        stats = getattr(env, "last_step_stats", {}) or {}
        if self._initial_coverage is None:
            self._initial_coverage = float(stats.get("coverage_before", 0.0))
        if any(stats.get("collisions", {}).values()):
            self.collision_steps += 1
        if stats.get("new_cells", 1) == 0:
            self.no_new_cell_steps += 1
        self.coverage_sum += float(stats.get("coverage_after", 0.0))

    def as_dict(self, env, obs) -> dict:
        remaining = len(env.goals_pos)
        discovered = max(0, self.initial_goals - remaining)
        obstacles = np.asarray(env.map)
        free = int(np.count_nonzero(obstacles != 1))
        unseen = np.ones(obstacles.shape, dtype=bool)
        for agent_obs in obs.values():
            unseen &= np.asarray(agent_obs) == 0.5
        coverage = float(np.count_nonzero(~unseen & (obstacles != 1)) / free) if free else 0.0
        return {
            "targets_discovered": discovered,
            "discovery_rate": discovered / self.initial_goals if self.initial_goals else 0.0,
            "first_discovery_step": self.first_discovery_step,
            "coverage": coverage,
            # coverage AUC: mean coverage over the episode, so a curve that rises
            # early scores higher than one that only reaches the same end value late.
            "coverage_auc": self.coverage_sum / self.steps if self.steps else 0.0,
            # 回合内新增覆盖 C_T - C_0：课程门控只看这个，不看含初始视野的总 coverage。
            "coverage_gain": coverage - (self._initial_coverage or 0.0),
            "travel": self.moves,
            # three distinct signals, no longer one conflated "repeat":
            "repeat_ratio": self.repeat_steps / self.steps if self.steps else 0.0,
            "collision_rate": self.collision_steps / self.steps if self.steps else 0.0,
            "recoverage_rate": self.no_new_cell_steps / self.steps if self.steps else 0.0,
            "episode_steps": self.steps,
        }


def default_curriculum_stages():
    """Episode-limit ladder from 50 to 250 in steps of 25.

    起点 E=50：E=10 时回合太短，指标主要由初始视野决定（随机策略 coverage
    已达 0.068），无法区分策略好坏；E=50 给随机策略的 coverage_gain 约 0.1，
    给可学习策略留出了明确的上升空间。
    """
    return list(range(50, 251, 25))


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
    # --- 奖励与折扣 ---
    parser.add_argument(
        "--reward-mode",
        choices=("legacy", "v2", "potential"),
        default=configs.reward_mode,
        help="reward function; legacy keeps the old formula for A/B control",
    )
    parser.add_argument("--gamma", type=float, default=configs.GAMMA,
                        help="discount factor; V2 training recommends 0.99")
    parser.add_argument("--lmbda", type=float, default=configs.LMBDA, help="GAE lambda")
    parser.add_argument("--potential-beta", type=float, default=configs.POTENTIAL_BETA,
                        help="potential-based shaping coefficient (potential mode)")
    # --- 课程门控 ---
    parser.add_argument(
        "--curriculum",
        choices=("gate", "off"),
        default="gate",
        help="gate: task-metric curriculum; off: fixed episode limit",
    )
    parser.add_argument("--fixed-episode-limit", type=int, default=configs.T_MAX,
                        help="episode step limit when --curriculum off")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="validate on the fixed map every N training episodes")
    parser.add_argument("--required-passes", type=int, default=3,
                        help="consecutive passing validations required to advance a stage")
    parser.add_argument("--validation-episodes", type=int, default=20,
                        help="fixed-seed validation episodes averaged into each validation score")
    parser.add_argument("--validation-seed", type=int, default=None,
                        help="fixed seed for the deterministic validation rollout "
                             "(defaults to seed + 1_000_000_007, kept off the training RNG)")
    # --- 早停：确定性策略必须在窗口内超过 raw-random 并接近 legal-random ---
    parser.add_argument("--early-stop", dest="early_stop", action="store_true", default=True,
                        help="stop early when the deterministic policy cannot beat the "
                             "raw-random baseline within the window")
    parser.add_argument("--no-early-stop", dest="early_stop", action="store_false",
                        help="disable the random-baseline early stop")
    parser.add_argument("--early-stop-window", type=int, default=500,
                        help="episode by which the policy must clear the random baselines")
    parser.add_argument("--early-stop-margin", type=float, default=0.05,
                        help="'close to legal-random' tolerance on validation coverage")
    return parser


def build_search_environment(args: argparse.Namespace, seed: int = None) -> Environment:
    """Build the search environment from the run's own command-line settings.

    ``main`` builds it once before training and rebuilds it at every episode
    boundary. The rebuild used to pass ``num_agents`` alone, so a non-default
    ``--num-goals`` was silently replaced by ``configs.num_goals`` from the second
    episode on: the curriculum then trained against a different world than the one
    ``write_run_meta`` recorded. Both sites go through here so the two cannot drift
    apart again.

    ``seed`` pins the environment's own RNG stream; the validation rollout passes a
    fixed seed so the training process can never consume it.
    """
    return Environment(
        num_agents=args.num_agents,
        num_goals=args.num_goals,
        track_reward_terms=args.track_reward_terms,
        reward_mode=args.reward_mode,
        gamma=args.gamma,
        t_max=configs.T_MAX,
        potential_beta=args.potential_beta,
        seed=seed,
    )


def seed_all(seed: int) -> None:
    """Seed every random source a rollout touches.

    Python's ``random``, NumPy, Torch CPU and Torch CUDA all draw during training
    (env layout, vision noise, action sampling, weight init). Missing any one of
    them is why two runs on the same seed diverged. CUDA kernels can still be
    non-deterministic (cudnn.benchmark is on in the policy module), so this
    guarantees reproducibility on CPU and only statistical reproducibility on GPU;
    the run metadata records which device was used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_action_deterministic(two_channelmap, policy, env, env_agents, device):
    """Argmax over the masked policy probabilities -- no sampling, no RNG consumed.

    与训练采样、PPO 更新使用同一份合法动作掩码，保证验证评估的动作分布
    与训练时一致；内部只调 ``get_probs``，不会触碰 torch 采样 RNG。
    """
    actions = {}
    for agent_id in env_agents:
        obstacle_map = two_channelmap[agent_id]["obstacle_map"]
        explore_map = two_channelmap[agent_id]["explore_map"]
        agent_intid = int(agent_id.split("_")[-1])
        x, y = env.agents_pos[agent_intid]
        pheromone_map = get_local_ph(env.global_pheromone, x, y)
        pheromone_tensor = torch.tensor(pheromone_map, dtype=torch.float32).unsqueeze(0).to(device)
        obs = np.stack([obstacle_map, explore_map], axis=0)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = policy.agents.get_probs(obs_tensor, pheromone_tensor, batch_size=1)
        masked = apply_action_mask(probs, legal_action_mask(env, agent_intid), device)
        actions[agent_id] = int(torch.argmax(masked[0]).item())
    return actions


def run_random_baseline(args, episode_limit: int, seed: int, legal_only: bool) -> dict:
    """均匀随机 rollout 基线，跑在固定验证 seed 上（早停参照系）。

    legal_only=False：raw-random，4 个动作均匀采样（含碰撞，下界参照）。
    legal_only=True ：legal-random，只在合法动作上均匀采样（无碰撞上界参照）。
    使用独立的 ``np.random.RandomState``，绝不触碰训练 RNG 流。
    """
    rng = np.random.RandomState(seed)
    env_agents = [f"agent_{i}" for i in range(args.num_agents)]
    totals: dict = {}
    episodes = max(1, int(args.validation_episodes))
    for episode in range(episodes):
        env = build_search_environment(args, seed=seed + episode)
        obs = initialize_obs(env)
        two_channelmap = env.two_channel_map(obs, {})
        metrics = SearchEpisodeMetrics()
        metrics.reset(len(env.goals_pos))
        step_t = 0
        while step_t < episode_limit:
            step_t += 1
            if legal_only:
                actions = {}
                for agent_id in env_agents:
                    mask = legal_action_mask(env, int(agent_id.split("_")[1]))
                    legal = np.flatnonzero(mask > 0).tolist()
                    actions[agent_id] = int(rng.choice(legal)) if legal else 0
            else:
                actions = {agent_id: int(rng.randint(4)) for agent_id in env_agents}
            obs, _reward, terminated, marks, two_channelmap = env.step(
                actions, obs, {}, two_channelmap, episode_limit
            )
            metrics.observe(env, marks)
            if any(terminated.values()):
                break
        report = metrics.as_dict(env, obs)
        for key, value in report.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
    return {key: value / episodes for key, value in totals.items()}


def validation_score(report: dict) -> float:
    """A single task-metric score for choosing the best checkpoint.

    Deliberately built only from coverage / discovery / collision / repeat -- never
    the cumulative reward -- so the best model is the one that searches well, not
    the one that happened to collect reward on a long episode.
    """
    return (
        report["coverage_auc"]
        + report["discovery_rate"]
        - report["collision_rate"]
        - 0.5 * report["recoverage_rate"]
    )


def run_validation(args, policy, device, episode_limit: int, seed: int) -> dict:
    """Average deterministic task metrics over a few fixed-seed validation episodes.

    The environment is rebuilt from a fixed seed every call, so the map, robot
    starts and hidden targets are identical across validations and are never
    consumed by the training RNG.
    """
    env_agents = [f"agent_{i}" for i in range(args.num_agents)]
    totals: dict = {}
    episodes = max(1, int(args.validation_episodes))
    for episode in range(episodes):
        env = build_search_environment(args, seed=seed + episode)
        for agent in env_agents:
            env.action_space(agent).seed(seed=seed + episode)
        obs = initialize_obs(env)
        two_channelmap = env.two_channel_map(obs, {})
        metrics = SearchEpisodeMetrics()
        metrics.reset(len(env.goals_pos))
        step_t = 0
        while step_t < episode_limit:
            step_t += 1
            action = select_action_deterministic(two_channelmap, policy, env, env_agents, device)
            obs, _reward, terminated, repeat_mark, two_channelmap = env.step(
                action, obs, {}, two_channelmap, episode_limit
            )
            metrics.observe(env, repeat_mark)
            if any(terminated.values()):
                break
        report = metrics.as_dict(env, obs)
        for key, value in report.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
    return {key: value / episodes for key, value in totals.items()}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # 使用configs中的超参数；gamma / lmbda 可由命令行覆盖并写入元数据。
    gamma = args.gamma
    actor_lr = configs.ACTOR_LR
    critic_lr = configs.CRITIC_LR
    horizon = configs.HORIZON
    clip_param = configs.CLIP_PARAM
    k_epochs = configs.K_EPOCHS
    minibatch_size = configs.MINIBATCH_SIZE
    lmbda = args.lmbda
    huber_delta = configs.HUBER_DELTA
    device = resolve_device(args.device)
    trick = dict(configs.TRICK)

    directories = prepare_output_dir(args.output_dir)
    model_dir = str(directories["checkpoints"])
    model_dir1 = model_dir

    # One seed drives every random source the episode depends on, so a run is reproducible.
    seed_all(args.seed)

    validation_seed = (
        args.validation_seed if args.validation_seed is not None else args.seed + 1_000_000_007
    )

    start_time = time.time()
    writer = None if args.no_tensorboard else SummaryWriter(log_dir=str(directories["tb_logs"]))

    env = build_search_environment(args, seed=args.seed)
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

    if args.curriculum == "gate":
        stages = default_curriculum_stages()
        gate = CurriculumGate(
            stages=default_stages(stages),
            eval_every=args.eval_every,
            required_passes=args.required_passes,
        )
    else:
        stages = None
        # 评估节奏必须尊重 --eval-every：课程关闭时仍按该间隔跑确定性验证
        # 与随机基线早停（见 CurriculumGate.should_evaluate 注释）。
        gate = CurriculumGate(
            stages=None,
            fixed_limit=args.fixed_episode_limit,
            eval_every=args.eval_every,
        )

    run_settings = {
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
        # 奖励与课程：模式、全部权重、门控配置必须随运行落盘，
        # 否则事后无法判断一个 checkpoint 是哪套奖励训出来的。
        "reward_mode": args.reward_mode,
        "reward_v2_weights": dict(configs.REWARD_V2),
        "potential_beta": args.potential_beta,
        "t_max": configs.T_MAX,
        "curriculum": args.curriculum,
        "curriculum_stage_limits": stages if args.curriculum == "gate" else None,
        "eval_every": args.eval_every,
        "required_passes": args.required_passes,
        "validation_episodes": args.validation_episodes,
        "validation_seed": validation_seed,
        "fixed_episode_limit": args.fixed_episode_limit,
        "early_stop": args.early_stop,
        "early_stop_window": args.early_stop_window,
        "early_stop_margin": args.early_stop_margin,
        "cuda_deterministic": False,  # cudnn.benchmark=True in the policy module
    }
    # checkpoint 内嵌的最小可解释元数据：gamma、lambda、奖励模式与全部权重。
    checkpoint_meta = {
        "gamma": gamma,
        "lmbda": lmbda,
        "reward_mode": args.reward_mode,
        "reward_v2_weights": dict(configs.REWARD_V2),
        "potential_beta": args.potential_beta,
        "t_max": configs.T_MAX,
        "seed": args.seed,
        "device": str(device),
    }

    write_run_meta(
        directories["root"],
        settings=run_settings,
        seed=args.seed,
        device=device,
        dim_info=dim_info,
    )

    episode_num = 0
    global_step = 0
    best_validation_score = -float("inf")
    early_stop_triggered = False
    trainer_state_path = directories["root"] / "trainer_state.pkl"
    if args.resume is not None:
        episode_num = resume_policy(policy, args.resume, device)
        print(f"resumed from {args.resume} at episode {episode_num}")
        # 恢复课程阶段、全局步数、最优验证分与三条 RNG 流——否则任何 resume
        # 都事实上从 stage 0 重来，且后续训练动作序列整体偏移。
        if trainer_state_path.is_file():
            state = load_trainer_state(trainer_state_path)
            global_step = int(state.get("global_step", 0))
            best_validation_score = float(state.get("best_validation_score", -float("inf")))
            early_stop_triggered = bool(state.get("early_stop_triggered", False))
            restore_trainer_state(state, gate)
            print(
                f"restored trainer state: global_step={global_step} "
                f"stage={gate.stage_index} best={best_validation_score:.3f}"
            )
        else:
            print("no trainer_state.pkl found; curriculum/RNG state restarts from scratch")

    step = global_step # 记录总步数（resume 时从 trainer_state 恢复）
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
    # E 现在只是当前课程阶段的回合步数上限，由 CurriculumGate 决定；
    # 它不再代表“累计奖励刷新”，升级只看验证任务指标。
    E = gate.episode_limit()
    # 随机基线缓存：按 E 缓存 raw-random / legal-random 指标（早停参照系）。
    baseline_cache: dict = {}
    step_reward_t = 0
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
    while episode_num < args.episodes:
        step += 1
        step_t +=1
        step_reward_t += 1
        # 获取动作（含合法动作掩码，掩码随样本进 buffer）
        action, action_log_pi, legal_masks = select_action(two_channelmap, policy, env,env_agents, device)
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

        # done 的两种语义分开：terminated 是真实终止（全部目标完成），
        # step_t >= E 是时间上限截断。adv_dones 用截断后的 done（GAE 边界），
        # done_bool（critic bootstrap）只用真实终止，避免把截断当成价值吸收态。
        done = {agent_id: (terminated[agent_id] or step_t >= E) for agent_id in env_agents}
        done_bool = {agent_id: terminated[agent_id] for agent_id in env_agents}

        for agentid in two_channelmap.keys():
            obs1[agentid] = two_channelmap[agentid]['obstacle_map']
            obs2[agentid] = two_channelmap[agentid]['explore_map']
        for agentid in next_two_channelmap.keys():
            next_obs1[agentid] = next_two_channelmap[agentid]['obstacle_map']
            next_obs2[agentid] = next_two_channelmap[agentid]['explore_map']

        # 保存数据
        policy.add(obs1,obs2,obs_pheromone, action, reward, next_obs1,next_obs2, next_obs_pheromone,done_bool, action_log_pi, done, legal_masks)

        episode_reward = {agent_id: episode_reward[agent_id] + reward[agent_id] for agent_id in env_agents}
        obs = next_obs
        two_channelmap = next_two_channelmap
        # 完成目标 或者 达到最大环境步 ，这一个回合结束
        if any(done.values()) or step_t >= E:
            # 先用回合结束时的环境与观测统计搜索指标，环境重建之后就拿不到了
            episode_metrics = metrics.as_dict(env, obs)
            episode_reward_total = sum(episode_reward.values())
            episode_length = episode_metrics["episode_steps"]
            episode_metrics.update(
                {
                    "episode": episode_num + 1,
                    "env_steps": E,
                    "curriculum_stage": gate.stage_index,
                    "episode_return": float(episode_reward_total),
                    "mean_reward_per_step": float(episode_reward_total / episode_length)
                    if episode_length
                    else 0.0,
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
                    f"Episode {episode_num + 1} | stage={gate.stage_index} E={E} | "
                    f"discovered={episode_metrics['targets_discovered']} "
                    f"coverage={episode_metrics['coverage']:.3f} "
                    f"return={episode_metrics['episode_return']:.1f} "
                    f"mean_r/step={episode_metrics['mean_reward_per_step']:.3f} | "
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

            # 课程门控：只喂验证任务指标，绝不喂累计奖励。达到评估边界时做一次
            # 固定地图/固定 seed 的确定性验证，连续达标才升级。
            gate.register_episode()
            if gate.should_evaluate():
                report = run_validation(args, policy, device, gate.episode_limit(), validation_seed)
                # 只把任务指标喂给门控；门控 coverage 用 coverage_gain（C_T - C_0），
                # 初始视野覆盖率（~0.044）不计入；repeat_ratio 对应 repeat_rate 阈值。
                gate_metrics = {
                    "coverage_gain": report["coverage_gain"],
                    "collision_rate": report["collision_rate"],
                    "repeat_rate": report["repeat_ratio"],
                }
                advanced = gate.submit_validation(gate_metrics)
                score = validation_score(report)

                # 随机基线（同一批固定验证 seed）：早停的参照系，按 E 缓存。
                if E not in baseline_cache:
                    baseline_cache[E] = (
                        run_random_baseline(args, E, validation_seed, legal_only=False),
                        run_random_baseline(args, E, validation_seed, legal_only=True),
                    )
                raw_base, legal_base = baseline_cache[E]

                if writer is not None:
                    for key, value in report.items():
                        writer.add_scalar(f"validation/{key}", value, episode_num)
                    writer.add_scalar("validation/stage", gate.stage_index, episode_num)
                    writer.add_scalar("baseline/raw_random_coverage", raw_base["coverage"], episode_num)
                    writer.add_scalar("baseline/legal_random_coverage", legal_base["coverage"], episode_num)
                # 最优 checkpoint 按验证任务指标保存，不按训练累计回报。
                if score > best_validation_score:
                    best_validation_score = score
                    policy.save(str(directories["root"] / "best"), episode_num, meta=checkpoint_meta)
                    save_trainer_state(
                        trainer_state_path,
                        episode_num=episode_num,
                        global_step=step,
                        best_validation_score=best_validation_score,
                        gate=gate,
                    )
                print(
                    f"[validation] episode {episode_num} stage={gate.stage_index} "
                    f"score={score:.3f} coverage={report['coverage']:.3f} "
                    f"gain={report['coverage_gain']:.3f} "
                    f"discovery={report['discovery_rate']:.3f} advanced={advanced} | "
                    f"baselines coverage raw={raw_base['coverage']:.3f} "
                    f"legal={legal_base['coverage']:.3f}"
                )

                # 早停：窗口内确定性策略必须超过 raw-random，并接近 legal-random，
                # 否则策略已坍塌（比随机还差），不再跑满 5000 回合浪费算力。
                if (
                    args.early_stop
                    and not early_stop_triggered
                    and episode_num >= args.early_stop_window
                ):
                    below_raw = report["coverage"] <= raw_base["coverage"]
                    below_legal = report["coverage"] < legal_base["coverage"] - args.early_stop_margin
                    if below_raw or below_legal:
                        early_stop_triggered = True
                        print(
                            f"[early-stop] episode {episode_num}: validation coverage "
                            f"{report['coverage']:.3f} raw-random {raw_base['coverage']:.3f} "
                            f"legal-random {legal_base['coverage']:.3f} (margin "
                            f"{args.early_stop_margin}) -> stopping; policy failed to "
                            f"beat random within {args.early_stop_window} episodes"
                        )
                        save_trainer_state(
                            trainer_state_path,
                            episode_num=episode_num,
                            global_step=step,
                            best_validation_score=best_validation_score,
                            gate=gate,
                            early_stop_triggered=True,
                        )
                # 阶段可能已变，重新读取当前回合步数上限。
                E = gate.episode_limit()

            env = build_search_environment(args, seed=args.seed + episode_num) # 重置环境
            for agent in env_agents:
                env.action_space(agent).seed(seed=args.seed + episode_num)
            obs = initialize_obs(env)  # 重置时间戳地图
            metrics.reset(len(env.goals_pos))

            # 保存模型：只在回合边界，且跳过 episode 0（0 % every == 0 会让第一回合每步都存盘）
            if should_save_checkpoint(episode_num, args.save_every):
                policy.save(model_dir, episode_num, meta=checkpoint_meta)
                save_trainer_state(
                    trainer_state_path,
                    episode_num=episode_num,
                    global_step=step,
                    best_validation_score=best_validation_score,
                    gate=gate,
                    early_stop_triggered=early_stop_triggered,
                )

            episode_reward = {agent_id: 0 for agent_id in env_agents}
            reward_term_sums = {agent_id: {} for agent_id in env_agents}
            step_t = 0

            if early_stop_triggered:
                break

        # 每隔一定步数更新网络
        if step % horizon == 0:
            policy.learn(minibatch_size, gamma, lmbda, clip_param, k_epochs, entropy_coefficient=0.01, huber_delta=huber_delta)

    print(f"Training completed in {time.time() - start_time} seconds")
    policy.save(model_dir1, episode_num, meta=checkpoint_meta)

    # 保存训练结果
    train_return_ = np.array([train_return[agent_id] for agent_id in env.agents])
    np.save(os.path.join(model_dir, "train_return.npy"), train_return_)
    if writer is not None:
        writer.close()
    print(f"checkpoints, TensorBoard logs and metrics are in {directories['root']}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())


