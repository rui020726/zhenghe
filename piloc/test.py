"""PILOC 独立测试集终评工具。

与训练/验证完全隔离的评估协议：

- 测试 seed 清单来自 ``--seed-manifest``（JSON，含 ``seeds`` 列表），所有评估
  臂（各 best checkpoint、legal-random、raw-random、legacy）复用同一清单，
  因此看到完全相同的测试地图。清单必须与训练 seed
  （``args.seed + episode_num``）和验证 seed（``args.seed + 1_000_000_007 + k``）
  不相交。
- 环境构造与确定性验证一致：seeded 地图生成（``test=False``，不用图片地图）、
  通信参数取 Environment 默认值（与训练/验证相同）。
- 默认确定性动作（掩码后 argmax，不消耗 RNG）；``--sampled`` 切换为掩码后
  采样。纯策略评估默认关闭 RescueAlgorithm（``--rescue`` 可单独开启并单独报告）。
- 网络加载后用 eval() 模式（与部署一致；train.py 的 resume 路径刻意保持
  train() 模式以保证恢复等价，终评以部署口径为准）。
- 指标口径与训练时确定性验证完全一致（复用 train.SearchEpisodeMetrics），
  另补 last_discovery_step 与 all_found（全目标发现率）。

用法示例::

    python piloc/test.py --checkpoint runs/cur_v2rep05_s0/best/shared_mappo_model_21_2900.pth \
        --tag v2rep05_s0 --seed-manifest runs/final_eval_20260916/test_seeds.json \
        --out runs/final_eval_20260916/v2rep05_s0.json
    python piloc/test.py --baseline legal-random --tag legal_random \
        --seed-manifest runs/final_eval_20260916/test_seeds.json \
        --out runs/final_eval_20260916/legal_random.json

每个臂输出 JSON：逐回合指标 + 均值/std/95% CI + 验收判定所需的全部字段。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

# Agent.py does `from normalization import ...`, which only resolves with
# src/DDQN on the path (same fix as train.py).
_SRC_DDQN = str(Path(__file__).resolve().parent / "src" / "DDQN")
if _SRC_DDQN not in sys.path:
    sys.path.append(_SRC_DDQN)

from src.DDQN.Agent import MAPPO
from src.DDQN.rescue import RescueAlgorithm
from src.Map.timemap import Environment, legal_action_mask
import configs
from train import (
    SearchEpisodeMetrics,
    apply_action_mask,
    get_local_ph,
    initialize_obs,
    resume_policy,
)

DEFAULT_SEED_MANIFEST = os.path.join("runs", "final_eval_20260916", "test_seeds.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PILOC independent test-set final evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="policy checkpoint (.pth); omit when running a random baseline")
    parser.add_argument("--baseline", type=str, default=None, choices=["legal-random", "raw-random"],
                        help="random baseline arm instead of a policy checkpoint")
    parser.add_argument("--tag", type=str, required=True, help="arm name recorded in the output JSON")
    parser.add_argument("--episodes", type=int, default=None,
                        help="number of test episodes (default: all seeds in the manifest)")
    parser.add_argument("--E", "--episode-limit", type=int, default=250, dest="episode_limit",
                        help="episode step limit (default 250, final curriculum stage)")
    parser.add_argument("--seed-manifest", type=str, default=DEFAULT_SEED_MANIFEST,
                        help="JSON file with a 'seeds' list; created from --test-seed-base if missing")
    parser.add_argument("--test-seed-base", type=int, default=2_000_000_000,
                        help="used to synthesize the manifest when the file does not exist")
    parser.add_argument("--num-agents", type=int, default=configs.num_agents)
    parser.add_argument("--num-goals", type=int, default=configs.num_goals)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--sampled", action="store_true",
                        help="sample from the masked policy instead of deterministic argmax")
    parser.add_argument("--rescue", action="store_true",
                        help="enable RescueAlgorithm assistance; reported separately from pure policy")
    parser.add_argument("--out", type=str, default=None, help="output JSON path")
    return parser.parse_args(argv)


def load_seed_manifest(path: str, base: int, episodes: int | None) -> list[int]:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            seeds = [int(s) for s in json.load(handle)["seeds"]]
    else:
        count = episodes or 120
        seeds = [base + i for i in range(count)]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"test_seed_base": base, "count": count, "episode_limit": 250,
                       "seeds": seeds}, handle, indent=2)
        print(f"[final-eval] seed manifest created: {path} ({count} seeds, base {base})")
    if episodes is not None:
        seeds = seeds[:episodes]
    return seeds


def load_policy(checkpoint: str, device: torch.device):
    dim_info = {f"agent_{i}": [121, 4] for i in range(configs.num_agents)}
    policy = MAPPO(dim_info, actor_lr=0, critic_lr=0, horizon=0, device=str(device), trick=configs.TRICK)
    episode, _bundle = resume_policy(policy, checkpoint, device)
    # 部署口径：eval() 模式（dropout 关闭），与集成 SAR runtime 一致
    policy.agents.full_model.eval()
    policy.agents.actor.eval()
    policy.agents.critic.eval()
    return policy, episode


def policy_actions(two_channelmap, policy, env, env_agents, device, sampled: bool):
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
        if sampled:
            dist = Categorical(probs=masked)
            action = dist.sample()
        else:
            action = torch.argmax(masked[0])
        actions[agent_id] = int(action.item())
    return actions


def run_episode(seed: int, args, policy, device, rng, rescue_policy_factory):
    env = Environment(
        num_agents=args.num_agents,
        num_goals=args.num_goals,
        track_reward_terms=False,
        reward_mode="v2",
        gamma=args.gamma,
        t_max=configs.T_MAX,
        seed=seed,
    )
    env_agents = [f"agent_{i}" for i in range(args.num_agents)]
    for agent in env_agents:
        env.action_space(agent).seed(seed)

    obs = initialize_obs(env, num_agents=args.num_agents)
    two_channelmap = env.two_channel_map(obs, {})
    metrics = SearchEpisodeMetrics()
    metrics.reset(len(env.goals_pos))

    rescue_policy = rescue_policy_factory(env, obs) if rescue_policy_factory else None
    rescue_actions: dict = {}
    rescue_triggers = 0
    rescue_steps = 0

    prev_goals = len(env.goals_pos)
    last_discovery_step = None
    step_t = 0
    while step_t < args.episode_limit:
        step_t += 1
        if args.baseline == "legal-random":
            actions = {}
            for agent_id in env_agents:
                legal = np.flatnonzero(legal_action_mask(env, int(agent_id.split("_")[-1])) > 0).tolist()
                actions[agent_id] = int(rng.choice(legal)) if legal else 0
        elif args.baseline == "raw-random":
            actions = {agent_id: int(rng.randint(4)) for agent_id in env_agents}
        else:
            actions = policy_actions(two_channelmap, policy, env, env_agents, device, args.sampled)

        if rescue_actions:
            for agent_id, sequence in list(rescue_actions.items()):
                if sequence:
                    actions[agent_id] = sequence[0]
                    sequence.pop(0)
                    rescue_steps += 1
                if not sequence:
                    del rescue_actions[agent_id]

        obs, _reward, terminated, repeat_marks, two_channelmap = env.step(
            actions, obs, rescue_actions, two_channelmap, args.episode_limit
        )
        metrics.observe(env, repeat_marks)

        if len(env.goals_pos) < prev_goals:
            last_discovery_step = step_t
            prev_goals = len(env.goals_pos)

        if repeat_marks and rescue_policy is not None:
            for agent_id in repeat_marks:
                path = rescue_policy.rescue_action(int(agent_id.split("_")[1]), env, obs)
                rescue_actions[agent_id] = path
            rescue_triggers += 1

        if any(terminated.values()):
            break

    report = metrics.as_dict(env, obs)
    report["last_discovery_step"] = last_discovery_step
    report["all_targets_found"] = bool(
        metrics.initial_goals > 0 and report["targets_discovered"] == metrics.initial_goals
    )
    report["seed"] = seed
    report["rescue_triggers"] = rescue_triggers
    report["rescue_steps"] = rescue_steps
    return report


def summarize(rows: list[dict]) -> dict:
    keys = ["coverage", "coverage_auc", "coverage_gain", "discovery_rate",
            "targets_discovered", "all_targets_found", "first_discovery_step",
            "last_discovery_step", "repeat_ratio", "collision_rate",
            "recoverage_rate", "episode_steps", "travel"]
    summary = {"n": len(rows)}
    n = len(rows)
    for key in keys:
        values = [float(r[key]) if r[key] is not None else float("nan") for r in rows]
        values = [v for v in values if not np.isnan(v)]
        if not values:
            continue
        mean = st.mean(values)
        std = st.pstdev(values) if len(values) > 1 else 0.0
        ci = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
        summary[key] = {"mean": mean, "std": std, "ci95": float(ci)}
    return summary


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device("cpu")
    seeds = load_seed_manifest(args.seed_manifest, args.test_seed_base, args.episodes)
    torch.manual_seed(seeds[0])

    policy = None
    checkpoint_episode = None
    if args.checkpoint:
        policy, checkpoint_episode = load_policy(args.checkpoint, device)
    elif not args.baseline:
        raise SystemExit("either --checkpoint or --baseline is required")

    def rescue_factory(env, obs):
        return RescueAlgorithm(env.map_size, env, obs)

    # 基线随机源与 train.run_random_baseline 一致：单次 RandomState，跨回合顺序抽取
    manifest_base = seeds[0]
    rng = np.random.RandomState(manifest_base) if args.baseline else None

    rows = []
    start = time.time()
    for index, seed in enumerate(seeds):
        rows.append(run_episode(seed, args, policy, device, rng,
                                rescue_factory if args.rescue else None))
        if (index + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"[{args.tag}] {index + 1}/{len(seeds)} episodes, {elapsed:.0f}s", flush=True)

    result = {
        "tag": args.tag,
        "git_commit": git_commit(),
        "checkpoint": args.checkpoint,
        "checkpoint_episode": checkpoint_episode,
        "baseline": args.baseline,
        "episodes": len(seeds),
        "episode_limit": args.episode_limit,
        "seed_manifest": os.path.abspath(args.seed_manifest),
        "seeds": seeds,
        "num_agents": args.num_agents,
        "num_goals": args.num_goals,
        "action_selection": "sampled" if args.sampled else "deterministic",
        "model_mode": "eval",
        "rescue_enabled": bool(args.rescue),
        "wall_time_s": round(time.time() - start, 1),
        "summary": summarize(rows),
        "per_episode": rows,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    summary = result["summary"]
    print(json.dumps({k: (round(v["mean"], 4) if isinstance(v, dict) else v)
                      for k, v in summary.items()}, ensure_ascii=False))
    print(f"[{args.tag}] done -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
