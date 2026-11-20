"""w/o A* 场景下的「合法动作掩码 + 低温采样」实验入口（独立脚本）。

不修改任何原有源码与权重；只替换推理选动作阶段：
1. 根据智能体已知障碍通道（two_channelmap[agent]["obstacle_map"]，1=障碍，
   0=已知空地，-1=未知）与地图边界，屏蔽 4 个正交动作中的非法方向；
2. 对策略输出概率做温度缩放 p^(1/T) 后重归一化再采样，T<=0.05 退化为 argmax。

动作索引已按 src/Map/timemap.py 的
``action_list = [[-1,0],[1,0],[0,-1],[0,1]]`` 核对：
0=上(-1,0)，1=下(+1,0)，2=左(0,-1)，3=右(0,+1)。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
for path in (ROOT, os.path.join(ROOT, "src", "DDQN")):
    if path not in sys.path:
        sys.path.insert(0, path)

import configs  # noqa: E402
import test as paper_test  # noqa: E402
from src.DDQN.Agent import MAPPO  # noqa: E402
from src.Map.timemap import Environment  # noqa: E402
from src.DDQN.rescue import RescueAlgorithm  # noqa: E402
from src.metrics.revisit_metrics import RevisitCoverageMeter  # noqa: E402

torch.set_num_threads(int(os.environ.get("PAPER_TORCH_THREADS", "4")))
ACTION_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def cpu_load(dim_info, is_continue, model_dir, trick=None):
    policy = MAPPO(dim_info, actor_lr=0, critic_lr=0, horizon=0,
                   device="cpu", trick=trick)
    state = torch.load(os.path.join(model_dir, "shared_mappo_model_21_9232.pth"),
                       map_location="cpu")
    policy.agents.full_model.load_state_dict(state["full_model"])
    policy.agents.actor.load_state_dict(state["actor"])
    policy.agents.critic.load_state_dict(state["critic"])
    policy.agents.full_model.eval()
    policy.agents.actor.eval()
    policy.agents.critic.eval()
    return policy


def local_pheromone(global_ph, x, y, radius=5):
    """与 test.py::get_local_ph 相同的 11×11 局部窗口（越界补 0）。"""
    h, w = global_ph.shape
    x0, x1 = max(0, x - radius), min(h, x + radius + 1)
    y0, y1 = max(0, y - radius), min(w, y + radius + 1)
    patch = global_ph[x0:x1, y0:y1]
    size = 2 * radius + 1
    return np.pad(patch,
                  ((max(0, size - (x1 - x0)) if x - radius < 0 else 0,
                    max(0, size - (x1 - x0)) if x + radius + 1 > h else 0),
                   (max(0, size - (y1 - y0)) if y - radius < 0 else 0,
                    max(0, size - (y1 - y0)) if y + radius + 1 > w else 0)),
                  mode="constant")


def valid_mask(agents_pos, agent_index, obstacle_map):
    """1=方向合法。仅屏蔽已知障碍(=1)与越界；未知(-1)按可走处理。"""
    x, y = agents_pos[agent_index]
    h, w = obstacle_map.shape
    mask = np.zeros(4, dtype=np.float32)
    for i, (dx, dy) in enumerate(ACTION_DELTAS):
        nx, ny = int(x) + dx, int(y) + dy
        if 0 <= nx < h and 0 <= ny < w and obstacle_map[nx, ny] != 1:
            mask[i] = 1.0
    return mask


def actor_probs(policy, obstacle_map, explore_map, pheromone_map, device):
    """复现 Agent.get_action 的前向部分，返回 [4] 概率（Actor 末端已 softmax）。"""
    obs = np.stack([obstacle_map, explore_map], axis=0)
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    ph_t = torch.tensor(pheromone_map, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = policy.agents.full_model(obs_t, ph_t, 1)
        probs = policy.agents.actor(feature.view(1, -1))
    return probs.squeeze(0)


def select_actions(two_channelmap, policy, env, env_agents, device,
                   temperature=0.5, use_mask=True):
    actions = {}
    for agent_id in env_agents:
        obstacle_map = two_channelmap[agent_id]["obstacle_map"]
        explore_map = two_channelmap[agent_id]["explore_map"]
        idx = int(agent_id.split("_")[-1])
        x, y = env.agents_pos[idx]
        ph = local_pheromone(env.global_pheromone, int(x), int(y))
        probs = actor_probs(policy, obstacle_map, explore_map, ph, device).clone()
        if use_mask:
            m = torch.tensor(valid_mask(env.agents_pos, idx, obstacle_map),
                             dtype=torch.float32, device=device)
            masked = probs * m
            if float(masked.sum()) > 0:  # 全部被挡时保留原分布，避免 NaN
                probs = masked
        if temperature is None or temperature <= 0.05:
            action = int(torch.argmax(probs).item())
        elif abs(temperature - 1.0) < 1e-6:
            action = int(torch.multinomial(probs / probs.sum(), 1).item())
        else:
            q = probs.clamp_min(1e-12) ** (1.0 / temperature)
            q = q / q.sum()
            action = int(torch.multinomial(q, 1).item())
        actions[agent_id] = action
    return actions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="test_manifest.txt")
    p.add_argument("--episodes", type=int, default=250)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--seed-base", type=int, default=2000000000)
    p.add_argument("--no-rescue", action="store_true", default=True,
                   help="默认关闭 A*（本实验针对 w/o A*）")
    p.add_argument("--rescue", dest="no_rescue", action="store_false",
                   help="启用 A*（保持兼容）")
    p.add_argument("--action-temperature", type=float, default=0.5)
    p.add_argument("--no-action-mask", action="store_true")
    p.add_argument("--out", default="runs/ch3_paper_env/mask/mask_eval.json")
    return p.parse_args()


def main():
    args = parse_args()
    maps = [line.strip() for line in open(args.manifest, encoding="utf-8")
            if line.strip()][: args.episodes]
    paper_test.MAPPO.load = staticmethod(cpu_load)
    dim_info = {f"agent_{i}": [9, 5] for i in range(configs.num_agents)}
    policy = paper_test.MAPPO.load(dim_info, False, "path_to_save_model_1",
                                   trick=configs.TRICK)
    device = torch.device("cpu")
    env_agents = [f"agent_{i}" for i in range(configs.num_agents)]

    rows = []
    start = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    jsonl = open(os.path.abspath(args.out) + ".jsonl", "w", encoding="utf-8")
    for ep, map_name in enumerate(maps):
        seed = args.seed_base + ep
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        env = Environment(test=True, map_filename=map_name,
                          comm_failure_prob=0.0, comm_partial_loss_prob=0.0,
                          comm_delay_prob=0.0, enable_comm_noise=False)
        obs = paper_test.initialize_obs(env)
        two_channelmap = env.two_channel_map(obs, {})
        rescue_policy = (None if args.no_rescue
                         else RescueAlgorithm(env.map_size, env, obs))
        rescue_actions = {}
        meter = RevisitCoverageMeter(free_mask=(env.map == 0), radius=5,
                                     window=100, use_los=True)
        tracker = paper_test.CoverageTracker(meter)
        tracker.reset()

        success = False
        step_t = 0
        res_t = 0
        res_step = 0
        while step_t < args.max_steps:
            step_t += 1
            action = select_actions(two_channelmap, policy, env, env_agents,
                                    device, args.action_temperature,
                                    not args.no_action_mask)
            for agent_id, seq in list(rescue_actions.items()):
                if seq:
                    action[agent_id] = seq[0]
                    seq.pop(0)
                    res_step += 1
                if not seq:
                    del rescue_actions[agent_id]
            next_obs, _reward, terminated, repeat_val, next_map = env.step(
                action, obs, rescue_actions, two_channelmap, ep)
            tracker.update(env.agents_pos, env)
            obs, two_channelmap = next_obs, next_map
            if repeat_val and rescue_policy is not None:
                rescue_actions.update(
                    paper_test.repeat_check(repeat_val, env, obs, rescue_policy))
                res_t += 1
            if any(terminated.values()):
                success = True
                break

        goals = 6 if success else 6 - len(env.goals_pos)
        curve = tracker.coverage_history
        rows.append({
            "episode": ep, "map": map_name, "seed": seed,
            "success": int(success), "steps": step_t, "goals": goals,
            "coverage_final": float(curve[-1]) if curve else 0.0,
            "coverage_auc": float(np.mean(curve)) if curve else 0.0,
            "rescue_triggers": res_t, "rescue_steps": res_step,
        })
        jsonl.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
        jsonl.flush()
        if (ep + 1) % 10 == 0:
            sr = np.mean([r["success"] for r in rows])
            print(f"[{os.path.basename(args.out)}] {ep+1}/{len(maps)} "
                  f"SR={sr:.3f} {time.time()-start:.0f}s", flush=True)

    steps = np.array([r["steps"] for r in rows], dtype=float)
    goals = np.array([r["goals"] for r in rows], dtype=float)
    succ = np.array([r["steps"] for r in rows if r["success"]], dtype=float)
    summary = {
        "n": len(rows),
        "SR": float(np.mean([r["success"] for r in rows])),
        "AS": float(steps.mean()),
        "SV": float(steps.var()),
        "ANTO": float(goals.mean()),
        "AS_success": float(succ.mean()) if succ.size else None,
        "SV_success": float(succ.var()) if succ.size > 1 else None,
        "coverage_final": float(np.mean([r["coverage_final"] for r in rows])),
        "coverage_auc": float(np.mean([r["coverage_auc"] for r in rows])),
        "rescue_triggers_mean": float(np.mean([r["rescue_triggers"] for r in rows])),
        "wall_time_s": round(time.time() - start, 1),
    }
    out = {
        "environment": "修改6",
        "checkpoint": os.path.join("path_to_save_model_1",
                                   "shared_mappo_model_21_9232.pth"),
        "config": {
            "rescue": not args.no_rescue,
            "action_mask": not args.no_action_mask,
            "action_temperature": args.action_temperature,
        },
        "episodes": len(rows),
        "max_steps": args.max_steps,
        "seed_base": args.seed_base,
        "summary": summary,
        "per_episode": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(os.path.splitext(args.out)[0] + ".csv", "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
