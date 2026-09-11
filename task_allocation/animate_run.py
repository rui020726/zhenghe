# animate_run.py
# -*- coding: utf-8 -*-

import os
import argparse
import time
from typing import List, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

from env.task_env import TaskEnv
from network import CTDE_Network



# =========================
# Utility
# =========================

def pick_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Build env / net (match driver)
# =========================

def build_env(dt: float) -> TaskEnv:
    env = TaskEnv(
        prio_range=(0.3, 1.0),
        per_species_range=(3, 5),
        species_range=(2, 3),
        tasks_range=(20, 25),
        comm_range=0.2,
        dynamic_task_discovery=True,
        discovery_probability=0.05,
        max_dynamic_tasks=5,
        seed=None,
        dt=dt,
    )
    return env


def build_net(device: torch.device) -> CTDE_Network:
    net = CTDE_Network(
        self_dim=10,
        neighbor_dim=10,
        task_dim=15,
        global_agent_dim=13,
        global_task_dim=18,
        hidden=128,
    ).to(device)
    net.eval()
    return net


def load_checkpoint(net: CTDE_Network, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"], strict=True)
    net.eval()
    print(f"Loaded checkpoint: {ckpt_path}")


# =========================
# Snapshot (record richer info)
# =========================

def snapshot(env: TaskEnv) -> Dict[str, Any]:
    agent_ids = sorted(env.agent_dic.keys())
    task_ids = sorted(env.task_dic.keys())

    agent_pos = np.array([env.agent_dic[aid]["location"] for aid in agent_ids])
    task_pos = np.array([env.task_dic[tid]["location"] for tid in task_ids])
    task_finished = np.array([env.task_dic[tid]["finished"] for tid in task_ids])

    # === NEW: target task & species for visualization ===
    targets = []
    species = []
    for aid in agent_ids:
        targets.append(env.agent_dic[aid].get("target_task", -1))
        species.append(env.agent_dic[aid]["species"])

    decision_now = any(a.get("need_decision", False) for a in env.agent_dic.values())

    return dict(
        t=env.current_time,
        agent_ids=agent_ids,
        agent_pos=agent_pos,
        task_ids=task_ids,
        task_pos=task_pos,
        task_finished=task_finished,
        targets=np.array(targets),
        species=np.array(species),
        decision_frame=decision_now,
    )


# =========================
# Rollout with decision-aware recording
# =========================

@torch.no_grad()
def rollout(
    env: TaskEnv,
    net: CTDE_Network,
    device: torch.device,
    dt: float,
    max_tasks: int,
    env_time_limit: float,
    max_steps: int,
    greedy: bool,
    stride: int,
    decision_hold: int,
):
    frames: List[Dict[str, Any]] = []
    step = 0

    frames.append(snapshot(env))

    while not env.finished and step < max_steps and env.current_time <= env_time_limit:

        decision_ids = [aid for aid, a in env.agent_dic.items() if a.get("need_decision", False)]

        if decision_ids:
            batch_s, batch_n, batch_t, batch_m = [], [], [], []
            action_maps = []

            for aid in decision_ids:
                s, n, t, m, amap = env.get_local_obs(aid, max_tasks=max_tasks)
                batch_s.append(s)
                batch_n.append(n)
                batch_t.append(t)
                batch_m.append(m.astype(bool))
                action_maps.append(amap)

            ts = torch.tensor(batch_s, dtype=torch.float32, device=device)
            tn = torch.tensor(batch_n, dtype=torch.float32, device=device)
            tt = torch.tensor(batch_t, dtype=torch.float32, device=device)
            tm = torch.tensor(batch_m, dtype=torch.bool, device=device)

            dist = net.get_action(ts, tn, tt, tm)
            acts = torch.argmax(dist.probs, dim=-1) if greedy else dist.sample()

            for i, aid in enumerate(decision_ids):
                env.apply_action(aid, int(acts[i]), action_maps[i])

        env.step(dt)
        step += 1

        if step % stride == 0:
            fr = snapshot(env)
            frames.append(fr)

            # === NEW: slow down decision frames ===
            if fr["decision_frame"]:
                for _ in range(decision_hold):
                    frames.append(fr)

    return frames


def interpolate_frames(frames, interp_steps=4):
    """
    在相邻帧之间做线性插值，让动画更丝滑
    - frames: 原始 snapshot 列表
    - interp_steps: 每两帧之间插多少帧（>=1）
    """
    if interp_steps <= 1 or len(frames) < 2:
        return frames

    smooth = []

    for i in range(len(frames) - 1):
        f0 = frames[i]
        f1 = frames[i + 1]

        smooth.append(f0)

        # 只对连续变量插值
        for k in range(1, interp_steps):
            alpha = k / interp_steps

            interp = dict(f0)  # 浅拷贝元数据
            interp["t"] = (1 - alpha) * f0["t"] + alpha * f1["t"]

            # === 位置插值（关键）===
            interp["agent_pos"] = (1 - alpha) * f0["agent_pos"] + alpha * f1["agent_pos"]

            # task 位置不变（静态）
            interp["task_pos"] = f0["task_pos"]

            # finished 状态：保持前一帧（避免闪烁）
            interp["task_finished"] = f0["task_finished"]

            # 目标任务、species 不变
            interp["targets"] = f0["targets"]
            interp["species"] = f0["species"]

            # 决策帧标记：只有真实帧才算
            interp["decision_frame"] = False

            smooth.append(interp)

    smooth.append(frames[-1])
    return smooth


# =========================
# Render animation
# =========================

def render(frames: List[Dict[str, Any]], out_path: str, fps: int, comm_range: float):

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    agent_sc = ax.scatter([], [], s=60, label="Agent")
    task_sc = ax.scatter([], [], s=70, label="Task")
    intent_lines = []

    # === NEW: communication range circles ===
    comm_circles = []
    COMM_ALPHA = 0.15  # 透明度，推荐 0.1 ~ 0.2
    COMM_LW = 1.0  # 线宽
    COMM_LS = "--"  # 虚线

    # 先用第一帧初始化圆
    first = frames[0]
    for _ in range(len(first["agent_pos"])):
        c = Circle(
            (0.0, 0.0),
            radius=comm_range,  # 注意：下面会告诉你 comm_range 怎么传进来
            fill=False,
            linestyle=COMM_LS,
            linewidth=COMM_LW,
            alpha=COMM_ALPHA,
            edgecolor="gray",
        )
        ax.add_patch(c)
        comm_circles.append(c)

    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    ax.legend(loc="lower right")

    species_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    def update(i):
        fr = frames[i]
        agent_sc.set_offsets(fr["agent_pos"])
        task_sc.set_offsets(fr["task_pos"])

        # fade finished tasks
        colors = []
        for done in fr["task_finished"]:
            colors.append((0.5, 0.5, 0.5, 0.3) if done else (0.8, 0.2, 0.2, 1.0))
        task_sc.set_color(colors)

        # clear old intent lines
        nonlocal intent_lines
        for ln in intent_lines:
            ln.remove()
        intent_lines = []

        # draw intent lines
        for i, tgt in enumerate(fr["targets"]):
            if tgt >= 0 and tgt < len(fr["task_pos"]):
                a = fr["agent_pos"][i]
                t = fr["task_pos"][tgt]
                c = species_colors[fr["species"][i] % len(species_colors)]
                ln, = ax.plot(
                    [a[0], t[0]],
                    [a[1], t[1]],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.6,
                    color=c,
                )
                intent_lines.append(ln)

        # === NEW: update communication circles ===
        for c, pos in zip(comm_circles, fr["agent_pos"]):
            c.center = (pos[0], pos[1])

        done = fr["task_finished"].sum()
        info.set_text(f"t={fr['t']:.1f}\nfinished={done}/{len(fr['task_finished'])}")

        return agent_sc, task_sc, *intent_lines, info

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if out_path.endswith(".gif"):
        ani.save(out_path, writer="pillow", fps=fps)
    else:
        ani.save(out_path, writer="ffmpeg", fps=fps)

    plt.close(fig)
    print(f"Saved animation to {out_path}")


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()

    # === model / output ===
    ap.add_argument("--ckpt", type=str, default="./checkpoints/ctde_final.pth")
    ap.add_argument("--out", type=str, default="demo.gif", help="Output animation (.gif)")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--device", default="auto")

    # === env / rollout ===
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--max_tasks", type=int, default=65)
    ap.add_argument("--env_time_limit", type=float, default=1000)
    ap.add_argument("--max_steps", type=int, default=1000)

    # === visualization control ===
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--decision_hold", type=int, default=8)
    ap.add_argument("--greedy", action="store_true")

    args = ap.parse_args()

    # === enforce gif extension ===
    if not args.out.lower().endswith(".gif"):
        args.out = args.out + ".gif"

    device = pick_device(args.device)
    env = build_env(args.dt)
    net = build_net(device)
    load_checkpoint(net, args.ckpt, device)

    env.reset(seed=0)

    frames = rollout(
        env,
        net,
        device,
        args.dt,
        args.max_tasks,
        args.env_time_limit,
        args.max_steps,
        args.greedy,
        args.stride,
        args.decision_hold,
    )
    # === NEW: smooth animation by interpolation ===
    frames = interpolate_frames(frames, interp_steps=8)

    render(frames, args.out, args.fps, comm_range=env.comm_range)


if __name__ == "__main__":
    main()
