# -*- coding: utf-8 -*-
"""
CTDE 轨迹可视化（额外诊断版）
- 对齐当前: env/task_env.py + network.py + driver顺序决策逻辑
- 输出GIF并叠加诊断信息，帮助分析成功率低原因

诊断项（每帧）：
1) 完成率/失败数/拥挤度
2) 平均合法动作数、合法动作最小值、被全mask次数
3) 冲突任务（同一时刻target同一task的agent数>1）
4) 本步无效到达惩罚(last_step_reward)与累计step惩罚
5) 采样是否命中masked动作（理论应为0）
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation

from env.task_env import TaskEnv
from network import CTDE_Network


class VizConfig:
    USE_GPU = True

    CHECKPOINT_DIR = "./checkpoints_3_5"
    CHECKPOINT_BEST = "ctde_best.pth"
    CHECKPOINT_FINAL = "ctde_final.pth"

    ENV_CFG = dict(
        prio_range=(0.3, 1.0),
        per_species_range=(3, 5),
        species_range=(2, 3),
        tasks_range=(20, 40),
        comm_range=0.3,
        dynamic_task_discovery=True,
        discovery_probability=0.05,
        max_dynamic_tasks=5,
        seed=0,
        dt=1.0,
    )

    MAX_STEP_PER_EPISODE = 700
    ENV_TIME_LIMIT = 1000.0

    SAVE_PATH = "./animations/ctde_agent_trajectory_diagnostic.gif"
    FPS = 12
    INTERVAL_MS = 90
    DPI = 120

    # 若想更稳定复现，使用argmax；否则sample更贴近训练
    USE_GREEDY = False


def load_model(device, cfg: VizConfig):
    p_best = os.path.join(cfg.CHECKPOINT_DIR, cfg.CHECKPOINT_BEST)
    p_final = os.path.join(cfg.CHECKPOINT_DIR, cfg.CHECKPOINT_FINAL)
    ckpt_path = p_best if os.path.exists(p_best) else p_final
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {p_best} or {p_final}")

    net = CTDE_Network(
        self_dim=10,
        neighbor_dim=10,
        task_dim=15,
        global_agent_dim=13,
        global_task_dim=18,
        hidden=128,
        dropout=0.15,
        use_pos_encoding=True,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    print(f"✅ Loaded model: {ckpt_path}")
    return net


def _count_conflicts(agent_dic):
    """统计当前target冲突：同一task被>1个agent同时target。"""
    tgt = {}
    for aid, a in agent_dic.items():
        if a.get("target_type") == "task" and a.get("target_id") is not None:
            tid = int(a["target_id"])
            tgt.setdefault(tid, []).append(aid)

    conflict_map = {tid: aids for tid, aids in tgt.items() if len(aids) > 1}
    conflict_tasks = len(conflict_map)
    conflict_agents = sum(len(v) for v in conflict_map.values())
    max_same_target = max((len(v) for v in tgt.values()), default=0)
    return conflict_map, conflict_tasks, conflict_agents, max_same_target


def make_snapshot(env: TaskEnv, step_count: int, step_diag: dict, accum: dict):
    snap = {
        "time": float(env.current_time),
        "step": int(step_count),
        "agents": {},
        "tasks": {},
        "diag": {},
    }

    for aid, a in env.agent_dic.items():
        snap["agents"][aid] = {
            "location": a["location"].copy(),
            "species": int(a["species"]),
            "status": a.get("status", "idle"),
            "current_task": int(a.get("current_task", -1)),
            "target_type": a.get("target_type", None),
            "target_id": None if a.get("target_id", None) is None else int(a["target_id"]),
        }

    for tid, t in env.task_dic.items():
        snap["tasks"][tid] = {
            "location": t["location"].copy(),
            "finished": bool(t["finished"]),
            "feasible": bool(t["feasible_assignment"]),
            "members": list(t.get("members", [])),
            "priority": float(t.get("search_priority", 0.5)),
            "source": env.task_source.get(tid, "initial"),
        }

    total_tasks = len(env.task_dic)
    finished_tasks = sum(int(t["finished"]) for t in env.task_dic.values())
    failed_count = int(sum(len(a.get("failed_tasks", [])) for a in env.agent_dic.values()))

    conflict_map, conflict_tasks, conflict_agents, max_same_target = _count_conflicts(env.agent_dic)

    snap["diag"] = {
        "finished_tasks": int(finished_tasks),
        "total_tasks": int(total_tasks),
        "finish_rate": float(finished_tasks / total_tasks) if total_tasks > 0 else 0.0,
        "failed_count": failed_count,

        "step_avg_legal": float(step_diag.get("avg_legal", 0.0)),
        "step_min_legal": int(step_diag.get("min_legal", 0)),
        "step_full_mask_count": int(step_diag.get("full_mask_count", 0)),
        "step_mask_hit_count": int(step_diag.get("mask_hit_count", 0)),

        "step_last_step_reward": float(step_diag.get("last_step_reward", 0.0)),
        "acc_step_penalty_sum": float(accum.get("step_penalty_sum", 0.0)),

        "conflict_tasks": int(conflict_tasks),
        "conflict_agents": int(conflict_agents),
        "max_same_target": int(max_same_target),
        "conflict_map": {int(k): v for k, v in conflict_map.items()},
    }
    return snap


def run_episode_collect_frames(env: TaskEnv, net: CTDE_Network, device, cfg: VizConfig):
    env.reset(seed=cfg.ENV_CFG.get("seed", None))
    frames = []

    accum = {
        "step_penalty_sum": 0.0,
        "mask_hit_total": 0,
        "full_mask_total": 0,
        "total_decisions": 0,
    }

    # 初始帧
    init_step_diag = dict(
        avg_legal=0.0,
        min_legal=0,
        full_mask_count=0,
        mask_hit_count=0,
        last_step_reward=0.0,
    )
    frames.append(make_snapshot(env, 0, init_step_diag, accum))

    step_count = 0
    with torch.no_grad():
        while not env.finished and step_count < cfg.MAX_STEP_PER_EPISODE:
            if env.current_time > cfg.ENV_TIME_LIMIT:
                env.finished = True
                break

            decision_ids = sorted([aid for aid, a in env.agent_dic.items() if a.get("need_decision", False)])

            legal_counts = []
            step_full_mask_count = 0
            step_mask_hit_count = 0

            for aid in decision_ids:
                s, n, t, m, amap = env.get_local_obs(aid)

                legal_n = int((~m).sum())
                legal_counts.append(legal_n)
                if legal_n <= 0:
                    step_full_mask_count += 1

                ts = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                tn = torch.as_tensor(n, dtype=torch.float32, device=device).unsqueeze(0)
                tt = torch.as_tensor(t, dtype=torch.float32, device=device).unsqueeze(0)
                tm = torch.as_tensor(m, dtype=torch.bool, device=device).unsqueeze(0)

                dist = net.get_action(ts, tn, tt, tm)
                if cfg.USE_GREEDY:
                    act = int(torch.argmax(dist.probs[0]).item())
                else:
                    act = int(dist.sample()[0].item())

                # 诊断：是否命中masked动作
                if bool(m[act]):
                    step_mask_hit_count += 1
                    # 安全回退
                    act = 0

                env.apply_action(aid, act, amap)

                env.update_communication_graph()
                env.propagate_task_completion()

            env.step(dt=cfg.ENV_CFG["dt"])
            if step_count % 10 == 0:
                env.cleanup_failed_tasks_memory()

            # 记录本步诊断
            step_last_step_reward = float(getattr(env, "last_step_reward", 0.0))
            accum["step_penalty_sum"] += step_last_step_reward
            accum["mask_hit_total"] += step_mask_hit_count
            accum["full_mask_total"] += step_full_mask_count
            accum["total_decisions"] += len(decision_ids)

            if len(legal_counts) > 0:
                avg_legal = float(np.mean(legal_counts))
                min_legal = int(np.min(legal_counts))
            else:
                avg_legal = 0.0
                min_legal = 0

            step_diag = dict(
                avg_legal=avg_legal,
                min_legal=min_legal,
                full_mask_count=step_full_mask_count,
                mask_hit_count=step_mask_hit_count,
                last_step_reward=step_last_step_reward,
            )

            step_count += 1
            frames.append(make_snapshot(env, step_count, step_diag, accum))

    # 结束摘要
    if len(frames) > 0:
        final_diag = frames[-1]["diag"]
        print("====== Episode Diagnostic Summary ======")
        print(f"Steps: {len(frames)-1}")
        print(f"Finish: {final_diag['finished_tasks']}/{final_diag['total_tasks']} ({final_diag['finish_rate']:.2%})")
        print(f"Failed count: {final_diag['failed_count']}")
        print(f"Mask-hit total: {accum['mask_hit_total']} (should be 0)")
        print(f"Full-mask total: {accum['full_mask_total']}")
        print(f"Total decisions: {accum['total_decisions']}")
        print(f"Accum step penalty: {accum['step_penalty_sum']:.3f}")
        print("=======================================")

    return frames


def visualize_animation(env: TaskEnv, frames, save_path, cfg: VizConfig):
    if len(frames) == 0:
        print("❌ No frames.")
        return

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 8), dpi=cfg.DPI)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    # 右侧诊断文本区
    diag_text = fig.text(0.72, 0.5, "", fontsize=9, va="center", family="monospace")
    fig.subplots_adjust(right=0.70)

    n_species = len(env.depot_dic)
    colors = plt.cm.Set2(np.linspace(0, 1, max(3, n_species)))

    # depot
    for sid, depot in env.depot_dic.items():
        d = patches.Circle(
            depot["location"], 0.03,
            facecolor=colors[sid % len(colors)],
            edgecolor="black", linewidth=1.2
        )
        ax.add_patch(d)
        ax.text(depot["location"][0], depot["location"][1] - 0.045, f"Depot-{sid}",
                fontsize=8, ha="center")

    f0 = frames[0]

    task_patches = {}
    task_texts = {}
    prio_texts = {}

    for tid, t in f0["tasks"].items():
        loc = np.array(t["location"], dtype=float)
        poly = patches.RegularPolygon(
            xy=loc, numVertices=5, radius=0.020,
            facecolor="#3498DB", edgecolor="navy", alpha=0.85
        )
        ax.add_patch(poly)
        task_patches[tid] = poly
        task_texts[tid] = ax.text(loc[0], loc[1] + 0.030, f"T{tid}", fontsize=7, ha="center")
        prio_texts[tid] = ax.text(loc[0], loc[1] - 0.030, f"P:{t['priority']:.2f}", fontsize=7, ha="center", color="purple")

    agent_patches = {}
    agent_texts = {}
    for aid, a in f0["agents"].items():
        loc = np.array(a["location"], dtype=float)
        sp = int(a["species"])
        c = patches.Circle(
            loc, 0.014,
            facecolor=colors[sp % len(colors)],
            edgecolor="black", linewidth=1.0
        )
        ax.add_patch(c)
        agent_patches[aid] = c
        agent_texts[aid] = ax.text(loc[0], loc[1], f"A{aid}", fontsize=7, ha="center", va="center", color="white", weight="bold")

    title = ax.set_title("")

    def task_color(t):
        if t["finished"]:
            return "#2ECC71"
        if t["feasible"]:
            return "#F39C12"
        return "#3498DB"

    # 在 visualize_animation(...) 里，替换 update(i) 函数为这个版本

    def update(i):
        fr = frames[i]
        diag = fr["diag"]

        title.set_text(
            f"Step={fr['step']}  Time={fr['time']:.1f}  "
            f"Finish={diag['finished_tasks']}/{diag['total_tasks']} ({diag['finish_rate']:.1%})  "
            f"Fail={diag['failed_count']}"
        )

        # ===== 关键修复：对“新出现的动态任务”延迟创建patch =====
        for tid, t in fr["tasks"].items():
            if tid not in task_patches:
                loc = np.array(t["location"], dtype=float)
                poly = patches.RegularPolygon(
                    xy=loc, numVertices=5, radius=0.020,
                    facecolor="#3498DB", edgecolor="navy", alpha=0.85
                )
                ax.add_patch(poly)
                task_patches[tid] = poly
                task_texts[tid] = ax.text(loc[0], loc[1] + 0.030, f"T{tid}", fontsize=7, ha="center")
                prio_texts[tid] = ax.text(loc[0], loc[1] - 0.030, f"P:{t['priority']:.2f}", fontsize=7, ha="center",
                                          color="purple")

        # 更新任务
        for tid, t in fr["tasks"].items():
            loc = np.array(t["location"], dtype=float)
            p = task_patches[tid]
            p.xy = loc

            if t["finished"]:
                face = "#2ECC71"
            elif t["feasible"]:
                face = "#F39C12"
            else:
                face = "#3498DB"

            p.set_facecolor(face)
            p.set_alpha(0.95 if t["finished"] else 0.85)
            p.set_linestyle("--" if t.get("source", "initial") == "discovered" else "-")

            task_texts[tid].set_position((loc[0], loc[1] + 0.03))
            task_texts[tid].set_text(f"T{tid}")

            prio_texts[tid].set_position((loc[0], loc[1] - 0.03))
            prio_texts[tid].set_text(f"P:{t['priority']:.2f}")

        # 可选：隐藏已消失任务（一般不会发生，但防御性处理）
        current_tids = set(fr["tasks"].keys())
        for tid in list(task_patches.keys()):
            vis = (tid in current_tids)
            task_patches[tid].set_visible(vis)
            task_texts[tid].set_visible(vis)
            prio_texts[tid].set_visible(vis)

        # 更新agent
        for aid, a in fr["agents"].items():
            loc = np.array(a["location"], dtype=float)
            agent_patches[aid].center = (loc[0], loc[1])
            agent_texts[aid].set_position((loc[0], loc[1]))

            st = a.get("status", "idle")
            if st == "working":
                agent_patches[aid].set_linewidth(2.0)
            elif st == "moving":
                agent_patches[aid].set_linewidth(1.4)
            else:
                agent_patches[aid].set_linewidth(1.0)

        # 诊断文本
        conflict_map = diag.get("conflict_map", {})
        if len(conflict_map) > 0:
            items = list(conflict_map.items())[:3]
            conflict_str = ", ".join([f"T{k}:{len(v)}" for k, v in items])
        else:
            conflict_str = "None"

        diag_block = (
            f"[Mask Diagnostics]\\n"
            f"step_avg_legal      : {diag['step_avg_legal']:.2f}\\n"
            f"step_min_legal      : {diag['step_min_legal']}\\n"
            f"step_full_mask_cnt  : {diag['step_full_mask_count']}\\n"
            f"step_mask_hit_cnt   : {diag['step_mask_hit_count']}\\n\\n"
            f"[Conflict Diagnostics]\\n"
            f"conflict_tasks      : {diag['conflict_tasks']}\\n"
            f"conflict_agents     : {diag['conflict_agents']}\\n"
            f"max_same_target     : {diag['max_same_target']}\\n"
            f"conflict_map(top3)  : {conflict_str}\\n\\n"
            f"[Reward/Failure]\\n"
            f"step_last_reward    : {diag['step_last_step_reward']:.3f}\\n"
            f"acc_step_penalty    : {diag['acc_step_penalty_sum']:.3f}\\n"
            f"failed_count(total) : {diag['failed_count']}\\n"
        )
        diag_text.set_text(diag_block)
        return []

    ani = FuncAnimation(
        fig, update, frames=len(frames),
        interval=cfg.INTERVAL_MS, blit=False, repeat=True
    )
    ani.save(save_path, writer="pillow", fps=cfg.FPS, dpi=cfg.DPI)
    plt.close(fig)
    print(f"✅ Saved diagnostic GIF: {save_path}")

def evaluate_batch_by_task_ranges(
    net,
    device,
    base_env_cfg,
    task_ranges,
    num_episodes_per_range=30,
    max_step_per_episode=700,
    env_time_limit=1000.0,
    use_greedy=False,
    verbose=True,
):
    """
    批量评估：按不同任务数量区间统计结果

    返回:
    results: list[dict]
      每个dict对应一个task_range，包含:
      - success_rate
      - completion_rate_mean/std
      - failed_mean
      - mask_hit_mean
      - full_mask_mean
      - conflict_tasks_per_step_mean
      - conflict_agents_per_step_mean
      - max_same_target_mean
      - step_penalty_sum_mean
      - decisions_mean
      - env_time_mean
      - steps_mean
      - reward_mean/std
    """
    import copy
    import numpy as np
    import torch

    def run_one_episode(env_cfg, seed):
        env = TaskEnv(**env_cfg)
        env.reset(seed=seed)

        step_count = 0
        mask_hit_total = 0
        full_mask_total = 0
        total_decisions = 0
        step_penalty_sum = 0.0

        conflict_tasks_sum = 0.0
        conflict_agents_sum = 0.0
        max_same_target_sum = 0.0
        conflict_stat_steps = 0

        with torch.no_grad():
            while not env.finished and step_count < max_step_per_episode:
                if env.current_time > env_time_limit:
                    env.finished = True
                    break

                decision_ids = sorted([aid for aid, a in env.agent_dic.items() if a.get("need_decision", False)])
                total_decisions += len(decision_ids)

                # 当前步决策
                legal_counts = []
                for aid in decision_ids:
                    s, n, t, m, amap = env.get_local_obs(aid)

                    legal_n = int((~m).sum())
                    legal_counts.append(legal_n)
                    if legal_n <= 0:
                        full_mask_total += 1

                    ts = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    tn = torch.as_tensor(n, dtype=torch.float32, device=device).unsqueeze(0)
                    tt = torch.as_tensor(t, dtype=torch.float32, device=device).unsqueeze(0)
                    tm = torch.as_tensor(m, dtype=torch.bool, device=device).unsqueeze(0)

                    dist = net.get_action(ts, tn, tt, tm)
                    if use_greedy:
                        act = int(torch.argmax(dist.probs[0]).item())
                    else:
                        act = int(dist.sample()[0].item())

                    if bool(m[act]):
                        mask_hit_total += 1
                        act = 0  # fallback depot

                    env.apply_action(aid, act, amap)

                    # 顺序决策后的局部同步（与训练逻辑一致）
                    env.update_communication_graph()
                    env.propagate_task_completion()

                # 统计“当前目标冲突”（step级）
                tgt = {}
                for aid, a in env.agent_dic.items():
                    if a.get("target_type") == "task" and a.get("target_id") is not None:
                        tid = int(a["target_id"])
                        tgt[tid] = tgt.get(tid, 0) + 1
                conflict_tasks = sum(1 for _, c in tgt.items() if c > 1)
                conflict_agents = sum(c for _, c in tgt.items() if c > 1)
                max_same_target = max(tgt.values()) if len(tgt) > 0 else 0

                conflict_tasks_sum += float(conflict_tasks)
                conflict_agents_sum += float(conflict_agents)
                max_same_target_sum += float(max_same_target)
                conflict_stat_steps += 1

                env.step(dt=env_cfg.get("dt", 1.0))
                step_penalty_sum += float(getattr(env, "last_step_reward", 0.0))

                if step_count % 10 == 0:
                    env.cleanup_failed_tasks_memory()

                step_count += 1

        # episode统计
        ep_reward, finished_flags = env.get_episode_reward()
        total_tasks = len(env.task_dic)
        num_finished = int(sum(1 for t in env.task_dic.values() if t["finished"]))
        completion_rate = (num_finished / total_tasks) if total_tasks > 0 else 0.0
        success = 1.0 if completion_rate >= 0.9999 else 0.0
        failed_count = sum(len(a.get("failed_tasks", [])) for a in env.agent_dic.values())

        denom = max(conflict_stat_steps, 1)
        return dict(
            success=success,
            completion_rate=completion_rate,
            failed_count=float(failed_count),
            mask_hit_total=float(mask_hit_total),
            full_mask_total=float(full_mask_total),
            conflict_tasks_per_step=float(conflict_tasks_sum / denom),
            conflict_agents_per_step=float(conflict_agents_sum / denom),
            max_same_target_mean=float(max_same_target_sum / denom),
            step_penalty_sum=float(step_penalty_sum),
            total_decisions=float(total_decisions),
            env_time=float(env.current_time),
            steps=float(step_count),
            episode_reward=float(ep_reward),
            total_tasks=float(total_tasks),
            total_agents=float(len(env.agent_dic)),
        )

    results = []

    print("\n" + "=" * 90)
    print(f"Batch Evaluation by Task Ranges | episodes/range={num_episodes_per_range} | greedy={use_greedy}")
    print("=" * 90)

    for tr in task_ranges:
        env_cfg = copy.deepcopy(base_env_cfg)
        env_cfg["tasks_range"] = tuple(tr)

        ep_stats = []
        for ep in range(num_episodes_per_range):
            seed = ep  # 你也可以改成更随机的seed方案
            stat = run_one_episode(env_cfg, seed)
            ep_stats.append(stat)

            if verbose and ((ep + 1) % 10 == 0 or ep < 2):
                print(
                    f"[tasks={tr}] ep {ep+1}/{num_episodes_per_range} | "
                    f"succ={int(stat['success'])} compl={stat['completion_rate']:.2%} "
                    f"fail={stat['failed_count']:.0f} mask_hit={stat['mask_hit_total']:.0f} "
                    f"conf_t/step={stat['conflict_tasks_per_step']:.2f} "
                    f"rew={stat['episode_reward']:.1f}"
                )

        # 汇总
        def m(k):
            return float(np.mean([x[k] for x in ep_stats])) if len(ep_stats) else 0.0

        def s(k):
            return float(np.std([x[k] for x in ep_stats])) if len(ep_stats) else 0.0

        row = {
            "task_range": tuple(tr),
            "episodes": int(num_episodes_per_range),

            "success_rate": m("success"),
            "completion_rate_mean": m("completion_rate"),
            "completion_rate_std": s("completion_rate"),

            "failed_mean": m("failed_count"),
            "mask_hit_mean": m("mask_hit_total"),
            "full_mask_mean": m("full_mask_total"),

            "conflict_tasks_per_step_mean": m("conflict_tasks_per_step"),
            "conflict_agents_per_step_mean": m("conflict_agents_per_step"),
            "max_same_target_mean": m("max_same_target_mean"),

            "step_penalty_sum_mean": m("step_penalty_sum"),
            "step_penalty_sum_std": s("step_penalty_sum"),

            "decisions_mean": m("total_decisions"),
            "env_time_mean": m("env_time"),
            "steps_mean": m("steps"),

            "reward_mean": m("episode_reward"),
            "reward_std": s("episode_reward"),

            "tasks_mean": m("total_tasks"),
            "agents_mean": m("total_agents"),
        }
        results.append(row)

        print("-" * 90)
        print(
            f"TaskRange={row['task_range']} | Succ={row['success_rate']:.2%} | "
            f"Compl={row['completion_rate_mean']:.2%}±{row['completion_rate_std']:.2%} | "
            f"Fail={row['failed_mean']:.2f} | MaskHit={row['mask_hit_mean']:.2f} | "
            f"ConfTask/Step={row['conflict_tasks_per_step_mean']:.3f} | "
            f"Penalty={row['step_penalty_sum_mean']:.2f}±{row['step_penalty_sum_std']:.2f} | "
            f"Reward={row['reward_mean']:.1f}��{row['reward_std']:.1f}"
        )
        print("-" * 90)

    # 总表打印
    print("\n" + "=" * 90)
    print("Summary Table")
    print("=" * 90)
    for r in results:
        print(
            f"{r['task_range']}: "
            f"succ={r['success_rate']:.2%}, "
            f"compl={r['completion_rate_mean']:.2%}, "
            f"conf_t/step={r['conflict_tasks_per_step_mean']:.3f}, "
            f"mask_hit={r['mask_hit_mean']:.2f}, "
            f"penalty={r['step_penalty_sum_mean']:.2f}, "
            f"reward={r['reward_mean']:.1f}"
        )

    return results

# ===== 可选：把下面函数也加上，用于保存CSV并画一张任务数-成功率曲线 =====
def save_eval_results_and_plot(results, out_dir="./animations"):
    import os
    import csv
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "batch_eval_by_task_ranges.csv")

    # 保存CSV
    if len(results) > 0:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"✅ Saved CSV: {csv_path}")

    # 画图（x用task_range均值）
    x = [np.mean(r["task_range"]) for r in results]
    succ = [r["success_rate"] for r in results]
    compl = [r["completion_rate_mean"] for r in results]
    conf = [r["conflict_tasks_per_step_mean"] for r in results]

    plt.figure(figsize=(8, 5), dpi=130)
    plt.plot(x, succ, marker="o", label="Success Rate")
    plt.plot(x, compl, marker="s", label="Completion Rate")
    plt.xlabel("Average Task Count")
    plt.ylabel("Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    p1 = os.path.join(out_dir, "taskcount_vs_success_completion.png")
    plt.tight_layout()
    plt.savefig(p1)
    plt.close()
    print(f"✅ Saved plot: {p1}")

    plt.figure(figsize=(8, 5), dpi=130)
    plt.plot(x, conf, marker="^", color="tab:red", label="Conflict Tasks / Step")
    plt.xlabel("Average Task Count")
    plt.ylabel("Conflict Tasks / Step")
    plt.grid(alpha=0.3)
    plt.legend()
    p2 = os.path.join(out_dir, "taskcount_vs_conflict.png")
    plt.tight_layout()
    plt.savefig(p2)
    plt.close()
    print(f"✅ Saved plot: {p2}")

def main():
    cfg = VizConfig()
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.USE_GPU else "cpu")
    print(f"Using device: {device}")

    net = load_model(device, cfg)
    env = TaskEnv(**cfg.ENV_CFG)

    t0 = time.time()
    frames = run_episode_collect_frames(env, net, device, cfg)
    t1 = time.time()
    print(f"Collected {len(frames)} frames in {t1 - t0:.2f}s")

    # ===== 在 main() 末尾调用示例（直接粘进去） =====
    # 1) 先生成单条GIF（你原来的逻辑）
    visualize_animation(env, frames, cfg.SAVE_PATH, cfg)

    # 2) 再做批量评估（不同任务数量）
    task_ranges = [
        (8, 15),
        (12, 25),
        (20, 35),
        (35, 50),
        (45, 64),
        (55, 75),  # OOD
        (70, 90),  # OOD
    ]

    results_sample = evaluate_batch_by_task_ranges(
        net=net,
        device=device,
        base_env_cfg=cfg.ENV_CFG,
        task_ranges=task_ranges,
        num_episodes_per_range=30,
        max_step_per_episode=cfg.MAX_STEP_PER_EPISODE,
        env_time_limit=cfg.ENV_TIME_LIMIT,
        use_greedy=False,  # 采样策略
        verbose=True,
    )

    results_greedy = evaluate_batch_by_task_ranges(
        net=net,
        device=device,
        base_env_cfg=cfg.ENV_CFG,
        task_ranges=task_ranges,
        num_episodes_per_range=30,
        max_step_per_episode=cfg.MAX_STEP_PER_EPISODE,
        env_time_limit=cfg.ENV_TIME_LIMIT,
        use_greedy=True,  # 贪心策略
        verbose=False,
    )

    print("\n===== Compare Sample vs Greedy (success_rate) =====")
    for rs, rg in zip(results_sample, results_greedy):
        print(
            f"{rs['task_range']}: sample={rs['success_rate']:.2%}, "
            f"greedy={rg['success_rate']:.2%}, "
            f"delta={rg['success_rate'] - rs['success_rate']:+.2%}"
        )

    save_eval_results_and_plot(results_sample, out_dir="./animations")


if __name__ == "__main__":
    main()