# -*- coding: utf-8 -*-
"""
CTDE 模型测试程序（潮汐爆发 + Gossip 通信网络完全体）
- ✅ 动态长度输入（pad_to_batch_max 引入 pad_value）
- ✅ 修复 Mask 对齐崩溃（pad_value=1.0）
- ✅ 激活潮汐任务爆发 (Tidal Burst) 与信息传染
- ✅ 新增：多智能体规模扩展实验（small/medium/large）
- ✅ 新增：同时输出 total 与 initial 两套完成率
"""

import os
import re
import sys
import glob as _glob
import time
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from old_task_env import TaskEnv
from old_network import CTDE_Network


def pad_to_batch_max(batch_data_list, pad_value=0.0):
    if len(batch_data_list) == 0:
        return None, None

    batch_size = len(batch_data_list)
    max_len = max(x.shape[0] for x in batch_data_list)
    feature_dim = batch_data_list[0].shape[-1] if batch_data_list[0].dim() > 1 else 1

    if batch_data_list[0].dim() == 1:
        padded = torch.full((batch_size, max_len), pad_value,
                            dtype=batch_data_list[0].dtype,
                            device=batch_data_list[0].device)
    else:
        padded = torch.full((batch_size, max_len, feature_dim), pad_value,
                            dtype=batch_data_list[0].dtype,
                            device=batch_data_list[0].device)

    lengths = []
    for i, data in enumerate(batch_data_list):
        actual_len = data.shape[0]
        padded[i, :actual_len] = data
        lengths.append(actual_len)

    return padded, torch.tensor(lengths)


plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TestConfig:
    CHECKPOINT_PATH = "./checkpoints_old/7/1"
    USE_GPU = True
    # Actor KV 模式: "summary" / "tokens" / "auto"(自动尝试)
    ACTOR_KV_MODE = "tokens"

    DT = 1.0
    MAX_STEP_PER_EPISODE = 3000
    ENV_TIME_LIMIT = 3000.0
    NUM_TEST_EPISODES = 100

    EVAL_GREEDY = False

    DEADLINE_TIGHTNESS = (1.2, 1.8)
    PER_SPECIES_RANGE = (2, 5)
    SPECIES_RANGE = (3, 4)

    COMM_RANGE = 0.28
    DYNAMIC_TASK_DISCOVERY = True
    DISCOVERY_PROBABILITY = 0.07
    MAX_DYNAMIC_TASKS = 5

    DURATION_SCALE = 9

    SHAPING_CROWD_PENALTY = 0.15
    SHAPING_LONELY_BONUS = 0.10
    SHAPING_REDUNDANT_PENALTY = 0.30
    SHAPING_DISTANCE_PENALTY = 0.05

    WORKING_TIMEOUT_RANGE = (50.0, 120.0)
    WORKING_TIMEOUT_DECAY = 0.7
    WORKING_TIMEOUT_MIN = 15.0
    WORKING_TIMEOUT_PENALTY = -0.5
    COOLDOWN_DURATION = 10.0

    ENABLE_TIDAL_TASKS = True
    TIDAL_INTERVAL = 200.0
    TIDAL_BURST_RANGE = (8, 15)
    TIDAL_NOTIFY_RATIO = 0.2

    ENABLE_DISCOVERED_TASKS = False
    DISCOVERED_INTERVAL = 350.0
    DISCOVERED_BURST_RANGE = (3, 8)
    DISCOVERED_NOTIFY_RATIO = 0.1
    DISCOVERED_CLUSTER_RADIUS = 0.15


    # 任务负载档位（初始积压）
    TASK_RANGES = [
        (8, 15),
        (15, 25),
        (20, 30),
        (30, 40),
        (40, 50)
    ]

    # 新增：智能体规模档位
    AGENT_SCALES = [
        {"name": "small",  "species_range": (3, 4), "per_species_range": (4, 5)},   # avg 16 (方案B)
        {"name": "medium", "species_range": (3, 4), "per_species_range": (5, 7)},   # avg 21 (方案B)
        {"name": "large",  "species_range": (4, 5), "per_species_range": (5, 7)},   # avg 22 (不变)
    ]

    # success 阈值（基于 deadline 完成率）
    DEADLINE_SUCCESS_THRESHOLD = 0.70

    OUTPUT_DIR = "./test_results"
    FIGURE_DIR = "./test_results/figures"

    # checkpoint 自动选择优先级（当 CHECKPOINT_PATH 是目录或不存在时）
    # 顺序：最新的 ctde_ep_*.pth > ctde_final.pth > ctde_best.pth
    CHECKPOINT_CANDIDATES = [
        "ctde_final.pth",
        "ctde_best.pth",
    ]
    # 优先选取最新 episode 的 ctde_ep_*.pth（按 episode 编号）
    PREFER_LATEST_EP_CKPT = True

class MetricsCollector:
    def __init__(self, initial_task_range):
        self.metrics = defaultdict(list)
        self.initial_task_range = initial_task_range

    def add_episode(self, env, episode_reward, wall_time, env_time, step_count):
        task_ids = list(env.task_dic.keys())
        finished_flags = [env.task_dic[tid]["finished"] for tid in task_ids]

        num_total_tasks = len(task_ids)
        num_finished_tasks = sum(finished_flags)
        task_completion = num_finished_tasks / num_total_tasks if num_total_tasks > 0 else 0.0

        # deadline 完成（按时完成，不含过期）
        def _deadline_done(tid):
            t = env.task_dic[tid]
            return t.get("finished", False) and not t.get("expired", False)

        num_deadline_done = sum(1 for tid in task_ids if _deadline_done(tid))
        deadline_success_rate = num_deadline_done / num_total_tasks if num_total_tasks > 0 else 0.0

        # initial / tidal / discovered 分拆统计
        init_ids = [tid for tid, src in env.task_source.items() if src == "initial" and tid in env.task_dic]
        tidal_ids = [tid for tid, src in env.task_source.items() if src == "tidal" and tid in env.task_dic]
        disc_ids = [tid for tid, src in env.task_source.items() if src == "discovered" and tid in env.task_dic]

        init_finished = sum(1 for tid in init_ids if env.task_dic[tid]["finished"])
        tidal_finished = sum(1 for tid in tidal_ids if env.task_dic[tid]["finished"])
        disc_finished = sum(1 for tid in disc_ids if env.task_dic[tid]["finished"])

        init_completion = init_finished / len(init_ids) if len(init_ids) > 0 else 0.0
        tidal_completion = tidal_finished / len(tidal_ids) if len(tidal_ids) > 0 else 0.0
        disc_completion = disc_finished / len(disc_ids) if len(disc_ids) > 0 else 0.0

        init_deadline = sum(1 for tid in init_ids if _deadline_done(tid))
        tidal_deadline = sum(1 for tid in tidal_ids if _deadline_done(tid))
        disc_deadline = sum(1 for tid in disc_ids if _deadline_done(tid))

        init_deadline_rate = init_deadline / len(init_ids) if len(init_ids) > 0 else 0.0
        tidal_deadline_rate = tidal_deadline / len(tidal_ids) if len(tidal_ids) > 0 else 0.0
        disc_deadline_rate = disc_deadline / len(disc_ids) if len(disc_ids) > 0 else 0.0

        # binary success: 基于 deadline 完成率
        success = 1.0 if deadline_success_rate >= TestConfig.DEADLINE_SUCCESS_THRESHOLD else 0.0

        self.metrics["num_agents"].append(len(env.agent_dic))
        self.metrics["num_tasks"].append(num_total_tasks)
        self.metrics["num_initial_tasks"].append(len(init_ids))
        self.metrics["num_tidal_tasks"].append(len(tidal_ids))
        self.metrics["num_discovered_tasks"].append(len(disc_ids))

        self.metrics["task_completion_rate"].append(task_completion)
        self.metrics["deadline_success_rate"].append(deadline_success_rate)
        self.metrics["initial_task_completion_rate"].append(init_completion)
        self.metrics["initial_deadline_rate"].append(init_deadline_rate)
        self.metrics["tidal_task_completion_rate"].append(tidal_completion)
        self.metrics["tidal_deadline_rate"].append(tidal_deadline_rate)
        self.metrics["discovered_task_completion_rate"].append(disc_completion)
        self.metrics["discovered_deadline_rate"].append(disc_deadline_rate)

        self.metrics["episode_success"].append(success)
        self.metrics["episode_reward"].append(episode_reward)
        self.metrics["env_time"].append(env_time)
        self.metrics["wall_time"].append(wall_time)
        self.metrics["step_count"].append(step_count)

        res_util = float(getattr(env, "get_resource_utilization", lambda: 0.0)())
        idle_rate = max(0.0, 1.0 - res_util)
        self.metrics["resource_utilization"].append(res_util)
        self.metrics["idle_rate"].append(idle_rate)

        # deadline 指标
        n_expired = sum(1 for t in env.task_dic.values() if t.get("expired", False))
        self.metrics["expired_rate"].append(n_expired / max(1, num_total_tasks))
        slack_vals = [
            max(0.0, t["deadline"] - t.get("time_finish", env.current_time))
            for t in env.task_dic.values()
            if t.get("finished") and not t.get("expired")
        ]
        self.metrics["avg_slack"].append(np.mean(slack_vals) if slack_vals else 0.0)

    def get_summary(self):
        out = {}
        for k, v in self.metrics.items():
            out[f"{k}_mean"] = float(np.mean(v)) if v else 0.0
            out[f"{k}_std"] = float(np.std(v)) if v else 0.0

        out["success_rate"] = out.get("episode_success_mean", 0.0)
        out["success_count"] = int(np.sum(self.metrics["episode_success"])) if self.metrics["episode_success"] else 0
        out["total_episodes"] = len(self.metrics["episode_success"])
        out["success_threshold_deadline"] = TestConfig.DEADLINE_SUCCESS_THRESHOLD
        return out


def build_env(task_range, config: TestConfig, seed=None, species_range=None, per_species_range=None):
    env = TaskEnv(
        deadline_tightness=config.DEADLINE_TIGHTNESS,
        per_species_range=per_species_range if per_species_range is not None else config.PER_SPECIES_RANGE,
        species_range=species_range if species_range is not None else config.SPECIES_RANGE,
        tasks_range=task_range,
        duration_scale=config.DURATION_SCALE,
        comm_range=config.COMM_RANGE,
        dynamic_task_discovery=config.DYNAMIC_TASK_DISCOVERY,
        discovery_probability=config.DISCOVERY_PROBABILITY,
        max_dynamic_tasks=config.MAX_DYNAMIC_TASKS,
        dt=config.DT,
        seed=seed,

        shaping_crowd_penalty=config.SHAPING_CROWD_PENALTY,
        shaping_lonely_bonus=config.SHAPING_LONELY_BONUS,
        shaping_redundant_penalty=config.SHAPING_REDUNDANT_PENALTY,
        shaping_distance_penalty=config.SHAPING_DISTANCE_PENALTY,

        working_timeout_range=config.WORKING_TIMEOUT_RANGE,
        working_timeout_decay=config.WORKING_TIMEOUT_DECAY,
        working_timeout_min=config.WORKING_TIMEOUT_MIN,
        working_timeout_penalty=config.WORKING_TIMEOUT_PENALTY,
        cooldown_duration=config.COOLDOWN_DURATION,

        enable_tidal_tasks=config.ENABLE_TIDAL_TASKS,
        tidal_interval=config.TIDAL_INTERVAL,
        tidal_burst_range=config.TIDAL_BURST_RANGE,
        tidal_notify_ratio=config.TIDAL_NOTIFY_RATIO,

        enable_discovered_tasks=config.ENABLE_DISCOVERED_TASKS,
        discovered_interval=config.DISCOVERED_INTERVAL,
        discovered_burst_range=config.DISCOVERED_BURST_RANGE,
        discovered_notify_ratio=config.DISCOVERED_NOTIFY_RATIO,
        discovered_cluster_radius=config.DISCOVERED_CLUSTER_RADIUS,
        hard_expire=getattr(config, 'HARD_EXPIRE', False),
    )
    return env


def resolve_checkpoint_path(checkpoint_path):
    """支持文件路径、目录路径、以及不存在时的候选自动查找。
    目录模式下：优先选最新 episode 编号的 ctde_ep_*.pth，
    否则按 CHECKPOINT_CANDIDATES 顺序回退（final/best）。
    """
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    def _ep_num(path):
        m = re.search(r'ctde_ep_(\d+)\.pth', os.path.basename(path))
        return int(m.group(1)) if m else -1

    candidate_paths = []
    if os.path.isdir(checkpoint_path):
        if TestConfig.PREFER_LATEST_EP_CKPT:
            ep_ckpts = _glob.glob(os.path.join(checkpoint_path, "ctde_ep_*.pth"))
            ep_ckpts = [p for p in ep_ckpts if _ep_num(p) >= 0]
            ep_ckpts.sort(key=_ep_num, reverse=True)
            candidate_paths.extend(ep_ckpts)

        for name in TestConfig.CHECKPOINT_CANDIDATES:
            candidate_paths.append(os.path.join(checkpoint_path, name))

        other_pths = [
            os.path.join(checkpoint_path, f)
            for f in os.listdir(checkpoint_path)
            if f.endswith(".pth")
        ]
        other_pths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        candidate_paths.extend(other_pths)
    else:
        base_dir = os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else "."
        if os.path.isdir(base_dir) and TestConfig.PREFER_LATEST_EP_CKPT:
            ep_ckpts = _glob.glob(os.path.join(base_dir, "ctde_ep_*.pth"))
            ep_ckpts = [p for p in ep_ckpts if _ep_num(p) >= 0]
            ep_ckpts.sort(key=_ep_num, reverse=True)
            candidate_paths.extend(ep_ckpts)
        for name in TestConfig.CHECKPOINT_CANDIDATES:
            candidate_paths.append(os.path.join(base_dir, name))
        candidate_paths.append(checkpoint_path)

    seen = set()
    deduped = []
    for p in candidate_paths:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    for p in deduped:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}. Tried candidates: {deduped[:6]}"
    )


def _build_net(device, actor_kv_mode):
    return CTDE_Network(
        self_dim=10,
        neighbor_dim=10,
        task_dim=20,
        global_agent_dim=13,
        global_task_dim=23,
        hidden=128,
        dropout=0.15,
        use_pos_encoding=True,
        actor_kv_mode=actor_kv_mode,
    ).to(device)


def load_model(checkpoint_path, device):
    resolved_path = resolve_checkpoint_path(checkpoint_path)
    ckpt = torch.load(resolved_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    mode_cfg = str(TestConfig.ACTOR_KV_MODE).lower()
    modes_to_try = ["tokens", "summary"] if mode_cfg == "auto" else [mode_cfg]

    load_errors = []
    net = None
    used_mode = None

    # 先严格加载，确保结构完全对齐
    for mode in modes_to_try:
        try:
            trial_net = _build_net(device, mode)
            trial_net.load_state_dict(state_dict, strict=True)
            net = trial_net
            used_mode = mode
            break
        except Exception as e:
            load_errors.append((mode, str(e)))

    # 兜底：非严格加载（兼容轻微结构变更）
    if net is None:
        fallback_mode = modes_to_try[0]
        net = _build_net(device, fallback_mode)
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        used_mode = fallback_mode
        print("[WARN] Strict load failed, fallback to strict=False")
        print(f"   missing_keys: {len(missing)} | unexpected_keys: {len(unexpected)}")
        for mode, err in load_errors[:2]:
            print(f"   strict_fail[{mode}]: {err.splitlines()[0] if err else 'unknown'}")

    net.eval()

    print(f"[LOADED] {resolved_path}")
    print(f"   actor_kv_mode: {used_mode} (config={TestConfig.ACTOR_KV_MODE})")
    print(f"   episode: {ckpt.get('episode', 'N/A') if isinstance(ckpt, dict) else 'N/A'}")
    if isinstance(ckpt, dict) and "best_metric" in ckpt:
        print(f"   best_metric: {ckpt['best_metric']:.4f}")
    return net


@torch.no_grad()
def run_episode(env, net, device, config: TestConfig, seed=None):
    env.reset(seed=seed)

    step_count = 0
    t0 = time.time()

    while not env.finished and step_count < config.MAX_STEP_PER_EPISODE:
        if env.current_time >= config.ENV_TIME_LIMIT:
            env.finished = True
            break

        decision_ids = [aid for aid, a in env.agent_dic.items() if a.get("need_decision", False)]

        if decision_ids:
            batch_s, batch_n, batch_t, batch_m, action_maps = [], [], [], [], []

            for aid in decision_ids:
                s, n, t, m, amap = env.get_local_obs(aid)
                batch_s.append(torch.as_tensor(s, dtype=torch.float32))
                batch_n.append(torch.as_tensor(n, dtype=torch.float32))
                batch_t.append(torch.as_tensor(t, dtype=torch.float32))
                batch_m.append(torch.as_tensor(m, dtype=torch.bool))
                action_maps.append(amap)

            ts = torch.stack(batch_s).to(device)
            tn = torch.stack(batch_n).to(device)

            tt, _ = pad_to_batch_max(batch_t, pad_value=0.0)
            tt = tt.to(device)

            tm_exp = [m.unsqueeze(-1).float() for m in batch_m]
            tm, _ = pad_to_batch_max(tm_exp, pad_value=1.0)
            tm = tm.squeeze(-1).bool().to(device)

            dist = net.get_action(ts, tn, tt, tm)

            if config.EVAL_GREEDY:
                actions = torch.argmax(dist.probs, dim=-1)
            else:
                actions = dist.sample()

            for i, aid in enumerate(decision_ids):
                env.apply_action(aid, int(actions[i].item()), action_maps[i])

        env.step(dt=config.DT)
        if step_count % 10 == 0:
            env.cleanup_failed_tasks_memory()
        step_count += 1

    wall_time = time.time() - t0
    episode_reward, _ = env.get_episode_reward()
    return episode_reward, wall_time, env.current_time, step_count


def test_scalability_multi_agents(net, device, config: TestConfig):
    print("\n=== Test: Multi-Agent Scale × Task Load (Tidal Burst POMDP) ===")
    results = []

    for scale in config.AGENT_SCALES:
        s_name = scale["name"]
        s_species = scale["species_range"]
        s_per_species = scale["per_species_range"]

        print(f"\n================ Agent Scale: {s_name} | species={s_species}, per_species={s_per_species} ================")

        for task_range in config.TASK_RANGES:
            print(f"\nTask range (Initial): {task_range}")
            collector = MetricsCollector(initial_task_range=task_range)

            for ep in range(config.NUM_TEST_EPISODES):
                env = build_env(
                    task_range=task_range,
                    config=config,
                    seed=ep,
                    species_range=s_species,
                    per_species_range=s_per_species
                )
                r, wt, et, steps = run_episode(env, net, device, config, seed=ep)
                collector.add_episode(env, r, wt, et, steps)

                if (ep + 1) % 10 == 0:
                    dsr = collector.metrics['deadline_success_rate'][-1]
                    tc = collector.metrics['task_completion_rate'][-1]
                    total_tasks = collector.metrics['num_tasks'][-1]
                    er = collector.metrics['expired_rate'][-1]
                    sl = collector.metrics['avg_slack'][-1]
                    print(f"  Ep {ep + 1}/{config.NUM_TEST_EPISODES} | TotalTasks={total_tasks} | "
                          f"R={r:.2f} | DeadlineSucc={dsr:.2%} | TotalComp={tc:.2%} | "
                          f"Exp={er:.1%} | Slack={sl:.1f} | env_t={et:.1f}")

            s = collector.get_summary()
            s["agent_scale_name"] = s_name
            s["agent_species_range"] = list(s_species)
            s["agent_per_species_range"] = list(s_per_species)
            s["task_range"] = list(task_range)
            results.append(s)

            print(f"  Summary -> Success={s['success_rate']:.2%}, "
                  f"Agents={s['num_agents_mean']:.1f}, "
                  f"AvgTotalTasks={s['num_tasks_mean']:.1f}, "
                  f"DeadlineSucc={s['deadline_success_rate_mean']:.2%}, "
                  f"TotalComp={s['task_completion_rate_mean']:.2%}, "
                  f"InitDL={s['initial_deadline_rate_mean']:.2%}, "
                  f"TidalDL={s['tidal_deadline_rate_mean']:.2%}, "
                  f"ExpRate={s['expired_rate_mean']:.2%}, "
                  f"Slack={s['avg_slack_mean']:.1f}, "
                  f"Reward={s['episode_reward_mean']:.2f}")

    return results


def _collect_checkpoints(checkpoint_path, only):
    """根据 --checkpoint 和 --only 解析出要测试的 (resolved_path, tag_suffix) 列表。

    规则：
      - checkpoint 指向具体文件：只测它一个，tag_suffix 用文件名（去后缀）
      - checkpoint 指向目录：根据 only 决定测 latest / best / both
          * latest = resolve_checkpoint_path(dir)（受 PREFER_LATEST_EP_CKPT 控制）
          * best   = <dir>/ctde_best.pth
    """
    if os.path.isfile(checkpoint_path):
        base = os.path.splitext(os.path.basename(checkpoint_path))[0]
        return [(checkpoint_path, base)]

    if not os.path.isdir(checkpoint_path):
        # 让下游 resolve_checkpoint_path 抛 FileNotFoundError
        return [(resolve_checkpoint_path(checkpoint_path), "single")]

    pairs = []
    if only in ("latest", "both"):
        latest = resolve_checkpoint_path(checkpoint_path)
        pairs.append((latest, "latest"))

    if only in ("best", "both"):
        best_path = os.path.join(checkpoint_path, "ctde_best.pth")
        if os.path.isfile(best_path):
            # 避免与 latest 重复（极端情况下两者指向同一文件）
            if not any(os.path.abspath(p) == os.path.abspath(best_path) for p, _ in pairs):
                pairs.append((best_path, "best"))
        elif only == "best":
            raise FileNotFoundError(f"ctde_best.pth not found in {checkpoint_path}")
        else:
            print(f"[WARN] ctde_best.pth not found in {checkpoint_path}, skipping best.")

    return pairs


def _run_one_checkpoint(resolved_path, tag, device, config: TestConfig):
    print("\n" + "=" * 60)
    print(f"[TEST] checkpoint={resolved_path} | tag={tag}")
    print("=" * 60)

    net = load_model(resolved_path, device)
    results = test_scalability_multi_agents(net, device, config)

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config.OUTPUT_DIR, f"test_results_{tag}_{ts}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Results saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="CTDE Model Test")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file or directory (overrides TestConfig.CHECKPOINT_PATH)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag prefix for output filename. 目录模式下会自动追加 _latest / _best 区分。")
    parser.add_argument("--only", type=str, default="both", choices=["latest", "best", "both"],
                        help="目录模式下要测试哪一类 checkpoint（默认 both = latest + best）")
    args = parser.parse_args()

    config = TestConfig()
    if args.checkpoint:
        config.CHECKPOINT_PATH = args.checkpoint

    user_tag = args.tag or os.path.basename(config.CHECKPOINT_PATH.rstrip("/\\"))
    print(f"[START] CTDE Test | checkpoint={config.CHECKPOINT_PATH} | tag_prefix={user_tag} | only={args.only}")

    device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.FIGURE_DIR, exist_ok=True)

    ckpt_pairs = _collect_checkpoints(config.CHECKPOINT_PATH, args.only)
    if not ckpt_pairs:
        raise RuntimeError(f"No checkpoints to test under {config.CHECKPOINT_PATH}")

    print(f"[PLAN] Will test {len(ckpt_pairs)} checkpoint(s):")
    for p, suffix in ckpt_pairs:
        print(f"   - [{suffix}] {p}")

    output_paths = []
    for resolved_path, suffix in ckpt_pairs:
        # 单文件输入时 suffix 已经是文件 basename，不再追加
        if os.path.isfile(config.CHECKPOINT_PATH):
            run_tag = user_tag
        else:
            run_tag = f"{user_tag}_{suffix}"
        try:
            out = _run_one_checkpoint(resolved_path, run_tag, device, config)
            output_paths.append(out)
        except Exception as e:
            print(f"[ERROR] Test failed for {resolved_path}: {e}")

    print("\n" + "=" * 60)
    print(f"[ALL DONE] {len(output_paths)} result file(s):")
    for p in output_paths:
        print(f"   {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()