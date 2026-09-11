# -*- coding: utf-8 -*-
"""
CTDE 时序任务协同分配 - 视频渲染程序 (超丝滑轨迹版)
- ✅ 修复了多智能体贪心对称性死锁（使用 sample 采样）
- ✅ 新增轨迹帧插值（Interpolation），解决瞬移问题，实现丝滑移动
- ✅ 支持导出为高清 MP4 视频，并带自动兼容回退
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

from old_task_env import TaskEnv
from old_network import CTDE_Network


# ================= 1. 环境与模型配置 =================
class DemoConfig:
    # ⚠️ 确保这里是你真实的模型路径
    CHECKPOINT_PATH = "./checkpoints_old/3/14/ctde_final.pth"
    USE_GPU = True

    # 测试场景配置
    TASKS_RANGE = (25, 35)
    SPECIES_RANGE = (3, 4)
    PER_SPECIES_RANGE = (2, 4)

    # 环境参数
    DT = 1.0
    MAX_STEP_PER_EPISODE = 1500
    ENV_TIME_LIMIT = 1500.0
    PRIO_RANGE = (0.3, 1.0)
    COMM_RANGE = 0.28
    DURATION_SCALE = 9

    # ================= 视频渲染核心设置 =================
    SAVE_FORMAT = "mp4"  # 'mp4' 或 'gif'
    VIDEO_NAME = "marl_rescue_smooth.mp4"
    FIGURE_DIR = "./test_results/figures"

    FPS = 30  # 视频帧率（30帧非常流畅）
    INTERPOLATE_FRAMES = 6  # 🌟 核心：每1个RL决策步之间，插入6帧补间动���！解决“瞬移”问题！
    EVAL_GREEDY = False  # 必须用 False，打破同质智能体死锁


# ================= 2. 动态Padding工具 =================
def pad_to_batch_max(batch_data_list):
    if len(batch_data_list) == 0:
        return None, None
    batch_size = len(batch_data_list)
    max_len = max(x.shape[0] for x in batch_data_list)
    feature_dim = batch_data_list[0].shape[-1] if batch_data_list[0].dim() > 1 else 1

    if batch_data_list[0].dim() == 1:
        padded = torch.zeros(batch_size, max_len, dtype=batch_data_list[0].dtype, device=batch_data_list[0].device)
    else:
        padded = torch.zeros(batch_size, max_len, feature_dim, dtype=batch_data_list[0].dtype,
                             device=batch_data_list[0].device)

    lengths = []
    for i, data in enumerate(batch_data_list):
        actual_len = data.shape[0]
        padded[i, :actual_len] = data
        lengths.append(actual_len)
    return padded, torch.tensor(lengths)


# ================= 3. 数据采集与平滑插值 =================
@torch.no_grad()
def collect_episode_data(config):
    print("⏳ 正在运行仿真并采集底层数据...")
    device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")

    net = CTDE_Network(self_dim=10, neighbor_dim=10, task_dim=15, global_agent_dim=13, global_task_dim=18, hidden=128,
                       dropout=0.15, use_pos_encoding=True).to(device)
    ckpt = torch.load(config.CHECKPOINT_PATH, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    env = TaskEnv(
        prio_range=config.PRIO_RANGE, per_species_range=config.PER_SPECIES_RANGE,
        species_range=config.SPECIES_RANGE, tasks_range=config.TASKS_RANGE,
        duration_scale=config.DURATION_SCALE, comm_range=config.COMM_RANGE,
        dynamic_task_discovery=True, discovery_probability=0.05, max_dynamic_tasks=5,
        dt=config.DT, seed=123
    )
    env.reset(seed=123)

    raw_history = []
    step_count = 0

    while not env.finished and step_count < config.MAX_STEP_PER_EPISODE:
        if env.current_time > config.ENV_TIME_LIMIT: break

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
            tt, _ = pad_to_batch_max(batch_t);
            tt = tt.to(device)
            tm_exp = [m.unsqueeze(-1).float() for m in batch_m]
            tm, _ = pad_to_batch_max(tm_exp);
            tm = tm.squeeze(-1).bool().to(device)

            dist = net.get_action(ts, tn, tt, tm)
            actions = torch.argmax(dist.logits, dim=-1) if config.EVAL_GREEDY else dist.sample()

            for i, aid in enumerate(decision_ids):
                env.apply_action(aid, int(actions[i].item()), action_maps[i])

        env.step(dt=config.DT)
        if step_count % 10 == 0: env.cleanup_failed_tasks_memory()

        # 记录每一步的离散坐标
        frame_data = {"time": env.current_time, "agents": [], "tasks": [], "depots": []}
        for d in env.depot_dic.values():
            frame_data["depots"].append(d["location"].copy())

        for aid, a in env.agent_dic.items():
            tgt_pos = a.get("target_pos")
            frame_data["agents"].append({
                "id": aid, "species": a["species"], "loc": a["location"].copy(),
                "status": a["status"], "target_pos": tgt_pos.copy() if tgt_pos is not None else None
            })

        for tid, t in env.task_dic.items():
            frame_data["tasks"].append({
                "id": tid, "loc": t["location"].copy(), "stage_idx": t["current_stage_index"],
                "total_stages": t["total_stages"], "finished": t["finished"],
                "is_dynamic": env.task_source.get(tid) == "discovered"
            })
        raw_history.append(frame_data)
        step_count += 1

    total_stages = sum(t["total_stages"] for t in env.task_dic.values())
    completed_stages = sum(t["current_stage_index"] for t in env.task_dic.values())
    stage_comp = completed_stages / total_stages if total_stages > 0 else 0

    print(f"✅ 底层数据采集完毕！环境步数: {len(raw_history)}。开始生成平滑补间帧...")

    # 🌟 轨迹插值处理，彻底告别瞬移！
    smooth_history = []
    for i in range(len(raw_history) - 1):
        f1 = raw_history[i]
        f2 = raw_history[i + 1]

        for k in range(config.INTERPOLATE_FRAMES):
            alpha = k / config.INTERPOLATE_FRAMES
            interp_frame = {
                "time": f1["time"] + alpha * (f2["time"] - f1["time"]),
                "depots": f1["depots"], "tasks": f1["tasks"], "agents": []
            }

            # 对小车坐标进行数学线性平滑
            for a1, a2 in zip(f1["agents"], f2["agents"]):
                interp_loc = a1["loc"] * (1 - alpha) + a2["loc"] * alpha
                interp_frame["agents"].append({
                    "id": a1["id"], "species": a1["species"], "loc": interp_loc,
                    "status": a1["status"], "target_pos": a1["target_pos"]
                })
            smooth_history.append(interp_frame)

    # 补上最后一帧
    smooth_history.append(raw_history[-1])
    print(f"🎬 补间完成！共生成 {len(smooth_history)} 帧高清画面。")
    return smooth_history, stage_comp


# ================= 4. 高清视频渲染逻辑 =================
def render_animation(history, final_comp, config):
    print(f"🎥 正在渲染视频 (共 {len(history)} 帧，预计耗时2-3分钟)...")

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)

    SPECIES_MARKERS = ['o', '^', 's', 'P', 'D']
    SPECIES_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#f8f9fa')

        data = history[frame_idx]

        # 画基地
        for loc in data["depots"]:
            ax.scatter(loc[0], loc[1], s=400, marker='H', color='#555555', alpha=0.4, zorder=1)
            ax.text(loc[0], loc[1] - 0.03, "BASE", ha='center', fontsize=10, fontweight='bold', color='#555555')

        # 画任务
        for t in data["tasks"]:
            x, y = t["loc"]
            is_fin = t["finished"]
            stage_idx = t["stage_idx"]
            total = t["total_stages"]

            if is_fin:
                ax.scatter(x, y, s=150, marker='*', color='#cccccc', edgecolors='black', zorder=2)
                ax.text(x, y + 0.02, f"T{t['id']} [✔]", ha='center', fontsize=8, color='gray')
            else:
                color = plt.cm.autumn(1.0 - (stage_idx / total))
                ax.scatter(x, y, s=250, marker='*', color=color, edgecolors='black', zorder=2)
                label = f"T{t['id']} ({stage_idx + 1}/{total})"
                if t["is_dynamic"]: label = "⚠️" + label
                ax.text(x, y + 0.02, label, ha='center', fontsize=8, fontweight='bold', color='black')

        # 画智能体及连线
        for a in data["agents"]:
            x, y = a["loc"]
            sp = a["species"]
            status = a["status"]
            tgt = a["target_pos"]

            color = SPECIES_COLORS[sp % len(SPECIES_COLORS)]
            marker = SPECIES_MARKERS[sp % len(SPECIES_MARKERS)]

            # 丝滑的移动引导线
            if tgt is not None:
                if status == "working":
                    ax.plot([x, tgt[0]], [y, tgt[1]], color=color, linestyle='-', linewidth=3.0, alpha=0.9, zorder=1)
                elif status == "moving":
                    ax.plot([x, tgt[0]], [y, tgt[1]], color=color, linestyle='--', linewidth=1.5, alpha=0.4, zorder=1)

            # 智能体本体
            size = 150 if status == "working" else 100
            edge = 'black' if status == "working" else 'white'
            lw = 2 if status == "working" else 1
            ax.scatter(x, y, s=size, marker=marker, color=color, edgecolors=edge, linewidths=lw, zorder=3)

            if status == "working":
                ax.text(x, y - 0.03, "Work", ha='center', fontsize=8, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))

        # 标题与进度条
        progress = frame_idx / max(1, len(history) - 1)
        title = f"Multi-Agent Sequential Rescue Assignment\n" \
                f"Env Time: {data['time']:.1f}  |  Stage Completion: {final_comp * progress:.1%}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

        # 图例
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', label='Unfinished Task Stage', markerfacecolor='red', markersize=15,
                   markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='Rescue Agent (Moving)', markerfacecolor='#1f77b4',
                   markersize=10),
            Line2D([0], [0], color='gray', lw=3, ls='-', label='Agent Working')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=1000 / config.FPS, blit=False)

    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    save_path = os.path.join(config.FIGURE_DIR, config.VIDEO_NAME)

    # 尝试打包为 MP4
    if config.SAVE_FORMAT == "mp4":
        print(f"🎬 正在编码导出 MP4 视频... (要求电脑安装了 ffmpeg)")
        try:
            writer = animation.FFMpegWriter(fps=config.FPS, bitrate=2500)
            ani.save(save_path, writer=writer, dpi=150)
            print(f"\n🎉 帅呆了！高清 MP4 视频已保存至：\n👉 {os.path.abspath(save_path)}")
        except Exception as e:
            print(f"\n⚠️ 警告: 导出 MP4 失败！可能因为系统未安装 ffmpeg 或未配环境变量。")
            print(f"   错误信息: {e}")
            print(f"🚀 系统正在为您自动降级：转存为高质量平滑 GIF 动图...")
            gif_path = save_path.replace(".mp4", ".gif")
            ani.save(gif_path, writer='pillow', fps=config.FPS, dpi=120)
            print(f"🎉 高质量 GIF 动图已成功保存：\n👉 {os.path.abspath(gif_path)}")
    else:
        gif_path = save_path.replace(".mp4", ".gif")
        ani.save(gif_path, writer='pillow', fps=config.FPS, dpi=120)
        print(f"🎉 GIF 动图已保存：\n👉 {os.path.abspath(gif_path)}")


if __name__ == "__main__":
    config = DemoConfig()
    if not os.path.exists(config.CHECKPOINT_PATH):
        print(f"❌ 找不到模型：{config.CHECKPOINT_PATH}。请检查路径。")
    else:
        history_data, final_stage_comp = collect_episode_data(config)
        if len(history_data) > 0:
            render_animation(history_data, final_stage_comp, config)