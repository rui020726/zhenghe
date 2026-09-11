# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_multi_agents_results(
    input_json="./test_results/test_results_tidal_multi_agents.json",
    out_dir="./test_results/figures_multi_agents"
):
    os.makedirs(out_dir, exist_ok=True)

    sns.set(style="whitegrid", font="DejaVu Sans")
    plt.rcParams["axes.unicode_minus"] = False

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["task_range_label"] = df["task_range"].apply(lambda x: f"{x[0]}-{x[1]}")
    df["task_mid"] = df["task_range"].apply(lambda x: (x[0] + x[1]) / 2.0)

    scale_order = ["small", "medium", "large"]
    df["agent_scale_name"] = pd.Categorical(df["agent_scale_name"], categories=scale_order, ordered=True)
    df = df.sort_values(["agent_scale_name", "task_mid"]).reset_index(drop=True)

    # 保存汇总表
    table_cols = [
        "agent_scale_name", "task_range_label", "num_agents_mean", "num_tasks_mean",
        "task_completion_rate_mean", "stage_completion_rate_mean", "success_rate",
        "episode_reward_mean", "wall_time_mean", "initial_task_completion_rate_mean",
        "tidal_task_completion_rate_mean"
    ]
    summary_csv = os.path.join(out_dir, "summary_table.csv")
    df[table_cols].to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # 图1：完成率
    plt.figure(figsize=(10, 6))
    for s in scale_order:
        sub = df[df["agent_scale_name"] == s]
        plt.plot(sub["task_mid"], sub["task_completion_rate_mean"], marker="o", linewidth=2, label=f"{s} total")
        plt.plot(sub["task_mid"], sub["tidal_task_completion_rate_mean"], marker="x", linestyle="--", linewidth=1.8, label=f"{s} tidal")
    plt.title("Completion Rate vs Initial Task Load")
    plt.xlabel("Initial Task Range Midpoint")
    plt.ylabel("Completion Rate")
    plt.ylim(0, 1.02)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    fig1 = os.path.join(out_dir, "fig_completion_vs_load.png")
    plt.savefig(fig1, dpi=220)
    plt.close()

    # 图2：Success
    plt.figure(figsize=(10, 6))
    for s in scale_order:
        sub = df[df["agent_scale_name"] == s]
        plt.plot(sub["task_mid"], sub["success_rate"], marker="o", linewidth=2, label=s)
    plt.title("Success Rate vs Initial Task Load")
    plt.xlabel("Initial Task Range Midpoint")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    fig2 = os.path.join(out_dir, "fig_success_vs_load.png")
    plt.savefig(fig2, dpi=220)
    plt.close()

    # 图3：Reward
    plt.figure(figsize=(10, 6))
    for s in scale_order:
        sub = df[df["agent_scale_name"] == s]
        plt.plot(sub["task_mid"], sub["episode_reward_mean"], marker="o", linewidth=2, label=s)
    plt.title("Episode Reward vs Initial Task Load")
    plt.xlabel("Initial Task Range Midpoint")
    plt.ylabel("Episode Reward (mean)")
    plt.legend()
    plt.tight_layout()
    fig3 = os.path.join(out_dir, "fig_reward_vs_load.png")
    plt.savefig(fig3, dpi=220)
    plt.close()

    # 图4：Wall Time
    plt.figure(figsize=(10, 6))
    for s in scale_order:
        sub = df[df["agent_scale_name"] == s]
        plt.plot(sub["task_mid"], sub["wall_time_mean"], marker="o", linewidth=2, label=s)
    plt.title("Wall Time vs Initial Task Load")
    plt.xlabel("Initial Task Range Midpoint")
    plt.ylabel("Wall Time (s)")
    plt.legend()
    plt.tight_layout()
    fig4 = os.path.join(out_dir, "fig_walltime_vs_load.png")
    plt.savefig(fig4, dpi=220)
    plt.close()

    print("✅ 输出完成：")
    print(" - 表格:", summary_csv)
    print(" - 图1 :", fig1)
    print(" - 图2 :", fig2)
    print(" - 图3 :", fig3)
    print(" - 图4 :", fig4)


def main():
    plot_multi_agents_results(
        input_json="./test_results/test_results_tidal_multi_agents.json",
        out_dir="./test_results/figures_multi_agents"
    )


if __name__ == "__main__":
    main()