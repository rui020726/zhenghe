"""ROUND35 任务 3 补充（0 环境启动）：修复在真实数据上到底改变了什么、没改变什么。

对照对象是本轮 launch1 自己的行/tape/批记录与归档 ROUND32 `ALL_ROWS_s11` 同一批格：

- **没变**：episode 总数、每批样本数、minibatch 数、`g_hat`、λ（g_hat 只用 episode 总量，
  与逐行归属无关）；A 级格里连动作与 features 都逐行相同。
- **变了**：task_stream 的键（早一个 tick）、哪些动作行拿到入账（真实数据逐格计数）、
  由归属驱动的 surrogate / value loss / 梯度范数 / 更新后参数哈希。

这条区分很重要：它说明修复改的是**逐决策归属**，不是约束统计量本身；因此不能把后续任何
行为差异说成"约束被满足得更好"。所有数字只用于证明链路走通，不作为学习效果证据。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import r35_common as smoke

LAUNCH = "launch1"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(tag_directory: Path, pattern: str, episode: int) -> dict[str, Any]:
    paths = sorted(tag_directory.glob(pattern.format(episode=episode)))
    if len(paths) != 1:
        raise RuntimeError(f"{pattern} e{episode}: expected one row, found {len(paths)}")
    return load(paths[0])


def tape_lines(directory: Path, pattern: str, episode: int) -> list[dict[str, Any]]:
    paths = sorted(directory.glob(pattern.format(episode=episode)))
    if len(paths) != 1:
        raise RuntimeError(f"tape e{episode}: expected one file, found {len(paths)}")
    return [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    smoke_rows_dir = smoke.HERE / LAUNCH / "rows"
    smoke_logs_dir = smoke.HERE / LAUNCH / "logs"
    archived_rows_dir = smoke.ROUND32 / "rows"
    archived_logs_dir = smoke.ROUND32 / "logs"
    per_episode = {}
    moved_total = 0
    for episode in range(smoke.SMOKE_EPISODES):
        ours = rows(smoke_rows_dir, f"v2_train_s11_e{{episode}}_s*_a{{episode}}.json", episode)
        theirs = rows(archived_rows_dir,
                      f"r32_train_ALL_ROWS_s11_e{{episode}}_s*_a{{episode}}.json", episode)
        tape_ours = tape_lines(smoke_logs_dir,
                               f"v2_train_s11_e{{episode}}_*_tape.jsonl", episode)
        tape_theirs = tape_lines(archived_logs_dir,
                                 f"r32_train_ALL_ROWS_s11_e{{episode}}_*_tape.jsonl", episode)
        shared = min(len(tape_ours), len(tape_theirs))
        new_keys = sorted(int(key) for key in ours["task_stream"])
        old_keys = sorted(int(key) for key in theirs["task_stream"])
        steps = [int(line["step"]) for line in tape_ours]

        def owner(tick: int) -> Any:
            earlier = [step for step in steps if step <= tick]
            return max(earlier) if earlier else "prefix"

        moved = [key for key in old_keys if owner(key) != owner(key - 1)]
        moved_total += len(moved)
        per_episode[f"e{episode}"] = {
            "scenario_smoke": int(ours["scenario_seed"]),
            "scenario_archived": int(theirs["scenario_seed"]),
            "tape_rows": {"smoke": len(tape_ours), "archived": len(tape_theirs)},
            "features_identical_row_by_row": (len(tape_ours) == len(tape_theirs)
                                              and all(a["features"] == b["features"] for a, b
                                                      in zip(tape_ours, tape_theirs))),
            "actions_identical_row_by_row": (len(tape_ours) == len(tape_theirs)
                                             and all(a["actions"] == b["actions"] for a, b
                                                     in zip(tape_ours, tape_theirs))),
            "max_abs_log_prob_gap": (max(abs(float(a["log_prob"]) - float(b["log_prob"]))
                                        for a, b in zip(tape_ours, tape_theirs)) if shared else
                                     None),
            "episode_totals_unchanged": (int(ours["on_time"]) == int(theirs["on_time"])
                                         and int(ours["emissions"]) == int(theirs["emissions"])
                                         and int(ours["spent_bytes"]) == int(theirs["spent_bytes"])),
            "archived_keys": old_keys, "smoke_keys": new_keys,
            "keys_one_tick_earlier": new_keys == [key - 1 for key in old_keys],
            "credits_whose_owner_row_changed": len(moved),
            "owner_rows_affected": sorted({str(owner(key)) for key in moved}
                                          | {str(owner(key - 1)) for key in moved}),
            "weights_differ_at_this_rollout": bool(shared) and max(
                abs(float(a["log_prob"]) - float(b["log_prob"]))
                for a, b in zip(tape_ours, tape_theirs)) > 0.0,
        }
    batches = {}
    for batch in (0, 1):
        ours = load(smoke.HERE / LAUNCH / "checkpoints" / f"v2_batch_s11_b{batch}.json")
        theirs = load(smoke.ROUND32 / "checkpoints" / f"r32_batch_ALL_ROWS_s11_b{batch}.json")
        batches[f"b{batch}"] = {
            "samples": {"smoke": ours["update"]["samples"], "archived": theirs["update"]["samples"],
                        "equal": ours["update"]["samples"] == theirs["update"]["samples"]},
            "minibatches_equal": ours["update"]["minibatches"] == theirs["update"]["minibatches"],
            "optimizer_steps_equal": (ours["update"]["optimizer_steps"]
                                      == theirs["update"]["optimizer_steps"]),
            "g_hat": {"smoke": ours["dual"]["g_hat"], "archived": theirs["dual"]["g_hat"],
                      "equal": ours["dual"]["g_hat"] == theirs["dual"]["g_hat"]},
            "lambda_next": {"smoke": ours["dual"]["lambda_next"],
                            "archived": theirs["dual"]["lambda_next"],
                            "equal": ours["dual"]["lambda_next"] == theirs["dual"]["lambda_next"]},
            "lambda_at_rollout_equal": ours["lambda_at_rollout"] == theirs["lambda_at_rollout"],
            "episodes_and_scenarios_equal": (ours["episodes"] == theirs["episodes"]
                                             and ours["scenarios"] == theirs["scenarios"]),
            "unassigned_prefix_task_reward_equal": (
                ours["update"]["unassigned_prefix_task_reward"]
                == theirs["update"]["unassigned_prefix_task_reward"]),
            "changed_readings": {key: {"smoke": ours["update"][key],
                                       "archived": theirs["update"][key]}
                                 for key in ("policy_surrogate", "value_loss_task",
                                             "value_loss_cost", "entropy", "approx_kl",
                                             "clip_fraction", "gradient_norm_mean")
                                 if ours["update"][key] != theirs["update"][key]},
            "parameters_after_equal": (ours["controller_parameters_after"]["parameter_sha256"]
                                       == theirs["controller_parameters_after"]["parameter_sha256"]),
            "module_digests_after": {"smoke": ours["controller_module_digests_after"],
                                     "archived": theirs["controller_module_digests_after"]},
        }
    payload = {
        "schema": "round35-attribution-effect/1", "round": 35, "launch": LAUNCH,
        "environment_launches_used_by_this_script": 0,
        "compared": "this launch's own rows/tapes/batch records against the archived ROUND32 "
                    "ALL_ROWS_s11 cells of the same episodes and scenarios",
        "per_episode": per_episode,
        "credits_whose_owner_row_changed_total_over_eight_episodes": moved_total,
        "batches": batches,
        "what_did_not_change": "episode totals (on_time, emissions, spent bytes), the batch "
                               "sample count, the minibatch count, the optimizer-step count, "
                               "g_hat and lambda: g_hat is a paired episode-level statistic, so "
                               "the timing fix does not move it",
        "what_did_change": "which tick and which action row carries each credit, and therefore "
                           "the surrogate, both critic losses, the gradient norm and the "
                           "post-update parameter hash",
        "not_claimed": "no statement about learning quality; a 24-step smoke cannot show "
                       "whether the corrected attribution helps or hurts learning",
    }
    out = smoke.write_json("ATTRIBUTION_EFFECT.json", payload, refresh=True)
    print(json.dumps({"wrote": str(out), "credits_moved": moved_total,
                      "all_keys_shifted": all(value["keys_one_tick_earlier"]
                                              for value in per_episode.values()),
                      "g_hat_equal_batches": {key: value["g_hat"]["equal"]
                                              for key, value in batches.items()},
                      "params_after_equal": {key: value["parameters_after_equal"]
                                             for key, value in batches.items()},
                      "weights_differ_by_episode": {key: value["weights_differ_at_this_rollout"]
                                                    for key, value in per_episode.items()}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from pathlib import Path  # noqa: E402  (used by the loader annotations)
    raise SystemExit(main())
