"""ROUND30 §4.2/§7 训练驱动：每个 (编码臂, 训练 seed) 一条 128 集运行。

契约（全部引用 ROUND24 冻结件，本文件不重新定义任何训练数学）：
- 新鲜初始化：`v2_controller.fresh_controller(training_seed)`，与 ROUND24/ROUND22 同规
  （"every arm starts at Controller(training_seed) and nothing else"）；不加载任何旧权重。
- optimizer：Adam(lr=3e-4) over `controller.parameters`（共享主干 + 动作头 + 任务价值头 +
  成本头四个模块，不含搜索/分配参数）。
- 每 4 集一批：`v2_trainer.ppo_update`（含 1e-6 log_prob 对齐）、`v2_trainer.batch_g_hat`
  （配对 immediate reference 标量，无梯度）、`v2_trainer.dual_update`；λ 批内冻结。
- 场景访问顺序：`common.scenario_order()`（ROUND24 冻结算法），同 seed 两臂共用。
- 每 episode 动作 RNG：`v2_run.action_seed(training_seed, episode_index)`，两臂同 seed 同
  派生（注意：同 seed 不意味着干预后每条消息遭遇相同丢包——信道 RNG 来自场景 seed 且发送
  序列分叉后消耗不同）。
- checkpoint：滚动 + ep0/64/128 里程碑（只写一次）；freeze 记录带 arm 与观测语义版本，
  跨臂/跨轮恢复由比较整个 freeze 字典显式拒绝。
- 被动诊断（启动前冻结）：每批 task/cost/g_hat/λ、双动作与 masked 样本数、critic 损失与
  逐模块梯度读数；每 episode 首报点的原始剩余期限、实际编码值、p(SEND) 与原动作——全部
  从既有 tape/rp 落盘件计算，不追加任何仿真采集。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

import r30_common as common
import r30_check as check
import r30_encoding as enc
import r30_run as runner
import v2_trainer as trainer
from v2_controller import fresh_controller, parameter_identity


def _milestone_identity(controller: Any) -> dict[str, Any]:
    return parameter_identity(controller)


def rolling_path(arm: str, training_seed: int) -> Path:
    return common.CHECKPOINTS / f"r30_train_{arm}_s{training_seed}.pt"


def milestone_path(arm: str, training_seed: int, episodes: int) -> Path:
    return common.CHECKPOINTS / f"r30_train_{arm}_s{training_seed}_ep{episodes}.pt"


def batch_record_path(arm: str, training_seed: int, batch: int) -> Path:
    return common.CHECKPOINTS / f"r30_batch_{arm}_s{training_seed}_b{batch}.json"


def module_digests(controller: Any) -> dict[str, str]:
    import hashlib

    digests: dict[str, str] = {}
    for module_name, module in (("trunk", controller.inner.trunk),
                                ("action_head", controller.inner.action_head),
                                ("task_value_head", controller.inner.value_head),
                                ("cost_head", controller.cost_head)):
        digest = hashlib.sha256()
        for name, tensor in sorted(module.named_parameters()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digests[module_name] = digest.hexdigest()[:16]
    return digests


def save_milestone(controller: Any, optimizer: Any, state: dict[str, Any], arm: str,
                   training_seed: int, episodes: int, *, freeze: dict[str, Any]) -> Path:
    """A write-once immutable milestone checkpoint; an existing file is never overwritten."""
    path = milestone_path(arm, training_seed, episodes)
    if path.exists():
        restored, _restored_optimizer, restored_state = trainer.load_checkpoint(path, freeze=freeze)
        same = (restored_state["training_seed"] == int(training_seed)
                and int(restored_state["episodes_done"]) == int(episodes)
                and int(restored_state["updates"]) == int(state["updates"])
                and int(restored_state["optimizer_steps"]) == int(state["optimizer_steps"])
                and float(restored_state["lambda"]) == float(state["lambda"])
                and _milestone_identity(restored)["parameter_sha256"]
                == _milestone_identity(controller)["parameter_sha256"])
        if not same:
            raise RuntimeError(f"milestone {path.name} already exists with a different state; "
                               "refusing to overwrite an immutable milestone")
        return path
    common.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    return trainer.save_checkpoint(controller, optimizer, state, path, freeze=freeze)


def first_report_records(rp_path: Path, tapes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Passive per-target first-candidate diagnostics from the already-written artefacts.

    ``remaining_time`` is the raw sender-side value (registry deadline minus the decision clock
    ``tape step - 1``, the rule verified against 756 unsaturated rows in CLOCK_RECHECK_3_2.json);
    ``encoded_column`` is what this arm's encoder put into the tape's feature row.
    """
    deadlines: dict[str, float] = {}
    with rp_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("event_type") == "detection" and row["target_id"] not in deadlines:
                detail = row["detail"]
                deadlines[row["target_id"]] = (float(row["timestamp"])
                                               + float(detail["deadline_offset"]))
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for tape in sorted(tapes, key=lambda item: int(item["step"])):
        step = int(tape["step"])
        for index, target in enumerate(tape["targets"]):
            if target in seen or target not in deadlines:
                continue
            seen.add(target)
            remaining = deadlines[target] - (step - 1)
            records.append({
                "target": target, "step": step,
                "raw_remaining_time": round(remaining, 6),
                "encoded_deadline_column": float(tape["features"][index][enc.DEADLINE_COLUMN]),
                "send_probability": float(tape["send_probabilities"].get(target, 0.0)),
                "original_action": ("SEND" if int(tape["actions"][target]) == 1 else "HOLD"),
                "actionable": target in tape["actionable"],
            })
    return records


def train_arm_seed(arm: str, training_seed: int, *, episodes: int, freeze: dict[str, Any],
                   reference_cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Train one (arm, seed) for ``episodes`` episodes, resuming from this round's rolling state."""
    checkpoint = rolling_path(arm, training_seed)
    if checkpoint.exists():
        controller, optimizer, state = trainer.load_checkpoint(checkpoint, freeze=freeze)
        print(f"{arm} seed {training_seed}: resumed at episode {state['episodes_done']} "
              f"lambda={state['lambda']:.4f} updates={state['updates']}", flush=True)
    else:
        controller = fresh_controller(training_seed)
        optimizer = torch.optim.Adam(controller.parameters, lr=common.v2c.LEARNING_RATE)
        state = trainer.fresh_training_state(training_seed)
        print(f"{arm} seed {training_seed}: fresh init Controller({training_seed}), "
              f"optimizer Adam lr={common.v2c.LEARNING_RATE}", flush=True)
        # The registered initialisation itself is the first immutable milestone; evaluation's
        # comparison of the two arms' initial parameters reads exactly these weights.
        save_milestone(controller, optimizer, state, arm, training_seed, 0, freeze=freeze)
        record_init_pairing(arm, training_seed, controller, optimizer, state)

    references = {scenario: float(row["on_time"]) for scenario, row in reference_cache.items()}
    order = common.scenario_order()
    before = parameter_identity(controller)
    digests_before = module_digests(controller)
    history: list[dict[str, Any]] = []
    started = time.time()

    while state["episodes_done"] < episodes:
        batch_index = state["episodes_done"] // common.v2c.EPISODES_PER_UPDATE
        lam = float(state["lambda"])          # frozen for the whole batch
        episodes_in_batch: list[dict[str, Any]] = []
        for _offset in range(common.v2c.EPISODES_PER_UPDATE):
            episode_index = state["episodes_done"]
            if episode_index >= episodes:
                break
            scenario = order[episode_index]
            if scenario not in references:
                raise RuntimeError(
                    f"{arm} seed {training_seed} episode {episode_index}: scenario {scenario} "
                    "has no cached immediate reference; the reference block must be complete "
                    "before training starts")
            episode = runner.run_training_episode(
                controller, arm=arm, training_seed=training_seed, episode_index=episode_index,
                scenario_seed=scenario, lam=lam)
            episodes_in_batch.append(episode)
            state["episodes_done"] += 1
            history.append({"episode_index": episode_index, "scenario": scenario,
                            "on_time": episode["on_time"], "spent_bytes": episode["spent_bytes"],
                            "emissions": episode["emissions"],
                            "decisions": (episode.get("collector_summary") or {}).get(
                                "decision_steps")})
            print(f"{arm} seed {training_seed} episode {episode_index}: scenario {scenario} "
                  f"on_time={episode['on_time']} bytes={episode['spent_bytes']} "
                  f"lambda={lam:.4f}", flush=True)

        if not episodes_in_batch:
            break
        update_report = trainer.ppo_update(controller, optimizer, episodes_in_batch,
                                           update_index=batch_index)
        g_record = trainer.batch_g_hat(episodes_in_batch, references)
        dual_record = trainer.dual_update(state["lambda"], g_record["g_hat"])
        state["lambda"] = dual_record["lambda_next"]
        state["lambda_history"].append({"batch": batch_index,
                                        "g_hat": dual_record["g_hat"],
                                        "lambda_before": dual_record["lambda_before"],
                                        "lambda_next": dual_record["lambda_next"]})
        state["updates"] += 1
        state["optimizer_steps"] += int(update_report.get("optimizer_steps", 0))
        after = parameter_identity(controller)
        if after["parameter_sha256"] == before["parameter_sha256"]:
            raise RuntimeError(
                f"{arm} seed {training_seed} batch {batch_index}: the optimizer did not change "
                "the communication controller")
        first_reports = []
        for episode in episodes_in_batch:
            rp_path = Path(episode["artefacts"]["lifecycle"])
            tapes = list(common.iter_jsonl(Path(episode["artefacts"]["tape"])))
            first_reports.append({
                "episode_index": int(episode["episode_index"]),
                "scenario": int(episode["scenario_seed"]),
                "records": first_report_records(rp_path, tapes),
            })
        batch_record = {
            "batch": batch_index, "arm": arm, "training_seed": training_seed,
            "episodes": [int(episode["episode_index"]) for episode in episodes_in_batch],
            "scenarios": [int(episode["scenario_seed"]) for episode in episodes_in_batch],
            "lambda_at_rollout": lam, "update": update_report,
            "constraint": g_record, "dual": dual_record,
            "controller_parameters_before": before, "controller_parameters_after": after,
            "controller_module_digests_before": digests_before,
            "controller_module_digests_after": module_digests(controller),
            "frozen_checkpoints": {
                str(path.relative_to(common.ROOT)): common.sha256_file(path)
                for path in common.v2c.CHECKPOINT_FILES},
            "optimizer_parameter_shapes": [list(parameter.shape) for group in optimizer.param_groups
                                           for parameter in group["params"]],
            "controller_parameter_shapes": [list(parameter.shape)
                                            for parameter in controller.parameters],
            "first_report_passive_records": first_reports,
        }
        common.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        batch_record_path(arm, training_seed, batch_index).write_text(
            json.dumps(batch_record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        trainer.save_checkpoint(controller, optimizer, state, checkpoint, freeze=freeze)
        if state["episodes_done"] in common.v2c.MILESTONE_EPISODES:
            save_milestone(controller, optimizer, state, arm, training_seed,
                           int(state["episodes_done"]), freeze=freeze)
        before = after
        digests_before = module_digests(controller)
        print(f"{arm} seed {training_seed} batch {batch_index}: samples="
              f"{update_report['samples']} opt_steps={update_report['optimizer_steps']} "
              f"g_hat={dual_record['g_hat']:+.4f} lambda {dual_record['lambda_before']:.4f} -> "
              f"{dual_record['lambda_next']:.4f}", flush=True)

    return {"arm": arm, "training_seed": training_seed, "episodes_done": state["episodes_done"],
            "updates": state["updates"], "optimizer_steps": state["optimizer_steps"],
            "lambda": state["lambda"], "lambda_history": state["lambda_history"],
            "seconds": round(time.time() - started, 2),
            "checkpoint": str(checkpoint),
            "controller_parameters": parameter_identity(controller),
            "history": history}


def record_init_pairing(arm: str, training_seed: int, controller: Any, optimizer: Any,
                        state: dict[str, Any]) -> None:
    """Evidence that both arms of one seed start from byte-identical parameters/optimizer/λ."""
    import hashlib

    import torch as _torch

    payload = {
        "arm": arm, "training_seed": training_seed,
        "controller_parameters": parameter_identity(controller),
        "module_digests": module_digests(controller),
        "optimizer_state_sha256": hashlib.sha256(
            json.dumps(optimizer.state_dict(), sort_keys=True, default=str)
            .encode("utf-8")).hexdigest(),
        "lambda": float(state["lambda"]),
        "fresh_state": {key: state[key] for key in
                        ("training_seed", "lambda", "updates", "episodes_done",
                         "optimizer_steps")},
    }
    path = common.OUTPUT_DIR / f"INIT_PAIRING_{arm}_s{training_seed}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")


def load_reference_cache() -> dict[int, dict[str, Any]]:
    cache: dict[int, dict[str, Any]] = {}
    for path in sorted(common.ROWS.glob("r30_reference_s*_a1.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        scenario = int(row["scenario_seed"])
        if scenario in cache and cache[scenario]["behaviour_sha256"] != row["behaviour_sha256"]:
            raise RuntimeError(f"scenario {scenario} has two different cached references")
        cache[scenario] = row
    return cache


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="train one (encoder arm, seed) run")
    parser.add_argument("--arm", required=True, choices=sorted(enc.ENCODERS))
    parser.add_argument("--seed", type=int, required=True, choices=list(common.TRAIN_SEEDS))
    parser.add_argument("--episodes", type=int, default=common.EPISODES_PER_SEED)
    args = parser.parse_args(argv)
    freeze = check.freeze_record(args.arm)
    cache = load_reference_cache()
    needed = set(common.scenario_order()[:args.episodes])
    missing = sorted(scenario for scenario in needed if scenario not in cache)
    if missing:
        raise SystemExit(f"reference cache is missing scenarios {missing[:8]}; run the "
                         "reference block first (it is part of the 1040 normal launches)")
    summary = train_arm_seed(args.arm, args.seed, episodes=args.episodes, freeze=freeze,
                             reference_cache=cache)
    print(json.dumps({key: summary[key] for key in
                      ("arm", "training_seed", "episodes_done", "updates", "optimizer_steps",
                       "lambda", "seconds")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
