"""ROUND32 任务 C/D 训练驱动：每个 (归约臂, 训练 seed) 一条 128 集运行。

契约（全部引用冻结件，本文件不重新定义任何训练数学）：
- 新鲜初始化：`v2_controller.fresh_controller(training_seed)`，与 ROUND22/24/30 同规
  （"every arm starts at Controller(training_seed) and nothing else"）；不加载任何旧权重，
  也不从 ROUND30/31 的 checkpoint 热启动（`r32_trainer.load_checkpoint` 会拒绝无归约标签的旧件）；
- optimizer：Adam(lr=3e-4) over `controller.parameters`（共享主干 + 动作头 + 任务价值头 +
  成本头四个模块，不含搜索/分配参数）；
- 每 4 集一批：`r32_trainer.ppo_update(reduction=...)`（内部逐字复用冻结的 batch_samples /
  GAE / likelihood 对齐 / critic / clip / 全局裁剪），`v2_trainer.batch_g_hat`
  （配对 immediate reference 标量，无梯度）、`v2_trainer.dual_update`；λ 批内冻结；
- 场景访问顺序：`common.scenario_order()`（ROUND24 冻结算法），同 seed 两臂共用；
- 每 episode 动作 RNG：`v2_run.action_seed(training_seed, episode_index)`，两臂同 seed 同派生
  （注意：同 seed 不意味着干预后每条消息遭遇相同丢包——信道 RNG 来自场景 seed 且发送序列
  分叉后消耗不同）；
- **任务 D 的额外证据**：批索引 0/15/31 在 `ppo_update` **之前**写一次"更新前完整 checkpoint"
  （模型 + optimizer + λ + RNG + 数据位置 + 编码器与归约模式），对应 tape 本来就已落盘；
  每批还落盘逐 minibatch 的 N/M/M·N、各 loss、KL、clip fraction、裁剪前后与模块级梯度范数、
  非有限标记，以及权重前后哈希。写这些读数不触碰 RNG、动作或参数更新。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

import r32_common as common  # noqa: F401  (installs the shared sys.path before any archived-round import)
import r30_encoding as enc
import r32_run as runner
from r32_freeze import freeze_record
import r32_trainer as trainer
import v2_trainer as frozen_trainer
from v2_controller import fresh_controller, parameter_identity

#: 逐 minibatch 读数在批记录里的字段名（任务 D）；批记录里其余字段沿用 ROUND30 的结构。
MODULES = ("trunk", "action_head", "task_value_head", "cost_head")


def rolling_path(reduction: str, training_seed: int) -> Path:
    return common.CHECKPOINTS / f"r32_train_{reduction}_s{training_seed}.pt"


def milestone_path(reduction: str, training_seed: int, episodes: int) -> Path:
    return common.CHECKPOINTS / f"r32_train_{reduction}_s{training_seed}_ep{episodes}.pt"


def pre_update_path(reduction: str, training_seed: int, batch: int) -> Path:
    return common.CHECKPOINTS / f"r32_pre_{reduction}_s{training_seed}_b{batch}.pt"


def init_pairing_path(reduction: str, training_seed: int) -> Path:
    return common.OUTPUT_DIR / f"INIT_PAIRING_{reduction}_s{training_seed}.json"


def batch_record_path(reduction: str, training_seed: int, batch: int) -> Path:
    return common.CHECKPOINTS / f"r32_batch_{reduction}_s{training_seed}_b{batch}.json"


def module_digests(controller: Any) -> dict[str, str]:
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


def save_immutable(controller: Any, optimizer: Any, state: dict[str, Any], reduction: str,
                   training_seed: int, path: Path, *, freeze: dict[str, Any],
                   what: str) -> Path:
    """A write-once checkpoint; an existing file is never overwritten."""
    if path.exists():
        restored, _restored_optimizer, restored_state = trainer.load_checkpoint(
            path, freeze=freeze, reduction=reduction)
        same = (restored_state["training_seed"] == int(training_seed)
                and int(restored_state["updates"]) == int(state["updates"])
                and int(restored_state["optimizer_steps"]) == int(state["optimizer_steps"])
                and float(restored_state["lambda"]) == float(state["lambda"])
                and int(restored_state["episodes_done"]) == int(state["episodes_done"])
                and parameter_identity(restored)["parameter_sha256"]
                == parameter_identity(controller)["parameter_sha256"])
        if not same:
            raise RuntimeError(f"{path.name}: a {what} checkpoint already exists with a "
                               "different state; refusing to overwrite an immutable artefact")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    return trainer.save_checkpoint(controller, optimizer, state, path, freeze=freeze,
                                   reduction=reduction)


def save_milestone(controller: Any, optimizer: Any, state: dict[str, Any], reduction: str,
                   training_seed: int, episodes: int, *, freeze: dict[str, Any]) -> Path:
    return save_immutable(controller, optimizer, state, reduction, training_seed,
                          milestone_path(reduction, training_seed, episodes), freeze=freeze,
                          what=f"ep{episodes} milestone")


def first_report_records(rp_path: Path, tapes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Passive per-target first-candidate diagnostics from the already-written artefacts."""
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


def summarize_reads(reads: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-minibatch readings of one batch (pure arithmetic on stored numbers)."""
    if not reads:
        return {"minibatches": 0}
    def stat(key: str) -> dict[str, Any]:
        values = [float(read[key]) for read in reads if read.get(key) is not None]
        if not values:
            return {"available": False}
        return {"available": True, "min": round(min(values), 9), "max": round(max(values), 9),
                "mean": round(sum(values) / len(values), 9)}
    n_values = [int(read["n_rows"]) for read in reads]
    m_values = [int(read["m_active_rows"]) for read in reads]
    return {
        "minibatches": len(reads),
        "n_rows": {"min": min(n_values), "max": max(n_values), "sum": sum(n_values)},
        "m_active_rows": {"min": min(m_values), "max": max(m_values), "sum": sum(m_values)},
        "m_over_n": stat("m_over_n"),
        "pre_clip_total_norm": stat("pre_clip_total_norm"),
        "post_clip_total_norm": stat("post_clip_total_norm"),
        "clipping_minibatches": sum(1 for read in reads if read["clipping_triggered"]),
        "clip_scale": stat("clip_scale"),
        "approx_kl": stat("approx_kl"),
        "clip_fraction": stat("clip_fraction"),
        "entropy_term": stat("entropy_term"),
        "value_loss_task": stat("value_loss_task"),
        "value_loss_cost": stat("value_loss_cost"),
        "actor_surrogate_mean": stat("actor_surrogate_mean"),
        "actor_surrogate_mean_all_rows_reference": stat(
            "actor_surrogate_mean_all_rows_reference"),
        "non_finite_minibatches": sum(1 for read in reads if not read["all_finite"]),
        "module_pre_clip_l1_mean": {
            name: round(sum(float(read["pre_clip_module_l1"][name]) for read in reads)
                        / len(reads), 9) for name in MODULES},
        "module_pre_clip_l2_mean": {
            name: round(sum(float(read["pre_clip_module_l2"][name]) for read in reads)
                        / len(reads), 9) for name in MODULES},
    }


def train_arm_seed(reduction: str, training_seed: int, *, episodes: int,
                   freeze: dict[str, Any],
                   reference_cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Train one (reduction, seed) run for ``episodes`` episodes, resuming this round's state."""
    checkpoint = rolling_path(reduction, training_seed)
    if checkpoint.exists():
        controller, optimizer, state = trainer.load_checkpoint(
            checkpoint, freeze=freeze, reduction=reduction)
        print(f"{reduction} seed {training_seed}: resumed at episode {state['episodes_done']} "
              f"lambda={state['lambda']:.4f} updates={state['updates']}", flush=True)
    else:
        controller = fresh_controller(training_seed)
        optimizer = torch.optim.Adam(controller.parameters, lr=common.v2c.LEARNING_RATE)
        state = frozen_trainer.fresh_training_state(training_seed)
        print(f"{reduction} seed {training_seed}: fresh init Controller({training_seed}), "
              f"optimizer Adam lr={common.v2c.LEARNING_RATE}", flush=True)
        save_immutable(controller, optimizer, state, reduction, training_seed,
                       milestone_path(reduction, training_seed, 0), freeze=freeze,
                       what="ep0 initial milestone")
        record_init_pairing(reduction, training_seed, controller, optimizer, state)

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
                    f"{reduction} seed {training_seed} episode {episode_index}: scenario "
                    f"{scenario} has no cached immediate reference; the reference block must be "
                    "complete before training starts")
            episode = runner.run_training_episode(
                controller, reduction=reduction, training_seed=training_seed,
                episode_index=episode_index, scenario_seed=scenario, lam=lam)
            episodes_in_batch.append(episode)
            state["episodes_done"] += 1
            history.append({"episode_index": episode_index, "scenario": scenario,
                            "on_time": episode["on_time"], "spent_bytes": episode["spent_bytes"],
                            "emissions": episode["emissions"],
                            "decisions": (episode.get("collector_summary") or {}).get(
                                "decision_steps")})
            print(f"{reduction} seed {training_seed} episode {episode_index}: "
                  f"scenario {scenario} on_time={episode['on_time']} "
                  f"bytes={episode['spent_bytes']} lambda={lam:.4f}", flush=True)

        if not episodes_in_batch:
            break
        # ---- task D: the pre-update checkpoint is written BEFORE any weight moves, so a later
        # round can analyse the true update start instead of a post-update snapshot
        pre_update_saved = None
        if batch_index in common.PRE_UPDATE_BATCHES:
            pre_update_saved = save_immutable(
                controller, optimizer, state, reduction, training_seed,
                pre_update_path(reduction, training_seed, batch_index), freeze=freeze,
                what=f"batch {batch_index} pre-update checkpoint")
        update_report = trainer.ppo_update(controller, optimizer, episodes_in_batch,
                                           update_index=batch_index, reduction=reduction)
        g_record = frozen_trainer.batch_g_hat(episodes_in_batch, references)
        dual_record = frozen_trainer.dual_update(state["lambda"], g_record["g_hat"])
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
                f"{reduction} seed {training_seed} batch {batch_index}: the optimizer did not "
                "change the communication controller")
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
            "batch": batch_index, "reduction": reduction, "arm": reduction,
            "encoder_arm": runner.ENCODER_ARM,
            "feature_layout": enc.ENCODERS[runner.ENCODER_ARM]["layout"],
            "training_seed": training_seed,
            "episodes": [int(episode["episode_index"]) for episode in episodes_in_batch],
            "scenarios": [int(episode["scenario_seed"]) for episode in episodes_in_batch],
            "lambda_at_rollout": lam, "update": update_report,
            "update_read_summary": summarize_reads(update_report.get("minibatch_reads") or []),
            "constraint": g_record, "dual": dual_record,
            "pre_update_checkpoint": (str(pre_update_saved.relative_to(common.ROOT))
                                      if pre_update_saved else None),
            "pre_update_checkpoint_saved_for_batches": list(common.PRE_UPDATE_BATCHES),
            "controller_parameters_before": before, "controller_parameters_after": after,
            "controller_module_digests_before": digests_before,
            "controller_module_digests_after": module_digests(controller),
            "frozen_checkpoints": {
                str(path.relative_to(common.ROOT)): common.sha256_file(path)
                for path in common.CHECKPOINT_FILES},
            "optimizer_parameter_shapes": [list(parameter.shape) for group in optimizer.param_groups
                                           for parameter in group["params"]],
            "controller_parameter_shapes": [list(parameter.shape)
                                            for parameter in controller.parameters],
            "first_report_passive_records": first_reports,
        }
        common.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        batch_record_path(reduction, training_seed, batch_index).write_text(
            json.dumps(batch_record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        trainer.save_checkpoint(controller, optimizer, state, checkpoint, freeze=freeze,
                                reduction=reduction)
        if state["episodes_done"] in common.v2c.MILESTONE_EPISODES:
            save_milestone(controller, optimizer, state, reduction, training_seed,
                           int(state["episodes_done"]), freeze=freeze)
        before = after
        digests_before = module_digests(controller)
        reads = update_report.get("minibatch_reads") or []
        print(f"{reduction} seed {training_seed} batch {batch_index}: samples="
              f"{update_report['samples']} opt_steps={update_report['optimizer_steps']} "
              f"N={sum(int(read['n_rows']) for read in reads)} "
              f"M={sum(int(read['m_active_rows']) for read in reads)} "
              f"g_hat={dual_record['g_hat']:+.4f} lambda {dual_record['lambda_before']:.4f} -> "
              f"{dual_record['lambda_next']:.4f}", flush=True)

    return {"reduction": reduction, "training_seed": training_seed,
            "episodes_done": state["episodes_done"],
            "updates": state["updates"], "optimizer_steps": state["optimizer_steps"],
            "lambda": state["lambda"], "lambda_history": state["lambda_history"],
            "seconds": round(time.time() - started, 2),
            "checkpoint": str(checkpoint),
            "controller_parameters": parameter_identity(controller),
            "history": history}


def record_init_pairing(reduction: str, training_seed: int, controller: Any, optimizer: Any,
                        state: dict[str, Any]) -> None:
    """Evidence that both arms of one seed start from byte-identical parameters/optimizer/λ."""
    payload = {
        "reduction": reduction, "arm": reduction, "training_seed": training_seed,
        "encoder_arm": runner.ENCODER_ARM,
        "feature_layout": enc.ENCODERS[runner.ENCODER_ARM]["layout"],
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
    path = common.OUTPUT_DIR / f"INIT_PAIRING_{reduction}_s{training_seed}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")


def load_reference_cache() -> dict[int, dict[str, Any]]:
    cache: dict[int, dict[str, Any]] = {}
    for path in sorted(common.ROWS.glob("r32_reference_s*_a1.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        scenario = int(row["scenario_seed"])
        if scenario in cache and cache[scenario]["behaviour_sha256"] != row["behaviour_sha256"]:
            raise RuntimeError(f"scenario {scenario} has two different cached references")
        cache[scenario] = row
    return cache


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="train one (reduction arm, seed) run")
    parser.add_argument("--reduction", required=True, choices=list(common.ARM_ORDER))
    parser.add_argument("--seed", type=int, required=True, choices=list(common.TRAIN_SEEDS))
    parser.add_argument("--episodes", type=int, default=common.EPISODES_PER_SEED)
    args = parser.parse_args(argv)
    freeze = freeze_record(args.reduction)
    cache = load_reference_cache()
    needed = set(common.scenario_order()[:args.episodes])
    missing = sorted(scenario for scenario in needed if scenario not in cache)
    if missing:
        raise SystemExit(f"reference cache is missing scenarios {missing[:8]}; run the "
                         "reference block first (it is part of the 1040 normal launches)")
    summary = train_arm_seed(args.reduction, args.seed, episodes=args.episodes, freeze=freeze,
                             reference_cache=cache)
    print(json.dumps({key: summary[key] for key in
                      ("reduction", "training_seed", "episodes_done", "updates",
                       "optimizer_steps", "lambda", "seconds")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
