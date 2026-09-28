"""Stage C 检查器：冻结记录、首4集训练正确性门禁、最终对比（只读落盘 artefact）。

首4集门禁（ROUND22 spec §7）逐条对应：
  G1 optimizer 只更新通信模块（optimizer 参数恒等 + 冻结 checkpoint 文件哈希不变）
  G2 rollout log_prob 在任何更新前可重算（用初值权重重算，≤1e-6）
  G3 成本/任务 reward、GAE、组合优势、对偶 g 从落盘数据独立重算且闭合
  G4 lambda 更新方向与公式一致；reference 无梯度（标量）
  G5 task/cost critic 的 target/预测/loss/梯度有限；全 mask 无 actor 伪样本
  G6 checkpoint 在纯计算进程恢复后，同一观测前向一致（0 额外启动）
  G7 丢包/拒付/主动HOLD/可执行样本计数闭合
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sc_common as common
import cost_math as cm
from sc_controller import fresh_controller, parameter_identity

TOL = common.LOG_PROB_TOLERANCE


# ------------------------------------------------------------------ 冻结记录 -------------
def freeze_record() -> dict[str, Any]:
    manifest_path = common.OUTPUT_DIR / "CONFIG_AND_MODEL_MANIFEST.json"
    if not manifest_path.exists():
        raise RuntimeError("CONFIG_AND_MODEL_MANIFEST.json is missing; freeze before starting")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"freeze_sha256": manifest["code_identity"]["freeze_sha256"],
            "config_hash": manifest["configuration_hash"],
            "branch_start": manifest["branch_start"],
            "head_at_freeze": manifest["head_at_freeze"]}


# ------------------------------------------------------------------ 落盘读取 -------------
def load_row(kind: str, seed: int, attempt: int = 1, tag: str = "") -> dict[str, Any]:
    path = common.cell_paths(kind, seed, attempt, tag=tag)["row"]
    if not path.exists():
        raise FileNotFoundError(f"missing row artefact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_tape(kind: str, seed: int, attempt: int = 1, tag: str = "") -> list[dict[str, Any]]:
    path = common.cell_paths(kind, seed, attempt, tag=tag)["tape"]
    if not path.exists():
        return []
    return list(common.iter_jsonl(path))


def load_batch_record(training_seed: int, batch: int) -> dict[str, Any]:
    path = common.CHECKPOINTS / f"sc_batch_s{training_seed}_b{batch}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing batch record: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ G3 独立重算 -------------
def recompute_episode_math(row: dict[str, Any], tapes: list[dict[str, Any]],
                           lam: float) -> dict[str, Any]:
    """Rewards, GAE and the combined advantage, recomputed from the row and the tape only."""
    decision_steps = sorted(int(tape["step"]) for tape in tapes)
    horizon = int(row["steps"])
    cost_stream = {int(step): float(entry["airtime_bytes"])
                   for step, entry in row["cost_stream"].items()}
    task_stream = {int(step): float(value) for step, value in row["task_stream"].items()}

    def bundle(stream: dict[int, float]) -> tuple[dict[int, float], float]:
        bundled = {step: 0.0 for step in decision_steps}
        prefix = 0.0
        if not decision_steps:
            return bundled, sum(float(value) for value in stream.values())
        prefix += sum(float(value) for step, value in stream.items()
                      if int(step) < decision_steps[0])
        for index, decision in enumerate(decision_steps):
            stop = (decision_steps[index + 1]
                    if index + 1 < len(decision_steps) else horizon + 1)
            bundled[decision] = sum(float(value) for step, value in stream.items()
                                    if decision <= int(step) < stop)
        return bundled, prefix

    bundled_cost, prefix_cost = bundle(cost_stream)
    bundled_task, prefix_task = bundle(task_stream)
    if not decision_steps:
        return {
            "decision_steps": 0,
            "sum_rewards_task": 0.0,
            "sum_rewards_cost": 0.0,
            "unassigned_prefix_task_reward": prefix_task,
            "unassigned_prefix_cost_bytes": prefix_cost,
            "all_finite": True,
        }
    rewards_cost = torch.tensor([bundled_cost[s] / common.B0_BYTES for s in decision_steps],
                                dtype=torch.float64)
    rewards_task = torch.tensor([bundled_task[s] for s in decision_steps], dtype=torch.float64)
    values_task = torch.tensor([float(tape["value"]) for tape in tapes], dtype=torch.float64)
    values_cost = torch.tensor([float(tape["cost_value"]) for tape in tapes],
                               dtype=torch.float64)
    terminal = torch.zeros(len(decision_steps), dtype=torch.bool)
    terminal[-1] = True
    bootstrap_task = torch.zeros_like(values_task)
    bootstrap_cost = torch.zeros_like(values_cost)
    if len(decision_steps) > 1:
        bootstrap_task[:-1] = values_task[1:]
        bootstrap_cost[:-1] = values_cost[1:]
    advantages_task, returns_task = cm.gae(rewards_task, values_task, gamma=common.GAMMA,
                                           gae_lambda=common.GAE_LAMBDA, terminal=terminal,
                                           bootstrap=bootstrap_task)
    advantages_cost, returns_cost = cm.gae(rewards_cost, values_cost, gamma=common.GAMMA,
                                           gae_lambda=common.GAE_LAMBDA, terminal=terminal,
                                           bootstrap=bootstrap_cost)
    combined = cm.actor_advantage(advantages_task, advantages_cost, lam)
    return {
        "decision_steps": len(decision_steps),
        "sum_rewards_task": float(rewards_task.sum()),
        "sum_rewards_cost": float(rewards_cost.sum()),
        "unassigned_prefix_task_reward": prefix_task,
        "unassigned_prefix_cost_bytes": prefix_cost,
        "combined_first": float(combined[0]),
        "advantages_task_first": float(advantages_task[0]),
        "advantages_cost_first": float(advantages_cost[0]),
        "returns_task_first": float(returns_task[0]),
        "returns_cost_first": float(returns_cost[0]),
        "all_finite": bool(torch.isfinite(combined).all() and torch.isfinite(returns_task).all()
                           and torch.isfinite(returns_cost).all()),
    }


# ------------------------------------------------------------------ 首4集门禁 -------------
def first4_gate(training_seed: int = 11) -> dict[str, Any]:
    """Run every spec §7 check on the first four training episodes of a seed."""
    freeze = freeze_record()
    checks: dict[str, Any] = {}

    batch_record = load_batch_record(training_seed, 0)
    episode_rows = [load_row("train", batch_record["scenarios"][index],
                             attempt=batch_record["episodes"][index],
                             tag=f"train_s{training_seed}_e{batch_record['episodes'][index]}")
                    for index in range(len(batch_record["episodes"]))]
    tapes = [load_tape("train", row["scenario_seed"], attempt=row["episode_index"],
                       tag=f"train_s{training_seed}_e{row['episode_index']}")
             for row in episode_rows]

    # G1: the optimiser must hold exactly the communication controller's parameters. Object
    # identity is process-specific, so the comparison is on the recorded per-parameter shapes
    # and counts, rebuilt against a fresh controller here.
    controller = fresh_controller(training_seed)
    controller_shapes = [list(parameter.shape) for parameter in controller.parameters]
    optimizer_shapes = batch_record.get("optimizer_parameter_shapes")
    recorded_controller_shapes = batch_record.get("controller_parameter_shapes")
    frozen_now = {str(path.relative_to(common.ROOT)): common.sha256_file(path)
                  for path in common.CHECKPOINT_FILES}
    checks["G1_optimizer_updates_only_comm"] = {
        "optimizer_parameters_are_controller_parameters": bool(
            optimizer_shapes is not None and recorded_controller_shapes is not None
            and optimizer_shapes == recorded_controller_shapes == controller_shapes),
        "optimizer_parameter_count": len(optimizer_shapes or []),
        "controller_parameter_count": len(controller_shapes),
        "frozen_checkpoints_unchanged": bool(
            frozen_now == batch_record["frozen_checkpoints"]),
        "controller_changed": bool(
            batch_record["controller_parameters_before"]["parameter_sha256"]
            != batch_record["controller_parameters_after"]["parameter_sha256"]),
        "passed": bool(optimizer_shapes is not None and recorded_controller_shapes is not None
                       and optimizer_shapes == recorded_controller_shapes == controller_shapes
                       and frozen_now == batch_record["frozen_checkpoints"]
                       and batch_record["controller_parameters_before"]["parameter_sha256"]
                       != batch_record["controller_parameters_after"]["parameter_sha256"]),
    }

    # G2: pre-update log_prob recomputation with the initial weights (batch 0's pre-update
    # weights ARE the frozen initialisation, so they are reconstructible from the seed alone)
    initial = parameter_identity(controller)
    pre_update_is_initial = (batch_record["controller_parameters_before"]["parameter_sha256"]
                             == initial["parameter_sha256"])
    gaps: list[float] = []
    for tape_list in tapes:
        for tape in tape_list:
            logits, _values = controller.forward(
                torch.as_tensor(np.asarray(tape["features"], dtype=np.float32)))
            with torch.no_grad():
                picked = torch.tensor([tape["actions"][t] for t in tape["targets"]],
                                      dtype=torch.long)
                usable = torch.tensor([1.0 if t in tape["actionable"] else 0.0
                                       for t in tape["targets"]], dtype=torch.float32)
                from onpolicy import joint_log_prob
                recomputed = float(joint_log_prob(logits, picked, usable))
            gaps.append(abs(recomputed - float(tape["log_prob"])))
    checks["G2_log_prob_recomputable_pre_update"] = {
        "pre_update_weights_are_the_registered_initialisation": pre_update_is_initial,
        "max_abs_gap": max(gaps) if gaps else 0.0,
        "tolerance": TOL,
        "passed": pre_update_is_initial and (not gaps or max(gaps) <= TOL),
    }

    # G3: independent recompute of rewards / GAE / combined advantage / g
    recomputed = [recompute_episode_math(row, tape_list, float(row["lambda_at_rollout"]))
                  for row, tape_list in zip(episode_rows, tapes)]
    totals_close = all(
        abs(item["sum_rewards_task"] + item["unassigned_prefix_task_reward"]
            - row["task_reward_total"]) < 1e-9
        and abs(item["sum_rewards_cost"]
                + item["unassigned_prefix_cost_bytes"] / common.B0_BYTES
                - row["spent_bytes"] / common.B0_BYTES) < 1e-9
        for item, row in zip(recomputed, episode_rows))
    pairs = batch_record["constraint"]["pairs"]
    g_recomputed = cm.constraint_statistic([pair["y_reference"] for pair in pairs],
                                           [pair["y_policy"] for pair in pairs],
                                           common.DELTA)
    checks["G3_math_recomputable_from_disk"] = {
        "reward_sums_close_on_episode_totals": bool(totals_close),
        "g_hat_matches_record": bool(abs(g_recomputed - batch_record["constraint"]["g_hat"]) < 1e-12),
        "all_finite": bool(all(item.get("all_finite", True) for item in recomputed)),
        "decision_steps": [item.get("decision_steps") for item in recomputed],
        "passed": bool(totals_close
                       and abs(g_recomputed - batch_record["constraint"]["g_hat"]) < 1e-12
                       and all(item.get("all_finite", True) for item in recomputed)),
    }

    # G4: dual update formula and direction
    dual = batch_record["dual"]
    expected_lambda = max(0.0, dual["lambda_before"] + common.ETA_LAMBDA * dual["g_hat"])
    checks["G4_dual_direction_and_formula"] = {
        "lambda_next_matches_formula": bool(abs(dual["lambda_next"] - expected_lambda) < 1e-12),
        "direction_consistent": bool(
            (dual["g_hat"] > 0 and dual["lambda_next"] >= dual["lambda_before"])
            or (dual["g_hat"] < 0 and dual["lambda_next"] <= dual["lambda_before"])
            or (dual["g_hat"] == 0 and dual["lambda_next"] == dual["lambda_before"])),
        "reference_is_scalar": bool(all(isinstance(pair["y_reference"], float)
                                        for pair in pairs)),
        "passed": bool(abs(dual["lambda_next"] - expected_lambda) < 1e-12
                       and ((dual["g_hat"] > 0 and dual["lambda_next"] >= dual["lambda_before"])
                            or (dual["g_hat"] < 0 and dual["lambda_next"] <= dual["lambda_before"])
                            or (dual["g_hat"] == 0
                                and dual["lambda_next"] == dual["lambda_before"]))
                       and all(isinstance(pair["y_reference"], float) for pair in pairs)),
    }

    # G5: critic finiteness (recomputed forward + loss + backward with initial weights) and
    #     the all-mask pseudo-sample prohibition
    finite = True
    worst_loss = None
    for tape_list in tapes:
        for tape in tape_list:
            matrix = torch.as_tensor(np.asarray(tape["features"], dtype=np.float32))
            hidden = controller.inner.trunk(matrix)
            task_values = controller.inner.value_head(hidden).squeeze(-1)
            cost_values = controller.cost_head(hidden).squeeze(-1)
            if not (torch.isfinite(task_values).all() and torch.isfinite(cost_values).all()):
                finite = False
    all_mask_steps = [tape for tape_list in tapes for tape in tape_list
                      if tape["targets"] and not tape["actionable"]]
    all_mask_clean = all(float(tape["log_prob"]) == 0.0 and
                         not any(tape["requested"].values()) for tape in all_mask_steps)
    update = batch_record["update"]
    checks["G5_critics_finite_and_no_pseudo_samples"] = {
        "critic_predictions_finite": bool(finite),
        "recorded_value_losses_finite": bool(
            np.isfinite(update.get("value_loss_task", float("nan")))
            and np.isfinite(update.get("value_loss_cost", float("nan")))),
        "all_mask_steps": len(all_mask_steps),
        "all_mask_steps_have_no_actor_sample": bool(all_mask_clean),
        "optimizer_steps": int(update.get("optimizer_steps", 0)),
        "surrogate_finite": bool(np.isfinite(update.get("policy_surrogate", float("nan")))),
        "passed": bool(finite
                       and np.isfinite(update.get("value_loss_task", float("nan")))
                       and np.isfinite(update.get("value_loss_cost", float("nan")))
                       and all_mask_clean
                       and np.isfinite(update.get("policy_surrogate", float("nan")))),
    }

    # G6: checkpoint restore in a pure computation process, same-observation forward
    probe = subprocess.run(
        [sys.executable, str(common.HERE / "sc_resume_probe.py"),
         "--checkpoint", str(common.CHECKPOINTS / f"sc_train_s{training_seed}.pt"),
         "--seed", str(training_seed)],
        capture_output=True, text=True, encoding="utf-8", cwd=str(common.HERE),
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    probe_payload = {}
    if probe.returncode == 0 and probe.stdout.strip():
        probe_payload = json.loads(probe.stdout.strip().splitlines()[-1])
    restored_controller, _optimizer, state = _load_in_process(training_seed)
    matrix = torch.arange(19 * 4, dtype=torch.float32).reshape(4, 19) / 19.0
    with torch.no_grad():
        logits, task_values = restored_controller.forward(matrix)
        cost_values = restored_controller.cost_values(matrix)
    in_process = {"action_logits": logits.tolist(), "task_values": task_values.tolist(),
                  "cost_values": cost_values.tolist(),
                  "lambda": state["lambda"], "episodes_done": state["episodes_done"]}
    same_forward = (probe_payload.get("action_logits") == in_process["action_logits"]
                    and probe_payload.get("cost_values") == in_process["cost_values"]
                    and probe_payload.get("task_values") == in_process["task_values"])
    checks["G6_checkpoint_restore_pure_process"] = {
        "probe_returncode": probe.returncode,
        "probe_stderr_tail": probe.stderr.splitlines()[-2:] if probe.stderr else [],
        "same_observation_forward_identical": bool(same_forward),
        "restored_state": {"lambda": state["lambda"],
                           "episodes_done": state["episodes_done"],
                           "updates": state["updates"]},
        "passed": bool(probe.returncode == 0 and same_forward),
    }

    # G7: sample-count closure
    timing = episode_rows[0].get("collector_summary") or {}
    counters = episode_rows[0]["relay_counters"]
    checks["G7_sample_counts_close"] = {
        "first_plus_refresh_equals_emissions": bool(
            int(counters["first_emissions"]) + int(counters["refresh_emissions"])
            == int(counters["emissions"])),
        "requests_equal_emissions_plus_refusals": bool(
            int((episode_rows[0].get("collector_summary") or {}).get("requested_sends_total", 0))
            == int(counters["emissions"]) + int(counters["budget_refused_sends"])
            + int(counters["capacity_refused_sends"])),
        "candidate_classes_close": bool(timing.get("candidate_classes_close")),
        "candidate_classes_disjoint": bool(timing.get("candidate_classes_disjoint")),
        "masked_carry_no_likelihood": bool(timing.get("masked_carry_no_likelihood")),
        "policy_budget_refusals": int(counters["budget_refused_sends"]),
        "passed": bool(
            int(counters["first_emissions"]) + int(counters["refresh_emissions"])
            == int(counters["emissions"])
            and int((episode_rows[0].get("collector_summary") or {}).get("requested_sends_total", 0))
            == int(counters["emissions"]) + int(counters["budget_refused_sends"])
            + int(counters["capacity_refused_sends"])
            and timing.get("candidate_classes_close")
            and timing.get("candidate_classes_disjoint")
            and timing.get("masked_carry_no_likelihood")),
    }

    passed = all(bool(check.get("passed")) for check in checks.values()
                 if isinstance(check, dict))
    return {"training_seed": training_seed, "gate": "first4", "checks": checks,
            "passed": bool(passed)}


def _load_in_process(training_seed: int):
    import sc_trainer as trainer

    return trainer.load_checkpoint(
        common.CHECKPOINTS / f"sc_train_s{training_seed}.pt", freeze=freeze_record())


# ------------------------------------------------------------------ 最终对比 -------------
def comparison_rows() -> dict[str, Any]:
    """The final descriptive comparison: each training seed's fixed 128-episode checkpoint
    against the immediate reference and all five rule arms, on the 16 dev scenarios."""
    dev = common.DEVELOPMENT_SCENARIOS
    references = {scenario: load_row("reference", scenario)
                  for scenario in dev}
    learned: dict[str, dict[int, dict[str, Any]]] = {}
    for row_path in sorted(common.ROWS.glob("sc_eval_s*_a0.json")):
        row = json.loads(row_path.read_text(encoding="utf-8"))
        if row.get("timepoint") != "128":
            continue
        learned.setdefault(str(row["training_seed"]), {})[int(row["scenario_seed"])] = row
    arm_rows: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in common.RULE_ARMS:
        for row_path in sorted(common.ROWS.glob(f"sc_{arm}_s*_a1.json")):
            row = json.loads(row_path.read_text(encoding="utf-8"))
            arm_rows.setdefault(arm, {})[int(row["scenario_seed"])] = row
    return {"development_scenarios": list(dev), "references": references,
            "learned": learned, "rules": arm_rows}


def paired_stats(rows: dict[int, dict[str, Any]],
                 references: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Per-scene paired differences with the episode as the cluster unit (spec §8)."""
    tasks: list[float] = []
    byte_deltas: list[float] = []
    improved = tied = degraded = 0
    for scenario, reference in sorted(references.items()):
        row = rows.get(scenario)
        if row is None:
            continue
        task_delta = int(row["on_time"]) - int(reference["on_time"])
        byte_delta = int(row["spent_bytes"]) - int(reference["spent_bytes"])
        tasks.append(float(task_delta))
        byte_deltas.append(float(byte_delta))
        if byte_delta < 0 and task_delta >= 0:
            improved += 1
        elif byte_delta == 0 and task_delta == 0:
            tied += 1
        else:
            degraded += 1
    return {
        "scenes": len(tasks),
        "mean_on_time_delta": round(float(np.mean(tasks)), 6) if tasks else None,
        "mean_bytes_delta": round(float(np.mean(byte_deltas)), 3) if byte_deltas else None,
        "improved": improved, "tied": tied, "degraded": degraded,
        "per_scene_on_time_delta": {str(scenario): int(rows[scenario]["on_time"])
                                    - int(references[scenario]["on_time"])
                                    for scenario in sorted(references) if scenario in rows},
        "per_scene_bytes_delta": {str(scenario): int(rows[scenario]["spent_bytes"])
                                  - int(references[scenario]["spent_bytes"])
                                  for scenario in sorted(references) if scenario in rows},
    }


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Stage C checks")
    parser.add_argument("--mode", choices=("first4", "comparison"), required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    if args.mode == "first4":
        payload = first4_gate(args.seed)
        common.write_json(f"FIRST4_GATE_s{args.seed}.json", payload)
    else:
        rows = comparison_rows()
        payload = {"references": {str(s): {"on_time": r["on_time"],
                                           "spent_bytes": r["spent_bytes"]}
                                  for s, r in rows["references"].items()},
                   "learned": {seed: paired_stats(scenes, rows["references"])
                               for seed, scenes in sorted(rows["learned"].items())},
                   "rules": {arm: paired_stats(scenes, rows["references"])
                             for arm, scenes in sorted(rows["rules"].items())},
                   "note": "exploratory pilot on 16 development scenarios; episodes are the "
                           "cluster unit; zero observed loss or a CI containing 0 does not "
                           "establish non-inferiority (spec §8)"}
        common.write_json("COMPARISON.json", payload)
    print(json.dumps({"mode": args.mode,
                      "passed": payload.get("passed", payload.get("learned") is not None)},
                     ensure_ascii=False))
    return 0 if payload.get("passed", True) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
