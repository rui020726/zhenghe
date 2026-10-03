"""ROUND42 independent first-batch gate and paired development analysis."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

RUN_DIR = Path(__file__).resolve().parent
SEMANTICS = "one-shot-send-skip/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float, np.number)):
        return math.isfinite(float(value))
    return True


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _stream_by_tick(row: dict[str, Any]) -> tuple[dict[int, float], dict[int, float]]:
    task = {int(step): float(value) for step, value in row["task_stream"].items()}
    cost = {int(step): float(value["airtime_bytes"])
            for step, value in row["cost_stream"].items()}
    return task, cost


def _validate_reward_timing(row: dict[str, Any]) -> dict[str, Any]:
    task_stream, cost_stream = _stream_by_tick(row)
    events = row.get("task_reward_events", [])
    by_tick: Counter[int] = Counter()
    credited_ids: list[str] = []
    for event in events:
        if event.get("credited"):
            task_id = str(event["task_id"])
            _assert(bool(event.get("accepted")) and bool(event.get("on_time"))
                    and not bool(event.get("event_expired"))
                    and event.get("snapshot_expired") is False
                    and float(event["event_time"]) <= float(event["deadline"]),
                    f"invalid credited completion in {row['label']}: {event}")
            credited_ids.append(task_id)
            by_tick[int(event["relay_tick"])] += 1
    _assert(len(credited_ids) == len(set(credited_ids)),
            f"duplicate credited target id in {row['label']}")
    expected = {tick: float(count) for tick, count in by_tick.items()}
    _assert(task_stream == expected,
            f"task event ticks do not reproduce task_stream in {row['label']}: "
            f"events={expected}, stream={task_stream}")
    _assert(sum(task_stream.values()) == int(row["on_time"]),
            f"task stream does not close to on_time in {row['label']}")
    _assert(sum(cost_stream.values()) == float(row["spent_bytes"])
            == float(row["airtime_tap_total"]),
            f"cost stream does not close in {row['label']}")
    return {
        "credited_on_time_ids": sorted(credited_ids),
        "credited_count": len(credited_ids),
        "reward_ticks": {str(tick): count for tick, count in sorted(by_tick.items())},
        "task_stream_closed": True,
        "cost_stream_closed": True,
    }


def _independent_bundle(stream: dict[int, float], decision_steps: list[int],
                        horizon: int) -> tuple[dict[int, float], float]:
    """Small independent transcription of the registered tick-to-decision mapping."""
    result = {step: 0.0 for step in decision_steps}
    prefix = 0.0
    for tick in range(horizon + 1):
        previous = -1
        for index, decision_step in enumerate(decision_steps):
            if decision_step <= tick:
                previous = index
            else:
                break
        reward = float(stream.get(tick, 0.0))
        if previous < 0:
            prefix += reward
        else:
            result[decision_steps[previous]] += reward
    return result, prefix


def _independent_gae(rewards: list[float], values: list[float], *, gamma: float,
                     gae_lambda: float) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        terminal = index == len(rewards) - 1
        bootstrap = 0.0 if terminal else values[index + 1]
        delta = rewards[index] + gamma * bootstrap - values[index]
        running = delta if terminal else delta + gamma * gae_lambda * running
        advantages[index] = running
    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return advantages, returns


def _independent_episode_targets(row: dict[str, Any], tapes: list[dict[str, Any]], *,
                                 lambda_at_rollout: float, b0: float,
                                 gamma: float, gae_lambda: float
                                 ) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    tapes = sorted(tapes, key=lambda tape: int(tape["step"]))
    decisions = [int(tape["step"]) for tape in tapes]
    task_stream, cost_stream = _stream_by_tick(row)
    task_rewards, task_prefix = _independent_bundle(task_stream, decisions, int(row["steps"]))
    cost_rewards, cost_prefix = _independent_bundle(cost_stream, decisions, int(row["steps"]))
    task_values = [float(tape["value"]) for tape in tapes]
    cost_values = [float(tape["cost_value"]) for tape in tapes]
    task_adv, task_return = _independent_gae(
        [task_rewards[step] for step in decisions], task_values,
        gamma=gamma, gae_lambda=gae_lambda)
    cost_adv, cost_return = _independent_gae(
        [cost_rewards[step] / b0 for step in decisions], cost_values,
        gamma=gamma, gae_lambda=gae_lambda)
    expected = {}
    for index, step in enumerate(decisions):
        expected[step] = {
            "advantage": lambda_at_rollout * task_adv[index] - cost_adv[index],
            "return_task": task_return[index],
            "return_cost": cost_return[index],
        }
    return expected, {
        "prefix_task_reward": task_prefix,
        "prefix_cost_bytes": cost_prefix,
        "credited_task_reward": sum(task_rewards.values()) + task_prefix,
        "measured_cost_bytes": sum(cost_rewards.values()) + cost_prefix,
    }


def _close(left: float, right: float, *, tolerance: float = 1e-6) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def first_batch_gate(seed: int, *, runtime: dict[str, Any],
                     frozen: dict[str, Any]) -> dict[str, Any]:
    """Audit the real four-episode rollout and real first PPO update before continuation."""
    import r42_run
    optimizer_calls_before_gate = int(r42_run._OPTIMIZER_STEP_CALLS)
    checkpoint_writes_before_gate = int(runtime["r42_versions"].CHECKPOINT_WRITE_CALLS)
    checkpoints = RUN_DIR / "checkpoints"
    records = sorted(checkpoints.glob(f"r40_batch_s{int(seed)}_b0_attempt*.json"))
    _assert(len(records) == 1, f"seed {seed}: expected one committed batch-0 record, got {records}")
    record_path = records[0]
    record = _read(record_path)
    _assert(record.get("training_seed") == int(seed) and record.get("batch") == 0,
            f"seed {seed}: wrong first batch identity")
    _assert(record.get("training_version") == frozen["round42"]["training_version"],
            f"seed {seed}: first batch has wrong training version")
    _assert(record.get("comm_action_eligibility") == "first-report-only/1",
            f"seed {seed}: first batch eligibility version mismatch")
    commit_path = checkpoints / f"r40_batch_commit_s{seed}_b0.json"
    _assert(commit_path.is_file(), f"seed {seed}: first batch commit receipt is missing")
    commit = _read(commit_path)
    _assert(commit.get("batch_record") == record_path.name
            and commit.get("batch_record_sha256") == _sha256(record_path),
            f"seed {seed}: first batch record does not match commit receipt")

    trainer = runtime["trainer"]
    round42_freeze = runtime["round42_freeze"]
    common = runtime["common"]
    pre_path = checkpoints / str(record["preupdate_checkpoint"])
    post_path = checkpoints / str(record["postupdate_checkpoint"])
    _assert(pre_path.is_file() and post_path.is_file(),
            f"seed {seed}: pre/post update checkpoint missing")
    _assert(_sha256(pre_path) == record["preupdate_checkpoint_sha256"],
            f"seed {seed}: preupdate checkpoint hash mismatch")
    _assert(_sha256(post_path) == commit["postupdate_checkpoint_sha256"],
            f"seed {seed}: postupdate checkpoint hash mismatch")
    pre_controller, pre_state, pre_identity = trainer.load_checkpoint_read_only(
        pre_path, freeze=round42_freeze)
    post_controller, post_state, post_identity = trainer.load_checkpoint_read_only(
        post_path, freeze=round42_freeze)
    expected_initial = frozen["round42"]["new_initial_models"][str(seed)]["parameter_sha256"]
    before_parameter = pre_controller.parameter_summary()["parameter_sha256"]
    after_parameter = post_controller.parameter_summary()["parameter_sha256"]
    _assert(before_parameter == expected_initial == record["controller_parameters_before"][
        "parameter_sha256"], f"seed {seed}: first batch did not start at registered ep0")
    _assert(after_parameter == record["controller_parameters_after"]["parameter_sha256"]
            and after_parameter != before_parameter,
            f"seed {seed}: actual first PPO batch did not match its parameter chain")
    _assert(int(pre_state.get("episodes_done", -1)) == 0
            and int(pre_state.get("updates", -1)) == 0
            and int(post_state.get("episodes_done", -1)) == 4
            and int(post_state.get("updates", -1)) == 1,
            f"seed {seed}: first batch checkpoint cursors are inconsistent")
    _assert(pre_identity.get("optimizer_restored") is False
            and pre_identity.get("rng_restored") is False
            and post_identity.get("optimizer_restored") is False
            and post_identity.get("rng_restored") is False,
            f"seed {seed}: gate checkpoint reads restored optimizer/RNG")

    payload = torch.load(post_path, map_location="cpu", weights_only=False)
    optimizer = torch.optim.Adam(post_controller.parameters, lr=float(common.LEARNING_RATE))
    optimizer.load_state_dict(payload["optimizer"])
    expected_parameters = list(post_controller.parameters)
    optimizer_parameters = [parameter for group in optimizer.param_groups
                            for parameter in group["params"]]
    _assert(len(optimizer_parameters) == len(expected_parameters)
            and [id(value) for value in optimizer_parameters]
            == [id(value) for value in expected_parameters],
            f"seed {seed}: optimizer parameter identities do not equal communication controller")
    module_parameters = []
    for module in (post_controller.inner.trunk, post_controller.inner.action_head,
                   post_controller.inner.value_head, post_controller.cost_head):
        module_parameters.extend(list(module.parameters()))
    _assert([id(value) for value in module_parameters]
            == [id(value) for value in expected_parameters],
            f"seed {seed}: optimizer scope is not exactly trunk/action/task/cost")
    _assert(set(payload["training_state"].keys()) >= {
        "training_seed", "episodes_done", "updates", "optimizer_steps", "lambda",
        "lambda_history", "training_version", "reward_timing_version",
        "comm_action_eligibility", "comm_action_semantics",
    }, f"seed {seed}: checkpoint training state is incomplete")
    _assert(payload.get("training_version") == frozen["round42"]["training_version"]
            and payload.get("comm_action_semantics") == SEMANTICS
            and payload.get("freeze") == round42_freeze,
            f"seed {seed}: checkpoint identity does not match the frozen ROUND42 version")

    reference_rows = {}
    for path in common.ROWS.glob("v2_reference_s*_a*.json"):
        row = _read(path)
        reference_rows[int(row["scenario_seed"])] = float(row["on_time"])
    _assert(len(reference_rows) == 64,
            f"seed {seed}: expected all 64 immediate reference rows before training")

    episodes: list[dict[str, Any]] = []
    tape_count = 0
    free_choices_total = 0
    skipped_total = 0
    send_choices_total = 0
    masked_total = 0
    closed_total = 0
    task_event_count = 0
    max_log_gap = 0.0
    expected_samples_by_key: dict[tuple[int, int], dict[str, float]] = {}
    independent_prefix_task = 0.0
    independent_prefix_cost = 0.0
    max_entropy_gap = 0.0
    from onpolicy import action_entropy
    for artifact in record["episode_artifacts"]:
        row_path = Path(artifact["row_path"])
        tape_path = Path(artifact["tape_path"])
        _assert(row_path.is_file() and tape_path.is_file(),
                f"seed {seed}: first batch artifact is missing")
        _assert(_sha256(row_path) == artifact["row_sha256"]
                and _sha256(tape_path) == artifact["tape_sha256"],
                f"seed {seed}: first batch row/tape hash mismatch")
        row = _read(row_path)
        _assert(row.get("training_seed") == seed
                and row.get("episode_index") in record["episodes"]
                and row.get("training_version") == frozen["round42"]["training_version"]
                and row.get("comm_action_semantics") == SEMANTICS,
                f"seed {seed}: wrong rollout row identity: {row_path}")
        _assert(int(row["scenario_seed"]) in reference_rows,
                f"seed {seed}: batch scenario lacks paired immediate reference")
        reward_check = _validate_reward_timing(row)
        task_event_count += reward_check["credited_count"]
        tapes = _read_jsonl(tape_path)
        _assert(tapes and len(tapes) == int(row["collector_summary"]["decision_steps"]),
                f"seed {seed}: tape row count differs from collector summary")
        per_episode_free: Counter[str] = Counter()
        for tape in tapes:
            _assert(tape.get("comm_action_semantics") == SEMANTICS
                    and tape.get("schema") == "round42-one-shot-send-skip-tape/1",
                    f"seed {seed}: wrong one-shot tape identity")
            targets = [str(target) for target in tape["targets"]]
            _assert(len(targets) == len(set(targets)),
                    f"seed {seed}: duplicated candidate id within tape row")
            categories = [set(tape["forced"]), set(tape["masked"]),
                          set(tape["fixed_hold_one_shot_closed"]), set(tape["actionable"])]
            _assert(set(targets) == set().union(*categories)
                    and all(not (categories[i] & categories[j])
                            for i in range(4) for j in range(i + 1, 4)),
                    f"seed {seed}: one-shot candidate classes do not close/disjoint")
            _assert(not tape["forced"], f"seed {seed}: unexpected forced SEND in one-shot path")
            _assert(set(tape["free_one_shot_choices"]) == set(tape["actionable"])
                    and set(tape["free_first_reports"]) == set(tape["actionable"]),
                    f"seed {seed}: free one-shot action set differs from actionable mask")
            for target in tape["actionable"]:
                candidate = tape["candidate_audit"][target]
                _assert(candidate.get("sequentially_affordable_before_action") is True,
                        f"seed {seed}: a free action was not sequentially affordable")
                _assert(candidate.get("one_shot_state_after") in (
                    "send_now_chosen", "skip_episode_chosen"),
                    f"seed {seed}: free choice failed to latch")
                per_episode_free[str(target)] += 1
                action = int(tape["actions"][target])
                _assert(action in (0, 1), f"seed {seed}: invalid one-shot action")
                _assert(bool(tape["requested"][target]) == (action == 1),
                        f"seed {seed}: request flag disagrees with action")
                if action == 0:
                    skipped_total += 1
                    _assert(target in tape["free"] and candidate[
                        "one_shot_state_after"] == "skip_episode_chosen",
                        f"seed {seed}: first SKIP is missing from policy likelihood set")
                else:
                    send_choices_total += 1
            for target in tape["masked"]:
                _assert(tape["actions"][target] == 0
                        and target not in tape["actionable"],
                        f"seed {seed}: budget/capacity mask leaked into action set")
                masked_total += 1
            for target in tape["fixed_hold_one_shot_closed"]:
                _assert(tape["actions"][target] == 0 and not tape["requested"][target]
                        and target not in tape["actionable"],
                        f"seed {seed}: closed target leaked into actor likelihood or SEND set")
                closed_total += 1
            if any(target in tape["actionable"] for target in tape["masked"]):
                raise RuntimeError("masked target entered the one-shot free action set")

            features = torch.as_tensor(np.asarray(tape["features"], dtype=np.float32))
            with torch.no_grad():
                logits, _value = pre_controller.forward(features)
                log_probs = torch.log_softmax(logits, dim=-1)
                independent_log_prob = sum(
                    float(log_probs[targets.index(str(target)), int(tape["actions"][target])])
                    for target in tape["actionable"]
                )
                probabilities = torch.softmax(logits, dim=-1)
                independent_entropy = sum(
                    float(-(probabilities[targets.index(str(target))]
                            * log_probs[targets.index(str(target))]).sum())
                    for target in tape["actionable"]
                )
                free_mask = torch.tensor(
                    [1.0 if target in set(tape["actionable"]) else 0.0
                     for target in targets], dtype=torch.float32)
                actual_entropy_helper = float(action_entropy(logits, free_mask))
            gap = abs(independent_log_prob - float(tape["log_prob"]))
            entropy_gap = abs(independent_entropy - actual_entropy_helper)
            max_log_gap = max(max_log_gap, gap)
            max_entropy_gap = max(max_entropy_gap, entropy_gap)
            _assert(gap <= 1e-6, f"seed {seed}: logged likelihood mismatch at tick {tape['step']}")
            _assert(_finite(independent_entropy) and entropy_gap <= 1e-6,
                    f"seed {seed}: entropy helper does not match the independent free-set sum")
            free_choices_total += len(tape["actionable"])
            tape_count += 1
        _assert(all(count <= 1 for count in per_episode_free.values()),
                f"seed {seed}: target received more than one free choice in one episode")
        _assert(row["collector_summary"]["skip_choices_in_free_set"] is True
                and row["collector_summary"]["closed_targets_excluded_from_likelihood"] is True,
                f"seed {seed}: collector summary reports a one-shot action-set violation")
        emissions_by_target = Counter(str(entry[1]) for entry in row["emission_sequence"])
        _assert(all(count <= 1 for count in emissions_by_target.values())
                and all(bool(entry[5]) for entry in row["emission_sequence"]),
                f"seed {seed}: a target refreshed or emitted more than once")
        _assert(int(row["relay_counters"].get("budget_refused_sends", 0)) == 0
                and int(row["relay_counters"].get("capacity_refused_sends", 0)) == 0,
                f"seed {seed}: one-shot SEND was refused by the real relay")
        episode = {
            "collector": SimpleNamespace(records=tapes),
            "steps": int(row["steps"]),
            "cost_stream": row["cost_stream"],
            "task_stream": row["task_stream"],
            "lambda": float(row["lambda_at_rollout"]),
            "episode_index": int(row["episode_index"]),
            "scenario_seed": int(row["scenario_seed"]),
            "on_time": int(row["on_time"]),
        }
        episodes.append(episode)
        independent, prefixes = _independent_episode_targets(
            row, tapes, lambda_at_rollout=float(row["lambda_at_rollout"]),
            b0=float(common.B0_BYTES), gamma=float(common.GAMMA),
            gae_lambda=float(common.GAE_LAMBDA))
        independent_prefix_task += prefixes["prefix_task_reward"]
        independent_prefix_cost += prefixes["prefix_cost_bytes"]
        for step, targets in independent.items():
            expected_samples_by_key[(int(row["episode_index"]), int(step))] = targets

    _assert(len(episodes) == 4 and sorted(ep["episode_index"] for ep in episodes) == [0, 1, 2, 3],
            f"seed {seed}: first update does not contain episodes 0-3 exactly once")
    _assert(task_event_count == sum(int(ep["on_time"]) for ep in episodes),
            f"seed {seed}: credited true completion-event total differs from training rows")
    _assert(send_choices_total + skipped_total == free_choices_total,
            f"seed {seed}: SEND/SKIP counts do not cover every free choice")

    actual_samples = trainer.batch_samples(episodes)
    actual_by_key = {(int(sample["episode_index"]), int(sample["step"])): sample
                     for sample in actual_samples}
    _assert(set(actual_by_key) == set(expected_samples_by_key),
            f"seed {seed}: actual batch_samples row keys differ from independent GAE rows")
    max_gae_gap = 0.0
    for key, expected in expected_samples_by_key.items():
        actual = actual_by_key[key]
        for field, expected_value in expected.items():
            gap = abs(float(actual[field]) - float(expected_value))
            max_gae_gap = max(max_gae_gap, gap)
            _assert(gap <= 1e-6,
                    f"seed {seed}: batch_samples {field} differs from independent GAE at {key}: "
                    f"{actual[field]} vs {expected_value}")
    reported_alignment = record["update"]["likelihood_alignment"]
    _assert(reported_alignment.get("passed") is True
            and float(reported_alignment.get("max_abs_gap", float("inf"))) <= 1e-6
            and max_log_gap <= 1e-6,
            f"seed {seed}: actual preupdate likelihood-alignment gate failed")

    reference_values = {int(pair["scenario"]): float(pair["y_reference"])
                        for pair in record["constraint"]["pairs"]}
    _assert(all(_close(reference_values[int(ep["scenario_seed"])],
                       reference_rows[int(ep["scenario_seed"])]) for ep in episodes),
            f"seed {seed}: g_hat reference pair does not match the actual immediate row")
    policy_values = [float(pair["y_policy"]) for pair in record["constraint"]["pairs"]]
    reference_batch = [reference_values[int(ep["scenario_seed"])] for ep in episodes]
    policy_batch = [float(ep["on_time"]) for ep in episodes]
    independent_g = sum(ref - policy for ref, policy in zip(reference_batch, policy_batch)) / 4.0
    independent_g -= float(common.DELTA)
    dual = record["dual"]
    expected_lambda = max(0.0, float(dual["lambda_before"])
                          + float(common.ETA_LAMBDA) * independent_g)
    _assert(_close(independent_g, float(record["constraint"]["g_hat"]))
            and _close(independent_g, float(dual["g_hat"]))
            and _close(expected_lambda, float(dual["lambda_next"]))
            and _close(float(dual["lambda_before"]), 1.0),
            f"seed {seed}: paired g_hat or dual update direction/value mismatch")
    _assert(len(policy_values) == 4 and len(reference_batch) == 4,
            f"seed {seed}: g_hat is not paired on the same four scenes")
    _assert(policy_values == policy_batch,
            f"seed {seed}: constraint report policy outcomes differ from the four rollout rows")

    update = record["update"]
    _assert(_finite(update), f"seed {seed}: non-finite PPO metrics/gradients")
    gradient_means = update.get("module_gradient_l1_mean", {})
    gradient_mins = update.get("module_gradient_l1_min", {})
    expected_modules = {"trunk", "action_head", "task_value_head", "cost_head"}
    _assert(set(gradient_means) == expected_modules
            and set(gradient_mins) == expected_modules,
            f"seed {seed}: first PPO update did not inspect both critics and shared actor modules")
    _assert(int(update["optimizer_steps"]) == int(update["minibatches"])
            and int(post_state["optimizer_steps"]) == int(update["optimizer_steps"]),
            f"seed {seed}: optimizer step count differs from actual batch/checkpoint")
    committed_steps_so_far = sum(
        int(_read(path).get("update", {}).get("optimizer_steps", 0))
        for path in checkpoints.glob("r40_batch_s*_b*_attempt*.json")
    )
    _assert(committed_steps_so_far == int(r42_run._OPTIMIZER_STEP_CALLS),
            f"seed {seed}: live Adam.step hook differs from committed PPO batch records")
    _assert(int(r42_run._OPTIMIZER_STEP_CALLS) == optimizer_calls_before_gate
            and int(runtime["r42_versions"].CHECKPOINT_WRITE_CALLS)
            == checkpoint_writes_before_gate,
            f"seed {seed}: read-only first-batch gate changed optimizer/checkpoint write counts")
    _assert(set(record["frozen_checkpoints"]) == {
        Path(item["path"]).relative_to(Path(frozen["execution_identity"][
            "canonical_read_only_input_root"])).as_posix()
        for item in frozen["round42"]["frozen_search_allocation_checkpoints"].values()
    }, f"seed {seed}: frozen PILOC/CTDE checkpoint path set changed")
    for item in frozen["round42"]["frozen_search_allocation_checkpoints"].values():
        relative = Path(item["path"]).relative_to(Path(frozen["execution_identity"][
            "canonical_read_only_input_root"])).as_posix()
        _assert(record["frozen_checkpoints"].get(relative) == item["sha256"],
                f"seed {seed}: frozen search/allocation checkpoint hash changed: {relative}")

    return {
        "kind": "round42-first-four-real-batch-gate/1",
        "status": "passed",
        "training_seed": int(seed),
        "episodes": [0, 1, 2, 3],
        "batch": 0,
        "batch_record": {"path": str(record_path), "sha256": _sha256(record_path)},
        "preupdate_parameter_sha256": before_parameter,
        "postupdate_parameter_sha256": after_parameter,
        "preupdate_checkpoint": {"path": str(pre_path), "sha256": _sha256(pre_path),
                                 "read_only": pre_identity},
        "postupdate_checkpoint": {"path": str(post_path), "sha256": _sha256(post_path),
                                  "read_only": post_identity,
                                  "episodes_done": int(post_state["episodes_done"]),
                                  "updates": int(post_state["updates"]),
                                  "optimizer_steps": int(post_state["optimizer_steps"])},
        "candidate_and_action_audit": {
            "decision_tape_rows": tape_count,
            "free_choices": free_choices_total,
            "send_now_choices": send_choices_total,
            "skip_episode_choices": skipped_total,
            "budget_or_capacity_masked_candidates": masked_total,
            "closed_fixed_hold_candidates": closed_total,
            "all_skip_choices_in_free_likelihood_set": True,
            "closed_and_masked_excluded": True,
            "one_free_choice_per_target_per_episode": True,
        },
        "independent_likelihood": {
            "max_abs_logged_vs_recomputed": max_log_gap,
            "max_abs_entropy_helper_vs_independent_free_set_sum": max_entropy_gap,
            "trainer_preupdate_alignment": reported_alignment,
            "tolerance": 1e-6,
        },
        "independent_reward_gae": {
            "max_abs_sample_target_gap": max_gae_gap,
            "independent_prefix_task_reward": independent_prefix_task,
            "independent_prefix_cost_bytes": independent_prefix_cost,
            "actual_batch_samples": len(actual_samples),
            "task_event_tick_and_cost_closure": True,
        },
        "paired_constraint": {
            "same_scene_reference_values": reference_batch,
            "policy_on_time_values": policy_batch,
            "independent_g_hat": independent_g,
            "recorded_g_hat": float(record["constraint"]["g_hat"]),
            "lambda_before": float(dual["lambda_before"]),
            "lambda_next": float(dual["lambda_next"]),
            "direction": dual["direction"],
        },
        "optimizer_and_gradient_scope": {
            "optimizer_parameter_count": len(optimizer_parameters),
            "communication_modules": sorted(expected_modules),
            "task_critic_included": True,
            "cost_critic_included": True,
            "finite_metrics_gradients_and_critic_targets": True,
            "module_gradient_l1_mean": gradient_means,
            "module_gradient_l1_min": gradient_mins,
            "gradient_norm_mean": update["gradient_norm_mean"],
            "optimizer_steps": int(update["optimizer_steps"]),
            "extra_optimizer_steps_or_checkpoint_writes_during_gate": 0,
        },
        "frozen_search_allocation_hashes_unchanged": True,
    }


def _on_time_ids(row: dict[str, Any]) -> set[str]:
    return {str(event["task_id"]) for event in row.get("task_reward_events", [])
            if event.get("credited") and event.get("on_time")}


def _row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    summary = row.get("collector_summary") or {}
    counts = summary.get("one_shot_counts", {})
    counters = row.get("relay_counters", {})
    emissions = row.get("emission_sequence", [])
    return {
        "scenario_seed": int(row["scenario_seed"]),
        "on_time": int(row["on_time"]),
        "on_time_task_ids": sorted(_on_time_ids(row)),
        "spent_bytes": int(row["spent_bytes"]),
        "first_report_emissions": int(counters.get("first_emissions", 0)),
        "first_report_bytes": sum(int(entry[3]) for entry in emissions
                                   if len(entry) > 5 and bool(entry[5])),
        "emissions": int(counters.get("emissions", row.get("emissions", 0))),
        "packets_delivered": int(counters.get("packets_delivered", 0)),
        "packets_lost": int(counters.get("packets_lost", 0)),
        "deliveries_applied": int(counters.get("deliveries_applied", 0)),
        "budget_refused_sends": int(counters.get("budget_refused_sends", 0)),
        "capacity_refused_sends": int(counters.get("capacity_refused_sends", 0)),
        "send_now_choices": int(counts.get("send_now_choices", 0)),
        "skip_episode_choices": int(counts.get("skip_episode_choices", 0)),
        "free_choices": int(counts.get("free_choices", 0)),
        "masked_budget_or_capacity": int(counts.get("masked_budget_or_capacity", 0)),
        "fixed_holds_after_choice": int(counts.get("fixed_holds_after_choice", 0)),
        "closed_by_prior_real_emission": int(counts.get("closed_by_prior_real_emission", 0)),
        "accepted_emission_records": [
            {"step": int(entry[0]), "target_id": str(entry[1]), "revision": int(entry[2]),
             "cost_bytes": int(entry[3]), "subscribers": int(entry[4]), "first": bool(entry[5])}
            for entry in emissions
        ],
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [_row_metrics(row) for row in rows]
    return {
        "n": len(metrics),
        "on_time_total": sum(item["on_time"] for item in metrics),
        "on_time_per_scene": {str(item["scenario_seed"]): item["on_time"] for item in metrics},
        "spent_bytes_total": sum(item["spent_bytes"] for item in metrics),
        "spent_bytes_per_scene": {str(item["scenario_seed"]): item["spent_bytes"]
                                  for item in metrics},
        "first_report_emissions_total": sum(item["first_report_emissions"] for item in metrics),
        "first_report_bytes_total": sum(item["first_report_bytes"] for item in metrics),
        "packet_delivered_total": sum(item["packets_delivered"] for item in metrics),
        "packet_lost_total": sum(item["packets_lost"] for item in metrics),
        "delivery_applied_total": sum(item["deliveries_applied"] for item in metrics),
        "budget_refusals_total": sum(item["budget_refused_sends"] for item in metrics),
        "capacity_refusals_total": sum(item["capacity_refused_sends"] for item in metrics),
        "send_now_choices_total": sum(item["send_now_choices"] for item in metrics),
        "skip_episode_choices_total": sum(item["skip_episode_choices"] for item in metrics),
        "free_choices_total": sum(item["free_choices"] for item in metrics),
        "budget_or_capacity_masks_total": sum(item["masked_budget_or_capacity"] for item in metrics),
        "fixed_holds_after_choice_total": sum(item["fixed_holds_after_choice"] for item in metrics),
        "scenes": metrics,
    }


def _paired_rows(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], *,
                 left_name: str, right_name: str) -> dict[str, Any]:
    left = {int(row["scenario_seed"]): row for row in left_rows}
    right = {int(row["scenario_seed"]): row for row in right_rows}
    _assert(set(left) == set(right), f"paired scene sets differ: {left_name} vs {right_name}")
    scene_deltas = []
    recovered: list[dict[str, Any]] = []
    lost: list[dict[str, Any]] = []
    swaps = 0
    for scene in sorted(left):
        left_ids = _on_time_ids(left[scene])
        right_ids = _on_time_ids(right[scene])
        regained_ids = sorted(right_ids - left_ids)
        lost_ids = sorted(left_ids - right_ids)
        swaps += min(len(regained_ids), len(lost_ids))
        recovered.extend({"scene": scene, "task_id": target} for target in regained_ids)
        lost.extend({"scene": scene, "task_id": target} for target in lost_ids)
        delta = int(right[scene]["on_time"]) - int(left[scene]["on_time"])
        scene_deltas.append({
            "scene": scene,
            f"{left_name}_on_time": int(left[scene]["on_time"]),
            f"{right_name}_on_time": int(right[scene]["on_time"]),
            "delta_right_minus_left": delta,
            f"{left_name}_spent_bytes": int(left[scene]["spent_bytes"]),
            f"{right_name}_spent_bytes": int(right[scene]["spent_bytes"]),
            "recovered_task_ids": regained_ids,
            "lost_task_ids": lost_ids,
            "target_swaps": min(len(regained_ids), len(lost_ids)),
        })
    deltas = [item["delta_right_minus_left"] for item in scene_deltas]
    return {
        "left": left_name,
        "right": right_name,
        "comparison": "right_minus_left_paired_by_scene",
        "scene_count": len(scene_deltas),
        "on_time_total_left": sum(int(row["on_time"]) for row in left_rows),
        "on_time_total_right": sum(int(row["on_time"]) for row in right_rows),
        "on_time_total_delta": sum(deltas),
        "on_time_mean_scene_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "improved_scenes": sum(delta > 0 for delta in deltas),
        "tied_scenes": sum(delta == 0 for delta in deltas),
        "degraded_scenes": sum(delta < 0 for delta in deltas),
        "maximum_single_scene_loss": max([0] + [-delta for delta in deltas]),
        "recovered_task_ids": recovered,
        "lost_task_ids": lost,
        "target_swap_count": swaps,
        "spent_bytes_total_left": sum(int(row["spent_bytes"]) for row in left_rows),
        "spent_bytes_total_right": sum(int(row["spent_bytes"]) for row in right_rows),
        "scenes": scene_deltas,
    }


def analyze_all(*, runtime: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    rows = [_read(path) for path in sorted(runtime["common"].ROWS.glob("*.json"))]
    rows = [row for row in rows if row.get("training_version")
            == frozen["round42"]["training_version"]]
    refs = [row for row in rows if row.get("reference_arm") == "immediate"]
    training = [row for row in rows if row.get("training_arm") == "NEW"]
    evaluations = [row for row in rows if row.get("kind") == "eval"]
    rules = [row for row in rows if row.get("kind") == "rule"]
    _assert(len(refs) == 64, f"expected 64 immediate reference rows, found {len(refs)}")
    _assert(len(training) == 384, f"expected 384 NEW training rows, found {len(training)}")
    _assert(len(evaluations) == 192, f"expected 192 OLD/NEW evaluation rows, found {len(evaluations)}")
    _assert(len(rules) == 32, f"expected 32 rule rows, found {len(rules)}")
    for row in rows:
        _validate_reward_timing(row)
        _assert(row.get("comm_action_semantics") == SEMANTICS,
                f"wrong action semantics in result row {row['label']}")

    ref_dev = {int(row["scenario_seed"]): row for row in refs
               if int(row["scenario_seed"]) in range(1_301_000, 1_301_016)}
    _assert(len(ref_dev) == 16, "immediate development reference block is incomplete")
    paired: dict[str, Any] = {}
    absolute: dict[str, Any] = {}
    group_summaries: dict[str, Any] = {"immediate_reference": _aggregate(refs),
                                       "new_training_rollouts": {}}
    for seed in (11, 22, 33):
        train_rows = [row for row in training if int(row["training_seed"]) == seed]
        _assert(len(train_rows) == 128, f"seed {seed}: expected 128 training episodes")
        group_summaries["new_training_rollouts"][str(seed)] = _aggregate(train_rows)
        for mode in ("greedy", "sampled"):
            cohort = [row for row in evaluations if int(row["training_seed"]) == seed
                      and row.get("eval_mode") == mode]
            old_rows = [row for row in cohort if row.get("model_arm") == "OLD"]
            new_rows = [row for row in cohort if row.get("model_arm") == "NEW"]
            _assert(len(old_rows) == len(new_rows) == 16,
                    f"seed {seed}/{mode}: OLD or NEW development block is incomplete")
            key = f"s{seed}_{mode}"
            paired[key] = _paired_rows(old_rows, new_rows,
                                       left_name="OLD", right_name="NEW")
            absolute[key] = {
                "OLD_vs_immediate": _paired_rows(ref_dev.values(), old_rows,
                                                 left_name="immediate", right_name="OLD"),
                "NEW_vs_immediate": _paired_rows(ref_dev.values(), new_rows,
                                                 left_name="immediate", right_name="NEW"),
            }
            group_summaries[f"{key}_OLD"] = _aggregate(old_rows)
            group_summaries[f"{key}_NEW"] = _aggregate(new_rows)
    rule_summary: dict[str, Any] = {}
    for arm in ("late_discovery_gate", "always_hold"):
        arm_rows = [row for row in rules if row.get("model_arm") == arm]
        _assert(len(arm_rows) == 16, f"rule arm {arm}: expected 16 development rows")
        rule_summary[arm] = {
            "absolute": _aggregate(arm_rows),
            "vs_immediate": _paired_rows(ref_dev.values(), arm_rows,
                                         left_name="immediate", right_name=arm),
        }
    return {
        "kind": "round42-paired-results/1",
        "training_version": frozen["round42"]["training_version"],
        "comm_action_semantics": SEMANTICS,
        "data_scope": {
            "immediate_references": len(refs),
            "new_training_episodes": len(training),
            "old_new_development_cells": len(evaluations),
            "rule_development_cells": len(rules),
            "development_scene_ids": sorted(ref_dev),
            "independent_test": False,
        },
        "primary_new_minus_old_paired_by_seed_and_mode": paired,
        "secondary_absolute_paired_to_immediate": absolute,
        "rule_absolute_performance": rule_summary,
        "group_summaries": group_summaries,
        "interpretation_limits": [
            "ROUND42 reuses its registered development block and is not an independent test.",
            "Greedy and sampled outcomes are reported separately.",
            "No pooled population, significance, noninferiority, or generalization claim is made.",
            "A one-shot target-level decision is not a general solution to communication timing.",
        ],
    }


__all__ = ["analyze_all", "first_batch_gate"]
