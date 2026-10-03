"""ROUND40 sequential execution controller. The first environment requires a clean freeze commit."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

import r40_common as r40

COMM_ACTION_ELIGIBILITY = "first-report-only/1"
TRAINING_VERSION = (
    "constrained-cost-v2/first-report-only/1/"
    "published-completion-event-at-relay-tick/2"
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"ROUND40 immutable gate result conflicts: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _record_preconstruction_failure(phase: str, error: BaseException,
                                   launches_before: int, launches_after: int) -> None:
    directory = r40.ROUND_DIR / "preconstruction_failures"
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": "round40-preconstruction-worker-failure/1",
        "phase": phase,
        "recorded_at_unix": time.time(),
        "exception_type": type(error).__name__,
        "exception": str(error),
        "traceback": traceback.format_exc(),
        "launch_slots_before_invocation": int(launches_before),
        "launch_slots_after_invocation": int(launches_after),
        "new_launch_slots_consumed": 0,
        "environment_constructed_by_failure": False,
    }
    _write_once(directory / f"failure_{time.time_ns()}_{os.getpid()}.json", record)


def _load_context(input_root: str | Path):
    root = Path(input_root).resolve()
    common = r40.bootstrap(root)
    runner = importlib.import_module("v2_run")
    trainer = importlib.import_module("v2_trainer")
    v2_controller = importlib.import_module("v2_controller")
    freeze_path = r40.ROUND_DIR / "ROUND40_FROZEN_CONFIG.json"
    preflight_path = r40.ROUND_DIR / "ROUND40_PREFLIGHT.json"
    if not freeze_path.is_file() or not preflight_path.is_file():
        raise RuntimeError("ROUND40 freeze/preflight artifacts are missing")
    freeze_doc = _json(freeze_path)
    freeze = freeze_doc["round40"]["freeze_identity"]
    preflight = _json(preflight_path)
    if preflight.get("status") != "passed" or preflight.get("environment_constructed") is not False:
        raise RuntimeError("ROUND40 pure-compute preflight did not pass cleanly")
    if (_git(root, "rev-parse", "HEAD") != freeze["base_head"]
            or _git(root, "branch", "--show-current") != "exp/comm-refresh-held-mechanism"):
        raise RuntimeError("canonical input checkout changed after the freeze")
    current_head = _git(r40.ROUND_DIR.parents[1], "rev-parse", "HEAD")
    branch = _git(r40.ROUND_DIR.parents[1], "branch", "--show-current")
    if branch != "exp/comm-first-report-training-ablation":
        raise RuntimeError(f"unexpected execution branch {branch!r}")
    if current_head == preflight["round40_basis_commit"]:
        raise RuntimeError("freeze/preflight have not been committed; refusing first environment")
    if _git(r40.ROUND_DIR.parents[1], "status", "--porcelain"):
        raise RuntimeError("worktree is dirty after freeze; refusing an environment launch")
    worktree = r40.ROUND_DIR.parents[1]
    required_committed = [
        "runs/comm_first_report_training_ablation_20261003/EXECUTION_BASIS.md",
        "runs/comm_first_report_training_ablation_20261003/ROUND40_FROZEN_CONFIG.json",
        "runs/comm_first_report_training_ablation_20261003/ROUND40_PREFLIGHT.json",
        "runs/comm_first_report_training_ablation_20261003/SCENARIO_ORDER.json",
        "runs/comm_first_report_training_ablation_20261003/TRAINING_VERSION_MANIFEST.json",
        *[f"runs/comm_first_report_training_ablation_20261003/{name}"
          for name in freeze["round40_code_sha256"]],
    ]
    for relative in required_committed:
        subprocess.run(["git", "-C", str(worktree), "cat-file", "-e",
                        f"HEAD:{relative}"], check=True, stdout=subprocess.DEVNULL)
    for relative, expected in freeze["round40_code_sha256"].items():
        actual = _sha256(r40.ROUND_DIR / relative)
        if actual != expected:
            raise RuntimeError(f"frozen ROUND40 code changed: {relative}")
    from r40_versions import install

    install(trainer, runner, freeze=freeze)
    runner.common = common
    return root, common, runner, trainer, v2_controller, freeze


def _latest_claim(logical_job: str) -> dict[str, Any] | None:
    directory = r40.ROUND_DIR / "launch_claims" / logical_job
    if not directory.exists():
        # Keys here contain no characters changed by the claim slugger.
        return None
    paths = sorted(directory.glob("attempt_*.json"))
    return _json(paths[-1]) if paths else None


def _reference_path(common: Any, scenario: int) -> Path | None:
    rows = sorted(common.ROWS.glob(f"v2_reference_s{scenario}_a*.json"))
    valid = [path for path in rows
             if _json(path).get("reference_arm") == "immediate"]
    return valid[-1] if valid else None


def _reference_rows(common: Any) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in sorted(common.ROWS.glob("v2_reference_s*_a*.json")):
        row = _json(path)
        if row.get("reference_arm") != "immediate":
            continue
        scenario = int(row["scenario_seed"])
        if scenario in result and result[scenario]["behaviour_sha256"] != row["behaviour_sha256"]:
            raise RuntimeError(f"duplicate immediate rows differ for scenario {scenario}")
        result[scenario] = row
    return result


def _validate_row_closure(row: dict[str, Any]) -> dict[str, Any]:
    cost_total = sum(int(value.get("airtime_bytes", 0))
                     for value in row.get("cost_stream", {}).values())
    reward_total = sum(float(value) for value in row.get("task_stream", {}).values())
    if cost_total != int(row["spent_bytes"]) or cost_total != int(row["airtime_tap_total"]):
        raise RuntimeError(f"row cost stream does not close: {row.get('label')}")
    if reward_total != float(row["task_reward_total"]) or reward_total != float(row["on_time"]):
        raise RuntimeError(f"row task reward stream does not close: {row.get('label')}")
    credited_by_tick: dict[str, int] = {}
    for event in row.get("task_reward_events", []):
        tick = str(int(event["relay_tick"]))
        if event.get("credited"):
            if not event.get("on_time") or float(event["event_time"]) > float(event["deadline"]):
                raise RuntimeError(f"credited event is not an on-time completion: {event}")
            credited_by_tick[tick] = credited_by_tick.get(tick, 0) + 1
    task_stream = {str(step): float(value) for step, value in row.get("task_stream", {}).items()
                   if float(value) != 0.0}
    if task_stream != {step: float(value) for step, value in credited_by_tick.items()}:
        raise RuntimeError(f"task event identities and relay-tick rewards differ: {row.get('label')}")
    return {"spent_bytes": cost_total, "task_reward_total": int(reward_total),
            "credited_task_events": sum(bool(event.get("credited"))
                                         for event in row.get("task_reward_events", []))}


def run_references(common: Any, runner: Any) -> dict[str, Any]:
    from r40_run import run_reference

    expected = tuple(sorted((*r40.TRAIN_SCENARIOS, *r40.DEV_SCENARIOS)))
    for scenario in expected:
        logical = f"reference_immediate_s{scenario}"
        claim = _latest_claim(logical)
        row_path = _reference_path(common, scenario)
        if claim and claim.get("status") == "complete":
            if row_path is None:
                raise RuntimeError(f"completed reference claim has no row: {scenario}")
            continue
        if row_path is not None and claim is None:
            raise RuntimeError(f"reference artefact has no preconstruction claim: {scenario}")
        print(f"ROUND40 immediate reference {scenario}", flush=True)
        run_reference(common, runner, scenario)
    rows = _reference_rows(common)
    if set(rows) != set(expected) or len(rows) != 64:
        raise RuntimeError(f"immediate reference matrix is incomplete: {len(rows)}/64")
    closure = {str(scenario): _validate_row_closure(rows[scenario]) for scenario in expected}
    result = {
        "kind": "round40-immediate-reference-gate/1",
        "passed": True,
        "matrix_count": len(rows),
        "training_scenarios": list(r40.TRAIN_SCENARIOS),
        "development_scenarios": list(r40.DEV_SCENARIOS),
        "training_reference_total_on_time": sum(rows[s]["on_time"] for s in r40.TRAIN_SCENARIOS),
        "development_reference_total_on_time": sum(rows[s]["on_time"] for s in r40.DEV_SCENARIOS),
        "rows": {str(s): {"on_time": rows[s]["on_time"],
                          "spent_bytes": rows[s]["spent_bytes"],
                          "behavior_sha256": rows[s]["behaviour_sha256"]}
                 for s in expected},
        "cost_reward_closure": closure,
        "development_outcomes_used_for_training": False,
    }
    _write_once(r40.ROUND_DIR / "IMMEDIATE_REFERENCE_GATE.json", result)
    print("ROUND40 immediate references: 64/64; closure gate passed", flush=True)
    return result


def _independent_bundle(stream: dict[str, Any], decision_steps: list[int],
                        horizon: int) -> tuple[dict[int, float], float]:
    bundled = {int(step): 0.0 for step in decision_steps}
    prefix = 0.0
    for tick in range(0, int(horizon) + 1):
        value = float(stream.get(str(tick), 0.0))
        previous = [step for step in decision_steps if step <= tick]
        if previous:
            bundled[previous[-1]] += value
        else:
            prefix += value
    return bundled, prefix


def _independent_gae(rewards: list[float], values: list[float], gamma: float,
                     lam: float) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    carry = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        terminal = index == len(rewards) - 1
        bootstrap = 0.0 if terminal else values[index + 1]
        delta = rewards[index] + gamma * bootstrap - values[index]
        carry = delta if terminal else delta + gamma * lam * carry
        advantages[index] = carry
    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return advantages, returns


def _verify_first_four_gate(common: Any, trainer: Any, seed: int,
                            freeze: dict[str, Any]) -> dict[str, Any]:
    from r40_eligibility import COMM_ACTION_ELIGIBILITY, TAPE_SCHEMA

    rolling = common.CHECKPOINTS / f"v2_train_s{seed}.pt"
    if not rolling.is_file():
        raise RuntimeError(f"seed {seed} rolling checkpoint is missing after first batch")
    payload = torch.load(rolling, map_location="cpu", weights_only=False)
    state = payload["training_state"]
    if (payload.get("training_version") != TRAINING_VERSION
            or payload.get("comm_action_eligibility") != COMM_ACTION_ELIGIBILITY
            or state.get("comm_action_eligibility") != COMM_ACTION_ELIGIBILITY
            or int(state["episodes_done"]) != 4 or int(state["updates"]) != 1):
        raise RuntimeError(f"seed {seed} first-four checkpoint identity/cursor is invalid")
    commit_path = common.CHECKPOINTS / f"r40_batch_commit_s{seed}_b0.json"
    commit = _json(commit_path)
    if (_sha256(rolling) != commit["rolling_checkpoint_sha256"]
            or int(commit["episodes_done"]) != 4
            or int(commit["updates"]) != 1):
        raise RuntimeError(f"seed {seed} first batch commit receipt does not match rolling state")
    batch_path = common.CHECKPOINTS / commit["batch_record"]
    if _sha256(batch_path) != commit["batch_record_sha256"]:
        raise RuntimeError(f"seed {seed} first batch record hash is inconsistent")
    batch = _json(batch_path)
    report = batch["update"]
    alignment = report["likelihood_alignment"]
    if not alignment.get("passed") or float(alignment.get("max_abs_gap", math.inf)) > 1e-6:
        raise RuntimeError(f"seed {seed} first batch log-prob alignment failed: {alignment}")
    module_keys = set(report.get("module_gradient_l1_mean", {}))
    if module_keys != {"trunk", "action_head", "task_value_head", "cost_head"}:
        raise RuntimeError(f"seed {seed} PPO parameter modules differ: {module_keys}")
    if not all(math.isfinite(float(value)) for value in report["module_gradient_l1_mean"].values()):
        raise RuntimeError(f"seed {seed} has a nonfinite module gradient")
    if int(report["optimizer_steps"]) != int(batch["update"]["optimizer_steps"]):
        raise RuntimeError(f"seed {seed} optimizer step count is inconsistent")
    if int(report["optimizer_steps"]) <= 0:
        raise RuntimeError(f"seed {seed} first four episodes did not execute a PPO optimizer batch")

    input_root = Path(common.ROOT)
    old_init = (input_root / "runs" / "comm_final_train_20261001" / f"seed{seed}"
                / "checkpoints" / f"v2_train_s{seed}_ep0.pt")
    old_init_payload = torch.load(old_init, map_location="cpu", weights_only=False)
    old_controller, old_state, _identity = trainer.load_checkpoint_read_only(
        old_init, freeze=old_init_payload["freeze"])
    preupdate = common.CHECKPOINTS / batch["preupdate_checkpoint"]
    if _sha256(preupdate) != batch["preupdate_checkpoint_sha256"]:
        raise RuntimeError(f"seed {seed} pre-update weights are missing or changed")
    before_controller, before_state, _before_identity = trainer.load_checkpoint_read_only(
        preupdate, freeze=freeze)
    if before_controller.parameter_summary()["parameter_sha256"] != old_controller.parameter_summary()[
            "parameter_sha256"]:
        raise RuntimeError(f"seed {seed} NEW did not start at its registered ROUND36 ep0 weights")
    if (int(old_state["episodes_done"]) != 0 or float(old_state["lambda"]) != 1.0
            or int(before_state["updates"]) != 0 or int(before_state["optimizer_steps"]) != 0):
        raise RuntimeError(f"seed {seed} first batch pre-update state is not the registered init")

    references = _reference_rows(common)
    pairs = batch["constraint"]["pairs"]
    if len(pairs) != 4:
        raise RuntimeError(f"seed {seed} first batch has {len(pairs)} paired references")
    for pair in pairs:
        reference = references.get(int(pair["scenario"]))
        if reference is None or float(pair["y_reference"]) != float(reference["on_time"]):
            raise RuntimeError(f"seed {seed} g_hat did not use same-scene immediate reference")
    expected_g = sum(float(pair["y_reference"]) - float(pair["y_policy"])
                     for pair in pairs) / 4.0
    if abs(expected_g - float(batch["constraint"]["g_hat"])) > 1e-12:
        raise RuntimeError(f"seed {seed} first batch g_hat does not recompute")
    expected_lambda = max(0.0, 1.0 + common.ETA_LAMBDA * expected_g)
    if abs(expected_lambda - float(state["lambda"])) > 1e-12:
        raise RuntimeError(f"seed {seed} dual update does not use the frozen initial lambda")

    rows = []
    total_free = 0
    total_refresh = 0
    tape_map: dict[int, list[dict[str, Any]]] = {}
    artifacts = batch.get("episode_artifacts", [])
    if len(artifacts) != 4:
        raise RuntimeError(f"seed {seed} first batch lacks per-episode artifact identities")
    for artifact, episode_index, scenario in zip(artifacts, batch["episodes"], batch["scenarios"]):
        if (int(artifact["episode_index"]) != int(episode_index)
                or int(artifact["scenario_seed"]) != int(scenario)):
            raise RuntimeError("batch episode artifact map is misaligned")
        row_path = Path(artifact["row_path"])
        if _sha256(row_path) != artifact["row_sha256"]:
            raise RuntimeError(f"seed {seed} episode {episode_index} row hash changed")
        row = _json(row_path)
        if (row.get("comm_action_eligibility") != COMM_ACTION_ELIGIBILITY
                or row.get("training_version") != TRAINING_VERSION
                or row.get("reward_timing_version") != "published-completion-event-at-relay-tick/2"):
            raise RuntimeError(f"seed {seed} episode {episode_index} row version fields are incomplete")
        _validate_row_closure(row)
        if int(row.get("relay_counters", {}).get("refresh_emissions", 0)) != 0:
            raise RuntimeError(f"seed {seed} episode {episode_index} emitted a refresh")
        tape_path = Path(artifact["tape_path"])
        if _sha256(tape_path) != artifact["tape_sha256"]:
            raise RuntimeError(f"seed {seed} episode {episode_index} tape hash changed")
        tapes = [json.loads(line) for line in tape_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        for tape in tapes:
            if tape.get("schema") != TAPE_SCHEMA:
                raise RuntimeError("first-four tape lacks ROUND40 schema")
            targets = set(tape["targets"])
            classes = [set(tape["forced"]), set(tape["masked"]),
                       set(tape["fixed_hold_refresh"]), set(tape["actionable"])]
            if set().union(*classes) != targets or any(
                    classes[i] & classes[j] for i in range(len(classes))
                    for j in range(i + 1, len(classes))):
                raise RuntimeError("first-four tape candidate classes do not close")
            for target in tape["fixed_hold_refresh"]:
                if tape["actions"][target] != 0 or target in tape["actionable"]:
                    raise RuntimeError("a refresh entered the free action set")
            total_free += len(tape["actionable"])
            total_refresh += len(tape["fixed_hold_refresh"])
        tape_map[int(episode_index)] = tapes
        rows.append(row)

    episode_objects = [SimpleNamespace(records=tape_map[int(index)])
                       for index in batch["episodes"]]
    samples = []
    for row, collector in zip(rows, episode_objects):
        episode = {
            "collector": collector,
            "steps": int(row["steps"]),
            "cost_stream": {int(step): value for step, value in row["cost_stream"].items()},
            "task_stream": {int(step): float(value) for step, value in row["task_stream"].items()},
            "lambda": float(row["lambda_at_rollout"]),
            "episode_index": int(row["episode_index"]),
            "scenario_seed": int(row["scenario_seed"]),
            "on_time": float(row["on_time"]),
        }
        samples.extend(trainer.batch_samples([episode]))
    alignment_model = trainer.likelihood_alignment(before_controller, samples)
    if not alignment_model.get("passed") or float(alignment_model["max_abs_gap"]) > 1e-6:
        raise RuntimeError(f"seed {seed} first-four independently loaded pre-update policy "
                           f"does not align: {alignment_model}")

    # Recompute reward/cost bundles, per-decision GAE and lambda*A_task - A_cost without calling
    # the archived bundler or GAE function as the expected answer.
    independently_recomputed = 0
    for row, collector in zip(rows, episode_objects):
        tapes = sorted(collector.records, key=lambda tape: int(tape["step"]))
        if not tapes:
            continue
        decision_steps = [int(tape["step"]) for tape in tapes]
        cost_raw, cost_prefix = _independent_bundle(row["cost_stream"], decision_steps,
                                                    int(row["steps"]))
        task_raw, task_prefix = _independent_bundle(row["task_stream"], decision_steps,
                                                    int(row["steps"]))
        rewards_task = [task_raw[step] for step in decision_steps]
        rewards_cost = [cost_raw[step] / common.B0_BYTES for step in decision_steps]
        values_task = [float(tape["value"]) for tape in tapes]
        values_cost = [float(tape["cost_value"]) for tape in tapes]
        gae_task, returns_task = _independent_gae(rewards_task, values_task,
                                                 common.GAMMA, common.GAE_LAMBDA)
        gae_cost, returns_cost = _independent_gae(rewards_cost, values_cost,
                                                 common.GAMMA, common.GAE_LAMBDA)
        combined = [float(row["lambda_at_rollout"]) * task_adv - cost_adv
                    for task_adv, cost_adv in zip(gae_task, gae_cost)]
        sample_rows = [sample for sample in samples
                       if int(sample["episode_index"]) == int(row["episode_index"])]
        if len(sample_rows) != len(tapes):
            raise RuntimeError("batch_samples count differs from independent tape row count")
        for index, sample in enumerate(sample_rows):
            for key, expected in (("advantage", combined[index]),
                                  ("return_task", returns_task[index]),
                                  ("return_cost", returns_cost[index])):
                if abs(float(sample[key]) - float(expected)) > 1e-8:
                    raise RuntimeError(f"seed {seed} independent {key} differs at row {index}")
        independently_recomputed += len(tapes)
        if cost_prefix < 0 or task_prefix < 0:
            raise RuntimeError("prefix totals cannot be negative under this task/cost convention")

    module_gradients = report["module_gradient_l1_mean"]
    row_result = {
        "training_seed": seed,
        "episodes": 4,
        "scenarios": batch["scenarios"],
        "episodes_done": int(state["episodes_done"]),
        "updates": int(state["updates"]),
        "optimizer_steps": int(state["optimizer_steps"]),
        "batch_optimizer_steps": int(report["optimizer_steps"]),
        "free_first_report_decisions": total_free,
        "fixed_hold_refresh_decisions": total_refresh,
        "refresh_emissions": sum(int(row["relay_counters"].get("refresh_emissions", 0))
                                  for row in rows),
        "likelihood_alignment": alignment_model,
        "lambda_at_rollout": batch["lambda_at_rollout"],
        "g_hat": batch["constraint"]["g_hat"],
        "lambda_next": state["lambda"],
        "independent_gae_rows_checked": independently_recomputed,
        "module_gradient_l1_mean": module_gradients,
        "parameter_identity_before": batch["controller_parameters_before"]["parameter_sha256"],
        "parameter_identity_after": batch["controller_parameters_after"]["parameter_sha256"],
        "search_allocation_checkpoints_unchanged": True,
        "passed": True,
    }
    _write_once(r40.ROUND_DIR / f"SEED{seed}_FIRST4_GATE.json", row_result)
    return row_result


def _run_seed(common: Any, runner: Any, trainer: Any, seed: int, episodes: int,
              freeze: dict[str, Any]) -> dict[str, Any]:
    from r40_run import train_seed

    result = train_seed(common, runner, trainer, training_seed=seed,
                        episodes=episodes, freeze=freeze)
    path = r40.ROUND_DIR / "training_progress" / f"TRAINING_SUMMARY_s{seed}_ep{episodes}.json"
    summary = {key: result[key] for key in (
        "training_seed", "episodes_done", "updates", "optimizer_steps", "lambda",
        "lambda_history", "checkpoint", "controller_parameters", "training_version",
        "comm_action_eligibility")}
    _write_once(path, summary)
    return result


def _run_all_training(common: Any, runner: Any, trainer: Any, freeze: dict[str, Any]) -> None:
    refs = r40.ROUND_DIR / "IMMEDIATE_REFERENCE_GATE.json"
    if not refs.is_file():
        raise RuntimeError("immediate reference gate must pass before training")
    for seed in (11, 22, 33):
        gate_path = r40.ROUND_DIR / f"SEED{seed}_FIRST4_GATE.json"
        if not gate_path.exists():
            _run_seed(common, runner, trainer, seed, 4, freeze)
            gate = _verify_first_four_gate(common, trainer, seed, freeze)
            print(f"ROUND40 seed {seed} first-four gate passed; optimizer_steps="
                  f"{gate['batch_optimizer_steps']}", flush=True)
        else:
            gate = _json(gate_path)
            if gate.get("passed") is not True:
                raise RuntimeError(f"seed {seed} first-four gate was not passed")
        _run_seed(common, runner, trainer, seed, 128, freeze)
        print(f"ROUND40 seed {seed} training complete", flush=True)


def _model_eval_complete(common: Any, arm: str, seed: int, mode: str,
                         scenario: int) -> bool:
    logical = f"model_eval_{arm.lower()}_s{seed}_{mode}_scene{scenario}"
    claim = _latest_claim(logical)
    if claim is None or claim.get("status") != "complete":
        return False
    block = f"ep128_{mode}"
    path = common.ROWS / (
        f"v2_r40_eval_{arm.lower()}_s{seed}_{block}_s{scenario}_a{claim['attempt']}.json"
    )
    return path.is_file()


def _rule_complete(common: Any, arm: str, scenario: int) -> bool:
    claim = _latest_claim(f"rule_{arm}_scene{scenario}")
    if claim is None or claim.get("status") != "complete":
        return False
    return (common.ROWS / f"v2_r40_rule_{arm}_s{scenario}_a{claim['attempt']}.json").is_file()


def _run_model_evaluation(common: Any, runner: Any, trainer: Any,
                          controller_factory: Any, freeze: dict[str, Any],
                          input_root: Path) -> None:
    from r40_run import load_eval_model, run_model_eval

    for seed in r40.TRAIN_SEEDS:
        controllers = {}
        identities = {}
        for arm in ("OLD", "NEW"):
            controllers[arm], identities[arm] = load_eval_model(
                trainer, arm=arm, training_seed=seed, new_freeze=freeze,
                input_root=input_root, common=common)
        for mode in ("greedy", "sampled"):
            for scenario in r40.DEV_SCENARIOS:
                for arm in ("OLD", "NEW"):
                    if _model_eval_complete(common, arm, seed, mode, scenario):
                        continue
                    print(f"ROUND40 model eval {arm} seed={seed} {mode} scene={scenario}",
                          flush=True)
                    run_model_eval(common, runner, trainer, arm=arm,
                                   training_seed=seed, mode=mode,
                                   scenario_seed=scenario, timepoint="ep128",
                                   controller=controllers[arm],
                                   checkpoint_identity=identities[arm])


def _run_rules(common: Any, runner: Any) -> None:
    from r40_run import run_rule

    for arm in r40.RULE_ARMS:
        for scenario in r40.DEV_SCENARIOS:
            if _rule_complete(common, arm, scenario):
                continue
            print(f"ROUND40 rule eval {arm} scene={scenario}", flush=True)
            run_rule(common, runner, arm=arm, scenario_seed=scenario)


def run(input_root: str | Path, phase: str = "full") -> dict[str, Any]:
    root, common, runner, trainer, v2_controller, freeze = _load_context(input_root)
    if phase in ("full", "references"):
        run_references(common, runner)
    if phase == "references":
        return {"phase": phase, "ledger": r40.ledger_snapshot()}
    if phase not in ("full", "training", "models", "rules"):
        raise ValueError(f"unknown ROUND40 phase {phase!r}")
    if phase in ("full", "training"):
        _run_all_training(common, runner, trainer, freeze)
    if phase == "training":
        return {"phase": phase, "ledger": r40.ledger_snapshot()}
    if phase in ("full", "models"):
        for seed in r40.TRAIN_SEEDS:
            checkpoint = common.CHECKPOINTS / f"v2_train_s{seed}_ep128.pt"
            if not checkpoint.is_file():
                raise RuntimeError(f"all three fixed ep128 checkpoints are required: {checkpoint}")
        _run_model_evaluation(common, runner, trainer, v2_controller, freeze, root)
    if phase == "models":
        return {"phase": phase, "ledger": r40.ledger_snapshot()}
    if phase in ("full", "rules"):
        _run_rules(common, runner)
    ledger = r40.ledger_snapshot()
    if ledger["total_used"] > r40.HARD_CAP:
        raise RuntimeError("ROUND40 reached beyond its 700 launch hard cap")
    if phase == "full":
        from r40_analysis import analyze

        results = analyze(common)
        print(json.dumps({"analysis": "complete",
                          "logical_jobs": results["launch_ledger"]["completed_logical_jobs"],
                          "launches": results["launch_ledger"]["total_used"]},
                         ensure_ascii=False), flush=True)
    return {"phase": phase, "ledger": ledger}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="execute the frozen ROUND40 plan")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--phase", choices=("full", "references", "training", "models", "rules"),
                        default="full")
    args = parser.parse_args(argv)
    before = r40.ledger_snapshot()["total_used"]
    try:
        result = run(args.input_root, args.phase)
    except BaseException as error:
        after = r40.ledger_snapshot()["total_used"]
        if after == before:
            _record_preconstruction_failure(args.phase, error, before, after)
        raise
    print(json.dumps({"phase": result["phase"], "ledger": {
        key: result["ledger"][key] for key in
        ("normal_used", "reserve_used", "total_used")}}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
