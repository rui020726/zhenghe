"""Execute the frozen ROUND41 first-scene gate, then the remaining 45 hazard cells."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

RUN_DIR = Path(__file__).resolve().parent
R41_ROOT = RUN_DIR.parents[1]
ROUND40_OUTPUT = Path(r"D:\code\整合\zhenghe-round40\runs\comm_first_report_training_ablation_20261003")
ROUND40_WORKTREE = Path(r"D:\code\整合\zhenghe-round40")
CANONICAL_INPUT_ROOT = Path(r"D:\code\整合\zhenghe")
SEEDS = (11, 22, 33)
SCENARIOS = tuple(range(1_301_000, 1_301_016))
FIRST_SCENE = SCENARIOS[0]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(R41_ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def is_finite_tree(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite JSON value at {path}: {value}")
    if isinstance(value, dict):
        for key, nested in value.items():
            is_finite_tree(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            is_finite_tree(nested, f"{path}[{index}]")


def configure_imports() -> None:
    directories = [
        CANONICAL_INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928",
        CANONICAL_INPUT_ROOT / "runs" / "comm_cost_stage_b_20260927",
        CANONICAL_INPUT_ROOT / "runs" / "comm_budget_action_mask_fix_20260922",
        CANONICAL_INPUT_ROOT / "runs" / "comm_budget_pilot_20260919",
        CANONICAL_INPUT_ROOT / "runs" / "comm_cost_controller_stage_a_20260927",
        CANONICAL_INPUT_ROOT / "runs" / "comm_tight_budget_20260919",
        CANONICAL_INPUT_ROOT / "runs" / "search_trace_and_policy_20260925",
        ROUND40_OUTPUT,
        CANONICAL_INPUT_ROOT,
    ]
    for directory in reversed(directories):
        if not directory.is_dir():
            raise FileNotFoundError(f"ROUND41 frozen dependency is missing: {directory}")
        sys.path.insert(0, str(directory))


def verify_code_and_branch(frozen: dict[str, Any]) -> str:
    if git("branch", "--show-current") != frozen["execution_identity"]["round41_branch"]:
        raise RuntimeError("ROUND41 branch changed")
    if git("status", "--porcelain"):
        raise RuntimeError("ROUND41 source worktree is dirty; refusing to launch")
    head = git("rev-parse", "HEAD")
    expected = frozen["round41_code_sha256"]
    for filename, source_hash in expected.items():
        path = RUN_DIR / filename
        if not path.is_file() or digest(path) != source_hash:
            raise RuntimeError(f"ROUND41 frozen source changed: {filename}")
        subprocess.run(["git", "-C", str(R41_ROOT), "cat-file", "-e",
                        f"HEAD:runs/{RUN_DIR.name}/{filename}"],
                       check=True, stdout=subprocess.DEVNULL)
    for filename in ("ROUND41_FROZEN_CONFIG.json", "ROUND41_PURE_COMPUTE_GATE.json"):
        subprocess.run(["git", "-C", str(R41_ROOT), "cat-file", "-e",
                        f"HEAD:runs/{RUN_DIR.name}/{filename}"],
                       check=True, stdout=subprocess.DEVNULL)
    return head


def verify_external_record(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"frozen ROUND40 input missing: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise RuntimeError(f"frozen ROUND40 input size changed: {path}")
    if digest(path) != record["sha256"]:
        raise RuntimeError(f"frozen ROUND40 input SHA256 changed: {path}")


def verify_all_reused_inputs(frozen: dict[str, Any]) -> None:
    seen: set[str] = set()
    for record in frozen["baseline_reuse"]["rows"].values():
        for key in ("row", "tape", "lifecycle", "transcript"):
            item = record.get(key)
            if not item:
                continue
            normalized = str(Path(item["path"]).resolve()).casefold()
            if normalized in seen:
                continue
            verify_external_record(item)
            seen.add(normalized)
    for record in frozen["models"].values():
        verify_external_record(record)
    for name, record in frozen["source_artifacts"].items():
        verify_external_record(record)
    expected_head = frozen["execution_identity"]["round40_head"]
    if subprocess.run(["git", "-C", str(ROUND40_WORKTREE), "rev-parse", "HEAD"],
                      check=True, capture_output=True, text=True).stdout.strip() != expected_head:
        raise RuntimeError("ROUND40 branch tip changed after prelaunch freeze")


@contextlib.contextmanager
def forbid_training_and_checkpoint_writes(torch: Any, counters: dict[str, int]) -> Iterator[None]:
    originals: list[tuple[Any, str, Any]] = []
    visited: set[type] = set()

    def descendants(base: type) -> list[type]:
        output = [base]
        for child in base.__subclasses__():
            if child not in visited:
                visited.add(child)
                output.extend(descendants(child))
        return output

    for cls in descendants(torch.optim.Optimizer):
        original = cls.__dict__.get("step")
        if original is None:
            continue

        def blocked_step(*_args: Any, _cls: type = cls, **_kwargs: Any) -> None:
            counters["optimizer_step_calls"] += 1
            raise RuntimeError(f"ROUND41 forbids optimizer step: {_cls.__name__}.step")

        originals.append((cls, "step", original))
        setattr(cls, "step", blocked_step)

    original_save = torch.save

    def blocked_save(*_args: Any, **_kwargs: Any) -> None:
        counters["checkpoint_write_calls"] += 1
        raise RuntimeError("ROUND41 forbids checkpoint writes")

    torch.save = blocked_save
    try:
        yield
    finally:
        torch.save = original_save
        for cls, name, original in reversed(originals):
            setattr(cls, name, original)


def load_runtime(frozen: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    configure_imports()
    import torch
    import r40_common

    common = r40_common.bootstrap(CANONICAL_INPUT_ROOT)
    common.DEVICE = torch.device("cpu")
    common.OUTPUT_DIR = RUN_DIR
    common.LOGS = RUN_DIR / "logs"
    common.ROWS = RUN_DIR / "rows"
    common.REJECTED = RUN_DIR / "rejected"
    common.CHECKPOINTS = RUN_DIR / "checkpoints"
    for path in (common.LOGS, common.ROWS, common.REJECTED):
        path.mkdir(parents=True, exist_ok=True)

    runner = importlib.import_module("v2_run")
    trainer = importlib.import_module("v2_trainer")
    controller_module = importlib.import_module("v2_controller")
    round40_runner = importlib.import_module("r40_run")
    versions = importlib.import_module("r40_versions")
    r40_freeze = json_read(ROUND40_OUTPUT / "ROUND40_FROZEN_CONFIG.json")["round40"]["freeze_identity"]
    versions.install(trainer, runner, freeze=r40_freeze)
    runner.common = common
    return common, runner, trainer, controller_module, round40_runner, r40_freeze


def load_model(seed: int, frozen: dict[str, Any], trainer: Any,
               r40_freeze: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    source = frozen["models"][str(seed)]
    checkpoint = Path(source["path"])
    verify_external_record(source)
    controller, state, identity = trainer.load_checkpoint_read_only(checkpoint, freeze=r40_freeze)
    identity.update({
        "arm": "NEW",
        "path": str(checkpoint),
        "file_sha256": digest(checkpoint),
        "training_seed": int(seed),
        "episodes_done": int(state["episodes_done"]),
        "updates": int(state["updates"]),
        "optimizer_steps": int(state["optimizer_steps"]),
        "lambda": float(state["lambda"]),
        "parameter_identity": json_canonical(controller.parameter_summary()),
    })
    if identity["parameter_identity"] != source["parameter_identity"]:
        raise RuntimeError(f"read-only model parameter identity changed for seed {seed}")
    if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
        raise RuntimeError(f"seed {seed} model did not load read-only")
    return controller, identity


def normalized_world_configuration(common: Any, runner: Any, scene: int,
                                   *, tag: str, attempt: int) -> tuple[dict[str, Any], str]:
    fields, _substituted = common.stage_b_inherited_config(scene)
    fields["comm_policy"] = "learned"
    paths = common.cell_paths("eval", scene, attempt=attempt, tag=tag)
    fields["output"] = str(paths["lifecycle"])
    fields["comm_transcript"] = str(paths["transcript"])
    configuration = runner.RunConfiguration(**fields)
    configuration.validate()
    actual = asdict(configuration)
    actual = {key: value for key, value in actual.items()
              if key not in {"output", "comm_transcript", "record_stream", "replay_stream",
                             "search_backend", "allocation_backend"}}
    serialized = json.dumps(actual, ensure_ascii=False, sort_keys=True, default=str)
    return actual, hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def closure(row: dict[str, Any]) -> list[str]:
    cost_total = sum(int(item.get("airtime_bytes", 0))
                     for item in row.get("cost_stream", {}).values())
    reward_total = sum(float(value) for value in row.get("task_stream", {}).values())
    if cost_total != int(row["spent_bytes"]) or cost_total != int(row["airtime_tap_total"]):
        raise RuntimeError(f"ROUND41 row cost closure failed: {row.get('label')}")
    if reward_total != int(row["on_time"]) or int(row["task_reward_total"]) != int(row["on_time"]):
        raise RuntimeError(f"ROUND41 row task closure failed: {row.get('label')}")
    if int(row.get("optimizer_steps", -1)) != 0:
        raise RuntimeError("evaluation row reports optimizer steps")
    ids = sorted({str(event["task_id"]) for event in row.get("task_reward_events", [])
                  if event.get("on_time") and event.get("credited")})
    if len(ids) != int(row["on_time"]):
        raise RuntimeError("ROUND41 target identities do not close onto task reward")
    if len(row.get("emission_sequence", [])) != int(row["emissions"]):
        raise RuntimeError("ROUND41 emissions do not close")
    return ids


def latest_complete_job(logical: str, tag: str, scene: int) -> dict[str, Any] | None:
    directory = RUN_DIR / "launch_claims" / logical
    if not directory.exists():
        return None
    completed = []
    for claim_path in sorted(directory.glob("attempt_*.json")):
        claim = json_read(claim_path)
        if claim.get("status") != "complete":
            continue
        row_path = RUN_DIR / "rows" / f"v2_{tag}_s{scene}_a{int(claim['attempt'])}.json"
        if row_path.is_file():
            completed.append((int(claim["attempt"]), row_path, claim))
    if not completed:
        return None
    _attempt, row_path, claim = completed[-1]
    row = json_read(row_path)
    row["_local_row_path"] = str(row_path)
    row["_local_row_sha256"] = digest(row_path)
    row["_launch_claim"] = claim
    return row


def run_cell(seed: int, mode: str, scene: int, *, common: Any, runner: Any,
             trainer: Any, model: Any, identity: dict[str, Any],
             round40_runner: Any, frozen: dict[str, Any]) -> dict[str, Any]:
    role = "greedy_reproduction" if mode == "greedy" else "formal_hazard_median"
    logical_mode = "greedy_reproduction" if mode == "greedy" else "hazard_median"
    logical = f"r41_new_s{seed}_{logical_mode}_scene{scene}"
    tag = f"r41_new_s{seed}_{logical_mode}"
    existing = latest_complete_job(logical, tag, scene)
    if existing is not None:
        return existing

    world, world_hash = normalized_world_configuration(
        common, runner, scene, tag=tag, attempt=1
    )
    expected_world_hash = frozen["world_and_scenarios"][
        "effective_world_config_sha256_by_scenario"][str(scene)
    ]
    if world_hash != expected_world_hash:
        raise RuntimeError(f"ROUND41 world config mismatch before scene {scene}")

    attempt = __import__("r41_common").next_attempt(logical)
    claim = __import__("r41_common").claim_launch(
        logical, attempt=attempt,
        metadata={
            "kind": "frozen_ep128_decoder_evaluation",
            "mode": mode,
            "role": role,
            "training_seed": seed,
            "scenario_seed": scene,
            "checkpoint_sha256": identity["file_sha256"],
            "parameter_sha256": identity["parameter_identity"]["parameter_sha256"],
            "world_configuration_sha256": world_hash,
            "round40_head": frozen["execution_identity"]["round40_head"],
        },
    )
    label = f"eval_new_s{seed}_ep128_{logical_mode}_s{scene}_a{attempt}"
    collector = (
        importlib.import_module("r40_eligibility").FirstReportOnlyCollector(model, greedy=True)
        if mode == "greedy"
        else importlib.import_module("r41_hazard_median").HazardMedianCollector(model)
    )
    # Both modes share ROUND40's greedy control-stream seed.  hazard_median itself draws no
    # random value, so other Torch-driven decisions begin from the same seed as the baseline.
    action_seed = runner.eval_action_seed(seed, scene, "greedy")
    original_builder = runner.build_joint_run
    import r41_common

    class RunProxy:
        def __init__(self, run: Any) -> None:
            self._run = run

        def __getattr__(self, name: str) -> Any:
            return getattr(self._run, name)

        def run(self, *args: Any, **kwargs: Any) -> Any:
            r41_common.mark_simulation_started(claim)
            return self._run.run(*args, **kwargs)

    def tracked_builder(configuration: Any) -> Any:
        r41_common.mark_environment_constructor(claim)
        return RunProxy(original_builder(configuration))

    runner.build_joint_run = tracked_builder
    try:
        row = runner._run_episode(
            label, "eval", int(scene), policy="learned", attach=collector.policy(),
            attempt=attempt, controller=model, collector=collector,
            action_rng_seed=action_seed,
            tag=tag,
            checkpoint_load_identity=identity,
        )
        row.update({
            "training_seed": int(seed),
            "model_arm": "NEW",
            "timepoint": "ep128",
            "eval_mode": mode,
            "eval_block": f"ep128_{logical_mode}",
            "action_rng_seed": int(action_seed),
            "checkpoint_parameter_identity": identity["parameter_identity"],
            "round41_decoder": mode,
            "round41_role": role,
            "round41_decoder_schema": (
                "round41-cumulative-risk-decoder/1" if mode == "hazard_median" else None
            ),
            "round41_world_configuration_sha256": world_hash,
            "round41_world_configuration": world,
            "optimizer_step_calls": 0,
            "checkpoint_write_calls": 0,
        })
        row["controller_parameters_after"] = json_canonical(row["controller_parameters_after"])
        versions = importlib.import_module("r40_versions")
        versions.add_row_identity(row, checkpoint=identity)
        on_time_ids = closure(row)
        if json_canonical(row.get("controller_parameters_after")) != identity["parameter_identity"]:
            raise RuntimeError(f"frozen model parameters changed during {logical}")
        if digest(Path(identity["path"])) != identity["file_sha256"]:
            raise RuntimeError(f"frozen checkpoint bytes changed during {logical}")
        runner._booked(label, "eval", row, round40_runner._BookedInR40Ledger())
        r41_common.finish_launch(
            claim, ok=True,
            detail=(f"mode={mode} seed={seed} scene={scene} on_time={row['on_time']} "
                    f"bytes={row['spent_bytes']} targets={len(on_time_ids)}"),
        )
        row_path = Path(row["artefacts"]["row"])
        row["_local_row_path"] = str(row_path)
        row["_local_row_sha256"] = digest(row_path)
        row["_launch_claim"] = json_read(claim)
        return row
    except Exception as error:
        r41_common.finish_launch(
            claim, ok=False, detail=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        runner.build_joint_run = original_builder


def load_local_tapes(row: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(row["artefacts"]["tape"])
    output = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            is_finite_tree(record, f"{path.name}:{line_number}")
            output.append(record)
    return output


def verify_hazard_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("round41_decoder") != "hazard_median":
        raise RuntimeError("hazard row is mislabeled")
    if row.get("eval_mode") in ("greedy", "sampled"):
        raise RuntimeError("hazard_median was mislabeled as greedy/sample")
    if row.get("collector_summary", {}).get("decoder") != "hazard_median":
        raise RuntimeError("hazard collector summary is mislabeled")
    if row.get("collector_summary", {}).get("action_log_prob_semantics") != "not_a_sampling_likelihood":
        raise RuntimeError("hazard decoder log-probability semantics are missing")
    tapes = load_local_tapes(row)
    previous: dict[str, float] = {}
    closed: set[str] = set()
    updated_by_tick: set[tuple[str, int]] = set()
    update_count = 0
    masked_count = 0
    free_count = 0
    tau = math.log(2.0)
    for tape in tapes:
        tick = int(tape["step"])
        for target in tape["targets"]:
            target = str(target)
            audit = tape["hazard_audit"][target]
            is_first = bool(tape["candidate_audit"].get(target, {}).get("first_emission"))
            before = float(previous.get(target, 0.0))
            if not math.isfinite(float(audit["H_before"])) or not math.isfinite(float(audit["H_after"])):
                raise RuntimeError(f"non-finite H in {target} at tick {tick}")
            if not math.isclose(float(audit["H_before"]), before, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"H continuity failed for {target} at tick {tick}")
            if not is_first:
                closed.add(target)
                if audit["H_updated"] or int(tape["actions"][target]) != 0:
                    raise RuntimeError(f"refresh escaped fixed HOLD for {target} at tick {tick}")
            if audit["H_updated"]:
                if target not in tape["free_first_reports"] or not is_first or target in closed:
                    raise RuntimeError(f"masked/closed target accumulated H for {target} at tick {tick}")
                marker = (target, tick)
                if marker in updated_by_tick:
                    raise RuntimeError(f"same target accumulated twice in relay tick {marker}")
                updated_by_tick.add(marker)
                p = float(tape["send_probabilities"][target])
                if not math.isfinite(p) or p < 0.0 or p > 1.0:
                    raise RuntimeError(f"invalid saved P(SEND) for {target} at tick {tick}")
                if p == 1.0:
                    expected_after = max(before, tau)
                    if not audit.get("p_one_saturated") or audit.get("hazard_increment") is not None:
                        raise RuntimeError(f"p=1 is not finitely saturated for {target} at tick {tick}")
                else:
                    expected_after = before - math.log1p(-p)
                    if audit.get("p_one_saturated"):
                        raise RuntimeError(f"false p=1 saturation for {target} at tick {tick}")
                if not math.isclose(float(audit["H_after"]), expected_after,
                                    rel_tol=1e-12, abs_tol=1e-12):
                    raise RuntimeError(f"H recurrence failed for {target} at tick {tick}")
                update_count += 1
            else:
                if not math.isclose(float(audit["H_after"]), before,
                                    rel_tol=0.0, abs_tol=1e-12):
                    raise RuntimeError(f"H changed without an eligible update for {target} at tick {tick}")
                if target in tape["masked"]:
                    masked_count += 1
            if target in tape["free_first_reports"]:
                free_count += 1
                expected_action = 1 if float(audit["H_after"]) >= tau else 0
                if int(tape["actions"][target]) != expected_action:
                    raise RuntimeError(f"threshold action mismatch for {target} at tick {tick}")
            if bool(tape["requested"][target]) != (int(tape["actions"][target]) == 1):
                raise RuntimeError(f"request/action mismatch for {target} at tick {tick}")
            previous[target] = float(audit["H_after"])

    sends = [tuple(row_item) for row_item in row["emission_sequence"]]
    if any(not bool(item[5]) for item in sends):
        raise RuntimeError("refresh emission appeared under first-report-only")
    if int(row["relay_counters"].get("emissions", -1)) != len(sends):
        raise RuntimeError("relay send ledger does not close")
    if int(row["spent_bytes"]) != sum(int(item[3]) for item in sends):
        raise RuntimeError("actual emitted bytes do not close to row bytes")

    transcript_path = Path(row["artefacts"]["transcript"])
    transcript = [json.loads(line) for line in transcript_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    terminal_send_keys = {
        (int(event["step"]), str(event["target"]))
        for event in transcript if event.get("kind") in ("send", "refused")
    }
    requested_keys = {
        (int(tape["step"]), str(target))
        for tape in tapes for target, requested in tape["requested"].items() if requested
    }
    if requested_keys != terminal_send_keys:
        raise RuntimeError("SEND requests do not match relay sends/refusals exactly")
    return {
        "tape_rows": len(tapes),
        "free_legal_opportunities": free_count,
        "H_updates": update_count,
        "masked_opportunities": masked_count,
        "unique_target_tick_updates": len(updated_by_tick),
        "request_send_or_refusal_events": len(requested_keys),
        "real_first_emissions": len(sends),
        "refresh_emissions": 0,
        "candidate_audit_sha256": digest(Path(row["artefacts"]["tape"])),
    }


def normalized_tape_prefix(greedy_row: dict[str, Any], hazard_row: dict[str, Any]) -> dict[str, Any]:
    greedy = load_local_tapes(greedy_row)
    hazard = load_local_tapes(hazard_row)
    limit = min(len(greedy), len(hazard))
    divergence: dict[str, Any] | None = None
    for index in range(limit):
        old, new = greedy[index], hazard[index]
        for key in ("step", "targets", "features", "free_first_reports", "masked",
                    "fixed_hold_refresh", "forced", "shadow_budget_before",
                    "shadow_capacity_before"):
            if old.get(key) != new.get(key):
                if divergence is None:
                    raise RuntimeError(f"first-scene action prefix diverged before action at tape row {index}: {key}")
        for target in old["targets"]:
            if old.get("send_probabilities", {}).get(target) != new.get("send_probabilities", {}).get(target):
                if divergence is None:
                    raise RuntimeError(f"model probability prefix changed before first action divergence at row {index}")
        changed = [str(target) for target in old["targets"]
                   if old["actions"].get(target) != new["actions"].get(target)]
        if changed:
            divergence = {
                "tape_index": index,
                "relay_tick": int(old["step"]),
                "first_differing_candidate": changed[0],
                "changed_candidates_in_original_order": changed,
                "same_observation_and_probability_prefix": True,
                "same_predecision_shadow_budget_and_capacity": True,
            }
            break
        if old["actions"] != new["actions"]:
            raise RuntimeError(f"action maps differ unexpectedly at first-scene tape row {index}")
    if divergence is None and len(greedy) != len(hazard):
        raise RuntimeError("greedy/hazard tapes ended at different decision steps without an action divergence")
    if divergence is None:
        divergence = {"tape_index": None, "relay_tick": None,
                      "first_differing_candidate": None,
                      "same_observation_and_probability_prefix": True,
                      "no_action_divergence": True}
    step = divergence["relay_tick"]
    if step is not None:
        def before_step(row: dict[str, Any]) -> list[list[Any]]:
            return [event for event in row["emission_sequence"] if int(event[0]) < int(step)]
        if before_step(greedy_row) != before_step(hazard_row):
            raise RuntimeError("emission transcript differs before the first action divergence")
    return divergence


def baseline_row(frozen: dict[str, Any], seed: int, mode: str,
                 scene: int) -> dict[str, Any]:
    key = f"NEW/s{seed}/{mode}/{scene}"
    record = frozen["baseline_reuse"]["rows"][key]
    verify_external_record(record["row"])
    row = json_read(Path(record["row"]["path"]))
    if row.get("behaviour_sha256") != record["behaviour_sha256"]:
        raise RuntimeError(f"frozen ROUND40 behavior identity changed for {key}")
    return row


def first_six_gate(rows: dict[str, dict[str, Any]], frozen: dict[str, Any]) -> dict[str, Any]:
    replay: dict[str, Any] = {}
    replay_results: dict[tuple[int, str], dict[str, Any]] = {}
    try:
        for seed in SEEDS:
            row = rows[f"greedy/{seed}"]
            base = baseline_row(frozen, seed, "greedy", FIRST_SCENE)
            base_identity = frozen["baseline_reuse"]["rows"][f"NEW/s{seed}/greedy/{FIRST_SCENE}"]
            if row.get("behaviour_sha256") != base_identity["behaviour_sha256"]:
                raise RuntimeError(f"greedy behavior SHA mismatch for seed {seed}")
            for key in ("on_time", "spent_bytes", "emissions", "emission_sequence",
                        "refusal_sequence", "cost_stream", "task_stream"):
                if row.get(key) != base.get(key):
                    raise RuntimeError(f"greedy replay field {key} differs for seed {seed}")
            if digest(Path(row["artefacts"]["tape"])) != base_identity["tape"]["sha256"]:
                raise RuntimeError(f"greedy replay tape bytes differ for seed {seed}")
            if row.get("action_rng_seed") != base.get("action_rng_seed"):
                raise RuntimeError(f"greedy replay action seed differs for seed {seed}")
            replay[str(seed)] = {
                "round40_row_sha256": base_identity["row"]["sha256"],
                "round40_tape_sha256": base_identity["tape"]["sha256"],
                "round41_row_sha256": digest(Path(row["artefacts"]["row"])),
                "round41_tape_sha256": digest(Path(row["artefacts"]["tape"])),
                "behavior_fields_equal": True,
                "behavior_sha256": row["behaviour_sha256"],
                "on_time": int(row["on_time"]),
                "spent_bytes": int(row["spent_bytes"]),
            }
            replay_results[(seed, "greedy")] = row

        hazard_checks: dict[str, Any] = {}
        prefixes: dict[str, Any] = {}
        for seed in SEEDS:
            row = rows[f"hazard_median/{seed}"]
            hazard_checks[str(seed)] = verify_hazard_row(row)
            prefixes[str(seed)] = normalized_tape_prefix(replay_results[(seed, "greedy")], row)
            greedy = replay_results[(seed, "greedy")]
            if row.get("round41_world_configuration_sha256") != greedy.get("round41_world_configuration_sha256"):
                raise RuntimeError(f"world config differs between first-six cells for seed {seed}")

        for seed in SEEDS:
            for mode in ("greedy", "hazard_median"):
                row = rows[f"{mode}/{seed}"]
                if int(row.get("on_time", -1)) != int(row.get("task_reward_total", -2)):
                    raise RuntimeError(f"first-six reward closure mismatch for {mode}/s{seed}")
                if int(row.get("spent_bytes", -1)) != int(row.get("airtime_tap_total", -2)):
                    raise RuntimeError(f"first-six cost closure mismatch for {mode}/s{seed}")
                if json_canonical(row.get("controller_parameters_after")) != frozen["models"][str(seed)]["parameter_identity"]:
                    raise RuntimeError(f"first-six model parameter identity changed for {mode}/s{seed}")
                if digest(Path(frozen["models"][str(seed)]["path"])) != frozen["models"][str(seed)]["sha256"]:
                    raise RuntimeError(f"checkpoint bytes changed for seed {seed}")
        ledger = importlib.import_module("r41_common").ledger_snapshot()
        return {
            "kind": "round41-first-six-gate/1",
            "status": "passed",
            "cells": 6,
            "greedy_reproduction": replay,
            "hazard_tape_recurrence": hazard_checks,
            "common_execution_prefix": prefixes,
            "task_and_cost_closure": True,
            "mask_H_invariance": True,
            "refresh_emissions": 0,
            "model_parameters_unchanged": True,
            "checkpoint_bytes_unchanged": True,
            "optimizer_step_calls": 0,
            "checkpoint_write_calls": 0,
            "environment_constructions": ledger["environment_constructor_calls"],
            "simulations_started": ledger["simulation_started"],
            "launch_attempts_including_retries": ledger["total_used"],
            "remaining_formal_hazard_cells": 45,
        }
    except Exception as error:
        ledger = importlib.import_module("r41_common").ledger_snapshot()
        return {
            "kind": "round41-first-six-gate/1",
            "status": "failed",
            "cells": len(rows),
            "error": f"{type(error).__name__}: {error}",
            "greedy_reproduction": replay,
            "hazard_tape_recurrence": locals().get("hazard_checks", {}),
            "common_execution_prefix": locals().get("prefixes", {}),
            "environment_constructions": ledger["environment_constructor_calls"],
            "simulations_started": ledger["simulation_started"],
            "launch_attempts_including_retries": ledger["total_used"],
        }


def write_progress(rows: dict[str, dict[str, Any]], *, phase: str,
                   frozen: dict[str, Any]) -> None:
    ledger = importlib.import_module("r41_common").ledger_snapshot()
    compact = []
    for key, row in sorted(rows.items()):
        compact.append({
            "cell": key,
            "row_path": row.get("_local_row_path", row.get("artefacts", {}).get("row")),
            "row_sha256": row.get("_local_row_sha256"),
            "on_time": int(row["on_time"]),
            "spent_bytes": int(row["spent_bytes"]),
            "checkpoint_parameter_identity": row.get("checkpoint_parameter_identity"),
        })
    json_write(RUN_DIR / "ROUND41_EXECUTION_PROGRESS.json", {
        "kind": "round41-execution-progress/1",
        "phase": phase,
        "prelaunch_head": frozen["execution_identity"]["round41_source_head_before_freeze_commit"],
        "ledger": ledger,
        "completed_rows": compact,
        "optimizer_step_calls": ledger["optimizer_step_calls"],
        "checkpoint_write_calls": ledger["checkpoint_write_calls"],
    })


def run(phase: str) -> None:
    frozen = json_read(RUN_DIR / "ROUND41_FROZEN_CONFIG.json")
    pure_gate = json_read(RUN_DIR / "ROUND41_PURE_COMPUTE_GATE.json")
    if frozen.get("status") != "passed" or pure_gate.get("status") != "passed":
        raise RuntimeError("ROUND41 prelaunch freeze or pure-compute gate did not pass")
    prelaunch_head = verify_code_and_branch(frozen)
    verify_all_reused_inputs(frozen)

    common, runner, trainer, _controller_module, round40_runner, r40_freeze = load_runtime(frozen)
    import torch
    r41_common = importlib.import_module("r41_common")
    ledger_before = r41_common.ledger_snapshot()
    if ledger_before["total_used"] != len(ledger_before["claims"]):
        raise RuntimeError("ROUND41 ledger slot/claim count is inconsistent")
    if phase == "all" and ledger_before["total_used"] > 56:
        raise RuntimeError("ROUND41 hard cap is already exhausted")
    models = {seed: load_model(seed, frozen, trainer, r40_freeze) for seed in SEEDS}
    counters = {"optimizer_step_calls": 0, "checkpoint_write_calls": 0}
    rows: dict[str, dict[str, Any]] = {}
    saved_builder = runner.build_joint_run
    # run_cell applies the constructor tracker around each isolated episode; training APIs
    # and checkpoint writers remain fail-closed across the entire execution.
    try:
        with forbid_training_and_checkpoint_writes(torch, counters):
            first_cells = [(seed, "greedy", FIRST_SCENE) for seed in SEEDS] + [
                (seed, "hazard_median", FIRST_SCENE) for seed in SEEDS
            ]
            if phase in ("all", "first_six"):
                for seed, mode, scene in first_cells:
                    key = f"{mode}/{seed}"
                    if key in rows:
                        continue
                    retry_count = 0
                    while True:
                        try:
                            rows[key] = run_cell(
                                seed, mode, scene, common=common, runner=runner,
                                trainer=trainer, model=models[seed][0], identity=models[seed][1],
                                round40_runner=round40_runner, frozen=frozen,
                            )
                            break
                        except Exception:
                            retry_count += 1
                            current = r41_common.ledger_snapshot()
                            reserve_used = current["reserve_used"]
                            if reserve_used >= 5 or retry_count > 5:
                                raise
                            continue
                    print(f"ROUND41 first-six {mode} seed={seed} scene={scene} "
                          f"on_time={rows[key]['on_time']} bytes={rows[key]['spent_bytes']}",
                          flush=True)
                    write_progress(rows, phase="first_six", frozen=frozen)

            if phase == "first_six":
                return
            if phase == "all":
                gate_path = RUN_DIR / "ROUND41_FIRST_SIX_GATE.json"
                gate = first_six_gate(rows, frozen)
                json_write(gate_path, gate)
                print(f"ROUND41 first-six gate: {gate['status']}", flush=True)
                if gate["status"] != "passed":
                    write_progress(rows, phase="stopped_first_six_gate", frozen=frozen)
                    raise RuntimeError(gate.get("error", "first-six gate failed"))

            if phase == "remaining45" or phase == "all":
                gate_path = RUN_DIR / "ROUND41_FIRST_SIX_GATE.json"
                gate = json_read(gate_path)
                if gate.get("status") != "passed":
                    raise RuntimeError("remaining45 cannot start until the six-cell gate passes")
                for seed in SEEDS:
                    for scene in SCENARIOS[1:]:
                        key = f"hazard_median/{seed}/{scene}"
                        retry_count = 0
                        while True:
                            try:
                                rows[key] = run_cell(
                                    seed, "hazard_median", scene, common=common, runner=runner,
                                    trainer=trainer, model=models[seed][0], identity=models[seed][1],
                                    round40_runner=round40_runner, frozen=frozen,
                                )
                                break
                            except Exception:
                                retry_count += 1
                                current = r41_common.ledger_snapshot()
                                if current["reserve_used"] >= 5 or retry_count > 5:
                                    raise
                                continue
                        print(f"ROUND41 hazard_median seed={seed} scene={scene} "
                              f"on_time={rows[key]['on_time']} bytes={rows[key]['spent_bytes']}",
                              flush=True)
                        write_progress(rows, phase="remaining45", frozen=frozen)
    finally:
        runner.build_joint_run = saved_builder

    final_ledger = r41_common.ledger_snapshot()
    if counters["optimizer_step_calls"] or counters["checkpoint_write_calls"]:
        raise RuntimeError(f"prohibited calls were observed: {counters}")
    if final_ledger["optimizer_step_calls"] or final_ledger["checkpoint_write_calls"]:
        raise RuntimeError("launch ledger records prohibited training or checkpoint writes")
    if phase == "all":
        expected_cells = 51
        complete_claims = sum(claim.get("status") == "complete" for claim in final_ledger["claims"])
        if complete_claims != expected_cells:
            raise RuntimeError(f"expected 51 completed cells, found {complete_claims}")
        if final_ledger["total_used"] > 56:
            raise RuntimeError("ROUND41 hard launch cap exceeded")
    write_progress(rows, phase="complete", frozen=frozen)
    print(json.dumps({
        "phase": phase,
        "prelaunch_head": prelaunch_head,
        "ledger": {key: final_ledger[key] for key in (
            "normal_used", "reserve_used", "total_used", "environment_constructor_calls",
            "simulation_started", "optimizer_step_calls", "checkpoint_write_calls",
        )},
        "guard_counters": counters,
    }, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "first_six", "remaining45"), default="all")
    args = parser.parse_args()
    run(args.phase)


if __name__ == "__main__":
    main()
