"""Serial ROUND39 runner: nine-cell technical gate, then the remaining fixed matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
R37 = ROOT / "runs" / "comm_final_evaluation_20261002"
STAGE_C = ROOT / "runs" / "comm_cost_v2_rerun_20260928"
BASE_HEAD = "233a44fc10048488a0c71bda9d88e977c99c800e"
BRANCH = "exp/comm-refresh-held-mechanism"
SCENES = [1600000, 1600001, 1600002, 1600003, 1600004, 1600005, 1600006,
          1600011, 1600018, 1600034, 1600036, 1600040, 1600049, 1600056,
          1600060, 1600062]
FAILED_SCENES = {1600000, 1600011, 1600018, 1600034, 1600036,
                 1600040, 1600049, 1600056, 1600060, 1600062}
ARMS = [
    "s11_original_greedy", "s11_refresh_held",
    "s22_original_greedy", "s22_refresh_held",
    "s33_original_greedy", "s33_refresh_held",
    "immediate", "late_discovery_gate_147", "always_hold",
]
SEEDS = [11, 22, 33]

sys.path[:0] = [
    str(ROOT), str(R37), str(STAGE_C),
    str(ROOT / "runs" / "comm_cost_stage_b_20260927"),
    str(ROOT / "runs" / "comm_budget_action_mask_fix_20260922"),
    str(ROOT / "runs" / "comm_tight_budget_20260919"), str(HERE),
]

import r37_run  # noqa: E402
import v2_common  # noqa: E402
import v2_run  # noqa: E402
from r39_collector import RefreshHeldCollector  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True,
        text=True, encoding="utf-8",
    ).stdout.strip()


def assert_prelaunch_commit() -> str:
    head = git("rev-parse", "HEAD")
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("ROUND39 runner is not on the dedicated experiment branch")
    if git("rev-parse", f"{head}^1") != BASE_HEAD:
        raise RuntimeError("prelaunch commit must be a single child of the required base")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked changes appeared after the prelaunch freeze commit")
    required = [
        "EXECUTION_BASIS.md", "FROZEN_SCENE_AND_ARM_PLAN.json", "CHECKPOINT_IDENTITY.json",
        "CODE_IDENTITY.json", "LAUNCH_BOOK.json", "PRELAUNCH_GATE.json",
        "r39_collector.py", "r39_preflight.py", "r39_freeze.py", "r39_run.py",
    ]
    for name in required:
        path = f"runs/comm_refresh_held_mechanism_20261002/{name}"
        subprocess.run(["git", "cat-file", "-e", f"{head}:{path}"], cwd=ROOT,
                       check=True, capture_output=True)
    gate = read_json(HERE / "PRELAUNCH_GATE.json")
    if not gate.get("all_passed") or gate.get("environment_constructions") != 0:
        raise RuntimeError("pure prelaunch gate is absent or did not pass")
    return head


def configure_output() -> None:
    v2_common.OUTPUT_DIR = HERE
    v2_common.LOGS = HERE / "logs"
    v2_common.ROWS = HERE / "rows"
    v2_common.REJECTED = HERE / "rejected"
    v2_common.CHECKPOINTS = HERE / "checkpoints"
    v2_common.LAUNCH_CAP = 153
    for path in (v2_common.OUTPUT_DIR, v2_common.LOGS, v2_common.ROWS,
                 v2_common.REJECTED, v2_common.CHECKPOINTS, HERE / "claims"):
        path.mkdir(parents=True, exist_ok=True)
    if v2_common.ledger_total() != 0:
        raise RuntimeError("ROUND39 launch ledger is not empty before the first cell")


def jobs() -> list[dict[str, Any]]:
    return [
        {"index": scene_index * len(ARMS) + arm_index,
         "scene": scene, "arm": arm}
        for scene_index, scene in enumerate(SCENES)
        for arm_index, arm in enumerate(ARMS)
    ]


def take_claim(job: dict[str, Any], *, retry: int = 0) -> None:
    path = HERE / "claims" / f"cell_{job['index']:03d}_retry_{retry}.claim"
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"cell was already claimed; duplicate run refused: {path.name}") from error
    with os.fdopen(fd, "w", encoding="utf-8") as sink:
        sink.write(json.dumps({"job": job, "retry": retry, "pid": os.getpid(),
                               "claimed_at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")


def original_r37_row_path(arm: str, scene: int) -> Path:
    if arm.startswith("s") and "_original_greedy" in arm:
        seed = int(arm[1:3])
        name = f"v2_eval_s{seed}_final_greedy_s{scene}_a0.json"
    else:
        rule = {"late_discovery_gate_147": "late_discovery_gate"}.get(arm, arm)
        name = f"v2_{rule}_s{scene}_a1.json"
    return R37 / "rows" / name


def label_and_kind(arm: str, scene: int) -> tuple[str, str]:
    if arm.startswith("s") and "_original_greedy" in arm:
        seed = int(arm[1:3])
        return f"eval_s{seed}_final_greedy_s{scene}", "eval"
    if arm.startswith("s") and "_refresh_held" in arm:
        seed = int(arm[1:3])
        return f"eval_s{seed}_refresh_held_s{scene}", "eval"
    return f"rule_{arm}_s{scene}", "rule"


def run_arm(arm: str, scene: int) -> dict[str, Any]:
    if arm.startswith("s") and "_original_greedy" in arm:
        seed = int(arm[1:3])
        controller, identity = r37_run.controller_for(seed, "greedy")
        return v2_run.run_eval(
            controller, training_seed=seed, timepoint="final", mode="greedy",
            scenario_seed=scene, checkpoint_load_identity=identity,
        )

    if arm.startswith("s") and "_refresh_held" in arm:
        seed = int(arm[1:3])
        controller, identity = r37_run.controller_for(seed, "greedy")
        collector = RefreshHeldCollector(
            controller, forced=lambda _item: False, mask=False, refresh_held=True
        )
        label = f"eval_s{seed}_refresh_held_s{scene}"
        ledger = v2_common.LaunchLedger()
        ledger.reserve("normal", label, scene, 0)
        try:
            row = v2_run._run_episode(
                label, "eval", scene, policy="learned", attach=collector.policy(),
                attempt=0, controller=controller, collector=collector,
                action_rng_seed=v2_run.eval_action_seed(seed, scene, "greedy"),
                tag=f"eval_s{seed}_refresh_held",
                checkpoint_load_identity=identity,
            )
        except Exception as error:
            ledger.record(label=label, seed=scene, attempt=0, ok=False,
                          detail=f"{type(error).__name__}: {error}")
            raise
        row["training_seed"] = int(seed)
        row["timepoint"] = "final"
        row["eval_mode"] = "greedy"
        row["eval_block"] = "final_greedy"
        row["intervention"] = "refresh_held_after_authoritative_first_emission"
        v2_run._booked(label, "eval", row, ledger)
        return row

    rule = {"late_discovery_gate_147": "late_discovery_gate"}.get(arm, arm)
    return v2_run.run_rule(rule, scene)


def row_path(arm: str, scene: int) -> Path:
    if arm.startswith("s") and "_original_greedy" in arm:
        seed = int(arm[1:3])
        return HERE / "rows" / f"v2_eval_s{seed}_final_greedy_s{scene}_a0.json"
    if arm.startswith("s") and "_refresh_held" in arm:
        seed = int(arm[1:3])
        return HERE / "rows" / f"v2_eval_s{seed}_refresh_held_s{scene}_a0.json"
    rule = {"late_discovery_gate_147": "late_discovery_gate"}.get(arm, arm)
    return HERE / "rows" / f"v2_{rule}_s{scene}_a1.json"


def add_attempt(job: dict[str, Any], *, status: str, detail: str = "",
                row: dict[str, Any] | None = None, error: str | None = None) -> None:
    book = read_json(HERE / "LAUNCH_BOOK.json")
    book["attempts"].append({
        "index": int(job["index"]), "scene": int(job["scene"]), "arm": job["arm"],
        "attempt": 0, "status": status, "detail": detail, "error": error,
        "row_path": str(row_path(job["arm"], job["scene"]).relative_to(ROOT)).replace("\\", "/"),
        "on_time": None if row is None else int(row["on_time"]),
        "spent_bytes": None if row is None else int(row["spent_bytes"]),
        "emissions": None if row is None else int(row["emissions"]),
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    book["starts"] = v2_common.ledger_total()
    book["completed"] = sum(item["status"] == "completed" for item in book["attempts"])
    book["technical_failures"] = sum(item["status"] == "technical_failure"
                                      for item in book["attempts"])
    book["environment_constructions"] = book["starts"]
    book["optimizer_steps"] = 0
    book["training_launches"] = 0
    book["checkpoint_writes"] = 0
    book["status"] = "running"
    atomic_json(HERE / "LAUNCH_BOOK.json", book)


BEHAVIOUR_FIELDS = (
    "policy", "spent_bytes", "budget_bytes", "emissions", "on_time", "completed",
    "expired", "discovered", "active", "unassigned", "task_reward_total",
    "training_version", "reward_timing_version", "b0_bytes", "normalised_cost_total",
    "emission_sequence", "refusal_sequence", "cost_stream", "task_stream",
    "task_reward_events", "ack_ledger", "transport_parameters", "relay_counters",
    "behaviour_sha256", "config_hash", "steps", "checkpoint_load_identity", "rule_identity",
)


def compare_r37_behavior(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    mismatches = {}
    for key in BEHAVIOUR_FIELDS:
        if key in new or key in old:
            if new.get(key) != old.get(key):
                mismatches[key] = {"round39": new.get(key), "round37": old.get(key)}
    return {"matches": not mismatches, "mismatches": mismatches}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def lifecycle_prefix(row: dict[str, Any], first_intervention_step: int) -> list[dict[str, Any]]:
    path = Path(row["artefacts"]["lifecycle"])
    records = read_jsonl(path)
    return [record for record in records
            if float(record.get("timestamp") or 0.0) <= float(first_intervention_step)]


def tape_by_step(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    path = Path(row["artefacts"]["tape"])
    return {int(record["step"]): record for record in read_jsonl(path)}


def validate_effective_budget(tape_rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    for tape in tape_rows:
        current = float(tape["relay_observation_remaining_bytes"])
        for target in tape["targets"]:
            # ROUND39 evaluation has forced=False. If a future harness changes that contract,
            # the row must be reviewed before this check can be relaxed.
            entry = tape["candidate_budget_audit"].get(target)
            if entry is None:
                raise RuntimeError(f"budget audit missing candidate {target} at step {tape['step']}")
            state = entry["effective_run"]
            if abs(float(state["budget_before"]) - current) > 1e-8:
                raise RuntimeError(
                    f"effective shadow order broke at step {tape['step']} target {target}"
                )
            charge = int(entry["attempt_charge_bytes"])
            should_charge = tape["effective_actions"][target] == 1
            if bool(state["charge_applied"]) != should_charge:
                raise RuntimeError(f"send charge mismatch at step {tape['step']} target {target}")
            expected_after = current - charge if should_charge else current
            if abs(float(state["budget_after"]) - expected_after) > 1e-8:
                raise RuntimeError(f"wrong effective shadow debit at step {tape['step']} target {target}")
            if expected_after < 0:
                raise RuntimeError(f"effective shadow went negative at step {tape['step']}")
            current = expected_after
            checked += 1
            if target in tape["intervened"]:
                if tape["effective_actions"][target] != 0 or should_charge:
                    raise RuntimeError("intervened refresh escaped HOLD/no-charge enforcement")
                if abs(float(state["budget_before"]) - float(state["budget_after"])) > 1e-8:
                    raise RuntimeError("refresh intervention consumed shadow credit")
        if abs(current - float(tape["shadow_budget_after"])) > 1e-8:
            raise RuntimeError(f"effective shadow end does not close at step {tape['step']}")
    return {"candidates_checked": checked, "nonnegative_and_sequential": True}


def validate_gate() -> dict[str, Any]:
    scene = 1600000
    rows: dict[str, dict[str, Any]] = {}
    round37_reproduction: dict[str, Any] = {}
    for arm in ARMS:
        path = row_path(arm, scene)
        if not path.exists():
            raise RuntimeError(f"first-scene cell missing row: {path}")
        row = read_json(path)
        rows[arm] = row
        if arm.endswith("original_greedy") or arm in {
            "immediate", "late_discovery_gate_147", "always_hold"
        }:
            old_path = original_r37_row_path(arm, scene)
            if not old_path.exists():
                raise RuntimeError(f"ROUND37 comparison row is missing: {old_path}")
            old = read_json(old_path)
            comparison = compare_r37_behavior(row, old)
            round37_reproduction[arm] = {
                "round37_row": str(old_path.relative_to(ROOT)).replace("\\", "/"),
                **comparison,
            }
            if not comparison["matches"]:
                raise RuntimeError(f"ROUND39 {arm} does not reproduce ROUND37: "
                                   f"{list(comparison['mismatches'])}")

        if row["optimizer_steps"] != 0:
            raise RuntimeError(f"optimizer step recorded in {arm}")
        if int(row["budget_bytes"]) != 29300 or int(row["spent_bytes"]) > int(row["budget_bytes"]):
            raise RuntimeError(f"budget mismatch in {arm}")
        if int(row["spent_bytes"]) != sum(
            int(item[1]["airtime_bytes"]) for item in row["cost_stream"].items()
        ):
            raise RuntimeError(f"cost-stream closure failed in {arm}")
        if int(row["on_time"]) != sum(int(value) for value in row["task_stream"].values()):
            raise RuntimeError(f"task-stream closure failed in {arm}")
        if len(row["emission_sequence"]) != int(row["emissions"]):
            raise RuntimeError(f"emission count closure failed in {arm}")

    checkpoint_expectations = read_json(HERE / "CHECKPOINT_IDENTITY.json")["models"]
    for seed in SEEDS:
        for suffix in ("original_greedy", "refresh_held"):
            arm = f"s{seed}_{suffix}"
            identity = rows[arm]["checkpoint_load_identity"]
            expected = checkpoint_expectations[f"s{seed}"]
            if identity["checkpoint_bytes_sha256"] != expected["bytes_sha256"]:
                raise RuntimeError(f"checkpoint bytes changed in {arm}")
            if identity["parameter_sha256"] != expected["parameter_sha256"]:
                raise RuntimeError(f"loaded parameters changed in {arm}")
            parameters_after = rows[arm]["controller_parameters_after"]
            if parameters_after["parameter_sha256"] != expected["parameter_sha256"]:
                raise RuntimeError(f"controller weights changed during {arm}")

    held_details = {}
    for seed in (11, 33):
        original = rows[f"s{seed}_original_greedy"]
        held = rows[f"s{seed}_refresh_held"]
        tapes = read_jsonl(Path(held["artefacts"]["tape"]))
        validation = validate_effective_budget(tapes)
        intervention_tapes = [tape for tape in tapes if tape["intervened"]]
        if not intervention_tapes:
            raise RuntimeError(f"seed {seed} gate path has no refresh intervention")
        if not any(tape["raw_model_actions"][target] == 1
                   for tape in intervention_tapes for target in tape["intervened"]):
            raise RuntimeError(f"seed {seed} gate path has no actual greedy refresh SEND suppressed")
        if held["spent_bytes"] >= original["spent_bytes"]:
            raise RuntimeError(f"seed {seed} refresh-held did not reduce actual billed bytes")
        if any(not bool(emission[5]) for emission in held["emission_sequence"]):
            raise RuntimeError(f"seed {seed} held run contains a real refresh emission")
        if any(
            tape["features"][i][5] == 1.0 and target in tape["intervened"]
            for tape in tapes for i, target in enumerate(tape["targets"])
        ):
            raise RuntimeError(f"seed {seed} first-emission candidate was intervened")

        first = min(tape["step"] for tape in intervention_tapes)
        base_tape = tape_by_step(original)
        held_tape = tape_by_step(held)
        common_steps = sorted(step for step in held_tape if step < first)
        for step in common_steps:
            if step not in base_tape:
                raise RuntimeError(f"baseline tape lacks pre-intervention step {step}")
            left, right = base_tape[step], held_tape[step]
            for key in ("targets", "features", "actions", "shadow_budget_before",
                        "shadow_budget_after"):
                if left.get(key) != right.get(key):
                    raise RuntimeError(
                        f"seed {seed} execution prefix first differs at step {step} ({key})"
                    )
        if first not in base_tape or base_tape[first]["targets"] != held_tape[first]["targets"] \
                or base_tape[first]["features"] != held_tape[first]["features"]:
            raise RuntimeError(f"seed {seed} first-intervention observation changed")
        for target in intervention_tapes[0]["intervened"]:
            if target in base_tape[first]["free"] and \
                    base_tape[first]["actions"][target] != \
                    intervention_tapes[0]["raw_model_actions"][target]:
                raise RuntimeError(f"seed {seed} raw greedy action differs before intervention")
        orig_sends = [item for item in original["emission_sequence"] if item[0] < first]
        held_sends = [item for item in held["emission_sequence"] if item[0] < first]
        if orig_sends != held_sends:
            raise RuntimeError(f"seed {seed} transmissions differ before first intervention")
        if lifecycle_prefix(original, first) != lifecycle_prefix(held, first):
            raise RuntimeError(f"seed {seed} lifecycle diverges before first intervention")
        held_details[str(seed)] = {
            "intervened_decision_steps": len(intervention_tapes),
            "intervened_candidate_count": sum(len(tape["intervened"]) for tape in tapes),
            "raw_model_send_recommendations_on_refresh_candidates": sum(
                1 for tape in tapes for target in tape["intervened"]
                if tape["raw_model_actions"][target] == 1
            ),
            "original_spent_bytes": original["spent_bytes"],
            "refresh_held_spent_bytes": held["spent_bytes"],
            "bytes_reduced": int(original["spent_bytes"]) - int(held["spent_bytes"]),
            "held_refresh_emissions": sum(1 for item in held["emission_sequence"]
                                          if not bool(item[5])),
            "first_intervention_step": first,
            "common_prefix_decision_steps_checked": len(common_steps),
            "effective_budget_order": validation,
        }

    original22, held22 = rows["s22_original_greedy"], rows["s22_refresh_held"]
    s22_tapes = read_jsonl(Path(held22["artefacts"]["tape"]))
    if any(tape["intervened"] for tape in s22_tapes):
        raise RuntimeError("seed22 negative-control run unexpectedly intervened")
    s22_fields = (
        "behaviour_sha256", "spent_bytes", "emissions", "on_time", "completed",
        "emission_sequence", "refusal_sequence", "cost_stream", "task_stream",
    )
    if any(original22[key] != held22[key] for key in s22_fields):
        raise RuntimeError("seed22 refresh-held negative control changed behavior")

    gate = {
        "schema": "round39-first-scene-gate/1",
        "scene": scene,
        "all_passed": True,
        "round37_reproduction": round37_reproduction,
        "high_refresh_seed_gates": held_details,
        "seed22_negative_control": {
            "passed": True,
            "intervention_candidates": 0,
            "behavior_task_emission_and_byte_fields_identical": True,
        },
        "first_emission_policy_not_overridden": True,
        "cost_task_and_emission_closure": True,
        "checkpoint_bytes_parameters_and_post_run_hashes_unchanged": True,
        "first_intervention_prefix_identical": True,
        "production_modules_modified": [],
        "optimizer_steps": 0,
    }
    atomic_json(HERE / "FIRST_SCENE_GATE.json", gate)
    return gate


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    take_claim(job)
    try:
        row = run_arm(job["arm"], int(job["scene"]))
    except Exception as error:
        add_attempt(job, status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    add_attempt(job, status="completed", row=row)
    print(json.dumps({"completed": job, "starts": v2_common.ledger_total(),
                      "on_time": row["on_time"], "spent_bytes": row["spent_bytes"]},
                     ensure_ascii=False), flush=True)
    return row


def run_all() -> None:
    prelaunch_commit = assert_prelaunch_commit()
    lock_path = HERE / ".runner.lock"
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("another ROUND39 runner holds the exclusive launch lock") from error
    with os.fdopen(lock_fd, "w", encoding="utf-8") as sink:
        sink.write(json.dumps({"pid": os.getpid(), "head": prelaunch_commit}) + "\n")

    try:
        configure_output()
        book = read_json(HERE / "LAUNCH_BOOK.json")
        if book["starts"] != 0 or book["attempts"]:
            raise RuntimeError("launch book was already used; this runner does not resume implicitly")
        book["status"] = "running_gate"
        book["prelaunch_commit"] = prelaunch_commit
        atomic_json(HERE / "LAUNCH_BOOK.json", book)

        all_jobs = jobs()
        first_scene_jobs = [job for job in all_jobs if job["scene"] == 1600000]
        rest_jobs = [job for job in all_jobs if job["scene"] != 1600000]
        for job in first_scene_jobs:
            run_one(job)
        gate = validate_gate()
        print(json.dumps({"first_scene_gate": "PASS", "gate": gate}, ensure_ascii=False),
              flush=True)
        book = read_json(HERE / "LAUNCH_BOOK.json")
        book["status"] = "full_matrix_running"
        atomic_json(HERE / "LAUNCH_BOOK.json", book)

        for job in rest_jobs:
            run_one(job)
        book = read_json(HERE / "LAUNCH_BOOK.json")
        if book["starts"] != 144 or book["completed"] != 144 or book["technical_failures"]:
            raise RuntimeError(
                f"normal matrix ledger does not close: starts={book['starts']} "
                f"completed={book['completed']} failures={book['technical_failures']}"
            )
        if v2_common.ledger_total() > 153:
            raise RuntimeError("the real Stage C launch ledger exceeded the hard cap")
        book["status"] = "complete"
        book["normal_cells_completed"] = 144
        book["retries_used"] = 0
        book["real_start_ledger"] = v2_common.ledger_report()
        atomic_json(HERE / "LAUNCH_BOOK.json", book)
        print(json.dumps({"matrix": "COMPLETE", "starts": book["starts"],
                          "completed": book["completed"], "retries_used": 0,
                          "optimizer_steps": 0}, ensure_ascii=False), flush=True)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true",
                        help="after prelaunch freeze, run gate then all remaining cells")
    args = parser.parse_args()
    if not args.run:
        parser.error("the environment runner requires explicit --run")
    run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
