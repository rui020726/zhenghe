"""Private R58 eval booking bridge; never modifies the archived R44/R45 modules."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

FAILED_KEY = "model_eval_new_s11_greedy_scene1700000"
FAILED_CLAIM_SHA256 = "d8dd8a7e91e77e6e3c02836be0b13103a8fbe5c5192e1e15ae10b9f5906f983c"
FAILED_SLOT_SHA256 = "88f349363eea12f9aab1d0677d81f2a1c330a8dc1abe485d4b982c2361c48a87"


def _closure(fn: Any) -> dict[str, Any]:
    return {name: cell.cell_contents for name, cell in
            zip(fn.__code__.co_freevars, fn.__closure__ or (), strict=True)}


def _verify_failed_first_eval(raw_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    path = raw_root / "launch_claims" / FAILED_KEY / "attempt_0001.json"
    slot = raw_root / "launch_slots/normal/0385.json"
    if not path.is_file() or not slot.is_file():
        raise RuntimeError("first failed evaluation claim/slot is missing")
    if (hashlib.sha256(path.read_bytes()).hexdigest() != FAILED_CLAIM_SHA256
            or hashlib.sha256(slot.read_bytes()).hexdigest() != FAILED_SLOT_SHA256):
        raise RuntimeError("first failed evaluation original bytes changed")
    claim = json.loads(path.read_text(encoding="utf-8"))
    booked = json.loads(slot.read_text(encoding="utf-8"))
    matching = [job for job in plan["jobs"] if job["logical_job"] == FAILED_KEY]
    if (len(matching) != 1 or matching[0]["normal_slot"] != 385
            or claim.get("logical_job") != FAILED_KEY
            or claim.get("status") != "failed_technical"
            or int(claim.get("attempt", -1)) != 1
            or claim.get("category") != "normal"
            or int(claim.get("launch_slot", -1)) != 385
            or int(claim.get("environment_constructor_calls", -1)) != 1
            or int(claim.get("run_return_calls", -1)) != 1
            or int(claim.get("completed_steps", -1)) != 1000
            or claim.get("round45_artifact_receipt") is not None
            or "artifact receipt key is outside the frozen matrix" not in str(claim.get("detail", ""))
            or booked.get("logical_job") != FAILED_KEY
            or int(booked.get("attempt", -1)) != 1
            or int(booked.get("slot", -1)) != 385
            or booked.get("category") != "normal"):
        raise RuntimeError("first evaluation failure differs from registered booking defect")
    retry_path = raw_root / "launch_claims" / FAILED_KEY / "attempt_0002.json"
    retry_slot = raw_root / "launch_slots/reserve/0004.json"
    if retry_path.exists() or retry_slot.exists():
        if not retry_path.is_file() or not retry_slot.is_file():
            raise RuntimeError("first evaluation reserve retry claim/slot is incomplete")
        retry = json.loads(retry_path.read_text(encoding="utf-8"))
        slot4 = json.loads(retry_slot.read_text(encoding="utf-8"))
        if (retry.get("logical_job") != FAILED_KEY
                or retry.get("status") != "complete"
                or retry.get("category") != "reserve"
                or int(retry.get("launch_slot", -1)) != 4
                or int(retry.get("attempt", -1)) != 2
                or not isinstance(retry.get("round45_artifact_receipt"), dict)
                or slot4.get("logical_job") != FAILED_KEY
                or slot4.get("category") != "reserve"
                or int(slot4.get("slot", -1)) != 4
                or int(slot4.get("attempt", -1)) != 2):
            raise RuntimeError("first evaluation reserve retry is not a complete qualified task")
        retry_status = "complete"
    else:
        retry_status = "pending"
    return {"claim_path": str(path.resolve()), "slot_path": str(slot.resolve()),
            "logical_job": FAILED_KEY, "next_attempt": 2,
            "expected_reserve_slot": 4, "retry_status": retry_status}


def install(runtime: dict[str, Any], r45: Any, plan: dict[str, Any],
            *, raw_root: Path) -> dict[str, Any]:
    """Route only frozen R58 eval cells around R44's archived 16-scene receipt.

    The actual row booker and R45 hash/companion receipt remain active. Only
    the one registered failed first cell bypasses R45's pre-completion retry
    restriction, consuming its next attempt under the existing reserve writer.
    """
    run, runner = runtime["r42_run"], runtime["runner"]
    r40 = runtime["r40_common"]
    old_booked, old_execute = runner._booked, run._execute_cell
    old_completed_row = r45._completed_row
    if (getattr(old_booked, "__name__", "") != "booked"
            or Path(old_booked.__code__.co_filename).resolve() != Path(r45.__file__).resolve()
            or getattr(old_execute, "__name__", "") != "round58_attempt3_execute_wrapper"):
        raise RuntimeError("V35 requires the actual R45 booker and R58 attempt3 execute seam")
    original_booked = getattr(runner, "_round44_original_booked", None)
    if not callable(original_booked):
        raise RuntimeError("V35 cannot locate original row booker beneath R44")
    frozen_jobs = {job["logical_job"]: job for job in plan["jobs"]
                   if job["scope"].startswith(("model_eval_", "rule_eval_"))}
    if len(frozen_jobs) != 640:
        raise RuntimeError("V35 evaluation plan is not 640 unique keys")
    if Path(r45.RUN_DIR).resolve() != Path(runtime["common"].OUTPUT_DIR).resolve():
        raise RuntimeError("V35 R45 receipt source root differs from actual artifact output")
    execute_cells = _closure(old_execute)
    base_execute = execute_cells.get("base_execute")
    if not callable(base_execute) or not callable(r45._record_artifact_receipt):
        raise RuntimeError("V35 cannot locate original actual execution/receipt functions")
    failure = _verify_failed_first_eval(Path(raw_root), plan)

    def v35_completed_row(context: dict[str, Any], attempts: list[tuple[Path, dict[str, Any]]],
                          *, collector=None, runner_kwargs=None, identity=None):
        logical = str(context.get("logical_job", ""))
        job = frozen_jobs.get(logical)
        complete = [(path, claim) for path, claim in attempts
                    if claim.get("status") == "complete"]
        if job is None or not complete:
            return old_completed_row(context, attempts, collector=collector,
                                     runner_kwargs=runner_kwargs, identity=identity)
        if (len(complete) != 1 or attempts[-1] != complete[0]
                or any(claim.get("status") not in ("complete", "failed_technical")
                       for _, claim in attempts)):
            raise RuntimeError("V35 completed evaluation has ambiguous attempts")
        claim_path, claim = complete[0]
        attempt = int(claim.get("attempt", -1))
        category = str(claim.get("category", ""))
        slot = int(claim.get("launch_slot", -1))
        expected_category = "normal" if attempt == 1 else "reserve"
        if (attempt not in (1, 2) or category != expected_category
                or (attempt == 1 and slot != int(job["normal_slot"]))
                or claim.get("logical_job") != logical
                or claim.get("round") != 45
                or claim.get("round45_key") != r45._serializable_key(context)
                or claim.get("training_version") != r45.RUNTIME["r42_versions"].TRAINING_VERSION
                or claim.get("runner_version") != r45.RUNNER_VERSION
                or claim.get("comm_action_semantics") != "one-shot-send-skip/1"):
            raise RuntimeError("V35 completed evaluation claim identity mismatch")
        slot_path = Path(raw_root) / "launch_slots" / category / f"{slot:04d}.json"
        slot_row = json.loads(slot_path.read_text(encoding="utf-8")) if slot_path.is_file() else {}
        if (slot_row.get("logical_job") != logical
                or int(slot_row.get("attempt", -1)) != attempt
                or int(slot_row.get("slot", -1)) != slot
                or slot_row.get("category") != category):
            raise RuntimeError("V35 completed evaluation slot identity mismatch")
        if any(int(claim.get(name, -1)) != value for name, value in {
                "environment_constructor_calls": 1, "run_invocation_calls": 1,
                "joint_run_method_calls": 1, "run_return_calls": 1,
                "run_failure_calls": 0, "completed_steps": 1000}.items()):
            raise RuntimeError("V35 completed evaluation lacks a full real run")
        artifacts = claim.get("round45_artifact_paths")
        row_path = Path(artifacts.get("row", "")) if isinstance(artifacts, dict) else Path()
        if (not row_path.is_file()
                or Path(r45.RUN_DIR).resolve() not in row_path.resolve().parents):
            raise RuntimeError("V35 completed evaluation row is outside its private root")
        row = json.loads(row_path.read_text(encoding="utf-8"))
        if (row.get("kind") != "eval"
                or int(row.get("attempt", -1)) != attempt
                or int(row.get("scenario_seed", -1)) != int(job["scene"])
                or int(row.get("steps", -1)) != 1000
                or row.get("training_version") != claim.get("training_version")
                or row.get("runner_version") != claim.get("runner_version")):
            raise RuntimeError("V35 completed evaluation row identity mismatch")
        if job["scope"].startswith("model_eval_"):
            expected_arm = "NEW" if job["scope"] == "model_eval_r58" else job["scope"].removeprefix("model_eval_").upper()
            if (row.get("model_arm") != expected_arm
                    or int(row.get("training_seed", -1)) != int(job["seed"])
                    or row.get("eval_mode") != job["mode"]):
                raise RuntimeError("V35 completed model evaluation key mismatch")
            expected_param = (identity or (runner_kwargs or {}).get("checkpoint_load_identity") or {}).get(
                "parameter_identity", {}).get("parameter_sha256")
            if (not expected_param or row.get("checkpoint_parameter_identity", {}).get(
                    "parameter_sha256") != expected_param):
                raise RuntimeError("V35 completed evaluation uses a different fixed model")
        elif row.get("model_arm") != job["rule_arm"].upper() + "_RULE":
            raise RuntimeError("V35 completed rule evaluation key mismatch")
        r45._verify_receipt_artifacts(claim, row)
        if claim.get("round44_eval_artifact_receipt") is not None:
            raise RuntimeError("V35 completed new-scene cell unexpectedly has an R44 receipt")
        if collector is not None:
            collector.records = r45._load_jsonl(Path(row["artefacts"]["tape"]))
        return row

    def v35_booked(label: str, kind: str, row: dict[str, Any], ledger: Any) -> None:
        if kind != "eval":
            return old_booked(label, kind, row, ledger)
        active = run._ACTIVE_CLAIM
        if active is None:
            raise RuntimeError("V35 eval booking has no active atomic claim")
        claim = json.loads(Path(active).read_text(encoding="utf-8"))
        logical = str(claim.get("logical_job", ""))
        job = frozen_jobs.get(logical)
        if (job is None or int(row.get("scenario_seed", -1)) != int(job["scene"])
                or row.get("kind") != "eval"
                or int(row.get("attempt", -1)) != int(claim.get("attempt", -2))):
            raise RuntimeError("V35 eval row/claim differs from frozen job")
        if job["scope"].startswith("model_eval_"):
            if (int(row.get("training_seed", -1)) != int(job["seed"])
                    or row.get("eval_mode") != job["mode"]
                    or row.get("model_arm") != job["scope"].removeprefix("model_eval_").upper()):
                # The R58 arm uses NEW rather than R58 as its row label.
                if not (job["scope"] == "model_eval_r58" and row.get("model_arm") == "NEW"
                        and int(row.get("training_seed", -1)) == int(job["seed"])
                        and row.get("eval_mode") == job["mode"]):
                    raise RuntimeError("V35 model eval role/seed/mode differs from frozen job")
        elif row.get("model_arm") != job["rule_arm"].upper() + "_RULE":
            raise RuntimeError("V35 rule eval arm differs from frozen job")
        # R44's 1301000..1301015 receipt is intentionally inapplicable here.
        # Retain the actual row writer and R45's full artifact hash validation.
        r45._record_artifact_receipt(kind, row, Path(active), original_booked, label, ledger)

    def v35_execute(label: str, logical_job: str, kind: str, scenario: int,
                    metadata: dict[str, Any], *, collector=None,
                    runner_kwargs=None, row_updates=None, identity=None):
        if logical_job != FAILED_KEY:
            return old_execute(label, logical_job, kind, scenario, metadata,
                               collector=collector, runner_kwargs=runner_kwargs,
                               row_updates=row_updates, identity=identity)
        registered = _verify_failed_first_eval(Path(raw_root), plan)
        if registered["retry_status"] == "complete":
            return old_execute(label, logical_job, kind, scenario, metadata,
                               collector=collector, runner_kwargs=runner_kwargs,
                               row_updates=row_updates, identity=identity)
        if kind != "eval" or int(scenario) != 1700000:
            raise RuntimeError("V35 retry requested outside the registered first evaluation")
        context = r45.CURRENT_CONTEXT.get()
        if context is None:
            context = r45._context_for_cell(kind, logical_job, int(scenario), metadata,
                                            dict(runner_kwargs or {}))
        context = dict(context)
        context.update({"logical_job": logical_job, "kind": kind,
                        "scenario_seed": int(scenario),
                        "tag": str((runner_kwargs or {}).get("tag", context.get("tag", "")))})
        r45._planned_key(context)
        if r40.next_attempt(logical_job) != registered["next_attempt"]:
            raise RuntimeError("V35 retry cursor is not attempt2")
        token = r45.CURRENT_CONTEXT.set(context)
        try:
            return base_execute(label, logical_job, kind, int(scenario), metadata,
                                collector=collector, runner_kwargs=runner_kwargs,
                                row_updates=row_updates, identity=identity)
        finally:
            r45.CURRENT_CONTEXT.reset(token)

    v35_booked.__name__ = "round58_v35_eval_booked"
    v35_execute.__name__ = "round58_v35_eval_execute"
    runner._booked = v35_booked
    run._execute_cell = v35_execute
    r45._completed_row = v35_completed_row

    def undo() -> None:
        if runner._booked is v35_booked:
            runner._booked = old_booked
        if run._execute_cell is v35_execute:
            run._execute_cell = old_execute
        if r45._completed_row is v35_completed_row:
            r45._completed_row = old_completed_row

    return {"kind": "round58-v35-frozen-eval-booking/1", "status": "installed",
            "registered_failed_retry": failure, "plan_eval_keys": len(frozen_jobs),
            "row_writer": str(Path(original_booked.__code__.co_filename).resolve()),
            "r45_receipt_source": str(Path(r45.__file__).resolve()),
            "completed_row_route": "R45_four_artifact_receipt_and_frozen_640_key_matrix",
            "undo": undo}
