"""Phase orchestration for the convergence stage: train until a plateau, in finite stages.

One phase per invocation; nothing chains, nothing self-starts and nothing resumes itself. There
is no retry and no slot substitution. The first failure writes a persistent stop marker before
any further slot can be claimed, cancels futures that have not started, and lets already-claimed
worlds finish and account for themselves.

Stage 2 is training only after full parent admission: six runs continue from milestone_1320
at global episode 1320 / update 330 and stop at cap 2040. Evaluation and finalisation are later,
separately frozen phases; in this stage they refuse with an explicit non-zero status instead of
running a half-trained plan.

Reward for the plateau criterion is the recorded on-time task count over the recorded generated
goal count (always 24). The criterion itself lives in convergence.py and is pure arithmetic.

Receipts: each phase writes an exclusive started receipt and an exclusive terminal (or error)
receipt with the same receipt_id, pid, ppid, argv, identity hash and slot-plan hash. The
process-level OS exit code is proven by the runner agent's external tool receipt; this module
never fills an outer code.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
import unittest
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from . import compact, constants as C, convergence, episode, guard, objective, parent_admission, trainer, world
from .ledger import LedgerGate, SlotLedger
from .paths import BaseLayout, Layout

GATE_SIGNAL = "GATE_FIRST_UPDATE.json"
GATE_TIMEOUT_SECONDS = 4 * 3600
STOP_SIGNAL = "GLOBAL_STOP.json"
PLAN_SCHEMA = "stage5b-comm-convergence-slot-plan/1"
FREEZE_SCHEMA = "stage5b-convergence-base-model-freeze/1"
PREFLIGHT_SCHEMA = "stage5b-convergence-resume-preflight/1"


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _worker_init():
    """One torch/numpy thread per worker, set before either library is imported."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _assert_single_thread(torch_module=None):
    """The resumed training must see the parent single-thread torch configuration."""
    if torch_module is None:
        import torch as torch_module
    threads = int(torch_module.get_num_threads())
    if threads != 1:
        raise RuntimeError("this stage requires one torch thread, found %d" % (threads,))
    return threads


def _phase_import_bootstrap(layout, *, init=None, bootstrap=None, thread_guard=None):
    """Thread environment, then the frozen import closure, then the one-thread assertion.

    Import only: no world, no controller, no checkpoint and no forward pass. The frozen
    episode.validate_runtime_imports binds the authority common and runs the native PILOC
    preflight, so a pure import defect fails before a slot is claimed.
    """
    (init or _worker_init)()
    (bootstrap or episode.validate_runtime_imports)(layout)
    return (thread_guard or _assert_single_thread)()


def _identity_sha(layout):
    return guard.sha256_file(layout.identity)


def _plan_sha(layout):
    return guard.sha256_file(layout.slot_plan) if Path(layout.slot_plan).is_file() else None


def _write_receipt(layout, phase, stage, receipt_id, code, detail):
    directory = Path(layout.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("%s_%s_%s.json" % (phase, stage, receipt_id))
    payload = {
        "schema": "stage5b-phase-receipt/1",
        "phase": phase,
        "stage": stage,
        "receipt_id": receipt_id,
        "argv": list(sys_argv()),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "identity_sha256": (_identity_sha(layout) if Path(layout.identity).is_file() else None),
        "slot_plan_sha256": _plan_sha(layout),
        "recorded_utc": _utc(),
        "returncode": (None if code is None else int(code)),
        "detail": detail or {},
        "outer_exit_code": None,
        "outer_exit_status": "UNKNOWN",
        "note": ("the outer tool exit code is recorded by the runner agent in a separate raw "
                 "receipt; this process never fills it"),
    }
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return path


def sys_argv():
    import sys
    return sys.argv


def _new_receipt_id(phase):
    return "%s-%d-%d" % (phase, os.getpid(), time.time_ns())


def stop_path(layout):
    return Path(layout.ledger) / STOP_SIGNAL


def gate_signal_path(layout):
    return Path(layout.ledger) / GATE_SIGNAL


def _budget_evidence():
    """The failed world's budget-authority evidence; it never replaces the original error."""
    try:
        return world.budget_authority_evidence()
    except Exception:  # pragma: no cover - defensive only
        return {"status": "UNKNOWN_EVIDENCE_ERROR"}


def request_stop(layout, reason, detail=None):
    """Persist the stop marker before any further slot may be claimed."""
    return LedgerGate(layout).request_stop(reason, detail=detail)


def stopped(layout):
    return stop_path(layout).exists()


def _record_abort(layout, cancelled):
    if not cancelled:
        return None
    path = Path(layout.ledger) / ("PHASE_ABORT_%d.json" % time.time_ns())
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "stage5b-phase-abort/1", "recorded_utc": _utc(),
               "pid": os.getpid(), "cancelled_or_unstarted": list(cancelled),
               "gate_signal_present": gate_signal_path(layout).exists(),
               "experiment_state": "STAGE_TERMINAL_UNRECOVERABLE",
               "note": ("unstarted slots are not claimed, no cancel event is fabricated and "
                        "no resume of this stage exists")}
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except FileExistsError:
        pass
    return path


def _pool_process_snapshot(executor):
    """Real multiprocessing.Process objects, taken before shutdown."""
    processes = getattr(executor, "_processes", None) or {}
    return list(processes.values())


def _record_pool_exits(layout, processes, phase):
    """Record real worker pids and exit codes; None stays UNKNOWN, never 0."""
    if not processes:
        payload = {"schema": "stage5b-pool-process-exits/1", "phase": phase,
                   "mode": "in_process",
                   "os_exit_code": "NOT_APPLICABLE_PHASE_OUTER_RECEIPT",
                   "note": "single-worker phases run in the parent; no separate worker OS code"}
    else:
        records = []
        for process in processes:
            code = process.exitcode
            records.append({"pid": process.pid,
                            "os_exit_code": (int(code) if isinstance(code, int) else "UNKNOWN"),
                            "note": "pool worker process, not a per-world exit code"})
        payload = {"schema": "stage5b-pool-process-exits/1", "phase": phase,
                   "mode": "process_pool", "workers": records}
    path = Path(layout.ledger) / ("POOL_EXITS_%d.json" % time.time_ns())
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except FileExistsError:
        pass
    return payload


def _task_slot(task):
    """Decode the normal stage or explicit recovery group payload."""
    if len(task) == 6:
        (_root, context_path, g3_report_path, world_authorization_path,
         arm, seed) = task
        if arm not in C.ARMS_TRAIN or type(seed) is not int \
                or int(seed) not in C.TRAIN_SEEDS:
            raise ValueError("unregistered recovery arm/seed")
        for value, label in ((context_path, "base recovery context"),
                             (g3_report_path, "G3 report"),
                             (world_authorization_path, "world authorization")):
            if not isinstance(value, str) or not value:
                raise ValueError(label + " path is missing")
        return "recovery:%s:%d" % (arm, int(seed))
    if len(task) == 4:
        raise ValueError("recovery training task lacks explicit G3/world-authorization refs")
    if len(task) != 3:
        raise ValueError("a training group payload has three or six entries")
    _root, arm, seed = task
    if arm not in C.ARMS_TRAIN:
        raise ValueError("unregistered training arm %r" % (arm,))
    if type(seed) is not int or int(seed) not in C.TRAIN_SEEDS:
        raise ValueError("unregistered training seed %r" % (seed,))
    return "train:%s:%d" % (arm, int(seed))


def _safe_task_slot(task):
    """A failure label must never replace the original exception."""
    try:
        return _task_slot(task)
    except Exception:
        return "INVALID_PAYLOAD:" + repr(task)


def _accepted_result(result):
    return isinstance(result, dict) and result.get("status") == "ok"


def _accepted_count(results):
    return sum(1 for item in results if _accepted_result(item))


def _require_world_result(result, slot):
    if not _accepted_result(result):
        raise RuntimeError("%s did not return an accepted status: %r" % (slot, result))
    return result


def _artifact_label(entry):
    return "s5b_train_%s_s%d_e%04d" % (entry["arm"], int(entry["seed"]),
                                       int(entry["episode"]))


def validate_artifact_paths(layout, plan):
    """Every planned world gets its own Windows-normalized artifact leaf."""
    from .world import artifact_directory
    paths = []
    for entry in plan:
        paths.append(str(artifact_directory(
            layout, _artifact_label(entry), artifact_namespace=entry["slot"])).casefold())
    if len(paths) != len(set(paths)):
        raise RuntimeError("Windows-normalized artifact paths are not unique")
    return paths


def plan_payload():
    """The exact bytes of the frozen launch plan."""
    return {
        "schema": PLAN_SCHEMA,
        "experiment_id": C.RUN_ID,
        "revision_id": C.REVISION_ID,
        "stage": C.STAGE_INDEX,
        "phase": "train",
        "world_cap": C.WORLD_CAP_TOTAL,
        "stage_start_episode": C.STAGE_START_EPISODE,
        "stage_end_episode": C.STAGE_END_EPISODE,
        "slots": C.slot_plan(),
    }


def plan_bytes():
    """One canonical serialisation, so the SHA written into the identity is the SHA on disk."""
    payload = plan_payload()
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return text.encode("utf-8")


PROTOCOL_CHECKS = (
    ("transport.budget_bytes", C.BUDGET_BYTES),
    ("transport.steps", C.STEPS),
    ("transport.subscribers", C.SUBSCRIBERS),
    ("transport.num_goals", C.NUM_GOALS),
    ("transport.b0_bytes", C.B0_BYTES),
    ("transport.feature_dim", C.FEATURE_DIM),
    ("training.stage_start_episode", C.STAGE_START_EPISODE),
    ("training.stage_end_episode", C.STAGE_END_EPISODE),
    ("training.episodes_per_update", C.EPISODES_PER_UPDATE),
    ("training.checkpoint_interval_episodes", C.CHECKPOINT_INTERVAL_EPISODES),
    ("convergence.window_episodes", C.CONVERGENCE_WINDOW_EPISODES),
    ("convergence.windows", C.CONVERGENCE_WINDOWS),
    ("convergence.interval_level", C.CONVERGENCE_INTERVAL_LEVEL),
    ("convergence.resamples", C.CONVERGENCE_BOOTSTRAP_REPS),
    ("convergence.pair_bound", C.CONVERGENCE_PAIR_BOUND),
    ("convergence.mean_range_bound", C.CONVERGENCE_MEAN_RANGE_BOUND),
    ("convergence.slope_bound", C.CONVERGENCE_SLOPE_BOUND),
    ("convergence.budget_range_bound", C.CONVERGENCE_BUDGET_RANGE_BOUND),
    ("convergence.budget_range_role", C.CONVERGENCE_BUDGET_RANGE_ROLE),
    ("convergence.first_check_episode", C.CONVERGENCE_FIRST_CHECK_EPISODE),
    ("convergence.check_step", C.CONVERGENCE_CHECK_STEP),
    ("convergence.required_consecutive", C.CONVERGENCE_REQUIRED_CONSECUTIVE),
    ("resources.stage_train_cap", C.STAGE_TRAIN_CAP),
    ("resources.world_cap_total", C.WORLD_CAP_TOTAL),
    ("resources.replay_worlds", C.REPLAY_WORLDS),
    ("resources.pacing_worlds", C.PACING_WORLDS),
    ("resources.eval_worlds_this_stage", C.NEW_EVAL_WORLDS),
    ("resources.eval_worlds_after_convergence", C.EVAL_WORLDS_AFTER_CONVERGENCE),
    ("resources.prior_approved_cumulative_cap", C.PRIOR_APPROVED_CUMULATIVE_CAP),
    ("resources.prior_recorded_actual_claims", C.PRIOR_RECORDED_ACTUAL_CLAIMS),
    ("resources.projected_actual_if_stage_full",
     C.PROJECTED_ACTUAL_CUMULATIVE_IF_STAGE_FULL),
    ("resources.cumulative_launch_slot_cap", C.APPROVED_CUMULATIVE_LAUNCH_CAP),
    ("resources.workers_max", C.WORKERS_MAX),
    ("resources.threads_per_worker", C.THREADS_PER_WORKER),
    ("history.episodes", list(C.PARENT_HISTORY_EPISODES)),
    ("history.sources", [dict(source) for source in C.HISTORY_SOURCES]),
    ("history.rows_pinned", C.HISTORY_ROWS_PINNED),
    ("revision.stage_index", C.STAGE_INDEX),
)


def _dig(protocol, dotted):
    node = protocol
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise RuntimeError("the pre-registration has no %r" % (dotted,))
        node = node[key]
    return node


def _assert_protocol_matches_constants(layout):
    """The protocol is the frozen statement; a constants/protocol disagreement stops prepare."""
    protocol = json.loads(Path(layout.preregistration).read_text(encoding="utf-8"))
    differences = []
    for dotted, expected in PROTOCOL_CHECKS:
        actual = _dig(protocol, dotted)
        if actual != expected:
            differences.append("%s: protocol %r != constants %r" % (dotted, actual, expected))
    for key, expected in (("experiment_id", C.RUN_ID),):
        if protocol.get(key) != expected:
            differences.append("%s: protocol %r != %r"
                               % (key, protocol.get(key), expected))
    revision = protocol.get("revision") or {}
    if revision.get("revision_id") != C.REVISION_ID:
        differences.append("revision.revision_id: %r != %r"
                           % (revision.get("revision_id"), C.REVISION_ID))
    if list(_dig(protocol, "training.scenes")) != list(C.TRAIN_SCENES):
        differences.append("training.scenes differ from the constants")
    if list(_dig(protocol, "training.seeds")) != list(C.TRAIN_SEEDS):
        differences.append("training.seeds differ from the constants")
    checkpoints = list(_dig(protocol, "training.checkpoint_episodes"))
    if checkpoints != list(C.STAGE_CHECKPOINT_EPISODES):
        differences.append("training.checkpoint_episodes %r != %r"
                           % (checkpoints, list(C.STAGE_CHECKPOINT_EPISODES)))
    checks = list(_dig(protocol, "convergence.check_episodes_this_stage"))
    if checks != list(C.convergence_check_episodes()):
        differences.append("convergence.check_episodes_this_stage %r != %r"
                           % (checks, list(C.convergence_check_episodes())))
    roots = list(_dig(protocol, "convergence.ancestor_certificate_roots"))
    expected_roots = [{"root": str(Path(group["root"])),
                       "experiment_id": str(group["experiment_id"])}
                      for group in C.CONVERGENCE_ANCESTOR_ROOTS]
    if roots != expected_roots:
        differences.append("convergence.ancestor_certificate_roots do not match constants")
    files = list(_dig(protocol, "convergence.ancestor_certificate_files"))
    expected_files = []
    for group in C.CONVERGENCE_ANCESTOR_ROOTS:
        for item in group.get("certificates", ()):
            expected_files.append({"root": str(Path(group["root"])),
                                   "experiment_id": str(group["experiment_id"]),
                                   "arm": str(item["arm"]), "seed": int(item["seed"]),
                                   "path": str(item["path"]), "bytes": int(item["bytes"]),
                                   "sha256": str(item["sha256"])})
    if files != expected_files:
        differences.append("convergence.ancestor_certificate_files do not match constants")
    if differences:
        raise RuntimeError("pre-registration and constants disagree: " + "; ".join(differences))
    return {"checked": len(PROTOCOL_CHECKS) + 5, "protocol_sha256":
            guard.sha256_file(layout.preregistration)}


def _assert_clean_start(layout):
    """No world of this stage may exist before the plan and identity are frozen."""
    offenders = []
    for folder in (Path(layout.ledger) / "claims", Path(layout.ledger) / "results",
                   Path(layout.rows, "train"), layout.checkpoints, layout.telemetry,
                   layout.convergence):
        path = Path(folder)
        if path.is_dir():
            found = sorted(item.name for item in path.rglob("*") if item.is_file())
            offenders.extend("%s/%s" % (path, name) for name in found[:5])
            if found:
                offenders.append("%s: %d file(s)" % (path, len(found)))
    for name in (STOP_SIGNAL, GATE_SIGNAL):
        if (Path(layout.ledger) / name).exists():
            offenders.append(str(Path(layout.ledger) / name))
    if offenders:
        raise RuntimeError("this stage is not a clean start: " + "; ".join(offenders))
    return True


def prepare(layout):
    """Freeze constants, plan and identity before any world exists.

    Order matters: protocol/constant agreement, then the read-only ancestor pins, then the
    in-memory plan's uniqueness and counts, then the plan's own bytes on disk, then the
    identity. Everything before the plan write is read-only.
    """
    guard.assert_experiment_root(layout)
    protocol = _assert_protocol_matches_constants(layout)
    _assert_clean_start(layout)
    base = guard.assert_base_pins(layout)
    plan = C.slot_plan()
    if len(plan) != C.WORLD_CAP_TOTAL:
        raise RuntimeError("the plan holds %d slots, expected %d"
                           % (len(plan), C.WORLD_CAP_TOTAL))
    if len({entry["slot"] for entry in plan}) != len(plan):
        raise RuntimeError("the plan contains duplicate slots")
    if Path(layout.slot_plan).exists():
        raise RuntimeError("the slot plan already exists: %s" % (layout.slot_plan,))
    if Path(layout.identity).exists():
        raise RuntimeError("the identity already exists: %s" % (layout.identity,))
    Path(layout.slot_plan).parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = plan_bytes()
    with Path(layout.slot_plan).open("xb") as handle:
        handle.write(payload_bytes)
    on_disk = Path(layout.slot_plan).read_bytes()
    if on_disk != payload_bytes:
        raise RuntimeError("the slot plan on disk is not the bytes that were hashed")
    plan_sha = hashlib.sha256(on_disk).hexdigest()
    identity = guard.freeze_identity(layout)
    if identity["slot_plan"]["sha256"] != plan_sha:
        raise RuntimeError("the identity records slot plan %s, disk holds %s"
                           % (identity["slot_plan"]["sha256"], plan_sha))
    verified = guard.verify_identity(layout)
    return {"protocol": protocol, "base": base, "plan_sha256": plan_sha,
            "plan_bytes_on_disk": len(on_disk), "slots": len(plan),
            "identity_sha256": guard.sha256_file(layout.identity),
            "identity": verified,
            "stages": {"start_episode": C.STAGE_START_EPISODE,
                       "end_episode": C.STAGE_END_EPISODE,
                       "new_episodes_per_run": C.NEW_EPISODES_PER_STAGE},
            "convergence_check_episodes": list(C.convergence_check_episodes()),
            "stage_verdict_if_no_certificate": C.STAGE_VERDICT_IF_NO_CERTIFICATE}


def puretests(layout):
    """Run the pure test suite inside this root; never touches a world or a checkpoint."""
    started = time.time()
    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(layout.root) / "tests"),
                            top_level_dir=str(Path(layout.root)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return {"tests": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped),
            "was_successful": bool(result.wasSuccessful()),
            "failure_names": [str(case) for case, _tb in result.failures],
            "error_names": [str(case) for case, _tb in result.errors],
            "seconds": round(time.time() - started, 2)}


def _verify_frozen_file(entry):
    """One frozen base checkpoint, byte for byte; drift is a hard error."""
    path = Path(entry["path"])
    if not path.is_file():
        raise RuntimeError("frozen base checkpoint is missing: %s" % (path,))
    actual_bytes = int(path.stat().st_size)
    actual_sha = guard.sha256_file(path)
    if actual_bytes != int(entry["bytes"]) or actual_sha != entry["sha256"]:
        raise guard.FreezeDrift("base checkpoint moved: %s (%d B, %s) != (%d B, %s)"
                                % (path, actual_bytes, actual_sha, entry["bytes"],
                                   entry["sha256"]))
    return path


def _base_freeze_entries(layout, episode_number=None):
    path = Path(layout.base_model_freeze)
    if not path.is_file():
        raise RuntimeError("the base model freeze has not run: %s" % (path,))
    payload = compact.read_json(path)
    if payload.get("status") != "FROZEN":
        raise RuntimeError("%s records %r, not FROZEN" % (path, payload.get("status")))
    entries = payload.get("entries") or []
    if len(entries) != len(C.ARMS_TRAIN) * len(C.TRAIN_SEEDS):
        raise RuntimeError("the base freeze holds %d entries, expected %d"
                           % (len(entries), len(C.ARMS_TRAIN) * len(C.TRAIN_SEEDS)))
    if episode_number is not None:
        entries = [item for item in entries
                   if int(item["episode"]) == int(episode_number)]
    return payload, entries


def freeze_base_models(layout):
    """After real parent admission, pin the six actual milestone_1320 files; no model load."""
    guard.assert_experiment_root(layout)
    parent_state = guard.assert_parent_admission(layout)
    layout.ensure()
    require_frozen(layout)
    if stopped(layout):
        raise RuntimeError("GLOBAL_STOP is present; the base freeze refuses to start")
    if Path(layout.base_model_freeze).exists():
        raise RuntimeError("the base model freeze already exists: %s"
                           % (layout.base_model_freeze,))
    base = BaseLayout()
    entries = []
    for item in guard.base_checkpoint_pins(layout):
        path = base.checkpoint(item["arm"], item["seed"], item["name"])
        if not path.is_file():
            raise guard.FreezeDrift("base checkpoint is missing: %s" % (path,))
        actual_bytes = int(path.stat().st_size)
        actual_sha = guard.sha256_file(path)
        if actual_bytes != int(item["bytes"]) or actual_sha != item["sha256"]:
            raise guard.FreezeDrift("base checkpoint moved: %s (%d B, %s) != (%d B, %s)"
                                    % (path, actual_bytes, actual_sha, item["bytes"],
                                       item["sha256"]))
        for name, value in (("parameter_fingerprint", item["parameter_sha256"]),
                            ("actor_fingerprint", item["actor_sha256"]),
                            ("full_state_fingerprint", item["full_state_sha256"])):
            guard.assert_hex64("%s/%d %s" % (item["arm"], item["seed"], name), value)
        entries.append({"file_arm": item["arm"], "arm": item["arm"],
                        "training_seed": int(item["seed"]),
                        "episode": int(item["episode"]), "name": item["name"],
                        "path": str(path), "bytes": actual_bytes, "sha256": actual_sha,
                        "parameter_fingerprint": item["parameter_sha256"],
                        "actor_fingerprint": item["actor_sha256"],
                        "full_state_fingerprint": item["full_state_sha256"]})
    payload = {"schema": FREEZE_SCHEMA, "status": "FROZEN",
               "experiment_id": C.RUN_ID, "stage": C.STAGE_INDEX,
               "recorded_utc": _utc(),
               "base_experiment_id": C.BASE_EXPERIMENT_ID,
               "base_identity_sha256": guard.parent_identity_pin(layout)["sha256"],
               "parent_admission_evidence_sha256": parent_state["evidence_sha256"],
               "parent_provenance": parent_state,
               "checkpoint_episode": C.STAGE_START_EPISODE,
               "entries": entries,
               "worlds": 0, "models_loaded": 0,
               "note": ("file bytes and the declared restored-state fingerprints only; no "
                        "checkpoint is loaded and no world is built in this phase")}
    compact.write_json_exclusive(Path(layout.base_model_freeze), payload)
    return {"status": "FROZEN", "entries": len(entries),
            "path": str(layout.base_model_freeze),
            "sha256": guard.sha256_file(layout.base_model_freeze)}


def resume_preflight(layout):
    """Zero-world: restore all six states and prove they equal the parent fingerprints."""
    guard.assert_experiment_root(layout)
    guard.assert_parent_admission(layout)
    layout.ensure()
    require_frozen(layout)
    if stopped(layout):
        raise RuntimeError("GLOBAL_STOP is present; the resume preflight refuses to start")
    if Path(layout.resume_preflight_receipt).exists():
        raise RuntimeError("the resume preflight already ran: %s"
                           % (layout.resume_preflight_receipt,))
    _freeze, entries = _base_freeze_entries(layout, episode_number=C.STAGE_START_EPISODE)
    _phase_import_bootstrap(layout)
    results = []
    for arm in C.ARMS_TRAIN:
        for seed in C.TRAIN_SEEDS:
            matches = [item for item in entries
                       if item["file_arm"] == arm and int(item["training_seed"]) == int(seed)]
            if len(matches) != 1:
                raise RuntimeError("the base freeze holds %d checkpoints for %s/%d"
                                   % (len(matches), arm, int(seed)))
            entry = matches[0]
            path = _verify_frozen_file(entry)
            restored = trainer.load_base_checkpoint(
                path, arm=arm, seed=int(seed), episode=C.STAGE_START_EPISODE,
                base_identity_sha256=_freeze["base_identity_sha256"],
                expected_parameter_sha256=entry["parameter_fingerprint"],
                expected_actor_sha256=entry["actor_fingerprint"],
                expected_full_state_sha256=entry["full_state_fingerprint"])
            state = restored["state"]
            results.append({
                "arm": arm, "seed": int(seed), "path": str(path),
                "sha256": entry["sha256"], "bytes": int(entry["bytes"]),
                "payload_arm": restored["payload_arm"],
                "payload_seed": restored["payload_seed"],
                "payload_episode": restored["payload_episode"],
                "state_episode": int(state.get("episode", -1)),
                "next_episode_index": int(state.get("episode", -1)),
                "next_update_index": int(state.get("episode", -1)) // C.EPISODES_PER_UPDATE,
                "parameter_sha256": restored["parameters"]["parameter_sha256"],
                "actor_sha256": restored["full_state"]["actor"],
                "full_state_sha256": restored["full_state"]["sha256"],
                "rng_sha256": restored["full_state"]["rng"],
                "optimizer_sha256": restored["full_state"]["optimizer"],
                "optimizer_restored": True, "reseeded": False,
            })
    payload = {"schema": PREFLIGHT_SCHEMA, "status": "PASS",
               "experiment_id": C.RUN_ID, "stage": C.STAGE_INDEX,
               "recorded_utc": _utc(), "workers_in_phase": 1,
               "worlds": 0, "checkpoints_loaded": len(results),
               "models_constructed": len(results),
               "entries": results,
               "note": ("each state is restored under the parent identity and compared with "
                        "the pinned parameter, actor and full-state digests; the full-state "
                        "digest covers the optimizer and all three RNGs, so a reset would "
                        "fail here. No world is built and no forward pass is run.")}
    compact.write_json_exclusive(Path(layout.resume_preflight_receipt), payload)
    return {"status": "PASS", "restored": len(results),
            "path": str(layout.resume_preflight_receipt),
            "sha256": guard.sha256_file(layout.resume_preflight_receipt)}


def assert_eval_dispatch_contract():
    """Pre-claim contract for the later evaluation phase.

    Every declared converged-run arm must resolve to the learned branch; a suffix test would
    accept names such as half_trained, and the parent round died exactly on a name that the
    whitelist did not contain. Pure: no world, no model, no checkpoint.
    """
    seen = []
    for name in C.EVAL_LEARNING_ARMS:
        if episode.eval_arm_kind(name) != "learned":
            raise RuntimeError("%r does not dispatch to the learned branch" % (name,))
        seen.append(name)
    for name in ("untrained", "T1_trained", "T2_trained"):
        if episode.eval_arm_kind(name) != "learned":
            raise RuntimeError("%r does not dispatch to the learned branch" % (name,))
    for name in C.BASELINE_ARMS + C.PACING_ARMS:
        if episode.eval_arm_kind(name) != "rule":
            raise RuntimeError("%r does not dispatch to the rule branch" % (name,))
    for name in ("half_trained", "T1_m401", "T1_conv", "T9_conv_s91001"):
        try:
            episode.eval_arm_kind(name)
        except ValueError:
            continue
        raise RuntimeError("%r was accepted although it is not a registered arm" % (name,))
    plan = C.all_eval_plan()
    if len(plan) != C.EVAL_WORLDS_AFTER_CONVERGENCE:
        raise RuntimeError("the declared evaluation plan holds %d worlds" % (len(plan),))
    if len({entry["slot"] for entry in plan}) != len(plan):
        raise RuntimeError("the declared evaluation plan repeats a slot")
    for entry in plan:
        if entry["arm"] not in C.ARMS_TRAIN:
            raise RuntimeError("evaluation plan arm %r is not a training arm"
                               % (entry["arm"],))
        if int(entry["scene"]) not in C.EVAL_SCENES:
            raise RuntimeError("evaluation plan scene %r is not registered"
                               % (entry["scene"],))
    return {"learned_arms": len(seen), "eval_worlds": len(plan),
            "worlds": 0, "models_loaded": 0}


def _assert_stage_contract(layout):
    """Everything the stage depends on, checked before the first slot is claimed."""
    plan = C.slot_plan()
    groups = {}
    episodes = {}
    for entry in plan:
        slot = entry["slot"]
        expected = "train:%s:%d:%04d" % (entry["arm"], int(entry["seed"]),
                                         int(entry["episode"]))
        if slot != expected:
            raise RuntimeError("plan slot %r is not %r" % (slot, expected))
        if int(entry["scene"]) != int(C.scene_for(entry["seed"], entry["episode"])):
            raise RuntimeError("plan slot %r carries scene %r, expected %d"
                               % (slot, entry["scene"],
                                  C.scene_for(entry["seed"], entry["episode"])))
        if entry["phase"] != "train":
            raise RuntimeError("stage 2 may only plan training worlds, found %r"
                               % (entry["phase"],))
        key = (entry["arm"], int(entry["seed"]))
        groups[key] = groups.get(key, 0) + 1
        episodes.setdefault(key, []).append(int(entry["episode"]))
    expected_groups = {(arm, int(seed)) for arm in C.ARMS_TRAIN for seed in C.TRAIN_SEEDS}
    if set(groups) != expected_groups:
        raise RuntimeError("the plan does not cover exactly the six registered runs")
    for key, count in groups.items():
        if count != C.NEW_EPISODES_PER_STAGE:
            raise RuntimeError("%s/%d plans %d episodes, expected %d"
                               % (key[0], key[1], count, C.NEW_EPISODES_PER_STAGE))
        wanted = list(range(C.STAGE_START_EPISODE, C.STAGE_END_EPISODE))
        if sorted(episodes[key]) != wanted:
            raise RuntimeError("%s/%d does not plan contiguous episodes from %d"
                               % (key[0], key[1], C.STAGE_START_EPISODE))
    _freeze, entries = _base_freeze_entries(layout, episode_number=C.STAGE_START_EPISODE)
    selectors = {(item["file_arm"], int(item["training_seed"])) for item in entries}
    if selectors != expected_groups:
        raise RuntimeError("the frozen checkpoint selector does not cover the six runs")
    dispatch = assert_eval_dispatch_contract()
    validate_artifact_paths(layout, plan)
    return {"slots": len(plan), "groups": len(groups),
            "episodes_per_group": C.NEW_EPISODES_PER_STAGE,
            "base_selectors": len(selectors), "dispatch": dispatch}


def _checkpoint_path(layout, arm, seed, episode_number):
    return layout.checkpoint(arm, seed, episode_number)


def _read_row(path):
    payload = compact.read_json(path)
    if int(Path(path).stat().st_size) > C.FILE_SIZE_CAP_BYTES:
        raise RuntimeError("%s exceeds the single-file cap" % (path,))
    return payload


def _ancestor_row_path(arm, seed, index):
    """One explicit source among [0,200), [200,600), [600,1320)."""
    return C.history_row_path(arm, seed, index)


def build_series(layout, arm, seed, end_episode, updates_by_index):
    """Rows and updates a check at end_episode needs; missing rows are simply absent.

    The convergence module reports a missing episode as a failure, so this loader never fills
    a gap with a neighbour, an interpolated value or a zero.
    """
    start = convergence.history_needed(end_episode)
    if start < 0:
        start = 0
    rows = parent_admission.history_rows(layout, arm, seed, int(start),
                                         min(C.STAGE_START_EPISODE, int(end_episode)))
    for index in range(max(int(start), C.STAGE_START_EPISODE), int(end_episode)):
        path = Path(layout.train_row("train:%s:%d:%04d" % (arm, int(seed), index)))
        if path.is_file():
            rows[int(index)] = _read_row(path)
    updates = parent_admission.history_updates(layout, arm, seed)
    for index, item in (updates_by_index or {}).items():
        if index in updates:
            raise RuntimeError("historical/new optimizer update conflict at %d" % index)
        updates[index] = item
    return {"rows": rows, "updates": updates}


def _read_check_certificates(layout, arm, seed):
    directory = Path(layout.convergence)
    candidates = []
    if directory.is_dir():
        prefix = "CHECK_%s_s%d_e" % (arm, int(seed))
        candidates.extend(("current", path, C.RUN_ID) for path in
                          sorted(directory.glob(prefix + "*.json")))
    for entry in guard.convergence_ancestor_entries():
        if entry["arm"] == arm and int(entry["seed"]) == int(seed):
            candidates.append(("pinned ancestor", Path(entry["path"]),
                               entry["experiment_id"]))
    by_key = {}
    for source, path, expected_experiment_id in candidates:
        payload = compact.read_json(path)
        if payload.get("experiment_id") != expected_experiment_id:
            raise RuntimeError("%s certificate %s belongs to experiment %r, expected %r"
                               % (source, path, payload.get("experiment_id"),
                                  expected_experiment_id))
        if payload.get("arm") != arm or int(payload.get("seed", -1)) != int(seed):
            raise RuntimeError("%s certificate %s does not describe %s/%d"
                               % (source, path, arm, int(seed)))
        if "end_episode" not in payload:
            raise RuntimeError("%s certificate %s has no end_episode" % (source, path))
        end_episode = int(payload["end_episode"])
        if (end_episode < C.CONVERGENCE_FIRST_CHECK_EPISODE
                or (end_episode - C.CONVERGENCE_FIRST_CHECK_EPISODE) % C.CONVERGENCE_CHECK_STEP):
            raise RuntimeError("certificate is not on the absolute convergence check ladder")
        if (type(payload.get("passed")) is not bool
                or (payload.get("passed") and payload.get("status") != convergence.STATUS_PASS)
                or (not payload.get("passed") and payload.get("status") != convergence.STATUS_FAIL)):
            raise RuntimeError("certificate status/passed fields are inconsistent")
        key = (arm, int(seed), end_episode)
        normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"))
        if key in by_key:
            if by_key[key]["_normalized"] != normalized:
                raise RuntimeError("conflicting duplicate check certificate %r" % (key,))
            continue
        by_key[key] = {"end_episode": end_episode,
                       "passed": bool(payload.get("passed")),
                       "status": payload.get("status"),
                       "path": str(path), "source": source,
                       "_normalized": normalized}
    result = []
    for key in sorted(by_key, key=lambda item: (item[2], item[0], item[1])):
        item = dict(by_key[key])
        item.pop("_normalized")
        result.append(item)
    return result


def _certificate_requires_stop(certificate):
    """Corrupt/incomplete history is a hard stop; an ordinary failed gate is not."""
    return certificate.get("status") == convergence.STATUS_INVALID


def _validate_prefix_results(results, *, stage_start=None, stage_end=None,
                             check_episodes=None):
    """Validate per-run accepted prefixes and their sum against the finite stage cap."""
    start = int(C.STAGE_START_EPISODE if stage_start is None else stage_start)
    end = int(C.STAGE_END_EPISODE if stage_end is None else stage_end)
    checks = set(C.convergence_check_episodes(end if check_episodes is None else end)
                 if check_episodes is None else (int(value) for value in check_episodes))
    expected_runs = {(arm, int(seed)) for arm in C.ARMS_TRAIN for seed in C.TRAIN_SEEDS}
    seen = set()
    total = 0
    for result in results:
        if not _accepted_result(result):
            raise RuntimeError("training result is not accepted: %r" % (result,))
        slot = str(result.get("slot", ""))
        pieces = slot.split(":")
        if len(pieces) != 3 or pieces[0] != "train":
            raise RuntimeError("training result has an invalid slot: %r" % (slot,))
        key = (pieces[1], int(pieces[2]))
        if key not in expected_runs or key in seen:
            raise RuntimeError("training result repeats or invents run %r" % (key,))
        seen.add(key)
        episodes = int(result.get("episodes", -1))
        terminal = int(result.get("terminal_episode", -1))
        planned = end - start
        if episodes <= 0 or episodes > planned or episodes % C.EPISODES_PER_UPDATE:
            raise RuntimeError("%s accepted an invalid prefix of %d episodes" % (slot, episodes))
        if terminal != start + episodes:
            raise RuntimeError("%s terminal episode %d does not equal %d + %d"
                               % (slot, terminal, start, episodes))
        if result.get("converged"):
            if terminal not in checks:
                raise RuntimeError("%s stopped early at %d without a registered check"
                                   % (slot, terminal))
        elif episodes != planned:
            raise RuntimeError("%s has no certificate and stopped before the stage end"
                               % (slot,))
        total += episodes
    if seen != expected_runs:
        raise RuntimeError("training results do not cover exactly the six registered runs")
    if total > C.WORLD_CAP_TOTAL:
        raise RuntimeError("accepted training prefix %d exceeds cap %d"
                           % (total, C.WORLD_CAP_TOTAL))
    return {"runs": len(seen), "episodes": total,
            "unused_world_slots": C.WORLD_CAP_TOTAL - total}


def convergence_check_at(layout, arm, seed, end_episode, updates_by_index):
    """Evaluate one check, write its exclusive certificate and report whether it passed."""
    series = build_series(layout, arm, seed, end_episode, updates_by_index)
    certificate = convergence.evaluate(series, arm=arm, seed=int(seed),
                                       end_episode=int(end_episode))
    path = Path(layout.convergence_check(arm, seed, end_episode))
    path.parent.mkdir(parents=True, exist_ok=True)
    compact.write_json_exclusive(path, certificate)
    certificate["path"] = str(path)
    return certificate


def _train_group(payload):
    """One (arm, seed) run: resume at episode 1320 and stop at cap2040 or a valid second pass."""
    root, arm, seed = payload
    if arm not in C.ARMS_TRAIN:
        raise ValueError("unregistered training arm %r" % (arm,))
    seed = int(seed)
    if seed not in C.TRAIN_SEEDS:
        raise ValueError("unregistered training seed %d" % (seed,))
    _worker_init()
    layout = Layout(root)
    guard.assert_experiment_root(layout)
    guard.verify_identity(layout)
    episode.native_import_preflight(layout)
    episode.ensure_import_paths(layout)
    guard.verify_identity(layout)
    receipt = Path(layout.resume_preflight_receipt)
    if not receipt.is_file():
        raise RuntimeError("the resume preflight has not run: %s" % (receipt,))
    if compact.read_json(receipt).get("status") != "PASS":
        raise RuntimeError("%s is not PASS" % (receipt,))
    _freeze, entries = _base_freeze_entries(layout, episode_number=C.STAGE_START_EPISODE)
    matches = [item for item in entries
               if item["file_arm"] == arm and int(item["training_seed"]) == seed]
    if len(matches) != 1:
        raise RuntimeError("the base freeze holds %d checkpoints for %s/%d"
                           % (len(matches), arm, seed))
    base_entry = matches[0]
    base_path = _verify_frozen_file(base_entry)
    restored = trainer.load_base_checkpoint(
        base_path, arm=arm, seed=seed, episode=C.STAGE_START_EPISODE,
        base_identity_sha256=_freeze["base_identity_sha256"],
        expected_parameter_sha256=base_entry["parameter_fingerprint"],
        expected_actor_sha256=base_entry["actor_fingerprint"],
        expected_full_state_sha256=base_entry["full_state_fingerprint"])
    controller, optimizer = restored["controller"], restored["optimizer"]
    identity_sha = _identity_sha(layout)
    gate = LedgerGate(layout)
    ledger = gate.ledger
    initial = controller.parameter_summary()
    initial_state = trainer.full_state_fingerprint(controller, optimizer)
    telemetry = {
        "schema": "stage5b-ppo-telemetry/2",
        "experiment_id": C.RUN_ID,
        "stage": C.STAGE_INDEX,
        "arm": arm,
        "training_seed": seed,
        "objective": objective.describe(arm),
        "resume": {
            "base_checkpoint": {"path": str(base_path), "sha256": base_entry["sha256"],
                                "bytes": int(base_entry["bytes"]),
                                "episode": C.STAGE_START_EPISODE,
                                "base_identity_sha256": _freeze["base_identity_sha256"]},
            "base_experiment_id": C.BASE_EXPERIMENT_ID,
            "restored_full_state": initial_state["sha256"],
            "reseeded": False,
            "optimizer_reset": False,
            "first_episode_index": C.STAGE_START_EPISODE,
            "first_update_index": C.RESUME_FROM_UPDATE,
            "pending_rollout": "EMPTY_NOT_STORED_IN_CHECKPOINT",
        },
        "schedule": {"scenes": [C.scene_for(seed, index)
                                for index in range(C.STAGE_START_EPISODE,
                                                   C.STAGE_END_EPISODE)],
                     "episodes_per_update": C.EPISODES_PER_UPDATE,
                     "updates": C.NEW_UPDATES_PER_STAGE,
                     "first_episode_index": C.STAGE_START_EPISODE,
                     "first_update_index": C.RESUME_FROM_UPDATE,
                     "stage_end_episode": C.STAGE_END_EPISODE},
        "initial_parameters": initial,
        "initial_full_state": initial_state,
        "episodes": [],
        "updates": [],
        "convergence": {"checks": [], "converged": False,
                        "criterion": {"window_episodes": C.CONVERGENCE_WINDOW_EPISODES,
                                      "windows": C.CONVERGENCE_WINDOWS,
                                      "required_consecutive":
                                          C.CONVERGENCE_REQUIRED_CONSECUTIVE}},
        "gate_group": (arm == C.STARTUP_GATE_ARM and seed == C.STARTUP_GATE_SEED),
    }
    pending_rows = []
    pending_records = []
    updates_by_index = {}
    update_index = C.RESUME_FROM_UPDATE
    check_episodes = set(C.convergence_check_episodes())
    terminal_episode = None
    started = time.time()
    try:
        for index in range(C.STAGE_START_EPISODE, C.STAGE_END_EPISODE):
            if stopped(layout):
                raise RuntimeError("GLOBAL_STOP is present before %s episode %d"
                                   % (arm, index))
            slot = "train:%s:%d:%04d" % (arm, seed, index)
            status = gate.claim(slot)
            if status == "stopped":
                raise RuntimeError("GLOBAL_STOP refused the claim of %s" % (slot,))
            if status == "consumed":
                raise RuntimeError("slot %s was already consumed; a slot is never re-run"
                                   % (slot,))
            before = trainer.full_state_fingerprint(controller, optimizer)
            scene = C.scene_for(seed, index)
            row = episode.run_training_world(layout, arm, seed, index, scene, controller)
            pending_rows.append(row)
            pending_records.append({
                "slot": slot, "episode_index": int(index), "scene": int(scene),
                "on_time": int(row["on_time"]),
                "generated_goals": int(row["generated_goals"]),
                "discovered": int(row["discovered"]),
                "charged_bytes": int(row["spent_bytes"]),
                "paid_emissions": int(row["emissions"]),
                "steps": int(row["steps"]),
                "seconds": float(row["seconds"]),
                "actor_sha256_before": before["actor"],
                "full_state_sha256_before": before["sha256"],
                "actor_sha256_after": None,
                "full_state_sha256_after": None,
                "update_index": None,
                "optimizer_steps": None,
            })
            batch_closed = (len(pending_rows) == C.EPISODES_PER_UPDATE
                            or index == C.STAGE_END_EPISODE - 1)
            if not batch_closed:
                continue
            update = trainer.ppo_update(controller, optimizer, pending_rows, arm=arm,
                                        update_index=update_index)
            if int(update.get("optimizer_steps", 0)) <= 0:
                raise RuntimeError("update %d performed no optimizer step" % (update_index,))
            if int(update["optimizer_steps"]) != int(update["minibatches"]):
                raise RuntimeError("update %d counts %d steps for %d minibatches"
                                   % (update_index, update["optimizer_steps"],
                                      update["minibatches"]))
            update["update_index"] = update_index
            update["episode_indices"] = [int(item["episode_index"])
                                         for item in pending_records]
            telemetry["updates"].append(update)
            updates_by_index[int(update_index)] = update
            after = trainer.full_state_fingerprint(controller, optimizer)
            for record, produced in zip(pending_records, pending_rows):
                record["actor_sha256_after"] = after["actor"]
                record["full_state_sha256_after"] = after["sha256"]
                record["update_index"] = update_index
                record["optimizer_steps"] = int(update["optimizer_steps"])
                telemetry["episodes"].append(dict(record))
                state_hashes = {
                    "actor_sha256_before": record["actor_sha256_before"],
                    "actor_sha256_after": record["actor_sha256_after"],
                    "full_state_sha256_before": record["full_state_sha256_before"],
                    "full_state_sha256_after": record["full_state_sha256_after"],
                    "optimizer_steps": record["optimizer_steps"],
                    "update_index": record["update_index"],
                }
                compacted = episode.compact_row(
                    produced, phase="train", arm=arm, seed=seed,
                    episode_index=record["episode_index"], scene=record["scene"],
                    slot=record["slot"], state_hashes=state_hashes)
                compacted["origin"] = {"revision": C.REVISION_ID, "kind": "NEW_TRAIN_WORLD",
                                       "stage": C.STAGE_INDEX}
                compacted["lambda"] = float(C.LAMBDA_FIXED)
                path = Path(layout.train_row(record["slot"]))
                path.parent.mkdir(parents=True, exist_ok=True)
                compact.write_json_exclusive(path, compacted)
                ledger.finish(record["slot"], "ok",
                              {"row": str(path), "on_time": compacted["on_time"]})
            pending_rows = []
            pending_records = []
            episode_number = index + 1
            terminal_episode = episode_number
            if episode_number in C.STAGE_CHECKPOINT_EPISODES:
                trainer.save_checkpoint(
                    _checkpoint_path(layout, arm, seed, episode_number), controller, optimizer,
                    {"arm": arm, "training_seed": seed, "episode": episode_number,
                     "stage": C.STAGE_INDEX},
                    identity_sha256=identity_sha)
            if (update_index == C.RESUME_FROM_UPDATE
                    and arm == C.STARTUP_GATE_ARM and seed == C.STARTUP_GATE_SEED):
                signal = {"schema": "stage5b-gate-first-update/1",
                          "arm": arm, "training_seed": seed,
                          "update_index": C.RESUME_FROM_UPDATE,
                          "episodes_accepted": int(episode_number),
                          "full_state_sha256": after["sha256"],
                          "actor_sha256": after["actor"],
                          "recorded_utc": _utc(),
                          "note": ("the first four resumed episodes and their update are "
                                   "complete; the other five groups may start while this one "
                                   "continues")}
                compact.write_json_exclusive(gate_signal_path(layout), signal)
            if episode_number in check_episodes:
                certificate = convergence_check_at(layout, arm, seed, episode_number,
                                                   updates_by_index)
                telemetry["convergence"]["checks"].append({
                    "end_episode": int(episode_number),
                    "passed": bool(certificate.get("passed")),
                    "status": certificate.get("status"),
                    "path": certificate["path"],
                })
                if _certificate_requires_stop(certificate):
                    reason = ("convergence history is invalid at %s/%d episode %d: %s"
                              % (arm, seed, episode_number,
                                 "; ".join(certificate.get("problems") or [])))
                    request_stop(layout, reason, detail=_budget_evidence())
                    raise RuntimeError(reason)
                if convergence.is_converged(_read_check_certificates(layout, arm, seed)):
                    telemetry["convergence"]["converged"] = True
                    telemetry["convergence"]["converged_at_episode"] = int(episode_number)
                    break
            update_index += 1
    except Exception as error:
        request_stop(layout, "%s/%d failed: %s" % (arm, seed, error),
                     detail=_budget_evidence())
        raise
    telemetry["final_parameters"] = controller.parameter_summary()
    telemetry["final_full_state"] = trainer.full_state_fingerprint(controller, optimizer)
    telemetry["terminal_episode"] = int(terminal_episode or C.STAGE_END_EPISODE)
    telemetry["seconds"] = round(time.time() - started, 2)
    telemetry["counts"] = {
        "episodes": len(telemetry["episodes"]),
        "updates": len(telemetry["updates"]),
        "optimizer_steps": sum(int(item["optimizer_steps"])
                               for item in telemetry["updates"]),
        "expected_updates": C.NEW_UPDATES_PER_STAGE,
        "expected_episodes": C.NEW_EPISODES_PER_STAGE,
        "parent_episodes": C.STAGE_START_EPISODE,
        "parent_updates": C.RESUME_FROM_UPDATE,
    }
    if telemetry["convergence"]["converged"]:
        if int(telemetry["terminal_episode"]) != int(
                telemetry["convergence"]["converged_at_episode"]):
            raise RuntimeError("%s/%d converged at episode %r but stopped at %r"
                               % (arm, seed,
                                  telemetry["convergence"]["converged_at_episode"],
                                  telemetry["terminal_episode"]))
    else:
        if len(telemetry["episodes"]) != C.NEW_EPISODES_PER_STAGE:
            raise RuntimeError("%s/%d recorded %d episodes, expected %d"
                               % (arm, seed, len(telemetry["episodes"]),
                                  C.NEW_EPISODES_PER_STAGE))
        if len(telemetry["updates"]) != C.NEW_UPDATES_PER_STAGE:
            raise RuntimeError("%s/%d recorded %d updates, expected %d"
                               % (arm, seed, len(telemetry["updates"]),
                                  C.NEW_UPDATES_PER_STAGE))
        if int(telemetry["terminal_episode"]) != C.STAGE_END_EPISODE:
            raise RuntimeError("%s/%d stopped at episode %r, expected the stage cap %d"
                               % (arm, seed, telemetry["terminal_episode"],
                                  C.STAGE_END_EPISODE))
    path = Path(layout.telemetry) / ("%s_s%d.json" % (arm, seed))
    compact.write_json_exclusive(path, telemetry)
    certificate = {"schema": C.CONVERGENCE_SCHEMA, "experiment_id": C.RUN_ID,
                   "arm": arm, "seed": seed, "stage": C.STAGE_INDEX,
                   "status": ("CONVERGED" if telemetry["convergence"]["converged"]
                              else "NOT_YET_CONVERGED"),
                   "terminal_episode": int(telemetry["terminal_episode"]),
                   "checks": telemetry["convergence"]["checks"],
                   "consecutive_passes": convergence.consecutive_passes(
                       _read_check_certificates(layout, arm, seed)),
                   "telemetry": str(path),
                   "telemetry_sha256": guard.sha256_file(path),
                   "stage_end_episode": C.STAGE_END_EPISODE,
                   "note": ("a converged run stops at its first certificate and is not trained "
                            "further to match another run's length")}
    if telemetry["convergence"]["converged"]:
        checkpoint = _checkpoint_path(layout, arm, seed, telemetry["terminal_episode"])
        if not checkpoint.is_file():
            raise RuntimeError("the converged run has no checkpoint at episode %d"
                               % (telemetry["terminal_episode"],))
        certificate["checkpoint"] = {"path": str(checkpoint),
                                     "bytes": int(checkpoint.stat().st_size),
                                     "sha256": guard.sha256_file(checkpoint)}
    final_path = Path(layout.convergence_certificate(arm, seed))
    final_path.parent.mkdir(parents=True, exist_ok=True)
    compact.write_json_exclusive(final_path, certificate)
    return {"slot": "train:%s:%d" % (arm, seed), "status": "ok",
            "episodes": len(telemetry["episodes"]), "telemetry": str(path),
            "converged": bool(telemetry["convergence"]["converged"]),
            "terminal_episode": int(telemetry["terminal_episode"])}


def _run_train_pool(layout, tasks, workers, function):
    """Gate group first; the other five are released after its first update.

    Active workers never exceed the real width. A gate failure is noticed while the other
    groups run, because the gate future stays in the wait set, and it requests STOP at once.
    """
    tasks = list(tasks)
    if not tasks:
        raise RuntimeError("no training group to run")
    if int(workers) <= 1:
        return _run_serial(layout, tasks, function)
    gate_task, rest = tasks[0], tasks[1:]
    results = []
    cancelled = []
    failure = None
    aborted = False
    executor = ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init)
    futures = {executor.submit(function, gate_task): gate_task}
    gate_done = False
    iterator = iter(rest)
    try:
        deadline = time.time() + GATE_TIMEOUT_SECONDS
        while not gate_done:
            if stopped(layout):
                aborted = True
                break
            done, _pending = wait(list(futures), timeout=5.0, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                try:
                    result = _require_world_result(future.result(), _safe_task_slot(task))
                except BaseException as error:
                    failure = error
                    request_stop(layout, "%s failed: %s" % (_safe_task_slot(task), error))
                    raise
                result.setdefault("slot", _safe_task_slot(task))
                results.append(result)
                gate_done = True
            if gate_done:
                break
            if gate_signal_path(layout).exists():
                break
            if time.time() > deadline:
                failure = RuntimeError("the startup gate did not report its first update")
                request_stop(layout, "startup gate timed out")
                raise failure
        if aborted or failure is not None:
            raise RuntimeError("training aborted before the non-gate groups were released")
        while True:
            # The gate future stays in `futures`, so it already counts against the width.
            limit = int(workers)
            while len(futures) < max(1, limit) and failure is None and not stopped(layout):
                try:
                    task = next(iterator)
                except StopIteration:
                    break
                futures[executor.submit(function, task)] = task
            if not futures:
                break
            done, _pending = wait(list(futures), return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                try:
                    result = _require_world_result(future.result(), _safe_task_slot(task))
                except BaseException as error:
                    failure = error
                    request_stop(layout, "%s failed: %s" % (_safe_task_slot(task), error))
                    for pending, pending_task in list(futures.items()):
                        if pending.cancel():
                            cancelled.append(_safe_task_slot(pending_task))
                        else:
                            try:
                                pending_result = _require_world_result(
                                    pending.result(), _safe_task_slot(pending_task))
                                pending_result.setdefault("slot",
                                                          _safe_task_slot(pending_task))
                                results.append(pending_result)
                            except BaseException as pending_error:
                                failure = failure or pending_error
                                request_stop(layout, "%s failed: %s"
                                             % (_safe_task_slot(pending_task), pending_error))
                    futures.clear()
                    break
                result.setdefault("slot", _safe_task_slot(task))
                results.append(result)
                if _safe_task_slot(task) == _safe_task_slot(gate_task):
                    gate_done = True
            if failure is not None:
                break
            if stopped(layout):
                aborted = True
                for future, task in list(futures.items()):
                    if future.cancel():
                        cancelled.append(_safe_task_slot(task))
                    else:
                        try:
                            result = _require_world_result(future.result(),
                                                           _safe_task_slot(task))
                            result.setdefault("slot", _safe_task_slot(task))
                            results.append(result)
                        except BaseException as error:
                            failure = failure or error
                futures.clear()
                break
    except BaseException as error:
        failure = failure or error
        try:
            request_stop(layout, "training pool failed: %s" % (error,))
        except BaseException:
            pass
    finally:
        try:
            for task in iterator:
                label = _safe_task_slot(task)
                cancelled.append(label)
                results.append({"slot": label, "status": "stopped_not_started"})
            for future, task in list(futures.items()):
                label = _safe_task_slot(task)
                if future.cancel():
                    cancelled.append(label)
                else:
                    try:
                        pending = _require_world_result(future.result(), label)
                        pending.setdefault("slot", label)
                        results.append(pending)
                    except BaseException as error:
                        failure = failure or error
        except BaseException as error:
            failure = failure or error
        finally:
            processes = _pool_process_snapshot(executor)
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except BaseException as error:
                failure = failure or error
            for record in (lambda: _record_pool_exits(layout, processes, "train"),
                           lambda: _record_abort(layout, cancelled)):
                try:
                    record()
                except BaseException as error:
                    failure = failure or error
    if failure is not None:
        raise failure
    if aborted or cancelled:
        raise RuntimeError("training aborted before all groups were accepted")
    if len(results) != len(tasks):
        raise RuntimeError("training returned %d results for %d groups"
                           % (len(results), len(tasks)))
    return results


def _run_serial(layout, tasks, function):
    """One worker in this process; a failure stops the run and keeps the real traceback."""
    results = []
    primary_failure = None
    try:
        for task in tasks:
            if stopped(layout):
                raise RuntimeError("GLOBAL_STOP refused %s" % (_safe_task_slot(task),))
            results.append(_require_world_result(function(task), _safe_task_slot(task)))
    except BaseException as error:
        primary_failure = error
        try:
            request_stop(layout, "serial training failed: %s" % (error,))
        except BaseException:
            pass
        raise
    finally:
        try:
            _record_pool_exits(layout, [], "train")
        except BaseException:
            # The original world failure is the evidence that must survive. A successful
            # serial run still surfaces an evidence-write failure below because primary_failure
            # is None in that branch.
            if primary_failure is None:
                raise
    return results


def require_frozen(layout):
    guard.assert_experiment_root(layout)
    guard.assert_parent_admission(layout)
    layout.ensure()
    return guard.verify_identity(layout)


def _require_clean_training(layout):
    """A stage is never replayed: no row, checkpoint or telemetry may exist yet."""
    offenders = sorted(Path(layout.train_rows).glob("*.json"))
    if offenders:
        raise RuntimeError("training rows already exist (%d); this stage is never replayed"
                           % (len(offenders),))
    for arm in C.ARMS_TRAIN:
        for seed in C.TRAIN_SEEDS:
            for episode_number in C.STAGE_CHECKPOINT_EPISODES:
                path = _checkpoint_path(layout, arm, seed, episode_number)
                if path.exists():
                    raise RuntimeError("checkpoint already exists: %s" % (path,))
    for arm in C.ARMS_TRAIN:
        for seed in C.TRAIN_SEEDS:
            path = Path(layout.telemetry) / ("%s_s%d.json" % (arm, seed))
            if path.exists():
                raise RuntimeError("telemetry already exists: %s" % (path,))
    return True


def _training_groups(root):
    """The one gate group first and the other five after it, each exactly once."""
    gate = (str(root), C.STARTUP_GATE_ARM, C.STARTUP_GATE_SEED)
    others = [(str(root), arm, int(seed)) for arm in C.ARMS_TRAIN for seed in C.TRAIN_SEEDS
              if not (arm == C.STARTUP_GATE_ARM and int(seed) == C.STARTUP_GATE_SEED)]
    if len(others) != 5:
        raise RuntimeError("the stage has %d non-gate groups, expected 5" % (len(others),))
    return gate, others


def recovery_g3(layout, context_path):
    """Explicit zero-World recovery G3; no normal-parent or normal-identity gate."""
    from . import recovery_runtime
    _worker_init()
    episode.validate_runtime_imports(layout)
    _assert_single_thread()
    return recovery_runtime.g3(layout, context_path)


def recovery_train(layout, context_path, g3_report_path,
                    world_authorization_path, workers):
    """Explicit finite recovery training with two validated execution sidecars."""
    from . import recovery_runtime
    _worker_init()
    _assert_single_thread()
    return recovery_runtime.run_train(
        layout, context_path, g3_report_path, world_authorization_path, workers)


def train(layout, workers):
    guard.assert_experiment_root(layout)
    require_frozen(layout)
    if stopped(layout):
        raise RuntimeError("GLOBAL_STOP is present; the training phase refuses to start")
    for gate_path in (layout.base_model_freeze, layout.resume_preflight_receipt):
        if not Path(gate_path).is_file():
            raise RuntimeError("a zero-world gate has not run: %s is missing" % (gate_path,))
        payload = compact.read_json(gate_path)
        expected = "FROZEN" if gate_path == layout.base_model_freeze else "PASS"
        if payload.get("status") != expected:
            raise RuntimeError("%s records %r, not %s"
                               % (gate_path, payload.get("status"), expected))
    if gate_signal_path(layout).exists():
        raise RuntimeError("a startup-gate signal already exists; training is never replayed")
    _require_clean_training(layout)
    contract = _assert_stage_contract(layout)
    episode.validate_runtime_imports(layout)
    gate, others = _training_groups(layout.root)
    results = _run_train_pool(layout, [gate] + others, workers, _train_group)
    if stopped(layout):
        raise RuntimeError("GLOBAL_STOP is present after the training phase")
    if len(results) != 6:
        raise RuntimeError("training returned %d groups, expected 6" % (len(results),))
    prefix = _validate_prefix_results(results)
    episodes = int(prefix["episodes"])
    counts = SlotLedger(layout).counts()
    rows = sorted(Path(layout.train_rows).glob("*.json"))
    if (counts["claimed"] != episodes or counts["results"] != episodes
            or counts["by_status"]["ok"] != episodes):
        raise RuntimeError("accepted prefix and ledger do not close: %r for %d episodes"
                           % (counts, episodes))
    if len(rows) != episodes:
        raise RuntimeError("accepted prefix and rows do not close: %d rows for %d episodes"
                           % (len(rows), episodes))
    ledger = SlotLedger(layout)
    expected_slots = set()
    for result in results:
        run_arm, run_seed = str(result["slot"]).split(":")[1:]
        terminal = int(result["terminal_episode"])
        if result.get("converged"):
            checks = _read_check_certificates(layout, run_arm, int(run_seed))
            if (not convergence.is_converged(checks) or not checks
                    or int(checks[-1]["end_episode"]) != terminal
                    or not _checkpoint_path(layout, run_arm, int(run_seed), terminal).is_file()):
                raise RuntimeError("converged prefix lacks two valid checks and its terminal checkpoint")
        for episode_index in range(C.STAGE_START_EPISODE, terminal):
            slot = "train:%s:%s:%04d" % (run_arm, run_seed, episode_index)
            expected_slots.add(slot)
            if not ledger.claimed(slot) or not ledger.finished(slot):
                raise RuntimeError("accepted prefix names an unclaimed or unfinished slot: %s"
                                   % (slot,))
            if not Path(layout.train_row(slot)).is_file():
                raise RuntimeError("accepted prefix names a slot without its row: %s" % (slot,))
    if len(expected_slots) != episodes:
        raise RuntimeError("accepted prefix does not form a unique slot set")
    return {"groups": len(results), "episodes": episodes, "contract": contract,
            "unused_world_slots": int(prefix["unused_world_slots"]),
            "slot_counts": counts,
            "converged_runs": sorted(item["slot"] for item in results if item.get("converged")),
            "results": results}


def convergence_report(layout):
    """Aggregate the six run certificates; never claims completion that did not happen."""
    guard.assert_experiment_root(layout)
    layout.ensure()
    require_frozen(layout)
    path = Path(layout.convergence_report)
    if path.exists():
        raise RuntimeError("the convergence report already exists: %s" % (path,))
    runs = []
    converged = 0
    for arm in C.ARMS_TRAIN:
        for seed in C.TRAIN_SEEDS:
            certificate_path = Path(layout.convergence_certificate(arm, seed))
            if certificate_path.is_file():
                certificate = compact.read_json(certificate_path)
                status = certificate.get("status")
                entry = {"arm": arm, "seed": int(seed), "status": status,
                         "terminal_episode": certificate.get("terminal_episode"),
                         "consecutive_passes": certificate.get("consecutive_passes"),
                         "checkpoint": certificate.get("checkpoint"),
                         "certificate": str(certificate_path),
                         "certificate_sha256": guard.sha256_file(certificate_path)}
            else:
                status = "ABSENT_NOT_YET_CHECKED"
                entry = {"arm": arm, "seed": int(seed), "status": status,
                         "terminal_episode": None, "checkpoint": None,
                         "certificate": None}
            if status == "CONVERGED":
                converged += 1
            runs.append(entry)
    check_episodes = list(C.convergence_check_episodes())
    if converged == len(runs):
        verdict = "CONVERGED_ALL"
        note = ("all six runs hold two consecutive passing certificates at 480-episode "
                "windows; the goal is complete for this stage")
    elif not check_episodes:
        verdict = C.STAGE_VERDICT_IF_NO_CERTIFICATE
        note = C.STAGE_VERDICT_NOTE
    else:
        verdict = "NOT_YET_CONVERGED"
        note = ("the goal stays active; the main model freezes the next finite stage and only "
                "the unconverged runs continue")
    ledger = SlotLedger(layout)
    payload = {"schema": C.CONVERGENCE_REPORT_SCHEMA, "experiment_id": C.RUN_ID,
               "stage": C.STAGE_INDEX, "recorded_utc": _utc(),
               "verdict": verdict, "verdict_note": note,
               "stage_start_episode": C.STAGE_START_EPISODE,
               "stage_end_episode": C.STAGE_END_EPISODE,
               "check_episodes_this_stage": check_episodes,
               "converged_runs": converged, "total_runs": len(runs),
               "criterion": {
                   "window_episodes": C.CONVERGENCE_WINDOW_EPISODES,
                   "windows": C.CONVERGENCE_WINDOWS,
                   "pairs": [list(pair) for pair in C.CONVERGENCE_PAIRS],
                   "interval_level": C.CONVERGENCE_INTERVAL_LEVEL,
                   "resamples": C.CONVERGENCE_BOOTSTRAP_REPS,
                   "pair_bound": C.CONVERGENCE_PAIR_BOUND,
                   "mean_range_bound": C.CONVERGENCE_MEAN_RANGE_BOUND,
                   "slope_bound": C.CONVERGENCE_SLOPE_BOUND,
                   "slope_basis": C.CONVERGENCE_SLOPE_BASIS,
                   "budget_range_bound": C.CONVERGENCE_BUDGET_RANGE_BOUND,
                   "budget_range_role": C.CONVERGENCE_BUDGET_RANGE_ROLE,
                   "required_consecutive": C.CONVERGENCE_REQUIRED_CONSECUTIVE,
                   "check_step": C.CONVERGENCE_CHECK_STEP,
               },
               "slots": ledger.counts(),
               "runs": runs,
               "stopped": stopped(layout),
               "note": ("per-run equivalence over three 480-episode windows; not a global "
                        "error rate, not a proof of optimality, and a stable low plateau is "
                        "still a low plateau. Training stability is not generalisation.")}
    compact.write_json_exclusive(path, payload)
    return {"path": str(path), "verdict": verdict, "converged_runs": converged,
            "sha256": guard.sha256_file(path), "slots": payload["slots"]}


def monitor(layout):
    """Read-only progress snapshot; an unreadable input stays UNKNOWN."""
    from . import monitor as monitor_module
    guard.assert_experiment_root(layout)
    return monitor_module.snapshot(layout)


def _not_authorised(name):
    raise RuntimeError(
        "phase %r is not authorised in stage %d: this stage trains only. The greedy evaluation, "
        "analysis and finalisation are separately frozen later phases of this convergence "
        "goal; no half-trained plan may be evaluated." % (name, C.STAGE_INDEX))


def run_phase(name, layout, workers=1, recovery_context=None,
              recovery_g3_report=None, recovery_world_authorization=None):
    """Acquire one root-level recovery lease before any receipt, claim, or World dispatch."""
    recovery_phase = name in ("recovery-g3", "recovery-train")
    if recovery_phase:
        from . import recovery_runtime
        from .recovery_cutover import RootPhaseLock
        recovery_runtime.assert_runtime_root(layout)
        with RootPhaseLock(layout.root):
            return _run_phase_unlocked(
                name, layout, workers=workers, recovery_context=recovery_context,
                recovery_g3_report=recovery_g3_report,
                recovery_world_authorization=recovery_world_authorization)
    return _run_phase_unlocked(
        name, layout, workers=workers, recovery_context=recovery_context,
        recovery_g3_report=recovery_g3_report,
        recovery_world_authorization=recovery_world_authorization)


def _run_phase_unlocked(name, layout, workers=1, recovery_context=None,
                        recovery_g3_report=None, recovery_world_authorization=None):
    recovery_phase = name in ("recovery-g3", "recovery-train")
    recovery_start_detail = {}
    if recovery_phase:
        if not isinstance(recovery_context, str) or not recovery_context:
            raise RuntimeError("%s requires --recovery-context" % (name,))
        from . import recovery_runtime
        recovery_runtime.assert_runtime_root(layout)
        if name == "recovery-train":
            for value, label in (
                    (recovery_g3_report, "--recovery-g3-report"),
                    (recovery_world_authorization, "--recovery-world-authorization")):
                if not isinstance(value, str) or not value:
                    raise RuntimeError("recovery-train requires " + label)
            context = recovery_runtime.load_context(layout, recovery_context, full=True)
            recovery_start_detail["execution_bindings"] = (
                recovery_runtime.load_execution_bindings(
                    context, recovery_g3_report, recovery_world_authorization))
        else:
            if recovery_g3_report is not None or recovery_world_authorization is not None:
                raise RuntimeError("recovery-g3 does not accept training execution sidecars")
            context = recovery_runtime.load_context(layout, recovery_context, full=False)
            recovery_start_detail["base_context"] = context["context_pin"]
    else:
        if recovery_g3_report is not None or recovery_world_authorization is not None:
            raise RuntimeError("execution sidecars are accepted only by recovery-train")
        guard.assert_experiment_root(layout)
        if name in ("prepare", "freeze-base-models", "resume-preflight", "train",
                    "convergence-report"):
            guard.assert_parent_admission(layout)
    started = _utc()
    receipt_id = _new_receipt_id(name)
    _write_receipt(layout, name, "started", receipt_id, None, recovery_start_detail)
    code = 0
    detail = {}
    try:
        if name == "prepare":
            detail = prepare(layout)
        elif name == "puretests":
            detail = puretests(layout)
            code = 0 if detail.get("was_successful") else 1
        elif name == "freeze-base-models":
            detail = freeze_base_models(layout)
        elif name == "resume-preflight":
            detail = resume_preflight(layout)
        elif name == "train":
            detail = train(layout, workers)
        elif name == "recovery-g3":
            detail = recovery_g3(layout, recovery_context)
        elif name == "recovery-train":
            detail = recovery_train(
                layout, recovery_context, recovery_g3_report,
                recovery_world_authorization, workers)
        elif name == "convergence-report":
            detail = convergence_report(layout)
        elif name == "monitor":
            detail = monitor(layout)
        elif name in ("eval", "analysis", "finalize", "freeze-eval-models"):
            _not_authorised(name)
        else:
            raise ValueError("unknown phase %r" % (name,))
    except Exception as error:
        code = 1
        detail = {"error": type(error).__name__ + ": " + str(error),
                  "traceback": traceback.format_exc()}
        try:
            if recovery_phase:
                from . import recovery_runtime
                recovery_view = (context if "context" in locals()
                                 else recovery_runtime.load_context(
                                     layout, recovery_context, full=False))
                detail["slot_accounting"] = recovery_runtime.recovery_ledger_accounting(
                    recovery_view, layout)
            else:
                detail["slot_accounting"] = SlotLedger(layout).accounting(
                    None if name != "train" else "train")
        except Exception as accounting_error:
            detail["slot_accounting_error"] = (type(accounting_error).__name__ + ": "
                                               + str(accounting_error))
        _write_receipt(layout, name, "error", receipt_id, code, detail)
        raise
    _write_receipt(layout, name, "terminal", receipt_id, code, detail)
    return {"phase": name, "code": int(code), "receipt_id": receipt_id,
            "started_utc": started, "detail": detail}

