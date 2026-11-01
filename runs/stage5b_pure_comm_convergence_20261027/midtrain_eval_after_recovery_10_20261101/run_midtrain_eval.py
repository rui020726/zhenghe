from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

SCENES = tuple(range(2301901, 2301909))
WORLD_SEED = 20261026
IDENTITY_SHA = "FC31F4A4F5FDBE7252762445590C44EC8E97381F07D477D57F92F47C99431892"
MODEL_SPECS = {
    ("T1", 91002): (6680, "milestone_6680.pth", "9D5990924B3B079C28430CD40D43127A03899AE99B67683A524222CAE9D09542"),
    ("T2", 91002): (6800, "milestone_6800.pth", "A3778BAE3E2BB7602C9CB759248D5D3054AA93BE3187300AC76B0D291F3144FA"),
}

def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(str(tmp), str(path))

def write_exclusive(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def slot_name(arm, seed, scene):
    return "eval:%s:%d:%d" % (arm, int(seed), int(scene))

def fname(slot):
    return slot.replace(":", "_") + ".json"

def layout_for(root, run_dir):
    from stage5b_comm.paths import Layout
    layout = Layout(Path(run_dir))
    layout.execution = Path(run_dir)
    layout.world_logs = Path(run_dir) / "world_logs"
    layout.rows = Path(run_dir) / "rows"
    layout.ledger = Path(run_dir) / "ledger"
    layout.receipts = Path(run_dir) / "receipts"
    return layout

def worker_task(task):
    sys.dont_write_bytecode = True
    root = Path(task["root"])
    run_dir = Path(task["run_dir"])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from stage5b_comm import constants as C
    from stage5b_comm import episode, trainer

    slot = task["slot"]
    claim_path = write_exclusive(run_dir / "ledger" / "claims" / fname(slot), {
        "schema": "stage5b-midtrain-eval-claim/1", "slot": slot,
        "arm": task["arm"], "training_seed": task["training_seed"], "scene": task["scene"],
        "claimed_utc": utc(), "worker_pid": os.getpid(), "status": "CLAIMED",
    })
    layout = layout_for(root, run_dir)
    episode.ensure_import_paths(layout)
    episode.native_import_preflight(layout)
    loaded_utc = utc()
    controller, _optimizer, state = trainer.load_checkpoint(
        Path(task["checkpoint"]["path"]), arm=task["arm"],
        identity_sha256=task["identity_sha256"], expected_seed=int(task["training_seed"]),
        expected_episode=int(task["checkpoint"]["episode"]))
    parameter_sha = controller.parameter_summary()["parameter_sha256"]
    logical_arm = C.eval_arm_name(task["arm"], task["training_seed"])
    row = episode.run_eval_world(layout, logical_arm, int(task["training_seed"]),
                                 int(task["scene"]), controller)
    row["attach_kind"] = "greedy"
    origin = dict(task["checkpoint"])
    origin.update({"logical_arm": logical_arm, "parameter_sha256": parameter_sha})
    row["checkpoint_origin"] = origin
    record = episode.compact_row(
        row, phase="eval", arm=logical_arm, seed=int(task["training_seed"]),
        episode_index=None, scene=int(task["scene"]), slot=slot,
        state_hashes={"attach_kind": "greedy", "actor_sha256": parameter_sha})
    checks = {
        "generated_goals_24": int(record["generated_goals"]) == 24,
        "steps_1000": int(record["steps"]) == 1000,
        "budget_58600": int(record["budget_bytes"]) == 58600,
        "eval_world_seed": int(record["eval_world_seed"]) == 20261026,
    }
    if not all(checks.values()):
        raise RuntimeError("%s fixed-protocol check failed: %r" % (slot, checks))
    record.update({"logical_arm": logical_arm, "checkpoint_episode": int(origin["episode"]),
                   "checkpoint_origin": origin, "model_quality_label": origin["quality_label"],
                   "greedy_eval": True})
    row_path = write_exclusive(run_dir / "rows" / fname(slot), record)
    result = {
        "schema": "stage5b-midtrain-eval-world-result/1", "slot": slot, "status": "OK",
        "arm": task["arm"], "training_seed": int(task["training_seed"]), "scene": int(task["scene"]),
        "worker_pid": os.getpid(), "model_loaded": True,
        "loaded_state_episode": int(state.get("episode", -1)),
        "loaded_training_seed": int(state.get("training_seed", -1)),
        "parameter_sha256": parameter_sha, "checkpoint": origin,
        "claim": str(claim_path), "row": str(row_path), "row_sha256": sha(row_path),
        "on_time": int(record["on_time"]), "completed": int(record["completed"]),
        "generated_goals": int(record["generated_goals"]),
        "success_rate": float(record["success_rate"]), "paid_bytes": int(record["paid_bytes"]),
        "budget_used_fraction": float(record["budget_used_fraction"]),
        "seconds": float(record["seconds"]), "checks": checks,
        "loaded_utc": loaded_utc, "finished_utc": utc(),
    }
    result_path = write_exclusive(run_dir / "ledger" / "results" / fname(slot), result)
    result["result_path"] = str(result_path)
    result["result_sha256"] = sha(result_path)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    root, run_dir = Path(args.root).resolve(), Path(args.run_dir).resolve()
    if run_dir.exists():
        if (run_dir / "EVAL_PLAN.json").exists() or (run_dir / "progress.json").exists():
            raise RuntimeError("evaluation directory already contains a run: %s" % run_dir)
    else:
        run_dir.mkdir(parents=False, exist_ok=False)
    for name in ("rows", "world_logs", "ledger", "ledger/claims", "ledger/results", "receipts"):
        target = run_dir / name
        if target.exists():
            if any(p.is_file() for p in target.rglob("*")):
                raise RuntimeError("evaluation subdirectory is not empty: %s" % target)
        else:
            target.mkdir(parents=False, exist_ok=False)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from stage5b_comm import constants as C

    fixed = {"scenes": list(SCENES), "eval_world_seed": int(C.EVAL_WORLD_SEED),
             "num_goals": int(C.NUM_GOALS), "steps": int(C.STEPS),
             "budget_bytes": int(C.BUDGET_BYTES), "b0_bytes": float(C.B0_BYTES),
             "feature_dim": int(C.FEATURE_DIM)}
    expected = {"scenes": list(SCENES), "eval_world_seed": WORLD_SEED, "num_goals": 24,
                "steps": 1000, "budget_bytes": 58600, "b0_bytes": 10000.0, "feature_dim": 19}
    if fixed != expected or tuple(C.EVAL_SCENES) != SCENES:
        raise RuntimeError("fixed protocol constants mismatch: %r" % fixed)
    identity_path = root / "protocol" / "RECOVERY_EXPERIMENT_IDENTITY.json"
    identity_sha = sha(identity_path)
    if identity_sha.lower() != IDENTITY_SHA.lower():
        raise RuntimeError("recovery identity mismatch: %s" % identity_sha)

    models, planned = [], []
    for arm, seed in sorted(MODEL_SPECS):
        episode_no, name, expected_sha = MODEL_SPECS[(arm, seed)]
        checkpoint = root / "execution" / "checkpoints" / ("%s_s%d" % (arm, seed)) / name
        actual_sha = sha(checkpoint) if checkpoint.is_file() else None
        pin = {"arm": arm, "training_seed": seed, "path": str(checkpoint),
               "experiment_identity_sha256": IDENTITY_SHA,
               "checkpoint_payload_identity_sha256": IDENTITY_SHA.lower(),
               "bytes": int(checkpoint.stat().st_size) if checkpoint.is_file() else None,
               "sha256": actual_sha, "expected_sha256": expected_sha,
               "matches_expected": actual_sha is not None and actual_sha.lower() == expected_sha.lower(), "episode": episode_no,
               "quality_label": "MIDTRAIN_RECOVERY_10_NOT_AT_TARGET"}
        models.append(pin)
        for scene in SCENES:
            slot = slot_name(arm, seed, scene)
            planned.append({"slot": slot, "arm": arm, "training_seed": seed, "scene": scene,
                            "root": str(root), "run_dir": str(run_dir),
                            "checkpoint": pin, "identity_sha256": pin["checkpoint_payload_identity_sha256"],
                            "model_quality_label": pin["quality_label"],
                            "status": "PLANNED" if pin["matches_expected"] else "BLOCKED_CHECKPOINT_PIN_MISMATCH"})
    if len(planned) != 16 or len({x["slot"] for x in planned}) != 16:
        raise RuntimeError("evaluation plan is not 16 unique model-scene slots")
    plan = {"schema": "stage5b-midtrain-dev-reused-plan/1", "created_utc": utc(),
            "root": str(root), "run_dir": str(run_dir), "identity_path": str(identity_path),
            "identity_sha256": identity_sha, "authorization": {"authority": "USER_EXPLICIT_INSTRUCTION",
                      "decision_ref": "USER_20261101_SIDECHAT_MIDTRAIN_EVAL",
                      "scope": "16 DEV_REUSED eval Worlds (2 live runs x 8 scenes), greedy actor"}, 
            "label": "MIDTRAIN_DEV_REUSED",
            "not_formal_validation": True, "not_overall_go": True, "holdout_sealed": True,
            "original_no_go_preserved": True, "goal_complete": False, "requested_worlds": 16,
            "fixed_protocol": fixed, "models": models, "slots": planned}
    dump(run_dir / "EVAL_PLAN.json", plan)
    if not all(x["matches_expected"] for x in models):
        raise RuntimeError("checkpoint pin mismatch; no world started")

    progress = {"schema": "stage5b-midtrain-dev-reused-progress/1",
                "started_utc": utc(), "updated_utc": utc(), "requested_worlds": 16,
                "claims": 0, "completed_worlds": 0, "model_loads": 0, "failures": 0,
                "per_scene": [], "status": "RUNNING", "label": "MIDTRAIN_DEV_REUSED"}
    dump(run_dir / "progress.json", progress)
    results, failures, pending = [], [], {}
    iterator = iter(planned)
    executor = ProcessPoolExecutor(max_workers=int(args.workers))
    stop_scheduling = False
    try:
        for _ in range(min(int(args.workers), 16)):
            task = next(iterator)
            pending[executor.submit(worker_task, task)] = task
        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    result = future.result()
                except BaseException as exc:
                    stop_scheduling = True
                    failures.append({"slot": task["slot"], "error_type": type(exc).__name__,
                                     "error": str(exc), "traceback": traceback.format_exc(),
                                     "recorded_utc": utc()})
                else:
                    results.append(result)
                    progress["model_loads"] += int(bool(result.get("model_loaded")))
                    progress["per_scene"].append({
                        "slot": result["slot"], "arm": result["arm"],
                        "training_seed": result["training_seed"], "scene": result["scene"],
                        "checkpoint_episode": result["checkpoint"]["episode"],
                        "on_time": result["on_time"], "completed": result["completed"],
                        "generated_goals": result["generated_goals"],
                        "success_rate": result["success_rate"], "paid_bytes": result["paid_bytes"],
                        "seconds": result["seconds"], "row": result["row"],
                        "result_path": result["result_path"]})
                progress.update({"updated_utc": utc(), "completed_worlds": len(results),
                                 "failures": len(failures),
                                 "claims": len(list((run_dir / "ledger" / "claims").glob("*.json")))})
                dump(run_dir / "progress.json", progress)
                if failures:
                    dump(run_dir / "FAILURE.json", {
                        "schema": "stage5b-midtrain-dev-reused-failure/1", "recorded_utc": utc(),
                        "failures": failures, "claims_started": progress["claims"],
                        "completed_worlds": len(results),
                        "note": "No failed world is retried and no claim is replaced."})
                if not stop_scheduling:
                    try:
                        next_task = next(iterator)
                    except StopIteration:
                        pass
                    else:
                        pending[executor.submit(worker_task, next_task)] = next_task
    except BaseException as exc:
        dump(run_dir / "FAILURE.json", {
            "schema": "stage5b-midtrain-dev-reused-failure/1", "recorded_utc": utc(),
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "claims_started": len(list((run_dir / "ledger" / "claims").glob("*.json"))),
            "completed_worlds": len(results)})
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=bool(failures))

    by_model = {}
    for item in results:
        key = (item["arm"], int(item["training_seed"]))
        row = by_model.setdefault(key, {
            "arm": item["arm"], "training_seed": int(item["training_seed"]),
            "checkpoint": item["checkpoint"], "quality_label": item["checkpoint"]["quality_label"],
            "completed_worlds": 0, "on_time_tasks": 0, "completed_tasks": 0,
            "generated_tasks": 0, "paid_bytes": 0})
        row["completed_worlds"] += 1
        row["on_time_tasks"] += int(item["on_time"])
        row["completed_tasks"] += int(item["completed"])
        row["generated_tasks"] += int(item["generated_goals"])
        row["paid_bytes"] += int(item["paid_bytes"])
    for row in by_model.values():
        row["success_rate"] = round(row["on_time_tasks"] / float(row["generated_tasks"]), 9)
    total_on = sum(int(x["on_time"]) for x in results)
    total_completed = sum(int(x["completed"]) for x in results)
    total_tasks = sum(int(x["generated_goals"]) for x in results)
    complete = len(results) == 16 and not failures
    summary = {
        "schema": "stage5b-midtrain-dev-reused-summary/1", "started_utc": progress["started_utc"],
        "finished_utc": utc(), "root": str(root), "run_dir": str(run_dir),
        "identity_sha256": identity_sha,
        "status": "MIDTRAIN_EVALUATION_COMPLETE" if complete else "MIDTRAIN_EVALUATION_INCOMPLETE",
        "authorization": {"authority": "USER_EXPLICIT_INSTRUCTION",
                      "decision_ref": "USER_20261101_SIDECHAT_MIDTRAIN_EVAL",
                      "scope": "16 DEV_REUSED eval Worlds (2 live runs x 8 scenes), greedy actor"}, 
        "label": "MIDTRAIN_DEV_REUSED", "not_formal_validation": True, "not_overall_go": True,
        "holdout_sealed": True, "original_no_go_preserved": True, "goal_complete": False,
        "fixed_protocol": fixed, "greedy_actor": True, "requested_worlds": 16,
        "completed_worlds": len(results), "failed_worlds": len(failures),
        "claims_started": len(list((run_dir / "ledger" / "claims").glob("*.json"))),
        "model_loads": sum(1 for x in results if x.get("model_loaded")),
        "primary_metric": {"sum_on_time_tasks": total_on, "sum_completed_tasks": total_completed,
                           "sum_total_tasks": total_tasks,
                           "on_time_success_rate": round(total_on / float(total_tasks), 9) if total_tasks else None,
                           "completed_success_rate": round(total_completed / float(total_tasks), 9) if total_tasks else None},
        "per_model": list(by_model.values()), "per_world": results, "failures": failures,
        "note": ("Mid-training recovery_10 checkpoints (T1/91002 at 6,680, T2/91002 at 6,800) "
                 "on the eight used DEV scenes; both runs are still training toward 10,000, so this is neither "
                 "a converged result nor formal validation, and the sealed holdout/original NO_GO remain.")
    }
    dump(run_dir / "SUMMARY.json", summary)
    progress.update({"status": summary["status"], "updated_utc": summary["finished_utc"],
                     "primary_metric": summary["primary_metric"]})
    dump(run_dir / "progress.json", progress)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if not complete:
        raise RuntimeError("mid-train evaluation incomplete: %d/16 worlds, %d failures" % (len(results), len(failures)))

if __name__ == "__main__":
    main()
