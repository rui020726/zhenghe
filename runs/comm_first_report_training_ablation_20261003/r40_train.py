"""ROUND40 batch driver: reuses Stage C PPO math with write-once batch records."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

import r40_common as r40

common: Any = None
runner: Any = None
trainer: Any = None


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item, key=lambda entry: repr(entry)):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(type(item).__name__.encode("utf-8"))
            digest.update(repr(item).encode("utf-8"))
            digest.update(b";")

    visit(value)
    return digest.hexdigest()


def _module_digests(controller: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, module in (("trunk", controller.inner.trunk),
                         ("action_head", controller.inner.action_head),
                         ("task_value_head", controller.inner.value_head),
                         ("cost_value_head", controller.cost_head)):
        digest = hashlib.sha256()
        for parameter_name, tensor in sorted(module.named_parameters()):
            digest.update(parameter_name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def _milestone_path(training_seed: int, episodes: int,
                    freeze: dict[str, Any]) -> Path:
    path = common.CHECKPOINTS / f"v2_train_s{training_seed}_ep{episodes}.pt"
    if episodes == 0 and path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("freeze") != freeze:
            freeze_key = hashlib.sha256(json.dumps(
                freeze, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest()[:12]
            path = common.CHECKPOINTS / (
                f"v2_train_s{training_seed}_ep0_r40f{freeze_key}.pt"
            )
    return path


def _rolling_path(training_seed: int) -> Path:
    return common.CHECKPOINTS / f"v2_train_s{training_seed}.pt"


def _save_milestone(controller: Any, optimizer: torch.optim.Optimizer,
                    state: dict[str, Any], training_seed: int, episodes: int,
                    freeze: dict[str, Any]) -> Path:
    path = _milestone_path(training_seed, episodes, freeze)
    if path.exists():
        restored, restored_state, identity = trainer.load_checkpoint_read_only(path,
                                                                                freeze=freeze)
        expected = controller.parameter_summary()["parameter_sha256"]
        actual = restored.parameter_summary()["parameter_sha256"]
        _assert_same = (
            int(restored_state.get("training_seed", -1)) == int(training_seed)
            and int(restored_state.get("episodes_done", -1)) == int(episodes)
            and int(restored_state.get("updates", -1)) == int(state["updates"])
            and int(restored_state.get("optimizer_steps", -1)) == int(state["optimizer_steps"])
            and float(restored_state.get("lambda", float("nan"))) == float(state["lambda"])
            and actual == expected
            and identity.get("read_only_for_round40") is True
        )
        if not _assert_same:
            raise RuntimeError(f"immutable milestone differs from current R40 state: {path}")
        return path

    temporary = path.with_name(path.name + f".{os.getpid()}.milestone-tmp")
    if temporary.exists():
        raise RuntimeError(f"stale milestone temporary requires inspection: {temporary}")
    trainer.save_checkpoint(controller, optimizer, state, temporary, freeze=freeze)
    try:
        # Windows os.rename refuses to replace an existing destination, preserving milestones.
        os.rename(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        return _save_milestone(controller, optimizer, state, training_seed, episodes, freeze)
    return path


def _batch_record_path(training_seed: int, batch_index: int) -> Path:
    stem = f"r40_batch_s{training_seed}_b{batch_index}"
    existing = list(common.CHECKPOINTS.glob(stem + "_attempt*.json"))
    attempt = len(existing) + 1
    path = common.CHECKPOINTS / f"{stem}_attempt{attempt:02d}.json"
    while path.exists():
        attempt += 1
        path = common.CHECKPOINTS / f"{stem}_attempt{attempt:02d}.json"
    return path


def _batch_commit_path(training_seed: int, batch_index: int) -> Path:
    return common.CHECKPOINTS / f"r40_batch_commit_s{training_seed}_b{batch_index}.json"


def _preupdate_checkpoint_path(training_seed: int, batch_index: int) -> Path:
    return common.CHECKPOINTS / f"v2_train_s{training_seed}_before_b{batch_index}.pt"


def _postupdate_checkpoint_path(training_seed: int, batch_index: int) -> Path:
    return common.CHECKPOINTS / f"v2_train_s{training_seed}_after_b{batch_index}.pt"


def _write_immutable_snapshot(path: Path, controller: Any, optimizer: torch.optim.Optimizer,
                              state: dict[str, Any], freeze: dict[str, Any]) -> str:
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        restored, restored_state, identity = trainer.load_checkpoint_read_only(path,
                                                                               freeze=freeze)
        if (restored.parameter_summary()["parameter_sha256"]
                != controller.parameter_summary()["parameter_sha256"]
                or int(restored_state.get("episodes_done", -1)) != int(state["episodes_done"])
                or int(restored_state.get("updates", -1)) != int(state["updates"])
                or int(restored_state.get("optimizer_steps", -1)) != int(state["optimizer_steps"])
                or float(restored_state.get("lambda", float("nan"))) != float(state["lambda"])
                or _state_hash(payload.get("optimizer")) != _state_hash(optimizer.state_dict())
                or identity.get("read_only_for_round40") is not True):
            raise RuntimeError(f"immutable checkpoint snapshot differs from current state: {path}")
        return _sha256(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.snapshot-tmp")
    if temporary.exists():
        raise RuntimeError(f"stale checkpoint temporary requires inspection: {temporary}")
    trainer.save_checkpoint(controller, optimizer, state, temporary, freeze=freeze)
    try:
        os.rename(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        return _write_immutable_snapshot(path, controller, optimizer, state, freeze)
    return _sha256(path)


def _save_preupdate_checkpoint(controller: Any, optimizer: torch.optim.Optimizer,
                               state: dict[str, Any], training_seed: int,
                               batch_index: int, freeze: dict[str, Any]) -> tuple[Path, str]:
    path = _preupdate_checkpoint_path(training_seed, batch_index)
    if int(state["episodes_done"]) != batch_index * common.EPISODES_PER_UPDATE:
        raise RuntimeError("pre-update checkpoint cursor is not at a batch boundary")
    return path, _write_immutable_snapshot(path, controller, optimizer, state, freeze)


def _write_batch_commit(training_seed: int, batch_index: int, record_path: Path,
                        checkpoint: Path, postupdate_checkpoint: Path,
                        state: dict[str, Any],
                        parameter_identity: dict[str, Any]) -> None:
    path = _batch_commit_path(training_seed, batch_index)
    receipt = {
        "kind": "round40-batch-checkpoint-commit/1",
        "training_seed": int(training_seed),
        "batch": int(batch_index),
        "episodes_done": int(state["episodes_done"]),
        "updates": int(state["updates"]),
        "optimizer_steps": int(state["optimizer_steps"]),
        "lambda": float(state["lambda"]),
        "training_version": state["training_version"],
        "comm_action_eligibility": state["comm_action_eligibility"],
        "batch_record": record_path.name,
        "batch_record_sha256": _sha256(record_path),
        "rolling_checkpoint": checkpoint.name,
        "rolling_checkpoint_sha256": _sha256(checkpoint),
        "postupdate_checkpoint": postupdate_checkpoint.name,
        "postupdate_checkpoint_sha256": _sha256(postupdate_checkpoint),
        "controller_parameters": parameter_identity,
    }
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != receipt:
            raise RuntimeError(f"batch commit receipt conflicts with checkpoint state: {path}")
        return
    _write_json_exclusive(path, receipt)


def _recover_final_batch_commit(training_seed: int, state: dict[str, Any],
                                controller: Any, checkpoint: Path) -> None:
    updates = int(state["updates"])
    if updates <= 0:
        return
    batch_index = updates - 1
    receipt_path = _batch_commit_path(training_seed, batch_index)
    if receipt_path.exists():
        return
    parameter = controller.parameter_summary()
    candidates = []
    for record_path in sorted(common.CHECKPOINTS.glob(
            f"r40_batch_s{training_seed}_b{batch_index}_attempt*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (record.get("episodes_done_after") == int(state["episodes_done"])
                and record.get("controller_parameters_after", {}).get("parameter_sha256")
                == parameter["parameter_sha256"]):
            candidates.append(record_path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"cannot prove checkpoint commit for seed {training_seed} batch {batch_index}; "
            f"matching immutable records={len(candidates)}"
        )
    record = json.loads(candidates[0].read_text(encoding="utf-8"))
    postupdate = common.CHECKPOINTS / record["postupdate_checkpoint"]
    if _sha256(postupdate) != record["postupdate_checkpoint_sha256"]:
        raise RuntimeError("cannot prove the post-update recovery snapshot")
    _write_batch_commit(training_seed, batch_index, candidates[0], checkpoint, postupdate,
                        state, parameter)


def _latest_resume_source(training_seed: int, freeze: dict[str, Any]) -> Path | None:
    rolling = _rolling_path(training_seed)
    candidates = ([rolling] if rolling.exists() else []) + sorted(
        common.CHECKPOINTS.glob(f"v2_train_s{training_seed}_after_b*.pt"),
        key=lambda path: int(path.stem.rsplit("b", 1)[-1]),
    )
    valid: list[tuple[int, int, Path, str]] = []
    for priority, path in enumerate(candidates):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            if path == rolling:
                # A later immutable post-update snapshot can still establish the committed
                # cursor; this path is retained and reported as a damaged rolling cache.
                continue
            raise RuntimeError(f"immutable post-update checkpoint is unreadable: {path}") from error
        state = payload.get("training_state")
        if (payload.get("training_version") != trainer.TRAINING_VERSION
                or payload.get("comm_action_eligibility") != "first-report-only/1"
                or not isinstance(state, dict)
                or state.get("training_version") != trainer.TRAINING_VERSION
                or state.get("comm_action_eligibility") != "first-report-only/1"
                or payload.get("freeze") != freeze):
            raise RuntimeError(f"checkpoint is outside the ROUND40 resume identity: {path}")
        controller = trainer._restore_controller(payload, state)
        parameter = controller.parameter_summary()["parameter_sha256"]
        valid.append((int(state["updates"]), priority, path, parameter))
    if not valid:
        return None
    max_update = max(item[0] for item in valid)
    latest = [item for item in valid if item[0] == max_update]
    if len({item[3] for item in latest}) != 1:
        raise RuntimeError("ROUND40 checkpoints conflict at the same committed update count")
    # Prefer rolling when it agrees with the newest immutable post-update snapshot.
    latest.sort(key=lambda item: (item[2] != rolling, item[1]))
    return latest[0][2]


def _save_progress(training_seed: int, state: dict[str, Any], *, status: str,
                   parameter_identity: dict[str, Any]) -> None:
    path = common.OUTPUT_DIR / "training_progress" / (
        f"seed{training_seed}_ep{int(state['episodes_done']):03d}_"
        f"u{int(state['updates']):02d}_{status}.json"
    )
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior.get("parameter_identity") != parameter_identity:
            raise RuntimeError(f"training progress receipt exists with a different state: {path}")
        return
    _write_json_exclusive(path, {
        "training_seed": int(training_seed),
        "episodes_done": int(state["episodes_done"]),
        "updates": int(state["updates"]),
        "optimizer_steps": int(state["optimizer_steps"]),
        "lambda": float(state["lambda"]),
        "training_version": state["training_version"],
        "comm_action_eligibility": state["comm_action_eligibility"],
        "parameter_identity": parameter_identity,
        "status": status,
    })


def train_seed(training_seed: int, *, episodes: int, freeze: dict[str, Any],
               reference_cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if training_seed not in common.TRAIN_SEEDS:
        raise ValueError(f"unregistered training seed {training_seed}")
    if episodes not in (4, common.EPISODES_PER_SEED) and episodes % common.EPISODES_PER_UPDATE:
        raise ValueError("ROUND40 training limit must end on a 4-episode PPO batch")
    if not isinstance(reference_cache, dict):
        raise TypeError("immediate cache must be a scenario-keyed dictionary")

    checkpoint = _rolling_path(training_seed)
    milestone_paths: dict[str, str] = {}
    resume_source = _latest_resume_source(training_seed, freeze)
    if resume_source is not None:
        controller, optimizer, state = trainer.load_checkpoint(resume_source, freeze=freeze)
        if resume_source != checkpoint:
            # A completed immutable snapshot is the source of truth if a crash happened after
            # its save and before the rolling checkpoint/commit receipt.
            trainer.save_checkpoint(controller, optimizer, state, checkpoint, freeze=freeze)
        _recover_final_batch_commit(training_seed, state, controller, checkpoint)
        print(f"R40 resumed seed {training_seed} at episode {state['episodes_done']} "
              f"lambda={state['lambda']:.4f} updates={state['updates']}", flush=True)
    else:
        from v2_controller import fresh_controller, parameter_identity

        controller = fresh_controller(training_seed)
        optimizer = torch.optim.Adam(controller.parameters, lr=common.LEARNING_RATE)
        state = trainer.fresh_training_state(training_seed)
        initial_milestone = _save_milestone(controller, optimizer, state, training_seed, 0, freeze)
        milestone_paths["0"] = str(initial_milestone)
        print(f"R40 seed {training_seed}: fresh Controller({training_seed}); "
              "no old weights loaded", flush=True)

    if int(state["training_seed"]) != int(training_seed):
        raise RuntimeError("rolling checkpoint seed does not match this training job")
    if state.get("comm_action_eligibility") != "first-report-only/1":
        raise RuntimeError("rolling checkpoint is missing ROUND40 action eligibility")
    order = common.scenario_order()
    needed_scenarios = set(order[:episodes])
    missing = sorted(needed_scenarios - set(reference_cache))
    if missing:
        raise RuntimeError(f"immediate reference cache is incomplete: {missing}")
    references = {int(scenario): float(row["on_time"])
                  for scenario, row in reference_cache.items()}

    before = parameter_identity(controller)
    module_before = _module_digests(controller)
    history: list[dict[str, Any]] = []
    started = time.time()

    # Record only immutable milestones whose content carries the current freeze identity.
    for milestone in (0, 64, 128):
        if milestone > int(state["episodes_done"]):
            continue
        milestone_path = _milestone_path(training_seed, milestone, freeze)
        if milestone_path.is_file():
            milestone_paths[str(milestone)] = str(milestone_path)
    # Repair only a missing ep64/ep128 immutable copy from a same-version rolling checkpoint.
    if int(state["episodes_done"]) in common.MILESTONE_EPISODES:
        milestone_path = _save_milestone(controller, optimizer, state, training_seed,
                                         int(state["episodes_done"]), freeze)
        milestone_paths[str(int(state["episodes_done"]))] = str(milestone_path)

    from r40_run import run_training_episode

    while int(state["episodes_done"]) < episodes:
        batch_index = int(state["updates"])
        if int(state["episodes_done"]) != batch_index * common.EPISODES_PER_UPDATE:
            raise RuntimeError("episode cursor and optimizer update index are inconsistent")
        preupdate_checkpoint, preupdate_sha = _save_preupdate_checkpoint(
            controller, optimizer, state, training_seed, batch_index, freeze)
        optimizer_state_before_hash = _state_hash(optimizer.state_dict())
        lambda_at_rollout = float(state["lambda"])
        episodes_in_batch: list[dict[str, Any]] = []
        for _ in range(common.EPISODES_PER_UPDATE):
            episode_index = int(state["episodes_done"])
            if episode_index >= episodes:
                break
            scenario = int(order[episode_index])
            if scenario not in references:
                raise RuntimeError(f"scenario {scenario} has no paired immediate reference")
            episode = run_training_episode(
                common, runner, controller,
                training_seed=training_seed,
                episode_index=episode_index,
                scenario_seed=scenario,
                lam=lambda_at_rollout,
            )
            episodes_in_batch.append(episode)
            state["episodes_done"] += 1
            history.append({
                "episode_index": episode_index,
                "scenario": scenario,
                "on_time": episode["on_time"],
                "spent_bytes": episode["spent_bytes"],
                "emissions": episode["emissions"],
                "free_first_reports": (episode.get("collector_summary") or {}).get(
                    "eligibility_counts", {}).get("free_first_reports"),
                "fixed_hold_refreshes": (episode.get("collector_summary") or {}).get(
                    "eligibility_counts", {}).get("fixed_hold_refreshes"),
            })
            print(f"R40 seed {training_seed} episode {episode_index}: scenario {scenario} "
                  f"on_time={episode['on_time']} bytes={episode['spent_bytes']} "
                  f"lambda={lambda_at_rollout:.4f}", flush=True)
        if len(episodes_in_batch) != common.EPISODES_PER_UPDATE:
            raise RuntimeError("ROUND40 unexpectedly ended with a partial PPO batch")

        update_report = trainer.ppo_update(controller, optimizer, episodes_in_batch,
                                           update_index=batch_index)
        g_record = trainer.batch_g_hat(episodes_in_batch, references)
        dual_record = trainer.dual_update(state["lambda"], g_record["g_hat"])
        state["lambda"] = dual_record["lambda_next"]
        state["lambda_history"].append({
            "batch": batch_index,
            "g_hat": dual_record["g_hat"],
            "lambda_before": dual_record["lambda_before"],
            "lambda_next": dual_record["lambda_next"],
        })
        state["updates"] += 1
        state["optimizer_steps"] += int(update_report.get("optimizer_steps", 0))
        after = parameter_identity(controller)
        if after["parameter_sha256"] == before["parameter_sha256"]:
            raise RuntimeError(f"seed {training_seed} batch {batch_index}: parameters did not move")

        batch_record = {
            "kind": "round40-training-batch-record/1",
            "batch": batch_index,
            "training_seed": int(training_seed),
            "episodes": [int(row["episode_index"]) for row in episodes_in_batch],
            "scenarios": [int(row["scenario_seed"]) for row in episodes_in_batch],
            "episode_artifacts": [
                {
                    "episode_index": int(row["episode_index"]),
                    "scenario_seed": int(row["scenario_seed"]),
                    "attempt": int(row["attempt"]),
                    "row_path": row["artefacts"]["row"],
                    "row_sha256": _sha256(Path(row["artefacts"]["row"])),
                    "tape_path": row["artefacts"]["tape"],
                    "tape_sha256": _sha256(Path(row["artefacts"]["tape"])),
                }
                for row in episodes_in_batch
            ],
            "lambda_at_rollout": lambda_at_rollout,
            "update": update_report,
            "constraint": g_record,
            "dual": dual_record,
            "controller_parameters_before": before,
            "controller_parameters_after": after,
            "preupdate_checkpoint": preupdate_checkpoint.name,
            "preupdate_checkpoint_sha256": preupdate_sha,
            "optimizer_state_sha256_before": optimizer_state_before_hash,
            "episodes_done_after": int(state["episodes_done"]),
            "updates_after": int(state["updates"]),
            "optimizer_steps_after": int(state["optimizer_steps"]),
            "postupdate_checkpoint": _postupdate_checkpoint_path(training_seed,
                                                                   batch_index).name,
            "controller_module_digests_before": module_before,
            "controller_module_digests_after": _module_digests(controller),
            "frozen_checkpoints": {
                str(path.relative_to(common.ROOT)): common.sha256_file(path)
                for path in common.CHECKPOINT_FILES
            },
            "optimizer_parameter_shapes": [
                list(parameter.shape) for group in optimizer.param_groups
                for parameter in group["params"]
            ],
            "controller_parameter_shapes": [list(parameter.shape)
                                            for parameter in controller.parameters],
            "training_version": state["training_version"],
            "comm_action_eligibility": state["comm_action_eligibility"],
            "record_status": "computed_before_checkpoint_commit",
        }
        batch_record_path = _batch_record_path(training_seed, batch_index)
        _write_json_exclusive(batch_record_path, batch_record)
        postupdate_checkpoint = _postupdate_checkpoint_path(training_seed, batch_index)
        _write_immutable_snapshot(postupdate_checkpoint, controller, optimizer, state, freeze)
        # The immutable post-update checkpoint is the recovery commit point. The rolling copy
        # and receipt follow; a prior batch record is never overwritten on recovery.
        trainer.save_checkpoint(controller, optimizer, state, checkpoint, freeze=freeze)
        _write_batch_commit(training_seed, batch_index, batch_record_path, checkpoint,
                            postupdate_checkpoint, state, after)
        if int(state["episodes_done"]) in common.MILESTONE_EPISODES:
            milestone_path = _save_milestone(controller, optimizer, state, training_seed,
                                             int(state["episodes_done"]), freeze)
            milestone_paths[str(int(state["episodes_done"]))] = str(milestone_path)
        _save_progress(training_seed, state, status="batch_checkpoint_committed",
                       parameter_identity=after)
        before = after
        module_before = _module_digests(controller)
        print(f"R40 seed {training_seed} batch {batch_index}: "
              f"samples={update_report['samples']} "
              f"optimizer_steps={update_report['optimizer_steps']} "
              f"g_hat={dual_record['g_hat']:+.4f} "
              f"lambda {dual_record['lambda_before']:.4f} -> "
              f"{dual_record['lambda_next']:.4f}", flush=True)

    final_identity = parameter_identity(controller)
    _save_progress(training_seed, state,
                   status="first_four_gate_ready" if episodes == 4 else "requested_limit_complete",
                   parameter_identity=final_identity)
    return {
        "training_seed": int(training_seed),
        "episodes_done": int(state["episodes_done"]),
        "updates": int(state["updates"]),
        "optimizer_steps": int(state["optimizer_steps"]),
        "lambda": float(state["lambda"]),
        "lambda_history": state["lambda_history"],
        "seconds": round(time.time() - started, 2),
        "checkpoint": str(checkpoint),
        "controller_parameters": final_identity,
        "history": history,
        "training_version": state["training_version"],
        "comm_action_eligibility": state["comm_action_eligibility"],
    }


__all__ = ["train_seed"]
