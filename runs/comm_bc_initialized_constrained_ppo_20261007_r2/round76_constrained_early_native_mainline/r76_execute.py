"""Frozen real-training entry for ROUND76; imported only after the CLI permit gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import r76_contracts as contracts
import r76_runtime

HERE = Path(__file__).resolve().parent
R66_DIR = HERE.parent / "round66_bc_early_path_preserving_late_adaptation"
SEEDS = (11, 22, 33)
EPISODES_PER_SEED = 128
EPISODES_PER_UPDATE = 4
NORMAL_CAP = 0
RESERVE_CAP = 0
HARD_CAP = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    """Atomic-replace JSON.  File contents are fsynced; directory-entry durability is not claimed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_execution_permit(permit: dict[str, Any]) -> dict[str, Any]:
    attempt_root = (HERE / "raw" / "attempt_dirs").resolve()
    requested = Path(permit["resource_root"]).resolve()
    if requested != (HERE / "raw" / "runtime").resolve():
        raise RuntimeError("the R76 permit resource root is not the canonical runtime root")
    return {"attempt_root": attempt_root,
            "resource_root": Path(permit["resource_root"]).resolve()}


def _fresh_architecture_controller(native_module: Any, seed: int) -> Any:
    """Build the R66 partitioned architecture only; never replay INIT or restore ep0."""
    return native_module.r66.fresh_partitioned_controller(
        int(seed), ep0_state=None, verify_ep0=False)


def _validated_source_manifest(native_module: Any, r68_module: Any,
                               cfg: dict[str, Any]) -> dict[str, str]:
    """Resolve, hash, and freeze the actual upstream files bound to this runtime."""
    repo = contracts.RUN_ROOT.parent.parent
    paths = {
        "r71_trial_execute.py": str(
            contracts.RUN_ROOT / "round71_critic_initialization_calibration" /
            "r71_trial_execute.py"),
        "r71_runtime_binding.py": str(
            repo / "analysis" / "round71-critic-calibration-readiness" /
            "r71_runtime_binding.py"),
        "r68_execute.py": str(Path(r68_module.__file__).resolve()),
        "r75_frozen.py": str(
            contracts.RUN_ROOT / "round75_native_ppo_qualification" / "r75_frozen.py"),
        "v2_trainer.py": str(Path(native_module.trainer.__file__).resolve()),
        "r42_one_shot.py": str(
            contracts.RUN_ROOT.parent / "comm_one_shot_full_trace_ablation_20261004" /
            "r42_one_shot.py"),
    }
    return contracts.validate_source_manifest(paths, sha256_file=sha256_file)


def configure_r76_runtime(contract: dict[str, Any],
                          resource_contract_sha256: str) -> dict[str, Any]:
    if str(R66_DIR) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(R66_DIR))
    import r66_execute
    import r68_execute
    import r76_native
    import r76_guard

    raw_root = HERE / "raw"
    cfg = r66_execute.configure_and_inject(raw_root)
    r40 = cfg["runtime"]["r40_common"]
    common = cfg["common"]
    environment_caps = contract["resource_caps"]["environment_construction"]
    for holder, names in ((r40, ("NORMAL_CAP", "RESERVE_CAP", "HARD_CAP")),
                          (common, ("NORMAL_LAUNCHES", "RESERVE_LAUNCHES", "LAUNCH_CAP"))):
        for name, value in zip(names, (environment_caps["normal"],
                                       environment_caps["reserve"],
                                       environment_caps["hard"])):
            setattr(holder, name, value)
    r40.plan = lambda: {"normal_total": environment_caps["normal"],
                        "technical_reserve_max": environment_caps["reserve"],
                        "hard_cap": environment_caps["hard"],
                        "task_id": contracts.TASK_ID}
    resources = r76_guard.R76Resources.production_from_contract(contract)
    resources.verify_or_fail()
    cfg["r76_source_manifest"] = _validated_source_manifest(r76_native,
                                                            r68_execute, cfg)
    cfg["r76_controller_factory"] = lambda seed: _fresh_architecture_controller(
        r76_native, seed)
    identity = r76_runtime.install_r76_runtime_bindings(
        cfg, resources, r76_native, request_id=contracts.REQUEST_ID)
    cfg["r76_resource_contract_sha256"] = resource_contract_sha256
    cfg["r76_resource_contract"] = contract
    identity["environment_normal_cap"] = environment_caps["normal"]
    identity["environment_reserve_cap"] = environment_caps["reserve"]
    identity["environment_hard_cap"] = environment_caps["hard"]
    identity["central_resource_kind"] = resources.snapshots()["kind"]
    cfg["r76_runtime_identity"] = identity
    return cfg


def r76_fresh_controller(cfg: dict[str, Any], seed: int) -> tuple[Any, dict[str, Any]]:
    import r76_native

    source_path = contracts.R71_INIT_ROOT / f"v2_train_s{int(seed)}_init.pt"
    identity_path = contracts.R71_INIT_ROOT / f"r71_init_identity_s{int(seed)}.json"
    r76_runtime.validate_r71_init_source_file(source_path, seed=int(seed))
    registered_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    payload = r76_native.torch.load(source_path, map_location="cpu", weights_only=False)
    controller, report = r76_runtime.build_r76_controller_from_r71_init(
        payload, registered_identity, seed=int(seed),
        controller_factory=cfg.get(
            "r76_controller_factory",
            lambda value: _fresh_architecture_controller(r76_native, value)),
        module_sha=r76_native.module_sha,
        freeze_trunk_only=r76_native.freeze_trunk_only,
        source_file=source_path)
    report["identity_file"] = str(identity_path)
    report["identity_file_sha256"] = sha256_file(identity_path)
    report["training_version"] = contracts.R76_TRAINING_VERSION
    return controller, report


def _reference_identity(cfg: dict[str, Any], order: list[int]) -> dict[str, Any]:
    import r68_execute

    files = {}
    for scene in sorted(set(order)):
        path = r68_execute.R66_DIR / "raw" / "rows" / f"v2_r66_ref_s{scene}_s{scene}_a1.json"
        if not path.is_file():
            raise RuntimeError(f"R76 reference identity row is missing for scene {scene}")
        files[str(scene)] = sha256_file(path)
    return {"kind": "round68-reused-r66-immediate-references/1",
            "reference_row_sha256": files}


def _batch_path(seed: int, batch: int, phase: str,
                runtime_root: Path | None = None) -> Path:
    root = HERE / "raw" / "runtime" if runtime_root is None else Path(runtime_root)
    return root / f"r76_batch_s{int(seed)}_b{int(batch)}_{phase}.json"


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_durable_history(seed: int, state: dict[str, Any],
                            data_identity: dict[str, Any],
                            resource_contract_sha256: str,
                            rolling: Path,
                            runtime_root: Path | None = None) -> None:
    """Refuse to continue past a dangling update or an unverified durable cursor."""
    updates = int(state["updates"])
    rolling_sha = sha256_file(rolling) if rolling.is_file() else None
    previous_committed = None
    for batch in range(32):
        pre = _read_json_if_exists(_batch_path(seed, batch, "preupdate", runtime_root))
        ready = _read_json_if_exists(_batch_path(seed, batch, "ready", runtime_root))
        post = _read_json_if_exists(_batch_path(seed, batch, "postupdate", runtime_root))
        committed = _read_json_if_exists(_batch_path(seed, batch, "committed", runtime_root))
        if pre is None and ready is None and post is None and committed is None:
            if batch >= updates:
                break
            raise RuntimeError(f"R76 seed {seed}: durable history is missing committed batch {batch}")
        for name, record in (("preupdate", pre), ("ready", ready),
                             ("postupdate", post),
                             ("committed", committed)):
            if record is not None and (
                    int(record.get("seed", -1)) != int(seed)
                    or int(record.get("batch", -1)) != int(batch)
                    or record.get("data_identity") != data_identity
                    or str(record.get("resource_contract_sha256", ""))
                    != resource_contract_sha256):
                raise RuntimeError(
                    f"R76 seed {seed}: durable {name} record identity mismatch at batch {batch}")
        if batch < updates:
            if pre is None or ready is None or post is None or committed is None:
                raise RuntimeError(
                    f"R76 seed {seed}: batch {batch} is inside the rolling cursor but lacks a "
                    "pre/ready/post/committed chain")
            pre_state = pre.get("state", {})
            post_state = post.get("state", {})
            committed_state = committed.get("state", {})
            def state_without_rollout(value):
                return {key: item for key, item in value.items()
                        if key != "lambda_at_rollout"}

            if previous_committed is not None and (
                    pre.get("parameters_before_sha")
                    != previous_committed.get("parameters_after_sha")
                    or state_without_rollout(pre_state)
                    != state_without_rollout(previous_committed.get("state", {}))):
                raise RuntimeError(
                    f"R76 seed {seed}: durable state/parameter chain is broken at batch {batch}")
            pre_state_sha = contracts.sha256_text(pre_state)
            ready_state_sha = contracts.sha256_text(ready.get("state", {}))
            post_state_sha = contracts.sha256_text(post_state)
            committed_state_sha = contracts.sha256_text(committed_state)
            if int(pre_state.get("updates", -1)) != batch \
                    or int(pre_state.get("episodes_done", -1)) != batch * EPISODES_PER_UPDATE \
                    or pre.get("state_sha256") != pre_state_sha \
                    or ready.get("state_sha256") != ready_state_sha \
                    or ready_state_sha != pre_state_sha \
                    or ready.get("parameters_sha") != pre.get("parameters_before_sha") \
                    or int(post_state.get("updates", -1)) != batch + 1 \
                    or int(post_state.get("episodes_done", -1)) \
                    != (batch + 1) * EPISODES_PER_UPDATE \
                    or post.get("parameters_before_sha") != pre.get("parameters_before_sha") \
                    or post.get("parameters_after_sha") == pre.get("parameters_before_sha") \
                    or committed_state != post_state \
                    or post.get("state_sha256") != post_state_sha \
                    or committed.get("state_sha256") != committed_state_sha \
                    or committed_state_sha != post_state_sha \
                    or committed.get("parameters_after_sha") != post.get("parameters_after_sha"):
                raise RuntimeError(
                    f"R76 seed {seed}: durable state/parameter chain is broken at batch {batch}")
            previous_committed = committed
            if batch == updates - 1:
                if committed.get("rolling_checkpoint_sha256") != rolling_sha:
                    raise RuntimeError(
                        f"R76 seed {seed}: rolling checkpoint does not match the last "
                        f"committed batch {batch}")
            if batch == 0:
                gate = _read_json_if_exists(
                    (HERE / "raw" / "runtime" if runtime_root is None else runtime_root)
                    / f"r76_first_batch_gate_s{seed}.json")
                if gate is None or gate.get("status") != "passed":
                    raise RuntimeError(
                        f"R76 seed {seed}: durable first-batch gate is missing before resume")
        else:
            if pre is not None or ready is not None or post is not None \
                    or committed is not None:
                raise RuntimeError(
                    f"R76 seed {seed}: dangling pre/post batch state found at batch {batch}; "
                    "manual reconciliation is required")
    if int(state["episodes_done"]) != updates * EPISODES_PER_UPDATE:
        raise RuntimeError("R76 durable state has a mismatched episode/update cursor")


def _persist_episode(cfg: dict[str, Any], row: dict[str, Any]) -> None:
    import r76_episode_protocol as protocol

    row_path = Path(row["artefacts"]["row"])
    tape_path = Path(row["artefacts"]["tape"])
    protocol.write_row(row_path, row)
    protocol.write_tape(tape_path, row["collector"].records)


def _reload_episode(cfg: dict[str, Any], row_path: Path, tape_path: Path, *,
                    seed: int, ep: int, scenario: int, lam: float) -> dict[str, Any]:
    import r76_episode_protocol as protocol

    row = protocol.read_row(row_path)
    tape = protocol.read_tape(tape_path)
    for field, expected in (("training_seed", int(seed)),
                            ("episode_index", int(ep)),
                            ("scenario_seed", int(scenario)),
                            ("lambda", float(lam))):
        got = row.get(field)
        same = int(got) == expected if isinstance(expected, int) and field != "lambda" \
            else float(got) == expected
        if not same:
            raise RuntimeError(
                f"R76 reload identity mismatch for {field}: row={got!r} "
                f"expected={expected!r}")
    return protocol.rebuild_episode(row, tape)


def _run_episode(cfg: dict[str, Any], controller: Any, *, seed: int, ep: int,
                 scenario: int, lam: float) -> dict[str, Any]:
    import r68_execute
    from r42_one_shot import OneShotSendSkipCollector

    runner = cfg["runner"]
    collector = OneShotSendSkipCollector(controller, greedy=False)
    tag = f"r76_train_s{int(seed)}_e{int(ep)}"
    logical = f"r76_new_training_s{int(seed)}_e{int(ep)}"
    action_seed = runner.action_seed(int(seed), int(ep))
    metadata = {"arm": "R76_NATIVE_EARLY_MAINLINE", "training_seed": int(seed),
                "episode_index": int(ep), "scenario_seed": int(scenario),
                "lambda_at_rollout": float(lam)}

    def run():
        cfg["r76"]["resources"].charge_env_construction(logical)
        row = runner._run_episode(tag, "train", int(scenario), policy="learned",
                                  attach=collector.policy(), controller=controller,
                                  collector=collector, action_rng_seed=action_seed,
                                  tag=tag, attempt=r68_execute._active_attempt(cfg))
        row["collector"] = collector
        row["training_seed"] = int(seed)
        row["episode_index"] = int(ep)
        row["scenario_seed"] = int(scenario)
        row["lambda"] = float(lam)
        _persist_episode(cfg, row)
        return row

    row = r68_execute._r68_atomic_run(cfg, logical, metadata, run,
                                  detail=lambda value:
                                  f"on_time={value['on_time']} bytes={value['spent_bytes']}")
    return row


def _first_batch_gate(cfg: dict[str, Any], controller: Any, optimizer: Any,
                      state: dict[str, Any], report: dict[str, Any], episodes: list[dict[str, Any]],
                      before_sha: str) -> dict[str, Any]:
    import r76_native

    mask = r76_native.assert_mask(controller)
    checks = {
        "episode_count_is_four": len(episodes) == EPISODES_PER_UPDATE,
        "episode_cursor_is_four": int(state["episodes_done"]) == EPISODES_PER_UPDATE,
        "update_cursor_is_one": int(state["updates"]) == 1,
        "optimizer_stepped": int(report.get("optimizer_steps", 0)) > 0,
        "parameters_moved": controller.parameter_summary()["parameter_sha256"] != before_sha,
        "trunk_frozen": mask["frozen_parameter_object_count"] == 4
                        and mask["trainable_parameter_object_count"] == 8,
        "resource_reconciled": bool(cfg["r76"]["resources"].verify_or_fail()["consistent"]),
    }
    return {"kind": "round76-first-batch-gate/1", "status":
            "passed" if all(checks.values()) else "failed", "checks": checks,
            "report": report}


def train_seed(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    import r68_execute
    import r76_native

    common = cfg["common"]
    trainer = cfg["trainer"]
    order = [int(value) for value in common.scenario_order()]
    ledger = r68_execute.R68Ledger(Path(cfg["raw_root"]))
    data_identity = {
        "data_order_seed": int(common.DATA_ORDER_SEED),
        "b0_bytes": float(common.B0_BYTES),
        "episodes_per_seed": int(common.EPISODES_PER_SEED),
        "scenario_order_sha256": contracts.sha256_text(order),
        "seeds": list(SEEDS),
    }
    references = r68_execute.reuse_r66_references(cfg, order)
    reference_identity = _reference_identity(cfg, order)
    resource_contract_sha256 = str(cfg["r76_resource_contract_sha256"])
    config_identity = dict(cfg["r76_runtime_identity"]["live_config"])
    cfg["r76_reference_identity"] = reference_identity
    cfg["r76_resume_context"] = {
        "seed": int(seed), "data_identity": dict(data_identity),
        "reference_identity": dict(reference_identity),
        "config_identity": dict(config_identity),
        "resource_contract_sha256": resource_contract_sha256,
    }
    cfg["r76_data_identity"] = data_identity
    rolling = Path(common.CHECKPOINTS) / f"r76_train_s{int(seed)}.pt"
    ep128 = Path(common.CHECKPOINTS) / f"r76_train_s{int(seed)}_ep128.pt"
    if rolling.exists():
        controller, optimizer, state = trainer.load_checkpoint(rolling)
        init_report = {"resumed": True, "rolling_checkpoint": str(rolling),
                       "rolling_checkpoint_sha256": sha256_file(rolling)}
    else:
        controller, init_report = r76_fresh_controller(cfg, seed)
        optimizer = r76_native.build_optimizer(controller, common.LEARNING_RATE)
        state = r76_runtime.build_r76_state(
            seed, data_order_seed=common.DATA_ORDER_SEED, b0_bytes=common.B0_BYTES)
    _verify_durable_history(seed, state, data_identity, resource_contract_sha256,
                            rolling)
    resources = cfg["r76"]["resources"]
    resources.ensure_trunk_meter(controller, what_prefix=f"r76_train_s{int(seed)}")
    for batch in range(int(state["updates"]), EPISODES_PER_SEED // EPISODES_PER_UPDATE):
        if int(state["episodes_done"]) != batch * EPISODES_PER_UPDATE:
            raise RuntimeError("R76 episode cursor is not aligned with the update cursor")
        lam = float(state["lambda"])
        state["lambda_at_rollout"] = lam
        before_sha = controller.parameter_summary()["parameter_sha256"]
        _write_json(_batch_path(seed, batch, "preupdate"),
                    {"kind": "round76-training-batch-preupdate/1", "seed": int(seed),
                     "batch": int(batch), "phase": "preupdate", "lambda_at_rollout": lam,
                     "parameters_before_sha": before_sha, "state": state,
                     "state_sha256": contracts.sha256_text(state),
                     "data_identity": data_identity,
                     "resource_contract_sha256": resource_contract_sha256})
        episodes = []
        for offset in range(EPISODES_PER_UPDATE):
            ep = int(state["episodes_done"]) + offset
            scenario = order[ep]
            logical = f"r76_new_training_s{int(seed)}_e{ep}"
            if ledger.is_complete(logical):
                attempt = r68_execute._qualified_attempt(cfg["runtime"]["r40_common"], logical)
                paths = common.cell_paths("train", scenario, attempt,
                                          tag=f"r76_train_s{seed}_e{ep}")
                episodes.append(_reload_episode(cfg, paths["row"], paths["tape"], seed=seed,
                                                ep=ep, scenario=scenario, lam=lam))
                continue
            episodes.append(_run_episode(cfg, controller, seed=seed, ep=ep,
                                         scenario=scenario, lam=lam))
        episode_identities = []
        for row in episodes:
            episode_identities.append({
                "seed": int(row["training_seed"]),
                "episode": int(row["episode_index"]),
                "scenario": int(row["scenario_seed"]),
                "row_sha256": sha256_file(row["artefacts"]["row"]),
                "tape_sha256": sha256_file(row["artefacts"]["tape"]),
            })
        _write_json(_batch_path(seed, batch, "ready"),
                    {"kind": "round76-training-batch-ready-record/1",
                     "seed": int(seed), "batch": int(batch), "lambda_at_rollout": lam,
                     "parameters_sha": before_sha, "state": state,
                     "state_sha256": contracts.sha256_text(state),
                     "episodes": episode_identities,
                     "data_identity": data_identity,
                     "resource_contract_sha256": resource_contract_sha256})
        report = trainer.ppo_update(controller, optimizer, episodes, update_index=batch)
        g_record = trainer.batch_g_hat(episodes, references)
        dual_record = trainer.dual_update(state["lambda"], g_record["g_hat"])
        state["lambda"] = dual_record["lambda_next"]
        state["lambda_history"].append({**dual_record, "batch": batch})
        state["updates"] += 1
        state["optimizer_steps"] += int(report.get("optimizer_steps", 0))
        state["episodes_done"] += EPISODES_PER_UPDATE
        after_sha = controller.parameter_summary()["parameter_sha256"]
        if after_sha == before_sha:
            raise RuntimeError(f"R76 seed {seed} batch {batch}: parameters did not move")
        record = {"kind": "round76-training-batch-after-record/1", "seed": int(seed),
                  "batch": int(batch), "episodes_done_after": int(state["episodes_done"]),
                  "optimizer_steps_after": int(state["optimizer_steps"]),
                  "lambda_at_rollout": lam, "update": report, "constraint": g_record,
                  "dual": dual_record, "parameters_before_sha": before_sha,
                  "parameters_after_sha": after_sha, "init_report": init_report}
        record["data_identity"] = data_identity
        record["resource_contract_sha256"] = resource_contract_sha256
        record["state"] = state
        record["state_sha256"] = contracts.sha256_text(state)
        _write_json(Path(common.CHECKPOINTS) /
                    f"r76_batch_s{seed}_b{batch}_after.json", record)
        _write_json(_batch_path(seed, batch, "postupdate"), record)
        trainer.save_checkpoint(controller, optimizer, state, rolling)
        _write_json(_batch_path(seed, batch, "committed"),
                    {"kind": "round76-training-batch-committed/1", "seed": int(seed),
                     "batch": int(batch), "lambda_at_rollout": lam,
                     "parameters_after_sha": after_sha, "state": state,
                     "state_sha256": contracts.sha256_text(state),
                     "rolling_checkpoint_sha256": sha256_file(rolling),
                     "data_identity": data_identity,
                     "resource_contract_sha256": resource_contract_sha256})
        if batch == 0:
            gate = _first_batch_gate(cfg, controller, optimizer, state, report,
                                     episodes, before_sha)
            _write_json(HERE / "raw" / "runtime" / f"r76_first_batch_gate_s{seed}.json", gate)
            if gate["status"] != "passed":
                raise RuntimeError(f"R76 seed {seed}: first-batch gate failed")
    trainer.save_checkpoint(controller, optimizer, state, ep128)
    return {"seed": int(seed), "status": "ep128_complete", "episodes_done": int(state["episodes_done"]),
            "optimizer_steps": int(state["optimizer_steps"]), "lambda": float(state["lambda"])}


def run_real_training(args: Any, permit: dict[str, Any]) -> int:
    contract = contracts.validate_real_execution_contract(
        permit["contract_path"], source_head=args.source_head,
        driver_path=HERE / "r76_execute.py")
    if permit["real_contract_sha256"] != sha256_file(Path(permit["contract_path"])):
        raise RuntimeError("the real-training permit contract hash does not match the contract file")
    identity = validate_execution_permit(permit)
    attempt_dir = Path(args.attempt_dir).resolve()
    if attempt_dir != identity["attempt_root"] and \
            identity["attempt_root"] not in attempt_dir.parents:
        raise RuntimeError("the R76 attempt directory is outside the private attempt root")
    cfg = configure_r76_runtime(contract, permit["real_contract_sha256"])
    results = [train_seed(cfg, seed) for seed in SEEDS]
    _write_json(HERE / "raw" / "runtime" / "r76_training_results.json",
                {"kind": "round76-training-results/1", "results": results})
    return 0
