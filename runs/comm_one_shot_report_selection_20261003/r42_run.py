"""ROUND42 runner adapters over the frozen ROUND40 environment/training implementation."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

RUN_DIR = Path(__file__).resolve().parent
ROUND42_ROOT = RUN_DIR.parents[1]
WORKSPACE_ROOT = ROUND42_ROOT.parent
INPUT_ROOT = WORKSPACE_ROOT / "zhenghe"
ROUND40_ROOT = WORKSPACE_ROOT / "zhenghe-round40"
ROUND40_OUTPUT = ROUND40_ROOT / "runs" / "comm_first_report_training_ablation_20261003"
ROUND41_OUTPUT = ROUND42_ROOT / "runs" / "comm_temporal_decoder_diagnostic_20261003"
SEEDS = (11, 22, 33)
DEV_SCENARIOS = tuple(range(1_301_000, 1_301_016))
RULE_ARMS = ("late_discovery_gate", "always_hold")

_RUNTIME: dict[str, Any] | None = None
_ACTIVE_CLAIM: Path | None = None
_ACTIVE_COLLECTOR: Any = None
_LAST_RUN: Any = None
_OPTIMIZER_STEP_CALLS = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_module_path(module: Any, expected: Path, label: str) -> None:
    actual = Path(module.__file__).resolve() if getattr(module, "__file__", None) else None
    if actual != expected.resolve():
        raise RuntimeError(f"runtime import provenance mismatch for {label}: {actual} != {expected}")


def _dependency_path(root: Path, relative: str) -> Path:
    direct = root / relative
    if direct.exists():
        return direct
    return root / "runs" / "comm_cost_v2_rerun_20260928" / relative


def _write_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROUND42_ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _git_at(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def verify_frozen_source(frozen: dict[str, Any]) -> str:
    if _git("branch", "--show-current") != frozen["execution_identity"]["branch"]:
        raise RuntimeError("ROUND42 branch changed after prelaunch freeze")
    if _git("status", "--porcelain"):
        raise RuntimeError("ROUND42 tracked/untracked source worktree is dirty; refusing launch")
    head = _git("rev-parse", "HEAD")
    canonical_head = _git_at(INPUT_ROOT, "rev-parse", "HEAD")
    if canonical_head != frozen["execution_identity"]["canonical_read_only_input_head"]:
        raise RuntimeError("canonical read-only input HEAD changed after ROUND42 preflight")
    if _git_at(ROUND40_ROOT, "rev-parse", "HEAD") != frozen[
            "execution_identity"]["round40_delivery_head"]:
        raise RuntimeError("ROUND40 source worktree HEAD changed after ROUND42 preflight")
    round41_root = Path(frozen["execution_identity"]["round41_worktree"])
    if _git_at(round41_root, "rev-parse", "HEAD") != frozen[
            "execution_identity"]["round41_delivery_head"]:
        raise RuntimeError("ROUND41 source worktree HEAD changed after ROUND42 preflight")
    prelaunch_identity_path = RUN_DIR / "runtime" / "ROUND42_PRELAUNCH_IDENTITY.json"
    if prelaunch_identity_path.is_file():
        prelaunch_identity = _json_read(prelaunch_identity_path)
        if head != prelaunch_identity.get("head"):
            raise RuntimeError("ROUND42 HEAD changed after prelaunch identity was frozen")
    freeze_identity = dict(frozen["round42"]["freeze_identity"])
    declared_freeze_sha = freeze_identity.pop("freeze_sha256", None)
    freeze_body = json.dumps(freeze_identity, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"))
    actual_freeze_sha = hashlib.sha256(freeze_body.encode("utf-8")).hexdigest()
    if actual_freeze_sha != declared_freeze_sha:
        raise RuntimeError("ROUND42 freeze identity content hash mismatch")
    basis = frozen["execution_identity"]["round42_basis_commit"]
    subprocess.run(["git", "-C", str(ROUND42_ROOT), "merge-base", "--is-ancestor",
                    basis, head], check=True, stdout=subprocess.DEVNULL)
    for relative, expected in frozen["execution_identity"]["round42_code_sha256"].items():
        path = RUN_DIR / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"ROUND42 frozen code changed: {relative}")
        subprocess.run(["git", "-C", str(ROUND42_ROOT), "cat-file", "-e",
                        f"HEAD:runs/{RUN_DIR.name}/{relative}"],
                       check=True, stdout=subprocess.DEVNULL)
    for relative, expected in frozen["execution_identity"][
            "frozen_dependency_sha256"].items():
        path = _dependency_path(INPUT_ROOT, relative)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen inherited dependency changed: {relative}")
    for relative, record in frozen["execution_identity"][
            "round42_checkout_dependency_sha256"].items():
        local_path = _dependency_path(ROUND42_ROOT, relative)
        expected_checkout_hash = record["sha256"]
        if not local_path.is_file() or sha256(local_path) != expected_checkout_hash:
            raise RuntimeError(f"ROUND42 committed checkout source snapshot changed: {relative}")
    for name, expected in frozen["execution_identity"]["round40_code_sha256"].items():
        path = ROUND40_OUTPUT / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"ROUND40 frozen runner source changed: {name}")
    round41_output = ROUND42_ROOT / "runs" / "comm_temporal_decoder_diagnostic_20261003"
    for name, expected in frozen["execution_identity"]["round41_code_sha256"].items():
        path = round41_output / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"ROUND41 frozen diagnostic source changed: {name}")
    for name in ("EXECUTION_BASIS.md", "ROUND42_FROZEN_CONFIG.json",
                 "ROUND42_PURE_COMPUTE_GATE.json", "ROUND42_PREFLIGHT.json",
                 "ROUND42_CONFIGURATION_DIFF.json", "TRAINING_VERSION_MANIFEST.json"):
        subprocess.run(["git", "-C", str(ROUND42_ROOT), "cat-file", "-e",
                        f"HEAD:runs/{RUN_DIR.name}/{name}"],
                       check=True, stdout=subprocess.DEVNULL)
    diff = _json_read(RUN_DIR / "ROUND42_CONFIGURATION_DIFF.json")
    diff_body = json.dumps(diff, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    actual_diff_sha = hashlib.sha256(diff_body.encode("utf-8")).hexdigest()
    if actual_diff_sha != frozen["round42"]["freeze_identity"]["configuration_diff_sha256"]:
        raise RuntimeError("ROUND42 configuration diff differs from its frozen identity")
    return head


def configure(frozen: dict[str, Any], round40_freeze: dict[str, Any]) -> dict[str, Any]:
    """Bind the audited code inputs and ROUND42-only output/ledger paths."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    for directory in (ROUND40_OUTPUT, INPUT_ROOT):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import r40_common

    # ROUND40's process-safe claim implementation is reused with its frozen 672+28 cap,
    # redirected exclusively into this round's output directory before bootstrap.
    r40_common.ROUND_DIR = RUN_DIR
    common = r40_common.bootstrap(INPUT_ROOT)
    r40_common.ROUND_DIR = RUN_DIR

    import torch
    import v2_run as runner
    import v2_trainer as trainer
    import r40_run
    import r40_train
    import r42_versions
    import r42_one_shot

    expected_modules = {
        "v2_common": INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_common.py",
        "v2_controller": INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_controller.py",
        "v2_run": INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_run.py",
        "v2_trainer": INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_trainer.py",
        "v2_rules": INPUT_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_rules.py",
        "budget_masked": INPUT_ROOT / "runs" / "comm_budget_action_mask_fix_20260922" / "budget_masked.py",
        "sb_collector": INPUT_ROOT / "runs" / "comm_cost_stage_b_20260927" / "sb_collector.py",
        "onpolicy": INPUT_ROOT / "runs" / "comm_tight_budget_20260919" / "onpolicy.py",
        "cost_math": INPUT_ROOT / "runs" / "comm_cost_controller_stage_a_20260927" / "cost_math.py",
        "r40_eligibility": ROUND40_OUTPUT / "r40_eligibility.py",
        "integrated_sar.comm": INPUT_ROOT / "integrated_sar" / "comm.py",
        "integrated_sar.comm_budget": INPUT_ROOT / "integrated_sar" / "comm_budget.py",
        "integrated_sar.comm_learning": INPUT_ROOT / "integrated_sar" / "comm_learning.py",
        "integrated_sar.comm_policy": INPUT_ROOT / "integrated_sar" / "comm_policy.py",
        "integrated_sar.joint_driver": INPUT_ROOT / "integrated_sar" / "joint_driver.py",
        "integrated_sar.lifecycle_metrics": INPUT_ROOT / "integrated_sar" / "lifecycle_metrics.py",
        "integrated_sar.task_registry": INPUT_ROOT / "integrated_sar" / "task_registry.py",
        "integrated_sar.adapters.task03": INPUT_ROOT / "integrated_sar" / "adapters" / "task03.py",
        "r40_common": ROUND40_OUTPUT / "r40_common.py",
        "r40_run": ROUND40_OUTPUT / "r40_run.py",
        "r40_train": ROUND40_OUTPUT / "r40_train.py",
        "r40_versions": ROUND40_OUTPUT / "r40_versions.py",
        "r42_one_shot": RUN_DIR / "r42_one_shot.py",
    }
    for name, expected_path in expected_modules.items():
        module = sys.modules.get(name)
        if module is None:
            raise RuntimeError(f"runtime module was not imported before provenance check: {name}")
        _assert_module_path(module, expected_path, name)

    common.DEVICE = torch.device("cpu")
    common.OUTPUT_DIR = RUN_DIR
    common.LOGS = RUN_DIR / "logs"
    common.ROWS = RUN_DIR / "rows"
    common.REJECTED = RUN_DIR / "rejected"
    common.CHECKPOINTS = RUN_DIR / "checkpoints"
    common.CHECKPOINT_FILES = (
        INPUT_ROOT / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth",
        INPUT_ROOT / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth",
    )
    for directory in (common.LOGS, common.ROWS, common.REJECTED, common.CHECKPOINTS):
        directory.mkdir(parents=True, exist_ok=True)
    runner.common = common
    r42_versions.install(trainer, runner, common=common,
                         freeze=frozen["round42"]["freeze_identity"],
                         round40_freeze=round40_freeze)

    checkpoint_journal = RUN_DIR / "runtime" / "checkpoint_writes.jsonl"
    checkpoint_journal.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_journal.is_file():
        with checkpoint_journal.open("r", encoding="utf-8") as handle:
            r42_versions.CHECKPOINT_WRITE_CALLS = sum(1 for _line in handle)
    original_save_checkpoint = trainer.save_checkpoint

    def journaled_save_checkpoint(*args: Any, **kwargs: Any):
        result = original_save_checkpoint(*args, **kwargs)
        state = args[2] if len(args) > 2 else kwargs["state"]
        path = Path(args[3] if len(args) > 3 else kwargs["path"])
        entry = {
            "path_at_write": str(path),
            "sha256_at_write": sha256(path),
            "training_version": r42_versions.TRAINING_VERSION,
            "episodes_done": int(state.get("episodes_done", -1)),
            "updates": int(state.get("updates", -1)),
        }
        with checkpoint_journal.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return result

    trainer.save_checkpoint = journaled_save_checkpoint

    def counted_build(configuration: Any):
        global _LAST_RUN
        if _ACTIVE_CLAIM is None:
            raise RuntimeError("environment construction attempted without an atomic quota claim")
        claim_data = _json_read(_ACTIVE_CLAIM)
        if claim_data.get("environment_constructor_calls"):
            raise RuntimeError(f"claim already constructed an environment: {_ACTIVE_CLAIM}")
        r40_common._update_claim(_ACTIVE_CLAIM,
                                 environment_constructor_calls=1,
                                 constructor_called_at_unix=time.time())
        _LAST_RUN = None
        run = _ORIGINAL_BUILD(configuration)
        original_run = run.run

        def counted_run():
            r40_common._update_claim(_ACTIVE_CLAIM,
                                     simulation_started=True,
                                     simulation_started_at_unix=time.time())
            if _ACTIVE_COLLECTOR is not None:
                from r42_one_shot import stop_on_policy_refusal
                with stop_on_policy_refusal():
                    return original_run()
            return original_run()

        run.run = counted_run
        _LAST_RUN = run
        return run

    _ORIGINAL_BUILD = runner.build_joint_run
    runner.build_joint_run = counted_build
    r40_run.run_training_episode = run_training_episode
    _RUNTIME = {
        "common": common,
        "r40_common": r40_common,
        "runner": runner,
        "trainer": trainer,
        "r40_run": r40_run,
        "r40_train": r40_train,
        "r42_versions": r42_versions,
        "round40_freeze": round40_freeze,
        "round42_freeze": frozen["round42"]["freeze_identity"],
    }

    import torch.optim

    original_step = torch.optim.Adam.step

    def count_optimizer_step(self: Any, *args: Any, **kwargs: Any):
        global _OPTIMIZER_STEP_CALLS
        _OPTIMIZER_STEP_CALLS += 1
        return original_step(self, *args, **kwargs)

    torch.optim.Adam.step = count_optimizer_step
    _RUNTIME["restore_adam_step"] = original_step
    return _RUNTIME


def restore_runtime_hooks() -> None:
    if _RUNTIME is None:
        return
    import torch.optim
    torch.optim.Adam.step = _RUNTIME["restore_adam_step"]


def _claim(logical_job: str, *, attempt: int, metadata: dict[str, Any]) -> Path:
    global _ACTIVE_CLAIM
    if _ACTIVE_CLAIM is not None:
        raise RuntimeError("nested environment claims are not allowed")
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("ROUND42 runtime is not configured")
    head = verify_frozen_source(_json_read(RUN_DIR / "ROUND42_FROZEN_CONFIG.json"))
    category = "normal" if int(attempt) == 1 else "reserve"
    claim = runtime["r40_common"].claim_launch(
        logical_job, category=category, attempt=int(attempt), metadata=metadata)
    runtime["r40_common"]._update_claim(
        claim,
        status="claimed_before_environment",
        environment_constructor_calls=0,
        simulation_started=False,
        optimizer_step_calls=0,
        checkpoint_write_calls=0,
        prelaunch_head=head,
        round=42,
        comm_action_semantics="one-shot-send-skip/1",
    )
    _ACTIVE_CLAIM = claim
    return claim


def _finish(claim: Path, *, ok: bool, detail: str) -> None:
    global _ACTIVE_CLAIM
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("ROUND42 runtime was lost while finishing a claim")
    runtime["r40_common"].finish_launch(claim, ok=ok, detail=detail)
    if _ACTIVE_CLAIM == claim:
        _ACTIVE_CLAIM = None


@contextlib.contextmanager
def _active(claim: Path) -> Iterator[None]:
    if _ACTIVE_CLAIM != claim:
        raise RuntimeError("active claim does not match the environment cell")
    yield


def _persist_failure(*, kind: str, scenario: int, attempt: int, label: str, tag: str,
                     collector: Any, error: BaseException) -> dict[str, Any]:
    runtime = _RUNTIME
    if runtime is None:
        return {"error": repr(error)}
    common = runtime["common"]
    paths = common.cell_paths(kind, int(scenario), int(attempt), tag=tag)
    persisted: dict[str, Any] = {"exception": f"{type(error).__name__}: {error}"}
    if collector is not None:
        tape_path = paths.get("tape")
        if tape_path is not None and not Path(tape_path).exists():
            Path(tape_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(tape_path).open("x", encoding="utf-8", newline="\n") as handle:
                for tape in collector.records:
                    handle.write(json.dumps(tape, ensure_ascii=False, sort_keys=True) + "\n")
            persisted["partial_tape"] = str(tape_path)
            persisted["partial_tape_sha256"] = sha256(Path(tape_path))
    if _LAST_RUN is not None:
        relay = _LAST_RUN.task_adapter.comm_relay
        transcript_path = paths.get("transcript")
        transcript = getattr(relay, "transcript", None)
        if transcript is not None and transcript_path is not None and not Path(transcript_path).exists():
            transcript_path = transcript.write(transcript_path)
            persisted["partial_transcript"] = str(transcript_path)
            persisted["partial_transcript_sha256"] = sha256(Path(transcript_path))
        persisted["relay_counters_at_failure"] = {
            key: int(value) for key, value in relay.counters.items()
        }
        persisted["relay_tick_at_failure"] = int(relay._tick)
    failure_path = RUN_DIR / "failures" / f"{label}.json"
    if not failure_path.exists():
        _write_new(failure_path, persisted)
        persisted["failure_record"] = str(failure_path)
    return persisted


class _NoOpLedger:
    def record(self, **_kwargs: Any) -> None:
        return None


def _execute_cell(label: str, logical_job: str, kind: str, scenario: int,
                  metadata: dict[str, Any], *, collector: Any = None,
                  runner_kwargs: dict[str, Any] | None = None,
                  row_updates: dict[str, Any] | None = None,
                  identity: dict[str, Any] | None = None) -> dict[str, Any]:
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("ROUND42 runtime is not configured")
    common, runner = runtime["common"], runtime["runner"]
    attempt = runtime["r40_common"].next_attempt(logical_job)
    claim = _claim(logical_job, attempt=attempt, metadata=metadata)
    kwargs = dict(runner_kwargs or {})
    actual_tag = str(kwargs["tag"]) if "tag" in kwargs else label
    global _ACTIVE_COLLECTOR
    if _ACTIVE_COLLECTOR is not None:
        raise RuntimeError("nested collector-backed environment runs are not allowed")
    _ACTIVE_COLLECTOR = collector
    try:
        with _active(claim):
            row = runner._run_episode(
                label, kind, int(scenario), attempt=attempt, **kwargs)
        if row_updates:
            row.update(row_updates)
        if identity is not None:
            runtime["r42_versions"].add_row_identity(row, checkpoint=identity)
        else:
            runtime["r42_versions"].add_row_identity(row)
        runner._booked(label, kind, row, _NoOpLedger())
        _finish(claim, ok=True,
                detail=f"{kind} scene={scenario} on_time={row['on_time']} "
                       f"bytes={row['spent_bytes']}")
        write_execution_ledger()
        return row
    except BaseException as error:
        detail = _persist_failure(kind=kind, scenario=scenario, attempt=attempt,
                                  label=label, tag=actual_tag, collector=collector,
                                  error=error)
        _finish(claim, ok=False,
                detail=f"{type(error).__name__}: {error}; preserved={json.dumps(detail, ensure_ascii=False)}")
        try:
            write_execution_ledger(strict=False)
        except BaseException as ledger_error:
            detail["ledger_write_error"] = repr(ledger_error)
        raise
    finally:
        _ACTIVE_COLLECTOR = None


def run_reference(scenario: int) -> dict[str, Any]:
    label = f"ref_immediate_s{int(scenario)}_r42"
    row = _execute_cell(
        label, f"immediate_reference_scene{int(scenario)}", "reference", int(scenario),
        {"arm": "immediate", "scenario_seed": int(scenario)},
        runner_kwargs={"policy": "immediate", "tag": ""},
        row_updates={"reference_arm": "immediate", "action_rng_seed": None},
    )
    return row


def run_training_episode(common: Any, runner: Any, controller: Any, *, training_seed: int,
                         episode_index: int, scenario_seed: int, lam: float) -> dict[str, Any]:
    from r42_one_shot import OneShotSendSkipCollector

    collector = OneShotSendSkipCollector(controller, greedy=False)
    logical = f"new_training_s{int(training_seed)}_e{int(episode_index)}"
    label = f"train_s{int(training_seed)}_e{int(episode_index)}_r42"
    row = _execute_cell(
        label, logical, "train", int(scenario_seed),
        {"arm": "NEW", "training_seed": int(training_seed),
         "episode_index": int(episode_index), "scenario_seed": int(scenario_seed)},
        collector=collector,
        runner_kwargs={
            "policy": "learned", "attach": collector.policy(), "controller": controller,
            "collector": collector,
            "action_rng_seed": runner.action_seed(training_seed, episode_index),
            "tag": f"r42_train_s{training_seed}_e{episode_index}",
        },
        row_updates={
            "training_seed": int(training_seed), "episode_index": int(episode_index),
            "lambda_at_rollout": float(lam), "lambda": float(lam),
            "training_arm": "NEW",
            "action_rng_seed": runner.action_seed(training_seed, episode_index),
        },
    )
    row["collector"] = collector
    row["controller"] = controller
    return row


def reference_cache() -> dict[int, dict[str, Any]]:
    common = _RUNTIME["common"]
    cache: dict[int, dict[str, Any]] = {}
    for path in sorted(common.ROWS.glob("v2_reference_s*_a*.json")):
        row = _json_read(path)
        scenario = int(row["scenario_seed"])
        if row.get("reference_arm") != "immediate":
            continue
        if scenario in cache and cache[scenario]["behaviour_sha256"] != row["behaviour_sha256"]:
            raise RuntimeError(f"scenario {scenario} has conflicting immediate references")
        cache[scenario] = row
    return cache


def train_seed(training_seed: int, *, episodes: int, frozen: dict[str, Any]) -> dict[str, Any]:
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("ROUND42 runtime is not configured")
    r40_train = runtime["r40_train"]
    cache_rows = reference_cache()
    cache = {scene: row for scene, row in cache_rows.items()}
    scenario_order = runtime["common"].scenario_order()
    missing = sorted(set(scenario_order[:episodes]) - set(cache))
    if missing:
        raise RuntimeError(f"ROUND42 immediate reference cache is incomplete: {missing}")
    r40_train.common = runtime["common"]
    r40_train.runner = runtime["runner"]
    r40_train.trainer = runtime["trainer"]
    report = r40_train.train_seed(
        int(training_seed), episodes=int(episodes),
        freeze=runtime["round42_freeze"], reference_cache=cache)
    report["comm_action_semantics"] = "one-shot-send-skip/1"
    report["training_version"] = frozen["round42"]["training_version"]
    return report


def load_eval_model(arm: str, training_seed: int) -> tuple[Any, dict[str, Any]]:
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("ROUND42 runtime is not configured")
    trainer = runtime["trainer"]
    if int(training_seed) not in SEEDS:
        raise ValueError(f"unregistered training seed {training_seed}")
    if arm == "OLD":
        source = _json_read(ROUND40_OUTPUT / "CHECKPOINT_RECONCILIATION.json")[
            "checkpoints"]["NEW_training_outputs"][str(training_seed)]["ep128"]
        path = Path(source["path"])
        if sha256(path) != source["file_sha256"]:
            raise RuntimeError(f"ROUND40 OLD read-only checkpoint hash changed: {path}")
        controller, state, identity = trainer.load_checkpoint_read_only(path)
        expected = source["parameter_sha256"]
        if controller.parameter_summary()["parameter_sha256"] != expected:
            raise RuntimeError(f"ROUND40 OLD parameter identity changed: seed {training_seed}")
    elif arm == "NEW":
        path = runtime["common"].CHECKPOINTS / f"v2_train_s{training_seed}_ep128.pt"
        if not path.is_file():
            raise FileNotFoundError(f"ROUND42 NEW ep128 checkpoint missing: {path}")
        payload = torch_load_cpu(path)
        if (payload.get("training_version") != runtime["r42_versions"].TRAINING_VERSION
                or payload.get("comm_action_semantics") != "one-shot-send-skip/1"):
            raise RuntimeError(f"ROUND42 NEW checkpoint identity mismatch: {path}")
        controller, state, identity = trainer.load_checkpoint_read_only(
            path, freeze=runtime["round42_freeze"])
        if int(state.get("episodes_done", -1)) != 128:
            raise RuntimeError(f"ROUND42 NEW model is not ep128: {path}")
    else:
        raise ValueError(f"unknown ROUND42 model arm {arm!r}")
    identity.update({
        "arm": arm,
        "path": str(path),
        "file_sha256": sha256(path),
        "training_seed": int(training_seed),
        "episodes_done": int(state.get("episodes_done", -1)),
        "updates": int(state.get("updates", -1)),
        "optimizer_steps": int(state.get("optimizer_steps", -1)),
        "lambda": float(state.get("lambda", float("nan"))),
        "parameter_identity": controller.parameter_summary(),
        "evaluation_semantics": "one-shot-send-skip/1",
    })
    if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
        raise RuntimeError(f"{arm} seed {training_seed}: read-only evaluation restored state")
    return controller, identity


def torch_load_cpu(path: Path) -> dict[str, Any]:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def run_model_eval(*, arm: str, training_seed: int, mode: str, scenario: int,
                   controller: Any, identity: dict[str, Any]) -> dict[str, Any]:
    from r42_one_shot import OneShotSendSkipCollector

    if mode not in ("greedy", "sampled"):
        raise ValueError(f"unknown evaluation mode {mode!r}")
    runtime = _RUNTIME
    runner = runtime["runner"]
    collector = OneShotSendSkipCollector(controller, greedy=(mode == "greedy"))
    logical = (f"model_eval_{arm.lower()}_s{training_seed}_{mode}_scene{int(scenario)}")
    label = f"eval_{arm.lower()}_s{training_seed}_{mode}_s{int(scenario)}_r42"
    action_seed = runner.eval_action_seed(training_seed, scenario, mode)
    return _execute_cell(
        label, logical, "eval", int(scenario),
        {"arm": arm, "training_seed": int(training_seed), "mode": mode,
         "scenario_seed": int(scenario)},
        collector=collector,
        runner_kwargs={
            "policy": "learned", "attach": collector.policy(), "controller": controller,
            "collector": collector, "action_rng_seed": action_seed,
            "tag": f"r42_eval_{arm.lower()}_s{training_seed}_{mode}",
            "checkpoint_load_identity": identity,
        },
        row_updates={
            "model_arm": arm, "training_seed": int(training_seed),
            "eval_mode": mode, "eval_block": f"one_shot_{mode}",
            "action_rng_seed": action_seed,
            "checkpoint_parameter_identity": identity["parameter_identity"],
        },
        identity=identity,
    )


def run_rule(arm: str, scenario: int) -> dict[str, Any]:
    if arm not in RULE_ARMS:
        raise ValueError(f"ROUND42 rule arm not authorized: {arm!r}")
    import v2_rules

    runtime = _RUNTIME
    policy = v2_rules.build_rule(arm, int(scenario))
    label = f"rule_{arm}_s{int(scenario)}_r42"
    row = _execute_cell(
        label, f"rule_{arm}_scene{int(scenario)}", "rule", int(scenario),
        {"arm": arm, "scenario_seed": int(scenario)},
        runner_kwargs={
            "policy": "learned", "attach": policy, "tag": f"r42_rule_{arm}",
        },
        row_updates={"model_arm": arm},
    )
    row["rule_identity"] = v2_rules.rule_identity(arm)
    row["rule_identity"]["built_parameters"] = dict(policy.parameters())
    # The registered row file is rewritten only within this new run's own namespace to include
    # the explicit, deterministic rule identity used by the completed cell.
    row_path = Path(row["artefacts"]["row"])
    row_path.write_text(json.dumps(row, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return row


def ledger_snapshot(*, strict: bool = True) -> dict[str, Any]:
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("ROUND42 runtime is not configured")
    ledger = runtime["r40_common"].ledger_snapshot()
    claims = ledger["claims"]
    batch_records = list((RUN_DIR / "checkpoints").glob("r40_batch_s*_b*_attempt*.json"))
    optimizer_steps_from_batches = sum(
        int(_json_read(path).get("update", {}).get("optimizer_steps", 0))
        for path in batch_records)
    optimizer_steps_actual = int(_OPTIMIZER_STEP_CALLS)
    if strict and optimizer_steps_from_batches != optimizer_steps_actual:
        raise RuntimeError(
            f"optimizer step count mismatch: batch records={optimizer_steps_from_batches}, "
            f"observed torch Adam.step={optimizer_steps_actual}"
        )
    ledger["counts"] = {
        "normal_used": int(ledger["normal_used"]),
        "reserve_used": int(ledger["reserve_used"]),
        "total_used": int(ledger["total_used"]),
        "environment_constructor_calls": sum(
            int(item.get("environment_constructor_calls", 0)) for item in claims),
        "simulation_started": sum(bool(item.get("simulation_started")) for item in claims),
        "technical_failed_claims": sum(item.get("status") == "failed_technical"
                                        for item in claims),
        "retry_claims": sum(int(item.get("attempt", 1)) > 1 for item in claims),
        "optimizer_step_calls": optimizer_steps_actual,
        "checkpoint_write_calls": int(runtime["r42_versions"].CHECKPOINT_WRITE_CALLS),
        "checkpoint_write_journal_entries": sum(
            1 for _line in (RUN_DIR / "runtime" / "checkpoint_writes.jsonl").open(
                "r", encoding="utf-8")
        ) if (RUN_DIR / "runtime" / "checkpoint_writes.jsonl").is_file() else 0,
        "optimizer_step_count_matches_committed_batches": (
            optimizer_steps_from_batches == optimizer_steps_actual),
        "hard_cap": int(runtime["r40_common"].HARD_CAP),
    }
    return ledger


def write_execution_ledger(*, strict: bool = True) -> dict[str, Any]:
    ledger = ledger_snapshot(strict=strict)
    path = RUN_DIR / "runtime" / "ROUND42_EXECUTION_LEDGER.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)
    return ledger


__all__ = [
    "DEV_SCENARIOS", "INPUT_ROOT", "ROUND40_OUTPUT", "ROUND42_ROOT", "RUN_DIR",
    "RULE_ARMS", "SEEDS", "configure", "ledger_snapshot", "load_eval_model",
    "reference_cache", "restore_runtime_hooks", "run_model_eval", "run_reference",
    "run_rule", "run_training_episode", "train_seed", "verify_frozen_source",
    "write_execution_ledger",
]
