"""ROUND43 reuses the counted ROUND42 runner with the full-trace GAE version overlay."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

RUN_DIR = Path(__file__).resolve().parent
ROUND42_ROOT = RUN_DIR.parents[1]
WORKSPACE_ROOT = ROUND42_ROOT.parent
INPUT_ROOT = WORKSPACE_ROOT / "zhenghe"
ROUND40_ROOT = WORKSPACE_ROOT / "zhenghe-round40"
ROUND40_OUTPUT = ROUND40_ROOT / "runs" / "comm_first_report_training_ablation_20261003"
ROUND41_OUTPUT = ROUND42_ROOT / "runs" / "comm_temporal_decoder_diagnostic_20261003"
ROUND42_OUTPUT = ROUND42_ROOT / "runs" / "comm_one_shot_gate_resume_20261003_v2"
SEEDS = (11, 22, 33)
DEV_SCENARIOS = tuple(range(1_301_000, 1_301_016))
RULE_ARMS = ("late_discovery_gate", "always_hold")
RUNNER_VERSION = "counted-jointrun-proxy/1"

_RUNTIME: dict[str, Any] | None = None
_ACTIVE_CLAIM: Path | None = None
_ACTIVE_COLLECTOR: Any = None
_ACTIVE_PROGRESS: dict[str, int] | None = None
_LAST_RUN: Any = None
_OPTIMIZER_STEP_CALLS = 0
_PPO_UPDATE_CALLS = 0


def _new_run_progress() -> dict[str, int]:
    return {
        "environment_constructor_calls": 0,
        "joint_run_build_failures": 0,
        "joint_run_constructed_calls": 0,
        "run_invocation_calls": 0,
        "joint_run_method_calls": 0,
        "run_return_calls": 0,
        "run_failure_calls": 0,
        "completed_steps": 0,
    }


def _step_index(target: Any) -> int | None:
    try:
        return int(target.orchestrator.step_index)
    except (AttributeError, TypeError, ValueError):
        return None


class _CountedJointRunProxy:
    """Run-local transparent adapter; collaborator objects remain owned by target."""

    __slots__ = (
        "_target", "_progress", "_claim", "_update_claim", "_collector",
        "_dispatch_run", "_initial_step_index", "_on_progress",
    )

    def __init__(self, target: Any, progress: dict[str, int], claim: Path,
                 update_claim: Callable[..., Any], collector: Callable[[], Any],
                 dispatch_run: Callable[[Callable[[], Any], Any], Any],
                 on_progress: Callable[[Path, dict[str, int]], None]) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_progress", progress)
        object.__setattr__(self, "_claim", claim)
        object.__setattr__(self, "_update_claim", update_claim)
        object.__setattr__(self, "_collector", collector)
        object.__setattr__(self, "_dispatch_run", dispatch_run)
        object.__setattr__(self, "_initial_step_index", _step_index(target))
        object.__setattr__(self, "_on_progress", on_progress)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _CountedJointRunProxy.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_target"), name, value)

    def _call_target_run(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        progress = object.__getattribute__(self, "_progress")
        progress["joint_run_method_calls"] += 1
        object.__getattribute__(self, "_on_progress")(
            object.__getattribute__(self, "_claim"), progress)
        return object.__getattribute__(self, "_target").run(*args, **kwargs)

    def _record_completed_steps(self) -> None:
        initial = object.__getattribute__(self, "_initial_step_index")
        current = _step_index(object.__getattribute__(self, "_target"))
        if initial is not None and current is not None:
            progress = object.__getattribute__(self, "_progress")
            progress["completed_steps"] = max(
                progress["completed_steps"], max(0, current - initial))

    def run(self, *args: Any, **kwargs: Any) -> Any:
        progress = object.__getattribute__(self, "_progress")
        progress["run_invocation_calls"] += 1
        object.__getattribute__(self, "_on_progress")(
            object.__getattribute__(self, "_claim"), progress)
        try:
            result = object.__getattribute__(self, "_dispatch_run")(
                lambda: self._call_target_run(args, kwargs),
                object.__getattribute__(self, "_collector")(),
            )
        except BaseException:
            self._record_completed_steps()
            progress["run_failure_calls"] += 1
            object.__getattribute__(self, "_on_progress")(
                object.__getattribute__(self, "_claim"), progress)
            raise
        self._record_completed_steps()
        progress["run_return_calls"] += 1
        object.__getattribute__(self, "_on_progress")(
            object.__getattribute__(self, "_claim"), progress)
        return result


def _make_counted_builder(
    original_builder: Callable[[Any], Any], *,
    active_claim: Callable[[], Path | None],
    active_progress: Callable[[], dict[str, int] | None],
    update_claim: Callable[..., Any],
    collector: Callable[[], Any],
    dispatch_run: Callable[[Callable[[], Any], Any], Any],
    write_progress: Callable[[Path, dict[str, int]], None],
    on_built: Callable[[Any], None] | None = None,
) -> Callable[[Any], _CountedJointRunProxy]:
    """The production builder seam used by both configure() and regression tests."""

    def counted_build(configuration: Any) -> _CountedJointRunProxy:
        claim = active_claim()
        progress = active_progress()
        if claim is None or progress is None:
            raise RuntimeError("environment construction attempted without an atomic quota claim")
        if progress["environment_constructor_calls"]:
            raise RuntimeError(f"claim already attempted an environment build: {claim}")
        progress["environment_constructor_calls"] += 1
        write_progress(claim, progress)
        try:
            target = original_builder(configuration)
        except BaseException:
            progress["joint_run_build_failures"] += 1
            write_progress(claim, progress)
            raise
        progress["joint_run_constructed_calls"] += 1
        write_progress(claim, progress)
        proxy = _CountedJointRunProxy(
            target, progress, claim, update_claim, collector, dispatch_run, write_progress)
        if on_built is not None:
            on_built(proxy)
        return proxy

    return counted_build


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


def _append_journal(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROUND42_ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _git_at(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def verify_frozen_source(frozen: dict[str, Any]) -> str:
    """Fail closed on the committed ROUND43 source, inputs, and one-variable freeze."""
    identity = frozen["execution_identity"]
    experiment = frozen["round43"]
    if Path(sys.executable).resolve() != Path(identity["python_executable"]).resolve():
        raise RuntimeError(
            f"ROUND43 Python executable differs from freeze: {sys.executable}")
    if (identity.get("runner_version") != RUNNER_VERSION
            or experiment.get("runner_version") != RUNNER_VERSION):
        raise RuntimeError("ROUND43 counted runner differs from its frozen identity")
    if experiment.get("comm_action_semantics") != "one-shot-send-skip/1":
        raise RuntimeError("ROUND43 action protocol differs from the frozen version")
    if _git("branch", "--show-current") != identity["branch"]:
        raise RuntimeError("ROUND43 branch changed after prelaunch freeze")
    if _git("status", "--porcelain"):
        raise RuntimeError("ROUND43 worktree is dirty; refusing environment launch")
    head = _git("rev-parse", "HEAD")
    expected_head = identity.get("prelaunch_head")
    if expected_head and head != expected_head:
        raise RuntimeError("ROUND43 HEAD changed after its source freeze")
    for ancestor in (identity["source_delivery_head"], identity["source_execution_head"]):
        subprocess.run(["git", "-C", str(ROUND42_ROOT), "merge-base", "--is-ancestor",
                        ancestor, head], check=True, stdout=subprocess.DEVNULL)
    for key, root in (("canonical_read_only_input_head", INPUT_ROOT),
                      ("round40_delivery_head", ROUND40_ROOT),
                      ("round41_delivery_head", Path(identity["round41_worktree"]))):
        actual = _git_at(root, "rev-parse", "HEAD")
        if actual != identity[key]:
            raise RuntimeError(f"ROUND43 frozen source checkout changed: {key}={actual}")

    freeze_identity = dict(frozen["round42"]["freeze_identity"])
    declared_freeze_sha = freeze_identity.pop("freeze_sha256", None)
    freeze_body = json.dumps(freeze_identity, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"))
    if hashlib.sha256(freeze_body.encode("utf-8")).hexdigest() != declared_freeze_sha:
        raise RuntimeError("ROUND43 freeze identity content hash mismatch")
    for relative, expected in identity["source_code_sha256"].items():
        path = RUN_DIR / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"ROUND43 frozen source code changed: {relative}")
        subprocess.run(["git", "-C", str(ROUND42_ROOT), "cat-file", "-e",
                        f"HEAD:runs/{RUN_DIR.name}/{relative}"],
                       check=True, stdout=subprocess.DEVNULL)
    for relative, expected in identity["inherited_dependency_sha256"].items():
        path = _dependency_path(INPUT_ROOT, relative)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"ROUND43 inherited dependency changed: {relative}")
    for name, expected in identity["round40_code_sha256"].items():
        path = ROUND40_OUTPUT / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"ROUND40 source dependency changed: {name}")

    basis = RUN_DIR / "EXECUTION_BASIS.md"
    if sha256(basis) != identity["execution_basis_sha256"]:
        raise RuntimeError("ROUND43 archived execution basis changed")
    diff = _json_read(RUN_DIR / "ROUND43_CONFIGURATION_DIFF.json")
    diff_body = json.dumps(diff, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    actual_diff_sha = hashlib.sha256(diff_body.encode("utf-8")).hexdigest()
    if actual_diff_sha != frozen["round42"]["freeze_identity"]["configuration_diff_sha256"]:
        raise RuntimeError("ROUND43 configuration diff differs from its freeze")
    prelaunch_path = RUN_DIR / "runtime" / "ROUND43_PRELAUNCH_IDENTITY.json"
    if prelaunch_path.is_file():
        prelaunch = _json_read(prelaunch_path)
        expected_receipt = {
            "head": head,
            "branch": identity["branch"],
            "execution_basis_sha256": identity["execution_basis_sha256"],
            "configuration_diff_sha256": actual_diff_sha,
            "frozen_config_sha256": sha256(RUN_DIR / "ROUND43_FROZEN_CONFIG.json"),
            "source_code_sha256": identity["source_code_sha256"],
            "preflight_sha256": sha256(RUN_DIR / "ROUND43_PREFLIGHT.json"),
        }
        for key, value in expected_receipt.items():
            if prelaunch.get(key) != value:
                raise RuntimeError(
                    f"ROUND43 prelaunch identity receipt mismatch for {key}")
        if prelaunch.get("status") != "passed":
            raise RuntimeError("ROUND43 prelaunch identity does not record a passed gate")
    return head


def configure(frozen: dict[str, Any], round40_freeze: dict[str, Any]) -> dict[str, Any]:
    """Bind the reused runner to ROUND43 paths and the registered GAE override."""
    global _RUNTIME, _PPO_UPDATE_CALLS, _OPTIMIZER_STEP_CALLS
    if _RUNTIME is not None:
        return _RUNTIME
    for directory in (ROUND40_OUTPUT, INPUT_ROOT):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import r40_common

    # Keep ROUND40's atomic claim implementation, while applying ROUND43's separate quota.
    r40_common.ROUND_DIR = RUN_DIR
    r40_common.NORMAL_CAP = 480
    r40_common.RESERVE_CAP = 24
    r40_common.HARD_CAP = 600
    r40_common.plan = lambda: {
        "normal": {"new_training": 384, "new_final_evaluation": 96},
        "normal_total": 480, "technical_reserve_max": 24, "hard_cap": 600,
    }
    common = r40_common.bootstrap(INPUT_ROOT)
    r40_common.ROUND_DIR = RUN_DIR

    import torch
    import v2_run as runner
    import v2_controller
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
        "r42_versions": RUN_DIR / "r42_versions.py",
    }
    for name, expected_path in expected_modules.items():
        module = sys.modules.get(name)
        if module is None:
            raise RuntimeError(f"runtime module was not imported before provenance check: {name}")
        _assert_module_path(module, expected_path, name)

    intervention = frozen["round43"]
    if float(common.GAE_LAMBDA) != float(intervention["gae_lambda_old"]):
        raise RuntimeError("active training module no longer declares ROUND42 GAE lambda")
    if float(common.GAMMA) != float(intervention["gamma"]):
        raise RuntimeError("active training module gamma differs from ROUND42")
    if float(intervention["gae_lambda"]) != 1.0:
        raise RuntimeError("ROUND43 requires the registered full-trace coefficient 1.0")
    common.GAE_LAMBDA = float(intervention["gae_lambda"])
    if trainer.common is not common:
        raise RuntimeError("GAE override is not connected to the active batch/runner module")
    if float(trainer.common.GAE_LAMBDA) != 1.0 or float(trainer.common.GAMMA) != float(
            intervention["gamma"]):
        raise RuntimeError("active batch_samples path did not receive the frozen GAE override")
    if r42_versions.TRAINING_VERSION != frozen["round42"]["training_version"]:
        raise RuntimeError("ROUND43 checkpoint gate version differs from the registered version")

    common.DEVICE = torch.device("cpu")
    common.NORMAL_LAUNCHES = 480
    common.RESERVE_LAUNCHES = 24
    common.LAUNCH_CAP = 600
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
    # Rehydrate durable call counters before installing instrumentation. Individual call
    # journals preserve attempts that did not reach a committed training batch.
    batch_records = list(common.CHECKPOINTS.glob("r40_batch_s*_b*_attempt*.json"))
    optimizer_call_journal = RUN_DIR / "runtime" / "optimizer_step_calls.jsonl"
    ppo_call_journal = RUN_DIR / "runtime" / "ppo_update_calls.jsonl"
    if optimizer_call_journal.is_file():
        with optimizer_call_journal.open("r", encoding="utf-8") as handle:
            _OPTIMIZER_STEP_CALLS = sum(1 for line in handle if line.strip())
    else:
        _OPTIMIZER_STEP_CALLS = sum(
            int(_json_read(path).get("update", {}).get("optimizer_steps", 0))
            for path in batch_records)
    if ppo_call_journal.is_file():
        with ppo_call_journal.open("r", encoding="utf-8") as handle:
            _PPO_UPDATE_CALLS = sum(1 for line in handle if line.strip())
    else:
        _PPO_UPDATE_CALLS = len(batch_records)
    committed_ppo_calls = len(batch_records)
    committed_optimizer_steps = sum(
        int(_json_read(path).get("update", {}).get("optimizer_steps", 0))
        for path in batch_records)
    if batch_records and (not optimizer_call_journal.is_file() or not ppo_call_journal.is_file()):
        raise RuntimeError("ROUND43 has committed batches but a durable training-call journal is missing")
    if (_PPO_UPDATE_CALLS != committed_ppo_calls
            or _OPTIMIZER_STEP_CALLS != committed_optimizer_steps):
        raise RuntimeError(
            "ROUND43 call journals do not close against committed batch records; "
            "stop before another episode")
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

    def write_run_progress(claim: Path, progress: dict[str, int]) -> None:
        updates: dict[str, Any] = dict(progress)
        started = int(progress.get("completed_steps", 0)) > 0
        updates["simulation_started"] = started
        if started:
            existing = _json_read(claim)
            if not existing.get("simulation_started_at_unix"):
                updates["simulation_started_at_unix"] = time.time()
        if int(progress.get("environment_constructor_calls", 0)):
            existing = _json_read(claim)
            if not existing.get("constructor_called_at_unix"):
                updates["constructor_called_at_unix"] = time.time()
        r40_common._update_claim(claim, **updates)

    def dispatch_run(call_target: Callable[[], Any], active_collector: Any) -> Any:
        if active_collector is not None:
            from r42_one_shot import stop_on_policy_refusal
            with stop_on_policy_refusal():
                return call_target()
        return call_target()

    def remember_run(run: Any) -> None:
        global _LAST_RUN
        _LAST_RUN = run

    _ORIGINAL_BUILD = runner.build_joint_run
    runner.build_joint_run = _make_counted_builder(
        _ORIGINAL_BUILD,
        active_claim=lambda: _ACTIVE_CLAIM,
        active_progress=lambda: _ACTIVE_PROGRESS,
        update_claim=r40_common._update_claim,
        collector=lambda: _ACTIVE_COLLECTOR,
        dispatch_run=dispatch_run,
        write_progress=write_run_progress,
        on_built=remember_run,
    )
    r40_run.run_training_episode = run_training_episode
    _RUNTIME = {
        "common": common,
        "r40_common": r40_common,
        "runner": runner,
        "controller_module": v2_controller,
        "trainer": trainer,
        "r40_run": r40_run,
        "r40_train": r40_train,
        "r42_versions": r42_versions,
        "round40_freeze": round40_freeze,
        "round42_freeze": frozen["round42"]["freeze_identity"],
        "frozen": frozen,
    }

    import torch.optim

    original_step = torch.optim.Adam.step

    def count_optimizer_step(self: Any, *args: Any, **kwargs: Any):
        global _OPTIMIZER_STEP_CALLS
        _OPTIMIZER_STEP_CALLS += 1
        _append_journal(optimizer_call_journal, {
            "call_index": _OPTIMIZER_STEP_CALLS,
            "optimizer_class": type(self).__qualname__,
            "call_status": "step_invoked",
            "unix_time": time.time(),
        })
        return original_step(self, *args, **kwargs)

    torch.optim.Adam.step = count_optimizer_step
    _RUNTIME["restore_adam_step"] = original_step

    original_ppo_update = trainer.ppo_update

    def count_ppo_update(*args: Any, **kwargs: Any):
        global _PPO_UPDATE_CALLS
        _PPO_UPDATE_CALLS += 1
        update_index = args[3] if len(args) > 3 else kwargs.get("update_index")
        _append_journal(ppo_call_journal, {
            "call_index": _PPO_UPDATE_CALLS,
            "update_index": int(update_index) if update_index is not None else None,
            "call_status": "ppo_update_invoked",
            "unix_time": time.time(),
        })
        return original_ppo_update(*args, **kwargs)

    trainer.ppo_update = count_ppo_update
    _RUNTIME["restore_ppo_update"] = original_ppo_update
    return _RUNTIME


def restore_runtime_hooks() -> None:
    if _RUNTIME is None:
        return
    import torch.optim
    torch.optim.Adam.step = _RUNTIME["restore_adam_step"]
    _RUNTIME["trainer"].ppo_update = _RUNTIME["restore_ppo_update"]


def _claim(logical_job: str, *, attempt: int, metadata: dict[str, Any]) -> Path:
    global _ACTIVE_CLAIM
    if _ACTIVE_CLAIM is not None:
        raise RuntimeError("nested environment claims are not allowed")
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("ROUND42 runtime is not configured")
    head = verify_frozen_source(_json_read(RUN_DIR / "ROUND43_FROZEN_CONFIG.json"))
    category = "normal" if int(attempt) == 1 else "reserve"
    claim = runtime["r40_common"].claim_launch(
        logical_job, category=category, attempt=int(attempt), metadata=metadata)
    runtime["r40_common"]._update_claim(
        claim,
        status="claimed_before_environment",
        environment_constructor_calls=0,
        joint_run_build_failures=0,
        joint_run_constructed_calls=0,
        run_invocation_calls=0,
        joint_run_method_calls=0,
        run_return_calls=0,
        run_failure_calls=0,
        completed_steps=0,
        simulation_started=False,
        optimizer_step_calls=0,
        checkpoint_write_calls=0,
        prelaunch_head=head,
        round=43,
        comm_action_semantics="one-shot-send-skip/1",
        training_version=runtime["r42_versions"].TRAINING_VERSION,
        runner_version=RUNNER_VERSION,
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
    failure_path = RUN_DIR / "failures" / f"{label}_a{int(attempt)}.json"
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
    global _ACTIVE_COLLECTOR, _ACTIVE_PROGRESS, _LAST_RUN
    if _ACTIVE_COLLECTOR is not None:
        raise RuntimeError("nested collector-backed environment runs are not allowed")
    _ACTIVE_COLLECTOR = collector
    _ACTIVE_PROGRESS = _new_run_progress()
    _LAST_RUN = None
    try:
        with _active(claim):
            row = runner._run_episode(
                label, kind, int(scenario), attempt=attempt, **kwargs)
        if int(row.get("steps", -1)) != int(_ACTIVE_PROGRESS["completed_steps"]):
            raise RuntimeError(
                f"{label}: returned step count {row.get('steps')} differs from observed "
                f"orchestrator completed steps {_ACTIVE_PROGRESS['completed_steps']}")
        if row_updates:
            row.update(row_updates)
        if identity is not None:
            runtime["r42_versions"].add_row_identity(row, checkpoint=identity)
        else:
            runtime["r42_versions"].add_row_identity(row)
        row["runner_version"] = RUNNER_VERSION
        runner._booked(label, kind, row, _NoOpLedger())
        _finish(claim, ok=True,
                detail=f"{kind} scene={scenario} on_time={row['on_time']} "
                       f"bytes={row['spent_bytes']}")
        write_execution_ledger()
        return row
    except BaseException as error:
        _RUNTIME["r40_common"]._update_claim(claim, **_ACTIVE_PROGRESS)
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
        _ACTIVE_PROGRESS = None


def run_reference(scenario: int) -> dict[str, Any]:
    label = f"ref_immediate_s{int(scenario)}_r42"
    metadata = {"arm": "immediate", "scenario_seed": int(scenario)}
    if int(scenario) == 1_300_000:
        metadata["retry_of"] = (
            "runs/comm_one_shot_report_selection_20261003/launch_claims/"
            "immediate_reference_scene1300000/attempt_0001.json"
        )
    row = _execute_cell(
        label, f"immediate_reference_scene{int(scenario)}", "reference", int(scenario),
        metadata,
        runner_kwargs={"policy": "immediate", "tag": ""},
        row_updates={"reference_arm": "immediate", "action_rng_seed": None},
    )
    return row


def run_training_episode(common: Any, runner: Any, controller: Any, *, training_seed: int,
                         episode_index: int, scenario_seed: int, lam: float) -> dict[str, Any]:
    from r42_one_shot import OneShotSendSkipCollector

    collector = OneShotSendSkipCollector(controller, greedy=False)
    logical = f"new_training_s{int(training_seed)}_e{int(episode_index)}"
    label = f"train_s{int(training_seed)}_e{int(episode_index)}_r43"
    row = _execute_cell(
        label, logical, "train", int(scenario_seed),
        {"arm": "NEW", "training_seed": int(training_seed),
         "episode_index": int(episode_index), "scenario_seed": int(scenario_seed)},
        collector=collector,
        runner_kwargs={
            "policy": "learned", "attach": collector.policy(), "controller": controller,
            "collector": collector,
            "action_rng_seed": runner.action_seed(training_seed, episode_index),
            "tag": f"r43_train_s{training_seed}_e{episode_index}",
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
    reference_root = ROUND42_OUTPUT / "rows"
    for path in sorted(reference_root.glob("v2_reference_s*_a*.json")):
        row = _json_read(path)
        scenario = int(row["scenario_seed"])
        if row.get("reference_arm") != "immediate":
            continue
        if scenario in cache and cache[scenario]["behaviour_sha256"] != row["behaviour_sha256"]:
            raise RuntimeError(f"scenario {scenario} has conflicting immediate references")
        cache[scenario] = row
    if len(cache) != 64:
        raise RuntimeError(f"ROUND43 requires the 64 identity-checked ROUND42 references, found {len(cache)}")
    return cache


@contextlib.contextmanager
def _registered_ep0_rng_after_init(runtime: dict[str, Any], training_seed: int,
                                   rng_state: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Apply the registered training RNG after the same-seed fresh model is constructed."""
    module = runtime["controller_module"]
    trainer = runtime["trainer"]
    original = module.fresh_controller
    status: dict[str, Any] = {"done": False}

    def fresh_controller_with_registered_rng(seed: int) -> Any:
        controller = original(seed)
        if not status["done"] and int(seed) == int(training_seed):
            status["fresh_controller_rng"] = trainer.rng_state()
            trainer.restore_rng(rng_state)
            status["done"] = True
        return controller

    module.fresh_controller = fresh_controller_with_registered_rng
    try:
        yield status
    finally:
        module.fresh_controller = original


def train_seed(training_seed: int, *, episodes: int, frozen: dict[str, Any]) -> dict[str, Any]:
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("ROUND42 runtime is not configured")
    r40_train = runtime["r40_train"]
    checkpoints = runtime["common"].CHECKPOINTS
    rolling = checkpoints / f"v2_train_s{int(training_seed)}.pt"
    prior_updates = list(checkpoints.glob(f"v2_train_s{int(training_seed)}_after_b*.pt"))
    fresh_start = not rolling.exists() and not prior_updates
    start_payload: dict[str, Any] | None = None
    if fresh_start:
        registered = frozen["round43"]["registered_ep0_inputs"][str(int(training_seed))]
        path = Path(registered["path"])
        expected_path = ROUND42_OUTPUT / "checkpoints" / f"v2_train_s{int(training_seed)}_ep0.pt"
        if path.resolve() != expected_path.resolve() or not path.is_file():
            raise RuntimeError(f"seed {training_seed}: registered ROUND42 ep0 RNG input path changed")
        if sha256(path) != registered["file_sha256"]:
            raise RuntimeError(f"seed {training_seed}: registered ROUND42 ep0 RNG input hash changed")
        payload = torch_load_cpu(path)
        state = payload.get("training_state", {})
        if (int(state.get("episodes_done", -1)) != 0
                or int(state.get("updates", -1)) != 0
                or int(state.get("optimizer_steps", -1)) != 0
                or int(state.get("training_seed", -1)) != int(training_seed)
                or state.get("training_version") != frozen["round42"]["source_training_version"]
                or payload.get("training_version") != frozen["round42"]["source_training_version"]
                or float(state.get("gae_lambda", float("nan"))) != 0.95
                or float(state.get("gamma", float("nan"))) != float(
                    frozen["round43"]["gamma"])
                or float(state.get("lambda", float("nan"))) != 1.0
                or payload.get("comm_action_semantics") != "one-shot-send-skip/1"
                or state.get("comm_action_semantics") != "one-shot-send-skip/1"):
            raise RuntimeError(f"seed {training_seed}: registered RNG input is not ROUND42 ep0")
        # The model and optimizer are freshly built by the frozen ROUND40 adapter. Restore the
        # registered ep0 RNG immediately after fresh_controller(seed), so initialization cannot
        # overwrite the registered rollout/update RNG before the first episode.
        start_payload = payload
        runtime.setdefault("rng_priming", {})[str(int(training_seed))] = {
            "source": str(path), "source_sha256": registered["file_sha256"],
            "loaded_model": False, "loaded_optimizer": False,
            "episodes_done": 0, "updates": 0, "optimizer_steps": 0,
        }
    cache_rows = reference_cache()
    cache = {scene: row for scene, row in cache_rows.items()}
    scenario_order = runtime["common"].scenario_order()
    missing = sorted(set(scenario_order[:episodes]) - set(cache))
    if missing:
        raise RuntimeError(f"ROUND42 immediate reference cache is incomplete: {missing}")
    r40_train.common = runtime["common"]
    r40_train.runner = runtime["runner"]
    r40_train.trainer = runtime["trainer"]
    if fresh_start:
        assert start_payload is not None
        prime_context = _registered_ep0_rng_after_init(
            runtime, int(training_seed), start_payload["rng"])
    else:
        prime_context = contextlib.nullcontext({"done": True})
    with prime_context as priming:
        report = r40_train.train_seed(
            int(training_seed), episodes=int(episodes),
            freeze=runtime["round42_freeze"], reference_cache=cache)
    if fresh_start and not priming["done"]:
        raise RuntimeError(f"seed {training_seed}: registered ep0 RNG was not applied after initialization")
    if fresh_start:
        runtime["rng_priming"][str(int(training_seed))]["applied_after_fresh_controller"] = True
    report["comm_action_semantics"] = "one-shot-send-skip/1"
    report["training_version"] = frozen["round42"]["training_version"]
    report["rng_priming"] = dict(runtime.get("rng_priming", {}).get(
        str(int(training_seed)), {"source": "same_round43_process_resume",
                                  "loaded_model": False, "loaded_optimizer": False}))
    return report


def load_eval_model(arm: str, training_seed: int) -> tuple[Any, dict[str, Any]]:
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("ROUND42 runtime is not configured")
    trainer = runtime["trainer"]
    if int(training_seed) not in SEEDS:
        raise ValueError(f"unregistered training seed {training_seed}")
    if arm == "OLD":
        source = _json_read(ROUND42_OUTPUT / "ROUND42_CHECKPOINT_INDEX.json")[
            "new_checkpoints"][str(training_seed)]["128"]
        path = Path(source["path"])
        if sha256(path) != source["file_sha256_after"]:
            raise RuntimeError(f"ROUND42 OLD read-only checkpoint hash changed: {path}")
        controller, state, identity = trainer.load_checkpoint_read_only(path)
        expected = source["parameter_identity"]["parameter_sha256"]
        if controller.parameter_summary()["parameter_sha256"] != expected:
            raise RuntimeError(f"ROUND42 OLD parameter identity changed: seed {training_seed}")
        if (state.get("training_version") != runtime["frozen"]["round42"][
                "source_training_version"]
                or int(state.get("episodes_done", -1)) != 128):
            raise RuntimeError(f"ROUND42 OLD checkpoint is not NEW ep128: seed {training_seed}")
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
        if (int(state.get("episodes_done", -1)) != 128
                or float(state.get("gae_lambda", -1)) != 1.0
                or float(state.get("gamma", -1)) != 1.0):
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
    label = f"eval_{arm.lower()}_s{training_seed}_{mode}_s{int(scenario)}_r43"
    action_seed = runner.eval_action_seed(training_seed, scenario, mode)
    return _execute_cell(
        label, logical, "eval", int(scenario),
        {"arm": arm, "training_seed": int(training_seed), "mode": mode,
         "scenario_seed": int(scenario)},
        collector=collector,
        runner_kwargs={
            "policy": "learned", "attach": collector.policy(), "controller": controller,
            "collector": collector, "action_rng_seed": action_seed,
            "tag": f"r43_eval_{arm.lower()}_s{training_seed}_{mode}",
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
    ppo_update_calls = int(_PPO_UPDATE_CALLS)
    carryover_path = RUN_DIR / "ROUND42R_CARRYOVER.json"
    carryover = _json_read(carryover_path) if carryover_path.is_file() else {}
    legacy_constructed = int(carryover.get("old_failure", {}).get(
        "joint_run_constructed_calls", 0))
    completed_jobs = {
        str(item.get("logical_job")) for item in claims
        if item.get("status") == "complete"
    }
    if strict and optimizer_steps_from_batches != optimizer_steps_actual:
        raise RuntimeError(
            f"optimizer step count mismatch: batch records={optimizer_steps_from_batches}, "
            f"observed torch Adam.step={optimizer_steps_actual}"
        )
    if strict and len(batch_records) != ppo_update_calls:
        raise RuntimeError(
            f"PPO update count mismatch: batch records={len(batch_records)}, "
            f"observed r42 trainer.ppo_update={ppo_update_calls}"
        )
    ledger["counts"] = {
        "normal_used": int(ledger["normal_used"]),
        "reserve_used": int(ledger["reserve_used"]),
        "total_used": int(ledger["total_used"]),
        "environment_constructor_calls": sum(
            int(item.get("environment_constructor_calls", 0)) for item in claims),
        "joint_run_build_failures": sum(
            int(item.get("joint_run_build_failures", 0)) for item in claims),
        "joint_run_constructed_calls": sum(
            int(item.get("joint_run_constructed_calls", 0)) for item in claims)
            + legacy_constructed,
        "run_invocation_calls": sum(
            int(item.get("run_invocation_calls", 0)) for item in claims),
        "joint_run_method_calls": sum(
            int(item.get("joint_run_method_calls", 0)) for item in claims),
        "run_return_calls": sum(
            int(item.get("run_return_calls", 0)) for item in claims),
        "run_failure_calls": sum(
            int(item.get("run_failure_calls", 0)) for item in claims),
        "completed_steps": sum(int(item.get("completed_steps", 0)) for item in claims),
        "completed_logical_jobs": len(completed_jobs),
        "simulation_started": sum(bool(item.get("simulation_started")) for item in claims),
        "technical_failed_claims": sum(item.get("status") == "failed_technical"
                                        for item in claims),
        "retry_claims": sum(int(item.get("attempt", 1)) > 1 for item in claims),
        "optimizer_step_calls": optimizer_steps_actual,
        "optimizer_step_journal_entries": sum(
            1 for _line in (RUN_DIR / "runtime" / "optimizer_step_calls.jsonl").open(
                "r", encoding="utf-8")
        ) if (RUN_DIR / "runtime" / "optimizer_step_calls.jsonl").is_file() else 0,
        "ppo_update_calls": ppo_update_calls,
        "ppo_update_journal_entries": sum(
            1 for _line in (RUN_DIR / "runtime" / "ppo_update_calls.jsonl").open(
                "r", encoding="utf-8")
        ) if (RUN_DIR / "runtime" / "ppo_update_calls.jsonl").is_file() else 0,
        "ppo_update_count_matches_committed_batches": len(batch_records) == ppo_update_calls,
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
    path = RUN_DIR / "runtime" / "ROUND43_EXECUTION_LEDGER.json"
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
