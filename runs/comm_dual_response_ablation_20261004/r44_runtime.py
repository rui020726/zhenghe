"""ROUND44 runtime adapter for the verified ROUND43 runner and R40 resume seam."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from r44_eval_index import create_claim_artifact_receipt
from r44_prepare import RUN_DIR, ROOT, R43_DIR, TRAINING_VERSION, RUNNER_VERSION, ETA_NEW


INPUT_ROOT = ROOT.parent / "zhenghe"
ROUND40_ROOT = ROOT.parent / "zhenghe-round40"
ROUND40_OUTPUT = ROUND40_ROOT / "runs" / "comm_first_report_training_ablation_20261003"
ROUND42_OUTPUT = ROOT / "runs" / "comm_one_shot_gate_resume_20261003_v2"
R43_FREEZE_PATH = R43_DIR / "ROUND43_FROZEN_CONFIG.json"
R44_FREEZE_PATH = RUN_DIR / "ROUND44_FROZEN_CONFIG.json"
PRELAUNCH_PATH = RUN_DIR / "runtime" / "ROUND44_PRELAUNCH_IDENTITY.json"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_frozen_source(frozen: dict[str, Any]) -> str:
    """Check committed R44 code/config identities before each atomic environment claim."""
    if frozen.get("status") != "prelaunch_frozen":
        raise RuntimeError("ROUND44 configuration is not in prelaunch_frozen state")
    if not PRELAUNCH_PATH.is_file():
        raise RuntimeError("ROUND44 exact prelaunch identity receipt is missing")
    receipt = _read(PRELAUNCH_PATH)
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    if head != receipt.get("head") or branch != "exp/comm-dual-response-ablation":
        raise RuntimeError("ROUND44 branch/HEAD differs from the committed prelaunch identity")
    if _git("status", "--porcelain"):
        raise RuntimeError("ROUND44 tracked worktree is dirty; refusing an environment claim")
    if receipt.get("branch") != branch or receipt.get("status") != "passed":
        raise RuntimeError("ROUND44 prelaunch receipt branch/status mismatch")
    for relative, expected in receipt.get("source_code_sha256", {}).items():
        path = RUN_DIR / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"ROUND44 frozen executor source changed: {relative}")
    for name, path in (
        ("frozen_config_sha256", RUN_DIR / "ROUND44_FROZEN_CONFIG.json"),
        ("configuration_diff_sha256", RUN_DIR / "ROUND44_CONFIGURATION_DIFF.json"),
        ("baseline_reuse_sha256", RUN_DIR / "ROUND44_BASELINE_REUSE.json"),
        ("zero_environment_gate_sha256", RUN_DIR / "ROUND44_ZERO_ENV_GATES.json"),
    ):
        if not path.is_file() or _sha256(path) != receipt.get(name):
            raise RuntimeError(f"ROUND44 frozen identity mismatch: {name}")
    source_identity = _read(R43_FREEZE_PATH)["execution_identity"]
    r43_branch_head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "exp/comm-one-shot-full-trace-ablation"],
        check=True, capture_output=True, text=True).stdout.strip()
    if r43_branch_head != receipt.get("reused_round43_delivery_head"):
        raise RuntimeError("ROUND44 read-only OLD source branch no longer matches its frozen head")
    for key, root in (
        ("canonical_read_only_input_head", INPUT_ROOT),
        ("round40_delivery_head", ROUND40_ROOT),
    ):
        expected = source_identity[key]
        actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
        if actual != expected:
            raise RuntimeError(f"ROUND44 frozen source checkout changed: {key}={actual}")
    round41_root = Path(source_identity["round41_worktree"])
    actual_round41 = subprocess.run(["git", "-C", str(round41_root), "rev-parse", "HEAD"],
                                    check=True, capture_output=True, text=True).stdout.strip()
    if actual_round41 != source_identity["round41_delivery_head"]:
        raise RuntimeError(f"ROUND44 frozen ROUND41 source checkout changed: {actual_round41}")
    return head


def _load_r43_runner():
    """Load the archived runner source file by its frozen path, without copying it."""
    sys.path[:] = [str(RUN_DIR), str(R43_DIR), str(ROUND40_OUTPUT), str(INPUT_ROOT)] + [
        entry for entry in sys.path
        if entry not in {str(RUN_DIR), str(R43_DIR), str(ROUND40_OUTPUT), str(INPUT_ROOT)}
    ]
    spec = importlib.util.spec_from_file_location("r42_run", R43_DIR / "r42_run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen R43 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules["r42_run"] = module
    spec.loader.exec_module(module)
    module.RUN_DIR = RUN_DIR
    module.ROUND42_ROOT = ROOT
    module.INPUT_ROOT = INPUT_ROOT
    module.RUNNER_VERSION = RUNNER_VERSION
    return module


def _install_claim_and_ledger_adapter(run_module: Any, frozen: dict[str, Any]) -> None:
    def claim(logical_job: str, *, attempt: int, metadata: dict[str, Any]) -> Path:
        if run_module._ACTIVE_CLAIM is not None:
            raise RuntimeError("nested ROUND44 environment claims are not allowed")
        runtime = run_module._RUNTIME
        if runtime is None:
            raise RuntimeError("ROUND44 runtime is not configured")
        head = verify_frozen_source(frozen)
        category = "normal" if int(attempt) == 1 else "reserve"
        claim_path = runtime["r40_common"].claim_launch(
            logical_job, category=category, attempt=int(attempt), metadata=metadata)
        runtime["r40_common"]._update_claim(
            claim_path,
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
            round=44,
            comm_action_semantics="one-shot-send-skip/1",
            training_version=runtime["r42_versions"].TRAINING_VERSION,
            runner_version=RUNNER_VERSION,
        )
        run_module._ACTIVE_CLAIM = claim_path
        return claim_path

    def write_ledger(*, strict: bool = True) -> dict[str, Any]:
        ledger = run_module.ledger_snapshot(strict=strict)
        ledger["kind"] = "round44-execution-ledger/1"
        ledger["round"] = 44
        path = RUN_DIR / "runtime" / "ROUND44_EXECUTION_LEDGER.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
        return ledger

    run_module.verify_frozen_source = verify_frozen_source
    run_module._claim = claim
    run_module.write_execution_ledger = write_ledger


def _install_r44_cells(run_module: Any, runtime: dict[str, Any]) -> None:
    from r42_one_shot import OneShotSendSkipCollector

    def training_episode(common: Any, runner: Any, controller: Any, *,
                         training_seed: int, episode_index: int,
                         scenario_seed: int, lam: float) -> dict[str, Any]:
        collector = OneShotSendSkipCollector(controller, greedy=False)
        logical = f"new_training_s{int(training_seed)}_e{int(episode_index)}"
        label = f"train_s{int(training_seed)}_e{int(episode_index)}_r44"
        row = run_module._execute_cell(
            label, logical, "train", int(scenario_seed),
            {"arm": "NEW", "training_seed": int(training_seed),
             "episode_index": int(episode_index), "scenario_seed": int(scenario_seed)},
            collector=collector,
            runner_kwargs={
                "policy": "learned", "attach": collector.policy(), "controller": controller,
                "collector": collector,
                "action_rng_seed": runner.action_seed(training_seed, episode_index),
                "tag": f"r44_train_s{training_seed}_e{episode_index}",
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

    def model_eval(*, arm: str, training_seed: int, mode: str, scenario: int,
                   controller: Any, identity: dict[str, Any]) -> dict[str, Any]:
        if arm != "NEW":
            raise ValueError("ROUND44 launches only NEW ep128 evaluations")
        if mode not in ("greedy", "sampled"):
            raise ValueError(f"unknown evaluation mode {mode!r}")
        collector = OneShotSendSkipCollector(controller, greedy=(mode == "greedy"))
        logical = f"model_eval_new_s{training_seed}_{mode}_scene{int(scenario)}"
        label = f"eval_new_s{training_seed}_{mode}_s{int(scenario)}_r44"
        action_seed = runtime["runner"].eval_action_seed(training_seed, scenario, mode)
        return run_module._execute_cell(
            label, logical, "eval", int(scenario),
            {"arm": arm, "training_seed": int(training_seed), "mode": mode,
             "scenario_seed": int(scenario)},
            collector=collector,
            runner_kwargs={
                "policy": "learned", "attach": collector.policy(), "controller": controller,
                "collector": collector, "action_rng_seed": action_seed,
                "tag": f"r44_eval_new_s{training_seed}_{mode}",
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

    def load_eval_model(arm: str, training_seed: int):
        trainer = runtime["trainer"]
        seed = int(training_seed)
        if seed not in run_module.SEEDS:
            raise ValueError(f"unregistered seed {seed}")
        if arm == "OLD":
            old = runtime["frozen"]["round42"]["old_models"][str(seed)]
            path = Path(old["path"])
            if not path.is_file() or run_module.sha256(path) != old["file_sha256"]:
                raise RuntimeError(f"R43 OLD ep128 file identity changed: seed={seed}")
            controller, state, identity = trainer.load_checkpoint_read_only(path)
            if (state.get("training_version") != runtime["frozen"]["round42"][
                    "old_training_version"]
                    or int(state.get("episodes_done", -1)) != 128
                    or controller.parameter_summary()["parameter_sha256"]
                    != old["parameter_sha256"]
                    or identity.get("optimizer_restored") is not False
                    or identity.get("rng_restored") is not False):
                raise RuntimeError(f"R43 OLD read-only ep128 identity mismatch: seed={seed}")
        elif arm == "NEW":
            return run_module._round44_base_load_eval_model("NEW", seed)
        else:
            raise ValueError(f"unknown model arm {arm!r}")
        identity.update({
            "arm": "OLD_R43_NEW_EP128",
            "path": str(path), "file_sha256": old["file_sha256"],
            "training_seed": seed, "episodes_done": 128,
            "updates": 32, "optimizer_steps": int(old["optimizer_steps"]),
            "lambda": float(old["lambda_dual"]),
            "parameter_identity": controller.parameter_summary(),
            "evaluation_semantics": "one-shot-send-skip/1",
            "source_training_version": old["training_version"],
        })
        return controller, identity

    runtime["runner"]._round44_original_booked = runtime["runner"]._booked
    base_booked = runtime["runner"]._booked

    def booked(label: str, kind: str, row: dict[str, Any], ledger: Any) -> None:
        base_booked(label, kind, row, ledger)
        if kind != "eval":
            return
        active = run_module._ACTIVE_CLAIM
        if active is None:
            raise RuntimeError("ROUND44 eval artifact receipt has no active atomic claim")
        receipt = create_claim_artifact_receipt(
            RUN_DIR, row,
            training_version=runtime["r42_versions"].TRAINING_VERSION,
            runner_version=RUNNER_VERSION,
            checkpoint_parameter_sha256=row["checkpoint_parameter_identity"][
                "parameter_sha256"],
        )
        runtime["r40_common"]._update_claim(active,
                                             round44_eval_artifact_receipt=receipt)

    runtime["runner"]._booked = booked
    runtime["r40_run"].run_training_episode = training_episode
    run_module.run_training_episode = training_episode
    run_module.run_model_eval = model_eval
    run_module._round44_base_load_eval_model = run_module.load_eval_model
    run_module.load_eval_model = load_eval_model


def configure(frozen: dict[str, Any]) -> dict[str, Any]:
    run_module = _load_r43_runner()
    run_module.verify_frozen_source = verify_frozen_source
    run_module.RUN_DIR = RUN_DIR
    run_module.ROUND42_ROOT = ROOT
    run_module.ROUND42_OUTPUT = ROUND42_OUTPUT
    run_module.RUNNER_VERSION = RUNNER_VERSION
    _install_claim_and_ledger_adapter(run_module, frozen)

    # Inject eta at the common-module boundary before r42_versions installs the trainer gate.
    sys.path.insert(0, str(ROUND40_OUTPUT))
    import r40_common
    base_bootstrap = r40_common.bootstrap

    def bootstrap_with_r44_eta(input_root: Any):
        common = base_bootstrap(input_root)
        common.ETA_LAMBDA = ETA_NEW
        return common

    r40_common.bootstrap = bootstrap_with_r44_eta
    round40_frozen = _read(ROUND40_OUTPUT / "ROUND40_FROZEN_CONFIG.json")[
        "round40"]["freeze_identity"]
    runtime = run_module.configure(frozen, round40_frozen)

    common = runtime["common"]
    if (float(common.ETA_LAMBDA) != ETA_NEW
            or float(runtime["trainer"].common.ETA_LAMBDA) != ETA_NEW
            or float(common.GAE_LAMBDA) != 1.0 or float(common.GAMMA) != 1.0):
        raise RuntimeError("actual batch/dual path did not receive frozen eta/GAE/gamma")
    common.ETA_LAMBDA = ETA_NEW
    common.NORMAL_LAUNCHES = 480
    common.RESERVE_LAUNCHES = 24
    common.LAUNCH_CAP = 504
    r40_common.ROUND_DIR = RUN_DIR
    r40_common.NORMAL_CAP = 480
    r40_common.RESERVE_CAP = 24
    r40_common.HARD_CAP = 504
    r40_common.plan = lambda: {
        "normal": {"new_training": 384, "new_final_evaluation": 96},
        "normal_total": 480, "technical_reserve_max": 24, "hard_cap": 504,
    }
    if (int(r40_common.NORMAL_CAP) != 480 or int(r40_common.RESERVE_CAP) != 24
            or int(r40_common.HARD_CAP) != 504
            or int(common.LAUNCH_CAP) != 504):
        raise RuntimeError("ROUND44 launch cap was not connected to the atomic claim layer")

    _install_r44_cells(run_module, runtime)
    import r42_analysis
    r42_analysis.RUN_DIR = RUN_DIR
    runtime["r42_run"] = run_module
    runtime["frozen"] = frozen
    runtime["r42_analysis"] = r42_analysis
    runtime["round42_freeze"] = frozen["round42"]["freeze_identity"]
    return runtime


__all__ = ["INPUT_ROOT", "PRELAUNCH_PATH", "RUN_DIR", "RUNNER_VERSION",
           "TRAINING_VERSION", "configure", "verify_frozen_source"]

