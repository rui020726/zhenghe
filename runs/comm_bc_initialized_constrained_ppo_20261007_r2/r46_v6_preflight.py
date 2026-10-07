"""ROUND46 /6 zero-environment preflight.

The /6 fix does not change the frozen runner: `r46_execute_r2.py` is byte
identical to its /5 identity, and every /5 receipt is preserved in place. The
only /6 change is the filesystem isolation of the three colliding /5 stop
artifacts, which the runner itself refuses to start with.

Because runner and inputs are unchanged, the historical /5 zero-environment
checks are reused by identity rather than re-derived. This preflight instead

  1. re-asserts the runner source identity is the frozen /5 one,
  2. re-verifies the /6 isolation registry and that the active receipt
     namespace and active checkpoint path are clean,
  3. exercises the runner's REAL pre-execution gate
     (`_verify_prelaunch_identity(require_r2_v5_evidence=True)`, the first thing
     `--execute` runs) with zero environment, updates, optimizer steps or
     checkpoint saves,
  4. proves the real ledger is unchanged by the preflight itself.

It constructs no environment, runs no simulation, performs no PPO update, calls
no optimiser step, and saves no checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


R2_DIR = Path(__file__).resolve().parent
REPO = R2_DIR.parents[1]
RUN_DIR = R2_DIR

RUNNER_PATH = RUN_DIR / "r46_execute_r2.py"
RUNNER_SHA256_FROZEN = (
    "ea5597d0ccbe0562d0fc4b87a81708ea0f525747d31900ef9bf46b4a7d228b86"
)
ISOLATION_MODULE_PATH = RUN_DIR / "r46_v6_ep0_receipt_isolation.py"
ISOLATION_REGISTRY_PATH = RUN_DIR / "ROUND46_R2_V6_EP0_RECEIPT_ISOLATION.json"
V6_PREFLIGHT_PATH = RUN_DIR / "ROUND46_R2_V6_PREFLIGHT_GATE.json"
V6_PREFLIGHT_SHA_PATH = RUN_DIR / "ROUND46_R2_V6_PREFLIGHT_GATE.sha256"
ACTIVE_RECEIPT_PATH = RUN_DIR / "runtime" / "ROUND45_R2_EP0_RECOVERY_s11.json"
ACTIVE_EP0_PATH = RUN_DIR / "checkpoints" / "v2_train_s11_ep0.pt"
ACTIVE_ROLLING_PATH = RUN_DIR / "checkpoints" / "v2_train_s11.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _counts(ledger: dict[str, Any]) -> dict[str, Any]:
    counts = ledger.get("counts", {})
    return {
        "normal_used": int(ledger.get("normal_used", -1)),
        "reserve_used": int(ledger.get("reserve_used", -1)),
        "environment_constructor_calls": int(counts.get("environment_constructor_calls", -1)),
        "run_invocation_calls": int(counts.get("run_invocation_calls", -1)),
        "completed_steps": int(counts.get("completed_steps", -1)),
        "ppo_update_calls": int(counts.get("ppo_update_calls", -1)),
        "optimizer_step_calls": int(counts.get("optimizer_step_calls", -1)),
        "checkpoint_write_calls": int(counts.get("checkpoint_write_calls", -1)),
        "checkpoint_write_journal_entries": int(
            counts.get("checkpoint_write_journal_entries", -1)),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    if V6_PREFLIGHT_PATH.exists() or V6_PREFLIGHT_SHA_PATH.exists():
        raise RuntimeError("ROUND46 /6 preflight already exists; refusing to rewrite it")

    runner_sha = _sha256(RUNNER_PATH)
    if runner_sha != RUNNER_SHA256_FROZEN:
        raise RuntimeError(
            "frozen runner source changed; /6 must not re-derive the /5 identity")
    if ISOLATION_MODULE_PATH.parent != RUN_DIR or not ISOLATION_MODULE_PATH.is_file():
        raise FileNotFoundError(ISOLATION_MODULE_PATH)

    isolation = _load("r46_v6_isolation", ISOLATION_MODULE_PATH)
    isolation_state = isolation.verify()
    if isolation_state.get("status") != "active_namespace_clean":
        raise RuntimeError("active namespace is not clean; run --isolate first")
    registry = _json(ISOLATION_REGISTRY_PATH)
    isolation._assert_registry(registry)

    # The runner's own loader must see the identical frozen source.
    runner = _load("r46_v6_runner", RUNNER_PATH)
    if _sha256(RUNNER_PATH) != RUNNER_SHA256_FROZEN:
        raise RuntimeError("runner source changed while importing")

    module = runner._configure_runtime_module()
    runtime = runner._RUNTIME
    assert runtime is not None

    ledger_before = runtime["r42_run"].ledger_snapshot(strict=True)
    counts_before = _counts(ledger_before)

    # The exact first gate --execute runs, with full /5 evidence reuse.
    runner._verify_prelaunch_identity(runtime, require_r2_v5_evidence=True)
    runner._verify_status_guard_regression_receipt()

    ledger_after = runtime["r42_run"].ledger_snapshot(strict=True)
    counts_after = _counts(ledger_after)
    if counts_before != counts_after:
        raise RuntimeError("ROUND46 /6 preflight changed the real ledger")

    active_clean = {
        "receipt_present": ACTIVE_RECEIPT_PATH.is_file(),
        "ep0_present": ACTIVE_EP0_PATH.is_file(),
        "rolling_present": ACTIVE_ROLLING_PATH.is_file(),
    }
    if any(active_clean.values()):
        raise RuntimeError("active namespace collides again; /6 preflight refuses")

    receipt = {
        "kind": "round46-r2-v6-zero-environment-preflight/1",
        "status": "passed",
        "round": 46,
        "runner_version": runner.ROUND46_RUNNER_VERSION,
        "training_version": runner.TRAINING_VERSION,
        "branch": runner._run_git("branch", "--show-current"),
        "head_at_preflight": runner._run_git("rev-parse", "HEAD"),
        "runner_path": RUNNER_PATH.relative_to(REPO).as_posix(),
        "runner_source_sha256": runner_sha,
        "runner_source_unchanged_from_v5": True,
        "reused_preflight_evidence": {
            "v5_prelaunch_gate_path": runner.PRELAUNCH_GATE_PATH.relative_to(REPO).as_posix(),
            "v5_prelaunch_gate_sha256": _sha256(runner.PRELAUNCH_GATE_PATH),
            "v5_prelaunch_sha_pointer_ok": (
                runner.PRELAUNCH_SHA_PATH.read_text(encoding="ascii").strip()
                == _sha256(runner.PRELAUNCH_GATE_PATH)),
            "actual_prelaunch_identity_reverified": True,
            "status_guard_regression_reverified": True,
            "new_preflight_checkpoint_writes": 0,
        },
        "isolation": {
            "registry_path": ISOLATION_REGISTRY_PATH.relative_to(REPO).as_posix(),
            "registry_sha256": _sha256(ISOLATION_REGISTRY_PATH),
            "status": registry["status"],
            "moved_file_count": len(registry.get("moved_files", [])),
            "all_moves_byte_identical": all(
                item.get("byte_identical") is True
                for item in registry.get("moved_files", [])),
            "family_journal_unchanged": registry.get("family_journal_unchanged") is True,
            "family_journal_entries": registry["family_after"]["family_journal_entries"],
            "family_ledger_sha256": registry["family_after"]["family_ledger_sha256"],
        },
        "active_namespace_clean": active_clean,
        "ledger_before": counts_before,
        "ledger_after": counts_after,
        "ledger_unchanged_by_preflight": counts_before == counts_after,
        "environment_constructed": 0,
        "run_invocation_calls": 0,
        "completed_steps": 0,
        "ppo_updates": 0,
        "optimizer_step_calls": 0,
        "claims": 0,
        "network_or_push": False,
    }
    _write_json(V6_PREFLIGHT_PATH, receipt)
    V6_PREFLIGHT_SHA_PATH.write_text(_sha256(V6_PREFLIGHT_PATH) + "\n", encoding="ascii")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
