"""Build the ROUND46 /6 freeze artifacts for the relocated execution namespace.

The /6 runner changed (the live output root moved to `v6/`), so its
`runner_source_sha256` and namespace-scoped artifacts must be regenerated. All
*inputs* (R44/R45/R43 data, the frozen training semantics, the fixed key plan)
are unchanged and are re-derived rather than hand-copied. Historical `/5`
receipts and stop evidence are preserved on disk untouched.

Writes: V6 key plan, config diff, training-version manifest, input identity,
runtime freeze, text-identity regression, status-guard reuse, execution freeze
and prelaunch gate. Zero environment, updates, optimizer steps, checkpoint
writes and claims.
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


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    r = _load("r46_v6_runner", R2_DIR / "r46_execute_r2.py")
    for path in (r.V6_KEY_PLAN_PATH, r.V6_FROZEN_CONFIG_PATH, r.V6_PRELAUNCH_GATE_PATH):
        if path.exists():
            raise RuntimeError(f"/6 freeze already exists: {path}")

    order, sequence_record = r._source_scenario_order()
    plan = r._logical_key_plan(order, sequence_record)
    _write(r.V6_KEY_PLAN_PATH, plan)

    text = r.run_r2_v5_text_identity_regression()
    if text.get("status") != "passed":
        raise RuntimeError("V6 text identity regression did not pass")

    r._configure_runtime_module()
    r45_module = r._RUNTIME["module"]

    v5_diff = _json(R2_DIR / "ROUND46_R2_V5_CONFIGURATION_DIFF.json")
    diff = dict(v5_diff)
    diff["runner_source_sha256"] = _sha256(R2_DIR / "r46_execute_r2.py")
    diff["v6_private_output_namespace"] = str(r.V6_DIR)
    _write(r.V6_CONFIG_DIFF_PATH, diff)

    v5_tv = _json(R2_DIR / "ROUND46_R2_V5_TRAINING_VERSION_MANIFEST.json")
    tv = dict(v5_tv)
    tv["runner_source_sha256"] = _sha256(R2_DIR / "r46_execute_r2.py")
    _write(r.V6_TRAINING_VERSION_PATH, tv)

    inputs = r._input_identity()
    _write(r.V6_INPUT_IDENTITY_PATH, inputs)

    runtime_freeze = {
        "kind": "round46-r2-v6-runtime-freeze/1",
        "runner_version": r.ROUND46_RUNNER_VERSION,
        "training_version": r.TRAINING_VERSION,
        "frozen_runtime_source_sha256": r45_module._source_hashes(),
        "v5_runtime_freeze_sha256": _sha256(R2_DIR / "ROUND46_R2_V5_RUNTIME_FREEZE.json"),
        "v6_private_output_namespace": str(r.V6_DIR),
        "output_roots": {
            "run_dir": str(r.V6_DIR), "runtime": str(r.V6_RUNTIME_DIR),
            "checkpoints": str(r.V6_CHECKPOINTS_DIR), "progress": str(r.V6_PROGRESS_DIR),
            "logs": str(r.V6_LOGS_DIR), "rows": str(r.V6_ROWS_DIR),
            "rejected": str(r.V6_REJECTED_DIR),
            "family_ledger": str(r.FAMILY_LEDGER_PATH),
            "family_journal": str(r.FAMILY_JOURNAL_PATH),
            "checkpoint_journal": str(r.CHECKPOINT_JOURNAL_PATH),
        },
    }
    _write(r.V6_RUNTIME_FREEZE_PATH, runtime_freeze)

    text_manifest: dict[str, Any] = {}
    for path in (
        R2_DIR / "r46_execute_r2.py",
        r.READBACK_REGRESSION_PATH,
        r.CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH,
        r.READBACK_REGRESSION_RESULT_PATH,
        r.STATUS_GUARD_REGRESSION_PATH,
        r.V3_FROZEN_CONFIG_PATH,
        r.V3_CONFIG_DIFF_PATH,
        r.V3_INPUT_IDENTITY_PATH,
        r.V3_KEY_PLAN_PATH,
        r.V3_TRAINING_VERSION_PATH,
        r.V3_RUNTIME_FREEZE_PATH,
        r.V3_PRELAUNCH_GATE_PATH,
        r.V3_PRELAUNCH_SHA_PATH,
        r.V3_STOP_ARCHIVE_PATH,
        r.V4_BINDING_AUDIT_PATH,
    ):
        relative = path.resolve().relative_to(REPO).as_posix()
        text_manifest[relative] = r._capture_committed_text_identity(path)

    frozen = {
        "kind": "round46-r2-v5-execution-freeze/1",
        "status": "prepared_before_environment",
        "branch": r.ROUND46_BRANCH,
        "runner_version": r.ROUND46_RUNNER_VERSION,
        "training_version": r.TRAINING_VERSION,
        "runner_path": (R2_DIR / "r46_execute_r2.py").relative_to(REPO).as_posix(),
        "runner_source_sha256": _sha256(R2_DIR / "r46_execute_r2.py"),
        "readback_regression_path": r.READBACK_REGRESSION_PATH.relative_to(REPO).as_posix(),
        "readback_regression_sha256": _sha256(r.READBACK_REGRESSION_PATH),
        "text_identity_regression_sha256": _sha256(r.V5_TEXT_IDENTITY_REGRESSION_PATH),
        "key_plan_sha256": _sha256(r.V6_KEY_PLAN_PATH),
        "configuration_diff_sha256": _sha256(r.V6_CONFIG_DIFF_PATH),
        "training_version_manifest_sha256": _sha256(r.V6_TRAINING_VERSION_PATH),
        "input_identity_sha256": _sha256(r.V6_INPUT_IDENTITY_PATH),
        "runtime_freeze_sha256": _sha256(r.V6_RUNTIME_FREEZE_PATH),
        "inputs": inputs,
        "text_file_identity_manifest": text_manifest,
        "v6_private_output_namespace": str(r.V6_DIR),
        "v5_preserved": {
            "v5_frozen_config_sha256": _sha256(R2_DIR / "ROUND46_R2_V5_FROZEN_CONFIG.json"),
            "v5_prelaunch_gate_sha256": _sha256(R2_DIR / "ROUND46_R2_V5_PRELAUNCH_GATE.json"),
            "v5_binding_regression_sha256": _sha256(R2_DIR / "ROUND46_R2_V5_BINDING_REGRESSION.json"),
        },
    }
    _write(r.V6_FROZEN_CONFIG_PATH, frozen)

    readback = _json(r.READBACK_REGRESSION_RESULT_PATH)
    route = _json(r.CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH)
    committed_text_checks: dict[str, dict[str, str]] = {}
    for relative in text_manifest:
        committed_text_checks[relative] = r._frozen_text_identity(REPO / Path(relative))

    before = r._RUNTIME
    runtime = r._configure_runtime_module() if before is None else before
    ledger_before = runtime["r42_run"].ledger_snapshot(strict=True)

    gate = {
        "kind": "round46-r2-v5-zero-environment-preflight/1",
        "status": "passed",
        "round": 46,
        "branch": r.ROUND46_BRANCH,
        "head_at_preflight": r._run_git("rev-parse", "HEAD"),
        "runner_version": r.ROUND46_RUNNER_VERSION,
        "training_version": r.TRAINING_VERSION,
        "runner_path": frozen["runner_path"],
        "runner_source_sha256": frozen["runner_source_sha256"],
        "frozen_config_sha256": _sha256(r.V6_FROZEN_CONFIG_PATH),
        "key_plan_sha256": _sha256(r.V6_KEY_PLAN_PATH),
        "input_identity_sha256": _sha256(r.V6_INPUT_IDENTITY_PATH),
        "text_identity_regression_sha256": _sha256(r.V5_TEXT_IDENTITY_REGRESSION_PATH),
        "text_identity_regression_status": text["status"],
        "text_file_identity_manifest": _json(r.V6_FROZEN_CONFIG_PATH).get(
            "text_file_identity_manifest", {}),
        "committed_text_identity_checks": committed_text_checks,
        "normal_cap": r.NORMAL_CAP, "technical_reserve_max": r.RESERVE_CAP,
        "hard_cap": r.HARD_CAP,
        "checkpoint_readback_regression": {
            "kind": readback["kind"], "status": readback["status"],
            "source_sha256": frozen["readback_regression_sha256"],
            "receipt_sha256": _sha256(r.READBACK_REGRESSION_RESULT_PATH),
            "reused_prior_receipt": True,
        },
        "checkpoint_journal_route_regression": {
            "kind": route["kind"], "status": route["status"],
            "receipt_path": r.CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH.relative_to(REPO).as_posix(),
            "receipt_sha256": _sha256(r.CHECKPOINT_JOURNAL_ROUTE_REGRESSION_PATH),
            "checkpoint_sha256": route["checkpoint_sha256"],
            "trainer_save_calls_delta": route["trainer_save_calls_delta"],
            "claims": 0, "environment_constructor_calls": 0,
            "optimizer_step_calls": 0, "reused_prior_receipt": True,
        },
        "v4_stop_preserved": {
            "failure_sha256": _sha256(r.V4_PREFLIGHT_FAILURE_PATH),
            "stop_report_sha256": _sha256(r.V4_STOP_REPORT_PATH),
            "status": _json(r.V4_PREFLIGHT_FAILURE_PATH)["status"],
            "claims": 0, "environment_constructor_calls": 0, "optimizer_step_calls": 0,
            "v4_prelaunch_gate_absent": not r.V4_PRELAUNCH_GATE_PATH.exists(),
            "v4_binding_regression_absent": not r.V4_BINDING_REGRESSION_PATH.exists(),
        },
        "v6_private_output_namespace": str(r.V6_DIR),
        "v6_isolation_registry_sha256": _sha256(
            R2_DIR / "ROUND46_R2_V6_EP0_RECEIPT_ISOLATION.json"),
        "environment_constructed": 0,
        "run_invocation_calls": 0,
        "completed_episodes": 0,
        "ppo_updates": 0,
        "optimizer_step_calls": 0,
        "claims": 0,
        "network_or_push": False,
    }
    _write(r.V6_PRELAUNCH_GATE_PATH, gate)
    r.V6_PRELAUNCH_SHA_PATH.write_text(_sha256(r.V6_PRELAUNCH_GATE_PATH) + "\n",
                                       encoding="ascii")

    ledger_after = runtime["r42_run"].ledger_snapshot(strict=True)
    if ledger_before != ledger_after:
        raise RuntimeError("/6 freeze changed the real ledger")
    print(json.dumps({
        "status": "frozen",
        "frozen_config": str(r.V6_FROZEN_CONFIG_PATH),
        "prelaunch_gate": str(r.V6_PRELAUNCH_GATE_PATH),
        "prelaunch_gate_sha256": _sha256(r.V6_PRELAUNCH_GATE_PATH),
        "runner_source_sha256": frozen["runner_source_sha256"],
        "counts": ledger_after.get("counts"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
