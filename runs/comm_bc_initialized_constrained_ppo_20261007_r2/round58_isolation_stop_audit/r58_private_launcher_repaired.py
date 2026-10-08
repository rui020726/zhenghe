"""Private, execution-disabled ROUND58 adapter using the real R58/R40 route.

The historic launcher remains byte-for-byte untouched. This audit launcher can
configure and bind the exact live modules in a private output root, but cannot
start work: a new authorization, freeze identity, and cumulative quota receipt
are not present in this checkout.
"""
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
GEN2 = HERE.parent / "round58_gen2"
R53 = HERE.parent / "round53_critic_isolation_trial_readiness"
R45 = HERE.parents[2] / "runs" / "comm_rule_warmstart_cost_learning_20261006_r2"
CANON = HERE.parents[2].parent / "zhenghe" / "runs" / "comm_cost_v2_rerun_20260928"

for _path in (GEN2, R53, R45, CANON):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import r58_gate_prior_repaired as repaired_adapter  # noqa: E402
import r58_repaired_entry as resume_guard  # noqa: E402


def execute_authorized(*, authorization: dict[str, Any] | None = None,
                       freeze_identity: dict[str, Any] | None = None,
                       cumulative_quota: dict[str, Any] | None = None) -> None:
    """Fail closed; this audit build deliberately has no execution capability."""
    if not isinstance(authorization, dict) or authorization.get("status") != "explicitly_authorized":
        raise PermissionError("ROUND58 execution denied: new explicit authorization is required")
    if not isinstance(freeze_identity, dict) or not isinstance(cumulative_quota, dict):
        raise PermissionError("ROUND58 execution denied: freeze identity and cumulative quota receipt are required")
    raise RuntimeError("ROUND58 execution disabled in the zero-environment audit adapter")


def load_legacy_launcher():
    """Load the preserved launcher module without invoking its main entry."""
    path = GEN2 / "r58_execute.py"
    spec = importlib.util.spec_from_file_location("round58_legacy_launcher_readonly", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load preserved ROUND58 launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_actual_runtime_for_audit(raw_root: Path, *, writer=None) -> dict[str, Any]:
    """Configure actual R46/R45/R40 objects and bind the repaired R58 adapter.

    This configures module objects and private paths only. It does not call the
    episode provider, optimizer update, checkpoint restore/save, or task runner.
    """
    import torch
    legacy = load_legacy_launcher()
    raw_root = Path(raw_root).resolve()
    if raw_root == GEN2.resolve() or GEN2.resolve() not in raw_root.parents:
        raise RuntimeError("audit runtime output must be a fresh child of, not the root of, gen2")
    # bind_runner_paths enforces containment under the launcher's immutable HERE.
    legacy.RAW = raw_root
    base = legacy.load_base_runner()
    legacy.bind_runner_paths(base, raw_root)
    legacy.install_private_configure_seams(base, allow_existing_journal=False)
    legacy.install_protected_dirty_file_guard(base)
    base._R45_MODULE = None
    base._RUNTIME = None
    # The legacy runtime configure mutates GAE_LAMBDA to the frozen full-trace
    # value. A prior in-process audit bind must not be mistaken for source input.
    import r45_bc  # noqa: PLC0415
    r45_bc.configure_source_imports()
    import v2_common  # noqa: PLC0415
    import r42_versions  # noqa: PLC0415
    source_frozen = legacy.read_json(legacy.R44_DIR / "ROUND44_FROZEN_CONFIG.json")
    v2_common.GAE_LAMBDA = float(source_frozen["round43"]["gae_lambda_old"])
    r42_versions.TRAINING_VERSION = source_frozen["round42"]["training_version"]
    for alias in legacy.RUNTIME_CODE_ALIASES:
        sys.modules.pop(Path(alias).stem, None)
    if str(raw_root) in sys.path:
        sys.path.remove(str(raw_root))
    sys.path.insert(0, str(raw_root))
    original_installer = legacy.install_gate_prior
    installed_receipts = []
    def install_repaired(trainer, detach_context, output_root):
        kwargs = {} if writer is None else {"writer": writer}
        receipt = repaired_adapter.install_gate_prior(
            trainer, detach_context, Path(output_root), **kwargs)
        installed_receipts.append(receipt)
        return receipt
    legacy.install_gate_prior = install_repaired
    original_write_bytes = Path.write_bytes
    original_open = Path.open
    original_touch = Path.touch
    journal_target = (raw_root / "runtime" / "checkpoint_writes.jsonl").resolve()
    journal_route_spy = []
    def intercept_empty_journal(path, data):
        if Path(path).resolve() == journal_target:
            journal_route_spy.append({"path": str(journal_target), "bytes": len(data),
                                      "action": "in-memory-only"})
            return len(data)
        return original_write_bytes(path, data)
    def intercept_journal_open(path, mode="r", *args, **kwargs):
        if Path(path).resolve() == journal_target and any(flag in mode for flag in ("w", "a", "x", "+")):
            journal_route_spy.append({"path": str(journal_target), "mode": mode,
                                      "action": "in-memory-only"})
            return io.BytesIO() if "b" in mode else io.StringIO()
        return original_open(path, mode, *args, **kwargs)
    def intercept_journal_touch(path, *args, **kwargs):
        if Path(path).resolve() == journal_target:
            journal_route_spy.append({"path": str(journal_target), "action": "in-memory-only-touch"})
            return path
        return original_touch(path, *args, **kwargs)
    Path.write_bytes = intercept_empty_journal
    Path.open = intercept_journal_open
    Path.touch = intercept_journal_touch
    try:
        base._configure_runtime_module()
        runtime = base._RUNTIME
        if not isinstance(runtime, dict):
            raise RuntimeError("actual Round46 configure did not produce runtime object map")
        plan = legacy.plan_from_frozen_sources()
        live = legacy.bind_live_runtime(base, runtime, plan, checkpoint_mode="forbid_existing")
    finally:
        Path.write_bytes = original_write_bytes
        Path.open = original_open
        Path.touch = original_touch
        legacy.install_gate_prior = original_installer
    if len(installed_receipts) != 1:
        raise RuntimeError(f"expected one repaired prior installation, found {len(installed_receipts)}")
    runtime["r58_repaired_adapter"] = installed_receipts[0]

    trainer = runtime["trainer"]
    r40 = runtime["r40_train"]
    if r40.trainer is not trainer:
        raise RuntimeError("actual R40 training module is not bound to the configured trainer object")
    closure_values = [cell.cell_contents for cell in (getattr(trainer.ppo_update, "__closure__", None) or ())]
    compiled_updates = [value for value in closure_values
                        if callable(value) and getattr(value, "__code__", None)
                        and value.__code__.co_filename.startswith("<r58-gate-ppo-update-repaired")]
    if len(compiled_updates) != 1 or compiled_updates[0].__globals__.get("common") is not runtime["common"]:
        raise RuntimeError("actual trainer ppo_update globals do not point at configured common")
    if runtime.get("r58_repaired_adapter", {}).get("kind") != "r58-gate-prior-repaired/1":
        raise RuntimeError("actual R58 runtime did not install the repaired prior adapter")
    return {"legacy": legacy, "base": base, "runtime": runtime,
            "live": live, "torch": torch, "raw_root": raw_root,
            "trainer_object_id": id(trainer), "r40_trainer_object_id": id(r40.trainer),
            "compiled_ppo_object_id": id(compiled_updates[0]),
            "checkpoint_journal_io_spy": journal_route_spy,
            "trainer_source": str(Path(trainer.ppo_update.__wrapped__.__code__.co_filename).resolve())
            if hasattr(trainer.ppo_update, "__wrapped__") else str(Path(trainer.ppo_update.__code__.co_filename).resolve())}


def resume_from_payload_before_any_restore(payload: dict[str, Any],
                                           expected_repair_identity: dict[str, Any],
                                           *, write_state, restore_model, restore_optimizer,
                                           restore_rng, construct_environment,
                                           optimizer_step):
    """The real repaired resume branch; legacy attestation is checked first."""
    resume_guard.require_repaired_resume(payload, expected_repair_identity)
    write_state(payload)
    model = restore_model(payload)
    optimizer = restore_optimizer(payload)
    restore_rng(payload)
    env = construct_environment()
    optimizer_step()
    return model, optimizer, env


def close_audit_runtime(binding: dict[str, Any]) -> None:
    runtime = binding["runtime"]
    adapter = runtime.get("r58_repaired_adapter")
    if adapter and callable(adapter.get("undo")):
        adapter["undo"]()
    undo = binding.get("live", {}).get("route_undo")
    if callable(undo):
        undo()
