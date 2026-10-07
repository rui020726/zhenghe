"""ROUND46 /6 ep0 recovery-receipt scope isolation (zero-environment).

The frozen /5 run stopped before its first atomic claim because the reused R45
promotion seam (`r45_r2_execute_r3.py:_promote_ep0_to_rolling`) found a prior
ep0 recovery receipt at its output path and rejected the different checkpoint
identity it was given:

    RuntimeError: seed 11: prior ep0 recovery receipt conflicts

The runner also refuses to start while any `v2_train_s*.pt` sits in
`R46_DIR/checkpoints` (`r46_execute_r2.py:_prior_r2_preclaim_attempt_identity`),
and training writes exactly those files. The /6 fix therefore moves the live
output root to a private directory (`v6/`) while this module keeps the
`R46_DIR` root free of stray untrained ep0 payloads and receipts.

Artifacts are moved with a registered, recoverable isolation: original path,
preserved path, byte size, and before/after SHA256. Nothing is deleted or
overwritten, and every preserved file is re-verified byte-identical. The single
family execution ledger and its checkpoint journal are never touched here.

This module only reads, hashes and moves explicitly identified files. It never
constructs an environment, runs a simulation, performs a PPO update, calls
Adam, or saves a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


R2_DIR = Path(__file__).resolve().parent
REPO = R2_DIR.parents[1]
RUN_DIR = R2_DIR

ACTIVE_RECEIPT_PATH = RUN_DIR / "runtime" / "ROUND45_R2_EP0_RECOVERY_s11.json"
ACTIVE_CHECKPOINTS_DIR = RUN_DIR / "checkpoints"
FAMILY_LEDGER_PATH = RUN_DIR / "runtime" / "ROUND46_EXECUTION_LEDGER.json"
FAMILY_JOURNAL_PATH = RUN_DIR / "runtime" / "checkpoint_writes.jsonl"

ISOLATION_DIR = RUN_DIR / "attempts" / "runner_v5_ep0_receipt_conflict"
REGISTRY_PATH = RUN_DIR / "ROUND46_R2_V6_EP0_RECEIPT_ISOLATION.json"

# Recorded /5 stop identities (wave 1).
WAVE1_RECEIPT_SHA256 = (
    "1a259177a313312ff9686a4831937a304205808656e00e2231577e991b81c300"
)
WAVE1_PAYLOAD_SHA256 = (
    "2ca8f764315795f5aadc89117901f23442c43c15e4d4802f5ae268765467950a"
)
EXPECTED_PAYLOAD_BYTES = 57605
EXPECTED_RECEIPT_KIND = "round45-r2-ep0-rolling-copy-receipt/1"

RUNNER_SHA256_FROZEN = (
    "ea5597d0ccbe0562d0fc4b87a81708ea0f525747d31900ef9bf46b4a7d228b86"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _journal_entries(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _family_state() -> dict[str, Any]:
    return {
        "family_ledger_path": str(FAMILY_LEDGER_PATH.resolve()),
        "family_ledger_sha256": (
            _sha256(FAMILY_LEDGER_PATH) if FAMILY_LEDGER_PATH.is_file() else None),
        "family_journal_path": str(FAMILY_JOURNAL_PATH.resolve()),
        "family_journal_sha256": (
            _sha256(FAMILY_JOURNAL_PATH) if FAMILY_JOURNAL_PATH.is_file() else None),
        "family_journal_entries": _journal_entries(FAMILY_JOURNAL_PATH),
    }


def _active_payloads() -> list[Path]:
    if not ACTIVE_CHECKPOINTS_DIR.is_dir():
        return []
    return sorted(ACTIVE_CHECKPOINTS_DIR.glob("v2_train_s*.pt"))


def _active_state() -> dict[str, Any]:
    return {
        "receipt": {
            "path": str(ACTIVE_RECEIPT_PATH.resolve()),
            "present": ACTIVE_RECEIPT_PATH.is_file(),
            "sha256": (_sha256(ACTIVE_RECEIPT_PATH)
                       if ACTIVE_RECEIPT_PATH.is_file() else None),
        },
        "payloads": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size,
             "sha256": _sha256(path)} for path in _active_payloads()],
    }


def _active_is_clean() -> bool:
    return not ACTIVE_RECEIPT_PATH.is_file() and not _active_payloads()


def _assert_family_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before != after:
        raise RuntimeError("isolation changed the family ledger or checkpoint journal")


def _assert_untrained_ep0_receipt(path: Path) -> dict[str, Any]:
    receipt = _json(path)
    if (receipt.get("kind") != EXPECTED_RECEIPT_KIND
            or receipt.get("status") != "passed"
            or int(receipt.get("seed", -1)) != 11
            or int(receipt.get("episodes_done", -1)) != 0
            or int(receipt.get("updates", -1)) != 0
            or int(receipt.get("optimizer_steps", -1)) != 0
            or receipt.get("byte_identical") is not True):
        raise RuntimeError("active ep0 recovery receipt is not an untrained ep0 stop receipt")
    return receipt


def _move(relative: str, source: Path, wave: str) -> dict[str, Any]:
    before_sha = _sha256(source)
    before_bytes = source.stat().st_size
    destination = ISOLATION_DIR / wave / relative.replace("/", os.sep)
    if destination.exists():
        raise RuntimeError(f"preserved isolation target already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    after_sha = _sha256(destination)
    if after_sha != before_sha or destination.stat().st_size != before_bytes:
        destination.unlink()
        raise RuntimeError(f"preserved copy is not byte-identical: {relative}")
    source.unlink()
    return {
        "wave": wave,
        "original_relative_path": relative,
        "preserved_relative_path": destination.relative_to(RUN_DIR).as_posix(),
        "original_path": str(source.resolve()),
        "preserved_path": str(destination.resolve()),
        "bytes": before_bytes,
        "sha256_before": before_sha,
        "sha256_after": after_sha,
        "byte_identical": after_sha == before_sha,
    }


def verify() -> dict[str, Any]:
    state = _active_state()
    return {
        "kind": "round46-r2-v6-ep0-receipt-isolation-verify/1",
        "status": "active_namespace_clean" if _active_is_clean() else "active_namespace_collides",
        "registry_path": str(REGISTRY_PATH.resolve()),
        "registry_present": REGISTRY_PATH.is_file(),
        "active": state,
        "family": _family_state(),
        "environment_constructor_calls": 0,
        "run_invocation_calls": 0,
        "completed_steps": 0,
        "ppo_updates": 0,
        "optimizer_step_calls": 0,
        "checkpoint_write_calls": 0,
    }


def archive_active(wave: str) -> dict[str, Any]:
    """Archive every untrained-ep0 artifact currently in the active root.

    A receipt must be the recorded wave-1 receipt, or a fresh untrained ep0
    receipt (identical shape, zero episodes/updates/optimizer steps). Payloads
    must all be the 57605-byte untrained ep0 file. Anything else is refused.
    """
    if _active_is_clean():
        raise RuntimeError("active namespace is already clean; nothing to archive")
    family_before = _family_state()

    if ACTIVE_RECEIPT_PATH.is_file():
        receipt = _assert_untrained_ep0_receipt(ACTIVE_RECEIPT_PATH)
        receipt_sha = _sha256(ACTIVE_RECEIPT_PATH)
        receipt_matches_wave1 = receipt_sha == WAVE1_RECEIPT_SHA256
    else:
        receipt = None
        receipt_sha = None
        receipt_matches_wave1 = None

    payloads = _active_payloads()
    for payload in payloads:
        if payload.stat().st_size != EXPECTED_PAYLOAD_BYTES:
            raise RuntimeError(
                f"active payload {payload.name} is not the untrained ep0 size")

    moved: list[dict[str, Any]] = []
    if ACTIVE_RECEIPT_PATH.is_file():
        moved.append(_move("runtime/ROUND45_R2_EP0_RECOVERY_s11.json",
                           ACTIVE_RECEIPT_PATH, wave))
    for payload in payloads:
        moved.append(_move(f"checkpoints/{payload.name}", payload, wave))

    if not _active_is_clean():
        raise RuntimeError("archive left a colliding artifact in the active namespace")
    _assert_family_unchanged(family_before, _family_state())

    entry = {
        "wave": wave,
        "receipt_sha256": receipt_sha,
        "receipt_matches_recorded_wave1": receipt_matches_wave1,
        "payload_count": len(payloads),
        "payload_shas": [_sha256(Path(item["preserved_path"])) for item in moved
                         if item["original_relative_path"].startswith("checkpoints/")],
        "moved_files": moved,
    }

    if REGISTRY_PATH.is_file():
        registry = _json(REGISTRY_PATH)
        registry.setdefault("additional_waves", []).append(entry)
        registry["active_receipt_absent_after"] = True
        registry["active_checkpoints_absent_after"] = True
        registry["family_journal_unchanged_after_last_wave"] = True
        registry["family_journal_entries_after_last_wave"] = (
            _family_state()["family_journal_entries"])
    else:
        registry = {
            "kind": "round46-r2-v6-ep0-receipt-isolation/1",
            "status": "isolated",
            "reason": "prior ep0 recovery receipt conflict at the reused R45 promotion seam",
            "runner_source_sha256": _sha256(RUN_DIR / "r46_execute_r2.py"),
            "active_receipt_absent_after": True,
            "active_checkpoints_absent_after": True,
            "preserved_stop_evidence_never_current_success": True,
            "moved_files": moved,
            "additional_waves": [],
            "temporary_copies_used": len(moved),
            "family_before": family_before,
            "family_after": _family_state(),
            "family_journal_unchanged": True,
            "environment_constructor_calls": 0,
            "run_invocation_calls": 0,
            "completed_steps": 0,
            "ppo_updates": 0,
            "optimizer_step_calls": 0,
            "checkpoint_write_calls": 0,
        }
    registry["wave2_entry"] = entry if wave != "wave1" else registry.get("wave2_entry")
    registry["active_namespace_clean"] = True
    _write_json(REGISTRY_PATH, registry)
    return registry


def _assert_registry(registry: dict[str, Any]) -> None:
    if (registry.get("kind") != "round46-r2-v6-ep0-receipt-isolation/1"
            or registry.get("active_receipt_absent_after") is not True
            or registry.get("active_checkpoints_absent_after") is not True):
        raise RuntimeError("isolation registry identity is not the recorded /6 result")
    for item in registry.get("moved_files", []):
        preserved = Path(item["preserved_path"])
        if not preserved.is_file() or _sha256(preserved) != item["sha256_after"]:
            raise RuntimeError(f"preserved stop evidence changed: {preserved}")
    for wave in registry.get("additional_waves", []):
        for item in wave.get("moved_files", []):
            preserved = Path(item["preserved_path"])
            if not preserved.is_file() or _sha256(preserved) != item["sha256_after"]:
                raise RuntimeError(f"preserved stop evidence changed: {preserved}")


CALL_JOURNAL_NAMES = ("optimizer_step_calls.jsonl", "ppo_update_calls.jsonl")


def relocate_call_journals() -> dict[str, Any]:
    """Move the /6 run's training-call journals into the private namespace.

    The R42/R43 instrumentation binds `RUN_DIR/runtime/{optimizer_step,ppo_update}`
    at configure time. During the /6 run those bound to the R45 root while the
    batch records went to the private root, so the journal/batch closure check
    cannot close. Both journals carry only /6 counts; move them, byte-identical,
    so journals and batch records share one root again.
    """
    source_root = REPO / "runs" / "comm_rule_warmstart_cost_learning_20261006_r2" / "runtime"
    target_root = RUN_DIR / "v6" / "runtime"
    family_before = _family_state()
    moved: list[dict[str, Any]] = []
    for name in CALL_JOURNAL_NAMES:
        source = source_root / name
        if not source.is_file():
            continue
        destination = target_root / name
        if destination.exists():
            existing = destination.read_bytes()
            if existing != source.read_bytes():
                raise RuntimeError(f"/6 {name} already exists with different bytes")
            continue
        item = _move(f"v6/runtime/{name}", source, "call_journal")
        moved.append(item)
    if not moved:
        raise RuntimeError("no /6 call journal needed relocation")
    _assert_family_unchanged(family_before, _family_state())
    for item in moved:
        if not Path(item["preserved_path"]).is_file():
            raise RuntimeError("relocated call journal is missing")
    entry = {
        "reason": "journals bound to the R45 root while batch records are in the /6 root",
        "moved_files": moved,
        "journal_line_counts": {
            item["original_relative_path"].rsplit("/", 1)[-1]:
                sum(1 for line in Path(item["preserved_path"]).read_text(
                    encoding="utf-8").splitlines() if line.strip())
            for item in moved
        },
    }
    registry = _json(REGISTRY_PATH) if REGISTRY_PATH.is_file() else {}
    registry.setdefault("relocated_call_journals", []).append(entry)
    _write_json(REGISTRY_PATH, registry)
    return entry


def quarantine_stale_claim(logical_job: str) -> dict[str, Any]:
    """Register and relocate a claim written by the pre-fix adapter.

    The pre-fix /6 adapter overwrote the round-45 claim identity fields, so the
    R45 completed-claim readback correctly refuses it. Nothing was committed
    from it (no PPO update), so the affected logical job is re-run instead of
    hand-editing the record. The stale claim and its row are preserved with
    hashes so the aborted attempt stays auditable.
    """
    claim_dir = RUN_DIR / "v6" / "launch_claims" / logical_job
    if not claim_dir.is_dir():
        raise FileNotFoundError(claim_dir)
    claim_files = sorted(claim_dir.glob("attempt_*.json"))
    if len(claim_files) != 1:
        raise RuntimeError(f"expected exactly one stale claim in {claim_dir}")
    claim_path = claim_files[0]
    claim = _json(claim_path)
    if (claim.get("status") not in {"failed_technical", "failed"}
            or claim.get("round46_key") is None
            or claim.get("round45_artifact_receipt") is not None
            or claim.get("round44_eval_artifact_receipt") is not None):
        raise RuntimeError("named claim is not the pre-fix /6 record; refusing to quarantine")
    counts = {
        "normal": int(claim.get("status") == "complete"),
        "environment_constructor_calls": int(claim.get("environment_constructor_calls", 0)),
        "run_invocation_calls": int(claim.get("run_invocation_calls", 0)),
        "completed_steps": int(claim.get("completed_steps", 0)),
        "ppo_update_calls": int(claim.get("ppo_update_calls", 0)),
        "optimizer_step_calls": int(claim.get("optimizer_step_calls", 0)),
    }
    if counts["ppo_update_calls"] != 0 or counts["optimizer_step_calls"] != 0:
        raise RuntimeError("stale claim committed an update; do not quarantine, audit first")

    family_before = _family_state()
    ISOLATION_DIR.mkdir(parents=True, exist_ok=True)
    preserved = []
    for path in [claim_path, *[p for p in sorted(claim_dir.iterdir()) if p != claim_path]]:
        if not path.is_file():
            continue
        relative = f"v6/launch_claims/{logical_job}/{path.name}"
        preserved.append(_move(relative, path, "stale_claim"))
    claim_dir.rmdir()
    _assert_family_unchanged(family_before, _family_state())

    entry = {
        "logical_job": logical_job,
        "reason": "pre-fix /6 claim identity fields; R45 readback correctly refused it",
        "claim_round_written": 46,
        "aborted_attempt_counts": counts,
        "moved_files": preserved,
        "rerun_authorized_because_no_update_committed": True,
    }
    registry = _json(REGISTRY_PATH) if REGISTRY_PATH.is_file() else {}
    registry.setdefault("quarantined_stale_claims", []).append(entry)
    _write_json(REGISTRY_PATH, registry)
    return entry


def install_call_journals() -> dict[str, Any]:
    """Place the /6 run's call journals in the private namespace.

    R42 rehydrates its live call counters from `RUN_DIR/runtime/*.jsonl` at
    configure time and then requires them to equal the batch-record totals. The
    /6 journals must therefore live in the private root alongside the batch
    records. Copies are made from the preserved archive (byte-identical); the
    archived originals stay untouched.
    """
    target_root = RUN_DIR / "v6" / "runtime"
    target_root.mkdir(parents=True, exist_ok=True)
    installed: list[dict[str, Any]] = []
    for name in CALL_JOURNAL_NAMES:
        preserved = ISOLATION_DIR / "call_journal" / "v6" / "runtime" / name
        if not preserved.is_file():
            raise FileNotFoundError(f"preserved /6 call journal is missing: {preserved}")
        destination = target_root / name
        source_bytes = preserved.read_bytes()
        if destination.exists():
            if destination.read_bytes() != source_bytes:
                raise RuntimeError(f"/6 {name} exists with different bytes")
        else:
            destination.write_bytes(source_bytes)
        digest = _sha256(destination)
        if digest != _sha256(preserved):
            raise RuntimeError(f"/6 {name} copy is not byte-identical")
        installed.append({
            "path": str(destination.resolve()),
            "sha256": digest,
            "entries": sum(1 for line in destination.read_text(
                encoding="utf-8").splitlines() if line.strip()),
            "copied_from_preserved": str(preserved.resolve()),
        })
    entry = {
        "reason": "R42 rehydrates live call counters from RUN_DIR/runtime at configure time",
        "installed": installed,
    }
    registry = _json(REGISTRY_PATH) if REGISTRY_PATH.is_file() else {}
    registry.setdefault("installed_call_journals", []).append(entry)
    _write_json(REGISTRY_PATH, registry)
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true",
                       help="read-only check of the active namespace and family ledger")
    group.add_argument("--archive-active", metavar="WAVE",
                       help="register and relocate the untrained-ep0 artifacts now in the active root")
    group.add_argument("--quarantine-stale-claim", metavar="LOGICAL_JOB",
                       help="register and relocate a pre-fix claim so its logical job can be re-run")
    group.add_argument("--relocate-call-journals", action="store_true",
                       help="move the /6 training-call journals into the private namespace")
    group.add_argument("--install-call-journals", action="store_true",
                       help="place byte-identical /6 call journals under the private root")
    args = parser.parse_args(argv)
    if args.verify:
        result = verify()
    elif args.archive_active is not None:
        result = archive_active(args.archive_active)
    elif args.relocate_call_journals:
        result = relocate_call_journals()
    elif args.install_call_journals:
        result = install_call_journals()
    else:
        result = quarantine_stale_claim(args.quarantine_stale_claim)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
