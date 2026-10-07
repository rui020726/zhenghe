"""ROUND46 /6 ep0 recovery-receipt scope isolation (zero-environment).

The frozen /5 run stopped before its first atomic claim because the reused R45
promotion seam (`r45_r2_execute_r3.py:_promote_ep0_to_rolling`) found a prior
ep0 recovery receipt at its output path and rejected the different checkpoint
identity it was given:

    RuntimeError: seed 11: prior ep0 recovery receipt conflicts

The same stopped attempt also left two untrained checkpoint payloads in the
active checkpoint path, which the frozen runner refuses to resume
(`r46_execute_r2.py:_prior_r2_preclaim_attempt_identity`, line 366).

The R45 seam resolves BOTH its recovery receipt (`module.RUN_DIR/runtime`) and
its rolling checkpoint (`common.CHECKPOINTS`) from one output root, and
`r42_run.ledger_snapshot` resolves the batch records and the checkpoint journal
from that same root. A *separate* receipt namespace is therefore not reachable
without relocating the whole checkpoint root, which would fork the single
family journal that the task forbids resetting. The minimal correct fix is a
registered, recoverable isolation of exactly the three colliding artifacts out
of the active path, leaving the frozen runner and every /5 receipt byte
identical.

After isolation the active receipt namespace contains no prior receipt, so the
seam writes a fresh one; the family journal keeps its 19 entries and its
existing SHA256; the preserved /5 stop evidence moves to a registered location
with before/after hashes and is never consulted as current-execution evidence.

This module only reads and moves the three explicitly named files. It never
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
ACTIVE_EP0_PATH = RUN_DIR / "checkpoints" / "v2_train_s11_ep0.pt"
ACTIVE_ROLLING_PATH = RUN_DIR / "checkpoints" / "v2_train_s11.pt"
FAMILY_LEDGER_PATH = RUN_DIR / "runtime" / "ROUND46_EXECUTION_LEDGER.json"
FAMILY_JOURNAL_PATH = RUN_DIR / "runtime" / "checkpoint_writes.jsonl"

ISOLATION_DIR = RUN_DIR / "attempts" / "runner_v5_ep0_receipt_conflict"
REGISTRY_PATH = RUN_DIR / "ROUND46_R2_V6_EP0_RECEIPT_ISOLATION.json"

# Recorded /5 stop identities. Isolation refuses to act if the live bytes differ.
EXPECTED_RECEIPT_SHA256 = (
    "1a259177a313312ff9686a4831937a304205808656e00e2231577e991b81c300"
)
EXPECTED_RECEIPT_RECORDED_SOURCE_SHA256 = (
    "330a80eb646bf422fbe31024951cc88c15740cc97579c903b0686a78107f97ed"
)
EXPECTED_PAYLOAD_SHA256 = (
    "2ca8f764315795f5aadc89117901f23442c43c15e4d4802f5ae268765467950a"
)
EXPECTED_PAYLOAD_BYTES = 57605
EXPECTED_FAMILY_LEDGER_SHA256 = (
    "b3b76ae5ce524348152f41dbdd49800b1715436bdf7045ba74a53b83d1a9a2a3"
)
EXPECTED_FAMILY_JOURNAL_SHA256 = (
    "0c612fda36556d589704b1df20ed7a5c06eae811ebeed2206f046d4df978f6d1"
)
EXPECTED_FAMILY_JOURNAL_ENTRIES = 19

MOVED_FILES = (
    ("runtime/ROUND45_R2_EP0_RECOVERY_s11.json", ACTIVE_RECEIPT_PATH),
    ("checkpoints/v2_train_s11_ep0.pt", ACTIVE_EP0_PATH),
    ("checkpoints/v2_train_s11.pt", ACTIVE_ROLLING_PATH),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _journal_entry_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _preserved_path(relative: str) -> Path:
    return ISOLATION_DIR / relative.replace("/", os.sep)


def _family_state() -> dict[str, Any]:
    ledger_sha = _sha256(FAMILY_LEDGER_PATH) if FAMILY_LEDGER_PATH.is_file() else None
    journal_sha = _sha256(FAMILY_JOURNAL_PATH) if FAMILY_JOURNAL_PATH.is_file() else None
    entries = _journal_entry_count(FAMILY_JOURNAL_PATH)
    return {
        "family_ledger_path": str(FAMILY_LEDGER_PATH.resolve()),
        "family_ledger_sha256": ledger_sha,
        "family_journal_path": str(FAMILY_JOURNAL_PATH.resolve()),
        "family_journal_sha256": journal_sha,
        "family_journal_entries": entries,
    }


def _assert_family_unchanged(state: dict[str, Any]) -> None:
    """The single family ledger and journal must keep their /5 bytes and count."""
    if state["family_ledger_sha256"] != EXPECTED_FAMILY_LEDGER_SHA256:
        raise RuntimeError("family execution ledger bytes changed; refusing to isolate")
    if state["family_journal_sha256"] != EXPECTED_FAMILY_JOURNAL_SHA256:
        raise RuntimeError("family checkpoint journal bytes changed; refusing to isolate")
    if state["family_journal_entries"] != EXPECTED_FAMILY_JOURNAL_ENTRIES:
        raise RuntimeError("family checkpoint journal entry count changed; refusing to isolate")


def _active_state() -> dict[str, Any]:
    receipt = {
        "path": str(ACTIVE_RECEIPT_PATH.resolve()),
        "present": ACTIVE_RECEIPT_PATH.is_file(),
        "sha256": _sha256(ACTIVE_RECEIPT_PATH) if ACTIVE_RECEIPT_PATH.is_file() else None,
    }
    payloads = []
    for label, path in (("ep0", ACTIVE_EP0_PATH), ("rolling", ACTIVE_ROLLING_PATH)):
        payloads.append({
            "role": label,
            "path": str(path.resolve()),
            "present": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": _sha256(path) if path.is_file() else None,
        })
    return {"receipt": receipt, "payloads": payloads}


def _assert_stop_identity(state: dict[str, Any]) -> None:
    """Every artifact we intend to move must be the recorded /5 stop evidence."""
    if state["receipt"]["present"] and state["receipt"]["sha256"] != EXPECTED_RECEIPT_SHA256:
        raise RuntimeError(
            "active ep0 recovery receipt is not the recorded /5 stop receipt; "
            "inspect before isolating"
        )
    for payload in state["payloads"]:
        if not payload["present"]:
            continue
        if payload["sha256"] != EXPECTED_PAYLOAD_SHA256 or payload["bytes"] != EXPECTED_PAYLOAD_BYTES:
            raise RuntimeError(
                f"active {payload['role']} payload is not the recorded untrained /5 ep0 file; "
                "inspect before isolating"
            )
    if state["receipt"]["present"]:
        receipt = _json(ACTIVE_RECEIPT_PATH)
        if (receipt.get("kind") != "round45-r2-ep0-rolling-copy-receipt/1"
                or receipt.get("status") != "passed"
                or receipt.get("seed") != 11
                or receipt.get("episodes_done") != 0
                or receipt.get("updates") != 0
                or receipt.get("optimizer_steps") != 0
                or receipt.get("source_sha256") != EXPECTED_RECEIPT_RECORDED_SOURCE_SHA256):
            raise RuntimeError("active ep0 recovery receipt is not the preserved /5 identity")


def verify() -> dict[str, Any]:
    state = _active_state()
    family = _family_state()
    _assert_family_unchanged(family)
    _assert_stop_identity(state)
    clean = (not state["receipt"]["present"]
             and not any(item["present"] for item in state["payloads"]))
    result = {
        "kind": "round46-r2-v6-ep0-receipt-isolation-verify/1",
        "status": "active_namespace_clean" if clean else "active_namespace_collides",
        "registry_path": str(REGISTRY_PATH.resolve()),
        "registry_present": REGISTRY_PATH.is_file(),
        "active": state,
        "family": family,
        "environment_constructor_calls": 0,
        "run_invocation_calls": 0,
        "completed_steps": 0,
        "ppo_updates": 0,
        "optimizer_step_calls": 0,
        "checkpoint_write_calls": 0,
    }
    return result


def isolate() -> dict[str, Any]:
    """Perform the registered recoverable relocation (idempotent)."""
    before = _active_state()
    family_before = _family_state()
    _assert_family_unchanged(family_before)
    _assert_stop_identity(before)

    already_clean = (not before["receipt"]["present"]
                     and not any(item["present"] for item in before["payloads"]))
    if already_clean:
        if not REGISTRY_PATH.is_file():
            raise RuntimeError(
                "active namespace is clean but no isolation registry exists; "
                "refusing to fabricate one"
            )
        registry = _json(REGISTRY_PATH)
        _assert_registry(registry)
        return {
            "kind": "round46-r2-v6-ep0-receipt-isolation/1",
            "status": "already_isolated",
            "registry": registry,
        }

    ISOLATION_DIR.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, Any]] = []
    temporary_copies = 0
    for relative, source in MOVED_FILES:
        if not source.is_file():
            continue
        before_sha = _sha256(source)
        before_bytes = source.stat().st_size
        destination = _preserved_path(relative)
        if destination.exists():
            raise RuntimeError(f"preserved isolation target already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Copy then verify then remove the active original: byte-identical move
        # with no window in which the evidence exists only in one place.
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
            dst.flush()
            os.fsync(dst.fileno())
        temporary_copies += 1
        after_sha = _sha256(destination)
        if after_sha != before_sha or destination.stat().st_size != before_bytes:
            destination.unlink()
            raise RuntimeError(f"preserved copy is not byte-identical: {relative}")
        os.replace(destination, destination)
        source.unlink()
        moved.append({
            "original_relative_path": relative,
            "preserved_relative_path": destination.relative_to(RUN_DIR).as_posix(),
            "original_path": str(source.resolve()),
            "preserved_path": str(destination.resolve()),
            "bytes": before_bytes,
            "sha256_before": before_sha,
            "sha256_after": after_sha,
            "byte_identical": after_sha == before_sha,
        })

    if not moved:
        raise RuntimeError("isolation found nothing to move but the namespace was not clean")

    after = _active_state()
    family_after = _family_state()
    _assert_family_unchanged(family_after)
    if after["receipt"]["present"] or any(item["present"] for item in after["payloads"]):
        raise RuntimeError("isolation left a colliding artifact in the active namespace")

    registry = {
        "kind": "round46-r2-v6-ep0-receipt-isolation/1",
        "status": "isolated",
        "reason": "prior ep0 recovery receipt conflict at the reused R45 promotion seam",
        "runner_source_sha256": _sha256(RUN_DIR / "r46_execute_r2.py"),
        "runner_source_sha256_expected":
            "ea5597d0ccbe0562d0fc4b87a81708ea0f525747d31900ef9bf46b4a7d228b86",
        "active_receipt_absent_after": True,
        "active_checkpoints_absent_after": True,
        "preserved_stop_evidence_never_current_success": True,
        "moved_files": moved,
        "temporary_copies_used": temporary_copies,
        "family_before": family_before,
        "family_after": family_after,
        "family_journal_unchanged": family_before == family_after,
        "environment_constructor_calls": 0,
        "run_invocation_calls": 0,
        "completed_steps": 0,
        "ppo_updates": 0,
        "optimizer_step_calls": 0,
        "checkpoint_write_calls": 0,
    }
    if registry["runner_source_sha256"] != registry["runner_source_sha256_expected"]:
        raise RuntimeError("frozen runner source changed; isolation is no longer minimal")
    if not registry["family_journal_unchanged"]:
        raise RuntimeError("isolation changed the family ledger or journal")
    _write_json(REGISTRY_PATH, registry)
    return registry


def _assert_registry(registry: dict[str, Any]) -> None:
    if (registry.get("kind") != "round46-r2-v6-ep0-receipt-isolation/1"
            or registry.get("status") not in {"isolated", "already_isolated"}
            or registry.get("active_receipt_absent_after") is not True
            or registry.get("active_checkpoints_absent_after") is not True
            or registry.get("family_journal_unchanged") is not True):
        raise RuntimeError("isolation registry identity is not the recorded /6 result")
    for item in registry.get("moved_files", []):
        preserved = Path(item["preserved_path"])
        if not preserved.is_file():
            raise RuntimeError(f"preserved stop evidence is missing: {preserved}")
        if _sha256(preserved) != item["sha256_after"]:
            raise RuntimeError(f"preserved stop evidence changed: {preserved}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true",
                       help="read-only check of the active namespace and family ledger")
    group.add_argument("--isolate", action="store_true",
                       help="register and relocate the three colliding /5 artifacts")
    args = parser.parse_args(argv)
    result = verify() if args.verify else isolate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
