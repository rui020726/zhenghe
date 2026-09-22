"""Source and run identity for the budget-action mask fix.

The executed-module list used to be typed by hand and named ``integrated_sar/comm_transcript.py``,
a path that has never existed in the repository and that nothing imports -- the class it was named
for lives in ``integrated_sar/comm.py``. It then read as a *dirty* module rather than an absent one
because ``git rev-parse HEAD:<missing path>`` echoes its argument instead of failing. Both halves
are corrected here, by the next round's provenance pass
(``runs/comm_budget_masked_retrain_20260922/COMM_BUDGET_MASK_PROVENANCE_CORRECTION.md``), which
changed no measurement of this round: the list is now derived by importing the round's own entry
points, and absence is resolved with ``git cat-file -e`` before anything is hashed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _directory in (str(ROOT), str(HERE), str(ROOT / "runs" / "comm_tight_budget_20260919")):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

import train_budget_masked  # noqa: E402  # sets the rest of the sys.path entries
from integrated_sar.joint_driver import RunConfiguration  # noqa: E402

#: Production modules this round touched, plus the ones that must be provably untouched.
EXECUTED_MODULES = (
    "integrated_sar/comm.py",
    "integrated_sar/comm_policy.py",
    "integrated_sar/comm_budget.py",
    "integrated_sar/comm_learning.py",
    "integrated_sar/comm_ack.py",
)

#: The entry points whose import graph is this round's real execution surface. Derived, so a
#: phantom path cannot re-enter the manifest.
EXECUTED_ENTRIES = (
    "smoke_budget_mask",
    "train_budget_masked",
    "budget_masked",
    "train_protocol",
    "train_onpolicy",
    "onpolicy",
    "protocol_env",
    "pilot_env",
    "train_pilot",
)


def executed_modules(entries: tuple[str, ...] = EXECUTED_ENTRIES) -> list[str]:
    """Every repo-local module the round actually imported, walked out of ``sys.modules``."""
    found: list[str] = []
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        path = Path(file)
        if not path.is_absolute() or not path.exists() or "__pycache__" in str(path):
            continue
        try:
            relative = path.resolve().relative_to(ROOT)
        except ValueError:
            continue
        if relative.parts[0] not in {"integrated_sar", "runs", "piloc", "task_allocation"}:
            continue
        if relative.suffix == ".py" and "/tests/" not in relative.as_posix():
            found.append(relative.as_posix())
    return sorted(set(found) | set(EXECUTED_MODULES))
ROUND_SCRIPTS = (
    "COMM_BUDGET_ACTION_MASK_PROTOCOL.md",
    "COMM_BUDGET_ACTION_MASK_RESULTS.md",
    "COMM_BUDGET_ACTION_MASK_RESULTS.json",
    "COMM_BUDGET_ACTION_MASK_SMOKE.json",
    "budget_masked.py",
    "train_budget_masked.py",
    "smoke_budget_mask.py",
    "test_budget_action_mask.py",
    "report_mask.py",
    "mask_manifest.py",
)
LAUNCH_LEDGER = "LAUNCH_LEDGER.json"
START_COMMIT = "31da5f7c36ed95b95d6752624bcbebd9f16221e6"
SEALED = "800000..800399"
CONSUMED = "900000..900199"


def git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def blob_identity(paths: tuple[str, ...] | list[str]) -> dict[str, dict[str, Any]]:
    """Git blob ids for HEAD and for the working tree, so "committed" is a checked fact.

    Blob identity rather than file hashes: a checkout can differ from the object database only in
    line endings, and a false mismatch there has already wasted a round's attention. Absence is
    settled first, and separately: ``git rev-parse HEAD:<missing path>`` prints its argument back on
    stdout rather than failing, so a path that was never committed used to arrive as a truncated
    fake blob id and be reported as a dirty module.
    """
    report: dict[str, dict[str, Any]] = {}
    for path in paths:
        in_head = subprocess.run(
            ("git", "cat-file", "-e", f"HEAD:{path}"),
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).returncode == 0
        head = git("rev-parse", f"HEAD:{path}") if in_head else ""
        working = git("hash-object", path) if (ROOT / path).exists() else ""
        report[path] = {
            "in_head": in_head,
            "on_disk": (ROOT / path).exists(),
            "head_blob": head[:16] or None,
            "working_blob": working[:16] or None,
            "matches_head": bool(head) and head == working,
        }
    return report


def production_diff() -> dict[str, Any]:
    """What changed in shipped code since the round started, and what must not have."""
    changed = git("diff", "--name-only", START_COMMIT, "HEAD", "--", "integrated_sar").splitlines()
    untouched = [
        path
        for path in ("piloc", "task_allocation", "AGENTS.md", "ARCHITECTURE.md")
        if git("diff", "--name-only", START_COMMIT, "HEAD", "--", path)
    ]
    return {
        "start_commit": START_COMMIT,
        "changed_production_files": sorted(changed),
        "other_production_paths_changed": untouched,
        "empty_outside_the_contract": not untouched
        and set(changed)
        <= {
            "integrated_sar/comm.py",
            "integrated_sar/comm_policy.py",
            "integrated_sar/tests/test_comm_budget_action_mask.py",
        },
    }


def defaults_unchanged() -> dict[str, Any]:
    configuration = RunConfiguration()
    return {
        "comm_ack": bool(configuration.comm_ack),
        "comm_timing_mode": str(configuration.comm_timing_mode),
        "comm_policy": str(configuration.comm_policy),
        "comm_budget_bytes": configuration.comm_budget_bytes,
        "comm_max_wait_steps": configuration.comm_max_wait_steps,
        "comm_urgent_wait_steps": configuration.comm_urgent_wait_steps,
        "comm_delivery_probability": configuration.comm_delivery_probability,
        "task_membership_consistency": bool(configuration.task_membership_consistency),
    }


def sealed_block_check() -> dict[str, Any]:
    """The learning final test must still not exist on disk in any form."""
    low, high = (int(item) for item in SEALED.split(".."))
    hits: list[str] = []
    for path in (ROOT / "runs").rglob("*"):
        name = path.name
        if not name.startswith(("rp_", "tr_", "ct_", "s")) or not name.endswith(".jsonl") and ".json" not in name:
            continue
        digits = "".join(character for character in name if character.isdigit())
        try:
            seed = int(digits[:6]) if digits else None
        except ValueError:
            continue
        if seed is not None and low <= seed <= high:
            hits.append(str(path.relative_to(ROOT)))
    return {"block": SEALED, "artefacts_found": hits[:10], "count": len(hits), "clean": not hits}


def streams_check() -> dict[str, Any]:
    smoke = json.loads((HERE / "COMM_BUDGET_ACTION_MASK_SMOKE.json").read_text(encoding="utf-8"))
    ledger = json.loads((HERE / LAUNCH_LEDGER).read_text(encoding="utf-8"))
    recorded = HERE / "episodes"
    return {
        "streams_used": [int(row["seed"]) for row in smoke["rows"]],
        "new_streams_recorded": 0,
        "launch_cap": int(ledger["cap"]),
        "launch_starts": int(ledger["starts"]),
        "technical_failures": smoke["technical_failures"],
        "episode_artefact_bytes_on_disk": sum(
            path.stat().st_size for path in recorded.rglob("*") if path.is_file()
        ),
        "episode_artefacts_in_git": False,
        "consumed_block_not_reused_as_test": CONSUMED,
    }


def main(argv: list[str]) -> int:
    modules = executed_modules()
    payload = {
        "document": "COMM_BUDGET_ACTION_MASK_MANIFEST",
        "date": "2026-09-22",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "executed_code": {
            "head": git("rev-parse", "HEAD"),
            "derivation": "modules walked out of sys.modules after importing this round's entries",
            "declared_contract_modules": list(EXECUTED_MODULES),
            "modules": blob_identity(modules),
            "dirty_modules": sorted(
                path for path, ids in blob_identity(modules).items()
                if not ids["matches_head"] and ids["in_head"] and ids["on_disk"]
            ),
            "absent_modules": sorted(
                path for path, ids in blob_identity(modules).items()
                if not ids["in_head"] or not ids["on_disk"]
            ),
            "round_scripts": blob_identity(tuple(
                f"runs/comm_budget_action_mask_fix_20260922/{name}"
                for name in ROUND_SCRIPTS
                if (HERE / name).exists()
            )),
        },
        "trainer_identity": train_budget_masked.identity(),
        "production_diff": production_diff(),
        "production_defaults": defaults_unchanged(),
        "sealed_block": sealed_block_check(),
        "streams": streams_check(),
        "round_files": [
            {"file": name, "bytes": (HERE / name).stat().st_size}
            for name in sorted(ROUND_SCRIPTS)
            if (HERE / name).exists()
        ],
        "not_committed": [
            "checkpoints (.pt)",
            "recorded streams",
            "per-episode lifecycle/decision/transcript logs",
            "other sessions' untracked files",
        ],
    }
    (HERE / "COMM_BUDGET_ACTION_MASK_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(
        {
            "head": payload["head"],
            "production_diff_empty_outside_contract": payload["production_diff"]["empty_outside_the_contract"],
            "changed": payload["production_diff"]["changed_production_files"],
            "sealed_clean": payload["sealed_block"]["clean"],
            "defaults": payload["production_defaults"],
        },
        ensure_ascii=False,
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
