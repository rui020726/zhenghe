"""Prove this round changed nothing it promised not to change.

Run:  python topology_verify.py      # writes COMM_RECIPIENT_TOPOLOGY_VERIFICATION.json

Three claims have to be checkable rather than asserted: no episode ran, the audited round's files
are untouched, and production code is byte-identical to the commit this branch started from. The
last one matters most here -- a topology audit is exactly the round where someone might be tempted
to make `subscriber_count` vary in order to pass a gate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import topology_common as T

CALLS = T.install_env_tripwire()
START_SHA = "efa942857f5bce1af01b84c1739c8fc9583dcca7"
PRODUCTION_PATHS = (
    "integrated_sar/comm.py",
    "integrated_sar/comm_policy.py",
    "integrated_sar/comm_budget.py",
    "integrated_sar/joint_driver.py",
    "integrated_sar/adapters/task03.py",
    "integrated_sar/coordinator.py",
    "integrated_sar/assembly.py",
    "integrated_sar/task_registry.py",
    "integrated_sar/contracts.py",
    "task_allocation/old_task_env.py",
    "task_allocation/old_driver.py",
    "task_allocation/old_network.py",
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=T.ROOT, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def production_drift() -> dict[str, Any]:
    """Has any production file changed since the start commit, or since HEAD?

    Uses `git diff` rather than comparing blob hashes of worktree bytes: the repository is checked
    out with line-ending conversion, so a raw `hash-object` would report a difference that is only
    a checkout artefact. The earlier round recorded exactly this trap.
    """
    rows: dict[str, Any] = {}
    against_start = git(
        "diff", "--name-only", START_SHA, "--", *PRODUCTION_PATHS
    ).splitlines()
    against_head = git("status", "--porcelain", "--", *PRODUCTION_PATHS).splitlines()
    dirty = {line.split()[-1].replace("\\", "/") for line in against_head}
    for relative in PRODUCTION_PATHS:
        normalised = relative.replace("\\", "/")
        rows[relative] = {
            "exists": (T.ROOT / relative).exists(),
            "differs_from_start_commit": any(
                normalised in line.replace("\\", "/") for line in against_start
            ),
            "differs_from_worktree_vs_head": any(
                normalised in line.replace("\\", "/") for line in dirty
            ),
        }
        rows[relative]["unchanged"] = (
            rows[relative]["exists"]
            and not rows[relative]["differs_from_start_commit"]
            and not rows[relative]["differs_from_worktree_vs_head"]
        )
    return rows


def resolve(key: str) -> Any:
    """Manifest keys are recorded relative to three different roots; route each one correctly."""
    if key.startswith("previous_round/"):
        return T.PREVIOUS_ROUND / key[len("previous_round/"):]
    if key.startswith(("integrated_sar/", "task_allocation/")):
        return T.ROOT / key
    return T.SOURCE_ROUND / key


def main(argv: list[str]) -> int:
    import pressure_common as common

    manifest = T.read_json(T.HERE / "COMM_RECIPIENT_TOPOLOGY_INPUT_MANIFEST.json")
    frozen: dict[str, str] = manifest["input_sha256"]
    changed: list[str] = []
    missing: list[str] = []
    for key, digest in sorted(frozen.items()):
        path = resolve(key)
        if not path.exists():
            missing.append(key)
        elif T.sha256_file(path) != digest:
            changed.append(key)
    ledger = common.ledger_totals()
    before = manifest["launch_ledger_at_freeze"]
    drift = production_drift()
    result = {
        "files_frozen": len(frozen),
        "files_changed": changed,
        "files_missing_now": missing,
        "launches_at_freeze": before,
        "launches_now": {key: ledger.get(key) for key in before},
        "launches_added_by_this_round": int(ledger.get("starts", 0))
        - int(before.get("starts", 0)),
        "episode_entry_point_calls": list(CALLS),
        "production_files_unchanged": {
            key: value["unchanged"] for key, value in drift.items()
        },
        "production_files_changed": sorted(
            key for key, value in drift.items() if not value["unchanged"]
        ),
        "sealed_block_800000_800399_artefacts_on_disk": len(
            list(T.ROOT.glob("runs/**/s8000*"))
        ),
        "gate_files_byte_identical_to_their_recorded_sha": not changed and not missing,
    }
    result["passed"] = (
        not changed
        and not missing
        and result["launches_added_by_this_round"] == 0
        and not CALLS
        and not result["production_files_changed"]
        and result["sealed_block_800000_800399_artefacts_on_disk"] == 0
    )
    T.write_json(T.HERE / "COMM_RECIPIENT_TOPOLOGY_VERIFICATION.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
