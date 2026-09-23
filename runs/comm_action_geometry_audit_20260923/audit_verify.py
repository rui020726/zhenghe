"""Prove the audit left the delivered round untouched, and re-check the no-episode claim.

Run:  python audit_verify.py         # writes AUDIT_VERIFICATION.json

The freeze manifest recorded a SHA256 for every input file before any analysis ran. This re-reads
them and reports any difference, so "we only analysed stored data" is a checked statement. Two
things have to hold: every hashed file is unchanged, and no launch was added to any ledger.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import audit_common as A

CALLS = A.install_env_tripwire()
A.enter_source_round()

import pressure_common as common  # noqa: E402


def main(argv: list[str]) -> int:
    manifest = A.read_json(A.HERE / "COMM_ACTION_GEOMETRY_INPUT_MANIFEST.json")
    frozen: dict[str, str] = manifest["input_sha256"]
    changed: list[str] = []
    disappeared: list[str] = []
    appeared: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for relative, digest in sorted(frozen.items()):
        path = A.SOURCE_ROUND / relative
        if not path.exists():
            disappeared.append(relative)
            continue
        current = A.sha256_file(path)
        if current != digest:
            changed.append(relative)
            mismatches.append({"path": relative, "frozen": digest, "now": current})
    frozen_set = set(frozen)
    for path in sorted((A.SOURCE_ROUND / "rows").rglob("*")):
        if path.is_file():
            key = str(path.relative_to(A.SOURCE_ROUND)).replace("\\", "/")
            if key not in frozen_set:
                appeared.append(key)
    for path in sorted((A.SOURCE_ROUND / "runs").rglob("*")):
        if path.is_file():
            key = str(path.relative_to(A.SOURCE_ROUND)).replace("\\", "/")
            if key not in frozen_set:
                appeared.append(key)
    for path in sorted(A.SOURCE_ROUND.glob("*.json")):
        if path.name not in frozen_set:
            appeared.append(path.name)

    ledger = common.ledger_totals()
    before = manifest["launch_ledger_at_freeze"]
    result = {
        "files_frozen": len(frozen),
        "files_changed": changed,
        "files_missing_now": disappeared,
        "files_added_since_freeze": appeared,
        "ledger_at_freeze": {
            key: before.get(key) for key in ("starts", "matrix_starts", "gate_starts", "failures")
        },
        "ledger_now": {
            key: ledger.get(key) for key in ("starts", "matrix_starts", "gate_starts", "failures")
        },
        "launches_added_by_this_round": int(ledger.get("starts", 0))
        - int(before.get("starts", 0)),
        "episode_entry_point_calls": list(CALLS),
        "sealed_block_artefacts_on_disk": len(
            list(A.ROOT.glob("runs/**/s8000*"))
            + list(A.ROOT.glob("runs/**/*8000[0-3][0-9][0-9].json"))
        ),
    }
    result["passed"] = (
        not changed
        and not disappeared
        and not appeared
        and result["launches_added_by_this_round"] == 0
        and not CALLS
    )
    A.write_json(A.HERE / "AUDIT_VERIFICATION.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
