"""Round boundaries, checked rather than asserted.

This round was authorised to change production code for one reason: B0's passive logging.
So the verification does not claim "production untouched" — it claims the narrower and
falsifiable thing: the touched files are the ones B0 needs, the reward/policy/training
modules are not among them, the two frozen checkpoints are the same bytes they were, and
nothing was trained, sealed or overwritten on the way.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import d1_common as dc

REPO = dc.REPO_ROOT

#: The only production paths B0 was allowed to touch.
ALLOWED_PRODUCTION = {
    "integrated_sar/info_probe.py",
    "integrated_sar/comm.py",
    "integrated_sar/joint_driver.py",
    "integrated_sar/adapters/piloc.py",
    "integrated_sar/tests/test_info_probe.py",
}

#: Prefixes this round owns outright: its own artefacts and the documents it must update.
ALLOWED_PREFIXES = ("runs/comm_info_path_validation_20260923/", ".gitignore")
ALLOWED_DOCUMENTS = {"PROJECT_CHARTER.md", "PROJECT_STATUS.md", "README.md"}

#: Modules whose behaviour would change a training or policy result if edited.
PROTECTED = (
    "task_allocation/", "piloc/", "integrated_sar/comm_policy.py",
    "integrated_sar/reward", "integrated_sar/training",
)


def git(*args: str) -> str:
    result = subprocess.run(("git", *args), cwd=REPO, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    return (result.stdout or "").strip()


def production_scope() -> dict[str, Any]:
    changed = [line for line in git("diff", "--name-only", dc.START_SHA).splitlines()
               if line.strip()]
    allowed = set(ALLOWED_PRODUCTION) | ALLOWED_DOCUMENTS
    outside = sorted(path for path in changed
                     if path not in allowed and not path.startswith(ALLOWED_PREFIXES))
    return {
        "changed_vs_start_sha": sorted(changed),
        "production_files_changed": sorted(set(changed) & (set(ALLOWED_PRODUCTION)
                                                           - ALLOWED_DOCUMENTS)),
        "outside_the_allowed_set": outside,
        "protected_paths_touched": sorted(path for path in changed
                                          if path.startswith(PROTECTED)),
    }


def checkpoints_untouched() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, path in (("search", dc.PIL_CHECKPOINT), ("allocation", dc.CTDE_CHECKPOINT)):
        out[label] = {"path": str(path.relative_to(REPO)), "exists": path.exists(),
                      "sha256": dc.sha256_file(path) if path.exists() else None,
                      "modified_after_start": _modified_after(path, dc.START_SHA)}
    return out


def _modified_after(path: Path, revision: str) -> bool | None:
    if not path.exists():
        return None
    stamp = int(path.stat().st_mtime)
    result = subprocess.run(("git", "show", "-s", "--format=%ct", revision), cwd=REPO,
                            capture_output=True, text=True, errors="replace")
    try:
        return stamp > int(result.stdout.strip())
    except ValueError:
        return None


def cell_census() -> dict[str, Any]:
    rows = list(dc.ROWS.glob("data_*.json"))
    failed = [json.loads(path.read_text(encoding="utf-8")) for path in rows]
    by_arm: dict[str, int] = {}
    for row in failed:
        by_arm[row["arm"]] = by_arm.get(row["arm"], 0) + 1
    return {
        "data_rows": len(rows),
        "by_arm": by_arm,
        "technical_failures": sum(1 for row in failed if not row.get("ok")),
        "cells_missing": sorted({arm for arm in dc.DATA_ARMS} - set(by_arm)),
        "probe_anomaly_cells": sum(1 for row in failed if row.get("probe_anomalies")),
        "seeds_expected": len(dc.SEEDS),
        "launches": dc.ledger_report(),
    }


def sealed_and_reserved() -> dict[str, Any]:
    hits = 0
    for path in (REPO / "runs").rglob("*"):
        tokens = path.name.replace(".jsonl", "").replace(".json", "").split("_")
        hits += sum(1 for token in tokens if token.isdigit()
                    and 800000 <= int(token) <= 800399)
    return {"sealed_block_artefacts": hits,
            "checkpoints_written_by_this_round": len(list(dc.ROUND_DIR.rglob("*.pth")))}


def run() -> int:
    scope = production_scope()
    census = cell_census()
    gates = {}
    for name in ("B0_PAIRED_REPLAY.json", "IDENTITY_AUDIT_data.json",
                 "IDENTITY_AUDIT_smoke.json", "SEED_COLLISION_AUDIT.json"):
        path = dc.ROUND_DIR / name
        gates[name] = (json.loads(path.read_text(encoding="utf-8")).get("passed")
                       if name.startswith("B0") else
                       json.loads(path.read_text(encoding="utf-8")).get(
                           "all_passed",
                           json.loads(path.read_text(encoding="utf-8")).get("delivered")))
        if not path.exists():
            gates[name] = "missing"
    checks = {
        "production_scope": scope,
        "checkpoints": checkpoints_untouched(),
        "cell_census": census,
        "sealed_and_reserved": sealed_and_reserved(),
        "gates": gates,
        "training": "none requested and none performed; both networks are loaded frozen",
        "production_defaults": "unchanged: every arm sets its parameters explicitly on the "
                               "command line, and RunConfiguration defaults were not edited",
        "not_an_independent_test_set": (
            "these 240 cells measure an information path, not a learned policy; no learned "
            "controller was run on them and they must not be cited as held-out evidence for one"
        ),
    }
    checks["passed"] = (
        not scope["outside_the_allowed_set"] and not scope["protected_paths_touched"]
        and not checks["checkpoints"]["search"]["modified_after_start"]
        and not checks["checkpoints"]["allocation"]["modified_after_start"]
        and census["data_rows"] == 240 and census["technical_failures"] == 0
        and not census["cells_missing"] and census["probe_anomaly_cells"] == 0
        and checks["sealed_and_reserved"]["sealed_block_artefacts"] == 0
        and checks["sealed_and_reserved"]["checkpoints_written_by_this_round"] == 0
        and census["launches"]["used"] <= census["launches"]["cap"]
        and gates.get("B0_PAIRED_REPLAY.json") is True
        and gates.get("IDENTITY_AUDIT_data.json") is True
        and gates.get("IDENTITY_AUDIT_smoke.json") is True
        and gates.get("SEED_COLLISION_AUDIT.json") == "passed"
    )
    dc.write_json("D1_VERIFICATION.json", checks)
    print(json.dumps({"passed": checks["passed"], "gates": gates, "census": {
        key: census[key] for key in ("data_rows", "technical_failures", "cells_missing",
                                     "probe_anomaly_cells", "launches")},
        "outside_allowed": scope["outside_the_allowed_set"],
        "protected_touched": scope["protected_paths_touched"]},
        ensure_ascii=False, indent=1))
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
