"""ROUND28 原始产物 hash 清单 + 启动账本汇总：不入 Git 的产物逐件登记 (path,size,sha256)。"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import r28_common as common


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scan(base: Path) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    if not base.exists():
        return manifest
    for path in sorted(base.rglob("*")):
        if path.is_file():
            manifest[path.relative_to(common.ROOT).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": _sha256(path)}
    return manifest


def build_manifest() -> dict[str, Any]:
    sections = {name: _scan(getattr(common, name))
                for name in ("ROWS", "LOGS", "CLAIMS", "REJECTED")}
    totals = {name: {"files": len(entries),
                     "bytes": sum(int(entry["bytes"]) for entry in entries.values())}
              for name, entries in sections.items()}
    report = common.ledger_report()
    return {
        "round": "28",
        "run_id": common.OUTPUT_DIR.name,
        "sections": sections,
        "totals": totals,
        "launch_ledger": {key: value for key, value in report.items()
                          if key not in ("entries", "results")},
        "ledger_entries": report["entries"],
        "ledger_results": report["results"],
        "git_policy": "rows, logs, tapes and claims stay out of version control; this manifest is "
                      "the external hash list that lets a reviewer find and verify the originals",
        "read_only_sources": [
            "runs/comm_cost_firstreport_intervention_20260929 (ROUND27 artefacts, never written)",
            "runs/comm_cost_v2_rerun_20260928 (ROUND24 execution artefacts, never written)",
            "runs/comm_cost_greedy_replication_20260928 (ROUND25 execution artefacts, never "
            "written)",
        ],
    }


def main() -> int:
    payload = build_manifest()
    path = common.OUTPUT_DIR / "ARTEFACT_MANIFEST.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps({"totals": payload["totals"],
                      "ledger": payload["launch_ledger"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    import json

    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
