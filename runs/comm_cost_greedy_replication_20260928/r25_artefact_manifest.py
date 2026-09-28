"""ROUND25 原始产物 hash 清单：rows/logs/claims 等不入 Git 的产物逐件登记 (path,size,sha256)。

执行依据 §四/§七：原始日志、tape、checkpoint 不入 Git，保存外部 hash 清单，确保远端复核有原件可找。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import r25_common as common


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


def build_manifest(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    sections = {name: _scan(getattr(common, name))
                for name in ("ROWS", "LOGS", "CLAIMS", "REJECTED")}
    totals = {name: {"files": len(entries),
                     "bytes": sum(int(entry["bytes"]) for entry in entries.values())}
              for name, entries in sections.items()}
    payload: dict[str, Any] = {
        "round": "25",
        "run_id": common.OUTPUT_DIR.name,
        "sections": sections,
        "totals": totals,
        "git_policy": "rows, logs, tapes, claims and checkpoints stay out of version control; "
                      "this manifest is the external hash list that lets a remote reviewer find "
                      "and verify the originals",
        "read_only_sources": [
            "runs/comm_cost_v2_rerun_20260928 (ROUND24 execution artefacts, never written)",
        ],
    }
    if extra:
        payload.update(extra)
    return payload


def main() -> int:
    payload = build_manifest()
    path = common.OUTPUT_DIR / "ARTEFACT_MANIFEST.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps({"totals": payload["totals"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
