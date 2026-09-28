"""ROUND24 §8 原始产物清单：rows/logs/checkpoints 的路径、大小、sha256（0 启动，只读）。

原始权重、逐 episode 日志与 tape 不入 Git；本清单让任何复核者能在 Git 之外逐文件校验本轮
672 次启动的落盘件。已提交的机读交付件（manifest、门禁、对比、判定）单独列节并复验哈希。

    /d/project_env/python.exe artefact_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHUNK = 1 << 20

DELIVERED_JSON = (
    "CONFIG_AND_MODEL_MANIFEST.json", "SCENE_REGISTRY.json", "START_LEDGER.json",
    "SYNTHETIC_GATE.json", "START_EQUIVALENCE.json", "FIRST4_GATE_s11.json",
    "BATCH_AUDIT_s11.json", "BATCH_AUDIT_s22.json", "BATCH_AUDIT_s33.json",
    "BATCH_CHAIN_REVIEW_s11.json", "BATCH_CHAIN_REVIEW_s22.json", "BATCH_CHAIN_REVIEW_s33.json",
    "TRAINING_CURVES.json", "CHECKPOINT_INDEX.json", "COMPARISON.json", "UNCERTAINTY.json",
    "ROUND24_VERDICT.json",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            value.update(block)
    return value.hexdigest()


def inventory(directory: Path) -> dict[str, Any]:
    files = []
    total = 0
    if not directory.is_dir():
        return {"files": [], "count": 0, "bytes": 0}
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            continue
        size = path.stat().st_size
        total += size
        files.append({"path": str(path.relative_to(HERE)).replace("\\", "/"),
                      "bytes": size, "sha256": digest(path)})
    return {"files": files, "count": len(files), "bytes": total}


def main() -> int:
    payload = {
        "round": "24",
        "run_id": HERE.name,
        "inventory": {name: inventory(HERE / name)
                      for name in ("rows", "logs", "checkpoints")},
        "delivered_artefacts": {
            name: {"bytes": (HERE / name).stat().st_size, "sha256": digest(HERE / name)}
            for name in DELIVERED_JSON if (HERE / name).exists()},
        "note": "raw weights, per-episode logs and tapes stay out of Git; this manifest records "
                "their path, size and sha256 so they remain auditable outside the repository",
    }
    counts = {name: section["count"] for name, section in payload["inventory"].items()}
    payload["counts"] = counts
    (HERE / "ARTEFACT_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts,
                      "delivered": len(payload["delivered_artefacts"]),
                      "bytes": {name: section["bytes"] for name, section in
                                payload["inventory"].items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
