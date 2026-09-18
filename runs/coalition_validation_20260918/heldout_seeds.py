"""Pre-registered held-out seed draw (brief section three), written and committed
before any held-out recording or replay exists.

Algorithm, fixed in advance: candidate seeds are ``30000, 30001, ...`` ascending; a
candidate is kept unless it appears in ANY historical manifest (listed by path with its
sha256 below, plus the three split blocks in ``integrated_sar/data_splits.py`` and the
development blocks this round created); the first 400 kept candidates form the set. No
sampling with rejection after the fact, no difficulty filtering, no re-draw.

    D:/project_env/python.exe runs/coalition_validation_20260918/heldout_seeds.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from integrated_sar.data_splits import SPLITS  # noqa: E402

COUNT = 400
BASE = 30000
# Every manifest whose seeds are already spent. Paths are repo-relative; each is
# hashed into the output so the exclusion set itself is checkable.
HISTORICAL_MANIFESTS = [
    "runs/final_eval_20260916/test_seeds.json",
    "runs/learnability_20260917/manifest.json",
    "runs/coalition_validation_20260918/ablation_seeds.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def used_seeds() -> tuple[set[int], list[dict]]:
    used: set[int] = set()
    sources: list[dict] = []
    for name, split in SPLITS.items():
        used.update(int(seed) for seed in split.seeds)
        sources.append({"source": f"data_splits:{name}", "count": len(split.seeds)})
    for relative in HISTORICAL_MANIFESTS:
        path = REPO / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        found: set[int] = set()

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "seeds" and isinstance(value, list):
                        found.update(int(item) for item in value)
                    else:
                        walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(payload)
        used.update(found)
        sources.append(
            {"source": relative, "sha256": sha256(path), "count": len(found)}
        )
    return used, sources


def draw() -> dict:
    used, sources = used_seeds()
    seeds: list[int] = []
    candidate = BASE
    while len(seeds) < COUNT:
        if candidate not in used:
            seeds.append(candidate)
        candidate += 1
    return {
        "algorithm": (
            "ascending integers from 30000, kept unless present in any listed historical "
            "manifest; the first 400 kept form the held-out set; no post-hoc rejection"
        ),
        "count": COUNT,
        "seeds": seeds,
        "excluded_union_size": len(used),
        "excluded_ranges": {
            "min": min(used) if used else None,
            "max": max(used) if used else None,
        },
        "historical_sources": sources,
        "overlap_with_used": sorted(set(seeds) & used),
    }


def main() -> int:
    payload = draw()
    assert not payload["overlap_with_used"], "held-out set collides with a used seed"
    (HERE / "HELDOUT_SEEDS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"seeds {payload['seeds'][0]}..{payload['seeds'][-1]} ({payload['count']}), "
          f"exclusion union {payload['excluded_union_size']}, overlap none")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
