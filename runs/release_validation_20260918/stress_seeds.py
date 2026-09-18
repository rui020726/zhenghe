"""Pre-registered stress seeds (release brief section 4).

Written, committed and hashed before any stress stream is recorded or any stress episode is
replayed. Three conditions, 60 seeds each, drawn from three disjoint hundred-wide bands far
from every seed this project has spent. A candidate is kept unless it appears in ANY
historical manifest or committed seed block; the first 60 kept in each band form that
condition's set. No sampling with rejection after the fact, no difficulty filtering, no
re-draw, and no seed shared between conditions, so each block stands on its own.

    D:/project_env/python.exe runs/release_validation_20260918/stress_seeds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from integrated_sar.data_splits import SPLITS  # noqa: E402

COUNT = 60
BANDS = {
    "A_tighter_deadline": 32000,
    "B_higher_load": 33000,
    "C_tight_and_heavy": 34000,
}
SPENT_MANIFESTS = [
    "runs/final_eval_20260916/test_seeds.json",
    "runs/learnability_20260917/manifest.json",
    "runs/coalition_validation_20260918/ablation_seeds.json",
    "runs/coalition_validation_20260918/HELDOUT_SEEDS.json",
    "runs/coalition_validation_20260918/SYSTEM_SEEDS.json",
]


def collect_seeds(node: object, found: set[int]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "seeds" and isinstance(value, list):
                found.update(int(item) for item in value)
            else:
                collect_seeds(value, found)
    elif isinstance(node, list):
        found.update(int(item) for item in node if isinstance(item, (int, float)))


def spent() -> tuple[set[int], list[dict]]:
    used: set[int] = set()
    sources = []
    for name, split in SPLITS.items():
        used.update(int(seed) for seed in split.seeds)
        sources.append({"source": f"data_splits:{name}", "count": len(split.seeds)})
    for relative in SPENT_MANIFESTS:
        found: set[int] = set()
        collect_seeds(json.loads((REPO / relative).read_text(encoding="utf-8")), found)
        used.update(found)
        sources.append({"source": relative, "count": len(found)})
    return used, sources


def main() -> int:
    used, sources = spent()
    spent_before = set(used)
    conditions = {}
    for name, base in BANDS.items():
        seeds: list[int] = []
        candidate = base
        while len(seeds) < COUNT:
            if candidate not in used:
                seeds.append(candidate)
                used.add(candidate)  # a later band may not take a seed an earlier band kept
            candidate += 1
        conditions[name] = seeds
    payload = {
        "algorithm": (
            "for each condition, ascending integers from its own band start (32000, 33000, "
            "34000), kept unless already spent or taken by an earlier band; the first 60 kept "
            "per condition; fixed before any stress recording or replay"
        ),
        "count_per_condition": COUNT,
        "conditions": conditions,
        "exclusion_sources": sources,
        "overlap_with_spent": sorted(
            {seed for seeds in conditions.values() for seed in seeds} & spent_before
        ),
        "shared_across_conditions": sorted(
            set(conditions["A_tighter_deadline"])
            & set(conditions["B_higher_load"])
            | set(conditions["A_tighter_deadline"]) & set(conditions["C_tight_and_heavy"])
            | set(conditions["B_higher_load"]) & set(conditions["C_tight_and_heavy"])
        ),
    }
    assert not payload["shared_across_conditions"], "two stress conditions share a seed"
    (HERE / "STRESS_SEEDS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    for name, seeds in conditions.items():
        print(f"{name}: {seeds[0]}..{seeds[-1]} ({len(seeds)})")
    print(f"exclusion union at draw time: {len(used)}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
