"""The seed draw for the corrected-environment independent block, fixed before any run.

Rule, chosen in advance and not tunable afterwards: candidates are the ascending integers
from ``BASE`` upwards; a candidate is kept unless it appears anywhere this script can find a
seed having been used -- the registered data splits, every ``seeds`` list in every manifest
under ``runs/``, and every artefact or stream file named ``...s<digits>...``. The first 400
kept candidates form the block. No sampling with post-hoc rejection, no difficulty filter,
no re-draw after results are seen, and no reuse of the old environment's 400.

The exclusion sweep is deliberately wider than the previous round's: the round that drew
``30000..`` is itself now spent history, and so is every seed the stress block and the
membership scans touched.

    D:/project_env/python.exe runs/corrected_env_final_20260918/seeds_v2.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from integrated_sar.data_splits import SPLITS  # noqa: E402

COUNT = 400
BASE = 500000
# The conditional online-system block: fixed here, before either block has produced a number.
SYSTEM_COUNT = 60
SYSTEM_BASE = 500400


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def collect() -> tuple[set[int], dict[str, int]]:
    used: set[int] = set()
    provenance: dict[str, int] = {}

    def add(name: str, found: set[int]) -> None:
        if found:
            provenance[name] = len(found)
            used.update(found)

    for label, split in SPLITS.items():
        add(f"data_splits:{label}", {int(seed) for seed in split.seeds})

    from_names: set[int] = set()
    from_manifests: set[int] = set()
    scanned = 0
    for root, directories, files in os.walk(REPO / "runs"):
        # This block's own products are not history: its streams and per-seed records would
        # otherwise count as spent seeds and move the block every time it is re-run.
        directories[:] = [
            name for name in directories if name != "__pycache__" and (Path(root) / name) != HERE
        ]
        if Path(root) == HERE:
            files = [name for name in files if not re.match(r"(record_)?s\d+\.", name)]
        for name in files:
            path = Path(root) / name
            lowered = name.lower()
            for match in re.finditer(r"[^a-z0-9]s(\d{2,8})(?:\.|$)", lowered):
                from_names.add(int(match.group(1)))
            if not lowered.endswith(".json") or path.stat().st_size > 4_000_000:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            scanned += 1

            def walk(node: object) -> None:
                if isinstance(node, dict):
                    for key, value in node.items():
                        if key in ("seeds", "failure_seeds", "control_seeds") and isinstance(
                            value, list
                        ):
                            from_manifests.update(
                                int(item)
                                for item in value
                                if isinstance(item, int) and 0 < item < 10**7
                            )
                        elif key == "seed" and isinstance(value, int) and 0 < value < 10**7:
                            from_manifests.add(value)
                        else:
                            walk(value)
                elif isinstance(node, list):
                    for item in node:
                        walk(item)

            walk(payload)

    add("runs artefact and stream names", from_names)
    provenance["manifests scanned"] = scanned
    add("runs manifests", from_manifests)
    return used, provenance


def draw() -> dict:
    used, provenance = collect()
    seeds: list[int] = []
    candidate = BASE
    while len(seeds) < COUNT:
        if candidate not in used:
            seeds.append(candidate)
        candidate += 1
    system: list[int] = []
    candidate = SYSTEM_BASE
    while len(system) < SYSTEM_COUNT:
        if candidate not in used and candidate not in seeds:
            system.append(candidate)
        candidate += 1
    committed = json.loads(HERE.joinpath("HELDOUT_SEEDS_V2.json").read_text(encoding="utf-8"))["seeds"] if (HERE / "HELDOUT_SEEDS_V2.json").exists() else []
    assert not committed or committed == seeds, (
        "the registered block of 400 may not shift once it exists: "
        f"{len(set(committed) ^ set(seeds))} seeds differ"
    )
    return {
        "system_seeds": system,
        "system_rule": (
            f"the same ascending-kept rule from {SYSTEM_BASE}, {SYSTEM_COUNT} seeds, also "
            "disjoint from the block above and from every historical source"
        ),
        "system_overlap_with_used": sorted(set(system) & used),
        "algorithm": (
            f"ascending integers from {BASE}, kept unless present in any historical manifest "
            "or artefact name under runs/ (plus the registered data splits); the first "
            f"{COUNT} kept candidates form the block; no post-hoc rejection and no re-draw"
        ),
        "count": COUNT,
        "seeds": seeds,
        "scenario": {
            "num_goals": 6,
            "steps": 1000,
            "grid_size": 60,
            "obstacle_density": 0.15,
            "detection_radius": 5,
            "search_agents": 2,
            "deadline_tightness": [0.5, 2.0],
            "note": "the standard distribution, unchanged from the recording protocol",
        },
        "exclusion_union_size": len(used),
        "exclusion_digest": sha256_text(",".join(str(seed) for seed in sorted(used))),
        "exclusion_provenance": provenance,
        "spent_400_block_of_20260918_kept_apart": True,
        "overlap_with_used": sorted(set(seeds) & used),
    }


def main() -> int:
    payload = draw()
    assert not payload["overlap_with_used"], "the drawn block collides with a used seed"
    assert not payload["system_overlap_with_used"], "a system seed collides with a used seed"
    assert not set(payload["system_seeds"]) & set(payload["seeds"]), "a seed is in both blocks"
    destination = HERE / "HELDOUT_SEEDS_V2.json"
    part = destination.with_suffix(".json.part")
    part.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(part, destination)
    print(
        f"seeds {payload['seeds'][0]}..{payload['seeds'][-1]} ({payload['count']}), "
        f"exclusion union {payload['exclusion_union_size']} seeds, digest "
        f"{payload['exclusion_digest'][:12]}, overlap none"
    )
    for name, count in sorted(payload["exclusion_provenance"].items()):
        print(f"  excluded from {name}: {count}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
