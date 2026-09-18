"""Pre-registered seeds for the live-system closed loop (brief section six).

Drawn before any system-loop result exists: 60 ascending integers from 31000, keeping
only seeds absent from every historical manifest AND from the 400 held-out replay seeds.
These episodes run the real PILOC search side, so discovery order is part of the
measurement and the two arms are not required to share detection streams.

    D:/project_env/python.exe runs/coalition_validation_20260918/system_seeds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from heldout_seeds import used_seeds  # noqa: E402

COUNT = 60
BASE = 31000


def main() -> int:
    used, sources = used_seeds()
    heldout = [int(seed) for seed in json.loads(
        (HERE / "HELDOUT_SEEDS.json").read_text(encoding="utf-8")
    )["seeds"]]
    used.update(heldout)
    seeds: list[int] = []
    candidate = BASE
    while len(seeds) < COUNT:
        if candidate not in used:
            seeds.append(candidate)
        candidate += 1
    payload = {
        "algorithm": (
            "ascending from 31000, kept unless present in a historical manifest or in the "
            "400 held-out replay seeds; 60 kept; fixed before any live-system result"
        ),
        "count": COUNT,
        "seeds": seeds,
        "excluded_union_size": len(used),
        "overlap": sorted(set(seeds) & used),
        "historical_sources": [source["source"] for source in sources] + ["HELDOUT_SEEDS.json"],
    }
    (HERE / "SYSTEM_SEEDS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"system seeds {seeds[0]}..{seeds[-1]}, overlap none, exclusion union {len(used)}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
