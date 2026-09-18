"""Pre-registration of the mechanism-ablation seed set (written before any ablation ran).

Rule, fixed before looking at any result: the five development failure seeds, plus ten
"normal" seeds drawn from ``analysis_legacy`` by taking, in ascending order, every seed
with ``seed % 10 == 0`` that is not a failure seed, then continuing with
``seed % 10 == 1`` until ten are collected. Content-independent, reproducible, and drawn
only from blocks already used for development -- this set analyses mechanisms and
supports no performance claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

FAILURE_SEEDS = [20003, 20008, 20052, 20070, 20083]
LEGACY = range(20000, 20100)
NORMAL_TAKE = 10


def normal_seeds() -> list[int]:
    failures = set(FAILURE_SEEDS)
    picked: list[int] = []
    for residue in (0, 1):
        for seed in LEGACY:
            if len(picked) >= NORMAL_TAKE:
                break
            if seed in failures or seed % 10 != residue:
                continue
            picked.append(seed)
    return picked[:NORMAL_TAKE]


def main() -> int:
    payload = {
        "rule": (
            "five development failure seeds + ten normal seeds drawn ascending by "
            "seed % 10 == 0 (failures excluded), continuing with seed % 10 == 1; "
            "fixed before any ablation run, drawn only from development-read blocks"
        ),
        "failure_seeds": FAILURE_SEEDS,
        "normal_seeds": normal_seeds(),
    }
    payload["seeds"] = FAILURE_SEEDS + [s for s in payload["normal_seeds"] if s not in FAILURE_SEEDS]
    (HERE / "ablation_seeds.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
