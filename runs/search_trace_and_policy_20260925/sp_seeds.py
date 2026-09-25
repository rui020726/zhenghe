"""Control-seed selection and the four-source seed collision audit.

Both outputs are produced *before* any episode runs, and both are derived from artefacts that
already exist, so neither can be chosen after seeing a result.

Control-seed rule (registered in SEARCH_TRACE_AND_POLICY_PROTOCOL §五): among the 60 seeds of
the delivered round-17 batch ``B1_d1_20260923``, take the seeds with zero undetected goals in
that batch's search-record arm, sort ascending, keep the first four. The rule reads only the
delivered attribution file, whose hash is recorded here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import sp_common as sc

CONTROL_COUNT = 4
S1_PATTERN = re.compile(r"12000[0-5][0-9]")
S1_BYTES = re.compile(rb"12000[0-5][0-9]")


def control_seeds() -> dict[str, Any]:
    attribution = sc.read_json(sc.ROUND17_ATTRIBUTION)
    batch = attribution["batches"]["B1_d1_20260923"]
    rows = batch["rows"]
    episodes = int(batch["denominators"]["episodes"])
    seeds = list(range(1100000, 1100000 + episodes))
    missed = sorted({int(row["seed"]) for row in rows if row["batch"] == "B1_d1_20260923"})
    clean = [seed for seed in seeds if seed not in missed]
    return {
        "rule": ("among the 60 seeds of the delivered round-17 batch B1_d1_20260923, "
                 "seeds with zero undetected goals, ascending, first four"),
        "input_file": str(sc.ROUND17_ATTRIBUTION.relative_to(sc.REPO_ROOT)).replace("\\", "/"),
        "input_sha256": sc.sha256_file(sc.ROUND17_ATTRIBUTION),
        "batch_episodes": episodes,
        "batch_undetected_goals": int(batch["denominators"]["undetected"]),
        "seeds_with_a_miss": missed,
        "seeds_with_no_miss_ascending": clean,
        "selected": clean[:CONTROL_COUNT],
        "selected_count": CONTROL_COUNT,
        "note": ("the rule does not look at any new run; it is computed from a delivered "
                 "artefact and frozen here before the first episode"),
    }


def collision_audit() -> dict[str, Any]:
    """Four sources: on-disk artefacts, recorded streams, ledgers, declared ranges."""
    candidates = list(sc.S1_SEEDS)
    hits: dict[str, list[str]] = {"artefacts": [], "recorded_streams": [], "ledgers": [],
                                   "declared_in_documents": []}
    roots = [sc.REPO_ROOT / "runs", sc.REPO_ROOT / "integrated_sar", sc.REPO_ROOT / "task_allocation"]
    stream_extensions = {".stream", ".jsonl", ".json"}
    document_extensions = {".md", ".txt", ".yaml", ".yml"}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            text_hit = S1_PATTERN.search(path.name) is not None
            content_hit = False
            if path.suffix in stream_extensions and size <= 64 * 1024 * 1024:
                with path.open("rb") as handle:
                    content_hit = S1_BYTES.search(handle.read()) is not None
            elif path.suffix in document_extensions and size <= 4 * 1024 * 1024:
                with path.open("rb") as handle:
                    content_hit = S1_BYTES.search(handle.read()) is not None
            if not (text_hit or content_hit):
                continue
            relative = str(path.relative_to(sc.REPO_ROOT)).replace("\\", "/")
            if path.suffix == ".stream":
                hits["recorded_streams"].append(relative)
            elif "ledger" in path.name:
                hits["ledgers"].append(relative)
            elif path.suffix in document_extensions:
                hits["declared_in_documents"].append(relative)
            else:
                hits["artefacts"].append(relative)
    claimed_overlap = [
        {"label": label, "low": low, "high": high,
         "overlaps_candidate_block": not (high < candidates[0] or low > candidates[-1])}
        for label, low, high in sc.CLAIMED_SEED_RANGES
    ]
    blocked = [key for key, rows in hits.items() if rows]
    return {
        "candidate_block": [candidates[0], candidates[-1]],
        "candidate_count": len(candidates),
        "sealed_block_untouched": {
            "range": [800000, 800399],
            "artefacts_on_disk": [
                str(path.relative_to(sc.REPO_ROOT)).replace("\\", "/")
                for path in (sc.REPO_ROOT / "runs").rglob("*")
                if path.is_file() and re.search(rb"800[0-3][0-9][0-9]", path.name.encode())
            ],
        },
        "claimed_ranges": claimed_overlap,
        "hits": hits,
        "clean": not blocked and not any(row["overlaps_candidate_block"] for row in claimed_overlap),
        "blocked_by": blocked,
        "gate": ("if this is not clean the formal matrix stops; the block is not renumbered to "
                 "find a clean interval"),
    }


def main(argv: list[str]) -> int:
    sc.write_json("CONTROL_SEEDS.json", control_seeds())
    audit = collision_audit()
    sc.write_json("SEED_COLLISION_AUDIT.json", audit)
    print("clean:", audit["clean"], "used:", sc.ledger_total())
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
