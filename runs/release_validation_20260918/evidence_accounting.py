"""Evidence accounting for the redundant-release candidate (release brief section 1.5).

Every number quoted for this candidate has to say which block it came from and whether that
block was already read during development. This script enumerates the blocks, their seeds,
how many replay episodes each produced, and the overlaps between them, so a "620 paired
episodes" style total cannot quietly mix independent evidence with reused development seeds.

    D:/project_env/python.exe runs/release_validation_20260918/evidence_accounting.py
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

VALIDATION = REPO / "runs" / "coalition_validation_20260918"
DISPATCH = REPO / "runs" / "dispatch_trigger_audit_20260918"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def block(name: str, role: str, seeds: list[int], episodes: int, arms: int, source: str, note: str) -> dict:
    return {
        "name": name,
        "role": role,  # development | independent | live-system
        "seeds": sorted(set(int(seed) for seed in seeds)),
        "unique_seeds": len(set(int(seed) for seed in seeds)),
        "paired_scenarios": episodes,
        "replay_episodes": episodes * arms,
        "arms": arms,
        "source": source,
        "note": note,
    }


def main() -> int:
    legacy = list(SPLITS["analysis_legacy"].seeds)
    fresh = list(SPLITS["analysis_fresh"].seeds)
    dev = list(SPLITS["dev"].seeds)
    failure = [20003, 20008, 20052, 20070, 20083]

    ablation = load(VALIDATION / "ablation_seeds.json")
    heldout = load(VALIDATION / "HELDOUT_SEEDS.json")
    system_seeds = load(VALIDATION / "SYSTEM_SEEDS.json")
    stress = load(HERE / "STRESS_SEEDS.json")
    expansion = load(DISPATCH / "paired" / "expansion.json")

    gate = load(DISPATCH / "CLOSED_LOOP_GATE.json")
    ablation_rows = load(VALIDATION / "ABLATION.json")
    system_rows = load(VALIDATION / "SYSTEM_LOOP.json")

    expansion_seeds = {
        split: [int(row["seed"]) for row in payload["paired"]]
        for split, payload in expansion["splits"].items()
    }

    blocks = [
        block(
            "stress_A_tighter_deadline",
            "stress",
            stress["conditions"]["A_tighter_deadline"],
            len(stress["conditions"]["A_tighter_deadline"]),
            2,
            "runs/release_validation_20260918/STRESS_SEEDS.json",
            "60 new seeds, deadline band (0.4, 1.2)",
        ),
        block(
            "stress_B_higher_load",
            "stress",
            stress["conditions"]["B_higher_load"],
            len(stress["conditions"]["B_higher_load"]),
            2,
            "runs/release_validation_20260918/STRESS_SEEDS.json",
            "60 new seeds, nine goals",
        ),
        block(
            "stress_C_tight_and_heavy",
            "stress",
            stress["conditions"]["C_tight_and_heavy"],
            len(stress["conditions"]["C_tight_and_heavy"]),
            2,
            "runs/release_validation_20260918/STRESS_SEEDS.json",
            "60 new seeds, both stressors",
        ),
        block(
            "coverage_verdict_development_rerun",
            "development",
            list(ablation["seeds"]),
            len(ablation["seeds"]),
            2,
            "runs/release_validation_20260918/DEV_RERUN.json",
            "reuses the gate's 5 and the ablation's 15 seeds; re-measures them under the "
            "coverage-preserving verdict, so it adds no new seed evidence",
        ),
        block(
            "closed_loop_gate",
            "development",
            list(gate["seeds"]),
            len(gate["seeds"]),
            2,
            "runs/dispatch_trigger_audit_20260918/CLOSED_LOOP_GATE.json",
            "the 5 failure seeds the fix was developed on; smallest closed loop",
        ),
        block(
            "mechanism_ablation",
            "development",
            list(ablation["seeds"]),
            len(ablation["seeds"]),
            len(ablation_rows["arms"]),
            "runs/coalition_validation_20260918/ABLATION.json",
            "5 failure + 10 normal seeds, all inside analysis_legacy; 4 arms A/B/C/D",
        ),
        block(
            "paired_expansion_legacy",
            "development",
            expansion_seeds["analysis_legacy"],
            len(expansion_seeds["analysis_legacy"]),
            2,
            "runs/dispatch_trigger_audit_20260918/paired/expansion.json",
            "analysis_legacy is a previously read block; explicitly not held-out",
        ),
        block(
            "paired_expansion_fresh",
            "development",
            expansion_seeds["analysis_fresh"],
            len(expansion_seeds["analysis_fresh"]),
            2,
            "runs/dispatch_trigger_audit_20260918/paired/expansion.json",
            "analysis_fresh was read when the learnability gate was specified",
        ),
        block(
            "heldout_replay",
            "independent",
            list(heldout["seeds"]),
            int(heldout["count"]),
            2,
            "runs/coalition_validation_20260918/HELDOUT_RESULTS.json",
            "400 seeds never used before the protocol was committed",
        ),
        block(
            "live_system_closed_loop",
            "live-system",
            list(system_seeds["seeds"]),
            len(system_seeds["seeds"]),
            2,
            "runs/coalition_validation_20260918/SYSTEM_LOOP.json",
            "60 new seeds, real PILOC search, no replayed streams",
        ),
    ]

    # Overlap matrix, so "reused" is stated as a set relation rather than a claim.
    overlaps = {}
    for i, left in enumerate(blocks):
        for right in blocks[i + 1 :]:
            shared = sorted(set(left["seeds"]) & set(right["seeds"]))
            if shared:
                overlaps[f"{left['name']} ∩ {right['name']}"] = {
                    "count": len(shared),
                    "seeds": shared[:12],
                }

    independent = [b for b in blocks if b["role"] == "independent"]
    live = [b for b in blocks if b["role"] == "live-system"]
    development = [b for b in blocks if b["role"] == "development"]
    stress_blocks = [b for b in blocks if b["role"] == "stress"]
    union_independent_live = set().union(*(set(b["seeds"]) for b in independent + live))
    union_all = set().union(*(set(b["seeds"]) for b in blocks))

    # The blocks this candidate never touched, recorded so the exclusion set is checkable.
    spent_elsewhere = {
        "dev_split": sorted(dev),
        "analysis_legacy": sorted(legacy),
        "analysis_fresh": sorted(fresh),
    }
    payload = {
        "blocks": [
            {key: value for key, value in item.items() if key != "seeds"} | {"seed_sample": item["seeds"][:6]}
            for item in blocks
        ],
        "block_seed_lists": {item["name"]: item["seeds"] for item in blocks},
        "overlaps": overlaps,
        "totals": {
            "paired_scenarios_all_blocks": sum(b["paired_scenarios"] for b in blocks),
            "replay_episodes_all_blocks": sum(b["replay_episodes"] for b in blocks),
            "unique_seeds_all_blocks": len(union_all),
            "independent_paired_scenarios": sum(b["paired_scenarios"] for b in independent),
            "live_system_paired_scenarios": sum(b["paired_scenarios"] for b in live),
            "unique_seeds_independent_or_live": len(union_independent_live),
            "development_paired_scenarios": sum(b["paired_scenarios"] for b in development),
            "development_unique_seeds": len(set().union(*(set(b["seeds"]) for b in development))),
            "stress_paired_scenarios": sum(b["paired_scenarios"] for b in stress_blocks),
            "stress_unique_seeds": len(set().union(*(set(b["seeds"]) for b in stress_blocks))),
        },
        "already_spent_blocks_excluded_by_the_held_out_draw": {
            name: len(seeds) for name, seeds in spent_elsewhere.items()
        },
        "heldout_excluded_union_size": heldout["excluded_union_size"],
        "system_excluded_union_size": system_seeds["excluded_union_size"],
    }
    (HERE / "EVIDENCE_ACCOUNTING.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    print("block                         role          scen  arms  episodes  uniq seeds")
    for item in blocks:
        print(
            f"{item['name']:<28} {item['role']:<12} {item['paired_scenarios']:>5} {item['arms']:>4} "
            f"{item['replay_episodes']:>9} {item['unique_seeds']:>9}"
        )
    print()
    for key, value in payload["totals"].items():
        print(f"{key}: {value}")
    print()
    print("overlaps:")
    for key, value in overlaps.items():
        print(f"  {key}: {value['count']} {value['seeds']}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
