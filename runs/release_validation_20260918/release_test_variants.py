"""Which redundancy test should the release use? Compared offline.

``_joint_redundant`` asks whether the candidate can still claim a unit of the requirement it
believes, after the *other* on-site members are subtracted. The requirement it reads is the
task's remaining status, which the environment already computed with this candidate counted
among the members -- so a robot that is the sole provider of a requirement believes that
requirement is zero, and the test calls it redundant. This script recomputes the verdict on
the same recorded data with the coverage-preserving alternative -- release only if taking me
away leaves the believed stage requirement exactly as covered as it was -- and reports how
many judgements and how many executed releases each test allows.

Nothing here runs an episode or changes production code: it reads the provenance scan.

    D:/project_env/python.exe runs/release_validation_20260918/release_test_variants.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "provenance"


def covered(requirement, abilities):
    total = np.sum(abilities, axis=0) if len(abilities) else np.zeros(len(requirement))
    return np.minimum(requirement, total)


def alternative_verdict(records):
    """Re-run the joint decision with the coverage-preserving test, per advise call.

    Records arrive one per candidate per call, tagged by (seed, now, task_id); calls are
    replayed in the coordinator's own deterministic order with each accepted release leaving
    the pool before the next robot is judged.
    """
    grouped: dict[tuple[int, float, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["seed"], record["now"], record["task_id"])].append(record)
    verdicts: dict[tuple[int, float, str, str], bool] = {}
    unknown_gross = 0
    for key, group in grouped.items():
        first = group[0]
        gross = first.get("believed_gross_requirement")
        pool = {member: np.array(abilities, dtype=float) for member, abilities in first["pool_abilities"].items()}
        if gross is None:
            # The belief carries no stage requirement: the conservative answer is no release.
            unknown_gross += len(group)
            for record in group:
                verdicts[(record["seed"], record["now"], record["task_id"], record["agent_id"])] = False
            continue
        requirement = np.array(gross, dtype=float)
        for record in sorted(group, key=lambda row: row["agent_id"]):
            agent_id = record["agent_id"]
            mine = pool.get(agent_id)
            if mine is None:
                verdicts[(record["seed"], record["now"], record["task_id"], agent_id)] = False
                continue
            with_me = covered(requirement, list(pool.values()))
            others = [abilities for member, abilities in pool.items() if member != agent_id]
            without_me = covered(requirement, others)
            releasable = bool(np.array_equal(with_me, without_me))
            verdicts[(record["seed"], record["now"], record["task_id"], agent_id)] = releasable
            if releasable:
                pool.pop(agent_id)  # banked, exactly as the shipped test banks a release
    return verdicts, unknown_gross


def main() -> int:
    rows: list[dict] = []
    for path in sorted(OUT.glob("scan_s*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        trace_path = OUT / f"candidate_tr_s{seed}.jsonl"
        redecided = set()
        if trace_path.exists():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("event_type") == "decision" and row.get("replanned"):
                    redecided.add((str(row["agent_id"]), float(row["timestamp"])))
        for record in payload["checks"]:
            record["redecided_this_tick"] = (record["agent_id"], record["now"]) in redecided
            rows.append(record)
    verdicts, unknown_gross = alternative_verdict(rows)
    tally = Counter()
    executed = Counter()
    disagreements: list[dict] = []
    for record in rows:
        key = (record["seed"], record["now"], record["task_id"], record["agent_id"])
        proposed = verdicts.get(key, False)
        current = record["released"]
        tally[f"current={current}_proposed={proposed}"] += 1
        if record.get("redecided_this_tick"):
            tally[f"EXECUTED current={current}_proposed={proposed}"] += 1
            executed["total"] += 1
            if record.get("truth_marginal_contribution", 0.0) > 1e-9:
                executed["harmful"] += 1
                executed["harmful_current_allows" if current else "harmful_current_blocks"] += 1
                executed["harmful_proposed_allows" if proposed else "harmful_proposed_blocks"] += 1
        if current and not proposed:
            disagreements.append(
                {
                    "seed": record["seed"],
                    "now": record["now"],
                    "agent_id": record["agent_id"],
                    "task_id": record["task_id"],
                    "believed_remaining": record["believed_requirement"],
                    "believed_gross": record.get("believed_gross_requirement"),
                    "pool": record["onsite_pool"],
                    "truth_marginal_contribution": record.get("truth_marginal_contribution"),
                    "redecided_this_tick": record.get("redecided_this_tick"),
                }
            )
    payload = {
        "release_candidates": len(rows),
        "tally": dict(tally),
        "executed_releases": dict(executed),
        "calls_without_a_believed_gross_requirement": unknown_gross,
        "current_allows_proposed_blocks": len(disagreements),
        "disagreement_sample": sorted(
            disagreements, key=lambda row: (not row["redecided_this_tick"], -(row["truth_marginal_contribution"] or 0))
        )[:25],
    }
    (HERE / "RELEASE_TEST_VARIANTS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "disagreement_sample"}, indent=1))
    print("\nexecuted releases the current test allows and the coverage-preserving test blocks:")
    for row in payload["disagreement_sample"]:
        if row["redecided_this_tick"]:
            print("   ", json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
