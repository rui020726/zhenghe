"""Emit RUN_LEDGER.json: the frozen launch budget plus the live tally.

The plan block is registered before the first episode and is re-checked on every refresh, so
the one file this round is allowed to rewrite cannot also widen its own budget.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import sp_common as sc

PURPOSE = {
    "b1": "B1 probe on/off x two candidate seeds x two repeats (passivity gate)",
    "repro": "ten historical miss seeds replayed + four registered no-miss controls",
    "smoke": "S1 interface, repeatability and configuration checks (simulation included)",
    "data": "formal comparison: 60 new paired seeds x two arms",
}


def _commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(sc.REPO_ROOT),
                            capture_output=True, text=True, timeout=30)
    return result.stdout.strip() or "<no-git>"


def plan_block() -> dict[str, Any]:
    return {
        "round": sc.ROUND,
        "start_branch": sc.START_BRANCH,
        "start_sha": sc.START_SHA,
        "registered_at_commit": _commit(),
        "cap": sc.LAUNCH_CAP,
        "phases": {name: {"planned": count, "purpose": PURPOSE[name]}
                   for name, count in sc.PHASE_PLAN.items()},
        "normal_total": sum(sc.PHASE_PLAN.values()),
        "reserve": sc.RESERVE,
        "reserve_may_be_used_for": ["technical failure", "correctness re-test"],
        "reserve_may_not_be_used_for": ["selecting more cases", "raising statistical significance",
                                        "converting the formal matrix into an open debugging budget"],
        "counting_rule": ("every joint-driver episode launch is booked, including failures, "
                          "smoke starts and early terminations; synthetic unit tests -- including "
                          "the ones that step the real timemap environment on hand-built maps -- "
                          "are not episode launches and produce no experimental evidence"),
        "unit_statement": {
            "primary": "undetected goals per episode (个/episode), denominator 60 episodes",
            "batch": "undetected goals per batch of 60 episodes (个/批)",
            "conversion": "1 个/批 = 1/60 ≈ 0.016667 个/episode",
            "task_level": "on-time completed tasks per episode (个/episode); different "
                          "denominator from goal counts, never added to them",
        },
        "hard_gate": ("B1 passivity and the detection-correctness verdict gate the S1 matrix "
                      "(protocol revision R1): with either failing, no search-side effect "
                      "number is produced at all"),
        "machine_books": "logs/ledger_<phase>_p<pid>.json",
    }


def main(argv: list[str]) -> int:
    payload = {"plan": plan_block(), "live": sc.ledger_report()}
    path = sc.write_json("RUN_LEDGER.json", payload, refresh=True)
    print(path)
    print(sys.stderr.write("") or "")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
