"""Development re-run under the coverage-preserving verdict (release brief section 3).

The redundancy test changed, so every development number that named the old verdict has to
be re-measured before anything is claimed about the new one. This re-runs exactly the two
development blocks the candidate was built on, with the same configurations, the same
recorded streams and the same metrics, writing to its own directory so the earlier
artefacts stay readable:

* the five failure seeds, binding baseline versus release candidate (the closed-loop gate);
* the fifteen pre-registered ablation seeds, all four mechanism arms.

    D:/project_env/python.exe runs/release_validation_20260918/dev_rerun.py --worker 0 --workers 8
    D:/project_env/python.exe runs/release_validation_20260918/dev_rerun.py --aggregate
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
sys.path.insert(0, str(REPO / "runs" / "coalition_validation_20260918"))
sys.path.insert(0, str(REPO / "runs" / "dispatch_trigger_audit_20260918"))

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
try:
    import torch

    torch.set_num_threads(1)
except Exception:  # torch is optional in dependency-free environments
    pass

import closed_loop_gate  # noqa: E402
import mechanism_ablation  # noqa: E402
from ablation_seeds import FAILURE_SEEDS, normal_seeds  # noqa: E402

OUT = HERE / "dev"
GATE_ARMS = (("baseline", closed_loop_gate.BASE), ("candidate", closed_loop_gate.CANDIDATE))
ABLATION_ARMS = list(mechanism_ablation.ARMS)
JOBS = [("gate", seed, tag) for seed in FAILURE_SEEDS for tag, _ in GATE_ARMS] + [
    ("ablation", seed, arm) for seed in FAILURE_SEEDS + normal_seeds() for arm in ABLATION_ARMS
]
TAGS = {tag: policy for tag, policy in GATE_ARMS}


def run_job(kind: str, seed: int, arm: str) -> dict:
    directory = OUT / kind
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{arm}_s{seed}.json"
    if kind == "gate":
        # Point the archived gate harness at this directory only; its configuration is
        # the one that produced the numbers being compared.
        closed_loop_gate.HERE = directory
        row = closed_loop_gate.run_one(seed, TAGS[arm], arm)
        part = destination.with_suffix(".json.part")
        part.write_text(json.dumps(row, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        os.replace(part, destination)
    else:
        # The ablation harness writes its own summary and returns the same record.
        mechanism_ablation.OUT = directory
        row = mechanism_ablation.run_arm(seed, arm)
    print(
        f"[{kind}/{arm} s{seed}] completed={row.get('completed')} expired={row.get('expired')} "
        f"resets={row.get('coverage_resets')} lost={row.get('lost_work')} "
        f"releases={row.get('releases_planned', '-')} recommands={row.get('recommands')}",
        flush=True,
    )
    return row


def aggregate() -> int:
    rows = {"gate": [], "ablation": []}
    for kind in rows:
        for path in sorted((OUT / kind).glob("*.json")):
            rows[kind].append(json.loads(path.read_text(encoding="utf-8")))

    v1 = json.loads(
        (REPO / "runs" / "dispatch_trigger_audit_20260918" / "CLOSED_LOOP_GATE.json").read_text(
            encoding="utf-8"
        )
    )
    v1_ablation = json.loads(
        (REPO / "runs" / "coalition_validation_20260918" / "ABLATION.json").read_text(encoding="utf-8")
    )

    def totals(kind: str, key: str, arm: str) -> float:
        wanted = TAGS.get(arm, arm) if kind == "gate" else arm
        return float(sum(row[key] for row in rows[kind] if row.get(arm_key(kind)) == wanted))

    def arm_key(kind: str) -> str:
        return "policy" if kind == "gate" else "arm"

    gate = {
        arm: {
            key: totals("gate", key, arm)
            for key in ("completed", "expired", "coverage_resets", "lost_work", "recommands",
                        "rejected_assignments", "executed_override_unjustified")
        }
        for arm, _ in GATE_ARMS
    }
    ablation = {
        arm: {
            key: totals("ablation", key, arm)
            for key in ("completed", "expired", "coverage_resets", "lost_work", "releases_planned",
                        "redundant_working_ticks", "recommands", "accepted_reassignments",
                        "replan_declined", "same_task_noop", "rejected_assignments",
                        "executed_override_unjustified")
        }
        for arm in ABLATION_ARMS
    }
    # Per-target regressions against the baseline arm, the number that actually matters.
    regressions: list[str] = []
    rescues: list[str] = []
    by_key: dict[tuple, dict] = {}
    for row in rows["gate"]:
        by_key[(row["seed"], row["policy"])] = row
    for seed in FAILURE_SEEDS:
        base = by_key.get((seed, closed_loop_gate.BASE), {}).get("targets", {})
        cand = by_key.get((seed, closed_loop_gate.CANDIDATE), {}).get("targets", {})
        for target in set(base) | set(cand):
            def on_time(source):
                entry = source.get(target, {})
                return float(entry.get("final_progress", 0.0)) >= 1.0 and not entry.get(
                    "finished_after_deadline"
                )

            if on_time(cand) and not on_time(base):
                rescues.append(f"{seed}/{target}")
            if on_time(base) and not on_time(cand):
                regressions.append(f"{seed}/{target}")

    payload = {
        "code_note": "coverage-preserving redundancy verdict",
        "gate": {
            "seeds": FAILURE_SEEDS,
            "arms": gate,
            "rescued_targets": sorted(rescues),
            "regressed_targets": sorted(regressions),
            "previous_verdict_for_reference": {
                "arms": v1["comparison"],
                "rescued": v1["rescued"],
                "new_failures": v1["new_failures"],
            },
        },
        "ablation": {
            "seeds": FAILURE_SEEDS + normal_seeds(),
            "arms": ablation,
            "previous_verdict_for_reference": {
                arm: {key: value for key, value in summary.items() if key in dict(ablation.get(ABLATION_ARMS[0], {}))}
                for arm, summary in v1_ablation["arms"].items()
            },
        },
    }
    (HERE / "DEV_RERUN.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate": gate, "ablation": ablation, "rescues": sorted(rescues),
                      "regressions": sorted(regressions)}, indent=1))
    return 0


def main(argv: list[str]) -> int:
    if "--worker" in argv:
        index = int(argv[argv.index("--worker") + 1])
        total = int(argv[argv.index("--workers") + 1])
        for position, (kind, seed, arm) in enumerate(JOBS):
            if position % total != index:
                continue
            destination = OUT / kind / f"{arm}_s{seed}.json"
            if not destination.exists():
                run_job(kind, seed, arm)
        return 0
    if "--aggregate" in argv:
        return aggregate()
    OUT.mkdir(parents=True, exist_ok=True)
    processes = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(i), "--workers", str(8)],
            cwd=str(REPO),
        )
        for i in range(8)
    ]
    codes = [process.wait() for process in processes]
    if any(codes):
        print(f"worker failures: {codes}")
        return 1
    return aggregate()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
