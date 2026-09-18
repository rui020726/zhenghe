"""Paired stress replays under one frozen commit (release brief section 4).

Three pre-registered conditions, 60 unseen scenario seeds each, two arms per seed reading
the same recorded detection stream. Metric definitions come from the ablation harness so a
stress number and a development number mean the same thing.

Guards, in this order:
* ``--identity``  the all-off configuration must still reproduce the 868d68a-era trajectory;
* ``--worker``    shard the 360 replays; each worker writes only its own files;
* ``--preflight`` every artefact must name the frozen commit, its own seed, and the digest
  of the stream it actually replayed;
* ``--aggregate`` the pre-declared statistics: seed-level paired bootstrap, nominal 95 %
  and Bonferroni-adjusted intervals, count closure, and the correctness checks kept apart
  from performance.

    D:/project_env/python.exe runs/release_validation_20260918/stress_replay.py --identity
    D:/project_env/python.exe runs/release_validation_20260918/stress_replay.py
    D:/project_env/python.exe runs/release_validation_20260918/stress_replay.py --aggregate
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
try:  # applied before any checkpoint load
    import torch

    torch.set_num_threads(1)
except Exception:  # torch absent only in dependency-free environments, where it is moot
    pass

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (REPO, HERE, REPO / "runs" / "coalition_validation_20260918"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

import mechanism_ablation as ablation  # noqa: E402  (metric definitions are shared)

# The runs record the HEAD they executed under, which is the freeze commit itself (it
# differs from ``fd0dd76`` only by this line: ``git diff fd0dd76 3fcef63 -- integrated_sar``
# is empty, so the production tree is the frozen one).
FROZEN_COMMIT = "3fcef63"  # full: 3fcef636d6dfc1dca599877238bca4e4006fe81f
STEPS = 1000
WORKERS = 8
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260918
CONDITIONS = {
    "A_tighter_deadline": {"deadline_tightness": (0.4, 1.2), "num_goals": 6},
    "B_higher_load": {"deadline_tightness": (0.5, 2.0), "num_goals": 9},
    "C_tight_and_heavy": {"deadline_tightness": (0.4, 1.2), "num_goals": 9},
}
BASELINE = {"policy": "coalition_planner", "release": False, "suppress": False}
CANDIDATE = {"policy": "coalition_planner", "release": True, "suppress": True}
ARMS = {"baseline": BASELINE, "candidate": CANDIDATE}
OUT = HERE / "stress"
DEV_SEEDS = [20003, 20008, 20052, 20070, 20083]
COMMITTED_TR = (
    REPO / "runs" / "coordinator_baselines_20260917" / "heldout" / "arms" / "coalplan0.25sticky"
)
FIELDS = (
    "step", "agent_id", "chosen_task", "actual_assignment", "needs_decision",
    "raw_choice", "post_mask_choice", "chosen_index",
)


def seeds_for(condition: str) -> list[int]:
    payload = json.loads((HERE / "STRESS_SEEDS.json").read_text(encoding="utf-8"))
    return [int(seed) for seed in payload["conditions"][condition]]


def stream_path(condition: str, seed: int) -> Path:
    return HERE / "stress_streams" / condition / f"s{seed}.json"


def configuration_for(condition: str, seed: int, arm: str, stream: Path) -> RunConfiguration:
    spec = ARMS[arm]
    scenario = CONDITIONS[condition]
    directory = OUT / condition
    directory.mkdir(parents=True, exist_ok=True)
    return RunConfiguration(
        seed=seed,
        steps=STEPS,
        device="cpu",
        output=str(directory / f"{arm}_rp_s{seed}.jsonl"),
        decision_trace=str(directory / f"{arm}_tr_s{seed}.jsonl"),
        task_checkpoint=str(REPO / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth"),
        mock=False,
        coordinator_policy=spec["policy"],
        sticky_commitments=True,
        use_coordinator=True,
        release_redundant_members=spec["release"],
        suppress_recommand_own_task=spec["suppress"],
        residual_scale=0.25,
        allocation_interval=5,
        coordination_interval=5,
        command_ttl=10.0,
        dt=1.0,
        detection_radius=5,
        search_agents=2,
        num_goals=scenario["num_goals"],
        deadline_tightness=tuple(scenario["deadline_tightness"]),
        grid_size=60,
        obstacle_density=0.15,
        rescue_species=3,
        rescue_per_species=2,
        actor_kv_mode="tokens",
        comm_delivery_probability=1.0,
        comm_delay_ticks=0,
        comm_subscribers="all",
        quiet=True,
        replay_stream=str(stream),
    )


def run_pair(condition: str, seed: int, arm: str) -> dict:
    stream = stream_path(condition, seed)
    counters = {
        "releases_planned": 0,
        "redundant_working_ticks": 0,
        "working_ticks": 0,
    }
    timings: list[float] = []
    run = build_joint_run(configuration_for(condition, seed, arm, stream))
    _instrument(run, counters, timings)
    summary = run.run()
    directory = OUT / condition
    metrics = ablation.measure(
        seed, arm, directory / f"{arm}_rp_s{seed}.jsonl", directory / f"{arm}_tr_s{seed}.jsonl",
        summary, counters,
    )
    metrics["condition"] = condition
    metrics["stream_sha256"] = hashlib.sha256(stream.read_bytes()).hexdigest()
    metrics["rescue_travel"] = round(
        float(sum(agent.get("travel_dist", 0.0) for agent in run.task_adapter.env.agent_dic.values())), 3
    )
    metrics["decision_milliseconds"] = _quantiles(timings)
    part = (directory / f"{arm}_s{seed}.json").with_suffix(".json.part")
    part.write_text(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(part, directory / f"{arm}_s{seed}.json")
    return metrics


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "p99": None, "calls": 0}
    array = np.array(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 3),
        "p50": round(float(np.percentile(array, 50)), 3),
        "p95": round(float(np.percentile(array, 95)), 3),
        "p99": round(float(np.percentile(array, 99)), 3),
        "calls": int(array.size),
    }


def _instrument(run, counters: dict, timings: list[float]) -> None:
    env = run.task_adapter.env
    original_step = run.orchestrator.step
    coordinator = run.orchestrator.coordinator
    inner_advise = coordinator.advise

    def timed_advise(state):
        started = time.perf_counter()
        advice = inner_advise(state)
        timings.append((time.perf_counter() - started) * 1000.0)
        return advice

    coordinator.advise = timed_advise

    def step():
        report = original_step()
        abilities = {str(aid): np.array(agent["abilities"], dtype=float) for aid, agent in env.agent_dic.items()}
        for task in env.task_dic.values():
            members = [str(member) for member in task["members"]]
            index = min(int(task["current_stage_index"]), len(task["stages"]) - 1)
            requirement = np.array(task["stages"][index]["requirements"], dtype=float)
            for member in members:
                if member not in abilities or env.agent_dic.get(int(member)) is None:
                    continue
                if env.agent_dic[int(member)].get("status") != "working":
                    continue
                others = [abilities[other] for other in members if other != member and other in abilities]
                covered = np.sum(others, axis=0) if others else np.zeros(requirement.shape)
                gap_without = np.maximum(requirement - covered, 0.0)
                counters["working_ticks"] += 1
                if float(np.minimum(gap_without, abilities[member]).sum()) <= 1e-9:
                    counters["redundant_working_ticks"] += 1
        counters["releases_planned"] += int((report.advice.metadata or {}).get("redundant_releases", 0) or 0)
        return report

    run.orchestrator.step = step


def identity_check() -> int:
    """The all-off configuration must reproduce the committed binding arm, field by field."""
    import dataclasses

    scratch = HERE / "identity_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    ok = True
    for seed in DEV_SEEDS:
        configuration = dataclasses.replace(
            configuration_for(
                "A_tighter_deadline",
                seed,
                "baseline",
                REPO / "runs" / "coordinator_baselines_20260917" / "heldout" / "streams" / f"s{seed}.json",
            ),
            output=str(scratch / f"rp_s{seed}.jsonl"),
            decision_trace=str(scratch / f"tr_s{seed}.jsonl"),
            num_goals=6,
            deadline_tightness=(0.5, 2.0),
        )
        build_joint_run(configuration).run()
        produced = _decision_rows(scratch / f"tr_s{seed}.jsonl")
        committed = _decision_rows(COMMITTED_TR / f"tr_coalplan0.25sticky_s{seed}.jsonl")
        same = produced == committed
        ok = ok and same
        print(
            f"identity seed {seed}: {len(produced)} rows vs {len(committed)} committed -> "
            f"{'IDENTICAL' if same else 'DIVERGED'}",
            flush=True,
        )
    print(f"switches-off still reproduces the 868d68a-era trajectory: {ok}")
    return 0 if ok else 1


def _decision_rows(path: Path) -> list[tuple]:
    return [
        tuple(row.get(name) for name in FIELDS)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and (row := json.loads(line)).get("event_type") == "decision"
    ]


def preflight() -> list[str]:
    problems: list[str] = []
    for condition, scenario in CONDITIONS.items():
        index = json.loads(
            (HERE / "stress_streams" / condition / "index.json").read_text(encoding="utf-8")
        )
        for seed in seeds_for(condition):
            for arm in ARMS:
                rp = OUT / condition / f"{arm}_rp_s{seed}.jsonl"
                metrics = OUT / condition / f"{arm}_s{seed}.json"
                if not rp.exists() or not metrics.exists():
                    problems.append(f"missing {rp.name}")
                    continue
                head = json.loads(rp.read_text(encoding="utf-8").splitlines()[0])
                signature = head["detail"].get("run_signature") or {}
                if not str(signature.get("code_commit", "")).startswith(FROZEN_COMMIT):
                    problems.append(f"{rp.name}: commit {signature.get('code_commit')}")
                configuration = head["detail"]["configuration"]
                if int(configuration["seed"]) != seed:
                    problems.append(f"{rp.name}: seed mismatch")
                if int(configuration.get("num_goals", -1)) != scenario["num_goals"]:
                    problems.append(f"{rp.name}: num_goals {configuration.get('num_goals')}")
                if [float(v) for v in configuration.get("deadline_tightness", [])] != [
                    float(v) for v in scenario["deadline_tightness"]
                ]:
                    problems.append(f"{rp.name}: deadline band mismatch")
                digest = hashlib.sha256(stream_path(condition, seed).read_bytes()).hexdigest()
                if digest != index[str(seed)]["sha256"]:
                    problems.append(f"{rp.name}: stream digest differs from the recorded index")
                if json.loads(metrics.read_text(encoding="utf-8")).get("stream_sha256") != digest:
                    problems.append(f"{rp.name}: metric stream digest mismatch")
    return problems


def bootstrap_ci(differences: np.ndarray, level: float) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, differences.size, size=(BOOTSTRAP_RESAMPLES, differences.size))
    means = differences[draws].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(np.percentile(means, 100 * tail)), float(np.percentile(means, 100 * (1 - tail)))


def aggregate() -> int:
    rows = {arm: {} for arm in ARMS}
    for path in sorted(OUT.glob("*/")):
        condition = path.name.rstrip("/")
        for arm in ARMS:
            for file in path.glob(f"{arm}_s*.json"):
                row = json.loads(file.read_text(encoding="utf-8"))
                rows[arm][(condition, int(row["seed"]))] = row

    keys = sorted(set(rows["baseline"]) & set(rows["candidate"]))
    missing = sorted(set(rows["baseline"]) ^ set(rows["candidate"]))
    adjusted_level = 1.0 - 0.05 / len(CONDITIONS)
    report: dict = {
        "frozen_commit": FROZEN_COMMIT,
        "paired_scenarios": len(keys),
        "unpaired": [list(key) for key in missing],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_rng_seed": BOOTSTRAP_SEED,
        "bonferroni_level": round(adjusted_level, 6),
        "conditions": {},
    }

    def outcomes(row: dict) -> tuple[int, int]:
        """(detected targets, targets finished by their own deadline)."""
        targets = row["targets"]
        return len(targets), sum(1 for entry in targets.values() if entry["on_time"])

    for condition in CONDITIONS:
        pairs = [key for key in keys if key[0] == condition]
        zero = [
            key[1] for key in pairs
            if outcomes(rows["baseline"][key])[0] == 0 and outcomes(rows["candidate"][key])[0] == 0
        ]
        used = [key for key in pairs if key[1] not in zero]
        differences = np.array(
            [
                outcomes(rows["candidate"][key])[1] / max(1, outcomes(rows["candidate"][key])[0])
                - outcomes(rows["baseline"][key])[1] / max(1, outcomes(rows["baseline"][key])[0])
                for key in used
            ],
            dtype=float,
        )
        totals = {
            arm: {
                "detections": sum(outcomes(rows[arm][key])[0] for key in used),
                "on_time": sum(outcomes(rows[arm][key])[1] for key in used),
                "expired": int(sum(rows[arm][key]["expired"] for key in used)),
            }
            for arm in ARMS
        }
        better = sum(1 for value in differences if value > 0)
        worse = sum(1 for value in differences if value < 0)
        regressions = []
        for key in pairs:
            base = rows["baseline"][key]["targets"]
            cand = rows["candidate"][key]["targets"]
            for target in set(base) & set(cand):
                if base[target]["on_time"] and not cand[target]["on_time"]:
                    regressions.append(f"{key[1]}/{target}")
        rescue_count = sum(
            1
            for key in pairs
            for target in set(rows["baseline"][key]["targets"]) & set(rows["candidate"][key]["targets"])
            if rows["candidate"][key]["targets"][target]["on_time"]
            and not rows["baseline"][key]["targets"][target]["on_time"]
        )
        ceiling = min(
            totals["baseline"]["on_time"] / max(1, totals["baseline"]["detections"]),
            totals["candidate"]["on_time"] / max(1, totals["candidate"]["detections"]),
        ) > 0.98
        report["conditions"][condition] = {
            "scenario": CONDITIONS[condition],
            "seeds": len(pairs),
            "excluded_zero_detection": zero,
            "paired_seeds_analysed": len(used),
            "closure_ok": len(used) + len(zero) == len(pairs) and not missing,
            "totals": totals,
            "mean_paired_difference": round(float(differences.mean()), 6) if differences.size else None,
            "ci95": [round(value, 6) for value in bootstrap_ci(differences, 0.95)] if differences.size else None,
            "ci_bonferroni": [
                round(value, 6) for value in bootstrap_ci(differences, adjusted_level)
            ] if differences.size else None,
            "seeds_candidate_better": better,
            "seeds_candidate_worse": worse,
            "ceiling_effect": bool(ceiling),
            "secondary": {
                metric: {
                    arm: round(float(np.mean([rows[arm][key][metric] for key in used])), 4)
                    for arm in ARMS
                }
                for metric in (
                    "coverage_resets", "lost_work", "redundant_working_ticks", "releases_planned",
                    "replan_attempts", "accepted_reassignments", "replan_declined", "same_task_noop",
                    "recommands", "rejected_assignments", "executed_override_unjustified",
                    "longest_idle_streak", "rescue_travel",
                )
            },
            "decision_milliseconds": {
                arm: {
                    statistic: round(
                        float(np.mean([rows[arm][key]["decision_milliseconds"][statistic] for key in used])), 3
                    )
                    for statistic in ("mean", "p50", "p95", "p99")
                }
                for arm in ARMS
            },
            "candidate_new_failures": sorted(regressions),
            "candidate_rescued_targets": rescue_count,
        }
        print(
            f"[{condition}] paired={len(used)}+{len(zero)} excluded  baseline_on_time="
            f"{totals['baseline']['on_time']}/{totals['baseline']['detections']} "
            f"candidate_on_time={totals['candidate']['on_time']}/{totals['candidate']['detections']} "
            f"mean_diff={report['conditions'][condition]['mean_paired_difference']} "
            f"ci_bonf={report['conditions'][condition]['ci_bonferroni']} "
            f"better={better} worse={worse} new_failures={len(regressions)}",
            flush=True,
        )
    part = (HERE / "STRESS_RESULTS.json").with_suffix(".json.part")
    part.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(part, HERE / "STRESS_RESULTS.json")
    return 0


def main(argv: list[str]) -> int:
    if "--identity" in argv:
        return identity_check()
    if "--preflight" in argv:
        problems = preflight()
        print(f"preflight problems: {len(problems)}")
        for problem in problems[:20]:
            print("   ", problem)
        return 1 if problems else 0
    if "--aggregate" in argv:
        return aggregate()

    jobs = [(condition, seed, arm) for condition in CONDITIONS for seed in seeds_for(condition) for arm in ARMS]
    if "--worker" in argv:
        index = int(argv[argv.index("--worker") + 1])
        total = int(argv[argv.index("--workers") + 1])
        for position, (condition, seed, arm) in enumerate(jobs):
            if position % total != index:
                continue
            destination = OUT / condition / f"{arm}_s{seed}.json"
            if destination.exists():
                continue
            row = run_pair(condition, seed, arm)
            print(
                f"[{condition}/{arm} s{seed}] completed={row['completed']} expired={row['expired']} "
                f"resets={row['coverage_resets']} lost={row['lost_work']} releases={row['releases_planned']} "
                f"rejected={row['rejected_assignments']} idle={row['longest_idle_streak']:.0f}",
                flush=True,
            )
        return 0

    processes = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(i), "--workers", str(WORKERS)],
            cwd=str(REPO),
        )
        for i in range(WORKERS)
    ]
    codes = [process.wait() for process in processes]
    if any(codes):
        print(f"worker failures: {codes}")
        return 1
    problems = preflight()
    print(f"preflight problems: {len(problems)}")
    for problem in problems[:20]:
        print("   ", problem)
    if problems:
        return 1
    return aggregate()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
