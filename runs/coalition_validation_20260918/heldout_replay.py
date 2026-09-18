"""Held-out replay of the two frozen arms (brief section four).

Arms, on one frozen commit, replaying byte-identical streams per seed:

  baseline   coalition_planner + sticky commitments, both new switches off
  candidate  the same planner + redundant-member release + own-task suppression

Before any held-out run, ``--identity`` proves the all-switches-off configuration still
reproduces the committed 868d68a-era trajectory on the five development seeds, field by
field, from the decision trace. The preflight then checks every held-out run's
run_signature (commit, arm, seed) and the stream digest against the recorded index.

    D:/project_env/python.exe runs/coalition_validation_20260918/heldout_replay.py --identity
    D:/project_env/python.exe runs/coalition_validation_20260918/heldout_replay.py
    ... heldout_replay.py --worker 3 --workers 8
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Same reasoning as heldout_record: sharded workers each defaulting to a 20-thread
# torch pool oversubscribe the box badly; single-threaded inference is faster in
# aggregate and identical within a process.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
try:
    import torch

    torch.set_num_threads(1)
except Exception:  # torch is optional in dependency-free environments
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from integrated_sar.expired_attribution import build_timelines  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402
from integrated_sar.lifecycle_metrics import load_records  # noqa: E402

import mechanism_ablation as ablation  # noqa: E402  (metric definitions are shared)
import heldout_record  # noqa: E402

FROZEN_COMMIT = "304fc60"  # full: 304fc6044218f8032d708cbef0f19cf640dc4add
BASELINE = {"policy": "coalition_planner", "release": False, "suppress": False}
CANDIDATE = {"policy": "coalition_planner", "release": True, "suppress": True}
ARMS = {"baseline": BASELINE, "candidate": CANDIDATE}
OUT = HERE / "heldout"
DEV_SEEDS = [20003, 20008, 20052, 20070, 20083]
COMMITTED_TR = (
    REPO / "runs" / "coordinator_baselines_20260917" / "heldout" / "arms" / "coalplan0.25sticky"
)
FIELDS = (
    "step", "agent_id", "chosen_task", "actual_assignment", "needs_decision",
    "raw_choice", "post_mask_choice", "chosen_index",
)


def configuration_for(seed: int, arm: str, stream: Path) -> RunConfiguration:
    spec = ARMS[arm]
    OUT.mkdir(parents=True, exist_ok=True)
    return RunConfiguration(
        seed=seed,
        steps=1000,
        device="cpu",
        output=str(OUT / f"{arm}_rp_s{seed}.jsonl"),
        decision_trace=str(OUT / f"{arm}_tr_s{seed}.jsonl"),
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
        num_goals=6,
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


def run_pair(seed: int, arm: str, stream: Path) -> dict:
    counters = {"releases_planned": 0, "redundant_working_ticks": 0, "working_ticks": 0}
    run = build_joint_run(configuration_for(seed, arm, stream))
    _instrument(run, counters)
    summary = run.run()
    rp = OUT / f"{arm}_rp_s{seed}.jsonl"
    trace = OUT / f"{arm}_tr_s{seed}.jsonl"
    metrics = ablation.measure(seed, arm, rp, trace, summary, counters)
    metrics["stream_sha256"] = heldout_record.sha256(stream)
    part = (OUT / f"{arm}_s{seed}.json").with_suffix(".json.part")
    part.write_text(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(part, OUT / f"{arm}_s{seed}.json")
    return metrics


def _instrument(run, counters: dict) -> None:
    env = run.task_adapter.env
    original_step = run.orchestrator.step

    def step():
        report = original_step()
        abilities = {str(aid): np.array(agent["abilities"], dtype=float) for aid, agent in env.agent_dic.items()}
        for task in env.task_dic.values():
            members = [str(m) for m in task["members"]]
            index = min(int(task["current_stage_index"]), len(task["stages"]) - 1)
            requirement = np.array(task["stages"][index]["requirements"], dtype=float)
            for member in members:
                agent = env.agent_dic.get(int(member)) if member.isdigit() else None
                if agent is None or agent.get("status") != "working" or member not in abilities:
                    continue
                others = [abilities[m] for m in members if m != member and m in abilities]
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

    # A fixed scratch directory rather than a temp dir: Windows refuses to unlink an
    # output while a writer still holds it, and the teardown error hides the verdict.
    scratch = HERE / "identity_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    ok = True
    for seed in DEV_SEEDS:
        configuration = dataclasses.replace(
            configuration_for(
                seed,
                "baseline",
                REPO / "runs" / "coordinator_baselines_20260917" / "heldout" / "streams" / f"s{seed}.json",
            ),
            output=str(scratch / f"rp_s{seed}.jsonl"),
            decision_trace=str(scratch / f"tr_s{seed}.jsonl"),
        )
        run = build_joint_run(configuration)
        run.run()
        produced = [
            tuple(row.get(name) for name in FIELDS)
            for line in (scratch / f"tr_s{seed}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() and (row := json.loads(line)).get("event_type") == "decision"
        ]
        committed = [
            tuple(row.get(name) for name in FIELDS)
            for line in (COMMITTED_TR / f"tr_coalplan0.25sticky_s{seed}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() and (row := json.loads(line)).get("event_type") == "decision"
        ]
        same = produced == committed
        ok = ok and same
        print(f"identity seed {seed}: {len(produced)} rows vs {len(committed)} committed -> {'IDENTICAL' if same else 'DIVERGED'}", flush=True)
    print(f"all-off reproduces the 868d68a-era trajectory: {ok}")
    return 0 if ok else 1


def preflight(seeds: list[int]) -> list[str]:
    import hashlib

    index = json.loads((HERE / "stream_index.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    for seed in seeds:
        for arm in ARMS:
            rp = OUT / f"{arm}_rp_s{seed}.jsonl"
            metrics = OUT / f"{arm}_s{seed}.json"
            if not rp.exists() or not metrics.exists():
                problems.append(f"missing {rp.name}")
                continue
            head = json.loads(rp.read_text(encoding="utf-8").splitlines()[0])
            signature = head["detail"].get("run_signature") or {}
            if not str(signature.get("code_commit", "")).startswith(FROZEN_COMMIT):
                problems.append(f"{rp.name}: commit {signature.get('code_commit')}")
            if int(head["detail"]["configuration"]["seed"]) != seed:
                problems.append(f"{rp.name}: seed mismatch")
            digest = hashlib.sha256((HERE / "streams" / f"s{seed}.json").read_bytes()).hexdigest()
            if digest != index[str(seed)]["sha256"]:
                problems.append(f"{rp.name}: stream digest differs from the recorded index")
            if json.loads(metrics.read_text(encoding="utf-8")).get("stream_sha256") != digest:
                problems.append(f"{rp.name}: metric stream digest mismatch")
    return problems


def main(argv: list[str]) -> int:
    seeds = heldout_record.heldout_seeds()
    if "--identity" in argv:
        return identity_check()
    if "--worker" in argv:
        index = int(argv[argv.index("--worker") + 1])
        total = int(argv[argv.index("--workers") + 1])
        jobs = [(seed, arm) for seed in seeds for arm in ARMS]
        for position, (seed, arm) in enumerate(jobs):
            if position % total != index or (OUT / f"{arm}_s{seed}.json").exists():
                continue
            metrics = run_pair(seed, arm, HERE / "streams" / f"s{seed}.json")
            print(
                f"[{arm} s{seed}] completed={metrics['completed']} expired={metrics['expired']} "
                f"active={metrics['active']} redundant={metrics['redundant_working_ticks']} "
                f"releases={metrics['releases_planned']} recommands={metrics['recommands']}",
                flush=True,
            )
        return 0

    processes = [
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(i), "--workers", "8"], cwd=str(REPO))
        for i in range(8)
    ]
    codes = [p.wait() for p in processes]
    if any(codes):
        print(f"worker failures: {codes}")
        return 1
    problems = preflight(seeds)
    if problems:
        print("PREFLIGHT FAILED")
        for problem in problems[:20]:
            print("  ", problem)
        return 1
    print(f"preflight clean over {len(seeds)} seeds x {len(ARMS)} arms")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
