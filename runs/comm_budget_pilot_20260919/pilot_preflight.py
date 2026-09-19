"""§1: prove what the pilot actually runs, and re-read the gates the design left open.

Two things get checked here rather than assumed:

* the training environment is the corrected rescue environment -- the default-off
  compatibility path is what the rest of the repository was measured on, and a pilot run on it
  would be measuring an environment this project has already judged wrong;
* the allocation side is the frozen C arm, point for point: same coordinator object, same
  binding commitments, same residual scale and intervals, same fleet, same deadline band, same
  checkpoint bytes. Nothing is allowed to change alongside the communication policy.

It then prints the design document's three training-entry gates with the status of each,
because the honest answer to "is there enough evidence to train" is a list, not a mood.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import pilot_common as common

from integrated_sar.coordinator import CoalitionPlannerCoordinator  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402

EXPECTED_ALLOCATION = {
    "coordinator_policy": "coalition_planner",
    "sticky_commitments": True,
    "release_redundant_members": False,
    "suppress_recommand_own_task": False,
    "check_release_source_stage": False,
    "task_membership_consistency": True,
    "residual_scale": 0.25,
    "coordination_interval": 5,
    "allocation_interval": 5,
    "command_ttl": 10.0,
    "dt": 1.0,
    "steps": 1000,
    "search_agents": 2,
    "num_goals": 6,
    "grid_size": 60,
    "obstacle_density": 0.15,
    "detection_radius": 5,
    "rescue_species": 3,
    "rescue_per_species": 2,
    "actor_kv_mode": "tokens",
    "deadline_tightness": (0.5, 2.0),
}

EXPECTED_COMMUNICATION_DEFAULTS = {
    "comm_policy": "immediate",
    "comm_policy_params": "",
    "comm_sends_per_tick": None,
    "comm_budget_bytes": None,
    "comm_transcript": None,
    "comm_range": 0.3,
    "gossip_interval": 2,
    "comm_delay_ticks": 0,
    "comm_delivery_probability": 1.0,
    "comm_subscribers": "all",
}

GATES = (
    {
        "gate": "场景可分辨",
        "requirement": (
            "under the pre-registered budget, the published rules separate on on-time "
            "completions with a paired CI that excludes zero"
        ),
        "status_at_preflight": (
            "open -- last round's unbudgeted dev set separated nothing (best-minus-reference "
            "0.25 targets/episode, CI lower bound 0); the budget tiers are what makes this "
            "checkable"
        ),
    },
    {
        "gate": "信息确有后果",
        "requirement": (
            "withholding or delaying a message changes receiver belief and task outcomes, "
            "with the environment's own sensing accounted for"
        ),
        "status_at_preflight": (
            "met in the isolated setting (belief age 70.0 vs 4.2 steps, requirement error "
            "14776 vs 0 robot-target-steps, silence informs nobody); only partially met in "
            "the deployed setting, where line-of-sight sensing supplies 95.6% of first "
            "knowledge anyway"
        ),
    },
    {
        "gate": "旁路已说明并受控",
        "requirement": (
            "no claim of communication value is made in a setting where the delayed "
            "information can still reach the receiver by another path"
        ),
        "status_at_preflight": (
            "met -- training and mechanism checks run in the isolated setting, the deployed "
            "setting is reported separately, and arrival verification is named as the one "
            "channel no send policy can replace"
        ),
    },
)


def check(seed: int, directory: Path) -> dict[str, Any]:
    values = common.configuration_for(seed, directory=directory, label="preflight")
    configuration = RunConfiguration(**values)
    configuration.validate()
    mismatches = []
    for name, expected in EXPECTED_ALLOCATION.items():
        found = getattr(configuration, name)
        if found != expected:
            mismatches.append(f"{name}: {found!r} != {expected!r}")
    # The transcript and the replay stream are this round's own settings, not allocation
    # parameters; everything else must equal the frozen arm exactly.
    as_built = {field.name: getattr(configuration, field.name) for field in dataclasses.fields(configuration)}
    run = build_joint_run(configuration)
    try:
        env = run.task_adapter.env
        relay = run.task_adapter.comm_relay
        runtime = {
            "environment_membership_consistency": bool(env.membership_consistency),
            "coordinator": type(run.orchestrator.coordinator).__name__,
            "coordinator_is_planner": isinstance(
                run.orchestrator.coordinator, CoalitionPlannerCoordinator
            )
            or isinstance(
                getattr(run.orchestrator.coordinator, "inner", None), CoalitionPlannerCoordinator
            ),
            "binding_commitments": bool(run.task_adapter.binding_commitments),
            "release_flags": [
                bool(run.task_adapter.suppress_recommand_own_task),
                bool(run.gate.release_roster_of is not None),
            ],
            "robots": len(env.agent_dic),
            "robot_abilities": sorted(
                {tuple(int(value) for value in agent["abilities"]) for agent in env.agent_dic.values()}
            ),
            "species": sorted({int(agent["species"]) for agent in env.agent_dic.values()}),
            "deadline_tightness_runtime": [
                float(value) for value in env.ranges["deadline_tightness"]
            ],
            "relay": relay.transport_parameters(),
            "arm_label": run.run_signature["arm"],
            "config_hash": run.run_signature["config_hash"],
            "code_commit": run.run_signature["code_commit"],
            "stream_sha256": run.run_signature["stream"]["sha256"],
            "environment_module": type(env).__module__,
        }
    finally:
        pass
    if not runtime["environment_membership_consistency"]:
        mismatches.append("the rescue environment is running the OLD bookkeeping")
    if not runtime["coordinator_is_planner"]:
        mismatches.append(f"coordinator is {runtime['coordinator']}, not the coalition planner")
    if tuple(runtime["deadline_tightness_runtime"]) != (0.5, 2.0):
        mismatches.append(f"deadline band {runtime['deadline_tightness_runtime']} != (0.5, 2.0)")
    if runtime["robots"] != 6:
        mismatches.append(f"fleet size {runtime['robots']} != 6")
    if runtime["release_flags"] != [False, False]:
        mismatches.append(f"release mechanisms are active: {runtime['release_flags']}")
    if "+memfix" not in runtime["arm_label"]:
        mismatches.append(f"arm label lacks the corrected-environment marker: {runtime['arm_label']}")
    if "+comm" in runtime["arm_label"] or "+bud" in runtime["arm_label"]:
        mismatches.append(f"preflight must be the unbudgeted default: {runtime['arm_label']}")
    if relay.policy.name != "immediate" or relay.transport_parameters()["budget"]["bytes_per_episode"] is not None:
        mismatches.append("the pilot's default must be the unbudgeted historical uplink")
    checkpoint_hashes = {
        name: common.sha256(Path(path)) for name, path in common.CHECKPOINTS.items()
    }
    for name, digest in checkpoint_hashes.items():
        if digest != common.CHECKPOINT_SHA256[name]:
            mismatches.append(f"{name} checkpoint bytes changed: {digest[:12]}")
    return {
        "seed": seed,
        "configuration": {
            # Read from the built RunConfiguration, not from the dict that asked for it: a
            # field the arm table leaves to its default is still a field the run uses.
            name: as_built[name]
            for name in sorted({*EXPECTED_ALLOCATION, *EXPECTED_COMMUNICATION_DEFAULTS})
        },
        "expected_allocation": EXPECTED_ALLOCATION,
        "expected_communication_defaults": EXPECTED_COMMUNICATION_DEFAULTS,
        "runtime": runtime,
        "mismatches": mismatches,
        "gates": list(GATES),
        "what_the_pilot_controls": {
            "channel": "A: search-side registry to rescue belief, CommRelay uplink",
            "decided": (
                "whether the message currently held for a target goes on the air this step -- "
                "covering both a first detection and a later task-state view, because both "
                "travel as the same DetectionMessage contract"
            ),
            "not_decided": (
                "payload fields, recipients, merge rule, allocator, coordinator, fleet, "
                "deadline, sensing"
            ),
            "sender_side_truth_is_legitimate_because": (
                "ARCHITECTURE.md names CommRelay the only channel from the registry side into "
                "rescue belief, and the registry is the ground station's own state; the "
                "receiver still learns nothing it was not sent"
            ),
            "bypasses": [
                "TaskEnv line-of-sight refresh (deployed setting only; switched off for the pilot)",
                "one-hop gossip (same switch)",
                "verify_task_on_arrival, which is physical presence rather than a message",
                "assignment-gate refusals feeding replan_agents, which carry no task state",
            ],
        },
    }


def main(argv: list[str]) -> int:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=common.TRAIN_SEEDS[0])
    parser.add_argument("--out", default=str(common.HERE / "PILOT_PREFLIGHT.json"))
    args = parser.parse_args(argv)
    directory = common.HERE / "preflight"
    directory.mkdir(parents=True, exist_ok=True)
    if not common.stream_path(args.seed).exists():
        print(f"stream for seed {args.seed} is not recorded yet; run record_streams.py first")
        return 2
    result = check(args.seed, directory)
    common.write_json(Path(args.out), result)
    print(json.dumps({"mismatches": result["mismatches"], "runtime": result["runtime"]},
                     ensure_ascii=False, indent=1, default=str))
    for gate in result["gates"]:
        print(f"- {gate['gate']}: {gate['status_at_preflight']}")
    return 1 if result["mismatches"] else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
