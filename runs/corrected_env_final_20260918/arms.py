"""The two frozen arms, as construction facts rather than as names.

An arm label is a claim. Two earlier rounds of this project produced artefacts whose file
names said one arm and whose code ran another, which is why the run signature exists. This
module adds the other half for the corrected-environment block: the truth table each arm must
have, and an assertion that the objects actually built match it -- the environment variant
that is really instantiated, the callbacks really wired into the gate, the flags really set on
the coordinator and the allocator. A CLI flag or an arm string that disagrees with the built
object stops the run before it produces a single row.

    D:/project_env/python.exe -c "import sys; sys.path.insert(0,'runs/corrected_env_final_20260918'); import arms"
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from integrated_sar.assignment_gate import AssignmentGate, ReplanCoordinator  # noqa: E402
from integrated_sar.coordinator import CoalitionPlannerCoordinator  # noqa: E402
from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402

PIL_CHECKPOINT = REPO / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth"
CTDE_CHECKPOINT = REPO / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth"
STEPS = 1000
WORKERS = 6
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260918
STREAMS = HERE / "streams"
OUT = HERE / "block"
FREEZE_FILE = HERE / "FREEZE.json"


def freeze_commit() -> str:
    """The commit the block is frozen at, read from FREEZE.json -- never typed by hand."""
    if not FREEZE_FILE.exists():
        raise AssertionError(
            "no FREEZE.json: run freeze.py to register the commit before building an arm"
        )
    return json.loads(FREEZE_FILE.read_text(encoding="utf-8"))["code_commit"]

# The frozen facts. Any change to one of these ends the acceptance run of this version.
FROZEN = {
    "branch": "fix/task-membership-consistency",
    "piloc_checkpoint": str(PIL_CHECKPOINT.relative_to(REPO)),
    "ctde_checkpoint": str(CTDE_CHECKPOINT.relative_to(REPO)),
    "steps": STEPS,
    "dt": 1.0,
    "grid_size": 60,
    "obstacle_density": 0.15,
    "detection_radius": 5,
    "search_agents": 2,
    "num_goals": 6,
    "deadline_tightness": [0.5, 2.0],
    "residual_scale": 0.25,
    "allocation_interval": 5,
    "coordination_interval": 5,
    "command_ttl": 10.0,
    "communication": {
        "comm_delivery_probability": 1.0,
        "comm_delay_ticks": 0,
        "comm_subscribers": "all",
    },
    "device": "cpu",
    "thread_environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    "trace_schema": "allocation-decision-trace/3",
    "recording_policy": "live PILOC search, coordinator off, membership-consistent environment",
}


@dataclass(frozen=True, slots=True)
class Arm:
    """One arm of the block: its configuration, and what must be true of the built run."""

    label: str
    description: str
    configuration: dict[str, Any]
    expects: dict[str, Any]

    def build(self, seed: int, stream: Path, directory: Path) -> Any:
        directory.mkdir(parents=True, exist_ok=True)
        values = dict(self.configuration)
        values.update(
            seed=seed,
            replay_stream=str(stream),
            output=str(directory / f"rp_{self.label}_s{seed}.jsonl"),
            decision_trace=str(directory / f"tr_{self.label}_s{seed}.jsonl"),
        )
        run = build_joint_run(RunConfiguration(**values))
        assert_built(run, self)
        return run


def _base() -> dict[str, Any]:
    return {
        "steps": STEPS,
        "device": "cpu",
        # Only the allocation network: a replay stream already carries the detections of the
        # recorded search policy, and the configuration refuses the two together.
        "task_checkpoint": str(CTDE_CHECKPOINT),
        "mock": False,
        "use_coordinator": True,
        "coordinator_policy": "coalition_planner",
        "sticky_commitments": True,
        "residual_scale": 0.25,
        "allocation_interval": 5,
        "coordination_interval": 5,
        "command_ttl": 10.0,
        "dt": 1.0,
        "detection_radius": 5,
        "search_agents": 2,
        "num_goals": 6,
        "deadline_tightness": (0.5, 2.0),
        "grid_size": 60,
        "obstacle_density": 0.15,
        "rescue_species": 3,
        "rescue_per_species": 2,
        "actor_kv_mode": "tokens",
        "comm_delivery_probability": 1.0,
        "comm_delay_ticks": 0,
        "comm_subscribers": "all",
        "quiet": True,
        # Both arms run on the corrected environment: the block measures controllers, and an
        # environment that credits absent robots would measure the residue as well.
        "task_membership_consistency": True,
    }


C = Arm(
    label="C",
    description="corrected environment + binding coalition planner, no release mechanism",
    configuration=_base() | {
        "release_redundant_members": False,
        "suppress_recommand_own_task": False,
        "check_release_source_stage": False,
    },
    expects={
        "task_membership_consistency": True,
        "release_redundant_members": False,
        "suppress_recommand_own_task": False,
        "renew_commitments": True,
        "gate_release_callbacks": False,
        "arm_suffixes": ["+memfix"],
        "arm_absent_suffixes": ["+redrel", "+noself", "+srccheck"],
    },
)

D = Arm(
    label="D",
    description=(
        "corrected environment + binding planner + coverage-preserving redundant release + "
        "own-task suppression + source stage and support checks at the execution boundary"
    ),
    configuration=_base() | {
        "release_redundant_members": True,
        "suppress_recommand_own_task": True,
        "check_release_source_stage": True,
    },
    expects={
        "task_membership_consistency": True,
        "release_redundant_members": True,
        "suppress_recommand_own_task": True,
        "renew_commitments": True,
        "gate_release_callbacks": True,
        "arm_suffixes": ["+redrel", "+noself", "+srccheck", "+memfix"],
        "arm_absent_suffixes": [],
    },
)

ARMS = {arm.label: arm for arm in (C, D)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_HEAD: str | None = None


def current_head() -> str:
    """The commit the artefacts must name. The freeze check is about the files, not this."""
    global _HEAD
    if _HEAD is None:
        import subprocess

        _HEAD = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
    return _HEAD


def unwrapped_coordinator(run: Any) -> CoalitionPlannerCoordinator:
    coordinator = run.orchestrator.coordinator
    while isinstance(coordinator, ReplanCoordinator):
        coordinator = coordinator.inner
    if not isinstance(coordinator, CoalitionPlannerCoordinator):
        raise AssertionError(f"the built coordinator is {type(coordinator).__name__}, not the planner")
    return coordinator


def assert_built(run: Any, arm: Arm) -> None:
    """Fail loudly when the object graph disagrees with the arm's truth table."""
    expects = arm.expects
    env = run.task_adapter.env
    if bool(getattr(env, "membership_consistency", False)) is not bool(
        expects["task_membership_consistency"]
    ):
        raise AssertionError(
            f"arm {arm.label}: the environment actually built has "
            f"membership_consistency={getattr(env, 'membership_consistency', None)!r}"
        )
    from integrated_sar.task03_runtime import load_task_env_module

    # The loader execs the file under its own module name, so class identity across two
    # loader calls is not a usable test. Name and source file are.
    module = load_task_env_module()
    if (
        type(env).__name__ != "TaskEnv"
        or type(env).__module__ != str(module.__name__)
        or Path(str(module.__file__)).name != "old_task_env.py"
    ):
        raise AssertionError(
            f"arm {arm.label}: the environment is {type(env).__module__}.{type(env).__name__} "
            f"from {module.__file__}, expected TaskEnv from old_task_env.py"
        )
    coordinator = unwrapped_coordinator(run)
    holders = {
        "release_redundant_members": coordinator,
        "renew_commitments": coordinator,
        # The allocator, not the coordinator, is the object that withholds a command sending
        # a robot back onto its own task.
        "suppress_recommand_own_task": run.task_adapter,
    }
    for field, carrier in holders.items():
        actual = bool(getattr(carrier, field, False))
        if actual is not bool(expects[field]):
            raise AssertionError(
                f"arm {arm.label}: {field} is {actual!r} on the built object, "
                f"the truth table says {expects[field]!r}"
            )
    gate = run.gate
    if not isinstance(gate, AssignmentGate):
        raise AssertionError(f"arm {arm.label}: the executor is not wrapped by the gate")
    wired = gate.release_state_of is not None or gate.release_roster_of is not None
    if bool(wired) is not bool(expects["gate_release_callbacks"]):
        raise AssertionError(
            f"arm {arm.label}: release screening wired={wired!r} but the truth table says "
            f"{expects['gate_release_callbacks']!r}"
        )
    if expects["gate_release_callbacks"] and not (
        gate.release_state_of is not None and gate.release_roster_of is not None
    ):
        raise AssertionError(
            f"arm {arm.label}: the source check needs both the stage and the roster callback"
        )
    signature = run.run_signature
    if signature.get("code_commit") != current_head():
        raise AssertionError(
            f"arm {arm.label}: the artefact names {signature.get('code_commit')!r} but the "
            f"tree it was built from is {current_head()!r}"
        )
    for suffix in expects["arm_suffixes"]:
        if suffix not in signature["arm"]:
            raise AssertionError(f"arm {arm.label}: signature arm {signature['arm']!r} lacks {suffix}")
    for suffix in expects["arm_absent_suffixes"]:
        if suffix in signature["arm"]:
            raise AssertionError(f"arm {arm.label}: signature arm {signature['arm']!r} carries {suffix}")
    for checkpoint in (PIL_CHECKPOINT, CTDE_CHECKPOINT):
        if not checkpoint.exists():
            raise AssertionError(f"arm {arm.label}: checkpoint {checkpoint} is missing")


def assert_frozen_tree() -> str:
    """The freeze is only real while the production tree still matches the commit it names.

    Compared against the freeze commit, not against HEAD: the protocol that names the freeze
    is committed after the code it describes, so a documentation commit on top is expected and
    a production edit on top is not.
    """
    import subprocess

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()

    dirty = git(
        "status", "--porcelain", "integrated_sar", "task_allocation",
        ":(exclude)integrated_sar/tests",
    )
    if dirty:
        raise AssertionError(
            "the production tree has uncommitted changes, so the block may not be run: "
            + dirty.replace("\n", "; ")
        )
    commit = freeze_commit()
    drift = git(
        "diff", commit, "--", "integrated_sar", "task_allocation",
        ":(exclude)integrated_sar/tests",
    )
    if drift:
        raise AssertionError(
            f"the production files differ from the freeze commit {commit}: "
            f"{len(drift.splitlines())} diff lines"
        )
    return commit


def truth_table() -> dict[str, Any]:
    """The two arms side by side, field by field, for the protocol document."""
    keys = sorted(set(ARMS["C"].configuration) | set(ARMS["D"].configuration))
    return {
        "frozen": {**FROZEN, "code_commit": freeze_commit()},
        "checkpoint_sha256": {
            "piloc": sha256(PIL_CHECKPOINT),
            "ctde": sha256(CTDE_CHECKPOINT),
        },
        "fields": {
            key: {"C": ARMS["C"].configuration.get(key), "D": ARMS["D"].configuration.get(key)}
            for key in keys
        },
        "construction_expectations": {
            label: {key: value for key, value in arm.expects.items()} for label, arm in ARMS.items()
        },
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(truth_table(), ensure_ascii=False, indent=1))
