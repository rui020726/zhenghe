"""Joint inference driver: PILOC search -> registry -> coordination -> CTDE allocation.

This is the first entry point that runs the two mainline simulators against one shared
scene and one global clock. It performs inference only -- no network in this process is
ever trained, which keeps architecture invariant 8 (weights change only at rollout
boundaries) trivially satisfied.

Typical use::

    python -m integrated_sar.joint_driver --mock --steps 20 --seed 42
    python -m integrated_sar.joint_driver --piloc-checkpoint model/shared_mappo_model_21_9232.pth \
        --task-checkpoint checkpoints_old/7/1/ctde_best.pth --steps 200 --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters.piloc import PilocSearchAdapter
from .adapters.task03 import Task03Adapter
from .assignment_gate import AssignmentGate, ReplanCoordinator
from .clock import GlobalClock
from .comm import CommRelay
from .contracts import DetectionEvent, ProgressEvent, TaskSnapshot
from .coordinator import (
    CoalitionAwareCoordinator,
    CoalitionPlannerCoordinator,
    DeadlineAwareCoalitionCoordinator,
    HeuristicCoordinator,
    PassthroughCoordinator,
)
from .detection_replay import scenario_map_sha256
from .errors import ConfigurationError, IntegratedSARError
from .event_bus import InMemoryEventBus
from .lifecycle_log import LifecycleLogger
from .orchestrator import JointSAROrchestrator
from .profiles import DeterministicTargetProfileProvider
from .scenarios import PILOC_GRID_SIZE, PILOC_OBS_RADIUS, SearchScenario, build_scenario
from .search_runtime import FrontierSearchPolicy, build_mock_search_side
from .task_registry import TaskRegistry

DEFAULT_OUTPUT = "joint_lifecycle.jsonl"


@dataclass(frozen=True, slots=True)
class RunConfiguration:
    """Everything a run depends on, in a form that can be serialised into the log."""

    seed: int = 42
    steps: int = 20
    device: str = "cpu"
    output: str = DEFAULT_OUTPUT
    mock: bool = False
    piloc_checkpoint: str | None = None
    task_checkpoint: str | None = None
    map_path: str | None = None
    grid_size: int = PILOC_GRID_SIZE
    obstacle_density: float = 0.15
    search_agents: int = 2
    num_goals: int = 6
    rescue_species: int = 3
    rescue_per_species: int = 2
    # The band TaskEnv draws each task's deadline multiplier from, as (low, high)
    # multiples of the estimated travel-and-work time. The default is TaskEnv's own, so
    # leaving it unset reproduces every scenario recorded before this field existed;
    # lowering the upper bound is how a run asks for less slack, not how the sampler
    # itself changes.
    deadline_tightness: tuple[float, float] = (0.5, 2.0)
    actor_kv_mode: str = "tokens"
    use_coordinator: bool = True
    residual_scale: float = 0.5
    # Scalar calibration of the CTDE task/depot preference, in logit units: added to
    # every legal task action before the coordinator residual, never to the depot and
    # never to a masked action. Zero keeps a run bit-for-bit identical to one from
    # before this parameter existed.
    ctde_task_logit_offset: float = 0.0
    # Which residual policy produces the coordination advice. "heuristic" is the
    # frozen v3 scoring function and the default; "coalition" scores each robot by
    # the share of a task's *uncovered* requirement it would add, so the ablations
    # can tell "a better bias" apart from "the same bias, better scaled".
    coordinator_policy: str = "heuristic"
    # Whether a commitment the advice still holds binds the allocator: a robot already
    # executing its committed task is not re-commanded, an idle one is pointed at the
    # committed task rather than whatever the residual would have picked, and the planner
    # keeps the commitment alive while the robot's own belief says the task is open. Off
    # by default, so the advisory arm stays what every earlier number was measured on.
    sticky_commitments: bool = False
    # Whether the planner may re-decide a working robot when taking it off its stage
    # would leave the believed stage requirement covered no worse. Off by default; the flag
    # only ever names a robot the plan also has somewhere else for.
    release_redundant_members: bool = False
    # Whether the allocator stays silent when a replan-woke robot re-picks the task it is
    # already executing. The release needs it (a re-issued command would restart the work
    # clock), but the two switches are independent so an ablation can tell the planner
    # mechanism apart from the suppression.
    suppress_recommand_own_task: bool = False
    # The deliberate exception: running the release without the suppression churns the
    # work clock by design, which is a measurement worth making but never a configuration
    # someone should reach for by accident.
    allow_release_without_suppression: bool = False
    command_ttl: float = 10.0
    coordination_interval: int = 5
    allocation_interval: int = 5
    dt: float = 1.0
    detection_radius: int = 5
    comm_delivery_probability: float = 1.0
    comm_delay_ticks: int = 0
    comm_subscribers: str = "all"
    quiet: bool = False
    repository_root: str | None = None
    # Detection-stream recording/replay. Exactly one of the two may be set: the
    # recording runs the frozen PILOC once and writes the ordered event stream;
    # the replay feeds that stream to every arm, so coordinator ablations compare
    # allocator behaviour on an identical search history.
    record_stream: str | None = None
    replay_stream: str | None = None
    # Write one line per allocation decision: the action space, the hard mask, the CTDE
    # logits, the coordinator residual, the choice, and the ground truth the decision is
    # later judged against. Off unless a path is given, and never read back by a
    # decision -- see integrated_sar.allocation_trace.
    decision_trace: str | None = None

    @property
    def search_backend(self) -> str:
        if self.replay_stream is not None:
            return "replayed-detections"
        return "mock-frontier" if self.piloc_checkpoint is None else "piloc-trained"

    @property
    def allocation_backend(self) -> str:
        return "heuristic" if self.task_checkpoint is None else "ctde-network"

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in self.__slots__:  # type: ignore[attr-defined]
            value = getattr(self, name)
            payload[name] = value if not isinstance(value, float) else round(value, 6)
        payload["search_backend"] = self.search_backend
        payload["allocation_backend"] = self.allocation_backend
        return payload

    def validate(self) -> None:
        if self.steps < 1:
            raise ConfigurationError(f"--steps must be at least 1, got {self.steps}")
        if self.dt <= 0:
            raise ConfigurationError(f"--dt must be positive, got {self.dt}")
        if self.search_agents < 1:
            raise ConfigurationError(f"--search-agents must be at least 1, got {self.search_agents}")
        if self.num_goals < 0:
            raise ConfigurationError(f"--num-goals cannot be negative, got {self.num_goals}")
        low, high = (float(value) for value in self.deadline_tightness)
        if low <= 0 or high <= 0 or low > high:
            raise ConfigurationError(
                f"--deadline-tightness must be two positive numbers with low <= high, "
                f"got {self.deadline_tightness}"
            )
        if self.detection_radius < 0:
            raise ConfigurationError(
                f"--detection-radius cannot be negative, got {self.detection_radius}; 0 is "
                f"allowed (a robot then only observes its own cell), but a negative radius "
                f"would build negative pheromone windows and crash deep inside numpy"
            )
        if self.grid_size != PILOC_GRID_SIZE and self.piloc_checkpoint is not None:
            raise ConfigurationError(
                f"the trained PILOC policy is a convolutional network built for "
                f"{PILOC_GRID_SIZE}x{PILOC_GRID_SIZE} timestamp maps, so --grid-size "
                f"{self.grid_size} cannot be used with --piloc-checkpoint"
            )
        if self.detection_radius != PILOC_OBS_RADIUS and self.piloc_checkpoint is not None:
            raise ConfigurationError(
                f"the trained PILOC policy derives its timestamp-map, pheromone and "
                f"last-action windows from obs_radius={PILOC_OBS_RADIUS}, so "
                f"--detection-radius {self.detection_radius} cannot be combined with "
                f"--piloc-checkpoint: the environment would remove targets on one radius "
                f"while the adapter detects them on another"
            )
        if self.dt != 1.0 and self.piloc_checkpoint is not None:
            raise ConfigurationError(
                f"timemap.Environment advances its own clock by exactly 1 per step, so "
                f"--dt {self.dt} would put the search environment on a different time "
                f"base from the global clock and the rescue environment; use --dt 1.0 "
                f"with --piloc-checkpoint"
            )
        if not 0.0 <= self.residual_scale <= 1.0:
            raise ConfigurationError(
                f"--residual-scale must stay in [0, 1] so the coordinator can only add a "
                f"bounded residual to the allocator logits, got {self.residual_scale}"
            )
        if not math.isfinite(self.ctde_task_logit_offset) or self.ctde_task_logit_offset < 0.0:
            raise ConfigurationError(
                f"--ctde-task-logit-offset must be a finite, non-negative number of "
                f"logit units (it is only ever added to legal task actions, never "
                f"subtracted from the depot), got {self.ctde_task_logit_offset}"
            )
        if self.command_ttl <= 0:
            raise ConfigurationError(f"--command-ttl must be positive, got {self.command_ttl}")
        if not 0.0 <= self.comm_delivery_probability <= 1.0:
            raise ConfigurationError(
                f"--comm-delivery-probability must stay in [0, 1] so the search-to-rescue "
                f"uplink stays a seedable packet-loss process, got {self.comm_delivery_probability}"
            )
        if self.comm_delay_ticks < 0:
            raise ConfigurationError(
                f"--comm-delay-ticks cannot be negative, got {self.comm_delay_ticks}"
            )
        # "all" broadcasts each detection to the whole rescue team; a count or an
        # explicit id list narrows the uplink to a subset, which is what a
        # restricted-topology experiment needs. Either way the link keeps its
        # delay and loss model, and robots outside the subset may still learn a
        # task later through sensing or gossip.
        subscribers = parse_comm_subscribers(self.comm_subscribers)
        if subscribers is not None and not subscribers:
            raise ConfigurationError(
                "--comm-subscribers must be 'all', 'first:N' with N >= 1, or a "
                f"comma-separated list of rescue robot ids, got {self.comm_subscribers!r}"
            )
        if self.coordination_interval < 1 or self.allocation_interval < 1:
            raise ConfigurationError("decision intervals must be at least 1")
        if self.mock and (self.piloc_checkpoint or self.task_checkpoint):
            raise ConfigurationError(
                "--mock runs the dependency-free heuristic pipeline and cannot be combined "
                "with --piloc-checkpoint or --task-checkpoint; drop --mock to use real weights"
            )
        if self.actor_kv_mode not in {"tokens", "summary"}:
            raise ConfigurationError(
                f"--actor-kv-mode must be 'tokens' or 'summary', got {self.actor_kv_mode!r}"
            )
        if self.coordinator_policy not in {
            "heuristic",
            "coalition",
            "coalition_planner",
            "coalition_assembly_scheduler",
            "coalition_redundant_release",
            "deadline_aware",
        }:
            raise ConfigurationError(
                f"--coordinator must be 'heuristic', 'coalition', 'coalition_planner', "
                f"'coalition_assembly_scheduler', 'coalition_redundant_release' or "
                f"'deadline_aware', got {self.coordinator_policy!r}"
            )
        if self.sticky_commitments and not self.use_coordinator:
            raise ConfigurationError(
                "--sticky-commitments needs a coordinator to hold the commitment; "
                "--no-coordinator removes the residual and with it anything to bind"
            )
        if self.sticky_commitments and self.coordinator_policy == "heuristic":
            raise ConfigurationError(
                "--sticky-commitments binds coalition reservations, but --coordinator "
                "heuristic never issues one; ask for --coordinator coalition_planner"
            )
        if self.release_redundant_members and self.coordinator_policy not in {
            "coalition_planner",
            "coalition_assembly_scheduler",
            "coalition_redundant_release",
        }:
            raise ConfigurationError(
                "--release-redundant-members re-decides working robots from the "
                "coalition planner's reservations, but --coordinator "
                f"{self.coordinator_policy} never issues one; ask for "
                "--coordinator coalition_redundant_release"
            )
        if self.suppress_recommand_own_task and not self.use_coordinator:
            raise ConfigurationError(
                "--suppress-recommand-own-task only ever applies to a decision a "
                "coordinator replan woke; --no-coordinator removes that trigger"
            )
        if self.release_redundant_members and not self.suppress_recommand_own_task \
                and self.coordinator_policy != "coalition_redundant_release" \
                and not self.allow_release_without_suppression:
            raise ConfigurationError(
                "--release-redundant-members without --suppress-recommand-own-task "
                "re-issues a command for a task the robot is already executing, which "
                "restarts its work clock and disarms the environment's working timeout. "
                "Use --coordinator coalition_redundant_release for the paired arm, or "
                "--allow-release-without-suppression when measuring that churn is the "
                "experiment"
            )
        if not self.use_coordinator and self.coordinator_policy != "heuristic":
            raise ConfigurationError(
                f"--no-coordinator already removes the residual, so asking for "
                f"--coordinator {self.coordinator_policy} would silently do nothing; "
                f"drop one of the two"
            )
        if self.record_stream is not None and self.replay_stream is not None:
            raise ConfigurationError(
                "--record-stream and --replay-stream are alternatives: a recording runs "
                "the live search side and writes the stream, a replay consumes it"
            )
        if self.replay_stream is not None and self.piloc_checkpoint is not None:
            raise ConfigurationError(
                f"--replay-stream already carries the detections of a trained PILOC "
                f"(recorded from {self.replay_stream}); combining it with "
                f"--piloc-checkpoint would run two search policies at once"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="integrated_sar.joint_driver",
        description="Run PILOC search and CTDE task allocation as one joint episode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--piloc-checkpoint", default=None,
                        help="PILOC search policy: a .pth file or a directory of shared_mappo_model_*.pth")
    parser.add_argument("--task-checkpoint", default=None,
                        help="CTDE allocation network: a .pth file or a checkpoint directory")
    parser.add_argument("--map", dest="map_path", default=None,
                        help="obstacle map (.png/.bmp/.npy/.csv/.txt); generated deterministically when omitted")
    parser.add_argument("--seed", type=int, default=42, help="seed for scenario, search and rescue environments")
    parser.add_argument("--steps", type=int, default=20, help="number of joint steps to run")
    parser.add_argument("--device", default="cpu", help="torch device for the checkpoints")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="JSONL lifecycle log path, or '-' to write to stdout")
    parser.add_argument("--mock", action="store_true",
                        help="force the dependency-free heuristic pipeline (no torch, no checkpoints)")
    parser.add_argument("--search-agents", type=int, default=2, help="number of PILOC search robots")
    parser.add_argument("--num-goals", type=int, default=6, help="number of hidden targets to search for")
    parser.add_argument("--grid-size", type=int, default=PILOC_GRID_SIZE, help="square search grid side length")
    parser.add_argument("--obstacle-density", type=float, default=0.15,
                        help="obstacle ratio used when no --map is given")
    parser.add_argument("--rescue-species", type=int, default=3, help="rescue robot species (ability profiles)")
    parser.add_argument("--rescue-per-species", type=int, default=2, help="rescue robots per species")
    parser.add_argument("--deadline-tightness", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        default=[0.5, 2.0],
                        help="multiplier band TaskEnv draws each deadline from; the default "
                             "is its own (0.5 2.0), lowering HIGH asks for less slack")
    parser.add_argument("--actor-kv-mode", default="tokens", choices=("tokens", "summary"),
                        help="CTDE actor key/value mode the checkpoint was trained with")
    parser.add_argument("--no-coordinator", action="store_true",
                        help="disable the coordinator residual (ablation baseline)")
    parser.add_argument("--residual-scale", type=float, default=0.5,
                        help="alpha in final_logits = ctde_logits + alpha * coordinator_bias")
    parser.add_argument("--ctde-task-logit-offset", type=float, default=0.0,
                        help="scalar calibration added to every legal task logit before the "
                             "coordinator residual (depot and masked actions unchanged); "
                             "0.0 keeps the uncalibrated behaviour")
    parser.add_argument("--coordinator", dest="coordinator_policy",
                        choices=("heuristic", "coalition", "coalition_planner",
                                 "coalition_assembly_scheduler",
                                 "coalition_redundant_release", "deadline_aware"),
                        default="heuristic",
                        help="residual policy: 'heuristic' scores how much of a task's whole "
                             "remaining requirement a robot covers, 'coalition' scores the share "
                             "of the *uncovered* part it would add, 'coalition_planner' plans a "
                             "greedy set-cover of the free robots over each task's open "
                             "capability slots and publishes a commitment per selected robot, "
                             "'coalition_assembly_scheduler' adds an arrival-synchronising term "
                             "on top of that plan so the covering set completes together, "
                             "'coalition_redundant_release' may also re-decide a working robot "
                             "whose own belief says it fills no slot on its stage")
    parser.add_argument("--sticky-commitments", dest="sticky_commitments",
                        action="store_true",
                        help="bind a coalition commitment while it stays valid: the "
                             "planner renews the commitment of a robot already on its "
                             "task, and the allocator does not silently send that robot "
                             "elsewhere or to the depot")
    parser.add_argument("--release-redundant-members", dest="release_redundant_members",
                        action="store_true",
                        help="let the coalition planner re-decide a working robot whose own "
                             "belief says it fills no slot on its current stage, and only "
                             "when the plan has a different task for it")
    parser.add_argument("--suppress-recommand-own-task", dest="suppress_recommand_own_task",
                        action="store_true",
                        help="issue no command when a replan-woke robot re-picks the task it "
                             "is already executing (re-issuing one restarts its work clock)")
    parser.add_argument("--allow-release-without-suppression", dest="allow_release_without_suppression",
                        action="store_true",
                        help="research escape: run the redundant-member release on its own so "
                             "the command churn it causes is measurable rather than prevented")
    parser.add_argument("--command-ttl", type=float, default=10.0,
                        help="how long an assignment command stays valid, in global-clock units")
    parser.add_argument("--coordination-interval", type=int, default=5, help="steps between coordinator calls")
    parser.add_argument("--allocation-interval", type=int, default=5, help="steps between allocation calls")
    parser.add_argument("--dt", type=float, default=1.0, help="global clock increment per joint step")
    parser.add_argument("--detection-radius", type=int, default=5, help="Chebyshev radius for target detection")
    parser.add_argument("--comm-delivery-probability", type=float, default=1.0,
                        help="per-packet success probability of the search-to-rescue uplink")
    parser.add_argument("--comm-delay-ticks", type=int, default=0,
                        help="joint steps a detection message spends on the uplink before delivery")
    parser.add_argument("--comm-subscribers", default="all",
                        help="uplink receivers: 'all' (default, whole rescue team), 'first:N' "
                             "(robots 0..N-1), or a comma-separated id list")
    parser.add_argument("--quiet", action="store_true", help="do not print per-step progress lines")
    parser.add_argument("--repository-root", default=None,
                        help="override the repository root used to locate piloc/ and task_allocation/")
    parser.add_argument("--record-stream", default=None,
                        help="run the live search side for --steps and write the ordered "
                             "detection stream (JSON) to this path; the task side still runs")
    parser.add_argument("--replay-stream", default=None,
                        help="replay a recorded detection stream instead of running PILOC, so "
                             "every coordinator arm sees byte-identical search events")
    parser.add_argument("--decision-trace", default=None,
                        help="write one line per allocation decision (action space, hard mask, "
                             "CTDE logits, coordinator residual, choice, ground truth) to this "
                             "path; a statistics-only artifact, never read back by a decision")
    return parser


def configuration_from_args(args: argparse.Namespace) -> RunConfiguration:
    return RunConfiguration(
        seed=int(args.seed),
        steps=int(args.steps),
        device=str(args.device),
        output=str(args.output),
        mock=bool(args.mock),
        piloc_checkpoint=args.piloc_checkpoint,
        task_checkpoint=args.task_checkpoint,
        map_path=args.map_path,
        grid_size=int(args.grid_size),
        obstacle_density=float(args.obstacle_density),
        search_agents=int(args.search_agents),
        num_goals=int(args.num_goals),
        rescue_species=int(args.rescue_species),
        rescue_per_species=int(args.rescue_per_species),
        deadline_tightness=(float(args.deadline_tightness[0]), float(args.deadline_tightness[1])),
        actor_kv_mode=str(args.actor_kv_mode),
        use_coordinator=not args.no_coordinator,
        residual_scale=float(args.residual_scale),
        ctde_task_logit_offset=float(args.ctde_task_logit_offset),
        coordinator_policy=str(args.coordinator_policy),
        sticky_commitments=bool(getattr(args, "sticky_commitments", False)),
        release_redundant_members=bool(getattr(args, "release_redundant_members", False)),
        suppress_recommand_own_task=bool(getattr(args, "suppress_recommand_own_task", False)),
        allow_release_without_suppression=bool(
            getattr(args, "allow_release_without_suppression", False)
        ),
        command_ttl=float(args.command_ttl),
        coordination_interval=int(args.coordination_interval),
        allocation_interval=int(args.allocation_interval),
        dt=float(args.dt),
        detection_radius=int(args.detection_radius),
        comm_delivery_probability=float(args.comm_delivery_probability),
        comm_delay_ticks=int(args.comm_delay_ticks),
        comm_subscribers=str(args.comm_subscribers),
        quiet=bool(args.quiet),
        repository_root=args.repository_root,
        record_stream=args.record_stream,
        replay_stream=args.replay_stream,
        decision_trace=args.decision_trace,
    )


def parse_comm_subscribers(spec: str) -> tuple[str, ...] | None:
    """Interpret ``--comm-subscribers``: ``all``, ``first:N``, or an id list.

    ``None`` means "every rescue robot" (the default broadcast uplink). The count
    form is spelled ``first:N`` rather than a bare ``N`` because rescue robot ids
    are themselves numbers: ``--comm-subscribers 0`` unambiguously names robot 0,
    while ``--comm-subscribers first:2`` names robots 0 and 1.
    """
    text = str(spec).strip().lower()
    if text in ("", "all"):
        return None
    if text.startswith("first:"):
        suffix = text[len("first:"):].strip()
        if suffix.isdigit() and int(suffix) >= 1:
            return tuple(str(index) for index in range(int(suffix)))
        return ()
    ids = [token.strip() for token in text.split(",") if token.strip()]
    if not ids or any(not token.isdigit() for token in ids):
        return ()
    return tuple(dict.fromkeys(ids))


def resolve_device(name: str, *, checkpoint_requested: bool) -> str:
    """Validate ``--device`` against what this machine can actually run."""
    normalized = str(name or "cpu").strip().lower()
    if normalized in ("", "cpu"):
        return "cpu"
    if not normalized.startswith("cuda"):
        raise ConfigurationError(
            f"unsupported --device {name!r}; use 'cpu', 'cuda' or 'cuda:<index>'"
        )
    if not checkpoint_requested:
        # Nothing loads weights, so torch is never imported and no GPU is required.
        return normalized

    try:
        import torch
    except ImportError as exc:
        raise ConfigurationError(
            f"--device {normalized} needs torch, which is not installed: pip install torch"
        ) from exc
    if not torch.cuda.is_available():
        raise ConfigurationError(
            f"--device {normalized} was requested but torch reports no usable CUDA device; "
            "rerun with --device cpu"
        )
    if ":" in normalized:
        suffix = normalized.split(":", 1)[1]
        if not suffix.isdigit() or int(suffix) >= torch.cuda.device_count():
            raise ConfigurationError(
                f"--device {normalized} is not available; torch sees "
                f"{torch.cuda.device_count()} CUDA device(s)"
            )
    return normalized


def run_signature_of(configuration: RunConfiguration) -> dict[str, Any]:
    """The identity of one run: see :mod:`integrated_sar.run_signature`.

    Computed from the run's own configuration, which is the only thing that knows the arm,
    the flags, the checkpoint and the stream it was built from. Callers that need it more
    than once cache it; callers on the decision path never build it at all.
    """
    from .allocation_trace import arm_name
    from .run_signature import build

    return build(
        configuration,
        arm=arm_name(
            use_coordinator=bool(configuration.use_coordinator),
            residual_scale=float(configuration.residual_scale),
            task_logit_offset=float(configuration.ctde_task_logit_offset),
            coordinator_policy=str(configuration.coordinator_policy),
            binding_commitments=bool(configuration.sticky_commitments),
            release_redundant_members=bool(
                configuration.release_redundant_members
                or configuration.coordinator_policy == "coalition_redundant_release"
            ),
            suppress_recommand_own_task=bool(
                configuration.suppress_recommand_own_task
                or configuration.coordinator_policy == "coalition_redundant_release"
            ),
        ),
        repo=Path(__file__).resolve().parent.parent,
    )


@dataclass(slots=True)
class JointRun:
    """A fully wired episode, ready to step and to log."""

    configuration: RunConfiguration
    scenario: SearchScenario
    orchestrator: JointSAROrchestrator
    registry: TaskRegistry
    gate: AssignmentGate
    logger: LifecycleLogger
    search_adapter: PilocSearchAdapter
    task_adapter: Task03Adapter
    stream_recorder: Any = None
    decision_trace: Any = None
    counters: dict[str, int] = field(default_factory=dict)
    _signature: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def run_signature(self) -> dict[str, Any]:
        """What this run is, written into every artefact it produces.

        Cached because the trace writer wants it at construction and the lifecycle log wants
        it at the first step, and both must carry the same truth.
        """
        if self._signature is None:
            self._signature = run_signature_of(self.configuration)
        return self._signature

    def run(self, steps: int | None = None) -> dict[str, Any]:
        total = self.configuration.steps if steps is None else int(steps)
        scenario_detail = {**self.scenario.as_dict(), "map_sha256": scenario_map_sha256(self.scenario)}
        self.logger.log_run_config(
            self.configuration.as_dict(), scenario_detail, self.run_signature
        )

        for report in self.orchestrator.run(total):
            for command in report.assignments:
                self.logger.log_assignment(command, step=report.step)
            for rejection in self.gate.drain_rejections():
                self.logger.log_rejection(rejection, step=report.step)
                self._count("rejected_assignments")
            self._count("detections", len(report.detections))
            self._count("progress_events", len(report.progress_events))
            self._count("assignments", len(report.assignments))
            if not self.configuration.quiet:
                print(
                    f"step={report.step:>4} t={report.current_time:>7.2f} "
                    f"detections={len(report.detections)} "
                    f"assignments={len(report.assignments)} "
                    f"progress={len(report.progress_events)} "
                    f"active_tasks={len(report.active_tasks)}",
                    file=sys.stderr,
                )

        summary = self.summarize(total)
        if self.stream_recorder is not None:
            written = self.stream_recorder.to_stream().write(self.configuration.record_stream)
            summary["recorded_stream"] = str(written)
            summary["recorded_detections"] = self.stream_recorder.detections_recorded
        if self.decision_trace is not None:
            self.decision_trace.finish(summary)
        self.logger.log_final(
            self.registry.snapshots(),
            step=total,
            current_time=self.orchestrator.current_time,
            metrics=summary,
        )
        return summary

    def reset(self) -> None:
        """Start a fresh episode on the same wired collaborators.

        Every clock, simulator, registry, adapter mapping, gate rejection, relay
        outbox, logger dedup entry and run counter restarts; nothing of the
        previous episode's state survives.
        """
        self.orchestrator.reset()
        self.logger.reset()
        self.counters.clear()

    def summarize(self, steps: int) -> dict[str, Any]:
        snapshots = self.registry.snapshots()
        by_status: dict[str, int] = {}
        for snapshot in snapshots:
            key = snapshot.status.value
            by_status[key] = by_status.get(key, 0) + 1
        return {
            "steps": int(steps),
            "current_time": round(self.orchestrator.current_time, 6),
            "targets_discovered": len(snapshots),
            "targets_active": len(self.registry.active_snapshots()),
            "status_counts": by_status,
            "detection_events": self.counters.get("detections", 0),
            "progress_events": self.counters.get("progress_events", 0),
            "assignment_commands": self.counters.get("assignments", 0),
            "rejected_assignments": self.counters.get("rejected_assignments", 0),
            "accepted_event_count": self.registry.accepted_event_count,
            "rejected_event_count": self.registry.rejected_event_count,
            "search_backend": self.configuration.search_backend,
            "allocation_backend": self.configuration.allocation_backend,
            "ctde_task_logit_offset": round(
                float(self.configuration.ctde_task_logit_offset), 6
            ),
            "coordinator_policy": (
                self.configuration.coordinator_policy
                if self.configuration.use_coordinator
                else "none"
            ),
            "remaining_search_targets": len(self.search_adapter.env.goals_pos),
            "rescue_robots": len(self.task_adapter.env.agent_dic),
            "resource_utilization": round(
                float(self.task_adapter.env.get_resource_utilization()), 6
            ),
        }

    def _count(self, key: str, amount: int = 1) -> None:
        if amount:
            self.counters[key] = self.counters.get(key, 0) + amount


def build_joint_run(configuration: RunConfiguration) -> JointRun:
    """Wire one shared scene into the search, registry, coordination and execution ports."""
    configuration.validate()
    device = resolve_device(
        configuration.device,
        checkpoint_requested=bool(configuration.piloc_checkpoint or configuration.task_checkpoint),
    )

    scenario = build_scenario(
        map_path=configuration.map_path,
        size=configuration.grid_size,
        seed=configuration.seed,
        num_agents=configuration.search_agents,
        num_goals=configuration.num_goals,
        obstacle_density=configuration.obstacle_density,
    )

    search_adapter = _build_search_side(configuration, scenario, device)
    recorder = None
    if configuration.replay_stream is not None:
        from .detection_replay import (
            ReplaySearchModule,
            assert_bundle_matches_configuration,
            load_detection_stream,
        )

        stream = load_detection_stream(configuration.replay_stream)
        assert_bundle_matches_configuration(stream, configuration, configuration.steps)
        search_adapter = ReplaySearchModule(stream)
    elif configuration.record_stream is not None:
        from .detection_replay import RecordingSearchModule

        recorder = RecordingSearchModule(
            search_adapter, scenario=scenario, configuration=configuration
        )
        search_adapter = recorder
    task_adapter = _build_task_side(configuration, scenario, device)

    registry = TaskRegistry()
    bus = InMemoryEventBus()
    clock = GlobalClock()
    gate = AssignmentGate(
        task_adapter,
        clock=clock,
        revision_of=lambda task_id: _revision_of(registry, task_id),
    )
    coordinator = (
        _build_coordinator(configuration)
        if configuration.use_coordinator
        else PassthroughCoordinator()
    )

    orchestrator = JointSAROrchestrator(
        search=search_adapter,
        task_sink=task_adapter,
        agent_source=task_adapter,
        coordinator=ReplanCoordinator(coordinator, gate.pending_replan),
        allocator=task_adapter,
        executor=gate,
        registry=registry,
        event_bus=bus,
        clock=clock,
        replan_source=gate,
        dt=configuration.dt,
        coordination_interval=configuration.coordination_interval,
        allocation_interval=configuration.allocation_interval,
    )

    # The search-to-rescue communication bridge. Subscribed after the registry so a
    # queued message snapshots exactly the task view the registry accepted, and so
    # an event the registry refuses is never sent. Delivery obeys an explicit
    # topology (--comm-subscribers), a per-packet delay and a seeded loss process;
    # positions play no role in it (invariant 10).
    requested = parse_comm_subscribers(configuration.comm_subscribers)
    available = tuple(str(agent_id) for agent_id in task_adapter.env.agent_dic)
    if requested is None:
        subscribers = available
    else:
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ConfigurationError(
                f"--comm-subscribers names rescue robots that do not exist: {', '.join(unknown)}; "
                f"this run has {', '.join(available)}"
            )
        subscribers = requested
    comm_relay = CommRelay(
        subscribers=subscribers,
        writer=task_adapter,
        snapshot_of=registry.get,
        accepted_of=registry.was_accepted,
        delivery_probability=configuration.comm_delivery_probability,
        delay_ticks=configuration.comm_delay_ticks,
        seed=configuration.seed,
    )
    bus.subscribe(DetectionEvent, comm_relay.handle_detection)
    task_adapter.comm_relay = comm_relay

    logger = LifecycleLogger(
        configuration.output,
        snapshot_of=registry.get,
        accepted_of=registry.was_accepted,
    )
    # Subscribed after the registry handlers so a logged record already sees the new revision.
    bus.subscribe(DetectionEvent, logger.handle_detection)
    bus.subscribe(ProgressEvent, logger.handle_progress)

    # Built once here, before either artefact exists: the trace's config record and the
    # lifecycle log's run_config record must state the same identity, and the only way to
    # guarantee that is to build it once and hand it to both.
    signature = run_signature_of(configuration)

    trace_writer = None
    if configuration.decision_trace is not None:
        from .allocation_trace import DecisionTraceWriter, trace_meta

        trace_writer = DecisionTraceWriter(
            configuration.decision_trace, meta=trace_meta(configuration, signature)
        )
        task_adapter.decision_trace = trace_writer

    return JointRun(
        configuration=configuration,
        scenario=scenario,
        orchestrator=orchestrator,
        registry=registry,
        gate=gate,
        logger=logger,
        search_adapter=search_adapter,
        task_adapter=task_adapter,
        stream_recorder=recorder,
        decision_trace=trace_writer,
        _signature=signature,
    )


def _build_search_side(
    configuration: RunConfiguration, scenario: SearchScenario, device: str
) -> PilocSearchAdapter:
    # One radius for the whole search side: it decides which cells an agent observes,
    # which targets the environment removes and which detections the adapter publishes.
    radius = configuration.detection_radius
    if configuration.piloc_checkpoint is None:
        environment, observations, channels = build_mock_search_side(
            scenario, view_radius=radius, dt=configuration.dt
        )
        action_provider: Any = FrontierSearchPolicy(view_radius=radius)
        # The mock environment restores the scenario itself.
        reset_world: Any = environment.reset
    else:
        from .piloc_runtime import build_piloc_search_side, reset_piloc_environment

        environment, action_provider, observations, channels = build_piloc_search_side(
            scenario,
            checkpoint=configuration.piloc_checkpoint,
            device=device,
            seed=configuration.seed,
            root=configuration.repository_root,
            obs_radius=radius,
        )

        def reset_world() -> None:
            # timemap.Environment.reset would generate a random map; a faithful
            # reset reseeds both RNGs, re-loads the original scenario and rebuilds
            # the pheromone, position history, timestamp maps and comm counters.
            reset_piloc_environment(environment, scenario, seed=configuration.seed)

    return PilocSearchAdapter(
        env=environment,
        action_provider=action_provider,
        profile_provider=DeterministicTargetProfileProvider(),
        obs=observations,
        two_channelmap=channels,
        episode_horizon=max(configuration.steps, 1),
        detection_radius=radius,
        reset_world=reset_world,
    )


def _rescue_depot_positions(
    scenario: SearchScenario, species: int
) -> list[tuple[float, float]]:
    """Stage each rescue species where a search team started, in the shared scene.

    Both teams respond to the same disaster, so their staging points have to be inside
    it. TaskEnv would otherwise scatter depots uniformly over the unit square, which
    can put every robot outside ``comm_range`` of every target forever.
    """
    cells = scenario.agent_positions
    transform = scenario.transform
    positions: list[tuple[float, float]] = []
    for index in range(max(1, int(species))):
        world = transform.to_world(cells[index % len(cells)])
        positions.append((world.x, world.y))
    return positions


def _build_coordinator(configuration: RunConfiguration) -> Any:
    """The residual policy named by ``--coordinator``; all of them are belief-only."""
    if configuration.coordinator_policy == "coalition":
        return CoalitionAwareCoordinator()
    if configuration.coordinator_policy == "coalition_planner":
        # ``release_redundant_members`` switches the same planner's release mechanism on
        # directly, so an ablation can pair it with or without the allocator suppression;
        # the policy shorthand below turns both halves on at once.
        return CoalitionPlannerCoordinator(
            renew_commitments=configuration.sticky_commitments,
            release_redundant_members=configuration.release_redundant_members,
        )
    if configuration.coordinator_policy == "coalition_assembly_scheduler":
        # The assembly-scheduler prototype: the same planner plus an arrival-synchronising
        # term. Binding commitments stay on, so the arm differs from the binding baseline
        # in exactly one mechanism and the comparison isolates that mechanism.
        return CoalitionPlannerCoordinator(
            renew_commitments=configuration.sticky_commitments,
            assembly_scheduling=True,
        )
    if configuration.coordinator_policy == "coalition_redundant_release":
        # The redundant-member release: the same planner, allowed to re-decide a working
        # robot whose own belief says it fills no slot on its stage. Binding commitments
        # stay on so the arm differs from the binding baseline in exactly one mechanism.
        return CoalitionPlannerCoordinator(
            renew_commitments=configuration.sticky_commitments,
            release_redundant_members=True,
        )
    if configuration.coordinator_policy == "deadline_aware":
        # The same-information oracle of the learnability gate: identical candidates and
        # slots, a different selection. Diagnostic, but a legal coordinator all the same.
        return DeadlineAwareCoalitionCoordinator()
    return HeuristicCoordinator()


def _build_task_side(
    configuration: RunConfiguration, scenario: SearchScenario, device: str
) -> Task03Adapter:
    from .task03_runtime import (
        build_task_env,
        load_ctde_network,
        load_task_env_module,
        resolve_task_checkpoint,
    )

    # Resolved first: a missing checkpoint must be reported before any module import.
    resolved_checkpoint = (
        resolve_task_checkpoint(configuration.task_checkpoint)
        if configuration.task_checkpoint is not None
        else None
    )

    module = load_task_env_module(
        configuration.repository_root,
        require_torch=resolved_checkpoint is not None,
    )
    environment = build_task_env(
        module,
        seed=configuration.seed,
        species=configuration.rescue_species,
        per_species=configuration.rescue_per_species,
        deadline_tightness=tuple(float(value) for value in configuration.deadline_tightness),
        dt=configuration.dt,
        depot_positions=_rescue_depot_positions(scenario, configuration.rescue_species),
    )
    network = (
        load_ctde_network(
            resolved_checkpoint,
            device=device,
            actor_kv_mode=configuration.actor_kv_mode,
            root=configuration.repository_root,
        )
        if resolved_checkpoint is not None
        else None
    )
    return Task03Adapter(
        env=environment,
        network=network,
        deterministic=True,
        residual_scale=configuration.residual_scale,
        command_ttl=configuration.command_ttl,
        seed=configuration.seed,
        ctde_task_logit_offset=configuration.ctde_task_logit_offset,
        binding_commitments=configuration.sticky_commitments,
        suppress_recommand_own_task=bool(
            configuration.suppress_recommand_own_task
            or configuration.coordinator_policy == "coalition_redundant_release"
        ),
    )


def _revision_of(registry: TaskRegistry, task_id: str | None) -> int | None:
    if task_id is None:
        return None
    snapshot = registry.get(task_id)
    return snapshot.revision if snapshot is not None else None


def run_from_configuration(configuration: RunConfiguration) -> dict[str, Any]:
    """Build, run and close one episode; used by the CLI and by the tests alike."""
    run = build_joint_run(configuration)
    try:
        return run.run()
    finally:
        run.logger.close()
        if run.decision_trace is not None:
            run.decision_trace.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configuration = configuration_from_args(args)

    output_is_file = configuration.output != "-"
    if output_is_file:
        directory = os.path.dirname(os.path.abspath(configuration.output))
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            print(
                f"[joint-driver] ConfigurationError: cannot create --output directory "
                f"{directory}: {exc}",
                file=sys.stderr,
            )
            return 2

    try:
        summary = run_from_configuration(configuration)
    except IntegratedSARError as exc:
        print(f"[joint-driver] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[joint-driver] interrupted", file=sys.stderr)
        return 130

    print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)
    if output_is_file:
        print(f"[joint-driver] lifecycle log written to {configuration.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
