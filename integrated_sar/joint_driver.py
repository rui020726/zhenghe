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
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .adapters.piloc import PilocSearchAdapter
from .adapters.task03 import Task03Adapter
from .assignment_gate import AssignmentGate, ReplanCoordinator
from .clock import GlobalClock
from .comm import CommRelay
from .contracts import DetectionEvent, ProgressEvent, TaskSnapshot
from .coordinator import HeuristicCoordinator, PassthroughCoordinator
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
    actor_kv_mode: str = "tokens"
    use_coordinator: bool = True
    residual_scale: float = 0.5
    command_ttl: float = 10.0
    coordination_interval: int = 5
    allocation_interval: int = 5
    dt: float = 1.0
    detection_radius: int = 5
    comm_delivery_probability: float = 1.0
    comm_delay_ticks: int = 0
    quiet: bool = False
    repository_root: str | None = None

    @property
    def search_backend(self) -> str:
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
    parser.add_argument("--actor-kv-mode", default="tokens", choices=("tokens", "summary"),
                        help="CTDE actor key/value mode the checkpoint was trained with")
    parser.add_argument("--no-coordinator", action="store_true",
                        help="disable the coordinator residual (ablation baseline)")
    parser.add_argument("--residual-scale", type=float, default=0.5,
                        help="alpha in final_logits = ctde_logits + alpha * coordinator_bias")
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
    parser.add_argument("--quiet", action="store_true", help="do not print per-step progress lines")
    parser.add_argument("--repository-root", default=None,
                        help="override the repository root used to locate piloc/ and task_allocation/")
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
        actor_kv_mode=str(args.actor_kv_mode),
        use_coordinator=not args.no_coordinator,
        residual_scale=float(args.residual_scale),
        command_ttl=float(args.command_ttl),
        coordination_interval=int(args.coordination_interval),
        allocation_interval=int(args.allocation_interval),
        dt=float(args.dt),
        detection_radius=int(args.detection_radius),
        comm_delivery_probability=float(args.comm_delivery_probability),
        comm_delay_ticks=int(args.comm_delay_ticks),
        quiet=bool(args.quiet),
        repository_root=args.repository_root,
    )


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
    counters: dict[str, int] = field(default_factory=dict)

    def run(self, steps: int | None = None) -> dict[str, Any]:
        total = self.configuration.steps if steps is None else int(steps)
        self.logger.log_run_config(self.configuration.as_dict(), self.scenario.as_dict())

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
        self.logger.log_final(
            self.registry.snapshots(),
            step=total,
            current_time=self.orchestrator.current_time,
            metrics=summary,
        )
        return summary

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
            "search_backend": self.configuration.search_backend,
            "allocation_backend": self.configuration.allocation_backend,
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
        HeuristicCoordinator() if configuration.use_coordinator else PassthroughCoordinator()
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
    # queued message snapshots exactly the task view the registry accepted. Delivery
    # obeys an explicit topology (the subscribed rescue robots), a per-packet delay
    # and a seeded loss process; positions play no role in it (invariant 10).
    comm_relay = CommRelay(
        subscribers=tuple(str(agent_id) for agent_id in task_adapter.env.agent_dic),
        writer=task_adapter,
        snapshot_of=registry.get,
        delivery_probability=configuration.comm_delivery_probability,
        delay_ticks=configuration.comm_delay_ticks,
        seed=configuration.seed,
    )
    bus.subscribe(DetectionEvent, comm_relay.handle_detection)
    task_adapter.comm_relay = comm_relay

    logger = LifecycleLogger(configuration.output, snapshot_of=registry.get)
    # Subscribed after the registry handlers so a logged record already sees the new revision.
    bus.subscribe(DetectionEvent, logger.handle_detection)
    bus.subscribe(ProgressEvent, logger.handle_progress)

    return JointRun(
        configuration=configuration,
        scenario=scenario,
        orchestrator=orchestrator,
        registry=registry,
        gate=gate,
        logger=logger,
        search_adapter=search_adapter,
        task_adapter=task_adapter,
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
    else:
        from .piloc_runtime import build_piloc_search_side

        environment, action_provider, observations, channels = build_piloc_search_side(
            scenario,
            checkpoint=configuration.piloc_checkpoint,
            device=device,
            seed=configuration.seed,
            root=configuration.repository_root,
            obs_radius=radius,
        )

    return PilocSearchAdapter(
        env=environment,
        action_provider=action_provider,
        profile_provider=DeterministicTargetProfileProvider(),
        obs=observations,
        two_channelmap=channels,
        episode_horizon=max(configuration.steps, 1),
        detection_radius=radius,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configuration = configuration_from_args(args)

    output_is_file = configuration.output != "-"
    if output_is_file:
        directory = os.path.dirname(os.path.abspath(configuration.output))
        if not os.path.isdir(directory):
            print(
                f"[joint-driver] ConfigurationError: --output directory does not exist: {directory}",
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
