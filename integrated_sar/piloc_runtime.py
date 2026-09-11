"""Runtime wiring for the real PILOC search module.

Every PILOC import happens inside a function: importing ``piloc.src.Map.timemap``
pulls in ``configs``, ``gym`` and Pillow, and ``piloc.src.DDQN.Agent`` pulls in
tensorflow and torch. Keeping those imports lazy is what allows the mock driver and
the unit tests to run on a machine that has none of them installed.
"""

from __future__ import annotations

import os
import random
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .errors import AssetMissingError, CheckpointIncompatibleError, DependencyMissingError
from .scenarios import SearchScenario
from .search_runtime import initialize_timestamp_observations, initialize_two_channel_map, local_pheromone_window

PILOC_SUBDIR = "piloc"
SHARED_MODEL_PATTERN = re.compile(r"^shared_mappo_model_(\d+)_(\d+)\.pth$")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_piloc_on_path(root: str | os.PathLike[str] | None = None) -> Path:
    """Make ``configs``, ``src.*`` and ``normalization`` importable, as PILOC expects."""
    base = Path(root) if root is not None else repository_root()
    piloc_dir = base / PILOC_SUBDIR
    if not piloc_dir.is_dir():
        raise AssetMissingError(
            f"PILOC sources not found at {piloc_dir}; the search module cannot be loaded"
        )
    for entry in (piloc_dir, piloc_dir / "src" / "DDQN"):
        text = str(entry)
        if entry.is_dir() and text not in sys.path:
            sys.path.append(text)
    return piloc_dir


def import_timemap_module(root: str | os.PathLike[str] | None = None) -> Any:
    """Import the PILOC environment module, which only needs gym and Pillow."""
    ensure_piloc_on_path(root)
    try:
        from src.Map import timemap  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise DependencyMissingError(
            f"importing piloc.src.Map.timemap failed ({exc}). "
            "PILOC needs gym and Pillow: pip install gym Pillow"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced as an actionable driver error
        raise DependencyMissingError(f"importing piloc.src.Map.timemap failed: {exc}") from exc
    return timemap


def import_agent_module(root: str | os.PathLike[str] | None = None) -> Any:
    """Import the PILOC policy module, which pulls in torch and tensorflow."""
    ensure_piloc_on_path(root)
    try:
        from src.DDQN import Agent as agent_module  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise DependencyMissingError(
            f"importing piloc.src.DDQN.Agent failed ({exc}). "
            "The search policy needs torch and tensorflow: pip install torch tensorflow"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise DependencyMissingError(f"importing piloc.src.DDQN.Agent failed: {exc}") from exc
    return agent_module


def build_piloc_environment(
    scenario: SearchScenario,
    *,
    root: str | os.PathLike[str] | None = None,
    num_agents: int | None = None,
    num_goals: int | None = None,
    seed: int = 0,
    obs_radius: int = 5,
    enable_comm_noise: bool = False,
) -> Any:
    """
    Construct ``timemap.Environment`` around the shared scenario.

    ``Environment.__init__`` reads ``self.map`` before assigning it whenever
    ``test=False`` (see the known-issues list in AGENTS.md), so the instance is
    created in two steps and the scenario map is installed first. ``load`` then
    replaces the positions the constructor randomly drew, which keeps one scene
    shared with task-03 as architecture invariant 10 requires.
    """
    timemap = import_timemap_module(root)
    agents = len(scenario.agent_positions) if num_agents is None else int(num_agents)
    goals = len(scenario.goal_positions) if num_goals is None else int(num_goals)
    if agents < 1:
        raise AssetMissingError("the scenario provides no search-robot start position")
    if goals < 1:
        raise AssetMissingError("the scenario provides no hidden target to search for")

    rows, cols = scenario.rows, scenario.cols
    if (rows, cols) != (60, 60):
        raise CheckpointIncompatibleError(
            f"PILOC's convolutional policy is built for a 60x60 timestamp map, "
            f"but the scenario from {scenario.source} is {rows}x{cols}"
        )

    # PILOC draws obstacle densities and start cells from both RNGs during __init__.
    random.seed(seed)
    np.random.seed(seed)

    environment = timemap.Environment.__new__(timemap.Environment)
    environment.map = np.asarray(scenario.obstacles, dtype=np.int32)
    timemap.Environment.__init__(
        environment,
        num_agents=agents,
        map_length=rows,
        obs_radius=obs_radius,
        num_goals=goals,
        curriculum=False,
        test=False,
        enable_comm_noise=enable_comm_noise,
    )
    environment.load(
        np.asarray(scenario.obstacles, dtype=np.int32),
        np.asarray([[p.row, p.col] for p in scenario.agent_positions], dtype=np.int32),
        np.asarray([[p.row, p.col] for p in scenario.goal_positions], dtype=np.int32),
    )
    return environment


def resolve_piloc_checkpoint(checkpoint: str | os.PathLike[str]) -> Path:
    """Accept either a ``.pth`` file or a directory of ``shared_mappo_model_*.pth``."""
    path = Path(os.path.abspath(os.fspath(checkpoint)))
    if path.is_file():
        return path
    if not path.is_dir():
        raise AssetMissingError(
            f"PILOC checkpoint not found: {path}. Pass a .pth file with --piloc-checkpoint, "
            "or a directory containing shared_mappo_model_*.pth, or drop the flag to run --mock."
        )

    candidates = sorted(entry for entry in path.iterdir() if entry.suffix == ".pth")
    if not candidates:
        raise AssetMissingError(f"directory {path} contains no .pth checkpoint")

    def episode_key(entry: Path) -> tuple[int, int]:
        match = SHARED_MODEL_PATTERN.match(entry.name)
        return (int(match.group(1)), int(match.group(2))) if match else (-1, -1)

    shared = [entry for entry in candidates if episode_key(entry)[0] >= 0]
    if shared:
        return max(shared, key=episode_key)
    return max(candidates, key=lambda entry: entry.stat().st_mtime)


def load_piloc_policy(
    checkpoint: str | os.PathLike[str],
    *,
    device: str = "cpu",
    num_agents: int = 2,
    root: str | os.PathLike[str] | None = None,
) -> Any:
    """
    Load a trained PILOC policy without touching ``MAPPO.load``.

    ``MAPPO.load`` hardcodes ``device='cuda'`` and the filename
    ``shared_mappo_model_21_9232.pth``, which AGENTS.md forbids, so the policy object
    is built directly and the requested weights are loaded strictly.
    """
    resolved = resolve_piloc_checkpoint(checkpoint)
    agent_module = import_agent_module(root)
    try:
        import torch
    except ImportError as exc:
        raise DependencyMissingError(
            f"loading PILOC checkpoint {resolved} requires torch: pip install torch"
        ) from exc

    try:
        import configs as piloc_configs  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DependencyMissingError(f"piloc configs module is not importable: {exc}") from exc

    dim_info = {f"agent_{index}": [9, 5] for index in range(num_agents)}
    policy = agent_module.MAPPO(
        dim_info,
        actor_lr=0.0,
        critic_lr=0.0,
        horizon=0,
        device=device,
        trick=dict(piloc_configs.TRICK),
        num_agent=num_agents,
    )

    try:
        state = torch.load(resolved, map_location=device, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise AssetMissingError(f"cannot read PILOC checkpoint {resolved}: {exc}") from exc
    if not isinstance(state, Mapping):
        raise CheckpointIncompatibleError(
            f"PILOC checkpoint {resolved} should hold a mapping of state dicts, got {type(state).__name__}"
        )

    missing_sections = [key for key in ("full_model", "actor", "critic") if key not in state]
    if missing_sections:
        raise CheckpointIncompatibleError(
            f"PILOC checkpoint {resolved} is missing {', '.join(missing_sections)}; "
            "expected the layout written by MAPPO.save"
        )

    for key, module in (
        ("full_model", policy.agents.full_model),
        ("actor", policy.agents.actor),
        ("critic", policy.agents.critic),
    ):
        try:
            module.load_state_dict(state[key], strict=True)
        except Exception as exc:  # noqa: BLE001
            raise CheckpointIncompatibleError(
                f"PILOC checkpoint {resolved} does not match the current {key} architecture: {exc}"
            ) from exc
        module.eval()

    policy.checkpoint_path = str(resolved)
    return policy


class TrainedPilocPolicy:
    """Action provider that runs the trained PILOC network deterministically.

    It mirrors ``piloc/test.select_action`` but takes the argmax instead of sampling,
    so one seed produces one episode.
    """

    def __init__(self, policy: Any, *, device: str = "cpu", view_radius: int = 5) -> None:
        self.policy = policy
        self.device = device
        self.view_radius = int(view_radius)

    def __call__(
        self,
        env: Any,
        obs: Mapping[str, np.ndarray],
        two_channelmap: Mapping[str, Any],
    ) -> dict[str, int]:
        import torch

        del obs  # the trained policy consumes the two-channel map, not the raw timestamps
        actions: dict[str, int] = {}
        for agent_id in env.agents:
            index = int(str(agent_id).rsplit("_", 1)[-1])
            row, col = (int(value) for value in env.agents_pos[index])
            channels = two_channelmap[agent_id]
            observation = np.stack(
                [channels["obstacle_map"], channels["explore_map"]], axis=0
            ).astype(np.float32)
            pheromone = local_pheromone_window(
                np.asarray(env.global_pheromone), row, col, self.view_radius
            )
            with torch.no_grad():
                _, probabilities = self.policy.agents.get_action(
                    torch.as_tensor(observation, device=self.device).unsqueeze(0),
                    torch.as_tensor(pheromone.astype(np.float32), device=self.device).unsqueeze(0),
                    batch_size=1,
                )
            values = np.asarray(probabilities.detach().cpu(), dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(values)):
                raise CheckpointIncompatibleError(
                    f"PILOC policy returned non-finite action probabilities for {agent_id}"
                )
            actions[str(agent_id)] = int(np.argmax(values))
        return actions


def build_piloc_search_side(
    scenario: SearchScenario,
    *,
    checkpoint: str | os.PathLike[str],
    device: str = "cpu",
    seed: int = 0,
    root: str | os.PathLike[str] | None = None,
    enable_comm_noise: bool = False,
) -> tuple[Any, Any, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Build the real PILOC environment, policy and initial observations."""
    # Resolved first: a missing checkpoint must be reported before any heavy import.
    resolved_checkpoint = resolve_piloc_checkpoint(checkpoint)
    environment = build_piloc_environment(
        scenario,
        root=root,
        seed=seed,
        enable_comm_noise=enable_comm_noise,
    )
    policy = load_piloc_policy(
        resolved_checkpoint,
        device=device,
        num_agents=len(scenario.agent_positions),
        root=root,
    )
    observations = initialize_timestamp_observations(environment)
    return (
        environment,
        TrainedPilocPolicy(policy, device=device),
        observations,
        initialize_two_channel_map(observations),
    )
