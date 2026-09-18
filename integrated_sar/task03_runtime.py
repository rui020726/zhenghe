"""Runtime wiring for the task-03 CTDE allocation and execution module.

``task_allocation/old_task_env.py`` is pure numpy at runtime but imports torch at
module scope, so the loader stubs torch when it is genuinely absent and no checkpoint
was requested. That keeps the mock driver usable on a CI machine with numpy only.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .errors import AssetMissingError, CheckpointIncompatibleError, DependencyMissingError
from .piloc_runtime import repository_root

TASK_ENV_CANDIDATES = (
    ("task_allocation", "old_task_env.py"),
    ("task-03-修改为CTDE范式", "old_task_env.py"),
)
CTDE_FILENAME_CANDIDATES = ("ctde_final.pth", "ctde_best.pth")
EPISODE_CHECKPOINT_PATTERN = re.compile(r"^ctde_ep_(\d+)\.pth$")

DEFAULT_NETWORK_DIMS: dict[str, Any] = {
    "self_dim": 10,
    "neighbor_dim": 10,
    "task_dim": 20,
    "global_agent_dim": 13,
    "global_task_dim": 23,
    "hidden": 128,
    "dropout": 0.15,
    "use_pos_encoding": True,
}


def task_env_source_path(root: str | os.PathLike[str] | None = None) -> Path:
    """Locate the mainline ``old_task_env.py`` described in AGENTS.md."""
    base = Path(root) if root is not None else repository_root()
    for parts in TASK_ENV_CANDIDATES:
        candidate = base.joinpath(*parts)
        if candidate.is_file():
            return candidate
    expected = base.joinpath(*TASK_ENV_CANDIDATES[0])
    raise AssetMissingError(
        f"task-03 environment source not found; looked for {expected} "
        f"and {base.joinpath(*TASK_ENV_CANDIDATES[1])}"
    )


def load_task_env_module(
    root: str | os.PathLike[str] | None = None,
    *,
    require_torch: bool = False,
) -> Any:
    """
    Import ``old_task_env.py`` as a standalone module.

    It is loaded by path rather than by package import because ``task_allocation`` is
    not a package and its sibling modules pull in training-only dependencies.
    """
    path = task_env_source_path(root)
    has_torch = importlib.util.find_spec("torch") is not None
    if not has_torch and require_torch:
        raise DependencyMissingError(
            "a task checkpoint was requested but torch is not installed: pip install torch"
        )

    torch_stub = None
    if not has_torch:
        # Only the three module-level statements in old_task_env.py touch torch.
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = types.SimpleNamespace(empty_cache=lambda: None)
        torch_stub.backends = types.SimpleNamespace(
            cudnn=types.SimpleNamespace(benchmark=False, deterministic=True)
        )
        sys.modules["torch"] = torch_stub

    spec = importlib.util.spec_from_file_location("integrated_sar_task_env", path)
    if spec is None or spec.loader is None:
        raise AssetMissingError(f"cannot create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise DependencyMissingError(f"importing {path} failed: {exc}") from exc
    finally:
        if torch_stub is not None:
            sys.modules.pop("torch", None)

    if not hasattr(module, "TaskEnv"):
        raise AssetMissingError(f"{path} does not define TaskEnv")
    module.__integrated_sar_source__ = str(path)
    return module


def build_task_env(
    module: Any,
    *,
    seed: int = 0,
    species: int = 3,
    per_species: int = 2,
    dt: float = 1.0,
    comm_range: float = 0.3,
    traits_dim: int = 5,
    deadline_tightness: tuple[float, float] = (0.5, 2.0),
    membership_consistency: bool = False,
    depot_positions: Sequence[Sequence[float]] | None = None,
) -> Any:
    """
    Create a rescue environment that starts empty and waits for PILOC detections.

    ``tasks_range=(0, 0)`` plus both burst generators disabled makes the registry the
    only source of tasks, and a fixed species count covering all five ability
    dimensions keeps injected tasks feasible for some robot.

    ``deadline_tightness`` picks the multiplier band TaskEnv already samples every
    deadline from. The default is TaskEnv's own, so a caller that says nothing gets the
    scenarios recorded before this argument existed; the sampler itself is untouched.

    ``membership_consistency`` makes the member roster mean what it says: an accepted
    reassignment detaches the robot from the task it is leaving, and re-selecting the task
    a working robot is already on becomes a true no-op. Off by default, because every
    existing number was produced under the old bookkeeping.

    ``depot_positions`` stages the rescue teams inside the shared scene. Without it
    TaskEnv scatters depots uniformly over the unit square, which can leave them
    permanently outside ``comm_range`` of every PILOC target: nothing is sensed,
    gossip has nothing to spread, and the belief-gated pipeline never starts.
    """
    env = module.TaskEnv(
        tasks_range=(0, 0),
        species_range=(species, species),
        per_species_range=(per_species, per_species),
        traits_dim=traits_dim,
        deadline_tightness=deadline_tightness,
        membership_consistency=membership_consistency,
        enable_tidal_tasks=False,
        enable_discovered_tasks=False,
        dynamic_task_discovery=False,
        comm_range=comm_range,
        dt=dt,
        seed=seed,
    )
    if depot_positions is not None:
        place_depots(env, depot_positions)
    return env


def place_depots(env: Any, depot_positions: Sequence[Sequence[float]]) -> None:
    """Move each species depot, and the robots staged on it, to a normalized position."""
    positions = [np.asarray(point, dtype=float) for point in depot_positions]
    if not positions:
        raise AssetMissingError("depot_positions must name at least one staging point")

    for index, species_id in enumerate(sorted(env.depot_dic)):
        point = positions[index % len(positions)]
        env.depot_dic[species_id]["location"] = point.copy()
        for member in env.depot_dic[species_id].get("members", ()):
            agent = env.agent_dic[member]
            agent["location"] = point.copy()
            agent["depot"] = point.copy()


def resolve_task_checkpoint(checkpoint: str | os.PathLike[str]) -> Path:
    """Accept a ``.pth`` file or a checkpoint directory written by ``old_driver.py``."""
    path = Path(os.path.abspath(os.fspath(checkpoint)))
    if path.is_file():
        return path
    if not path.is_dir():
        raise AssetMissingError(
            f"task checkpoint not found: {path}. Pass a .pth file with --task-checkpoint, "
            "or a directory containing ctde_final.pth / ctde_best.pth / ctde_ep_*.pth, "
            "or drop the flag to run the heuristic allocator."
        )

    for name in CTDE_FILENAME_CANDIDATES:
        candidate = path / name
        if candidate.is_file():
            return candidate

    episode_checkpoints = [
        (int(EPISODE_CHECKPOINT_PATTERN.match(entry.name).group(1)), entry)
        for entry in path.iterdir()
        if EPISODE_CHECKPOINT_PATTERN.match(entry.name)
    ]
    if episode_checkpoints:
        return max(episode_checkpoints)[1]

    others = [entry for entry in path.iterdir() if entry.suffix == ".pth" and entry.is_file()]
    if others:
        return max(others, key=lambda entry: entry.stat().st_mtime)

    raise AssetMissingError(
        f"directory {path} contains no .pth checkpoint. Expected ctde_final.pth / "
        "ctde_best.pth / ctde_ep_*.pth, pass a .pth file directly with --task-checkpoint, "
        "or drop the flag to run the heuristic allocator."
    )


def load_ctde_network(
    checkpoint: str | os.PathLike[str],
    *,
    device: str = "cpu",
    actor_kv_mode: str = "tokens",
    root: str | os.PathLike[str] | None = None,
) -> Any:
    """
    Load the CTDE allocation network strictly.

    ``old_test.load_model`` silently retries with ``strict=False`` when the structure
    does not match, which would run the episode with partly random weights. Here a
    mismatch is fatal, and the network is probed once before it is allowed to decide
    anything.
    """
    resolved = resolve_task_checkpoint(checkpoint)
    try:
        import torch
    except ImportError as exc:
        raise DependencyMissingError(
            f"loading task checkpoint {resolved} requires torch: pip install torch"
        ) from exc

    network_module = _load_network_module(root)
    network = network_module.CTDE_Network(
        actor_kv_mode=actor_kv_mode, **DEFAULT_NETWORK_DIMS
    )

    try:
        state = torch.load(resolved, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise AssetMissingError(f"cannot read task checkpoint {resolved}: {exc}") from exc

    recorded_mode = state.get("actor_kv_mode") if isinstance(state, dict) else None
    if recorded_mode is not None and recorded_mode != actor_kv_mode:
        raise CheckpointIncompatibleError(
            f"task checkpoint {resolved} was trained with actor_kv_mode={recorded_mode!r} "
            f"but {actor_kv_mode!r} was requested. The two modes have identical parameter "
            f"shapes, so a strict load would succeed while silently changing what the "
            f"network computes."
        )

    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    try:
        network.load_state_dict(state_dict, strict=True)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointIncompatibleError(
            f"task checkpoint {resolved} does not match CTDE_Network(actor_kv_mode={actor_kv_mode!r}): {exc}"
        ) from exc

    network.eval().to(device)
    network.checkpoint_path = str(resolved)
    network.checkpoint_episode = (
        state.get("episode") if isinstance(state, dict) else None
    )
    _probe_network(network, torch, device)
    return network


def _load_network_module(root: str | os.PathLike[str] | None) -> Any:
    """Import ``old_network.py`` by path, next to the environment it serves."""
    env_path = task_env_source_path(root)
    network_path = env_path.with_name("old_network.py")
    if not network_path.is_file():
        raise AssetMissingError(f"CTDE network source not found next to {env_path}")

    spec = importlib.util.spec_from_file_location("integrated_sar_ctde_network", network_path)
    if spec is None or spec.loader is None:
        raise AssetMissingError(f"cannot create an import spec for {network_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise DependencyMissingError(f"importing {network_path} failed: {exc}") from exc
    return module


def _probe_network(network: Any, torch: Any, device: str) -> None:
    """
    Reject weights that cannot produce a usable decision.

    The probe reproduces the padding-only neighbour batch documented in AGENTS.md and
    also checks that a masked slot never wins, so a checkpoint that would emit NaN or
    ignore the environment's hard mask fails at load time instead of mid-episode.
    """
    task_count = 4
    self_state = torch.zeros(1, DEFAULT_NETWORK_DIMS["self_dim"], device=device)
    neighbors = torch.zeros(
        1, 20, DEFAULT_NETWORK_DIMS["neighbor_dim"], device=device
    )
    tasks = torch.zeros(1, task_count, DEFAULT_NETWORK_DIMS["task_dim"], device=device)
    tasks[0, :, 0] = torch.arange(task_count, dtype=torch.float32, device=device)
    mask = torch.zeros(1, task_count, dtype=torch.bool, device=device)
    mask[0, -1] = True

    try:
        with torch.no_grad():
            distribution = network.get_action(self_state, neighbors, tasks, mask)
            logits = distribution.logits
    except Exception as exc:  # noqa: BLE001 - torch rejects NaN logits before we can inspect them
        raise CheckpointIncompatibleError(
            f"CTDE network {getattr(network, 'checkpoint_path', '?')} could not score a "
            f"padding-only neighbour batch (actor_kv_mode="
            f"{getattr(network, 'actor_kv_mode', '?')}): {exc}"
        ) from exc

    values = logits.detach().cpu().numpy().reshape(-1)
    if not np.all(np.isfinite(values)):
        raise CheckpointIncompatibleError(
            f"CTDE network {getattr(network, 'checkpoint_path', '?')} produced non-finite "
            f"logits {values.tolist()} for a padding-only neighbour batch "
            f"(actor_kv_mode={getattr(network, 'actor_kv_mode', '?')})"
        )
    chosen = int(max(range(len(values)), key=lambda index: values[index]))
    if bool(mask[0, chosen]):
        raise CheckpointIncompatibleError(
            f"CTDE network {getattr(network, 'checkpoint_path', '?')} selected masked "
            f"action {chosen}; the environment hard mask must stay authoritative"
        )
