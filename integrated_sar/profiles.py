"""Target-profile providers bridge PILOC point targets to heterogeneous tasks."""

from __future__ import annotations

import hashlib
import random
from typing import Protocol

from .contracts import GridPosition, TargetProfile, TaskStageSpec


class TargetProfileProvider(Protocol):
    def profile_for(
        self,
        target_id: str,
        position: GridPosition,
        detected_at: float,
    ) -> TargetProfile: ...


class DeterministicTargetProfileProvider:
    """Reproducible stand-in until the simulator exposes semantic target data."""

    def __init__(
        self,
        traits_dim: int = 5,
        max_stages: int = 3,
        deadline_range: tuple[float, float] = (60.0, 180.0),
    ) -> None:
        if traits_dim < 1 or max_stages < 1:
            raise ValueError("traits_dim and max_stages must be positive")
        if deadline_range[0] <= 0 or deadline_range[1] < deadline_range[0]:
            raise ValueError("invalid deadline range")
        self.traits_dim = int(traits_dim)
        self.max_stages = int(max_stages)
        self.deadline_range = (float(deadline_range[0]), float(deadline_range[1]))

    def profile_for(
        self,
        target_id: str,
        position: GridPosition,
        detected_at: float,
    ) -> TargetProfile:
        # Hash the physical location, not the track id: track ids are namespaced
        # by episode ("piloc-e2-target-3"), and a replayed episode must give the
        # same hidden target the same stage plan, severity and deadline.
        payload = f"{position.row}:{position.col}".encode("utf-8")
        seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
        rng = random.Random(seed)
        stage_count = rng.randint(1, self.max_stages)
        stages: list[TaskStageSpec] = []

        for _ in range(stage_count):
            required_count = rng.randint(1, min(3, self.traits_dim))
            required_dims = rng.sample(range(self.traits_dim), required_count)
            requirements = tuple(1 if index in required_dims else 0 for index in range(self.traits_dim))
            stages.append(TaskStageSpec(requirements, duration=rng.uniform(3.0, 12.0)))

        severity = rng.uniform(0.3, 1.0)
        deadline_offset = rng.uniform(*self.deadline_range)
        return TargetProfile(
            stages=tuple(stages),
            deadline_offset=deadline_offset,
            severity=severity,
            metadata={
                "profile_source": "deterministic-placeholder",
                "profile_seed": seed,
            },
        )
