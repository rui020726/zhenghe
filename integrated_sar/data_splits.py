"""The data split registry: which seeds exist, what each one is for, and what may touch it.

A split is a promise about how information was allowed to flow. The previous phase read
seeds 20000-20099 while designing the arm that was then measured on them, so calling that
block a test set any longer would be the leak it is. This module is where the promise is
written down once, so that a later phase cannot quietly promote a development block into
an evaluation one and a test can hold the line.

Every split carries its purpose, whether it is still frozen, and the protocol its streams
were recorded with. Nothing here reads or writes data: it only names it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parent.parent

# The frozen checkpoints every arm of every split is replayed against. The search policy
# is part of the data: a different PILOC checkpoint invalidates every stream recorded with
# this one and requires recording again.
PILOC_CHECKPOINT = REPO_DEFAULT / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth"
CTDE_CHECKPOINT = REPO_DEFAULT / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth"

STEPS = 1000
RECORDING_PROTOCOL = (
    "one live PILOC pass per seed at --steps 1000 with --no-coordinator, writing the "
    "ordered DetectionEvent stream; every arm of the split then replays that stream"
)


@dataclass(frozen=True, slots=True)
class DataSplit:
    """One named block of seeds, and what it is allowed to be used for."""

    name: str
    seeds: tuple[int, ...]
    purpose: str
    frozen: bool
    stream_dir_name: str
    # Where the split's arm logs and traces live. Two of the three blocks were produced
    # by earlier phases and stay where they were written; pointing at them rather than
    # copying avoids re-running a hundred seeds to read data that already exists.
    run_dir_name: str = ""

    def stream_dir(self, repo: Path = REPO_DEFAULT) -> Path:
        # Relative to runs/, because two of the three blocks were recorded before this
        # module existed and still live where they were written.
        return repo / "runs" / self.stream_dir_name

    def run_dir(self, repo: Path = REPO_DEFAULT) -> Path:
        return repo / "runs" / (self.run_dir_name or f"learnability_20260917/{self.name}")

    def contains(self, seed: int) -> bool:
        return seed in self.seeds


SPLITS: dict[str, DataSplit] = {
    # Read repeatedly while the calibration offset and the coalition term were designed,
    # so it selects arms and nothing else.
    "dev": DataSplit(
        name="dev",
        seeds=tuple(range(10000, 10030)),
        purpose="development: arm and threshold selection only, never an evaluation set",
        frozen=True,
        stream_dir_name="joint_eval_20260916/streams",
        run_dir_name="coordinator_baselines_20260917",
    ),
    # This block was recorded before the reservation arm existed and was reported as a
    # held-out test set in RESERVATION_VERDICT.md. It has since been read while the
    # learnability gate was designed, so it is demoted: it is an *analysis* block from
    # here on, and the arm was already selected on it.
    "analysis_legacy": DataSplit(
        name="analysis_legacy",
        seeds=tuple(range(20000, 20100)),
        purpose="analysis: previously reported as held-out, now demoted because the "
        "learnability gate was designed while reading it",
        frozen=True,
        stream_dir_name="coordinator_baselines_20260917/heldout/streams",
        run_dir_name="coordinator_baselines_20260917/heldout",
    ),
    # Recorded after the gate was specified, for the same reason every fresh block is:
    # the oracle and the attribution classes were fixed before these seeds were touched.
    "analysis_fresh": DataSplit(
        name="analysis_fresh",
        seeds=tuple(range(21000, 21060)),
        purpose="analysis: fresh seeds recorded after the gate was specified, used to "
        "confirm the attribution and the oracle away from the legacy block",
        frozen=True,
        stream_dir_name="learnability_20260917/analysis_fresh",
    ),
}

# A name that must never describe a split again: the 20000-20099 block was reported as a
# test set once and has been read since. Keeping the word here is what stops a later
# phase from using it in a verdict title.
RETIRED_NAMES = ("held-out", "heldout", "test")


def split_of(name: str) -> DataSplit:
    try:
        return SPLITS[name]
    except KeyError as error:
        known = ", ".join(sorted(SPLITS))
        raise KeyError(f"unknown split {name!r}; known splits: {known}") from error


def all_seeds() -> set[int]:
    """Every seed any split claims. Two splits sharing a seed is a leak by definition."""
    seeds: set[int] = set()
    for split in SPLITS.values():
        seeds.update(split.seeds)
    return seeds


def run_dir(repo: Path, split_name: str) -> Path:
    return split_of(split_name).run_dir(repo)


def stream_path(repo: Path, split_name: str, seed: int) -> Path:
    return split_of(split_name).stream_dir(repo) / f"s{seed}.json"


# Where one arm's logs live inside a split's run directory. An arm is a policy plus the
# flags it was invoked with, and two arms differ by construction -- the advisory planner and
# the binding one differ by exactly one flag -- so each gets its own directory *and* its own
# file names. A shared directory with a label in the name was enough for a batch script to
# point the wrong flag at the wrong output and leave files whose names said one arm and
# whose contents were another.
ARMS_DIR = "arms"


def arm_dir(repo: Path, split_name: str, arm: str) -> Path:
    return run_dir(repo, split_name) / ARMS_DIR / arm


def arm_paths(
    repo: Path, split_name: str, arm: str, seed: int
) -> tuple[Path, Path]:
    """The lifecycle log and the decision trace of one arm on one seed."""
    directory = arm_dir(repo, split_name, arm)
    return directory / f"rp_{arm}_s{seed}.jsonl", directory / f"tr_{arm}_s{seed}.jsonl"
