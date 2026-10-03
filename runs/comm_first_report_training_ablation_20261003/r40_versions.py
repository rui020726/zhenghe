"""ROUND40 training identity and resume boundary.

This module installs its versioned seam onto the archived Stage C modules at the R40 entry
point. The archived files and their historical behavior remain intact in Git history; all
ROUND40 writers go through these wrappers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from r40_eligibility import COMM_ACTION_ELIGIBILITY

REWARD_TIMING_VERSION = "published-completion-event-at-relay-tick/2"
TRAINING_VERSION = (
    "constrained-cost-v2/first-report-only/1/"
    "published-completion-event-at-relay-tick/2"
)


def install(trainer: Any, runner: Any, *, freeze: dict[str, Any]) -> None:
    """Install ROUND40 fields and make the real training restore entry fail closed."""
    if getattr(trainer, "_round40_first_report_installed", False):
        runner.TRAINING_VERSION = TRAINING_VERSION
        runner.REWARD_TIMING_VERSION = REWARD_TIMING_VERSION
        return
    trainer.TRAINING_VERSION = TRAINING_VERSION
    trainer.REWARD_TIMING_VERSION = REWARD_TIMING_VERSION
    runner.TRAINING_VERSION = TRAINING_VERSION
    runner.REWARD_TIMING_VERSION = REWARD_TIMING_VERSION

    old_fresh_state = trainer.fresh_training_state

    def fresh_training_state(training_seed: int) -> dict[str, Any]:
        state = old_fresh_state(training_seed)
        state["comm_action_eligibility"] = COMM_ACTION_ELIGIBILITY
        return state

    trainer.fresh_training_state = fresh_training_state

    def save_checkpoint(controller: Any, optimizer: torch.optim.Optimizer,
                        state: dict[str, Any], path: Path, *,
                        freeze: dict[str, Any]) -> Path:
        if state.get("comm_action_eligibility") != COMM_ACTION_ELIGIBILITY:
            raise RuntimeError("ROUND40 save refused a training state without action eligibility")
        payload = {
            "schema": trainer.STATE_SCHEMA,
            "trainer_tag": trainer.TRAINER_TAG,
            "training_version": TRAINING_VERSION,
            "comm_action_eligibility": COMM_ACTION_ELIGIBILITY,
            "reward_timing_version": REWARD_TIMING_VERSION,
            "loss_version": trainer.LOSS_VERSION,
            "gae_schema": trainer.GAE_SCHEMA,
            "optimizer_schema": trainer.OPTIMIZER_SCHEMA,
            "reward_timing_schema": trainer.REWARD_TIMING_SCHEMA,
            "freeze": freeze,
            "controller": {key: value for key, value in controller.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "training_state": state,
            "rng": trainer.rng_state(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.writing")
        try:
            with temporary.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    trainer.save_checkpoint = save_checkpoint

    old_load_checkpoint = trainer.load_checkpoint

    def load_checkpoint(path: Path, *, freeze: dict[str, Any]):
        # Check the intervention identity before restoring controller, optimizer, or RNG.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("training_state")
        if (payload.get("comm_action_eligibility") != COMM_ACTION_ELIGIBILITY
                or not isinstance(state, dict)
                or state.get("comm_action_eligibility") != COMM_ACTION_ELIGIBILITY):
            raise RuntimeError(
                f"{path.name}: action eligibility version mismatch; this checkpoint is "
                "read-only under ROUND40 first-report-only/1"
            )
        if (payload.get("training_version") != TRAINING_VERSION
                or state.get("training_version") != TRAINING_VERSION):
            raise RuntimeError(
                f"{path.name}: training version mismatch; this checkpoint cannot resume "
                "ROUND40 training"
            )
        return old_load_checkpoint(path, freeze=freeze)

    trainer.load_checkpoint = load_checkpoint

    old_read_only = trainer.load_checkpoint_read_only

    def load_checkpoint_read_only(path: Path, *, freeze: dict[str, Any] | None = None):
        # Legacy OLD weights use their embedded ROUND36 freeze identity. This path restores
        # neither optimizer nor RNG and is only exposed to evaluation callers.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        stored_freeze = payload.get("freeze")
        if freeze is None:
            if not isinstance(stored_freeze, dict):
                raise RuntimeError(f"{path.name}: checkpoint has no embedded freeze identity")
            freeze = stored_freeze
        controller, state, identity = old_read_only(path, freeze=freeze)
        eligibility = payload.get("comm_action_eligibility")
        identity.update({
            "comm_action_eligibility": eligibility,
            "action_eligibility_interpretation": (
                COMM_ACTION_ELIGIBILITY if eligibility == COMM_ACTION_ELIGIBILITY
                else "legacy_first_and_refresh_action_space"
            ),
            "read_only_for_round40": True,
        })
        return controller, state, identity

    trainer.load_checkpoint_read_only = load_checkpoint_read_only
    trainer._round40_first_report_installed = True


def add_row_identity(row: dict[str, Any], *, checkpoint: dict[str, Any] | None = None) -> None:
    row["training_version"] = TRAINING_VERSION
    row["reward_timing_version"] = REWARD_TIMING_VERSION
    row["comm_action_eligibility"] = COMM_ACTION_ELIGIBILITY
    if checkpoint is not None:
        row["checkpoint_identity"] = dict(checkpoint)


__all__ = [
    "COMM_ACTION_ELIGIBILITY", "REWARD_TIMING_VERSION", "TRAINING_VERSION",
    "add_row_identity", "install",
]
