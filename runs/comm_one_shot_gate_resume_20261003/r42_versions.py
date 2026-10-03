"""ROUND42 checkpoint and training identity gate, layered over the ROUND40 loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from r42_one_shot import COMM_ACTION_SEMANTICS
from r40_versions import COMM_ACTION_ELIGIBILITY, REWARD_TIMING_VERSION


TRAINING_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/"
    "published-completion-event-at-relay-tick/2"
)
CHECKPOINT_WRITE_CALLS = 0


def install(trainer: Any, runner: Any, *, common: Any, freeze: dict[str, Any],
            round40_freeze: dict[str, Any]) -> None:
    """Install strict one-shot identity while retaining ROUND40's frozen base checks."""
    global CHECKPOINT_WRITE_CALLS
    if getattr(trainer, "_round42_one_shot_installed", False):
        trainer.TRAINING_VERSION = TRAINING_VERSION
        runner.TRAINING_VERSION = TRAINING_VERSION
        runner.REWARD_TIMING_VERSION = REWARD_TIMING_VERSION
        return

    import r40_versions

    r40_versions.install(trainer, runner, freeze=round40_freeze)
    trainer.TRAINING_VERSION = TRAINING_VERSION
    trainer.REWARD_TIMING_VERSION = REWARD_TIMING_VERSION
    runner.TRAINING_VERSION = TRAINING_VERSION
    runner.REWARD_TIMING_VERSION = REWARD_TIMING_VERSION

    old_fresh = trainer.fresh_training_state

    def fresh_training_state(training_seed: int) -> dict[str, Any]:
        state = old_fresh(training_seed)
        state.update({
            "training_version": TRAINING_VERSION,
            "reward_timing_version": REWARD_TIMING_VERSION,
            "comm_action_eligibility": COMM_ACTION_ELIGIBILITY,
            "comm_action_semantics": COMM_ACTION_SEMANTICS,
        })
        return state

    trainer.fresh_training_state = fresh_training_state

    def save_checkpoint(controller: Any, optimizer: torch.optim.Optimizer,
                        state: dict[str, Any], path: Path, *,
                        freeze: dict[str, Any]) -> Path:
        global CHECKPOINT_WRITE_CALLS
        expected = {
            "training_version": TRAINING_VERSION,
            "reward_timing_version": REWARD_TIMING_VERSION,
            "comm_action_eligibility": COMM_ACTION_ELIGIBILITY,
            "comm_action_semantics": COMM_ACTION_SEMANTICS,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(f"ROUND42 save refused state with {key}={state.get(key)!r}")
        if not isinstance(freeze, dict) or freeze.get("comm_action_semantics") != COMM_ACTION_SEMANTICS:
            raise RuntimeError("ROUND42 save refused a checkpoint without its one-shot freeze identity")
        payload = {
            "schema": trainer.STATE_SCHEMA,
            "trainer_tag": trainer.TRAINER_TAG,
            **expected,
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
        if temporary.exists():
            raise RuntimeError(f"stale ROUND42 checkpoint temporary needs inspection: {temporary}")
        try:
            with temporary.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            CHECKPOINT_WRITE_CALLS += 1
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    trainer.save_checkpoint = save_checkpoint

    def _load_payload(path: Path) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("training_state")
        if not isinstance(state, dict):
            raise RuntimeError(f"{path.name}: missing training_state")
        for key, value in {
            "training_version": TRAINING_VERSION,
            "reward_timing_version": REWARD_TIMING_VERSION,
            "comm_action_eligibility": COMM_ACTION_ELIGIBILITY,
            "comm_action_semantics": COMM_ACTION_SEMANTICS,
        }.items():
            if payload.get(key) != value or state.get(key) != value:
                raise RuntimeError(
                    f"{path.name}: {key} mismatch; checkpoint is read-only under "
                    f"ROUND42 {COMM_ACTION_SEMANTICS}"
                )
        return payload

    def load_checkpoint(path: Path, *, freeze: dict[str, Any]):
        # Semantic identity is checked before constructing a controller/optimizer or restoring
        # RNG.  The pre-existing base validator then checks all ROUND40 loss/GAE/freeze fields.
        payload = _load_payload(path)
        state = trainer._validate_checkpoint_base(path, payload, freeze=freeze)
        if freeze.get("comm_action_semantics") != COMM_ACTION_SEMANTICS:
            raise RuntimeError("ROUND42 resume caller supplied a non-one-shot freeze")
        controller = trainer._restore_controller(payload, state)
        optimizer = torch.optim.Adam(controller.parameters, lr=common.LEARNING_RATE)
        optimizer.load_state_dict(payload["optimizer"])
        trainer.restore_rng(payload.get("rng"))
        return controller, optimizer, state

    trainer.load_checkpoint = load_checkpoint

    base_read_only = trainer.load_checkpoint_read_only

    def load_checkpoint_read_only(path: Path, *, freeze: dict[str, Any] | None = None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        stored_state = payload.get("training_state")
        stored_semantics = payload.get("comm_action_semantics")
        state_semantics = (stored_state.get("comm_action_semantics")
                           if isinstance(stored_state, dict) else None)
        if (stored_semantics is not None and stored_semantics != COMM_ACTION_SEMANTICS):
            raise RuntimeError(f"{path.name}: unexpected action semantics in read-only load")
        if state_semantics is not None and state_semantics != COMM_ACTION_SEMANTICS:
            raise RuntimeError(f"{path.name}: training_state action semantics mismatch")
        selected_freeze = freeze
        if selected_freeze is None:
            embedded = payload.get("freeze")
            if not isinstance(embedded, dict):
                raise RuntimeError(f"{path.name}: checkpoint has no embedded freeze identity")
            selected_freeze = embedded
        # The legacy controller constructor seeds torch from the stored training seed. Preserve
        # the caller's ambient streams around that construction; this still does not restore the
        # checkpoint RNG and keeps read-only evaluation/gates from changing a live action stream.
        ambient_rng = trainer.rng_state()
        try:
            controller, state, identity = base_read_only(path, freeze=selected_freeze)
        finally:
            trainer.restore_rng(ambient_rng)
        identity.update({
            "comm_action_semantics": stored_semantics,
            "action_semantics_interpretation": (
                COMM_ACTION_SEMANTICS if stored_semantics == COMM_ACTION_SEMANTICS
                else "round40-first-report-only/1_adapted_to_one-shot-send-skip/1"
            ),
            "read_only_for_round40": True,
            "read_only_for_round42": True,
            "ambient_rng_preserved": True,
        })
        if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
            raise RuntimeError(f"{path.name}: read-only loader restored optimizer or RNG")
        return controller, state, identity

    trainer.load_checkpoint_read_only = load_checkpoint_read_only
    trainer._round42_one_shot_installed = True


def add_row_identity(row: dict[str, Any], *, checkpoint: dict[str, Any] | None = None) -> None:
    row["training_version"] = TRAINING_VERSION
    row["reward_timing_version"] = REWARD_TIMING_VERSION
    row["comm_action_eligibility"] = COMM_ACTION_ELIGIBILITY
    row["comm_action_semantics"] = COMM_ACTION_SEMANTICS
    if checkpoint is not None:
        row["checkpoint_identity"] = dict(checkpoint)


__all__ = [
    "CHECKPOINT_WRITE_CALLS", "COMM_ACTION_ELIGIBILITY", "COMM_ACTION_SEMANTICS",
    "REWARD_TIMING_VERSION", "TRAINING_VERSION", "add_row_identity", "install",
]
