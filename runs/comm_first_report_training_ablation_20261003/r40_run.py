"""ROUND40 runner seam: shared first-report collector, claims, and row identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import r40_common as r40
from r40_eligibility import COMM_ACTION_ELIGIBILITY, FirstReportOnlyCollector
from r40_versions import TRAINING_VERSION, add_row_identity, install


def install_versions(common: Any, runner: Any, trainer: Any, *, freeze: dict[str, Any]) -> None:
    install(trainer, runner, freeze=freeze)
    runner.common = common


def _finish(path: Path, *, ok: bool, detail: str) -> None:
    r40.finish_launch(path, ok=ok, detail=detail)


class _BookedInR40Ledger:
    """Satisfy the archived row writer without creating a second launch ledger."""

    def record(self, **_kwargs: Any) -> None:
        return None


def _claim(logical_job: str, *, attempt: int, metadata: dict[str, Any]) -> Path:
    category = "normal" if int(attempt) == 1 else "reserve"
    return r40.claim_launch(logical_job, category=category, attempt=attempt,
                            metadata=metadata)


def run_reference(common: Any, runner: Any, scenario_seed: int) -> dict[str, Any]:
    logical = f"reference_immediate_s{int(scenario_seed)}"
    attempt = r40.next_attempt(logical)
    claim = _claim(logical, attempt=attempt,
                   metadata={"kind": "immediate_reference", "scenario_seed": scenario_seed})
    label = f"ref_s{int(scenario_seed)}"
    try:
        row = runner._run_episode(label, "reference", int(scenario_seed),
                                  policy="immediate", attempt=attempt)
        row["reference_arm"] = "immediate"
        row["action_rng_seed"] = None
        add_row_identity(row)
        runner._booked(label, "reference", row, _BookedInR40Ledger())
        _finish(claim, ok=True,
                detail=f"immediate scenario={scenario_seed} on_time={row['on_time']} "
                       f"bytes={row['spent_bytes']}")
        return row
    except Exception as error:
        _finish(claim, ok=False, detail=f"{type(error).__name__}: {error}")
        raise


def run_training_episode(common: Any, runner: Any, controller: Any, *, training_seed: int,
                         episode_index: int, scenario_seed: int, lam: float) -> dict[str, Any]:
    logical = f"new_training_s{int(training_seed)}_e{int(episode_index)}"
    attempt = r40.next_attempt(logical)
    claim = _claim(logical, attempt=attempt,
                   metadata={"kind": "new_training", "training_seed": training_seed,
                             "episode_index": episode_index, "scenario_seed": scenario_seed})
    label = f"train_s{int(training_seed)}_e{int(episode_index)}_a{attempt}"
    collector = FirstReportOnlyCollector(controller, greedy=False)
    try:
        row = runner._run_episode(
            label, "train", int(scenario_seed), policy="learned", attach=collector.policy(),
            attempt=attempt, controller=controller, collector=collector,
            action_rng_seed=runner.action_seed(training_seed, episode_index),
            tag=f"r40_train_s{training_seed}_e{episode_index}",
        )
        row.update({
            "training_seed": int(training_seed),
            "episode_index": int(episode_index),
            "lambda_at_rollout": float(lam),
            "lambda": float(lam),
            "training_arm": "NEW",
            "action_rng_seed": runner.action_seed(training_seed, episode_index),
        })
        add_row_identity(row)
        runner._booked(label, "train", row, _BookedInR40Ledger())
        _finish(claim, ok=True,
                detail=f"NEW seed={training_seed} episode={episode_index} "
                       f"on_time={row['on_time']} bytes={row['spent_bytes']}")
    except Exception as error:
        _finish(claim, ok=False, detail=f"{type(error).__name__}: {error}")
        raise
    row["collector"] = collector
    row["controller"] = controller
    return row


def reference_cache(common: Any) -> dict[int, dict[str, Any]]:
    cache: dict[int, dict[str, Any]] = {}
    for path in sorted(common.ROWS.glob("v2_reference_s*_a*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        scenario = int(row["scenario_seed"])
        if row.get("reference_arm") != "immediate":
            continue
        if scenario in cache and cache[scenario]["behaviour_sha256"] != row["behaviour_sha256"]:
            raise RuntimeError(f"scenario {scenario} has conflicting immediate references")
        cache[scenario] = row
    return cache


def train_seed(common: Any, runner: Any, trainer: Any, *, training_seed: int,
               episodes: int, freeze: dict[str, Any]) -> dict[str, Any]:
    import r40_train

    install_versions(common, runner, trainer, freeze=freeze)
    r40_train.common = common
    r40_train.runner = runner
    r40_train.trainer = trainer
    cache_rows = reference_cache(common)
    cache = {scenario: row for scenario, row in cache_rows.items()}
    missing = sorted(set(common.scenario_order()[:episodes]) - set(cache))
    if missing:
        raise RuntimeError(f"ROUND40 immediate reference cache missing {missing}")
    return r40_train.train_seed(training_seed, episodes=episodes, freeze=freeze,
                                reference_cache=cache)


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_eval_model(trainer: Any, *, arm: str, training_seed: int,
                    new_freeze: dict[str, Any], input_root: Path,
                    common: Any) -> tuple[Any, dict[str, Any]]:
    if arm == "OLD":
        path = (input_root / "runs" / "comm_final_train_20261001" / f"seed{training_seed}"
                / "checkpoints" / f"v2_train_s{training_seed}_ep128.pt")
        controller, state, identity = trainer.load_checkpoint_read_only(path)
    elif arm == "NEW":
        path = common.CHECKPOINTS / f"v2_train_s{training_seed}_ep128.pt"
        if not path.is_file():
            raise RuntimeError(f"NEW final milestone is missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload.get("training_version") != TRAINING_VERSION
                or payload.get("comm_action_eligibility") != COMM_ACTION_ELIGIBILITY):
            raise RuntimeError(f"NEW checkpoint identity mismatch: {path}")
        controller, state, identity = trainer.load_checkpoint_read_only(path, freeze=new_freeze)
    else:
        raise ValueError(f"unknown model arm {arm!r}")
    identity.update({
        "arm": arm,
        "path": str(path),
        "file_sha256": _checkpoint_digest(path),
        "training_seed": int(training_seed),
        "episodes_done": int(state.get("episodes_done", -1)),
        "updates": int(state.get("updates", -1)),
        "optimizer_steps": int(state.get("optimizer_steps", -1)),
        "lambda": float(state.get("lambda", float("nan"))),
        "parameter_identity": controller.parameter_summary(),
    })
    return controller, identity


def run_model_eval(common: Any, runner: Any, trainer: Any, *, arm: str,
                   training_seed: int, mode: str, scenario_seed: int, timepoint: str,
                   controller: Any, checkpoint_identity: dict[str, Any]) -> dict[str, Any]:
    logical = (f"model_eval_{arm.lower()}_s{training_seed}_{mode}_"
               f"scene{scenario_seed}")
    attempt = r40.next_attempt(logical)
    claim = _claim(logical, attempt=attempt,
                   metadata={"kind": "model_development_eval", "arm": arm,
                             "training_seed": training_seed, "mode": mode,
                             "scenario_seed": scenario_seed})
    greedy = mode == "greedy"
    collector = FirstReportOnlyCollector(controller, greedy=greedy)
    block = f"{timepoint}_{mode}"
    label = f"eval_{arm.lower()}_s{training_seed}_{block}_s{scenario_seed}_a{attempt}"
    action_seed = runner.eval_action_seed(training_seed, scenario_seed, mode)
    try:
        row = runner._run_episode(
            label, "eval", int(scenario_seed), policy="learned", attach=collector.policy(),
            attempt=attempt, controller=controller, collector=collector,
            action_rng_seed=action_seed,
            tag=f"r40_eval_{arm.lower()}_s{training_seed}_{block}",
            checkpoint_load_identity=checkpoint_identity,
        )
        row.update({"training_seed": int(training_seed), "model_arm": arm,
                    "timepoint": timepoint, "eval_mode": mode, "eval_block": block,
                    "action_rng_seed": action_seed,
                    "checkpoint_parameter_identity": checkpoint_identity["parameter_identity"]})
        add_row_identity(row, checkpoint=checkpoint_identity)
        runner._booked(label, "eval", row, _BookedInR40Ledger())
        _finish(claim, ok=True,
                detail=f"{arm} seed={training_seed} mode={mode} scene={scenario_seed} "
                       f"on_time={row['on_time']} bytes={row['spent_bytes']}")
    except Exception as error:
        _finish(claim, ok=False, detail=f"{type(error).__name__}: {error}")
        raise
    return row


def run_rule(common: Any, runner: Any, *, arm: str, scenario_seed: int) -> dict[str, Any]:
    import v2_rules

    if arm not in ("late_discovery_gate", "always_hold"):
        raise ValueError(f"ROUND40 rule arm is not authorized: {arm!r}")
    logical = f"rule_{arm}_scene{int(scenario_seed)}"
    attempt = r40.next_attempt(logical)
    claim = _claim(logical, attempt=attempt,
                   metadata={"kind": "rule_development_eval", "arm": arm,
                             "scenario_seed": scenario_seed})
    policy = v2_rules.build_rule(arm, scenario_seed)
    label = f"rule_{arm}_s{int(scenario_seed)}_a{attempt}"
    try:
        row = runner._run_episode(
            label, "rule", int(scenario_seed), policy="learned", attach=policy,
            attempt=attempt,
            tag=f"r40_rule_{arm}",
        )
        row["rule_identity"] = v2_rules.rule_identity(arm)
        row["rule_identity"]["built_parameters"] = dict(policy.parameters())
        row["model_arm"] = arm
        add_row_identity(row)
        runner._booked(label, "rule", row, _BookedInR40Ledger())
        _finish(claim, ok=True,
                detail=f"rule={arm} scene={scenario_seed} on_time={row['on_time']} "
                       f"bytes={row['spent_bytes']}")
    except Exception as error:
        _finish(claim, ok=False, detail=f"{type(error).__name__}: {error}")
        raise
    return row


__all__ = [
    "FirstReportOnlyCollector", "load_eval_model", "reference_cache", "run_model_eval",
    "run_reference", "run_rule", "run_training_episode", "train_seed",
]
