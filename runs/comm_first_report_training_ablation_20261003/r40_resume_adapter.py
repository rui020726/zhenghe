"""ROUND40 continuation shim for a missing checkpoint-index return field.

This wrapper calls the frozen training function unchanged. After it returns, it adds the
versioned milestone paths expected by the worker summary writer. It does not modify samples,
controller state, optimizer state, RNG state, or launch behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _install_summary_field(common: Any, runner: Any, trainer: Any, freeze: dict[str, Any]) -> None:
    import r40_train

    r40_train.common = common
    r40_train.runner = runner
    r40_train.trainer = trainer
    if getattr(r40_train.train_seed, "_round40_milestone_return_adapter", False):
        return
    frozen_train_seed = r40_train.train_seed

    def train_seed_with_milestone_paths(training_seed: int, *, episodes: int,
                                        freeze: dict[str, Any],
                                        reference_cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
        result = frozen_train_seed(training_seed, episodes=episodes, freeze=freeze,
                                   reference_cache=reference_cache)
        paths = {}
        for milestone in (0, 64, 128):
            path = r40_train._milestone_path(training_seed, milestone, freeze)
            if path.is_file():
                paths[str(milestone)] = str(path)
        result = dict(result)
        result["milestone_checkpoints"] = paths
        return result

    train_seed_with_milestone_paths._round40_milestone_return_adapter = True
    r40_train.train_seed = train_seed_with_milestone_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="continue the frozen ROUND40 worker")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--phase", choices=("full", "training", "models", "rules"),
                        default="full")
    args = parser.parse_args(argv)

    import r40_worker

    _root, common, runner, trainer, _controller, freeze = r40_worker._load_context(
        args.input_root)
    _install_summary_field(common, runner, trainer, freeze)
    result = r40_worker.run(args.input_root, args.phase)
    print(json.dumps({"phase": result["phase"], "ledger": {
        key: result["ledger"][key] for key in ("normal_used", "reserve_used", "total_used")}},
        ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
