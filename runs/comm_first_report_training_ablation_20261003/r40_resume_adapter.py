"""ROUND40 continuation adapter for a committed batch's reporting/resume seams.

The frozen training function is executed from its source unchanged apart from moving one
existing import above its checkpoint-resume branch. The wrapper then adds only the milestone
paths expected by the frozen worker's summary writer. It does not change samples, actions,
returns, loss, optimizer, RNG, or launch order.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _install_summary_and_resume_adapter(common: Any, runner: Any, trainer: Any,
                                         freeze: dict[str, Any]) -> dict[str, Any]:
    import r40_train

    r40_train.common = common
    r40_train.runner = runner
    r40_train.trainer = trainer
    if getattr(r40_train.train_seed, "_round40_continuation_adapter", False):
        return {"already_installed": True}

    original_source = inspect.getsource(r40_train.train_seed)
    old_line = "    checkpoint = _rolling_path(training_seed)\n"
    new_line = "    from v2_controller import parameter_identity\n" + old_line
    if original_source.count(old_line) != 1:
        raise RuntimeError("frozen train_seed source does not match the audited resume patch")
    patched_source = original_source.replace(old_line, new_line, 1)
    original_source_sha = hashlib.sha256(original_source.encode("utf-8")).hexdigest()
    patched_source_sha = hashlib.sha256(patched_source.encode("utf-8")).hexdigest()
    exec(compile(patched_source, str(Path(r40_train.__file__)), "exec"), r40_train.__dict__)
    resumed_train_seed = r40_train.train_seed
    resumed_train_seed._round40_resume_import_patch = True

    def train_seed_with_milestone_paths(training_seed: int, *, episodes: int,
                                        freeze: dict[str, Any],
                                        reference_cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
        result = resumed_train_seed(training_seed, episodes=episodes, freeze=freeze,
                                    reference_cache=reference_cache)
        paths = {}
        for milestone in (0, 64, 128):
            path = r40_train._milestone_path(training_seed, milestone, freeze)
            if path.is_file():
                paths[str(milestone)] = str(path)
        result = dict(result)
        result["milestone_checkpoints"] = paths
        return result

    train_seed_with_milestone_paths._round40_continuation_adapter = True
    r40_train.train_seed = train_seed_with_milestone_paths

    rolling = common.CHECKPOINTS / "v2_train_s11.pt"
    if not rolling.is_file():
        raise RuntimeError("ROUND40 rolling checkpoint is missing before continuation adapter setup")
    _controller, state, identity = trainer.load_checkpoint_read_only(rolling, freeze=freeze)
    use_record = {
        "kind": "round40-continuation-adapter-use/1",
        "recorded_at_unix": time.time(),
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": _sha256(Path(__file__).resolve()),
        "training_version": state["training_version"],
        "comm_action_eligibility": state["comm_action_eligibility"],
        "resume_source": str(rolling),
        "resume_source_sha256": _sha256(rolling),
        "episodes_done_before_resume": int(state["episodes_done"]),
        "updates_before_resume": int(state["updates"]),
        "optimizer_steps_before_resume": int(state["optimizer_steps"]),
        "read_only_identity": identity,
        "function_source_sha256_before_import_patch": original_source_sha,
        "function_source_sha256_after_import_patch": patched_source_sha,
        "exact_source_patch": {
            "before": old_line.rstrip("\n"),
            "inserted_before": "from v2_controller import parameter_identity",
            "purpose": "make the existing parameter_identity local available on the checkpoint-resume branch",
        },
        "post_return_field": "milestone_checkpoints",
        "scope_excludes_rollouts_actions_returns_loss_optimizer_rng_and_checkpoint_bytes": True,
        "environment_constructed_by_adapter_setup": False,
        "optimizer_steps_by_adapter_setup": 0,
    }
    use_path = common.OUTPUT_DIR / "training_progress" / (
        f"continuation_adapter_use_{time.time_ns()}_{os.getpid()}.json"
    )
    _write_exclusive(use_path, use_record)
    return {"already_installed": False, "runtime_use_record": str(use_path),
            "function_source_sha256_before": original_source_sha,
            "function_source_sha256_after": patched_source_sha,
            "episodes_done_before_resume": int(state["episodes_done"]),
            "updates_before_resume": int(state["updates"]),
            "optimizer_steps_before_resume": int(state["optimizer_steps"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="continue the frozen ROUND40 worker")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--phase", choices=("full", "training", "models", "rules"),
                        default="full")
    args = parser.parse_args(argv)

    import r40_worker

    _root, common, runner, trainer, _controller, freeze = r40_worker._load_context(
        args.input_root)
    adapter = _install_summary_and_resume_adapter(common, runner, trainer, freeze)
    print(json.dumps({"continuation_adapter": adapter}, ensure_ascii=False), flush=True)
    result = r40_worker.run(args.input_root, args.phase)
    print(json.dumps({"phase": result["phase"], "ledger": {
        key: result["ledger"][key] for key in ("normal_used", "reserve_used", "total_used")}},
        ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
