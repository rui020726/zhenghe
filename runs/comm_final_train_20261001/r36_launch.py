"""ROUND36 的受保护训练启动器：给一次 seed 训练套上 wall-clock 上限并记账。

父进程做四件事：核对额度与起点目录、按 ROUND32 归档同 seed 实际耗时的 1.5 倍设上限并以子进程
超时强制执行、把子进程输出整段落盘为训练日志、按真实 ledger 与训练返回值记账。训练本身全部
发生在 `r36_worker.py` 调用的真实 `v2_train.train_seed` 里。

    /d/project_env/python.exe r36_launch.py --seed 11
    /d/project_env/python.exe r36_launch.py --seed 22
    /d/project_env/python.exe r36_launch.py --seed 33

失败储备：`--reserve` 只允许一次，且必须先有登记在册的失败原因（`record_failure`）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import r36_common as r36

PYTHON = Path(sys.executable)


def _tail(text: str, lines: int = 40) -> list[str]:
    return [line for line in text.splitlines()[-lines:] if line.strip()]


def run(training_seed: int, *, reserve: bool = False) -> int:
    state = r36.book()
    tag = f"seed{training_seed}" + ("_reserve" if reserve else "")
    if not reserve and state["training_launches"] >= r36.MAX_TRAINING_LAUNCHES:
        raise SystemExit(f"{r36.MAX_TRAINING_LAUNCHES} training launches are authorised and "
                         f"{state['training_launches']} are already booked")
    budget = r36.round_constructions()
    if budget["total"] + r36.EPISODES_PER_SEED > budget["allowed_now"]:
        raise SystemExit(f"this launch would need {r36.EPISODES_PER_SEED} episode constructions "
                         f"but only {budget['allowed_now'] - budget['total']} of the authorised "
                         f"{budget['allowed_now']} remain (booked: {budget['per_directory']})")
    ceiling = r36.wall_clock_ceiling(training_seed)
    directory = r36.seed_directory(training_seed, reserve=reserve)
    directory.mkdir(parents=True, exist_ok=True)
    console = directory / "launch_console.log"
    identity = r36.code_identity()
    freeze = r36.freeze_identity()
    r36.write_seed_json(training_seed, "FREEZE_MANIFEST.json", {
        "kind": "round36-per-launch-freeze-manifest", "launch_tag": tag,
        "training_seed": int(training_seed), "reserve": bool(reserve),
        "training_version": r36.TRAINING_VERSION,
        "reward_timing_version": r36.REWARD_TIMING_VERSION,
        "trainer_tag": r36.v2_trainer.TRAINER_TAG,
        "loss_version": r36.v2_trainer.LOSS_VERSION,
        "state_schema": r36.v2_trainer.STATE_SCHEMA,
        "gae_schema": r36.v2_trainer.GAE_SCHEMA,
        "optimizer_schema": r36.v2_trainer.OPTIMIZER_SCHEMA,
        "reward_timing_schema": r36.v2_trainer.REWARD_TIMING_SCHEMA,
        "inherited_freeze_token": freeze,
        "episodes_per_seed": r36.EPISODES_PER_SEED,
        "batches_per_seed": r36.BATCHES_PER_SEED,
        "seeds_frozen_from_stage_c": list(r36.SEEDS),
        "code_identity_at_launch": identity,
        "wall_clock_ceiling_seconds": ceiling,
        "written_at": time.strftime("%F %T")}, reserve=reserve, refresh=True)

    command = [str(PYTHON), str(r36.HERE / "r36_worker.py"), "--seed", str(training_seed)]
    if reserve:
        command.append("--reserve")
    print(f"launching {tag}: ceiling {ceiling}s, command {' '.join(command)}", flush=True)
    started = time.time()
    timed_out = False
    return_code = -1
    with console.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# ROUND36 launch {tag}\n# command: {' '.join(command)}\n"
                     f"# wall-clock ceiling: {ceiling}s (1.5x ROUND32 archived actual for this "
                     f"seed)\n")
        handle.flush()
        try:
            child = subprocess.run(command, cwd=str(r36.HERE), stdout=handle,
                                   stderr=subprocess.STDOUT, timeout=ceiling, check=False)
            return_code = int(child.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = -9
            handle.write("\n*** WALL-CLOCK CEILING EXCEEDED: the process was killed ***\n")
        handle.write(f"\n# exit code {return_code}, parent-measured wall clock "
                     f"{round(time.time() - started, 2)}s\n")
    wall = round(time.time() - started, 2)

    record_path = directory / "TRAINING_RECORD.json"
    training_record = r36.read_json(record_path) if record_path.exists() else None
    ledger = r36.starts_in(training_seed, reserve=reserve)
    steps = int(training_record["result"]["optimizer_steps"]) if training_record else 0
    outcome = "completed" if return_code == 0 and training_record else (
        "wall_clock_ceiling_exceeded" if timed_out else "failed")
    detail = {
        "launch_tag": tag, "pid": os.getpid(), "worker_command": command,
        "exit_code": return_code, "outcome": outcome,
        "parent_wall_clock_seconds": wall, "wall_clock_ceiling_seconds": ceiling,
        "within_ceiling": wall <= ceiling,
        "episode_constructions_in_this_seed_directory": ledger["starts"],
        "failed_starts_in_this_seed_directory": ledger["failed"],
        "optimizer_steps": steps,
        "training_record": training_record,
        "console_log": str(console),
        "console_tail_last": _tail(console.read_text("utf-8"), 12),
        "freeze_identity_used": freeze, "code_identity_at_launch": identity,
        "note": "loss and reward values are recorded only as evidence that the frozen protocol "
                "executed; this round reports no learning-effect numbers"}
    r36.write_seed_json(training_seed, "LAUNCH_RECORD.json", detail, reserve=reserve,
                        refresh=True)
    booked = r36.book_launch(tag=tag, training_seed=training_seed,
                             constructions=int(ledger["starts"]), optimizer_steps=steps,
                             wall_clock_seconds=wall, ceiling_seconds=ceiling, reserve=reserve,
                             detail={"outcome": outcome, "exit_code": return_code})
    print(json.dumps({"launch_tag": tag, "outcome": outcome, "exit_code": return_code,
                      "wall_clock_seconds": wall, "ceiling_seconds": ceiling,
                      "constructions": ledger["starts"], "optimizer_steps": steps,
                      "book_totals": {key: booked[key] for key in
                                      ("training_launches", "reserve_reruns",
                                       "episode_constructions", "optimizer_steps")}},
                     ensure_ascii=False, indent=1), flush=True)
    return 0 if outcome == "completed" else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="one guarded ROUND36 training launch")
    parser.add_argument("--seed", type=int, required=True, choices=list(r36.SEEDS))
    parser.add_argument("--reserve", action="store_true")
    arguments = parser.parse_args(sys.argv[1:])
    raise SystemExit(run(arguments.seed, reserve=arguments.reserve))
