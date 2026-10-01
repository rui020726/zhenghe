"""ROUND36 训练启动的工作进程：一次调用真实 `v2_train.train_seed`，跑满一个 seed 的 128 集。

这里没有任何数学：真实 rollout（`v2_run.run_training_episode` → `_install_taps` →
`PublishedProgressRewardCollector`）、真实 PPO/对偶更新（`v2_trainer`）、真实 checkpoint 保存
（`v2_trainer.save_checkpoint` / `v2_train.save_milestone`）都在 ROUND24+ROUND34 的模块里执行。
本进程只做三件事：把输出路径重绑定到本 seed 自己的目录、证明起点是登记初值（不是任何旧
checkpoint 的热启动）、把训练返回值原样落盘。

    /d/project_env/python.exe r36_worker.py --seed 11

由 `r36_launch.py` 以子进程方式调用并施加固 wall-clock 上限；不要手工重复运行（每次运行都消耗
一整份额度）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import r36_common as r36
import v2_common
import v2_train
import v2_trainer


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="one authorised ROUND36 training launch")
    parser.add_argument("--seed", type=int, required=True, choices=list(r36.SEEDS))
    parser.add_argument("--reserve", action="store_true",
                        help="the single authorised re-run, into its own directory")
    arguments = parser.parse_args(argv)

    paths = r36.configure(arguments.seed, reserve=arguments.reserve)
    prior_weights = sorted(path.name for path in paths["CHECKPOINTS"].glob("*.pt"))
    if prior_weights:
        raise SystemExit(
            f"refusing to start seed {arguments.seed} in a directory that already holds "
            f"{prior_weights}: every authorised launch must begin from the registered "
            "initialisation, and a crashed run's artefacts stay where they are")

    freeze = r36.freeze_identity()
    cache = r36.reference_cache()
    if cache["missing"]:
        raise SystemExit(f"the read-only reference cache does not cover {cache['missing'][:8]}")
    references = cache["by_scenario"]
    needed = set(v2_common.scenario_order()[:r36.EPISODES_PER_SEED])
    missing = sorted(scenario for scenario in needed if scenario not in references)
    if missing:
        raise SystemExit(f"reference cache is missing scenarios {missing[:8]}")

    from v2_controller import fresh_controller, parameter_identity

    registered_init = parameter_identity(fresh_controller(arguments.seed))["parameter_sha256"]
    print(f"seed {arguments.seed}: registered initialisation parameter_sha256={registered_init}",
          flush=True)

    started = time.time()
    result: dict[str, Any] = v2_train.train_seed(arguments.seed,
                                                 episodes=r36.EPISODES_PER_SEED,
                                                 freeze=freeze, reference_cache=references)
    wall = round(time.time() - started, 2)
    payload = {
        "kind": "round36-seed-training-record", "round": "36",
        "training_seed": int(arguments.seed),
        "reserve": bool(arguments.reserve),
        "pid": __import__("os").getpid(),
        "episodes_requested": r36.EPISODES_PER_SEED,
        "registered_initialisation": registered_init,
        "worker_wall_clock_seconds": wall,
        "result": {key: result[key] for key in
                   ("training_seed", "episodes_done", "updates", "optimizer_steps", "lambda",
                    "seconds", "checkpoint", "controller_parameters")},
        "lambda_history": result["lambda_history"],
        "freeze_identity_used": freeze,
        "reference_cache_files_read_only": cache["files_read_only"],
        "ledger_after": {"directory": paths["LOGS"].name,
                         "starts": r36.starts_in(arguments.seed, reserve=arguments.reserve)},
        "bound_paths": {key: str(value) for key, value in paths.items()},
        "note": "loss and reward values in this record show that the frozen training protocol "
                "executed; they are not evidence about learning quality and are not reported "
                "as results in this round"}
    r36.write_seed_json(arguments.seed, "TRAINING_RECORD.json", payload,
                        reserve=arguments.reserve, refresh=True)
    print(json.dumps({"seed": arguments.seed, "episodes_done": result["episodes_done"],
                      "updates": result["updates"], "optimizer_steps": result["optimizer_steps"],
                      "lambda": result["lambda"], "worker_seconds": wall,
                      "parameters": result["controller_parameters"]["parameter_sha256"]},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
