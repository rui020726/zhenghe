"""ROUND35 的两次真实启动（授权上限：2 次调用 / 16 次 episode 构造 / 100 次 optimizer.step）。

调用的是真实训练入口 `v2_train.train_seed`，因此 rollout 走 `v2_run.run_training_episode` →
`v2_run._install_taps` → `PublishedProgressRewardCollector`，更新走 `v2_trainer.ppo_update` /
`batch_g_hat` / `dual_update`，checkpoint 走 `v2_trainer.save_checkpoint` / `load_checkpoint`。
本文件不含任何数学：只有目录绑定、记账与上限守卫。

    python r35_launch.py --launch-tag launch1     # 连续 8 集（2 批），接线与账本
    python r35_launch.py --launch-tag launch2     # 4 集 -> 存 -> 真实恢复 -> 再 4 集

smoke 的 loss/回报数字只作为链路走通的记录，不作为学习效果证据（见报告）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from typing import Any

import r35_common as smoke
import v2_train  # the real training entry point (imported after the path binding)
import v2_trainer
from v2_controller import parameter_identity


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result[key] for key in
            ("training_seed", "episodes_done", "updates", "optimizer_steps", "lambda",
             "seconds", "checkpoint", "controller_parameters")}


def run(launch_tag: str, *, with_probe: bool = False) -> int:
    budget = smoke.ledger_starts_across_launches()
    remaining = budget["remaining"]
    planned = {"launch1": smoke.SMOKE_EPISODES, "launch2": 7}[launch_tag]
    paths = smoke.configure(launch_tag, cap=remaining)
    if planned > remaining:
        raise SystemExit(f"{launch_tag} plans {planned} episode constructions but only "
                         f"{remaining} of the authorised {smoke.MAX_EPISODE_CONSTRUCTIONS} "
                         f"remain (ledger: {budget})")
    probe = None
    if with_probe:
        import r35_probe

        probe = r35_probe.Probe()
        probe.install()
    identity = smoke.code_identity()
    freeze = smoke.freeze_identity()
    cache = smoke.reference_cache()
    references = {scenario: row for scenario, row in cache["by_scenario"].items()}
    needed = smoke.v2_common.scenario_order()[:smoke.SMOKE_EPISODES]
    missing = sorted(scenario for scenario in needed if scenario not in references)
    if missing:
        raise SystemExit(f"the read-only reference cache does not cover {missing[:8]}")
    book = smoke.smoke_book()
    if book["invocations"] >= smoke.MAX_LAUNCH_INVOCATIONS:
        raise SystemExit(f"only {smoke.MAX_LAUNCH_INVOCATIONS} invocations are authorised and "
                         f"{book['invocations']} are already booked")

    started = time.time()
    phases: list[dict[str, Any]] = []
    if launch_tag == "launch1":
        result = v2_train.train_seed(smoke.SMOKE_SEED, episodes=smoke.SMOKE_EPISODES,
                                     freeze=freeze, reference_cache=references)
        phases.append({"phase": "uninterrupted_8_episodes", "result": _summary(result)})
        constructions = smoke.SMOKE_EPISODES
    elif launch_tag == "launch2":
        first = v2_train.train_seed(smoke.SMOKE_SEED, episodes=4, freeze=freeze,
                                    reference_cache=references)
        phases.append({"phase": "first_half_then_save", "result": _summary(first)})
        saved = paths["CHECKPOINTS"] / "v2_train_s11_ep4_saved_for_resume.pt"
        shutil.copy2(v2_train.rolling_path(smoke.SMOKE_SEED), saved)
        # the real resume path: train_seed finds the rolling checkpoint and calls
        # v2_trainer.load_checkpoint, which restores controller + optimizer + RNG + state.
        # 7 rather than 8 because the crashed first attempt already spent one of the
        # authorised sixteen episode constructions (no refunds).
        second = v2_train.train_seed(smoke.SMOKE_SEED, episodes=7, freeze=freeze,
                                     reference_cache=references)
        phases.append({"phase": "resumed_after_a_cross_process_reload", "result": _summary(second),
                       "resume_source_checkpoint": str(saved),
                       "episodes_covered_after_resume": [4, 5, 6],
                       "note": "the resumed batch holds three episodes rather than four, so its "
                               "own update is a different batch from launch1's; what has to match "
                               "is the rollout state of episodes 4..6"})
        constructions = 7
    else:
        raise SystemExit(f"unknown launch tag {launch_tag!r}")

    steps = sum(int(phase["result"]["optimizer_steps"]) for phase in phases)
    detail = {"launch_tag": launch_tag, "pid": smoke.os.getpid(),
              "bound_paths": {key: str(value) for key, value in paths.items()},
              "phases": phases, "phases_optimizer_steps": steps,
              "wall_clock_seconds": round(time.time() - started, 2),
              "freeze_identity_used": freeze,
              "code_identity": identity,
              "reference_cache_files_read_only": cache["files_read_only"],
              "ledger_after": smoke.v2_common.ledger_report()["entries"],
              "note": "loss and reward values in these records show that the chain executed; "
                      "they are not evidence about learning quality"}
    booked = smoke.book_invocation(launch_tag, episode_constructions=constructions,
                                   optimizer_steps=steps, detail=detail)
    out = smoke.HERE / f"{launch_tag.upper()}_SUMMARY.json"
    out.write_text(json.dumps(detail, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    if probe is not None:
        r35_probe_dump = smoke.write_json(f"{launch_tag.upper()}_COLLECTOR_PROBE.json",
                                          {"schema": "round35-collector-probe/1",
                                           "launch_tag": launch_tag,
                                           "instrumentation": "class-level read-only wrapper "
                                                              "around "
                                                              "PublishedProgressRewardCollector"
                                                              ".handle_progress and "
                                                              "InMemoryEventBus.publish, "
                                                              "installed in this process only",
                                           **probe.summary()},
                                          refresh=True)
        print("probe written to", r35_probe_dump, flush=True)
    print(json.dumps({"launch_tag": launch_tag, "phases": [
        {key: phase["result"][key] for key in
         ("episodes_done", "updates", "optimizer_steps", "lambda")} for phase in phases],
        "wall_clock_seconds": detail["wall_clock_seconds"],
        "book_totals": {key: booked[key] for key in
                        ("invocations", "episode_constructions", "optimizer_steps")}},
        ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="one authorised ROUND35 smoke launch")
    parser.add_argument("--launch-tag", required=True, choices=("launch1", "launch2"))
    parser.add_argument("--with-probe", action="store_true",
                        help="attach the read-only collector probe (launch 2 only)")
    arguments = parser.parse_args(sys.argv[1:])
    raise SystemExit(run(arguments.launch_tag, with_probe=arguments.with_probe))
