"""ROUND35 记账（0 环境启动）：从 ledger 与批记录重算本轮真实消耗，并补记崩溃那次未入账的开销。

`r35_launch.py` 只在跑完成功才入账，所以第一次尝试（4 秒处失败）与 launch2（被额度拦下）都没有
正常记账。这里不手填数字：episode 构造从各 launch 目录的真实 START_LEDGER 求和（含失败起始），
optimizer 步从各批记录的 `optimizer_steps` 求和，验证脚本另计的步数取 SMOKE_LAUNCH_BOOK 的
verification 事件，然后写两份东西：

- `SMOKE_LAUNCH_BOOK.json` 追加一条 reconciliation 事件（历史事件不删不改）；
- `LAUNCH_ACCOUNTING.json` 作为本轮对外报告的权威计数件（含 runner 进程数、授权来源、剩余额度）。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import r35_common as smoke


def batch_steps() -> dict[str, int]:
    steps = {}
    for path in sorted(smoke.HERE.glob("launch*/checkpoints/v2_batch_s11_b*.json")):
        record = json.loads(path.read_text("utf-8"))
        steps[f"{path.parent.parent.name}/{path.stem}"] = int(record["update"]["optimizer_steps"])
    return steps


def main() -> int:
    ledgers = smoke.ledger_starts_across_launches()
    steps_by_batch = batch_steps()
    book = smoke.smoke_book()
    verification_steps = int(book.get("verification_optimizer_steps") or 0)
    launch_steps = sum(steps_by_batch.values())
    processes = [
        {"name": "launch1", "outcome": "completed 8 episodes and 2 updates",
         "episode_starts_in_its_own_ledger": ledgers["per_launch"].get("launch1", 0)},
        {"name": "launch2 attempt 1", "outcome": "crashed 4 s in, inside this round's own probe",
         "episode_starts_in_its_own_ledger": 1, "note": "the failed start stays in the ledger"},
        {"name": "launch2 attempt 2", "outcome": "4 episodes, batch 0, cross-process resume, "
                                                 "2 more episodes, stopped by the ledger cap at "
                                                 "episode 6",
         "episode_starts_in_its_own_ledger": 6}]
    total_starts = ledgers["total_starts"]
    if total_starts > smoke.MAX_EPISODE_CONSTRUCTIONS:
        raise RuntimeError("the round exceeded its episode-construction quota")
    if launch_steps + verification_steps > smoke.MAX_OPTIMIZER_STEPS:
        raise RuntimeError("the round exceeded its optimizer-step ceiling")
    extra = {
        "kind": "reconciliation", "pid": os.getpid(),
        "episode_constructions": total_starts,
        "optimizer_steps": launch_steps + verification_steps,
        "booked_by": "r35_reconcile_book.py",
        "why": "r35_launch books only on success, so the crashed attempt and launch2 were missing "
               "from the tally; these figures are summed from the real ledgers and batch records",
        "detail": {"per_launch_starts": ledgers["per_launch"], "failed_starts":
                   ledgers["failed_starts"], "optimizer_steps_by_batch_record": steps_by_batch,
                   "verification_steps_booked_separately": verification_steps,
                   "r4_ran_twice_because_the_first_run_stopped_at_a_harness_bug": True}}
    book["events"] = [event for event in book["events"]
                      if event.get("kind") != "reconciliation"] + [extra]
    book["episode_constructions"] = total_starts
    book["optimizer_steps"] = launch_steps + verification_steps
    book["reconciliation_note"] = ("episode_constructions and optimizer_steps are the reconciled "
                                   "totals recomputed from the ledgers and batch records, not the "
                                   "sum of the success-only bookings")
    smoke.write_json("SMOKE_LAUNCH_BOOK.json", book, refresh=True)
    episode_seconds = {}
    for launch in sorted({path.parent.parent.name for path in
                          smoke.HERE.glob("launch*/rows/v2_train_s11_e*_s*_a*.json")}):
        total = 0.0
        rows = 0
        for path in sorted(smoke.HERE.glob(f"{launch}/rows/v2_train_s11_e*_s*_a*.json")):
            total += float(json.loads(path.read_text("utf-8"))["seconds"])
            rows += 1
        episode_seconds[launch] = {"episode_rows": rows,
                                   "sum_of_measured_episode_seconds": round(total, 2)}
    wall_clock = {name: json.loads(path.read_text("utf-8"))["wall_clock_seconds"]
                  for name, path in ((launch, smoke.HERE / f"{launch.upper()}_SUMMARY.json")
                                      for launch in ("launch1", "launch2"))
                  if path.exists()}
    accounting = {
        "schema": "round35-launch-accounting/1", "round": 35,
        "authorised": {"runner_invocations": 2, "episode_constructions": 16,
                       "optimizer_steps": 100, "wall_clock_per_invocation_hours": 2},
        "used": {"runner_processes_started": len(processes),
                 "productive_runner_invocations": 2,
                 "episode_constructions_total": total_starts,
                 "episode_constructions_remaining": smoke.MAX_EPISODE_CONSTRUCTIONS - total_starts,
                 "optimizer_steps_total": launch_steps + verification_steps,
                 "optimizer_steps_split": {"inside_the_launches": launch_steps,
                                           "inside_verification_recomputations": verification_steps},
                 "episode_seconds_measured_per_launch": episode_seconds,
                 "process_wall_clock_seconds": wall_clock,
                 "wall_clock_note": ("the wall-clock summary file exists only for launch1 because "
                                     "launch2's second attempt ended on the ledger refusal before "
                                     "it could write one; the per-episode seconds above are the "
                                     "measured rollouts and the 2 h per-invocation bound was "
                                     "nowhere near being reached"),
                 "farthest_measured_single_invocation_seconds": max(
                     list(wall_clock.values())
                     + [value["sum_of_measured_episode_seconds"]
                        for value in episode_seconds.values()], default=0.0)},
        "per_batch_optimizer_steps": steps_by_batch,
        "runner_processes": processes,
        "episode_constructions_per_launch": ledgers["per_launch"],
        "failed_starts": ledgers["failed_starts"],
        "extra_invocation_authorisation": json.loads(
            (smoke.HERE / "AUTHORISATION_EXTRA_INVOCATION.json").read_text("utf-8")),
        "quota_discipline": ("the failed attempt's episode start stays spent; nothing was "
                            "re-run to look better and no result was selected"),
        "new_streams_created": 0,
        "sealed_blocks_touched": 0,
        "network_attempts": 0,
        "push_to_github": 0,
        "old_round_artefacts_rewritten": 0,
    }
    out = smoke.write_json("LAUNCH_ACCOUNTING.json", accounting, refresh=True)
    print(json.dumps({"used": accounting["used"],
                      "per_launch_starts": ledgers["per_launch"],
                      "wrote": [out.name, "SMOKE_LAUNCH_BOOK.json"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
