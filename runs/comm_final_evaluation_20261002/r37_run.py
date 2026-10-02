"""ROUND37 评估矩阵运行器（可分片、可重入、原子 claim，禁止重复启动）。

作业清单来自冻结件：9 臂 x 64 个新场景 = 576 格，按 `index % shards == shard` 明确分片，
每格先用 O_CREAT|O_EXCL 的 claim 文件原子领取；领不到就跳过，因此四个进程并行也不会重复启动，
中断后重跑同一分片也不会二次消耗。真实记账仍由 `v2_common.LaunchLedger` 在构造 driver 之前完成。

    /d/project_env/python.exe r37_run.py --shard 0 --shards 4
    /d/project_env/python.exe r37_run.py --shard 0 --shards 4 --scenario 1600000   # 技术门禁

本轮 0 训练、0 optimizer.step：模型只经 `load_checkpoint_read_only` 装载（不建 optimizer、
不动 RNG），规则臂走 `v2_run.run_rule`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import r37_common as r37
import v2_run

CONTROLLERS: dict[str, tuple[Any, dict[str, Any]]] = {}
PLAN = None


def load_plan() -> dict[str, Any]:
    global PLAN
    if PLAN is None:
        path = r37.HERE / "FROZEN_EVALUATION_PLAN.json"
        if not path.exists():
            raise SystemExit("FROZEN_EVALUATION_PLAN.json is missing: freeze before launching")
        PLAN = r37.read_json(path)
        arms = [item["id"] for item in PLAN["arms"]]
        if arms != list(r37.ALL_ARMS) or PLAN["scenarios"] != list(r37.SCENARIOS):
            raise SystemExit("the frozen plan does not match the registered arms/scenarios")
    return PLAN


def controller_for(seed: int, mode: str) -> tuple[Any, dict[str, Any]]:
    """The frozen ep128 model, loaded read-only; identical weights serve every scene of a block."""
    key = f"s{seed}_{mode}"
    if key not in CONTROLLERS:
        import torch
        import v2_trainer

        entry = load_plan()["models"][f"s{seed}"]
        path = r37.ROOT / entry["path"]
        stored = r37.stored_freeze_token(path)
        controller, state, identity = v2_trainer.load_checkpoint_read_only(path, freeze=stored)
        if int(state["episodes_done"]) != r37.MILESTONE_EPISODES:
            raise SystemExit(f"seed {seed}: not the {r37.MILESTONE_EPISODES}-episode milestone")
        if identity["timing_interpretation"] != "versioned_capture_semantics":
            raise SystemExit(f"seed {seed}: unversioned capture semantics")
        from v2_controller import parameter_identity

        identity.update({"checkpoint": entry["path"], "checkpoint_bytes_sha256":
                         r37.sha256_file(path),
                         "parameter_sha256": parameter_identity(controller)["parameter_sha256"],
                         "episodes_done": int(state["episodes_done"]),
                         "updates": int(state["updates"]),
                         "lambda_in_state": float(state["lambda"]),
                         "training_seed": int(state["training_seed"]),
                         "decoding_mode": mode})
        CONTROLLERS[key] = (controller, identity)
    return CONTROLLERS[key]


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    arm, scenario = job["arm"], job["scenario"]
    kind, seed, mode = r37.parse_arm(arm)
    started = time.time()
    if r37.row_path(job).exists():
        r37.bump_reused_rows()
        return {"job": job, "reused": True, "ok": True}
    try:
        if kind == "model":
            controller, identity = controller_for(seed, mode)
            row = v2_run.run_eval(controller, training_seed=seed, timepoint="final", mode=mode,
                                  scenario_seed=scenario, checkpoint_load_identity=identity)
        else:
            row = v2_run.run_rule(arm, scenario)
        r37.book_event(kind, arm=arm, scenario=scenario, ok=True,
                       detail=f"on_time={row['on_time']} bytes={row['spent_bytes']}",
                       seconds=time.time() - started)
        return {"job": job, "ok": True, "on_time": row["on_time"],
                "bytes": row["spent_bytes"], "seconds": round(time.time() - started, 2)}
    except Exception as error:                      # 技术失败照计，不静默跳过
        r37.book_event(kind, arm=arm, scenario=scenario, ok=False,
                       detail=f"{type(error).__name__}: {error}", seconds=time.time() - started)
        return {"job": job, "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "seconds": round(time.time() - started, 2)}


def retry_claim_held(index: int) -> bool:
    return (r37.HERE / "claims" / f"job_{index:04d}.retry.claim").exists()


def claim_retry(index: int) -> bool:
    path = r37.HERE / "claims" / f"job_{index:04d}.retry.claim"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        sink.write(json.dumps({"index": index, "pid": os.getpid(), "kind": "retry",
                               "claimed_at": time.strftime("%F %T")}) + "\n")
    return True


def retry_targets() -> list[dict[str, Any]]:
    """Cells whose last booked event failed technically and that still have no row.

    Performance is never a reason to retry: a cell that ran and completed is left alone even if
    its scenario is an easy or a hard one.
    """
    last: dict[tuple[str, int], bool] = {}
    for event in r37.book()["events"]:
        last[(str(event["arm"]), int(event["scenario"]))] = bool(event["ok"])
    out = []
    for job in r37.jobs():
        key = (job["arm"], job["scenario"])
        if last.get(key) is False and not r37.row_path(job).exists() \
                and not retry_claim_held(job["index"]):
            out.append(job)
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="one shard of the frozen ROUND37 matrix")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=r37.MAX_PARALLEL_WORKERS)
    parser.add_argument("--scenario", type=int, default=None,
                        help="restrict the shard to one scenario (the technical gate)")
    parser.add_argument("--limit", type=int, default=None, help="stop after N jobs (dry planning)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="re-run only cells that ended in a technical failure and have no row")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the selected cells without starting any environment")
    arguments = parser.parse_args(argv)
    if not 0 <= arguments.shard < arguments.shards:
        raise SystemExit("shard must be in [0, shards)")
    if arguments.shards > r37.MAX_PARALLEL_WORKERS:
        raise SystemExit(f"at most {r37.MAX_PARALLEL_WORKERS} simulation processes are authorised")
    r37.configure()
    load_plan()

    if arguments.retry_failed:
        mine = [job for job in retry_targets()
                if job["index"] % arguments.shards == arguments.shard]
        if arguments.dry_run:
            print(json.dumps({"retry_selected": len(mine),
                              "cells": [{"arm": job["arm"], "scenario": job["scenario"]}
                                        for job in mine]}, ensure_ascii=False))
            return 0
        for job in mine:
            if not claim_retry(job["index"]):
                continue
            outcome = run_job(job)
            print(json.dumps({"retry": job["arm"], "scenario": job["scenario"],
                              "ok": outcome["ok"]}, ensure_ascii=False), flush=True)
        return 0

    mine = [job for job in r37.jobs()
            if job["index"] % arguments.shards == arguments.shard
            and (arguments.scenario is None or job["scenario"] == arguments.scenario)]
    if arguments.limit:
        mine = mine[:arguments.limit]
    if arguments.dry_run:
        print(json.dumps({"shard": arguments.shard, "shards": arguments.shards,
                          "cells_selected": len(mine)}, ensure_ascii=False))
        return 0
    ran = reused = failed = 0
    started = time.time()
    for job in mine:
        if not r37.claim(job["index"]):
            continue
        outcome = run_job(job)
        if outcome.get("reused"):
            reused += 1
        elif outcome["ok"]:
            ran += 1
        else:
            failed += 1
            print(f"TECHNICAL FAILURE {job['arm']} s{job['scenario']}: {outcome['error']}",
                  flush=True)
        if (ran + reused + failed) % 10 == 0:
            print(f"shard {arguments.shard}: {ran + reused + failed}/{len(mine)} cells, "
                  f"ledger {r37.real_ledger_total()}", flush=True)
    summary = {"shard": arguments.shard, "shards": arguments.shards,
               "scenario_filter": arguments.scenario, "cells_in_shard": len(mine),
               "ran": ran, "reused_existing_rows": reused, "technical_failures": failed,
               "claims_held_round_wide": r37.claims_held(),
               "real_ledger_total": r37.real_ledger_total(),
               "seconds": round(time.time() - started, 2),
               "training_launches": 0, "optimizer_steps": 0}
    path = r37.HERE / "shards" / f"shard_{arguments.shard}_{'gate' if arguments.scenario else 'full'}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
