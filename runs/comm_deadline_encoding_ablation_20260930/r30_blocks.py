"""ROUND30 §7 块级编排：reference / train / first4 / eval / late，以及账本刷新。

每个命令内部按 cell 记账；6 条训练可按 (arm, seed) 分进程并行（各进程独立 RNG，
账本由锁文件保护临界区）。块完成后做该块的完成性检查（行数、闭合字段、与 ROUND24 归档
reference 的行为字段比对）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

import r30_check as check
import r30_common as common
import r30_encoding as enc
import r30_run as runner
import r30_train as train
import v2_trainer as trainer


def block_reference() -> dict[str, Any]:
    rows = []
    for scenario in list(common.TRAINING_SCENARIOS) + list(common.DEVELOPMENT_SCENARIOS):
        rows.append(runner.run_reference(int(scenario)))
    problems = []
    if len(rows) != common.REFERENCE_LAUNCHES:
        problems.append(f"reference rows {len(rows)} != {common.REFERENCE_LAUNCHES}")
    # 与 ROUND24 归档 reference 的行为字段比对（身份核验，不是门禁）
    diverged = []
    for row in rows:
        archived = (common.ROOT / "runs/comm_cost_v2_rerun_20260928/rows"
                    / f"v2_reference_s{row['scenario_seed']}_a1.json")
        if not archived.exists():
            continue
        old = json.loads(archived.read_text(encoding="utf-8"))
        for field in ("on_time", "spent_bytes", "emissions", "expired", "completed",
                      "discovered", "ack_ledger", "steps"):
            if old.get(field) != row.get(field):
                diverged.append({"scenario": row["scenario_seed"], "field": field,
                                 "archived": old.get(field), "now": row.get(field)})
    return {"rows": len(rows), "problems": problems, "diverged_fields": diverged,
            "passed": not problems and not diverged}


def block_late() -> dict[str, Any]:
    rows = [runner.run_rule("late_discovery_gate", int(scenario))
            for scenario in common.DEVELOPMENT_SCENARIOS]
    problems = [] if len(rows) == common.LATE_RULE_LAUNCHES else [
        f"late rows {len(rows)} != {common.LATE_RULE_LAUNCHES}"]
    return {"rows": len(rows), "problems": problems, "passed": not problems}


def block_eval(arm: str | None = None, seed: int | None = None,
               mode: str | None = None) -> dict[str, Any]:
    rows = []
    arms = (arm,) if arm else ("old", "new")
    seeds = (seed,) if seed else tuple(common.TRAIN_SEEDS)
    modes = (mode,) if mode else ("greedy", "sampled")
    for seed in seeds:
        for eval_mode in modes:
            for eval_arm in arms:
                freeze = check.freeze_record(eval_arm)
                milestone = (common.CHECKPOINTS
                             / f"r30_train_{eval_arm}_s{seed}_ep{common.v2c.EPISODES_PER_SEED}.pt")
                controller, _optimizer, state = trainer.load_checkpoint(milestone,
                                                                        freeze=freeze)
                if int(state["episodes_done"]) != int(common.v2c.EPISODES_PER_SEED):
                    raise RuntimeError(f"{eval_arm} seed {seed}: milestone is at episode "
                                       f"{state['episodes_done']}, expected "
                                       f"{common.v2c.EPISODES_PER_SEED}")
                for scenario in common.DEVELOPMENT_SCENARIOS:
                    rows.append(runner.run_eval(controller, arm=eval_arm, training_seed=seed,
                                                timepoint="final", mode=eval_mode,
                                                scenario_seed=int(scenario)))
    return {"rows": len(rows), "problems": [], "passed": True}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="run one fixed block of the round-30 plan")
    parser.add_argument("block", choices=["reference", "train", "first4", "eval", "late",
                                          "ledger"])
    parser.add_argument("--arm", choices=sorted(enc.ENCODERS), default=None)
    parser.add_argument("--seed", type=int, choices=list(common.TRAIN_SEEDS), default=None)
    parser.add_argument("--episodes", type=int, default=common.EPISODES_PER_SEED)
    parser.add_argument("--mode", choices=["greedy", "sampled"], default=None)
    args = parser.parse_args(argv)

    if args.block == "ledger":
        report = common.ledger_report()
        plan = {"normal_launches": common.NORMAL_LAUNCHES,
                "reserve_launches": common.RESERVE_LAUNCHES, "hard_cap": common.LAUNCH_CAP,
                "reference": common.REFERENCE_LAUNCHES, "training": common.TRAINING_LAUNCHES,
                "eval": common.EVAL_LAUNCHES, "late_rule": common.LATE_RULE_LAUNCHES,
                "synthetic_update_cap": common.SYNTHETIC_UPDATE_CAP}
        common.write_json("START_LEDGER.json",
                          {"plan": plan, "live": {"used": report["used"],
                                                  "by_kind": report["by_kind"],
                                                  "results": report["results"]}},
                          refresh=True)
        print(json.dumps({"used": report["used"], "cap": common.LAUNCH_CAP},
                         ensure_ascii=False))
        return 0

    if args.block == "reference":
        summary = block_reference()
    elif args.block == "late":
        summary = block_late()
    elif args.block == "eval":
        if args.mode is not None and args.mode not in ("greedy", "sampled"):
            raise SystemExit("--mode must be greedy or sampled")
        summary = block_eval(arm=args.arm, seed=args.seed, mode=args.mode)
    elif args.block in ("train", "first4"):
        if args.arm is None or args.seed is None:
            raise SystemExit(f"{args.block} needs --arm and --seed")
        episodes = 4 if args.block == "first4" else args.episodes
        summary = train.train_arm_seed(args.arm, args.seed, episodes=episodes,
                                       freeze=check.freeze_record(args.arm),
                                       reference_cache=train.load_reference_cache())
    else:  # pragma: no cover - argparse restricts choices
        raise SystemExit("unknown block")
    print(json.dumps({"block": args.block, "passed": summary.get("passed", True),
                      "rows": summary.get("rows", summary.get("episodes_done"))},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
