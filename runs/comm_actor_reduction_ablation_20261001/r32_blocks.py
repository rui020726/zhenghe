"""ROUND32 块级编排：reference / first4 / train / eval / late / ledger，以及账本刷新。

每个命令内部按 cell 记账；6 条训练按 (reduction, seed) 分进程并行（各进程独立 RNG，
账本由锁文件保护临界区），并行度上限 4。块完成后做该块的完成性检查（行数、闭合字段、
与 ROUND24/ROUND30 归档 reference 的行为字段比对）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import r32_common as common
import r32_run as runner
import r32_trainer as trainer
from r32_train import load_reference_cache, milestone_path, pre_update_path, train_arm_seed

#: 与归档件的行为字段（身份核验，不是表现门禁）
BEHAVIOUR_FIELDS = ("on_time", "spent_bytes", "emissions", "expired", "completed",
                    "discovered", "ack_ledger", "steps")
ARCHIVES = {"round24": common.ROOT / "runs/comm_cost_v2_rerun_20260928/rows",
            "round30": common.ROOT / "runs/comm_deadline_encoding_ablation_20260930/rows"}


def _archive_rows(directory: Path, scenario: int) -> Path | None:
    candidate = directory / f"v2_reference_s{scenario}_a1.json"
    if candidate.exists():
        return candidate
    candidate = directory / f"r30_reference_s{scenario}_a1.json"
    return candidate if candidate.exists() else None


def block_reference() -> dict[str, Any]:
    rows = []
    for scenario in list(common.TRAINING_SCENARIOS) + list(common.DEVELOPMENT_SCENARIOS):
        rows.append(runner.run_reference(int(scenario)))
    problems = []
    if len(rows) != common.REFERENCE_LAUNCHES:
        problems.append(f"reference rows {len(rows)} != {common.REFERENCE_LAUNCHES}")
    diverged = []
    for row in rows:
        for name, directory in ARCHIVES.items():
            archived_path = _archive_rows(directory, int(row["scenario_seed"]))
            if archived_path is None:
                continue
            old = common.read_json(archived_path)
            for field in BEHAVIOUR_FIELDS:
                if old.get(field) != row.get(field):
                    diverged.append({"archive": name, "scenario": row["scenario_seed"],
                                     "field": field, "archived": old.get(field),
                                     "now": row.get(field)})
    return {"rows": len(rows), "problems": problems, "diverged_fields": diverged,
            "passed": not problems and not diverged}


def block_late() -> dict[str, Any]:
    rows = [runner.run_rule(common.LATE_RULE_ARM, int(scenario))
            for scenario in common.DEVELOPMENT_SCENARIOS]
    problems = [] if len(rows) == common.LATE_RULE_LAUNCHES else [
        f"late rows {len(rows)} != {common.LATE_RULE_LAUNCHES}"]
    return {"rows": len(rows), "problems": problems, "passed": not problems}


def block_eval(reduction: str | None = None, seed: int | None = None,
               mode: str | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    reductions = (reduction,) if reduction else common.ARM_ORDER
    seeds = (seed,) if seed else tuple(common.TRAIN_SEEDS)
    modes = (mode,) if mode else tuple(common.MODES)
    from r32_check import freeze_record
    for seed in seeds:
        for eval_mode in modes:
            for eval_reduction in reductions:
                freeze = freeze_record(eval_reduction)
                milestone = milestone_path(eval_reduction, int(seed), common.EPISODES_PER_SEED)
                controller, _optimizer, state = trainer.load_checkpoint(
                    milestone, freeze=freeze, reduction=eval_reduction)
                if int(state["episodes_done"]) != int(common.EPISODES_PER_SEED):
                    raise RuntimeError(f"{eval_reduction} seed {seed}: milestone is at episode "
                                       f"{state['episodes_done']}, expected "
                                       f"{common.EPISODES_PER_SEED}")
                if int(state["optimizer_steps"]) == 0:
                    raise RuntimeError(f"{eval_reduction} seed {seed}: an untrained milestone "
                                       "cannot be evaluated as the final policy")
                for scenario in common.DEVELOPMENT_SCENARIOS:
                    rows.append(runner.run_eval(controller, reduction=eval_reduction,
                                                training_seed=int(seed), timepoint="final",
                                                mode=eval_mode, scenario_seed=int(scenario)))
    return {"rows": len(rows), "problems": [], "passed": True}


def refresh_ledger() -> dict[str, Any]:
    report = common.ledger_report()
    plan = {"normal_launches": common.NORMAL_LAUNCHES,
            "reserve_launches": common.RESERVE_LAUNCHES, "hard_cap": common.LAUNCH_CAP,
            "reference": common.REFERENCE_LAUNCHES, "training": common.TRAINING_LAUNCHES,
            "eval": common.EVAL_LAUNCHES, "late_rule": common.LATE_RULE_LAUNCHES,
            "synthetic_update_cap": common.SYNTHETIC_UPDATE_CAP,
            "max_parallel_workers": common.MAX_PARALLEL_WORKERS}
    common.write_json("START_LEDGER.json",
                      {"plan": plan, "live": {"used": report["used"],
                                              "by_kind": report["by_kind"],
                                              "results": report["results"]}},
                      refresh=True)
    return {"used": report["used"], "cap": common.LAUNCH_CAP, "by_kind": report["by_kind"]}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="run one fixed block of the round-32 plan")
    parser.add_argument("block", choices=["reference", "train", "first4", "eval", "late",
                                          "ledger"])
    parser.add_argument("--reduction", choices=list(common.ARM_ORDER), default=None)
    parser.add_argument("--seed", type=int, choices=list(common.TRAIN_SEEDS), default=None)
    parser.add_argument("--episodes", type=int, default=common.EPISODES_PER_SEED)
    parser.add_argument("--mode", choices=list(common.MODES), default=None)
    args = parser.parse_args(argv)

    if args.block == "ledger":
        print(json.dumps(refresh_ledger(), ensure_ascii=False))
        return 0
    if args.block == "reference":
        summary = block_reference()
    elif args.block == "late":
        summary = block_late()
    elif args.block == "eval":
        summary = block_eval(args.reduction, args.seed, args.mode)
    elif args.block in ("train", "first4"):
        if args.reduction is None or args.seed is None:
            raise SystemExit(f"{args.block} needs --reduction and --seed")
        from r32_check import freeze_record
        episodes = 4 if args.block == "first4" else args.episodes
        summary = train_arm_seed(args.reduction, args.seed, episodes=episodes,
                                 freeze=freeze_record(args.reduction),
                                 reference_cache=load_reference_cache())
        summary["passed"] = summary["episodes_done"] == episodes
    else:  # pragma: no cover - argparse restricts choices
        raise SystemExit("unknown block")
    print(json.dumps({"block": args.block, "passed": summary.get("passed", True),
                      "rows": summary.get("rows", summary.get("episodes_done"))},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
