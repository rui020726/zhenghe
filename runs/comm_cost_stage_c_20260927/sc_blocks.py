"""Stage C 块运行器：reference / eval / rule 三个块的续跑式执行。

每个块都可重入：已有 row 的 cell 跳过（不重复消耗启动）。所有启动照常记账。
后台运行示例（日志落盘）：

    /d/project_env/python.exe sc_blocks.py references
    /d/project_env/python.exe sc_blocks.py evals --timepoints 0 64 128
    /d/project_env/python.exe sc_blocks.py rules
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import sc_common as common
import sc_run as runner


def _done_scenarios(pattern: str) -> set[int]:
    done: set[int] = set()
    for path in sorted(common.ROWS.glob(pattern)):
        done.add(int(json.loads(path.read_text(encoding="utf-8"))["scenario_seed"]))
    return done


def run_references() -> dict[str, Any]:
    scenarios = list(common.TRAINING_SCENARIOS) + list(common.DEVELOPMENT_SCENARIOS)
    done = _done_scenarios("sc_reference_s*_a1.json")
    remaining = [scenario for scenario in scenarios if scenario not in done]
    print(f"references: {len(done)} done, {len(remaining)} remaining", flush=True)
    for index, scenario in enumerate(remaining, start=1):
        row = runner.run_reference(scenario)
        print(f"[{index}/{len(remaining)}] reference s{scenario}: "
              f"on_time={row['on_time']} bytes={row['spent_bytes']}", flush=True)
    return {"block": "references", "done": len(done), "ran": len(remaining)}


def run_evals(timepoints: list[str]) -> dict[str, Any]:
    import sc_trainer as trainer

    ran = 0
    for training_seed in common.TRAIN_SEEDS:
        for timepoint in timepoints:
            if timepoint == "0":
                controller = __import__("sc_controller").fresh_controller(training_seed)
            else:
                checkpoint = common.CHECKPOINTS / f"sc_train_s{training_seed}.pt"
                manifest = json.loads(
                    (common.OUTPUT_DIR / "CONFIG_AND_MODEL_MANIFEST.json").read_text(encoding="utf-8"))
                freeze = {"freeze_sha256": manifest["code_identity"]["freeze_sha256"],
                          "config_hash": manifest["configuration_hash"],
                          "branch_start": manifest["branch_start"],
                          "head_at_freeze": manifest["head_at_freeze"]}
                controller, _optimizer, state = trainer.load_checkpoint(checkpoint, freeze=freeze)
                if int(state["episodes_done"]) < int(timepoint):
                    raise SystemExit(
                        f"seed {training_seed} has {state['episodes_done']} episodes, cannot "
                        f"evaluate timepoint {timepoint}")
            for scenario in common.DEVELOPMENT_SCENARIOS:
                tag = f"eval_s{training_seed}_{timepoint}"
                if (common.ROWS / f"sc_{tag}_s{scenario}_a0.json").exists():
                    continue
                row = runner.run_eval(controller, training_seed=training_seed,
                                      timepoint=timepoint, scenario_seed=scenario)
                ran += 1
                print(f"eval s{training_seed}@{timepoint} scene {scenario}: "
                      f"on_time={row['on_time']} bytes={row['spent_bytes']}", flush=True)
    return {"block": "evals", "ran": ran}


def run_rules() -> dict[str, Any]:
    ran = 0
    for arm in common.RULE_ARMS:
        for scenario in common.DEVELOPMENT_SCENARIOS:
            if (common.ROWS / f"sc_{arm}_s{scenario}_a1.json").exists():
                continue
            row = runner.run_rule(arm, scenario)
            ran += 1
            print(f"rule {arm} scene {scenario}: on_time={row['on_time']} "
                  f"bytes={row['spent_bytes']}", flush=True)
    return {"block": "rules", "ran": ran}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Stage C blocks")
    parser.add_argument("block", choices=("references", "evals", "rules"))
    parser.add_argument("--timepoints", nargs="*", default=["0", "64", "128"])
    args = parser.parse_args(argv)
    if args.block == "references":
        summary = run_references()
    elif args.block == "evals":
        summary = run_evals([str(value) for value in args.timepoints])
    else:
        summary = run_rules()
    print(json.dumps({**summary, "ledger_used": common.ledger_total()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
