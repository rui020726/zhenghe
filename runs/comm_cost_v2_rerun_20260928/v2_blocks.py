"""ROUND24 v2 重跑块运行器：reference / eval / rule 三个块的续跑式执行。

每个块都可重入：已有 row 的 cell 跳过（不重复消耗启动）。所有启动照常记账。
后台运行示例（日志落盘，注意后台 shell 的 CWD 会重置，必须显式 cd）：

    cd /d/code/整合/zhenghe/runs/comm_cost_v2_rerun_20260928 && /d/project_env/python.exe v2_blocks.py references
    cd /d/code/整合/zhenghe/runs/comm_cost_v2_rerun_20260928 && /d/project_env/python.exe v2_blocks.py evals
    cd /d/code/整合/zhenghe/runs/comm_cost_v2_rerun_20260928 && /d/project_env/python.exe v2_blocks.py rules

评估块（ROUND24 spec §5，首次启动前固定的三个块，每 seed 16 场景 × 3 = 48 次）：
    init_greedy   —— 登记初值 Controller(训练种子)，greedy；
    final_greedy  —— 128 集里程碑 checkpoint，greedy；
    final_sampled —— 同一 128 集里程碑 checkpoint，sampled（与训练的 masked 随机策略一致）。
不跑 64 集开发评估；里程碑 ep64 只保存供审计。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import v2_common as common
import v2_run as runner


def _done_scenarios(pattern: str) -> set[int]:
    done: set[int] = set()
    for path in sorted(common.ROWS.glob(pattern)):
        done.add(int(json.loads(path.read_text(encoding="utf-8"))["scenario_seed"]))
    return done


def run_references() -> dict[str, Any]:
    scenarios = list(common.TRAINING_SCENARIOS) + list(common.DEVELOPMENT_SCENARIOS)
    done = _done_scenarios("v2_reference_s*_a1.json")
    remaining = [scenario for scenario in scenarios if scenario not in done]
    print(f"references: {len(done)} done, {len(remaining)} remaining", flush=True)
    for index, scenario in enumerate(remaining, start=1):
        row = runner.run_reference(scenario)
        print(f"[{index}/{len(remaining)}] reference s{scenario}: "
              f"on_time={row['on_time']} bytes={row['spent_bytes']}", flush=True)
    return {"block": "references", "done": len(done), "ran": len(remaining)}


def _eval_controller(training_seed: int, block: str):
    """The registered initialisation or an explicitly read-only immutable milestone."""
    import v2_controller
    import v2_trainer as trainer

    spec = common.EVAL_BLOCKS[block]
    if spec["checkpoint"] == "initialisation":
        return v2_controller.fresh_controller(training_seed), {
            "load_mode": "fresh_initialization",
            "training_version": trainer.TRAINING_VERSION,
            "reward_timing_version": trainer.REWARD_TIMING_VERSION,
            "optimizer_restored": False,
            "rng_restored": False,
        }
    manifest = json.loads(
        (common.OUTPUT_DIR / "CONFIG_AND_MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    freeze = {"freeze_sha256": manifest["code_identity"]["freeze_sha256"],
              "config_hash": manifest["configuration_hash"],
              "branch_start": manifest["branch_start"],
              "head_at_freeze": manifest["head_at_freeze"]}
    # Only the immutable milestone is evaluated: the rolling resume checkpoint is never a
    # candidate model (spec §5/§8).
    import v2_train as train_driver

    path = train_driver.milestone_path(training_seed, common.EPISODES_PER_SEED)
    controller, state, identity = trainer.load_checkpoint_read_only(path, freeze=freeze)
    if int(state["episodes_done"]) != common.EPISODES_PER_SEED:
        raise SystemExit(
            f"seed {training_seed}: milestone holds {state['episodes_done']} episodes, expected "
            f"{common.EPISODES_PER_SEED}")
    identity.update({"checkpoint": str(path.relative_to(common.ROOT)),
                     "training_seed": int(state["training_seed"]),
                     "episodes_done": int(state["episodes_done"])})
    return controller, identity


def run_evals(blocks: list[str]) -> dict[str, Any]:
    ran = 0
    for training_seed in common.TRAIN_SEEDS:
        for block in blocks:
            spec = common.EVAL_BLOCKS[block]
            tag = f"eval_s{training_seed}_{block}"
            # The controller is rebuilt once per (seed, block); the same frozen weights serve
            # every scene of that block.
            controller = None
            checkpoint_load_identity = None
            for scenario in common.DEVELOPMENT_SCENARIOS:
                if (common.ROWS / f"v2_{tag}_s{scenario}_a0.json").exists():
                    continue
                if controller is None:
                    controller, checkpoint_load_identity = _eval_controller(training_seed, block)
                row = runner.run_eval(controller, training_seed=training_seed,
                                      timepoint=spec["timepoint"], mode=spec["mode"],
                                      scenario_seed=scenario,
                                      checkpoint_load_identity=checkpoint_load_identity)
                ran += 1
                print(f"eval s{training_seed} {block} scene {scenario}: "
                      f"on_time={row['on_time']} bytes={row['spent_bytes']}", flush=True)
    return {"block": "evals", "blocks": list(blocks), "ran": ran}


def run_rules() -> dict[str, Any]:
    ran = 0
    for arm in common.RULE_ARMS:
        for scenario in common.DEVELOPMENT_SCENARIOS:
            if (common.ROWS / f"v2_{arm}_s{scenario}_a1.json").exists():
                continue
            row = runner.run_rule(arm, scenario)
            ran += 1
            print(f"rule {arm} scene {scenario}: on_time={row['on_time']} "
                  f"bytes={row['spent_bytes']}", flush=True)
    return {"block": "rules", "ran": ran}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="ROUND24 v2 rerun blocks")
    parser.add_argument("block", choices=("references", "evals", "rules"))
    parser.add_argument("--blocks", nargs="*",
                        default=["init_greedy", "final_greedy", "final_sampled"])
    args = parser.parse_args(argv)
    if args.block == "references":
        summary = run_references()
    elif args.block == "evals":
        summary = run_evals([str(value) for value in args.blocks])
    else:
        summary = run_rules()
    print(json.dumps({**summary, "ledger_used": common.ledger_total()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
