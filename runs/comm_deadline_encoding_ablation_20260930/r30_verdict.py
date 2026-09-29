"""ROUND30 判定：执行状态、五问回答与诊断汇总（只读落盘件，0 环境启动）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import r30_common as common


def read(name: str) -> Any:
    return json.loads((common.OUTPUT_DIR / name).read_text(encoding="utf-8"))


def main() -> int:
    comparison = read("PAIRED_COMPARISON.json")
    gates = {}
    for path in sorted(common.OUTPUT_DIR.glob("FIRST4_GATE_*.json")):
        gates[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    ledger = common.ledger_report()
    preflight = read("RUN_PREFLIGHT.json")
    manifest = read("CONFIG_AND_MODEL_MANIFEST.json")
    differences = read("CONFLICT_RECHECK_3_1.json")["resolution"]
    clock = read("CLOCK_RECHECK_3_2.json")["totals"]

    greedy = {seed: comparison["comparisons"][f"s{seed}_greedy"] for seed in common.TRAIN_SEEDS}
    sampled = {seed: comparison["comparisons"][f"s{seed}_sampled"] for seed in common.TRAIN_SEEDS}

    def direction(value: float) -> str:
        return "positive" if value > 0 else ("negative" if value < 0 else "zero")

    verdict: dict[str, Any] = {
        "round": "30",
        "section": "verdict: old vs new deadline encoding, paired from-initialisation ablation",
        "execution_status": {
            "branch": "exp/comm-deadline-encoding-ablation",
            "start_sha": "99eb746baaa073550f9062fec5c87a4b4711827b",
            "launches_used": ledger["used"], "normal_launches": ledger["normal_launches"],
            "reserve_launches": ledger["reserve_launches"], "hard_cap": ledger["hard_cap"],
            "reserve_used": ledger["used"] - ledger["normal_launches"],
            "training_episodes_per_arm_per_seed": 128,
            "optimizer_steps": {f"{arm}_s{seed}": read("TRAINING_CURVES.json")["curves"][
                f"{arm}_s{seed}"]["optimizer_steps"]
                for arm in ("old", "new") for seed in common.TRAIN_SEEDS},
            "synthetic_optimizer_steps": preflight["synthetic_optimizer_steps"],
            "synthetic_update_cap": preflight["synthetic_update_cap"],
            "preflight_passed": preflight["passed"],
            "first4_gates": {key: value["passed"] for key, value in gates.items()},
            "publication_status": "PENDING_NETWORK",
            "network_retries_this_round": 0,
        },
        "q1_only_registered_encoding_difference": {
            "answer": "yes",
            "evidence": [
                "the scoped collector differs from the archived parent in exactly one source "
                "line (the feature-row construction), registered by a programmatic source diff "
                "in the freeze manifest",
                "OLD reproduces the archived collector's tape field for field in the preflight",
                "everything else -- controller class, trainer, PPO hyperparameters, budget, "
                "ACK-off, mask semantics, scenario order, action/eval RNG derivations, "
                "evaluation decoders -- is the shared frozen code, and the production/imported "
                "module identity is byte-identical to ROUND24's recorded freeze",
                "the two arms of one seed start from byte-identical parameters, optimizer "
                "state and lambda (INIT_PAIRING artefacts)",
            ],
        },
        "q2_task_and_cost_changes": {
            "greedy_primary": {
                f"s{seed}": {
                    "delta_y_mean": greedy[seed]["delta_y"]["mean"],
                    "delta_y_counts": greedy[seed]["delta_y"]["counts"],
                    "delta_y_ci95": greedy[seed]["delta_y"]["ci95"],
                    "delta_c_mean_bytes": greedy[seed]["delta_c"]["mean"],
                    "delta_c_counts": greedy[seed]["delta_c"]["counts"],
                    "delta_c_ci95": greedy[seed]["delta_c"]["ci95"],
                    "joint_counts": greedy[seed]["joint_counts"],
                    "goal_set_swaps": greedy[seed]["goal_set_swaps"],
                    "vs_immediate_and_late": greedy[seed]["vs_immediate"],
                } for seed in common.TRAIN_SEEDS},
            "sampled_separate": {
                f"s{seed}": {
                    "delta_y_mean": sampled[seed]["delta_y"]["mean"],
                    "delta_y_counts": sampled[seed]["delta_y"]["counts"],
                    "delta_c_mean_bytes": sampled[seed]["delta_c"]["mean"],
                    "delta_c_counts": sampled[seed]["delta_c"]["counts"],
                } for seed in common.TRAIN_SEEDS},
        },
        "q3_seed_consistency": {
            "greedy_delta_y_means": {f"s{seed}": greedy[seed]["delta_y"]["mean"]
                                     for seed in common.TRAIN_SEEDS},
            "greedy_delta_c_means": {f"s{seed}": greedy[seed]["delta_c"]["mean"]
                                     for seed in common.TRAIN_SEEDS},
            "consistent_sign": None,
            "note": "three seeds shown separately; consistency is read as sign agreement of the "
                    "per-seed means, never as a statistical stability or non-inferiority claim",
        },
        "q4_greedy_vs_sampled": {
            "greedy_delta_y_means": {f"s{seed}": greedy[seed]["delta_y"]["mean"]
                                     for seed in common.TRAIN_SEEDS},
            "sampled_delta_y_means": {f"s{seed}": sampled[seed]["delta_y"]["mean"]
                                      for seed in common.TRAIN_SEEDS},
            "greedy_delta_c_means": {f"s{seed}": greedy[seed]["delta_c"]["mean"]
                                     for seed in common.TRAIN_SEEDS},
            "sampled_delta_c_means": {f"s{seed}": sampled[seed]["delta_c"]["mean"]
                                      for seed in common.TRAIN_SEEDS},
            "same_direction": None,
            "note": "sampled observes the trained policy itself; its constraint estimate is not "
                    "transferred to greedy",
        },
        "q5_not_established": [
            "no confirmatory significance claim, no delta=0 guarantee and no independent "
            "generalisation claim: the development block is already-used data and each block "
            "runs once per cell",
            "a descriptive 95% interval covers scene uncertainty under the fixed trained "
            "weights only, not training-seed uncertainty, and one sampled rollout per scene "
            "does not estimate its full sampling variance",
            "a better new-encoding result attributes to the whole encoding replacement (both "
            "information retention and numerical scale changed together); no claim that "
            "removing saturation alone is the cause",
            "the feasibility, effort or deployment value of the candidate layout is not "
            "established; no new checkpoint is authorised for any deployment path",
        ],
        "conflict_resolutions": {"3_1_development_block_counts": differences,
                                 "3_2_clock_relation": clock},
        "artefacts": {
            "report": "ROUND30_REPORT.md",
            "freeze_manifest": "CONFIG_AND_MODEL_MANIFEST.json",
            "preflight": "RUN_PREFLIGHT.json",
            "first4_gates": sorted(gates),
            "paired_comparison": "PAIRED_COMPARISON.json",
            "checkpoint_index": "CHECKPOINT_INDEX.json",
            "training_curves": "TRAINING_CURVES.json",
            "artefact_manifest": "ARTEFACT_MANIFEST.json",
            "start_ledger": "START_LEDGER.json",
        },
        "stop": "local delivery and stop: no added training, no tuning, no candidate change, no "
                "confirmation experiment, no deployment, no automatic push",
    }
    greedy_means = [greedy[seed]["delta_y"]["mean"] for seed in common.TRAIN_SEEDS]
    cost_means = [greedy[seed]["delta_c"]["mean"] for seed in common.TRAIN_SEEDS]
    verdict["q3_seed_consistency"]["consistent_sign"] = (
        len({direction(value) for value in greedy_means}) == 1
        and len({direction(value) for value in cost_means}) == 1)
    verdict["q4_greedy_vs_sampled"]["same_direction"] = (
        all(direction(greedy[seed]["delta_y"]["mean"])
            == direction(sampled[seed]["delta_y"]["mean"]) for seed in common.TRAIN_SEEDS))
    (common.OUTPUT_DIR / "ROUND30_VERDICT.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"greedy_dY": greedy_means, "greedy_dC": cost_means,
                      "first4": verdict["execution_status"]["first4_gates"],
                      "launches": ledger["used"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
