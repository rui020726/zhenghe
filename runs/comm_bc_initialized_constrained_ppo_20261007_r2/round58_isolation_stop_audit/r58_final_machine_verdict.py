"""Assemble the final R58 verdict from already checked source-aware products."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str):
    path = HERE / name
    return json.loads(path.read_text(encoding="utf-8"))


def sha(name: str):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def main() -> None:
    stats = load("ROUND58_FINAL_STATISTICS.json")
    accounting = load("ROUND58_FINAL_ACCOUNTING.json")
    diagnostics = load("ROUND58_FINAL_DIAGNOSTIC_COVERAGE.json")
    checkpoint_index = load("ROUND58_FINAL_CHECKPOINT_INDEX.json")
    index = load("ROUND58_FINAL_SOURCE_INDEX.json")
    quota = accounting["quota"]
    if (stats["training_counts"] != {"11": 128, "22": 128, "33": 128}
            or len(index["sources"]) != 1024
            or quota["normal_used"] != 1024 or quota["reserve_used"] != 5
            or quota["cumulative_attempt_slots"] != 1045
            or quota["cumulative_actual_environment_constructions"] != 1043
            or accounting["optimizer"]["gen3_actual_hook_calls"] != 556
            or accounting["ppo_updates"]["effective_committed_batches"] != 96
            or diagnostics["passive_records"]["count"] != 556
            or diagnostics["gradient_points"]["source_verified_points"] != 9):
        raise RuntimeError("final source graph, quota, optimizer or diagnostics do not close")
    if (len(checkpoint_index["batches"]) != 96
            or sum(item["before_location"] == "sha_verified_v18_archive_relocation"
                   for item in checkpoint_index["batches"]) != 9
            or sum(item["after_location"] == "sha_verified_v18_archive_relocation"
                   for item in checkpoint_index["batches"]) != 9):
        raise RuntimeError("96-batch checkpoint byte identity/approved archive index does not close")
    r56 = [row for row in stats["comparisons"] if row["control"] == "r56"]
    if len(r56) != 6:
        raise RuntimeError("missing a frozen R58-vs-R56 seed/mode comparison")
    primary = []
    for row in r56:
        task_change = [item for item in row["target_id_changes"] if not item["identical"]]
        cost = row["differences"]["spent_bytes"]
        primary.append({"seed": row["seed"], "mode": row["mode"],
                        "cost_mean_delta_B_per_scene": cost["mean"],
                        "cost_ci95_B_per_scene": cost["ci95_linear"],
                        "task_mean_delta_per_scene": row["differences"]["on_time"]["mean"],
                        "on_time_id_changes": task_change})
    cost_directions = {"lower": sum(item["cost_mean_delta_B_per_scene"] < 0 for item in primary),
                       "equal": sum(item["cost_mean_delta_B_per_scene"] == 0 for item in primary),
                       "higher": sum(item["cost_mean_delta_B_per_scene"] > 0 for item in primary)}
    target_loss_comparisons = sum(any(change["lost"] for change in item["on_time_id_changes"])
                                  for item in primary)
    output = {
        "kind": "round58-fixed-plan-final-machine-verdict/1",
        "execution": "complete_1024_of_1024_on_frozen_development_block",
        "correctness": "source_claim_receipt_checkpoint_ledger_diagnostics_closed",
        "scientific_result": {"comparison": "r58-minus-r56",
                              "cost_direction_counts_across_six_seed_modes": cost_directions,
                              "cross_seed_and_mode_cost_direction_consistent": (
                                  cost_directions["lower"] == 6 or cost_directions["higher"] == 6),
                              "target_loss_comparisons": target_loss_comparisons},
        "historical_development_go_unchanged": False,
        "new_block_development_go": "not_defined_by_frozen_plan",
        "no_deployment_or_population_noninferiority_claim": True,
        "quota": quota,
        "primary_r58_minus_r56": primary,
        "bootstrap": stats["resampling"],
        "evidence": {name: sha(name) for name in (
            "ROUND58_FINAL_STATISTICS.json", "ROUND58_FINAL_SOURCE_INDEX.json",
            "ROUND58_FINAL_ACCOUNTING.json", "ROUND58_FINAL_DIAGNOSTIC_COVERAGE.json",
            "ROUND58_FINAL_CHECKPOINT_INDEX.json")},
        "limitations": [
            "development scenes are not independent deployment or non-inferiority evidence",
            "historical gen1/gen2 48 optimizer calls are evidence-supported, partly reconstructed, not gen3 effective updates",
            "external interruption at seed11 episode43 has unknown exact simulation step count",
            "CI containing zero does not establish equivalence",
        ],
    }
    (HERE / "ROUND58_FINAL_MACHINE_VERDICT.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"execution": output["execution"],
                      "science": output["scientific_result"],
                      "primary": primary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
