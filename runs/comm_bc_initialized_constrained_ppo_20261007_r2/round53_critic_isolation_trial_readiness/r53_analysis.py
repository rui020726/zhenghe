"""ROUND53 same-source statistics, machine verdict, report, and git-external manifest.

All compared rows come from the hash-closed registries built by
``r53_execute.verify_baseline_reuse`` plus the R53 private raw rows.  No row is
regenerated; the statistics are computed once from the persisted originals.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
RUN_ROOT = HERE.parent
SEEDS = (11, 22, 33)
SCENES = tuple(range(1_301_000, 1_301_016))
RNG_SEED = 20261008
DRAWS = 10_000
BILLED_BYTES_THRESHOLD = 4838.25

sys.path.insert(0, str(HERE))
from r53_contracts import TRAINING_VERSION, sha256_file  # noqa: E402
from r53_execute import read_json, verify_baseline_reuse  # noqa: E402


def on_time_ids(row: dict[str, Any]) -> list[str]:
    return sorted({event["task_id"] for event in row.get("task_reward_events", [])
                   if event.get("on_time") and event.get("credited")})


def first_reports(row: dict[str, Any]) -> int:
    return int(sum(1 for item in row.get("emission_sequence", [])
                   if len(item) >= 6 and item[5] is True))


def row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "on_time": int(row["on_time"]),
        "spent_bytes": int(row["spent_bytes"]),
        "first_reports": first_reports(row),
        "on_time_task_ids": on_time_ids(row),
    }


def load_verified_row(entry: dict[str, Any]) -> dict[str, Any]:
    path = Path(entry["path"])
    sha = entry.get("sha256") or entry.get("source_sha256")
    if sha and sha256_file(path) != sha:
        raise RuntimeError(f"row hash mismatch in statistics input: {path}")
    return read_json(path)


def collect_isolated_rows() -> dict[tuple[int, str], dict[int, dict[str, Any]]]:
    arms: dict[tuple[int, str], dict[int, dict[str, Any]]] = {}
    for path in sorted((RAW / "rows").glob("v2_r44_eval_new_s*_a1.json")):
        row = read_json(path)
        if (row.get("model_arm") != "NEW" or row.get("eval_mode") not in ("greedy", "sampled")
                or int(row.get("steps", -1)) != 1000
                or row.get("training_version") != TRAINING_VERSION
                or int(row.get("scenario_seed", -1)) not in SCENES
                or int(row.get("training_seed", -1)) not in SEEDS):
            raise RuntimeError(f"unexpected isolated eval row: {path}")
        key = (int(row["training_seed"]), str(row["eval_mode"]))
        arms.setdefault(key, {})[int(row["scenario_seed"])] = row_metrics(row)
    expected = {(seed, mode): set(SCENES) for seed in SEEDS for mode in ("greedy", "sampled")}
    if {key: set(value) for key, value in arms.items()} != expected or len(arms) != 6:
        raise RuntimeError("isolated NEW eval rows do not close 3x2x16")
    return arms


def collect_bc_sampled_rows() -> dict[int, dict[int, dict[str, Any]]]:
    arms: dict[int, dict[int, dict[str, Any]]] = {}
    for path in sorted((RAW / "rows").glob("*.json")):
        row = read_json(path)
        if (row.get("model_arm") == "BC_ONLY_R2" and row.get("eval_mode") == "sampled"
                and row.get("r53_evaluation_role") == "BC_POST_SAMPLED_EVAL"):
            seed = int(row["training_seed"]); scene = int(row["scenario_seed"])
            if (row.get("training_version") != TRAINING_VERSION
                    or int(row.get("steps", -1)) != 1000 or scene not in SCENES or seed not in SEEDS):
                raise RuntimeError(f"unexpected BC sampled row: {path}")
            arms.setdefault(seed, {})[scene] = row_metrics(row)
    if {seed: set(value) for seed, value in arms.items()} != {seed: set(SCENES) for seed in SEEDS}:
        raise RuntimeError("BC sampled rows do not close 3x16")
    return arms


def matrix(values: dict[int, dict[str, float]]) -> tuple[list[str], np.ndarray]:
    scenes = sorted(values)
    names = [str(scene) for scene in scenes]
    columns = list(values[scenes[0]].keys())
    array = np.array([[values[scene][column] for column in columns] for scene in scenes],
                     dtype=float)
    return names, array


def main() -> dict[str, Any]:
    baseline = verify_baseline_reuse()
    isolated = collect_isolated_rows()
    bc_sampled = collect_bc_sampled_rows()

    bc_greedy: dict[int, dict[int, dict[str, Any]]] = {}
    for entry in baseline["bc_greedy"]["keys"]:
        row = load_verified_row(entry)
        bc_greedy.setdefault(int(entry["seed"]), {})[int(entry["scene"])] = row_metrics(row)
    shared_ppo: dict[tuple[int, str], dict[int, dict[str, Any]]] = {}
    for entry in baseline["shared_ppo"]["keys"]:
        row = load_verified_row(entry)
        key = (int(entry["seed"]), str(entry["mode"]))
        shared_ppo.setdefault(key, {})[int(entry["scene"])] = row_metrics(row)
    dev_rows: dict[str, dict[int, dict[str, Any]]] = {}
    for label in ("immediate_dev", "late_dev"):
        for entry in baseline["development_references"][label]:
            row = load_verified_row(entry)
            dev_rows.setdefault(label, {})[int(entry["scene"])] = row_metrics(row)

    rng = np.random.Generator(np.random.PCG64(RNG_SEED))
    index_matrix = rng.integers(0, len(SCENES), size=(DRAWS, len(SCENES)))
    check_values = np.arange(1.0, 17.0)
    hand = float(check_values[index_matrix[0]].mean())
    vectorized = float(check_values[index_matrix[0]].mean())
    independent = float(sum(float(check_values[i]) for i in index_matrix[0]) / len(SCENES))
    if abs(vectorized - independent) > 1e-9 or abs(hand - independent) > 1e-9:
        raise RuntimeError("bootstrap indices usage check failed")
    indices_check = {"values": "arange(1..16)", "first_draw_mean_vectorized": vectorized,
                     "first_draw_mean_independent_loop": independent, "passed": True}

    def boot_ci(values: np.ndarray) -> dict[str, Any]:
        means = values[index_matrix].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5], method="linear")
        return {"point": float(values.mean()), "ci95_linear": [float(lo), float(hi)]}

    comparisons: list[dict[str, Any]] = []
    for seed in SEEDS:
        for mode in ("greedy", "sampled"):
            iso = isolated[(seed, mode)]
            iso_scene_names, iso_arr = matrix(
                {s: {"on_time": m["on_time"], "spent_bytes": m["spent_bytes"],
                     "first_reports": m["first_reports"]} for s, m in iso.items()})
            controls: dict[str, dict[int, dict[str, Any]]] = {
                "shared_ppo": shared_ppo.get((seed, mode), {}),
                "immediate_dev": dev_rows["immediate_dev"],
                "late_dev": dev_rows["late_dev"],
            }
            if mode == "greedy":
                controls["bc_greedy"] = bc_greedy[seed]
            if mode == "sampled":
                controls["bc_sampled"] = bc_sampled[seed]
            for control_name, control in controls.items():
                _, ctrl_arr = matrix(
                    {s: {"on_time": m["on_time"], "spent_bytes": m["spent_bytes"],
                         "first_reports": m["first_reports"]} for s, m in control.items()})
                for column, column_name in enumerate(("on_time", "spent_bytes", "first_reports")):
                    diff = iso_arr[:, column] - ctrl_arr[:, column]
                    comparisons.append({
                        "seed": seed, "mode": mode, "control_name": control_name,
                        "metric": column_name, "direction": "isolated - control",
                        "isolated": boot_ci(iso_arr[:, column]),
                        "control": boot_ci(ctrl_arr[:, column]),
                        "difference": boot_ci(diff),
                    })

    per_seed_summary = []
    for seed in SEEDS:
        iso_greedy = isolated[(seed, "greedy")]
        immediate = dev_rows["immediate_dev"]
        matching_scenes = [scene for scene in SCENES
                           if iso_greedy[scene]["on_time_task_ids"] == immediate[scene]["on_time_task_ids"]]
        mean_billed = float(np.mean([iso_greedy[s]["spent_bytes"] for s in SCENES]))
        greedy_vs_shared = next(c for c in comparisons
                                if c["seed"] == seed and c["mode"] == "greedy"
                                and c["control_name"] == "shared_ppo" and c["metric"] == "on_time")
        sampled_vs_bc = next(c for c in comparisons
                             if c["seed"] == seed and c["mode"] == "sampled"
                             and c["control_name"] == "bc_sampled" and c["metric"] == "on_time")
        lost = sorted({t for s in SCENES for t in immediate[s]["on_time_task_ids"]}
                      - {t for s in SCENES for t in iso_greedy[s]["on_time_task_ids"]})
        saved = sorted({t for s in SCENES for t in iso_greedy[s]["on_time_task_ids"]}
                       - {t for s in SCENES for t in immediate[s]["on_time_task_ids"]})
        per_seed_summary.append({
            "seed": seed,
            "greedy_on_time_total": int(sum(iso_greedy[s]["on_time"] for s in SCENES)),
            "greedy_mean_billed_bytes": mean_billed,
            "greedy_mean_billed_bytes_below_threshold": mean_billed < BILLED_BYTES_THRESHOLD,
            "greedy_scenes_with_immediate_identical_on_time_ids": len(matching_scenes),
            "greedy_vs_immediate_lost_targets": lost,
            "greedy_vs_immediate_saved_targets": saved,
            "greedy_vs_shared_ppo_on_time_difference_point": greedy_vs_shared["difference"]["point"],
            "sampled_vs_bc_sampled_on_time_difference_point": sampled_vs_bc["difference"]["point"],
        })

    development_go = all(item["greedy_scenes_with_immediate_identical_on_time_ids"] == 16
                         and item["greedy_mean_billed_bytes_below_threshold"]
                         for item in per_seed_summary)

    statistics = {
        "kind": "round53-statistics/1",
        "status": "complete",
        "training_version": TRAINING_VERSION,
        "unit": "development scenario",
        "scenes": [str(scene) for scene in SCENES],
        "resampling": {"rng": "numpy Generator PCG64", "seed": RNG_SEED,
                       "draws": DRAWS, "sampling": "with replacement over 16 scenes",
                       "indices_usage": "values[indices].mean(axis=1)",
                       "indices_usage_check": indices_check,
                       "interval": "linear percentile 95% descriptive interval"},
        "difference_direction": "isolated candidate - control",
        "comparisons": comparisons,
        "per_seed_summary": per_seed_summary,
    }
    (HERE / "ROUND53_STATISTICS.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checkpoint_index = {"kind": "round53-checkpoint-index/1", "seeds": {}}
    for seed in SEEDS:
        summary = read_json(RAW / f"ROUND46_TRAINING_s{seed}.json")
        entries = {}
        for name, path in (
            ("ep0", RAW / "checkpoints" / f"v2_train_s{seed}_ep0.pt"),
            ("ep64_after_b15", RAW / "checkpoints" / f"v2_train_s{seed}_after_b15.pt"),
            ("ep128", RAW / "checkpoints" / f"v2_train_s{seed}_ep128.pt"),
            ("rolling", RAW / "checkpoints" / f"v2_train_s{seed}.pt"),
        ):
            if path.is_file():
                entries[name] = {"path": str(path), "sha256": sha256_file(path),
                                 "size_bytes": path.stat().st_size}
            else:
                entries[name] = {"path": str(path), "missing": True}
        checkpoint_index["seeds"][str(seed)] = {
            "checkpoints": entries,
            "lambda_chain": summary["fixed_ep128_report"]["lambda_history"],
            "ep128_checkpoint_sha256": summary["ep128_checkpoint_sha256"],
        }
    (HERE / "ROUND53_CHECKPOINT_INDEX.json").write_text(
        json.dumps(checkpoint_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    verdict = {
        "kind": "round53-machine-verdict/1",
        "technical_execution_complete": True,
        "claim_closure": {"normal": 528, "reserve": 0, "hard_cap": 552},
        "critic_isolation_route": "critic_representation_stop_gradient/1",
        "training_version": TRAINING_VERSION,
        "development_go": development_go,
        "development_gate": {
            "criterion": "per seed: greedy on-time target ID sets identical to immediate in 16/16 scenes AND mean billed bytes < 4838.25 B/episode",
            "per_seed": per_seed_summary,
        },
        "claims_not_established": [
            "no independent confirmation run",
            "no formal non-inferiority test",
            "no population-zero-loss or generalization claim",
            "CI containing 0 is not equivalence",
        ],
    }
    (HERE / "ROUND53_MACHINE_VERDICT.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_entries = []
    seen = set()
    def add(path: Path):
        path = Path(path)
        if not path.is_file() or path in seen or "__pycache__" in path.parts:
            return
        seen.add(path)
        manifest_entries.append({"path": str(path), "size_bytes": path.stat().st_size,
                                 "sha256": sha256_file(path)})
    for path in RAW.rglob("*"):
        add(path)
    for entry in (baseline["bc_greedy"]["keys"] + baseline["shared_ppo"]["keys"]
                  + baseline["development_references"]["immediate_dev"]
                  + baseline["development_references"]["late_dev"]):
        add(Path(entry["path"]))
    for model in read_json(RUN_ROOT / "ROUND46_R2_V6_INPUT_IDENTITY.json")["bc_models_by_seed"].values():
        add(Path(model["checkpoint_path"]))
    manifest = {"kind": "round53-git-external-manifest/1", "entries": manifest_entries}
    (HERE / "ROUND53_GIT_EXTERNAL_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {"status": "complete", "development_go": development_go,
            "comparisons": len(comparisons), "manifest_entries": len(manifest_entries)}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
