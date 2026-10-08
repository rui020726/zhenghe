"""ROUND56 statistics: R56 candidate vs R53 critic-isolation PPO (paired), plus references; frozen bootstrap protocol."""
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Any
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
R53 = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2" / "round53_critic_isolation_trial_readiness"
RAW = HERE / "raw"
SEEDS = (11, 22, 33)
SCENES = tuple(range(1_301_000, 1_301_016))
RNG_SEED, DRAWS, THRESHOLD = 20261008, 10_000, 4838.25

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "round53_critic_isolation_trial_readiness"))
from r56_contracts import TRAINING_VERSION, sha256_file  # noqa: E402
from r53_execute import read_json, verify_baseline_reuse  # noqa: E402


def metrics(row):
    ids = sorted({e["task_id"] for e in row.get("task_reward_events", [])
                  if e.get("on_time") and e.get("credited")})
    firsts = int(sum(1 for it in row.get("emission_sequence", []) if len(it) >= 6 and it[5] is True))
    return {"on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
            "first_reports": firsts, "on_time_task_ids": ids}


def load(entry):
    p = Path(entry["path"])
    if entry.get("sha256") and sha256_file(p) != entry["sha256"]:
        raise RuntimeError(f"hash mismatch {p}")
    return read_json(p)


def main():
    baseline = verify_baseline_reuse()
    arms = {}
    for path in sorted((RAW / "rows").glob("v2_r44_eval_new_s*_a1.json")):
        row = read_json(path)
        if row.get("training_version") != TRAINING_VERSION:
            raise RuntimeError(f"bad version {path}")
        arms.setdefault(("r56", int(row["training_seed"]), str(row["eval_mode"])), {})[
            int(row["scenario_seed"])] = metrics(row)
    for path in sorted((R53 / "raw" / "rows").glob("v2_r44_eval_new_s*_a1.json")):
        row = read_json(path)
        if row.get("training_version") != TRAINING_VERSION.replace("+r56-soft-early-report-prior-beta0p10/1", ""):
            raise RuntimeError(f"bad version {path}")
        arms.setdefault(("r53", int(row["training_seed"]), str(row["eval_mode"])), {})[
            int(row["scenario_seed"])] = metrics(row)
    for entry in baseline["bc_greedy"]["keys"]:
        row = load(entry)
        arms.setdefault(("bc", int(entry["seed"]), "greedy"), {})[int(entry["scene"])] = metrics(row)
    for entry in baseline["bc_sampled_rows"] if "bc_sampled_rows" in baseline else []:
        pass
    # BC sampled rows from R53 raw
    for path in sorted((R53 / "raw" / "rows").glob("*.json")):
        row = read_json(path)
        if (row.get("model_arm") == "BC_ONLY_R2" and row.get("eval_mode") == "sampled"):
            arms.setdefault(("bc", int(row["training_seed"]), "sampled"), {})[
                int(row["scenario_seed"])] = metrics(row)
    for entry in baseline["shared_ppo"]["keys"]:
        row = load(entry)
        arms.setdefault(("shared_ppo", int(entry["seed"]), str(entry["mode"])), {})[
            int(entry["scene"])] = metrics(row)
    for label, name in (("immediate_dev", "immediate"), ("late_dev", "late")):
        for entry in baseline["development_references"][label]:
            row = load(entry)
            arms.setdefault((name, 0, "greedy"), {})[int(entry["scene"])] = metrics(row)
    for key, scenes in arms.items():
        if set(scenes) != set(SCENES):
            raise RuntimeError(f"incomplete arm {key}")

    rng = np.random.Generator(np.random.PCG64(RNG_SEED))
    idx = rng.integers(0, 16, size=(DRAWS, 16))

    def boot(vals):
        m = vals[idx].mean(axis=1)
        lo, hi = np.percentile(m, [2.5, 97.5], method="linear")
        return {"point": float(vals.mean()), "ci95_linear": [float(lo), float(hi)],
                "significant": bool(lo > 0 or hi < 0)}

    def arr(arm, seed, mode, metric):
        d = arms[(arm, seed, mode)]
        return np.array([float(d[s][metric]) for s in SCENES])

    comparisons = []
    controls = {"r53": "r53", "bc": "bc", "shared_ppo": "shared_ppo",
                "immediate": "immediate", "late": "late"}
    for seed in SEEDS:
        for mode in ("greedy", "sampled"):
            for cname, ckey in controls.items():
                ckey_eff = (ckey, seed, mode) if ckey not in ("immediate", "late") else (ckey, 0, "greedy")
                for metric in ("on_time", "spent_bytes", "first_reports"):
                    comparisons.append({
                        "seed": seed, "mode": mode, "control_name": cname, "metric": metric,
                        "isolated": boot(arr("r56", seed, mode, metric)),
                        "control": boot(arr(*ckey_eff, metric)),
                        "difference": boot(arr("r56", seed, mode, metric) - arr(*ckey_eff, metric))})

    per_seed = []
    gate = {}
    for seed in SEEDS:
        g = arms[("r56", seed, "greedy")]
        imm = arms[("immediate", 0, "greedy")]
        match = sum(1 for s in SCENES if g[s]["on_time_task_ids"] == imm[s]["on_time_task_ids"])
        mean_b = float(np.mean([g[s]["spent_bytes"] for s in SCENES]))
        lost = sorted({t for s in SCENES for t in imm[s]["on_time_task_ids"]}
                      - {t for s in SCENES for t in g[s]["on_time_task_ids"]})
        saved = sorted({t for s in SCENES for t in g[s]["on_time_task_ids"]}
                       - {t for s in SCENES for t in imm[s]["on_time_task_ids"]})
        per_seed.append({"seed": seed, "greedy_on_time_total": int(sum(g[s]["on_time"] for s in SCENES)),
                         "greedy_mean_billed_bytes": mean_b,
                         "mean_billed_strictly_below_threshold": bool(mean_b < THRESHOLD),
                         "identical_on_time_id_scenes": match, "lost_targets": lost, "saved_targets": saved})
        sm = {m: float(np.mean([arms[("r53", seed, m)][s]["spent_bytes"] for s in SCENES])) for m in ("greedy", "sampled")}
        gate[str(seed)] = {"r53_mean_billed": sm, "r56_mean_billed": mean_b}
    development_go = all(p["identical_on_time_id_scenes"] == 16
                         and p["mean_billed_strictly_below_threshold"] for p in per_seed)
    out = {"kind": "round56-statistics/1", "training_version": TRAINING_VERSION,
           "scenes": [str(s) for s in SCENES],
           "resampling": {"rng": "numpy Generator PCG64", "seed": RNG_SEED, "draws": DRAWS,
                          "indices_usage": "values[indices].mean(axis=1)",
                          "interval": "linear percentile 95% descriptive"},
           "difference_direction": "R56 - control",
           "comparisons": comparisons, "per_seed_summary": per_seed,
           "development_go": development_go, "r53_vs_r56_means": gate}
    (HERE / "ROUND56_STATISTICS.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"per_seed": per_seed, "development_go": development_go}, ensure_ascii=False, indent=1))
    key_cells = [(c["seed"], c["mode"], c["control_name"], c["metric"], c["difference"])
                 for c in comparisons
                 if c["control_name"] in ("r53", "late") and c["metric"] == "spent_bytes"]
    for k in key_cells:
        print(k[0], k[1], k[2], "diff", round(k[3]["point"], 3), [round(x, 3) for x in k[3]["ci95_linear"]])


if __name__ == "__main__":
    main()
