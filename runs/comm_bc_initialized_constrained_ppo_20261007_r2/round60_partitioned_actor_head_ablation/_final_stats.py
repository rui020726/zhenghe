"""ROUND60 final paired statistics: NEW vs R58 (primary), and NEW vs immediate/late, with exact
on-time target ID sets and lost/rescued targets. Reuses R58's frozen scene order, PCG64(20261008)
10000-draw bootstrap index and linear interval. Greedy is primary; sampled reported separately;
seeds not pooled; the 256 reused cells are never counted as new launches."""
import json, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
NEW_ROWS = RAW / "rows"
R58_ROWS = HERE.parent / "round58_gen3_resume_v35" / "raw" / "rows"
SCENES = list(range(1700000, 1700032))
SEEDS = (11, 22, 33)
MODES = ("greedy", "sampled")

def on_time_ids(lifecycle):
    ids = set()
    p = Path(lifecycle)
    if not p.is_file():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event_type") == "final" and e.get("target_id") and e["target_id"] != "__run__":
            if e.get("status") == "completed":
                ids.add(e["target_id"])
    return ids

def load_model(root, pat):
    out = {}
    for p in Path(root).glob(pat):
        r = json.loads(p.read_text(encoding="utf-8"))
        key = (int(r["training_seed"]), r["eval_mode"], int(r["scenario_seed"]))
        out[key] = {"on_time": int(r["on_time"]), "spent": int(r["spent_bytes"]),
                    "ids": on_time_ids(r.get("artefacts", {}).get("lifecycle"))}
    return out

def load_rule(root, arm):
    out = {}
    for p in Path(root).glob(f"v2_r58_rule_{arm}_s*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        scene = int(r["scenario_seed"])
        if scene in out:
            continue
        out[scene] = {"on_time": int(r["on_time"]), "spent": int(r["spent_bytes"]),
                      "ids": on_time_ids(r.get("artefacts", {}).get("lifecycle"))}
    return out

idx = np.random.Generator(np.random.PCG64(20261008)).integers(0, 32, size=(10000, 32))
idx_sha = hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest()

new = load_model(NEW_ROWS, "v2_r60_eval_new_s*_*_s*.json")
r58 = load_model(R58_ROWS, "v2_r44_eval_new_s*_*_s*.json")
imm = load_rule(R58_ROWS, "immediate")
late = load_rule(R58_ROWS, "late")

def compare(ref, ref_is_per_seedmode):
    per = {}
    for seed in SEEDS:
        for mode in MODES:
            dc, dt = [], []
            lost, rescued, scene_changes = set(), set(), []
            for scene in SCENES:
                n = new[(seed, mode, scene)]
                r = ref[(seed, mode, scene)] if ref_is_per_seedmode else ref[scene]
                dc.append(n["spent"] - r["spent"]); dt.append(n["on_time"] - r["on_time"])
                if n["ids"] is not None and r["ids"] is not None:
                    lo = r["ids"] - n["ids"]; re = n["ids"] - r["ids"]
                    if lo or re:
                        scene_changes.append({"scene": scene, "lost": sorted(lo), "rescued": sorted(re)})
                        lost |= lo; rescued |= re
            dc = np.asarray(dc, float); dt = np.asarray(dt, float)
            bc, bt = dc[idx].mean(1), dt[idx].mean(1)
            per[f"{seed}/{mode}"] = {
                "ref_on_time_total": int(sum((ref[(seed, mode, s)] if ref_is_per_seedmode else ref[s])["on_time"] for s in SCENES)),
                "new_on_time_total": int(sum(new[(seed, mode, s)]["on_time"] for s in SCENES)),
                "task_delta_per_scene": float(dt.mean()),
                "task_ci95": [float(np.quantile(bt, .025)), float(np.quantile(bt, .975))],
                "ref_bytes_per_scene": float(np.mean([(ref[(seed, mode, s)] if ref_is_per_seedmode else ref[s])["spent"] for s in SCENES])),
                "new_bytes_per_scene": float(np.mean([new[(seed, mode, s)]["spent"] for s in SCENES])),
                "cost_delta_B_per_scene": float(dc.mean()),
                "cost_ci95": [float(np.quantile(bc, .025)), float(np.quantile(bc, .975))],
                "lost_targets": sorted(lost), "rescued_targets": sorted(rescued),
                "scene_id_changes": scene_changes}
    return per

result = {"kind": "round60-final-statistics/1",
          "primary_comparison": "NEW_minus_R58_same_seed_mode_scene",
          "greedy_primary": True, "sampled_separate": True, "seeds_not_pooled": True,
          "scene_unit": 32, "reused_cells_not_new_launches": True,
          "bootstrap": {"rng": "PCG64", "seed": 20261008, "draws": 10000, "index_sha256": idx_sha,
                        "matches_r58_frozen_index": idx_sha == "8421defbdc5cf4ba5485e7c1cfd0fb2952c2446720a490ff195d4637ccc7445b",
                        "quantile": "numpy linear 0.025/0.975"},
          "new_minus_r58": compare(r58, True),
          "new_minus_immediate": compare(imm, False),
          "new_minus_late": compare(late, False),
          "cell_counts": {"new": len(new), "r58": len(r58), "immediate": len(imm), "late": len(late)}}
(RAW / "ROUND60_FINAL_STATISTICS.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
# concise console summary
for cmp_name in ("new_minus_r58", "new_minus_immediate", "new_minus_late"):
    print("==", cmp_name)
    for k, v in result[cmp_name].items():
        print(f"  {k}: task {v['ref_on_time_total']}->{v['new_on_time_total']} (d{v['task_delta_per_scene']}) "
              f"cost_delta {v['cost_delta_B_per_scene']} CI[{v['cost_ci95'][0]},{v['cost_ci95'][1]}] "
              f"lost={v['lost_targets']} rescued={v['rescued_targets']}")
print("bootstrap index matches R58:", result["bootstrap"]["matches_r58_frozen_index"])
