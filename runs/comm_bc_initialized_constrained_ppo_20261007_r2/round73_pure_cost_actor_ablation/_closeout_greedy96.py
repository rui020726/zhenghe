# ROUND73 read-only close-out: greedy 96-cell identity + descriptive NEW-R71 paired comparison.
# stdlib only; parses existing rows/tapes; writes only the OUT json. No torch, no model, no environment.
import json, os, sys, hashlib, collections

BASE = os.path.join("D:" + os.sep, "code", "整合", "zhenghe-round42", "runs",
                    "comm_bc_initialized_constrained_ppo_20261007_r2")
R73 = os.path.join(BASE, "round73_pure_cost_actor_ablation", "raw")
R71 = os.path.join(BASE, "round71_critic_initialization_calibration", "raw")
R66 = os.path.join(BASE, "round66_bc_early_path_preserving_late_adaptation", "raw")
OUT = sys.argv[1]

SCENES = list(range(1800000, 1800032))
SEEDS = [11, 22, 33]
CUT = 147

def jload(p):
    with open(p, "rb") as f:
        return json.loads(f.read().decode("utf-8"))

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

R73_VERSION = jload(os.path.join(os.path.dirname(R73), "ROUND73_FREEZE.json"))["intervention"]["training_version"]
R71_VERSION = jload(os.path.join(R71, "ROUND71_STATISTICS.json"))["training_version"]

def on_time_ids(row):
    out = set()
    for e in row.get("task_reward_events") or []:
        if e.get("on_time") and e.get("credited"):
            out.add(e.get("task_id"))
    return out

def tape_stats(path):
    st = {"early_send": 0, "early_skip": 0, "late_send": 0, "late_skip": 0,
          "unknown": 0, "lines": 0, "free_total": 0, "events": []}
    if not os.path.exists(path):
        st["missing"] = True
        return st
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            o = json.loads(raw.decode("utf-8"))
            st["lines"] += 1
            free = o.get("free_one_shot_choices") or []
            targets = o.get("targets") or []
            feats = o.get("features") or []
            actions = o.get("actions") or {}
            probs = o.get("send_probabilities") or {}
            for t in free:
                if t not in targets:
                    st["unknown"] += 1
                    continue
                i = targets.index(t)
                if i >= len(feats):
                    st["unknown"] += 1
                    continue
                fv = feats[i]
                ready = int(round(fv[18] * 1000)) - int(round(fv[0]))
                act = actions.get(t)
                st["free_total"] += 1
                if act == 1:
                    if ready <= CUT:
                        st["early_send"] += 1
                    else:
                        st["late_send"] += 1
                elif act == 0:
                    if ready <= CUT:
                        st["early_skip"] += 1
                    else:
                        st["late_skip"] += 1
                else:
                    st["unknown"] += 1
                st["events"].append({"step": o.get("step"), "target": t, "action": act,
                                     "ready_step": ready,
                                     "region": "early" if ready <= CUT else "late",
                                     "send_prob": probs.get(t)})
    return st
def load_arm(rows_dir, prefix_filter, mode, seeds):
    """Discover rows by content, not by guessed filenames."""
    found = {}
    for name in sorted(os.listdir(rows_dir)):
        if not name.endswith(".json"):
            continue
        if not prefix_filter(name):
            continue
        row = jload(os.path.join(rows_dir, name))
        if row.get("kind") != "eval":
            continue
        if mode is not None and row.get("eval_mode") != mode:
            continue
        key = (row.get("training_seed"), row.get("scenario_seed"), row.get("attempt"))
        found.setdefault(key, []).append((name, row))
    return found

def identity_check(row, expect_version, expect_actor_objective, seed, scene):
    problems = []
    cli = row.get("checkpoint_load_identity") or {}
    if cli.get("training_version") != expect_version:
        problems.append("training_version_mismatch")
    if expect_actor_objective is None:
        if cli.get("actor_objective"):
            problems.append("unexpected_actor_objective")
    elif cli.get("actor_objective") != expect_actor_objective:
        problems.append("actor_objective_mismatch")
    if cli.get("optimizer_restored") is not False:
        problems.append("optimizer_restored_not_false")
    if cli.get("rng_restored") is not False:
        problems.append("rng_restored_not_false")
    pi = cli.get("parameter_identity") or {}
    if row.get("training_seed") != seed:
        problems.append("seed_mismatch")
    if row.get("scenario_seed") != scene:
        problems.append("scene_mismatch")
    if row.get("model_arm") != "NEW_EP128":
        problems.append("arm_mismatch")
    if row.get("policy") != "learned":
        problems.append("policy_mismatch")
    if row.get("budget_bytes") != 29300:
        problems.append("budget_mismatch")
    if row.get("b0_bytes") != 10000.0:
        problems.append("b0_mismatch")
    if row.get("steps") != 1000:
        problems.append("steps_mismatch")
    if row.get("discovered") != 6:
        problems.append("discovered_not_6")
    if row.get("spent_bytes") != row.get("airtime_tap_total"):
        problems.append("bytes_tap_mismatch")
    if len(on_time_ids(row)) != row.get("on_time"):
        problems.append("on_time_ids_count_mismatch")
    if (row.get("completed") or 0) + (row.get("expired") or 0) + (row.get("active") or 0) + (row.get("unassigned") or 0) != row.get("discovered"):
        problems.append("task_count_closure_mismatch")
    return problems, cli, pi

def world_check(row):
    """Read the frozen world configuration out of the row's own lifecycle log."""
    art = row.get("artefacts") or {}
    lp = art.get("lifecycle")
    if not lp or not os.path.exists(lp):
        return {"missing": True}
    with open(lp, "rb") as f:
        first = f.readline()
    o = json.loads(first.decode("utf-8"))
    cfg = ((o.get("detail") or {}).get("configuration") or {})
    sig = ((o.get("detail") or {}).get("run_signature") or {})
    return {"num_goals": cfg.get("num_goals"), "comm_budget_bytes": cfg.get("comm_budget_bytes"),
            "comm_ack": cfg.get("comm_ack"), "steps": cfg.get("steps"),
            "comm_delivery_probability": cfg.get("comm_delivery_probability"),
            "config_hash": sig.get("config_hash"),
            "piloc_checkpoint_sha": (sig.get("checkpoints") or {}).get("piloc"),
            "ctde_checkpoint_sha": (sig.get("checkpoints") or {}).get("ctde"),
            "seed": (o.get("detail") or {}).get("scenario", {}).get("seed")}

report = {"read_only": True, "eval_block": "1800000..1800031 (already-used development block)",
          "r73_training_version": R73_VERSION, "r71_training_version": R71_VERSION}

# ---------- R73 NEW greedy 96 ----------
r73 = {}
ident = {"r73": {"cells": 0, "problems": [], "param_sha_by_seed": {}, "files_missing": []}}
for seed in SEEDS:
    shas = set()
    for sc in SCENES:
        cands = []
        for name in sorted(os.listdir(os.path.join(R73, "rows"))):
            if not name.startswith("v2_r73eval_model_new_ep128_s%d_greedy" % seed):
                continue
            if ("_s%d_a" % sc) not in name:
                continue
            cands.append(name)
        if len(cands) != 1:
            ident["r73"]["problems"].append({"seed": seed, "scene": sc, "issue": "row_file_count_%d" % len(cands)})
            continue
        row = jload(os.path.join(R73, "rows", cands[0]))
        probs, cli, pi = identity_check(row, R73_VERSION, "pure-cost/1", seed, sc)
        if row.get("eval_mode") != "greedy":
            probs.append("mode_not_greedy")
        art = row.get("artefacts") or {}
        for k in ("row", "tape", "transcript", "lifecycle"):
            v = art.get(k)
            if not v or not os.path.exists(v):
                ident["r73"]["files_missing"].append({"seed": seed, "scene": sc, "artefact": k})
        ts = tape_stats(art.get("tape"))
        osh = row.get("collector_summary", {}).get("one_shot_counts", {})
        if ts.get("free_total") != osh.get("free_choices"):
            probs.append("tape_free_total_mismatch")
        if ts.get("early_send", 0) + ts.get("late_send", 0) != osh.get("send_now_choices"):
            probs.append("tape_send_count_mismatch")
        if ts.get("early_skip", 0) + ts.get("late_skip", 0) != osh.get("skip_episode_choices"):
            probs.append("tape_skip_count_mismatch")
        shas.add(pi.get("parameter_sha256"))
        if probs:
            ident["r73"]["problems"].append({"seed": seed, "scene": sc, "issues": probs})
        r73[(seed, sc)] = {"row": row, "tape": ts, "file": cands[0]}
        ident["r73"]["cells"] += 1
    ident["r73"]["param_sha_by_seed"][str(seed)] = sorted(x for x in shas if x)
# ---------- R71 control greedy 96 ----------
r71 = {}
ident["r71"] = {"cells": 0, "problems": [], "param_sha_by_seed": {}, "files_missing": []}
for seed in SEEDS:
    shas = set()
    for sc in SCENES:
        cands = [n for n in sorted(os.listdir(os.path.join(R71, "rows")))
                 if n.startswith("v2_r71eval_model_new_ep128_s%d_greedy" % seed) and ("_s%d_a" % sc) in n]
        if len(cands) != 1:
            ident["r71"]["problems"].append({"seed": seed, "scene": sc, "issue": "row_file_count_%d" % len(cands)})
            continue
        row = jload(os.path.join(R71, "rows", cands[0]))
        probs, cli, pi = identity_check(row, R71_VERSION, None, seed, sc)
        if row.get("eval_mode") != "greedy":
            probs.append("mode_not_greedy")
        art = row.get("artefacts") or {}
        for k in ("row", "tape", "transcript", "lifecycle"):
            v = art.get(k)
            if not v or not os.path.exists(v):
                ident["r71"]["files_missing"].append({"seed": seed, "scene": sc, "artefact": k})
        ts = tape_stats(art.get("tape"))
        osh = row.get("collector_summary", {}).get("one_shot_counts", {})
        if ts.get("free_total") != osh.get("free_choices"):
            probs.append("tape_free_total_mismatch")
        if ts.get("early_send", 0) + ts.get("late_send", 0) != osh.get("send_now_choices"):
            probs.append("tape_send_count_mismatch")
        if ts.get("early_skip", 0) + ts.get("late_skip", 0) != osh.get("skip_episode_choices"):
            probs.append("tape_skip_count_mismatch")
        shas.add(pi.get("parameter_sha256"))
        if probs:
            ident["r71"]["problems"].append({"seed": seed, "scene": sc, "issues": probs})
        r71[(seed, sc)] = {"row": row, "tape": ts, "file": cands[0]}
        ident["r71"]["cells"] += 1
    ident["r71"]["param_sha_by_seed"][str(seed)] = sorted(x for x in shas if x)

# ---------- rule arms (read-only R66) ----------
rules = {}
for arm, prefix in (("IMMEDIATE", "v2_r66eval2_rule_immediate_"),
                    ("LATE_DISCOVERY_GATE", "v2_r66eval2_rule_late_discovery_gate_"),
                    ("ALWAYS_HOLD", "v2_r66eval2_rule_always_hold_")):
    cells = {}
    dup = []
    for name in sorted(os.listdir(os.path.join(R66, "rows"))):
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        row = jload(os.path.join(R66, "rows", name))
        sc = row.get("scenario_seed")
        att = row.get("attempt")
        if sc in cells:
            dup.append({"scene": sc, "kept": cells[sc]["file"], "dropped": name,
                        "kept_attempt": cells[sc]["row"].get("attempt"), "dropped_attempt": att})
            if att == 1:
                dup[-1]["note"] = "attempt-1 row kept; later duplicate dropped from the mean"
            continue
        cells[sc] = {"row": row, "file": name}
    on_time_total = sum(c["row"].get("on_time") or 0 for c in cells.values())
    bytes_total = sum(c["row"].get("spent_bytes") or 0 for c in cells.values())
    ids = set()
    for c in cells.values():
        ids |= on_time_ids(c["row"])
    rules[arm] = {"scenes": len(cells), "duplicate_rows_dropped": dup,
                  "on_time_total": on_time_total, "bytes_total": bytes_total,
                  "bytes_per_scene": (bytes_total / len(cells)) if cells else None,
                  "distinct_on_time_target_ids": sorted(ids),
                  "files": sorted(c["file"] for c in cells.values())}
report["rule_arms"] = rules

# ---------- paired comparison ----------
per_seed = {}
for seed in SEEDS:
    scenes = []
    lost_all, gained_all = set(), set()
    r73_bytes, r71_bytes = [], []
    agg = {"r73": {"early_send": 0, "early_skip": 0, "late_send": 0, "late_skip": 0},
           "r71": {"early_send": 0, "early_skip": 0, "late_send": 0, "late_skip": 0}}
    for sc in SCENES:
        a = r73.get((seed, sc))
        b = r71.get((seed, sc))
        if not a or not b:
            scenes.append({"scene": sc, "missing": True})
            continue
        ra, rb = a["row"], b["row"]
        ia, ib = on_time_ids(ra), on_time_ids(rb)
        lost = sorted(ib - ia)
        gained = sorted(ia - ib)
        lost_all |= set(lost)
        gained_all |= set(gained)
        r73_bytes.append(ra.get("spent_bytes"))
        r71_bytes.append(rb.get("spent_bytes"))
        for k in agg["r73"]:
            agg["r73"][k] += a["tape"].get(k, 0)
            agg["r71"][k] += b["tape"].get(k, 0)
        scenes.append({
            "scene": sc,
            "r73_on_time": ra.get("on_time"), "r71_on_time": rb.get("on_time"),
            "on_time_delta": (ra.get("on_time") or 0) - (rb.get("on_time") or 0),
            "r73_bytes": ra.get("spent_bytes"), "r71_bytes": rb.get("spent_bytes"),
            "bytes_delta": (ra.get("spent_bytes") or 0) - (rb.get("spent_bytes") or 0),
            "lost_vs_r71": lost, "gained_vs_r71": gained,
            "r73_on_time_ids": sorted(ia), "r71_on_time_ids": sorted(ib),
            "r73_emissions": ra.get("emissions"), "r71_emissions": rb.get("emissions"),
            "r73_first_emission_targets": [e[1] for e in (ra.get("emission_sequence") or [])],
            "r71_first_emission_targets": [e[1] for e in (rb.get("emission_sequence") or [])],
            "r73_early_send": a["tape"].get("early_send"), "r73_late_send": a["tape"].get("late_send"),
            "r73_early_skip": a["tape"].get("early_skip"), "r73_late_skip": a["tape"].get("late_skip"),
            "r71_early_send": b["tape"].get("early_send"), "r71_late_send": b["tape"].get("late_send"),
            "r71_early_skip": b["tape"].get("early_skip"), "r71_late_skip": b["tape"].get("late_skip"),
        })
    n = len([s for s in scenes if not s.get("missing")])
    per_seed[str(seed)] = {
        "cells": n,
        "r73_on_time_total": sum(s["r73_on_time"] or 0 for s in scenes if not s.get("missing")),
        "r71_on_time_total": sum(s["r71_on_time"] or 0 for s in scenes if not s.get("missing")),
        "on_time_delta_total": sum(s["on_time_delta"] for s in scenes if not s.get("missing")),
        "r73_bytes_mean": (sum(r73_bytes) / n) if n else None,
        "r71_bytes_mean": (sum(r71_bytes) / n) if n else None,
        "bytes_delta_mean": ((sum(r73_bytes) - sum(r71_bytes)) / n) if n else None,
        "bytes_delta_per_scene": [s["bytes_delta"] for s in scenes if not s.get("missing")],
        "scenes_with_cost_decrease": sum(1 for s in scenes if not s.get("missing") and s["bytes_delta"] < 0),
        "scenes_with_cost_increase": sum(1 for s in scenes if not s.get("missing") and s["bytes_delta"] > 0),
        "scenes_with_cost_equal": sum(1 for s in scenes if not s.get("missing") and s["bytes_delta"] == 0),
        "scenes_with_task_loss": [s["scene"] for s in scenes if not s.get("missing") and s["on_time_delta"] < 0],
        "scenes_with_task_gain": [s["scene"] for s in scenes if not s.get("missing") and s["on_time_delta"] > 0],
        "targets_lost_vs_r71": sorted(lost_all),
        "targets_gained_vs_r71": sorted(gained_all),
        "lost_detail": [{"scene": s["scene"], "targets": s["lost_vs_r71"]} for s in scenes
                        if not s.get("missing") and s["lost_vs_r71"]],
        "gained_detail": [{"scene": s["scene"], "targets": s["gained_vs_r71"]} for s in scenes
                          if not s.get("missing") and s["gained_vs_r71"]],
        "send_skip_early_late": agg,
        "send_skip_delta": {k: agg["r73"][k] - agg["r71"][k] for k in agg["r73"]},
        "per_scene": scenes,
    }
report["identity_audit"] = ident
report["per_seed_greedy"] = per_seed

# ---------- gaps that must not be silently filled ----------
r73_rows = [n for n in os.listdir(os.path.join(R73, "rows")) if n.endswith(".json")]
sampled_cells = [n for n in r73_rows if n.startswith("v2_r73eval_model_new_ep128_") and "_sampled_" in n]
focal_cells = [n for n in r73_rows if n.startswith("v2_r73eval_focal_")]
report["gaps"] = {
    "sampled_cells_on_disk": len(sampled_cells),
    "focal_cells_on_disk": len(focal_cells),
    "note": "sampled and focal cells are reported separately and never mixed into the greedy primary comparison",
}
report["gaps"]["sampled_missing_cells"] = sorted(
    [["s%d" % s, sc] for s in SEEDS for sc in SCENES
     if not any(n.startswith("v2_r73eval_model_new_ep128_s%d_sampled" % s) and ("_s%d_a" % sc) in n
                for n in r73_rows)])
report["gaps"]["sampled_present_cells"] = sorted(
    [["s%d" % s, sc] for s in SEEDS for sc in SCENES
     if any(n.startswith("v2_r73eval_model_new_ep128_s%d_sampled" % s) and ("_s%d_a" % sc) in n
            for n in r73_rows)])
report["gaps"]["focal_present_cells"] = sorted(focal_cells)

with open(OUT, "wb") as f:
    f.write((json.dumps(report, indent=1, sort_keys=True) + "\n").encode("utf-8"))
print("wrote", OUT)