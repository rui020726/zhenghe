import json, os
BASE = os.path.join("D:" + os.sep, "code", "\u6574\u5408", "zhenghe-round42", "runs",
                    "comm_bc_initialized_constrained_ppo_20261007_r2")
R73 = os.path.join(BASE, "round73_pure_cost_actor_ablation", "raw")
R71 = os.path.join(BASE, "round71_critic_initialization_calibration", "raw")
def jload(p):
    with open(p, "rb") as f:
        return json.loads(f.read().decode("utf-8"))
def tape_events(path):
    evs = []
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            o = json.loads(raw.decode("utf-8"))
            free = o.get("free_one_shot_choices") or []
            targets = o.get("targets") or []
            feats = o.get("features") or []
            actions = o.get("actions") or {}
            for t in free:
                if t not in targets:
                    continue
                i = targets.index(t)
                if i >= len(feats):
                    continue
                fv = feats[i]
                ready = int(round(fv[18] * 1000)) - int(round(fv[0]))
                evs.append({"target": t, "action": actions.get(t), "ready": ready,
                            "region": "early" if ready <= 147 else "late", "step": o.get("step")})
    return evs
out = {}
for seed in (11, 22, 33):
    per = []
    for sc in range(1800000, 1800032):
        a = jload(os.path.join(R73, "rows", "v2_r73eval_model_new_ep128_s%d_greedy_s%d_a1.json" % (seed, sc)))
        b = jload(os.path.join(R71, "rows", "v2_r71eval_model_new_ep128_s%d_greedy_a1_s%d_a1.json" % (seed, sc)))
        ia = set(e["task_id"] for e in (a.get("task_reward_events") or []) if e.get("on_time") and e.get("credited"))
        ib = set(e["task_id"] for e in (b.get("task_reward_events") or []) if e.get("on_time") and e.get("credited"))
        lost = sorted(ib - ia)
        if not lost:
            continue
        ea = {e["target"]: e for e in tape_events(a["artefacts"]["tape"])}
        eb = {e["target"]: e for e in tape_events(b["artefacts"]["tape"])}
        for t in lost:
            per.append({"scene": sc, "target": t,
                        "r71_action": (eb.get(t) or {}).get("action"),
                        "r71_region": (eb.get(t) or {}).get("region"),
                        "r71_ready": (eb.get(t) or {}).get("ready"),
                        "r73_action": (ea.get(t) or {}).get("action"),
                        "r73_region": (ea.get(t) or {}).get("region"),
                        "r73_ready": (ea.get(t) or {}).get("ready"),
                        "r73_bytes": a.get("spent_bytes"), "r71_bytes": b.get("spent_bytes")})
    out["s%d" % seed] = {"lost_events": per,
                         "all_lost_were_late_sends_in_r71": all(
                             e["r71_action"] == 1 and e["r71_region"] == "late" for e in per),
                         "all_lost_are_late_skips_in_r73": all(
                             e["r73_action"] == 0 and e["r73_region"] == "late" for e in per)}
with open(os.path.join(R73, "_closeout_lost_target_mechanism.json"), "wb") as f:
    f.write((json.dumps(out, indent=1, sort_keys=True) + "\n").encode("utf-8"))
print(json.dumps({k: {"n": len(v["lost_events"]),
                      "all_late_send_in_r71": v["all_lost_were_late_sends_in_r71"],
                      "all_late_skip_in_r73": v["all_lost_are_late_skips_in_r73"]}
                  for k, v in out.items()}, indent=1))
print(json.dumps(out["s33"]["lost_events"][:4], indent=1))