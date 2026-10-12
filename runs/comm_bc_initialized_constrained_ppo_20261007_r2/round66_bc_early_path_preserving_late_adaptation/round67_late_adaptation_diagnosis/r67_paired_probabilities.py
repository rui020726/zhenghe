"""ROUND67 - paired NEW vs BC saved probabilities per tape record (pure data, no forward).

Task 1 established that NEW and BC take the SAME action for every free first-report candidate in
all 192 eval cells (and the 6 focal cells), and that the two tapes of a cell are step-aligned (the
only per-step difference ever found was a probability value). That makes a per-record pairing of the
saved send probabilities legitimate: same seed, same mode, same scene, same step, same target.

This quantifies how much probability mass actually changed sides, and therefore what a sampled draw
was exposed to. A uniform multinomial draw flips with probability |dp| on the affected row, so the
sum of |dp| over the sampled free rows is the expected number of draw flips; the observed number is
taken from the Task 1 action comparison (zero).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "raw" / "logs"
SEEDS = (11, 22, 33)
MODES = ("greedy", "sampled")
SCENES = [1800000 + i for i in range(32)]
FOCAL = 1700027
HORIZON, READY_CUT = 1000, 147


def ready_step(vec):
    return round(float(vec[18]) * HORIZON) - round(float(vec[0]))


def tape_path(arm, seed, mode, scene):
    return LOGS / f"v2_r66eval2_model_{arm}_s{seed}_{mode}_a1_s{scene}_a1_tape.jsonl"


def focal_tape(arm, seed):
    return LOGS / f"v2_r66eval2_focal_{arm}_s{seed}_a1_s{FOCAL}_a1_tape.jsonl"


def read(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    out = {}
    detail = []
    for seed in SEEDS:
        for mode in MODES:
            for scene in SCENES:
                pairs = list(zip(read(tape_path("new", seed, mode, scene)),
                                 read(tape_path("bc", seed, mode, scene))))
                for rn, rb in pairs:
                    if rn.get("step") != rb.get("step"):
                        raise RuntimeError("tape step misalignment")
                    free = set(rn.get("free") or [])
                    pn = rn.get("send_probabilities") or {}
                    pb = rb.get("send_probabilities") or {}
                    for i, vec in enumerate(rn.get("features") or []):
                        tgt = (rn.get("targets") or [])[i]
                        if tgt not in free or tgt not in pn or tgt not in pb:
                            continue
                        rs = ready_step(vec)
                        route = "late" if rs > READY_CUT else "early"
                        detail.append({"seed": seed, "mode": mode, "scene": scene,
                                       "step": rn.get("step"), "target": tgt, "route": route,
                                       "ready_step": rs, "p_new": pn[tgt], "p_bc": pb[tgt],
                                       "abs_dp": abs(pn[tgt] - pb[tgt]),
                                       "action_new": (rn.get("actions") or {}).get(tgt),
                                       "action_bc": (rb.get("actions") or {}).get(tgt)})
        # focal
        for arm in ("new", "bc"):
            pass
        pairs = list(zip(read(focal_tape("new", seed)), read(focal_tape("bc", seed))))
        for rn, rb in pairs:
            free = set(rn.get("free") or [])
            pn = rn.get("send_probabilities") or {}
            pb = rb.get("send_probabilities") or {}
            for i, vec in enumerate(rn.get("features") or []):
                tgt = (rn.get("targets") or [])[i]
                if tgt not in free or tgt not in pn or tgt not in pb:
                    continue
                rs = ready_step(vec)
                detail.append({"seed": seed, "mode": "greedy", "scene": FOCAL, "focal": True,
                               "step": rn.get("step"), "target": tgt,
                               "route": "late" if rs > READY_CUT else "early",
                               "ready_step": rs, "p_new": pn[tgt], "p_bc": pb[tgt],
                               "abs_dp": abs(pn[tgt] - pb[tgt]),
                               "action_new": (rn.get("actions") or {}).get(tgt),
                               "action_bc": (rb.get("actions") or {}).get(tgt)})

    def block(rows):
        if not rows:
            return None
        return {"n": len(rows),
                "n_action_diff": sum(1 for r in rows if r["action_new"] != r["action_bc"]),
                "n_prob_diff": sum(1 for r in rows if r["abs_dp"] > 0.0),
                "sum_abs_dp": sum(r["abs_dp"] for r in rows),
                "max_abs_dp": max(r["abs_dp"] for r in rows),
                "p_new_min": min(r["p_new"] for r in rows),
                "p_new_max": max(r["p_new"] for r in rows),
                "p_bc_min": min(r["p_bc"] for r in rows),
                "p_bc_max": max(r["p_bc"] for r in rows),
                "n_p_new_ge_0p99": sum(1 for r in rows if r["p_new"] >= 0.99),
                "n_p_bc_ge_0p99": sum(1 for r in rows if r["p_bc"] >= 0.99),
                "n_cross_0p5": sum(1 for r in rows if (r["p_new"] > 0.5) != (r["p_bc"] > 0.5))}

    for seed in SEEDS:
        for mode in MODES:
            for route in ("early", "late"):
                rows = [d for d in detail if d["seed"] == seed and d["mode"] == mode
                        and d["route"] == route]
                out[f"s{seed}/{mode}/{route}"] = block(rows)
    focal_rows = [d for d in detail if d.get("focal")]
    out["focal_1700027"] = block(focal_rows)
    out["focal_1700027_target5"] = block([d for d in focal_rows if d["target"] == "piloc-e1-target-5"])
    out["all_late_free"] = block([d for d in detail if d["route"] == "late"])
    out["all_early_free"] = block([d for d in detail if d["route"] == "early"])

    sampled_late = [d for d in detail if d["mode"] == "sampled" and d["route"] == "late"]
    out["sampled_late_free_expected_draw_flips"] = {
        "expected_flips_sum_abs_dp": sum(d["abs_dp"] for d in sampled_late),
        "rows": len(sampled_late),
        "observed_action_differences": sum(1 for d in sampled_late if d["action_new"] != d["action_bc"]),
        "note": ("a uniform multinomial draw on a 2-way distribution flips with probability "
                 "|dp| on an affected row, so sum|dp| is the expected number of flips; the "
                 "observed count comes from the Task 1 action comparison"),
    }
    greedy_late = [d for d in detail if d["mode"] == "greedy" and d["route"] == "late"]
    out["greedy_late_free"] = {
        "rows": len(greedy_late),
        "observed_action_differences": sum(1 for d in greedy_late if d["action_new"] != d["action_bc"]),
        "n_p_new_below_0p5": sum(1 for d in greedy_late if d["p_new"] < 0.5),
        "n_p_bc_below_0p5": sum(1 for d in greedy_late if d["p_bc"] < 0.5),
    }

    result = {"kind": "round67-paired-saved-probabilities/1",
              "comparison": "NEW (R66 ep128) vs BC (R60 ep0), same seed/mode/scene/step/target",
              "source": "raw/logs/v2_r66eval2_model_{new,bc}_s{seed}_{mode}_a1_s{scene}_a1_tape.jsonl"
                        " (+ the 6 focal tapes)",
              "blocks": out,
              "records": detail}
    dest = HERE / "r67_paired_probabilities.json"
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, ensure_ascii=False)
    for key, val in out.items():
        if isinstance(val, dict):
            print(key, "->", json.dumps(val, ensure_ascii=False))
    print("wrote", dest)


if __name__ == "__main__":
    main()
