"""ROUND61 case freeze: select the focus + 7 controls from the R60 seed33 greedy saved early
first-report candidates, using the fixed selection order and the legal-input distance over
columns 13/14/15/18 only. Read-only; no environment, no model forward, no outcome-based sorting."""
import json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
R60 = HERE.parent / "round60_partitioned_actor_head_ablation" / "raw"
cands = json.loads((HERE / "_candidates.json").read_text(encoding="utf-8"))
COLS = (13, 14, 15, 18)
FOCUS = (1700027, "piloc-e1-target-5")

ranges = {}
for c in COLS:
    vals = [cand["features"][c] for cand in cands]
    ranges[c] = max(vals) - min(vals)

def dist(a, b):
    d = 0.0
    for c in COLS:
        r = ranges[c]
        if r > 0:
            d += ((a["features"][c] - b["features"][c]) / r) ** 2
    return d

focus = next(c for c in cands if (c["scene"], c["target"]) == FOCUS)
sends = [c for c in cands if c["action"] == "SEND"]
skips = [c for c in cands if c["action"] == "SKIP"]

selected = [focus]
used_scenes = {focus["scene"]}
# rule 2: up to 3 SEND candidates from different scenes (0 available in this model/mode)
send_picks = []
for c in sorted(sends, key=lambda x: (dist(x, focus), x["scene"], x["target"])):
    if c["scene"] in used_scenes:
        continue
    send_picks.append(c); used_scenes.add(c["scene"])
    if len(send_picks) == 3:
        break
selected += send_picks
# rule 3: fill to 8 from SKIP candidates by distance, different scenes
skip_picks = []
for c in sorted(skips, key=lambda x: (dist(x, focus), x["scene"], x["target"])):
    if c["scene"] in used_scenes:
        continue
    skip_picks.append(c); used_scenes.add(c["scene"])
    if len(selected) + len(skip_picks) == 8:
        break
selected += skip_picks

if len(selected) != 8:
    raise SystemExit(f"only {len(selected)} qualifying cases across distinct scenes; stopping per rules")

def case_rec(c, role):
    return {"role": role, "scene": c["scene"], "target": c["target"], "ready_step": c["ready"],
            "baseline_action": c["action"], "decision_step": c["decision_step"],
            "features_19d": c["features"], "source_tape": c["tape"].replace("\\", "/"),
            "distance_to_focus": dist(c, focus),
            "cols_13_14_15_18": [c["features"][i] for i in COLS]}

cases = [case_rec(focus, "focus")] + [case_rec(c, "control") for c in send_picks + skip_picks]

model = {"path": "round60_partitioned_actor_head_ablation/raw/checkpoints/v2_train_s33_ep128.pt",
         "file_sha256": "573e4c26a4fee6f01405d4386ca73b518eeb1adb9cd8409bc83661e4113f6c5a",
         "parameter_sha256": "bce71a0e9810122c146950f12cf818a9a359406312d96dfc9c7a457ebb654e2a",
         "structure": "partitioned-early-late-actor-head/1", "mode": "greedy", "training_seed": 33,
         "training_version": "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/dual-eta-lambda-0p5/1/rule-bc-late147-init/1/published-completion-event-at-relay-tick/2+r53-critic-representation-stop-gradient/1+r56-soft-early-report-prior-beta0p10/1+r58-prior-strength-beta1p00/1+r60-partitioned-early-late-actor-head/1"}

freeze = {"kind": "round61-case-freeze/1",
          "selection_algorithm": {
            "candidate_filter": "R60 seed33 greedy saved eval tapes; real discovered target; baseline first-report free_one_shot_choice (free, legal, payable); first_emission; ready_step=round(f18*1000)-round(f0)<=147; distinct scene per case",
            "distance": "range-scaled squared difference summed over legal input columns 13,14,15,18 only; range computed over all qualifying early candidates; constant columns contribute 0",
            "column_ranges": {str(c): ranges[c] for c in COLS},
            "order": "focus mandatory; up to 3 SEND (different scenes) by distance; fill to 8 from SKIP by distance; ties by scene int then target id",
            "outcome_blind": True,
            "note": "0 SEND early first-report candidates exist in this model/mode (all 146 are SKIP), so rule 2 selects 0 and rule 3 fills 7 SKIP controls; no labels forced"},
          "candidate_counts": {"total_early": len(cands), "send": len(sends), "skip": len(skips),
                               "distinct_scenes": len({c['scene'] for c in cands})},
          "model": model,
          "world_config": {"max_steps": 1000, "goal_count": 6, "communication_budget_bytes": 29300,
                           "ack_enabled": False, "feature_layout": "comm-send-state/1", "greedy_tie": "SEND on tie"},
          "cases": cases,
          "focus_ready_step": focus["ready"]}
(HERE / "ROUND61_CASE_FREEZE.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
for c in cases:
    print(f"{c['role']:8s} scene={c['scene']} {c['target']} ready={c['ready_step']} base={c['baseline_action']} dist={c['distance_to_focus']:.4f} cols={c['cols_13_14_15_18']}")
print("froze", len(cases), "cases -> ROUND61_CASE_FREEZE.json")
