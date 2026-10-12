"""ROUND67 Task 1 - action / emission equivalence between R66 NEW and BC.

Read-only. Parses the already-persisted R66 eval rows and decision-step tapes. No environment,
no simulation, no model forward, no sampling, no checkpoint write.

For every eval cell (3 seeds x 2 modes x 32 scenes) plus the 6 focal cells it compares NEW vs BC
on the fields that decide whether "identical task count / cost" also means "identical behaviour":
  * outcome fields: on_time, spent_bytes, emissions, refusal_sequence, emission_sequence,
    discovered / expired / unassigned / completed / active, task_reward_events, behaviour_sha256
  * trajectory fields from the tape: decision-step count, per-step target list, per-step free
    first-report target list, the per-target action actually taken, and the saved per-target
    send probability.

Mismatches are reported, never force-matched. A cell whose tape is missing is reported as
missing_input, not as a match.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "raw"
ROWS = RAW / "rows"
LOGS = RAW / "logs"
SEEDS = (11, 22, 33)
MODES = ("greedy", "sampled")
SCENES = [1800000 + i for i in range(32)]
FOCAL_SCENE = 1700027

OUTCOME_FIELDS = ("on_time", "spent_bytes", "emissions", "discovered", "expired",
                  "unassigned", "completed", "active", "budget_used_fraction",
                  "normalised_cost_total", "task_reward_total", "steps", "seconds")


def load_json(path: Path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_tape(path: Path):
    if not path.is_file():
        return None
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def row_name(arm: str, seed: int, mode: str, scene: int) -> str:
    return f"v2_r66eval2_model_{arm}_s{seed}_{mode}_a1_s{scene}_a1.json"


def tape_name(arm: str, seed: int, mode: str, scene: int) -> str:
    return f"v2_r66eval2_model_{arm}_s{seed}_{mode}_a1_s{scene}_a1_tape.jsonl"


def focal_name(arm: str, seed: int) -> str:
    return f"v2_r66eval2_focal_{arm}_s{seed}_a1_s{FOCAL_SCENE}_a1.json"


def action_of(rec: dict) -> dict:
    """Per-target action actually taken at this decision step."""
    return {t: rec["actions"].get(t) for t in rec.get("targets", [])}


def free_targets(rec: dict) -> list:
    return sorted(rec.get("free_first_reports") or [])


def send_probs(rec: dict) -> dict:
    return {t: (rec.get("send_probabilities") or {}).get(t) for t in rec.get("targets", [])}


def compare_cell(arm_a: str, arm_b: str, seed: int, mode: str, scene: int,
                 row_a=None, row_b=None) -> dict:
    """Compare one NEW cell against one BC cell (or a focal row against its pair)."""
    tag = f"s{seed}/{mode}/s{scene}"
    pa = ROWS / (row_a or row_name(arm_a, seed, mode, scene))
    pb = ROWS / (row_b or row_name(arm_b, seed, mode, scene))
    ra, rb = load_json(pa), load_json(pb)
    res = {"cell": tag, "row_paths": [str(pa), str(pb)], "row_missing": []}
    if ra is None or rb is None:
        res["row_missing"] = [p for p, r in ((str(pa), ra), (str(pb), rb)) if r is None]
        return res

    diffs = {}
    for field in OUTCOME_FIELDS:
        va, vb = ra.get(field), rb.get(field)
        if isinstance(va, float) or isinstance(vb, float):
            same = (va is not None and vb is not None and abs(float(va) - float(vb)) < 1e-9)
        else:
            same = va == vb
        if not same:
            diffs[field] = [va, vb]
    for field in ("emission_sequence", "refusal_sequence", "task_reward_events"):
        va, vb = ra.get(field), rb.get(field)
        if va != vb:
            diffs[field] = [va, vb]
    if ra.get("behaviour_sha256") != rb.get("behaviour_sha256"):
        diffs["behaviour_sha256"] = [ra.get("behaviour_sha256"), rb.get("behaviour_sha256")]
    res["outcome_diffs"] = diffs

    ta_name = pa.name.replace(".json", "_tape.jsonl")
    tb_name = pb.name.replace(".json", "_tape.jsonl")
    ta, tb = load_tape(LOGS / ta_name), load_tape(LOGS / tb_name)
    if ta is None or tb is None:
        res["tape_missing"] = [p for p, t in ((str(LOGS / ta_name), ta), (str(LOGS / tb_name), tb)) if t is None]
        return res

    res["tape_records"] = [len(ta), len(tb)]
    traj = {"decision_step_count_differs": len(ta) != len(tb)}
    step_diffs = []
    for idx, (xa, xb) in enumerate(zip(ta, tb)):
        d = {}
        if xa.get("step") != xb.get("step"):
            d["step"] = [xa.get("step"), xb.get("step")]
        if list(xa.get("targets") or []) != list(xb.get("targets") or []):
            d["targets"] = [xa.get("targets"), xb.get("targets")]
        fa, fb = free_targets(xa), free_targets(xb)
        if fa != fb:
            d["free_first_reports"] = [fa, fb]
        if list(xa.get("free_one_shot_choices") or []) != list(xb.get("free_one_shot_choices") or []):
            d["free_one_shot_choices"] = [xa.get("free_one_shot_choices"), xb.get("free_one_shot_choices")]
        aa, ab = action_of(xa), action_of(xb)
        if aa != ab:
            d["actions"] = [aa, ab]
        pa_, pb_ = send_probs(xa), send_probs(xb)
        if set(pa_) != set(pb_):
            d["send_probability_keys"] = [sorted(pa_), sorted(pb_)]
        else:
            for t in pa_:
                if pa_[t] is None or pb_[t] is None:
                    if pa_[t] != pb_[t]:
                        d.setdefault("send_probability_none", []).append([t, pa_[t], pb_[t]])
                elif abs(float(pa_[t]) - float(pb_[t])) > 1e-12:
                    d.setdefault("send_probability_value", []).append(
                        [t, pa_[t], pb_[t], abs(float(pa_[t]) - float(pb_[t]))])
        if d:
            d["index"] = idx
            step_diffs.append(d)
    traj["step_diffs"] = step_diffs
    traj["free_first_report_target_sets_equal"] = [free_targets(r) for r in ta] == [free_targets(r) for r in tb]
    traj["action_maps_equal"] = [action_of(r) for r in ta] == [action_of(r) for r in tb]
    res["trajectory"] = traj
    return res


def main() -> None:
    cells = []
    for seed in SEEDS:
        for mode in MODES:
            for scene in SCENES:
                cells.append(compare_cell("new", "bc", seed, mode, scene))
    focal = []
    for seed in SEEDS:
        focal.append(compare_cell("focal_new", "focal_bc", seed, "greedy", FOCAL_SCENE,
                                  row_a=focal_name("new", seed), row_b=focal_name("bc", seed)))

    def summarise(items):
        out = {"cells": len(items), "row_missing": 0, "tape_missing": 0,
               "outcome_diff_cells": 0, "action_map_diff_cells": 0,
               "free_set_diff_cells": 0, "emission_sequence_diff_cells": 0,
               "behaviour_sha_diff_cells": 0, "step_diff_cells": 0}
        for it in items:
            if it.get("row_missing"):
                out["row_missing"] += 1
                continue
            if it.get("tape_missing"):
                out["tape_missing"] += 1
                continue
            if it.get("outcome_diffs"):
                out["outcome_diff_cells"] += 1
            if "emission_sequence" in (it.get("outcome_diffs") or {}):
                out["emission_sequence_diff_cells"] += 1
            if "behaviour_sha256" in (it.get("outcome_diffs") or {}):
                out["behaviour_sha_diff_cells"] += 1
            traj = it.get("trajectory") or {}
            if not traj.get("action_maps_equal", True):
                out["action_map_diff_cells"] += 1
            if not traj.get("free_first_report_target_sets_equal", True):
                out["free_set_diff_cells"] += 1
            if traj.get("step_diffs"):
                out["step_diff_cells"] += 1
        return out

    result = {
        "kind": "round67-task1-action-emission-equivalence/1",
        "scope": {"seeds": list(SEEDS), "modes": list(MODES), "scenes": SCENES,
                  "focal_scene": FOCAL_SCENE, "focal_mode": "greedy"},
        "comparison": "NEW (R66 ep128, frozen early path) vs BC (R60 ep0 model-only)",
        "eval_cells_summary": summarise(cells),
        "focal_cells_summary": summarise(focal),
        "eval_cells_with_outcome_diffs": [c["cell"] for c in cells if c.get("outcome_diffs")],
        "eval_cells_with_action_diffs": [c["cell"] for c in cells
                                         if not (c.get("trajectory") or {}).get("action_maps_equal", True)],
        "eval_cells_with_step_diffs": [c["cell"] for c in cells if (c.get("trajectory") or {}).get("step_diffs")],
        "focal_cells_with_outcome_diffs": [c["cell"] for c in focal if c.get("outcome_diffs")],
        "focal_cells_with_action_diffs": [c["cell"] for c in focal
                                          if not (c.get("trajectory") or {}).get("action_maps_equal", True)],
        "detail_eval": cells,
        "detail_focal": focal,
    }
    out = Path(__file__).resolve().parent / "r67_action_equivalence.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, ensure_ascii=False)
    print("eval summary:", json.dumps(result["eval_cells_summary"], ensure_ascii=False))
    print("focal summary:", json.dumps(result["focal_cells_summary"], ensure_ascii=False))
    print("outcome diff cells:", result["eval_cells_with_outcome_diffs"][:10],
          "focal:", result["focal_cells_with_outcome_diffs"][:10])
    print("action diff cells:", result["eval_cells_with_action_diffs"][:10],
          "focal:", result["focal_cells_with_action_diffs"][:10])
    print("wrote", out)


if __name__ == "__main__":
    main()
