"""ROUND67 Task 2/3 - joint probability panel for the R66 NEW vs BC controllers.

READ-ONLY mechanism analysis. No environment, no simulation, no sampling, no optimizer step, no
PPO replay, no checkpoint write, no RNG restore. Two CPU forward passes per distinct input row
under torch.no_grad(); the total number of input rows pushed through a forward is capped by
FORWARD_ROW_CAP and the plan is counted and checked BEFORE the first forward runs.

Why a panel at all: R66 froze trunk+early_head and trained late_head + both critics. The
parameters really moved (ROUND66_REVIEW_PARAM_DIFF.json: late_head L2 0.090/0.237/0.116; trunk and
early_head bit-identical), yet NEW and BC agree on task count, cost and target-ID sets in all 192
eval cells. This script measures what the moved head does to the routed probabilities on the
states the frozen policy really visited, using the pre-action 19-dim inputs saved in the persisted
decision-step tapes.

Panel construction (deterministic, outcome-independent):
  * distinct key = (seed, exact float32 bytes of the 19-dim input); the same input reached from any
    arm/mode/scene is forwarded once per model and is never filtered by outcome;
  * Panel A = COMPLETE census of every distinct FREE first-report input (the only rows where the
    actor probability can change the action); early and late both included;
  * Panel B = quota-limited sample of distinct LATE-routed non-free (forced-hold) inputs;
  * Panel C = quota-limited sample of distinct EARLY-routed non-free inputs (control: the early
    path is bit-identical between the arms, so this must come out exactly equal).
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
R66_DIR = HERE.parent
sys.path.insert(0, str(R66_DIR))
import r66_frozen as r66  # noqa: E402

RAW = R66_DIR / "raw"
LOGS = RAW / "logs"
SEEDS = (11, 22, 33)
MODES = ("greedy", "sampled")
SCENES = [1800000 + i for i in range(32)]
FOCAL_SCENE = 1700027
FOCAL_TARGET = "piloc-e1-target-5"
HORIZON, READY_CUT = 1000, 147
SKIP_INDEX, SEND_INDEX = 0, 1
FORWARD_ROW_CAP = 5000
LATE_NONFREE_QUOTA_PER_SEED = 620
EARLY_NONFREE_QUOTA_PER_SEED = 40
TOL = 1e-6


def bentropy(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def f32_key(vec):
    return struct.pack("<" + "f" * len(vec), *[float(x) for x in vec])


def ready_step(vec):
    return round(float(vec[18]) * HORIZON) - round(float(vec[0]))


def eval_tape(arm, seed, mode, scene):
    return LOGS / f"v2_r66eval2_model_{arm}_s{seed}_{mode}_a1_s{scene}_a1_tape.jsonl"


def focal_tape(arm, seed):
    return LOGS / f"v2_r66eval2_focal_{arm}_s{seed}_a1_s{FOCAL_SCENE}_a1_tape.jsonl"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RowGuard:
    """Counts every input row pushed through any forward; refuses to exceed the cap."""

    def __init__(self, cap):
        self.cap = cap
        self.used = 0
        self.calls = 0

    def charge(self, rows, what):
        if self.used + rows > self.cap:
            raise RuntimeError(f"forward row guard: {what} would take the total to "
                               f"{self.used + rows} > cap {self.cap}")
        self.used += rows
        self.calls += 1


def load_controllers():
    """Read-only load of the R66 ep128 (NEW) and R60 ep0 (BC) controllers, identity verified."""
    mods = r66.configure_source_imports()
    index = json.loads((RAW / "runtime" / "ROUND66_CHECKPOINT_INDEX.json").read_text(encoding="utf-8"))
    out = {}
    for seed in SEEDS:
        entry = index["checkpoints"][f"s{seed}"]
        path = (R66_DIR / entry["path"]).resolve()
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise RuntimeError(f"{path.name}: file sha256 drifted from the checkpoint index")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if set(payload["controller"]) != {"trunk", "early_head", "late_head",
                                          "value_task_head", "value_cost_head"}:
            raise RuntimeError(f"{path.name}: not the five-module partitioned schema")
        ts = payload["training_state"]
        controller = r66.fresh_partitioned_controller(seed, ep0_state=None, verify_ep0=False)
        controller.load_state_dict(payload["controller"])
        out[seed] = {
            "NEW": {"controller": controller, "path": str(path), "file_sha256": digest,
                    "episodes_done": ts.get("episodes_done"), "updates": ts.get("updates"),
                    "structure": payload.get("structure"),
                    "training_version": payload.get("training_version"),
                    "parameter_identity": r66.frozen_parameter_summary(controller)},
        }
        bc_controller, bc_identity = r66.load_r60_checkpoint_read_only(seed, "ep0")
        out[seed]["BC"] = {"controller": bc_controller, "path": bc_identity["source_path"],
                           "file_sha256": bc_identity["source_file_sha256"],
                           "parameter_identity": bc_identity["parameter_identity"],
                           "episodes_done": bc_identity["episodes_done"],
                           "updates": bc_identity["updates"],
                           "structure": "partitioned-early-late-actor-head/1",
                           "training_version": bc_identity["training_version"]}
    out["_canonical_modules"] = mods.get("canonical_modules")
    return out


def collect_records():
    """Every tape row that carries a model probability, plus the panel keys it contributes."""
    records = []
    distinct = {}
    for seed in SEEDS:
        bucket = {}
        sources = []
        for mode in MODES:
            for scene in SCENES:
                for arm in ("new", "bc"):
                    sources.append((arm, mode, scene, eval_tape(arm, seed, mode, scene)))
        for arm in ("new", "bc"):
            sources.append((arm, "greedy", FOCAL_SCENE, focal_tape(arm, seed)))
        for arm, mode, scene, path in sources:
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    feats = rec.get("features") or []
                    tgts = rec.get("targets") or []
                    free = set(rec.get("free") or [])
                    ffr = set(rec.get("free_first_reports") or [])
                    probs = rec.get("send_probabilities") or {}
                    for i, vec in enumerate(feats):
                        if len(vec) != 19:
                            raise RuntimeError(f"{path.name}: feature row is not 19-dim")
                        tgt = tgts[i] if i < len(tgts) else None
                        key = f32_key(vec)
                        rs = ready_step(vec)
                        entry = bucket.setdefault(key, {"vec": [float(x) for x in vec],
                                                        "route": "late" if rs > READY_CUT else "early",
                                                        "ready_step": rs, "free": False,
                                                        "free_first_report": False})
                        if tgt in free:
                            entry["free"] = True
                            if tgt in ffr:
                                entry["free_first_report"] = True
                        if tgt in probs:
                            records.append({"arm": arm, "seed": seed, "mode": mode,
                                            "scene": scene, "step": rec.get("step"),
                                            "target": tgt, "key": key,
                                            "route": entry["route"], "free": tgt in free,
                                            "saved_p": probs.get(tgt),
                                            "saved_action": (rec.get("actions") or {}).get(tgt),
                                            "saved_log_prob": rec.get("log_prob")})
        distinct[seed] = bucket
    return records, distinct


def plan_panels(distinct):
    plan = {"A_free_census": [], "B_late_nonfree": [], "C_early_nonfree": []}
    for seed in SEEDS:
        bucket = distinct[seed]
        free_keys = sorted(k for k, v in bucket.items() if v["free"])
        late_nf = sorted(k for k, v in bucket.items() if v["route"] == "late" and not v["free"])
        early_nf = sorted(k for k, v in bucket.items() if v["route"] == "early" and not v["free"])
        plan["A_free_census"].extend((seed, k) for k in free_keys)
        plan["B_late_nonfree"].extend((seed, k) for k in late_nf[:LATE_NONFREE_QUOTA_PER_SEED])
        plan["C_early_nonfree"].extend((seed, k) for k in early_nf[:EARLY_NONFREE_QUOTA_PER_SEED])
    return plan


def stats_block(sub):
    if not sub:
        return None
    pn = [x["NEW"]["p_send"] for x in sub]
    pb = [x["BC"]["p_send"] for x in sub]
    dp = [a - b for a, b in zip(pn, pb)]
    dl = [(x["NEW"]["logit_send"] - x["NEW"]["logit_skip"])
          - (x["BC"]["logit_send"] - x["BC"]["logit_skip"]) for x in sub]
    n = len(sub)
    sat = lambda p: sum(1 for v in p if v >= 0.99 or v <= 0.01)
    return {
        "n": n,
        "p_send_NEW": {"min": min(pn), "max": max(pn), "mean": sum(pn) / n},
        "p_send_BC": {"min": min(pb), "max": max(pb), "mean": sum(pb) / n},
        "abs_dp": {"max": max(abs(v) for v in dp), "mean": sum(abs(v) for v in dp) / n,
                   "n_nonzero": sum(1 for v in dp if v > 0.0)},
        "abs_dlogit_margin": {"max": max(abs(v) for v in dl),
                              "mean": sum(abs(v) for v in dl) / n},
        "entropy_NEW_bits_mean": sum(bentropy(v) for v in pn) / n,
        "entropy_BC_bits_mean": sum(bentropy(v) for v in pb) / n,
        "distance_to_half_NEW": {"min": min(abs(v - 0.5) for v in pn),
                                 "max": max(abs(v - 0.5) for v in pn)},
        "distance_to_half_BC": {"min": min(abs(v - 0.5) for v in pb),
                                "max": max(abs(v - 0.5) for v in pb)},
        "saturated_0p99_NEW": sat(pn), "saturated_0p99_BC": sat(pb),
        "argmax_flip_vs_BC": sum(1 for x in sub if (x["NEW"]["p_send"] > 0.5) != (x["BC"]["p_send"] > 0.5)),
    }


def main():
    print("loading controllers (read-only) ...")
    controllers = load_controllers()
    print("collecting tape records ...")
    records, distinct = collect_records()
    plan = plan_panels(distinct)
    planned_rows = 2 * sum(len(v) for v in plan.values())
    print("distinct inputs per seed: " + ", ".join(f"s{s}={len(distinct[s])}" for s in SEEDS))
    for name, items in plan.items():
        print(f"  panel {name}: {len(items)} distinct inputs -> {2 * len(items)} forward rows")
    print(f"planned forward rows = {planned_rows} (cap {FORWARD_ROW_CAP})")
    if planned_rows > FORWARD_ROW_CAP:
        raise RuntimeError("panel plan exceeds the forward row cap; refusing to run")

    guard = RowGuard(FORWARD_ROW_CAP)
    results = {}
    for name, items in plan.items():
        by_seed = {}
        for seed, key in items:
            by_seed.setdefault(seed, []).append(key)
        for seed, keys in sorted(by_seed.items()):
            mat = torch.tensor([distinct[seed][k]["vec"] for k in keys], dtype=torch.float32)
            guard.charge(2 * mat.shape[0], f"panel {name} seed {seed}")
            with torch.no_grad():
                for arm in ("NEW", "BC"):
                    logits, _ = controllers[seed][arm]["controller"].forward(mat)
                    probs = torch.softmax(logits, dim=-1)
                    for j, key in enumerate(keys):
                        results.setdefault((seed, key), {})[arm] = {
                            "logit_skip": float(logits[j, SKIP_INDEX]),
                            "logit_send": float(logits[j, SEND_INDEX]),
                            "p_send": float(probs[j, SEND_INDEX]),
                            "panel": name,
                        }
    print(f"forward rows used: {guard.used} in {guard.calls} charged batches")

    check = {"records": 0, "matched": 0, "mismatched": 0, "missing": 0,
             "max_abs_diff": 0.0, "examples": []}
    for rec in records:
        got = results.get((rec["seed"], rec["key"]), {}).get(rec["arm"].upper())
        if got is None or rec["saved_p"] is None:
            check["missing"] += 1
            continue
        check["records"] += 1
        diff = abs(got["p_send"] - float(rec["saved_p"]))
        check["max_abs_diff"] = max(check["max_abs_diff"], diff)
        if diff <= TOL:
            check["matched"] += 1
        else:
            check["mismatched"] += 1
            if len(check["examples"]) < 10:
                check["examples"].append({"seed": rec["seed"], "mode": rec["mode"],
                                          "scene": rec["scene"], "step": rec["step"],
                                          "target": rec["target"], "route": rec["route"],
                                          "recomputed": got["p_send"], "saved": rec["saved_p"],
                                          "abs_diff": diff})
    print("recompute-vs-saved:", {k: v for k, v in check.items() if k != "examples"})

    mode_of, scene_of = {}, {}
    for rec in records:
        mode_of.setdefault((rec["seed"], rec["key"]), set()).add(rec["mode"])
        scene_of.setdefault((rec["seed"], rec["key"]), set()).add(rec["scene"])
    blocks = {}
    for (seed, key), arms in results.items():
        if "NEW" not in arms or "BC" not in arms:
            continue
        info = distinct[seed][key]
        for mode in sorted(mode_of.get((seed, key), [])):
            tag = f"s{seed}/{mode}/{info['route']}/{'free' if info['free'] else 'nonfree'}"
            blocks.setdefault(tag, []).append(arms)
    panel_stats = {tag: stats_block(sub) for tag, sub in sorted(blocks.items())}

    route_summary = {}
    for route in ("early", "late"):
        sub = [arms for (seed, key), arms in results.items()
               if "NEW" in arms and "BC" in arms and distinct[seed][key]["route"] == route]
        route_summary[route] = stats_block(sub)
    early_bit_identical = all(
        arms["NEW"]["p_send"] == arms["BC"]["p_send"] and arms["NEW"]["logit_send"] == arms["BC"]["logit_send"]
        for (seed, key), arms in results.items()
        if "NEW" in arms and "BC" in arms and distinct[seed][key]["route"] == "early")

    focal = []
    for seed in SEEDS:
        with focal_tape("new", seed).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for i, vec in enumerate(rec.get("features") or []):
                    tgt = (rec.get("targets") or [])[i]
                    if tgt != FOCAL_TARGET:
                        continue
                    key = f32_key(vec)
                    arms = results.get((seed, key))
                    if arms is None:
                        continue
                    rs = ready_step(vec)
                    focal.append({
                        "seed": seed, "step": rec.get("step"), "ready_step": rs,
                        "route": "late" if rs > READY_CUT else "early",
                        "free": tgt in set(rec.get("free") or []),
                        "p_NEW": arms["NEW"]["p_send"], "p_BC": arms["BC"]["p_send"],
                        "abs_dp": abs(arms["NEW"]["p_send"] - arms["BC"]["p_send"]),
                        "action": (rec.get("actions") or {}).get(tgt),
                        "panel": arms["NEW"]["panel"],
                    })

    canonical = controllers.pop("_canonical_modules")
    out = {
        "kind": "round67-task23-joint-probability-panel/1",
        "comparison": "NEW (R66 ep128 frozen early path) vs BC (R60 ep0 model-only)",
        "canonical_modules": canonical,
        "forward_row_cap": FORWARD_ROW_CAP,
        "forward_rows_used": guard.used,
        "forward_batches_charged": guard.calls,
        "panel_sizes": {name: {"distinct_inputs": len(items), "forward_rows": 2 * len(items)}
                        for name, items in plan.items()},
        "quotas": {"late_nonfree_per_seed": LATE_NONFREE_QUOTA_PER_SEED,
                   "early_nonfree_per_seed": EARLY_NONFREE_QUOTA_PER_SEED,
                   "free_census": "complete (no quota)"},
        "recompute_vs_saved": check,
        "route_summary": route_summary,
        "early_path_bit_identical_on_panel": early_bit_identical,
        "panel_stats_by_seed_mode_route_free": panel_stats,
        "focal_1700027_target5": focal,
        "controllers": {f"s{seed}": {arm: {k: v for k, v in info.items()
                                           if k not in ("controller", "parameter_identity")}
                                     for arm, info in arms.items()}
                        for seed, arms in controllers.items()},
        "note": ("p(SEND)=softmax(logits)[...,1]; SKIP_INDEX=0, SEND_INDEX=1; the greedy tie goes to "
                 "SEND, so the decision boundary is p=0.5. Entropy is the binary entropy of p(SEND) "
                 "in bits. Saturation is the descriptive p>=0.99 or p<=0.01 rule; no threshold was "
                 "fitted. Panels B and C are quota-limited by construction and labelled as such."),
    }
    dest = HERE / "r67_probability_panel.json"
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, ensure_ascii=False)
    print("route summary:", json.dumps(route_summary, ensure_ascii=False, indent=1))
    print("early bit identical on panel:", early_bit_identical)
    print("focal target-5 rows:", len(focal))
    print("wrote", dest)


if __name__ == "__main__":
    main()
