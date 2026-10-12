"""ROUND65 read-only analyzer: training coverage (T0/T1/T2) + learning-signal audit.

Zero-environment. Reads ONLY the compliant R60 artifacts: the 96 attempt01 batch records
(source index), the 384 training row files and their tapes; and the R64 L baseline tape for the
authoritative focal encoding. It never constructs an environment, touches a provider/claim/slot,
runs an optimizer/PPO update, restores training RNG/optimizer/lambda/cursor, or writes a
checkpoint. The only computed quantities beyond persisted fields are clearly-labelled indicative
notes; per-sample A_task/A_cost are marked not_persisted and never backfilled.
"""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r65_common as C

RUN = HERE.parent
R60 = RUN / "round60_partitioned_actor_head_ablation"
R64 = RUN / "round64_controlled_deadline_intervention"
RAW = R60 / "raw"
CK = RAW / "checkpoints"
ROWS = RAW / "rows"
LOGS = RAW / "logs"
SEEDS = [11, 22, 33]
NBATCH = 32
FOCAL_TAPE = R64 / "raw" / "logs" / "v2_L_a03_s1700027_a3_tape.jsonl"
FOCAL_TARGET = "piloc-e1-target-5"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_focal():
    """AUTHORITATIVE focal 19-D: parsed from the R64 L tape, cross-checked on source path, file
    SHA256, tick and target index, then validated against the hand-entered constant. Abort on any
    mismatch so the constant can never silently become the science anchor."""
    if not FOCAL_TAPE.exists():
        raise SystemExit(f"focal source tape missing: {FOCAL_TAPE}")
    file_sha = sha256_file(FOCAL_TAPE)
    found = None
    with open(FOCAL_TAPE, encoding="utf-8") as fh:
        for ln in fh:
            if '"free_first_reports": []' in ln or '"free_first_reports":[]' in ln:
                continue
            rec = json.loads(ln)
            ffr = rec.get("free_first_reports") or []
            if FOCAL_TARGET in ffr:
                idx = rec["targets"].index(FOCAL_TARGET)
                found = {"features": rec["features"][idx], "step": int(rec["step"]),
                         "target_index": idx, "targets": rec["targets"],
                         "action": rec["actions"].get(FOCAL_TARGET),
                         "p_send": rec["send_probabilities"].get(FOCAL_TARGET),
                         "quote_bytes": rec["attempt_charge_bytes"].get(FOCAL_TARGET)}
                break
    if found is None:
        raise SystemExit("focal target-5 free first report not found in R64 L tape")
    focal32 = C.f32(found["features"])
    focal_bytes = focal32.tobytes()
    const_bytes = bytes.fromhex(C.FOCAL_F32_HEX)
    if focal_bytes != const_bytes:
        raise SystemExit("FATAL: parsed R64 focal float32 != r65_common constant; refusing to "
                         "match by constant. parsed=" + focal_bytes.hex())
    prov = {
        "source_path": str(FOCAL_TAPE), "source_file_sha256": file_sha,
        "tick": found["step"], "target": FOCAL_TARGET, "target_index": found["target_index"],
        "recorded_action": found["action"], "recorded_p_send": found["p_send"],
        "recorded_quote_bytes": found["quote_bytes"],
        "float32_hex": focal_bytes.hex(),
        "constant_matches_parsed": True,
        "note": "parsed value is authoritative; constant validated equal, used only as test expectation",
    }
    return focal32, focal_bytes, prov


def build_index():
    index, batches, problems = {}, {}, []
    per_seed = defaultdict(set)
    for seed in SEEDS:
        for b in range(NBATCH):
            rec = json.load(open(CK / f"r40_batch_s{seed}_b{b}_attempt01.json", encoding="utf-8"))
            batches[(seed, b)] = rec
            eps, scenes = rec["episodes"], rec["scenarios"]
            if len(eps) != 4 or len(scenes) != 4:
                problems.append(f"seed{seed} b{b}: len(episodes/scenes)!=4")
            for ep, sc in zip(eps, scenes):
                key = (seed, int(ep))
                if key in index:
                    problems.append(f"duplicate episode {key}")
                index[key] = {"scene": int(sc), "batch": b}
                per_seed[seed].add(int(ep))
    for seed in SEEDS:
        if per_seed[seed] != set(range(128)):
            problems.append(f"seed{seed}: coverage != 0..127 (n={len(per_seed[seed])})")
    return index, batches, problems


def scan_tapes(index, focal32, focal_bytes):
    cands, mism, row_cache = [], [], {}
    for (seed, ep), meta in sorted(index.items()):
        scene = meta["scene"]
        tape = LOGS / f"v2_r60_train_s{seed}_e{ep}_s{scene}_a1_tape.jsonl"
        rowp = ROWS / f"v2_r60_train_s{seed}_e{ep}_s{scene}_a1.json"
        if not tape.exists():
            mism.append(f"missing tape {tape.name}"); continue
        row = row_cache.get(rowp)
        if row is None:
            row = json.load(open(rowp, encoding="utf-8")); row_cache[rowp] = row
        ep_send = ep_skip = 0
        for ln in open(tape, encoding="utf-8"):
            if '"free_first_reports": []' in ln or '"free_first_reports":[]' in ln:
                continue
            rec = json.loads(ln)
            ffr = rec.get("free_first_reports") or []
            if not ffr:
                continue
            targets = rec["targets"]; step = int(rec["step"])
            val = rec.get("value"); cval = rec.get("cost_value"); jlp = rec.get("log_prob")
            for tgt in ffr:
                if tgt not in targets:
                    mism.append(f"s{seed}e{ep}: {tgt} ffr not in targets"); continue
                i = targets.index(tgt)
                feats = rec["features"][i]
                act = rec["actions"].get(tgt)
                tier = C.classify_free_first_report(feats, focal32, focal_bytes)
                if act == 1: ep_send += 1
                elif act == 0: ep_skip += 1
                cands.append({
                    "seed": seed, "episode": ep, "scene": scene, "batch": meta["batch"],
                    "step": step, "target": tgt, "tier": tier,
                    "ready_step": C.ready_step(feats), "action": act,
                    "backlog": float(C.f32(feats)[C.IDX_BACKLOG]),
                    "has_future": float(C.f32(feats)[C.IDX_HAS_FUTURE]),
                    "budget": float(C.f32(feats)[C.IDX_BUDGET]),
                    "p_send": rec["send_probabilities"].get(tgt),
                    "quote_bytes": rec["attempt_charge_bytes"].get(tgt),
                    "row_value_task": val, "row_value_cost": cval, "row_log_prob": jlp,
                    "differing_dims": (C.differing_dims(feats, focal32)
                                       if tier in ("T0_exact", "T1_partial") else None),
                })
        cs = row.get("collector_summary", {}) or {}
        elig = cs.get("eligibility_counts", {}) or {}
        osc = cs.get("one_shot_counts", {}) or {}
        ref_send = elig.get("requested_first_report_sends", osc.get("send_now_choices"))
        ref_skip = osc.get("skip_episode_choices")
        if ref_send is not None and int(ref_send) != ep_send:
            mism.append(f"s{seed}e{ep}: send {ep_send}!=summary {ref_send}")
        if ref_skip is not None and int(ref_skip) != ep_skip:
            mism.append(f"s{seed}e{ep}: skip {ep_skip}!=summary {ref_skip}")
        row_cache[rowp] = {"on_time": row.get("on_time"),
                           "task_reward_total": row.get("task_reward_total"),
                           "spent_bytes": row.get("spent_bytes")}
    return cands, mism, row_cache


def tier_summary(cands):
    def leaf():
        return {"candidates": 0, "episodes": set(), "scenes": set(), "send": 0, "skip": 0}
    names = ("T0_exact", "T1_partial", "T2_early_other", "late_or_other")
    tiers = {k: {**leaf(), "by_seed": {s: leaf() for s in SEEDS}} for k in names}
    for c in cands:
        t = tiers[c["tier"]]
        for d in (t, t["by_seed"][c["seed"]]):
            d["candidates"] += 1; d["episodes"].add(c["episode"]); d["scenes"].add(c["scene"])
            if c["action"] == 1: d["send"] += 1
            elif c["action"] == 0: d["skip"] += 1
    def fin(v):
        return {"candidates": v["candidates"], "episodes": len(v["episodes"]),
                "scenes": len(v["scenes"]), "send": v["send"], "skip": v["skip"]}
    def fingroup(v):
        return {**fin(v), "by_seed": {str(s): fin(v["by_seed"][s]) for s in SEEDS}}
    return {k: fingroup(v) for k, v in tiers.items()}

def signal_summary(batches):
    rows, gz, gp = [], 0, 0
    te = tl = ts = to = 0
    lam = defaultdict(list)
    for (seed, b), rec in sorted(batches.items()):
        u, con, du = rec["update"], rec["constraint"], rec["dual"]
        g = float(con.get("g_hat", 0.0))
        gz += (g == 0.0); gp += (g > 0)
        ef, lf, sm, op = int(u.get("early_free",0)), int(u.get("late_free",0)), int(u.get("samples",0)), int(u.get("optimizer_steps",0))
        te += ef; tl += lf; ts += sm; to += op
        lam[seed].append({"batch": b, "lambda_before": du.get("lambda_before"),
                          "lambda_next": du.get("lambda_next"), "g_hat": g, "direction": du.get("direction")})
        pairs = con.get("pairs", [])
        rows.append({"seed": seed, "batch": b, "lambda_at_rollout": rec.get("lambda_at_rollout"),
            "samples": sm, "early_free": ef, "late_free": lf, "minibatches": u.get("minibatches"),
            "optimizer_steps": op, "epochs": u.get("epochs"), "policy_surrogate": u.get("policy_surrogate"),
            "value_loss_task": u.get("value_loss_task"), "value_loss_cost": u.get("value_loss_cost"),
            "entropy": u.get("entropy"), "approx_kl": u.get("approx_kl"), "clip_fraction": u.get("clip_fraction"),
            "gradient_norm_mean": u.get("gradient_norm_mean"), "module_gradient_l1_mean": u.get("module_gradient_l1_mean"),
            "g_hat": g, "delta": con.get("delta"), "y_reference": [p.get("y_reference") for p in pairs],
            "y_policy": [p.get("y_policy") for p in pairs], "lambda_before": du.get("lambda_before"),
            "lambda_next": du.get("lambda_next"), "episodes_done_after": rec.get("episodes_done_after"),
            "optimizer_steps_after": rec.get("optimizer_steps_after"),
            "parameters_before_sha": rec.get("parameters_before_sha"), "parameters_after_sha": rec.get("parameters_after_sha")})
    agg = {"n_batches": len(rows), "g_hat_zero_batches": gz, "g_hat_positive_batches": gp,
           "total_early_free": te, "total_late_free": tl, "total_samples_all_rows": ts,
           "total_optimizer_steps": to, "lambda_history": {str(s): lam[s] for s in SEEDS},
           "lambda_final": {str(s): lam[s][-1]["lambda_next"] for s in SEEDS}}
    return rows, agg


def traceback(cands, batches):
    out = []
    def pref(seed, batch, scene):
        for p in batches[(seed, batch)]["constraint"].get("pairs", []):
            if int(p.get("scenario")) == int(scene):
                return p.get("y_reference"), p.get("y_policy")
        return None, None
    for c in cands:
        if c["tier"] not in ("T0_exact", "T1_partial"):
            continue
        rec = batches[(c["seed"], c["batch"])]
        yref, ypol = pref(c["seed"], c["batch"], c["scene"])
        out.append({"seed": c["seed"], "episode": c["episode"], "scene": c["scene"], "batch": c["batch"],
            "step": c["step"], "target": c["target"], "tier": c["tier"], "ready_step": c["ready_step"],
            "action": c["action"], "p_send": c["p_send"], "quote_bytes": c["quote_bytes"],
            "old_value_task": c["row_value_task"], "old_value_cost": c["row_value_cost"],
            "row_log_prob": c["row_log_prob"], "batch_g_hat": float(rec["constraint"].get("g_hat", 0.0)),
            "batch_lambda_before": rec["dual"].get("lambda_before"), "paired_y_reference": yref,
            "paired_y_policy": ypol, "differing_dims": c["differing_dims"],
            "A_task_A_cost": "not_persisted", "combined_advantage": "not_persisted"})
    return out



def distributions(cands):
    from collections import Counter
    early=[c for c in cands if c["tier"] in ("T0_exact","T1_partial","T2_early_other")]
    late=[c for c in cands if c["tier"]=="late_or_other"]
    def dist(group):
        return {"n":len(group),
            "ready_step_hist":dict(sorted(Counter(c["ready_step"] for c in group).items())),
            "backlog_hist":dict(sorted(Counter(c["backlog"] for c in group).items())),
            "has_future_hist":dict(sorted(Counter(c["has_future"] for c in group).items())),
            "budget_min":min((c["budget"] for c in group),default=None),
            "budget_max":max((c["budget"] for c in group),default=None),
            "send":sum(1 for c in group if c["action"]==1),
            "skip":sum(1 for c in group if c["action"]==0)}
    return {"early":dist(early),"late":dist(late)}


def near_miss(cands, focal32):
    """Diagnostic ONLY (NOT T1). Among early candidates, how many share the focal non-time
    flags, and the nearest ready_steps to the focal. No tolerance is loosened for T0/T1."""
    from collections import Counter
    fb=float(focal32[C.IDX_BACKLOG]); fh=float(focal32[C.IDX_HAS_FUTURE]); frs=C.ready_step(focal32)
    early=[c for c in cands if c["tier"] in ("T0_exact","T1_partial","T2_early_other")]
    both=[c for c in early if c["backlog"]==fb and c["has_future"]==fh]
    bl1=[c for c in early if c["backlog"]==fb]
    hf1=[c for c in early if c["has_future"]==fh]
    rs=Counter(c["ready_step"] for c in early)
    nearest=sorted(rs.items(), key=lambda kv:(abs(kv[0]-frs), kv[0]))[:6]
    return {"focal_ready_step":frs,"focal_backlog":fb,"focal_has_future":fh,
        "early_total":len(early),"early_backlog_eq_focal":len(bl1),
        "early_has_future_eq_focal":len(hf1),"early_backlog_and_future_eq_focal":len(both),
        "early_ready_step_eq_focal":rs.get(frs,0),
        "nearest_ready_steps":[{"ready_step":k,"count":v} for k,v in nearest],
        "note":"diagnostic near-miss; NOT the T1 tier; no tolerance loosened"}


def ghat_positive(batches):
    out=[]
    for (seed,b),rec in sorted(batches.items()):
        g=float(rec["constraint"].get("g_hat",0.0))
        if g>0:
            du=rec["dual"]; con=rec["constraint"]
            out.append({"seed":seed,"batch":b,"g_hat":g,"lambda_before":du.get("lambda_before"),
                "lambda_next":du.get("lambda_next"),"direction":du.get("direction"),
                "pairs":con.get("pairs")})
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args(); out = Path(args.out)
    focal32, focal_bytes, prov = parse_focal()
    index, batches, problems = build_index()
    cands, mism, _ = scan_tapes(index, focal32, focal_bytes)
    tiers = tier_summary(cands)
    sig_rows, sig_agg = signal_summary(batches)
    tb = traceback(cands, batches)

    coverage = {"kind": "round65-training-coverage/1",
        "focal": {"scene": 1700027, "target": FOCAL_TARGET, "ready_step": C.ready_step(focal32),
                  "float32_hex": focal_bytes.hex(), "features": [float(x) for x in focal32],
                  "provenance": prov},
        "compliant_index": {"n_batches": len(batches), "n_episodes": len(index),
            "per_seed_episodes": {str(s): sum(1 for (sd,_) in index if sd==s) for s in SEEDS},
            "problems": problems},
"total_free_first_report_candidates": len(cands), "tiers": tiers,
        "distributions": distributions(cands), "diagnostic_near_miss": near_miss(cands, focal32),
        "action_encoding_mismatch_count": len(mism), "action_encoding_mismatches": mism[:50],
        "notes": ["Repeated scenes are not independent samples; identical 19-D float32 != identical world.",
            "T0 = exact 19-D float32 byte identity AND free first report.",
            "T1 = same early route + ready_step==134 + queue_backlog==1 + has_future_requirements==1 (PARTIAL, never same-state).",
            "T2 = all free early first reports (ready_step<=147); distributional context only."]}
    json.dump(coverage, open(out/"TRAINING_COVERAGE.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)

    signal = {"kind": "round65-learning-signal-audit/1", "aggregate": sig_agg, "g_hat_positive_batches": ghat_positive(batches), "batch_rows": sig_rows,
        "t0_t1_traceback": tb,
        "interpretation_guardrails": [
            "Multiple free candidates on one decision step share ONE decision-row advantage; not independent feedback.",
            "A joint decision-row advantage is not a single-target causal contribution.",
            "Equal episode reward sums do not prove event attribution; use the same-tick contract.",
            "Combined-advantage sign only indicates the initial surrogate local tendency; without retained minibatch/ratio/clip/entropy/prior/net-update data the total gradient direction cannot be inferred.",
            "g_hat == 0 means the task-preservation constraint held that batch; it does NOT mean per-message task value is 0.",
            "Masked/fixed rows stay in the ALL_ROWS denominator (samples) and are counted separately from free actions.",
            "Per-sample A_task/A_cost and the combined advantage are NOT persisted by R60 (marked not_persisted); not backfilled; no autograd/gradient replay."]}
    json.dump(signal, open(out/"LEARNING_SIGNAL_AUDIT.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)

    print("focal provenance:", json.dumps({k:prov[k] for k in ('source_file_sha256','tick','target_index','recorded_p_send','constant_matches_parsed')}, ensure_ascii=False))
    print("INDEX problems:", problems)
    print("action mismatches:", len(mism), mism[:8])
    print("total free-first-report candidates:", len(cands))
    print("tiers:", json.dumps(tiers, ensure_ascii=False))
    print("signal agg:", {k:v for k,v in sig_agg.items() if k!='lambda_history'})
    print("lambda_final:", sig_agg['lambda_final'])
    print("T0/T1 traceback records:", len(tb))


if __name__ == "__main__":
    main()
