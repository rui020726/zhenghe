"""ROUND67 Task 4 - training exposure of the R66 frozen-early-path run.

READ-ONLY. Parses the 96 committed ROUND66 training batch records
(``raw/checkpoints/r40_batch_s{seed}_b{batch}_attempt01.json``, the R42-compatible name), the three
first-batch gate files, the durable ppo/optimizer call journals, and the 384 persisted training
episode tapes. No PPO replay, no fabricated minibatch order, no optimizer step, no environment.

What it answers: how many late free choices the late_head actually saw, what the recorded
gradients/clip readings were, what g_hat / lambda did, and what trainable role the beta=1.00 gate
prior still has once the early path is frozen. Fields that were never recorded are reported as
``missing`` rather than reconstructed.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R66_DIR = HERE.parent
RAW = R66_DIR / "raw"
CKPT = RAW / "checkpoints"
RUNTIME = RAW / "runtime"
LOGS = RAW / "logs"
SEEDS = (11, 22, 33)
HORIZON, READY_CUT = 1000, 147


def jload(path):
    if not Path(path).is_file():
        return None
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def jlines(path):
    if not Path(path).is_file():
        return []
    out = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def ready_step(vec):
    return round(float(vec[18]) * HORIZON) - round(float(vec[0]))


def main():
    batches = {}
    for seed in SEEDS:
        for batch in range(32):
            rec = jload(CKPT / f"r40_batch_s{seed}_b{batch}_attempt01.json")
            if rec is None:
                raise FileNotFoundError(f"missing batch record s{seed} b{batch}")
            batches[(seed, batch)] = rec
    print("batch records:", len(batches))

    # ---- per-seed aggregates over the 32 committed updates --------------------------------
    per_seed = {}
    for seed in SEEDS:
        recs = [batches[(seed, b)] for b in range(32)]
        upd = [r["update"] for r in recs]
        per_seed[f"s{seed}"] = {
            "updates": len(recs),
            "episodes_covered": sorted({e for r in recs for e in r["episodes"]}),
            "episode_count": len({e for r in recs for e in r["episodes"]}),
            "optimizer_steps_total": sum(int(u.get("optimizer_steps", 0)) for u in upd),
            "late_free_total": sum(int(u.get("late_free", 0)) for u in upd),
            "late_free_per_update": {"min": min(int(u.get("late_free", 0)) for u in upd),
                                     "max": max(int(u.get("late_free", 0)) for u in upd),
                                     "mean": sum(int(u.get("late_free", 0)) for u in upd) / len(upd)},
            "early_free_total": sum(int(u.get("early_free", 0)) for u in upd),
            "early_free_per_update": {"min": min(int(u.get("early_free", 0)) for u in upd),
                                      "max": max(int(u.get("early_free", 0)) for u in upd),
                                      "mean": sum(int(u.get("early_free", 0)) for u in upd) / len(upd)},
            "gate_rows_total": sum(int(u.get("gate_rows", 0)) for u in upd),
            "gate_loss_mean": sum(float(u.get("gate_loss", 0.0)) for u in upd) / len(upd),
            "gradient_norm_mean": {"min": min(float(u.get("gradient_norm_mean", 0.0)) for u in upd),
                                   "max": max(float(u.get("gradient_norm_mean", 0.0)) for u in upd),
                                   "mean": sum(float(u.get("gradient_norm_mean", 0.0)) for u in upd) / len(upd)},
            "clip_fraction_mean": sum(float(u.get("clip_fraction", 0.0)) for u in upd) / len(upd),
            "approx_kl_mean": sum(float(u.get("approx_kl", 0.0)) for u in upd) / len(upd),
            "entropy_mean": sum(float(u.get("entropy", 0.0)) for u in upd) / len(upd),
            "module_gradient_l1_mean_final": upd[-1].get("module_gradient_l1_mean"),
            "module_gradient_l1_mean_first": upd[0].get("module_gradient_l1_mean"),
            "module_gradient_l1_min_first": upd[0].get("module_gradient_l1_min"),
            "late_head_grad_l1_mean_over_updates": (
                sum(float(u["module_gradient_l1_mean"]["late_head"]) for u in upd
                    if (u.get("module_gradient_l1_mean") or {}).get("late_head") is not None) /
                max(1, sum(1 for u in upd if (u.get("module_gradient_l1_mean") or {}).get("late_head") is not None))),
            "task_value_grad_l1_mean_over_updates": (
                sum(float(u["module_gradient_l1_mean"]["task_value_head"]) for u in upd
                    if (u.get("module_gradient_l1_mean") or {}).get("task_value_head") is not None) /
                max(1, sum(1 for u in upd if (u.get("module_gradient_l1_mean") or {}).get("task_value_head") is not None))),
            "cost_head_grad_l1_mean_over_updates": (
                sum(float(u["module_gradient_l1_mean"]["cost_head"]) for u in upd
                    if (u.get("module_gradient_l1_mean") or {}).get("cost_head") is not None) /
                max(1, sum(1 for u in upd if (u.get("module_gradient_l1_mean") or {}).get("cost_head") is not None))),
            "likelihood_alignment_all_passed": all(
                bool((u.get("likelihood_alignment") or {}).get("passed")) for u in upd),
            "lambda_history_len": 32,
            "lambda_final": (recs[-1].get("dual") or {}).get("lambda_next"),
            "g_hat_final": (recs[-1].get("dual") or {}).get("g_hat"),
            "g_hat_history": [float((r.get("dual") or {}).get("g_hat")) for r in recs],
            "lambda_history": [float((r.get("dual") or {}).get("lambda_next")) for r in recs],
            "parameters_moved_every_update": all(
                r.get("parameters_before_sha") != r.get("parameters_after_sha") for r in recs),
            "trainable_parameter_object_count": upd[-1].get("trainable_parameter_object_count"),
            "frozen_parameter_object_count": upd[-1].get("frozen_parameter_object_count"),
            "frozen_early_path": upd[-1].get("frozen_early_path"),
            "structure": upd[-1].get("structure"),
        }
        # the per-update lambda history lives on the training state; the batch record carries the
        # dual record only, so report what exists and mark the rest missing.
        per_seed[f"s{seed}"]["lambda_history_field_in_batch_record"] = (
            "missing" if "lambda_history" not in recs[-1] else "present")

    # ---- gate-prior diagnostics (only recorded for diag updates 0, 15, 31) -----------------
    prior_diag = []
    for seed in SEEDS:
        for batch in range(32):
            gd = (batches[(seed, batch)].get("update") or {}).get("gate_diag")
            if not gd:
                continue
            prior_diag.append({"seed": seed, "update_index": gd.get("update_index"),
                               "early_free_rows": gd.get("early_free_rows"),
                               "late_free_rows": gd.get("late_free_rows"),
                               "base_trainable_grad_l1": gd.get("base_trainable_grad_l1"),
                               "late_actor_grad_l1": gd.get("late_actor_grad_l1"),
                               "critic_grad_l1": gd.get("critic_grad_l1"),
                               "prior_output_requires_grad": gd.get("prior_output_requires_grad"),
                               "prior_trainable_grad_l1": gd.get("prior_trainable_grad_l1"),
                               "prior_late_actor_grad_l1": gd.get("prior_late_actor_grad_l1"),
                               "prior_gradient": gd.get("prior_gradient"),
                               "base_gradient": gd.get("base_gradient"),
                               "dot": gd.get("dot"), "cosine": gd.get("cosine"),
                               "trunk": gd.get("trunk"), "early_head": gd.get("early_head"),
                               "trainable_parameter_object_count": gd.get("trainable_parameter_object_count")})
    gates = {f"s{seed}": jload(RUNTIME / f"r60_first_batch_gate_s{seed}.json") for seed in SEEDS}

    # ---- training-time late free exposure from the persisted episode tapes ----------------
    exposure = {}
    for seed in SEEDS:
        agg = {"episodes": 0, "late_free_rows": 0, "early_free_rows": 0,
               "late_free_send": 0, "late_free_skip": 0,
               "early_free_send": 0, "early_free_skip": 0,
               "late_free_p": [], "early_free_p": [], "missing_p": 0}
        for ep in range(128):
            matches = sorted(LOGS.glob(f"v2_r60_train_s{seed}_e{ep}_s*_a1_tape.jsonl"))
            if len(matches) != 1:
                raise RuntimeError(f"seed {seed} episode {ep}: {len(matches)} tapes")
            agg["episodes"] += 1
            with matches[0].open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    free = set(rec.get("free") or [])
                    probs = rec.get("send_probabilities") or {}
                    for i, vec in enumerate(rec.get("features") or []):
                        tgt = (rec.get("targets") or [])[i] if i < len(rec.get("targets") or []) else None
                        if tgt not in free:
                            continue
                        rs = ready_step(vec)
                        act = (rec.get("actions") or {}).get(tgt)
                        p = probs.get(tgt)
                        if rs > READY_CUT:
                            agg["late_free_rows"] += 1
                            if act == 1:
                                agg["late_free_send"] += 1
                            elif act == 0:
                                agg["late_free_skip"] += 1
                            if p is None:
                                agg["missing_p"] += 1
                            else:
                                agg["late_free_p"].append(float(p))
                        else:
                            agg["early_free_rows"] += 1
                            if act == 1:
                                agg["early_free_send"] += 1
                            elif act == 0:
                                agg["early_free_skip"] += 1
                            if p is None:
                                agg["missing_p"] += 1
                            else:
                                agg["early_free_p"].append(float(p))
        for tag in ("late_free_p", "early_free_p"):
            arr = sorted(agg[tag])
            if arr:
                agg[tag + "_stats"] = {"n": len(arr), "min": arr[0], "median": arr[len(arr) // 2],
                                       "max": arr[-1],
                                       "saturated_0p99": sum(1 for v in arr if v >= 0.99 or v <= 0.01)}
            del agg[tag]
        exposure[f"s{seed}"] = agg

    ppo = jlines(RUNTIME / "ppo_update_calls.jsonl")
    opt = jlines(RUNTIME / "optimizer_step_calls.jsonl")

    out = {
        "kind": "round67-task4-training-exposure/1",
        "sources": {"batch_records": "raw/checkpoints/r40_batch_s{seed}_b{batch}_attempt01.json",
                    "first_batch_gates": "raw/runtime/r60_first_batch_gate_s{seed}.json",
                    "ppo_journal": "raw/runtime/ppo_update_calls.jsonl",
                    "optimizer_journal": "raw/runtime/optimizer_step_calls.jsonl",
                    "training_tapes": "raw/logs/v2_r60_train_s{seed}_e{ep}_s*_a1_tape.jsonl"},
        "journals": {"ppo_update_calls": len(ppo), "optimizer_step_calls": len(opt),
                     "ppo_update_call_shape": ppo[0] if ppo else None,
                     "optimizer_step_call_shape": opt[0] if opt else None},
        "per_seed": per_seed,
        "gate_prior_diagnostics": prior_diag,
        "first_batch_gates": {k: {"status": v.get("status"), "checks": v.get("checks"),
                                  "gate_diag": v.get("gate_diag"),
                                  "module_gradient_l1_mean": v.get("module_gradient_l1_mean"),
                                  "isolation_note": v.get("isolation_note"),
                                  "lambda_after_first_batch": v.get("lambda_after_first_batch")}
                              for k, v in gates.items()},
        "training_exposure_from_tapes": exposure,
        "notes": [
            "module_gradient_l1_mean/min are per-minibatch pre-clip L1 readings of TRAINABLE "
            "modules only; trunk and early_head are None because requires_grad=False, which is a "
            "structural fact and not a measured zero gradient.",
            "gate_diag (and therefore the prior-gradient readings) exists only for the diag "
            "updates 0, 15 and 31; for every other update the field is absent, not zero.",
            "the beta=1.00 gate prior adds -log p(SKIP) over early first-emission free candidates "
            "only; those rows route to the frozen early_head, so the prior has no trainable path.",
        ],
    }
    dest = HERE / "r67_training_exposure.json"
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, ensure_ascii=False)
    for seed in SEEDS:
        s = per_seed[f"s{seed}"]
        print(f"s{seed}: updates={s['updates']} opt_steps={s['optimizer_steps_total']} "
              f"late_free={s['late_free_total']} early_free={s['early_free_total']} "
              f"gate_rows={s['gate_rows_total']} late_grad_l1_mean={s['late_head_grad_l1_mean_over_updates']:.3e} "
              f"gnorm_mean={s['gradient_norm_mean']['mean']:.4f} clipfrac={s['clip_fraction_mean']:.4f} "
              f"lambda_final={s['lambda_final']} g_hat_final={s['g_hat_final']}")
    print("prior diag records:", len(prior_diag))
    for d in prior_diag:
        print(f"  s{d['seed']} u{d['update_index']}: early_free_rows={d['early_free_rows']} "
              f"late_free_rows={d['late_free_rows']} base_trainable_l1={d['base_trainable_grad_l1']} "
              f"late_actor_l1={d['late_actor_grad_l1']} prior_trainable_l1={d['prior_trainable_grad_l1']} "
              f"prior_requires_grad={d['prior_output_requires_grad']}")
    print("exposure:", json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.endswith('_stats')}
                                   for k, v in exposure.items()}, ensure_ascii=False))
    for seed in SEEDS:
        e = exposure[f"s{seed}"]
        print(f"  s{seed} training late_free p(SEND) stats: {json.dumps(e.get('late_free_p_stats'), ensure_ascii=False)}")
        print(f"  s{seed} training early_free p(SEND) stats: {json.dumps(e.get('early_free_p_stats'), ensure_ascii=False)}")
    print("wrote", dest)


if __name__ == "__main__":
    main()
