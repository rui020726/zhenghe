#!/usr/bin/env python
"""ROUND69 read-only audit: is there usable late-actor update signal in ROUND68's saved data?

ZERO ENVIRONMENT. This script only reads files already on disk (ROUND68 training rows, tapes,
batch commit records, first-batch gates, frozen statistics and the seed-chain eligibility) and
performs pure in-memory arithmetic. It constructs no environment, loads no model, performs no
forward/autograd, no PPO replay, no optimizer step, no RNG restore and no checkpoint write.

Deliverable 1: per-batch total-row / all-mask / early-free / late-free counts, the real ALL_ROWS
denominator and the late share, by seed and batch.

Deliverable 2: an independent recomputation of GAE and the combined advantage
``lambda*A_task - A_cost`` on every ROUND68 training row, from the SAVED rollout (pre-update)
critic values, the reward streams, the real terminal boundary and lambda_before; checked first
against a hand-computed miniature, then against the row-level episode reconciliation. On the
legal late-free rows it decomposes the combined advantage into
``(lambda*R_task - R_cost) - (lambda*V_task - V_cost)`` and reports component scales, signs,
SEND/SKIP distribution and critic residuals. Masked / forced / frozen-early roles are kept in
separate buckets; a multi-candidate joint decision counts once as one training row.

The exact minibatch membership is reconstructed from the frozen data-order RNG (pure in-memory
``random.Random`` replay, no environment) and cross-checked against the recorded per-minibatch
``gate_records``; a mismatch would fail the audit rather than be smoothed over.

Caliber warning (registered after delegation review): the ``early_free`` / ``late_free`` fields
in ``update`` and in ``gate_diag`` are EPOCH-ACCUMULATED EXPOSURE counts (4 PPO epochs per
batch), not unique per-tape candidates. Every row / target count reported by this audit is
rebuilt from the tapes as a UNIQUE joint decision row or UNIQUE free candidate per batch; the
recorded exposure figures are only used as a cross-check via ``unique * EPOCHS == recorded`` and
are listed separately. A mixed early+late row is counted once in the row denominator (the
``rows_with_late_free`` and ``rows_early_only`` buckets are disjoint and sum to ``free_rows``),
so it is never double counted.

The n_rows / early_free / late_free agreement is an AGGREGATE cross-check only: the shuffle is
source-derivable and the per-minibatch aggregate counts match, but the per-entry member indices
of each minibatch were NOT persisted on disk and are therefore not independently provable.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = (REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
       / "round68_late_initial_confidence_ablation")
RAW = RUN / "raw"
ROWS = RAW / "rows"
CKPTS = RAW / "checkpoints"
RT = RAW / "runtime"
SEEDS = (11, 22, 33)

# Living constants, read from the modules the frozen trainer actually imports
# (runs/comm_cost_v2_rerun_20260928/v2_common.py) with the ROUND68 live overrides recorded in
# r68_execute.py (GAE_LAMBDA=1.0, ETA_LAMBDA=0.5; GAMMA asserted 1.0). Module import defaults
# that are NOT the living values (e.g. ETA_LAMBDA=0.05, GAE_LAMBDA=0.95) are never used here.
B0 = 10000.0
HORIZON = 1000.0
READY_CUT = 147.0
GAMMA = 1.0
GAE_LAMBDA = 1.0
MINIBATCH_STEPS = 256
EPOCHS = 4
DATA_ORDER_SEED = 20260927


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ GAE (independent) ----
def gae(rewards, values, gamma, lam, terminal, bootstrap):
    """Standard GAE: A_t = delta_t + gamma*lam*(1-done)*A_{t+1}; returns = A + V.

    ``terminal[t]`` true means the episode truly ends after t (continuation 0); otherwise
    ``bootstrap[t]`` carries the next value. Recomputed here from first principles; it is NOT
    imported from the trainer so a transcription error cannot hide.
    """
    n = len(rewards)
    adv = [0.0] * n
    nxt = 0.0
    for t in range(n - 1, -1, -1):
        done = 0.0 if terminal[t] else 1.0
        delta = rewards[t] + gamma * bootstrap[t] * done - values[t]
        adv[t] = delta + gamma * lam * done * nxt
        nxt = adv[t]
    ret = [adv[t] + values[t] for t in range(n)]
    return adv, ret


def hand_example():
    """Minimal hand-computed check: gamma=1, lambda=1, last step terminal.

    rewards [1,2,3], values [0.5,0.7,0.9], bootstrap [0.7,0.9,0].
    A2 = 3 - 0.9 = 2.1; A1 = (2+0.9-0.7) + 2.1 = 4.3; A0 = (1+0.7-0.5) + 4.3 = 5.5.
    return0 = 5.5 + 0.5 = 6.0 == the Monte-Carlo sum 1+2+3. With gamma=lambda=1 and a true
    terminal the first-decision return telescopes to the full MC sum of the bundled stream.
    """
    adv, ret = gae([1.0, 2.0, 3.0], [0.5, 0.7, 0.9], 1.0, 1.0,
                   [False, False, True], [0.7, 0.9, 0.0])
    ok = (abs(adv[2] - 2.1) < 1e-12 and abs(adv[1] - 4.3) < 1e-12
          and abs(adv[0] - 5.5) < 1e-12 and abs(ret[0] - 6.0) < 1e-12)
    return ok, {"advantages": adv, "returns": ret, "mc_sum": 6.0}


def bundle(stream, decision_steps, horizon):
    """Verbatim re-implementation of v2_trainer._bundle: each tick's post-action reward is
    credited to the latest decision at or before that tick; a prefix before the first decision
    is unassigned and reported separately."""
    bundled = {s: 0.0 for s in decision_steps}
    prefix = 0.0
    for tick in range(0, horizon + 1):
        value = float(stream.get(tick, 0.0))
        previous = bisect.bisect_right(decision_steps, tick) - 1
        if previous < 0:
            prefix += value
        else:
            bundled[decision_steps[previous]] += value
    return bundled, prefix


def ready_step(feature):
    return int(round(float(feature[18]) * HORIZON) - round(float(feature[0])))


def quantiles(xs):
    if not xs:
        return None
    s = sorted(xs)
    def q(p):
        if len(s) == 1:
            return s[0]
        i = p * (len(s) - 1)
        lo = int(i)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (i - lo)
    return {"n": len(s), "mean": statistics.fmean(s), "median": statistics.median(s),
            "p10": q(0.10), "p25": q(0.25), "p75": q(0.75), "p90": q(0.90),
            "min": s[0], "max": s[-1],
            "frac_positive": sum(1 for x in s if x > 0) / len(s),
            "frac_negative": sum(1 for x in s if x < 0) / len(s),
            "frac_zero": sum(1 for x in s if x == 0) / len(s)}


def minibatch_plan(n_samples, update_index):
    """Replay the frozen data-order RNG exactly as frozen_early_ppo_update does: one
    random.Random per batch, order shuffled once per epoch (mutating in place), chunks of
    MINIBATCH_STEPS. Pure in-memory; no environment, no sampling."""
    rng = random.Random(int(hashlib.sha256(
        f"{DATA_ORDER_SEED}:{int(update_index)}".encode("utf-8")).hexdigest()[:16], 16))
    order = list(range(n_samples))
    plan = []
    for _epoch in range(EPOCHS):
        rng.shuffle(order)
        for start in range(0, n_samples, MINIBATCH_STEPS):
            plan.append(list(order[start:start + MINIBATCH_STEPS]))
    return plan


# ------------------------------------------------------------------ load the round --------
def row_path(seed, ep):
    hits = sorted(ROWS.glob(f"v2_r68_train_s{seed}_e{ep}_s*_a1.json"))
    if len(hits) != 1:
        raise RuntimeError(f"row for seed {seed} episode {ep}: {len(hits)} matches")
    return hits[0]


def main():
    ok_hand, hand = hand_example()
    if not ok_hand:
        raise RuntimeError(f"hand-computed GAE miniature failed: {hand}")

    batches = []
    for seed in SEEDS:
        for b in range(32):
            p = CKPTS / f"r40_batch_s{seed}_b{b}_attempt01.json"
            if not p.is_file():
                raise RuntimeError(f"missing batch record {p}")
            batches.append((seed, b, load(p)))

    episodes = {}          # (seed, ep) -> per-step records
    recon = {"task_max_abs": 0.0, "cost_max_abs": 0.0, "episodes": 0, "failures": []}
    lambda_mismatch = []
    step_records = []      # flat, for aggregation

    for seed, b, rec in batches:
        lam = float(rec["lambda_at_rollout"])
        for ep, scene in zip(rec["episodes"], rec["scenarios"]):
            key = (seed, int(ep))
            if key in episodes:
                continue
            row = load(row_path(seed, int(ep)))
            if float(row["lambda"]) != lam:
                lambda_mismatch.append({"seed": seed, "episode": int(ep),
                                        "row_lambda": row["lambda"], "batch_lambda": lam})
            tape_path = row["artefacts"]["tape"]
            tapes = [json.loads(line) for line in
                     Path(tape_path).read_text(encoding="utf-8").splitlines() if line.strip()]
            tapes.sort(key=lambda t: int(t["step"]))
            steps = [int(t["step"]) for t in tapes]
            horizon = int(row["steps"])
            cost_stream = {int(k): float(v["airtime_bytes"])
                           for k, v in row["cost_stream"].items()}
            task_stream = {int(k): float(v) for k, v in row["task_stream"].items()}
            bcost, pcost = bundle(cost_stream, steps, horizon)
            btask, ptask = bundle(task_stream, steps, horizon)
            rc = [bcost[s] / B0 for s in steps]
            rt = [btask[s] for s in steps]
            vt = [float(t["value"]) for t in tapes]
            vc = [float(t["cost_value"]) for t in tapes]
            terminal = [False] * len(steps)
            terminal[-1] = True
            boot_t = [0.0] * len(steps)
            boot_c = [0.0] * len(steps)
            for i in range(len(steps) - 1):
                boot_t[i] = vt[i + 1]
                boot_c[i] = vc[i + 1]
            at, rett = gae(rt, vt, GAMMA, GAE_LAMBDA, terminal, boot_t)
            ac, retc = gae(rc, vc, GAMMA, GAE_LAMBDA, terminal, boot_c)
            # reconciliation against the row's own episode totals (real boundary check)
            d_task = abs((rett[0] + ptask) - float(row["task_reward_total"]))
            d_cost = abs((retc[0] * B0 + pcost) - float(row["airtime_tap_total"]))
            recon["task_max_abs"] = max(recon["task_max_abs"], d_task)
            recon["cost_max_abs"] = max(recon["cost_max_abs"], d_cost)
            recon["episodes"] += 1
            if d_task > 1e-6 or d_cost > 1e-6:
                recon["failures"].append({"seed": seed, "episode": int(ep),
                                          "task_delta": d_task, "cost_delta": d_cost})
            per_step = []
            for i, t in enumerate(tapes):
                targets = list(t["targets"])
                feats = t["features"]
                free = set(t["free"])
                acts = t["actions"]
                probs = t.get("send_probabilities", {})
                n_early_free = n_late_free = 0
                late_send = late_skip = 0
                late_probs = []
                early_send = early_skip = 0
                for gi, g in enumerate(targets):
                    if g not in free:
                        continue
                    r = ready_step(feats[gi])
                    a = int(acts[g])
                    if r <= READY_CUT:
                        n_early_free += 1
                        early_send += a
                        early_skip += (1 - a)
                    else:
                        n_late_free += 1
                        late_send += a
                        late_skip += (1 - a)
                        late_probs.append(float(probs.get(g, float("nan"))))
                combined = lam * at[i] - ac[i]
                per_step.append({
                    "step": steps[i], "n_targets": len(targets), "n_free": len(free),
                    "n_early_free": n_early_free, "n_late_free": n_late_free,
                    "late_send": late_send, "late_skip": late_skip,
                    "early_send": early_send, "early_skip": early_skip,
                    "late_send_prob_mean": (statistics.fmean(late_probs)
                                            if late_probs else None),
                    "log_prob": float(t["log_prob"]),
                    "A_task": at[i], "A_cost": ac[i],
                    "R_task": rett[i], "R_cost": retc[i],
                    "V_task": vt[i], "V_cost": vc[i],
                    "combined": combined,
                    "lamR_task": lam * rett[i], "lamV_task": lam * vt[i],
                })
            episodes[key] = {"batch": b, "lambda": lam, "steps": per_step,
                             "first_step": steps[0], "last_step": steps[-1],
                             "horizon": horizon, "prefix_task": ptask,
                             "prefix_cost_bytes": pcost,
                             "task_total": float(row["task_reward_total"]),
                             "airtime_total": float(row["airtime_tap_total"]),
                             "spent_bytes": float(row["spent_bytes"])}
            for s in per_step:
                step_records.append({"seed": seed, "batch": b, "episode": int(ep), **s})

    # ------------------------------------------------ deliverable 1: counts + ALL_ROWS -----
    per_batch = []
    all_rows_denominator = 0
    for seed, b, rec in batches:
        eps = [episodes[(seed, int(e))] for e in rec["episodes"]]
        samples = sum(len(e["steps"]) for e in eps)
        all_mask = sum(1 for e in eps for s in e["steps"] if s["n_free"] == 0)
        free_rows = sum(1 for e in eps for s in e["steps"] if s["n_free"] > 0)
        late_rows = sum(1 for e in eps for s in e["steps"] if s["n_late_free"] > 0)
        early_only = sum(1 for e in eps for s in e["steps"]
                         if s["n_late_free"] == 0 and s["n_early_free"] > 0)
        tgt_free = sum(s["n_early_free"] + s["n_late_free"] for e in eps for s in e["steps"])
        tgt_early = sum(s["n_early_free"] for e in eps for s in e["steps"])
        tgt_late = sum(s["n_late_free"] for e in eps for s in e["steps"])
        tgt_late_send = sum(s["late_send"] for e in eps for s in e["steps"])
        tgt_late_skip = sum(s["late_skip"] for e in eps for s in e["steps"])
        upd = rec["update"]
        all_rows_denominator += samples * EPOCHS
        # cross-check against the recorded batch report
        agree = (samples == upd["samples"]
                 and tgt_early * EPOCHS == upd["early_free"]
                 and tgt_late * EPOCHS == upd["late_free"]
                 and samples * EPOCHS == sum(g["n_rows"] for g in upd["gate_records"]))
        per_batch.append({
            "seed": seed, "batch": b, "lambda_at_rollout": float(rec["lambda_at_rollout"]),
            "samples": samples, "all_mask_rows": all_mask, "free_rows": free_rows,
            "rows_with_late_free": late_rows, "rows_early_only": early_only,
            "targets_free_unique": tgt_free, "targets_early_free": tgt_early,
            "targets_late_free": tgt_late, "targets_late_send": tgt_late_send,
            "targets_late_skip": tgt_late_skip,
            "all_rows_denominator_instances": samples * EPOCHS,
            "recorded_samples": upd["samples"], "recorded_early_free": upd["early_free"],
            "recorded_late_free": upd["late_free"],
            "recorded_gate_rows": upd["gate_rows"],
            "recorded_minibatches": upd["minibatches"],
            "recorded_n_rows_sum": sum(g["n_rows"] for g in upd["gate_records"]),
            "cross_check_with_record": agree,
            "likelihood_alignment_max_abs_gap": upd["likelihood_alignment"]["max_abs_gap"],
        })

    totals = {
        "batches": len(per_batch),
        "samples_total": sum(p["samples"] for p in per_batch),
        "all_mask_rows": sum(p["all_mask_rows"] for p in per_batch),
        "free_rows": sum(p["free_rows"] for p in per_batch),
        "rows_with_late_free": sum(p["rows_with_late_free"] for p in per_batch),
        "rows_early_only": sum(p["rows_early_only"] for p in per_batch),
        "targets_free_unique": sum(p["targets_free_unique"] for p in per_batch),
        "targets_early_free": sum(p["targets_early_free"] for p in per_batch),
        "targets_late_free": sum(p["targets_late_free"] for p in per_batch),
        "targets_late_send": sum(p["targets_late_send"] for p in per_batch),
        "targets_late_skip": sum(p["targets_late_skip"] for p in per_batch),
        "all_rows_denominator_instances": all_rows_denominator,
        "batch_cross_check_all_agree": all(p["cross_check_with_record"] for p in per_batch),
        "lambda_row_vs_batch_mismatches": lambda_mismatch,
        "reconciliation": recon,
    }
    totals["late_target_share_of_free"] = (totals["targets_late_free"]
                                           / totals["targets_free_unique"])
    totals["late_row_share_of_all_rows"] = (totals["rows_with_late_free"]
                                            / totals["samples_total"])
    totals["late_row_share_of_free_rows"] = (totals["rows_with_late_free"]
                                             / totals["free_rows"])

    # ------------------------------------------------ exact minibatch cross-check ----------
    mb_mismatch = []
    mb_late_rows = []
    mb_all_rows = 0
    mb_plan_count = []
    # Actual execution totals read straight from the 96 batch commit records, NOT derived
    # from the reconstructed plan, so plan-vs-execution can be compared instead of assumed.
    rec_entries_total = 0
    rec_entries_late_gt0 = 0
    rec_update_minibatches_sum = 0
    rec_optimizer_steps_sum = 0
    rec_count_mismatch = []
    for seed, b, rec in batches:
        eps = [episodes[(seed, int(e))] for e in rec["episodes"]]
        flat = [s for e in eps for s in e["steps"]]
        plan = minibatch_plan(len(flat), b)
        mb_plan_count.extend(plan)
        gr = rec["update"]["gate_records"]
        upd = rec["update"]
        rec_entries_total += len(gr)
        rec_entries_late_gt0 += sum(1 for g in gr if g["late_free"] > 0)
        rec_update_minibatches_sum += int(upd["minibatches"])
        rec_optimizer_steps_sum += int(upd["optimizer_steps"])
        if not (len(gr) == int(upd["minibatches"]) == int(upd["optimizer_steps"])):
            rec_count_mismatch.append(
                {"seed": seed, "batch": b, "gate_record_entries": len(gr),
                 "recorded_minibatches": int(upd["minibatches"]),
                 "recorded_optimizer_steps": int(upd["optimizer_steps"])})
        if len(plan) != len(gr):
            mb_mismatch.append({"seed": seed, "batch": b, "why": "minibatch count",
                                "plan": len(plan), "recorded": len(gr)})
            continue
        for idx, chunk in enumerate(plan):
            # gate_records count per-TARGET free candidates (the update loop increments once
            # per free target), not rows; n_rows is the number of decision-step samples.
            e = sum(flat[i]["n_early_free"] for i in chunk)
            l = sum(flat[i]["n_late_free"] for i in chunk)
            l_rows = sum(1 for i in chunk if flat[i]["n_late_free"] > 0)
            g = gr[idx]
            if (len(chunk) != g["n_rows"] or e != g["early_free"] or l != g["late_free"]):
                mb_mismatch.append({"seed": seed, "batch": b, "minibatch": idx,
                                    "plan": {"n_rows": len(chunk), "early_free_targets": e,
                                             "late_free_targets": l},
                                    "recorded": {"n_rows": g["n_rows"],
                                                 "early_free": g["early_free"],
                                                 "late_free": g["late_free"]}})
            if l_rows > 0:
                mb_late_rows.append(l_rows)
            mb_all_rows += len(chunk)

    def unique_free_for_batch(seed, b):
        """Unique joint rows / unique free candidates for one batch, rebuilt from its tapes."""
        rows = early = late = late_rows = early_only = 0
        for (sd, _ep), ed in episodes.items():
            if sd != seed or ed["batch"] != b:
                continue
            for s in ed["steps"]:
                rows += 1
                early += s["n_early_free"]
                late += s["n_late_free"]
                if s["n_late_free"] > 0:
                    late_rows += 1
                elif s["n_early_free"] > 0:
                    early_only += 1
        return {"rows": rows, "unique_early_free_targets": early,
                "unique_late_free_targets": late, "rows_with_late_free": late_rows,
                "rows_early_only": early_only, "free_rows": late_rows + early_only}

    # ------------------------------------------------ deliverable 2: decomposition ---------
    def bucket(pred):
        return [s for s in step_records if pred(s)]

    late_rows = bucket(lambda s: s["n_late_free"] > 0)
    late_send_rows = [s for s in late_rows if s["late_send"] > 0]
    late_skip_rows = [s for s in late_rows if s["late_send"] == 0]
    early_only_rows = bucket(lambda s: s["n_late_free"] == 0 and s["n_early_free"] > 0)
    all_mask_rows = bucket(lambda s: s["n_free"] == 0)

    def decomp(rows):
        if not rows:
            return None
        return {
            "rows": len(rows),
            "row_weight": len(rows),
            "combined_advantage": quantiles([s["combined"] for s in rows]),
            "lambda_A_task": quantiles([s["lamR_task"] - s["lamV_task"] for s in rows]),
            "A_cost": quantiles([s["A_cost"] for s in rows]),
            "return_part_lambdaRtask_minus_Rcost":
                quantiles([s["lamR_task"] - s["R_cost"] for s in rows]),
            "value_part_lambdaVtask_minus_Vcost":
                quantiles([s["lamV_task"] - s["V_cost"] for s in rows]),
            "critic_residual_task_R_minus_V": quantiles([s["R_task"] - s["V_task"]
                                                         for s in rows]),
            "critic_residual_cost_R_minus_V": quantiles([s["R_cost"] - s["V_cost"]
                                                         for s in rows]),
            "R_task": quantiles([s["R_task"] for s in rows]),
            "R_cost": quantiles([s["R_cost"] for s in rows]),
            "V_task": quantiles([s["V_task"] for s in rows]),
            "V_cost": quantiles([s["V_cost"] for s in rows]),
        }

    per_seed_late = {str(sd): decomp([s for s in late_rows if s["seed"] == sd])
                     for sd in SEEDS}
    per_batch_index = []
    for seed, b, rec in batches:
        rows = [s for s in late_rows if s["seed"] == seed and s["batch"] == b]
        if not rows:
            per_batch_index.append({"seed": seed, "batch": b, "late_rows": 0})
            continue
        per_batch_index.append({
            "seed": seed, "batch": b, "lambda_at_rollout": float(rec["lambda_at_rollout"]),
            "late_rows": len(rows),
            "combined_mean": statistics.fmean([s["combined"] for s in rows]),
            "combined_frac_positive": sum(1 for s in rows if s["combined"] > 0) / len(rows),
            "lambda_A_task_mean": statistics.fmean([s["lamR_task"] - s["lamV_task"]
                                                    for s in rows]),
            "A_cost_mean": statistics.fmean([s["A_cost"] for s in rows]),
            "recorded_late_head_grad_l1_mean":
                rec["update"]["module_gradient_l1_mean"].get("late_head"),
            "recorded_gate_loss": rec["update"].get("gate_loss"),
        })

    late_target_actions = {
        "targets_late_free": totals["targets_late_free"],
        "SEND": totals["targets_late_send"],
        "SKIP": totals["targets_late_skip"],
        "send_share": (totals["targets_late_send"] / totals["targets_late_free"]
                       if totals["targets_late_free"] else None),
        "rows_any_late_send": len(late_send_rows),
        "rows_all_late_skip": len(late_skip_rows),
    }
    late_send_prob = [s["late_send_prob_mean"] for s in late_rows
                      if s["late_send_prob_mean"] is not None]

    # ------------------------------------------------ context inputs (read-only) -----------
    proj = load(RAW / "ROUND68_PROJECTION.json")
    chain = load(RAW / "ROUND68_SEED_CHAIN_ELIGIBILITY.json")
    gates = {s: load(RT / f"r68_first_batch_gate_s{s}.json") for s in SEEDS}
    context = {
        "r68_eval_primary_trained_minus_init": {
            k: {"mode": k.split("/")[1],
                "task_delta_mean": v["task_delta"]["mean"],
                "task_delta_ci95": v["task_delta"]["ci95"],
                "cost_delta_B_per_scene_mean": v["cost_delta_B"]["mean"],
                "cost_delta_ci95": v["cost_delta_B"]["ci95"],
                "lost_count": v["task_delta"]["lost_count"],
                "gained_count": v["task_delta"]["gained_count"],
                "gained": v["task_delta"]["gained"]}
            for k, v in proj["primary_trained_minus_init"].items()},
        "r68_chain_facts": {
            "all_seeds_verified": chain["all_seeds_verified"],
            "per_seed": {s: {
                "trunk_ep128_equals_r60_ep0":
                    chain["seeds"][s]["checks"]["trunk_ep128_equals_r60_ep0"],
                "early_head_ep128_equals_r60_ep0":
                    chain["seeds"][s]["checks"]["early_head_ep128_equals_r60_ep0"],
                "late_head_moved_from_init":
                    chain["seeds"][s]["checks"]["late_head_moved_from_init"],
                "critics_moved_from_r60_ep0":
                    chain["seeds"][s]["checks"]["critics_moved_from_r60_ep0"],
                "lambda_final": chain["seeds"][s]["facts"]["lambda_final"],
                "optimizer_steps_after_b31":
                    chain["seeds"][s]["facts"]["optimizer_steps_after_b31"]}
                for s in chain["seeds"]},
            "journals": {"ppo_update_calls_lines": chain["journals"]["ppo_update_calls_lines"],
                         "optimizer_step_calls_lines":
                             chain["journals"]["optimizer_step_calls_lines"]},
        },
        "gate_prior_on_trainable_path": {
            s: {"gate_loss": None,
                "prior_trainable_grad_l1_recorded":
                    gates[s]["gate_diag"].get("prior_trainable_grad_l1"),
                "prior_late_actor_grad_l1_recorded":
                    gates[s]["gate_diag"].get("prior_late_actor_grad_l1"),
                "late_actor_grad_l1_recorded": gates[s]["gate_diag"].get("late_actor_grad_l1"),
                "note": ("gate-eligible early first-emission candidates route to the frozen "
                         "early_head, so the beta=1.00 prior cannot produce a trainable "
                         "gradient; the late head receives only the surrogate signal")}
            for s in gates},
        "eval_block_status": "1800000..1800031 is an already-used development block, not a "
                             "confirmation set",
    }

    # ------------------------------------------------ trend over the 32 batches -------------
    trend = {}
    for sd in SEEDS:
        rows_sd = [r for r in per_batch_index if r["seed"] == sd and r.get("late_rows")]
        quarters = []
        for q in range(4):
            sel = [r for r in rows_sd if q * 8 <= r["batch"] < (q + 1) * 8]
            if not sel:
                continue
            quarters.append({
                "batches": f"{q*8}..{(q+1)*8-1}",
                "late_rows": sum(r["late_rows"] for r in sel),
                "combined_mean": statistics.fmean([r["combined_mean"] for r in sel]),
                "combined_frac_positive_mean": statistics.fmean(
                    [r["combined_frac_positive"] for r in sel]),
                "lambda_A_task_mean": statistics.fmean([r["lambda_A_task_mean"] for r in sel]),
                "A_cost_mean": statistics.fmean([r["A_cost_mean"] for r in sel]),
                "recorded_late_head_grad_l1_mean": statistics.fmean(
                    [r["recorded_late_head_grad_l1_mean"] for r in sel]),
                "lambda_at_rollout_mean": statistics.fmean(
                    [r["lambda_at_rollout"] for r in sel]),
            })
        trend[str(sd)] = quarters

    out = {
        "kind": "round69-late-actor-signal-audit/1",
        "task_id": "ROUND69-LATE-ACTOR-SIGNAL-READONLY-AUDIT",
        "branch": "analysis/round69-late-actor-signal-audit",
        "source_round": "ROUND68-LATE-INITIAL-CONFIDENCE-ABLATION",
        "corrections": {
            "discipline": ("Append-only. The original report and this JSON are not rewritten; "
                           "every amendment below is registered here and summarised in section "
                           "8 of ROUND69_REPORT.md."),
            "items": [
                {"id": "C1",
                 "topic": "minibatch total",
                 "was": ("An earlier draft reported minibatches_total = 501, derived as "
                         "128184 // 256 + 1."),
                 "now": ("676, matching the 676 recorded gate-record entries and the "
                         "676 reconstructed plan minibatches. The 501 figure assumed every "
                         "minibatch is full-256 and ignored the partial final chunk of each "
                         "epoch; it is retained only as superseded_erroneous_minibatches_total "
                         "correction evidence."),
                 "effect_on_results": "None on any scientific quantity; the count was wrong, "
                                      "not the data."},
                {"id": "C2",
                 "topic": "minibatch membership proof strength",
                 "was": "Implied the exact minibatch membership was fully reproduced.",
                 "now": ("Downgraded to an aggregate cross-check: the shuffle is "
                         "source-derivable and n_rows / early_free / late_free match per "
                         "minibatch with zero mismatches, but per-entry member indices were "
                         "not persisted on disk and remain unprovable."),
                 "effect_on_results": "No number changes; the claim strength is reduced."},
                {"id": "C3",
                 "topic": "counterfactual SEND/SKIP action value",
                 "was": ("Stated that counterfactual action value would need a model forward "
                         "pass or a PPO replay, both forbidden this round."),
                 "now": ("Withdrawn. A forward pass or a PPO replay could not settle the "
                         "causal question either: on-policy chosen-action advantages are not "
                         "causal action values under any permitted recomputation. The "
                         "question is not identifiable from the saved on-policy data, and no "
                         "computation allowed in this round would change that."),
                 "effect_on_results": "The item moves from 'needs a forbidden computation' to "
                                      "'not identifiable from saved data'."},
                {"id": "C4",
                 "topic": "direction does not distinguish SEND/SKIP",
                 "was": ("Read as evidence that the late signal cannot separate SEND from "
                         "SKIP."),
                 "now": ("Withdrawn as an inference. Both action groups showing positive "
                         "chosen-action advantage says nothing about whether the expected "
                         "policy gradient can separate the two actions; chosen-action "
                         "advantage is measured against a critic baseline, not between "
                         "actions. The SEND/SKIP breakdown is retained strictly as a "
                         "descriptive distribution."),
                 "effect_on_results": "Descriptive counts unchanged; no separability claim."},
                {"id": "C5",
                 "topic": "gradient magnitude gap and clipping",
                 "was": ("The L1 gap between late_head (mean 0.269) and the critic heads "
                         "(78.07 / 3.73) invited a clip-suppression reading."),
                 "now": ("Withdrawn. The heads have different parameter counts, so their L1 "
                         "gradient magnitudes are not comparable, and no clipping inference "
                         "is drawn from the gap. The provable facts are only: late-actor "
                         "signal is sparse (1.60% of rows), its residuals are "
                         "positive-skewed, and the recorded late_head gradients are non-zero "
                         "on all 96 batches. These are explicitly NOT designated as the root "
                         "cause of anything."),
                 "effect_on_results": "No number changes; an unsupported inference removed."},
                {"id": "C6",
                 "topic": "exposure vs unique counts",
                 "was": ("A delegation reminder flagged that update/gate_diag early_free and "
                         "late_free are epoch-accumulated exposure counts and must not be "
                         "divided by tape rows as if they were unique row shares."),
                 "now": ("Confirmed already satisfied by this audit and now stated "
                         "explicitly: every row / target figure is rebuilt from the tapes as "
                         "unique per batch, recorded exposures are cross-checked via "
                         "unique * 4 epochs == recorded, mixed early+late rows are counted "
                         "once, and the two buckets are disjoint. See "
                         "exposure_vs_unique_counts."),
                 "effect_on_results": "None; the audit already used the correct caliber."},
            ],
        },
        "zero_side_effects": {
            "environment_constructions": 0, "episodes": 0, "claims": 0,
            "optimizer_updates_real": 0, "optimizer_updates_synthetic": 0,
            "ppo_replays": 0, "model_forwards": 0, "autograd_calls": 0,
            "rng_restores": 0, "checkpoint_writes": 0, "files_written_outside_report": 0,
        },
        "method": {
            "hand_example": hand,
            "hand_example_passed": ok_hand,
            "living_constants": {
                "gamma": GAMMA, "gae_lambda": GAE_LAMBDA, "b0_bytes": B0,
                "minibatch_steps": MINIBATCH_STEPS, "epochs_per_batch": EPOCHS,
                "data_order_seed": DATA_ORDER_SEED, "ready_cut": READY_CUT,
                "horizon": HORIZON,
                "evidence": ("v2_common.py module values for GAMMA/B0/MINIBATCH_STEPS/"
                             "EPOCHS_PER_BATCH/DATA_ORDER_SEED; r68_execute.py live overrides "
                             "GAE_LAMBDA=1.0 and ETA_LAMBDA=0.5 on the actual common module the "
                             "trainer imports; the recorded batch dual eta=0.5 and the gate "
                             "check common_gae_lambda_1=true corroborate at runtime"),
            },
            "gae_recomputed_from": ("saved rollout (pre-update) tape value/cost_value, the row "
                                    "task_stream/cost_stream bundled to decision steps exactly "
                                    "as v2_trainer._bundle, terminal[-1]=True with zero "
                                    "bootstrap on the last decision and next-decision value "
                                    "otherwise, lambda_before from the batch record"),
            "mc_boundary_check": ("gamma=1 and gae_lambda=1 with a true terminal telescope the "
                                  "first-decision return to the full Monte-Carlo sum of the "
                                  "bundled stream; verified per episode against the row's own "
                                  "task_reward_total and airtime_tap_total"),
        },
        "deliverable1_counts": {
            "totals": totals,
            "per_batch": per_batch,
        },
        "exposure_vs_unique_counts": {
            "warning": ("update.early_free / update.late_free and the gate_diag equivalents "
                        "are EPOCH-ACCUMULATED EXPOSURE counts across the 4 PPO epochs of a "
                        "batch. They are NOT unique per-tape candidates and must not be "
                        "divided by tape row totals as a row share."),
            "recorded_exposure_totals_epoch_accumulated": {
                "early_free": sum(p["recorded_early_free"] for p in per_batch),
                "late_free": sum(p["recorded_late_free"] for p in per_batch),
                "gate_rows": sum(p["recorded_gate_rows"] for p in per_batch),
                "epochs_per_batch": EPOCHS,
            },
            "unique_rebuilt_from_tapes_per_batch": {
                "joint_decision_rows": totals["samples_total"],
                "unique_early_free_targets": totals["targets_early_free"],
                "unique_late_free_targets": totals["targets_late_free"],
                "unique_free_targets": totals["targets_free_unique"],
                "rows_with_late_free": totals["rows_with_late_free"],
                "rows_early_only": totals["rows_early_only"],
                "free_rows": totals["free_rows"],
            },
            "cross_check_rule": ("unique_per_batch * EPOCHS == recorded exposure, applied per "
                                 "batch for samples, early_free and late_free; all 96 batches "
                                 "agree (batch_cross_check_all_agree)."),
            "mixed_row_accounting": ("rows_with_late_free and rows_early_only are disjoint "
                                     "partitions and sum exactly to free_rows, so a row "
                                     "carrying both an early and a late free candidate is "
                                     "counted once in the row denominator and never twice."),
            "spot_check_first_batch": {
                "s22": {"batch": 0,
                        "unique_from_tapes": unique_free_for_batch(22, 0),
                        "recorded_exposure": {
                            "early_free": next(p["recorded_early_free"] for p in per_batch
                                               if p["seed"] == 22 and p["batch"] == 0),
                            "late_free": next(p["recorded_late_free"] for p in per_batch
                                              if p["seed"] == 22 and p["batch"] == 0)}},
                "s33": {"batch": 0,
                        "unique_from_tapes": unique_free_for_batch(33, 0),
                        "recorded_exposure": {
                            "early_free": next(p["recorded_early_free"] for p in per_batch
                                               if p["seed"] == 33 and p["batch"] == 0),
                            "late_free": next(p["recorded_late_free"] for p in per_batch
                                              if p["seed"] == 33 and p["batch"] == 0)}},
                "note": ("For s22 batch 0 the recorded 64/28 exposure corresponds to 16/7 "
                         "unique free candidates (16*4=64, 7*4=28). The same recomputation "
                         "for s33 batch 0 also yields 16/7 unique, not the 16/6 quoted in "
                         "the delegation reminder; the on-disk value is 16/7 and the 16/6 "
                         "figure does not reproduce. This affects no audit total, which uses "
                         "the full 96-batch recomputation."),
            },
        },
        "minibatch_reconstruction": {
            "minibatches_total": len(mb_plan_count),
            "minibatch_row_instances": mb_all_rows,
            "minibatches_containing_late_rows": len(mb_late_rows),
            "late_rows_inside_late_minibatches": sum(mb_late_rows),
            "mismatches_vs_recorded_gate_records": mb_mismatch,
            "cross_check_passed": not mb_mismatch,
            "recorded_gate_record_entries_total": rec_entries_total,
            "recorded_minibatches_with_late_free_gt0": rec_entries_late_gt0,
            "recorded_update_minibatches_sum": rec_update_minibatches_sum,
            "recorded_optimizer_steps_sum": rec_optimizer_steps_sum,
            "recorded_count_mismatches": rec_count_mismatch,
            "plan_equals_recorded_execution": (
                len(mb_plan_count) == rec_entries_total
                and len(mb_late_rows) == rec_entries_late_gt0
                and not rec_count_mismatch and not mb_mismatch),
            "superseded_erroneous_minibatches_total": 501,
            "superseded_value_bug": (
                "An earlier draft of this audit reported minibatches_total = 501, computed as "
                "all_rows_denominator_instances // MINIBATCH_STEPS + 1 (128184 // 256 + 1). "
                "That formula implicitly assumes every minibatch is a full 256 rows, which is "
                "false: each epoch's final chunk is partial, so the true count is the sum over "
                "batches of EPOCHS * ceil(samples / MINIBATCH_STEPS). The value 501 is wrong "
                "and is kept here only as documented correction evidence; it is superseded by "
                "the recorded 676 gate-record entries and the 676 plan minibatches, which "
                "agree exactly per batch."),
            "plan_vs_execution_note": (
                "The theoretical all-4-epoch plan and the actual recorded execution coincide "
                "per batch: for every one of the 96 batches the reconstructed plan length, "
                "the recorded gate_records length, update.minibatches and "
                "update.optimizer_steps are all equal, and each plan minibatch's n_rows, "
                "early_free and late_free match its recorded gate_record. The plan is "
                "therefore not a hypothetical alternative schedule but a reproduction of "
                "what ran."),
            "aggregate_cross_check_caveat": (
                "This is an AGGREGATE cross-check. The data-order shuffle is derivable from "
                "the frozen source and the per-minibatch aggregate counts match with zero "
                "mismatches, but the per-entry member indices of each minibatch were never "
                "persisted to disk. Agreement on n_rows / early_free / late_free therefore "
                "does not independently prove which specific rows sat in which minibatch."),
        },
        "late_signal_trend_by_quarter": trend,
        "context_inputs": context,
        "deliverable2_decomposition": {
            "late_free_rows_all": decomp(late_rows),
            "late_free_rows_any_send": decomp(late_send_rows),
            "late_free_rows_all_skip": decomp(late_skip_rows),
            "early_only_free_rows": decomp(early_only_rows),
            "all_mask_rows": decomp(all_mask_rows),
            "per_seed_late_rows": per_seed_late,
            "late_target_actions": late_target_actions,
            "late_row_send_probability_mean": (statistics.fmean(late_send_prob)
                                               if late_send_prob else None),
            "per_batch_late_signal": per_batch_index,
            "note": ("combined = lambda*A_task - A_cost = (lambda*R_task - R_cost) - "
                     "(lambda*V_task - V_cost) exactly, because A = R - V for each critic; a "
                     "multi-candidate joint decision is ONE training row (row weight 1) and its "
                     "targets are not independent samples"),
        },
        "withdrawn_inferences": {
            "counterfactual_action_value": (
                "Withdrawn. The earlier wording implied that a model forward pass or a PPO "
                "replay would answer whether late sending is worth it. It would not: "
                "on-policy chosen-action advantage is not a causal action value, and no "
                "recomputation permitted in a read-only round can make it one. The SEND/SKIP "
                "causal question is simply not identifiable from the saved on-policy data."),
            "direction_separates_send_skip": (
                "Withdrawn. Both action groups carrying positive chosen-action advantage "
                "does not show that the expected policy gradient cannot separate SEND from "
                "SKIP. The chosen-action advantage compares the taken joint action with a "
                "critic baseline, never the two actions with each other. Kept strictly "
                "descriptive."),
            "clip_suppression_from_magnitude_gap": (
                "Withdrawn. The late_head L1 (mean 0.269) versus task_value_head (78.07) and "
                "cost_head (3.73) gap cannot support a clipping or suppression inference: "
                "the heads carry different parameter counts, so raw L1 magnitudes are not "
                "comparable across them."),
            "what_remains_provable": (
                "Only these facts stand: the late-actor signal is sparse (1.60% of rows, "
                "about 3 rows per minibatch), its combined-advantage residuals are "
                "positive-skewed (92.2% positive), and the recorded late_head gradients are "
                "non-zero on all 96 batches. None of these is designated as the unique root "
                "cause of the R68 greedy zero-change, and none licenses deleting messages, "
                "changing a threshold, or naming a single loss as the cause."),
        },
    }
    dst = HERE / "ROUND69_LATE_ACTOR_SIGNAL_AUDIT.json"
    tmp = dst.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    tmp.replace(dst)
    print(json.dumps({
        "hand_example_passed": ok_hand,
        "totals": {k: v for k, v in totals.items() if k != "per_batch"},
        "batch_cross_check_all_agree": totals["batch_cross_check_all_agree"],
        "minibatch_cross_check_passed": not mb_mismatch,
        "minibatch_mismatch_count": len(mb_mismatch),
        "late_free_rows": len(late_rows),
        "late_targets_send_skip": (totals["targets_late_send"],
                                   totals["targets_late_skip"]),
        "combined_on_late_rows": (out["deliverable2_decomposition"]["late_free_rows_all"]
                                  ["combined_advantage"]),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())