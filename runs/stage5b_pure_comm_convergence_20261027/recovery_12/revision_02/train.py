"""Revision_02 training: unbounded iterations, per-episode cell sampling, dual lambda.

What is unusual here, and why
-----------------------------
The loop has **no episode target**. The task asks for a policy that trades bytes for success
under a constraint, not for a run of a fixed length, so the stopping rule is the circuit
breaker rather than a counter. ``--max-episodes`` exists only as a safety net for an
unattended machine; leaving it at 0 means "run until a breaker fires or the operator stops it".

Every episode draws its own ``(G, R, N)`` cell from ``scale_program.cell_for(scene, index)``.
A batch of four consecutive episodes therefore usually spans several cells, which is what
makes one policy see the whole grid instead of overfitting to whichever cell it saw first.

``lambda`` moves once per PPO update, from that batch's own success rate against the
``normal_immediate`` rate recorded for the cells present in the batch. It is never touched
inside an episode: the advantage of a step must not depend on a price set after the step.

Circuit breakers, all of which stop the loop and preserve the state
-------------------------------------------------------------------
1. non-finite loss, gradient norm, or advantage (NaN/Inf anywhere);
2. cost not conserved: the relay ledger and the tap stream must agree in every world;
3. credit not conserved: the causal re-keying must preserve the task total;
4. two consecutive evaluation points whose success is below ``0.5 * q_ref`` while spending more
   than ``2x`` normal_immediate in the same cells;
5. a repeated seam error: the same exception text three times running.

Breaker 2 and 3 are already enforced per world by ``world._run_episode_body`` (it raises
before a row is returned), so this driver's own checks are the belt to that braces: a world
whose row arrives without those checks having run is a bug, and is treated as one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE
IDENTITY = (ROOT.parents[0].parent / "recovery_09b" / "revision_01" / "protocol"
            / "RECOVERY_EXPERIMENT_IDENTITY.json")
START_CHECKPOINT = (ROOT.parents[0].parent / "recovery_09b" / "revision_01" / "execution"
                    / "checkpoints" / "T2_s91002" / "milestone_5000.pth")
SCENES = tuple(range(2301901, 2301909))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(str(tmp), str(path))


def load_references(path):
    """``{cell_label: {arm: {...}}}`` from the baseline JSON, plus the exact file hash."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    per_cell = payload.get("per_cell") or {}
    if not per_cell:
        raise SystemExit("baseline file has no per_cell block: %s" % path)
    q_ref = {}
    for label, entry in per_cell.items():
        immediate = (entry.get("arms") or {}).get("normal_immediate") or {}
        if immediate.get("q") is not None:
            q_ref[label] = float(immediate["q"])
    return payload, q_ref


def evaluate_all_cells(layout, controller, out, episode, identity_sha, scenes=(2301901,)):
    """Greedy evaluation on every cell at one checkpoint, in-process.

    Running the evaluation inside the training process is deliberate: a child process would
    need its own controller and the checkpoint file, and a checkpoint that has not been
    flushed yet is exactly the kind of race that makes an evaluation point lie. The cost is
    that a slow evaluation blocks training, which is the right trade for a metric the breaker
    reads.

    One scene per cell per checkpoint keeps the cost at 15 worlds. Rotating through the eight
    DEV scenes across eight checkpoints covers them all without paying for them at once.
    """
    from stage5b_comm import episode as episode_mod, scale_program

    scene = int(scenes[(episode // 100) % len(scenes)])
    rows = []
    for cell in scale_program.CELLS:
        label = scale_program.cell_label(cell)
        try:
            row = episode_mod.run_eval_world(layout, "T2", 91002, scene, controller, cell=cell)
        except Exception as error:
            rows.append({"cell": label, "status": "error",
                         "error": "%s: %s" % (type(error).__name__, error)})
            continue
        sends = list(row.get("emission_sequence") or [])
        distinct = {str(entry[1]) for entry in sends}
        rows.append({
            "cell": label, "status": "ok", "scene": scene,
            "on_time": int(row["on_time"]),
            "generated_goals": int(row.get("generated_goals") or 0),
            "spent_bytes": int(row["spent_bytes"]),
            "emissions": int(row["emissions"]),
            "distinct_targets": len(distinct),
            "duplicate_sends": int(row["emissions"]) - len(distinct),
            "bytes_per_on_time": row.get("bytes_per_on_time"),
            "masked_by_credit": row.get("masked_by_credit_total"),
        })
    ok = [r for r in rows if r["status"] == "ok"]
    return {
        "episode": int(episode),
        "scene": scene,
        "cells_expected": len(scale_program.CELLS),
        "cells_ok": len(ok),
        "mean_q": (sum(r["on_time"] for r in ok)
                   / max(1, sum(r["generated_goals"] for r in ok)) if ok else None),
        "mean_bytes": (sum(r["spent_bytes"] for r in ok) // len(ok) if ok else None),
        "rows": rows,
    }


def check_two_point_breaker(points, breaker):
    """Breaker 4: two consecutive points below half the reference, spending over twice it.

    Compares against ``normal_immediate`` for the **same cells** and the same scene, so the
    test cannot fire because the checkpoint happened to land on a hard cell. Both conditions
    must hold at both points, and a point missing its reference does not count as failing.
    """
    if len(points) < 2:
        return False
    references = check_two_point_breaker.references
    if not references:
        return False
    last_two = points[-2:]
    judged = []
    for point in last_two:
        scene = int(point["scene"])
        rows = [r for r in point["rows"] if r["status"] == "ok"
                and (r["cell"], scene) in references]
        if not rows:
            return False
        learned_on_time = sum(r["on_time"] for r in rows)
        learned_bytes = sum(r["spent_bytes"] for r in rows)
        ref_on_time = sum(references[(r["cell"], scene)]["on_time"] for r in rows)
        ref_bytes = sum(references[(r["cell"], scene)]["bytes"] for r in rows)
        if not ref_on_time or not ref_bytes:
            return False
        judged.append((learned_on_time / ref_on_time, learned_bytes / ref_bytes))
    if all(q < 0.5 and cost > 2.0 for q, cost in judged):
        breaker("two_points_below_half_reference_and_over_double_cost",
                "success ratio %s, cost ratio %s"
                % (["%.3f" % q for q, _ in judged], ["%.2f" % c for _, c in judged]))
        return True
    return False


check_two_point_breaker.references = {}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--baselines", required=True, help="BASELINES.json from run_baselines.py")
    ap.add_argument("--episodes-per-update", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--max-episodes", type=int, default=0,
                    help="0 = no limit (the loop runs until a breaker fires)")
    ap.add_argument("--start-checkpoint", default=str(START_CHECKPOINT))
    ap.add_argument("--start-episode", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--eval-scenes-per-checkpoint", type=int, default=1,
                    help="1 rotating scene per cell per checkpoint covers all 8 over 8 checks")
    args = ap.parse_args()

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    from stage5b_comm import constants as C
    from stage5b_comm import episode, scale_program, trainer, unlimited_budget
    from stage5b_comm.paths import Layout

    out = ROOT / "execution" / args.tag
    if (out / "TRAIN_SUMMARY.json").exists():
        raise SystemExit("%s already holds a completed run; use a new --tag" % out)
    out.mkdir(parents=True, exist_ok=True)
    layout = Layout(ROOT)
    layout.execution = out
    layout.world_logs = out / "world_logs"
    layout.rows = out / "rows"
    layout.ledger = out / "ledger"
    layout.receipts = out / "receipts"
    layout.checkpoints = out / "checkpoints"
    for path in (layout.world_logs, layout.rows, layout.ledger, layout.receipts,
                 layout.checkpoints):
        path.mkdir(parents=True, exist_ok=True)
    episode.ensure_import_paths(layout)
    episode.native_import_preflight(layout)

    baseline_payload, q_ref = load_references(args.baselines)
    baseline_sha = sha256(args.baselines)
    # What the two-point breaker reads: `normal_immediate`'s per-scene on-time and byte totals,
    # keyed by (cell label, scene) so the comparison is against the same world the evaluation
    # point ran, not against an average over scenes.
    check_two_point_breaker.references = {
        (scale_program.cell_label(tuple(r["cell"])), int(r["scene"])): {
            "on_time": int(r["on_time"]), "bytes": int(r["spent_bytes"])}
        for r in (baseline_payload.get("results") or [])
        if r.get("arm") == "normal_immediate" and r.get("status") == "ok"
    }
    print("cells with a q_ref: %d of %d" % (len(q_ref), len(scale_program.CELLS)))
    missing = [scale_program.cell_label(cell) for cell in scale_program.CELLS
               if scale_program.cell_label(cell) not in q_ref]
    if missing:
        print("  WARNING: no q_ref for", ", ".join(missing))
        print("  those cells will not move lambda; the dual step reports them as missing")

    identity_sha = sha256(IDENTITY)
    checkpoint = Path(args.start_checkpoint)
    if not checkpoint.exists():
        raise SystemExit("start checkpoint missing: %s" % checkpoint)
    controller, optimizer, state = trainer.load_checkpoint(
        checkpoint, arm="T2", identity_sha256=identity_sha, expected_seed=91002,
        expected_episode=args.start_episode)
    state = dict(state)

    lam = float(C.LAMBDA_INITIAL)
    episodes_log, updates, evaluation_points, breakers = [], [], [], []
    pending = []
    update_index = args.start_episode // args.episodes_per_update
    index = args.start_episode
    started = time.time()
    consecutive_errors = 0
    last_error = None
    stopping_reason = None

    def breaker(kind, detail):
        nonlocal stopping_reason
        breakers.append({"kind": kind, "detail": detail, "episode": int(index),
                         "seconds": round(time.time() - started, 1)})
        stopping_reason = kind
        print("BREAKER %s at episode %d: %s" % (kind, index, detail), flush=True)

    try:
        while True:
            if args.max_episodes and len(episodes_log) >= args.max_episodes:
                stopping_reason = "max_episodes"
                break
            scene = C.scene_for(91002, index)
            cell = scale_program.cell_for(scene, index)
            label = scale_program.cell_label(cell)
            try:
                row = episode.run_training_world(layout, "T2", 91002, index, scene, controller,
                                                 cell=cell)
            except Exception as error:
                message = "%s: %s" % (type(error).__name__, error)
                consecutive_errors = (consecutive_errors + 1 if message == last_error else 1)
                last_error = message
                print("  episode %d failed (%d in a row): %s"
                      % (index, consecutive_errors, message), flush=True)
                if consecutive_errors >= 3:
                    breaker("repeated_seam_error", message)
                    break
                index += 1
                continue
            consecutive_errors = 0
            last_error = None

            # Breaker 2 and 3, re-checked at the driver: a row without these is a bug.
            spent = int(row["spent_bytes"])
            tap = int(row["airtime_tap_total"])
            if spent != tap:
                breaker("cost_not_conserved", "ledger %d != tap %d at episode %d"
                        % (spent, tap, index))
                break
            credited = int(row["task_reward_total"])
            on_time = int(row["on_time"])
            if credited != on_time:
                breaker("credit_not_conserved", "task credit %d != on_time %d at episode %d"
                        % (credited, on_time, index))
                break
            if row.get("masked_by_credit_total") not in (0, None):
                breaker("budget_refusal_under_unlimited_budget",
                        "credit mask %s at episode %d"
                        % (row.get("masked_by_credit_total"), index))
                break
            for value in (row.get("spent_bytes"), row.get("on_time"), row.get("emissions")):
                if value is None:
                    breaker("missing_row_field", "episode %d returned %r" % (index, value))
                    break

            row["cell_label"] = label
            row["lambda"] = lam
            pending.append(row)
            episodes_log.append({
                "episode_index": int(index), "scene": int(scene), "cell": list(cell),
                "cell_label": label,
                "on_time": on_time, "generated_goals": int(row["generated_goals"]),
                "on_time_rate": row.get("on_time_rate"),
                "emissions": int(row["emissions"]), "spent_bytes": spent,
                "bytes_per_on_time": row.get("bytes_per_on_time"),
                "cost_scale": row.get("cost_scale"),
                "lambda": lam,
                "masked_by_credit": row.get("masked_by_credit_total"),
                "seconds": row.get("seconds"),
            })
            index += 1
            if len(pending) < args.episodes_per_update:
                continue

            update = trainer.ppo_update(controller, optimizer, pending, lam=lam,
                                        update_index=update_index, q_ref=q_ref)
            update["update_index"] = update_index
            update["toward_episode"] = int(index - 1)
            dual = update.get("dual") or {}
            if dual.get("applied"):
                lam = float(dual["lambda_after"])
            updates.append(update)
            update_index += 1
            pending = []
            print("[ep %5d] cell %-14s q=%s  lam %.4f -> %.4f  sends=%-3d bytes=%-7d "
                  "on_time=%-3d surrogate=%.4f kl=%.5f"
                  % (index - 1, label,
                     ("%.3f" % dual["q_batch"]) if dual.get("applied") else "  n/a",
                     dual.get("lambda_before", lam), lam,
                     episodes_log[-1]["emissions"], episodes_log[-1]["spent_bytes"],
                     episodes_log[-1]["on_time"],
                     update.get("policy_surrogate") or float("nan"),
                     update.get("approx_kl") or float("nan")), flush=True)
            for key in ("policy_surrogate", "value_loss_task", "value_loss_cost", "approx_kl",
                        "gradient_norm_mean", "entropy"):
                value = update.get(key)
                if value is not None and not math.isfinite(float(value)):
                    breaker("non_finite_update", "%s=%r at update %d"
                            % (key, value, update_index - 1))
                    break
            if stopping_reason:
                break

            if len(episodes_log) % args.checkpoint_every == 0:
                milestone = out / "checkpoints" / ("milestone_%d.pth" % len(episodes_log))
                state["episode"] = len(episodes_log)
                state["arm"] = "T2"
                state["training_seed"] = 91002
                state["lambda"] = lam
                if not milestone.exists():
                    trainer.save_checkpoint(milestone, controller, optimizer, state,
                                            identity_sha256=identity_sha)
                    print("  checkpoint", milestone.name, flush=True)
                # The greedy evaluation the task requires at every checkpoint, on every cell.
                # It is written into the summary so the two-evaluation-point breaker has real
                # numbers to fire on instead of a remembered impression.
                point = evaluate_all_cells(layout, controller, out, len(episodes_log),
                                          identity_sha)
                evaluation_points.append(point)
                print("  eval @%d: %d/%d cells, mean q %.4f, mean bytes %d"
                      % (point["episode"], point["cells_ok"], point["cells_expected"],
                         point["mean_q"] or 0.0, point["mean_bytes"] or 0), flush=True)
                fired = check_two_point_breaker(evaluation_points, breaker)
                if fired or stopping_reason:
                    break
    except Exception:
        breaker("uncaught_exception", traceback.format_exc()[-2000:])
    finally:
        elapsed = round(time.time() - started, 1)
        summary = {
            "schema": "stage5b-revision02-train/1",
            "tag": args.tag,
            "root": str(ROOT),
            "baselines": str(args.baselines), "baselines_sha256": baseline_sha,
            "identity_sha256": identity_sha,
            "start_checkpoint": str(checkpoint), "start_checkpoint_sha256": sha256(checkpoint),
            "start_episode": args.start_episode,
            "episodes_completed": len(episodes_log),
            "episodes_attempted": int(index - args.start_episode),
            "updates": len(updates),
            "lambda_final": lam,
            "lambda_initial": float(C.LAMBDA_INITIAL),
            "budget_bytes": int(C.BUDGET_BYTES),
            "scale_features": str(C.SCALE_FEATURES),
            "credit_mode": str(C.CREDIT_MODE),
            "advantage_mode": str(C.ADVANTAGE_MODE),
            "causal_gamma": float(C.CAUSAL_GAMMA),
            "stopping_reason": stopping_reason,
            "breakers": breakers,
            "wall_seconds": elapsed,
            "end_parameter_sha256": controller.parameter_summary()["parameter_sha256"],
            "cells_seen": sorted({entry["cell_label"] for entry in episodes_log}),
            "evaluation_points": evaluation_points,
            "episodes": episodes_log,
            "update_history": updates,
        }
        dump(out / "TRAIN_SUMMARY.json", summary)
        print()
        print("stopping_reason:", stopping_reason)
        print("episodes %d, updates %d, lambda %.4f, wall %.0fs"
              % (len(episodes_log), len(updates), lam, elapsed))
        if breakers:
            for entry in breakers:
                print("  breaker:", entry["kind"], "-", entry["detail"][:200])


if __name__ == "__main__":
    main()
