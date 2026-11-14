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


def evaluate_all_cells(layout, controller, out, episode, identity_sha,
                       scenes=(2301901,), caps=None, floors=None, compute_waste=False):
    """Greedy evaluation on every cell at one checkpoint, in-process.

    Running the evaluation inside the training process is deliberate: a child process would
    need its own controller and the checkpoint file, and a checkpoint that has not been
    flushed yet is exactly the kind of race that makes an evaluation point lie. The cost is
    that a slow evaluation blocks training, which is the right trade for a metric the breaker
    and the dual both read.

    ``scenes`` is a tuple. train_v4g passes three, the same three ``evaluate.py`` runs, so the
    success rate the dual integrates over is the quantity the registered verdict measures --
    one scene per cell was measured to be a lottery (v4f: the same R=6 cell read 0.25 on one
    scene and 0.58 on another), and a dual fed by a lottery moves on noise.

    Returns one row per (cell, scene) plus a per-cell aggregate. The aggregate's ``q`` is
    ``sum(on_time) / sum(generated)`` over the scenes, which is exactly the denominator rule
    the external report uses.
    """
    from stage5b_comm import episode as episode_mod, scale_program

    scenes = tuple(int(scene) for scene in scenes)
    if not scenes:
        raise ValueError("at least one evaluation scene is required")
    rows = []
    for scene in scenes:
        for cell in scale_program.CELLS:
            label = scale_program.cell_label(cell)
            try:
                row = episode_mod.run_eval_world(
                    layout, "T2", 91002, scene, controller, cell=cell,
                    episode_index=episode,
                    cap=(caps or {}).get(label),
                    floor=(floors or {}).get(label))
            except Exception as error:
                rows.append({"cell": label, "scene": scene, "status": "error",
                             "error": "%s: %s" % (type(error).__name__, error)})
                continue
            sends = list(row.get("emission_sequence") or [])
            distinct = {str(entry[1]) for entry in sends}
            audit = row.get("scale_invariant") or {}
            record = {
                "cell": label, "scene": scene, "status": "ok",
                "on_time": int(row["on_time"]),
                "generated_goals": int(row.get("generated_goals") or 0),
                "spent_bytes": int(row["spent_bytes"]),
                "emissions": int(row["emissions"]),
                "distinct_targets": len(distinct),
                "duplicate_sends": int(row["emissions"]) - len(distinct),
                "bytes_per_on_time": row.get("bytes_per_on_time"),
                "masked_by_credit": row.get("masked_by_credit_total"),
                # Proof that the world saw the revision's own feature layer, not the frozen one.
                "horizon_denominator": (audit.get("horizon_scale") or {}).get("denominator"),
                "capacity_ratio": (audit.get("divisors") or {}).get("#16 capacity_ratio"),
            }
            if compute_waste:
                # The v4v dual prices useless broadcasts, so the scan has to know how many it
                # made. This is the paper's own definition, read from the two logs the world
                # just wrote; a failure here must not silently look like "no waste".
                from stage5b_comm import waste as waste_mod
                archived = list((row.get("artefacts") or {}).values())
                if not archived:
                    raise RuntimeError("%s scene %d: row carries no artefacts, so the scan "
                                       "cannot measure broadcast waste" % (label, scene))
                measured = waste_mod.world_waste(Path(archived[0]).parent)
                record.update({"waste_n": measured["waste_n"],
                               "waste_d": measured["distinct_broadcast"],
                               "waste_ratio": measured["waste_ratio"]})
            rows.append(record)
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        # An evaluation point that measured nothing is not a data point. Reporting 0/15 as if
        # it were one is how the episode-200 collision almost passed as a result.
        raise RuntimeError("evaluation at episode %d produced no successful world; first "
                           "error: %s" % (int(episode),
                                          (rows[0].get("error") if rows else "no rows")))
    cells = []
    for cell in scale_program.CELLS:
        label = scale_program.cell_label(cell)
        own = [r for r in ok if r["cell"] == label]
        if not own:
            cells.append({"cell": label, "status": "error", "scenes_ok": 0})
            continue
        on_time = sum(r["on_time"] for r in own)
        generated = sum(r["generated_goals"] for r in own)
        spent = sum(r["spent_bytes"] for r in own)
        cell_row = {
            "cell": label, "status": "ok", "scenes_ok": len(own),
            "on_time": on_time, "generated_goals": generated,
            "spent_bytes": spent, "emissions": sum(r["emissions"] for r in own),
            "q": (on_time / generated) if generated else None,
            "bytes_per_on_time": (spent / on_time) if on_time else None,
        }
        if compute_waste:
            broadcast = sum(int(r.get("waste_d") or 0) for r in own)
            useless = sum(int(r.get("waste_n") or 0) for r in own)
            cell_row.update({"waste_d": broadcast, "waste_n": useless,
                             "waste_ratio": (useless / broadcast) if broadcast else None})
        cells.append(cell_row)
    cells_ok = [c for c in cells if c["status"] == "ok"]
    return {
        "episode": int(episode),
        "scenes": list(scenes),
        "cells_expected": len(scale_program.CELLS),
        "cells_ok": len(cells_ok),
        "worlds_ok": len(ok),
        "mean_q": (sum(c["on_time"] for c in cells_ok)
                   / max(1, sum(c["generated_goals"] for c in cells_ok)) if cells_ok else None),
        "mean_bytes": (sum(c["spent_bytes"] for c in cells_ok) // len(cells_ok)
                       if cells_ok else None),
        "cells": cells,
        "rows": rows,
    }


def check_two_point_breaker(points, breaker):
    """Breaker 4: two consecutive points below half the reference, spending over twice it.

    Compares against ``normal_immediate`` for the **same cells** and the **same scenes**, so
    the test cannot fire because the checkpoint happened to land on a hard cell. Both
    conditions must hold at both points, and a point missing its reference does not count as
    failing.
    """
    if len(points) < 2:
        return False
    references = check_two_point_breaker.references
    if not references:
        return False
    last_two = points[-2:]
    judged = []
    for point in last_two:
        scenes = [int(scene) for scene in point["scenes"]]
        rows = [r for r in point["rows"] if r["status"] == "ok"
                and (r["cell"], int(r["scene"])) in references]
        if not rows:
            return False
        if {int(r["scene"]) for r in rows} != set(scenes):
            return False
        learned_on_time = sum(r["on_time"] for r in rows)
        learned_bytes = sum(r["spent_bytes"] for r in rows)
        ref_on_time = sum(references[(r["cell"], int(r["scene"]))]["on_time"] for r in rows)
        ref_bytes = sum(references[(r["cell"], int(r["scene"]))]["bytes"] for r in rows)
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
    ap.add_argument("--start-episode", type=int, default=5000,
                    help="episode recorded in the start checkpoint; the checkpoint check is exact")
    ap.add_argument("--start-index", type=int, default=None,
                    help="world index to continue from when it differs from the checkpoint's "
                         "recorded episode (a continuation keeps the same (scene, cell, seed) "
                         "stream rather than replaying the start of the run)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--eval-scenes-per-checkpoint", type=int, default=1,
                    help="1 rotating scene per cell per checkpoint covers all 8 over 8 checks")
    ap.add_argument("--eval-scenes", type=int, default=1,
                    help="how many of the three DEV scenes the in-process checkpoint evaluation "
                         "runs per cell (1 = v4f's rotating scene; 3 = the external report's "
                         "scenes, which is what train_v4g's dual anchor reads)")
    ap.add_argument("--lambda-mode", choices=("shared", "per_cell"), default="per_cell",
                    help="per_cell (train_v4_cell_lambda) keeps one lambda per (G, R, N) cell; "
                         "shared reproduces train_v3's single price")
    ap.add_argument("--episodes", type=int, default=0,
                    help="stop after this many episodes of this run (0 = until a breaker)")
    ap.add_argument("--lambda-step", type=float, default=None,
                    help="per-cell dual step; default is C.LAMBDA_STEP_PER_CELL (0.25)")
    ap.add_argument("--send-cap", choices=("off", "per_cell"), default="off",
                    help="per_cell switches the actuator: instead of a scalar price the dual "
                         "moves an integer page budget per cell (one page = one unit of the "
                         "verdict). Measured reason: at the same lambda one cell cut one page "
                         "while another cut ninety-seven, so a price cannot place 15 cells in "
                         "the 1-5 page window")
    ap.add_argument("--freeze-actor", action="store_true",
                    help="set every optimizer learning rate to 0 and skip nothing else: the "
                         "policy stays at the warm-start checkpoint while the send budget still "
                         "acts. Measured reason: with the budget alone the actor keeps drifting "
                         "(train_v4r moved G48_R6_N6 from 46 to 141 pages in 100 episodes) and "
                         "gives back the bytes the 18%% baseline had saved")
    ap.add_argument("--send-cap-lead", type=int, default=0,
                    help="start the page cap this many pages BELOW normal_immediate, so the "
                         "constraint binds from the first episode instead of after the first "
                         "scan. One page is one unit of the verdict")
    ap.add_argument("--send-cap-step", type=int, default=1,
                    help="pages added or removed per scan in --send-cap per_cell")
    ap.add_argument("--send-floor-auto", action="store_true",
                    help="set each cell's floor from its own deficit instead of a flat lead: "
                         "floor = measured pages x (0.965*q_ref / q), bounded by the cap. A flat "
                         "floor of immediate-2 forced 138 of 142 pages in cells v4t deliberately "
                         "cut to 57, so the byte ratio rose to 0.961")
    ap.add_argument("--send-floor-lead", type=int, default=0,
                    help="also enforce a MINIMUM of (normal_immediate pages - this) pages, paced "
                         "linearly over the episode. The cap alone cannot make a policy that "
                         "declines to send send; measured on v4t three cells under-sent and the "
                         "unseen-scene on-time rate fell to 0.894. 0 keeps the one-sided budget")
    ap.add_argument("--lambda-initial", type=float, default=None,
                    help="lambda every cell starts at when --resume-lambda is off; default is "
                         "unlimited_budget.LAMBDA_BY_CELL_INITIAL (3.0). Only for arms that "
                         "know where the usable band is: at total temperature 8 the band is "
                         "about 4-5.5, and 3.0 sits just below it (measured on train_v4m)")
    ap.add_argument("--resume-lambda", action="store_true",
                    help="continue the per-cell lambda and q_ema tables stored in the start "
                         "checkpoint instead of resetting them to the initial value")
    ap.add_argument("--action-temperature", type=float, default=1.0,
                    help="divide the action head's logits by this factor at warm start. argmax "
                         "is unchanged (so the greedy policy is identical at t), but the sampled "
                         "policy becomes trainable: measured send/hold margins are 12-15 logits "
                         "while the price term is <=0.12, so at t=1 the actor cannot move")
    ap.add_argument("--entropy-bonus", type=float, default=None,
                    help="override C.ENTROPY_BONUS (0.01). The measured bottleneck is a "
                         "near-one-hot send/hold margin of ~12 logits, so the entropy term is "
                         "the only lever on it: the price term is bounded by lambda <= 8")
    ap.add_argument("--advantage-scale", choices=("none", "std_only"), default="none",
                    help="none reproduces train_v3's gradient scale; std_only divides each "
                         "minibatch's advantage by its own sd, which is 10-20x here because the "
                         "critic residual is small next to a near-one-hot send/hold margin")
    ap.add_argument("--dual-greedy-anchor", action="store_true",
                    help="[v4f, kept for reproduction] at every checkpoint, overwrite a cell's "
                         "q_ema with the single-scene greedy success rate when the greedy rate "
                         "is worse. Superseded by --dual-anchor scene3, which is bilateral")
    ap.add_argument("--dual-mode", choices=("verdict", "verdict_waste"), default="verdict",
                    help="verdict = the registered per-cell price (raise while the cell beats "
                         "0.965*q_ref, otherwise lower it towards 0.25). verdict_waste = "
                         "train_v4v: when a cell is short of on-time the price decays to zero "
                         "and no cost term is added; otherwise the price rises on margin and on "
                         "broadcast waste above 0.10")
    ap.add_argument("--dual-anchor", choices=("off", "single", "scene3"), default="off",
                    help="what the dual integrates over between scans. off = the training "
                         "batches' sampling rate (v4d/v4e). single = v4f's one-way, "
                         "one-scene greedy floor. scene3 = train_v4g: the 3-scene greedy rate "
                         "the external report measures, assigned to q_ema in both directions "
                         "and then stepped once by the dual")
    args = ap.parse_args()
    if args.dual_greedy_anchor and args.dual_anchor != "off":
        raise SystemExit("--dual-greedy-anchor is the single-scene form of --dual-anchor; "
                         "pass only one of them")
    if args.dual_greedy_anchor:
        args.dual_anchor = "single"
    anchor_mode = args.dual_anchor
    # The anchor reads the same three scenes ``evaluate.py`` runs, so its success rate and the
    # verdict's are the same quantity. A single-scene anchor keeps v4f's rotating scene.
    eval_scene_count = 3 if anchor_mode == "scene3" else max(1, int(args.eval_scenes))

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    from stage5b_comm import constants as C
    from stage5b_comm import episode, scale_program, send_cap, trainer, unlimited_budget
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
    #: --send-cap per_cell: the ceiling is the page count of ``normal_immediate`` in the same
    #: cell, summed over the worlds that baseline ran. Measuring the ceiling from the current
    #: policy instead ratchets down whenever the actor drifts to sending less (measured on
    #: train_v4p: 17 pages/episode became a cap of 6, locking the over-cut in).
    immediate_pages = {}
    for label, entry in (baseline_payload.get("per_cell") or {}).items():
        immediate = (entry.get("arms") or {}).get("normal_immediate") or {}
        worlds = immediate.get("worlds")
        emissions = immediate.get("emissions")
        if worlds and emissions is not None:
            immediate_pages[label] = max(1, int(round(float(emissions) / float(worlds))))
    if args.send_cap == "per_cell":
        missing_pages = [label for label in q_ref if label not in immediate_pages]
        if missing_pages:
            raise SystemExit("no normal_immediate page count for %s; the cap has no ceiling"
                             % ", ".join(sorted(missing_pages)))
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

    if args.action_temperature != 1.0:
        if args.action_temperature <= 0.0:
            raise SystemExit("--action-temperature must be positive")
        import torch as _torch
        head = controller.inner.action_head
        with _torch.no_grad():
            before_norm = float(sum(float(p.norm()) for p in head.parameters()))
            for parameter in head.parameters():
                parameter.div_(float(args.action_temperature))
            after_norm = float(sum(float(p.norm()) for p in head.parameters()))
        print("action head divided by %.3f: parameter norm %.4f -> %.4f (argmax unchanged)"
              % (args.action_temperature, before_norm, after_norm), flush=True)
    if args.freeze_actor:
        frozen_groups = [group.get("lr") for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        print("actor frozen: learning rates %s -> 0.0 (policy stays at the warm start)"
              % (["%.2e" % value if value is not None else "None"
                  for value in frozen_groups]), flush=True)
    lam = float(C.LAMBDA_INITIAL)
    #: train_v4_cell_lambda: one price per (G, R, N) cell, and one smoothed success rate per
    #: cell to move it. A single shared lambda is driven by whichever cells a 4-episode batch
    #: happened to draw, which is how train_v3 ended up hitting a clamp 191 times.
    lambda_initial = float(unlimited_budget.LAMBDA_BY_CELL_INITIAL
                           if args.lambda_initial is None else args.lambda_initial)
    lambda_by_cell = {scale_program.cell_label(cell): lambda_initial
                      for cell in scale_program.CELLS}
    q_ema_by_cell = {}
    #: --send-cap per_cell: pages allowed per episode, one integer per cell. Empty means "no
    #: cap yet", which is what the first scan measures before it removes the first page.
    cap_by_cell = ({str(key): int(value)
                    for key, value in (state.get("cap_by_cell") or {}).items()}
                   if args.send_cap == "per_cell" else {})
    floor_by_cell = ({str(key): int(value)
                      for key, value in (state.get("floor_by_cell") or {}).items()}
                     if args.send_cap == "per_cell" else {})
    cap_trace = []
    if args.send_cap == "per_cell" and int(args.send_cap_lead):
        cap_by_cell.update({label: max(1, int(pages) - int(args.send_cap_lead))
                            for label, pages in immediate_pages.items()})
    if args.send_cap == "per_cell" and int(args.send_floor_lead):
        floor_by_cell.update({label: max(1, int(pages) - int(args.send_floor_lead))
                              for label, pages in immediate_pages.items()})
        over = [label for label, floor in floor_by_cell.items()
                if floor > cap_by_cell.get(label, 0)]
        if over:
            raise SystemExit("floor exceeds cap on %s; raise --send-cap-lead above the floor"
                             % ", ".join(sorted(over)))
    lam_step = float(C.LAMBDA_STEP_PER_CELL if args.lambda_step is None else args.lambda_step)
    if args.resume_lambda and args.lambda_mode == "per_cell":
        resumed = {str(key): float(value)
                   for key, value in (state.get("lambda_by_cell") or {}).items()}
        if not resumed:
            raise SystemExit("--resume-lambda asked for a lambda table the start checkpoint "
                             "does not carry")
        lambda_by_cell.update(resumed)
        q_ema_by_cell.update({str(key): float(value)
                              for key, value in (state.get("q_ema_by_cell") or {}).items()})
        print("resumed lambda table from %s: %d cells, mean %.3f"
              % (checkpoint.name, len(resumed),
                 sum(resumed.values()) / len(resumed)), flush=True)
    lambda_trace = []
    anchor_log = []
    episodes_log, updates, evaluation_points, breakers = [], [], [], []
    pending = []
    index = int(args.start_index if args.start_index is not None else args.start_episode)
    # The minibatch shuffle is seeded from the update index only, so a continuation must carry
    # on from its own position rather than repeating the start of the run's ordering.
    update_index = index // args.episodes_per_update
    if args.start_index is not None:
        print("world stream continues at index %d (checkpoint episode %d)"
              % (index, args.start_episode), flush=True)
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
                                                 cell=cell,
                                                 cap=cap_by_cell.get(label),
                                                 floor=floor_by_cell.get(label))
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
            row["lambda"] = (lambda_by_cell.get(label, lam) if args.lambda_mode == "per_cell"
                             else lam)
            pending.append(row)
            episodes_log.append({
                "episode_index": int(index), "scene": int(scene), "cell": list(cell),
                "cell_label": label,
                "on_time": on_time, "generated_goals": int(row["generated_goals"]),
                "on_time_rate": row.get("on_time_rate"),
                "emissions": int(row["emissions"]), "spent_bytes": spent,
                "bytes_per_on_time": row.get("bytes_per_on_time"),
                "cost_scale": row.get("cost_scale"),
                "cost_reward_factor": row.get("cost_reward_factor"),
                "lambda": row["lambda"],
                "masked_by_credit": row.get("masked_by_credit_total"),
                "seconds": row.get("seconds"),
            })
            index += 1
            if len(pending) < args.episodes_per_update:
                continue

            price = dict(lambda_by_cell) if args.lambda_mode == "per_cell" else lam
            update = trainer.ppo_update(controller, optimizer, pending, lam=price,
                                        update_index=update_index, q_ref=q_ref,
                                        q_ema_by_cell=(q_ema_by_cell
                                                       if args.lambda_mode == "per_cell"
                                                       else None),
                                        lambda_step=(lam_step
                                                     if args.lambda_mode == "per_cell"
                                                     else None),
                                        advantage_scale=args.advantage_scale,
                                        entropy_bonus=args.entropy_bonus,
                                        q_ema_frozen=(anchor_mode != "off"))
            update["update_index"] = update_index
            update["toward_episode"] = int(index - 1)
            dual = update.get("dual") or {}
            if dual.get("applied"):
                if args.lambda_mode == "per_cell":
                    lambda_by_cell = dict(dual["lambda_by_cell"])
                    q_ema_by_cell = dict(dual["q_ema_by_cell"])
                else:
                    lam = float(dual["lambda_after"])
            updates.append(update)
            update_index += 1
            pending = []
            prices = list(lambda_by_cell.values()) if args.lambda_mode == "per_cell" else [lam]
            price_mean = sum(prices) / len(prices)
            price_sd = (sum((value - price_mean) ** 2 for value in prices)
                        / max(1, len(prices) - 1)) ** 0.5
            lambda_trace.append({
                "update_index": int(update_index - 1),
                "toward_episode": int(index - 1),
                "mode": args.lambda_mode,
                "cell": label,
                "cell_lambda": float(lambda_by_cell.get(label, lam)
                                     if args.lambda_mode == "per_cell" else lam),
                "lambda_mean": round(price_mean, 6),
                "lambda_sd": round(price_sd, 6),
                "lambda_min": round(min(prices), 6),
                "lambda_max": round(max(prices), 6),
                "cells_moved": (sorted((dual.get("cells") or {})) if args.lambda_mode == "per_cell"
                                else []),
                "lambda_by_cell": ({key: round(float(value), 6)
                                    for key, value in sorted(lambda_by_cell.items())}
                                   if args.lambda_mode == "per_cell" else None),
                "q_ema_by_cell": ({key: round(float(value), 6)
                                   for key, value in sorted(q_ema_by_cell.items())}
                                  if args.lambda_mode == "per_cell" else None),
            })
            kl = update.get("approx_kl")
            surrogate = update.get("policy_surrogate")
            print("[ep %5d] cell %-14s q=%s  lam %.4f  mean %.4f sd %.3f  sends=%-3d "
                  "bytes=%-7d on_time=%-3d surrogate=%s kl=%s"
                  % (index - 1, label,
                     ("%.3f" % dual["q_batch"]) if "q_batch" in dual else "  n/a",
                     float(lambda_by_cell.get(label, lam))
                     if args.lambda_mode == "per_cell" else lam,
                     price_mean, price_sd,
                     episodes_log[-1]["emissions"], episodes_log[-1]["spent_bytes"],
                     episodes_log[-1]["on_time"],
                     ("n/a" if surrogate is None else "%.4f" % float(surrogate)),
                     # Negative zero is a legitimate KL reading; only None means "not computed".
                     ("n/a" if kl is None else "%.6f" % float(kl))), flush=True)
            dump(out / "lambda_trace.json", {
                "schema": "stage5b-revision02-lambda-trace/2",
                "mode": args.lambda_mode,
                "step": float(C.LAMBDA_STEP_PER_CELL if args.lambda_mode == "per_cell"
                              else C.LAMBDA_STEP),
                "step_effective": (lam_step if args.lambda_mode == "per_cell" else None),
                "ema_alpha": (float(unlimited_budget.Q_EMA_ALPHA)
                              if args.lambda_mode == "per_cell" else None),
                "q_target_fraction_dual": float(unlimited_budget.Q_TARGET_FRACTION_DUAL),
                "q_target_fraction_verdict": float(unlimited_budget.Q_TARGET_FRACTION),
                "dual_anchor_mode": anchor_mode,
                "initial": (float(unlimited_budget.LAMBDA_BY_CELL_INITIAL)
                            if args.lambda_mode == "per_cell" else float(C.LAMBDA_INITIAL)),
                "bounds": [float(C.LAMBDA_MIN), float(C.LAMBDA_MAX)],
                "initial_checkpoint": str(checkpoint),
                "initial_checkpoint_sha256": sha256(checkpoint),
                "entries": lambda_trace,
            })
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
                state["lambda_mode"] = args.lambda_mode
                state["lambda_by_cell"] = {key: float(value)
                                           for key, value in sorted(lambda_by_cell.items())}
                state["q_ema_by_cell"] = {key: float(value)
                                          for key, value in sorted(q_ema_by_cell.items())}
                state["dual_mode"] = args.dual_mode
                state["send_cap_mode"] = args.send_cap
                state["cap_by_cell"] = {key: int(value)
                                        for key, value in sorted(cap_by_cell.items())}
                state["floor_by_cell"] = {key: int(value)
                                          for key, value in sorted(floor_by_cell.items())}
                if not milestone.exists():
                    trainer.save_checkpoint(milestone, controller, optimizer, state,
                                            identity_sha256=identity_sha)
                    print("  checkpoint", milestone.name, flush=True)
                # The greedy evaluation the task requires at every checkpoint, on every cell.
                # It is written into the summary so the two-evaluation-point breaker has real
                # numbers to fire on instead of a remembered impression.
                if eval_scene_count > 1:
                    eval_scenes = tuple(SCENES[:eval_scene_count])
                else:
                    eval_scenes = (SCENES[(len(episodes_log) // 100) % len(SCENES)],)
                point = evaluate_all_cells(layout, controller, out, len(episodes_log),
                                          identity_sha, scenes=eval_scenes,
                                          caps=(cap_by_cell if args.send_cap == "per_cell"
                                                else None),
                                          floors=(floor_by_cell or None),
                                          compute_waste=(args.dual_mode == "verdict_waste"))
                evaluation_points.append(point)
                print("  eval @%d: %d/%d cells x %d scene(s), mean q %.4f, mean bytes %d"
                      % (point["episode"], point["cells_ok"], point["cells_expected"],
                         len(point["scenes"]), point["mean_q"] or 0.0,
                         point["mean_bytes"] or 0), flush=True)
                fired = check_two_point_breaker(evaluation_points, breaker)
                if anchor_mode != "off":
                    # The dual must integrate over the *deployed* policy's rate, measured the
                    # way the verdict measures it. v4f proved the direction was wrong (the
                    # sampling rate sat at 0.58 while the argmax achieved 0.25) but fed the
                    # dual a single scene, which is a lottery, and clamped it one-way so no
                    # cell could ever be told "you have margin, keep trimming". scene3 reads
                    # three scenes and assigns the aggregate in both directions; the dual then
                    # steps once on the refreshed rate, and between scans keeps stepping on it
                    # (q_ema_frozen) instead of on whatever the training batches happened to
                    # sample.
                    anchored = []
                    for cell_row in point["cells"]:
                        if cell_row.get("status") != "ok" or cell_row.get("q") is None:
                            continue
                        label = str(cell_row["cell"])
                        rate = q_ref.get(label)
                        if rate is None:
                            continue
                        greedy_q = float(cell_row["q"])
                        before_q = q_ema_by_cell.get(label)
                        before_lambda = lambda_by_cell.get(label,
                                                           unlimited_budget.LAMBDA_BY_CELL_INITIAL)
                        if anchor_mode == "single":
                            new_q = (greedy_q if before_q is None
                                     else min(float(before_q), greedy_q))
                        else:
                            new_q = greedy_q
                        q_ema_by_cell[label] = new_q
                        target = float(unlimited_budget.Q_TARGET_FRACTION_DUAL) * rate
                        if args.send_cap == "per_cell":
                            # Integer actuator: one step is one page, the same unit the verdict
                            # counts. The first scan measures what the uncapped policy spends per
                            # episode and removes the first page in the same visit.
                            measured = immediate_pages.get(label)
                            if measured is None:
                                scenes_ok = max(1, int(cell_row.get("scenes_ok") or 1))
                                measured = int(round(float(cell_row.get("emissions") or 0)
                                                     / float(scenes_ok)))
                            before_cap = cap_by_cell.get(label, measured)
                            after_cap, cap_detail = send_cap.update_cap(
                                before_cap, new_q, rate, step=args.send_cap_step,
                                measured=measured)
                            cap_by_cell[label] = int(after_cap)
                            direction = ("cut_page" if after_cap < before_cap
                                         else ("add_page" if after_cap > before_cap else "hold"))
                            cap_trace.append({
                                "cell": label, "episode": point["episode"],
                                "q_greedy_3": greedy_q, "q_ref": rate, "q_target": target,
                                "measured_pages": measured, "cap_before": before_cap,
                                "cap_after": int(after_cap), "direction": direction,
                                "clamped": bool(cap_detail["clamped"]),
                            })
                            anchored.append({
                                "cell": label, "episode": point["episode"],
                                "mode": "send_cap", "q_greedy_3": greedy_q,
                                "q_ema_before": before_q, "q_ema_after": new_q,
                                "cap_before": before_cap, "cap_after": int(after_cap),
                                "direction": direction,
                            })
                            continue
                        if args.dual_mode == "verdict_waste":
                            new_lambda, detail = unlimited_budget.update_lambda_verdict(
                                before_lambda, new_q, rate,
                                waste_ratio=cell_row.get("waste_ratio"), step=lam_step)
                        else:
                            new_lambda, detail = unlimited_budget.update_lambda_cell(
                                before_lambda, new_q, rate, step=lam_step)
                        lambda_by_cell[label] = new_lambda
                        anchored.append({
                            "cell": label, "episode": point["episode"],
                            "mode": anchor_mode, "dual_mode": args.dual_mode,
                            "q_greedy_3": greedy_q,
                            "q_ema_before": before_q, "q_ema_after": new_q,
                            "q_ref": rate,
                            "q_target": detail["q_target"],
                            "waste_ratio": detail.get("waste_ratio"),
                            "region": detail.get("region"),
                            "lambda_before": before_lambda,
                            "lambda_after": new_lambda,
                            "error": detail["error"],
                            "direction": detail["direction"],
                        })
                    if args.send_floor_auto:
                        # Only the cells that are actually short get a floor. The others keep 0,
                        # so the deep cuts v4t earned (G48_R6_N3: 57 of 142 pages) survive.
                        for entry in anchored:
                            label = entry["cell"]
                            cell_row = next((c for c in point["cells"]
                                             if str(c.get("cell")) == label), None)
                            if cell_row is None or cell_row.get("status") != "ok":
                                continue
                            scenes_ok = max(1, int(cell_row.get("scenes_ok") or 1))
                            measured = int(round(float(cell_row.get("emissions") or 0)
                                                 / float(scenes_ok)))
                            rate = q_ref.get(label)
                            q_now = float(cell_row["q"])
                            if rate is None or q_now <= 0.0:
                                continue
                            target = float(unlimited_budget.Q_TARGET_FRACTION_DUAL) * rate
                            if q_now >= target:
                                floor_by_cell[label] = 0
                                continue
                            need = int(math.ceil(measured * target / q_now))
                            floor_by_cell[label] = max(0, min(int(cap_by_cell.get(label,
                                                                                  measured)),
                                                              need))
                            cap_trace.append({
                                "cell": label, "episode": point["episode"],
                                "auto_floor": int(floor_by_cell[label]),
                                "measured_pages": measured, "q_now": q_now,
                                "q_target": target,
                                "cap_now": int(cap_by_cell.get(label, measured)),
                            })
                    anchor_log.extend(anchored)
                    moved = [a for a in anchored
                             if (a["cap_after"] != a["cap_before"] if "cap_after" in a
                                 else abs(a["lambda_after"] - a["lambda_before"]) > 0.0)]
                    print("  dual anchor @%d (%s, %d scene(s)): %d cell(s) assigned, %d moved; "
                          "lambda mean %.3f sd %.3f"
                          % (point["episode"], anchor_mode, len(point["scenes"]),
                             len(anchored), len(moved),
                             sum(lambda_by_cell.values()) / max(1, len(lambda_by_cell)),
                             (sum((v - sum(lambda_by_cell.values())
                                   / max(1, len(lambda_by_cell))) ** 2
                                  for v in lambda_by_cell.values())
                              / max(1, len(lambda_by_cell) - 1)) ** 0.5), flush=True)
                if fired or stopping_reason:
                    break
            if args.episodes and len(episodes_log) >= args.episodes:
                stopping_reason = "reached_episodes"
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
            "lambda_mode": args.lambda_mode,
            "lambda_by_cell_final": ({key: float(value)
                                      for key, value in sorted(lambda_by_cell.items())}
                                     if args.lambda_mode == "per_cell" else None),
            "q_ema_by_cell_final": ({key: float(value)
                                     for key, value in sorted(q_ema_by_cell.items())}
                                    if args.lambda_mode == "per_cell" else None),
            "lambda_step": float(C.LAMBDA_STEP_PER_CELL if args.lambda_mode == "per_cell"
                                 else C.LAMBDA_STEP),
            "lambda_step_effective": (lam_step if args.lambda_mode == "per_cell" else None),
            "dual_mode": args.dual_mode,
            "dual_anchor_mode": anchor_mode,
            "dual_anchor_scenes": (int(eval_scene_count) if anchor_mode != "off" else 0),
            "q_target_fraction_dual": float(unlimited_budget.Q_TARGET_FRACTION_DUAL),
            "q_target_fraction_verdict": float(unlimited_budget.Q_TARGET_FRACTION),
            "dual_anchor_log": anchor_log,
            "send_cap_mode": args.send_cap,
            "freeze_actor": bool(args.freeze_actor),
            "send_cap_step": int(args.send_cap_step),
            "send_floor_lead": int(args.send_floor_lead),
            "send_floor_auto": bool(args.send_floor_auto),
            "floor_by_cell": ({key: int(value) for key, value in sorted(floor_by_cell.items())}
                              if floor_by_cell else None),
            "cap_by_cell_final": ({key: int(value)
                                   for key, value in sorted(cap_by_cell.items())}
                                  if args.send_cap == "per_cell" else None),
            "cap_trace": cap_trace,
            "resumed_lambda": bool(args.resume_lambda),
            "q_ema_alpha": (float(unlimited_budget.Q_EMA_ALPHA)
                            if args.lambda_mode == "per_cell" else None),
            "budget_bytes": int(C.BUDGET_BYTES),
            "scale_features": str(C.SCALE_FEATURES),
            "advantage_scale": args.advantage_scale,
            "entropy_bonus": (float(C.ENTROPY_BONUS) if args.entropy_bonus is None
                              else float(args.entropy_bonus)),
            "action_temperature": float(args.action_temperature),
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
