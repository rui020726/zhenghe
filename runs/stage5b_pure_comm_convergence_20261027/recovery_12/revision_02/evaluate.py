"""Greedy per-cell evaluation of a trained checkpoint, paired with the same-cell baselines.

For every cell in the grid and every DEV scene, this runs one greedy world and reports the
task's own metric set per cell:

    on_time / generated_goals, raw bytes, sends, distinct targets, duplicate sends,
    bytes per on-time task

and then applies the registered improvement test (``>= 95% of normal_immediate's success count
AND strictly fewer bytes``) per cell, with a paired difference and interval against **both**
``normal_immediate`` and ``dedup_immediate``.

What it refuses to do
---------------------
* It never compares across cells. Each cell's verdict uses only that cell's own two baselines.
* It never turns "fewer bytes" into "better" on its own: a cell where bytes fell and success
  fell below the 95 % bar is reported as not improved, in those words.
* It never re-runs a baseline. The baselines come from the JSON the training run used, and
  that file's sha256 is recorded so the pairing is auditable.

Seeds. The learned arm's world for ``(scene, cell)`` uses ``scale_program.eval_seed``, and the
baseline worlds for the same pair used the same function, so the comparison is pair-by-pair
rather than average-to-average.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

BASE = Path(__file__).resolve().parent          # .../recovery_12/revision_02
ROOT = BASE                                     # the revision package root
STAGE = BASE.parents[1]                         # .../stage5b_pure_comm_convergence_20261027
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


def worker(task):
    sys.dont_write_bytecode = True
    scene = int(task["scene"])
    cell = tuple(int(v) for v in task["cell"])
    result = {"scene": scene, "cell": list(cell)}
    started = time.time()
    try:
        root, run_dir = Path(task["root"]), Path(task["run_dir"])
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from stage5b_comm import episode, scale_program, trainer
        from stage5b_comm.paths import Layout

        layout = Layout(root)
        layout.execution = run_dir
        layout.world_logs = run_dir / "world_logs"
        layout.rows = run_dir / "rows"
        layout.ledger = run_dir / "ledger"
        layout.receipts = run_dir / "receipts"
        for path in (layout.world_logs, layout.rows, layout.ledger, layout.receipts):
            path.mkdir(parents=True, exist_ok=True)
        episode.ensure_import_paths(layout)
        episode.native_import_preflight(layout)

        controller, _opt, state = trainer.load_checkpoint(
            Path(task["checkpoint"]), arm="T2", identity_sha256=task["identity_sha"],
            expected_seed=91002, expected_episode=int(task["checkpoint_episode"]))
        row = episode.run_eval_world(layout, "T2", 91002, scene, controller, cell=cell)
        sends = list(row.get("emission_sequence") or [])
        distinct = {str(entry[1]) for entry in sends}
        result.update({
            "status": "ok",
            "on_time": int(row["on_time"]),
            "generated_goals": int(row.get("generated_goals") or 0),
            "completed": int(row.get("completed") or 0),
            "expired": int(row.get("expired") or 0),
            "emissions": int(row["emissions"]),
            "distinct_targets": len(distinct),
            "duplicate_sends": int(row["emissions"]) - len(distinct),
            "spent_bytes": int(row["spent_bytes"]),
            "bytes_per_on_time": row.get("bytes_per_on_time"),
            "on_time_rate": row.get("on_time_rate"),
            "cost_scale": row.get("cost_scale"),
            "masked_by_credit": row.get("masked_by_credit_total"),
            "capacity_refused": row.get("masked_by_capacity_total"),
            "subscribers_live": row.get("subscribers_live"),
            "rescue_robots": row.get("rescue_robots"),
            "cell_label": row.get("cell_label") or scale_program.cell_label(cell),
            "budget_bytes": int(row.get("budget_bytes") or 0),
            "budget_restored": row.get("budget_authority_restored"),
            "scale_mode": (row.get("scale_invariant") or {}).get("mode")
                          or (row.get("scale_features") or {}).get("mode"),
            "checkpoint_episode": state.get("episode") if isinstance(state, dict) else None,
        })
    except Exception as error:
        result["status"] = "error"
        result["error"] = "%s: %s" % (type(error).__name__, error)
        result["traceback"] = traceback.format_exc()[-2500:]
    result["elapsed_seconds"] = round(time.time() - started, 2)
    return result


def sign_test(differences):
    nonzero = [d for d in differences if d != 0]
    n = len(nonzero)
    if n == 0:
        return {"n": 0, "p_two_sided": None}
    positive = sum(1 for d in nonzero if d > 0)
    tail = sum(math.comb(n, k) for k in range(0, min(positive, n - positive) + 1))
    return {"n": n, "positive": positive, "negative": n - positive,
            "p_two_sided": round(min(1.0, 2.0 * tail / (2 ** n)), 6)}


def summarize_pair(learned, baseline):
    """Paired difference over the scenes both arms ran; t-interval when n >= 3."""
    pairs = []
    for entry in learned:
        base = baseline.get(entry["scene"])
        if base is None:
            continue
        pairs.append({
            "scene": entry["scene"],
            "learned_on_time": entry["on_time"], "baseline_on_time": base["on_time"],
            "delta_on_time": entry["on_time"] - base["on_time"],
            "learned_bytes": entry["spent_bytes"], "baseline_bytes": base["spent_bytes"],
            "delta_bytes": entry["spent_bytes"] - base["spent_bytes"],
        })
    if not pairs:
        return {"pairs": [], "mean_delta_on_time": None}
    deltas = [p["delta_on_time"] for p in pairs]
    mean = sum(deltas) / len(deltas)
    interval = None
    if len(deltas) >= 3:
        sd = statistics.stdev(deltas)
        se = sd / math.sqrt(len(deltas))
        # t(0.975, n-1) via a small table; more degrees of freedom fall back to 1.96.
        table = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
        critical = table.get(len(deltas) - 1, 1.96)
        interval = [round(mean - critical * se, 4), round(mean + critical * se, 4)]
    return {
        "pairs": pairs,
        "mean_delta_on_time": round(mean, 4),
        "total_delta_on_time": sum(deltas),
        "total_delta_bytes": sum(p["delta_bytes"] for p in pairs),
        "ci95": interval,
        "sign_test": sign_test(deltas),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="training tag whose checkpoint is scored")
    ap.add_argument("--episode", type=int, required=True, help="checkpoint episode index")
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--cells", default="", help="limit to these cells (semicolon separated)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    from stage5b_comm import scale_program

    checkpoint = ROOT / "execution" / args.tag / "checkpoints" / (
        "milestone_%d.pth" % args.episode)
    if not checkpoint.exists():
        raise SystemExit("checkpoint missing: %s" % checkpoint)
    scenes = (tuple(int(x) for x in args.scenes.split(",")) if args.scenes else SCENES)
    if args.cells:
        cells = [tuple(int(p) for p in chunk.split(",")) for chunk in args.cells.split(";")]
    else:
        cells = list(scale_program.CELLS)
    run_dir = ROOT / "execution" / args.tag / (
        "eval_%d%s" % (args.episode, ("_" + args.out_tag) if args.out_tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    identity_sha = sha256(STAGE / "recovery_09b" / "revision_01" / "protocol"
                          / "RECOVERY_EXPERIMENT_IDENTITY.json")

    baseline_payload = json.loads(Path(args.baselines).read_text(encoding="utf-8"))
    baseline_sha = sha256(args.baselines)

    tasks = [{"root": str(ROOT), "run_dir": str(run_dir), "scene": scene, "cell": cell,
              "checkpoint": str(checkpoint), "checkpoint_episode": args.episode,
              "identity_sha": identity_sha}
             for cell in cells for scene in scenes]
    print("eval: %d cells x %d scenes = %d worlds" % (len(cells), len(scenes), len(tasks)))
    print("checkpoint:", checkpoint)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=__import__("multiprocessing").get_context("spawn")) as pool:
        pending = {pool.submit(worker, task): task for task in tasks}
        done = 0
        while pending:
            finished, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in finished:
                task = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = {"scene": task["scene"], "cell": list(task["cell"]),
                              "status": "executor_error",
                              "error": "%s: %s" % (type(error).__name__, error)}
                results.append(result)
                done += 1
                print("  [%d/%d] %-16s s%d %-6s on_time=%-3s/%s sends=%-3s dup=%-2s bytes=%-7s"
                      % (done, len(tasks),
                         scale_program.cell_label(tuple(task["cell"])), result["scene"],
                         result.get("status"), result.get("on_time"),
                         result.get("generated_goals"), result.get("emissions"),
                         result.get("duplicate_sends"), result.get("spent_bytes")),
                      flush=True)

    ok = [r for r in results if r.get("status") == "ok"]
    per_cell = {}
    for cell in cells:
        label = scale_program.cell_label(cell)
        learned_rows = [r for r in ok if tuple(r["cell"]) == tuple(cell)]
        learned_on_time = sum(r["on_time"] for r in learned_rows)
        learned_generated = sum(r["generated_goals"] for r in learned_rows)
        learned_bytes = sum(r["spent_bytes"] for r in learned_rows)
        learned_emissions = sum(r["emissions"] for r in learned_rows)
        learned_distinct = sum(r["distinct_targets"] for r in learned_rows)
        base_entry = (baseline_payload.get("per_cell") or {}).get(label) or {}
        base_arms = base_entry.get("arms") or {}

        verdicts = {}
        pairings = {}
        for arm in ("normal_immediate", "dedup_immediate", "always_hold"):
            arm_data = base_arms.get(arm) or {}
            base_on_time = arm_data.get("on_time")
            base_bytes = arm_data.get("spent_bytes")
            verdict = None
            if arm == "normal_immediate" and base_on_time is not None:
                verdict = improvement_verdict(
                    cell=label, learned_on_time=learned_on_time,
                    learned_bytes=learned_bytes, immediate_on_time=base_on_time,
                    immediate_bytes=base_bytes)
            verdicts[arm] = verdict or {
                "cell": label, "arm": arm,
                "base_on_time": base_on_time, "base_bytes": base_bytes}
            # Per-scene pairing needs the baseline rows, not just the totals.
            base_rows = {int(r["scene"]): r for r in
                         (baseline_payload.get("results") or [])
                         if r.get("arm") == arm and tuple(r.get("cell") or ()) == tuple(cell)
                         and r.get("status") == "ok"}
            pairings[arm] = summarize_pair(learned_rows, base_rows)

        per_cell[label] = {
            "cell": list(cell),
            "worlds": len(learned_rows),
            "learned": {
                "on_time": learned_on_time,
                "generated_goals": learned_generated,
                "q": (learned_on_time / learned_generated) if learned_generated else None,
                "spent_bytes": learned_bytes,
                "emissions": learned_emissions,
                "distinct_targets": learned_distinct,
                "duplicate_sends": learned_emissions - learned_distinct,
                "bytes_per_on_time": (learned_bytes / learned_on_time
                                      if learned_on_time else None),
            },
            "baselines": {arm: {"on_time": (base_arms.get(arm) or {}).get("on_time"),
                               "q": (base_arms.get(arm) or {}).get("q"),
                               "spent_bytes": (base_arms.get(arm) or {}).get("spent_bytes"),
                               "emissions": (base_arms.get(arm) or {}).get("emissions")}
                          for arm in ("normal_immediate", "dedup_immediate", "always_hold")},
            "verdict_vs_immediate": verdicts["normal_immediate"],
            "paired_vs_immediate": pairings["normal_immediate"],
            "paired_vs_dedup": pairings["dedup_immediate"],
            "paired_vs_hold": pairings["always_hold"],
        }

    failures = [r for r in results if r.get("status") != "ok"]
    payload = {
        "schema": "stage5b-revision02-eval/1",
        "tag": args.tag,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_episode": args.episode,
        "baselines": str(args.baselines), "baselines_sha256": baseline_sha,
        "scenes": list(scenes),
        "cells": [list(cell) for cell in cells],
        "worlds": len(results), "worlds_ok": len(ok), "failures": failures,
        "per_cell": per_cell,
        "results": sorted(results, key=lambda r: (tuple(r["cell"]), r["scene"])),
        "caveat": ("Cells are never compared to each other. R changes execution capability and "
                   "N changes communication cost; a cell's verdict uses only that cell's own "
                   "baselines."),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    target = run_dir / "EVAL.json"
    dump(target, payload)

    print()
    print("%-16s %-10s %-10s %-10s %-10s %s"
          % ("cell", "q_learned", "q_imm", "bytes_learn", "bytes_imm", "improved"))
    improved_cells = 0
    for label, entry in per_cell.items():
        verdict = entry["verdict_vs_immediate"]
        improved = bool(verdict.get("improved"))
        improved_cells += int(improved)
        print("%-16s %-10s %-10s %-10d %-10s %s"
              % (label,
                 "%.4f" % entry["learned"]["q"] if entry["learned"]["q"] is not None else "n/a",
                 "%.4f" % entry["baselines"]["normal_immediate"]["q"]
                 if entry["baselines"]["normal_immediate"]["q"] is not None else "n/a",
                 entry["learned"]["spent_bytes"],
                 entry["baselines"]["normal_immediate"]["spent_bytes"],
                 ("YES" if improved else "no") + "  " + str(verdict.get("reason", ""))))
    print()
    print("cells improved: %d / %d" % (improved_cells, len(per_cell)))
    print("wrote", target)
    print("sha256", sha256(target))
    if failures:
        print("FAILURES:", len(failures))
        for failure in failures[:8]:
            print("  ", failure.get("cell"), failure.get("scene"), failure.get("error"))


def improvement_verdict(*, cell, learned_on_time, learned_bytes, immediate_on_time,
                        immediate_bytes):
    """Local copy of the registered test, so the scorer does not import the trainer package."""
    if learned_on_time is None:
        return {"cell": cell, "improved": False, "reason": "no evaluation point yet"}
    if not immediate_on_time:
        return {"cell": cell, "improved": False, "reason": "no immediate reference"}
    ratio = float(learned_on_time) / float(immediate_on_time)
    fewer = float(learned_bytes) < float(immediate_bytes)
    improved = ratio >= 0.95 and fewer
    if improved:
        reason = "success %.3f of immediate and %.0f B fewer" % (
            ratio, float(immediate_bytes) - float(learned_bytes))
    elif not fewer:
        reason = "bytes not lower (%.0f vs %.0f)" % (learned_bytes, immediate_bytes)
    else:
        reason = "success %.3f of immediate, below 0.95" % ratio
    return {"cell": cell, "improved": improved, "success_ratio_vs_immediate": ratio,
            "bytes_saved": float(immediate_bytes) - float(learned_bytes), "reason": reason}


if __name__ == "__main__":
    main()
