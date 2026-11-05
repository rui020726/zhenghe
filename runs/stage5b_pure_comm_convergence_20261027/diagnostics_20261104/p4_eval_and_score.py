"""P4 evaluation: greedy, 8 DEV_REUSED scenes, learned arm vs the same-caliber strong rule.

The baseline is not re-run.  P2 already measured `normal_immediate` at N=12 with the same
seam, the same 8 scenes, the same 58,600 B and the same eval action seeds; re-running it
would spend worlds to reproduce a number already on disk.  It is read from
`execution/scale_probe/P2_12_fixed_p2/RESULTS.json` and its per-scene values are asserted
against the numbers P2 published, so a drifted source file fails instead of silently
changing the comparison.

What this run adds
------------------
  on-time /192          both arms, per scene and paired
  paired difference     per scene, plus mean and a sign test
  paid bytes            total and per completed task (the cost side of the trade)
  emissions             total, and completed/send conversion
  credit mask           first tick a candidate was masked by credit, and how many

The N=12 caveat is inherited from P2 and restated in the output: the roster is 12 robots, so
on-time differences confound selection with labour.  The honest claim is about the cost and
on-time trade against the strong rule at identical N, not about selection quality alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE.parent
ROOT = STAGE / "recovery_12" / "revision_01"
PROBE = ROOT / "execution" / "scale_probe"
IDENTITY = (STAGE / "recovery_09b" / "revision_01" / "protocol"
            / "RECOVERY_EXPERIMENT_IDENTITY.json")
SCENES = tuple(range(2301901, 2301909))

#: P2's published N=12 `normal_immediate` totals, asserted against the source file.
P2_EXPECT = {"on_time": 90, "emissions": 90, "spent_bytes": 450192, "worlds": 8}

N_CONFIG = {"comm_subscribers": "all", "rescue_per_species": 4,
            "expectation_overrides": {"rescue_robots": 12}}


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
    t0 = time.time()
    result = {"scene": scene}
    try:
        root, run_dir = Path(task["root"]), Path(task["run_dir"])
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from stage5b_comm import constants as C
        from stage5b_comm import episode, trainer, world
        from stage5b_comm.paths import Layout

        world.SCALE_SUBSCRIBERS = N_CONFIG["comm_subscribers"]
        world.SCALE_RESCUE_PER_SPECIES = N_CONFIG["rescue_per_species"]
        world.SCALE_EXPECTATION_OVERRIDES = dict(N_CONFIG["expectation_overrides"])
        C.SCALE_INVARIANT = "norm_only"

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
        row = episode.run_eval_world(layout, "T2", 91002, scene, controller)
        result.update({
            "status": "ok",
            "on_time": int(row["on_time"]),
            "completed": int(row.get("completed", 0)),
            "expired": int(row.get("expired", 0)),
            "generated_goals": int(row.get("generated_goals", 0)),
            "emissions": int(row["emissions"]),
            "spent_bytes": int(row["spent_bytes"]),
            "subscribers_live": row.get("subscribers_live"),
            "masked_by_credit_total": row.get("masked_by_credit_total"),
            "first_masked_credit_tick": row.get("first_masked_credit_tick"),
            "scale_mode": (row.get("scale_invariant") or {}).get("mode"),
            "scale_matrix_calls": (row.get("scale_invariant") or {}).get("matrix_calls"),
            "checkpoint_episode": state.get("episode") if isinstance(state, dict) else None,
        })
    except Exception as error:
        result["status"] = "error"
        result["error"] = "%s: %s" % (type(error).__name__, error)
        result["traceback"] = traceback.format_exc()[-3000:]
    result["elapsed_seconds"] = round(time.time() - t0, 2)
    return result


def load_baseline():
    """P2's N=12 normal_immediate rows, verified against the totals it published."""
    path = PROBE / "P2_12_fixed_p2" / "RESULTS.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for record in payload["results"]:
        if record.get("arm") != "normal_immediate":
            continue
        row = record["row"]
        rows[int(record["scene"])] = {
            "on_time": int(row["on_time"]), "emissions": int(row["emissions"]),
            "spent_bytes": int(row["spent_bytes"]),
            "completed": int(row.get("completed", 0)),
        }
    totals = {key: sum(item[key] for item in rows.values())
              for key in ("on_time", "emissions", "spent_bytes")}
    mismatches = {key: (totals[key], P2_EXPECT[key]) for key in totals
                  if totals[key] != P2_EXPECT[key]}
    if len(rows) != P2_EXPECT["worlds"]:
        mismatches["worlds"] = (len(rows), P2_EXPECT["worlds"])
    return path, sha256(path), rows, totals, mismatches


def sign_test(differences):
    """Two-sided exact sign test on the non-zero paired differences."""
    nonzero = [d for d in differences if d != 0]
    n = len(nonzero)
    if n == 0:
        return {"n": 0, "positive": 0, "negative": 0, "p_two_sided": None,
                "note": "all paired differences are zero"}
    positive = sum(1 for d in nonzero if d > 0)
    tail = sum(math.comb(n, k) for k in range(0, min(positive, n - positive) + 1))
    p = min(1.0, 2.0 * tail / (2 ** n))
    return {"n": n, "positive": positive, "negative": n - positive,
            "p_two_sided": round(p, 6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="training tag whose final checkpoint is scored")
    ap.add_argument("--episode", type=int, required=True, help="checkpoint episode to load")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--out", default="", help="output dir; default execution/<tag>/eval_<episode>")
    args = ap.parse_args()

    checkpoint = ROOT / "execution" / args.tag / "T2" / ("milestone_%d.pth" % args.episode)
    if not checkpoint.exists():
        raise SystemExit("checkpoint missing: %s" % checkpoint)
    scenes = (tuple(int(x) for x in args.scenes.split(",")) if args.scenes else SCENES)
    run_dir = (Path(args.out) if args.out
               else ROOT / "execution" / args.tag / ("eval_%d" % args.episode))
    run_dir.mkdir(parents=True, exist_ok=True)
    identity_sha = sha256(IDENTITY)

    tasks = [{"root": str(ROOT), "run_dir": str(run_dir), "scene": scene,
              "checkpoint": str(checkpoint), "checkpoint_episode": args.episode,
              "identity_sha": identity_sha} for scene in scenes]

    print("P4 eval  %s @ %d  %d scenes  workers=%d"
          % (args.tag, args.episode, len(tasks), args.workers))
    lock = multiprocessing.Lock()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(worker, task): task for task in tasks}
        while pending:
            done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = {"scene": task["scene"], "status": "executor_error",
                              "error": "%s: %s" % (type(error).__name__, error)}
                results.append(result)
                print("  s%d %-6s on_time=%-3s sends=%-4s paid=%-7s masks=%-5s %.0fs"
                      % (result["scene"], result["status"], result.get("on_time"),
                         result.get("emissions"), result.get("spent_bytes"),
                         result.get("masked_by_credit_total"),
                         result.get("elapsed_seconds") or 0))
    del lock

    baseline_path, baseline_sha, baseline, baseline_totals, baseline_mismatch = load_baseline()

    ok = [r for r in results if r["status"] == "ok"]
    paired = []
    for result in sorted(ok, key=lambda r: r["scene"]):
        base = baseline.get(result["scene"])
        if base is None:
            continue
        paired.append({
            "scene": result["scene"],
            "learned_on_time": result["on_time"], "baseline_on_time": base["on_time"],
            "delta_on_time": result["on_time"] - base["on_time"],
            "learned_bytes": result["spent_bytes"], "baseline_bytes": base["spent_bytes"],
            "delta_bytes": result["spent_bytes"] - base["spent_bytes"],
            "learned_emissions": result["emissions"],
            "baseline_emissions": base["emissions"],
        })

    deltas = [row["delta_on_time"] for row in paired]
    learned_on_time = sum(r["on_time"] for r in ok)
    learned_bytes = sum(r["spent_bytes"] for r in ok)
    learned_emissions = sum(r["emissions"] for r in ok)
    completed = sum(r.get("completed", 0) for r in ok)
    denominators_ok = all(int(r.get("generated_goals", 0)) == 24 for r in ok)

    payload = {
        "schema": "stage5b-p4-tight-budget-eval/1",
        "tag": args.tag, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_episode": args.episode,
        "identity": "local experiment level, no identity built",
        "n_subscribers": 12, "budget_bytes": 58600,
        "seam": dict(N_CONFIG), "scale_invariant": "norm_only",
        "scenes": list(scenes), "replays": len(results),
        "learned": {
            "worlds_ok": len(ok),
            "on_time": learned_on_time, "denominator": 24 * len(ok),
            "every_world_generated_24_goals": denominators_ok,
            "spent_bytes": learned_bytes, "emissions": learned_emissions,
            "completed": completed,
            "completed_per_send": (round(completed / learned_emissions, 4)
                                   if learned_emissions else None),
            "bytes_per_completed_task": (round(learned_bytes / completed, 1)
                                         if completed else None),
            "first_masked_credit_ticks": {str(r["scene"]): r.get("first_masked_credit_tick")
                                          for r in ok},
            "masked_by_credit_total": sum(int(r.get("masked_by_credit_total") or 0)
                                          for r in ok),
        },
        "baseline": {
            "arm": "normal_immediate", "n": 12, "source": str(baseline_path),
            "source_sha256": baseline_sha,
            "totals_verified_against_P2": not baseline_mismatch,
            "mismatch": baseline_mismatch,
            "on_time": baseline_totals["on_time"], "denominator": 192,
            "spent_bytes": baseline_totals["spent_bytes"],
            "emissions": baseline_totals["emissions"],
            "completed_per_send": round(sum(r["completed"] for r in baseline.values())
                                        / baseline_totals["emissions"], 4),
            "per_scene": {str(k): v for k, v in sorted(baseline.items())},
        },
        "paired": {
            "mean_delta_on_time": (round(sum(deltas) / len(deltas), 4) if deltas else None),
            "total_delta_on_time": sum(deltas) if deltas else None,
            "total_delta_bytes": sum(row["delta_bytes"] for row in paired) if paired else None,
            "per_scene": paired,
            "sign_test": sign_test(deltas),
        },
        "caveat": ("N=12 moves the roster to 12 robots (rescue_per_species=4), so on-time "
                   "differences confound target selection with execution labour. P2 already "
                   "records this. The licensed claim is about the cost/on-time trade against "
                   "normal_immediate at identical N, not about selection quality in isolation."),
        "results": sorted(results, key=lambda r: r["scene"]),
    }
    dump(run_dir / "EVAL.json", payload)

    print()
    print("learned  on_time %3d/%-3d  sends %3d  paid %7d B  completed %3d  %.4f per send"
          % (learned_on_time, 24 * len(ok), learned_emissions, learned_bytes, completed,
             (completed / learned_emissions) if learned_emissions else 0.0))
    print("baseline on_time %3d/%-3d  sends %3d  paid %7d B  (normal_immediate, P2 @N=12)"
          % (baseline_totals["on_time"], 192, baseline_totals["emissions"],
             baseline_totals["spent_bytes"]))
    print("paired   mean delta %+.4f /scene  total %+d tasks  %+d B"
          % (sum(deltas) / len(deltas) if deltas else 0.0, sum(deltas) if deltas else 0,
             sum(row["delta_bytes"] for row in paired) if paired else 0))
    print("sign test", json.dumps(sign_test(deltas)))
    if baseline_mismatch:
        print("BASELINE MISMATCH", json.dumps(baseline_mismatch))
    bad = [r for r in results if r["status"] != "ok"]
    for r in bad:
        print("  FAIL", r["scene"], r.get("error"))
    print("wrote", run_dir / "EVAL.json")
    print("sha256", sha256(run_dir / "EVAL.json"))


if __name__ == "__main__":
    main()
