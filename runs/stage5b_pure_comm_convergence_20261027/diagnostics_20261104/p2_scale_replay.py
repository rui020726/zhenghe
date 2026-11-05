"""P2: per-N upper/lower bound and cost-side measurement, fixed calibre, rule arms only.

Runs N in {3,6,12} x 8 frozen scenes x {normal_immediate, always_hold} through the
revision's own episode entry point, bypassing runner.py (recovery_12/revision_01/protocol/
is empty, so guard.verify_identity() has nothing to verify and would raise).

How N is moved, and what that costs in cleanliness:
  * N=3 and N=6 narrow/keep the uplink through joint_driver's own registered config field
    comm_subscribers, with rescue_per_species untouched at 2 -- so the roster stays at 6
    robots and the two differ only in fan-out. Neither frozen expectation set names
    comm_subscribers, so the divergence gate sees nothing new.
  * N=12 cannot be reached that way: joint_driver validates every requested subscriber id
    against the real roster and raises on an unknown one. It needs rescue_per_species=4,
    i.e. 12 robots, so rescue_robots diverges and is registered as a second override with
    its reason in world.SCALE_EXPECTATION_OVERRIDE_REASONS. On-time at N=12 is therefore a
    confounded indicator -- labour moves with fan-out -- and is recorded, not concluded on.

Every replay appends to LEDGER.jsonl before it starts, under a cross-process lock, so a
crash still leaves the quota spent on the record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
LEDGER = PROBE / "LEDGER.jsonl"
SCENES = tuple(range(2301901, 2301909))
BUDGET = 58600
FROZEN_RULE_ROWS = (STAGE / ".." / "stage5b_pure_comm_budget58600_20261026"
                    / "revision_02" / "execution" / "rows" / "eval").resolve()

#: N -> (comm_subscribers spec, rescue_per_species or None, registered extra expectations).
#: Only N=12 has to move the roster; see the module docstring for why.
N_CONFIG = {
    3: {"comm_subscribers": "first:3", "rescue_per_species": None,
        "expectation_overrides": {}, "labour_coupled": False},
    6: {"comm_subscribers": "all", "rescue_per_species": None,
        "expectation_overrides": {}, "labour_coupled": False},
    12: {"comm_subscribers": "all", "rescue_per_species": 4,
         "expectation_overrides": {"rescue_robots": 12}, "labour_coupled": True},
}

#: Keys worth keeping from the world row; everything else is bulk tape.
KEEP = ("kind", "label", "scenario_seed", "spent_bytes", "generated_goals",
        "num_goals_placed", "budget_bytes", "b0_bytes", "airtime_tap_total", "emissions",
        "on_time", "completed", "expired", "discovered", "active", "unassigned",
        "task_reward_total", "budget_refused_sends", "capacity_refused_sends",
        "masked_by_credit_total", "masked_by_capacity_total", "first_masked_credit_tick",
        "budget_authority_before", "budget_authority_after_exit",
        "budget_authority_restored", "budget_credit", "right_censored")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def ledger_append(lock, record):
    with lock:
        with LEDGER.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


_LOCK = None


def _init_worker(lock):
    """A spawn pool cannot pickle a Lock into task args, so it arrives by inheritance."""
    global _LOCK
    _LOCK = lock


def worker(task):
    """One world. Sets the revision-local scale seam, runs, and returns the metered row."""
    sys.dont_write_bytecode = True
    lock = _LOCK
    root, run_dir = Path(task["root"]), Path(task["run_dir"])
    n, arm, scene = int(task["n"]), task["arm"], int(task["scene"])
    started = utc()
    t0 = time.time()
    ledger_append(lock, {"ts": started, "stage": "P2", "n": n, "mode": "fixed",
                         "arm": arm, "scene": scene, "phase": "start",
                         "quota": "counted"})
    result = {"n": n, "arm": arm, "scene": scene, "started": started}
    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from stage5b_comm import episode, world
        from stage5b_comm.paths import Layout

        cfg = N_CONFIG[n]
        world.SCALE_SUBSCRIBERS = cfg["comm_subscribers"]
        world.SCALE_RESCUE_PER_SPECIES = cfg["rescue_per_species"]
        world.SCALE_EXPECTATION_OVERRIDES = dict(cfg["expectation_overrides"])

        layout = Layout(root)
        layout.execution = run_dir
        layout.world_logs = run_dir / "world_logs"
        layout.rows = run_dir / "rows"
        layout.ledger = run_dir / "ledger"
        layout.receipts = run_dir / "receipts"
        for path in (layout.world_logs, layout.rows, layout.ledger, layout.receipts):
            path.mkdir(parents=True, exist_ok=True)

        row = episode.run_rule_world(layout, arm, scene, phase="eval")
        result["row"] = {key: row.get(key) for key in KEEP if key in row}
        result["row_keys_not_kept"] = sorted(
            k for k in row if k not in KEEP and not isinstance(row[k], (list, dict)))
        result["all_scalar_keys"] = sorted(
            k for k in row if not isinstance(row[k], (list, dict)))
        result["scale_seam"] = {
            "comm_subscribers": world.SCALE_SUBSCRIBERS,
            "rescue_per_species": world.SCALE_RESCUE_PER_SPECIES,
            "expectation_overrides": world.SCALE_EXPECTATION_OVERRIDES,
            "budget_authority_live": world.last_budget_authority(),
        }
        result["status"] = "ok"
    except Exception as error:  # a failed replay still consumes quota
        result["status"] = "error"
        result["error"] = "%s: %s" % (type(error).__name__, error)
        result["traceback"] = traceback.format_exc()[-3000:]
    result["elapsed_seconds"] = round(time.time() - t0, 2)
    result["finished"] = utc()
    ledger_append(lock, {"ts": result["finished"], "stage": "P2", "n": n, "mode": "fixed",
                         "arm": arm, "scene": scene, "phase": "end",
                         "status": result["status"],
                         "elapsed_seconds": result["elapsed_seconds"],
                         "emissions": (result.get("row") or {}).get("emissions"),
                         "spent_bytes": (result.get("row") or {}).get("spent_bytes"),
                         "on_time": (result.get("row") or {}).get("on_time")})
    return result


def frozen_reference(arm, scene):
    """The frozen N=6 rule row this stage must reproduce when arm/scene match."""
    name = {"normal_immediate": "eval_normal_immediate_none_%d.json",
            "always_hold": "eval_always_hold_none_%d.json"}[arm] % scene
    path = FROZEN_RULE_ROWS / name
    if not path.exists():
        return None, str(path)
    row = json.loads(path.read_text(encoding="utf-8"))
    return row, str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, choices=(3, 6, 12))
    parser.add_argument("--tag", default="p2")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--scenes", default="")
    args = parser.parse_args()

    n = args.n
    scenes = (tuple(int(x) for x in args.scenes.split(",")) if args.scenes else SCENES)
    run_dir = PROBE / ("P2_%d_fixed_%s" % (n, args.tag))
    run_dir.mkdir(parents=True, exist_ok=True)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)

    arms = ("normal_immediate", "always_hold")
    tasks = [{"root": str(ROOT), "run_dir": str(run_dir), "n": n, "arm": arm,
              "scene": scene}
             for arm in arms for scene in scenes]
    print("P2 N=%d  %d replays (%d arms x %d scenes)  workers=%d"
          % (n, len(tasks), len(arms), len(scenes), args.workers))
    print("  seam:", json.dumps(N_CONFIG[n], ensure_ascii=False, sort_keys=True))
    print("  out :", run_dir)

    lock = multiprocessing.Lock()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(lock,)) as pool:
        pending = {pool.submit(worker, task): task for task in tasks}
        while pending:
            done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = {"n": n, "arm": task["arm"], "scene": task["scene"],
                              "status": "executor_error",
                              "error": "%s: %s" % (type(error).__name__, error)}
                results.append(result)
                row = result.get("row") or {}
                print("  %-17s s%d  %-6s  emissions=%-4s spent=%-7s on_time=%-3s  %.0fs"
                      % (result["arm"], result["scene"], result["status"],
                         row.get("emissions"), row.get("spent_bytes"),
                         row.get("on_time"), result.get("elapsed_seconds") or 0))

    dump(run_dir / "RESULTS.json", {
        "schema": "scale-probe-p2-worlds/1", "stage": "P2", "n": n, "mode": "fixed",
        "identity": "local experiment level, no identity built (protocol/ is empty; "
                    "runner.py bypassed, episode.run_rule_world called directly)",
        "seam": N_CONFIG[n],
        "run_dir": str(run_dir),
        "replays": len(results),
        "results": sorted(results, key=lambda r: (r["arm"], r["scene"])),
    })
    print("wrote", run_dir / "RESULTS.json")
    print("sha256", sha256(run_dir / "RESULTS.json"))

    ok = [r for r in results if r["status"] == "ok"]
    print("ok %d/%d" % (len(ok), len(results)))
    for r in results:
        if r["status"] != "ok":
            print("  FAIL", r["arm"], r["scene"], r.get("error"))


if __name__ == "__main__":
    main()
