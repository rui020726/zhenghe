"""P3: feature N-covariance transform and its self-control, learned greedy, <=32 replays.

Four configurations, 8 frozen scenes each = the whole P3 allowance:
  off        @ N=6   -- must reproduce the archived smoke-eval rows exactly (stop-loss 3)
  norm_only  @ N=6   -- the self-control: what the normalisation itself costs in on-time
  norm_only  @ N=3   -- feature stats + cost side only (on-time stays a confounded indicator
  norm_only  @ N=12     only for N=3, where labour is 6; at N=12 it is confounded by roster)

The arm is a learned checkpoint rather than a rule, on purpose. A rule arm's policy never
reads the feature row, so an on-time difference there is 0/192 by construction and the
"drop > 2 tasks" test could not fail even if the normalisation were harmful. The learned
greedy path is the only one where the rewritten dims can actually change a decision.

Nothing here trains: the checkpoint is loaded read-only and evaluated greedily.
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
IDENTITY = STAGE / "recovery_09b" / "revision_01" / "protocol" / "RECOVERY_EXPERIMENT_IDENTITY.json"

#: The archived N=6 reference: same revision, same scenes, same checkpoint, no scale seam.
REFERENCE = ROOT / "execution" / "smoke500" / "eval_control" / "RESULTS.json"
CHECKPOINT = ROOT / "execution" / "smoke500" / "control" / "milestone_5500.pth"

#: (tag, scale mode, N). N=6 keeps the roster at 6; N=3 narrows the uplink only.
CONFIGS = (
    ("off_n6", "off", 6),
    ("norm_n6", "norm_only", 6),
    ("norm_n3", "norm_only", 3),
    ("norm_n12", "norm_only", 12),
    #: scene 2301901 alone, re-run because the batched off_n6 attempt hit an existing
    #: artifact directory for that one scene (the world itself had already completed in the
    #: single-scene probe). Kept separate so no existing evidence is overwritten.
    ("off_n6b", "off", 6),
)

N_CONFIG = {
    3: {"comm_subscribers": "first:3", "rescue_per_species": None,
        "expectation_overrides": {}},
    6: {"comm_subscribers": "all", "rescue_per_species": None,
        "expectation_overrides": {}},
    12: {"comm_subscribers": "all", "rescue_per_species": 4,
         "expectation_overrides": {"rescue_robots": 12}},
}


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


_LOCK = None


def _init_worker(lock):
    global _LOCK
    _LOCK = lock


def ledger_append(lock, record):
    with lock:
        with LEDGER.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def worker(task):
    sys.dont_write_bytecode = True
    lock = _LOCK
    tag, mode, n, scene = task["tag"], task["mode"], int(task["n"]), int(task["scene"])
    t0 = time.time()
    ledger_append(lock, {"ts": utc(), "stage": "P3", "n": n, "mode": mode,
                         "arm": task["tag"], "scene": scene, "phase": "start",
                         "quota": "counted"})
    result = {"tag": tag, "mode": mode, "n": n, "scene": scene}
    try:
        root, run_dir = Path(task["root"]), Path(task["run_dir"])
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from stage5b_comm import constants as C
        from stage5b_comm import episode, trainer, world
        from stage5b_comm.paths import Layout

        cfg = N_CONFIG[n]
        world.SCALE_SUBSCRIBERS = cfg["comm_subscribers"]
        world.SCALE_RESCUE_PER_SPECIES = cfg["rescue_per_species"]
        world.SCALE_EXPECTATION_OVERRIDES = dict(cfg["expectation_overrides"])
        C.SCALE_INVARIANT = mode

        layout = Layout(root)
        layout.execution = run_dir
        layout.world_logs = run_dir / "world_logs"
        layout.rows = run_dir / "rows"
        layout.ledger = run_dir / "ledger"
        layout.receipts = run_dir / "receipts"
        for path in (layout.world_logs, layout.rows, layout.ledger, layout.receipts):
            path.mkdir(parents=True, exist_ok=True)
        # The learned path needs the frozen repo modules on sys.path (cost_math and friends)
        # and the native-import preflight, exactly as eval_smoke_arms.py does.  A rule arm
        # never imports them, which is why P2 did not need this.
        episode.ensure_import_paths(layout)
        episode.native_import_preflight(layout)

        controller, _opt, state = trainer.load_checkpoint(
            Path(task["checkpoint"]), arm="T2", identity_sha256=task["identity_sha"],
            expected_seed=91002, expected_episode=5500)
        from budget_masked import BudgetMaskedCollector

        collector = BudgetMaskedCollector(controller, forced=lambda item: False, mask=False,
                                          greedy=True, budget_mask=True)
        row = world.run_episode(
            layout, label="s5b_p3_%s_s%d" % (tag, scene), kind="eval", scene=scene,
            policy="learned", attach=collector.policy(), controller=controller,
            collector=collector, action_rng_seed=episode.eval_action_seed(scene),
            tag="p3_%s_s%d" % (tag, scene),
            artifact_namespace="eval:%s:none:%d" % (task["ns"], scene))
        result.update({
            "status": "ok",
            "spent_bytes": int(row["spent_bytes"]),
            "emissions": int(row["emissions"]),
            "on_time": int(row["on_time"]),
            "generated_goals": int(row["generated_goals"]),
            "masked_by_credit_total": row.get("masked_by_credit_total"),
            "first_masked_credit_tick": row.get("first_masked_credit_tick"),
            "subscribers_live": row.get("subscribers_live"),
            "scale_invariant": row.get("scale_invariant"),
            "checkpoint_episode": state.get("episode") if isinstance(state, dict) else None,
        })
    except Exception as error:
        result["status"] = "error"
        result["error"] = "%s: %s" % (type(error).__name__, error)
        result["traceback"] = traceback.format_exc()[-3000:]
    result["elapsed_seconds"] = round(time.time() - t0, 2)
    ledger_append(lock, {"ts": utc(), "stage": "P3", "n": n, "mode": mode,
                         "arm": tag, "scene": scene, "phase": "end",
                         "status": result["status"],
                         "elapsed_seconds": result["elapsed_seconds"],
                         "emissions": result.get("emissions"),
                         "spent_bytes": result.get("spent_bytes"),
                         "on_time": result.get("on_time")})
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="off_n6")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--tag", default="", help="suffix for the output dir; keeps old evidence")
    args = ap.parse_args()

    spec = [c for c in CONFIGS if c[0] == args.config]
    if not spec:
        raise SystemExit("unknown config %r; choose from %r"
                         % (args.config, [c[0] for c in CONFIGS]))
    tag, mode, n = spec[0]
    scenes = (tuple(int(x) for x in args.scenes.split(",")) if args.scenes else SCENES)
    run_dir = PROBE / ("P3_%s%s" % (tag, ("_" + args.tag) if args.tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.exists():
        raise SystemExit("checkpoint missing: %s" % CHECKPOINT)
    identity_sha = sha256(IDENTITY)
    ns = tag + (("_" + args.tag) if args.tag else "")
    tasks = [{"root": str(ROOT), "run_dir": str(run_dir), "tag": tag, "ns": ns,
              "mode": mode, "n": n, "scene": scene, "checkpoint": str(CHECKPOINT),
              "identity_sha": identity_sha} for scene in scenes]

    print("P3 %s  mode=%s N=%d  %d replays  workers=%d" % (tag, mode, n, len(tasks),
                                                           args.workers))
    print("  checkpoint:", CHECKPOINT)
    print("  out       :", run_dir)

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
                    result = {"tag": tag, "mode": mode, "n": n, "scene": task["scene"],
                              "status": "executor_error",
                              "error": "%s: %s" % (type(error).__name__, error)}
                results.append(result)
                print("  s%d %-6s on_time=%-3s paid=%-7s emissions=%-4s %.0fs"
                      % (result["scene"], result["status"], result.get("on_time"),
                         result.get("spent_bytes"), result.get("emissions"),
                         result.get("elapsed_seconds") or 0))

    payload = {"schema": "scale-probe-p3-worlds/1", "stage": "P3", "tag": tag,
               "mode": mode, "n": n,
               "identity": "local experiment level, no identity built",
               "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
               "scenes": list(scenes), "replays": len(results),
               "results": sorted(results, key=lambda r: r["scene"])}
    dump(run_dir / "RESULTS.json", payload)
    print("wrote", run_dir / "RESULTS.json")
    print("sha256", sha256(run_dir / "RESULTS.json"))
    bad = [r for r in results if r["status"] != "ok"]
    print("ok %d/%d" % (len(results) - len(bad), len(results)))
    for r in bad:
        print("  FAIL", r["scene"], r.get("error"))


if __name__ == "__main__":
    main()
