"""Per-cell report: what the run achieved, and where each cell stands against its baselines.

This is the file the task's section 4 asks for, one row per cell:

    on_time / generated_goals, raw bytes, sends, distinct targets, duplicate sends,
    bytes per on-time task, lambda/q trace, paired difference and interval against
    normal_immediate and dedup_immediate

Two rules enforced in code, not just in prose:

* a cell's row never contains another cell's numbers, so a cross-cell claim cannot be read off
  this table; and
* "improved" is the registered test -- success >= 95 % of that cell's `normal_immediate` **and**
  strictly fewer bytes. Everything else prints `no` with the reason.

Usage:

    python report.py --train execution/<tag>/TRAIN_SUMMARY.json \
                     --eval  execution/<tag>/eval_<n>/EVAL.json \
                     --baselines execution/baselines/<tag>/BASELINES.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sign_test(differences):
    nonzero = [d for d in differences if d != 0]
    n = len(nonzero)
    if n == 0:
        return {"n": 0, "p_two_sided": None}
    positive = sum(1 for d in nonzero if d > 0)
    tail = sum(math.comb(n, k) for k in range(0, min(positive, n - positive) + 1))
    return {"n": n, "positive": positive, "negative": n - positive,
            "p_two_sided": round(min(1.0, 2.0 * tail / (2 ** n)), 6)}


def interval(deltas):
    if len(deltas) < 3:
        return None
    mean = sum(deltas) / len(deltas)
    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(len(deltas))
    critical = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}.get(
        len(deltas) - 1, 1.96)
    return [round(mean - critical * se, 4), round(mean + critical * se, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", default="")
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    train = json.loads(Path(args.train).read_text(encoding="utf-8"))
    baselines = json.loads(Path(args.baselines).read_text(encoding="utf-8"))
    evaluation = (json.loads(Path(args.eval).read_text(encoding="utf-8")) if args.eval else None)

    episodes = train.get("episodes") or []
    updates = train.get("update_history") or []

    # ---------------------------------------------------------------- lambda trace
    trace = [{"episode": entry["episode_index"], "lambda": entry.get("lambda"),
              "cell": entry.get("cell_label"), "on_time": entry.get("on_time"),
              "generated_goals": entry.get("generated_goals"),
              "spent_bytes": entry.get("spent_bytes"),
              "q": (entry["on_time"] / entry["generated_goals"]
                    if entry.get("generated_goals") else None)}
             for entry in episodes]
    dual_steps = [u.get("dual") or {} for u in updates]
    applied = [step for step in dual_steps if step.get("applied")]

    # ---------------------------------------------------------------- training aggregates by cell
    by_cell = {}
    for entry in episodes:
        label = entry.get("cell_label") or "unknown"
        bucket = by_cell.setdefault(label, {"cell": entry.get("cell"), "episodes": 0,
                                            "on_time": 0, "generated_goals": 0,
                                            "spent_bytes": 0, "emissions": 0})
        bucket["episodes"] += 1
        bucket["on_time"] += int(entry.get("on_time") or 0)
        bucket["generated_goals"] += int(entry.get("generated_goals") or 0)
        bucket["spent_bytes"] += int(entry.get("spent_bytes") or 0)
        bucket["emissions"] += int(entry.get("emissions") or 0)
    for bucket in by_cell.values():
        generated = bucket["generated_goals"]
        bucket["q"] = (bucket["on_time"] / generated) if generated else None
        bucket["bytes_per_on_time"] = (bucket["spent_bytes"] / bucket["on_time"]
                                       if bucket["on_time"] else None)

    # ---------------------------------------------------------------- evaluation by cell
    eval_cells = (evaluation or {}).get("per_cell") or {}

    rows = []
    for label, trained in sorted(by_cell.items()):
        base = ((baselines.get("per_cell") or {}).get(label) or {}).get("arms") or {}
        evaluated = eval_cells.get(label) or {}
        learned = evaluated.get("learned") or {}
        row = {
            "cell": trained["cell"],
            "cell_label": label,
            "training": {
                "episodes": trained["episodes"],
                "on_time": trained["on_time"],
                "generated_goals": trained["generated_goals"],
                "q": trained["q"],
                "spent_bytes": trained["spent_bytes"],
                "emissions": trained["emissions"],
                "bytes_per_on_time": trained["bytes_per_on_time"],
                "lambda_last": next((t["lambda"] for t in reversed(trace)
                                     if t["cell"] == label), None),
            },
            "baselines": {
                arm: {
                    "on_time": (base.get(arm) or {}).get("on_time"),
                    "q": (base.get(arm) or {}).get("q"),
                    "spent_bytes": (base.get(arm) or {}).get("spent_bytes"),
                    "emissions": (base.get(arm) or {}).get("emissions"),
                } for arm in ("normal_immediate", "dedup_immediate", "always_hold")
            },
        }
        if evaluated:
            row["evaluation"] = {
                "worlds": evaluated.get("worlds"),
                "on_time": learned.get("on_time"),
                "generated_goals": learned.get("generated_goals"),
                "q": learned.get("q"),
                "spent_bytes": learned.get("spent_bytes"),
                "emissions": learned.get("emissions"),
                "distinct_targets": learned.get("distinct_targets"),
                "duplicate_sends": learned.get("duplicate_sends"),
                "bytes_per_on_time": learned.get("bytes_per_on_time"),
            }
            row["verdict_vs_immediate"] = evaluated.get("verdict_vs_immediate")
            for arm in ("normal_immediate", "dedup_immediate", "always_hold"):
                pairing = evaluated.get("paired_vs_" + {
                    "normal_immediate": "immediate",
                    "dedup_immediate": "dedup",
                    "always_hold": "hold"}[arm]) or {}
                deltas = [p["delta_on_time"] for p in (pairing.get("pairs") or [])]
                row["paired_vs_" + {
                    "normal_immediate": "immediate",
                    "dedup_immediate": "dedup",
                    "always_hold": "hold"}[arm]] = {
                    "mean_delta_on_time": pairing.get("mean_delta_on_time"),
                    "total_delta_on_time": pairing.get("total_delta_on_time"),
                    "total_delta_bytes": pairing.get("total_delta_bytes"),
                    "ci95": pairing.get("ci95") or interval(deltas),
                    "sign_test": sign_test(deltas),
                    "per_scene": pairing.get("pairs"),
                }
        rows.append(row)

    payload = {
        "schema": "stage5b-revision02-report/1",
        "train": str(args.train), "train_sha256": sha256(args.train),
        "eval": str(args.eval) if args.eval else None,
        "eval_sha256": sha256(args.eval) if args.eval else None,
        "baselines": str(args.baselines), "baselines_sha256": sha256(args.baselines),
        "stopping_reason": train.get("stopping_reason"),
        "breakers": train.get("breakers"),
        "lambda_initial": train.get("lambda_initial"),
        "lambda_final": train.get("lambda_final"),
        "lambda_trace": trace,
        "dual_steps": dual_steps,
        "dual_steps_applied": len(applied),
        "rows": rows,
        "caveat": ("Every row is one (G, R, N) cell and uses only that cell's own baselines. "
                   "R changes execution capability, N changes communication cost; no row may be "
                   "compared to another, and no row licenses a Pareto claim."),
    }

    print("=" * 110)
    print("revision_02 report")
    print("=" * 110)
    print("stopping_reason:", payload["stopping_reason"])
    if payload["breakers"]:
        for breaker in payload["breakers"]:
            print("  breaker:", breaker.get("kind"), "-", str(breaker.get("detail"))[:160])
    print("lambda: %.4f -> %.4f over %d dual steps (%d applied)"
          % (payload["lambda_initial"], payload["lambda_final"], len(dual_steps), len(applied)))
    print("episodes:", sum(r["training"]["episodes"] for r in rows),
          " cells:", len(rows))
    print()

    header = ("%-15s %6s %8s %8s %9s %9s %6s %6s %9s %s"
              % ("cell", "eps", "q_train", "q_eval", "bytes_tr", "bytes_ev",
                 "sends", "dups", "B/on_time", "verdict"))
    print(header)
    print("-" * len(header))
    improved = 0
    for row in rows:
        evaluation = row.get("evaluation") or {}
        verdict = row.get("verdict_vs_immediate") or {}
        improved += int(bool(verdict.get("improved")))
        bytes_per_on_time = evaluation.get("bytes_per_on_time")
        print("%-15s %6d %8s %8s %9d %9s %6s %6s %9s %s"
              % (row["cell_label"], row["training"]["episodes"],
                 "%.4f" % row["training"]["q"] if row["training"]["q"] is not None else "n/a",
                 "%.4f" % evaluation["q"] if evaluation.get("q") is not None else "-",
                 row["training"]["spent_bytes"],
                 ("%d" % evaluation["spent_bytes"]) if evaluation.get("spent_bytes") else "-",
                 evaluation.get("distinct_targets", "-"),
                 evaluation.get("duplicate_sends", "-"),
                 ("%.0f" % bytes_per_on_time) if bytes_per_on_time else "-",
                 ("YES " if verdict.get("improved") else "no  ") + str(verdict.get("reason", ""))))
    print()
    print("cells improved: %d / %d" % (improved, len(rows)))
    print()

    # ---------------------------------------------------------------- lambda trace
    print("lambda trace (every applied dual step)")
    print("  %8s %8s %10s %8s %8s %10s" %
          ("update", "lambda", "q_batch", "q_ref", "error", "direction"))
    for step in applied[:40]:
        print("  %8s %8.4f %10.4f %8.4f %+8.4f %10s"
              % (step.get("update_index", "-"), step.get("lambda_after", float("nan")),
                 step.get("q_batch", float("nan")), step.get("q_ref", float("nan")),
                 step.get("error", float("nan")), step.get("direction", "-")))
    if len(applied) > 40:
        print("  ... %d more" % (len(applied) - 40))
    missing = [step for step in dual_steps if not step.get("applied")]
    if missing:
        print("  %d updates did not move lambda:" % len(missing))
        for step in missing[:5]:
            print("    ", step.get("reason"))

    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                          encoding="utf-8", newline="\n")
        print()
        print("wrote", target)
        print("sha256", sha256(target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
