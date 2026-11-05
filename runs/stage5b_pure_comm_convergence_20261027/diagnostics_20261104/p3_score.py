"""P3 scoring: the off/norm_only self-control, plus the pin-hash recheck.

Reads the four P3 result trees and the archived N=6 learned-eval rows. Runs no world.

Three verdicts come out of here:
  A. inertness  -- off @ N=6 must equal the archived rows on paid/emissions/on_time, exactly.
                   A mismatch is stop-loss rule 3 and voids P3 wholesale.
  B. drift      -- what rewriting #14 and #17 actually did to the input distribution and to
                   the on-time count at N=6. This is the one place the drop test can bite.
  C. pins       -- every hash-pinned source file the probe leans on still matches the
                   manifest, so none of the above was bought with an edit to a frozen file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE.parent
ROOT = STAGE / "recovery_12" / "revision_01"
PROBE = ROOT / "execution" / "scale_probe"
REFERENCE = ROOT / "execution" / "smoke500" / "eval_control" / "RESULTS.json"
SCENES = tuple(range(2301901, 2301909))
BUDGET = 58600
DROP_LIMIT = 2

#: The pin list the recovered identity was built from, and the files this stage touches.
MANIFEST = STAGE / "recovery_09b" / "revision_01" / "protocol" / "RECOVERY_SOURCE_MANIFEST.json"
PIN_CHECK = ("sb_collector.py", "comm_learning.py", "comm.py", "budget_masked.py",
             "v2_run.py", "v2_reward_timing.py", "a1_metrics.py", "joint_driver.py")

#: The norm_only cells come from the second run (tag v2): the first attempt died on a
#: numpy-vs-torch seam and produced no usable world. The off cell is split because scene
#: 2301901 completed in a single-scene probe before the batched run collided with its
#: own artifact directory.
CONFIGS = ("off_n6", "norm_n6_v2", "norm_n3_v2", "norm_n12_v2")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def worlds(tag):
    payload = load(PROBE / ("P3_%s" % tag) / "RESULTS.json")
    return {int(w["scene"]): w for w in payload["results"] if w["status"] == "ok"}, payload


def dim_of(world, name):
    audit = world.get("scale_invariant") or {}
    return (audit.get("dims") or {}).get(name) or {}


def recover_from_logs(namespace_dir: str, label: str, tag: str, mode: str, n: int,
                      scene: int) -> dict | None:
    """Rebuild one world from its archived transcript + lifecycle log (zero replays).

    Used only where a finished world's row was never written or was later overwritten:
    the world itself ran, its two event logs are on disk, and both on-time and spend are
    re-derivable from them. Nothing is re-simulated.
    """
    import gzip
    base = PROBE / namespace_dir / "world_logs" / label
    transcript, lifecycle = base / "relay_transcript.jsonl.gz", base / "lifecycle.jsonl.gz"
    if not (transcript.exists() and lifecycle.exists()):
        return None
    sends = spend = 0
    with gzip.open(transcript, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("kind") == "send":
                sends += 1
                spend += int(record.get("cost_bytes") or 0)
    completed = 0
    with gzip.open(lifecycle, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event_type") == "final" and record.get("status") == "completed":
                completed += 1
    return {
        "tag": tag, "mode": mode, "n": n, "scene": scene, "status": "ok",
        "spent_bytes": spend, "emissions": sends, "on_time": completed,
        "generated_goals": 24, "subscribers_live": n,
        "recovered_from_logs": True, "source": str(base),
        "note": "spend/emissions from the relay transcript, on_time from lifecycle "
                "final/status=completed; no replay spent",
    }


def main() -> None:
    loaded, payloads = {}, {}
    for tag in CONFIGS:
        loaded[tag], payloads[tag] = worlds(tag)
    # Two worlds ran but their rows were never persisted / were overwritten by a later
    # batched run: scene 2301901 in off_n6 (artifact-dir collision) and in norm_n6_v2
    # (single-scene run then overwritten). Both are recovered from their own logs.
    for tag, ns, label, mode, n, run_label in (
            ("off_n6", "P3_off_n6", "eval__off_n6__none__2301901",
             "off", 6, "s5b_p3_off_n6_s2301901"),
            ("norm_n6_v2", "P3_norm_n6_v2", "eval__norm_n6_v2__none__2301901",
             "norm_only", 6, "s5b_p3_norm_n6_s2301901")):
        if 2301901 not in loaded[tag]:
            recovered = recover_from_logs(ns, label + "/" + run_label,
                                         tag, mode, n, 2301901)
            if recovered is not None:
                loaded[tag][2301901] = recovered
    missing = {tag: [s for s in SCENES if s not in loaded[tag]] for tag in CONFIGS}
    if any(missing.values()):
        raise SystemExit("missing P3 worlds: %r" % (missing,))

    reference = {int(w["scene"]): w for w in load(REFERENCE)["worlds"]}
    off_n6 = loaded["off_n6"]
    norm_n6 = loaded["norm_n6_v2"]
    checks = []

    print("=" * 78)
    print("P3 acceptance  (learned greedy, 32 replays, 4 configs x 8 scenes)")
    print("=" * 78)

    # ---- A. inertness: off @ N=6 must equal the archived rows ------------------
    print("--- A. off @ N=6 vs archived smoke500/eval_control rows ---")
    mismatches, rowA = [], {}
    for scene in SCENES:
        ref, mine = reference[scene], off_n6[scene]
        pairs = {"paid_bytes": (int(ref["paid_bytes"]), int(mine["spent_bytes"])),
                 "emissions": (int(ref["emissions"]), int(mine["emissions"])),
                 "on_time": (int(ref["on_time"]), int(mine["on_time"]))}
        bad = [k for k, (a, b) in pairs.items() if a != b]
        rowA[str(scene)] = {"reference": {k: v[0] for k, v in pairs.items()},
                            "off_n6": {k: v[1] for k, v in pairs.items()},
                            "equal": not bad, "differing": bad}
        for key in bad:
            mismatches.append("%s/%s" % (scene, key))
        print("  s%d  ref paid=%-6d emit=%-3d on_time=%-3d | off paid=%-6d emit=%-3d "
              "on_time=%-3d  %s"
              % (scene, pairs["paid_bytes"][0], pairs["emissions"][0],
                 pairs["on_time"][0], pairs["paid_bytes"][1], pairs["emissions"][1],
                 pairs["on_time"][1], "OK" if not bad else "DIFF %r" % (bad,)))
    inert = not mismatches
    checks.append({"check": "A_off_N6_equals_archived_rows", "pass": inert,
                   "mismatches": mismatches,
                   "cells": "%d scenes x 3 quantities" % len(SCENES),
                   "note": "a mismatch is stop-loss rule 3 and voids P3"})
    print("  => %s" % ("identical on all 24 quantities" if inert
                       else "MISMATCH %r -- stop-loss rule 3" % (mismatches,)))
    print()

    # ---- B. drift at N=6 -------------------------------------------------------
    print("--- B. off vs norm_only at N=6: what the rewrite did ---")
    off_on = sum(off_n6[s]["on_time"] for s in SCENES)
    norm_on = sum(norm_n6[s]["on_time"] for s in SCENES)
    off_paid = sum(off_n6[s]["spent_bytes"] for s in SCENES)
    norm_paid = sum(norm_n6[s]["spent_bytes"] for s in SCENES)
    off_emit = sum(off_n6[s]["emissions"] for s in SCENES)
    norm_emit = sum(norm_n6[s]["emissions"] for s in SCENES)
    print("  on_time    off=%d/192  norm_only=%d/192  delta=%+d"
          % (off_on, norm_on, norm_on - off_on))
    print("  paid_bytes off=%d  norm_only=%d  delta=%+d" % (off_paid, norm_paid,
                                                             norm_paid - off_paid))
    print("  emissions  off=%d  norm_only=%d  delta=%+d" % (off_emit, norm_emit,
                                                            norm_emit - off_emit))
    per_scene = {}
    for name in ("queue_backlog_targets(#14)", "subscribers(#17)"):
        agg_off = {"count": 0, "min": None, "max": None, "mean": 0.0, "sd": 0.0}
        agg_norm = {"count": 0, "min": None, "max": None, "mean": 0.0, "sd": 0.0}
        hist, changed = {}, 0
        for scene in SCENES:
            o = dim_of(off_n6[scene], name).get("original") or {}
            rw = dim_of(norm_n6[scene], name)
            new = rw.get("rewritten") or {}
            changed += int(rw.get("values_changed") or 0)
            for key, target, src in (("off", agg_off, o), ("norm", agg_norm, new)):
                if not src.get("count"):
                    continue
                # pooled mean/sd from the per-world moments
                n0, m0 = target["count"], target["mean"]
                n1, m1 = src["count"], src["mean"]
                s1 = float(src["sd"])
                total = n0 + n1
                mean = (m0 * n0 + m1 * n1) / total
                var = ((n0 * (target["sd"] ** 2 + (m0 - mean) ** 2))
                       + (n1 * (s1 ** 2 + (m1 - mean) ** 2))) / total
                target.update({"count": total, "mean": mean, "sd": var ** 0.5,
                               "min": src["min"] if target["min"] is None
                               else min(target["min"], src["min"]),
                               "max": src["max"] if target["max"] is None
                               else max(target["max"], src["max"])})
            for k, v in (rw.get("diff_histogram") or {}).items():
                hist[float(k)] = hist.get(float(k), 0) + int(v)
        per_scene[name] = {"off": agg_off, "norm_only": agg_norm,
                           "values_changed": changed,
                           "diff_histogram_top": {str(k): v for k, v in
                                                  sorted(hist.items(),
                                                         key=lambda kv: -kv[1])[:6]}}
        print("  %s" % name)
        print("    off        count=%-6d min=%-8.3f max=%-8.3f mean=%-9.4f sd=%.4f"
              % (agg_off["count"], agg_off["min"] or 0, agg_off["max"] or 0,
                 agg_off["mean"], agg_off["sd"]))
        print("    norm_only  count=%-6d min=%-8.3f max=%-8.3f mean=%-9.4f sd=%.4f"
              % (agg_norm["count"], agg_norm["min"] or 0, agg_norm["max"] or 0,
                 agg_norm["mean"], agg_norm["sd"]))
        print("    values changed by the rewrite: %d" % changed)
        print("    top difference buckets: %r"
              % (sorted(hist.items(), key=lambda kv: -kv[1])[:6],))
    delta = norm_on - off_on
    drop_ok = delta >= -DROP_LIMIT
    checks.append({"check": "B_norm_only_drop_at_N6_within_limit", "pass": drop_ok,
                   "on_time_off": off_on, "on_time_norm": norm_on, "delta": delta,
                   "limit": -DROP_LIMIT, "denominator": 192,
                   "note": "the protocol voids the normalisation if it costs more than "
                           "%d tasks at N=6" % DROP_LIMIT})
    checks.append({"check": "B_dim17_is_identity_at_N6", "pass":
                   abs((per_scene["subscribers(#17)"]["norm_only"]["mean"] or 0) - 1.0) < 1e-6,
                   "norm_mean": per_scene["subscribers(#17)"]["norm_only"]["mean"],
                   "note": "#17 holds 6.0 at N=6, so n/6 must land on exactly 1.0"})
    checks.append({"check": "B_dim14_is_not_identity_at_N6", "pass":
                   per_scene["queue_backlog_targets(#14)"]["values_changed"] > 0,
                   "values_changed": per_scene["queue_backlog_targets(#14)"]["values_changed"],
                   "note": "backlog/24 must actually move at N=6; if it did not, the "
                           "transform never ran and B is vacuous"})
    print()

    # ---- C. N=3 / N=12: feature stats and cost side only ----------------------
    print("--- C. norm_only at N=3 / N=12: features and cost side ---")
    other = {}
    for tag, n in (("norm_n3_v2", 3), ("norm_n12_v2", 12)):
        cells = loaded[tag]
        paid = sum(cells[s]["spent_bytes"] for s in SCENES)
        emit = sum(cells[s]["emissions"] for s in SCENES)
        on_time = sum(cells[s]["on_time"] for s in SCENES)
        d14 = dim_of(cells[SCENES[0]], "queue_backlog_targets(#14)")
        d17 = dim_of(cells[SCENES[0]], "subscribers(#17)")
        other[tag] = {
            "n": n, "spent_bytes": paid, "emissions": emit,
            "cost_per_send": (paid / emit) if emit else None,
            "budget_share": paid / (BUDGET * len(SCENES)),
            "on_time": on_time, "on_time_denominator": 24 * len(SCENES),
            "dim17_original": (d17.get("original") or {}).get("mean"),
            "dim17_rewritten": (d17.get("rewritten") or {}).get("mean"),
            "dim14_original": d14.get("original"),
            "dim14_rewritten": d14.get("rewritten"),
            "subscribers_live": cells[SCENES[0]].get("subscribers_live"),
        }
        print("  N=%-2d paid=%-6d emissions=%-4d cost/send=%.2f B  budget_share=%.3f  "
              "on_time=%d/%d (recorded only)"
              % (n, paid, emit, other[tag]["cost_per_send"] or 0,
                 other[tag]["budget_share"], on_time, 24 * len(SCENES)))
        print("      #17 original mean=%r -> rewritten mean=%r  (n/6)"
              % (other[tag]["dim17_original"], other[tag]["dim17_rewritten"]))
        print("      #14 original mean=%r -> rewritten mean=%r  (backlog/24)"
              % ((other[tag]["dim14_original"] or {}).get("mean"),
                 (other[tag]["dim14_rewritten"] or {}).get("mean")))
    checks.append({"check": "C_N3_and_N12_report_only", "pass": True,
                   "note": "P3 asks for feature statistics and cost-side metrics at these N; "
                           "on-time is recorded but is only a clean indicator at N=3 "
                           "(labour held at 6) and is confounded at N=12 (roster is 12)"})
    print()

    # ---- D. pin hashes --------------------------------------------------------
    print("--- D. pin recheck ---")
    # The manifest records absolute paths; hash exactly those bytes. Guessing a relative
    # location would read as drift the moment the guess is wrong.
    manifest = load(MANIFEST) if MANIFEST.exists() else None
    pin_rows, pin_ok = {}, True
    if manifest is None:
        print("  manifest not found at", MANIFEST); pin_ok = False
    else:
        by_name = {Path(str(e.get("path"))).name: e for e in manifest.get("files", [])}
        for name in PIN_CHECK:
            entry = by_name.get(name) or {}
            expected = entry.get("sha256")
            path = Path(str(entry["path"])) if entry.get("path") else None
            actual = sha256(path) if path is not None and path.exists() else None
            same = expected is not None and actual == expected
            pin_ok = pin_ok and same
            pin_rows[name] = {"manifest_sha256": expected, "actual_sha256": actual,
                              "path": str(path) if path else None, "unchanged": same}
            print("  %-22s %s" % (name, "unchanged" if same else "MISMATCH"))
        all_rows, drift = {}, []
        for entry in manifest.get("files", []):
            path = Path(str(entry.get("path")))
            actual = sha256(path) if path.exists() else None
            if actual != entry.get("sha256"):
                drift.append(str(path))
            all_rows[path.name] = {"expected": entry.get("sha256"), "actual": actual}
        print("  full manifest: %d files, %d drifted" % (len(all_rows), len(drift)))
    checks.append({"check": "D_pin_files_unchanged", "pass": pin_ok, "files": pin_rows,
                   "manifest_sweep": {"files_checked": len(all_rows) if manifest else 0,
                                      "files_drifted": drift if manifest else None},
                   "note": "the rewrite lives in revision-local scale_features.py plus an "
                           "instance-level forward shadow, so no pinned file moved"})
    print()

    passed = sum(1 for c in checks if c["pass"])
    verdict = "pass" if passed == len(checks) else "fail"
    print("=" * 78)
    print("P3 verdict: %s  (%d/%d checks pass)" % (verdict, passed, len(checks)))
    for c in checks:
        if not c["pass"]:
            print("  FAIL", c["check"], json.dumps(c, ensure_ascii=False)[:400])
    print("=" * 78)

    report = {
        "schema": "scale-probe-p3-report/1", "stage": "P3",
        "identity": "local experiment level, no identity built",
        "reference": {"path": str(REFERENCE), "sha256": sha256(REFERENCE)},
        "configs": {tag: {"results_sha256": sha256(PROBE / ("P3_%s" % tag) / "RESULTS.json"),
                          "worlds": len(loaded[tag])} for tag in CONFIGS},
        "inertness_per_scene": rowA,
        "drift_at_n6": {"on_time_off": off_on, "on_time_norm": norm_on,
                        "on_time_delta": delta, "denominator": 192,
                        "paid_bytes_off": off_paid, "paid_bytes_norm": norm_paid,
                        "emissions_off": off_emit, "emissions_norm": norm_emit,
                        "per_dim": per_scene},
        "other_n": other,
        "pins": pin_rows,
        "checks": checks, "checks_passed": passed, "checks_total": len(checks),
        "verdict": verdict,
        "cannot_say": [
            "that the 19-dim distribution is aligned across N: only two dims were rewritten, "
            "and #10 revision_delta / #14 in its raw form / #2 sent_count still carry "
            "N-dependent scale by construction",
            "that normalisation preserves on-time: the N=6 delta above is one checkpoint on "
            "8 development scenes, not a population estimate",
        ],
    }
    out = PROBE / "P3_feature_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True),
                   encoding="utf-8")
    print("wrote", out)
    print("sha256", sha256(out))


if __name__ == "__main__":
    main()
