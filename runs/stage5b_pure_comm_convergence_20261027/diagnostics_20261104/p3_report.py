"""P3 terminal report: what completed, what did not, and which stop rule fired.

Zero replays. Reads the LEDGER, the partial P3 world trees, the archived N=6 reference rows
and the pin manifest, then writes P3_feature_report.json.

P3 is recorded as a partial/stopped stage, not a pass. The transform seam failed 24
consecutive times after the first eight worlds, which is stop-loss rule 1, and the stage's
attempt count also ran past its own 32-replay allocation. Both facts are reported with the
exact counts rather than smoothed over.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE.parent
ROOT = STAGE / "recovery_12" / "revision_01"
PROBE = ROOT / "execution" / "scale_probe"
REFERENCE = ROOT / "execution" / "smoke500" / "eval_control" / "RESULTS.json"
MANIFEST = STAGE / "recovery_09b" / "revision_01" / "protocol" / "RECOVERY_SOURCE_MANIFEST.json"
SCENES = tuple(range(2301901, 2301909))
BUDGET = 58600

#: Allocation and cap, from the protocol.
QUOTA = {"global_cap": 120, "P0": 24, "P2": 48, "P3": 32}

PIN_CHECK = ("sb_collector.py", "comm_learning.py", "comm.py", "budget_masked.py",
             "v2_run.py", "v2_reward_timing.py", "a1_metrics.py", "joint_driver.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ledger_path = PROBE / "LEDGER.jsonl"
    rows = [json.loads(line) for line in
            ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = [r for r in rows if r.get("phase") == "start"]
    ends = [r for r in rows if r.get("phase") == "end"]

    per_stage = {}
    for stage in ("P2", "P3"):
        s = [r for r in starts if r["stage"] == stage]
        e = [r for r in ends if r["stage"] == stage]
        per_stage[stage] = {
            "attempted": len(s),
            "completed": len(e),
            "ok": sum(1 for r in e if r.get("status") == "ok"),
            "failed": sum(1 for r in e if r.get("status") != "ok"),
            "failures_by_tag": dict(Counter(r.get("arm") for r in e
                                            if r.get("status") != "ok")),
        }
    total_attempted = len(starts)

    # ---- the eight off@N=6 worlds that did complete -------------------------
    reference = {int(w["scene"]): w for w in load(REFERENCE)["worlds"]}
    off = {}
    if (PROBE / "P3_off_n6" / "RESULTS.json").exists():
        for world in load(PROBE / "P3_off_n6" / "RESULTS.json")["results"]:
            if world["status"] == "ok":
                off[int(world["scene"])] = world
    # scene 2301901 completed in the single-scene probe before the batched attempt collided
    # with its own artifact directory; recover its numbers from the world's own archived
    # lifecycle log rather than spending another replay.
    single = PROBE / "P3_off_n6" / "world_logs" / "eval__off_n6__none__2301901" \
        / "s5b_p3_off_n6_s2301901"
    recovered = None
    if single.exists() and 2301901 not in off:
        recovered = {"scene": 2301901, "source": str(single),
                     "note": "world completed in the 1-scene probe; the batched rerun hit "
                             "FileExistsError on this scene's own artifact directory, so no "
                             "second world was run and none of its numbers were re-derived "
                             "here beyond what the transcript already holds"}
    inert_rows, mismatches = {}, []
    for scene, world in sorted(off.items()):
        ref = reference[scene]
        pairs = {"paid_bytes": (int(ref["paid_bytes"]), int(world["spent_bytes"])),
                 "emissions": (int(ref["emissions"]), int(world["emissions"])),
                 "on_time": (int(ref["on_time"]), int(world["on_time"]))}
        bad = [k for k, (a, b) in pairs.items() if a != b]
        inert_rows[str(scene)] = {"reference": {k: v[0] for k, v in pairs.items()},
                                  "off_n6": {k: v[1] for k, v in pairs.items()},
                                  "equal": not bad, "differing": bad}
        mismatches.extend("%s/%s" % (scene, k) for k in bad)

    # ---- pin recheck --------------------------------------------------------
    manifest = load(MANIFEST)
    # The manifest records absolute paths, so hash exactly those bytes rather than guessing
    # a relative location -- a guessed path that does not exist would read as drift.
    by_name = {}
    for entry in manifest.get("files", []):
        by_name[Path(str(entry.get("path"))).name] = entry
    pins, pins_ok = {}, True
    for name in PIN_CHECK:
        entry = by_name.get(name) or {}
        expected = entry.get("sha256")
        path = Path(str(entry.get("path"))) if entry.get("path") else None
        actual = sha256(path) if path is not None and path.exists() else None
        same = expected is not None and actual == expected
        pins_ok = pins_ok and same
        pins[name] = {"manifest_sha256": expected, "actual_sha256": actual,
                      "path": str(path) if path else None, "exists": bool(
                          path is not None and path.exists()), "unchanged": same}

    # Whole-manifest sweep: every one of the pinned files, not only the eight this stage
    # leans on, so the "no frozen file moved" claim covers the entire pin list.
    all_rows, all_drift = {}, []
    for entry in manifest.get("files", []):
        path = Path(str(entry.get("path")))
        expected = entry.get("sha256")
        actual = sha256(path) if path.exists() else None
        if actual != expected:
            all_drift.append(str(path))
        all_rows[path.name] = {"expected": expected, "actual": actual}

    verdict = "not_run"
    report = {
        "schema": "scale-probe-p3-report/1", "stage": "P3", "status": "stopped",
        "verdict": verdict,
        "identity": "local experiment level, no identity built",
        "stop_rules_fired": [
            {"rule": 1,
             "text": "the same seam/import/gate error reproduced 3 or more consecutive times "
                     "without a fix",
             "evidence": "scale_features.rewrite handed a numpy array to the trunk; "
                         "linear() rejected it on %d consecutive worlds across norm_n6, "
                         "norm_n3 and norm_n12"
                         % per_stage["P3"]["failures_by_tag"].get("norm_n6", 0)},
            {"rule": 4,
             "text": "the replay count reaches its cap",
             "evidence": "P3 attempted %d replays against its own 32-slot allocation while the "
                         "global ledger stands at %d/%d"
                         % (per_stage["P3"]["attempted"], total_attempted,
                            QUOTA["global_cap"])},
        ],
        "quota": {
            "allocation": QUOTA,
            "global_attempted": total_attempted,
            "global_remaining": QUOTA["global_cap"] - total_attempted,
            "per_stage": per_stage,
            "note": "failures consume quota and are not refunded; P2 finished exactly at its "
                    "48, and P3 overshot its 32 before the stop",
        },
        "what_completed": {
            "off_n6_worlds_ok": sorted(off),
            "off_n6_count": len(off),
            "off_n6_scene_2301901": recovered,
            "inertness_per_scene": inert_rows,
            "inertness_mismatches": mismatches,
            "inertness_verdict": ("all %d completed off@N=6 worlds reproduce the archived "
                                  "rows exactly on paid_bytes/emissions/on_time"
                                  % len(off)) if not mismatches else "MISMATCH",
        },
        "what_did_not_complete": {
            "norm_n6": "0/8 (all 8 failed on the tensor seam, plus 1 retry)",
            "norm_n3": "0/8 (all 8 failed on the same seam)",
            "norm_n12": "0/8 (all 8 failed on the same seam)",
            "consequence": "no feature-drift distribution, no N=6 drop measurement, no "
                           "N=3/N=12 feature statistics: the three P3 self-controls did not "
                           "produce usable data",
        },
        "implementation_state": {
            "scale_invariant_module": str((ROOT / "stage5b_comm" / "scale_invariant.py")
                                          .relative_to(STAGE)),
            "shadow_feature_module": str((ROOT / "stage5b_comm" / "scale_features.py")
                                         .relative_to(STAGE)),
            "shadow_feature_sha256": sha256(ROOT / "stage5b_comm" / "scale_features.py"),
            "switch": "constants.SCALE_INVARIANT in {'off','norm_only'}, default 'off'",
            "seam": "world.py installs the transform on the controller instance in "
                    "_run_episode_body and restores it on the single finally path, mirroring "
                    "causal_credit.py; no pinned file is edited",
            "off_path_status": "demonstrated inert on %d worlds (rows identical to the "
                               "archive)" % len(off),
            "norm_only_status": "implemented but not demonstrated: the first version returned "
                                "an ndarray and the stage stopped before the corrected build "
                                "could be run under its own allocation",
        },
        "pins": pins, "pins_unchanged": pins_ok,
        "pin_manifest_sweep": {"files_checked": len(all_rows),
                               "files_drifted": all_drift,
                               "all_unchanged": not all_drift},
        "cannot_say": [
            "anything about feature drift or on-time cost of the normalisation at any N",
            "that off is inert on all 8 scenes: 7 were re-measured and 1 is quoted from its "
            "single-scene probe run",
            "anything about N=3 or N=12 features under norm_only",
        ],
    }
    out = PROBE / "P3_feature_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True),
                   encoding="utf-8")

    print("=" * 78)
    print("P3 terminal report")
    print("=" * 78)
    print("quota: global %d/%d attempted, remaining %d"
          % (total_attempted, QUOTA["global_cap"], QUOTA["global_cap"] - total_attempted))
    for stage, block in per_stage.items():
        print("  %s: attempted=%d ok=%d failed=%d  %r"
              % (stage, block["attempted"], block["ok"], block["failed"],
                 block["failures_by_tag"]))
    print()
    print("off@N=6 completed worlds:", sorted(off))
    print("inertness:", report["what_completed"]["inertness_verdict"])
    for mismatch in mismatches:
        print("  MISMATCH", mismatch)
    print("named pins unchanged:", pins_ok,
          "| full manifest: %d files, %d drifted"
          % (len(all_rows), len(all_drift)))
    if not pins_ok:
        for name, row in pins.items():
            if not row["unchanged"]:
                print("  DRIFT", name, row)
    print()
    print("wrote", out)
    print("sha256", sha256(out))


if __name__ == "__main__":
    main()
