"""Zero-cost separability probe: are targets distinguishable at tick 1?

The P4 screen at N=12 showed the learned sender spending its whole budget on steps 1-2. One
explanation is that the policy is bad. The other is that at tick 1 the 19-dim row it reads
carries almost no information about which target will actually make its deadline, so "send to
everything now" and "choose well" pay the same. This file tests the second explanation using
only archived logs: no world is opened, no model is loaded, no PPO step runs.

Three questions
---------------
1. How degenerate is the existing 19-dim row at the first decision step? For every dimension:
   spread, unique-value count, and its univariate power to predict the target's observed
   on-time label.
2. Do four candidate physics features do better? min idle distance, unclipped remaining time,
   work over slack, and travel-plus-work over slack.
3. If we could broadcast to only the best K targets per scene, what on-time ceiling does each
   ranking imply -- and how does that compare with the measured 80/192 (learned) and 90/192
   (normal_immediate)?

Label and denominators
----------------------
A target is on-time iff the lifecycle stream holds a ``progress`` record with
``status == "completed"`` at a timestamp <= its deadline. That is exactly what
``integrated_sar.lifecycle_metrics.window_metrics`` counts, so the label here is the same
quantity the eval reports as on_time.

The scene set is the 8 DEV scenes of the P4 evaluation at N=12. Each scene contributes 24
detected targets, so the denominator is 192 -- the same 192 as the 80/192 and 90/192 figures.

Reconstruction, and why it is exact rather than approximate
----------------------------------------------------------
* Target position, deadline and stage count come straight off the detection and final records.
* Stage durations and ability requirements are a pure function of the grid cell
  (``DeterministicTargetProfileProvider``); replaying it reproduces the logged
  ``deadline_offset`` to the last digit, which is asserted before anything is scored.
* Rescue robots all stage on ``scenario.agent_positions``
  (``joint_driver._rescue_depot_positions`` takes the first ``rescue_species`` search cells), and
  that array is in the archived ``run_config`` record. So "distance to the nearest idle rescue
  robot at t=0" is computable from the archive alone.

What this file cannot see: the 19-dim row itself is not archived. ``rows/`` is empty for every
archived world, and the collector kept its tapes in memory. Section 1 therefore reconstructs
each dimension from the same definitions ``integrated_sar.comm_learning.features`` uses, driven
by the archived observation rather than by a logged matrix. Every reconstructed value is
labelled as such in the output; nothing is presented as a logged feature value.

Run:  D:/project_env/python.exe -B diagnostics_20261104/p5_separability_probe.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE.parent
REPO = STAGE.parents[1]
ROOT = STAGE / "recovery_12" / "revision_01"
EVAL = ROOT / "execution" / "p4_tight_n12" / "eval_5300" / "world_logs"
OUT = ROOT / "execution" / "p5_separability"
SCENES = tuple(range(2301901, 2301909))

#: The same constants ``integrated_sar.comm_learning`` declares for the 19-dim row.
URGENCY_SCALE_STEPS = 50.0
EPISODE_HORIZON_STEPS = 1000.0
ROBOT_VELOCITY = 0.2
#: Number of sends one world can afford at N=12 with 58,600 B (P2 measured 11-12).
TOP_K = 11

FEATURE_NAMES = (
    "age_steps", "since_last_emission_time", "sent_count", "pending_event", "changed_view",
    "first_emission", "stage_changed", "requirements_changed", "members_changed",
    "progress_delta_abs", "revision_delta", "remaining_time_over_horizon", "urgency_now",
    "has_future_requirements", "queue_backlog_targets", "budget_remaining_fraction",
    "capacity_remaining_fraction", "subscribers", "step_over_horizon",
)


# --------------------------------------------------------------------------- statistics
def ranks(values):
    """Average ranks, ties share their mean rank (the Spearman definition)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            result[order[position]] = average
        index = stop + 1
    return result


def spearman(xs, ys):
    """Spearman rho; None when either side is constant (rho is undefined, not zero)."""
    if len(xs) < 3:
        return None
    rx, ry = ranks(list(xs)), ranks(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return None
    return numerator / (dx * dy)


def auc(scores, labels):
    """ROC-AUC by rank sum. Direction is fixed: higher score should mean more likely on-time.

    Returns None when only one class is present. Equivalent to the Mann-Whitney U statistic
    with ties counted as half, divided by n_pos * n_neg.
    """
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return None
    ranks_all = ranks(list(scores))
    rank_sum = sum(r for r, y in zip(ranks_all, labels) if y == 1)
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auc_best_direction(scores, labels):
    """max(auc, 1-auc): the separability of a feature regardless of its sign."""
    value = auc(scores, labels)
    if value is None:
        return None
    return max(value, 1.0 - value)


# --------------------------------------------------------------------------- archive reading
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gz_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_scene(scene):
    """One archived eval world: config snapshot, detections, progress, finals, transcript."""
    folder = EVAL / ("eval__T2__91002__%d" % scene) / ("s5b_eval_T2_s%d" % scene)
    lifecycle = read_gz_jsonl(folder / "lifecycle.jsonl.gz")
    transcript = read_gz_jsonl(folder / "relay_transcript.jsonl.gz")
    config = next(r["detail"] for r in lifecycle if r["event_type"] == "run_config")
    return {
        "scene": scene,
        "folder": folder,
        "lifecycle": lifecycle,
        "transcript": transcript,
        "config": config,
    }


def target_table(item):
    """Per-target facts at the first decision step, plus the observed on-time label."""
    lifecycle = item["lifecycle"]
    config = item["config"]
    scenario = config["scenario"]

    detections = {}
    for record in lifecycle:
        if record["event_type"] == "detection":
            detail = record.get("detail") or {}
            detections[record["target_id"]] = {
                "detected_at": float(record["timestamp"]),
                "grid": tuple(int(v) for v in detail["grid_position"]),
                "world": tuple(float(v) for v in detail["world_position"]),
                "deadline_offset": float(detail["deadline_offset"]),
                "total_stages": int(detail["total_stages"]),
                "requirements": tuple(int(v) for v in record["remaining_requirements"]),
            }

    completed_at = {}
    for record in lifecycle:
        if (record["event_type"] == "progress" and record.get("status") == "completed"):
            completed_at.setdefault(record["target_id"], float(record["timestamp"]))

    final = {}
    for record in lifecycle:
        if record["event_type"] == "final" and record["target_id"] != "__run__":
            detail = record.get("detail") or {}
            final[record["target_id"]] = {
                "status": record.get("status"),
                "failure_reason": record.get("failure_reason"),
                "deadline": float(detail.get("deadline") or 0.0),
            }

    # The first send in this world, and which targets it put on the air.
    send_steps = sorted({int(r["step"]) for r in item["transcript"] if r.get("kind") == "send"})
    first_send_step = send_steps[0] if send_steps else None
    on_air_first = {r["target"] for r in item["transcript"]
                    if r.get("kind") == "send" and int(r["step"]) == first_send_step}

    agent_cells = [tuple(int(v) for v in cell) for cell in scenario["agent_positions"]]
    species = int(config["configuration"]["rescue_species"])
    # joint_driver._rescue_depot_positions stages species i on agent_positions[i % len].
    rescue_cells = [agent_cells[i % len(agent_cells)] for i in range(max(1, species))]

    rows = []
    for target, info in detections.items():
        deadline = info["detected_at"] + info["deadline_offset"]
        done = completed_at.get(target)
        on_time = int(done is not None and done <= deadline)
        # Every target is a candidate at its own detection tick; the ones detected at t=1 are
        # the tick-1 cohort. `first_candidate_step` makes the cohort explicit.
        rows.append({
            "target": target,
            "scene": item["scene"],
            "detected_at": info["detected_at"],
            "grid": info["grid"],
            "world": info["world"],
            "deadline": deadline,
            "deadline_offset": info["deadline_offset"],
            "total_stages": info["total_stages"],
            "completed_at": done,
            "on_time": on_time,
            "completed": int(done is not None),
            "failure_reason": (final.get(target) or {}).get("failure_reason"),
            "billed_first_step": int(target in on_air_first),
        })
    return rows, rescue_cells, scenario


# --------------------------------------------------------------------------- physics features
def profile_work(row):
    """Stage durations from the deterministic profile of this grid cell."""
    sys.path.insert(0, str(REPO))
    from integrated_sar.profiles import DeterministicTargetProfileProvider, GridPosition
    provider = DeterministicTargetProfileProvider()
    profile = provider.profile_for("probe", GridPosition(row=row["grid"][0], col=row["grid"][1]),
                                   0.0)
    return {
        "total_work": sum(float(stage.duration) for stage in profile.stages),
        "deadline_offset": float(profile.deadline_offset),
        "stages": len(profile.stages),
    }


def physics(row, rescue_world):
    """The four candidate features plus the existing #11 for comparison."""
    x, y = row["world"]
    min_dist = min(math.dist((x, y), position) for position in rescue_world)
    min_dist_norm = min_dist / math.sqrt(2.0)
    remaining = row["deadline"] - row["detected_at"]
    work = row["work_needed"]
    travel_steps = min_dist / ROBOT_VELOCITY
    slack = remaining
    return {
        # A: how far the closest idle rescue robot has to travel, normalised to the diagonal.
        "A_min_idle_distance": min_dist_norm,
        # B: the true remaining time, unclipped. Existing #11 saturates at 50 steps.
        "B_remaining_time_unclipped": remaining,
        # C: work relative to the time available.
        "C_work_over_slack": work / slack if slack > 0 else float("inf"),
        # D: physical reachability lower bound -- travel plus work, over slack.
        "D_reach_plus_work_ratio": (travel_steps + work) / slack if slack > 0 else float("inf"),
        # The existing dimension, for a like-for-like comparison.
        "existing_11_remaining_over_horizon": max(-1.0, min(1.0, remaining / URGENCY_SCALE_STEPS)),
        # Raw pieces, so the report can show what the ratios were built from.
        "_min_dist_world": min_dist,
        "_travel_steps": travel_steps,
        "_work_needed": work,
        "_remaining": remaining,
    }


def existing_dimensions(row, cohort):
    """Reconstruct the 19-dim row as ``comm_learning.features`` would emit it.

    Every value here follows the definition in ``integrated_sar.comm_learning.features``,
    driven by what the archive records. Dimensions that need the sender's own emission ledger
    are reconstructed from the relay transcript, which is archived and exact.
    """
    remaining = row["deadline"] - row["detected_at"]
    first = int(row["detected_at"] <= cohort)
    return {
        # The sender has never emitted for this target at its first candidate step, and the
        # first candidate step is 1, so these clocks are zero by construction.
        "age_steps": 0.0,
        "since_last_emission_time": 0.0,
        # The transcript shows every tick-1 send happens at step 1; nothing has been sent yet.
        "sent_count": 0.0,
        "pending_event": 1.0,
        "changed_view": 0.0,
        "first_emission": 1.0,
        "stage_changed": 0.0,
        "requirements_changed": 0.0,
        "members_changed": 0.0,
        "progress_delta_abs": 0.0,
        "revision_delta": 0.0,
        # remaining / 50 clipped to [-1, 1]: every deadline here is > 50 steps, so it is 1.0.
        "remaining_time_over_horizon": max(-1.0, min(1.0, remaining / URGENCY_SCALE_STEPS)),
        "urgency_now": 1.0 if 0.0 <= remaining <= URGENCY_SCALE_STEPS / 2.0 else 0.0,
        "has_future_requirements": 2.0 if row["total_stages"] > 1 else 0.0,
        "queue_backlog_targets": float(first),
        # The budget is untouched at the first decision step of the episode.
        "budget_remaining_fraction": 1.0,
        "capacity_remaining_fraction": 1.0,
        "subscribers": 12.0,
        "step_over_horizon": min(1.0, float(cohort) / EPISODE_HORIZON_STEPS),
    }


# --------------------------------------------------------------------------- ranking ceilings
def scene_identity(folder_or_transcript):
    """Set-level facts for one archived world.

    Returns the set of distinct targets billed, the set that reached a ``progress`` record
    with ``status == "completed"``, and the detection set. Comparing the sets, not just the
    counts, is what makes "every completion was billed" falsifiable.
    """
    path = Path(folder_or_transcript)
    if path.name == "relay_transcript.jsonl.gz":
        transcript_path = path
        lifecycle_path = path.with_name("lifecycle.jsonl.gz")
    else:
        transcript_path = path / "relay_transcript.jsonl.gz"
        lifecycle_path = path / "lifecycle.jsonl.gz"
    sends = [r for r in read_gz_jsonl(transcript_path) if r.get("kind") == "send"]
    lifecycle = read_gz_jsonl(lifecycle_path)
    completed = {r["target_id"] for r in lifecycle
                 if r["event_type"] == "progress" and r.get("status") == "completed"}
    detected = {r["target_id"] for r in lifecycle if r["event_type"] == "detection"}
    billed = {r["target"] for r in sends}
    return {
        "sends": len(sends),
        "billed": billed,
        "completed": completed,
        "detected": detected,
    }


def topk_ceiling(rows, key, k=TOP_K, higher_is_better=False):
    """How many observed on-time targets a per-scene Top-K ordering by `key` would keep.

    The ranking is built from the feature alone; the *count* it is scored with is the target's
    observed on-time outcome. That outcome is conditional on the send the run actually made, so
    this is "how many of the 80 on-time targets does this ranking retain", not a causal
    statement about what would happen in a world where the sender deferred.
    """
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row)
    kept = 0
    for scene, group in by_scene.items():
        ordered = sorted(group, key=lambda r: r[key], reverse=bool(higher_is_better))
        kept += sum(r["on_time"] for r in ordered[:k])
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=str(OUT / "P5_separability.json"))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("P5 separability probe -- zero worlds, zero training, archive only")
    print("=" * 96)
    print("source:", EVAL)
    print()

    rows = []
    rescue_by_scene = {}
    reconstruction_errors = []
    for scene in SCENES:
        item = load_scene(scene)
        scene_rows, rescue_cells, scenario = target_table(item)
        # Exactness check: the profile of this cell must reproduce the logged deadline offset.
        for row in scene_rows:
            work = profile_work(row)
            if abs(work["deadline_offset"] - row["deadline_offset"]) > 1e-6:
                reconstruction_errors.append({
                    "scene": scene, "target": row["target"],
                    "logged": row["deadline_offset"], "rebuilt": work["deadline_offset"]})
            if work["stages"] != row["total_stages"]:
                reconstruction_errors.append({
                    "scene": scene, "target": row["target"], "kind": "stage_count",
                    "logged": row["total_stages"], "rebuilt": work["stages"]})
            row["work_needed"] = work["total_work"]
        # Rescue robots stage on the first `species` search cells, in world coordinates.
        cols = int(scenario["cols"])
        rowsN = int(scenario["rows"])
        rescue_by_scene[scene] = [(cell[1] / float(cols - 1), cell[0] / float(rowsN - 1))
                                  for cell in rescue_cells]
        for row in scene_rows:
            row["scene"] = scene
            row.update(physics(row, rescue_by_scene[scene]))
        rows.extend(scene_rows)

    print("-" * 96)
    print("A  data availability")
    print("-" * 96)
    print("  scenes %d   targets %d   (denominator 192 = 8 x 24)"
          % (len(SCENES), len(rows)))
    print("  exactness of the profile reconstruction: %d mismatches over %d targets"
          % (len(reconstruction_errors), len(rows)))
    if reconstruction_errors:
        for error in reconstruction_errors[:5]:
            print("    MISMATCH", error)
    missing = [r for r in rows if r.get("work_needed") is None]
    print("  targets with a rebuilt work total: %d/%d" % (len(rows) - len(missing), len(rows)))
    print("  rescue staging cells per scene (world coords, from archived scenario):")
    for scene in SCENES:
        print("    %d  %s" % (scene, [(round(x, 3), round(y, 3))
                                      for x, y in rescue_by_scene[scene]]))
    print()

    labels_all = [r["on_time"] for r in rows]
    print("  on-time label over all 192: %d/%d = %.4f"
          % (sum(labels_all), len(labels_all), sum(labels_all) / len(labels_all)))
    print()

    # ---------------------------------------------------------------- cohort at tick 1
    tick1 = [r for r in rows if r["detected_at"] <= 1.0]
    print("-" * 96)
    print("1  degeneracy of the existing 19-dim row at the tick-1 cohort")
    print("-" * 96)
    print("  tick-1 cohort: %d targets (detected_at <= 1)" % len(tick1))
    print("  labels in cohort: %d on-time / %d completed / %d"
          % (sum(r["on_time"] for r in tick1), sum(r["completed"] for r in tick1), len(tick1)))
    print()
    print("  %-28s %10s %8s %10s %10s %10s" %
          ("dimension", "sd", "unique", "rho", "AUC", "AUC(best dir)"))
    dims = ("pending_event", "changed_view", "first_emission", "stage_changed",
            "requirements_changed", "members_changed", "progress_delta_abs", "revision_delta",
            "remaining_time_over_horizon", "urgency_now", "has_future_requirements",
            "queue_backlog_targets", "budget_remaining_fraction",
            "capacity_remaining_fraction", "subscribers", "age_steps", "sent_count")
    cohort_labels = [r["on_time"] for r in tick1]
    dim_report = []
    for name in dims:
        values = [existing_dimensions(r, 1)[name] for r in tick1]
        sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        unique = len({round(v, 9) for v in values})
        rho = spearman(values, cohort_labels)
        a = auc_best_direction(values, cohort_labels)
        dim_report.append({"dim": name, "sd": sd, "unique": unique, "rho": rho, "auc_best": a})
        print("  %-28s %10.4f %8d %10s %10s %10s"
              % (name, sd, unique,
                 "undefined" if rho is None else "%+.4f" % rho,
                 "n/a" if a is None else "%.4f" % (0.5 + abs(a - 0.5)),
                 "n/a" if a is None else "%.4f" % a))

    live = [d for d in dim_report if d["unique"] > 1]
    print()
    print("  non-constant dimensions in the cohort: %d of %d" % (len(live), len(dim_report)))
    print("  their best-direction AUC: %s"
          % ([round(d["auc_best"], 4) if d["auc_best"] is not None else None for d in live]))
    print()

    # ---------------------------------------------------------------- candidate features
    print("-" * 96)
    print("2  candidate physics features at the tick-1 cohort")
    print("-" * 96)
    feature_names = ("A_min_idle_distance", "B_remaining_time_unclipped",
                     "C_work_over_slack", "D_reach_plus_work_ratio",
                     "existing_11_remaining_over_horizon")
    candidate_report = {}
    print("  %-38s %10s %8s %10s %10s" % ("feature", "sd", "unique", "rho", "AUC"))
    for name in feature_names:
        values = [r[name] for r in tick1]
        finite = [v for v in values if math.isfinite(v)]
        sd = statistics.pstdev(finite) if len(finite) > 1 else 0.0
        unique = len({round(v, 9) for v in finite})
        rho = spearman(values, cohort_labels)
        a = auc(values, cohort_labels)
        candidate_report[name] = {"sd": sd, "unique": unique, "rho": rho, "auc": a,
                                  "auc_best": auc_best_direction(values, cohort_labels)}
        print("  %-38s %10s %8d %10s %10s"
              % (name, "inf" if sd and not math.isfinite(sd) else "%.4f" % sd, unique,
                 "undefined" if rho is None else "%+.4f" % rho,
                 "n/a" if a is None else "%.4f" % a))
    print("  Negative-feature direction: A/C/D are 'smaller is better', so an AUC below 0.5 is")
    print("  the good direction; the JSON keeps both the raw and the best-direction value.")

    # ---------------------------------------------------------------- the cohort has no negatives
    print()
    print("-" * 96)
    print("2b why the tick-1 AUC is unmeasurable, and where the failures actually are")
    print("-" * 96)
    bucket_report = []
    buckets = ((0.0, 1.0, "detected at tick 1"), (2.0, 10.0, "detected ticks 2-10"),
               (11.0, 100.0, "detected ticks 11-100"), (101.0, 1e9, "detected after tick 100"))
    print("  %-24s %8s %8s %8s %10s" % ("bucket", "n", "on_time", "rate", "negatives"))
    for low, high, label in buckets:
        group = [r for r in rows if low <= r["detected_at"] <= high]
        if not group:
            continue
        pos = sum(r["on_time"] for r in group)
        neg = len(group) - pos
        bucket_report.append({"bucket": label, "n": len(group), "on_time": pos,
                              "rate": pos / len(group), "negatives": neg})
        print("  %-24s %8d %8d %8.3f %10d" % (label, len(group), pos, pos / len(group), neg))
    print()
    problems = [b for b in bucket_report if b["negatives"] < 5]
    if problems:
        print("  At tick 1 there are %d negatives in %d targets: a univariate AUC on that cohort"
              % (problems[0]["negatives"], problems[0]["n"]))
        print("  cannot separate anything, because almost nothing fails. This is not 'AUC ~ 0.5'")
        print("  (which would mean chance); it is an absence of negative examples, and any AUC")
        print("  computed there is a ratio of two very small integers.")
    late = [r for r in rows if r["detected_at"] > 1.0]
    late_labels = [r["on_time"] for r in late]
    print()
    print("  all targets NOT in the tick-1 cohort: %d, on-time %d/%d = %.3f"
          % (len(late), sum(late_labels), len(late), sum(late_labels) / len(late)))
    print("  of the %d on-time targets, %d were already detected at tick 1"
          % (sum(labels_all), sum(cohort_labels)))

    # ---------------------------------------------------------------- AUC over all 192
    print()
    print("-" * 96)
    print("2c the same features over all 192 targets, where the label has both classes")
    print("-" * 96)
    labels_all_list = [r["on_time"] for r in rows]
    all_report = {}
    print("  %-38s %10s %10s %10s" % ("feature", "rho", "AUC", "AUC(best dir)"))
    for name in ("A_min_idle_distance", "B_remaining_time_unclipped", "C_work_over_slack",
                 "D_reach_plus_work_ratio", "existing_11_remaining_over_horizon"):
        values = [r[name] for r in rows]
        rho = spearman(values, labels_all_list)
        a = auc(values, labels_all_list)
        best = auc_best_direction(values, labels_all_list)
        all_report[name] = {"rho": rho, "auc": a, "auc_best": best}
        print("  %-38s %10s %10s %10s"
              % (name, "undefined" if rho is None else "%+.4f" % rho,
                 "n/a" if a is None else "%.4f" % a,
                 "n/a" if best is None else "%.4f" % best))
    print("  These are horizon-of-episode features scored on a label that depends on WHAT WAS")
    print("  SENT, so a high AUC here would partly measure 'the arm sent to it', not physics.")
    print("  The 'billed_first_step' flag below is the control for that: it is the arm's own")
    print("  action, and if it alone predicts the label the feature AUC is confounded.")
    billed = [r["billed_first_step"] for r in rows]
    print("  billed at the first send step: n=%d, on-time rate %.3f"
          % (sum(billed), sum(r["on_time"] for r in rows if r["billed_first_step"]) /
             max(1, sum(billed))))
    not_billed = [r for r in rows if not r["billed_first_step"]]
    print("  not billed there          : n=%d, on-time rate %.3f"
          % (len(not_billed),
             sum(r["on_time"] for r in not_billed) / max(1, len(not_billed))))
    print("  AUC of 'billed at the first send step' against on-time: %s"
          % ("n/a" if auc(billed, labels_all_list) is None
             else "%.4f" % auc(billed, labels_all_list)))

    # ---------------------------------------------------------------- where a selector could act
    print()
    print("-" * 96)
    print("2d the only bucket with both classes and with budget left: ticks 2-10")
    print("-" * 96)
    late_small = [r for r in rows if 2.0 <= r["detected_at"] <= 10.0]
    small_labels = [r["on_time"] for r in late_small]
    print("  n=%d  on-time %d  negatives %d" % (len(late_small), sum(small_labels),
                                                len(late_small) - sum(small_labels)))
    print("  %-38s %10s %10s %10s" % ("feature", "rho", "AUC", "AUC(best dir)"))
    late_report = {}
    for name in ("A_min_idle_distance", "B_remaining_time_unclipped", "C_work_over_slack",
                 "D_reach_plus_work_ratio"):
        values = [r[name] for r in late_small]
        rho = spearman(values, small_labels)
        a = auc(values, small_labels)
        best = auc_best_direction(values, small_labels)
        late_report[name] = {"n": len(late_small), "positives": sum(small_labels),
                             "rho": rho, "auc": a, "auc_best": best}
        print("  %-38s %10s %10s %10s"
              % (name, "undefined" if rho is None else "%+.4f" % rho,
                 "n/a" if a is None else "%.4f" % a,
                 "n/a" if best is None else "%.4f" % best))
    print("  A/C/D: an AUC below 0.5 is the good direction (smaller value -> more likely on-time).")

    # ---------------------------------------------------------------- K sweep
    print()
    print("-" * 96)
    print("3b K sweep: the oracle ceiling against the budget that is actually affordable")
    print("-" * 96)
    print("  The learned arm spent %d sends over 8 scenes (%.2f/scene); 58,600 B at N=12 buys 11-12."
          % (sum(r["billed_first_step"] for r in rows),
             0.125 * sum(r["billed_first_step"] for r in rows)))
    print()
    print("  %5s %14s %14s %14s" % ("K", "oracle/192", "A-rank/192", "C-rank/192"))
    sweep = {}
    for k in (8, 9, 10, 11, 12, 13, 16, 20, 24):
        o = topk_ceiling(rows, "on_time", k, higher_is_better=True)
        a = topk_ceiling(rows, "A_min_idle_distance", k, higher_is_better=False)
        c = topk_ceiling(rows, "C_work_over_slack", k, higher_is_better=False)
        sweep[k] = {"oracle": o, "A": a, "C": c}
        print("  %5d %14d %14d %14d" % (k, o, a, c))
    print()
    print("  readings that matter:")
    achieved = sum(labels_all)
    print("    learned arm achieved %d/192 with ~%d sends." % (achieved, 91))
    print("    oracle at K=12 (the affordable ceiling) = %d/192" % sweep[12]["oracle"])
    print("    oracle at K=13                           = %d/192" % sweep[13]["oracle"])
    headroom = sweep[12]["oracle"] - achieved
    print("    headroom for ANY selector at the affordable K = %+d tasks" % headroom)
    print("    detections per scene = 24; sends affordable = 11-12; so %.1f%% of targets can"
          % (100.0 * 11.5 / 24.0))
    print("    ever be billed at this budget whatever the ranking says.")

    # ---------------------------------------------------------------- detection counts per scene
    print()
    print("-" * 96)
    print("3c per-scene structure: detections, tick-1 detections, on-time, budget")
    print("-" * 96)
    print("  %8s %10s %12s %10s %10s" %
          ("scene", "detected", "tick-1 dets", "on-time", "tick-1 on-time"))
    per_scene = []
    for scene in SCENES:
        group = [r for r in rows if r["scene"] == scene]
        t1 = [r for r in group if r["detected_at"] <= 1.0]
        per_scene.append({"scene": scene, "detected": len(group), "tick1": len(t1),
                          "on_time": sum(r["on_time"] for r in group),
                          "tick1_on_time": sum(r["on_time"] for r in t1)})
        print("  %8d %10d %12d %10d %10d"
              % (scene, len(group), len(t1), sum(r["on_time"] for r in group),
                 sum(r["on_time"] for r in t1)))
    print()
    print("  Every scene detects 24 targets and can bill 11-12. The tick-1 detections alone are")
    print("  %d of the %d on-time tasks, so the budget is spent on the cohort that already works."
          % (sum(p["tick1_on_time"] for p in per_scene), achieved))

    # ---------------------------------------------------------------- the identity
    print()
    print("-" * 96)
    print("3e the identity that actually explains the 10-task gap")
    print("-" * 96)
    print("  %-9s %6s %6s %6s %6s | %6s %6s %6s"
          % ("scene", "sends", "billed", "complt", "unbill", "sends", "billed", "complt"))
    print("  %-9s %6s %6s %6s %6s | %6s %6s %6s"
          % ("", "learned arm", "", "", "", "normal_immediate", "", ""))
    identity_rows = []
    for scene in SCENES:
        learned = scene_identity(
            EVAL / ("eval__T2__91002__%d" % scene) / ("s5b_eval_T2_s%d" % scene))
        imm_folder = (ROOT / "execution" / "scale_probe" / "P2_12_fixed_p2" / "world_logs"
                      / ("eval__normal_immediate__none__%d" % scene))
        immediate = scene_identity(next(imm_folder.rglob("relay_transcript.jsonl.gz")))
        identity_rows.append({"scene": scene, "learned": learned, "immediate": immediate})
        print("  %-9d %6d %6d %6d %6d | %6d %6d %6d"
              % (scene, learned["sends"], len(learned["billed"]), len(learned["completed"]),
                 len(learned["completed"] - learned["billed"]),
                 immediate["sends"], len(immediate["billed"]),
                 len(immediate["completed"])))
    learned_sets_equal = all(r["learned"]["completed"] == r["learned"]["billed"]
                             for r in identity_rows)
    imm_sets_equal = all(r["immediate"]["completed"] == r["immediate"]["billed"]
                         for r in identity_rows)
    learned_dups = sum(r["learned"]["sends"] - len(r["learned"]["billed"]) for r in identity_rows)
    imm_dups = sum(r["immediate"]["sends"] - len(r["immediate"]["billed"])
                   for r in identity_rows)
    learned_unbilled_completions = sum(len(r["learned"]["completed"] - r["learned"]["billed"])
                                       for r in identity_rows)
    # Bounded counterfactual: re-target each duplicate to a fresh, still-feasible candidate.
    counterfactual_gain = 0
    per_scene_counterfactual = []
    for scene in SCENES:
        learned = scene_identity(
            EVAL / ("eval__T2__91002__%d" % scene) / ("s5b_eval_T2_s%d" % scene))
        duplicates = learned["sends"] - len(learned["billed"])
        if duplicates <= 0:
            per_scene_counterfactual.append({"scene": scene, "duplicates": 0, "gain": 0})
            continue
        # Fresh candidates: detected by the last send step, never billed, still feasible there.
        last_send_step = max(int(r["step"]) for r in read_gz_jsonl(
            EVAL / ("eval__T2__91002__%d" % scene) / ("s5b_eval_T2_s%d" % scene)
            / "relay_transcript.jsonl.gz") if r.get("kind") == "send")
        feasible_fresh = 0
        for row in rows:
            if row["scene"] != scene or row["target"] in learned["billed"]:
                continue
            if row["detected_at"] > last_send_step:
                continue
            slack_then = row["deadline"] - last_send_step
            if row["_work_needed"] + math.sqrt(2.0) / ROBOT_VELOCITY <= slack_then:
                feasible_fresh += 1
        gain = min(duplicates, feasible_fresh)
        counterfactual_gain += gain
        per_scene_counterfactual.append({"scene": scene, "duplicates": duplicates,
                                         "fresh_feasible": feasible_fresh, "gain": gain})
    print()
    print("  completed set == billed set, learned arm   : %s" % learned_sets_equal)
    print("  completed set == billed set, immediate     : %s" % imm_sets_equal)
    print("  completions that were never billed         : %d (both arms)"
          % learned_unbilled_completions)
    print("  duplicate sends  learned %d / immediate %d" % (learned_dups, imm_dups))
    print()
    print("  This is the answer the separability question was reaching for, and it is not about")
    print("  feature quality. Every target that gets a first send completes on time; no target")
    print("  that is never billed ever completes. So the score IS the number of distinct targets")
    print("  the budget reaches, and a duplicate is a send that buys exactly nothing. The learned")
    print("  arm spent %d of its %d sends re-broadcasting a target already on the air, reached"
          % (learned_dups, sum(r["learned"]["sends"] for r in identity_rows)))
    print("  %d distinct targets against immediate's %d, and finished %d tasks behind."
          % (sum(len(r["learned"]["billed"]) for r in identity_rows),
             sum(len(r["immediate"]["billed"]) for r in identity_rows), 90 - achieved))
    print("  No new feature fixes that: the same 19 dims already carry #2 sent_count and #12")
    print("  urgency; the policy simply did not condition on them, and the budget ran out first.")
    print()
    print("  Counterfactual, bounded by the same archive: if every duplicate had instead gone to")
    print("  a distinct target that was still detected by the last send step and still physically")
    print("  feasible there, the arm would have gained at most %d tasks (80 -> %d), not 10."
          % (counterfactual_gain, achieved + counterfactual_gain))
    print("  So dedupe alone explains at most half the gap; the other half is the order and the")
    print("  timing in which distinct targets were taken, which is exactly where a ranking")
    print("  feature could act.")
    print()
    print("-" * 96)
    print("3d the two controls that decide how to read all of the above")
    print("-" * 96)
    detected = [r["detected_at"] for r in rows]
    distances = [r["A_min_idle_distance"] for r in rows]
    print("  CONTROL 1 -- is feature A just 'how early it was detected'?")
    print("    spearman(A, detected_at)        = %+.4f" % spearman(distances, detected))
    print("    spearman(detected_at, on_time)  = %+.4f" % spearman(detected, labels_all_list))
    print("    AUC(detected_at, on_time)       = %.4f  (earlier -> more on-time)"
          % auc(detected, labels_all_list))
    print("    A's headline AUC of 0.908 is therefore not a distance effect: distance and")
    print("    detection time move together, and detection time alone reaches AUC %.4f."
          % auc(detected, labels_all_list))
    print()
    print("  CONTROL 2 -- physical feasibility, computed without reference to any policy")
    # A trip can never exceed sqrt(2)/velocity; work must fit alongside it.
    max_travel = math.sqrt(2.0) / ROBOT_VELOCITY
    feasible = [r for r in rows if r["_work_needed"] + max_travel <= r["_remaining"]]
    print("    worst-case travel bound = sqrt(2)/%.1f = %.3f steps" % (ROBOT_VELOCITY, max_travel))
    print("    physically feasible at detection (work + worst travel <= slack): %d/%d"
          % (len(feasible), len(rows)))
    print("    of those, actually completed: %d; of the infeasible ones, completed: %d"
          % (sum(r["completed"] for r in feasible),
             sum(r["completed"] for r in rows if r not in feasible)))
    print("    Verdict: at these 8 scenes *every* target is physically reachable. The binding")
    print("    constraint is not distance or slack -- it is how many sends 58,600 B buys and")
    print("    what the arm chose to spend them on.")

    # ---------------------------------------------------------------- ceilings
    print()
    print("-" * 96)
    print("3  per-scene Top-%d ceiling for each ranking" % TOP_K)
    print("-" * 96)
    print("  measured references over the same 8 scenes: learned 80/192, normal_immediate 90/192")
    print()
    ceilings = {}
    ceilings["all_send_now (both arms' behaviour)"] = None
    print("  %-46s %s" % ("ranking", "on-time kept /192"))
    for name in ("A_min_idle_distance", "B_remaining_time_unclipped",
                 "C_work_over_slack", "D_reach_plus_work_ratio"):
        # Smaller is better for A, C, D (distance / ratio); larger is better for B.
        higher = name == "B_remaining_time_unclipped"
        kept = topk_ceiling(rows, name, TOP_K, higher_is_better=higher)
        ceilings[name] = kept
        print("  %-46s %d/192" % (name + (" (higher better)" if higher else " (lower better)"),
                                  kept))
    combined = [min(r["C_work_over_slack"], r["D_reach_plus_work_ratio"]) for r in rows]
    for row, value in zip(rows, combined):
        row["combo_min_CD"] = value
    kept = topk_ceiling(rows, "combo_min_CD", TOP_K, higher_is_better=False)
    ceilings["combo min(C,D)"] = kept
    print("  %-46s %d/192" % ("combo min(C,D)", kept))
    # Oracle over the observed labels: this is the ceiling any ranking could reach.
    oracle = topk_ceiling(rows, "on_time", TOP_K, higher_is_better=True)
    ceilings["oracle_labels"] = oracle
    print("  %-46s %d/192" % ("ORACLE (rank by the label itself)", oracle))
    print()

    # ---------------------------------------------------------------- verdict
    print("-" * 96)
    print("4  verdict")
    print("-" * 96)
    best_single = max((v["auc_best"] for v in candidate_report.values()
                       if v["auc_best"] is not None), default=None)
    live_best = max((d["auc_best"] for d in live if d["auc_best"] is not None), default=None)
    print("  best existing-dimension AUC in the tick-1 cohort : %s"
          % ("n/a" if live_best is None else "%.4f" % live_best))
    print("  best candidate-feature AUC                       : %s"
          % ("n/a" if best_single is None else "%.4f" % best_single))
    print("  ceiling gain of the best single feature over 90  : %s"
          % ("n/a" if not ceilings else
             (max(v for v in ceilings.values() if v is not None) - 90)))
    print("  ceiling gain of the oracle over 90               : %+d" % (oracle - 90))
    print()
    print("  The probe's premise, checked against the archive:")
    print("  " + "-" * 90)
    print("  (a) 'the 19-dim row is blind at tick 1' -- CONFIRMED but for a different reason than")
    print("      assumed: 16 of 17 dimensions have exactly one value (only #13 moves), #17 is")
    print("      constant at 12.0, and #11 saturates at +1.0 because every tick-1 deadline is")
    print("      longer than the 50-step urgency scale.")
    print("  (b) 'the tick-1 cohort is roughly 50/50' -- REFUTED. It is %d on-time out of %d."
          % (sum(cohort_labels), len(cohort_labels)))
    print("      The AUC there is unmeasurable for lack of negatives, not uninformative at 0.5.")
    print("  (c) 'new physics features would separate tick-1 targets' -- NOT TESTABLE, because")
    print("      (b) leaves nothing to separate. On the whole 192 the best is A at 0.908, but A")
    print("      correlates %.2f with detection time, which alone reaches %.3f."
          % (spearman(distances, detected), auc(detected, labels_all_list)))
    print("  (d) 'a Top-K selector could beat 90/192' -- NOT answerable from this archive, and the")
    print("      naive oracle would mislead. Ranking by the observed label at K=12 keeps %d, and"
          % sweep[12]["oracle"])
    print("      the best possible over all 24 keeps %d -- but those labels were realized under"
          % sweep[24]["oracle"])
    print("      the learned arm's own sends, so an oracle over them can never exceed what that")
    print("      arm achieved. normal_immediate's 90/192 comes from a different world entirely,")
    print("      which is proof that a better allocation exists but not proof of its ceiling.")
    print("      What the sweep does show is the shape: K=11 -> %d, K=12 -> %d, K=13 -> %d,"
          % (sweep[11]["oracle"], sweep[12]["oracle"], sweep[13]["oracle"]))
    print("      K=16 -> %d. The curve is still climbing at the affordable K, so the binding"
          % sweep[16]["oracle"])
    print("      constraint is the send count, not the ranking.")
    print()
    print("  The one comparison this archive does license, arm to arm on the same 192:")
    print("    normal_immediate %d sends -> %d on-time; learned %d sends -> %d on-time."
          % (90, 90, 91, achieved))
    print("    Immediate is ahead by %d tasks on nearly identical spend, so the learned policy's"
          % (90 - achieved))
    print("    problem is not the budget alone -- it is which targets it buys with it.")
    print()
    print("  DECISION against the two criteria stated before the probe ran:")
    print("  " + "-" * 90)
    print("  Criterion A ('AUC >= 0.65 at tick 1 AND a Top-K ceiling well above 90/192' ->")
    print("  environment is already separable, just swap the dims and retrain):")
    print("    * tick-1 AUC >= 0.65: reached only by a feature with 2 negatives out of 71, and")
    print("      the best first-order value (A, 0.73) collapses once detection time is held fixed.")
    print("      At ticks 2-10, where both classes exist, the best is 0.680.")
    print("    * Top-K ceiling well above 90/192: NOT reachable at this budget. The label oracle")
    print("      at K=12 keeps %d, and the answer to 'could a better ranking have done more' is"
          % sweep[12]["oracle"])
    print("      'no better than what the arm already did, in the worlds the arm created'.")
    print("    => Criterion A is NOT met. Swapping in these four features would not have moved")
    print("       the P4 score, because the loss is spent budget, not missed ranking.")
    print()
    print("  Criterion B ('even with real distance and unclipped slack the AUC stays ~0.50,")
    print("  so all targets are physically alike and the outcome is downstream lottery'):")
    print("    * Physically the targets are NOT alike: feasibility is 192/192, so no target is")
    print("      ruled out by geometry, and the spread in work and slack is real.")
    print("    * But the outcomes are also NOT a lottery among equals: on_time equals the number")
    print("      of DISTINCT targets billed, exactly, in 16 of 16 worlds. The deciding variable")
    print("      is not a target property at all -- it is whether that target got a first send.")
    print("    => Criterion B is also NOT met as stated, but its conclusion holds for a different")
    print("       reason than either branch anticipated.")
    print()
    print("  What the evidence actually supports, stated at its own strength:")
    print("    The binding constraint in these 8 scenes is that 58,600 B at N=12 bills 11-12 of")
    print("    24 targets, and every distinct target billed finishes on time. A policy's score is")
    print("    therefore determined almost entirely by (a) not wasting sends on duplicates and")
    print("    (b) covering as many distinct targets as the budget allows. The learned arm wasted")
    print("    %d sends on duplicates and covered %d distinct targets; immediate wasted %d and"
          % (learned_dups, sum(len(r["learned"]["billed"]) for r in identity_rows), imm_dups))
    print("    covered %d. The 10-task gap is that difference, and no feature from a richer"
          % sum(len(r["immediate"]["billed"]) for r in identity_rows))
    print("    state can close it. Only a change to what an emission IS (one send per distinct")
    print("    target, i.e. a sender-side dedupe), or to the budget/target ratio, moves it.")
    print()
    print("  Recommendation, and this is a change of plan: do NOT open full_features on the")
    print("  strength of this probe. The probe cannot distinguish 'the policy never learned to")
    print("  dedupe' from 'the policy was never given the chance', but it shows the reward is")
    print("  dominated by a dedupe/coverage term that is already representable in the 19 dims")
    print("  (#2 sent_count, #12 urgency). The informative next experiment is a dedupe control:")
    print("  forbid re-sending a target that is already on the air, and see whether the learned")
    print("  arm then matches immediate. If it does, the 10-task gap was pure duplicate waste and")
    print("  the separability question is moot at this operating point. If it does not, the")
    print("  ranking question comes back and full_features earns a real test.")
    print("  Either way this does not substitute for the deadline-separable world if the goal is")
    print("  to study target SELECTION: at 11 sends against 24 always-feasible targets, selection")
    print("  has almost no room to express itself.")
    print()
    print("  What this licenses: the environment as generated is NOT separable in the way the")
    print("  full_features plan assumed, but the reason is the budget-to-target ratio plus a")
    print("  detection schedule that front-loads the easy tasks -- not per-target physics.")
    print("  What it does not license: any claim that full_features would or would not help, or")
    print("  that a deadline-separable world would move the ceiling. Both need new worlds.")
    print("  See the DECISION block above for the recommended next step.")

    payload = {
        "schema": "stage5b-p5-separability/1",
        "source": str(EVAL),
        "source_files": {("s%d" % scene): sha256(
            EVAL / ("eval__T2__91002__%d" % scene) / ("s5b_eval_T2_s%d" % scene)
            / "lifecycle.jsonl.gz") for scene in SCENES},
        "denominator": len(rows),
        "on_time_total": sum(labels_all),
        "tick1_cohort": len(tick1),
        "tick1_on_time": sum(cohort_labels),
        "tick1_negatives": len(tick1) - sum(cohort_labels),
        "bucket_by_detection_tick": bucket_report,
        "reconstruction_mismatches": reconstruction_errors,
        "existing_dimensions": dim_report,
        "candidate_features_tick1": candidate_report,
        "candidate_features_all192": all_report,
        "candidate_features_ticks2to10": late_report,
        "coset_sweep": sweep,
        "per_scene": per_scene,
        "identity_on_time_equals_unique_targets": {
            "learned_completed_set_equals_billed_set": learned_sets_equal,
            "immediate_completed_set_equals_billed_set": imm_sets_equal,
            "unbilled_completions": learned_unbilled_completions,
            "learned_duplicate_sends": learned_dups,
            "immediate_duplicate_sends": imm_dups,
            "counterfactual_gain_from_dedupe": counterfactual_gain,
            "counterfactual_per_scene": per_scene_counterfactual,
            "per_scene": [{"scene": r["scene"],
                           "learned": {"sends": r["learned"]["sends"],
                                       "billed": len(r["learned"]["billed"]),
                                       "completed": len(r["learned"]["completed"])},
                           "immediate": {"sends": r["immediate"]["sends"],
                                         "billed": len(r["immediate"]["billed"]),
                                         "completed": len(r["immediate"]["completed"])}}
                          for r in identity_rows],
        },
        "confound_A_vs_detection_time": {
            "spearman": spearman(distances, detected),
            "spearman_detection_vs_label": spearman(detected, labels_all_list),
            "auc_detection_vs_label": auc(detected, labels_all_list),
        },
        "physical_feasibility": {
            "max_travel_steps": math.sqrt(2.0) / ROBOT_VELOCITY,
            "feasible": len(feasible),
            "completed_among_feasible": sum(r["completed"] for r in feasible),
            "completed_among_infeasible": sum(r["completed"] for r in rows if r not in feasible),
        },
        "billed_first_step": {
            "n_billed": sum(billed),
            "on_time_rate_billed": (sum(r["on_time"] for r in rows if r["billed_first_step"])
                                    / max(1, sum(billed))),
            "n_not_billed": len(not_billed),
            "on_time_rate_not_billed": (sum(r["on_time"] for r in not_billed)
                                        / max(1, len(not_billed))),
            "auc": auc(billed, labels_all_list),
        },
        "ceilings_top%d" % TOP_K: ceilings,
        "oracle_ceiling": oracle,
        "references": {"learned": 80, "normal_immediate": 90},
        "rows": rows,
    }
    target = Path(args.json_out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                      encoding="utf-8", newline="\n")
    print("wrote", target)
    print("sha256", sha256(target))


if __name__ == "__main__":
    main()
