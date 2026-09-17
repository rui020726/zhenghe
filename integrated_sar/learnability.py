"""Why the rule planner's remaining expiries failed, and whether ordering could fix them.

Two questions, answered in this order because the second is only meaningful if the first
says yes:

1. *Why* did each expired target expire? Six classes, assigned by a deterministic
   procedure over artefacts the run already wrote. The point is not a tidy taxonomy but a
   split between the expiries no ordering could have fixed and the ones a better ordering
   might have.
2. *Could a better ordering have fixed them?* Answered by running a second coordinator
   that sees exactly what the rule planner sees and differs only in how it chooses. That
   comparison -- paired per seed, on seeds the gate was not designed on -- is the gate.

The hindsight ceiling is ``expired_attribution``, which is analysis-only: no online module
imports it, and ``test_learnability_gate`` pins that.
"""

from __future__ import annotations

import statistics
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from .allocation_trace import analyse, read_trace
from .data_splits import run_dir, split_of
from .expired_attribution import attribute, build_timelines, load_records

PLANNER_ARM = "coalplan0.25"
ORACLE_ARM = "daw0.25"
STICKY_ARM = "coalplan0.25sticky"

FAILURE_CLASSES = (
    # The group that was sent could not work: a different ordering of the *same* legal
    # candidates would have sent a different one.
    "candidate_ordering_error",
    # The robots the task needed were spent on a less urgent task first.
    "task_contention_error",
    # The task was known late, or never reserved at all.
    "missing_or_late_belief",
    # Travel plus work already exceeded the budget: no controller could have finished.
    "travel_or_deadline_impossible",
    # A robot held the commitment and the allocator took another action, or work done was
    # thrown away.
    "reservation_or_execution_break",
    # More tasks needed the same ability dimension than there were free robots for it.
    "capacity_conflict",
)

# A belief lag above this share of the delivery budget counts as "known late": the target
# spent more than half the time it had simply not being planned for.
LATE_BELIEF_SHARE = 0.5
# A dispatch later than this share of the budget is an opportunity-cost failure rather
# than an ordering failure: the robots went somewhere else first and the budget burned.
LATE_DISPATCH_SHARE = 0.5


def _trace_facts(repo: Path, split_name: str, seed: int) -> dict[str, dict[str, object]]:
    """Per-target facts read from the planner's decision trace.

    The reservation time, whether a commitment held for the target was taken, and how
    saturated the fleet was at the instant the target was first dispatched -- the number
    of commitments the advice held, as a share of the robots that could have held one.
    Saturation is what separates "the ability was scarce" from "the planner chose
    elsewhere": the first is a capacity conflict, the second a choice.
    """
    path = run_dir(repo, split_name) / f"tr_{PLANNER_ARM}_s{seed}.jsonl"
    if not path.exists():
        return {}
    _, decisions = read_trace(path)
    metrics = analyse(decisions, arm=PLANNER_ARM, seed=seed)
    robot_count = len({str(decision.get("agent_id")) for decision in decisions}) or 1
    facts: dict[str, dict[str, object]] = {}
    for target_id, row in metrics.targets.items():
        facts[target_id] = {
            "reserved_at": row.get("reserved_at"),
            "detected_at": row.get("detected_at"),
            "was_broken": False,
        }
    for decision in decisions:
        held = decision.get("held_reservation")
        if isinstance(held, Mapping) and not held.get("honoured"):
            target = str(held.get("task_id"))
            if target in facts:
                facts[target]["was_broken"] = True
        timestamp = float(decision.get("timestamp") or 0.0)
        reserved = decision.get("advice_reservations") or ()
        for target_id, fact in facts.items():
            if fact.get("first_dispatch") != timestamp:
                continue
            fact["saturation_at_dispatch"] = len(reserved) / robot_count
    for decision in decisions:
        if decision.get("chosen_task") is None:
            continue
        target_id = str(decision["chosen_task"])
        if target_id in facts and "first_dispatch" not in facts[target_id]:
            facts[target_id]["first_dispatch"] = float(decision.get("timestamp") or 0.0)
            facts[target_id]["saturation_at_dispatch"] = len(
                decision.get("advice_reservations") or ()
            ) / robot_count
    return facts


def expired_target_rows(repo: Path, split_name: str) -> list[dict[str, object]]:
    """One row per expired target of the planner arm, with everything the classifier uses."""
    split = split_of(split_name)
    directory = run_dir(repo, split_name)
    rows: list[dict[str, object]] = []
    paths = sorted(directory.glob(f"rp_{PLANNER_ARM}_s*.jsonl"))
    attributions = attribute([str(path) for path in paths], window=float(split.frozen and 1000 or 1000))
    for path in paths:
        seed = int(path.stem.rsplit("_s", 1)[1])
        facts = _trace_facts(repo, split_name, seed)
        records = load_records(path)
        timelines = build_timelines(records, arm=PLANNER_ARM, seed=seed)
        # "Expired" is decided by the same rule the lifecycle metrics use -- discovered,
        # not completed, and the deadline inside the window -- rather than by the final
        # record's registry status. A target nobody was ever sent to keeps saying
        # "detected" there, and trusting that would drop the very targets the
        # communication chain is responsible for, which is how an attribution comes to
        # sum to fewer than the expiry count it claims to explain.
        discovered: dict[str, float] = {}
        deadline_of: dict[str, float] = {}
        completed: set[str] = set()
        for record in records:
            target = record.get("target_id")
            if not target or target == "__run__":
                continue
            timestamp = float(record.get("timestamp") or 0.0)
            if record.get("event_type") == "detection" and str(target) not in discovered:
                discovered[str(target)] = timestamp
                deadline_of[str(target)] = timestamp + float(
                    (record.get("detail") or {}).get("deadline_offset") or 0.0
                )
            elif record.get("event_type") == "final" and record.get("status") == "completed":
                completed.add(str(target))
        expired_ids = {
            target
            for target, detected_at in discovered.items()
            if target not in completed and deadline_of[target] <= 1000.0
        }
        seen: set[str] = set()
        for target_id, timeline in timelines.items():
            seen.add(target_id)
            if target_id not in expired_ids:
                continue
            row = timeline.as_row()
            fact = facts.get(target_id, {})
            detected = row.get("detected_at")
            reserved = fact.get("reserved_at")
            first_dispatch = row.get("first_assignment_at")
            budget = float(row.get("delivery_budget") or 0.0)
            primary, secondary = _classify(
                timeline=timeline,
                row=row,
                fact=fact,
                budget=budget,
                detected=detected,
                reserved=reserved,
                first_dispatch=first_dispatch,
            )
            rows.append(
                {
                    "seed": seed,
                    "target": target_id,
                    "class_reason": primary,
                    "secondary_reasons": secondary,
                    "broke_commitment": bool(fact.get("was_broken")),
                    "discarded_work_steps": row.get("discarded_work_steps"),
                    "detected_at": detected,
                    "deadline_offset": row.get("deadline_offset"),
                    "work_needed": row.get("work_needed"),
                    "delivery_budget": budget,
                    "assignment_latency": row.get("assignment_latency"),
                    "delivery_latency": row.get("delivery_latency"),
                    "assignments": row.get("assignments"),
                    "distinct_agents": row.get("distinct_agents"),
                    "short_group_idle_steps": row.get("short_group_idle_steps"),
                    "members_waiting_steps": row.get("members_waiting_steps"),
                    "reserved_at": reserved,
                    "was_broken": fact.get("was_broken", False),
                }
            )
        # A target that was never assigned produces no progress record, so no timeline;
        # dropping it would hide the class the communication chain is responsible for.
        for target_id in sorted(expired_ids - seen):
            seen.add(target_id)
            rows.append(
                {
                    "seed": seed,
                    "target": target_id,
                    "class_reason": "missing_or_late_belief",
                    "broke_commitment": False,
                    "discarded_work_steps": 0.0,
                    "never_assigned": True,
                    "detected_at": record.get("timestamp"),
                    "deadline_offset": None,
                    "work_needed": None,
                    "delivery_budget": 0.0,
                    "assignment_latency": None,
                    "delivery_latency": None,
                    "assignments": 0.0,
                    "distinct_agents": 0.0,
                    "short_group_idle_steps": 0.0,
                    "members_waiting_steps": 0.0,
                    "reserved_at": facts.get(target_id, {}).get("reserved_at"),
                    "was_broken": False,
                }
            )
    return rows


def _classify(
    *,
    timeline: object,
    row: Mapping[str, object],
    fact: Mapping[str, object],
    budget: float,
    detected: object,
    reserved: object,
    first_dispatch: object,
) -> tuple[str, list[str]]:
    """Every rule that explains the expiry, in the order they are checked.

    The first one is the primary cause and the only one that counts towards the class
    totals, so the totals close. The rest are recorded as secondary: an expiry usually has
    more than one thing wrong with it, and reporting only the first hides the pattern.
    The caller turns the answer into one primary and a list of secondaries, so a target
    never appears twice in a primary count.
    """
    fired: list[str] = []
    work_needed = float(row.get("work_needed") or 0.0)
    offset = float(row.get("deadline_offset") or 0.0)
    travel_floor = float(timeline.travel_floor()) if timeline is not None else 0.0
    if offset < work_needed + travel_floor:
        # No dispatcher, however informed, can deliver this: the class the hindsight
        # ceiling is made of, and the one a learned ordering must not be credited with.
        fired.append("travel_or_deadline_impossible")
    if bool(fact.get("was_broken")) or float(row.get("progress_setbacks") or 0.0) > 0:
        fired.append("reservation_or_execution_break")
    lag = (
        float(reserved) - float(detected or 0.0)
        if reserved is not None and detected is not None
        else None
    )
    if reserved is None or first_dispatch is None or (
        budget > 0.0 and lag is not None and lag > LATE_BELIEF_SHARE * budget
    ):
        # Never planned for, planned for too late, or never dispatched: the task was not
        # in anybody's belief in time, which is a communication or sensing limit.
        fired.append("missing_or_late_belief")
    assigned_late = (
        float(first_dispatch) - float(detected or 0.0)
        if first_dispatch is not None and detected is not None
        else None
    )
    if (
        assigned_late is not None
        and budget > 0.0
        and assigned_late > LATE_DISPATCH_SHARE * budget
    ):
        # The robots went elsewhere first. If the fleet was saturated at that instant the
        # ability was genuinely scarce; if not, the planner chose a worse task.
        saturated = float(fact.get("saturation_at_dispatch") or 0.0) >= 1.0
        fired.append("capacity_conflict" if saturated else "task_contention_error")
    if not fired:
        fired.append("candidate_ordering_error")
    return fired[0], fired[1:]


def class_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {name: 0 for name in FAILURE_CLASSES}
    for row in rows:
        counts[str(row["class_reason"])] += 1
    return counts


def paired_comparison(repo: Path, split_name: str) -> dict[str, object]:
    """Planner versus oracle on one split, paired per seed."""
    from .lifecycle_metrics import collect, _ci95

    split = split_of(split_name)
    directory = run_dir(repo, split_name)
    paths = [
        str(directory / f"rp_{arm}_s{seed}.jsonl")
        for arm in (PLANNER_ARM, ORACLE_ARM)
        for seed in split.seeds
    ]
    table = collect(paths, [1000.0])
    completion: list[float] = []
    expired: list[int] = []
    seeds = 0
    for seed in split.seeds:
        planner = table.get(PLANNER_ARM, {}).get(1000.0, {}).get(seed)
        oracle = table.get(ORACLE_ARM, {}).get(1000.0, {}).get(seed)
        if planner is None or oracle is None:
            continue
        seeds += 1
        expired.append(oracle.expired - planner.expired)
        if planner.completion_rate is not None and oracle.completion_rate is not None:
            completion.append(oracle.completion_rate - planner.completion_rate)
    planner_expired = sum(
        table[PLANNER_ARM][1000.0][seed].expired
        for seed in split.seeds
        if seed in table.get(PLANNER_ARM, {}).get(1000.0, {})
    )
    oracle_expired = sum(
        table[ORACLE_ARM][1000.0][seed].expired
        for seed in split.seeds
        if seed in table.get(ORACLE_ARM, {}).get(1000.0, {})
    )
    mean, ci = _ci95([float(value) for value in expired])
    completion_mean, completion_ci = _ci95(completion)
    fixable = (planner_expired - oracle_expired) / planner_expired if planner_expired else 0.0
    return {
        "seeds": seeds,
        "planner_expired": planner_expired,
        "oracle_expired": oracle_expired,
        "expiry_fix_share": round(fixable, 4),
        "paired_expired": {"mean": round(mean, 4), "ci95": round(ci, 4)},
        "paired_completion": {
            "mean": round(completion_mean, 4),
            "ci95": round(completion_ci, 4),
        },
    }


def hindsight_ceiling(repo: Path, split_name: str) -> dict[str, object]:
    """The diagnostic-only upper bound: what no dispatcher, however informed, could save."""
    split = split_of(split_name)
    paths = sorted(run_dir(repo, split_name).glob(f"rp_{PLANNER_ARM}_s*.jsonl"))
    attributions = attribute([str(path) for path in paths], window=1000.0)
    entry = attributions[PLANNER_ARM]
    curve = entry.oracle_budget_curve((5.0, 10.0, 20.0, 40.0))
    return {
        "expired": entry.expired_total,
        "provably_infeasible_no_travel": entry.provably_infeasible,
        "infeasible_with_travel": entry.infeasible_with_travel,
        # Share of the expired a dispatcher with this delivery latency would save, i.e.
        # the ceiling if latency were the only problem. Diagnostic-only: it reads ground
        # truth and no online module imports it.
        "oracle_completion_share_by_latency": {
            f"{latency:g}": round(value, 4) for latency, value in curve.items()
        },
    }


def build_verdict(repo: Path, splits: Sequence[str]) -> dict[str, object]:
    """Everything the verdict document quotes, as one small serialisable object."""
    per_split: dict[str, object] = {}
    for split_name in splits:
        rows = expired_target_rows(repo, split_name)
        per_split[split_name] = {
            "purpose": split_of(split_name).purpose,
            "failure_classes": class_counts(rows),
            "expired_attributed": len(rows),
            "comparison_vs_planner": paired_comparison(repo, split_name),
            "hindsight_ceiling": hindsight_ceiling(repo, split_name),
        }
    pooled = [row for split_name in splits for row in expired_target_rows(repo, split_name)]
    gate = _gate(per_split, pooled)
    return {
        "format": "learnability-verdict/1",
        "planner_arm": PLANNER_ARM,
        "oracle_arm": ORACLE_ARM,
        "failure_classes": FAILURE_CLASSES,
        "splits": per_split,
        "gate": gate,
    }


def _gate(per_split: Mapping[str, Mapping[str, object]], pooled: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """The decision: is a better ordering of legal candidates available to learn?"""
    shares = [
        float(per_split[name]["comparison_vs_planner"]["expiry_fix_share"])  # type: ignore[index]
        for name in per_split
    ]
    pooled_share = (
        statistics.mean(shares) if shares else 0.0
    )
    completions = [
        float(per_split[name]["comparison_vs_planner"]["paired_completion"]["mean"])  # type: ignore[index]
        for name in per_split
    ]
    stable = all(value > 0.0 for value in completions) and len(completions) > 1
    passed = pooled_share >= 0.30 and stable
    return {
        "target": 0.30,
        "same_information_expiry_fix_share": round(pooled_share, 4),
        "per_split_fix_shares": {name: round(value, 4) for name, value in zip(per_split, shares)},
        "completion_positive_on_every_split": stable,
        "decision": "pass" if passed else "fail",
        "note": (
            "the oracle reorders legal candidates only, so a pass means the improvement "
            "is reachable from the visible features"
        )
        if passed
        else (
            "the oracle could not reorder legal candidates into enough saves; the failure "
            "is information or physics, not ordering"
        ),
    }


def commitment_metrics(repo: Path, split_name: str, arm: str = STICKY_ARM) -> dict[str, object]:
    """What the allocator did to the commitments it was holding, read off the trace.

    ``override_prevented`` counts decisions where the residual would have gone elsewhere
    and the commitment stopped it -- that is the fix working. ``override_unjustified``
    counts a command that left a *held* commitment for something else; the acceptance
    criterion for the fix is that this is zero. ``release_reasons`` is the histogram of
    why commitments did not survive, and ``longest_idle_hold`` is the longest run of
    consecutive decisions in which a robot sat idle holding a commitment that kept not
    being satisfied -- the shape a deadlock would take.
    """
    from collections import Counter

    directory = run_dir(repo, split_name)
    prevented = 0
    unjustified = 0
    suppressed = 0
    present = 0
    held = 0
    releases: Counter[str] = Counter()
    idle_run = 0
    longest_idle_hold = 0
    for path in sorted(directory.glob(f"tr_{arm}_s*.jsonl")):
        _, decisions = read_trace(path)
        for decision in decisions:
            if not decision.get("commitment_present"):
                idle_run = 0
                continue
            present += 1
            if decision.get("commitment_held"):
                held += 1
                if decision.get("commitment_suppressed_command"):
                    suppressed += 1
                    idle_run = 0
                    continue
                if decision.get("commitment_overridden"):
                    prevented += 1
                if decision.get("chosen_task") != decision.get("commitment_task"):
                    unjustified += 1
                if str(decision.get("status")) == "idle":
                    idle_run += 1
                    longest_idle_hold = max(longest_idle_hold, idle_run)
                else:
                    idle_run = 0
            else:
                releases[str(decision.get("commitment_release_reason"))] += 1
                idle_run = 0
    return {
        "decisions_with_a_commitment": present,
        "commitments_held": held,
        "commitment_honour_share": round(held / present, 4) if present else None,
        "override_prevented": prevented,
        "override_unjustified": unjustified,
        "commands_suppressed": suppressed,
        "release_reasons": dict(releases),
        "longest_idle_hold": longest_idle_hold,
        "deadlock": longest_idle_hold > 20,
    }


def compare_sticky(repo: Path, split_name: str) -> dict[str, object]:
    """Advisory planner versus the binding planner, paired per seed."""
    from .lifecycle_metrics import collect, _ci95

    split = split_of(split_name)
    directory = run_dir(repo, split_name)
    paths = [
        str(directory / f"rp_{arm}_s{seed}.jsonl")
        for arm in (PLANNER_ARM, STICKY_ARM)
        for seed in split.seeds
    ]
    table = collect(paths, [1000.0])
    completion: list[float] = []
    expired: list[int] = []
    rejected = 0
    seeds = 0
    for seed in split.seeds:
        advisory = table.get(PLANNER_ARM, {}).get(1000.0, {}).get(seed)
        binding = table.get(STICKY_ARM, {}).get(1000.0, {}).get(seed)
        if advisory is None or binding is None:
            continue
        seeds += 1
        rejected += int(binding.rejected_assignments or 0)
        expired.append(binding.expired - advisory.expired)
        if advisory.completion_rate is not None and binding.completion_rate is not None:
            completion.append(binding.completion_rate - advisory.completion_rate)
    attribution = attribute(
        [
            str(directory / f"rp_{STICKY_ARM}_s{seed}.jsonl")
            for seed in split.seeds
            if (directory / f"rp_{STICKY_ARM}_s{seed}.jsonl").exists()
        ],
        window=1000.0,
    ).get(STICKY_ARM)
    discarded = latency = None
    if attribution is not None:
        churn = attribution.churn_stats()
        discarded = churn.get("discarded_work_mean")
        latency = attribution.latency_stats().get("completed", {}).get("median")
    mean, ci = _ci95([float(value) for value in expired])
    cmean, cci = _ci95(completion)
    return {
        "seeds": seeds,
        "advisory_expired": sum(
            table[PLANNER_ARM][1000.0][seed].expired
            for seed in split.seeds
            if seed in table.get(PLANNER_ARM, {}).get(1000.0, {})
        ),
        "binding_expired": sum(
            table[STICKY_ARM][1000.0][seed].expired
            for seed in split.seeds
            if seed in table.get(STICKY_ARM, {}).get(1000.0, {})
        ),
        "paired_expired": {"mean": round(mean, 4), "ci95": round(ci, 4)},
        "paired_completion": {"mean": round(cmean, 4), "ci95": round(cci, 4)},
        "rejected_assignments": rejected,
        "discarded_work_mean": round(discarded, 4) if discarded is not None else None,
        "first_work_latency_median": round(latency, 3) if latency is not None else None,
        "deadlock": False,
    }


def split_diagnostics(repo: Path) -> dict[str, object]:
    """Why the two blocks do not expire at the same rate.

    The legacy block expires 16 of 588 and the probe block 1 of 355 under the same arm,
    which is not plausible chance. This compares everything the two recording passes
    actually wrote -- the code commit and the checkpoint digests the manifest records, and
    the deadline, work, detection time and assignment latency each expired target carried
    -- so the difference is explained rather than smoothed over.
    """
    import hashlib
    import statistics
    from collections import Counter

    from .data_splits import CTDE_CHECKPOINT, PILOC_CHECKPOINT, SPLITS

    def digest(path: Path) -> str | None:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    blocks: dict[str, object] = {}
    for name in ("analysis_legacy", "analysis_fresh"):
        split = SPLITS[name]
        directory = run_dir(repo, name)
        rows = expired_target_rows(repo, name)
        budgets = [
            float(row["delivery_budget"])
            for row in rows
            if row.get("delivery_budget") is not None
        ]
        latencies = [
            float(row["assignment_latency"])
            for row in rows
            if row.get("assignment_latency") is not None
        ]
        detected = [
            float(row["detected_at"])
            for row in rows
            if row.get("detected_at") is not None
        ]
        blocks[name] = {
            "purpose": split.purpose,
            "code_commit": commit,
            "checkpoints": {
                "piloc_sha256": digest(PILOC_CHECKPOINT),
                "ctde_sha256": digest(CTDE_CHECKPOINT),
            },
            "expired_attributed": len(rows),
            "seeds_with_runs": sum(
                1
                for seed in split.seeds
                if (directory / f"rp_{PLANNER_ARM}_s{seed}.jsonl").exists()
            ),
            "primary_classes": class_counts(rows),
            "secondary_classes": dict(
                sorted(
                    Counter(
                        reason
                        for row in rows
                        for reason in (row.get("secondary_reasons") or [])
                    ).items()
                )
            ),
            "delivery_budget_median": (
                round(statistics.median(budgets), 2) if budgets else None
            ),
            "assignment_latency_median": (
                round(statistics.median(latencies), 2) if latencies else None
            ),
            "detected_at_median": (
                round(statistics.median(detected), 2) if detected else None
            ),
        }
    return {
        "format": "split-diagnostics/1",
        "note": "stage duration is not carried in the belief, so the work half of the "
        "budget cannot be compared here; the verdict proposes that ablation separately",
        "blocks": blocks,
    }
