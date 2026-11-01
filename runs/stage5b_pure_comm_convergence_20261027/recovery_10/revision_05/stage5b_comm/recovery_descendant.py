"""Completed finite recovery segment -> next finite segment. Offline prepare only.
Initial candidate/normal-parent validators remain unchanged. A descendant explicitly pins
a completed parent and the original accepted prefix; no initial 18k source re-collection.
All CP pins are bytes-only until a separate G3 loads each payload once. No auto-approval.
"""
from __future__ import annotations
import argparse
import hashlib
import math
import re
import statistics
from pathlib import Path
import types
from . import recovery_runtime as RT, recovery_protocol as R, recovery_seed as S
from . import recovery_cutover as X
from . import constants as C, compact, convergence
from .paths import Layout

SCHEMA = "stage5b-completed-recovery-descendant/1"
CONTEXT_SCHEMA = "stage5b-recovery-descendant-context/1"
#: Only the descendant stage descriptor may exceed the frozen 10 MB protocol-artifact cap.
DESCRIPTOR_CAP_BYTES = 64 * 1024 * 1024
DESCRIPTOR = "protocol/RECOVERY_DESCENDANT_STAGE.json"
ADMISSION = "protocol/RECOVERY_DESCENDANT_ADMISSION.json"
LOSS_REVIEW_RELATIVE = "execution/recovery/loss_stability_review.json"
LOSS_REVIEW_SCHEMA = "stage5b-loss-stability-review/1"
LOSS_METRICS = ("policy_surrogate", "value_loss_task", "value_loss_cost")
INITIAL_ROOT = Path("D:/code/整合/zhenghe-round42/runs/stage5b_pure_comm_convergence_20261027/recovery_01/revision_01")
RECOVERY02_ROOT = Path("D:/code/整合/zhenghe-round42/runs/stage5b_pure_comm_convergence_20261027/recovery_02/revision_01")
RECOVERY02_RECEIPT_ID = "recovery-train-28840-1793195293700174800"
RECOVERY02_CHAT_POLL_REF = "user message COMM26_CONTROL_A_20261025_131328: runner assistant chat poll result"
RECONSTRUCTION_SCHEMA = "stage5b-recovery02-source-reconstruction/1"
GENERIC_RECONSTRUCTION_SCHEMA = "stage5b-completed-parent-provenance/1"
PREPARED_OBSERVER_MODE = "CONTEMPORANEOUS_OBSERVER_PREPARATION"
OBSERVERLESS_POSTHOC_MODE = "POSTHOC_OBSERVER_METADATA_UNKNOWN_NOT_CAPTURED"
TRAINING_INTERPRETER_UNKNOWN = "UNKNOWN_NOT_RECORDED"
INITIAL_CONTEXT_SHA = "527b35bca3020eb65968623023e5c96fbc469c7bbe0759b24d507b760e552857"
SCIENCE_SHA = "be5915d321c69d07158f045866c43205759d315d603bb8e0425673d2358cdb40"
ROSTER = tuple((a, s) for a, s, _ in R.RUN_CURSORS)


def canonical(root):
    root = Path(root).resolve()
    RT.need(root == Path(__file__).resolve().parents[1],
            "descendant root must match its own canonical source root")
    return root


class Reads:
    """One bytes/hash/strict-JSON read per path; retain the exact evidence closure."""
    def __init__(self):
        self.pins, self.raws = {}, {}

    def raw(self, path, pin=None, root=None):
        path = Path(path).resolve()
        if root is not None:
            RT.evidence.inside(root, path)
        key = str(path)
        if key not in self.raws:
            raw = path.read_bytes()
            self.raws[key] = raw
            self.pins[key] = {"path": key, "bytes": len(raw),
                              "sha256": RT.sha256_bytes(raw)}
        if pin is not None:
            RT.exact_keys(pin, RT.PIN_KEYS, "parent evidence pin")
            actual = self.pins[key]
            RT.need(type(pin["bytes"]) is int and pin["bytes"] >= 0
                    and Path(pin["path"]).resolve() == path
                    and actual["bytes"] == pin["bytes"]
                    and actual["sha256"] == RT.digest(pin["sha256"]),
                    "parent evidence pin drift: " + key)
        return self.raws[key]

    def json(self, path, pin=None, root=None):
        return RT.strict_json_bytes(self.raw(path, pin, root), str(path))

    def pinned(self, pin, root=None):
        return self.json(pin["path"], pin, root)


def validate_terminal(summary, receipt, raw):
    RT.need(summary.get("schema") == "stage5b-recovery-train-terminal-summary/1",
            "unknown parent terminal-summary schema")
    for key in ("child_exit", "tool_exit", "cli_returncode", "drift_count"):
        RT.need(type(summary.get(key)) is int and summary[key] == 0,
                "parent training needs actual zero " + key)
    RT.need(summary.get("stage_complete") is True
            and summary.get("global_stop") is False
            and summary.get("source_context_auth_drift") == [],
            "parent is incomplete/stopped/drifted")
    RT.need(receipt.get("schema") == "stage5b-phase-receipt/1"
            and receipt.get("phase") == "recovery-train"
            and receipt.get("stage") == "terminal"
            and type(receipt.get("returncode")) is int
            and receipt["returncode"] == 0,
            "parent terminal CLI receipt is not successful training")
    RT.need(type(raw.get("exit_code")) is int and raw["exit_code"] == 0,
            "parent raw outer tool exit is missing/nonzero")
    output = raw.get("output")
    RT.need(type(output) is str and "CHILD_EXIT=0" in output
            and "DRIFT_COUNT=0" in output, "parent raw child/drift terminal markers missing")
    lines = output.replace("\r", "").splitlines()
    cli = [RT.strict_json_bytes(line.encode(), "terminal CLI output")
           for line in lines if line.startswith('{"phase": "recovery-train"')]
    RT.need(len(cli) == 1 and cli[0].get("returncode") == 0
            and cli[0].get("receipt_id") == receipt.get("receipt_id"),
            "raw train terminal does not match CLI receipt ID")
    detail = receipt.get("detail") or {}
    RT.need(detail.get("status") == "FINITE_RECOVERY_STAGE_COMPLETE",
            "parent stage detail is not complete")
    return detail


def validate_closed_run(run, result, telemetry, segment_bound=120):
    arm, seed = run["arm"], run["training_seed"]
    start, cap_end = run["start_episode"], run["end_episode_cap"]
    n, end = result.get("episodes"), result.get("terminal_episode")
    RT.need(type(n) is int and 0 < n <= cap_end - start <= segment_bound and n % 4 == 0
            and type(end) is int and end == start + n
            and result.get("status") == "ok",
            "parent run does not close an exact finite prefix")
    RT.need(result.get("converged") is True or end == cap_end,
            "unfinished non-converged parent run")
    RT.need(telemetry.get("schema") == RT.TELEMETRY_SCHEMA
            and telemetry.get("arm") == arm and telemetry.get("training_seed") == seed
            and telemetry.get("start_episode") == start
            and telemetry.get("terminal_episode") == end
            and telemetry.get("end_episode_cap") == cap_end,
            "parent telemetry run/cursor mismatch")
    ep = telemetry.get("episodes")
    up = telemetry.get("updates")
    RT.need(type(ep) is list and [x.get("episode_index") for x in ep]
            == list(range(start, end)), "parent telemetry episode gap/duplicate")
    RT.need(type(up) is list and [x.get("update_index") for x in up]
            == list(range(start // 4, end // 4)), "parent telemetry update gap/duplicate")
    for x in up:
        RT.evidence.validate_update(x)
    RT.finite_tree(telemetry)
    return end


def merge_prefix(rows, updates, new_rows, new_updates, start, end, run):
    gap = R.known_gap(*run)
    row_gap = R.known_row_gap(*run)
    missing_rows = set(range(*row_gap["episodes"])) if row_gap else set()
    expected_rows = [x for x in range(start) if x not in missing_rows]
    RT.need(sorted(rows) == expected_rows,
            "ancestor history differs from the exact registered row gap for %r start=%d "
            "actual=%d expected=%d missing=%r"
            % (run, start, len(rows), len(expected_rows), sorted(missing_rows)[:8]))
    RT.need(set(new_rows) == set(range(start, end)), "closed parent row gap/duplicate")
    RT.need(not (set(rows) & set(new_rows)), "parent history repeats an episode")
    RT.need(set(new_updates) == set(range(start // 4, end // 4)),
            "closed parent journal update gap")
    RT.need(not (set(updates) & set(new_updates)), "parent repeats an update")
    combined_rows, combined_updates = dict(rows), dict(updates)
    combined_rows.update(new_rows)
    combined_updates.update(new_updates)
    expected = set(range(end // 4))
    missing = set(range(*gap["updates"])) if gap else set()
    RT.need(set(combined_updates) == expected - missing,
            "unexpected diagnostic gap/conflict beyond registered gaps")
    for x in combined_updates.values():
        RT.evidence.validate_update(x)
    return combined_rows, combined_updates


def next_runs(runs, states, loss_reviews=None, *, include_completion_counts=False, include_minimum_target=False, target_completed_episodes=None, retired_runs=frozenset()):
    """Build the next interval; long stages use actual completed episodes, never update indices."""
    RT.need(type(include_completion_counts) is bool,
            "completion-counter mode must be an exact boolean")
    RT.need(type(include_minimum_target) is bool,
            "minimum-target mode must be an exact boolean")
    if target_completed_episodes is not None:
        from . import long_stage as L
        target_completed_episodes, _interval = L.validate_parameters(target_completed_episodes)
        RT.need(include_completion_counts, "long stage requires actual completed episode counts")
    loss_reviews = loss_reviews or {}
    result = []
    for run in runs:
        key = RT.normalized_run(run["arm"], run["training_seed"])
        state = states[key]
        cursor = RT.integer(state["cursor"], "verified parent cursor")
        update = RT.integer(state["next_update"], "last journal next_update")
        RT.need(cursor % 4 == 0 and update == cursor // 4,
                "parent cursor/update mismatch")
        certified = state.get("certified")
        RT.need(type(certified) is bool, "task certificate disposition must be boolean")
        inherited_stable = state.get("loss_stability_stable") is True
        review = loss_reviews.get(key)
        newly_stable = review is not None and review.get("decision") == "LOSS_STABLE_CONFIRMED"
        stable = inherited_stable or newly_stable
        completed = None
        missing = None
        if include_completion_counts:
            completed = RT.integer(state.get("completed_episode_count"),
                                   "verified completed episode count")
            missing = RT.integer(state.get("registered_missing_episode_count"),
                                 "registered missing episode count")
            RT.need(completed + missing == cursor,
                    "episode cursor/completed-count accounting does not close")
        if target_completed_episodes is None:
            retired = certified and stable
            step = 0 if retired else 120
        else:
            step = max(0, target_completed_episodes - completed)
            retired = step == 0
            if key in retired_runs:
                # This revision deliberately stops a run at its parent completed target.
                step = 0
                retired = True
        review_record = review or state.get("loss_stability_review")
        if not stable and review_record is None:
            review_record = {"status": "REVIEW_REQUIRED" if certified
                             else "TASK_CERTIFICATE_PENDING"}
        output = {
            "arm": key[0], "training_seed": key[1], "start_episode": cursor,
            "end_episode_cap": cursor + step,
            "new_world_cap": step, "active": not retired,
            "next_update": update, "donor": state["donor"],
            "task_certified": certified, "loss_stability_stable": stable,
            "loss_stability_review": review_record,
            "latest_telemetry_sha256": state.get("latest_telemetry_sha256"),
        }
        if include_completion_counts:
            output.update({
                "episode_cursor_before_stage": cursor,
                "completed_episode_count_before_stage": completed,
                "registered_missing_episode_count_before_stage": missing,
            })
            if include_minimum_target:
                projected = completed + step
                output.update({
                    "minimum_episode_target": 5000,
                    "projected_completed_episode_count_after_stage": projected,
                    "episodes_below_minimum_5000_after_stage": max(0, 5000 - projected),
                })
        if target_completed_episodes is not None:
            output.update({
                "target_completed_episodes": target_completed_episodes,
                "target_basis": "ACTUAL_COMPLETED_EPISODE_COUNT",
                "checkpoint_interval_episodes": L.CHECKPOINT_INTERVAL_EPISODES,
            })
        if key in retired_runs:
            # Only a deliberately retired run carries this marker, so ancestor run entries
            # re-derived without retirement stay byte-identical.
            output["revision_retired"] = True
        result.append(output)
    RT.need([(x["arm"], x["training_seed"]) for x in result] == list(ROSTER),
            "next plan must retain the exact six-run roster")
    return result


def legacy_next_runs(runs, states):
    """Read pre-loss-review descendant contexts without rewriting their task-only policy."""
    output = []
    for run in runs:
        key = RT.normalized_run(run["arm"], run["training_seed"])
        state = states[key]
        cursor = RT.integer(state["cursor"], "legacy parent cursor")
        update = RT.integer(state["next_update"], "legacy parent next_update")
        retired = state.get("certified") is True
        output.append({"arm": key[0], "training_seed": key[1], "start_episode": cursor,
                       "end_episode_cap": cursor if retired else cursor + 120,
                       "new_world_cap": 0 if retired else 120, "active": not retired,
                       "next_update": update, "donor": state["donor"]})
    return output

def next_stage_index(parent_index):
    RT.need(type(parent_index) is int and parent_index >= 1,
            "parent stage index must be a positive exact integer")
    return parent_index + 1

def first_task_certificate_end(checks):
    """First endpoint at which the unchanged formal evaluator certified convergence."""
    for count in range(2, len(checks) + 1):
        if convergence.is_converged(checks[:count]):
            return checks[count - 1]["end_episode"]
    return None

def validate_task_termination(run, result, checks, end, loss_review_enabled, long_stage=False):
    """Keep the formal task event immutable while validating a later stage endpoint."""
    latest_streak = convergence.is_converged(checks)
    first_end = first_task_certificate_end(checks)
    if not loss_review_enabled:
        certified = latest_streak
        cert_end = checks[-1]["end_episode"] if certified else None
        if certified:
            RT.need(cert_end == end,
                    "legacy certified parent did not stop at its confirming check")
        RT.need(type(result.get("converged")) is bool
                and result["converged"] == certified,
                "parent run certification lacks two strict adjacent checks")
        return certified, cert_end

    inherited = run.get("task_certified") is True
    certified = inherited or first_end is not None
    cert_end = run.get("task_certificate_end_episode") if inherited else first_end
    RT.need(type(result.get("converged")) is bool
            and result["converged"] == certified
            and result.get("task_certificate_end_episode") == cert_end,
            "continued parent did not preserve its original task certificate")
    if inherited:
        RT.need(type(cert_end) is int and first_end == cert_end and end >= cert_end,
                "continued parent task-certificate source/cursor mismatch")
        continued = end > cert_end
        RT.need(result.get("continued_after_task_certificate") is continued,
                "continued-parent disposition marker mismatch")
        if result.get("stopped_on_task_certificate") is True:
            RT.need(latest_streak and checks[-1]["end_episode"] == end,
                    "continued run stopped without a new strict task check")
        else:
            RT.need(end == run["end_episode_cap"],
                    "continued run ended early without a new task check")
    elif certified:
        if long_stage:
            # A long target never stops on a certificate; the event stays immutable and the
            # run closes its cap instead.
            RT.need(end == run["end_episode_cap"] and end >= cert_end
                    and result.get("stopped_on_task_certificate") is False
                    and result.get("continued_after_task_certificate") is (end > cert_end),
                    "new long-stage task certificate must be preserved while the run closes its target")
        else:
            RT.need(end == cert_end and result.get("stopped_on_task_certificate") is True
                    and result.get("continued_after_task_certificate") is False,
                    "new task certificate did not stop at its confirming check")
    else:
        RT.need(end == run["end_episode_cap"]
                and result.get("stopped_on_task_certificate") is False
                and result.get("continued_after_task_certificate") is False,
                "uncertified parent did not reach its finite stage cap")
    return certified, cert_end

def _loss_window_evidence(parent, key):
    state = parent.states[key]
    cursor = state["next_update"]
    records = parent.updates[key]
    windows, flattened = [], []
    for start in (cursor - 60, cursor - 40, cursor - 20):
        indices = list(range(start, start + 20))
        RT.need(all(index in records for index in indices),
                "loss review needs three complete contiguous 20-update windows")
        points = []
        summary = {"start_update": start, "end_update_exclusive": start + 20,
                   "metrics": {}}
        for index in indices:
            item = records[index]
            point = {"update_index": index}
            for metric in LOSS_METRICS:
                value = item.get(metric)
                RT.need(type(value) in (int, float) and math.isfinite(value),
                        "loss review window has a missing/nonfinite metric")
                point[metric] = float(value)
            points.append(point)
        for metric in LOSS_METRICS:
            values = [item[metric] for item in points]
            median = float(statistics.median(values))
            mad = float(statistics.median([abs(value - median) for value in values]))
            summary["metrics"][metric] = {"median": median, "mad": mad}
        windows.append(summary)
        flattened.extend(points)
    return windows, RT.canonical_digest(flattened)

def validate_loss_stability_review(review, parent, root):
    """Validate an optional human decision; calculate windows, never apply a loss threshold."""
    RT.exact_keys(review, {"schema", "status", "parent_context_sha256", "reviews"},
                   "loss review")
    RT.need(review["schema"] == LOSS_REVIEW_SCHEMA
            and review["status"] == "HUMAN_REVIEW_COMPLETE",
            "loss review schema/status is invalid")
    RT.need(RT.digest(review["parent_context_sha256"])
            == parent.context.context_pin["sha256"],
            "loss review is bound to another parent context")
    entries = review["reviews"]
    RT.need(type(entries) is list, "loss review entries must be a list")
    decisions = {}
    expected_keys = {"arm", "training_seed", "decision", "reviewer", "reviewed_utc",
                     "rationale", "acknowledged_nonmonotonicity", "windows"}
    for entry in entries:
        RT.exact_keys(entry, expected_keys, "loss review run entry")
        key = RT.normalized_run(entry["arm"], entry["training_seed"])
        RT.need(key not in decisions and key in parent.states,
                "loss review has duplicate/unknown run")
        RT.need(parent.states[key].get("task_certified") is True,
                "loss review may only disposition a formally task-certified run")
        RT.need(entry["decision"] in ("LOSS_STABLE_CONFIRMED", "CONTINUE_TRAINING"),
                "unknown human loss-review decision")
        RT.need(type(entry["reviewer"]) is str and entry["reviewer"].strip()
                and type(entry["reviewed_utc"]) is str and entry["reviewed_utc"].strip()
                and type(entry["rationale"]) is str and entry["rationale"].strip()
                and entry["acknowledged_nonmonotonicity"] is True,
                "human review needs reviewer, time, rationale, and PPO nonmonotonicity acknowledgement")
        windows, evidence_sha = _loss_window_evidence(parent, key)
        expected_ranges = [{"start_update": x["start_update"],
                            "end_update_exclusive": x["end_update_exclusive"]}
                           for x in windows]
        RT.need(entry["windows"] == expected_ranges,
                "loss review must cite the three latest nonoverlapping 20-update windows")
        decisions[key] = {**entry, "windows": windows,
                          "source_updates_sha256": evidence_sha}
    return decisions

def load_loss_stability_review(root, parent):
    """Absent review is conservative continuation; malformed/stale review fails closed."""
    path = Path(root) / LOSS_REVIEW_RELATIVE
    if not path.exists():
        return None, {}
    reader = Reads()
    body = reader.json(path, root=root)
    decisions = validate_loss_stability_review(body, parent, root)
    return reader.pins[str(path.resolve())], decisions

def _long_stage_target_for_runs(runs):
    targets = {run.get("target_completed_episodes") for run in runs}
    if len(targets) != 1:
        raise RT.RecoveryRuntimeError("seed plan mixes long-stage and ordinary targets")
    target = next(iter(targets))
    if target is None:
        return None
    from . import long_stage as L
    target, interval = L.validate_parameters(
        target, runs[0].get("checkpoint_interval_episodes"))
    for run in runs:
        retired_here = (run.get("active") is False and run.get("new_world_cap") == 0
                        and run.get("projected_completed_episode_count_after_stage")
                        == run.get("completed_episode_count_before_stage"))
        RT.need(run.get("target_completed_episodes") == target
                and run.get("target_basis") == "ACTUAL_COMPLETED_EPISODE_COUNT"
                and run.get("checkpoint_interval_episodes") == interval
                and (run.get("projected_completed_episode_count_after_stage") == target
                     or retired_here),
                "long-stage seed plan does not close the actual-completed target")
    return target


def seed_plan(runs, old_keys):
    RT.need(old_keys, "complete pinned old claim-key snapshot is required")
    target = _long_stage_target_for_runs(runs)
    all_keys, plan = set(old_keys), []
    for run in runs:
        if not run["active"]:
            continue
        arm, seed = run["arm"], run["training_seed"]
        for ep in range(run["start_episode"], run["end_episode_cap"]):
            scene = C.scene_for(seed, ep)
            # Exactly the frozen v1 mapping, independent of arm; no RNG object/draw.
            payload = "%s:sc-action:%d:%d" % (R.WORLD_SEED_NAMESPACE, seed, ep)
            action = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
            key = (arm, seed, scene, action)
            RT.need(key not in all_keys, "next-stage physical key was already claimed/planned")
            all_keys.add(key)
            plan.append({"arm": arm, "training_seed": seed, "episode_index": ep,
                         "scene_seed": scene, "action_rng_seed": action})
    cap = sum(x["new_world_cap"] for x in runs)
    RT.need(len(plan) == cap and (target is not None or cap <= 720),
            "finite stage exceeds its active-run cap")
    return plan


def validate_seed_plan(plan, runs):
    RT.need(type(plan) is list, "descendant seed plan is not a list")
    expected = seed_plan(runs, {("__sentinel__", 0, 0, 0)})
    RT.need(plan == expected, "seed plan is not the deterministic per-run finite interval")
    return plan


def _base(root, reader, seen):
    """Read only explicit ancestry; no ancestor search and no re-running initial admission."""
    root = Path(root).resolve()
    stage_index = 1
    RT.need(root not in seen, "parent context cycle")
    seen = seen | {root}
    path = root / RT.CONTEXT_RELATIVE
    body = reader.json(path, root=root)
    RT.need(body.get("g3_report") is None and body.get("world_authorization") is None,
            "base context must be immutable, with null execution sidecars")
    if body.get("schema") == CONTEXT_SCHEMA:
        descriptor = reader.pinned(body["descendant"], root)
        RT.need(descriptor.get("schema") == SCHEMA and descriptor.get("status") == "FROZEN",
                "unknown descendant descriptor")
        RT.need(Path(body["root"]).resolve() == root
                and descriptor["root"] == body["root"]
                and body["namespace"] == R.WORLD_SEED_NAMESPACE,
                "descendant root/namespace mismatch")
        previous = Path(descriptor["parent_context"]["path"]).resolve()
        RT.need(previous.name == Path(RT.CONTEXT_RELATIVE).name,
                "parent-context filename changed")
        reader.raw(previous, descriptor["parent_context"])
        parent = completed_parent(previous.parents[1], descriptor["parent_run_directory"],
                                  provenance_path=descriptor.get("parent_provenance_path"), seen=seen)
        # The pinned parent descriptor closes its own ancestry transitively. Do not
        # flatten every earlier ledger into each new descriptor (bounded JSON writer).
        expected_stage = next_stage_index(parent.context.stage_index)
        stage_index = expected_stage
        if "stage_index" not in descriptor and "loss_stability_review" not in descriptor:
            # Recovery_02 was frozen before loss review/stage numbering existed.
            expected_runs = legacy_next_runs(parent.runs, parent.states)
        else:
            review_pin = descriptor.get("loss_stability_review")
            review_decisions = {}
            if review_pin is not None:
                review_body = reader.pinned(review_pin, root=root)
                review_decisions = validate_loss_stability_review(review_body, parent, root)
            expected_runs = next_runs(
                parent.runs, parent.states, review_decisions,
                include_completion_counts=parent.context.stage_index >= 5,
                include_minimum_target=expected_stage >= 7,
                target_completed_episodes=descriptor.get("target_completed_episodes"),
                retired_runs=frozenset(
                    RT.normalized_run(item.split(":")[0], int(item.split(":")[1]))
                    for item in descriptor.get("retired_runs", ())))
        consumed_keys = frozenset()
        cutover_pin = descriptor.get("recovery_05_consumed_evidence")
        cutover_schema = descriptor.get("cutover_schema")
        inherited_cutover_pin = descriptor.get("inherited_recovery_05_cutover_evidence")
        if cutover_pin is None:
            RT.need(cutover_schema is None, "cutover descriptor lacks its evidence pin")
            if inherited_cutover_pin is not None:
                RT.need(isinstance(inherited_cutover_pin, dict)
                        and Path(inherited_cutover_pin.get("path", "")).name
                            == Path(X.CUTOVER_RELATIVE).name,
                        "inherited cutover evidence pin is malformed")
                reader.pinned(inherited_cutover_pin)
        else:
            RT.need(inherited_cutover_pin is None,
                    "descendant descriptor mixes new and legacy cutover evidence")
            RT.need(cutover_schema == X.CUTOVER_SCHEMA
                    and Path(cutover_pin.get("path", "")).resolve() == root / X.CUTOVER_RELATIVE,
                    "cutover descriptor evidence path/schema mismatch")
            cutover_body = reader.pinned(cutover_pin, root=root)
            consumed_keys = X.validate_consumed_manifest(cutover_body)
            expected_runs = X.apply_donor_restart_runs(expected_runs)
        RT.need(body.get("stage_index", expected_stage) == expected_stage,
                "descendant body stage_index changed")
        RT.need(descriptor.get("stage_index", expected_stage) == expected_stage,
                "descendant descriptor stage_index changed")
        RT.need(body.get("loss_stability_review") == descriptor.get("loss_stability_review"),
                "descendant loss review pin changed")
        RT.need(body["runs"] == expected_runs,
                "descendant derived donor/restart plan changed")
        RT.need(body["certificate_sources"] == parent.certificates,
                "descendant certificate sources changed")
        RT.need(descriptor["parent_closure"] == parent.closure,
                "descendant parent closure changed")
        RT.need(descriptor.get("parent_source_reconstruction") == parent.source_reconstruction,
                "descendant parent source reconstruction changed")
        RT.need(descriptor["science_profile_sha256"] == SCIENCE_SHA
                and body["protocol"] == parent.context.payload["protocol"]
                and descriptor["science_profile"] == body["protocol"]
                and body["candidate_manifest"] == parent.context.payload["candidate_manifest"]
                and body["candidate_index"] == parent.context.payload["candidate_index"],
                "science profile/candidate lineage changed")
        rows, updates = parent.rows, parent.updates
        old_keys = set(parent.old_keys) | set(consumed_keys)
        checks = parent.certificates
        runs = tuple(expected_runs)
        protocol = {"stage": {"runs": list(runs)}}
        source = reader.pinned(body["source_manifest"], root)
        RT.need(descriptor["source_manifest_sha256"] == body["source_manifest"]["sha256"],
                "descriptor source manifest mismatch")
        report = reader.pinned(body["admission_report"], root)
        RT.need(report == admission_body(
            parent, expected_runs, include_cutover=(
                cutover_schema == X.CUTOVER_SCHEMA
                or inherited_cutover_pin is not None)),
            "descendant admission mismatch")
        identity = reader.pinned(body["identity"], root)
        RT.need(identity == identity_body(body, descriptor), "descendant identity cross-hash mismatch")
        plan = reader.pinned(body["slot_plan"], root)
        seeds = reader.pinned(body["seed_plan"], root)
        validate_seed_plan(seeds, runs)
        RT.need(plan == slot_body(root, body["recovery_experiment_id"], seeds),
                "descendant frozen slot plan mismatch")
        RT.need(descriptor["parent_context"] == parent.context.context_pin,
                "parent context pin mismatch")
    else:
        RT.need(root == INITIAL_ROOT.resolve()
                and reader.pins[str(path.resolve())]["sha256"] == INITIAL_CONTEXT_SHA,
                "initial accepted base context is not the registered immutable one")
        RT.need(body.get("schema") == RT.CONTEXT_SCHEMA, "unknown base context schema")
        protocol = reader.pinned(body["protocol"], root)
        RT.need(body["protocol"]["sha256"].lower() == SCIENCE_SHA,
                "initial science profile pin mismatch")
        runs = tuple(RT.evidence.validate_draft(protocol, root))
        source = reader.pinned(body["source_manifest"], root)
        identity = reader.pinned(body["identity"], root)
        report = reader.pinned(body["admission_report"], root)
        for key in ("approval", "evidence_index", "candidate_manifest", "candidate_index",
                    "slot_plan", "seed_plan"):
            reader.pinned(body[key], root)
        candidate = reader.pinned(body["candidate_manifest"], root)
        pins = {}
        for item in candidate["output_files"]:
            reader.raw(item["path"], item, root)
            pins[Path(item["path"]).name] = item
        initial = types.SimpleNamespace(bundle_pins=pins)
        rows = {key: RT.selected_history(initial, *key) for key in ROSTER}
        updates = {key: RT.selected_updates(initial, *key) for key in ROSTER}
        old_keys = set(RT.load_old_runtime_keys(initial))
        checks = []
        plan, seeds = reader.pinned(body["slot_plan"], root), reader.pinned(body["seed_plan"], root)
    RT.need(source.get("schema") == RT.SOURCE_MANIFEST_SCHEMA
            and source.get("status") == "FROZEN", "source/native freeze missing")
    for pin in source["files"]:
        reader.raw(pin["path"], {k: pin[k] for k in RT.PIN_KEYS},
                   root if pin["category"] == "code" else None)
    if body.get("schema") == CONTEXT_SCHEMA:
        parent_native = {x["path"]: x["sha256"] for x in
                         parent.context.source_manifest["files"] if x["category"] == "native"}
        actual_native = {x["path"]: x["sha256"] for x in source["files"]
                         if x["category"] == "native"}
        RT.need(actual_native == parent_native, "frozen native parent lineage mismatch")
        RT._validate_source_manifest(source, root, R.WORLD_SEED_NAMESPACE,
                                     layout=Layout(root))
        RT.need(source["files"] == RT._context_source_files(Layout(root)),
                "descendant source closure has additions/omissions/drift")
    context = RT.RecoveryContext(
        root=root, context_path=path, context_pin=reader.pins[str(path.resolve())],
        payload=body, admission_report=report, approval={}, candidate={}, index={},
        protocol=protocol, identity=identity, slot_plan=plan, source_manifest=source,
        seed_plan=seeds, bundle_pins={}, runs=runs, old_runtime_keys=frozenset(old_keys))
    context.stage_index = stage_index
    context.is_descendant = body.get("schema") == CONTEXT_SCHEMA
    context.loss_review_enabled = bool(context.is_descendant and "stage_index" in body)
    context.lineage_rows, context.lineage_updates = rows, updates
    context.lineage_certificates = checks
    if context.is_descendant:
        context.descendant = descriptor
    return context, seen


def validate_recovery02_reconstructed_terminal(root, run_dir, context, started, child,
                                                child_original, receipt, stdout_text,
                                                summary, raw_wrapper, before, after):
    """Accept only the named recovery_02 reconstruction artifacts, with raw gaps explicit."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    RT.need(root == RECOVERY02_ROOT.resolve()
            and run_dir == root / "execution/recovery_train_run_01",
            "reconstruction is restricted to the registered recovery_02 root/run directory")
    RT.need(summary.get("reconstruction_basis", {}).get("pre_run_snapshot")
            == "reconstructed; no historical wrapper snapshot claimed"
            and summary.get("reconstruction_basis", {}).get("tool_exit")
            == "actual session 26862 terminal poll exit_code 0"
            and raw_wrapper.get("recovery_kind")
            == "assembled_from_existing_evidence_not_byte_exact_original"
            and raw_wrapper.get("normalization")
            == "saved child logs plus verified markers; not claimed as original outer response",
            "recovery_02 wrapper reconstruction provenance is missing/changed")
    observed = raw_wrapper.get("observed_session_poll") or {}
    RT.need(observed.get("session_id") == 26862 and observed.get("chunk_id") == "130ac2"
            and observed.get("exit_code") == 0
            and observed.get("basis") == "actual terminal write_stdin response visible in execution history"
            and "CHILD_EXIT_CODE=0" in observed.get("output", ""),
            "recorded runner chat poll evidence does not prove outer exit zero")
    RT.need(before.get("reconstruction_basis") == {
                "kind": "frozen_source_manifest_context_and_started_receipt",
                "not_original_wrapper_snapshot": True}
            and type(before.get("files")) is list and len(before["files"]) == 11
            and after.get("reconstruction_basis")
                == "current bytes match the reconstructed frozen 11-file baseline"
            and before.get("files") == after.get("files"),
            "post-hoc startup/current source pin reconstruction is incomplete or inconsistent")
    RT.need(type(summary.get("claims")) is int and summary["claims"] == 720
            and summary.get("results") == 720 and summary.get("update_journals") == 180
            and summary.get("stage_complete") is True and summary.get("global_stop") is False
            and summary.get("convergence_complete") is False,
            "reconstructed terminal summary does not describe the exact finite parent run")
    rid, argv = RECOVERY02_RECEIPT_ID, started.get("argv")
    child_argv = child.get("argv")
    original_argv = child_original.get("argv")
    RT.need(type(argv) is list and argv == receipt.get("argv")
            and type(child_argv) is list and type(original_argv) is list
            and child_argv == original_argv == ["D:/project_env/python.exe", "-B"] + argv
            and "recovery-train" in argv,
            "reconstructed child/start/terminal argv chain mismatch")
    RT.need(started.get("stage") == "started" and started.get("phase") == "recovery-train"
            and started.get("receipt_id") == rid
            and receipt.get("schema") == "stage5b-phase-receipt/1"
            and receipt.get("stage") == "terminal" and receipt.get("phase") == "recovery-train"
            and receipt.get("receipt_id") == rid
            and type(receipt.get("returncode")) is int and receipt["returncode"] == 0,
            "recovery_02 terminal receipt chain is incomplete or unsuccessful")
    RT.need(child.get("exit_code") == 0 and child_original.get("exit_code") == 0
            and type(child.get("exit_code")) is int
            and child.get("argv") == child_original.get("argv"),
            "recovery_02 observed child exit is missing or inconsistent")
    RT.need(receipt.get("outer_exit_code") is None
            and receipt.get("outer_exit_status") == "UNKNOWN",
            "terminal receipt's absent outer exit must remain UNKNOWN")
    lines = [line for line in stdout_text.replace("\r", "").splitlines()
             if line.startswith('{"phase": "recovery-train"')]
    RT.need(len(lines) == 1, "saved child stdout must contain one terminal CLI line")
    cli = RT.strict_json_bytes(lines[0].encode("utf-8"), "recovery_02 saved child stdout")
    RT.need(cli.get("returncode") == 0 and cli.get("receipt_id") == rid,
            "saved child stdout does not match terminal receipt")
    raw_lines = [line for line in raw_wrapper.get("output", "").replace("\r", "").splitlines()
                 if line.startswith('{"phase": "recovery-train"')]
    RT.need(len(raw_lines) == 1 and raw_lines[0] == lines[0],
            "reconstructed wrapper text does not match saved child CLI output")
    detail = receipt.get("detail")
    RT.need(type(detail) is dict and detail.get("status") == "FINITE_RECOVERY_STAGE_COMPLETE"
            and detail.get("worlds") == 720 and detail.get("unused_world_slots") == 0
            and detail.get("unclaimed_slots") == 0 and detail.get("convergence_complete") is False
            and detail.get("goal_complete") is False
            and detail.get("slot_counts") == {"claimed":720,"ok":720,"planned":720,
                                                "remaining":0,"results":720},
            "terminal receipt does not close exactly 720 worlds and 180 updates")
    bindings = detail.get("execution_bindings")
    RT.need(type(bindings) is dict
            and bindings == (started.get("detail") or {}).get("execution_bindings")
            and bindings.get("base_context") == context.context_pin
            and type(bindings.get("g3_report")) is dict
            and type(bindings.get("world_authorization")) is dict,
            "startup/terminal context, G3 and authorization bindings differ")
    results = detail.get("results")
    RT.need(type(results) is list and len(results) == len(context.runs) == 6,
            "terminal receipt lacks the six-run roster")
    expected = {(r["arm"],r["training_seed"]):r for r in context.runs}
    seen = set()
    for item in results:
        parts = str(item.get("slot","")).split(":")
        RT.need(len(parts)==3 and parts[0]=="recovery", "terminal result label invalid")
        key = RT.normalized_run(parts[1],int(parts[2])); run=expected.get(key)
        RT.need(run is not None and key not in seen, "terminal run is unknown/duplicated")
        seen.add(key)
        RT.need(item.get("status")=="ok" and item.get("episodes")==120
                and item.get("terminal_episode")==run["end_episode_cap"]
                and item.get("converged") is False
                and item.get("stopped_on_task_certificate", False) is False,
                "terminal run is not its closed 120-episode nonconverged prefix")
    RT.need(seen==set(expected), "terminal run roster incomplete")
    RT.need(len(summary.get("checkpoints",[]))==6
            and len(summary.get("certificates",[]))==6,
            "reconstructed CP/certificate index is incomplete")
    return detail, bindings, {
        "schema": RECONSTRUCTION_SCHEMA,
        "status": "RECONSTRUCTED_PARENT_EVIDENCE_WITH_EXPLICIT_UNKNOWN_FIELDS",
        "root": str(root), "source_reference": RECOVERY02_CHAT_POLL_REF,
        "outer_tool_exit": {"exit_code": 0, "basis": "RECORDED_RUNNER_SESSION_POLL",
            "session_id": observed["session_id"], "chunk_id": observed["chunk_id"],
            "raw_tool_return": "UNKNOWN_NOT_PERSISTED",
            "verifiability": "POLL_RESULT_RECONSTRUCTION_NOT_ORIGINAL_OUTER_RESPONSE"},
        "child_exit_code": 0, "terminal_receipt_returncode": 0,
        "receipt_id": rid, "worlds": 720, "startup_execution_bindings": bindings,
        "wrapper_summary_raw_pins": "RECONSTRUCTED_FROM_CLOSED_EVIDENCE_NOT_ORIGINAL_WRAPPER",
        "source_manifest_current_pins": "POST_HOC_BASELINE_MATCH_ONLY",
        "source_pre_start_snapshot": "UNKNOWN_NOT_CAPTURED",
        "source_post_exit_snapshot": "UNKNOWN_NOT_CAPTURED",
        "source_context_auth_drift": "UNKNOWN_NOT_COMPUTABLE_FROM_PERSISTED_EVIDENCE",
        "child_pid_match": "UNKNOWN_NOT_RECORDED",
        "terminal_checkpoint_payload_to_final_state": "UNKNOWN_NOT_LOADED_NEXT_G3_REQUIRED",
        "checkpoint_bytes_basis": "HASHED_AT_RECONSTRUCTION_NOT_RUNNER_TERMINAL_SNAPSHOT",
    }

def _poll_tool_return(record):
    RT.need(type(record) is dict, "poll evidence is not an object")
    if "tool_return" in record:
        tool = record["tool_return"]
        RT.need(type(tool) is dict, "poll wrapper lacks its tool_return object")
        return tool
    RT.need("chunk_id" in record or "session_id" in record or "exit_code" in record,
            "raw poll object has no tool-return fields")
    return record

def validate_reconstructed_parent_provenance(root, run_dir, context, provenance, started,
                                               receipt, observer, preceding_poll_record, poll_record, stdout_text,
                                               summary, before, after, pins, *, observer_mode=None,
                                               training_interpreter_history=None,
                                               training_session_id=None):
    """Validate later-stage post-hoc evidence without claiming historical snapshots."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    RT.need(root != RECOVERY02_ROOT.resolve() and getattr(context, "is_descendant", False)
            and context.stage_index >= 2,
            "generic reconstructed provenance is only for completed descendant stages")
    RT.need(provenance.get("schema") == GENERIC_RECONSTRUCTION_SCHEMA
            and provenance.get("status") == "POSTHOC_EVIDENCE_INDEX"
            and provenance.get("parent_root") == str(root)
            and provenance.get("run_directory") == str(run_dir)
            and type(provenance.get("receipt_id")) is str,
            "generic parent provenance identity/schema mismatch")
    rid = provenance["receipt_id"]
    RT.need(rid.startswith("recovery-train-") and "/" not in rid and "\\" not in rid,
            "generic provenance receipt ID is invalid")
    RT.need(started.get("schema") == "stage5b-phase-receipt/1"
            and started.get("phase") == "recovery-train" and started.get("stage") == "started"
            and started.get("receipt_id") == rid
            and receipt.get("schema") == "stage5b-phase-receipt/1"
            and receipt.get("phase") == "recovery-train" and receipt.get("stage") == "terminal"
            and receipt.get("receipt_id") == rid
            and type(receipt.get("returncode")) is int and receipt["returncode"] == 0
            and type(started.get("argv")) is list and started["argv"] == receipt.get("argv"),
            "generic parent start/terminal receipt chain is not successful and exact")
    bindings = (receipt.get("detail") or {}).get("execution_bindings")
    RT.need(type(bindings) is dict and bindings == (started.get("detail") or {}).get("execution_bindings")
            and bindings.get("base_context") == context.context_pin
            and type(bindings.get("g3_report")) is dict
            and type(bindings.get("world_authorization")) is dict,
            "generic parent execution bindings differ from its started context/G3/authorization")
    RT.need(observer_mode in (PREPARED_OBSERVER_MODE, OBSERVERLESS_POSTHOC_MODE)
            and training_interpreter_history == TRAINING_INTERPRETER_UNKNOWN
            and type(training_session_id) is int and not isinstance(training_session_id, bool),
            "generic parent provenance must declare observer mode and unknown interpreter history")
    tool = _poll_tool_return(poll_record)
    preceding = _poll_tool_return(preceding_poll_record)
    terminal_name = Path(pins["terminal_poll"]["path"]).name
    preceding_name = Path(pins["preceding_live_poll"]["path"]).name
    if observer_mode == PREPARED_OBSERVER_MODE:
        RT.need(type(observer) is dict and type(pins.get("observer_preparation")) is dict,
                "prepared-observer mode requires its pinned preparation record")
        session = observer.get("tool_session_id")
        started_obs = observer.get("train_started_receipt") or {}
        terminal_match = re.fullmatch(r"poll_([0-9]{4})_terminal\.json", terminal_name)
        preceding_match = re.fullmatch(r"poll_([0-9]{4})_live\.json", preceding_name)
        bound = (observer.get("schema") == "stage5b-training-observer-preparation/1"
                 and type(session) is int and not isinstance(session, bool)
                 and started_obs.get("receipt_id") == rid
                 and started_obs.get("argv") == started.get("argv")
                 and session == training_session_id)
    else:
        RT.need(observer is None and pins.get("observer_preparation") is None,
                "observerless provenance may only use its explicit null preparation pin")
        session = training_session_id
        bound = (type(session) is int and not isinstance(session, bool)
                 and "exit_code" not in preceding and "session_id" not in tool)
        terminal_match = re.fullmatch(r"poll_raw_([0-9]{3,4})_terminal\.json", terminal_name)
        preceding_match = re.fullmatch(r"poll_raw_([0-9]{3,4})\.json", preceding_name)
    sequence_bound = (terminal_match is not None and preceding_match is not None
                      and int(terminal_match.group(1)) == int(preceding_match.group(1)) + 1)
    direct_session = "session_id" in tool
    RT.need(bound
            and sequence_bound and preceding.get("session_id") == session
            and type(tool.get("exit_code")) is int and tool["exit_code"] == 0
            and type(tool.get("chunk_id")) is str and type(tool.get("output")) is str
            and (not direct_session or (type(tool.get("session_id")) is int
                 and tool.get("session_id") == session)),
            "actual terminal poll is absent or not bound to the recorded training session")
    lines = stdout_text.replace("\r", "").splitlines()
    cli_lines = [line for line in lines if line.startswith('{"phase": "recovery-train"')]
    RT.need(len(cli_lines) == 1, "one saved recovery-train CLI terminal line is required")
    cli = RT.strict_json_bytes(cli_lines[0].encode("utf-8"), "reconstructed CLI terminal")
    RT.need(cli.get("returncode") == 0 and cli.get("receipt_id") == rid,
            "saved CLI terminal differs from the pinned terminal receipt")
    marker = {"schema": GENERIC_RECONSTRUCTION_SCHEMA,
              "status": "POST_RUN_RECONSTRUCTION_NOT_CONTEMPORANEOUS"}
    RT.need(summary.get("schema") == "stage5b-recovery-train-terminal-summary/1"
            and summary.get("reconstruction_basis") == marker
            and summary.get("stage_complete") is True and summary.get("global_stop") is False
            and summary.get("convergence_complete") is False,
            "generic reconstructed summary lacks its explicit post-run marker or is incomplete")
    before_marker, after_marker = before.get("reconstruction_basis"), after.get("reconstruction_basis")
    RT.need(before_marker == marker and after_marker == marker
            and type(before.get("files")) is list and before.get("files") == after.get("files")
            and after.get("source_context_auth_drift") == "UNKNOWN_NOT_COMPUTABLE_FROM_POSTHOC_PINS",
            "post-hoc source pins are not explicitly marked or unknowns were overwritten")
    expected = {}
    code_pins = [x for x in context.source_manifest["files"]
                 if x.get("category") == "code" and Path(x["path"]).name in
                 {"stage5b.py", "recovery_runtime.py", "runner.py", "trainer.py", "guard.py"}]
    RT.need(len(code_pins) == 5, "frozen parent code pin set is incomplete")
    for pin in code_pins:
        rel = Path(pin["path"]).resolve().relative_to(root).as_posix()
        expected[rel] = (pin["bytes"], pin["sha256"])
    for pin in (context.payload["source_manifest"], context.payload["slot_plan"],
                context.payload["identity"], context.context_pin,
                bindings["g3_report"], bindings["world_authorization"]):
        rel = Path(pin["path"]).resolve().relative_to(root).as_posix()
        RT.need(rel not in expected, "post-hoc pin closure repeats a path")
        expected[rel] = (pin["bytes"], pin["sha256"])
    actual = {}
    for pin in before["files"]:
        rel = Path(pin["path"]).as_posix()
        RT.need(rel not in actual and type(pin.get("bytes")) is int
                and type(pin.get("sha256")) is str,
                "post-hoc pin list has duplicate or malformed entries")
        actual[rel] = (pin["bytes"], pin["sha256"])
    RT.need(actual == expected, "post-hoc pins do not equal frozen source/context/start bindings")
    active = [r for r in context.runs if r.get("active", True)]
    # A completed long stage is admitted with its own per-run/world caps; ordinary
    # bounded stages keep the 120-episode/720-world reconstruction bounds.
    long_parent = context.payload.get("target_completed_episodes") is not None
    detail = receipt["detail"]
    items = detail.get("results")
    RT.need(type(items) is list and len(items) == len(active),
            "generic terminal receipt has missing/extra active-run results")
    expected_runs = {(r["arm"], r["training_seed"]): r for r in active}
    seen, worlds = set(), 0
    for item in items:
        parts = str(item.get("slot", "")).split(":")
        RT.need(len(parts) == 3 and parts[0] == "recovery", "generic result run label is invalid")
        key = RT.normalized_run(parts[1], int(parts[2]))
        run = expected_runs.get(key)
        RT.need(run is not None and key not in seen, "generic result run is unknown/duplicated")
        seen.add(key)
        n, end = item.get("episodes"), item.get("terminal_episode")
        run_bound = (run["end_episode_cap"] - run["start_episode"]) if long_parent else 120
        RT.need(item.get("status") == "ok" and type(n) is int and 0 < n <= run_bound and n % 4 == 0
                and type(end) is int and end == run["start_episode"] + n
                and end <= run["end_episode_cap"]
                and (item.get("converged") is True or end == run["end_episode_cap"]),
                "generic parent run is not a closed bounded segment")
        worlds += n
    slots = detail.get("slot_counts") or {}
    world_bound = (sum(r["end_episode_cap"] - r["start_episode"] for r in active)
                   if long_parent else 720)
    RT.need(seen == set(expected_runs) and detail.get("status") == "FINITE_RECOVERY_STAGE_COMPLETE"
            and detail.get("worlds") == worlds
            and slots.get("claimed") == slots.get("ok") == slots.get("results") == worlds
            and worlds > 0 and worlds <= world_bound
            and summary.get("claims") == summary.get("results") == worlds
            and summary.get("update_journals") == worlds // C.EPISODES_PER_UPDATE
            and type(summary.get("checkpoints")) is list and len(summary["checkpoints"]) == 6
            and type(summary.get("certificates")) is list and len(summary["certificates"]) == 6,
            "generic parent totals/CP/certificate inventory do not close")

    reconstruction = {
        "schema": GENERIC_RECONSTRUCTION_SCHEMA,
        "status": "POSTHOC_PARENT_EVIDENCE_VALIDATED_WITH_EXPLICIT_UNKNOWNS",
        "receipt_id": rid, "worlds": worlds, "terminal_poll_pin": pins["terminal_poll"],
        "outer_tool_exit": {"exit_code": 0, "basis": "PINNED_TERMINAL_POLL_REAL_SESSION_RETURN",
                             "raw_tool_return": "PERSISTED_IN_PINNED_TERMINAL_POLL"},
        "child_exit_code": "UNKNOWN_NOT_SEPARATELY_RECORDED",
        "source_pre_start_snapshot": "UNKNOWN_NOT_CAPTURED",
        "source_post_exit_snapshot": "POSTHOC_CURRENT_BYTES_MATCH_FROZEN_PINS_ONLY",
        "source_context_auth_drift": "UNKNOWN_NOT_COMPUTABLE_FROM_POSTHOC_PINS",
        "child_pid_match": "UNKNOWN_NOT_RECORDED",
        "terminal_checkpoint_payload_to_final_state": "UNKNOWN_NEXT_G3_REQUIRED",
        "checkpoint_bytes_basis": "RECONSTRUCTED_SUMMARY_PINS_RECHECKED_AS_CURRENT_BYTES",
    }
    if "observer_mode" in provenance:
        reconstruction.update({
            "observer_mode": observer_mode,
            "observer_preparation": ("PINNED_CONTEMPORANEOUS"
                                     if observer_mode == PREPARED_OBSERVER_MODE
                                     else "UNKNOWN_NOT_CAPTURED"),
            "training_interpreter_history": training_interpreter_history,
            "terminal_poll_session_id": session,
        })
    return detail, bindings, summary["checkpoints"], summary["certificates"], reconstruction

def _recovery02_reconstructed_inputs(root, run_dir, reader, context):
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    RT.need(root == RECOVERY02_ROOT.resolve()
            and run_dir == root / "execution/recovery_train_run_01",
            "no generic reconstructed-parent admission is permitted")
    receipts = root / "execution/receipts"
    started = reader.json(receipts / ("recovery-train_started_%s.json" % RECOVERY02_RECEIPT_ID), root=root)
    receipt = reader.json(receipts / ("recovery-train_terminal_%s.json" % RECOVERY02_RECEIPT_ID), root=root)
    child = reader.json(run_dir / "child_exit.json", root=root)
    child_original = reader.json(run_dir / "child_exit_observed_original.json", root=root)
    stdout = reader.raw(run_dir / "stdout.log", root=root).decode("utf-8", errors="strict")
    summary = reader.json(run_dir / "terminal_summary.json", root=root)
    raw = reader.json(run_dir / "terminal_raw_recovered_01.json", root=root)
    before = reader.json(run_dir / "pins_before.json", root=root)
    after = reader.json(run_dir / "result.json", root=root)
    detail, binding, reconstruction = validate_recovery02_reconstructed_terminal(
        root, run_dir, context, started, child, child_original, receipt, stdout,
        summary, raw, before, after)
    expected_pins = {}
    source_code = [x for x in context.source_manifest["files"]
                   if x.get("category") == "code" and
                   Path(x["path"]).name in {"stage5b.py", "recovery_runtime.py", "runner.py",
                                             "trainer.py", "guard.py"}]
    RT.need(len(source_code) == 5, "reconstructed baseline source closure is incomplete")
    for pin in source_code:
        path = Path(pin["path"]).resolve().relative_to(root).as_posix()
        expected_pins[path] = (pin["bytes"], pin["sha256"])
    for pin in (context.payload["source_manifest"], context.payload["slot_plan"],
                context.payload["identity"], context.context_pin,
                binding["g3_report"], binding["world_authorization"]):
        path = Path(pin["path"]).resolve().relative_to(root).as_posix()
        RT.need(path not in expected_pins, "reconstructed baseline pin path is duplicated")
        expected_pins[path] = (pin["bytes"], pin["sha256"])
    before_records = {}
    for pin in before["files"]:
        rel = Path(pin["path"]).as_posix()
        RT.need(rel not in before_records, "reconstructed baseline repeats a source pin")
        before_records[rel] = (pin["bytes"], pin["sha256"])
    RT.need(before_records == expected_pins,
            "reconstructed baseline pins differ from frozen source/context/start bindings")
    for name in ("g3_report", "world_authorization"):
        reader.pinned(binding[name], root)
    g3, auth = reader.pinned(binding["g3_report"], root), reader.pinned(binding["world_authorization"], root)
    RT.validate_g3(context, g3, binding["g3_report"])
    RT.validate_world_authorization(context, auth, binding["world_authorization"], binding["g3_report"])
    pins = {}
    for item in before["files"]:
        path = (root / item["path"]).resolve()
        pin = {"path": str(path), "bytes": item["bytes"], "sha256": item["sha256"]}
        reader.raw(path, pin, root=root)
        pins[path] = pin
    checkpoints, certificates = summary["checkpoints"], summary["certificates"]
    for result in detail["results"]:
        _, arm, seed_text = str(result["slot"]).split(":")
        arm, seed = RT.normalized_run(arm, int(seed_text))
        end = result["terminal_episode"]
        cp = Layout(root).checkpoint(arm, seed, end)
        cp_record = next(x for x in checkpoints if x.get("run") == "%s_%d" % (arm, seed))
        RT.need(cp_record.get("episode") == end, "reconstructed CP cursor mismatch")
        cp_pin = {"path": str(cp), "bytes": cp_record["bytes"], "sha256": cp_record["sha256"]}
        reader.raw(cp, cp_pin, root=root)
        cert_path = root / "execution/recovery/certificates" / ("CERTIFICATE_%s_s%d.json" % (arm, seed))
        cert_record = next(x for x in certificates if x.get("run") == "%s_%d" % (arm, seed))
        cert_pin = {"path": str(cert_path), "bytes": cert_record["bytes"],
                    "sha256": cert_record["sha256"]}
        reader.raw(cert_path, cert_pin, root=root)
    reconstruction["terminal_checkpoint_pins"] = checkpoints
    reconstruction["terminal_certificate_pins"] = certificates
    reconstruction["source_manifest_current_pins"] = "MATCHED_RECONSTRUCTED_BASELINE_DURING_ADMISSION"
    reconstruction["g3_report_sha256"] = binding["g3_report"]["sha256"]
    reconstruction["world_authorization_sha256"] = binding["world_authorization"]["sha256"]
    return detail, binding, checkpoints, certificates, reconstruction, g3

def _generic_reconstructed_inputs(root, run_dir, reader, context, provenance_path):
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    provenance_path = Path(provenance_path).resolve()
    evidence_dir = provenance_path.parent.resolve()
    evidence_root = evidence_dir.parent.parent.parent.resolve()
    RT.need(evidence_dir.name == root.name and evidence_dir.parent.name == "recovery_parent_evidence"
            and evidence_dir.parent.parent.name == "execution" and root != evidence_root
            and evidence_dir / "parent_provenance.json" == provenance_path,
            "generic provenance path is not the fixed parent-evidence layout")
    current_root = Path(__file__).resolve().parents[1]
    if evidence_root != current_root:
        # A completed ancestor child root must pin exactly this parent/index pair.
        layout_reader = Reads()
        child_context = layout_reader.json(evidence_root / RT.CONTEXT_RELATIVE, root=evidence_root)
        child_descriptor = layout_reader.pinned(child_context["descendant"], root=evidence_root)
        RT.need(Path(child_descriptor["parent_context"]["path"]).resolve().parents[1] == root
                and Path(child_descriptor["parent_provenance_path"]).resolve() == provenance_path,
                "generic provenance evidence root does not pin this parent and index")
    # During prepare the current root has not yet published its descendant context.
    RT.need(root != RECOVERY02_ROOT.resolve()
            and getattr(context, "is_descendant", False) and context.stage_index >= 2,
            "generic provenance cannot replace the registered recovery_02 path or initial parent gate")
    RT.need(provenance_path == evidence_dir / "parent_provenance.json",
            "generic provenance index must be in the canonical new-root evidence directory")
    provenance = reader.json(provenance_path, root=evidence_root)
    base_keys = {"schema", "status", "parent_root", "run_directory", "receipt_id", "pins"}
    explicit_keys = base_keys | {"observer_mode", "training_interpreter_history", "training_session_id"}
    RT.need(set(provenance) in (base_keys, explicit_keys),
            "generic parent provenance has missing or unknown fields")
    RT.need(provenance.get("schema") == GENERIC_RECONSTRUCTION_SCHEMA
            and provenance.get("status") == "POSTHOC_EVIDENCE_INDEX"
            and provenance.get("parent_root") == str(root)
            and provenance.get("run_directory") == str(run_dir),
            "generic parent provenance root/status mismatch")
    pins = provenance["pins"]
    required = {"started_receipt", "terminal_receipt", "observer_preparation", "preceding_live_poll",
                "terminal_poll", "terminal_summary", "pins_before", "pins_after"}
    RT.need(type(pins) is dict and set(pins) == required,
            "generic parent provenance pin fields are incomplete")
    if set(provenance) == base_keys:
        # Legacy observer-prepared indexes remain valid without inventing new evidence.
        observer_mode = PREPARED_OBSERVER_MODE
        training_interpreter_history = TRAINING_INTERPRETER_UNKNOWN
        training_session_id = None
        RT.need(pins["observer_preparation"] is not None,
                "legacy provenance lacks its contemporaneous observer preparation")
    else:
        observer_mode = provenance["observer_mode"]
        training_interpreter_history = provenance["training_interpreter_history"]
        training_session_id = provenance["training_session_id"]
    RT.need(observer_mode in (PREPARED_OBSERVER_MODE, OBSERVERLESS_POSTHOC_MODE)
            and training_interpreter_history == TRAINING_INTERPRETER_UNKNOWN
            and (training_session_id is None
                 or (type(training_session_id) is int
                     and not isinstance(training_session_id, bool))),
            "generic parent provenance mode/unknown-history mismatch")
    observerless = observer_mode == OBSERVERLESS_POSTHOC_MODE
    RT.need((pins["observer_preparation"] is None) is observerless,
            "observer preparation pin and explicit observer mode disagree")
    rid = provenance["receipt_id"]
    RT.need(type(rid) is str and rid.startswith("recovery-train-")
            and "/" not in rid and "\\" not in rid,
            "generic parent provenance receipt ID is invalid")
    expected = {
        "started_receipt": root / "execution/receipts" / ("recovery-train_started_%s.json" % rid),
        "terminal_receipt": root / "execution/receipts" / ("recovery-train_terminal_%s.json" % rid),
        "terminal_summary": evidence_dir / "terminal_summary.json",
        "pins_before": evidence_dir / "pins_before.json",
        "pins_after": evidence_dir / "pins_after.json",
    }
    observer_dir = None
    for name in ("observer_preparation", "preceding_live_poll", "terminal_poll"):
        pin = pins[name]
        if name == "observer_preparation" and pin is None:
            continue
        RT.exact_keys(pin, RT.PIN_KEYS, "generic provenance " + name)
        path = Path(pin["path"]).resolve()
        RT.evidence.inside(root / "execution", path)
        if name == "observer_preparation":
            RT.need(path.name == "observer_preparation.json"
                    and path.parent.name.startswith("recovery_train_observer_"),
                    "observer preparation path is not the recorded training observer")
            observer_dir = path.parent
        else:
            if observer_dir is None:
                RT.need(path.parent.name.startswith("recovery_train_observer_"),
                        "observerless poll is not in a recorded training-observer directory")
                observer_dir = path.parent
            else:
                RT.need(path.parent == observer_dir,
                        "poll evidence is outside the bound observer directory")
            expected[name] = path
    for name, path in expected.items():
        pin = pins[name]
        RT.exact_keys(pin, RT.PIN_KEYS, "generic provenance " + name)
        RT.need(Path(pin["path"]).resolve() == path.resolve(),
                "generic provenance path differs from its fixed evidence location")
    bodies = {}
    for name in required:
        pin = pins[name]
        if name == "observer_preparation" and pin is None:
            bodies[name] = None
            continue
        pin_root = evidence_root if name in {"terminal_summary", "pins_before", "pins_after"} else root
        bodies[name] = reader.json(pin["path"], pin, root=pin_root)
    if training_session_id is None:
        observer = bodies["observer_preparation"]
        RT.need(observer_mode == PREPARED_OBSERVER_MODE and type(observer) is dict
                and type(observer.get("tool_session_id")) is int
                and not isinstance(observer.get("tool_session_id"), bool),
                "legacy prepared observer lacks an exact training session")
        training_session_id = observer["tool_session_id"]
    detail, binding, checkpoints, certificates, reconstruction = \
        validate_reconstructed_parent_provenance(
            root, run_dir, context, provenance, bodies["started_receipt"],
            bodies["terminal_receipt"], bodies["observer_preparation"],
            bodies["preceding_live_poll"], bodies["terminal_poll"],
            _poll_tool_return(bodies["terminal_poll"]).get("output", ""),
            bodies["terminal_summary"], bodies["pins_before"], bodies["pins_after"], pins,
            observer_mode=observer_mode,
            training_interpreter_history=training_interpreter_history,
            training_session_id=training_session_id)
    for item in bodies["pins_before"]["files"]:
        path = (root / item["path"]).resolve()
        reader.raw(path, {"path": str(path), "bytes": item["bytes"],
                          "sha256": item["sha256"]}, root=root)
    for name in ("g3_report", "world_authorization"):
        reader.pinned(binding[name], root)
    g3 = reader.pinned(binding["g3_report"], root)
    auth = reader.pinned(binding["world_authorization"], root)
    RT.validate_g3(context, g3, binding["g3_report"])
    RT.validate_world_authorization(context, auth, binding["world_authorization"], binding["g3_report"])
    for result in detail["results"]:
        _, arm, seed_text = str(result["slot"]).split(":")
        arm, seed = RT.normalized_run(arm, int(seed_text))
        end = result["terminal_episode"]
        cp = Layout(root).checkpoint(arm, seed, end)
        cp_record = next(x for x in checkpoints if x.get("run") == "%s_%d" % (arm, seed))
        RT.need(cp_record.get("episode") == end, "reconstructed CP cursor mismatch")
        reader.raw(cp, {"path": str(cp), "bytes": cp_record["bytes"],
                        "sha256": cp_record["sha256"]}, root=root)
        cert_path = root / "execution/recovery/certificates" / ("CERTIFICATE_%s_s%d.json" % (arm, seed))
        cert = next(x for x in certificates if x.get("run") == "%s_%d" % (arm, seed))
        reader.raw(cert_path, {"path": str(cert_path), "bytes": cert["bytes"],
                               "sha256": cert["sha256"]}, root=root)
    reconstruction["terminal_checkpoint_pins"] = checkpoints
    reconstruction["terminal_certificate_pins"] = certificates
    reconstruction["g3_report_sha256"] = binding["g3_report"]["sha256"]
    reconstruction["world_authorization_sha256"] = binding["world_authorization"]["sha256"]
    return detail, binding, checkpoints, certificates, reconstruction, g3, bodies["terminal_summary"]

def _recovery06_terminal_inputs(root, run_dir, reader, context, *, summary_path=None, summary_payload=None, outer_poll_name=None, adapter=None):
    """Validate the writer-native terminal evidence used by completed recovery stages."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    summary_path = Path(summary_path or (run_dir / "stage_terminal_summary.json")).resolve()
    summary = summary_payload if summary_payload is not None else reader.json(summary_path, root=root)
    RT.need(summary.get("schema") == "stage5b-recovery-stage-terminal-summary/1"
            and Path(summary.get("root", "")).resolve() == root
            and summary.get("stage_complete") is True
            and summary.get("converged") is False
            and summary.get("result_count") == 720
            and summary.get("update_journal_count") == 180
            and summary.get("result_status_counts") == {"ok": 720},
            "completed recovery parent terminal summary is incomplete/noncanonical")

    child_pin = summary.get("child_exit")
    RT.need(isinstance(child_pin, dict), "parent terminal summary lacks child-exit evidence")
    child_path = Path(child_pin.get("path", "")).resolve()
    expected_child_name = ("child_exit.txt"
                           if adapter == "RECOVERY_08_MONITOR_WRAPPER/1"
                           else "child.exit.txt")
    RT.need(child_path == run_dir / expected_child_name,
            "parent child-exit evidence is not at its writer path")
    child_raw = reader.raw(child_path, {k: child_pin[k] for k in RT.PIN_KEYS}, root=root)
    RT.need(child_raw.decode("ascii").strip() == "0"
            and child_pin.get("value") == "0",
            "parent real child exit is not zero")

    outer_pin = summary.get("outer_terminal")
    RT.need(isinstance(outer_pin, dict), "parent terminal summary lacks outer exit evidence")
    outer_path = Path(outer_pin.get("path", "")).resolve()
    expected_outer_name = outer_poll_name or "outer_poll_177.json"
    RT.need(outer_path.parent == run_dir and outer_path.name == expected_outer_name,
            "parent outer exit evidence is not the registered final poll")
    outer = reader.json(outer_path, {k: outer_pin[k] for k in RT.PIN_KEYS}, root=root)
    if adapter == "RECOVERY_07_MONITOR_TERMINAL/1":
        response = outer
    elif adapter == "RECOVERY_08_MONITOR_WRAPPER/1":
        response = outer.get("poll") or {}
    else:
        response = outer.get("response") or {}
    common_outer = (response.get("exit_code") == 0
                    and isinstance(response.get("output"), str)
                    and outer_pin.get("exit_code") == 0
                    and outer_pin.get("output") == response.get("output")
                    and outer_pin.get("wall_time_seconds") == response.get("wall_time_seconds"))
    if adapter == "RECOVERY_07_MONITOR_TERMINAL/1":
        RT.need(common_outer and response.get("output") == "",
                "recovery_07 outer poll must be a clean exit zero with empty output")
    elif adapter == "RECOVERY_08_MONITOR_WRAPPER/1":
        RT.need(common_outer and "CHILD_EXIT=0" in response["output"]
                and "CHILD_PID=" in response["output"],
                "recovery_08 outer poll must close the child exit zero")
    else:
        RT.need(common_outer and "CHILD_EXIT_CODE=0" in response["output"],
                "parent real outer tool exit is not zero")

    stop_files = summary.get("stop_or_terminal_named_files")
    RT.need(isinstance(stop_files, list), "parent terminal evidence inventory is missing")
    for item in stop_files:
        RT.need(isinstance(item, dict), "parent terminal evidence pin is malformed")
        reader.raw(Path(item.get("path", "")), {k: item[k] for k in RT.PIN_KEYS}, root=root)
    receipt_items = [x for x in stop_files
                     if Path(x.get("path", "")).name.startswith("recovery-train_terminal_")]
    RT.need(len(receipt_items) == 1, "parent needs exactly one train terminal receipt")
    receipt_item = receipt_items[0]
    receipt_path = Path(receipt_item["path"]).resolve()
    RT.need(receipt_path.parent == root / "execution/receipts",
            "parent terminal receipt is outside its writer directory")
    receipt = reader.json(receipt_path, {k: receipt_item[k] for k in RT.PIN_KEYS}, root=root)
    RT.need(receipt.get("schema") == "stage5b-phase-receipt/1"
            and receipt.get("phase") == "recovery-train"
            and receipt.get("stage") == "terminal"
            and receipt.get("returncode") == 0,
            "parent terminal receipt is not successful training")
    rid = receipt.get("receipt_id")
    RT.need(receipt_path.name == "recovery-train_terminal_%s.json" % rid,
            "parent terminal receipt ID/path mismatch")
    started_receipt_path = root / "execution/receipts" / (
        "recovery-train_started_%s.json" % rid)
    started_receipt = reader.json(started_receipt_path, root=root)
    RT.need(started_receipt.get("phase") == "recovery-train"
            and started_receipt.get("stage") == "started"
            and started_receipt.get("receipt_id") == rid
            and started_receipt.get("argv") == receipt.get("argv"),
            "parent started/terminal receipt chain mismatch")
    detail = receipt.get("detail") or {}
    RT.need(detail.get("status") == "FINITE_RECOVERY_STAGE_COMPLETE"
            and detail.get("worlds") == 720
            and detail.get("unused_world_slots") == 0
            and detail.get("unclaimed_slots") == 0
            and detail.get("convergence_complete") is False
            and detail.get("slot_counts") == {
                "claimed": 720, "ok": 720, "planned": 720,
                "remaining": 0, "results": 720}
            and isinstance(detail.get("results"), list)
            and len(detail["results"]) == 6,
            "parent terminal receipt totals/results do not close")
    binding = detail.get("execution_bindings") or {}
    RT.need(binding == (started_receipt.get("detail") or {}).get("execution_bindings"),
            "parent execution inputs drifted between receipts")

    started_pin = summary.get("started")
    RT.need(isinstance(started_pin, dict), "parent terminal summary lacks started pin")
    started_path = Path(started_pin.get("path", "")).resolve()
    RT.need(started_path == run_dir / "started.json",
            "parent started record is not at its writer path")
    run_started = reader.json(started_path, {k: started_pin[k] for k in RT.PIN_KEYS}, root=root)
    if adapter == "RECOVERY_07_MONITOR_TERMINAL/1":
        RT.need(run_started.get("schema") == "stage5b-recovery-train-execution-start/2"
                and Path(run_started.get("root", "")).resolve() == root
                and run_started.get("context_sha256") == binding["base_context"]["sha256"]
                and run_started.get("g3_report_sha256") == binding["g3_report"]["sha256"]
                and run_started.get("authorization_sha256") == binding["world_authorization"]["sha256"]
                and run_started.get("workers") == 3
                and run_started.get("argv") == ["-B"] + list(receipt.get("argv", [])),
                "recovery_07 run-start record does not close the receipt bindings")
    elif adapter == "RECOVERY_08_MONITOR_WRAPPER/1":
        RT.need(set(run_started) == {
                    "utc", "phase", "pid", "cwd", "argv", "context_sha256",
                    "g3_report_sha256", "authorization_sha256", "stdout", "stderr", "workers"}
                and run_started.get("phase") == "recovery-train"
                and run_started.get("pid") == receipt.get("pid")
                and Path(run_started.get("cwd", "")).resolve() == root
                and run_started.get("context_sha256") == binding["base_context"]["sha256"]
                and run_started.get("g3_report_sha256") == binding["g3_report"]["sha256"]
                and run_started.get("authorization_sha256") == binding["world_authorization"]["sha256"]
                and run_started.get("workers") == 3
                and run_started.get("argv", [])[2:] == receipt.get("argv"),
                "recovery_08 run-start record does not close the receipt bindings")
    else:
        RT.need(run_started.get("schema") == "stage5b-recovery-train-started/4"
                and run_started.get("authorization_sha256") == binding["world_authorization"]["sha256"]
                and run_started.get("g3_report_sha256") == binding["g3_report"]["sha256"]
                and run_started.get("argv", [])[1:] == receipt.get("argv"),
                "parent run-start record does not close the receipt bindings")
    for name in ("g3_report", "world_authorization"):
        reader.pinned(binding[name], root)
    g3 = reader.pinned(binding["g3_report"], root)
    auth = reader.pinned(binding["world_authorization"], root)
    RT.validate_g3(context, g3, binding["g3_report"])
    RT.validate_world_authorization(context, auth, binding["world_authorization"],
                                    binding["g3_report"])

    roster = {"%s_%d" % key for key in ROSTER}

    def normalized_writer_run(label):
        match = re.fullmatch(r"(T[12])_s(9100[1-3])", str(label))
        RT.need(match is not None, "parent checkpoint/certificate run is unregistered")
        normalized = "%s_%s" % (match.group(1), match.group(2))
        RT.need(normalized in roster, "parent checkpoint/certificate run is unregistered")
        return normalized
    checkpoint_inventory = []
    raw_checkpoints = summary.get("checkpoints")
    RT.need(isinstance(raw_checkpoints, list) and len(raw_checkpoints) == 6,
            "parent checkpoint inventory is incomplete")
    for item in raw_checkpoints:
        RT.need(isinstance(item, dict), "parent checkpoint pin is malformed")
        writer_run = item.get("run")
        normalized_run = normalized_writer_run(writer_run)
        path = Path(item.get("path", "")).resolve()
        match = re.fullmatch(r"milestone_(\d+)\.pth", path.name)
        RT.need(match is not None and path.parent.name == str(writer_run)
                and path.parent.parent == root / "execution/checkpoints",
                "parent checkpoint cursor/path is malformed")
        reader.raw(path, {k: item[k] for k in RT.PIN_KEYS}, root=root)
        checkpoint_inventory.append({**item, "run": normalized_run,
                                     "writer_run": writer_run,
                                     "episode": int(match.group(1))})
    RT.need({x["run"] for x in checkpoint_inventory} == roster,
            "parent checkpoint roster is incomplete")

    certificate_inventory = summary.get("certificates")
    RT.need(isinstance(certificate_inventory, list) and len(certificate_inventory) == 6,
            "parent certificate inventory is incomplete")
    for index, item in enumerate(certificate_inventory):
        RT.need(isinstance(item, dict), "parent certificate pin is malformed")
        writer_run = item.get("run")
        normalized_run = normalized_writer_run(writer_run)
        reader.raw(Path(item.get("path", "")), {k: item[k] for k in RT.PIN_KEYS}, root=root)
        certificate_inventory[index] = {**item, "run": normalized_run,
                                        "writer_run": writer_run}
    RT.need({x["run"] for x in certificate_inventory} == roster,
            "parent certificate roster is incomplete")

    reconstruction = {
        "schema": "stage5b-recovery06-terminal-evidence/1",
        "status": "WRITER_NATIVE_TERMINAL_EVIDENCE_VALIDATED",
        "terminal_summary": reader.pins[str(summary_path.resolve())],
        "terminal_receipt": reader.pins[str(receipt_path)],
        "run_started": reader.pins[str(started_path)],
        "child_exit": reader.pins[str(child_path)],
        "outer_terminal": reader.pins[str(outer_path)],
        "scope": "COMPLETED_PARENT_ONLY_NO_PRE_EXECUTION_SNAPSHOT_RECREATED",
    }
    if adapter is not None:
        reconstruction["adapter"] = adapter
    return (detail, binding, checkpoint_inventory, certificate_inventory,
            reconstruction, g3, {**summary, "checkpoints": checkpoint_inventory})

def completed_parent(root, run_directory=None, *, provenance_path=None, reader=None, seen=frozenset()):
    """Validate a complete parent; recovery_02 has one explicit bounded reconstruction path."""
    root = Path(root).resolve()
    reader = reader or Reads()
    RT.need(not (root / "execution/ledger" / RT.STOP_SIGNAL).exists(),
            "STOP parent cannot be admitted as complete")
    context, ancestry = _base(root, reader, seen)
    run_dir = Path(run_directory or root / "execution/recovery_train_run_01").resolve()
    RT.evidence.inside(root / "execution", run_dir)
    completed_stage_path = run_dir / "stage_terminal_summary.json"
    summary_path = run_dir / "terminal_summary.json"
    summary = reader.json(summary_path, root=root) if summary_path.is_file() else None
    reconstruction = None
    provenance_path = Path(provenance_path).resolve() if provenance_path is not None else run_dir / "parent_provenance.json"
    if provenance_path.is_file():
        detail, binding, checkpoint_inventory, certificate_inventory, reconstruction, g3, summary = \
            _generic_reconstructed_inputs(root, run_dir, reader, context, provenance_path)
    elif (run_dir / "monitor_terminal_receipt.json").is_file():
        from . import recovery07_terminal
        summary_payload, outer_poll_name, monitor_path = \
            recovery07_terminal.normalize(root, run_dir, reader)
        detail, binding, checkpoint_inventory, certificate_inventory, reconstruction, g3, summary = \
            _recovery06_terminal_inputs(
                root, run_dir, reader, context, summary_path=monitor_path,
                summary_payload=summary_payload, outer_poll_name=outer_poll_name,
                adapter=summary_payload.get("adapter"))
    elif completed_stage_path.is_file():
        detail, binding, checkpoint_inventory, certificate_inventory, reconstruction, g3, summary = \
            _recovery06_terminal_inputs(root, run_dir, reader, context)
    elif summary is not None and summary.get("reconstruction_basis") is not None:
        detail, binding, checkpoint_inventory, certificate_inventory, reconstruction, g3 = \
            _recovery02_reconstructed_inputs(root, run_dir, reader, context)
    elif summary is not None:
        child = reader.json(run_dir / "child_exit.json", root=root)
        RT.need(type(child.get("exit_code")) is int and child["exit_code"] == 0
                and child.get("child_pid") == summary.get("child_pid"),
                "parent actual child exit/PID mismatch")
        raws = tuple(run_dir.glob("terminal_raw*.json"))
        RT.need(len(raws) == 1, "one explicit terminal raw poll is required")
        raw = reader.json(raws[0], root=root)
        lines = str(raw.get("output", "")).replace("\r", "").splitlines()
        cli_lines = [line for line in lines if line.startswith('{"phase": "recovery-train"')]
        RT.need(len(cli_lines) == 1, "one real terminal recovery-train CLI line required")
        cli = RT.strict_json_bytes(cli_lines[0].encode(), "parent terminal CLI")
        rid = cli.get("receipt_id")
        RT.need(type(rid) is str and rid.startswith("recovery-train-")
                and "/" not in rid and "\\" not in rid, "invalid parent receipt ID")
        receipt = reader.json(root / "execution/receipts" /
                              ("recovery-train_terminal_%s.json" % rid), root=root)
        started = reader.json(root / "execution/receipts" /
                              ("recovery-train_started_%s.json" % rid), root=root)
        detail = validate_terminal(summary, receipt, raw)
        RT.need(started.get("receipt_id") == rid and started.get("stage") == "started"
                and started.get("phase") == "recovery-train"
                and started.get("argv") == receipt.get("argv"),
                "parent started/terminal receipt chain mismatch")
        RT.need(detail["execution_bindings"] == started["detail"]["execution_bindings"],
                "parent execution inputs drifted between receipts")
        binding = detail["execution_bindings"]
        checkpoint_inventory, certificate_inventory = summary["checkpoints"], summary["certificates"]
        for name in ("g3_report", "world_authorization"):
            reader.pinned(binding[name], root)
        g3 = reader.pinned(binding["g3_report"], root)
        auth = reader.pinned(binding["world_authorization"], root)
        RT.validate_g3(context, g3, binding["g3_report"])
        RT.validate_world_authorization(context, auth, binding["world_authorization"],
                                        binding["g3_report"])
        before = reader.json(run_dir / "pins_before.json", root=root)
        after = reader.json(run_dir / "result.json", root=root)
        RT.need(before.get("files") == after.get("files")
                and after.get("source_context_auth_drift") == [],
                "parent source/context/auth pre/post drift")
        for item in before["files"]:
            path = (root / item["path"]).resolve()
            reader.raw(path, {"path": str(path), "bytes": item["bytes"],
                              "sha256": item["sha256"]}, root)
    else:
        raise RT.RecoveryRuntimeError(
            "parent has no terminal summary; only the registered recovery_02 reconstruction artifacts are admissible")
    g3_by_run = {RT.normalized_run(item["arm"], item["training_seed"]): item
                  for item in g3.get("runs", [])}
    RT.need(set(g3_by_run) == set(ROSTER), "validated G3 does not contain the six-run roster")
    # Append only closed parent rows and updates. All consumed claims are retained.
    rows = {key: dict(context.lineage_rows[key]) for key in ROSTER}
    updates = {key: dict(context.lineage_updates[key]) for key in ROSTER}
    new_updates_by_run = {key: {} for key in ROSTER}
    old_keys = set(context.old_runtime_keys)
    states, all_slots = {}, set()
    certificates = list(context.lineage_certificates)
    results = detail.get("results")
    active = [r for r in context.runs if r.get("active", True)]
    RT.need(type(results) is list and len(results) == len(active),
            "parent has missing/extra run terminal results")
    by_result = {}
    for result in results:
        parts = str(result.get("slot", "")).split(":")
        RT.need(len(parts) == 3 and parts[0] == "recovery", "parent run label invalid")
        key = RT.normalized_run(parts[1], int(parts[2]))
        RT.need(key not in by_result, "parent repeats a run result")
        by_result[key] = result
    for run in context.runs:
        key = run["arm"], run["training_seed"]
        if not run.get("active", True):
            RT.need(key not in by_result, "retired run consumed new worlds")
            checks = RT.read_check_certificates(context, Layout(root), *key)
            cert_end = first_task_certificate_end(checks)
            RT.need(cert_end is not None, "retired run has no preserved formal task certificate")
            states[key] = {"cursor": run["start_episode"], "next_update": run["next_update"],
                           "certified": True, "task_certified": True,
                           "task_certificate_end_episode": run.get("task_certificate_end_episode", cert_end),
                           "loss_stability_stable": run.get("loss_stability_stable") is True,
                           "loss_stability_review": run.get("loss_stability_review"),
                           "latest_telemetry_sha256": run.get("latest_telemetry_sha256"),
                           "donor": run["donor"]}
            states[key]["completed_episode_count"] = len(rows[key])
            states[key]["registered_missing_episode_count"] = (
                states[key]["cursor"] - states[key]["completed_episode_count"])
            continue
        RT.need(key in by_result, "active parent run has no terminal result")
        result = by_result[key]
        telemetry_path = root / "execution/recovery/telemetry" / ("%s_s%d.json" % key)
        RT.need(Path(result["telemetry"]).resolve() == telemetry_path,
                "parent telemetry is not at its writer path")
        telemetry = reader.json(telemetry_path, root=root)
        RT.need(telemetry.get("root") == str(root)
                and telemetry.get("namespace") == context.namespace
                and telemetry.get("recovery_experiment_id") == context.experiment_id
                and telemetry.get("execution_bindings") == binding,
                "parent telemetry identity/execution binding mismatch")
        # A completed long stage closes segments its own size; ordinary stages stay at 120.
        segment_bound = (run["end_episode_cap"] - run["start_episode"]
                         if context.payload.get("target_completed_episodes") is not None else 120)
        end = validate_closed_run(run, result, telemetry, segment_bound=segment_bound)
        start = run["start_episode"]
        resume = telemetry.get("resume") or {}
        initial_state = resume.get("initial_full_state") or {}
        g3_run = g3_by_run[key]
        g3_full_state = RT.digest(g3_run.get("full_state_fingerprint"))
        g3_actor = RT.digest(g3_run.get("actor_fingerprint"))
        donor_full_state = run["donor"].get("expected_full_state_sha256")
        donor_actor = run["donor"].get("expected_actor_sha256")
        if donor_full_state is not None:
            RT.need(RT.digest(donor_full_state) == g3_full_state,
                    "donor and validated G3 full-state pins disagree")
        if donor_actor is not None:
            RT.need(RT.digest(donor_actor) == g3_actor,
                    "donor and validated G3 actor pins disagree")
        donor_episode = run.get("donor_episode", start)
        resume_donor = resume.get("episode")
        resume_start = resume.get("start_episode", resume_donor)
        RT.need(resume_donor == donor_episode and resume_start == start
                and resume.get("next_update") == start // C.EPISODES_PER_UPDATE
                and RT.digest(initial_state.get("sha256")) == g3_full_state
                and RT.digest(initial_state.get("actor")) == g3_actor,
                "parent restored donor/start fingerprint does not match its validated G3")

        previous_full_state = initial_state["sha256"]
        new_rows, new_updates = {}, {}
        planned = {(x["arm"], x["training_seed"], x["episode_index"]): x
                   for x in context.seed_plan}
        plan_slots = {x["slot"]: x for x in context.slot_plan["slots"]}
        for ep, record in zip(range(start, end), telemetry["episodes"]):
            slot = "recovery-train:%s:%d:%04d" % (key[0], key[1], ep)
            flat = slot.replace(":", "_")
            claim = reader.json(root / "execution/ledger/recovery_claims" / (flat + ".claim"),
                                root=root)
            outcome = reader.json(root / "execution/ledger/recovery_results" / (flat + ".json"),
                                  root=root)
            row_path = root / "execution/rows/train" / (flat + ".json")
            row = reader.json(row_path, root=root)
            RT.need(slot not in all_slots, "repeated closed slot")
            all_slots.add(slot)
            RT.need(claim.get("schema") == RT.CLAIM_SCHEMA
                    and outcome.get("schema") == RT.RESULT_SCHEMA
                    and claim.get("slot") == slot and outcome.get("slot") == slot
                    and outcome.get("status") == "ok"
                    and Path(outcome.get("row", "")).resolve() == row_path,
                    "claim/result/row closure mismatch")
            for body in (claim, outcome):
                RT.need(body.get("root") == str(root)
                        and body.get("recovery_experiment_id") == context.experiment_id
                        and body.get("namespace") == context.namespace,
                        "claim/result identity mismatch")
            runtime = claim.get("runtime_key")
            physical = RT.runtime_key(runtime)
            expected_seed = planned.get((key[0], key[1], ep))
            RT.need(expected_seed is not None
                    and claim.get("entry") == plan_slots.get(slot)
                    and runtime == expected_seed
                    and physical[3] == expected_seed["action_rng_seed"],
                    "parent claim differs from its pre-frozen actual seed/slot")
            RT.need(physical[:2] == key
                    and runtime.get("episode_index") == ep
                    and physical[2] == C.scene_for(key[1], ep),
                    "claim runtime key/run/scene mismatch")
            RT.need(physical not in old_keys, "parent repeated an already consumed rollout")
            old_keys.add(physical)
            RT.need(row.get("slot") == slot and row.get("episode_index") == ep
                    and row.get("arm") == key[0] and row.get("seed") == key[1]
                    and row.get("scene") == physical[2]
                    and row.get("action_rng_seed") == physical[3]
                    and row.get("budget_bytes") == C.BUDGET_BYTES
                    and row.get("generated_goals") == 24 and row.get("steps") == 1000
                    and 0 <= row.get("on_time", -1) <= 24
                    and 0 <= row.get("charged_bytes", -1) <= C.BUDGET_BYTES,
                    "closed parent row science/runtime mismatch")
            if context.loss_review_enabled:
                origin = row.get("origin") or {}
                RT.need(origin.get("stage") == context.stage_index,
                        "parent row provenance stage does not match its frozen descendant")
            RT.need(record.get("slot") == slot and record.get("episode_index") == ep
                    and record.get("action_rng_seed") == physical[3],
                    "telemetry/claim runtime mismatch")
            new_rows[ep] = row
        for update, offset in zip(telemetry["updates"], range(start // 4, end // 4)):
            jp = (root / "execution/recovery/update_journal" / ("%s_s%d" % key)
                  / ("update_%06d.json" % offset))
            journal = reader.json(jp, root=root)
            RT.need(journal.get("schema") == RT.JOURNAL_SCHEMA
                    and journal.get("root") == str(root)
                    and journal.get("recovery_experiment_id") == context.experiment_id
                    and journal.get("arm") == key[0]
                    and journal.get("training_seed") == key[1]
                    and journal.get("update_index") == offset
                    and journal.get("update") == update
                    and journal.get("episode_indices") == list(range(offset * 4, offset * 4 + 4))
                    and journal.get("episodes") ==
                        telemetry["episodes"][(offset - start // 4)*4:(offset - start // 4 + 1)*4],
                    "journal/telemetry update closure mismatch")
            RT.digest(journal["full_state_before"])
            RT.digest(journal["full_state_after"])
            RT.need(journal["full_state_before"] == previous_full_state,
                    "parent update journal state chain has a gap/conflict")
            previous_full_state = journal["full_state_after"]
            for rec in journal["episodes"]:
                state = new_rows[rec["episode_index"]].get("state_hashes") or {}
                RT.need(state.get("full_state_sha256_after") == journal["full_state_after"]
                        and state.get("update_index") == offset,
                        "row state/update does not bind the journal")
            new_updates[offset] = update
        rows[key], updates[key] = merge_prefix(rows[key], updates[key], new_rows, new_updates,
                                              start, end, key)
        new_updates_by_run[key] = new_updates
        RT.need(telemetry.get("final_full_state", {}).get("sha256") == previous_full_state,
                "parent terminal fingerprint does not close the journal chain")
        parent_checks = RT.read_check_certificates(context, Layout(root), *key)
        for check in parent_checks:
            pin = reader.pins.get(str(Path(check["path"]).resolve()))
            if pin is None:
                reader.raw(check["path"])
                pin = reader.pins[str(Path(check["path"]).resolve())]
            body = reader.pinned(pin)
            source = {"arm": key[0], "training_seed": key[1],
                      "experiment_id": body["experiment_id"], **pin}
            if source not in certificates:
                certificates.append(source)
        certified, cert_end = validate_task_termination(
            run, result, parent_checks, end, context.loss_review_enabled,
            long_stage=context.payload.get("target_completed_episodes") is not None)
        cp = Layout(root).checkpoint(*key, end)
        reader.raw(cp, root=root)  # bytes only, never torch.load
        cp_pin = reader.pins[str(cp.resolve())]
        advertised = [x for x in checkpoint_inventory
                      if x.get("run") == "%s_%d" % key]
        RT.need(len(advertised) == 1 and advertised[0]["episode"] == end,
                "parent terminal summary CP cursor mismatch")
        reader.raw(cp, {"path": str(cp), "bytes": advertised[0]["bytes"],
                        "sha256": advertised[0]["sha256"]}, root)
        last_journal = reader.json(root / "execution/recovery/update_journal" /
                                   ("%s_s%d" % key) / ("update_%06d.json" % (end // 4 - 1)))
        full_state = telemetry.get("final_full_state") or {}
        RT.need(RT.digest(full_state.get("sha256")) == last_journal["full_state_after"],
                "terminal state does not match the last real update")
        certificate_path = root / "execution/recovery/certificates" / (
            "CERTIFICATE_%s_s%d.json" % key)
        RT.need(Path(result["certificate"]).resolve() == certificate_path,
                "run certificate writer path mismatch")
        certificate = reader.json(certificate_path, root=root)
        advertised_cert = [x for x in certificate_inventory
                           if x.get("run") == "%s_%d" % key]
        RT.need(len(advertised_cert) == 1, "parent summary certificate missing/duplicate")
        reader.raw(certificate_path, {"path": str(certificate_path),
                   "bytes": advertised_cert[0]["bytes"],
                   "sha256": advertised_cert[0]["sha256"]}, root)
        RT.need(certificate.get("status") ==
                ("CONVERGED" if certified else "NOT_YET_CONVERGED")
                and certificate.get("terminal_episode") == end
                and certificate.get("consecutive_passes") == convergence.consecutive_passes(parent_checks)
                and RT.digest(certificate.get("telemetry_sha256")) ==
                    reader.pins[str(telemetry_path)]["sha256"],
                "parent terminal certificate not closed")
        if context.loss_review_enabled:
            RT.need(certificate.get("task_certificate_end_episode") == cert_end
                    and certificate.get("continued_after_task_certificate")
                        is (certified and end > cert_end),
                    "parent stage certificate metadata does not preserve its task event")
        # CP cursor/next_update must later be observed inside the payload by G3.
        donor = {**cp_pin, "identity_sha256": context.payload["identity"]["sha256"],
                 "cursor_from_closed_journal": end, "next_update": end // 4,
                 "next_update_basis": "CLOSED_LAST_JOURNAL_REQUIRE_PAYLOAD_MATCH",
                 "expected_full_state_sha256": full_state["sha256"],
                 "expected_actor_sha256": full_state["actor"],
                 "stored_adam_parameter_step": None,
                 "payload_observation": "NOT_LOADED_G3_REQUIRED"}
        states[key] = {"cursor": end, "next_update": end // 4,
                       "certified": certified, "task_certified": certified,
                       "task_certificate_end_episode": cert_end,
                       "loss_stability_stable": False,
                       "loss_stability_review": {"status": "NEW_PARENT_DATA_REQUIRES_REVIEW"},
                       "latest_telemetry_sha256": reader.pins[str(telemetry_path.resolve())]["sha256"],
                       "donor": donor}
        states[key]["completed_episode_count"] = len(rows[key])
        states[key]["registered_missing_episode_count"] = (
            end - states[key]["completed_episode_count"])
    actual_claims = {p.stem.replace("_", ":") for p in
                     (root / "execution/ledger/recovery_claims").glob("*.claim")}
    # Filenames have no underscores inside run labels. Keep all claims; extras are fatal here.
    RT.need(actual_claims == all_slots, "parent has unresolved/extra/missing claims")
    for folder, suffix in (("execution/ledger/recovery_results", "*.json"),
                           ("execution/rows/train", "*.json")):
        RT.need(len(tuple((root / folder).glob(suffix))) == len(all_slots),
                "parent extra/missing result or row files")
    count = len(all_slots)
    if reconstruction is None:
        RT.need(summary.get("claims") == count == summary.get("results")
                and summary.get("update_journals") == count // 4
                and detail.get("worlds") == count
                and detail.get("slot_counts", {}).get("claimed") == count
                and detail.get("slot_counts", {}).get("ok") == count,
                "complete parent world/update totals do not close")
    elif reconstruction.get("schema") == "stage5b-recovery06-terminal-evidence/1":
        RT.need(detail.get("worlds") == count
                and detail.get("slot_counts", {}).get("claimed") == count
                and detail.get("slot_counts", {}).get("results") == count
                and detail.get("slot_counts", {}).get("ok") == count,
                "writer-native terminal parent counts do not close against claims and journals")
    elif reconstruction.get("schema") == GENERIC_RECONSTRUCTION_SCHEMA:
        RT.need(detail.get("worlds") == count
                and detail.get("slot_counts", {}).get("claimed") == count
                and detail.get("slot_counts", {}).get("results") == count
                and detail.get("slot_counts", {}).get("ok") == count
                and summary.get("claims") == summary.get("results") == count
                and summary.get("update_journals") == count // 4
                and sum(len(x) for x in new_updates_by_run.values()) == count // 4,
                "generic reconstructed parent counts do not close against claims and journals")
    else:
        RT.need(detail.get("worlds") == count == 720
                and detail.get("slot_counts", {}).get("claimed") == count
                and detail.get("slot_counts", {}).get("results") == count
                and detail.get("slot_counts", {}).get("ok") == count
                and sum(len(x) for x in new_updates_by_run.values()) == count // 4,
                "recovery_02 reconstructed parent counts do not close against claims and journals")
    # A long-stage parent is bounded by its own frozen world cap, not by 720.
    cap_bound = (context.slot_plan["world_cap"]
                 if context.payload.get("target_completed_episodes") is not None else 720)
    RT.need(count <= sum(r["new_world_cap"] for r in context.runs) <= cap_bound,
            "parent exceeds its finite registered cap")
    RT.need(len(tuple((root / "execution/recovery/update_journal").glob("*/*.json")))
            == count // 4, "parent extra/missing update-journal files")
    if reconstruction is not None:
        reconstruction["g3_status"] = "PASS_RESTORED_OBSERVED_VERIFIED_BY_FROZEN_VALIDATOR"
        if reconstruction.get("schema") == RECONSTRUCTION_SCHEMA:
            reconstruction["training_state_chain"] = "DONOR_FINGERPRINT_TO_180_JOURNALS_TO_TERMINAL_TELEMETRY_VERIFIED"
        else:
            reconstruction["training_state_chain"] = "DONOR_FINGERPRINT_TO_CLOSED_JOURNALS_TO_TERMINAL_TELEMETRY_VERIFIED"
        reconstruction["claim_result_row_update_closure"] = "%d_CLOSED_ROWS_%d_JOURNALS_ZERO_DUPLICATE_PHYSICAL_KEYS" % (count, count // 4)
    closure = sorted(reader.pins.values(), key=lambda p: p["path"])
    inherited_reconstruction = (context.descendant.get("parent_source_reconstruction")
                                if getattr(context, "is_descendant", False) else None)
    return types.SimpleNamespace(
        context=context, runs=context.runs, states=states, rows=rows, updates=updates,
        old_keys=frozenset(old_keys), certificates=certificates, closure=closure,
        run_directory=str(run_dir), reader=reader,
        source_reconstruction=(reconstruction or inherited_reconstruction))


def admission_body(parent, runs, include_cutover=False):
    unknown = 0
    for run in runs:
        gap = R.known_gap(run["arm"], run["training_seed"])
        if gap is not None and ((run["arm"], run["training_seed"]) in R.KNOWN_GAPS or include_cutover):
            unknown += int(gap["updates"][1]) - int(gap["updates"][0])
    body = {
        "schema": "stage5b-completed-parent-admission/1",
        "status": "COMPLETED_PARENT_PREFIX_VALIDATED_G3_REQUIRED",
        "admission": "NO_WORLD_AUTHORIZATION",
        "parent_context_sha256": parent.context.context_pin["sha256"],
        "parent_closure_sha256": RT.canonical_digest(parent.closure),
        "donor_set_sha256": RT.canonical_digest([r["donor"] for r in runs]),
        "selected_rows": sum(len(x) for x in parent.rows.values()),
        "actual_updates": sum(len(x) for x in parent.updates.values()),
        "unknown_unfilled_updates": unknown,
        "world_cap": sum(r["new_world_cap"] for r in runs),
    }
    reconstruction = getattr(parent, "source_reconstruction", None)
    if reconstruction is not None:
        body["parent_source_reconstruction"] = reconstruction
    return body


def slot_body(root, experiment_id, seeds):
    slots = [{"slot": "recovery-train:%s:%d:%04d" %
              (x["arm"], x["training_seed"], x["episode_index"]),
              "phase": "train", "arm": x["arm"], "training_seed": x["training_seed"],
              "episode": x["episode_index"], "scene": x["scene_seed"],
              "action_rng_seed": x["action_rng_seed"]} for x in seeds]
    return {"schema": RT.SLOT_PLAN_SCHEMA, "status": "FROZEN", "root": str(root),
            "recovery_experiment_id": experiment_id, "namespace": R.WORLD_SEED_NAMESPACE,
            "world_cap": len(seeds), "slots": slots,
            "note": "MECHANICALLY_DERIVED_COMPLETED_PARENT_FINITE_ACTIVE_RUNS"}


def identity_body(context, descriptor):
    body = {
        "schema": "stage5b-descendant-runtime-identity/1", "status": "FROZEN",
        "root": context["root"], "recovery_experiment_id": context["recovery_experiment_id"],
        "namespace": context["namespace"],
        "bindings": {name: context[name]["sha256"] for name in (
            "descendant", "admission_report", "source_manifest", "seed_plan",
            "slot_plan", "candidate_manifest", "candidate_index")},
        "parent_context_sha256": descriptor["parent_context"]["sha256"],
        "science_profile_sha256": SCIENCE_SHA,
    }
    # Keep legacy recovery_02 identity bytes compatible while reading that parent.
    if "stage_index" in descriptor:
        body["stage_index"] = descriptor["stage_index"]
        review = descriptor.get("loss_stability_review")
        body["loss_stability_review_sha256"] = None if review is None else review["sha256"]
    return body

def load_context(layout, path):
    root = canonical(layout.root)
    RT.need(Path(path).resolve() == root / RT.CONTEXT_RELATIVE,
            "descendant context must use its canonical frozen writer path")
    reader = Reads()
    context, _ = _base(root, reader, frozenset())
    RT.need(context.is_descendant, "explicit descendant schema required")
    context.world_cap = context.slot_plan["world_cap"]
    return context


def build_next_stage(parent_root, newroot, parent_run_directory=None, parent_provenance_path=None, target_completed_episodes=None, checkpoint_interval_episodes=None, retired_runs=frozenset()):
    """Build the next finite continuation from one completed recovery stage."""
    root = canonical(newroot)
    parent_root = Path(parent_root).resolve()
    RT.need(parent_root != root, "new stage cannot reuse the parent root")
    parent = completed_parent(parent_root, parent_run_directory,
                              provenance_path=parent_provenance_path)
    loss_review_pin, loss_reviews = load_loss_stability_review(root, parent)
    stage_index = next_stage_index(parent.context.stage_index)
    if target_completed_episodes is None:
        checkpoint_interval_episodes = None
    else:
        from . import long_stage as L
        target_completed_episodes, checkpoint_interval_episodes = L.validate_parameters(
            target_completed_episodes,
            checkpoint_interval_episodes or L.CHECKPOINT_INTERVAL_EPISODES)
        RT.need(checkpoint_interval_episodes == L.CHECKPOINT_INTERVAL_EPISODES,
                "long stage must reuse the frozen 120-episode checkpoint interval")
    retired_runs = frozenset(RT.normalized_run(arm, seed) for arm, seed in retired_runs)
    for key in retired_runs:
        RT.need(key in parent.states, "retired run is not in the parent roster: %r" % (key,))
        parent_target = parent.context.payload.get("target_completed_episodes")
        completed = parent.states[key].get("completed_episode_count")
        RT.need(parent_target is not None and completed is not None
                and int(completed) == int(parent_target),
                "a retired run must already sit at its parent completed target: %r" % (key,))
    runs = next_runs(parent.runs, parent.states, loss_reviews,
                     include_completion_counts=parent.context.stage_index >= 5,
                     include_minimum_target=stage_index >= 7 or target_completed_episodes is not None,
                     target_completed_episodes=target_completed_episodes,
                     retired_runs=retired_runs)
    seeds = seed_plan(runs, set(parent.old_keys))
    RT.need(seeds, "all six runs passed task and loss review; no training stage is permitted")
    RT.need(len(seeds) == sum(r["new_world_cap"] for r in runs) and len(seeds) > 0,
            "six-run continuation seed total does not close")
    if target_completed_episodes is None:
        RT.need(len(seeds) <= 720, "ordinary stage exceeds the finite 720-world cap")
    exp = "stage5b_comm_recovery_" + root.parent.name + "_" + root.name
    generated = set(RT.GENERATED_PROTOCOL_PATHS) | {DESCRIPTOR, ADMISSION,
                                                    "protocol/RECOVERY_NEXT_SEEDS.json"}
    files, pins = {}, {}

    def put(relative, value):
        data = compact.json_bytes(value)
        files[relative] = data
        pins[relative] = {"path": str(root / relative), "bytes": len(data),
                          "sha256": RT.sha256_bytes(data)}
        return pins[relative]

    source = RT.build_source_manifest(Layout(root))
    native = {x["path"]: x["sha256"] for x in source["files"]
              if x["category"] == "native"}
    expected_native = {x["path"]: x["sha256"] for x in
                       parent.context.source_manifest["files"] if x["category"] == "native"}
    RT.need(native == expected_native, "native files differ from the frozen parent/base")
    source_pin = put(RT.SOURCE_MANIFEST_RELATIVE, source)
    plan_pin = put(RT.SLOT_PLAN_RELATIVE, slot_body(root, exp, seeds))
    seed_pin = put("protocol/RECOVERY_NEXT_SEEDS.json", seeds)
    inherited_cutover = (parent.context.descendant.get("recovery_05_consumed_evidence")
                         if getattr(parent.context, "is_descendant", False) else None)
    admission_pin = put(ADMISSION, admission_body(
        parent, runs, include_cutover=inherited_cutover is not None))
    descriptor = {
        "schema": SCHEMA, "status": "FROZEN", "root": str(root),
        "stage_index": stage_index,
        "loss_stability_review": loss_review_pin,
        "science_profile_sha256": SCIENCE_SHA,
        "science_profile": parent.context.payload["protocol"],
        "parent_context": parent.context.context_pin,
        "parent_run_directory": parent.run_directory,
        "parent_provenance_path": (str(Path(parent_provenance_path).resolve())
                                   if parent_provenance_path else None),
        "parent_closure": parent.closure,
        "parent_source_reconstruction": getattr(parent, "source_reconstruction", None),
        "branch_mode": "COMPLETED_PARENT_CHECKPOINT_CONTINUATION",
        "inherited_recovery_05_cutover_evidence": inherited_cutover,
        "source_manifest_sha256": source_pin["sha256"],
        "source_equality_basis": "COMPLETE_PARENT_CLOSED_LEDGER_AND_JOURNALS",
        "checkpoint_payload_loading": "NOT_PERFORMED_G3_REQUIRED",
        "target_completed_episodes": target_completed_episodes,
        "checkpoint_interval_episodes": checkpoint_interval_episodes,
        "retired_runs": sorted("%s:%d" % RT.normalized_run(arm, seed)
                               for arm, seed in retired_runs),
    }
    desc_pin = put(DESCRIPTOR, descriptor)
    context = {
        "schema": CONTEXT_SCHEMA, "status": RT.CONTEXT_STATUS, "root": str(root),
        "recovery_experiment_id": exp, "namespace": R.WORLD_SEED_NAMESPACE,
        "stage_index": stage_index, "loss_stability_review": loss_review_pin,
        "target_completed_episodes": target_completed_episodes,
        "checkpoint_interval_episodes": checkpoint_interval_episodes,
        "descendant": desc_pin, "runs": runs,
        "admission_report": admission_pin,
        "protocol": parent.context.payload["protocol"],
        "candidate_manifest": parent.context.payload["candidate_manifest"],
        "candidate_index": parent.context.payload["candidate_index"],
        "source_manifest": source_pin, "seed_plan": seed_pin, "slot_plan": plan_pin,
        "certificate_sources": parent.certificates,
        "g3_report": None, "world_authorization": None,
    }
    context["identity"] = put(RT.IDENTITY_RELATIVE, identity_body(context, descriptor))
    put(RT.CONTEXT_RELATIVE, context)
    RT.need(set(files) == generated, "descendant artifact set incomplete")
    return {"files": files, "pins": pins, "world_cap": len(seeds),
            "runs": runs, "authorization": "G3_NOT_RUN_WORLD_NOT_AUTHORIZED"}

def write_bundle(bundle, newroot):
    root = canonical(newroot)
    RT.need(type(bundle) is dict and set(bundle.get("files", {}))
            == set(RT.GENERATED_PROTOCOL_PATHS), "wrong descendant artifact set")
    for relative in bundle["files"]:
        RT.evidence.inside(root, root / relative)
        RT.need(not (root / relative).exists(), "refuse overwrite of frozen descendant artifact")
    # Recheck the consumed parent/code pins immediately before publishing any artifact.
    reader = Reads()
    descriptor = RT.strict_json_bytes(bundle["files"][DESCRIPTOR], "built descriptor")
    parent_root = Path(descriptor["parent_context"]["path"]).resolve().parents[1]
    RT.need(not (parent_root / "execution/ledger" / RT.STOP_SIGNAL).exists(),
            "parent STOP appeared after prepare collection")
    for pin in descriptor["parent_closure"]:
        reader.raw(pin["path"], pin)
    review_pin = descriptor.get("loss_stability_review")
    if review_pin is not None:
        RT.need(Path(review_pin["path"]).resolve() == root / LOSS_REVIEW_RELATIVE,
                "loss-stability review moved after prepare")
        reader.raw(review_pin["path"], review_pin, root=root)
    inherited_cutover = descriptor.get("inherited_recovery_05_cutover_evidence")
    if inherited_cutover is not None:
        RT.need(isinstance(inherited_cutover, dict),
                "inherited recovery_05 cutover pin is malformed")
        reader.raw(inherited_cutover["path"], inherited_cutover)
    source = RT.strict_json_bytes(bundle["files"][RT.SOURCE_MANIFEST_RELATIVE],
                                  "built source manifest")
    for item in source["files"]:
        reader.raw(item["path"], {k: item[k] for k in RT.PIN_KEYS})
    # Base context is published last; partial failure cannot create a runnable base.
    for relative in [r for r in bundle["files"] if r != RT.CONTEXT_RELATIVE] + [
            RT.CONTEXT_RELATIVE]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # A long-stage parent carries one closed-row pin per consumed World in its closure,
        # so only the descendant descriptor gets the wider allowance; every other artifact
        # keeps the frozen 10 MB cap.
        cap = DESCRIPTOR_CAP_BYTES if relative == DESCRIPTOR else None
        compact.write_bytes_exclusive(path, bundle["files"][relative], cap_bytes=cap)
    return bundle["pins"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare a completed-parent finite continuation")
    parser.add_argument("--parent", required=True)
    parser.add_argument("--newroot", required=True)
    parser.add_argument("--parent-run-directory")
    parser.add_argument("--parent-provenance")
    parser.add_argument("--target-completed-episodes", type=int)
    parser.add_argument("--checkpoint-interval-episodes", type=int, default=120)
    parser.add_argument("--retire-run", action="append", default=[],
                        metavar="ARM:SEED",
                        help="do not continue this run; it must sit at its parent completed target")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    retired = []
    for item in args.retire_run:
        arm, _, seed = item.partition(":")
        retired.append((arm, int(seed)))
    bundle = build_next_stage(args.parent, args.newroot, args.parent_run_directory, args.parent_provenance, args.target_completed_episodes, args.checkpoint_interval_episodes, retired_runs=frozenset(retired))
    if args.write:
        write_bundle(bundle, args.newroot)
    print(compact.dumps({"mode": "WRITTEN_EXCLUSIVE" if args.write else "NOT_WRITTEN",
                         "world_cap": bundle["world_cap"],
                         "runs": bundle["runs"], "pins": bundle["pins"],
                         "authorization": bundle["authorization"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())