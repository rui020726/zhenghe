"""Pure stdlib R76 row/tape persistence and reconstruction contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import r76_contracts as contracts

TAPE_SCHEMA = "round42-one-shot-send-skip-tape/1"
SEND = 1
HOLD = 0
FEATURE_WIDTH = 19


class EpisodeProtocolError(RuntimeError):
    """A row or tape cannot be reconstructed without inventing missing data."""


def _validate_tape_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate an ordered tape without sorting or accepting future-state pollution."""
    validated: list[dict[str, Any]] = []
    known_targets: set[str] = set()
    previous_step: int | None = None
    for index, record in enumerate(records):
        step = int(record["step"])
        if previous_step is not None and step <= previous_step:
            raise EpisodeProtocolError(
                f"tape record {index} step {step} is not strictly increasing after "
                f"{previous_step}")
        validated.append(validate_tape_record(
            record, record_index=index, historical_target_ids=known_targets))
        known_targets.update(record["targets"])
        previous_step = step
    return validated


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EpisodeProtocolError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EpisodeProtocolError(f"{field} must be finite")
    return result


def validate_tape_record(tape: Any, *, record_index: int | None = None,
                         historical_target_ids: set[str] | None = None) -> dict[str, Any]:
    label = "tape record" if record_index is None else f"tape record {record_index}"
    contracts.validate_exact_keys(tape, contracts.R76_REQUIRED_TAPE_FIELDS, label)
    if tape["schema"] != TAPE_SCHEMA:
        raise EpisodeProtocolError(f"{label} schema is not {TAPE_SCHEMA}")
    step = int(tape["step"])
    if step < 0:
        raise EpisodeProtocolError(f"{label} has a negative step")
    targets = tape["targets"]
    if not isinstance(targets, list) or not targets or \
            any(not isinstance(target, str) or not target for target in targets) \
            or len(targets) != len(set(targets)):
        raise EpisodeProtocolError(f"{label} targets must be non-empty unique strings")
    features = tape["features"]
    if not isinstance(features, list) or len(features) != len(targets):
        raise EpisodeProtocolError(f"{label} features do not match targets")
    for row_index, row in enumerate(features):
        if not isinstance(row, list) or len(row) != FEATURE_WIDTH \
                or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                       or not math.isfinite(float(value)) for value in row):
            raise EpisodeProtocolError(f"{label} feature row {row_index} is not 19 finite numbers")
    actions = tape["actions"]
    if not isinstance(actions, dict) or set(actions) != set(targets) \
            or any(value not in (SEND, HOLD) for value in actions.values()):
        raise EpisodeProtocolError(f"{label} actions are not SEND/HOLD for every target")
    if actions != tape["effective_actions"]:
        raise EpisodeProtocolError(f"{label} effective actions do not match actions")
    requested = tape["requested"]
    if not isinstance(requested, dict) or set(requested) != set(targets) or \
            any(not isinstance(value, bool) for value in requested.values()):
        raise EpisodeProtocolError(f"{label} requested is not boolean for every target")
    # In the real collector, free_one_shot_choices/free_first_reports/free/actionable are
    # aliases of the same free_choices list.  The four mutually exclusive collector classes
    # are forced / masked / fixed_hold_one_shot_closed / actionable.
    required_sets = {
        "free": tape["free"],
        "actionable": tape["actionable"],
        "masked": tape["masked"],
        "fixed_hold_one_shot_closed": tape["fixed_hold_one_shot_closed"],
        "forced": tape["forced"],
    }
    for name, values in required_sets.items():
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise EpisodeProtocolError(f"{label} {name} is not a list of target ids")
    categories = [set(tape["forced"]), set(tape["masked"]),
                  set(tape["fixed_hold_one_shot_closed"]), set(tape["actionable"])]
    if any(categories[i] & categories[j] for i in range(len(categories))
           for j in range(i + 1, len(categories))):
        raise EpisodeProtocolError(f"{label} candidate classes overlap")
    if set(targets) != set().union(*categories):
        raise EpisodeProtocolError(f"{label} candidate classes do not cover all targets")
    if tape["free_one_shot_choices"] != tape["free"] \
            or tape["free_first_reports"] != tape["free"] \
            or tape["free"] != tape["actionable"]:
        raise EpisodeProtocolError(f"{label} free/actionable collector aliases disagree")
    forced_set = set(tape["forced"])
    candidate_audit = tape["candidate_audit"]
    if not isinstance(candidate_audit, dict) or \
            set(candidate_audit) != set(targets) - forced_set:
        raise EpisodeProtocolError(
            f"{label} candidate audit must cover free/masked/closed candidates and omit only "
            f"transport-forced candidates")
    charges = tape["attempt_charge_bytes"]
    quote_required = set(tape["free"]) | set(tape["masked"])
    if not isinstance(charges, dict) or \
            not quote_required.issubset(charges) or \
            any(target in charges for target in forced_set):
        raise EpisodeProtocolError(
            f"{label} relay quotes must cover free/masked candidates and omit forced candidates; "
            f"closed candidates may omit a relay-owned None quote")
    for target, charge in charges.items():
        if target not in targets:
            raise EpisodeProtocolError(f"{label} has a relay quote for an unknown target {target}")
        if charge is not None and (isinstance(charge, bool) or int(charge) < 0):
            raise EpisodeProtocolError(f"{label} has an invalid relay quote for {target}")
        if target in quote_required:
            audit = candidate_audit[target]
            if not isinstance(audit, dict) or audit.get("attempt_charge_bytes") != charge:
                raise EpisodeProtocolError(
                    f"{label} audit and relay quote disagree for {target}")
    executable = tape["send_executable_initial"]
    block_reason = tape["send_block_reason_initial"]
    if not isinstance(executable, dict) or set(executable) != set(targets) or \
            any(value is not None and not isinstance(value, bool) for value in executable.values()):
        raise EpisodeProtocolError(f"{label} executable-initial is not valid for every target")
    if not isinstance(block_reason, dict) or set(block_reason) != set(targets) or \
            any(value is not None and (not isinstance(value, str) or not value)
                for value in block_reason.values()):
        raise EpisodeProtocolError(f"{label} block-reason-initial is not valid for every target")
    decision_class = tape["decision_class"]
    if not isinstance(decision_class, dict) or set(decision_class) != set(targets):
        raise EpisodeProtocolError(f"{label} decision class does not cover every target")
    expected_classes = ({target: "transport_forced" for target in tape["forced"]} |
                        {target: "free_one_shot_choice" for target in tape["actionable"]} |
                        {target: "existing_budget_or_capacity_mask" for target in tape["masked"]} |
                        {target: "fixed_hold_one_shot_closed"
                         for target in tape["fixed_hold_one_shot_closed"]})
    if any(decision_class[target] != expected_classes[target] for target in targets):
        raise EpisodeProtocolError(f"{label} decision class disagrees with candidate classes")
    for name in ("shadow_budget_before", "shadow_budget_after"):
        if _finite_number(tape[name], f"{label}.{name}") < 0:
            raise EpisodeProtocolError(f"{label}.{name} is negative")
    for name in ("shadow_capacity_before", "shadow_capacity_after"):
        value = tape[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise EpisodeProtocolError(f"{label}.{name} is not a valid capacity")
    one_shot_state = tape["one_shot_state_after_step"]
    allowed_states = {"undecided", "send_now_chosen", "skip_episode_chosen",
                      "closed_by_prior_real_emission"}
    # The episode-level target_state snapshot retains already-decided targets after they
    # leave the current candidate set.  Restrict it to current or previously seen targets;
    # read/write/rebuild thread that history through the tape in step order.
    allowed_target_ids = set(targets)
    if historical_target_ids is not None:
        if any(not isinstance(target, str) or not target
               for target in historical_target_ids):
            raise EpisodeProtocolError(f"{label} historical target context is invalid")
        allowed_target_ids.update(historical_target_ids)
    if not isinstance(one_shot_state, dict) or \
            any(target not in allowed_target_ids or value not in allowed_states
                for target, value in one_shot_state.items()):
        raise EpisodeProtocolError(f"{label} one-shot state contains an invalid target or state")
    _finite_number(tape["log_prob"], f"{label}.log_prob")
    _finite_number(tape["value"], f"{label}.value")
    _finite_number(tape["cost_value"], f"{label}.cost_value")
    return tape


def validate_row(row: Any) -> dict[str, Any]:
    contracts.validate_exact_keys(row, contracts.R76_REQUIRED_ROW_FIELDS, "episode row")
    if row.get("collector") is not None:
        raise EpisodeProtocolError("a persisted row may not contain a live collector object")
    for name in ("cost_stream", "task_stream"):
        stream = row[name]
        if not isinstance(stream, dict):
            raise EpisodeProtocolError(f"row {name} must be a mapping")
        if name == "cost_stream":
            for step, entry in stream.items():
                _finite_number(entry.get("airtime_bytes"), f"cost_stream[{step}].airtime_bytes")
        else:
            for step, value in stream.items():
                try:
                    step_number = int(step)
                except (TypeError, ValueError) as error:
                    raise EpisodeProtocolError(
                        f"row {name} has a non-integer step {step!r}") from error
                if step_number < 0:
                    raise EpisodeProtocolError(f"row {name} has a negative step")
                _finite_number(value, f"task_stream[{step}]")
    steps = int(row["steps"])
    if steps <= 0:
        raise EpisodeProtocolError("row steps must be positive")
    for name in ("scenario_seed", "attempt", "training_seed", "episode_index"):
        if not isinstance(row[name], int) or isinstance(row[name], bool):
            raise EpisodeProtocolError(f"row {name} must be an integer")
    if _finite_number(row["spent_bytes"], "spent_bytes") < 0 \
            or _finite_number(row["budget_bytes"], "budget_bytes") <= 0:
        raise EpisodeProtocolError("row byte totals are not valid")
    if float(row["b0_bytes"]) != 10000.0:
        raise EpisodeProtocolError("row B0 is not 10000")
    _finite_number(row["lambda"], "lambda")
    artefacts = row["artefacts"]
    contracts.validate_exact_keys(artefacts, ("row", "tape", "lifecycle", "transcript"),
                                  "row artefacts")
    return row


def write_row(path, row: dict[str, Any]) -> str:
    safe = {key: value for key, value in row.items() if key != "collector"}
    validate_row(safe)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return str(path)


def read_row(path) -> dict[str, Any]:
    return validate_row(json.loads(Path(path).read_text(encoding="utf-8")))


def write_tape(path, records: list[dict[str, Any]]) -> str:
    validated = _validate_tape_records(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in validated:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return str(path)


def read_tape(path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return _validate_tape_records(records)


class PersistedCollector:
    def __init__(self, records: list[dict[str, Any]]):
        self.records = list(records)


def rebuild_episode(row: dict[str, Any], tape: list[dict[str, Any]]) -> dict[str, Any]:
    safe_row = validate_row({key: value for key, value in row.items()
                             if key != "collector"})
    validated_tape = _validate_tape_records(tape)
    steps = sorted(int(record["step"]) for record in validated_tape)
    if len(steps) != len(set(steps)):
        raise EpisodeProtocolError("tape steps are not unique")
    episode = dict(safe_row)
    episode["collector"] = PersistedCollector(validated_tape)
    return episode
