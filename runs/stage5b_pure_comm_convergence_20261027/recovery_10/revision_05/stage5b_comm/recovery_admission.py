"""Strict offline RecoveryCheckpointPrefix admission validation.

This module reads a collector-owned candidate, an independently pinned training-evidence
index, and a separately supplied main-model approval file. It never creates approval,
marks a candidate ADMITTED, writes a parent-admission document, loads a checkpoint, creates a
World/model, or performs materialization. Success means only that the bundle is eligible for
a later, separately authorized materialization step.

Operator/session authenticity and G3 restoration remain external proof obligations. A
`CANDIDATE_NOT_ADMITTED` status, UNKNOWN payload fields, and NOT_PERFORMED G3 are required and
are never converted into success here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from . import constants as C
from . import recovery_evidence as E
from . import recovery_protocol as R

CANONICAL_ROOT = E.CANONICAL_ROOT
DRAFT_SHA256 = E.DRAFT_SHA
COLLECTOR_CODE_SHA256 = (
    "b34433cf74376e7f8d809a377f8cb5afc09d932ce899988c8a9ce80529cac40d"
)
WORLD_SEED_NAMESPACE = R.WORLD_SEED_NAMESPACE
RECOVERY_EXPERIMENT_ID = R.RECOVERY_STAGE_ID
CANDIDATE_SCHEMA = "stage5b-recovery-evidence-candidate/1"
CANDIDATE_STATUS = "CANDIDATE_NOT_ADMITTED"
CANDIDATE_ADMISSION = "NOT_IMPLEMENTED_NOT_ADMITTED"
APPROVAL_SCHEMA = "stage5b-recovery-main-approval/1"
EVIDENCE_INDEX_SCHEMA = "stage5b-recovery-training-freeze-evidence-index/1"
RESULT_SCHEMA = "stage5b-recovery-admission-validation/1"
RESULT_STATUS = "ELIGIBLE_FOR_MATERIALIZATION_NOT_ADMITTED"
FINAL_ADMISSION = "NOT_ADMITTED"
SOURCE_ALIASES = E.SOURCE_ALIASES
HEX = re.compile(r"[0-9a-fA-F]{64}\Z")
CANDIDATE_NAME = re.compile(r"candidate_[0-9]+_[0-9]+\Z")
UINT64_MAX = (1 << 64) - 1

PRODUCTION_COUNTS = {
    "selected_rows": 6120,
    "logical_updates": 1530,
    "actual_updates": 1260,
    "unknown_unfilled_update_count": 270,
    "old_rollout_record_count": 6169,
    "old_unique_runtime_key_count": 6165,
    "controlled_replay_pairs": 4,
    "quarantined_successful_rows": 36,
    "quarantined_unclosed_claims": 9,
    "proposed_runtime_seed_keys": 720,
    "candidate_output_files": 139,
    "index_outputs": 138,
    "history_chunks": 51,
    "update_chunks": 14,
    "old_rollout_chunks": 52,
    "source_pin_chunks": 19,
    "source_pin_paths": 18538,
}

EXPECTED_HISTORY = {
    ("T1", 91001): (1320, 330, ()),
    ("T1", 91002): (1320, 330, ()),
    ("T1", 91003): (960, 150, tuple(range(150, 240))),
    ("T2", 91001): (960, 150, tuple(range(150, 240))),
    ("T2", 91002): (960, 150, tuple(range(150, 240))),
    ("T2", 91003): (600, 150, ()),
}

EXPECTED_KNOWN_GAPS = [
    {"arm": "T1", "training_seed": 91003,
     "episodes_half_open": [600, 960], "updates_half_open": [150, 240],
     "missing_updates": 90},
    {"arm": "T2", "training_seed": 91001,
     "episodes_half_open": [600, 960], "updates_half_open": [150, 240],
     "missing_updates": 90},
    {"arm": "T2", "training_seed": 91002,
     "episodes_half_open": [600, 960], "updates_half_open": [150, 240],
     "missing_updates": 90},
]

CANDIDATE_KEYS = frozenset((
    "G3", "actual_updates", "admission", "controlled_replay_records_retained",
    "donors", "draft_sha256", "inventory", "known_gaps", "limits",
    "logical_updates", "main_model_approval", "old_rollout_record_count",
    "old_unique_runtime_key_count", "output_files", "quarantined_successful_rows",
    "quarantined_unclosed_claims", "recovery_experiment_id", "root",
    "runtime_seed_wiring", "schema", "selected_rows", "source_coverage",
    "source_versions", "status", "unknown_unfilled_update_count",
))
DONOR_KEYS = frozenset((
    "adam_step_semantics", "bytes", "full_controller_adam_three_rng_restore",
    "identity_sha256", "observed_episode", "path",
    "payload_full_state_fingerprint", "payload_inventory_status",
    "payload_next_update", "python_numpy_torch_rng", "sha256",
    "stored_adam_parameter_step",
))
COVERAGE_KEYS = frozenset((
    "actual_relevant_claim_names", "claim_count", "coverage_basis", "plan_pin",
    "planned_train_replay_slots", "source_id",
))
ROLLOUT_KEYS = frozenset((
    "claim", "claim_pin", "field_origins", "outcome", "result", "result_pin",
    "row_pin", "runtime_key", "source_id", "source_root",
))
APPROVAL_KEYS = frozenset((
    "schema", "status", "candidate_manifest_sha256", "candidate_index_sha256", "protocol_sha256",
    "collector_code_sha256", "evidence_index_sha256", "root",
    "recovery_experiment_id", "namespace", "donor_set_sha256",
    "source_proof_set_sha256", "provenance",
))
APPROVAL_PROVENANCE = {
    "authority": "MAIN_MODEL_EXPLICIT_APPROVAL",
    "review": "RECOVERY_PREFIX_ACCEPTED_FOR_ZERO_WORLD_PREFLIGHT",
    "decision_ref": "COMM26_CONTROL_A_20261025_131328",
}

HANDOFF_RELATIVE = "execution/recovery_evidence_input/inventory02_handoff.json"
GRAND_OUTER_RELATIVE = (
    "execution/train_run_01/raw_receipts/CALL_183_TRAIN_POLL.json"
)
GRAND_TERMINAL_RELATIVE = (
    "execution/receipts/train_terminal_train-27328-1792979420977466300.json"
)
GRAND_SOURCE_CHECK_RELATIVE = (
    "execution/train_run_01/raw_receipts/CALL_185_TERMINAL_FACTS.json"
)
PARENT_OUTER_RELATIVE = (
    "execution/train_run_01/raw_receipts/CALL_293_TRAIN_POLL_60D.json"
)
PARENT_TERMINAL_RELATIVE = (
    "execution/receipts/train_terminal_train-40232-1793008010447671900.json"
)
PARENT_SOURCE_CHECK_RELATIVE = (
    "execution/train_run_01/raw_receipts/CALL_084_TRAIN_GROUP1_SOURCE_HASH_CHECK_V2.json"
)
R1_DIAGNOSTIC_RELATIVE = (
    "execution/train_run_01/failure_recoverability_diagnostic_20261027.json"
)
R1_MONITOR_RELATIVE = "execution/train_run_01/monitor_poll_returns_33.json"
R1_BEFORE_RELATIVE = "execution/prepare_run_01/source_hashes_before.json"
R1_AFTER_RELATIVE = "execution/prepare_run_01/source_hashes_after.json"


class RecoveryAdmissionError(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise RecoveryAdmissionError(message)


def exact_keys(value, expected, label):
    need(type(value) is dict, label + " must be a mapping")
    need(frozenset(value) == frozenset(expected), label + " has missing or unknown fields")
    return value


def strict_uint64(value, label):
    need(type(value) is int and 0 <= value <= UINT64_MAX,
         label + " must be an exact unsigned 64-bit integer")
    return value


def canonical_digest(value):
    E.finite_tree(value)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False,
                     sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def donor_set_digest(donors):
    need(type(donors) is list and len(donors) == 6, "exactly six donor pins are required")
    for item in donors:
        exact_keys(item, DONOR_KEYS, "donor")
    return canonical_digest(donors)


def source_proof_set_digest(source_versions):
    exact_keys(source_versions, SOURCE_ALIASES, "source proof set")
    return canonical_digest(source_versions)


def not_admitted(reason):
    need(type(reason) is str and reason, "NOT_ADMITTED requires a reason")
    return {"schema": RESULT_SCHEMA, "status": "NOT_ADMITTED",
            "admission": FINAL_ADMISSION, "reason": reason}


def _pin_parts(pin, label, allowed_root=None, expected_path=None):
    exact_keys(pin, {"path", "bytes", "sha256"}, label)
    path = Path(pin["path"])
    need(path.is_absolute(), label + " path must be absolute")
    path = path.resolve()
    need(type(pin["bytes"]) is int and pin["bytes"] >= 0,
         label + " bytes must be a nonnegative integer")
    need(type(pin["sha256"]) is str and HEX.fullmatch(pin["sha256"]),
         label + " SHA256 is malformed")
    if allowed_root is not None:
        E.inside(allowed_root, path)
    if expected_path is not None:
        need(path == Path(expected_path).resolve(), label + " path is not the fixed original")
    return path


def _json_lines(raw, label):
    need(type(raw) is bytes, label + " must be bytes")
    lines = raw.splitlines()
    need(lines and all(line.strip() for line in lines), label + " is empty or has blank lines")
    result = []
    for line in lines:
        item = E.strict_json(line)
        need(type(item) is dict, label + " line is not a JSON object")
        result.append(item)
    return result


def _same_pin(left, right):
    return (Path(left["path"]).resolve() == Path(right["path"]).resolve()
            and left["bytes"] == right["bytes"]
            and E.digest(left["sha256"]) == E.digest(right["sha256"]))


class AdmissionReads(E.Reads):
    """One immutable byte snapshot per pinned file, reused by all later checks."""

    def __init__(self):
        super().__init__()
        self._buffers = {}

    def raw(self, path, sha=None, size=None):
        path = Path(path).resolve()
        key = str(path)
        raw = self._buffers.get(key)
        if raw is None:
            need(path.is_file(), "missing evidence: " + key)
            raw = path.read_bytes()
            self._buffers[key] = raw
        pin = E.pin_for(path, raw)
        if sha is not None:
            need(pin["sha256"] == E.digest(sha), "file SHA mismatch: " + key)
        if size is not None:
            need(E.integer(size, "pin bytes") == len(raw), "file byte mismatch: " + key)
        old = self.pins.get(key)
        need(old is None or old == pin, "file changed during admission: " + key)
        self.pins[key] = pin
        return raw

TRAINING_SUCCESS_KEYS = frozenset((
    "status", "train_exit_basis", "outer_tool_pin", "terminal_receipt_pin",
    "source_check_pin",
))
R1_EVIDENCE_KEYS = frozenset((
    "status", "train_exit", "diagnostic_pin", "monitor_poll_pin",
    "source_hashes_before_pin", "source_hashes_after_pin",
))


def validate_approval(approval, *, candidate_sha256, candidate_index_sha256,
                      evidence_index_sha256, protocol_sha256, collector_code_sha256,
                      root, experiment_id, namespace, donor_digest, source_digest):
    """Validate a separately supplied approval; this function never creates one."""
    exact_keys(approval, APPROVAL_KEYS, "main-model approval")
    need(approval.get("schema") == APPROVAL_SCHEMA
         and approval.get("status") == "APPROVED",
         "explicit main-model approval has the wrong schema/status")
    expected = {
        "candidate_manifest_sha256": candidate_sha256,
        "candidate_index_sha256": candidate_index_sha256,
        "protocol_sha256": protocol_sha256,
        "collector_code_sha256": collector_code_sha256,
        "evidence_index_sha256": evidence_index_sha256,
        "donor_set_sha256": donor_digest,
        "source_proof_set_sha256": source_digest,
    }
    for key, value in expected.items():
        need(E.digest(approval.get(key)) == E.digest(value),
             "main-model approval binding mismatch: " + key)
    need(Path(str(approval.get("root"))).resolve() == Path(root).resolve(),
         "main-model approval root mismatch")
    need(approval.get("recovery_experiment_id") == experiment_id,
         "main-model approval experiment mismatch")
    need(approval.get("namespace") == namespace,
         "main-model approval seed namespace mismatch")
    exact_keys(approval.get("provenance"), APPROVAL_PROVENANCE,
               "main-model approval provenance")
    need(approval["provenance"] == APPROVAL_PROVENANCE,
         "main-model approval provenance values changed")
    return True


def _pinned_json(reader, root, pin, relative, label):
    path = _pin_parts(pin, label, root, Path(root) / relative)
    return E.strict_json(reader.pinned(pin, root))




def _validate_terminal_receipt(receipt, identity_sha, plan_sha, worlds, label):
    need(receipt.get("schema") == "stage5b-phase-receipt/1"
         and receipt.get("phase") == "train"
         and receipt.get("stage") == "terminal"
         and type(receipt.get("returncode")) is int
         and receipt["returncode"] == 0,
         label + " has no actual train child terminal returncode 0")
    need(receipt.get("identity_sha256") == identity_sha
         and receipt.get("slot_plan_sha256") == plan_sha,
         label + " identity/plan closure mismatch")
    need((receipt.get("detail") or {}).get("groups") == 6
         and (receipt.get("detail") or {}).get("worlds_completed") == worlds,
         label + " six-run/world completion mismatch")
    need(receipt.get("outer_exit_code") is None
         and receipt.get("outer_exit_status") == "UNKNOWN",
         label + " receipt incorrectly claims to contain the outer tool exit")
    need(type(receipt.get("receipt_id")) is str and receipt["receipt_id"],
         label + " receipt_id missing")
    return receipt["receipt_id"]


def _line_set(text):
    return {line.strip() for line in text.splitlines()}


def validate_training_evidence_index(index, root, source_roots, identity_shas, plan_shas,
                                     reader):
    """Validate the separately pinned train evidence; prep/preflight zero is not accepted."""
    exact_keys(index, {"schema", "status", "root", "sources"}, "training evidence index")
    need(index.get("schema") == EVIDENCE_INDEX_SCHEMA
         and index.get("status") == "COMPLETE_PINNED_TRAINING_EVIDENCE",
         "training evidence index schema/status mismatch")
    need(Path(str(index.get("root"))).resolve() == Path(root).resolve(),
         "training evidence index root mismatch")
    sources = index.get("sources")
    need(type(sources) is dict and frozenset(sources) == frozenset(SOURCE_ALIASES),
         "training evidence index must contain exactly grand/parent/R1")
    need(frozenset(source_roots) == frozenset(SOURCE_ALIASES)
         and frozenset(identity_shas) == frozenset(SOURCE_ALIASES)
         and frozenset(plan_shas) == frozenset(SOURCE_ALIASES),
         "internal source-evidence roster mismatch")

    grand_root = Path(source_roots["grand200"]).resolve()
    grand = exact_keys(sources["grand200"], TRAINING_SUCCESS_KEYS, "grand training evidence")
    need(grand.get("status") == "COMPLETE_TRAIN_SUCCESS"
         and grand.get("train_exit_basis") == "RAW_TOOL_ROOT",
         "grand source is not a fixed successful train source")
    grand_outer = _pinned_json(reader, grand_root, grand["outer_tool_pin"],
                               GRAND_OUTER_RELATIVE, "grand outer train receipt")
    need(type(grand_outer.get("exit_code")) is int and grand_outer["exit_code"] == 0,
         "grand outer train tool exit is not integer zero")
    grand_output = grand_outer.get("output")
    need(type(grand_output) is str
         and {"CHILD_EXIT_CODE=0", "WRAPPER_EXIT_CODE=0"} <= _line_set(grand_output),
         "grand raw train output does not contain child/wrapper exit zero")
    grand_receipt = _pinned_json(reader, grand_root, grand["terminal_receipt_pin"],
                                 GRAND_TERMINAL_RELATIVE, "grand terminal receipt")
    _validate_terminal_receipt(grand_receipt, identity_shas["grand200"],
                               plan_shas["grand200"], 1200, "grand terminal receipt")
    grand_check = _pinned_json(reader, grand_root, grand["source_check_pin"],
                               GRAND_SOURCE_CHECK_RELATIVE, "grand source check")
    need(type(grand_check.get("exit_code")) is int and grand_check["exit_code"] == 0,
         "grand train-time source check did not exit zero")
    grand_check_text = grand_check.get("output")
    need(type(grand_check_text) is str
         and re.search(r"SOURCE:\s*files=38\s+bytes=\d+\s+drift=0", grand_check_text)
         and re.search(r"STOP=False(?:\s|$)", grand_check_text),
         "grand successful-source closure/STOP evidence is incomplete")

    parent_root = Path(source_roots["parent600"]).resolve()
    parent = exact_keys(sources["parent600"], TRAINING_SUCCESS_KEYS,
                        "parent training evidence")
    need(parent.get("status") == "COMPLETE_TRAIN_SUCCESS"
         and parent.get("train_exit_basis") == "OPERATOR_RECORD_RAW_RESPONSE",
         "parent source is not a fixed successful train source")
    parent_wrapper = _pinned_json(reader, parent_root, parent["outer_tool_pin"],
                                  PARENT_OUTER_RELATIVE, "parent write_stdin train poll")
    exact_keys(parent_wrapper,
               {"call_index", "tool", "phase", "captured_at_utc", "raw_response"},
               "parent write_stdin outer receipt")
    need(parent_wrapper.get("call_index") == 293
         and parent_wrapper.get("tool") == "write_stdin"
         and parent_wrapper.get("phase") == "window-60 poll 4/4"
         and type(parent_wrapper.get("captured_at_utc")) is str
         and parent_wrapper["captured_at_utc"],
         "parent outer evidence must be the fixed write_stdin terminal poll shape")
    response = parent_wrapper.get("raw_response")
    exact_keys(response,
               {"chunk_id", "wall_time_seconds", "exit_code",
                "original_token_count", "output"},
               "parent write_stdin raw response")
    need(type(response.get("chunk_id")) is str and response["chunk_id"]
         and type(response.get("wall_time_seconds")) in (int, float)
         and type(response.get("exit_code")) is int and response["exit_code"] == 0
         and type(response.get("original_token_count")) is int
         and response["original_token_count"] >= 0,
         "parent operator-record raw_response train exit is not integer zero")
    parent_output = response.get("output")
    parent_lines = _line_set(parent_output) if type(parent_output) is str else set()
    need({"CHILD_EXIT_CODE=0", "WRAPPER_EXIT_CODE=0"} <= parent_lines
         and any(line.startswith("END_UTC=") and len(line) > len("END_UTC=")
                 for line in parent_lines),
         "parent write_stdin output lacks child/wrapper zero or END_UTC terminal markers")
    parent_receipt = _pinned_json(reader, parent_root, parent["terminal_receipt_pin"],
                                  PARENT_TERMINAL_RELATIVE, "parent terminal receipt")
    _validate_terminal_receipt(parent_receipt, identity_shas["parent600"],
                               plan_shas["parent600"], 2400, "parent terminal receipt")
    parent_check = _pinned_json(reader, parent_root, parent["source_check_pin"],
                                PARENT_SOURCE_CHECK_RELATIVE, "parent source check")
    check_response = parent_check.get("raw_response")
    need(type(check_response) is dict
         and type(check_response.get("exit_code")) is int
         and check_response["exit_code"] == 0,
         "parent train-time identity source check did not exit zero")
    check_text = check_response.get("output")
    need(type(check_text) is str
         and re.search(r"CHECKED=36\s+MISMATCHES=0\s+MISSING=0", check_text),
         "parent 36-pin identity source check is not closed at zero mismatches")

    r1_root = Path(source_roots["R1"]).resolve()
    r1 = exact_keys(sources["R1"], R1_EVIDENCE_KEYS, "R1 archive evidence")
    need(r1.get("status") == "INTERRUPTED_ARCHIVE_NOT_COMPLETE_PARENT"
         and r1.get("train_exit") == "UNKNOWN",
         "R1 must remain an interrupted archive with train exit UNKNOWN")
    diagnostic = _pinned_json(reader, r1_root, r1["diagnostic_pin"],
                              R1_DIAGNOSTIC_RELATIVE, "R1 interruption diagnostic")
    phases = diagnostic.get("phase_receipts") or {}
    source_freeze = diagnostic.get("source_freeze") or {}
    need("TRAIN_PHASE_TERMINAL_EXIT_UNKNOWN" in str(diagnostic.get("classification"))
         and phases.get("cli_or_outer_exit_code") == "UNKNOWN"
         and phases.get("cli_terminal_or_error_receipts") == []
         and phases.get("train_run_external_terminal_files") == []
         and source_freeze.get("files") == 33
         and source_freeze.get("differences") == 0,
         "R1 diagnostic does not prove the archived UNKNOWN train exit")
    monitor = _pinned_json(reader, r1_root, r1["monitor_poll_pin"],
                           R1_MONITOR_RELATIVE, "R1 monitor poll")
    observations = monitor.get("observations")
    need("exit_code" not in monitor and type(observations) is list and observations
         and all(type(item) is dict and item.get("terminal_exit_code") is None
                 for item in observations),
         "R1 monitor poll cannot substitute for a terminal train exit")
    before = _pinned_json(reader, r1_root, r1["source_hashes_before_pin"],
                          R1_BEFORE_RELATIVE, "R1 source hashes before")
    after = _pinned_json(reader, r1_root, r1["source_hashes_after_pin"],
                         R1_AFTER_RELATIVE, "R1 source hashes after")
    need(type(before) is list and len(before) == 33 and before == after,
         "R1 33-source before/after freeze evidence differs")
    for entry in after:
        exact_keys(entry, {"path", "sha256", "length"}, "R1 source hash entry")
        path = Path(entry["path"])
        E.inside(r1_root, path)
        need(type(entry["length"]) is int and entry["length"] >= 0,
             "R1 source length is malformed")
        reader.raw(path, entry["sha256"], entry["length"])
    return {
        "status": "TRAINING_EVIDENCE_COMPLETE_NOT_G3",
        "grand_terminal_receipt_id": grand_receipt["receipt_id"],
        "parent_terminal_receipt_id": parent_receipt["receipt_id"],
        "R1": "INTERRUPTED_ARCHIVE_NOT_COMPLETE_PARENT",
        "R1_train_exit": "UNKNOWN",
    }

def canonical_candidate_directory(root, candidate_directory):
    root = E.canonical_root(root)
    parent = (root / "execution/recovery_candidates").resolve()
    path = Path(candidate_directory).resolve(strict=False)
    need(path.parent == parent, "candidate directory is outside the owned candidate root")
    need(path.name.isascii() and CANDIDATE_NAME.fullmatch(path.name) is not None,
         "candidate directory name is not collector-owned")
    for ancestor in (root / "execution", parent, path):
        need(not ancestor.is_symlink(), "candidate path uses a symbolic link")
    return path


def validate_candidate_status(candidate, root):
    exact_keys(candidate, CANDIDATE_KEYS, "candidate manifest")
    need(candidate.get("schema") == CANDIDATE_SCHEMA
         and candidate.get("status") == CANDIDATE_STATUS,
         "candidate schema/status is not CANDIDATE_NOT_ADMITTED")
    need(candidate.get("main_model_approval") is None
         and candidate.get("admission") == CANDIDATE_ADMISSION,
         "candidate already claims approval/admission")
    need(candidate.get("G3") == "NOT_PERFORMED"
         and candidate.get("runtime_seed_wiring") == "NOT_IMPLEMENTED",
         "candidate G3/runtime wiring status changed")
    need(Path(str(candidate.get("root"))).resolve() == Path(root).resolve(),
         "candidate root mismatch")
    need(E.digest(candidate.get("draft_sha256")) == DRAFT_SHA256,
         "candidate DRAFT hash mismatch")
    need(candidate.get("recovery_experiment_id") == RECOVERY_EXPERIMENT_ID,
         "candidate recovery experiment identity mismatch")
    for key in ("selected_rows", "logical_updates", "actual_updates",
                "unknown_unfilled_update_count", "old_rollout_record_count",
                "old_unique_runtime_key_count", "quarantined_successful_rows",
                "quarantined_unclosed_claims"):
        need(type(candidate.get(key)) is int
             and candidate[key] == PRODUCTION_COUNTS[key],
             "candidate production count changed: " + key)
    need(candidate.get("controlled_replay_records_retained") is True,
         "candidate does not retain controlled replay records")
    need(candidate.get("known_gaps") == EXPECTED_KNOWN_GAPS,
         "candidate known gaps differ from the fixed three-run gap")
    donors = candidate.get("donors")
    need(type(donors) is list and len(donors) == 6,
         "candidate must pin exactly six donors")
    donor_set_digest(donors)
    coverage = candidate.get("source_coverage")
    need(type(coverage) is list and len(coverage) == 3,
         "candidate source coverage must have exactly three sources")
    for item in coverage:
        exact_keys(item, COVERAGE_KEYS, "source coverage")
    exact_keys(candidate.get("source_versions"), SOURCE_ALIASES, "source versions")
    for alias in SOURCE_ALIASES:
        exact_keys(candidate["source_versions"][alias],
                   {"action", "files", "scene", "source_identity"}, "source proof")
    exact_keys(candidate.get("inventory"),
               {"external_ingress_limit", "report_sha256", "status"},
               "candidate inventory summary")
    need(type(candidate.get("limits")) is list and len(candidate["limits"]) == 3,
         "candidate limitations must remain explicit")
    return True


def validate_candidate_outputs(candidate_directory, candidate, reader):
    """Read every owned output and verify candidate/index cross-pins."""
    directory = Path(candidate_directory).resolve()
    all_paths = tuple(directory.iterdir())
    need(all(path.is_file() and not path.is_symlink() for path in all_paths),
         "candidate directory must contain only regular files")
    listing = set(reader.listing(directory))
    pins = candidate.get("output_files")
    need(type(pins) is list and len(pins) == PRODUCTION_COUNTS["candidate_output_files"],
         "candidate.output_files must contain 139 pins")
    by_name = {}
    for pin in pins:
        path = _pin_parts(pin, "candidate output pin")
        need(path.parent == directory,
             "candidate output escapes its owned candidate directory")
        need(path.name != "candidate.json" and path.name not in by_name,
             "candidate output duplicate/self pin")
        by_name[path.name] = pin
        reader.pinned(pin, directory)
    need(listing == set(by_name) | {"candidate.json"},
         "candidate directory membership differs from its 139 output pins")
    need("index.json" in by_name, "candidate output pins do not include index.json")
    index_pin = by_name["index.json"]
    index = E.strict_json(reader.pinned(index_pin, directory))
    exact_keys(index, {"history", "outputs", "source_coverage"},
               "candidate index")
    index_pins = index.get("outputs")
    need(type(index_pins) is list and len(index_pins) == PRODUCTION_COUNTS["index_outputs"],
         "index.outputs must contain 138 pins")
    expected_names = set(by_name) - {"index.json"}
    actual_names = set()
    for pin in index_pins:
        path = _pin_parts(pin, "index output pin")
        need(path.parent == directory and path.name not in actual_names,
             "index output duplicate/escape")
        need(path.name != "candidate.json" and path.name != "index.json",
             "index cannot pin candidate.json or itself")
        need(path.name in by_name and _same_pin(pin, by_name[path.name]),
             "index output/ candidate output pin conflict")
        actual_names.add(path.name)
    need(actual_names == expected_names,
         "index outputs do not exactly cover the 138 data files")
    raws = {name: reader.pinned(pin, directory) for name, pin in by_name.items()}
    return index, raws, index_pin


def _chunks(count, per_chunk=120):
    return (count + per_chunk - 1) // per_chunk


def expected_chunk_names():
    names = set()
    for (arm, seed), (rows, updates, _missing) in EXPECTED_HISTORY.items():
        prefix = "history_%s_s%d" % (arm, seed)
        names.update("%s_%05d.jsonl" % (prefix, index)
                     for index in range(_chunks(rows)))
        prefix = "updates_%s_s%d" % (arm, seed)
        names.update("%s_%05d.jsonl" % (prefix, index)
                     for index in range(_chunks(updates)))
    names.update("old_rollouts_%05d.jsonl" % index
                 for index in range(PRODUCTION_COUNTS["old_rollout_chunks"]))
    names.update("source_pins_%05d.jsonl" % index
                 for index in range(PRODUCTION_COUNTS["source_pin_chunks"]))
    return names


def validate_index_summary(index):
    history = index.get("history")
    need(type(history) is list and len(history) == 6,
         "index history summary must contain six runs")
    seen = set()
    for item in history:
        exact_keys(item, {"arm", "training_seed", "rows", "updates",
                          "missing_update_indices"}, "index history")
        key = (item.get("arm"), item.get("training_seed"))
        need(key in EXPECTED_HISTORY and key not in seen,
             "index history has an unregistered/duplicate run")
        rows, updates, missing = EXPECTED_HISTORY[key]
        need(item.get("rows") == rows and item.get("updates") == updates
             and item.get("missing_update_indices") == list(missing),
             "index production history counts changed")
        seen.add(key)
    need(seen == set(EXPECTED_HISTORY), "index history is incomplete")
    return True


def validate_history_chunks(raws):
    rows_by_key = {}
    for (arm, seed), (row_count, _updates, _missing) in EXPECTED_HISTORY.items():
        prefix = "history_%s_s%d_" % (arm, seed)
        names = sorted(name for name in raws if name.startswith(prefix))
        need(len(names) == _chunks(row_count), "history chunk count mismatch: %s/%s" % (arm, seed))
        rows = []
        for name in names:
            rows.extend(_json_lines(raws[name], name))
        need(len(rows) == row_count, "history row count mismatch")
        for index, row in enumerate(rows):
            source = C.history_source_for_episode(index)
            E.validate_row(row, source, arm, seed, index)
            rows_by_key[(arm, seed, index)] = row
    need(len(rows_by_key) == PRODUCTION_COUNTS["selected_rows"],
         "history chunks do not close at 6120 rows")
    return rows_by_key


def validate_update_chunks(raws):
    actual = 0
    logical = 0
    missing_total = 0
    for (arm, seed), (_rows, update_count, missing) in EXPECTED_HISTORY.items():
        prefix = "updates_%s_s%d_" % (arm, seed)
        names = sorted(name for name in raws if name.startswith(prefix))
        need(len(names) == _chunks(update_count), "update chunk count mismatch")
        records = []
        for name in names:
            records.extend(_json_lines(raws[name], name))
        need(len(records) == update_count, "actual update record count mismatch")
        cursor = R.run_cursor(arm, seed)
        expected = [index for index in range(cursor // 4)
                    if index not in set(missing)]
        indices = [E.validate_update(item) for item in records]
        need(indices == expected, "update history is gapped, duplicated, reordered or unexpected")
        actual += len(records)
        logical += cursor // 4
        missing_total += len(missing)
    need(actual == PRODUCTION_COUNTS["actual_updates"]
         and logical == PRODUCTION_COUNTS["logical_updates"]
         and missing_total == PRODUCTION_COUNTS["unknown_unfilled_update_count"],
         "production update accounting changed")
    return actual, logical, missing_total

def _source_spec(alias):
    return dict(zip(SOURCE_ALIASES, C.HISTORY_SOURCES))[alias]


def _claim_filename(slot):
    need(type(slot) is str and slot.count(":") >= 3, "claim slot is malformed")
    return slot.replace(":", "_") + ".claim"


def _result_filename(slot):
    need(type(slot) is str and slot.count(":") >= 3, "result slot is malformed")
    return slot.replace(":", "_") + ".json"


def validate_old_rollout_chunks(raws, reader, rows_by_key, authorities, source_versions):
    """Recheck every retained claim/result/row and the selected prefix linkage."""
    names = sorted(name for name in raws if name.startswith("old_rollouts_"))
    need(len(names) == PRODUCTION_COUNTS["old_rollout_chunks"],
         "old-rollout chunk count changed")
    records = []
    for name in names:
        records.extend(_json_lines(raws[name], name))
    need(len(records) == PRODUCTION_COUNTS["old_rollout_record_count"],
         "old rollout records no longer total 6169")

    selected = {}
    seen_claims = set()
    for record in records:
        exact_keys(record, ROLLOUT_KEYS, "old rollout record")
        alias = record.get("source_id")
        need(alias in SOURCE_ALIASES, "old rollout uses an unexpected source")
        spec = _source_spec(alias)
        source_root = Path(str(record.get("source_root"))).resolve()
        need(source_root == Path(spec["root"]).resolve(),
             "old rollout source root mismatch")
        runtime = record.get("runtime_key")
        physical = E.runtime_physical_key(runtime)
        arm, seed, scene, action = physical
        episode = runtime.get("episode_index")
        need(type(episode) is int and episode >= 0
             and runtime.get("branch") == spec["source_id"],
             "old rollout episode/branch provenance mismatch")
        need(scene == authorities[alias].scene(seed, episode),
             "old rollout scene does not match the frozen source schedule")

        claim = record.get("claim")
        need(type(claim) is dict, "old rollout claim must be retained as a mapping")
        entry = claim.get("entry")
        need(type(entry) is dict, "old rollout claim entry missing")
        phase = entry.get("phase")
        need(phase in ("train", "replay")
             and entry.get("arm") == arm and entry.get("seed") == seed
             and entry.get("episode") == episode
             and entry.get("scene") == scene
             and type(entry.get("slot")) is str
             and claim.get("slot") == entry["slot"],
             "old rollout claim/plan key conflict")
        need(action == authorities[alias].action(seed, episode, phase),
             "old rollout action RNG key differs from its frozen source proof")
        need(claim.get("schema") == authorities[alias].literals["LEDGER_SCHEMA"],
             "old rollout claim schema mismatch")
        need(type(claim.get("worker_pid")) is int
             and type(claim.get("worker_ppid")) is int,
             "old rollout claim worker provenance malformed")
        claim_path = (source_root / "execution/ledger/claims" /
                      _claim_filename(entry["slot"]))
        claim_identity = (alias, claim_path.resolve())
        need(claim_identity not in seen_claims,
             "old snapshot repeats one source claim file")
        seen_claims.add(claim_identity)
        _pin_parts(record.get("claim_pin"), "claim pin", source_root, claim_path)
        claim_from_disk = E.strict_json(reader.pinned(record["claim_pin"], source_root))
        need(claim_from_disk == claim, "retained claim differs from its pinned original")

        result = record.get("result")
        result_pin = record.get("result_pin")
        outcome = record.get("outcome")
        result_path = (source_root / "execution/ledger/results" /
                       _result_filename(entry["slot"]))
        if result is None:
            need(result_pin is None and outcome == "UNKNOWN_UNCLOSED",
                 "unclosed claim acquired a result or a different outcome")
        else:
            need(type(result) is dict and result_pin is not None,
                 "closed claim result/pin missing")
            _pin_parts(result_pin, "result pin", source_root, result_path)
            result_from_disk = E.strict_json(reader.pinned(result_pin, source_root))
            need(result_from_disk == result, "retained result differs from its pinned original")
            need(result.get("schema") == authorities[alias].literals["LEDGER_SCHEMA"]
                 and result.get("slot") == entry["slot"]
                 and result.get("status") in ("ok", "failed", "terminated")
                 and result.get("worker_pid") == claim.get("worker_pid"),
                 "result/schema/claim worker conflict")
            need(outcome == result["status"], "retained outcome differs from result status")

        origins = record.get("field_origins")
        need(type(origins) is dict, "old rollout field origins missing")
        need(origins.get("source_functions") == {
                 "action": source_versions[alias]["action"],
                 "scene": source_versions[alias]["scene"],
                 "source_identity": source_versions[alias]["source_identity"]},
             "old rollout field origins do not match the source proof pins")
        need(origins.get("scene_seed") == "DIRECT_FROZEN_PLAN_AND_CLAIM"
             and origins.get("action_rng_seed") in ("DIRECT", "SOURCE_RECONSTRUCTION"),
             "old rollout runtime key origin is unsupported")

        row_pin = record.get("row_pin")
        row_object = None
        if row_pin is not None:
            need(phase == "train" and outcome == "ok",
                 "only successful train claims may carry a world row")
            row_path = (source_root / "execution/rows/train" /
                        ("train_%s_%d_%0*d.json" % (
                            arm, seed, spec["row_digits"], episode)))
            _pin_parts(row_pin, "row pin", source_root, row_path)
            row_object = E.strict_json(reader.pinned(row_pin, source_root))
            E.validate_row(row_object, spec, arm, seed, episode)
            payload = (result or {}).get("payload") or {}
            need(Path(str(payload.get("row", ""))).resolve() == row_path.resolve()
                 and payload.get("on_time") == row_object.get("on_time"),
                 "result/world-row linkage conflict")
        elif phase == "train" and outcome == "ok":
            raise RecoveryAdmissionError("successful train claim lacks its world-row pin")

        selected_key = (alias, arm, seed, episode)
        if phase == "train":
            need(selected_key not in selected,
                 "duplicate retained train claim for one source/run/episode")
            selected[selected_key] = (record, row_object)

    groups = {}
    for record in records:
        groups.setdefault(E.runtime_physical_key(record["runtime_key"]), []).append(record)
    replay_pairs = sum(len(group) == 2 for group in groups.values())
    need(len(groups) == PRODUCTION_COUNTS["old_unique_runtime_key_count"]
         and replay_pairs == PRODUCTION_COUNTS["controlled_replay_pairs"],
         "old rollout unique-key/replay-pair counts changed")
    E.validate_rollout_key_groups(records)

    expected_selected = 0
    for (arm, seed), (cursor, _updates, _missing) in EXPECTED_HISTORY.items():
        for episode in range(cursor):
            if episode < 200:
                alias = "grand200"
            elif episode < 600:
                alias = "parent600"
            else:
                alias = "R1"
            key = (alias, arm, seed, episode)
            need(key in selected, "selected history claim is absent from the old snapshot")
            record, row_object = selected[key]
            need(record["claim"]["entry"]["phase"] == "train"
                 and record["outcome"] == "ok"
                 and record["result_pin"] is not None
                 and record["row_pin"] is not None
                 and row_object == rows_by_key[(arm, seed, episode)],
                 "selected history differs from its pinned claim/result/row")
            expected_selected += 1
    need(expected_selected == PRODUCTION_COUNTS["selected_rows"],
         "selected history old-claim linkage does not close at 6120")
    return records


def validate_quarantine(raw_by_name, records, draft):
    raw = raw_by_name.get("quarantined_tail_and_unclosed_claims.json")
    need(raw is not None, "quarantine output missing")
    excluded = E.strict_json(raw)
    need(type(excluded) is list
         and len(excluded) == (PRODUCTION_COUNTS["quarantined_successful_rows"]
                               + PRODUCTION_COUNTS["quarantined_unclosed_claims"]),
         "quarantine must contain exactly 36 successful plus 9 unclosed records")
    canonical_snapshot = {canonical_digest(item) for item in records}
    canonical_excluded = [canonical_digest(item) for item in excluded]
    need(len(canonical_excluded) == len(set(canonical_excluded))
         and set(canonical_excluded) <= canonical_snapshot,
         "quarantine records are not exact retained old-claim originals")
    E.check_quarantine(excluded, draft)
    need(sum(item.get("outcome") == "ok" for item in excluded)
         == PRODUCTION_COUNTS["quarantined_successful_rows"]
         and sum(item.get("outcome") == "UNKNOWN_UNCLOSED" for item in excluded)
         == PRODUCTION_COUNTS["quarantined_unclosed_claims"],
         "quarantine closed/unclosed counts changed")
    for item in excluded:
        if item.get("outcome") == "UNKNOWN_UNCLOSED":
            need(item.get("result") is None and item.get("result_pin") is None
                 and item.get("row_pin") is None,
                 "unclosed claim was supplied with a result/row")
        else:
            need(item.get("result") is not None and item.get("result_pin") is not None
                 and item.get("row_pin") is not None,
                 "successful quarantined tail lacks result/row evidence")
    return excluded


def validate_proposed_seed_plan(raw_by_name, records):
    raw = raw_by_name.get("proposed_runtime_seed_keys.json")
    need(raw is not None, "proposed runtime seed plan missing")
    proposed = E.strict_json(raw)
    need(type(proposed) is list and len(proposed) == PRODUCTION_COUNTS["proposed_runtime_seed_keys"],
         "proposed recovery plan must contain exactly 720 keys")
    for item in proposed:
        exact_keys(item, {"arm", "training_seed", "episode_index",
                          "scene_seed", "action_rng_seed"}, "proposed runtime key")
        E.runtime_physical_key(item)
    runs = [{"arm": arm, "training_seed": seed, "start_episode": cursor,
             "end_episode_cap": cursor + R.EPISODES_PER_RUN_PER_ROOT}
            for arm, seed, cursor in R.RUN_CURSORS]
    expected = E.proposed_seed_plan(runs, records)
    need(proposed == expected,
         "proposed 720 seeds differ from the fixed namespace/exact old-key collision guard")
    old_keys = {E.runtime_physical_key(item["runtime_key"]) for item in records}
    proposed_keys = {E.runtime_physical_key(item) for item in proposed}
    need(len(proposed_keys) == len(proposed) and not (old_keys & proposed_keys),
         "new recovery seed collides with an old or another new runtime key")
    return proposed

def verify_source_pin(pin, allowed_roots, reader):
    path = _pin_parts(pin, "source pin")
    allowed_roots = [Path(item).resolve() for item in allowed_roots]
    need(any(path == parent or parent in path.parents for parent in allowed_roots),
         "source pin escapes every registered source/evidence root")
    old = reader.pins.get(str(path))
    if old is not None:
        need(old["bytes"] == pin["bytes"]
             and E.digest(old["sha256"]) == E.digest(pin["sha256"]),
             "source-pin table conflicts with a prior read")
    else:
        reader.raw(path, pin["sha256"], pin["bytes"])
    actual = reader.pins[str(path)]
    return path, (actual["bytes"], E.digest(actual["sha256"]))


def validate_source_pin_chunks(raws, reader):
    names = sorted(name for name in raws if name.startswith("source_pins_"))
    need(len(names) == PRODUCTION_COUNTS["source_pin_chunks"],
         "source-pin chunk count changed")
    pins = []
    for name in names:
        pins.extend(_json_lines(raws[name], name))
    need(len(pins) == PRODUCTION_COUNTS["source_pin_paths"],
         "source-pin table must contain 18538 unique paths")
    seen = {}
    allowed = [Path(root).resolve() for root in (
        CANONICAL_ROOT / "protocol", CANONICAL_ROOT / "stage5b_comm",
        CANONICAL_ROOT / "execution/checkpoint_inventory_02",
        CANONICAL_ROOT / "execution/recovery_evidence_input",
        C.BASE_ROOT, C.RUNS_ROOT / "comm_cost_v2_rerun_20260928",
        *(Path(spec["root"]) for spec in C.HISTORY_SOURCES))]
    for pin in pins:
        path, signature = verify_source_pin(pin, allowed, reader)
        need(path not in seen, "source-pin table contains a duplicate path")
        seen[path] = signature
    return seen


def validate_source_versions(candidate, reader):
    versions = candidate["source_versions"]
    authorities = {}
    plan_pins = {}
    plan_shas = {}
    identity_shas = {}
    for alias, spec in zip(SOURCE_ALIASES, C.HISTORY_SOURCES):
        identity_sha = E.R1_IDENTITY_SHA if alias == "R1" else spec["identity_sha256"]
        authority = E.FrozenSource(spec, identity_sha, reader)
        need(versions[alias] == authority.proofs,
             "candidate source proof differs from the frozen source identity/version")
        plan_path = (authority.root / "protocol/WORLD_SLOT_PLAN.json").resolve()
        plan_pin = reader.pins[str(plan_path)]
        plan_pins[alias] = plan_pin
        plan_shas[alias] = E.digest(plan_pin["sha256"])
        identity_shas[alias] = E.digest(authority.identity_pin["sha256"])
        authorities[alias] = authority
    return authorities, identity_shas, plan_pins, plan_shas


def validate_source_coverage(candidate, index, records, authorities, plan_pins):
    coverage = candidate.get("source_coverage")
    need(index.get("source_coverage") == coverage,
         "index/candidate source coverage differs")
    need([item.get("source_id") for item in coverage] == list(SOURCE_ALIASES),
         "source coverage order/roster mismatch")
    actual_names = {alias: set() for alias in SOURCE_ALIASES}
    for record in records:
        actual_names[record["source_id"]].add(_claim_filename(record["claim"]["slot"]))
    total = 0
    for item, alias in zip(coverage, SOURCE_ALIASES):
        names = item.get("actual_relevant_claim_names")
        need(type(names) is list and all(type(name) is str for name in names)
             and names == sorted(names) and len(names) == len(set(names)),
             "actual claim-name table is not a sorted unique string list")
        need(set(names) == actual_names[alias],
             "source coverage names differ from retained old claims")
        need(item.get("claim_count") == len(names),
             "source coverage claim count mismatch")
        authority = authorities[alias]
        slots = authority.plan.get("slots")
        need(type(slots) is list, "frozen source plan slots missing")
        relevant_slots = [slot for slot in slots
                          if slot.get("phase") in ("train", "replay")]
        plan_name_list = [_claim_filename(slot["slot"]) for slot in relevant_slots]
        plan_names = set(plan_name_list)
        need(len(plan_names) == len(plan_name_list),
             "frozen source plan contains duplicate train/replay slots")
        need(set(names) <= plan_names, "claimed name is absent from its frozen source plan")
        need(item.get("planned_train_replay_slots") == len(plan_names),
             "planned train/replay slot count mismatch")
        need(item.get("coverage_basis")
             == "FROZEN_PLAN_AND_STABLE_COMPLETE_LEDGER_DIRECTORY_MEMBERSHIP",
             "coverage basis changed")
        need(_same_pin(item.get("plan_pin"), plan_pins[alias]),
             "source coverage plan pin mismatch")
        total += len(names)
    need(total == PRODUCTION_COUNTS["old_rollout_record_count"],
         "source coverage does not total the 6169 retained claims")
    return coverage


def _check_separate_input(path, root, candidate_directory, label):
    resolved = Path(path).resolve(strict=False)
    E.inside(root, resolved)
    need(resolved.is_file(), "missing " + label + ": " + str(resolved))
    candidate_directory = Path(candidate_directory).resolve()
    need(resolved != candidate_directory and candidate_directory not in resolved.parents,
         label + " must be separate from the candidate bundle")
    return resolved


def validate_recovery_admission(root, candidate_directory, evidence_index_path,
                                approval_path):
    """Return an eligible validation object; never persist or widen admission."""
    root = E.canonical_root(root)
    candidate_directory = canonical_candidate_directory(root, candidate_directory)
    evidence_index_path = _check_separate_input(
        evidence_index_path, root, candidate_directory, "training evidence index")
    approval_path = _check_separate_input(
        approval_path, root, candidate_directory, "main-model approval")
    need(evidence_index_path != approval_path,
         "training evidence index and approval must be separate files")
    need(candidate_directory.is_dir(), "missing candidate directory")
    reader = AdmissionReads()

    # Missing candidate/evidence/approval fails before any admission object exists.
    candidate_path = candidate_directory / "candidate.json"
    candidate_raw = reader.raw(candidate_path)
    candidate = E.strict_json(candidate_raw)
    evidence_raw = reader.raw(evidence_index_path)
    evidence_index = E.strict_json(evidence_raw)
    approval = E.strict_json(reader.raw(approval_path))

    draft = reader.json(root / "protocol/RECOVERY_PROTOCOL.json", DRAFT_SHA256)
    runs = E.validate_draft(draft, root)
    validate_candidate_status(candidate, root)
    index, raws, index_pin = validate_candidate_outputs(
        candidate_directory, candidate, reader)
    validate_index_summary(index)

    expected = expected_chunk_names()
    actual_chunk_names = (set(raws) - {"index.json"}
                          - {"quarantined_tail_and_unclosed_claims.json",
                             "proposed_runtime_seed_keys.json"})
    need(actual_chunk_names == expected,
         "136 JSONL chunks do not have the fixed 51/14/52/19 shape")
    validate_source_pin_chunks(raws, reader)
    rows_by_key = validate_history_chunks(raws)
    validate_update_chunks(raws)

    inventory = E.inventory_evidence(root, draft, reader, HANDOFF_RELATIVE)
    need(candidate.get("inventory") == inventory,
         "candidate inventory summary differs from the pinned inventory evidence")
    expected_donors = [run["donor"] for run in runs]
    need(candidate.get("donors") == expected_donors,
         "candidate donors differ from the fixed DRAFT donor table")

    authorities, identity_shas, plan_pins, plan_shas = validate_source_versions(
        candidate, reader)
    records = validate_old_rollout_chunks(
        raws, reader, rows_by_key, authorities, candidate["source_versions"])
    validate_quarantine(raws, records, draft)
    validate_source_coverage(candidate, index, records, authorities, plan_pins)
    validate_proposed_seed_plan(raws, records)

    training = validate_training_evidence_index(
        evidence_index, root,
        {alias: spec["root"] for alias, spec in zip(SOURCE_ALIASES, C.HISTORY_SOURCES)},
        identity_shas, plan_shas, reader)

    donor_digest = donor_set_digest(candidate["donors"])
    source_digest = source_proof_set_digest(candidate["source_versions"])
    collector_raw = reader.raw(root / "stage5b_comm/recovery_evidence.py",
                               COLLECTOR_CODE_SHA256)
    candidate_sha = hashlib.sha256(candidate_raw).hexdigest()
    index_sha = E.digest(index_pin["sha256"])
    evidence_sha = hashlib.sha256(evidence_raw).hexdigest()
    validate_approval(
        approval,
        candidate_sha256=candidate_sha,
        candidate_index_sha256=index_sha,
        evidence_index_sha256=evidence_sha,
        protocol_sha256=DRAFT_SHA256,
        collector_code_sha256=hashlib.sha256(collector_raw).hexdigest(),
        root=root,
        experiment_id=RECOVERY_EXPERIMENT_ID,
        namespace=WORLD_SEED_NAMESPACE,
        donor_digest=donor_digest,
        source_digest=source_digest)

    return {
        "schema": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "admission": FINAL_ADMISSION,
        "G3": "NOT_PERFORMED",
        "runtime_seed_wiring": "NOT_IMPLEMENTED",
        "materialization": "NOT_PERFORMED",
        "root": str(root),
        "candidate_directory": str(candidate_directory),
        "candidate_manifest_sha256": candidate_sha,
        "candidate_index_sha256": index_sha,
        "evidence_index_sha256": evidence_sha,
        "protocol_sha256": DRAFT_SHA256,
        "collector_code_sha256": COLLECTOR_CODE_SHA256,
        "recovery_experiment_id": RECOVERY_EXPERIMENT_ID,
        "namespace": WORLD_SEED_NAMESPACE,
        "donor_set_sha256": donor_digest,
        "source_proof_set_sha256": source_digest,
        "selected_rows": PRODUCTION_COUNTS["selected_rows"],
        "logical_updates": PRODUCTION_COUNTS["logical_updates"],
        "actual_updates": PRODUCTION_COUNTS["actual_updates"],
        "unknown_unfilled_updates": PRODUCTION_COUNTS["unknown_unfilled_update_count"],
        "old_rollout_records": PRODUCTION_COUNTS["old_rollout_record_count"],
        "old_unique_runtime_keys": PRODUCTION_COUNTS["old_unique_runtime_key_count"],
        "proposed_runtime_keys": PRODUCTION_COUNTS["proposed_runtime_seed_keys"],
        "training_evidence": training,
        "remaining_requirements": [
            "separately authorized materialization of this pinned validation",
            "real six-controller/Adam/three-RNG G3 restore before any World",
            "guard/runner integration and a new independent world authorization",
        ],
    }
