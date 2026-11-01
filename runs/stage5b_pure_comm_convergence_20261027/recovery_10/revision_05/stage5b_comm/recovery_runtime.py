"""Explicit recovery runtime for finite, per-run checkpoint-prefix stages.

This module is deliberately separate from the normal complete-parent phase path. It consumes a
separately frozen execution context that pins a successful full recovery-admission validation,
the main-model approval, the candidate/index/evidence bundle, a source/native manifest, a frozen
recovery identity and slot plan, and the 720 proposed physical keys. It never creates approval,
ADMITTED, an identity, a protocol, a slot plan or a world authorization.

The only production entry points are g3(...) and run_train(...), called by the explicit
recovery-g3/recovery-train CLI phases. G3 loads each donor once and performs no forward pass;
training additionally requires a PASS G3 report and a separate main training-world authorization.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import random
from pathlib import Path
import time
import types

from . import compact, constants as C, convergence, episode, trainer
from . import recovery_admission as admission
from . import recovery_evidence as evidence
from . import recovery_protocol as R
from .ledger import CrossProcessLock

IDENTITY_RELATIVE = "protocol/RECOVERY_EXPERIMENT_IDENTITY.json"
SLOT_PLAN_RELATIVE = "protocol/RECOVERY_WORLD_SLOT_PLAN.json"
SOURCE_MANIFEST_RELATIVE = "protocol/RECOVERY_SOURCE_MANIFEST.json"
CONTEXT_RELATIVE = "protocol/RECOVERY_RUNTIME_CONTEXT.json"
G3_RELATIVE = "execution/recovery_runtime/G3_REPORT.json"
GENERATED_PROTOCOL_PATHS = frozenset((
    IDENTITY_RELATIVE, SLOT_PLAN_RELATIVE, SOURCE_MANIFEST_RELATIVE, CONTEXT_RELATIVE,
    "protocol/RECOVERY_DESCENDANT_STAGE.json",
    "protocol/RECOVERY_DESCENDANT_ADMISSION.json",
    "protocol/RECOVERY_NEXT_SEEDS.json",
))
CONTEXT_SCHEMA = "stage5b-recovery-runtime-context/1"
CONTEXT_STATUS = "FROZEN_RECOVERY_RUNTIME_CONTEXT"
SOURCE_MANIFEST_SCHEMA = "stage5b-recovery-source-manifest/1"
IDENTITY_SCHEMA = "stage5b-recovery-runtime-identity/1"
SLOT_PLAN_SCHEMA = "stage5b-recovery-world-slot-plan/1"
G3_SCHEMA = "stage5b-recovery-g3-report/1"
G3_STATUS = "PASS_RESTORED_OBSERVED"
WORLD_AUTH_SCHEMA = "stage5b-recovery-training-world-authorization/1"
WORLD_AUTH_STATUS = "APPROVED_FOR_RECOVERY_TRAINING_WORLDS"
CLAIM_SCHEMA = "stage5b-recovery-runtime-claim/1"
RESULT_SCHEMA = "stage5b-recovery-runtime-result/1"
JOURNAL_SCHEMA = "stage5b-recovery-update-journal/1"
CERTIFICATE_SCHEMA = "stage5b-recovery-run-certificate/1"
TELEMETRY_SCHEMA = "stage5b-recovery-run-telemetry/1"
GATE_SIGNAL = "GATE_FIRST_UPDATE.json"
STOP_SIGNAL = "GLOBAL_STOP.json"
HEX64 = frozenset("0123456789abcdefABCDEF")
UINT64_MAX = (1 << 64) - 1
SLOT_PREFIX = "recovery-train"
CONTEXT_KEYS = frozenset((
    "schema", "status", "root", "recovery_experiment_id", "namespace",
    "admission_report", "approval", "evidence_index", "candidate_manifest",
    "candidate_index", "protocol", "identity", "slot_plan", "seed_plan",
    "source_manifest", "certificate_sources", "g3_report", "world_authorization",
    "note",
))
PIN_KEYS = frozenset(("path", "bytes", "sha256"))
SOURCE_MANIFEST_KEYS = frozenset(("schema", "status", "root", "files", "note"))
SOURCE_FILE_KEYS = frozenset(("category", "role", "path", "bytes", "sha256"))
SLOT_PLAN_KEYS = frozenset((
    "schema", "status", "root", "recovery_experiment_id", "namespace", "world_cap",
    "slots", "note",
))
SLOT_KEYS = frozenset((
    "slot", "phase", "arm", "training_seed", "episode", "scene", "action_rng_seed",
))
G3_KEYS = frozenset((
    "schema", "status", "root", "recovery_experiment_id", "namespace",
    "context_base_sha256", "admission_report_sha256", "source_manifest_sha256",
    "donor_set_sha256", "runs", "recorded_utc",
    "claim_boundary",
    "note",
))
G3_RUN_KEYS = frozenset((
    "arm", "training_seed", "donor_path", "donor_bytes", "donor_sha256",
    "donor_identity_sha256", "episode", "payload_episode", "payload_seed",
    "payload_arm", "next_update", "next_update_basis",
    "stored_adam_parameter_step", "observed_adam_parameter_steps",
    "optimizer_parameter_state_count", "dual_critic_heads",
    "parameter_fingerprint", "actor_fingerprint", "full_state_fingerprint",
    "fingerprint_repeat", "rng_restore_observation",
    "restored_payload_comparison",
    "saved_fullstate_equality",
))
WORLD_AUTH_KEYS = frozenset((
    "schema", "status", "root", "recovery_experiment_id", "namespace",
    "authority", "decision_ref", "context_base_sha256", "admission_report_sha256",
    "g3_report_sha256", "candidate_manifest_sha256", "candidate_index_sha256",
    "donor_set_sha256", "max_worlds", "scope", "recorded_utc",
))
UPDATE_JOURNAL_KEYS = frozenset((
    "schema", "root", "recovery_experiment_id", "namespace", "arm", "training_seed",
    "update_index", "episode_indices", "full_state_before", "full_state_after",
    "episodes", "update", "recorded_utc",
    "claim_note",
))


class RecoveryRuntimeError(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise RecoveryRuntimeError(message)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def exact_keys(value, expected, label):
    need(type(value) is dict, label + " must be a mapping")
    need(frozenset(value) == frozenset(expected),
         label + " has missing or unknown fields: " + repr(sorted(set(value) ^ set(expected))))
    return value


def integer(value, label):
    need(type(value) is int, label + " must be an exact integer")
    return value


def uint64(value, label):
    need(type(value) is int and 0 <= value <= UINT64_MAX,
         label + " must be a non-bool uint64")
    return value


def digest(value, label="sha256"):
    need(type(value) is str and len(value) == 64 and all(ch in HEX64 for ch in value),
         label + " must be 64 ASCII hex characters")
    return value.lower()


def finite_tree(value):
    if type(value) is float:
        need(math.isfinite(value), "runtime data contains a nonfinite float")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def strict_json_bytes(raw, label):
    def pairs(items):
        body = {}
        for key, value in items:
            need(key not in body, label + " repeats JSON key " + key)
            body[key] = value
        return body
    try:
        value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               ValueError("invalid JSON constant " + value)))
    except (UnicodeError, ValueError) as error:
        raise RecoveryRuntimeError(label + " is not strict UTF-8 JSON: " + str(error)) from None
    finite_tree(value)
    return value


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def pin_file(path, root=None):
    path = Path(path).resolve()
    if root is not None:
        evidence.inside(root, path)
    need(path.is_file(), "pinned file is missing: " + str(path))
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": sha256_bytes(raw)}


def verify_pin(pin, label, root=None, expected_sha=None, expected_bytes=None):
    exact_keys(pin, PIN_KEYS, label + " pin")
    path = Path(str(pin["path"])).resolve(strict=False)
    if root is not None:
        evidence.inside(root, path)
    need(type(pin["bytes"]) is int and pin["bytes"] >= 0,
         label + " pin bytes must be an integer")
    expected = digest(pin["sha256"], label + " pin digest")
    if expected_sha is not None:
        need(expected == digest(expected_sha, label + " expected digest"),
             label + " pin digest differs from its binding")
    if expected_bytes is not None:
        need(pin["bytes"] == integer(expected_bytes, label + " expected bytes"),
             label + " pin byte count differs from its binding")
    need(path.is_file(), label + " pinned path is missing: " + str(path))
    actual_bytes = path.stat().st_size
    need(actual_bytes == pin["bytes"], label + " file bytes changed")
    actual_sha = sha256_bytes(path.read_bytes())
    need(actual_sha == expected, label + " file SHA-256 changed")
    return {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha}


def canonical_digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def runtime_key(record, label="runtime key"):
    required = frozenset(("arm", "training_seed", "scene_seed", "action_rng_seed"))
    allowed = required | frozenset(("episode_index", "branch"))
    need(type(record) is dict and required <= set(record) and set(record) <= allowed,
         label + " must contain exactly the physical key plus provenance only")
    arm = record["arm"]
    need(type(arm) is str and arm in C.ARMS_TRAIN, label + " arm is unregistered")
    if "episode_index" in record:
        integer(record["episode_index"], label + " episode_index")
    if "branch" in record:
        need(type(record["branch"]) is str and record["branch"],
             label + " branch provenance must be a non-empty string")
    return (arm, integer(record["training_seed"], label + " training_seed"),
            integer(record["scene_seed"], label + " scene_seed"),
            uint64(record["action_rng_seed"], label + " action_rng_seed"))


def assert_runtime_root(layout):
    """Reject an external same-shape copy before any receipt, claim or file write."""
    from .recovery_descendant import canonical
    return canonical(layout.root)


def normalized_run(arm, seed):
    need(type(arm) is str, "recovery arm must be an exact string")
    need(type(seed) is int, "recovery training seed must be a non-bool integer")
    key = (arm, seed)
    need(key in {(a, s) for a, s, _cursor in R.RUN_CURSORS},
         "unregistered recovery run %r/%r" % key)
    return key


@dataclass
class RecoveryContext:
    root: Path
    context_path: Path
    context_pin: dict
    payload: dict
    admission_report: dict
    approval: dict
    candidate: dict
    index: dict
    protocol: dict
    identity: dict
    slot_plan: dict
    source_manifest: dict
    seed_plan: list
    bundle_pins: dict
    runs: tuple
    old_runtime_keys: frozenset

    @property
    def experiment_id(self):
        return self.payload["recovery_experiment_id"]

    @property
    def namespace(self):
        return self.payload["namespace"]

    def base_sha256(self):
        """Actual SHA-256 of the immutable base-context file bytes."""
        exact_keys(self.context_pin, PIN_KEYS, "base recovery context pin")
        return digest(self.context_pin["sha256"])


ADMISSION_REPORT_KEYS = frozenset((
    "schema", "status", "admission", "G3", "runtime_seed_wiring",
    "materialization", "root", "candidate_directory", "candidate_manifest_sha256",
    "candidate_index_sha256", "evidence_index_sha256", "protocol_sha256",
    "collector_code_sha256", "recovery_experiment_id", "namespace",
    "donor_set_sha256", "source_proof_set_sha256", "selected_rows",
    "logical_updates", "actual_updates", "unknown_unfilled_updates",
    "old_rollout_records", "old_unique_runtime_keys", "proposed_runtime_keys",
    "training_evidence", "remaining_requirements",
))
IDENTITY_KEYS = frozenset((
    "schema", "status", "root", "recovery_experiment_id", "namespace",
    "protocol_sha256", "source_manifest_sha256", "admission_report_sha256",
    "approval_sha256", "candidate_manifest_sha256", "candidate_index_sha256",
    "evidence_index_sha256", "seed_plan_sha256", "slot_plan_sha256",
    "donor_set_sha256", "recorded_utc", "note",
))


def _read_json_pin(pin, label, root, full=True):
    checked = verify_pin(pin, label, root=root)
    raw = Path(checked["path"]).read_bytes()
    return checked, strict_json_bytes(raw, label)


def _validate_source_manifest(payload, root, namespace, *, layout=None):
    exact_keys(payload, SOURCE_MANIFEST_KEYS, "recovery source manifest")
    need(payload.get("schema") == SOURCE_MANIFEST_SCHEMA
         and payload.get("status") == "FROZEN",
         "recovery source manifest must be FROZEN")
    need(Path(payload.get("root", "")).resolve() == Path(root).resolve(),
         "recovery source manifest root mismatch")
    files = payload.get("files")
    need(type(files) is list and files, "recovery source manifest has no files")
    roles, categories = set(), set()
    for index, item in enumerate(files):
        exact_keys(item, SOURCE_FILE_KEYS, "source manifest file %d" % index)
        role = item.get("role")
        category = item.get("category")
        need(type(role) is str and role and role not in roles,
             "source manifest role is empty or duplicated")
        need(category in ("code", "native"), "source manifest category is unsupported")
        roles.add(role)
        categories.add(category)
        inside_root = category == "code"
        verify_pin({key: item[key] for key in PIN_KEYS}, "source " + role,
                   root=(root if inside_root else None))
        if inside_root:
            need(Path(item["path"]).resolve().is_relative_to(Path(root).resolve()),
                 "code pin escapes the recovery root")
    required_code = {"runtime", "runner", "guard", "cli", "episode", "trainer", "world",
                     "recovery_admission", "recovery_evidence"}
    need(required_code <= roles, "source manifest omits runtime/import-closure code roles")
    need("native" in categories, "source manifest omits the native source closure")
    need(payload.get("note") == "IMMUTABLE_PRE_RUN_SOURCE_AND_NATIVE_CLOSURE",
         "source manifest note/contract changed")
    if layout is not None:
        from . import guard
        expected_native = frozenset(
            str(Path(path).resolve()).casefold() for path in guard.allowlist_paths(layout))
        actual_native = frozenset(
            str(Path(item["path"]).resolve()).casefold()
            for item in files if item["category"] == "native")
        need(actual_native == expected_native,
             "source manifest native closure differs from the registered guard/base pins")
    return payload


def _validate_identity(identity, context, namespace):
    exact_keys(identity, IDENTITY_KEYS, "recovery runtime identity")
    need(identity.get("schema") == IDENTITY_SCHEMA
         and identity.get("status") == "FROZEN",
         "recovery runtime identity must be FROZEN")
    need(Path(identity.get("root", "")).resolve() == Path(context.root).resolve(),
         "recovery runtime identity root mismatch")
    need(identity.get("recovery_experiment_id") == context.payload["recovery_experiment_id"]
         and identity.get("namespace") == namespace,
         "recovery identity experiment/namespace mismatch")
    bindings = {
        "protocol_sha256": context.payload["protocol"]["sha256"],
        "source_manifest_sha256": context.payload["source_manifest"]["sha256"],
        "admission_report_sha256": context.payload["admission_report"]["sha256"],
        "approval_sha256": context.payload["approval"]["sha256"],
        "candidate_manifest_sha256": context.payload["candidate_manifest"]["sha256"],
        "candidate_index_sha256": context.payload["candidate_index"]["sha256"],
        "evidence_index_sha256": context.payload["evidence_index"]["sha256"],
        "seed_plan_sha256": context.payload["seed_plan"]["sha256"],
        "slot_plan_sha256": context.payload["slot_plan"]["sha256"],
        "donor_set_sha256": context.admission_report["donor_set_sha256"],
    }
    for key, expected in bindings.items():
        need(digest(identity.get(key), "identity " + key) == digest(expected),
             "recovery identity binding mismatch: " + key)
    return identity


def _load_seed_plan(path, root):
    raw = Path(path).read_bytes()
    value = strict_json_bytes(raw, "proposed runtime seed plan")
    need(type(value) is list and len(value) == 720,
         "proposed runtime seed plan must contain exactly 720 keys")
    by_run = {}
    physical = set()
    for index, item in enumerate(value):
        exact_keys(item, ("arm", "training_seed", "episode_index", "scene_seed",
                          "action_rng_seed"), "proposed runtime seed %d" % index)
        arm = item["arm"]
        seed = integer(item["training_seed"], "seed plan training_seed")
        episode = integer(item["episode_index"], "seed plan episode_index")
        scene = integer(item["scene_seed"], "seed plan scene_seed")
        action = uint64(item["action_rng_seed"], "seed plan action_rng_seed")
        run = normalized_run(arm, seed)
        start = R.run_cursor(*run)
        need(start <= episode < start + R.EPISODES_PER_RUN_PER_ROOT,
             "seed plan episode lies outside its finite run interval")
        need(scene == int(C.scene_for(seed, episode)),
             "seed plan scene differs from the frozen scene schedule")
        key = (arm, seed, scene, action)
        need(key not in physical, "seed plan repeats a physical runtime key")
        physical.add(key)
        by_run.setdefault(run, {})[episode] = dict(item)
    expected = {(arm, int(seed)) for arm, seed, _cursor in R.RUN_CURSORS}
    need(set(by_run) == expected, "seed plan does not cover exactly the six runs")
    for arm, seed in expected:
        start = R.run_cursor(arm, seed)
        expected_episodes = list(range(start, start + R.EPISODES_PER_RUN_PER_ROOT))
        need(sorted(by_run[(arm, seed)]) == expected_episodes,
             "seed plan is not a contiguous 120-episode interval for %s/%d" % (arm, seed))
    return value


def _validate_slot_plan(slot_plan, context):
    exact_keys(slot_plan, SLOT_PLAN_KEYS, "recovery slot plan")
    need(slot_plan.get("schema") == SLOT_PLAN_SCHEMA
         and slot_plan.get("status") == "FROZEN",
         "recovery slot plan must be FROZEN")
    need(Path(slot_plan.get("root", "")).resolve() == context.root
         and slot_plan.get("recovery_experiment_id") == context.experiment_id
         and slot_plan.get("namespace") == context.namespace,
         "recovery slot plan identity/root mismatch")
    need(integer(slot_plan.get("world_cap"), "slot plan world_cap") == 720,
         "recovery slot plan cap is not 720")
    slots = slot_plan.get("slots")
    need(type(slots) is list and len(slots) == 720,
         "recovery slot plan must contain exactly 720 slots")
    expected = {(item["arm"], int(item["training_seed"]), int(item["episode_index"])):
                item for item in context.seed_plan}
    seen_slots, seen_keys = set(), set()
    for index, item in enumerate(slots):
        exact_keys(item, SLOT_KEYS, "recovery slot %d" % index)
        arm = item["arm"]
        seed = integer(item["training_seed"], "slot training_seed")
        episode = integer(item["episode"], "slot episode")
        scene = integer(item["scene"], "slot scene")
        action = uint64(item["action_rng_seed"], "slot action_rng_seed")
        need(item["phase"] == "train", "recovery plan may hold only training slots")
        need(item["slot"] == "%s:%s:%d:%04d" % (SLOT_PREFIX, arm, seed, episode),
             "recovery slot id is not canonical")
        need(item["slot"] not in seen_slots, "recovery slot id is duplicated")
        seen_slots.add(item["slot"])
        planned = expected.get((arm, seed, episode))
        need(planned is not None and scene == planned["scene_seed"]
             and action == planned["action_rng_seed"],
             "recovery slot differs from the frozen proposed key plan")
        key = (arm, seed, scene, action)
        need(key not in seen_keys, "recovery slot plan repeats a physical key")
        seen_keys.add(key)
    need(len(seen_slots) == 720, "recovery slot set does not close")
    return slot_plan


def load_context(layout, context_path, *, full=True):
    """Load one explicitly pinned/FROZEN recovery context, fail closed."""
    root = assert_runtime_root(layout)
    path = Path(context_path).resolve(strict=False)
    evidence.inside(root, path)
    preview = strict_json_bytes(path.read_bytes(), "runtime context schema")
    if preview.get("schema") == "stage5b-recovery-descendant-context/1":
        from . import recovery_descendant
        result = recovery_descendant.load_context(layout, path)
        if full:
            return result
        return {"context_pin": result.context_pin}
    context_pin = pin_file(path, root=root)
    payload = strict_json_bytes(path.read_bytes(), "recovery runtime context")
    exact_keys(payload, CONTEXT_KEYS, "recovery runtime context")
    need(payload.get("schema") == CONTEXT_SCHEMA
         and payload.get("status") == CONTEXT_STATUS,
         "recovery runtime context is not FROZEN_RECOVERY_RUNTIME_CONTEXT")
    need(Path(payload.get("root", "")).resolve() == root,
         "recovery runtime context root mismatch")
    experiment_id = payload.get("recovery_experiment_id")
    namespace = payload.get("namespace")
    need(experiment_id == R.RECOVERY_STAGE_ID
         and namespace == R.WORLD_SEED_NAMESPACE,
         "recovery experiment identity or fixed namespace changed")

    protocol_pin, protocol = _read_json_pin(payload["protocol"], "recovery protocol",
                                            root, full)
    need(protocol_pin["sha256"] == evidence.DRAFT_SHA,
         "runtime accepts only the admitted recovery protocol bytes")
    admission_pin, report = _read_json_pin(
        payload["admission_report"], "recovery admission report", root, full)
    approval_pin, approval = _read_json_pin(payload["approval"], "main-model approval",
                                            root, full)
    evidence_pin, evidence_index = _read_json_pin(
        payload["evidence_index"], "training evidence index", root, full)
    candidate_pin, candidate = _read_json_pin(
        payload["candidate_manifest"], "candidate manifest", root, full)
    index_pin, index = _read_json_pin(payload["candidate_index"], "candidate index",
                                      root, full)
    source_pin, source_manifest = _read_json_pin(
        payload["source_manifest"], "recovery source manifest", root, full)
    identity_pin, identity = _read_json_pin(payload["identity"], "recovery identity",
                                            root, full)
    slot_pin, slot_plan = _read_json_pin(payload["slot_plan"], "recovery slot plan",
                                         root, full)
    seed_pin = verify_pin(payload["seed_plan"], "proposed seed plan", root=root)

    if full:
        exact_keys(report, ADMISSION_REPORT_KEYS, "recovery admission report")
        need(report.get("schema") == admission.RESULT_SCHEMA
             and report.get("status") == admission.RESULT_STATUS
             and report.get("admission") == "NOT_ADMITTED"
             and report.get("G3") == "NOT_PERFORMED"
             and report.get("materialization") == "NOT_PERFORMED",
             "no successful pre-G3 recovery admission report is pinned")
        need(Path(report.get("root", "")).resolve() == root,
             "admission report belongs to another root")
        need(report.get("recovery_experiment_id") == experiment_id
             and report.get("namespace") == namespace,
             "admission report identity/namespace mismatch")
        need(digest(report.get("candidate_manifest_sha256")) == candidate_pin["sha256"]
             and digest(report.get("candidate_index_sha256")) == index_pin["sha256"]
             and digest(report.get("evidence_index_sha256")) == evidence_pin["sha256"]
             and digest(report.get("protocol_sha256")) == protocol_pin["sha256"],
             "admission report does not bind the context artifacts")
        need(digest(report.get("collector_code_sha256"))
             == sha256_bytes((root / "stage5b_comm/recovery_evidence.py").read_bytes()),
             "admission report collector code changed")
        admission.validate_approval(
            approval,
            candidate_sha256=candidate_pin["sha256"],
            candidate_index_sha256=index_pin["sha256"],
            evidence_index_sha256=evidence_pin["sha256"],
            protocol_sha256=protocol_pin["sha256"],
            collector_code_sha256=report["collector_code_sha256"],
            root=root,
            experiment_id=experiment_id,
            namespace=namespace,
            donor_digest=report["donor_set_sha256"],
            source_digest=report["source_proof_set_sha256"])
        need(type(candidate) is dict and candidate.get("schema") == admission.CANDIDATE_SCHEMA
             and candidate.get("status") == admission.CANDIDATE_STATUS,
             "candidate status is not CANDIDATE_NOT_ADMITTED")
        need(type(index) is dict, "candidate index is not a mapping")
        admission.validate_index_summary(index)
        runs = evidence.validate_draft(protocol, root)
        _validate_source_manifest(source_manifest, root, namespace, layout=layout)
        seed_records = _load_seed_plan(seed_pin["path"], root)
    else:
        runs = None
        seed_records = None

    # The two JSON special files are part of the 139/138 artifact closure, while
    # expected_chunk_names intentionally describes only the 136 JSONL chunks.
    output_pins = candidate.get("output_files") if type(candidate) is dict else None
    index_outputs = index.get("outputs") if type(index) is dict else None
    need(type(output_pins) is list and len(output_pins) == 139,
         "candidate output pin list is not 139 files")
    need(type(index_outputs) is list and len(index_outputs) == 138,
         "candidate index output list is not 138 files")
    bundle_pins = {}
    candidate_names = set()
    for item in output_pins:
        checked = verify_pin(item, "candidate output", root=(root if full else None))
        name = Path(checked["path"]).name
        need(name not in candidate_names, "candidate output basename is duplicated")
        candidate_names.add(name)
        bundle_pins[name] = checked
    index_names = set()
    for item in index_outputs:
        checked = verify_pin(item, "index output", root=(root if full else None))
        name = Path(checked["path"]).name
        need(name not in index_names, "index output basename is duplicated")
        index_names.add(name)
        previous = bundle_pins.get(name)
        if previous is not None:
            need(previous == checked, "candidate/index pin disagree for " + name)
    need(len(index_names - candidate_names) == 0, "index output is absent from candidate pins")
    need({"quarantined_tail_and_unclosed_claims.json",
          "proposed_runtime_seed_keys.json"} <= candidate_names,
         "candidate omits the two required special JSON files")
    need(bundle_pins["index.json"] == index_pin, "index pin differs from candidate output")
    need(bundle_pins["proposed_runtime_seed_keys.json"] == seed_pin,
         "seed-plan pin differs from candidate output")
    need(seed_pin["sha256"] == (digest(payload["seed_plan"]["sha256"])),
         "seed-plan context pin differs from bundle pin")

    if full:
        context = RecoveryContext(
            root=root, context_path=path, context_pin=context_pin, payload=payload,
            admission_report=report, approval=approval, candidate=candidate,
            index=index, protocol=protocol, identity=identity, slot_plan=slot_plan,
            source_manifest=source_manifest, seed_plan=seed_records,
            bundle_pins=bundle_pins, runs=tuple(runs), old_runtime_keys=frozenset())
        _validate_identity(identity, context, namespace)
        _validate_slot_plan(slot_plan, context)
        context.old_runtime_keys = load_old_runtime_keys(context)
        return context

    return {
        "root": root, "context_path": path, "context_pin": context_pin,
        "payload": payload, "admission_report": report, "approval": approval,
        "candidate": candidate, "index": index, "protocol": protocol,
        "identity": identity, "slot_plan": slot_plan,
        "source_manifest": source_manifest, "seed_plan": seed_records,
        "bundle_pins": bundle_pins, "runs": tuple(runs or ()),
        "old_runtime_keys": frozenset(),
    }


def load_old_runtime_keys(context):
    keys = set()
    records = 0
    names = sorted(name for name in context.bundle_pins
                   if name.startswith("old_rollouts_") and name.endswith(".jsonl"))
    need(len(names) == admission.PRODUCTION_COUNTS["old_rollout_chunks"],
         "old rollout chunk count changed")
    for name in names:
        raw = Path(context.bundle_pins[name]["path"]).read_bytes()
        for line_number, line in enumerate(raw.splitlines(), 1):
            need(line.strip(), "blank line in old rollout chunk")
            record = strict_json_bytes(line, "old rollout %s:%d" % (name, line_number))
            need(type(record) is dict and type(record.get("runtime_key")) is dict,
                 "old rollout lacks a runtime_key")
            keys.add(runtime_key(record["runtime_key"], "old rollout runtime key"))
            records += 1
    need(records == admission.PRODUCTION_COUNTS["old_rollout_record_count"],
         "old rollout snapshot does not contain the fixed 6169 records")
    return frozenset(keys)


def selected_history(context, arm, seed):
    run = normalized_run(arm, seed)
    if getattr(context, "is_descendant", False):
        return dict(context.lineage_rows[run])
    start, logical_updates, missing_updates = admission.EXPECTED_HISTORY[run]
    rows = {}
    names = sorted(name for name in context.bundle_pins
                   if name.startswith("history_%s_s%d_" % run))
    expected_chunks = (start + 119) // 120
    need(len(names) == expected_chunks, "selected history chunk count changed")
    for name in names:
        raw = Path(context.bundle_pins[name]["path"]).read_bytes()
        for line_number, line in enumerate(raw.splitlines(), 1):
            need(line.strip(), "blank line in selected history")
            row = strict_json_bytes(line, "history %s:%d" % (name, line_number))
            need(type(row) is dict, "selected history row is not a mapping")
            index = integer(row.get("episode_index"), "history episode_index")
            need(row.get("arm") == run[0] and integer(row.get("seed"), "history seed") == run[1],
                 "selected history has the wrong arm/seed")
            need(index not in rows, "selected history repeats an episode")
            need(row.get("scene") == int(C.scene_for(run[1], index)),
                 "selected history scene differs from the frozen schedule")
            generated = integer(row.get("generated_goals"), "history generated_goals")
            steps = integer(row.get("steps"), "history steps")
            on_time = integer(row.get("on_time"), "history on_time")
            charged = integer(row.get("charged_bytes"), "history charged_bytes")
            need(generated == C.NUM_GOALS and steps == C.STEPS,
                 "selected history task count or 1000-step length changed")
            need(0 <= on_time <= generated and 0 <= charged <= C.BUDGET_BYTES,
                 "selected history success or charge is outside its domain")
            finite_tree(row)
            rows[index] = row
    need(sorted(rows) == list(range(start)),
         "selected history is gapped, reordered or conflicting")
    return rows


def selected_updates(context, arm, seed):
    run = normalized_run(arm, seed)
    if getattr(context, "is_descendant", False):
        return dict(context.lineage_updates[run])
    cursor, _logical, known_missing = admission.EXPECTED_HISTORY[run]
    names = sorted(name for name in context.bundle_pins
                   if name.startswith("updates_%s_s%d_" % run))
    expected_chunks = ((cursor // C.EPISODES_PER_UPDATE) + 119) // 120
    need(len(names) == expected_chunks, "selected update chunk count changed")
    records = {}
    for name in names:
        raw = Path(context.bundle_pins[name]["path"]).read_bytes()
        for line_number, line in enumerate(raw.splitlines(), 1):
            need(line.strip(), "blank line in selected update diagnostics")
            item = strict_json_bytes(line, "update %s:%d" % (name, line_number))
            index = evidence.validate_update(item)
            need(index not in records, "selected update diagnostics repeat/conflict")
            records[index] = item
    expected = set(range(cursor // C.EPISODES_PER_UPDATE))
    missing = sorted(expected - set(records))
    need(set(missing) == set(known_missing),
         "selected update diagnostics differ from the three registered gaps")
    need(set(records) == expected - set(missing),
         "selected update diagnostics contain an unexpected index")
    return records


def _required_check_indices(end_episode):
    windows = convergence.window_bounds(int(end_episode))
    rows = set()
    updates = set()
    for window in windows:
        rows.update(range(int(window["start"]), int(window["end"])))
        updates.update(range(int(window["start"]) // C.EPISODES_PER_UPDATE,
                             int(window["end"]) // C.EPISODES_PER_UPDATE))
    return windows, rows, updates


def validate_check_material(arm, seed, end_episode, rows, updates):
    """Prove all three windows exist or prove absence is exactly a registered update gap.

    Every non-gap duplicate/missing row or update, bad arm/seed/scene/task count, or nonfinite
    diagnostic is rejected before the qualification helper can label a check not evaluable.
    """
    run = normalized_run(arm, seed)
    windows, required_rows, required_updates = _required_check_indices(end_episode)
    gap = R.known_gap(*run)
    row_gap = R.known_row_gap(*run)
    allowed_updates, allowed_rows = set(), set()
    if gap is not None:
        allowed_updates = set(range(int(gap["updates"][0]), int(gap["updates"][1])))
    if row_gap is not None:
        allowed_rows = set(range(int(row_gap["episodes"][0]), int(row_gap["episodes"][1])))
    try:
        missing_rows = required_rows - set(rows)
        unexpected_rows = sorted(missing_rows - allowed_rows)
        need(not unexpected_rows,
             "unexpected missing history row at episode %d"
             % (unexpected_rows[0] if unexpected_rows else -1))
        for index in sorted(required_rows - missing_rows):
            row = rows[index]
            need(row.get("episode_index") == index and row.get("arm") == run[0]
                 and type(row.get("seed")) is int and row.get("seed") == run[1],
                 "history key/arm/seed mismatch at episode %d" % index)
            need(row.get("scene") == int(C.scene_for(run[1], index)),
                 "history scene mismatch at episode %d" % index)
            need(row.get("generated_goals") == C.NUM_GOALS
                 and row.get("steps") == C.STEPS,
                 "history task count or 1000-step length mismatch at episode %d" % index)
            finite_tree(row)
        missing_updates = required_updates - set(updates)
        unexpected = sorted(missing_updates - allowed_updates)
        need(not unexpected,
             "unexpected missing update diagnostic outside the registered gap: %r" % unexpected)
        for index in sorted(required_updates - missing_updates):
            evidence.validate_update(updates[index])
    except evidence.RecoveryEvidenceError as error:
        raise RecoveryRuntimeError(str(error)) from error
    return {
        "windows": windows,
        "required_rows": len(required_rows),
        "required_updates": len(required_updates),
        "missing_rows": sorted(missing_rows),
        "missing_updates": sorted(missing_updates),
        "allowed_known_gap_rows": sorted(missing_rows & allowed_rows),
        "allowed_known_gap_updates": sorted(missing_updates & allowed_updates),
        "unexpected_missing_rows": [],
        "unexpected_missing_updates": [],
    }


def certificate_path(layout, arm, seed, end_episode):
    return (Path(layout.root) / "execution" / "recovery" / "convergence"
            / ("CHECK_%s_s%d_e%04d.json" % (str(arm), int(seed), int(end_episode))))


def not_evaluable_path(layout, arm, seed, end_episode):
    return (Path(layout.root) / "execution" / "recovery" / "convergence"
            / ("NOT_EVALUABLE_%s_s%d_e%04d.json" % (str(arm), int(seed), int(end_episode))))


def read_check_certificates(context, layout, arm, seed):
    """Load current plus explicitly pinned ancestor certificates and deduplicate exactly."""
    run = normalized_run(arm, seed)
    candidates = []
    current = Path(layout.root) / "execution" / "recovery" / "convergence"
    if current.is_dir():
        for path in sorted(current.glob("CHECK_%s_s%d_e*.json" % run)):
            candidates.append(("current", path, context.experiment_id, None))
    ancestor_sources = context.payload.get("certificate_sources")
    need(type(ancestor_sources) is list, "certificate_sources must be a list")
    for index, source in enumerate(ancestor_sources):
        expected_keys = {"arm", "training_seed", "experiment_id", "path", "bytes", "sha256"}
        exact_keys(source, expected_keys, "certificate source %d" % index)
        source_run = normalized_run(source["arm"], source["training_seed"])
        if source_run != run:
            continue
        pin = verify_pin({key: source[key] for key in PIN_KEYS},
                         "certificate source %d" % index, root=None)
        candidates.append(("pinned ancestor", Path(pin["path"]),
                           str(source["experiment_id"]), pin))
    by_key = {}
    for source_name, path, expected_experiment, source_pin in candidates:
        raw = Path(path).read_bytes()
        if source_pin is not None:
            need(len(raw) == source_pin["bytes"]
                 and sha256_bytes(raw) == source_pin["sha256"],
                 "pinned certificate changed during its read")
        certificate = strict_json_bytes(raw, "check certificate " + str(path))
        need(certificate.get("schema") == C.CONVERGENCE_SCHEMA,
             "check certificate schema changed")
        need(certificate.get("experiment_id") == expected_experiment,
             "certificate experiment does not match its pinned source")
        need(certificate.get("arm") == run[0]
             and integer(certificate.get("seed"), "certificate seed") == run[1],
             "certificate belongs to another run")
        end_episode = integer(certificate.get("end_episode"), "certificate end_episode")
        R.absolute_check(end_episode)
        if source_name == "current":
            need(Path(path).name == "CHECK_%s_s%d_e%04d.json"
                 % (run[0], run[1], end_episode),
                 "current certificate filename/content episode mismatch")
        need(type(certificate.get("passed")) is bool,
             "certificate passed must be boolean")
        expected_status = (convergence.STATUS_PASS if certificate["passed"]
                           else convergence.STATUS_FAIL)
        need(certificate.get("status") == expected_status,
             "certificate status/passed fields disagree")
        key = (run[0], run[1], end_episode)
        normalized = canonical_digest(certificate)
        item = {"end_episode": end_episode, "passed": certificate["passed"],
                "status": certificate["status"], "path": str(path),
                "source": source_name, "sha256": sha256_bytes(raw),
                "_normalized": normalized}
        previous = by_key.get(key)
        if previous is not None:
            need(previous["_normalized"] == normalized,
                 "conflicting duplicate check certificate at episode %d" % end_episode)
            continue
        by_key[key] = item
    result = []
    for key in sorted(by_key, key=lambda value: (value[2], value[0], value[1])):
        item = dict(by_key[key])
        item.pop("_normalized")
        result.append(item)
    return result


def evaluate_check(context, layout, arm, seed, end_episode, rows, updates):
    run = normalized_run(arm, seed)
    material = validate_check_material(run[0], run[1], end_episode, rows, updates)
    eligibility = R.check_eligibility(run[0], run[1], end_episode)
    if eligibility["status"] == R.CHECK_NOT_EVALUABLE:
        payload = {
            "schema": "stage5b-recovery-not-evaluable/1",
            "experiment_id": context.experiment_id,
            "namespace": context.namespace,
            "arm": run[0], "training_seed": run[1],
            "end_episode": int(end_episode), "status": R.CHECK_NOT_EVALUABLE,
            "qualification": eligibility, "material": material,
            "recorded_utc": utc(),
            "note": ("qualification event only: no strict evaluator call, PASS, certificate "
                     "or consecutive-pass increment"),
        }
        path = not_evaluable_path(layout, run[0], run[1], end_episode)
        path.parent.mkdir(parents=True, exist_ok=True)
        compact.write_json_exclusive(path, payload)
        return {"evaluated": False, "status": R.CHECK_NOT_EVALUABLE,
                "path": str(path), "streak": convergence.consecutive_passes(
                    read_check_certificates(context, layout, run[0], run[1]))}
    need(eligibility["status"] == R.CHECK_EVALUABLE,
         "unknown check qualification disposition")
    certificate = convergence.evaluate({"rows": rows, "updates": updates},
                                       arm=run[0], seed=run[1],
                                       end_episode=int(end_episode))
    # The frozen evaluator computes the criterion; only the owning root identity changes here.
    certificate["source_evaluator_experiment_id"] = certificate.get("experiment_id")
    certificate["experiment_id"] = context.experiment_id
    certificate["recovery_namespace"] = context.namespace
    certificate["qualification"] = eligibility
    if getattr(context, "is_descendant", False):
        certificate["lineage"] = {
            "base_context_sha256": context.base_sha256(),
            "parent_context": context.descendant["parent_context"],
            "science_profile_sha256": context.descendant["science_profile_sha256"],
        }
    path = certificate_path(layout, run[0], run[1], end_episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    compact.write_json_exclusive(path, certificate)
    checks = read_check_certificates(context, layout, run[0], run[1])
    return {"evaluated": True, "status": certificate.get("status"),
            "passed": bool(certificate.get("passed")), "path": str(path),
            "streak": convergence.consecutive_passes(checks)}


class RecoveryLedger:
    """Recovery-plan claims over the same root ledger directories and OS lock.

    SlotLedger intentionally recognizes only the normal constants.slot_plan(); reusing it here
    would silently substitute the normal stage plan. This class retains exclusive-create and
    non-retry semantics while validating the independent 720-slot recovery plan.
    """

    def __init__(self, context, layout):
        self.context = context
        self.layout = layout
        self.plan = context.slot_plan
        self.entries = {item["slot"]: item for item in self.plan["slots"]}
        self.claims = Path(layout.ledger) / "recovery_claims"
        self.results = Path(layout.ledger) / "recovery_results"
        self.lock = CrossProcessLock(Path(layout.ledger) / "RECOVERY_LEDGER.lock")

    @staticmethod
    def _flat(slot):
        return str(slot).replace(":", "_")

    def ensure(self):
        self.claims.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)

    def claimed(self, slot):
        return (self.claims / (self._flat(slot) + ".claim")).exists()

    def finished(self, slot):
        return (self.results / (self._flat(slot) + ".json")).exists()

    def _claim_path(self, slot):
        return self.claims / (self._flat(slot) + ".claim")

    def _result_path(self, slot):
        return self.results / (self._flat(slot) + ".json")

    def existing_runtime_keys(self, arm, seed):
        keys = set()
        if not self.claims.is_dir():
            return keys
        for path in sorted(self.claims.glob("*.claim")):
            body = strict_json_bytes(path.read_bytes(), "existing recovery claim")
            key = body.get("runtime_key")
            if type(key) is dict and key.get("arm") == arm \
                    and key.get("training_seed") == int(seed):
                keys.add(runtime_key(key, "existing recovery claim key"))
        return keys

    def claim(self, slot, *, runtime_key_value):
        need(slot in self.entries, "slot is not in the frozen recovery plan")
        self.ensure()
        payload = {
            "schema": CLAIM_SCHEMA, "root": str(self.layout.root),
            "recovery_experiment_id": self.context.experiment_id,
            "namespace": self.context.namespace, "slot": slot,
            "entry": dict(self.entries[slot]), "runtime_key": dict(runtime_key_value),
            "claimed_utc": utc(), "worker_pid": os.getpid(),
            "worker_ppid": os.getppid(),
            "status_note": ("irreversible in-process claim; not an OS exit code and never "
                            "retried or substituted"),
        }
        with self.lock:
            if (Path(self.layout.ledger) / STOP_SIGNAL).exists():
                return "stopped"
            try:
                with self._claim_path(slot).open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False,
                                            sort_keys=True) + "\n")
            except FileExistsError:
                return "consumed"
        return "claimed"

    def finish(self, slot, row_path):
        need(self.claimed(slot), "cannot finish an unclaimed recovery slot")
        body = {
            "schema": RESULT_SCHEMA, "root": str(self.layout.root),
            "recovery_experiment_id": self.context.experiment_id,
            "namespace": self.context.namespace, "slot": slot, "status": "ok",
            "row": str(row_path), "ended_utc": utc(), "worker_pid": os.getpid(),
            "status_note": "in-process result; phase-level OS status is in the external receipt",
        }
        with self.lock:
            path = self._result_path(slot)
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(body, ensure_ascii=False,
                                        sort_keys=True) + "\n")
        return path

    def counts(self):
        self.ensure()
        claimed = len(list(self.claims.glob("*.claim")))
        results = len(list(self.results.glob("*.json")))
        ok = 0
        for path in self.results.glob("*.json"):
            if strict_json_bytes(path.read_bytes(), "recovery result").get("status") == "ok":
                ok += 1
        cap = self.context.slot_plan.get("world_cap", 720)
        return {"planned": cap, "claimed": claimed, "results": results, "ok": ok,
                "remaining": cap - results}


def recovery_ledger_accounting(context, layout):
    """Use the recovery slot plan in error receipts; never substitute normal WORLD_SLOT_PLAN."""
    if isinstance(context, dict):
        context = types.SimpleNamespace(
            slot_plan=context["slot_plan"],
            experiment_id=context["payload"]["recovery_experiment_id"],
            namespace=context["payload"]["namespace"])
    result = RecoveryLedger(context, layout).counts()
    result["basis"] = "RECOVERY_CONTEXT_SLOT_PLAN"
    return result


def request_stop(layout, reason):
    path = Path(layout.ledger) / STOP_SIGNAL
    path.parent.mkdir(parents=True, exist_ok=True)
    with CrossProcessLock(Path(layout.ledger) / "RECOVERY_STOP.lock"):
        if path.exists():
            return strict_json_bytes(path.read_bytes(), "existing recovery stop")
        payload = {
            "schema": "stage5b-recovery-global-stop/1", "status": "TERMINAL_UNRECOVERABLE",
            "reason": str(reason), "recorded_utc": utc(), "pid": os.getpid(),
            "note": ("no new recovery slot may be claimed; already-claimed slots finish; "
                     "there is no retry, replacement seed or resume of this phase"),
        }
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False,
                                    sort_keys=True) + "\n")
        return payload


def _optimizer_steps(optimizer):
    state = optimizer.state_dict().get("state", {})
    values = []
    for parameter_id in sorted(state, key=lambda value: int(value)):
        item = state[parameter_id]
        value = item.get("step")
        need(value is not None, "optimizer parameter state has no Adam step")
        need(type(value) is not bool, "Adam step cannot be boolean")
        try:
            step = int(value)
        except (TypeError, ValueError, OverflowError):
            raise RecoveryRuntimeError("Adam step is not an exact integer scalar") from None
        need(step >= 0, "Adam step is negative")
        values.append(step)
    return values


def _donor_identity_text(value, label):
    need(type(value) is str and len(value) == 64 and all(ch in HEX64 for ch in value),
         label + " must be 64 ASCII hex characters")
    # SHA-256 hex text is case-insensitive. Other identity fields and path/schema
    # comparisons remain exact and case-sensitive.
    return value.lower()


def _donor_pairs(context):
    stage = context.protocol.get("stage") or {}
    runs = stage.get("runs")
    need(type(runs) is list and len(runs) == 6,
         "recovery protocol does not hold six donors")
    result = []
    for index, run in enumerate(runs):
        arm, seed, initial_cursor = R.RUN_CURSORS[index]
        start = (run.get("start_episode") if getattr(context, "is_descendant", False)
                 else initial_cursor)
        donor = run.get("donor")
        need(type(donor) is dict, "recovery run has no donor pin")
        cursor = (donor.get("observed_episode", donor.get("cursor_from_closed_journal"))
                  if getattr(context, "is_descendant", False) else initial_cursor)
        need(type(cursor) is int and cursor >= initial_cursor and cursor % 4 == 0
             and type(start) is int and start >= cursor and start % 4 == 0
             and run.get("arm") == arm and run.get("training_seed") == seed
             and run.get("donor_episode", cursor) == cursor
             and run.get("donor_next_update", cursor // 4) == cursor // 4,
             "recovery donor/start branch differs for %r/%r start=%r cursor=%r donor_ep=%r donor_next=%r"
             % (run.get("arm"), run.get("training_seed"), start, cursor,
                run.get("donor_episode"), run.get("donor_next_update")))
        result.append(((arm, seed, cursor), donor))
    return result


def validate_g3(context, report, pin):
    exact_keys(report, G3_KEYS, "recovery G3 report")
    need(report.get("schema") == G3_SCHEMA and report.get("status") == G3_STATUS,
         "recovery training requires a PASS_RESTORED_OBSERVED G3 report")
    need(Path(report.get("root", "")).resolve() == context.root
         and report.get("recovery_experiment_id") == context.experiment_id
         and report.get("namespace") == context.namespace,
         "G3 report root/identity/namespace mismatch")
    need(digest(report.get("context_base_sha256")) == context.base_sha256(),
         "G3 report does not bind this recovery context")
    need(digest(report.get("admission_report_sha256"))
         == digest(context.payload["admission_report"]["sha256"]),
         "G3 report does not bind the admission report")
    need(digest(report.get("source_manifest_sha256"))
         == digest(context.payload["source_manifest"]["sha256"]),
         "G3 report does not bind the source/native manifest")
    need(digest(report.get("donor_set_sha256"))
         == digest(context.admission_report["donor_set_sha256"]),
         "G3 report does not bind the six donors")
    entries = report.get("runs")
    need(type(entries) is list and len(entries) == 6,
         "G3 report must hold six restored donors")
    expected = _donor_pairs(context)
    for entry, ((arm, seed, episode), donor) in zip(entries, expected):
        exact_keys(entry, G3_RUN_KEYS, "G3 run")
        need(entry.get("arm") == arm and entry.get("training_seed") == seed
             and entry.get("episode") == episode,
             "G3 report run order/cursor mismatch")
        need(Path(entry.get("donor_path", "")).resolve()
             == Path(str(donor.get("path", ""))).resolve()
             and entry.get("donor_bytes") == donor.get("bytes")
             and digest(entry.get("donor_sha256")) == digest(donor.get("sha256")),
             "G3 report donor pin mismatch")
        need(entry.get("payload_episode") == episode
             and entry.get("payload_seed") == seed
             and entry.get("payload_arm") == arm,
             "G3 report payload cursor mismatch")
        descendant = getattr(context, "is_descendant", False)
        basis = ("PAYLOAD_AND_CLOSED_LAST_JOURNAL" if descendant
                 else "FROZEN_STRIDE4_SOURCE_RECONSTRUCTION")
        need(entry.get("next_update") == episode // C.EPISODES_PER_UPDATE
             and entry.get("next_update_basis") == basis,
             "G3 report next_update does not bind its actual donor/journal")
        observed = entry.get("observed_adam_parameter_steps")
        need(type(observed) is list and len(observed) > 0
             and all(type(step) is int and step >= 0 for step in observed),
             "G3 observed Adam steps are missing/invalid")
        if descendant:
            need(entry.get("full_state_fingerprint") == donor["expected_full_state_sha256"]
                 and entry.get("actor_fingerprint") == donor["expected_actor_sha256"],
                 "G3 state differs from the completed parent's last real update")
        else:
            need(all(step == donor.get("stored_adam_parameter_step") for step in observed),
                 "G3 report Adam step does not equal the pinned stored step")
        need(entry.get("fingerprint_repeat") == "PASSIVE_IDENTICAL",
             "G3 fingerprint repeat check did not pass")
        comparison = entry.get("restored_payload_comparison")
        need(type(comparison) is dict and comparison.get("status") == "MATCH",
             "G3 raw-to-restored component comparison did not pass")
        need((comparison.get("controller") or {}).get("status") == "MATCH"
             and (comparison.get("optimizer") or {}).get("status") == "MATCH"
             and (comparison.get("rng") or {}).get("status") == "MATCH",
             "G3 controller/optimizer/RNG raw component mismatch")
        scope = comparison.get("scope") or {}
        need(scope.get("checkpoint_loads") == 1
             and scope.get("forward_passes") == 0
             and scope.get("comparison_rng_draws") == 0
             and scope.get("comparison_optimizer_steps") == 0
             and scope.get("comparison_mode_changes") == 0
             and comparison.get("comparison_preserved_rng") is True
             and comparison.get("comparison_preserved_model_mode") is True,
             "G3 comparison scope advanced RNG/model/optimizer or loaded twice")
        need(entry.get("saved_fullstate_equality") == "UNKNOWN_NOT_STORED",
             "G3 must not claim an unrecorded saved full-state fingerprint")
    return {"sha256": pin["sha256"], "status": report["status"], "runs": 6}


def completed_episode_target(context):
    """Return the frozen actual-completed target; ordinary stages return None."""
    from . import long_stage
    return long_stage.completed_target_from_payload(getattr(context, "payload", None))


def expected_world_authorization_scope(context):
    if completed_episode_target(context) is None:
        return "FINITE_RECOVERY_STAGE_MAX_720_WORLDS_ONE_ROOT_ONLY"
    return "LONG_RECOVERY_STAGE_TO_COMPLETED_EPISODE_TARGET_ONE_ROOT_ONLY"


def validate_world_authorization(context, authorization, pin, g3_pin):
    descendant = getattr(context, "is_descendant", False)
    fields = WORLD_AUTH_KEYS | ({"parent_context_sha256"} if descendant else set())
    exact_keys(authorization, fields, "recovery training-world authorization")
    if descendant:
        need(digest(authorization["parent_context_sha256"])
             == context.descendant["parent_context"]["sha256"],
             "world authorization does not bind the completed parent")
    need(authorization.get("schema") == WORLD_AUTH_SCHEMA
         and authorization.get("status") == WORLD_AUTH_STATUS,
         "separate main training-world authorization is missing")
    need(authorization.get("authority") == "MAIN_MODEL_EXPLICIT_APPROVAL"
         and type(authorization.get("decision_ref")) is str
         and authorization["decision_ref"],
         "training-world authorization authority/decision is missing")
    need(type(authorization.get("scope")) is str
         and authorization["scope"] == expected_world_authorization_scope(context),
         "training-world authorization scope is not this finite root")
    need(Path(authorization.get("root", "")).resolve() == context.root
         and authorization.get("recovery_experiment_id") == context.experiment_id
         and authorization.get("namespace") == context.namespace,
         "training-world authorization root/identity/namespace mismatch")
    need(digest(authorization.get("context_base_sha256")) == context.base_sha256(),
         "training-world authorization does not bind this context")
    need(digest(authorization.get("admission_report_sha256"))
         == digest(context.payload["admission_report"]["sha256"]),
         "training-world authorization does not bind admission")
    need(digest(authorization.get("g3_report_sha256")) == digest(g3_pin["sha256"]),
         "training-world authorization does not bind the exact G3 report")
    need(digest(authorization.get("candidate_manifest_sha256"))
         == digest(context.payload["candidate_manifest"]["sha256"])
         and digest(authorization.get("candidate_index_sha256"))
         == digest(context.payload["candidate_index"]["sha256"])
         and digest(authorization.get("donor_set_sha256"))
         == digest(context.admission_report["donor_set_sha256"]),
         "training-world authorization candidate/donor binding mismatch")
    cap = context.slot_plan["world_cap"] if descendant else 720
    need(integer(authorization.get("max_worlds"), "training authorization max_worlds") == cap,
         "training-world authorization must bind the exact active-run finite cap")
    return {"sha256": pin["sha256"], "max_worlds": cap,
            "decision_ref": authorization["decision_ref"]}


def g3(layout, context_path):
    """Zero-World gate: one load per donor; comparison adds no forward or RNG draw."""
    root = assert_runtime_root(layout)
    context = load_context(layout, context_path, full=True)
    if (Path(layout.ledger) / STOP_SIGNAL).exists():
        raise RecoveryRuntimeError("GLOBAL_STOP is present; G3 refuses to start")
    episode.validate_runtime_imports(layout)
    output = root / "execution" / "recovery_runtime" / "G3_REPORT.json"
    if output.exists():
        raise RecoveryRuntimeError("G3 report already exists; G3 is never replayed")
    entries = []
    for (arm, seed, episode_number), donor in _donor_pairs(context):
        checked = verify_pin({key: donor[key] for key in PIN_KEYS}, "donor " + arm, root=None)
        if getattr(context, "is_descendant", False):
            need(integer(donor.get("cursor_from_closed_journal"), "donor journal cursor")
                 == episode_number, "completed parent journal cursor changed")
        else:
            need(integer(donor.get("observed_episode"), "donor observed_episode")
                 == episode_number, "donor observed episode changed")
        identity_text = _donor_identity_text(donor.get("identity_sha256"),
                                             "donor identity_sha256")
        restored = trainer.load_base_checkpoint(
            checked["path"], arm=arm, seed=seed, episode=episode_number,
            base_identity_sha256=identity_text,
            expected_parameter_sha256=None,
            expected_full_state_sha256=donor.get("expected_full_state_sha256"),
            expected_actor_sha256=donor.get("expected_actor_sha256"))
        need(type(restored.get("payload")) is dict,
             "load_base_checkpoint did not return the raw payload from its single load; "
             "the G3 raw-to-restored comparison cannot run without that returned object")
        payload_comparison = compare_restored_payload(restored["payload"], restored)
        need(payload_comparison["status"] == "MATCH",
             "restored payload components differ from the raw checkpoint")
        need(payload_comparison["scope"]["checkpoint_loads"] == 1,
             "payload comparison did not consume the same single load")
        repeat = trainer.full_state_fingerprint(restored["controller"],
                                                restored["optimizer"])
        need(repeat == restored["full_state"],
             "passive full-state fingerprint changed during G3")
        observed_steps = _optimizer_steps(restored["optimizer"])
        validate_donor_state(context, donor, restored, episode_number)
        if getattr(context, "is_descendant", False):
            stored_step = None  # No initial inventory steps are reused for a new model.
            need(bool(observed_steps), "new donor has no restored Adam parameter state")
        else:
            stored_step = integer(donor.get("stored_adam_parameter_step"),
                                  "donor stored Adam step")
            need(observed_steps and all(step == stored_step for step in observed_steps),
                 "restored Adam parameter steps differ from the donor inventory pin")
        controller = restored["controller"]
        need(controller.cost_head is not None and controller.inner.value_head is not None,
             "restored controller does not expose both critic heads")
        entries.append({
            "arm": arm, "training_seed": seed,
            "donor_path": checked["path"], "donor_bytes": checked["bytes"],
            "donor_sha256": checked["sha256"],
            "donor_identity_sha256": identity_text,
            "episode": episode_number,
            "payload_episode": restored["payload_episode"],
            "payload_seed": restored["payload_seed"],
            "payload_arm": restored["payload_arm"],
            "next_update": episode_number // C.EPISODES_PER_UPDATE,
            "next_update_basis": ("PAYLOAD_AND_CLOSED_LAST_JOURNAL"
                                  if getattr(context, "is_descendant", False)
                                  else "FROZEN_STRIDE4_SOURCE_RECONSTRUCTION"),
            "stored_adam_parameter_step": stored_step,
            "observed_adam_parameter_steps": observed_steps,
            "optimizer_parameter_state_count": len(observed_steps),
            "dual_critic_heads": {
                "value_task": sorted(name for name, _ in
                                     controller.inner.value_head.named_parameters()),
                "value_cost": sorted(name for name, _ in
                                     controller.cost_head.named_parameters()),
            },
            "parameter_fingerprint": restored["parameters"]["parameter_sha256"],
            "actor_fingerprint": restored["full_state"]["actor"],
            "full_state_fingerprint": restored["full_state"]["sha256"],
            "fingerprint_repeat": ("PASSIVE_IDENTICAL"
                                   if repeat == restored["full_state"] else "MISMATCH"),
            "restored_payload_comparison": payload_comparison,
            "rng_restore_observation": {
                "status": "RESTORED_AND_OBSERVED",
                "digests": restored["full_state"]["rng"],
                "saved_payload_containers": "LIST",
                "saved_container_values": "UNKNOWN_NOT_STORED_AS_INDEPENDENT_PIN",
            },
            "saved_fullstate_equality": "UNKNOWN_NOT_STORED",
        })
    payload = {
        "schema": G3_SCHEMA, "status": G3_STATUS, "root": str(root),
        "recovery_experiment_id": context.experiment_id,
        "namespace": context.namespace,
        "context_base_sha256": context.base_sha256(),
        "admission_report_sha256": context.payload["admission_report"]["sha256"],
        "source_manifest_sha256": context.payload["source_manifest"]["sha256"],
        "donor_set_sha256": context.admission_report["donor_set_sha256"],
        "runs": entries, "recorded_utc": utc(),
        "claim_boundary": {
            "worlds": 0,
            "checkpoint_loads_per_donor": 1,
            "comparison_forward_passes": 0,
            "comparison_rng_draws": 0,
            "comparison_optimizer_steps": 0,
            "comparison_mode_changes": 0,
            "controller_initialization_rng_before_restore": "ALLOWED_NOT_COUNTED_AS_COMPARISON",
            "saved_fullstate_fingerprint": "UNKNOWN_NOT_STORED",
        },
        "note": ("raw controller buffers/parameters, optimizer state/param groups and the three "
                 "saved RNG streams were compared directly with the one loaded in-memory state; "
                 "controller construction may use initialization RNG before the saved state is "
                 "restored, while the comparison itself performs no draw, forward or mode change"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    compact.write_json_exclusive(output, payload)
    return {"status": G3_STATUS, "path": str(output), "runs": 6,
            "sha256": sha256_bytes(output.read_bytes()), "worlds": 0,
            "checkpoint_loads_per_donor": 1,
            "comparison_forward_passes": 0, "comparison_rng_draws": 0,
            "comparison_optimizer_steps": 0, "comparison_mode_changes": 0}


def load_execution_bindings(context, g3_report_path, world_authorization_path):
    """Validate two explicit training sidecars without mutating the frozen base context."""
    need(g3_report_path is not None,
         "recovery training requires an explicit G3 report sidecar")
    need(world_authorization_path is not None,
         "recovery training requires an explicit main world-authorization sidecar")
    root = Path(context.root).resolve()
    base = pin_file(context.context_path, root=root)
    need(base == context.context_pin,
         "base recovery context bytes changed before execution binding")
    g3_path = Path(g3_report_path).resolve()
    auth_path = Path(world_authorization_path).resolve()
    for path, label in ((g3_path, "G3 report sidecar"),
                        (auth_path, "world-authorization sidecar")):
        evidence.inside(root, path)
        need(path.is_file(), label + " is missing: " + str(path))
    need(len({str(context.context_path.resolve()), str(g3_path), str(auth_path)}) == 3,
         "base context, G3 report and world authorization must be distinct files")
    checked_g3, report = _read_json_pin(
        pin_file(g3_path, root=root), "recovery G3 report", root, True)
    g3_state = validate_g3(context, report, checked_g3)
    checked_auth, authorization = _read_json_pin(
        pin_file(auth_path, root=root), "recovery training-world authorization",
        root, True)
    auth_state = validate_world_authorization(context, authorization, checked_auth,
                                               checked_g3)
    final_base = pin_file(context.context_path, root=root)
    need(final_base == base,
         "base recovery context bytes changed during execution binding")
    return {
        "base_context": base,
        "g3_report": checked_g3,
        "world_authorization": checked_auth,
        "g3_validation": g3_state,
        "world_validation": auth_state,
    }


def validate_donor_state(context, donor, restored, cursor):
    if not getattr(context, "is_descendant", False):
        return
    state = restored.get("state") or {}
    need(type(state.get("next_update")) is int
         and state["next_update"] == donor["next_update"] == cursor // 4
         and state.get("episode") == cursor
         and restored["payload_episode"] == cursor,
         "new donor payload next_update/cursor does not match the closed last journal")
    need(restored["full_state"]["sha256"] == donor["expected_full_state_sha256"]
         and restored["full_state"]["actor"] == donor["expected_actor_sha256"],
         "new donor loaded state differs from terminal telemetry/journal")


def run_spec(context, arm, seed):
    """Read this stage's verified donor cursor/cap rather than a global loop constant."""
    run = normalized_run(arm, seed)
    matches = [item for item in context.runs
               if item.get("arm") == run[0]
               and item.get("training_seed") == run[1]]
    need(len(matches) == 1, "recovery run plan does not contain one exact donor entry")
    item = matches[0]
    start = integer(item.get("start_episode"), "run start_episode")
    end = integer(item.get("end_episode_cap"), "run end_episode_cap")
    need(start % C.EPISODES_PER_UPDATE == 0
         and end > start
         and (end - start) % C.EPISODES_PER_UPDATE == 0,
         "recovery run cursor/cap is not a finite four-episode interval")
    need(item.get("new_world_cap") == end - start,
         "recovery run cap differs from its start/end interval")
    return start, end


def planned_run_entry(context, arm, seed):
    """Return the single dict-shaped run-plan entry; never expose the matches list."""
    matches = [item for item in context.runs
               if item.get("arm") == arm and item.get("training_seed") == seed]
    need(len(matches) == 1, "recovery run plan is ambiguous for %s/%d" % (arm, seed))
    need(type(matches[0]) is dict, "recovery run-plan entry is not a mapping")
    return matches[0]


def resume_branch_fields(planned, donor_episode, start):
    """Derive telemetry-only branch provenance from the real run-plan schema."""
    need(type(planned) is dict, "planned recovery run must be a mapping")
    need(type(donor_episode) is int and type(start) is int,
         "donor/start episodes must be integers")
    return {
        "rng_restart_branch": donor_episode != start,
        "branch_mode": planned.get("branch_mode"),
        "history_gap": planned.get("known_history_gap"),
    }

def _prepared_run(context):
    by_run = {}
    for item in context.seed_plan:
        by_run.setdefault((item["arm"], item["training_seed"]), {})[
            item["episode_index"]] = item
    return by_run


def _assert_new_key(context, record, prior_keys, claimed_keys, emitted):
    key = (record["arm"], record["training_seed"], record["scene_seed"],
           record["action_rng_seed"])
    need(key not in context.old_runtime_keys,
         "proposed recovery key collides with any retained old physical rollout")
    need(key not in claimed_keys,
         "proposed recovery key collides with a prior recovery claim")
    need(key not in emitted,
         "proposed recovery key repeats inside this run")
    return key


def _update_journal_path(layout, arm, seed, update_index):
    return (Path(layout.root) / "execution" / "recovery" / "update_journal"
            / ("%s_s%d" % (arm, int(seed)))
            / ("update_%06d.json" % int(update_index)))


def recovery_stage_origin(context):
    stage_index = getattr(context, "stage_index", None)
    need(type(stage_index) is int and stage_index >= 1,
         "recovery row needs a validated positive stage index")
    from . import recovery_cutover as X
    return {"revision": context.experiment_id,
            "kind": X.BRANCH_MODE,
            "stage": stage_index, "namespace": context.namespace,
            "exact_continuation_claimed": False}


def _checkpoint_state(context, arm, seed, episode_number, next_update):
    stage_index = getattr(context, "stage_index", None)
    need(type(stage_index) is int and stage_index >= 1,
         "recovery checkpoint needs a validated positive stage index")
    return {
        "arm": arm, "training_seed": int(seed), "episode": int(episode_number),
        "next_update": int(next_update),
        "stage": stage_index, "mode": "RECOVERY_DONOR_RNG_RESTART_NEW_BRANCH",
        "namespace": R.WORLD_SEED_NAMESPACE,
    }


def write_update_journal(context, layout, arm, seed, update_index,
                         pending_records, after_state, update):
    arm, seed = normalized_run(arm, seed)
    need(type(pending_records) is list and len(pending_records) == C.EPISODES_PER_UPDATE,
         "an update journal requires exactly its four completed episodes")
    expected = list(range(int(update_index) * C.EPISODES_PER_UPDATE,
                          int(update_index) * C.EPISODES_PER_UPDATE
                          + C.EPISODES_PER_UPDATE))
    need([record.get("episode_index") for record in pending_records] == expected,
         "update journal episode indices are not the exact four-episode prefix")
    need(update.get("update_index") == int(update_index)
         and update.get("episode_indices") == expected,
         "update journal PPO diagnostics do not describe its four episodes")
    before = pending_records[0].get("full_state_sha256_before")
    after = after_state.get("sha256")
    digest(before, "journal full_state_before")
    digest(after, "journal full_state_after")
    finite_tree(pending_records)
    finite_tree(update)
    payload = {
        "schema": JOURNAL_SCHEMA, "root": str(layout.root),
        "recovery_experiment_id": context.experiment_id,
        "namespace": context.namespace, "arm": arm,
        "training_seed": seed, "update_index": int(update_index),
        "episode_indices": expected,
        "full_state_before": before, "full_state_after": after,
        "episodes": [dict(record) for record in pending_records],
        "update": dict(update), "recorded_utc": utc(),
        "claim_note": ("journal is durable before compact rows and result files; "
                       "claims remain consumed if any later write fails"),
    }
    path = _update_journal_path(layout, arm, seed, update_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    compact.write_json_exclusive(path, payload)
    return path


def first_task_certificate_end(checks):
    """First endpoint where the unchanged formal task evaluator certified."""
    for count in range(2, len(checks) + 1):
        if convergence.is_converged(checks[:count]):
            return checks[count - 1]["end_episode"]
    return None


def new_task_certificate_end(checks, inherited_certificate_end):
    """Return only a certificate event formed in this run, never an inherited certificate."""
    if inherited_certificate_end is not None:
        return None
    return first_task_certificate_end(checks)


def run_group(payload):
    """One registered run in one process; claims precede World construction."""
    (root_value, context_path, g3_report_path, world_authorization_path,
     arm, seed) = payload
    from .paths import Layout
    layout = Layout(root_value)
    try:
        context = load_context(layout, context_path, full=True)
        normalize = normalized_run(arm, seed)
        arm, seed = normalize
        execution_bindings = load_execution_bindings(
            context, g3_report_path, world_authorization_path)
        if (Path(layout.ledger) / STOP_SIGNAL).exists():
            raise RecoveryRuntimeError("GLOBAL_STOP is present before a recovery group starts")
        episode.native_import_preflight(layout)
        episode.ensure_import_paths(layout)
        _donor = dict(_donor_pairs(context))
        start, end = run_spec(context, arm, seed)
        run_plan = planned_run_entry(context, arm, seed)
        donor_episode = integer(run_plan.get("donor_episode", start),
                                "run donor_episode")
        donor_next_update = integer(run_plan.get("donor_next_update", donor_episode // 4),
                                    "run donor_next_update")
        need(donor_episode <= start and start - donor_episode in (0, 4),
             "only exact continuation or the registered 1800->1804 donor restart is allowed")
        if donor_episode != start:
            need((arm, seed, donor_episode, start) == ("T1", 91001, 1800, 1804),
                 "unregistered donor restart")
        donor = _donor[(arm, seed, donor_episode)]
        checked_donor = verify_pin({key: donor[key] for key in PIN_KEYS},
                                   "recovery donor " + arm, root=None)
        restored = trainer.load_base_checkpoint(
            checked_donor["path"], arm=arm, seed=seed,
            episode=donor_episode,
            base_identity_sha256=_donor_identity_text(
                donor.get("identity_sha256"), "donor identity_sha256"),
            expected_full_state_sha256=donor.get("expected_full_state_sha256"),
            expected_actor_sha256=donor.get("expected_actor_sha256"))
        validate_donor_state(context, donor, restored, donor_episode)
        controller, optimizer = restored["controller"], restored["optimizer"]
        rows = selected_history(context, arm, seed)
        updates = selected_updates(context, arm, seed)
        planned = _prepared_run(context)[(arm, seed)]
        ledger = RecoveryLedger(context, layout)
        claimed_keys = ledger.existing_runtime_keys(arm, seed)
        emitted = set()
        pending_rows, pending_records = [], []
        update_index = start // C.EPISODES_PER_UPDATE
        initial_state = trainer.full_state_fingerprint(controller, optimizer)
        gate_path = Path(layout.ledger) / GATE_SIGNAL
        check_episodes = {value for value in C.convergence_check_episodes(end)
                          if start < int(value) <= end}
        loss_review_enabled = getattr(context, "loss_review_enabled", False)
        from . import long_stage
        long_target = completed_episode_target(context)
        inherited_task_certified = False
        inherited_certificate_end = None
        if loss_review_enabled:
            inherited_task_certified = run_plan.get("task_certified") is True
            inherited_certificate_end = first_task_certificate_end(
                read_check_certificates(context, layout, arm, seed))
            need(inherited_task_certified == (inherited_certificate_end is not None),
                 "run plan task-certificate flag differs from pinned check history")
        telemetry = {
            "schema": TELEMETRY_SCHEMA, "root": str(layout.root),
            "recovery_experiment_id": context.experiment_id,
            "namespace": context.namespace, "arm": arm, "training_seed": seed,
            "execution_bindings": execution_bindings,
            "start_episode": start, "end_episode_cap": end,
            "new_world_cap": end - start,
            "resume": {
                "donor": checked_donor, "donor_episode": donor_episode,
                "donor_next_update": donor_next_update,
                "start_episode": start, "episode": donor_episode,
                "next_update": update_index,
                "next_update_basis": ("PAYLOAD_AND_CLOSED_LAST_JOURNAL"
                                      if getattr(context, "is_descendant", False)
                                      else "FROZEN_STRIDE4_SOURCE_RECONSTRUCTION"),
                "initial_full_state": initial_state,
                "optimizer_reset": False,
                **resume_branch_fields(run_plan, donor_episode, start),
                "branch_mode": (run_plan.get("branch_mode")
                                if getattr(context, "is_descendant", False)
                                else "FROZEN_INITIAL_CHECKPOINT_PREFIX"),
                "payload_full_state_fingerprint": "UNKNOWN_NOT_STORED",
                "note": ("donor state restored, then a new-branch episode stream starts at "
                         "start_episode; this is not exact continuation after the consumed gap"),
            },
            "episodes": [], "updates": [], "convergence": {
                "checks": [], "not_evaluable": [],
                "converged": inherited_task_certified,
                "task_certificate_end_episode": inherited_certificate_end,
                "stopped_on_task_certificate": False,
            },
        }
        terminal_episode = start
        try:
            for episode_number in range(start, end):
                if (Path(layout.ledger) / STOP_SIGNAL).exists():
                    raise RecoveryRuntimeError("GLOBAL_STOP is present before %s/%d episode %d"
                                               % (arm, seed, episode_number))
                planned_item = planned[episode_number]
                key = _assert_new_key(context, planned_item, context.old_runtime_keys,
                                      claimed_keys, emitted)
                need(planned_item["scene_seed"] == int(C.scene_for(seed, episode_number)),
                     "runtime scene differs from the frozen scene schedule")
                slot = "%s:%s:%d:%04d" % (SLOT_PREFIX, arm, seed, episode_number)
                status = ledger.claim(slot, runtime_key_value=planned_item)
                if status == "stopped":
                    raise RecoveryRuntimeError("GLOBAL_STOP refused claim of " + slot)
                if status == "consumed":
                    raise RecoveryRuntimeError("recovery slot was already consumed: " + slot)
                emitted.add(key)
                claimed_keys.add(key)
                before = trainer.full_state_fingerprint(controller, optimizer)
                row = episode.run_training_world(
                    layout, arm, seed, episode_number, int(planned_item["scene_seed"]),
                    controller, action_rng_seed=int(planned_item["action_rng_seed"]))
                need(int(row.get("generated_goals", -1)) == C.NUM_GOALS
                     and int(row.get("steps", -1)) == C.STEPS,
                     "new World row changed task count or 1000-step length")
                need(int(row.get("scene", -1)) == int(planned_item["scene_seed"])
                     and int(row.get("action_rng_seed", -1))
                     == int(planned_item["action_rng_seed"]),
                     "new World did not use the pinned scene/action keys")
                finite_tree(row)
                pending_rows.append(row)
                pending_records.append({
                    "slot": slot, "episode_index": int(episode_number),
                    "scene": int(planned_item["scene_seed"]),
                    "action_rng_seed": int(planned_item["action_rng_seed"]),
                    "on_time": int(row["on_time"]),
                    "generated_goals": int(row["generated_goals"]),
                    "discovered": int(row["discovered"]),
                    "charged_bytes": int(row["spent_bytes"]),
                    "paid_emissions": int(row["emissions"]),
                    "steps": int(row["steps"]), "seconds": float(row["seconds"]),
                    "actor_sha256_before": before["actor"],
                    "full_state_sha256_before": before["sha256"],
                })
                if len(pending_rows) < C.EPISODES_PER_UPDATE:
                    continue
                update = trainer.ppo_update(controller, optimizer, pending_rows,
                                            arm=arm, update_index=update_index)
                need(update.get("samples", 0) > 0
                     and update.get("minibatches", 0) > 0
                     and update.get("optimizer_steps") == update.get("minibatches"),
                     "real PPO update did not close optimizer-step accounting")
                expected_steps = C.EPOCHS_PER_BATCH * max(
                    1, math.ceil(int(update["samples"]) / float(C.MINIBATCH_STEPS)))
                need(update["optimizer_steps"] == expected_steps,
                     "optimizer steps do not follow samples/4 epochs/minibatch 256")
                update["update_index"] = update_index
                update["episode_indices"] = [record["episode_index"]
                                             for record in pending_records]
                finite_tree(update)
                after = trainer.full_state_fingerprint(controller, optimizer)
                for record in pending_records:
                    record["actor_sha256_after"] = after["actor"]
                    record["full_state_sha256_after"] = after["sha256"]
                    record["update_index"] = update_index
                    record["optimizer_steps"] = int(update["optimizer_steps"])
                write_update_journal(context, layout, arm, seed, update_index,
                                     pending_records, after, update)
                updates[update_index] = dict(update)
                telemetry["updates"].append(dict(update))
                for record, produced in zip(pending_records, pending_rows):
                    state_hashes = {
                        "actor_sha256_before": record["actor_sha256_before"],
                        "actor_sha256_after": record["actor_sha256_after"],
                        "full_state_sha256_before": record["full_state_sha256_before"],
                        "full_state_sha256_after": record["full_state_sha256_after"],
                        "optimizer_steps": record["optimizer_steps"],
                        "update_index": record["update_index"],
                    }
                    compacted = episode.compact_row(
                        produced, phase="train", arm=arm, seed=seed,
                        episode_index=record["episode_index"], scene=record["scene"],
                        slot=record["slot"], state_hashes=state_hashes)
                    compacted["origin"] = recovery_stage_origin(context)
                    compacted["lambda"] = float(C.LAMBDA_FIXED)
                    path = Path(layout.train_row(record["slot"]))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    compact.write_json_exclusive(path, compacted)
                    rows[record["episode_index"]] = compacted
                    ledger.finish(record["slot"], path)
                    telemetry["episodes"].append(dict(record))
                completed_episode = pending_records[-1]["episode_index"] + 1
                terminal_episode = completed_episode
                pending_rows, pending_records = [], []
                if (completed_episode - start) % C.CHECKPOINT_INTERVAL_EPISODES == 0:
                    trainer.save_checkpoint(
                        layout.checkpoint(arm, seed, completed_episode),
                        controller, optimizer,
                        _checkpoint_state(context, arm, seed, completed_episode, update_index + 1),
                        identity_sha256=context.payload["identity"]["sha256"])
                active_runs = [r for r in context.runs if r.get("active", True)]
                gate_run = (active_runs[0]["arm"], active_runs[0]["training_seed"])
                if ((arm, seed) == gate_run
                        and update_index == start // C.EPISODES_PER_UPDATE):
                    signal = {
                        "schema": "stage5b-recovery-gate-first-update/1",
                        "root": str(layout.root), "experiment_id": context.experiment_id,
                        "arm": arm, "training_seed": seed, "update_index": update_index,
                        "episodes_accepted": completed_episode,
                        "full_state_sha256": after["sha256"],
                        "actor_sha256": after["actor"], "recorded_utc": utc(),
                    }
                    gate_path.parent.mkdir(parents=True, exist_ok=True)
                    compact.write_json_exclusive(gate_path, signal)
                if completed_episode in check_episodes:
                    result = evaluate_check(context, layout, arm, seed,
                                            completed_episode, rows, updates)
                    if not result["evaluated"]:
                        telemetry["convergence"]["not_evaluable"].append(result)
                    else:
                        telemetry["convergence"]["checks"].append(result)
                        if result["status"] == convergence.STATUS_INVALID:
                            raise RecoveryRuntimeError(
                                "convergence history invalid at %s/%d episode %d: %s"
                                % (arm, seed, completed_episode, result["path"]))
                        checks = read_check_certificates(context, layout, arm, seed)
                        new_certificate_end = new_task_certificate_end(
                            checks, inherited_certificate_end)
                        should_stop = long_stage.certificate_stop_required(
                            long_target, new_certificate_end, completed_episode)
                        if new_certificate_end == completed_episode:
                            telemetry["convergence"]["converged"] = True
                            telemetry["convergence"]["task_certificate_end_episode"] = completed_episode
                            telemetry["convergence"]["stopped_on_task_certificate"] = should_stop
                            telemetry["convergence"]["converged_at_episode"] = completed_episode
                            if should_stop:
                                break
                update_index += 1
        except Exception as error:
            try:
                request_stop(layout, "%s/%d recovery group failed: %s" % (arm, seed, error))
            except Exception as stop_error:
                raise RecoveryRuntimeError(
                    "%s/%d failed: %s; stop-marker write also failed: %s"
                    % (arm, seed, error, stop_error)) from error
            raise
        need(not pending_rows and not pending_records,
             "group ended with an incomplete four-episode update")
        terminal_checkpoint = layout.checkpoint(arm, seed, terminal_episode)
        if not terminal_checkpoint.is_file():
            trainer.save_checkpoint(
                terminal_checkpoint, controller, optimizer,
                _checkpoint_state(context, arm, seed, terminal_episode,
                                  terminal_episode // C.EPISODES_PER_UPDATE),
                identity_sha256=context.payload["identity"]["sha256"])
        telemetry["terminal_episode"] = int(terminal_episode)
        telemetry["final_full_state"] = trainer.full_state_fingerprint(controller, optimizer)
        telemetry["counts"] = {
            "episodes": len(telemetry["episodes"]),
            "updates": len(telemetry["updates"]),
            "optimizer_steps": sum(int(item["optimizer_steps"])
                                   for item in telemetry["updates"]),
            "expected_new_episodes": end - start,
            "expected_new_updates": (end - start) // C.EPISODES_PER_UPDATE,
        }
        if long_target is not None:
            need(terminal_episode == end and len(telemetry["episodes"]) == end - start
                 and len(telemetry["updates"]) == (end - start) // C.EPISODES_PER_UPDATE,
                 "long stage did not reach its actual-completed episode target")
        elif telemetry["convergence"]["stopped_on_task_certificate"]:
            need(terminal_episode == telemetry["convergence"]["converged_at_episode"],
                 "new task certificate did not stop at its confirming check")
        else:
            need(terminal_episode == end and len(telemetry["episodes"]) == end - start
                 and len(telemetry["updates"]) == (end - start) // C.EPISODES_PER_UPDATE,
                 "recovery run without a new stopping check did not reach its finite stage end")
        telemetry_path = (Path(layout.root) / "execution" / "recovery" / "telemetry"
                          / ("%s_s%d.json" % (arm, seed)))
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        compact.write_json_exclusive(telemetry_path, telemetry)
        checks = read_check_certificates(context, layout, arm, seed)
        certificate = {
            "schema": CERTIFICATE_SCHEMA, "root": str(layout.root),
            "recovery_experiment_id": context.experiment_id,
            "namespace": context.namespace, "arm": arm, "training_seed": seed,
            "status": ("CONVERGED" if telemetry["convergence"]["converged"]
                       else "NOT_YET_CONVERGED"),
            "terminal_episode": int(terminal_episode),
            "task_certificate_end_episode": telemetry["convergence"]["task_certificate_end_episode"],
            "continued_after_task_certificate": bool(
                telemetry["convergence"]["converged"]
                and terminal_episode > telemetry["convergence"]["task_certificate_end_episode"]),
            "stage_end_episode_cap": int(end),
            "consecutive_passes": convergence.consecutive_passes(checks),
            "checks": telemetry["convergence"]["checks"],
            "not_evaluable": telemetry["convergence"]["not_evaluable"],
            "telemetry": str(telemetry_path),
            "telemetry_sha256": sha256_bytes(telemetry_path.read_bytes()),
            "checkpoint": None,
            "note": ("non-check stage end preserves the qualified strict streak; ordinary FAIL "
                     "resets it; a valid certificate stops only this run"),
        }
        if telemetry["convergence"]["converged"]:
            checkpoint = layout.checkpoint(arm, seed, terminal_episode)
            need(checkpoint.is_file(), "converged run has no terminal full checkpoint")
            certificate["checkpoint"] = {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256_bytes(checkpoint.read_bytes()),
            }
        certificate_path_value = Path(layout.root) / "execution" / "recovery" / "certificates" \
            / ("CERTIFICATE_%s_s%d.json" % (arm, seed))
        certificate_path_value.parent.mkdir(parents=True, exist_ok=True)
        compact.write_json_exclusive(certificate_path_value, certificate)
        return {
            "slot": "recovery:%s:%d" % (arm, seed), "status": "ok",
            "episodes": len(telemetry["episodes"]),
            "terminal_episode": int(terminal_episode),
            "converged": bool(telemetry["convergence"]["converged"]),
            "task_certificate_end_episode": telemetry["convergence"]["task_certificate_end_episode"],
            "stopped_on_task_certificate": telemetry["convergence"]["stopped_on_task_certificate"],
            "continued_after_task_certificate": bool(
                telemetry["convergence"]["converged"]
                and terminal_episode > telemetry["convergence"]["task_certificate_end_episode"]),
            "telemetry": str(telemetry_path),
            "certificate": str(certificate_path_value),
        }
    except Exception as error:
        try:
            request_stop(layout, "%s/%d setup or terminalization failed: %s"
                         % (arm, seed, error))
        except Exception as stop_error:
            raise RecoveryRuntimeError(
                "%s/%d failed: %s; stop-marker write also failed: %s"
                % (arm, seed, error, stop_error)) from error
        raise


def _require_clean_recovery_training(layout, context):
    ledger = RecoveryLedger(context, layout)
    if ledger.claims.is_dir() and any(ledger.claims.iterdir()):
        raise RecoveryRuntimeError("recovery claims already exist; this phase is never replayed")
    if ledger.results.is_dir() and any(ledger.results.iterdir()):
        raise RecoveryRuntimeError("recovery results already exist; this phase is never replayed")
    if Path(layout.train_rows).is_dir() and list(Path(layout.train_rows).glob("*.json")):
        raise RecoveryRuntimeError("training rows already exist; the recovery stage is not replayable")
    for run in context.runs:
        if not run.get("active", True):
            continue  # a run retired by this plan consumes no slot and writes no checkpoint
        start, end = run_spec(context, run["arm"], run["training_seed"])
        checkpoint = layout.checkpoint(run["arm"], run["training_seed"], end)
        if checkpoint.exists():
            raise RecoveryRuntimeError("recovery checkpoint already exists: " + str(checkpoint))
    telemetry = Path(layout.root) / "execution" / "recovery" / "telemetry"
    if telemetry.is_dir() and any(telemetry.iterdir()):
        raise RecoveryRuntimeError("recovery telemetry already exists")
    convergence_dir = Path(layout.root) / "execution" / "recovery" / "convergence"
    if convergence_dir.is_dir() and any(convergence_dir.iterdir()):
        raise RecoveryRuntimeError("recovery convergence output already exists")
    need(not (Path(layout.ledger) / GATE_SIGNAL).exists(),
         "a recovery first-update gate signal already exists")


def run_train(layout, context_path, g3_report_path,
              world_authorization_path, workers):
    """Run the finite stage with explicit, read-only G3 and world-authorization sidecars."""
    root = assert_runtime_root(layout)
    context = load_context(layout, context_path, full=True)
    execution_bindings = load_execution_bindings(
        context, g3_report_path, world_authorization_path)
    if (Path(layout.ledger) / STOP_SIGNAL).exists():
        raise RecoveryRuntimeError("GLOBAL_STOP is present; recovery training refuses to start")
    _require_clean_recovery_training(layout, context)
    episode.validate_runtime_imports(layout)
    base = (str(root), str(Path(context_path).resolve()),
            str(Path(g3_report_path).resolve()),
            str(Path(world_authorization_path).resolve()))
    active = [item for item in context.runs if item.get("active", True)]
    need(bool(active), "all six runs are certified; no training Worlds may be allocated")
    tasks = [base + (item["arm"], int(item["training_seed"])) for item in active]
    cap = context.slot_plan["world_cap"]
    need(cap > 0 and sum(run_spec(context, x["arm"], x["training_seed"])[1]
             - run_spec(context, x["arm"], x["training_seed"])[0] for x in active)
         == cap, "active-run cap does not match its frozen slot plan")
    from . import runner as runner_module
    results = runner_module._run_train_pool(layout, tasks, int(workers), run_group)
    need(len(results) == len(active), "recovery training did not close every active run")
    if (Path(layout.ledger) / STOP_SIGNAL).exists():
        raise RecoveryRuntimeError("GLOBAL_STOP is present after recovery training")
    seen = set()
    total = 0
    converged = [(x["arm"], x["training_seed"]) for x in context.runs
                 if not x.get("active", True) and x.get("revision_retired") is not True]
    for result in results:
        need(result.get("status") == "ok", "recovery group did not return accepted status")
        pieces = str(result.get("slot", "")).split(":")
        need(len(pieces) == 3 and pieces[0] == "recovery",
             "recovery result has an invalid run label")
        run = normalized_run(pieces[1], int(pieces[2]))
        need(run not in seen, "recovery results repeat a run")
        seen.add(run)
        episodes = integer(result.get("episodes"), "recovery result episodes")
        need(0 < episodes <= run_spec(context, *run)[1] - run_spec(context, *run)[0]
             and episodes % C.EPISODES_PER_UPDATE == 0,
             "recovery result does not name a four-episode complete prefix")
        terminal = integer(result.get("terminal_episode"), "recovery terminal_episode")
        need(terminal == run_spec(context, *run)[0] + episodes,
             "recovery terminal episode does not close its prefix")
        total += episodes
        if getattr(context, "loss_review_enabled", False):
            checks = read_check_certificates(context, layout, run[0], run[1])
            certificate_end = first_task_certificate_end(checks)
            planned_run = next(item for item in context.runs
                               if item.get("arm") == run[0]
                               and item.get("training_seed") == run[1])
            task_certified = planned_run.get("task_certified") is True or certificate_end is not None
            expected_end = (planned_run.get("task_certificate_end_episode")
                            if planned_run.get("task_certified") is True else certificate_end)
            need(result.get("converged") is task_certified
                 and result.get("task_certificate_end_episode") == expected_end,
                 "recovery result changed the preserved task certificate")
            if task_certified:
                need(certificate_end == expected_end
                     and layout.checkpoint(run[0], run[1], terminal).is_file(),
                     "task-certified run lacks its preserved check/cursor checkpoint")
                if result.get("stopped_on_task_certificate"):
                    need(convergence.is_converged(checks)
                         and checks[-1]["end_episode"] == terminal,
                         "new task certificate did not stop at its check")
                else:
                    need(terminal == run_spec(context, *run)[1],
                         "continued certified run did not close its finite stage")
                converged.append(run)
        elif result.get("converged"):
            checks = read_check_certificates(context, layout, run[0], run[1])
            need(convergence.is_converged(checks) and checks
                 and checks[-1]["end_episode"] == terminal
                 and layout.checkpoint(run[0], run[1], terminal).is_file(),
                 "converged run lacks its two strict checks and terminal checkpoint")
            converged.append(run)
    need(seen == {(item["arm"], int(item["training_seed"])) for item in active},
         "recovery results do not cover exactly the frozen active runs")
    need(total <= cap, "recovery accepted prefix exceeds its frozen finite cap")
    ledger = RecoveryLedger(context, layout)
    counts = ledger.counts()
    need(counts["claimed"] == total and counts["results"] == total
         and counts["ok"] == total,
         "recovery claims/results do not equal the accepted episode prefixes")
    rows = sorted(Path(layout.train_rows).glob("*.json")) if Path(layout.train_rows).is_dir() else []
    need(len(rows) == total, "recovery rows do not equal the accepted prefixes")
    final_bindings = load_execution_bindings(
        context, g3_report_path, world_authorization_path)
    need(final_bindings == execution_bindings,
         "base context, G3 report or world authorization changed during training")
    return {
        "status": "FINITE_RECOVERY_STAGE_COMPLETE",
        "worlds": total, "unused_world_slots": cap - total,
        "unclaimed_slots": cap - total,
        "execution_bindings": final_bindings,
        "g3": final_bindings["g3_validation"],
        "authorization": final_bindings["world_validation"],
        "converged_runs": sorted("%s/%d" % run for run in converged),
        "certified_run_count": len(converged),
        "convergence_complete": len(converged) == 6,
        "goal_complete": False,  # Separately registered final 48 DEV_REUSED eval is still required.
        "parent_context": (context.descendant["parent_context"]
                           if getattr(context, "is_descendant", False) else None),
        "slot_counts": counts, "results": results,
        "note": ("stage cap exhaustion is not convergence; unused slots remain unclaimed and "
                 "no cancellation record is fabricated"),
    }


def _context_source_files(layout):
    """Build the complete code/test/protocol plus native closure from frozen sources."""
    from . import guard
    root = Path(layout.root).resolve()
    code_paths = sorted(Path(path).resolve() for path in guard._new_source_paths(layout))
    native_paths = sorted(Path(path).resolve() for path in guard.allowlist_paths(layout))
    special_roles = {
        "stage5b.py": "cli",
        "stage5b_comm/recovery_runtime.py": "runtime",
        "stage5b_comm/runner.py": "runner",
        "stage5b_comm/guard.py": "guard",
        "stage5b_comm/episode.py": "episode",
        "stage5b_comm/trainer.py": "trainer",
        "stage5b_comm/world.py": "world",
        "stage5b_comm/recovery_admission.py": "recovery_admission",
        "stage5b_comm/recovery_evidence.py": "recovery_evidence",
    }
    files = []
    seen_roles = set()
    seen_paths = set()

    def add(path, category, preferred_role=None):
        path = Path(path).resolve()
        if category == "code":
            evidence.inside(root, path)
        key = str(path).casefold()
        need(key not in seen_paths, "source manifest repeats a physical path")
        seen_paths.add(key)
        role = preferred_role or ("%s_%04d" % (category, len(files)))
        need(role not in seen_roles, "source manifest repeats a role")
        seen_roles.add(role)
        pin = pin_file(path, root=(root if category == "code" else None))
        files.append({"category": category, "role": role,
                      "path": pin["path"], "bytes": pin["bytes"],
                      "sha256": pin["sha256"]})

    for path in code_paths:
        relative = path.relative_to(root).as_posix()
        if relative in GENERATED_PROTOCOL_PATHS:
            continue
        add(path, "code", special_roles.get(relative))
    for path in native_paths:
        add(path, "native")
    return files


def build_source_manifest(layout):
    root = assert_runtime_root(layout)
    payload = {
        "schema": SOURCE_MANIFEST_SCHEMA, "status": "FROZEN", "root": str(root),
        "files": _context_source_files(layout),
        "note": "IMMUTABLE_PRE_RUN_SOURCE_AND_NATIVE_CLOSURE",
    }
    _validate_source_manifest(payload, root, R.WORLD_SEED_NAMESPACE, layout=layout)
    return payload


def build_slot_plan(root, seed_plan):
    slots = []
    for item in seed_plan:
        slots.append({
            "slot": "%s:%s:%d:%04d" % (SLOT_PREFIX, item["arm"],
                                        item["training_seed"], item["episode_index"]),
            "phase": "train", "arm": item["arm"],
            "training_seed": item["training_seed"],
            "episode": item["episode_index"], "scene": item["scene_seed"],
            "action_rng_seed": item["action_rng_seed"],
        })
    payload = {
        "schema": SLOT_PLAN_SCHEMA, "status": "FROZEN", "root": str(Path(root).resolve()),
        "recovery_experiment_id": R.RECOVERY_STAGE_ID,
        "namespace": R.WORLD_SEED_NAMESPACE, "world_cap": 720, "slots": slots,
        "note": ("MECHANICALLY_DERIVED_FROM_PROPOSED_RUNTIME_SEED_KEYS; "
                 "NO_SEED_ADAPTATION_OR_UNUSED_SLOT"),
    }
    context = types.SimpleNamespace(
        root=Path(root).resolve(), experiment_id=R.RECOVERY_STAGE_ID,
        namespace=R.WORLD_SEED_NAMESPACE, seed_plan=seed_plan)
    _validate_slot_plan(payload, context)
    return payload


def build_recovery_context_bundle(layout, admission_report_path, approval_path,
                                  evidence_index_path, candidate_directory,
                                  protocol_path=None):
    """Derive the four frozen runtime artifacts in memory; this performs no writes.

    The full recovery admission validator is executed again so an old or unverifiable report
    cannot be turned into a context. The returned files mapping is relative filename to exact
    UTF-8 bytes for a later exclusive writer.
    """
    root = assert_runtime_root(layout)
    protocol = Path(protocol_path or (root / "protocol" / "RECOVERY_PROTOCOL.json")).resolve()
    evidence.inside(root, protocol)
    need(protocol == (root / "protocol" / "RECOVERY_PROTOCOL.json").resolve(),
         "context builder accepts only the registered recovery protocol")
    report_path = Path(admission_report_path).resolve()
    approval = Path(approval_path).resolve()
    evidence_index = Path(evidence_index_path).resolve()
    candidate_dir = Path(candidate_directory).resolve()
    for path, label in ((report_path, "admission report"),
                        (approval, "approval"),
                        (evidence_index, "evidence index"),
                        (candidate_dir, "candidate directory")):
        evidence.inside(root, path)
        need(path.exists(), "missing " + label + ": " + str(path))
    recomputed = admission.validate_recovery_admission(
        root, candidate_dir, evidence_index, approval)
    report_pin, report = _read_json_pin(
        pin_file(report_path, root=root), "admission report", root, True)
    need(report == recomputed,
         "admission report file differs from a fresh successful full validation")

    protocol_pin, protocol_payload = _read_json_pin(
        pin_file(protocol, root=root), "recovery protocol", root, True)
    need(protocol_pin["sha256"] == evidence.DRAFT_SHA,
         "context builder refuses a changed recovery protocol")
    evidence.validate_draft(protocol_payload, root)
    approval_pin = pin_file(approval, root=root)
    evidence_pin = pin_file(evidence_index, root=root)
    candidate_pin, candidate = _read_json_pin(
        pin_file(candidate_dir / "candidate.json", root=root),
        "candidate manifest", root, True)
    index_pin, index = _read_json_pin(
        pin_file(candidate_dir / "index.json", root=root),
        "candidate index", root, True)
    admission.validate_index_summary(index)
    output_pins = candidate.get("output_files")
    index_outputs = index.get("outputs")
    need(type(output_pins) is list and len(output_pins) == 139
         and type(index_outputs) is list and len(index_outputs) == 138,
         "candidate/index output sets are not the admitted 139/138 closure")
    seed_matches = [item for item in index_outputs
                    if Path(item.get("path", "")).name == "proposed_runtime_seed_keys.json"]
    need(len(seed_matches) == 1, "candidate index has no unique runtime seed plan")
    seed_pin = verify_pin(seed_matches[0], "runtime seed plan", root=root)
    seed_plan = _load_seed_plan(seed_pin["path"], root)
    need(digest(recomputed["candidate_manifest_sha256"]) == candidate_pin["sha256"]
         and digest(recomputed["candidate_index_sha256"]) == index_pin["sha256"]
         and digest(recomputed["evidence_index_sha256"]) == evidence_pin["sha256"]
         and digest(recomputed["protocol_sha256"]) == protocol_pin["sha256"],
         "fresh admission result does not bind the builder inputs")

    source_manifest = build_source_manifest(layout)
    source_bytes = compact.json_bytes(source_manifest)
    source_pin = {
        "path": str(root / SOURCE_MANIFEST_RELATIVE),
        "bytes": len(source_bytes), "sha256": sha256_bytes(source_bytes),
    }
    slot_plan = build_slot_plan(root, seed_plan)
    slot_bytes = compact.json_bytes(slot_plan)
    slot_pin = {
        "path": str(root / SLOT_PLAN_RELATIVE),
        "bytes": len(slot_bytes), "sha256": sha256_bytes(slot_bytes),
    }
    recorded = utc()
    identity = {
        "schema": IDENTITY_SCHEMA, "status": "FROZEN", "root": str(root),
        "recovery_experiment_id": R.RECOVERY_STAGE_ID,
        "namespace": R.WORLD_SEED_NAMESPACE,
        "protocol_sha256": protocol_pin["sha256"],
        "source_manifest_sha256": source_pin["sha256"],
        "admission_report_sha256": report_pin["sha256"],
        "approval_sha256": approval_pin["sha256"],
        "candidate_manifest_sha256": candidate_pin["sha256"],
        "candidate_index_sha256": index_pin["sha256"],
        "evidence_index_sha256": evidence_pin["sha256"],
        "seed_plan_sha256": seed_pin["sha256"],
        "slot_plan_sha256": slot_pin["sha256"],
        "donor_set_sha256": recomputed["donor_set_sha256"],
        "recorded_utc": recorded,
        "note": ("FROZEN recovery runtime identity; not a normal complete-parent identity and "
                 "not an ADMITTED parent"),
    }
    exact_keys(identity, IDENTITY_KEYS, "built recovery identity")
    identity_bytes = compact.json_bytes(identity)
    identity_pin = {
        "path": str(root / IDENTITY_RELATIVE), "bytes": len(identity_bytes),
        "sha256": sha256_bytes(identity_bytes),
    }
    context = {
        "schema": CONTEXT_SCHEMA, "status": CONTEXT_STATUS, "root": str(root),
        "recovery_experiment_id": R.RECOVERY_STAGE_ID,
        "namespace": R.WORLD_SEED_NAMESPACE,
        "admission_report": report_pin, "approval": approval_pin,
        "evidence_index": evidence_pin,
        "candidate_manifest": candidate_pin, "candidate_index": index_pin,
        "protocol": protocol_pin, "identity": identity_pin,
        "slot_plan": slot_pin, "seed_plan": seed_pin,
        "source_manifest": source_pin, "certificate_sources": [],
        "g3_report": None, "world_authorization": None,
        "note": ("FROZEN prepare-only context; no G3 or World authorization is implied"),
    }
    exact_keys(context, CONTEXT_KEYS, "built recovery context")
    runs = tuple(evidence.validate_draft(protocol_payload, root))
    built_context = RecoveryContext(
        root=root, context_path=root / CONTEXT_RELATIVE,
        context_pin={"path": str(root / CONTEXT_RELATIVE), "bytes": 0,
                     "sha256": "0" * 64},
        payload=context, admission_report=report, approval={},
        candidate=candidate, index=index, protocol=protocol_payload,
        identity=identity, slot_plan=slot_plan, source_manifest=source_manifest,
        seed_plan=seed_plan, bundle_pins={}, runs=runs,
        old_runtime_keys=frozenset())
    _validate_source_manifest(source_manifest, root, R.WORLD_SEED_NAMESPACE, layout=layout)
    _validate_identity(identity, built_context, R.WORLD_SEED_NAMESPACE)
    _validate_slot_plan(slot_plan, built_context)
    context_bytes = compact.json_bytes(context)
    return {
        "files": {
            SOURCE_MANIFEST_RELATIVE: source_bytes,
            SLOT_PLAN_RELATIVE: slot_bytes,
            IDENTITY_RELATIVE: identity_bytes,
            CONTEXT_RELATIVE: context_bytes,
        },
        "pins": {
            SOURCE_MANIFEST_RELATIVE: source_pin,
            SLOT_PLAN_RELATIVE: slot_pin,
            IDENTITY_RELATIVE: identity_pin,
            CONTEXT_RELATIVE: {
                "path": str(root / CONTEXT_RELATIVE), "bytes": len(context_bytes),
                "sha256": sha256_bytes(context_bytes),
            },
        },
        "authorization": {
            "G3": "NOT_RUN", "world": "NOT_AUTHORIZED", "training": "NOT_STARTED",
        },
    }


def write_recovery_context_bundle(bundle, layout):
    """Exclusively write the builder artifacts; never overwrite or auto-authorize."""
    root = assert_runtime_root(layout)
    files = bundle.get("files") if isinstance(bundle, dict) else None
    need(type(files) is dict
         and set(files) == set(GENERATED_PROTOCOL_PATHS),
         "recovery prepare bundle has the wrong file set")
    targets = {}
    for relative, data in files.items():
        need(type(data) is bytes and data, "bundle artifact is not non-empty bytes")
        path = (root / relative).resolve()
        evidence.inside(root, path)
        need(not path.exists(), "refusing to overwrite recovery artifact: " + str(path))
        targets[relative] = (path, data)
    for relative in (SOURCE_MANIFEST_RELATIVE, SLOT_PLAN_RELATIVE,
                     IDENTITY_RELATIVE, CONTEXT_RELATIVE):
        path, data = targets[relative]
        path.parent.mkdir(parents=True, exist_ok=True)
        compact.write_bytes_exclusive(path, data)
    return {relative: pin_file(path, root=root)
            for relative, (path, _data) in targets.items()}


def _normalise_rng_value(value):
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _normalise_rng_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalise_rng_value(item) for item in value]
    return value


def _rng_snapshot():
    torch, np, _cm, _action, _log_prob, _saved, _restore = trainer.deps()
    python = random.getstate()
    numpy = np.random.get_state()
    return {
        "python": [python[0], list(python[1]), python[2]],
        "numpy": [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]],
        "torch": torch.get_rng_state().tolist(),
    }


def _tensor_comparison(torch, raw, loaded, path):
    if torch.is_tensor(raw) or torch.is_tensor(loaded):
        if not (torch.is_tensor(raw) and torch.is_tensor(loaded)):
            return [{"path": path, "kind": "tensor-vs-non-tensor"}]
        if tuple(raw.shape) != tuple(loaded.shape):
            return [{"path": path, "kind": "shape",
                     "raw": list(raw.shape), "restored": list(loaded.shape)}]
        if raw.dtype != loaded.dtype:
            return [{"path": path, "kind": "dtype",
                     "raw": str(raw.dtype), "restored": str(loaded.dtype)}]
        raw_cpu = raw.detach().cpu()
        loaded_cpu = loaded.detach().cpu()
        if torch.equal(raw_cpu, loaded_cpu):
            return []
        difference = raw_cpu != loaded_cpu
        coordinates = difference.nonzero()[:5].tolist()
        return [{"path": path, "kind": "value",
                 "mismatch_count": int(difference.sum().item()),
                 "first_coordinates": coordinates}]
    return None


def _compare_tree(raw, loaded, path, mismatches, torch, np):
    tensor = _tensor_comparison(torch, raw, loaded, path)
    if tensor is not None:
        mismatches.extend(tensor)
        return
    if isinstance(raw, np.ndarray) or isinstance(loaded, np.ndarray):
        if not (isinstance(raw, np.ndarray) and isinstance(loaded, np.ndarray)):
            mismatches.append({"path": path, "kind": "array-vs-non-array"})
            return
        if raw.shape != loaded.shape or raw.dtype != loaded.dtype:
            mismatches.append({"path": path, "kind": "array-shape-or-dtype",
                               "raw_shape": list(raw.shape), "restored_shape": list(loaded.shape),
                               "raw_dtype": str(raw.dtype),
                               "restored_dtype": str(loaded.dtype)})
            return
        if not np.array_equal(raw, loaded):
            positions = np.argwhere(raw != loaded)[:5].tolist()
            mismatches.append({"path": path, "kind": "array-value",
                               "first_coordinates": positions})
        return
    if isinstance(raw, dict) or isinstance(loaded, dict):
        if not (isinstance(raw, dict) and isinstance(loaded, dict)):
            mismatches.append({"path": path, "kind": "mapping-vs-nonmapping"})
            return
        if set(raw) != set(loaded):
            mismatches.append({"path": path, "kind": "mapping-keys",
                               "raw_only": [repr(value) for value in sorted(
                                   set(raw) - set(loaded), key=repr)],
                               "restored_only": [repr(value) for value in sorted(
                                   set(loaded) - set(raw), key=repr)]})
            return
        for key in sorted(raw, key=repr):
            _compare_tree(raw[key], loaded[key], path + "." + str(key), mismatches,
                          torch, np)
        return
    if isinstance(raw, (list, tuple)) or isinstance(loaded, (list, tuple)):
        if not (isinstance(raw, (list, tuple)) and isinstance(loaded, (list, tuple))):
            mismatches.append({"path": path, "kind": "sequence-vs-nonsequence"})
            return
        if len(raw) != len(loaded):
            mismatches.append({"path": path, "kind": "sequence-length",
                               "raw": len(raw), "restored": len(loaded)})
            return
        for index, (left, right) in enumerate(zip(raw, loaded)):
            _compare_tree(left, right, path + "[%d]" % index, mismatches, torch, np)
        return
    if type(raw) is not type(loaded) or raw != loaded:
        mismatches.append({"path": path, "kind": "scalar-or-type",
                           "raw": repr(raw), "restored": repr(loaded)})


def _controller_state(controller):
    return {
        "trunk": controller.inner.trunk.state_dict(),
        "action_head": controller.inner.action_head.state_dict(),
        "value_task_head": controller.inner.value_head.state_dict(),
        "value_cost_head": controller.cost_head.state_dict(),
    }


def _model_mode_snapshot(controller):
    modules = {
        "trunk": controller.inner.trunk,
        "action_head": controller.inner.action_head,
        "value_task_head": controller.inner.value_head,
        "value_cost_head": controller.cost_head,
    }
    return {name: bool(getattr(module, "training"))
            for name, module in sorted(modules.items())}


def compare_restored_payload(payload, restored):
    """Direct raw-payload versus one restored in-memory state comparison.

    The caller must pass the payload retained by the same single load that produced restored.
    This function performs no checkpoint read, forward pass, RNG draw, optimizer step or mode
    change. Tensor equality includes shape, dtype and values; RNG wrappers are normalised only
    for tuple/list/array serialization differences.
    """
    need(type(payload) is dict, "restored payload is unavailable")
    need(type(restored) is dict and "controller" in restored and "optimizer" in restored,
         "restored runtime state is incomplete")
    torch, np, _cm, _action, _log_prob, _saved, _restore = trainer.deps()
    rng_before = _rng_snapshot()
    mode_before = _model_mode_snapshot(restored["controller"])
    mismatches = []

    raw_components = payload.get("controller")
    need(type(raw_components) is dict, "raw checkpoint controller is not a mapping")
    live_components = _controller_state(restored["controller"])
    need(set(raw_components) == set(live_components),
         "raw/live controller component names differ")
    controller_results = {}
    for name in sorted(raw_components):
        component_mismatches = []
        _compare_tree(raw_components[name], live_components[name],
                      "controller." + name, component_mismatches, torch, np)
        controller_results[name] = {
            "status": "MATCH" if not component_mismatches else "MISMATCH",
            "mismatches": component_mismatches,
            "includes_buffers": True,
        }
        mismatches.extend(component_mismatches)

    optimizer_mismatches = []
    _compare_tree(payload.get("optimizer"), restored["optimizer"].state_dict(),
                  "optimizer", optimizer_mismatches, torch, np)
    mismatches.extend(optimizer_mismatches)

    raw_rng = _normalise_rng_value(payload.get("rng"))
    live_rng = _normalise_rng_value(rng_before)
    rng_mismatches = []
    _compare_tree(raw_rng, live_rng, "rng", rng_mismatches, torch, np)
    mismatches.extend(rng_mismatches)
    rng_streams = {}
    for stream in ("python", "numpy", "torch"):
        stream_mismatches = [item for item in rng_mismatches
                             if item["path"].startswith("rng." + stream)]
        rng_streams[stream] = {
            "status": "MATCH" if not stream_mismatches else "MISMATCH",
            "mismatches": stream_mismatches,
        }

    rng_after = _rng_snapshot()
    mode_after = _model_mode_snapshot(restored["controller"])
    rng_preserved = (_normalise_rng_value(rng_after)
                     == _normalise_rng_value(rng_before))
    modes_preserved = mode_after == mode_before
    if not rng_preserved:
        mismatches.append({"path": "comparison.rng", "kind": "comparison-advanced-rng"})
    if not modes_preserved:
        mismatches.append({"path": "comparison.mode",
                           "kind": "comparison-changed-model-mode"})
    result = {
        "status": "MATCH" if not mismatches else "MISMATCH",
        "controller": {
            "status": ("MATCH" if all(item["status"] == "MATCH"
                                       for item in controller_results.values())
                       else "MISMATCH"),
            "components": controller_results,
        },
        "optimizer": {
            "status": "MATCH" if not optimizer_mismatches else "MISMATCH",
            "includes_state_and_param_groups": True,
            "mismatches": optimizer_mismatches,
        },
        "rng": {
            "status": "MATCH" if not rng_mismatches else "MISMATCH",
            "streams": rng_streams,
        },
        "comparison_preserved_rng": rng_preserved,
        "comparison_preserved_model_mode": modes_preserved,
        "model_mode_before": mode_before,
        "model_mode_after": mode_after,
        "scope": {
            "checkpoint_loads": 1,
            "forward_passes": 0,
            "comparison_rng_draws": 0,
            "comparison_optimizer_steps": 0,
            "comparison_mode_changes": 0,
        },
        "mismatches": mismatches,
    }
    # Raw component equality and comparison-scope preservation together decide PASS.
    return result


def prepare_context_main(argv=None):
    """Explicit prepare-only builder entry; no G3, World or training authorization."""
    import argparse
    from .paths import Layout, root_dir
    parser = argparse.ArgumentParser(
        description="Build the frozen recovery runtime context from a successful full admission")
    parser.add_argument("phase", choices=("prepare-recovery-context",))
    parser.add_argument("--root", required=True)
    parser.add_argument("--admission-report", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--candidate-directory", required=True)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--write", action="store_true",
                        help="exclusively create the four frozen protocol artifacts")
    args = parser.parse_args(argv)
    layout = Layout(root_dir(args.root))
    bundle = build_recovery_context_bundle(
        layout, args.admission_report, args.approval, args.evidence_index,
        args.candidate_directory, args.protocol)
    if args.write:
        pins = write_recovery_context_bundle(bundle, layout)
        mode = "WRITTEN_EXCLUSIVE"
    else:
        pins = bundle["pins"]
        mode = "DRY_RUN_NOT_WRITTEN"
    summary = {
        "phase": args.phase, "status": "PREPARED_NOT_G3_NOT_AUTHORIZED",
        "mode": mode, "context": pins[CONTEXT_RELATIVE],
        "files": pins, "authorization": bundle["authorization"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(prepare_context_main())