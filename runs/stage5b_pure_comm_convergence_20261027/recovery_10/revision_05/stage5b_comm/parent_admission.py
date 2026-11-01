"""Read-only admission of a normal completed stage-1 parent. No model imports or writes.

The checked-in bundle is deliberately PENDING. Filling it and recording main-model
acceptance are later authorised evidence operations; this module never manufactures pins.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from . import constants as C

SCHEMA = "stage5b-stage2-parent-admission/1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RAW_TOOL_EXIT_POINTER = "/exit_code"
DIAGNOSTICS = ("policy_surrogate", "value_loss_task", "value_loss_cost", "entropy",
               "approx_kl", "gradient_norm_mean")


class ParentAdmissionError(RuntimeError):
    pass


class PendingParentAdmission(ParentAdmissionError):
    pass


def _require(condition, message):
    if not condition:
        raise ParentAdmissionError(message)


def _integer(value):
    _require(type(value) is int, "required integer is missing or malformed: %r" % (value,))
    return value


def _hex(value):
    _require(isinstance(value, str) and HEX64.fullmatch(value) is not None,
             "required actual SHA256 is missing or malformed: %r" % (value,))
    return value


def _bad_constant(value):
    raise ParentAdmissionError("non-finite JSON constant: %s" % value)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_bad_constant)
    except (OSError, ValueError) as error:
        raise ParentAdmissionError("cannot read evidence %s: %s" % (path, error)) from error


def pin_path(root, pin, expected=None):
    _require(isinstance(pin, dict), "an actual file pin is required")
    relative = Path(str(pin.get("path", "")))
    _require(bool(str(pin.get("path", ""))) and not relative.is_absolute(),
             "evidence paths must be explicit and relative to their declared root")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ParentAdmissionError("evidence path escapes its explicit root: %s" % path) from error
    if expected is not None:
        _require(path == (root / expected).resolve(), "unexpected evidence path: %s" % path)
    return path


def verify_pin(root, pin, expected=None):
    path = pin_path(root, pin, expected)
    expected_bytes = _integer(pin.get("bytes"))
    _require(expected_bytes > 0, "evidence pin must name a positive byte size")
    expected_sha = _hex(pin.get("sha256"))
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ParentAdmissionError("pinned evidence missing: %s" % path) from error
    _require(len(raw) == expected_bytes and hashlib.sha256(raw).hexdigest() == expected_sha,
             "pinned evidence bytes/SHA changed: %s" % path)
    return path


def require_admitted(layout):
    path = Path(layout.parent_admission)
    if not path.is_file():
        raise PendingParentAdmission("PENDING_PARENT_ADMISSION: no completed admission bundle")
    document = _read_json(path)
    _require(isinstance(document, dict) and document.get("schema") == SCHEMA,
             "foreign parent-admission schema")
    _require(document.get("experiment_id") == C.RUN_ID, "admission belongs to another stage")
    if document.get("status") != "ADMITTED":
        raise PendingParentAdmission("PENDING_PARENT_ADMISSION: parent terminal evidence/main acceptance incomplete")
    parent = document.get("parent") or {}
    _require(parent.get("experiment_id") == C.BASE_EXPERIMENT_ID
             and Path(str(parent.get("root", ""))).resolve() == Path(C.BASE_ROOT).resolve(),
             "admission names a different direct parent")
    return document


def evidence_digest(document):
    body = {key: value for key, value in document.items() if key != "main_model_acceptance"}
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_source_intervals(sources=None, stop=None):
    sources = C.HISTORY_SOURCES if sources is None else sources
    stop = C.STAGE_START_EPISODE if stop is None else stop
    cursor = 0
    ids = set()
    for source in sources:
        _require(source["source_id"] not in ids, "duplicate history source id")
        ids.add(source["source_id"])
        start, end = source["episodes"]
        _require(type(start) is int and type(end) is int and start == cursor and end > start,
                 "history sources overlap, have a gap or carry an invalid interval")
        cursor = end
    _require(cursor == stop, "explicit history sources do not close at the resume episode")
    return True


def _source(source_id):
    matches = [item for item in C.HISTORY_SOURCES if item["source_id"] == source_id]
    _require(len(matches) == 1, "unknown or conflicting explicit history source %r" % source_id)
    return matches[0]


def validate_history_row(row, source, arm, seed, index):
    _require(row.get("schema") == source["row_schema"] and row.get("phase") == "train",
             "history row has the wrong source schema/phase")
    _require(row.get("arm") == arm and type(row.get("seed")) is int
             and row["seed"] == seed and type(row.get("episode_index")) is int
             and row["episode_index"] == index, "history row run/index conflict")
    _require(type(row.get("scene")) is int and row["scene"] == C.scene_for(seed, index),
             "history row has the wrong registered scene")
    _require(type(row.get("generated_goals")) is int and row["generated_goals"] == C.NUM_GOALS,
             "history row must record 24 generated goals")
    _require(type(row.get("steps")) is int and row["steps"] == C.STEPS,
             "history row must record 1000 steps")
    _require(type(row.get("on_time")) is int and 0 <= row["on_time"] <= C.NUM_GOALS,
             "history row has invalid on_time")
    charged = row.get("charged_bytes")
    _require(type(charged) in (int, float) and math.isfinite(charged)
             and 0 <= charged <= C.BUDGET_BYTES, "history row has invalid/non-finite charged bytes")
    return row


def _run_key(item):
    arm, seed = item.get("arm"), item.get("seed")
    _require(arm in C.ARMS_TRAIN and type(seed) is int and seed in C.TRAIN_SEEDS,
             "unregistered run selector")
    return arm, seed


def _keyed(items, key_function, label):
    _require(isinstance(items, list), "%s must be an explicit list" % label)
    result = {}
    for item in items:
        _require(isinstance(item, dict), "%s contains a non-mapping" % label)
        key = key_function(item)
        _require(key not in result, "%s contains a duplicate/conflicting key %r" % (label, key))
        result[key] = item
    return result


def validate_parent_terminal(payload, identity_sha, plan_sha):
    _require(payload.get("phase") == "train" and payload.get("stage") == "terminal"
             and type(payload.get("returncode")) is int and payload["returncode"] == 0,
             "parent train CLI has no real terminal returncode 0")
    _require(payload.get("identity_sha256") == identity_sha
             and payload.get("slot_plan_sha256") == plan_sha, "parent CLI identity/plan conflict")
    detail = payload.get("detail") or {}
    _require(detail.get("groups") == 6 and detail.get("episodes") == 4320,
             "parent CLI does not close six full runs / 4320 worlds")
    results = _keyed(detail.get("results"), lambda item: item.get("slot"), "parent CLI results")
    expected = {"train:%s:%d" % (arm, seed) for arm in C.ARMS_TRAIN for seed in C.TRAIN_SEEDS}
    _require(set(results) == expected, "parent CLI run terminals are missing or conflicting")
    for item in results.values():
        _require(item.get("status") == "ok" and item.get("episodes") == 720
                 and item.get("terminal_episode") == 1320 and item.get("converged") is False,
                 "parent run terminal is incomplete or claims an impossible stage-1 convergence")
    counts = detail.get("slot_counts") or {}
    _require(counts.get("claimed") == 4320 and counts.get("results") == 4320
             and counts.get("by_status") == {"ok": 4320, "failed": 0, "terminated": 0},
             "parent CLI ledger counts do not close")
    return True


def _pointer(payload, pointer):
    _require(isinstance(pointer, str) and pointer.startswith("/"), "explicit actual tool exit JSON pointer required")
    node = payload
    try:
        for part in pointer[1:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            node = node[int(part)] if isinstance(node, list) else node[part]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ParentAdmissionError("actual tool exit pointer does not resolve") from error
    return node


def validate_tool_exit(payload, pointer=RAW_TOOL_EXIT_POINTER):
    _require(pointer == RAW_TOOL_EXIT_POINTER,
             "raw tool outer exit must use the fixed /exit_code field")
    code = _pointer(payload, RAW_TOOL_EXIT_POINTER)
    _require(type(code) is int and code == 0, "actual parent train tool exit is not 0")
    return code


def _assert_identity_closure(identity):
    seen = {}
    tables = [identity.get(key) for key in ("files", "new_sources", "reference_files", "weights",
                                           "base_files", "base_checkpoints")]
    tables.append((identity.get("convergence_ancestors") or {}).get("files", []))
    for table in tables:
        _require(isinstance(table, list), "parent frozen identity lacks a required source/input table")
        for entry in table:
            path = Path(entry["path"]).resolve()
            signature = (_integer(entry.get("bytes")), _hex(entry.get("sha256")))
            if str(path) in seen:
                _require(seen[str(path)] == signature, "parent identity has conflicting file pins")
                continue
            seen[str(path)] = signature
            try:
                raw = path.read_bytes()
            except OSError as error:
                raise ParentAdmissionError("parent frozen dependency missing: %s" % path) from error
            _require((len(raw), hashlib.sha256(raw).hexdigest()) == signature,
                     "parent frozen source/input drift: %s" % path)


def checkpoint_pins(layout):
    return require_admitted(layout)["checkpoint_pins"]


def parent_identity_pin(layout):
    return require_admitted(layout)["parent"]["identity_pin"]


def _tool_evidence_root(layout, tool):
    tool_root = Path(C.BASE_ROOT)
    if tool.get("root") is not None:
        relative = Path(str(tool["root"]))
        _require(not relative.is_absolute(), "candidate tool evidence root must be relative to NEWROOT")
        _require(relative.parts == ("execution", "parent_evidence_input"),
                 "candidate tool evidence must use the fixed NEWROOT parent_evidence_input directory")
        tool_root = (Path(layout.root).resolve() / relative).resolve()
        allowed = (Path(layout.root).resolve() / "execution" / "parent_evidence_input").resolve()
        try:
            tool_root.relative_to(allowed)
        except ValueError as error:
            raise ParentAdmissionError("candidate tool evidence root escapes NEWROOT parent_evidence") from error
        _require(tool_root == allowed, "candidate tool evidence root does not match its registered ingress directory")
        tool_pin_path = Path(str((tool.get("file") or {}).get("path", "")))
        _require(tool_pin_path.parts == ("raw_parent_train_tool_result.json",),
                 "candidate tool evidence must pin the fixed raw parent tool result filename")
    return tool_root


def evidence_pin_pairs(layout, document=None):
    document = require_admitted(layout) if document is None else document
    parent = document["parent"]
    pairs = [(C.BASE_ROOT, parent[key]) for key in ("identity_pin", "slot_plan_pin", "preregistration_pin")]
    for name in ("source_identity_pins", "history_files", "telemetry_files", "legacy_evidence"):
        pairs.extend((_source(item["source_id"])["root"], item) for item in document[name])
    for name in ("checkpoint_pins", "ledger_files", "parent_certificates"):
        pairs.extend((C.BASE_ROOT, item) for item in document[name])
    if document.get("cli_started") is not None:
        pairs.append((C.BASE_ROOT, document["cli_started"]))
    tool = document["tool_terminal"]
    tool_root = _tool_evidence_root(layout, tool)
    pairs.extend(((C.BASE_ROOT, document["cli_terminal"]), (tool_root, tool["file"])))
    if document.get("main_model_acceptance") is not None:
        pairs.append((layout.root, document["main_model_acceptance"]))
    return pairs


def evidence_paths(layout, document=None):
    paths = {}
    for root, pin in evidence_pin_pairs(layout, document):
        path = pin_path(root, pin)
        signature = (pin.get("bytes"), pin.get("sha256"))
        if str(path) in paths:
            _require(paths[str(path)][1] == signature, "conflicting evidence path pins")
        else:
            paths[str(path)] = (path, signature)
    return [item[0] for item in paths.values()]


def validate(layout):
    document = require_admitted(layout)  # PENDING returns before any parent/live-output read.
    return _validate_document(layout, document, require_acceptance=True)


def validate_candidate(layout, candidate):
    """Validate complete pinned parent evidence while preserving its unaccepted status.

    This read-only path is for the offline collector. It cannot mark a candidate ADMITTED,
    create an acceptance record, or satisfy any production phase gate.
    """
    _require(isinstance(candidate, dict) and candidate.get("schema") == SCHEMA
             and candidate.get("experiment_id") == C.RUN_ID
             and candidate.get("status") == "CANDIDATE_NOT_ACCEPTED"
             and candidate.get("main_model_acceptance") is None,
             "parent evidence is not an unaccepted collector candidate")
    admitted_shape = dict(candidate)
    admitted_shape["status"] = "ADMITTED"
    summary = _validate_document(layout, admitted_shape, require_acceptance=False)
    summary["status"] = "CANDIDATE_EVIDENCE_COMPLETE"
    return summary


def _validate_document(layout, document, *, require_acceptance):
    validate_source_intervals()
    parent = document["parent"]
    _require(parent.get("episodes") == [600, 1320] and parent.get("worlds") == 4320,
             "parent interval/cap declaration is not stage 1")
    base = Path(C.BASE_ROOT)
    _require(not (base / "execution/ledger/GLOBAL_STOP.json").exists(),
             "direct stage-1 parent has a STOP; normal terminal admission refused")
    identity_path = verify_pin(base, parent["identity_pin"], "protocol/EXPERIMENT_IDENTITY.json")
    plan_path = verify_pin(base, parent["slot_plan_pin"], "protocol/WORLD_SLOT_PLAN.json")
    protocol_path = verify_pin(base, parent["preregistration_pin"], "protocol/STAGE5B_PREREGISTRATION.json")
    identity, plan, protocol = _read_json(identity_path), _read_json(plan_path), _read_json(protocol_path)
    _require(identity.get("experiment_id") == C.BASE_EXPERIMENT_ID
             and protocol.get("experiment_id") == C.BASE_EXPERIMENT_ID,
             "parent identity/protocol belongs to another experiment")
    _require((protocol.get("training") or {}).get("stage_start_episode") == 600
             and (protocol.get("training") or {}).get("stage_end_episode") == 1320,
             "parent protocol interval mismatch")
    _require((identity.get("slot_plan") or {}).get("sha256") == parent["slot_plan_pin"]["sha256"],
             "parent frozen identity does not pin the admitted plan")
    _assert_identity_closure(identity)

    sources = _keyed(document.get("source_identity_pins"), lambda item: item.get("source_id"), "source identities")
    _require(set(sources) == {item["source_id"] for item in C.HISTORY_SOURCES}, "missing/extra source identities")
    for source in C.HISTORY_SOURCES:
        pin = sources[source["source_id"]]
        path = verify_pin(source["root"], pin, "protocol/EXPERIMENT_IDENTITY.json")
        _require(_read_json(path).get("experiment_id") == source["experiment_id"], "history source identity mismatch")
        expected = source["identity_sha256"] or parent["identity_pin"]["sha256"]
        _require(pin["sha256"] == expected, "history source identity hash changed")

    history = _keyed(document.get("history_files"),
                     lambda item: (*_run_key(item), _integer(item.get("episode_index"))), "history rows")
    _require(len(history) == C.HISTORY_ROWS_PINNED, "full history must pin exactly 7920 rows")
    rows = {}
    for arm in C.ARMS_TRAIN:
        for seed in C.TRAIN_SEEDS:
            for index in range(C.STAGE_START_EPISODE):
                key = arm, seed, index
                _require(key in history, "history gap: %r" % (key,))
                pin = history[key]
                source = C.history_source_for_episode(index)
                _require(pin.get("source_id") == source["source_id"], "history row source conflict")
                relative = C.history_row_path(arm, seed, index).relative_to(Path(source["root"]))
                path = verify_pin(source["root"], pin, relative)
                rows[key] = validate_history_row(_read_json(path), source, arm, seed, index)

    checkpoints = _keyed(document.get("checkpoint_pins"), _run_key, "parent checkpoints")
    expected_runs = {(arm, seed) for arm in C.ARMS_TRAIN for seed in C.TRAIN_SEEDS}
    _require(set(checkpoints) == expected_runs, "all six actual terminal checkpoints are required")
    for (arm, seed), pin in checkpoints.items():
        _require(pin.get("episode") == 1320 and pin.get("name") == C.PARENT_CHECKPOINT_NAME,
                 "parent checkpoint selector must match the real milestone_1320 writer")
        verify_pin(base, pin, "execution/checkpoints/%s_s%d/%s" % (arm, seed, C.PARENT_CHECKPOINT_NAME))
        for key in ("parameter_sha256", "actor_sha256", "full_state_sha256", "optimizer_sha256"):
            _hex(pin.get(key))
        rng = pin.get("rng_sha256")
        _require(isinstance(rng, dict) and set(rng) == {"python", "numpy", "torch"},
                 "actual three-RNG fingerprint mapping is required")
        for value in rng.values():
            _hex(value)

    telemetry = _keyed(document.get("telemetry_files"),
                       lambda item: (item.get("source_id"), *_run_key(item)), "history telemetry")
    _require(set(telemetry) == {(source["source_id"], arm, seed) for source in C.HISTORY_SOURCES
                               for arm, seed in expected_runs}, "all 18 completed source telemetry files required")
    for (source_id, arm, seed), pin in telemetry.items():
        source = _source(source_id)
        path = verify_pin(source["root"], pin, "execution/telemetry/%s_s%d.json" % (arm, seed))
        record = _read_json(path)
        _require(record.get("schema") == source["telemetry_schema"] and record.get("arm") == arm
                 and record.get("training_seed") == seed, "telemetry schema/run conflict")
        start, stop = source["episodes"]
        episodes, updates = record.get("episodes"), record.get("updates")
        _require(isinstance(episodes, list) and [item.get("episode_index") for item in episodes]
                 == list(range(start, stop)), "completed telemetry has a missing/duplicate episode")
        _require(isinstance(updates, list) and [item.get("update_index") for item in updates]
                 == list(range(start // 4, stop // 4)), "completed telemetry has a missing/duplicate update")
        for item in episodes:
            row = rows[(arm, seed, item["episode_index"])]
            for field in ("scene", "on_time", "charged_bytes", "generated_goals", "steps"):
                _require(item.get(field) == row[field], "telemetry/world row metric conflict: %s" % field)
            _require(item.get("full_state_sha256_after")
                     == (row.get("state_hashes") or {}).get("full_state_sha256_after"),
                     "telemetry/world row state conflict")
        for item in updates:
            for field in DIAGNOSTICS:
                value = item.get(field)
                _require(type(value) in (int, float) and math.isfinite(value),
                         "missing/non-finite real optimizer diagnostic: %s" % field)
            samples = _integer(item.get("samples"))
            expected_steps = 4 * ((samples + 255) // 256)
            _require(0 < samples <= 4000 and item.get("epochs") == 4
                     and item.get("optimizer_steps") == expected_steps
                     and item.get("minibatches") == expected_steps, "optimizer step-count evidence is inconsistent")
            _require(item.get("episode_indices") == list(range(item["update_index"] * 4,
                                                               item["update_index"] * 4 + 4)),
                     "optimizer update covers the wrong global episodes")
        counts = record.get("counts") or {}
        _require(counts.get("episodes") == stop - start and counts.get("updates") == (stop - start) // 4
                 and counts.get("optimizer_steps") == sum(item["optimizer_steps"] for item in updates),
                 "telemetry counters do not close")
        if source_id == "stage_1":
            cp = checkpoints[(arm, seed)]
            state = record.get("final_full_state") or {}
            _require(record.get("terminal_episode") == 1320
                     and (record.get("convergence") or {}).get("converged") is False
                     and (record.get("convergence") or {}).get("checks") == [],
                     "stage-1 telemetry terminal/convergence mismatch")
            for field, state_field in (("actor_sha256", "actor"), ("full_state_sha256", "sha256"),
                                       ("optimizer_sha256", "optimizer"), ("rng_sha256", "rng")):
                _require(state.get(state_field) == cp[field], "terminal state/checkpoint pin mismatch: %s" % field)
            _require((record.get("final_parameters") or {}).get("parameter_sha256") == cp["parameter_sha256"],
                     "terminal parameter fingerprint mismatch")

    slots = [{"slot": "train:%s:%d:%04d" % (arm, seed, index), "phase": "train", "arm": arm,
              "seed": seed, "episode": index, "scene": C.scene_for(seed, index)}
             for arm in C.ARMS_TRAIN for seed in C.TRAIN_SEEDS for index in range(600, 1320)]
    _require(plan.get("world_cap") == 4320 and plan.get("slots") == slots, "parent plan is not the complete stage-1 plan")
    ledger = _keyed(document.get("ledger_files"),
                    lambda item: (item.get("kind"), *_run_key(item), _integer(item.get("episode_index"))),
                    "parent ledger pins")
    _require(len(ledger) == 8640, "exactly 4320 claims and 4320 result pins required")
    for item in slots:
        key = item["arm"], item["seed"], item["episode"]
        for kind, suffix in (("claim", ".claim"), ("result", ".json")):
            _require((kind, *key) in ledger, "parent ledger gap")
            pin = ledger[(kind, *key)]
            folder = "claims" if kind == "claim" else "results"
            path = verify_pin(base, pin, "execution/ledger/%s/%s%s"
                              % (folder, item["slot"].replace(":", "_"), suffix))
            body = _read_json(path)
            _require(body.get("slot") == item["slot"], "parent ledger slot conflict")
            if kind == "claim":
                _require(body.get("entry") == item, "parent claim entry conflicts with frozen plan")
            else:
                _require(body.get("status") == "ok", "parent result is not a successful terminal")
                result = body.get("payload") or {}
                _require(Path(str(result.get("row", ""))).resolve() == C.history_row_path(*key).resolve()
                         and result.get("on_time") == rows[key]["on_time"], "parent result/world row conflict")
    for folder, suffix in (("claims", ".claim"), ("results", ".json")):
        expected_names = {item["slot"].replace(":", "_") + suffix for item in slots}
        actual_names = {path.name for path in (base / "execution/ledger" / folder).glob("*" + suffix)}
        _require(actual_names == expected_names, "parent ledger has missing/extra slot files")

    certs = _keyed(document.get("parent_certificates"), _run_key, "parent run summaries")
    _require(set(certs) == expected_runs, "six completed parent run summaries required")
    for (arm, seed), pin in certs.items():
        path = verify_pin(base, pin, "execution/convergence/CERTIFICATE_%s_s%d.json" % (arm, seed))
        record = _read_json(path)
        _require(record.get("experiment_id") == C.BASE_EXPERIMENT_ID and record.get("arm") == arm
                 and record.get("seed") == seed and record.get("terminal_episode") == 1320
                 and record.get("status") == "NOT_YET_CONVERGED" and record.get("checks") == []
                 and record.get("consecutive_passes") == 0
                 and record.get("telemetry_sha256") == telemetry[("stage_1", arm, seed)]["sha256"],
                 "parent run summary is incomplete or invents a valid stage-1 convergence check")
    _require(not list((base / "execution/convergence").glob("CHECK_*.json")),
             "stage 1 ends before the first absolute check; unexpected CHECK file refused")

    cli = _read_json(verify_pin(base, document["cli_terminal"]))
    tool = document.get("tool_terminal") or {}
    _require(isinstance(cli.get("receipt_id"), str) and bool(cli["receipt_id"].strip())
             and tool.get("evidence_kind") == "RAW_EXEC_TOOL_RESULT"
             and tool.get("cli_receipt_id") == cli.get("receipt_id")
             and tool.get("exit_code_pointer") == RAW_TOOL_EXIT_POINTER,
             "actual tool/CLI linkage or fixed outer exit field missing")
    validate_parent_terminal(cli, parent["identity_pin"]["sha256"], parent["slot_plan_pin"]["sha256"])
    started_path = verify_pin(base, document.get("cli_started"),
                              "execution/receipts/train_started_%s.json" % cli.get("receipt_id"))
    started = _read_json(started_path)
    _require(started.get("phase") == "train" and started.get("stage") == "started"
             and started.get("receipt_id") == cli.get("receipt_id")
             and started.get("returncode") is None
             and started.get("pid") == cli.get("pid") and started.get("ppid") == cli.get("ppid")
             and started.get("argv") == cli.get("argv")
             and started.get("identity_sha256") == parent["identity_pin"]["sha256"]
             and started.get("slot_plan_sha256") == parent["slot_plan_pin"]["sha256"],
             "parent train started/terminal receipt handoff does not match")
    validate_tool_exit(_read_json(verify_pin(_tool_evidence_root(layout, tool), tool.get("file"))),
                       RAW_TOOL_EXIT_POINTER)
    legacy = document.get("legacy_evidence")
    _require(isinstance(legacy, list) and len(legacy) == 1, "legacy STOP identity pin must remain explicit")
    old = legacy[0]
    _require(old.get("source_id") == "longer_training" and old.get("sha256") == C.LEGACY_LONGER_STOP_SHA256
             and old.get("bytes") == C.LEGACY_LONGER_STOP_BYTES, "legacy STOP identity was changed")
    verify_pin(_source("longer_training")["root"], old, "execution/ledger/GLOBAL_STOP.json")
    if require_acceptance:
        approval = _read_json(verify_pin(layout.root, document.get("main_model_acceptance")))
        _require(approval.get("status") == "ACCEPTED" and approval.get("authority") == "MAIN_MODEL"
                 and approval.get("parent_experiment_id") == C.BASE_EXPERIMENT_ID
                 and approval.get("target_experiment_id") == C.RUN_ID
                 and approval.get("evidence_sha256") == evidence_digest(document),
                 "main-model acceptance does not bind this exact real evidence bundle")
    else:
        _require(document.get("main_model_acceptance") is None,
                 "unaccepted candidate must not contain a main-model acceptance")
    evidence_paths(layout, document)  # validate this exact bundle, including unaccepted candidates
    return {"status": "ADMITTED", "base_identity_sha256": parent["identity_pin"]["sha256"],
            "parent_train_rows": 4320, "history_rows_pinned": 7920, "checkpoints": 6,
            "checkpoint_episode": 1320, "normal_parent_stop_absent": True,
            "prior_recorded_actual_claims": C.LEGACY_RECORDED_ACTUAL_CLAIMS + 4320,
            "evidence_sha256": evidence_digest(document)}


def history_rows(layout, arm, seed, start, stop):
    document = require_admitted(layout)
    pins = _keyed(document.get("history_files"),
                  lambda item: (*_run_key(item), _integer(item.get("episode_index"))), "history rows")
    rows = {}
    for index in range(start, stop):
        source = C.history_source_for_episode(index)
        pin = pins.get((arm, seed, index))
        _require(pin is not None and pin.get("source_id") == source["source_id"], "history row pin missing/conflicting")
        relative = C.history_row_path(arm, seed, index).relative_to(Path(source["root"]))
        path = verify_pin(source["root"], pin, relative)
        rows[index] = validate_history_row(_read_json(path), source, arm, seed, index)
    return rows


def merge_update_histories(groups):
    result = {}
    for group in groups:
        for item in group:
            index = _integer(item.get("update_index"))
            _require(index not in result, "duplicate/conflicting historical optimizer update %d" % index)
            result[index] = item
    return result


def history_updates(layout, arm, seed):
    document = require_admitted(layout)
    pins = _keyed(document.get("telemetry_files"),
                  lambda item: (item.get("source_id"), *_run_key(item)), "history telemetry")
    groups = []
    for source in C.HISTORY_SOURCES:
        pin = pins.get((source["source_id"], arm, seed))
        _require(pin is not None, "missing pinned source telemetry")
        path = verify_pin(source["root"], pin, "execution/telemetry/%s_s%d.json" % (arm, seed))
        groups.append(_read_json(path)["updates"])
    return merge_update_histories(groups)
