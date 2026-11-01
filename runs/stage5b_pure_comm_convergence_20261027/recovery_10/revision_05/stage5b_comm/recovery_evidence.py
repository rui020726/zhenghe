"""Offline recovery candidate collection. No torch/native import, model load or phase.

Run from NEWROOT: python -m stage5b_comm.recovery_evidence --root NEWROOT
  --inventory-handoff execution/recovery_evidence_input/inventory02_handoff.json

The handoff is externally prepared observation data, schema recovery-inventory02-handoff/1:
argv must equal the inventory freeze argv; freeze_sha256/report_sha256 bind that invocation.
raw_tool_result and child_exit are absolute path/bytes/sha256 pins inside NEWROOT/execution.
They name the original raw tool response (root exit_code) and child-terminal receipt
(phase checkpoint_inventory_02, stage child_terminal, root exit_code).
This checks structure and byte bindings, not API signatures or operator/session authenticity.
The operator must independently establish the original terminal-poll/source handoff.
No approval is generated; missing handoff evidence stops before creating an output directory.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from . import constants as C, parent_admission as A, recovery_protocol as R, recovery_seed as S
from .compact import write_bytes_exclusive

CANONICAL_ROOT = Path("D:/code/整合/zhenghe-round42/runs/stage5b_pure_comm_convergence_20261027/recovery_01/revision_01")
SOURCE_ROOT = Path(__file__).resolve().parents[1]
DRAFT_SHA = "be5915d321c69d07158f045866c43205759d315d603bb8e0425673d2358cdb40"
REPORT_SHA = "c23b6828f028a0cdc7e819a9cacdd665902ed259e46d472f89f28cf108235501"
FREEZE_SHA = "7b245a9e11bbf9c05bc1392df13e339c35f52f5b289442e84623c5fde6c72008"
OUTER_TOOL_SHA = "1dc784dbe73bf023c52be9b89d7d7cbe00edecc7ed0ebb410ca8c119b9028d21"
CHILD_EXIT_SHA = "38e73ac2f70b44cd6e082089ad4bccbce3496723bd959d21595370e361f4e8a1"
R1_IDENTITY_SHA = "0717ed4dd25e1a5cbc8212b07cfd8fd2c03d8df2398a18577f0d23154701d5a0"
SOURCE_ALIASES = ("grand200", "parent600", "R1")
LEGACY_PLAN_SCHEMA = "stage5b-world-slot-plan/1"
LEGACY_PLAN_SOURCE_IDS = ("budget58600", "longer_training")
CURSORS = (1320, 1320, 960, 960, 960, 600)
HEX = re.compile(r"[0-9a-fA-F]{64}\Z")


class RecoveryEvidenceError(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise RecoveryEvidenceError(message)


def integer(value, label):
    need(type(value) is int, label + " must be an exact integer")
    return value


def digest(value):
    need(type(value) is str and HEX.fullmatch(value) is not None, "invalid SHA256 text")
    return value.lower()


def finite_tree(value):
    if type(value) is float:
        need(math.isfinite(value), "nonfinite evidence")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def _pairs(pairs):
    body = {}
    for key, value in pairs:
        need(key not in body, "duplicate JSON object key: " + key)
        body[key] = value
    return body


def strict_json(raw):
    body = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_pairs,
                      parse_constant=A._bad_constant)
    finite_tree(body)  # Includes exponent overflow; JSON integers remain Python ints.
    return body


def inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    need(path == root or root in path.parents, "path escapes its registered root: " + str(path))
    return path


def canonical_root(root):
    root = Path(root).resolve()
    need(root == CANONICAL_ROOT.resolve() == SOURCE_ROOT.resolve(),
         "canonical recovery source/root mismatch")
    return root


def pin_for(path, raw):
    return {"path": str(Path(path).resolve()), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


class Reads:
    """One-byte snapshot per file, with a final file-and-directory drift check."""
    def __init__(self):
        self.pins, self.directories = {}, {}

    def raw(self, path, sha=None, size=None):
        path = Path(path).resolve()
        need(path.is_file(), "missing evidence: " + str(path))
        raw = path.read_bytes()
        pin = pin_for(path, raw)
        if sha is not None:
            need(pin["sha256"] == digest(sha), "file SHA mismatch: " + str(path))
        if size is not None:
            need(integer(size, "pin bytes") == len(raw), "file byte mismatch: " + str(path))
        old = self.pins.get(str(path))
        need(old is None or old == pin, "file changed during collection: " + str(path))
        self.pins[str(path)] = pin
        return raw

    def json(self, path, sha=None, size=None):
        return strict_json(self.raw(path, sha, size))

    def pinned(self, pin, allowed_root=None):
        need(type(pin) is dict, "an explicit evidence pin is required")
        path = Path(pin["path"])
        need(path.is_absolute(), "input pin must use an absolute source path")
        if allowed_root is not None:
            inside(allowed_root, path)
        return self.raw(path, pin["sha256"], integer(pin["bytes"], "pin bytes"))

    def listing(self, path):
        path = Path(path).resolve()
        need(path.is_dir(), "missing evidence directory: " + str(path))
        names = tuple(sorted(p.name for p in path.iterdir() if p.is_file()))
        self.directories[path] = names
        return names

    def recheck(self):
        for path, names in self.directories.items():
            need(tuple(sorted(p.name for p in path.iterdir() if p.is_file())) == names,
                 "evidence directory membership changed: " + str(path))
        for pin in list(self.pins.values()):
            self.raw(pin["path"], pin["sha256"], pin["bytes"])


def function_body(text, name):
    node = next((n for n in ast.parse(text).body
                 if isinstance(n, ast.FunctionDef) and n.name == name), None)
    need(node is not None, "missing frozen pure function: " + name)
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[])), ast.get_source_segment(text, node)


def require_function(text, name, template):
    actual, segment = function_body(text, name)
    expected, _ = function_body(template, name)
    need(actual == expected, "unsupported frozen source function/version: " + name)
    return {"function": name, "function_sha256": hashlib.sha256(segment.encode()).hexdigest()}

def require_b0_constant(literals):
    need(type(literals.get("B0_BYTES")) is float and literals["B0_BYTES"] == 10000.0,
         "frozen source science mismatch: B0_BYTES")


SCENE_FUNCTION = """def scene_for(seed, episode_index):
    offset = SEED_OFFSET[int(seed)]
    return TRAIN_SCENES[(offset + int(episode_index)) % len(TRAIN_SCENES)]
"""
ACTION_FUNCTION = """def action_seed(training_seed, episode_index):
    return int(hashlib.sha256(f"sc-action:{int(training_seed)}:{int(episode_index)}".encode(
        "utf-8")).hexdigest()[:16], 16)
"""


def all_identity_pins(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= set(value):
            yield value
        for item in value.values():
            yield from all_identity_pins(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_identity_pins(item)


def validated_plan_experiment_id(spec, identity_experiment_id, plan):
    """Validate the fixed source/schema, then use its verified experiment identity."""
    source_id = spec.get("source_id")
    registered_spec = next((item for item in C.HISTORY_SOURCES
                            if item.get("source_id") == source_id), None)
    need(registered_spec is not None,
         "missing plan identity on a new/unregistered source")
    need(spec == registered_spec, "legacy plan source is not the fixed registered ancestor")
    if source_id in LEGACY_PLAN_SOURCE_IDS:
        expected_schema = LEGACY_PLAN_SCHEMA
        schema_error = "unknown legacy plan schema"
    else:
        need(source_id == "stage_1", "missing plan identity on a new/unregistered source")
        expected_schema = "stage5b-comm-convergence-slot-plan/1"
        schema_error = "unknown convergence plan schema"
    need(plan.get("schema") == expected_schema, schema_error)
    registered = spec.get("experiment_id")
    need(type(identity_experiment_id) is str and identity_experiment_id
         and identity_experiment_id == registered, "source identity/run mismatch")
    if "experiment_id" in plan:
        value = plan["experiment_id"]
        need(type(value) is str and value == identity_experiment_id,
             "frozen source plan/run mismatch")
        return identity_experiment_id, "PLAN_EXPERIMENT_ID"
    need(source_id in LEGACY_PLAN_SOURCE_IDS,
         "missing plan identity for registered convergence source")
    need(type(plan.get("inherited_slots")) is int and plan["inherited_slots"] == 0
         and type(plan.get("recovered_rows")) is int and plan["recovered_rows"] == 0,
         "legacy plan inherited/recovered counts must both be integer zero")
    return identity_experiment_id, "VERIFIED_LEGACY_SOURCE_IDENTITY"

class FrozenSource:
    def __init__(self, spec, identity_sha, reader):
        self.spec, self.root, self.reader = spec, Path(spec["root"]).resolve(), reader
        identity_path = self.root / "protocol/EXPERIMENT_IDENTITY.json"
        identity = reader.json(identity_path, identity_sha)
        need(identity.get("experiment_id") == spec["experiment_id"], "source identity/run mismatch")
        self.identity_pin = reader.pins[str(identity_path.resolve())]
        self.table = {}
        for pin in all_identity_pins(identity):
            path = str(Path(pin["path"]).resolve())
            signature = (integer(pin["bytes"], "source pin bytes"), digest(pin["sha256"]))
            need(path not in self.table or self.table[path][0] == signature, "conflicting frozen pins")
            self.table[path] = (signature, pin)
        self.proofs = {}
        constants = self.code(self.root / "stage5b_comm/constants.py")
        self.literals = {}
        for node in ast.parse(constants).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        try:
                            self.literals[target.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
        for key, expected in {"BUDGET_BYTES": 58600, "FEATURE_DIM": 19, "STEPS": 1000,
                              "NUM_GOALS": 24, "EPISODES_PER_UPDATE": 4}.items():
            need(type(self.literals.get(key)) is int and self.literals[key] == expected,
                 "frozen source science mismatch: " + key)
        require_b0_constant(self.literals)
        need(tuple(self.literals.get("TRAIN_SCENES", ())) == tuple(C.TRAIN_SCENES)
             and self.literals.get("SEED_OFFSET") == C.SEED_OFFSET, "frozen scene rotation mismatch")
        need(self.literals.get("ROW_SCHEMA") == spec["row_schema"],
             "source row schema differs from the registered source")
        need(type(self.literals.get("LEDGER_SCHEMA")) is str
             and bool(self.literals["LEDGER_SCHEMA"]), "source ledger schema missing")
        self.proofs["source_identity"] = dict(self.identity_pin)
        self.proofs["scene"] = require_function(constants, "scene_for", SCENE_FUNCTION)
        action_path = C.RUNS_ROOT / "comm_cost_v2_rerun_20260928/v2_run.py"
        action = self.code(action_path)
        self.proofs["action"] = require_function(action, "action_seed", ACTION_FUNCTION)
        episode = self.code(self.root / "stage5b_comm/episode.py")
        self.episode_tree = ast.parse(episode)
        world = self.code(self.root / "stage5b_comm/world.py")
        for fragment in ('fields["seed"] = int(scene)', "torch.manual_seed(int(action_rng_seed))"):
            need(fragment in world, "unsupported frozen world seed plumbing")
        need(any(isinstance(n, ast.keyword) and n.arg == "scene"
                 and ast.unparse(n.value) == "int(scene)" for n in ast.walk(ast.parse(world))),
             "missing frozen RuntimeSpec scene argument")
        self.code(self.root / "stage5b_comm/runner.py")
        self.plan = reader.json(self.frozen_path(self.root / "protocol/WORLD_SLOT_PLAN.json"))
        self.experiment_id, self.plan_identity_basis = validated_plan_experiment_id(
            spec, identity["experiment_id"], self.plan)
        self.proofs["files"] = [self.reader.pins[path] for path in self.proofs.pop("file_paths")]

    def frozen_path(self, path):
        path = Path(path).resolve()
        need(str(path) in self.table, "source/version not pinned by that source identity: " + str(path))
        (size, sha), _ = self.table[str(path)]
        self.reader.raw(path, sha, size)
        return path

    def code(self, path):
        path = self.frozen_path(path)
        self.proofs.setdefault("file_paths", []).append(str(path))
        return self.reader.raw(path).decode("utf-8-sig")

    def scene(self, seed, episode):
        return self.literals["TRAIN_SCENES"][(self.literals["SEED_OFFSET"][seed] + episode) % 12]

    def action(self, seed, episode, phase):
        name = "run_training_world" if phase == "train" else "run_replay_world"
        fn = next((n for n in self.episode_tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == name), None)
        need(fn is not None, "unverified frozen phase seed caller: " + phase)
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.keyword) and n.arg == "action_rng_seed"]
        need(len(calls) == 1 and ast.unparse(calls[0].value)
             == "v2_run.action_seed(int(seed), int(index))", "unsupported frozen action seed call")
        payload = "sc-action:%d:%d" % (seed, episode)
        return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)

def merge_prefix(groups, end):
    """Ordered half-open [0,end) rows; never deduplicate conflicting or repeated history."""
    merged = []
    for group in groups:
        for row in group:
            need(integer(row.get("episode_index"), "row episode_index") == len(merged),
                 "prefix gap, duplicate, conflict or order violation")
            finite_tree(row)
            merged.append(row)
    need(len(merged) == integer(end, "prefix endpoint"), "incomplete selected prefix")
    return merged


def validate_update(item):
    finite_tree(item)
    index = integer(item.get("update_index"), "update_index")
    need(index >= 0 and type(item.get("episode_indices")) is list
         and all(type(value) is int for value in item["episode_indices"])
         and item["episode_indices"] == list(range(index * 4, index * 4 + 4)),
         "update episode coverage mismatch")
    for name in A.DIAGNOSTICS:
        value = item.get(name)
        need(type(value) in (int, float) and math.isfinite(value),
             "missing/nonfinite real update diagnostic: " + name)
    samples = integer(item.get("samples"), "update samples")
    steps = 4 * ((samples + 255) // 256)
    need(0 < samples <= 4000 and type(item.get("epochs")) is int and item["epochs"] == 4
         and type(item.get("optimizer_steps")) is int and item["optimizer_steps"] == steps
         and type(item.get("minibatches")) is int and item["minibatches"] == steps,
         "update optimizer step closure mismatch")
    return index


def select_updates(records, start, stop, known_gap=None):
    """Only real telemetry updates; known absence is indexed, never filled."""
    need(type(records) is list, "telemetry updates must be a list")
    table = {}
    previous = -1
    for item in records:  # Validate even records inside a known gap or outside selected prefix.
        index = validate_update(item)
        need(index not in table, "duplicate/conflicting telemetry update")
        need(index > previous, "telemetry update order violation")
        previous = index
        table[index] = item
    expected = set(range(start // 4, stop // 4))
    missing = sorted(expected - table.keys())
    allowed = set(range(*known_gap)) if known_gap is not None else set()
    need(set(missing) <= allowed, "unexpected missing update outside registered known gap")
    return [table[index] for index in sorted(expected & table.keys())], missing


def validate_row(row, source, arm, seed, index):
    A.validate_history_row(row, source, arm, seed, index)
    finite_tree(row)
    need(type(row.get("budget_bytes")) is int and row["budget_bytes"] == 58600,
         "world row budget evidence mismatch")
    credit = row.get("budget_credit") or {}
    need(all(type(credit.get(name)) is int and credit[name] == 58600 for name in
             ("context_budget_bytes", "executor_budget_bytes", "protocol_budget_bytes",
              "run_configuration_budget_bytes")), "world row budget authority missing")
    obs = row.get("communication_network_observation") or {}
    need(obs.get("status") == "OBSERVED" and type(obs.get("last_dim")) is int
         and obs["last_dim"] == 19 and type(obs.get("calls")) is int and obs["calls"] > 0,
         "world row lacks actual 19-feature observation")



def runtime_record(authority, entry, direct_records):
    """Extract actual fields, reconstruct only through this source's verified frozen functions."""
    seed, episode = entry["seed"], entry["episode"]
    expected_scene = authority.scene(seed, episode)
    expected_action = authority.action(seed, episode, entry["phase"])
    scenes = [entry["scene"]]
    actions = []
    for record in direct_records:
        if not isinstance(record, dict):
            continue
        scenes += [record[key] for key in ("scene", "scenario_seed") if key in record]
        actions += [record["action_rng_seed"]] if "action_rng_seed" in record else []
    need(all(type(scene) is int and scene == expected_scene for scene in scenes),
         "direct/frozen scene key conflict")
    need(all(type(action) is int and 0 <= action < 2 ** 64 and action == expected_action
             for action in actions), "direct/frozen action RNG conflict")
    return {
        "arm": entry["arm"], "training_seed": seed, "episode_index": episode,
        "scene_seed": expected_scene, "action_rng_seed": expected_action,
        "branch": authority.spec["source_id"],
    }, {"scene_seed": "DIRECT_FROZEN_PLAN_AND_CLAIM",
        "action_rng_seed": "DIRECT" if actions else "SOURCE_RECONSTRUCTION",
        "source_functions": authority.proofs}


def validate_draft(draft, root):
    need(draft.get("schema") == "stage5b-recovery-protocol-draft/1"
         and draft.get("status") == "DRAFT_NOT_FROZEN_NOT_ADMITTED"
         and Path(draft.get("root", "")).resolve() == root, "wrong recovery DRAFT schema/status/root")
    science = draft.get("inherited_frozen_science") or {}
    need(type(science.get("B0")) is int and science["B0"] == 10000,
         "DRAFT inherited B0 science mismatch")
    stage = draft.get("stage") or {}
    runs = stage.get("runs")
    need(type(runs) is list and len(runs) == 6 and stage.get("max_new_worlds_this_root") == 720
         and stage.get("max_new_worlds_per_active_run") == 120, "wrong six-run finite stage")
    for index, run in enumerate(runs):
        arm, seed, cursor = R.RUN_CURSORS[index]
        need(run.get("arm") == arm and type(run.get("training_seed")) is int
             and run["training_seed"] == seed and run.get("active") is True
             and type(run.get("start_episode")) is int and run["start_episode"] == cursor
             and type(run.get("end_episode_cap")) is int and run["end_episode_cap"] == cursor + 120
             and run.get("new_world_cap") == 120, "DRAFT run/cursor/cap mismatch")
        expected = [("grand200", [0, 200]), ("parent600", [200, 600])]
        if cursor > 600:
            expected.append(("R1", [600, cursor]))
        need([(part["source_id"], part["episodes_half_open"]) for part in run["selected_history"]]
             == expected, "DRAFT selected source prefix mismatch")
    counts = draft.get("history_accounting") or {}
    need([counts.get(key) for key in ("selected_prefix_rows", "logical_updates",
                                    "available_update_diagnostics", "missing_update_diagnostics")]
         == [6120, 1530, 1260, 270], "DRAFT production history counts changed")
    draft_gaps = {}
    for gap in draft.get("known_diagnostic_gaps") or []:
        key = (gap.get("arm"), integer(gap.get("training_seed"), "gap training seed"))
        need(key not in draft_gaps, "duplicate DRAFT known gap")
        draft_gaps[key] = {
            "episodes": tuple(integer(value, "gap episode") for value in gap["episodes_half_open"]),
            "updates": tuple(integer(value, "gap update") for value in gap["updates_half_open"]),
        }
    expected_gaps = {key: {"episodes": tuple(value["episodes"]),
                           "updates": tuple(value["updates"])}
                     for key, value in R.KNOWN_GAPS.items()}
    need(draft_gaps == expected_gaps, "DRAFT known gaps differ from registered three runs")
    need((draft.get("archived_r1") or {}).get("status")
         == "INTERRUPTED_ARCHIVE_NOT_COMPLETE_PARENT", "DRAFT R1 archive status changed")
    need((draft.get("rollout_seed_contract") or {}).get("namespace") == R.WORLD_SEED_NAMESPACE,
         "DRAFT fixed seed namespace changed")
    return runs


def inventory_exit_records(probe_dir, handoff, reader):
    probe_dir = Path(probe_dir).resolve()
    pins = {}
    for key, name in (("raw_tool_result", "outer_tool_return_visible.json"),
                       ("child_exit", "child_exit.json")):
        pin = handoff.get(key)
        need(type(pin) is dict and {"path", "bytes", "sha256"} <= set(pin),
             "inventory exit handoff pin is incomplete")
        need(Path(pin.get("path", "")).resolve() == (probe_dir / name).resolve(),
             "inventory exit evidence must use probe_run_01 original files")
        pins[key] = pin
    raw_tool = strict_json(reader.pinned(pins["raw_tool_result"], probe_dir))
    child = strict_json(reader.pinned(pins["child_exit"], probe_dir))
    need(type(raw_tool.get("exit_code")) is int and raw_tool["exit_code"] == 0,
         "inventory raw outer /exit_code is not integer 0")
    need(child.get("phase") == "checkpoint_inventory_02"
         and child.get("stage") == "child_terminal"
         and type(child.get("exit_code")) is int and child["exit_code"] == 0,
         "inventory actual child exit_code is not integer 0")
    return raw_tool, child, pins

def inventory_evidence(root, draft, reader, handoff_path):
    directory = root / "execution/checkpoint_inventory_02"
    declared = draft.get("inventory_evidence") or {}
    report_decl = declared.get("report") or {}
    need(Path(report_decl.get("path", "")).resolve()
         == (directory / "inventory_report.json").resolve()
         and report_decl.get("bytes") == 60421
         and digest(report_decl.get("sha256")) == REPORT_SHA,
         "DRAFT inventory report reference mismatch")
    report = reader.json(directory / "inventory_report.json", REPORT_SHA)
    inputs = reader.json(directory / "expected_inputs.json",
                         draft["inventory_evidence"]["expected_inputs"]["sha256"])
    freeze_path = directory / "execution_freeze_receipt.json"
    freeze = reader.json(freeze_path, FREEZE_SHA)
    need(report.get("schema") == "checkpoint-inventory-report/1"
         and report.get("status") == "COMPLETE" and report.get("torch_load_calls") == 6,
         "inventory02 not complete six-data-load evidence")
    for field in ("model_instances_created", "forward_calls", "worlds_created", "optimizer_steps"):
        need(type(report.get(field)) is int and report[field] == 0, "inventory02 scope conflict")
    need(freeze.get("kind") == "checkpoint_inventory_02_execution_freeze"
         and freeze.get("checkpoints") == inputs.get("inputs"), "inventory freeze/input donor conflict")
    for key, filename in (("probe", "checkpoint_inventory_probe.py"),
                          ("helper", "checkpoint_inventory_digest.py"),
                          ("input", "expected_inputs.json")):
        expected = directory / filename
        pin = freeze.get(key) or {}
        need(Path(pin.get("path", "")).resolve() == expected.resolve(),
             "inventory freeze has unexpected input/source path")
        reader.raw(expected, pin.get("sha256"))
    need(freeze.get("three_positional_paths") == {
        "INPUT": str(directory / "expected_inputs.json"),
        "REPORT": str(directory / "inventory_report.json"),
        "PROGRESS": str(directory / "inventory_progress.json")},
        "inventory freeze positional output binding mismatch")
    handoff_path = inside(root / "execution/recovery_evidence_input", root / handoff_path)
    handoff = reader.json(handoff_path)
    need(handoff.get("schema") == "recovery-inventory02-handoff/1"
         and handoff.get("argv") == freeze.get("argv")
         and digest(handoff.get("freeze_sha256")) == reader.pins[str(freeze_path.resolve())]["sha256"]
         and digest(handoff.get("report_sha256")) == REPORT_SHA,
         "inventory terminal handoff does not bind this freeze/report/invocation")
    raw_tool, child, exit_pins = inventory_exit_records(
        directory / "probe_run_01", handoff, reader)
    need(digest(exit_pins["raw_tool_result"]["sha256"]) == OUTER_TOOL_SHA
         and digest(exit_pins["child_exit"]["sha256"]) == CHILD_EXIT_SHA,
         "inventory exit handoff pins differ from original probe files")
    need(child.get("argv") == freeze.get("argv"),
         "inventory child argv differs from frozen invocation")
    expected = inputs.get("inputs")
    need(type(expected) is list and len(expected) == 6 and len(report.get("items", [])) == 6,
         "six inventory inputs/items required")
    for run, pin, item in zip(draft["stage"]["runs"], expected, report["items"]):
        donor, observed = run["donor"], item.get("observed") or {}
        need(item.get("load_status") == "LOADED" and item.get("metadata_mismatches") == []
             and item.get("label") == pin["label"] and observed.get("schema") == inputs["checkpoint_schema_expected"]
             and observed.get("episode") == pin["episode"] == run["start_episode"]
             and observed.get("arm") == pin["arm"] == run["arm"]
             and observed.get("training_seed") == pin["seed"] == run["training_seed"],
             "inventory donor payload schema/run/cursor mismatch")
        need(Path(item["path"]).resolve() == Path(pin["path"]).resolve()
             == Path(donor["path"]).resolve()
             and pin["bytes"] == donor["bytes"] and digest(pin["sha256"]) == digest(donor["sha256"])
             and digest(observed.get("identity_sha256")) == digest(pin["identity_sha256"])
             == digest(donor["identity_sha256"]), "inventory/DRAFT donor pin-source conflict")
        reader.pinned(pin, C.RUNS_ROOT)
    for name in ("input_precheck", "input_postcheck"):
        checks = report.get(name)
        need(type(checks) is list and len(checks) == 6, "missing inventory before/after hash table")
        for check, pin in zip(checks, expected):
            need(check.get("matches") is True and check.get("label") == pin["label"]
                 and Path(check.get("path", "")).resolve() == Path(pin["path"]).resolve()
                 and check.get("bytes") == pin["bytes"]
                 and digest(check.get("sha256")) == digest(pin["sha256"]),
                 "inventory before/after donor integrity failure")
    return {"status": "DATA_INVENTORY_NOT_G3", "report_sha256": REPORT_SHA,
            "external_ingress_limit": "Operator/session authenticity requires independent original-poll handoff review."}


def collect_source(authority, alias, reader, run_table):
    """Read only train/replay ledger leaves; never open eval/holdout result data."""
    root, spec = authority.root, authority.spec
    slots = authority.plan.get("slots")
    need(type(slots) is list and authority.experiment_id == spec["experiment_id"],
         "frozen source plan/run mismatch")
    all_claims, all_results, relevant = {}, {}, {}
    for slot in slots:
        name = slot["slot"].replace(":", "_")
        need(name + ".claim" not in all_claims, "duplicate frozen source slot")
        all_claims[name + ".claim"], all_results[name + ".json"] = slot, slot
        if slot.get("phase") in ("train", "replay"):
            need(slot.get("arm") in ("T1", "T2") and type(slot.get("seed")) is int
                 and slot["seed"] in C.TRAIN_SEEDS and type(slot.get("episode")) is int,
                 "malformed frozen train/replay slot")
            need(type(slot.get("scene")) is int
                 and slot["scene"] == authority.scene(slot["seed"], slot["episode"]),
                 "frozen source plan scene conflict")
            relevant[name + ".claim"] = slot
    claim_dir, result_dir = root / "execution/ledger/claims", root / "execution/ledger/results"
    claim_names, result_names = reader.listing(claim_dir), reader.listing(result_dir)
    row_names = reader.listing(root / "execution/rows/train")
    reader.listing(root / "execution/telemetry")
    train_plan_names = {name[:-6] + ".json" for name, slot in all_claims.items()
                        if slot.get("phase") == "train"}
    need(set(row_names) <= train_plan_names,
         "world row leaf outside its frozen source plan")
    need(all(name[:-5] + ".claim" in claim_names for name in row_names),
         "world row exists without a matching frozen-plan claim")
    need(set(claim_names) <= all_claims.keys() and set(result_names) <= all_results.keys(),
         "ledger leaf outside its frozen plan; coverage cannot be established")
    relevant_names = sorted(set(claim_names) & relevant.keys())
    need(relevant_names, "empty source train/replay claim coverage")
    need(all(name[:-5] + ".claim" in claim_names for name in result_names
             if all_results[name].get("phase") in ("train", "replay")),
         "train/replay result exists without claim")
    selected, snapshot, excluded = {}, [], []
    for name in relevant_names:
        entry = relevant[name]
        arm, seed, index = entry["arm"], entry["seed"], entry["episode"]
        claim_path, result_path = claim_dir / name, result_dir / (name[:-6] + ".json")
        claim = reader.json(claim_path)
        need(claim.get("schema") == authority.literals["LEDGER_SCHEMA"]
             and claim.get("slot") == entry["slot"] and claim.get("entry") == entry,
             "claim/schema/frozen plan conflict")
        result = reader.json(result_path) if result_path.name in result_names else None
        if result is not None:
            need(result.get("schema") == authority.literals["LEDGER_SCHEMA"]
                 and result.get("slot") == entry["slot"]
                 and result.get("status") in ("ok", "failed", "terminated"),
                 "result/schema/claim conflict")
            need(type(result.get("worker_pid")) is int
                 and result["worker_pid"] == claim.get("worker_pid"), "claim/result worker mismatch")
        row = None
        row_path = root / ("execution/rows/train/train_%s_%d_%0*d.json"
                           % (arm, seed, spec["row_digits"], index))
        if entry["phase"] == "train" and row_path.is_file():
            row = reader.json(row_path)
            if result is not None and result.get("status") == "ok":
                validate_row(row, spec, arm, seed, index)
                need(row.get("slot") == entry["slot"], "world row/claim slot mismatch")
                payload = result.get("payload") or {}
                need(Path(payload.get("row", "")).resolve() == row_path.resolve()
                     and payload.get("on_time") == row["on_time"], "result/world row linkage conflict")
        if entry["phase"] == "train" and result is not None and result.get("status") == "ok":
            need(row is not None, "successful train result lacks its actual world row")
        runtime, derivation = runtime_record(authority, entry,
            (claim.get("extra"), (result or {}).get("payload"), row, entry))
        derivation["source_functions"] = {
            key: value for key, value in authority.proofs.items() if key != "files"}
        record = {"runtime_key": runtime, "field_origins": derivation,
                  "source_id": alias, "source_root": str(root), "claim": claim,
                  "claim_pin": reader.pins[str(claim_path.resolve())],
                  "result": result, "outcome": result.get("status") if result else "UNKNOWN_UNCLOSED",
                  "result_pin": reader.pins.get(str(result_path.resolve())),
                  "row_pin": reader.pins.get(str(row_path.resolve()))}
        snapshot.append(record)  # Replays/duplicate physical keys retain every original record.
        start, end = {"grand200": (0, 200), "parent600": (200, 600),
                      "R1": (600, run_table[(arm, seed)]["start_episode"])}[alias]
        if entry["phase"] == "train" and start <= index < end:
            need(result is not None and result.get("status") == "ok" and row is not None,
                 "selected prefix contains failed/unknown/missing claim-result-row")
            key = arm, seed, index
            need(key not in selected, "duplicate/conflicting selected prefix row")
            selected[key] = row
        elif alias == "R1" and entry["phase"] == "train":
            excluded.append(record)
    coverage = {"source_id": alias, "plan_pin": reader.pins[str((root / "protocol/WORLD_SLOT_PLAN.json").resolve())],
                "claim_count": len(snapshot), "planned_train_replay_slots": len(relevant),
                "actual_relevant_claim_names": relevant_names,
                "coverage_basis": "FROZEN_PLAN_AND_STABLE_COMPLETE_LEDGER_DIRECTORY_MEMBERSHIP"}
    return selected, snapshot, excluded, coverage

def collect_telemetry(authority, arm, seed, start, stop, rows, reader, gap):
    path = authority.root / ("execution/telemetry/%s_s%d.json" % (arm, seed))
    if not path.is_file():
        need(gap == (start // 4, stop // 4), "missing telemetry outside exact registered gap: " + str(path))
        return [], list(range(*gap))
    record = reader.json(path)
    need(record.get("schema") == authority.spec["telemetry_schema"]
         and record.get("arm") == arm and type(record.get("training_seed")) is int
         and record["training_seed"] == seed, "telemetry source schema/run mismatch")
    episodes, updates = record.get("episodes"), record.get("updates")
    need(type(episodes) is list and [item.get("episode_index") for item in episodes]
         == list(range(start, stop)), "telemetry episode prefix/duplicate/order mismatch")
    selected, missing = select_updates(updates, start, stop, gap)
    for item in episodes:
        row = rows[(arm, seed, item["episode_index"])]
        for field in ("scene", "on_time", "charged_bytes", "generated_goals", "steps"):
            need(item.get(field) == row[field], "telemetry/world row conflict: " + field)
        need(digest(item.get("full_state_sha256_after"))
             == digest((row.get("state_hashes") or {}).get("full_state_sha256_after")),
             "telemetry/world state fingerprint conflict")
    counts = record.get("counts") or {}
    need(counts.get("episodes") == len(episodes) and counts.get("updates") == len(updates)
         and counts.get("optimizer_steps") == sum(item["optimizer_steps"] for item in updates),
         "telemetry counters fail actual update-record closure")
    return selected, missing


def check_quarantine(excluded, draft):
    expected = set()
    for group in draft["archived_r1"]["isolated_tail_groups"]:
        arm, seed = group["arm"], group["training_seed"]
        expected.update((arm, seed, index, "ok") for index in
                        range(*group["successful_episodes_half_open"]))
        expected.update((arm, seed, index, "UNKNOWN_UNCLOSED") for index in
                        range(*group["unresolved_claims_half_open"]))
    actual = []
    for item in excluded:
        key = item["runtime_key"]
        actual.append((key["arm"], key["training_seed"], key["episode_index"], item["outcome"]))
    need(len(actual) == len(set(actual)) and set(actual) == expected,
         "R1 excluded tails/orphan claims differ from the registered archive boundary")
    need(sum(item["outcome"] == "ok" for item in excluded) == 36
         and sum(item["outcome"] == "UNKNOWN_UNCLOSED" for item in excluded) == 9,
         "R1 36-tail/9-orphan quarantine count mismatch")


def runtime_physical_key(runtime):
    need(type(runtime) is dict, "runtime key must be a mapping")
    arm, seed = runtime.get("arm"), runtime.get("training_seed")
    need(arm in C.ARMS_TRAIN and type(seed) is int and seed in C.TRAIN_SEEDS,
         "runtime key has an unregistered run")
    scene = runtime.get("scene_seed")
    action = runtime.get("action_rng_seed")
    need(type(scene) is int and scene > 0 and type(action) is int
         and 0 <= action < 2 ** 64, "runtime physical key is malformed")
    return arm, seed, scene, action


def validate_rollout_key_groups(records):
    """Keep every claim; reject physical duplicates except closed exact train/replay pairs."""
    groups = {}
    for record in records:
        key = runtime_physical_key(record.get("runtime_key"))
        groups.setdefault(key, []).append(record)
    for key, group in groups.items():
        phases = []
        for record in group:
            entry = (record.get("claim") or {}).get("entry") or {}
            phase = entry.get("phase")
            need(phase in ("train", "replay") and entry.get("arm") == key[0]
                 and entry.get("seed") == key[1], "claim/runtime phase or run conflict")
            phases.append(phase)
        if len(group) == 1:
            need(phases == ["train"], "isolated replay or unknown duplicate runtime key")
            continue
        need(len(group) == 2 and sorted(phases) == ["replay", "train"],
             "duplicate train-train or unknown runtime key")
        train = group[phases.index("train")]
        replay = group[phases.index("replay")]
        train_entry = train["claim"]["entry"]
        replay_entry = replay["claim"]["entry"]
        need(train_entry.get("episode") == replay_entry.get("episode")
             and str(train_entry.get("slot", "")).replace("train:", "replay:", 1)
             == replay_entry.get("slot"),
             "runtime duplicate is not an exact planned train/replay pair")
        need(train.get("outcome") == "ok" and train.get("result_pin")
             and train.get("row_pin") and replay.get("outcome") == "ok"
             and replay.get("result_pin"),
             "planned train/replay duplicate lacks closed evidence")


def proposed_seed_plan(runs, snapshot):
    """Pure proposals; this creates no slot claims or runtime seed override."""
    result = []
    for run in runs:
        arm, seed = run["arm"], run["training_seed"]
        prior = [item["runtime_key"] for item in snapshot
                 if item["runtime_key"]["arm"] == arm and item["runtime_key"]["training_seed"] == seed]
        need(prior, "missing same-run prior claim coverage; cannot certify seed disjointness")
        cumulative = [dict(item) for item in prior]
        seen = {runtime_physical_key(item) for item in cumulative}
        for episode in range(run["start_episode"], run["end_episode_cap"]):
            scene = C.scene_for(seed, episode)
            action = S.action_rng_seed(arm, seed, episode, scene, cumulative)
            item = {"arm": arm, "training_seed": seed, "episode_index": episode,
                    "scene_seed": scene, "action_rng_seed": action}
            key = runtime_physical_key(item)
            need(key not in seen, "generated recovery key repeats old/generated physical key")
            result.append(item)
            seen.add(key)
            cumulative.append(item)  # Every later key sees old plus already-generated keys.
    return result


def output_parent(root):
    root = canonical_root(root)
    path = inside(root, root / "execution/recovery_candidates")
    for parent in (root / "execution", root / "execution/recovery_candidates"):
        need(not parent.is_symlink(), "candidate output ancestor is a symbolic link")
    return path


def emit_candidate(root, reader, payload, history, snapshot, excluded):
    parent = output_parent(root)  # All canonical checks precede even mkdir.
    reader.recheck()
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / ("candidate_%d_%d" % (time.time_ns(), os.getpid()))
    directory.mkdir(exist_ok=False)
    outputs = []

    def emit(name, body, jsonl=False):
        if jsonl:
            raw = b"".join((json.dumps(item, ensure_ascii=False, allow_nan=False,
                                      sort_keys=True) + "\n").encode("utf-8") for item in body)
        else:
            raw = (json.dumps(body, ensure_ascii=False, allow_nan=False,
                              sort_keys=True, indent=2) + "\n").encode("utf-8")
        path = directory / name
        write_bytes_exclusive(path, raw)
        outputs.append(pin_for(path, raw))

    def chunks(label, items, count=120):
        for start in range(0, len(items), count):
            emit("%s_%05d.jsonl" % (label, start // count), items[start:start + count], True)

    for (arm, seed), series in history.items():
        chunks("history_%s_s%d" % (arm, seed), series["rows"])
        chunks("updates_%s_s%d" % (arm, seed), series["updates"])
    chunks("old_rollouts", snapshot)
    chunks("source_pins", list(reader.pins.values()), 1000)
    emit("quarantined_tail_and_unclosed_claims.json", excluded)
    emit("proposed_runtime_seed_keys.json", payload.pop("proposed_runtime_seed_keys"))
    emit("index.json", {"history": [{"arm": arm, "training_seed": seed,
          "rows": len(series["rows"]), "updates": len(series["updates"]),
          "missing_update_indices": series["missing_updates"]}
          for (arm, seed), series in history.items()],
          "source_coverage": payload["source_coverage"], "outputs": list(outputs)})
    reader.recheck()  # A drift leaves no candidate.json, never an admissible partial bundle.
    payload["output_files"] = outputs
    emit("candidate.json", payload)
    return directory


def collect(root, inventory_handoff):
    root = canonical_root(root)
    reader = Reads()
    draft = reader.json(root / "protocol/RECOVERY_PROTOCOL.json", DRAFT_SHA)
    runs = validate_draft(draft, root)
    inv = inventory_evidence(root, draft, reader, inventory_handoff)
    for key in ("base_protocol", "stride_source", "trainer_source"):
        reader.pinned(draft["inherited_frozen_science"][key], Path(C.BASE_ROOT))
    for filename in ("recovery_evidence.py", "recovery_seed.py", "recovery_protocol.py"):
        reader.raw(root / "stage5b_comm" / filename)
    run_table = {(run["arm"], run["training_seed"]): run for run in runs}
    selected, authorities, snapshot, excluded, coverage = {}, {}, [], [], []
    registered = draft["history_sources"]
    need([item["id"] for item in registered] == list(SOURCE_ALIASES), "unexpected ancestor source roster")
    for alias, spec, entry in zip(SOURCE_ALIASES, C.HISTORY_SOURCES, registered):
        need(Path(entry["root"]).resolve() == Path(spec["root"]).resolve(),
             "DRAFT ancestor root differs from explicitly registered source")
        identity_sha = R1_IDENTITY_SHA if alias == "R1" else spec["identity_sha256"]
        authority = FrozenSource(spec, identity_sha, reader)
        authorities[alias] = authority
        source_rows, records, tail, source_coverage = collect_source(authority, alias, reader, run_table)
        selected[alias] = source_rows
        snapshot.extend(records)
        excluded.extend(tail)
        coverage.append(source_coverage)
    validate_rollout_key_groups(snapshot)
    # All claims remain in snapshot; only exact closed train/replay physical pairs may repeat.
    history, total_updates, total_missing = {}, 0, 0
    for run in runs:
        arm, seed, stop = run["arm"], run["training_seed"], run["start_episode"]
        row_groups, update_groups, missing = [], [], []
        for interval in run["selected_history"]:
            alias = interval["source_id"]
            start, end = interval["episodes_half_open"]
            source_rows = selected[alias]
            keys = [(arm, seed, index) for index in range(start, end)]
            need(all(key in source_rows for key in keys), "selected history claim/result/row gap")
            row_groups.append([source_rows[key] for key in keys])
            gap = (150, 240) if alias == "R1" and (arm, seed) in R.KNOWN_GAPS else None
            updates, source_missing = collect_telemetry(
                authorities[alias], arm, seed, start, end, source_rows, reader, gap)
            update_groups.append(updates)
            missing.extend(source_missing)
        rows = merge_prefix(row_groups, stop)
        updates = list(A.merge_update_histories(update_groups).values())
        expected_missing = list(range(150, 240)) if (arm, seed) in R.KNOWN_GAPS else []
        need(missing == expected_missing, "actual diagnostic gaps differ from exact registered gaps")
        need(len(updates) + len(missing) == stop // 4, "logical update coverage does not close")
        history[(arm, seed)] = {"rows": rows, "updates": updates, "missing_updates": missing}
        total_updates += len(updates)
        total_missing += len(missing)
    need(sum(len(series["rows"]) for series in history.values()) == 6120
         and total_updates == 1260 and total_missing == 270, "production 6120/1260/270 totals failed")
    check_quarantine(excluded, draft)
    proposed = proposed_seed_plan(runs, snapshot)
    need(len(proposed) == 720, "production proposed finite stage does not contain 720 keys")
    unique = {(item["runtime_key"]["arm"], item["runtime_key"]["training_seed"],
               item["runtime_key"]["scene_seed"], item["runtime_key"]["action_rng_seed"])
              for item in snapshot}
    payload = {"schema": "stage5b-recovery-evidence-candidate/1",
               "status": "CANDIDATE_NOT_ADMITTED", "recovery_experiment_id": R.RECOVERY_STAGE_ID,
               "root": str(root), "draft_sha256": DRAFT_SHA,
               "main_model_approval": None, "admission": "NOT_IMPLEMENTED_NOT_ADMITTED",
               "G3": "NOT_PERFORMED", "runtime_seed_wiring": "NOT_IMPLEMENTED",
               "inventory": inv, "source_coverage": coverage,
               "source_versions": {alias: authority.proofs for alias, authority in authorities.items()},
               "selected_rows": 6120, "logical_updates": 1530, "actual_updates": 1260,
               "unknown_unfilled_update_count": 270, "known_gaps": draft["known_diagnostic_gaps"],
               "old_rollout_record_count": len(snapshot), "old_unique_runtime_key_count": len(unique),
               "controlled_replay_records_retained": True,
               "quarantined_successful_rows": 36, "quarantined_unclosed_claims": 9,
               "donors": [run["donor"] for run in runs],
               "proposed_runtime_seed_keys": proposed,
               "limits": ["Candidate-only collection; no approval, identity freeze, phase, model or world.",
                          "Raw tool ingress authenticity requires external original-session handoff review.",
                          "Payload next_update and independent full-state fingerprint remain UNKNOWN_NOT_STORED."]}
    return emit_candidate(root, reader, payload, history, snapshot, excluded)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--inventory-handoff", required=True)
    args = parser.parse_args(argv)
    try:
        directory = collect(args.root, args.inventory_handoff)
    except (RecoveryEvidenceError, A.ParentAdmissionError, OSError, ValueError, KeyError) as error:
        parser.exit(2, "RECOVERY_EVIDENCE_STOP: %s\n" % error)
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
