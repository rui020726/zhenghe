from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = Path(__file__).resolve().parent
R44_DIR = ROOT / "runs" / "comm_dual_response_ablation_20261004"
R43_DIR = ROOT / "runs" / "comm_one_shot_full_trace_ablation_20261004"
ROUND40_ROOT = ROOT.parent / "zhenghe-round40"
ROUND40_DIR = ROUND40_ROOT / "runs" / "comm_first_report_training_ablation_20261003"
R40_COMMON_PATH = ROUND40_DIR / "r40_common.py"
R43_RUN_PATH = ROOT / "runs" / "comm_one_shot_full_trace_ablation_20261004" / "r42_run.py"
R44_RUNTIME_PATH = R44_DIR / "r44_runtime.py"
CANONICAL_ROOT = ROOT.parent / "zhenghe"
TEACHER_PATH = (
    CANONICAL_ROOT / "runs" / "comm_cost_v2_rerun_20260928" / "v2_rules.py"
)
RELAY_SOURCE_PATH = CANONICAL_ROOT / "integrated_sar" / "comm.py"
FEATURE_SOURCE_PATH = CANONICAL_ROOT / "integrated_sar" / "comm_learning.py"
R25_COMMON_PATH = (
    CANONICAL_ROOT / "runs" / "comm_cost_greedy_replication_20260928" / "r25_common.py"
)
R25_RULES_PATH = (
    CANONICAL_ROOT / "runs" / "comm_cost_greedy_replication_20260928" / "r25_rules.py"
)
RAW_MANIFEST_PATH = R44_DIR / "ROUND44_RAW_ARTIFACT_MANIFEST.json"
FROZEN_R44_PATH = R44_DIR / "ROUND44_FROZEN_CONFIG.json"
FROZEN_R43_PATH = R43_DIR / "ROUND43_FROZEN_CONFIG.json"
ROUND44_REUSE_PATH = R44_DIR / "ROUND44_BASELINE_REUSE.json"
ROUND44_STATUS_PATH = R44_DIR / "ROUND44_STATUS_COMPLETE_EVAL_R1.json"

SEEDS = (11, 22, 33)
FIT_RANGE = (1300000, 1300031)
VALIDATION_RANGE = (1300032, 1300047)
FORBIDDEN_SCENES = set(range(1301000, 1301016))
TEACHER_ARM = "late_discovery_gate"
TEACHER_CUT = 147.0
TEACHER_SHA256 = "a7ce36de2d465cadaba90bd70b36539b5e865a73be8c4e075b9faf5a86c53a57"
TAPE_NAME = re.compile(
    r"^logs/v2_r44_train_s(?P<seed>\d+)_e(?P<episode>\d+)_"
    r"s(?P<scene>\d+)_a(?P<attempt>\d+)_tape\.jsonl$"
)
DATASET_PATH = RUN_DIR / "BC_DATASET.jsonl"
AUDIT_PATH = RUN_DIR / "BC_DATA_AUDIT_R2.json"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_new_or_match(path: Path, data: bytes) -> None:
    if path.exists():
        current = path.read_bytes()
        if current != data:
            raise RuntimeError(f"existing immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_teacher(frozen_r44: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    identity = frozen_r44["execution_identity"]
    expected_root = Path(identity["canonical_read_only_input_root"]).resolve()
    expected_head = str(identity["canonical_read_only_input_head"])
    if expected_root != CANONICAL_ROOT.resolve():
        raise RuntimeError(f"canonical input root mismatch: {expected_root}")
    actual_head = _git_head(CANONICAL_ROOT)
    if actual_head != expected_head:
        raise RuntimeError(
            f"canonical input HEAD changed: expected {expected_head}, got {actual_head}"
        )
    expected_hash = identity["inherited_dependency_sha256"]["v2_rules.py"]
    if expected_hash != TEACHER_SHA256:
        raise RuntimeError(f"R44 teacher hash record changed: {expected_hash}")
    actual_hash = sha256_file(TEACHER_PATH)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"canonical teacher file hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    relay_expected = identity["inherited_dependency_sha256"]["integrated_sar/comm.py"]
    relay_hash = sha256_file(RELAY_SOURCE_PATH)
    if relay_hash != relay_expected:
        raise RuntimeError(
            f"canonical relay source changed: expected {relay_expected}, got {relay_hash}"
        )
    feature_expected = identity["inherited_dependency_sha256"]["integrated_sar/comm_learning.py"]
    feature_hash = sha256_file(FEATURE_SOURCE_PATH)
    if feature_hash != feature_expected:
        raise RuntimeError(
            f"canonical feature source changed: expected {feature_expected}, got {feature_hash}"
        )
    relay_text = RELAY_SOURCE_PATH.read_text(encoding="utf-8")
    for fragment in (
        "self._tick += 1",
        "ready_step = self._tick + 1",
        "age_steps=max(0, self._tick - int(ready_step))",
        "step=self._tick",
    ):
        if fragment not in relay_text:
            raise RuntimeError(f"relay tick/ready-step source evidence is missing: {fragment}")
    feature_text = FEATURE_SOURCE_PATH.read_text(encoding="utf-8")
    if "float(item.age_steps)" not in feature_text or "float(observation.step)" not in feature_text:
        raise RuntimeError("19D feature source no longer exposes raw age and relay step")
    r25_common_text = R25_COMMON_PATH.read_text(encoding="utf-8")
    r25_rules_text = R25_RULES_PATH.read_text(encoding="utf-8")
    if ('"late_discovery_gate_147"' not in r25_common_text
            or '"cut_steps": 147.0' not in r25_common_text
            or "LateDiscoveryGatePolicy(cut_steps=cut)" not in r25_rules_text
            or "strict " not in r25_rules_text):
        raise RuntimeError("ROUND25 registered late teacher lineage no longer matches")

    module_dir = str(TEACHER_PATH.parent)
    canonical_root = str(CANONICAL_ROOT)
    sys.path[:] = [
        module_dir,
        canonical_root,
        *[entry for entry in sys.path if entry not in {module_dir, canonical_root}],
    ]
    rules = importlib.import_module("v2_rules")
    actual_path = Path(rules.__file__).resolve()
    if actual_path != TEACHER_PATH.resolve():
        raise RuntimeError(f"teacher import path mismatch: {actual_path}")
    policy = rules.build_rule(TEACHER_ARM, 1300000)
    if policy.parameters() != {"cut_steps": TEACHER_CUT}:
        raise RuntimeError(f"registered rule parameters mismatch: {policy.parameters()}")

    provenance = {
        "teacher_arm_in_round42_rows": "late_discovery_gate",
        "registered_alias": "late_discovery_gate_147",
        "policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "policy_parameters": policy.parameters(),
        "rule_source_path": str(actual_path),
        "rule_source_sha256": actual_hash,
        "r44_frozen_expected_sha256": expected_hash,
        "relay_source_path": str(RELAY_SOURCE_PATH),
        "relay_source_sha256": relay_hash,
        "feature_source_path": str(FEATURE_SOURCE_PATH),
        "feature_source_sha256": feature_hash,
        "round25_arm_registry_path": str(R25_COMMON_PATH),
        "round25_arm_registry_sha256": sha256_file(R25_COMMON_PATH),
        "round25_rule_factory_path": str(R25_RULES_PATH),
        "round25_rule_factory_sha256": sha256_file(R25_RULES_PATH),
        "round42_late_rule_row_identity": {
            "arm": "late_discovery_gate",
            "policy": "late_discovery_gate",
            "params": {"cut_steps": TEACHER_CUT},
        },
        "canonical_input_root": str(expected_root),
        "canonical_input_head": actual_head,
        "decision_fields_read_by_policy": [
            "observation.items",
            "item.target_id",
            "item.first_emission",
            "item.ready_step",
        ],
        "registered_predicate": "item.first_emission and float(item.ready_step) > 147.0",
        "time_basis": {
            "relay_observation_step_source": "CommRelay._observation returns step=self._tick",
            "age_source": "SendItem.age_steps=max(0, self._tick-int(ready_step))",
            "first_emission_ready_step_bound": (
                "event booking uses self._tick+1 before the next tick; observation increments "
                "the relay tick before exposing the candidate"
            ),
            "tape_step_source": "ROUND42 OneShotCollector records int(observation.step)",
            "reconstruction": "ready_step = tape relay tick - exact integer float32(age_steps)",
            "clock_now_used_for_label": False,
            "non_teacher_snapshot_fields": (
                "missing/not reconstructed; the policy is given only target_id, "
                "first_emission, ready_step"
            ),
        },
    }
    return policy, provenance


def _identity_from_manifest_entry(entry: dict[str, Any]) -> tuple[int, int, int, int]:
    match = TAPE_NAME.fullmatch(str(entry["relative_path"]).replace("\\", "/"))
    if match is None:
        raise ValueError(f"unexpected R44 tape path: {entry['relative_path']}")
    seed, episode, scene, attempt = (
        int(match.group(name)) for name in ("seed", "episode", "scene", "attempt")
    )
    if seed not in SEEDS or not 0 <= episode < 128 or attempt != 1:
        raise ValueError(f"unexpected training tape identity: {entry['relative_path']}")
    if scene not in range(1300000, 1300048) or scene in FORBIDDEN_SCENES:
        raise ValueError(f"out-of-scope training scene: {scene}")
    return seed, episode, scene, attempt


def _teacher_items(record: dict[str, Any]) -> tuple[SimpleNamespace, ...]:
    step = int(record["step"])
    items: list[SimpleNamespace] = []
    for target, raw_feature in zip(record["targets"], record["features"], strict=True):
        vector = np.asarray(raw_feature, dtype=np.float32)
        if vector.shape != (19,) or not np.isfinite(vector).all():
            raise ValueError(f"malformed 19D float32 feature row at step {step}: {target}")
        if vector[5] not in (np.float32(0.0), np.float32(1.0)):
            raise ValueError(f"first_emission feature is not boolean float32: {target}")
        first_emission = bool(vector[5] == np.float32(1.0))
        if first_emission:
            raw_age = float(vector[0])
            if not math.isfinite(raw_age) or raw_age < 0 or raw_age != int(raw_age):
                raise ValueError(f"first-emission age is not an exact nonnegative integer: {target}")
            ready_step = step - int(raw_age)
            if ready_step < 0 or ready_step > step:
                raise ValueError(f"cannot recover ready_step from legal age/time fields: {target}")
        else:
            # The production rule short-circuits on first_emission and never reads this field.
            ready_step = 0
        items.append(
            SimpleNamespace(
                target_id=str(target),
                first_emission=first_emission,
                ready_step=ready_step,
            )
        )
    return tuple(items)


def _feature(record: dict[str, Any], index: int, step: int, target: str) -> tuple[list[float], bytes]:
    vector = np.asarray(record["features"][index], dtype=np.float32)
    if vector.shape != (19,) or not np.isfinite(vector).all():
        raise ValueError(f"feature row must be finite float32[19]: {target} at step {step}")
    if vector[5] != np.float32(1.0):
        raise ValueError(f"free first-report sample is not first_emission: {target} at {step}")
    age = float(vector[0])
    if age < 0 or age != int(age):
        raise ValueError(f"age_steps is not an exact integer: {target} at {step}")
    encoded_step = float(np.float32(step / 1000.0))
    if vector[18] != np.float32(encoded_step):
        raise ValueError(f"feature step does not match tape relay step: {target} at {step}")
    little_endian = vector.astype("<f4", copy=False)
    return [float(value) for value in little_endian], little_endian.tobytes(order="C")


def _validate_record(record: dict[str, Any], relative_path: str, line_number: int) -> None:
    if record.get("schema") != "round42-one-shot-send-skip-tape/1":
        raise ValueError(f"unexpected tape schema in {relative_path}:{line_number}")
    if record.get("comm_action_semantics") != "one-shot-send-skip/1":
        raise ValueError(f"unexpected action semantics in {relative_path}:{line_number}")
    if record.get("comm_action_eligibility") != "first-report-only/1":
        raise ValueError(f"unexpected eligibility in {relative_path}:{line_number}")
    if record.get("base_action_mask_schema") != "budget-sequential-mask/1":
        raise ValueError(f"unexpected mask schema in {relative_path}:{line_number}")

    targets = record.get("targets")
    features = record.get("features")
    if not isinstance(targets, list) or not isinstance(features, list) or len(targets) != len(features):
        raise ValueError(f"target/feature rows do not align in {relative_path}:{line_number}")
    if len(set(targets)) != len(targets):
        raise ValueError(f"duplicate target in candidate order at {relative_path}:{line_number}")
    for index, raw in enumerate(features):
        vector = np.asarray(raw, dtype=np.float32)
        if vector.shape != (19,) or not np.isfinite(vector).all():
            raise ValueError(f"non-finite or non-19D input at {relative_path}:{line_number}")
        if vector[5] not in (np.float32(0.0), np.float32(1.0)):
            raise ValueError(f"first_emission column is not boolean at {relative_path}:{line_number}")
        target = targets[index]
        audit = record.get("candidate_audit", {}).get(target)
        if isinstance(audit, dict) and "first_emission" in audit:
            if bool(vector[5] == np.float32(1.0)) != bool(audit["first_emission"]):
                raise ValueError(
                    f"float32 first_emission disagrees with audit at "
                    f"{relative_path}:{line_number}:{target}"
                )

    decision_class = record.get("decision_class")
    candidate_audit = record.get("candidate_audit")
    if not isinstance(decision_class, dict) or not isinstance(candidate_audit, dict):
        raise ValueError(f"decision/audit map missing at {relative_path}:{line_number}")
    free_by_class = [target for target in targets
                     if decision_class.get(target) == "free_one_shot_choice"]
    if free_by_class != record.get("free_one_shot_choices"):
        raise ValueError(f"free action order differs from candidate order at {relative_path}:{line_number}")
    if free_by_class != record.get("free") or free_by_class != record.get("actionable"):
        raise ValueError(f"free/actionable aliases disagree at {relative_path}:{line_number}")
    actions = record.get("actions")
    if not isinstance(actions, dict):
        raise ValueError(f"action map missing at {relative_path}:{line_number}")
    for target in free_by_class:
        if actions.get(target) not in (0, 1):
            raise ValueError(f"free action is not one of the two policy actions: {target}")
        audit = candidate_audit.get(target)
        if not isinstance(audit, dict):
            raise ValueError(f"free candidate audit missing: {target}")
        if (audit.get("one_shot_state_before") != "undecided"
                or audit.get("first_emission") is not True
                or audit.get("sequentially_affordable_before_action") is not True
                or audit.get("effective_block_reason") != "allowed"
                or audit.get("relay_initial_executable") is not True
                or audit.get("relay_initial_block_reason") != "allowed"):
            raise ValueError(f"free opportunity failed pre-action legality audit: {target}")
        if int(audit.get("attempt_charge_bytes", 0)) <= 0:
            raise ValueError(f"free opportunity lacks a positive transport quote: {target}")
        index = targets.index(target)
        if np.asarray(features[index], dtype=np.float32)[5] != np.float32(1.0):
            raise ValueError(f"free row feature and audit disagree on first_emission: {target}")

    for target in record.get("masked", []):
        audit = candidate_audit.get(target)
        if decision_class.get(target) != "existing_budget_or_capacity_mask" \
                or not isinstance(audit, dict):
            raise ValueError(f"masked row classification/audit mismatch: {target}")
        if (audit.get("one_shot_state_before") != "undecided"
                or audit.get("first_emission") is not True
                or audit.get("sequentially_affordable_before_action") is not False):
            raise ValueError(f"masked row incorrectly appears to be a free choice: {target}")


def build_dataset() -> dict[str, Any]:
    for required in (RAW_MANIFEST_PATH, FROZEN_R44_PATH, FROZEN_R43_PATH,
                     ROUND44_REUSE_PATH, ROUND44_STATUS_PATH, TEACHER_PATH,
                     RELAY_SOURCE_PATH, FEATURE_SOURCE_PATH, R25_COMMON_PATH, R25_RULES_PATH,
                     R40_COMMON_PATH, R43_RUN_PATH, R44_RUNTIME_PATH):
        if not required.is_file():
            raise FileNotFoundError(required)

    frozen_r44 = read_json(FROZEN_R44_PATH)
    frozen_r43 = read_json(FROZEN_R43_PATH)
    r44_manifest = read_json(RAW_MANIFEST_PATH)
    reuse = read_json(ROUND44_REUSE_PATH)
    r44_status = read_json(ROUND44_STATUS_PATH)
    if r44_status.get("status") != "complete_local_delivery":
        raise RuntimeError(f"R44 is not recorded complete: {r44_status.get('status')}")
    if not reuse.get("verification", {}).get("r43_reuse_audit") == "passed":
        raise RuntimeError("ROUND44 reusable baseline audit is not passed")
    if r44_status.get("execution_identity_bound_manifest_sha256") != (
            "b3bb6f7c2488b2acb03f467c799b6eb74554795cf9041eb30529031774a3ebeb"):
        raise RuntimeError("R44 immutable input manifest binding changed")

    policy, teacher_provenance = load_teacher(frozen_r44)
    identity = frozen_r44["execution_identity"]
    r40_expected = identity["round40_code_sha256"]["r40_common.py"]
    r40_hash = sha256_file(R40_COMMON_PATH)
    if r40_hash != r40_expected:
        raise RuntimeError(f"R44 frozen r40_common changed: {r40_hash}")
    r43_expected = identity["source_code_sha256"]["r42_run.py"]
    r43_hash = sha256_file(R43_RUN_PATH)
    if r43_hash != r43_expected:
        raise RuntimeError(f"R44 frozen inherited r42_run changed: {r43_hash}")
    teacher_provenance["runtime_import_chain"] = {
        "r44_runtime_path": str(R44_RUNTIME_PATH),
        "r44_runtime_sha256": sha256_file(R44_RUNTIME_PATH),
        "r44_sys_path_prefix": [
            str(R44_DIR),
            str(R43_DIR),
            str(ROUND40_DIR),
            str(CANONICAL_ROOT),
        ],
        "r43_run_path": str(R43_RUN_PATH),
        "r43_run_sha256": r43_hash,
        "r40_common_path": str(R40_COMMON_PATH),
        "r40_common_sha256": r40_hash,
        "r43_import_assertion": (
            "r42_run._assert_module_path(v2_rules, "
            "INPUT_ROOT/runs/comm_cost_v2_rerun_20260928/v2_rules.py)"
        ),
        "interpretation": (
            "R44 completed through the frozen import-path assertion; the selected source is "
            "the canonical readonly file, not the same-name ROUND42 checkout copy"
        ),
    }

    late_rows = [
        row for row in reuse["round42_fixed_rules"]
        if row.get("arm") == "late_discovery_gate"
    ]
    if len(late_rows) != 16:
        raise RuntimeError(f"expected 16 registered late-rule baseline rows, found {len(late_rows)}")
    verified_rule_rows: list[dict[str, Any]] = []
    for entry in late_rows:
        row_path = Path(str(entry["path"]))
        if not row_path.is_file() \
                or row_path.stat().st_size != int(entry["size_bytes"]) \
                or sha256_file(row_path) != str(entry["sha256"]):
            raise RuntimeError(f"reused late-rule row identity mismatch: {entry['path']}")
        row = read_json(row_path)
        rule_identity = row.get("rule_identity", {})
        if (rule_identity.get("arm") != TEACHER_ARM
                or rule_identity.get("policy") != TEACHER_ARM
                or rule_identity.get("params") != {"cut_steps": TEACHER_CUT}
                or rule_identity.get("built_parameters") != {"cut_steps": TEACHER_CUT}):
            raise RuntimeError(f"late-rule row has different registered policy identity: {row_path}")
        verified_rule_rows.append({
            "path": str(row_path),
            "size_bytes": int(entry["size_bytes"]),
            "sha256": str(entry["sha256"]),
            "scene": int(entry["scene"]),
            "arm": str(entry["arm"]),
            "rule_identity": rule_identity,
        })
    teacher_provenance["reused_late_rule_rows"] = verified_rule_rows
    raw_manifest_sha = sha256_file(RAW_MANIFEST_PATH)

    tape_entries = [
        item for item in r44_manifest["files"]
        if str(item.get("relative_path", "")).replace("\\", "/").startswith(
            "logs/v2_r44_train_s"
        )
        and str(item.get("relative_path", "")).replace("\\", "/").endswith(
            "_tape.jsonl"
        )
    ]
    identities: dict[tuple[int, int], tuple[int, str]] = {}
    for item in tape_entries:
        seed, episode, scene, _ = _identity_from_manifest_entry(item)
        key = (seed, episode)
        if key in identities:
            raise RuntimeError(f"duplicate R44 training episode tape: {key}")
        identities[key] = (scene, str(item["path"]))
    expected_keys = {(seed, episode) for seed in SEEDS for episode in range(128)}
    if set(identities) != expected_keys:
        raise RuntimeError(
            f"R44 training tape matrix mismatch: missing={sorted(expected_keys-set(identities))[:8]}, "
            f"extra={sorted(set(identities)-expected_keys)[:8]}"
        )

    order = [int(scene) for scene in frozen_r43["round42"]["training_order"]]
    if len(order) != 128 or not order:
        raise RuntimeError("ROUND43 inherited training order is malformed")
    order_hash = hashlib.sha256(
        json.dumps(order).encode("utf-8")
    ).hexdigest()
    if order_hash != frozen_r43["round42"]["training_order_sha256"]:
        raise RuntimeError("ROUND43 inherited training-order hash mismatch")
    for seed in SEEDS:
        observed = [identities[(seed, episode)][0] for episode in range(128)]
        if observed != order:
            raise RuntimeError(f"R44 tape order differs from frozen ROUND42 order for seed {seed}")

    rows: list[dict[str, Any]] = []
    file_checks: list[dict[str, Any]] = []
    tape_line_counts: Counter[int] = Counter()
    masked_predecessors: Counter[int] = Counter()
    total_candidate_rows = 0
    free_rows = 0

    for seed, episode in sorted(identities):
        scene, path_text = identities[(seed, episode)]
        path = Path(path_text)
        manifest_item = next(
            item for item in tape_entries
            if _identity_from_manifest_entry(item)[:2] == (seed, episode)
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        size = 0
        seen_free: set[str] = set()
        closed: dict[str, str] = {}
        sample_number = 0
        tape_record_count = 0
        with path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                digest.update(raw_line)
                size += len(raw_line)
                if not raw_line.strip():
                    continue
                tape_record_count += 1
                record = json.loads(raw_line.decode("utf-8"))
                _validate_record(record, str(manifest_item["relative_path"]), line_number)
                tape_line_counts[seed] += 1
                targets = record["targets"]
                total_candidate_rows += len(targets)
                step = int(record["step"])
                if not 1 <= step <= 1000:
                    raise ValueError(f"out-of-range relay step {step} in {path}")
                teacher_items = _teacher_items(record)
                decisions = tuple(policy.decide(SimpleNamespace(items=teacher_items)))
                if len(set(decisions)) != len(decisions) or any(t not in targets for t in decisions):
                    raise ValueError(f"teacher returned invalid candidate sequence at {path}:{line_number}")

                for candidate in targets:
                    klass = record["decision_class"].get(candidate)
                    audit = record["candidate_audit"].get(candidate, {})
                    if klass == "existing_budget_or_capacity_mask":
                        if candidate in seen_free or candidate in closed:
                            raise ValueError(f"mask occurred after one-shot closure: {candidate}")
                        masked_predecessors[seed] += 1
                    elif klass == "fixed_hold_one_shot_closed":
                        state = str(audit.get("one_shot_state_before", ""))
                        if state not in {"send_now_chosen", "skip_episode_chosen"}:
                            raise ValueError(f"closed HOLD lacks an earlier one-shot choice: {candidate}")
                        if candidate not in seen_free:
                            raise ValueError(f"closed HOLD has no earlier free choice: {candidate}")
                        closed[candidate] = state
                    elif klass == "free_one_shot_choice":
                        if candidate in seen_free or candidate in closed:
                            raise ValueError(f"more than one free choice for target {candidate}")
                        seen_free.add(candidate)
                        free_rows += 1
                        index = targets.index(candidate)
                        values, encoded = _feature(record, index, step, candidate)
                        label = int(candidate in decisions)
                        sample_number += 1
                        feature_hash = sha256_bytes(encoded)
                        rows.append({
                            "split": (
                                "fit" if FIT_RANGE[0] <= scene <= FIT_RANGE[1]
                                else "validation"
                            ),
                            "seed": seed,
                            "episode_index": episode,
                            "scene_id": scene,
                            "target_id": str(candidate),
                            "episode_tape_relative_path": str(
                                manifest_item["relative_path"]
                            ),
                            "tape_line": line_number,
                            "candidate_index": index,
                            "step": step,
                            "ready_step": step - int(values[0]),
                            "age_steps": int(values[0]),
                            "candidate_order": list(targets),
                            "both_policy_actions_active": True,
                            "sequentially_affordable": True,
                            "attempt_charge_bytes": int(
                                record["candidate_audit"][candidate]["attempt_charge_bytes"]
                            ),
                            "teacher_arm": TEACHER_ARM,
                            "teacher_label": label,
                            "teacher_action": "SEND_NOW" if label else "SKIP_EPISODE",
                            "teacher_decision_order": list(decisions),
                            "feature_float32": values,
                            "feature_float32_sha256": feature_hash,
                        })
                for candidate in record.get("free_one_shot_choices", []):
                    state_after = record["candidate_audit"][candidate].get(
                        "one_shot_state_after"
                    )
                    expected_state = (
                        "send_now_chosen" if record["actions"][candidate] == 1
                        else "skip_episode_chosen"
                    )
                    if state_after != expected_state:
                        raise ValueError(f"one-shot latch post-action mismatch: {candidate}")

        observed_hash = digest.hexdigest()
        if size != int(manifest_item["size_bytes"]) \
                or observed_hash != str(manifest_item["sha256"]):
            raise RuntimeError(
                f"R44 tape identity mismatch {manifest_item['relative_path']}: "
                f"size={size}/{manifest_item['size_bytes']} "
                f"sha256={observed_hash}/{manifest_item['sha256']}"
            )
        file_checks.append({
            "relative_path": str(manifest_item["relative_path"]),
            "size_bytes": size,
            "sha256": observed_hash,
            "manifest_match": True,
            "seed": seed,
            "episode_index": episode,
            "scene_id": scene,
            "tape_records": tape_record_count,
            "selected_free_samples": sample_number,
        })

    if len(file_checks) != 384:
        raise RuntimeError(f"expected 384 R44 training tapes, found {len(file_checks)}")
    if any(scene in FORBIDDEN_SCENES for scene, _ in (value for value in identities.values())):
        raise RuntimeError("development scene leaked into BC sources")

    conflicts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_counts: dict[str, Counter[int]] = {
        split: Counter() for split in ("fit", "validation")
    }
    by_seed_split: dict[str, Counter[int]] = defaultdict(Counter)
    by_scene_split: dict[str, Counter[int]] = defaultdict(Counter)
    feature_rows: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        split = str(row["split"])
        label = int(row["teacher_label"])
        label_counts[split][label] += 1
        by_seed_split[f"{split}/s{row['seed']}"][label] += 1
        by_scene_split[f"{split}/{row['scene_id']}"][label] += 1
        feature_rows[row["feature_float32_sha256"]].append((index, row))
    for feature_hash, occurrences in feature_rows.items():
        labels = {int(row["teacher_label"]) for _, row in occurrences}
        if len(labels) > 1:
            conflicts[feature_hash] = [
                {
                    "label": int(row["teacher_label"]),
                    "split": row["split"],
                    "seed": row["seed"],
                    "episode_index": row["episode_index"],
                    "scene_id": row["scene_id"],
                    "target_id": row["target_id"],
                    "tape_line": row["tape_line"],
                    "feature_float32": row["feature_float32"],
                }
                for _, row in occurrences
            ]
    if conflicts:
        audit = {
            "kind": "round45-bc-data-audit/1",
            "status": "blocked_float32_feature_label_conflict",
            "conflicts": conflicts,
            "raw_manifest_sha256": raw_manifest_sha,
            "teacher": teacher_provenance,
        }
        write_new_or_match(
            AUDIT_PATH,
            (json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
        raise RuntimeError(f"conflicting teacher labels for {len(conflicts)} float32 vectors")
    if label_counts["fit"][0] == 0 or label_counts["fit"][1] == 0:
        raise RuntimeError(f"fit split lacks a teacher class: {label_counts['fit']}")
    if not rows:
        raise RuntimeError("no free one-shot teacher samples were found")

    feature_summary: dict[str, Any] = {}
    for split in ("fit", "validation"):
        subset = [row for row in rows if row["split"] == split]
        unique = len({row["feature_float32_sha256"] for row in subset})
        feature_summary[split] = {
            "samples": len(subset),
            "unique_scenes": len({row["scene_id"] for row in subset}),
            "unique_float32_features": unique,
            "repeated_feature_samples": len(subset) - unique,
            "feature_hashes_with_repeated_sources": sum(
                sum(1 for _, row in occurrences if row["split"] == split) > 1
                for occurrences in feature_rows.values()
            ),
            "feature_hashes_present_in_fit_and_validation": sum(
                any(row["split"] == "fit" for _, row in occurrences)
                and any(row["split"] == "validation" for _, row in occurrences)
                for occurrences in feature_rows.values()
            ),
        }

    audit = {
        "kind": "round45-bc-data-audit/1",
        "status": "passed",
        "supersedes": {
            "path": "BC_DATA_AUDIT.json",
            "reason": (
                "R1 stored selected free sample count under tape_records; R2 separates "
                "tape_records from selected_free_samples. Dataset rows and labels are unchanged."
            ),
        },
        "input_scope": "R44 train tapes only; no development tapes, outcomes, or post-action metadata",
        "input_artifact_manifest": {
            "path": str(RAW_MANIFEST_PATH),
            "sha256": raw_manifest_sha,
            "files_declared": int(r44_manifest["raw_files_count"]),
            "training_tape_files_verified": len(file_checks),
            "training_tapes": file_checks,
        },
        "frozen_training_order": {
            "source_path": str(FROZEN_R43_PATH),
            "source_sha256": sha256_file(FROZEN_R43_PATH),
            "length": len(order),
            "sha256": order_hash,
            "per_seed_tape_order_matches": True,
        },
        "teacher": teacher_provenance,
        "selection": {
            "unit": "first free one-shot choice per (seed, episode_index, stable_target_id)",
            "source_row_order_preserved": True,
            "free_choice_rows": free_rows,
            "all_candidate_rows_read": total_candidate_rows,
            "prior_budget_or_capacity_mask_rows_seen": dict(masked_predecessors),
            "forced_or_closed_rows_used_as_samples": 0,
            "action_taken_used_as_label": False,
            "label_rule_output_used": True,
            "first_emission_age_reconstructed_as": "ready_step = tape.step - float32(features[0])",
            "ready_step_is_exact_integer_for_samples": True,
            "feature_dtype": "float32",
            "feature_width": 19,
            "feature_hash": "SHA256 of 19 little-endian IEEE-754 float32 values in controller input order",
        },
        "split": {
            "fit_scene_range_inclusive": list(FIT_RANGE),
            "validation_scene_range_inclusive": list(VALIDATION_RANGE),
            "development_scenes_used_for_fit_or_validation": sorted(
                FORBIDDEN_SCENES.intersection(row["scene_id"] for row in rows)
            ),
            "label_counts": {
                split: {
                    "SKIP_EPISODE": int(counts[0]),
                    "SEND_NOW": int(counts[1]),
                }
                for split, counts in label_counts.items()
            },
            "by_seed": {
                key: {
                    "SKIP_EPISODE": int(counts[0]),
                    "SEND_NOW": int(counts[1]),
                }
                for key, counts in sorted(by_seed_split.items())
            },
            "by_scene": {
                key: {
                    "SKIP_EPISODE": int(counts[0]),
                    "SEND_NOW": int(counts[1]),
                }
                for key, counts in sorted(by_scene_split.items())
            },
            "feature_repeat_summary": feature_summary,
            "float32_label_conflicts": 0,
        },
        "output_dataset_sha256": None,
        "disclosure": {
            "repeated_on_policy_observations_are_retained": True,
            "source_repetition_is_reported_not_mislabeled_as_independent_scenes": True,
            "validation_is_not_an_independent_test_set": True,
        },
    }
    dataset_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        .encode("utf-8")
        for row in rows
    )
    audit["output_dataset_sha256"] = sha256_bytes(dataset_bytes)
    audit["output_dataset_bytes"] = len(dataset_bytes)
    write_new_or_match(DATASET_PATH, dataset_bytes)
    audit_bytes = (
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    write_new_or_match(AUDIT_PATH, audit_bytes)
    return audit


if __name__ == "__main__":
    result = build_dataset()
    print(json.dumps({
        "status": result["status"],
        "dataset_sha256": result["output_dataset_sha256"],
        "sample_counts": result["split"]["label_counts"],
        "unique_feature_counts": {
            split: values["unique_float32_features"]
            for split, values in result["split"]["feature_repeat_summary"].items()
        },
    }, ensure_ascii=False, sort_keys=True))
