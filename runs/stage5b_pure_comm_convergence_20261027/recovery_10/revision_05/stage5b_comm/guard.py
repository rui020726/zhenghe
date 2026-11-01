"""Freeze guard: the execution closure, the new sources and the experiment identity.

The execution closure is the set of read-only files this experiment actually imports or
reads: the integrated runtime, the reused Stage A/B/C harness, the A4 adapter, the A3 runner
and its metrics, the trial harness, the authority fixture, the inherited configuration archive
and the two frozen checkpoints. Reference-only analysis scripts are hashed under a separate
key because they are never executed here.

New experimental code (stage5b_comm/, tests/, the CLI entry, the protocol and this README) is
pinned by the same identity and is compared on every verify. It is not required to match the
historical 207-file communication manifest: this round is a new, separately pinned experiment.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

from . import constants as C
from .paths import BASE_ROOT, LEGACY_ROOTS, REPO_ROOT, REVISION_ROOT, BaseLayout
from . import parent_admission

INTEGRATED_SAR_FILES = (
    "integrated_sar/__init__.py",
    "integrated_sar/comm.py",
    "integrated_sar/comm_policy.py",
    "integrated_sar/comm_learning.py",
    "integrated_sar/comm_ack.py",
    "integrated_sar/comm_budget.py",
    "integrated_sar/contracts.py",
    "integrated_sar/coordinator.py",
    "integrated_sar/orchestrator.py",
    "integrated_sar/allocation_trace.py",
    "integrated_sar/clock.py",
    "integrated_sar/event_bus.py",
    "integrated_sar/ports.py",
    "integrated_sar/task_registry.py",
    "integrated_sar/assignment_gate.py",
    "integrated_sar/lifecycle_metrics.py",
    "integrated_sar/lifecycle_log.py",
    "integrated_sar/info_probe.py",
    "integrated_sar/search_probe.py",
    "integrated_sar/detection_replay.py",
    "integrated_sar/joint_driver.py",
    "integrated_sar/task03_runtime.py",
    "integrated_sar/assembly.py",
    "integrated_sar/scenarios.py",
    "integrated_sar/data_splits.py",
    "integrated_sar/profiles.py",
    "integrated_sar/piloc_runtime.py",
    "integrated_sar/search_runtime.py",
    "integrated_sar/run_signature.py",
    "integrated_sar/errors.py",
    "integrated_sar/expired_attribution.py",
)

#: The A4 Decoupled/K0 seam and the A3 runner it installs into.
A4_FILES = (
    "runs/a4_staged_pool_probe_20261020/a4_hooks.py",
    "runs/a4_staged_pool_probe_20261020/implementation/a4_world_adapter.py",
)
A3_FILES = (
    "runs/attribution_and_gossip_probe_20261020/a2_a3_runner.py",
    "runs/attribution_and_gossip_probe_20261020/a1_metrics.py",
    "runs/attribution_and_gossip_probe_20261020/a1_shadow_eval.py",
    "runs/attribution_and_gossip_probe_20261020/b_metrics.py",
    "runs/attribution_and_gossip_probe_20261020/b_preflight.py",
)
TRIAL_FILES = (
    "runs/communication_trial_20261019/trial_runner.py",
    "runs/communication_trial_20261019/trial_train.py",
    "runs/communication_trial_20261019/trial_eval.py",
    "runs/communication_trial_20261019/startup_identity.py",
)

REUSED_RUN_FILES = (
    "runs/comm_cost_v2_rerun_20260928/v2_common.py",
    "runs/comm_cost_v2_rerun_20260928/v2_run.py",
    "runs/comm_cost_v2_rerun_20260928/v2_trainer.py",
    "runs/comm_cost_v2_rerun_20260928/v2_controller.py",
    "runs/comm_cost_v2_rerun_20260928/v2_reward_timing.py",
    "runs/comm_cost_v2_rerun_20260928/v2_rules.py",
    "runs/comm_cost_stage_b_20260927/sb_controller.py",
    "runs/comm_cost_stage_b_20260927/sb_collector.py",
    "runs/comm_cost_stage_b_20260927/sb_common.py",
    "runs/comm_budget_pilot_20260919/train_pilot.py",
    "runs/comm_budget_pilot_20260919/pilot_common.py",
    "runs/comm_budget_pilot_20260919/pilot_env.py",
    "runs/comm_budget_action_mask_fix_20260922/budget_masked.py",
    "runs/comm_tight_budget_20260919/onpolicy.py",
    "runs/comm_cost_controller_stage_a_20260927/cost_math.py",
    "runs/search_trace_and_policy_20260925/sp_common.py",
)

#: Read at runtime but not imported: the authority fixture and the inherited D1 archive.
EXTERNAL_FILES = (
    r"D:\code\整合\zhenghe\runs\comm_control_20260919\common.py",
    r"D:\code\整合\zhenghe\runs\comm_info_path_validation_20260923\logs\data_A3_s1100000_rp.jsonl",
    r"D:\code\整合\zhenghe\runs\corrected_env_final_20260918\ARMS_CONFIG_TRUTH.json",
)

WEIGHTS = (
    r"D:\code\整合\zhenghe\runs\cur_v2rep05_s2\best\shared_mappo_model_21_3100.pth",
    r"D:\code\整合\zhenghe\runs\joint_eval_20260916\ctde_final_7_1.pth",
)

#: Analysis scripts that informed the design but are never executed by this experiment.
REFERENCE_FILES = (
    "runs/a4_staged_pool_probe_20261020/analysis_a4_sweep.py",
    "runs/stage5a_ceiling_and_feature_audit_20261020/stage5a_offline.py",
)


class FreezeDrift(RuntimeError):
    """The frozen experiment identity no longer matches the bytes on disk."""


def assert_experiment_root(layout):
    """Require the exact revision containing this source, before any phase writes.

    Matching protocol IDs in another directory do not authorise that directory.
    Pure fixtures mock the source anchor explicitly; there is no runtime bypass.
    """
    root = Path(layout.root).resolve()
    canonical = Path(REVISION_ROOT).resolve()
    if root != canonical:
        raise FreezeDrift("run root %s is not this source's canonical revision %s"
                          % (root, canonical))
    protocol_path = Path(layout.preregistration)
    if not protocol_path.is_file():
        raise FreezeDrift("run root %s has no pre-registration" % (root,))
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FreezeDrift("cannot read the run-root pre-registration: %s" % (error,)) from error
    if protocol.get("experiment_id") != C.RUN_ID:
        raise FreezeDrift("run root %s has experiment_id %r, expected %r"
                          % (root, protocol.get("experiment_id"), C.RUN_ID))
    revision = protocol.get("revision")
    if not isinstance(revision, dict) or revision.get("revision_id") != C.REVISION_ID:
        raise FreezeDrift("run root %s is not registered as %s" % (root, C.REVISION_ID))
    return root


def repo_root(layout):
    """The real repository, anchored by integrated_sar, independent of revision depth."""
    return REPO_ROOT


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry(path):
    resolved = Path(path)
    if not resolved.is_file():
        raise FreezeDrift("frozen dependency is missing: %s" % (resolved,))
    return {"path": str(resolved), "bytes": int(resolved.stat().st_size),
            "sha256": sha256_file(resolved)}


def _resolve(repo, item):
    path = Path(item)
    if not path.is_absolute():
        path = repo / item
    if not path.is_file():
        raise FreezeDrift("allowlist entry is missing: %s" % (path,))
    return path


def allowlist_paths(layout):
    """Every repository or external file this experiment reads, in deterministic order."""
    repo = repo_root(layout)
    relative = (list(INTEGRATED_SAR_FILES) + list(A4_FILES) + list(A3_FILES)
                + list(TRIAL_FILES) + list(REUSED_RUN_FILES))
    adapter_dir = repo / "integrated_sar" / "adapters"
    if adapter_dir.is_dir():
        for path in sorted(adapter_dir.glob("*.py")):
            relative.append(str(path.relative_to(repo)).replace("\\", "/"))
    paths = [_resolve(repo, item) for item in relative]
    paths.extend(_resolve(repo, item) for item in EXTERNAL_FILES)
    return paths


def reference_paths(layout):
    repo = repo_root(layout)
    return [_resolve(repo, item) for item in REFERENCE_FILES]


def _new_source_paths(layout):
    root = Path(layout.root)
    paths = [root / "stage5b.py", root / "README.md"]
    skip = {Path(layout.identity).resolve(), Path(layout.slot_plan).resolve()}
    for folder in ("stage5b_comm", "tests", "protocol"):
        for path in sorted((root / folder).glob("*")):
            # The identity cannot hash itself: it does not exist at freeze time and would
            # otherwise appear as "new file added" on the very verify that follows.
            if path.is_file() and path.resolve() not in skip:
                paths.append(path)
    return paths


def base_history_row_paths(layout):
    """Exactly 7920 explicitly assigned pre-resume rows; no ancestor glob."""
    parent_admission.validate_source_intervals()
    return [C.history_row_path(arm, seed, index)
            for arm in C.ARMS_TRAIN for seed in C.TRAIN_SEEDS
            for index in range(C.STAGE_START_EPISODE)]


def assert_parent_admission(layout):
    """Read-only hard gate; PENDING refuses before any execution write/import."""
    return parent_admission.validate(layout)


def parent_identity_pin(layout):
    return parent_admission.parent_identity_pin(layout)


def base_evidence_paths(layout):
    return [Path(layout.parent_admission)] + parent_admission.evidence_paths(layout)


def base_checkpoint_pins(layout):
    return parent_admission.checkpoint_pins(layout)


def base_checkpoint_paths(layout):
    base = BaseLayout()
    return [base.checkpoint(item["arm"], item["seed"], item["name"])
            for item in base_checkpoint_pins(layout)]


def base_revision_summary(layout):
    return {"experiment_id": C.BASE_EXPERIMENT_ID, "revision_id": C.BASE_REVISION_ID,
            "root": str(BaseLayout().root), "admission_mode": "NORMAL_TERMINAL_NO_STOP",
            "history_sources": [dict(source) for source in C.HISTORY_SOURCES]}


def _convergence_ancestor_entries_and_roots():
    """Explicitly pinned ancestor certificates; never discovered by a glob of run roots."""
    entries = []
    roots = []
    seen_roots = set()
    for group in C.CONVERGENCE_ANCESTOR_ROOTS:
        if not isinstance(group, dict):
            raise FreezeDrift("a convergence ancestor entry is not a mapping: %r" % (group,))
        if not group.get("root") or not group.get("experiment_id"):
            raise FreezeDrift("a convergence ancestor must name root and experiment_id")
        root = Path(group["root"]).resolve()
        if not root.is_dir():
            raise FreezeDrift("convergence ancestor root is missing: %s" % (root,))
        try:
            identity_path = parent_admission.verify_pin(
                root, group.get("identity_pin"), "protocol/EXPERIMENT_IDENTITY.json")
            ancestor_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except parent_admission.ParentAdmissionError as error:
            raise FreezeDrift(str(error)) from error
        if ancestor_identity.get("experiment_id") != group["experiment_id"]:
            raise FreezeDrift("convergence ancestor identity belongs to another run")
        root_key = str(root).casefold()
        if root_key not in seen_roots:
            seen_roots.add(root_key)
            roots.append({"root": str(root),
                          "experiment_id": str(group["experiment_id"]),
                          "identity_sha256": group["identity_pin"]["sha256"]})
        certificates = group.get("certificates")
        if not isinstance(certificates, (list, tuple)):
            raise FreezeDrift("convergence ancestor %s has no explicit certificate list"
                              % (root,))
        for item in certificates:
            if not isinstance(item, dict):
                raise FreezeDrift("a convergence certificate entry is not a mapping")
            for key in ("arm", "seed", "path", "sha256", "bytes"):
                if key not in item:
                    raise FreezeDrift("a convergence certificate entry has no %r" % (key,))
            relative = Path(str(item["path"]))
            if relative.is_absolute():
                raise FreezeDrift("an ancestor certificate path must be relative to its root")
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise FreezeDrift("ancestor certificate escapes its root: %s" % (path,)) from error
            if not path.is_file():
                raise FreezeDrift("ancestor certificate is missing: %s" % (path,))
            actual_bytes = int(path.stat().st_size)
            actual_sha = sha256_file(path)
            assert_hex64("ancestor certificate sha256", item["sha256"])
            if actual_bytes != int(item["bytes"]) or actual_sha != item["sha256"]:
                raise FreezeDrift("ancestor certificate moved: %s" % (path,))
            entries.append({"root": str(root),
                            "experiment_id": str(group["experiment_id"]),
                            "identity_sha256": group["identity_pin"]["sha256"],
                            "arm": str(item["arm"]), "seed": int(item["seed"]),
                            "path": str(path), "bytes": actual_bytes,
                            "sha256": actual_sha})
    return entries, roots


def convergence_ancestor_entries():
    """The explicitly pinned ancestor certificate files, in declaration order."""
    return _convergence_ancestor_entries_and_roots()[0]


def convergence_ancestor_roots():
    """The explicitly declared ancestor roots, deduplicated by resolved path."""
    return _convergence_ancestor_entries_and_roots()[1]


def collect_identity(layout):
    """Hash the execution closure, the protocol, new sources, weights and the slot plan."""
    parent_state = assert_parent_admission(layout)
    files = [_entry(path) for path in allowlist_paths(layout)]
    files.append(_entry(layout.preregistration))
    reference = [_entry(path) for path in reference_paths(layout)]
    new_sources = [_entry(path) for path in _new_source_paths(layout)]
    weights = [_entry(Path(item)) for item in WEIGHTS]
    base_files = [_entry(path) for path in base_evidence_paths(layout)]
    base_checkpoints = [_entry(path) for path in base_checkpoint_paths(layout)]
    ancestor_files, ancestor_roots = _convergence_ancestor_entries_and_roots()
    return {
        "schema": C.IDENTITY_SCHEMA,
        "experiment_id": C.RUN_ID,
        "root": str(Path(layout.root).resolve()),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_sha256": sha256_file(layout.preregistration),
        "attachment_sha256": C.ATTACHMENT_SHA256,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "scenes": {"training": list(C.TRAIN_SCENES), "evaluation": list(C.EVAL_SCENES),
                   "evaluation_class": C.EVAL_CLASS, "smoke": C.SMOKE_SCENE},
        "seeds": {"training": list(C.TRAIN_SEEDS),
                  "offsets": {str(k): v for k, v in C.SEED_OFFSET.items()}},
        "transport": {"budget_bytes": C.BUDGET_BYTES, "steps": C.STEPS,
                      "subscribers": C.SUBSCRIBERS, "b0_bytes": C.B0_BYTES},
        "slot_plan": _entry(layout.slot_plan),
        "launch_plan": {"world_cap": C.WORLD_CAP_TOTAL, "replay": C.REPLAY_WORLDS,
                        "train": C.TRAIN_WORLDS, "pacing": C.PACING_WORLDS,
                        "eval": C.NEW_EVAL_WORLDS, "science_worlds": C.SCIENCE_WORLDS,
                        "stage": C.STAGE_INDEX,
                        "stage_start_episode": C.STAGE_START_EPISODE,
                        "stage_end_episode": C.STAGE_END_EPISODE,
                        "inherited_slots": 0, "recovered_rows": 0},
        #: Historical launch accounting is recorded, never re-claimed.
        "launch_cap_proposal": {"prior_approved_cumulative_cap":
                                    C.PRIOR_APPROVED_CUMULATIVE_CAP,
                                "prior_approved_status": C.PRIOR_APPROVED_STATUS,
                                "prior_recorded_actual_claims":
                                    parent_state["prior_recorded_actual_claims"],
                                "own_claim_cap": C.OWN_CLAIM_CAP,
                                "approved_cumulative_cap":
                                    C.APPROVED_CUMULATIVE_LAUNCH_CAP,
                                "approved": C.LAUNCH_CAP_APPROVED,
                                "stage_authority": C.STAGE_AUTHORITY,
                                "note": ("the user goal has no fixed total episode count; each "
                                         "stage is a separately frozen finite cap and an "
                                         "unconverged run continues in the next stage")},
        "stage": {"index": C.STAGE_INDEX, "start_episode": C.STAGE_START_EPISODE,
                  "end_episode": C.STAGE_END_EPISODE,
                  "new_episodes_per_run": C.NEW_EPISODES_PER_STAGE,
                  "verdict_if_no_certificate": C.STAGE_VERDICT_IF_NO_CERTIFICATE,
                  "verdict_note": C.STAGE_VERDICT_NOTE},
        "history": {"episodes": list(C.PARENT_HISTORY_EPISODES),
                    "sources": [dict(source) for source in C.HISTORY_SOURCES],
                    "rows_pinned": int(C.HISTORY_ROWS_PINNED),
                    "note": ("complete continuous 0..1319 lineage; the criterion's first "
                             "window still starts at episode 120")},
        "convergence": {
            "window_episodes": C.CONVERGENCE_WINDOW_EPISODES,
            "windows": C.CONVERGENCE_WINDOWS,
            "pairs": [list(pair) for pair in C.CONVERGENCE_PAIRS],
            "interval_level": C.CONVERGENCE_INTERVAL_LEVEL,
            "resamples": C.CONVERGENCE_BOOTSTRAP_REPS,
            "order_statistics": [C.CONVERGENCE_LOWER_ORDER, C.CONVERGENCE_UPPER_ORDER],
            "seed_label": C.CONVERGENCE_SEED_LABEL,
            "pair_bound": C.CONVERGENCE_PAIR_BOUND,
            "mean_range_bound": C.CONVERGENCE_MEAN_RANGE_BOUND,
            "slope_bound": C.CONVERGENCE_SLOPE_BOUND,
            "slope_basis": C.CONVERGENCE_SLOPE_BASIS,
            "budget_range_bound": C.CONVERGENCE_BUDGET_RANGE_BOUND,
            "budget_range_role": C.CONVERGENCE_BUDGET_RANGE_ROLE,
            "first_check_episode": C.CONVERGENCE_FIRST_CHECK_EPISODE,
            "check_step": C.CONVERGENCE_CHECK_STEP,
            "required_consecutive": C.CONVERGENCE_REQUIRED_CONSECUTIVE,
            "check_episodes_this_stage": list(C.convergence_check_episodes()),
            "ancestor_certificate_roots": ancestor_roots,
            "ancestor_certificate_files": ancestor_files,
        },
        "parent_provenance": parent_state,
        "parent_admission": parent_state,
        "candidate_status": C.CANDIDATE_STATUS,
        "base_revision": base_revision_summary(layout),
        "files": files,
        "reference_files": reference,
        "new_sources": new_sources,
        "weights": weights,
        "base_files": base_files,
        "base_checkpoints": base_checkpoints,
        "convergence_ancestors": {"roots": ancestor_roots, "files": ancestor_files},
        "note": ("execution closure plus new sources; reference analysis scripts are hashed "
                 "separately and are never executed; base revision files are read-only "
                 "evidence; this revision inherits no launch slot"),
    }


def freeze_identity(layout):
    """Exclusive-create the identity; an existing identity is never overwritten."""
    assert_experiment_root(layout)
    assert_parent_admission(layout)
    payload = collect_identity(layout)
    path = layout.identity
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return payload


def _table(entries):
    return {item["path"]: (item["sha256"], item["bytes"]) for item in entries or []}


def _diff_tables(name, frozen, current, differences):
    old = _table(frozen)
    now = _table(current)
    for key in sorted(set(old) | set(now)):
        if key not in old:
            differences.append("%s: new file added: %s" % (name, key))
        elif key not in now:
            differences.append("%s: frozen file vanished: %s" % (name, key))
        elif old[key] != now[key]:
            differences.append("%s: bytes moved: %s" % (name, key))


def compare_identity(frozen, current):
    """Pure comparison used by verify_identity and by the synthetic drift tests."""
    differences = []
    for key in ("schema", "experiment_id", "root"):
        if frozen.get(key) != current.get(key):
            differences.append("%s moved: %r != %r"
                               % (key, frozen.get(key), current.get(key)))
    if frozen.get("protocol_sha256") != current.get("protocol_sha256"):
        differences.append("protocol hash moved")
    for name, key in (("execution", "files"), ("reference", "reference_files"),
                      ("new source", "new_sources"), ("weights", "weights"),
                      ("base evidence", "base_files"),
                      ("base checkpoint", "base_checkpoints")):
        _diff_tables(name, frozen.get(key), current.get(key), differences)
    if frozen.get("base_revision") != current.get("base_revision"):
        differences.append("base revision identity moved: %r != %r"
                           % (frozen.get("base_revision"), current.get("base_revision")))
    if frozen.get("history") != current.get("history"):
        differences.append("pre-resume history pin moved")
    if frozen.get("convergence_ancestors") != current.get("convergence_ancestors"):
        differences.append("convergence ancestor certificate pins moved")
    if frozen.get("parent_admission") != current.get("parent_admission"):
        differences.append("parent admission evidence moved")
    if frozen.get("convergence") != current.get("convergence"):
        differences.append("convergence criterion or absolute ladder moved")
    old_plan = frozen.get("slot_plan") or {}
    now_plan = current.get("slot_plan") or {}
    if (old_plan.get("sha256"), old_plan.get("bytes")) != (now_plan.get("sha256"),
                                                          now_plan.get("bytes")):
        differences.append("slot plan bytes moved")
    if frozen.get("launch_plan") != current.get("launch_plan"):
        differences.append("launch plan counts moved")
    if frozen.get("scenes") != current.get("scenes"):
        differences.append("scenes moved")
    if frozen.get("seeds") != current.get("seeds"):
        differences.append("seeds moved")
    if frozen.get("transport") != current.get("transport"):
        differences.append("transport constants moved")
    if differences:
        raise FreezeDrift("experiment identity drift: " + "; ".join(differences))
    return True


def verify_identity(layout):
    """Recompute and compare; raise FreezeDrift listing every difference."""
    assert_experiment_root(layout)
    path = layout.identity
    if not Path(path).is_file():
        raise FreezeDrift("experiment identity is missing; run prepare first")
    frozen = json.loads(Path(path).read_text(encoding="utf-8"))
    current = collect_identity(layout)
    compare_identity(frozen, current)
    return {"verified": True, "experiment_id": current["experiment_id"],
            "root": current["root"], "protocol_sha256": current["protocol_sha256"],
            "files": len(current["files"]), "new_sources": len(current["new_sources"]),
            "weights": len(current["weights"]),
            "slot_plan_sha256": current["slot_plan"]["sha256"]}


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def assert_hex64(name, value):
    """A frozen sha256 is exactly 64 lower-case hex digits; a short paste is a hard error."""
    if not isinstance(value, str) or not _HEX64.match(value):
        raise FreezeDrift("%s is not a 64-hex sha256: %r" % (name, value))
    return value


def assert_base_pins(layout):
    """Require full normal-parent admission, all 7920 history pins and actual terminal proof."""
    return assert_parent_admission(layout)

def source_pins(layout):
    """Relative path -> sha256 table used by receipts and the final manifest."""
    repo = repo_root(layout)
    pins = {}
    for path in allowlist_paths(layout):
        try:
            key = str(Path(path).relative_to(repo))
        except ValueError:
            key = str(path)
        pins[key.replace("\\", "/")] = sha256_file(path)
    return pins
