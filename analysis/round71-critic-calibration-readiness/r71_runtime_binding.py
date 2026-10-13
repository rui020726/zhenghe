#!/usr/bin/env python
"""ROUND71 real runtime binding for the future critic-calibration provider (ZERO EXECUTION).

This module is the ACTUAL wiring between the candidate critic-calibration initialisation and the
live ROUND68 runtime source.  It is preparation only: NOTHING in this module is called this
round.  Every entry point that would touch real state (runtime configuration, model load, .pt
open, RNG restore, teacher read, neural forward, fit, optimizer, checkpoint, claim, episode) is
gated behind a verified licence AND an explicit model-read approval, and the CLI stops at the
model read-in boundary.

What is REAL here (verified statically, not mocked):
  * the source manifest: exact files, SHA256 identities and function/line resolution for the
    controller, the model-only ep0 loader, the critic->trunk detach, the PPO/common live config,
    the five-module checkpoint/restore, the claim/slot ledger and the evaluation seam;
  * ``configure_r71_runtime``: stands up the live ROUND68 runtime through the real
    ``r68_execute.configure_and_inject`` and rebinds ONLY identity/version/checkpoint/quota,
    leaving the trainer math, the routed forward, the mask, the loss and the dual update alone;
  * ``CalibrationProvider``: the real trunk entry ``(N,19)`` charged per call through the
    corrected ``ForwardLedger`` with the 1024 guard, the mean-over-targets pooling to joint rows
    exactly as ``r66_frozen.frozen_early_ppo_update`` pools critic values, the fixed dual-output
    ridge, the exact inverse standardisation to the raw 64-D affine critic heads, and a
    ``copy_`` installation into a NEW model's two critic parameters only;
  * ``validate_calibration_source``: payload-bound role validation (R68 ep0 model-only INIT,
    R68 ep128 read-only CONTROL, this run's own NEW full restore), never caller flags alone.

What is NOT done here: no .pt is opened, no model is constructed or loaded, no RNG is restored,
no forward is run, no fit is executed, no optimizer step, no checkpoint write, no claim, no
episode.  The default trunk delegate does not exist: the provider calls the real controller
trunk, and tests inject a spy callable instead.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from r71_critic_calibration_seam import (  # noqa: E402
    BATCH_INPUT_ROW_CAP, EMBED_DIM, FEATURE_DIM, OFFLINE_INPUT_ROW_CAP, RIDGE,
    BatchTooLarge, BudgetExceeded, ForwardLedger, LicenseGate, LicenseInvalid,
    LicenseMissing, RealLicense, TrunkInputError, affine_to_critic_heads,
    standardized_to_raw_affine,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN_ROOT = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
R68_DIR = RUN_ROOT / "round68_late_initial_confidence_ablation"
R66_DIR = RUN_ROOT / "round66_bc_early_path_preserving_late_adaptation"
R58_DIR = RUN_ROOT / "round58_gen2"
R53_DIR = RUN_ROOT / "round53_critic_isolation_trial_readiness"
R60_DIR = RUN_ROOT / "round60_partitioned_actor_head_ablation"
R42_ONE_SHOT = RUN_ROOT / "r42_one_shot.py"
R46_RUNNER = RUN_ROOT / "r46_execute_r2.py"
# The LIVE canonical trainer tree (r66_frozen.configure_source_imports resolves it here).
CANON = REPO.parent / "zhenghe" / "runs" / "comm_cost_v2_rerun_20260928"
STAGE_B = REPO.parent / "zhenghe" / "runs" / "comm_cost_stage_b_20260927"
PILOT = REPO.parent / "zhenghe" / "runs" / "comm_tight_budget_20260919"

TASK_ID = "ROUND71-REAL-RUNTIME-BINDING-ZERO-EXEC"
BRANCH = "analysis/round71-critic-calibration-readiness"

# ---------------------------------------------------------------------------------------
# identity: the R71 training/model contract EXTENDS the frozen ROUND68 one (no science change)
# ---------------------------------------------------------------------------------------
#: ROUND68 training version, taken from the frozen ROUND68_FREEZE.json (verified read-only in
#: ``verify_source_manifest``; never guessed, never re-derived from an import default).
R68_TRAINING_VERSION_EXPECTED = (
    "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/"
    "dual-eta-lambda-0p5/1/rule-bc-late147-init/1/published-completion-event-at-relay-tick/2"
    "+r53-critic-representation-stop-gradient/1+r56-soft-early-report-prior-beta0p10/1"
    "+r58-prior-strength-beta1p00/1+r60-partitioned-early-late-actor-head/1"
    "+r66-frozen-early-path-late-adaptation/1+r68-late-initial-confidence-alpha0.10/1")

R71_TRAINING_VERSION = R68_TRAINING_VERSION_EXPECTED + "+r71-critic-calibration-init/1"
R71_LOSS_VERSION = "constrained-lagrangian/2+partitioned-actor-head/1+frozen-early-path/1"
R71_STATE_SCHEMA = ("constrained-cost-state/2+partitioned-actor-head/1+frozen-early-path/1"
                    "+r68-late-init-alpha/1+r71-critic-calibration-init/1")
R71_OPTIMIZER_SCHEMA = ("stage-c-frozen-early-path-six-module-parameter-set/1"
                        "+r68-late-init-alpha/1+r71-critic-calibration-init/1")
R71_STRUCTURE = ("partitioned-early-late-actor-head/1+frozen-early-path-late-adaptation/1"
                 "+late-initial-confidence-alpha0.10/1+r71-critic-calibration-heads/1")
R71_TRAINER_TAG = "constrained-cost-v2-frozen-early-path-late-init-alpha-critic-calibration"
R71_FREEZE_KIND = "round71-critic-calibration-freeze/1"
R71_CALIBRATION_ROUTE = "critic-representation-stop-gradient/1+r71-dual-output-ridge-init/1"

#: registered (NOT authorised) plan identity; any licensed run is counted separately and in full
R71_QUOTA_IDENTITY = ("round71-critic-calibration/plan-582-normal+18-reserve-hard-600/"
                      "offline-72000-input-rows-batch-1024/1")
CANDIDATE_STATUS = "APPROVAL_PENDING"

RESTORE_ROLES = ("r68_ep0_model_only_init", "r68_ep128_readonly_control",
                 "r71_new_full_restore", "rule_control_readonly")

#: frozen scientific fields and where each one comes from in the CONFIGURED runtime.  These are
#: living values on the actual common module (or module constants), never import defaults.
LIVE_CONFIG_CONTRACT = (
    ("common.GAMMA", 1.0, "v2_common.GAMMA; r68_execute.configure_and_inject asserts ==1.0"),
    ("common.GAE_LAMBDA", 1.0, "living override set by r68_execute.configure_and_inject"),
    ("common.ETA_LAMBDA", 0.5, "living override set by r68_execute.configure_and_inject"),
    ("common.B0_BYTES", 10000.0, "v2_common.B0_BYTES (positive cost normaliser)"),
    ("common.DATA_ORDER_SEED", 20260927, "v2_common.DATA_ORDER_SEED (frozen data order)"),
    ("common.EPOCHS_PER_BATCH", 4, "v2_common.EPOCHS_PER_BATCH"),
    ("common.MINIBATCH_STEPS", 256, "v2_common.MINIBATCH_STEPS"),
    ("r66.READY_CUT", 147.0, "r66_frozen.READY_CUT (ready_step cut, routing only)"),
    ("r66.HORIZON", 1000.0, "r66_frozen.HORIZON"),
    ("r66.GATE_BETA", 1.0, "r66_frozen.GATE_BETA (beta=1.00 gate prior)"),
    ("r68.LATE_INIT_ALPHA", 0.10, "r68_frozen.LATE_INIT_ALPHA (init-only, never re-applied)"),
)


class ModelReadBoundary(RuntimeError):
    """Raised at the model read-in point without an explicit recorded model-read approval."""


class SourceIdentityError(RuntimeError):
    """A bound source file's identity drifted from the registered manifest."""


class RoleValidationError(RuntimeError):
    """A payload failed its role-bound validation (not merely a declared flag)."""


# --------------------------------------------------------------------------------------- 1. manifest
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_line(path: Path, pattern: str) -> int:
    """1-based line number of the first line matching ``pattern`` (static source resolution)."""
    rx = re.compile(pattern)
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if rx.search(line):
            return i
    raise SourceIdentityError(f"{path.name}: pattern {pattern!r} not found")


def _parse_r66_expected_identity() -> dict:
    """Statically read R60_EXPECTED_IDENTITY out of r66_frozen.py (no import, no torch)."""
    src = (R66_DIR / "r66_frozen.py").read_text(encoding="utf-8")
    m = re.search(r"R60_EXPECTED_IDENTITY\s*=\s*(\{.*?\n\})\n", src, re.S)
    if not m:
        raise SourceIdentityError("R60_EXPECTED_IDENTITY not found in r66_frozen.py")
    return ast.literal_eval(m.group(1))


def _expected_identity_row(seed: int, role: str) -> dict:
    """One row of the registered R60_EXPECTED_IDENTITY table (int keys, statically parsed)."""
    table = _parse_r66_expected_identity()
    key = int(seed) if int(seed) in table else str(seed)
    if key not in table:
        raise SourceIdentityError(f"seed {seed} is not in the registered identity table")
    if role not in table[key]:
        raise SourceIdentityError(f"role {role!r} is not registered for seed {seed}")
    return table[key][role]


def build_source_manifest() -> dict:
    """Read-only manifest of every source this binding depends on, with live identities."""
    entries = []

    def add(role, path, functions=(), note=""):
        p = Path(path)
        entry = {"role": role, "path": str(p), "exists": p.is_file(),
                 "note": note}
        if p.is_file():
            entry["sha256"] = sha256_file(p)
            entry["functions"] = {name: _resolve_line(p, rf"^\s*(def|class)\s+{name}\b")
                                  for name in functions}
        entries.append(entry)

    add("canonical_trainer", CANON / "v2_trainer.py",
        ["ppo_update", "save_checkpoint", "load_checkpoint", "batch_samples",
         "_unassigned_prefix_totals"],
        "live trainer the R66/R68 chain imports (r66_frozen.configure_source_imports)")
    add("canonical_common", CANON / "v2_common.py", [],
        "live common module holding the configured GAE_LAMBDA/ETA_LAMBDA living values")
    add("canonical_controller", CANON / "v2_controller.py", ["fresh_controller"],
        "StageCController factory the runtime calls for a fresh seed controller")
    add("canonical_reward_timing", CANON / "v2_reward_timing.py", [],
        "owns TRAINING_VERSION / REWARD_TIMING_VERSION imported by v2_trainer")
    add("onpolicy", PILOT / "onpolicy.py", ["joint_log_prob", "action_entropy", "restore_rng"],
        "frozen PPO math primitives (ALL_ROWS reduction, joint log-prob, entropy)")
    add("stage_b_controller", STAGE_B / "sb_controller.py", ["StageBController"],
        "the base controller the partitioned controller wraps")
    add("partitioned_controller_and_ppo", R66_DIR / "r66_frozen.py",
        ["PartitionedController", "route_logits", "ready_steps", "load_r60_model",
         "r60_source_identity", "build_frozen_controller", "frozen_early_ppo_update",
         "build_optimizer", "fresh_training_state", "freeze_early_path",
         "load_r60_checkpoint_read_only", "configure_source_imports"],
        "five-module controller, model-only R60 loader, frozen-early-path PPO update")
    add("critic_trunk_detach", R53_DIR / "round53_trial_adapter.py",
        ["critic_input_detached", "install_on_verified_trainer", "validate_live_binding"],
        "critic->trunk gradient isolation (forward pre-hook detach on both critic heads)")
    add("r68_identity_and_checkpoint", R68_DIR / "r68_frozen.py",
        ["build_frozen_controller", "scale_late_head_init", "fresh_training_state",
         "save_checkpoint", "load_checkpoint", "load_checkpoint_read_only",
         "_reject_foreign_contract", "_validate_base", "_restore_controller"],
        "R68 late-init identity + five-module checkpoint contract this binding extends")
    add("r68_executor", R68_DIR / "r68_execute.py",
        ["configure_and_inject", "R68Ledger", "_r68_atomic_run",
         "_make_permit_guarded_builder", "zero_env_acceptance", "main"],
        "runtime stand-up, claim/slot/quota ledger, construction permits, acceptance")
    add("r58_base_runner", R58_DIR / "r58_execute.py",
        ["load_base_runner", "bind_runner_paths", "install_private_configure_seams"],
        "frozen R46/R42 sim runtime stand-up reused by configure_and_inject")
    add("frozen_runner_source", R46_RUNNER, ["main"],
        "the frozen ROUND46 runner module loaded by load_base_runner")
    add("frozen_collector", R42_ONE_SHOT, [],
        "the frozen one-shot collector module (imported for its side effects only)")
    add("r68_eval", R68_DIR / "r68_eval.py",
        ["configure_r68_eval", "quota_guard", "eval_model_arms", "eval_focal",
         "compute_stats", "_load_arm"],
        "evaluation seam the future licensed evaluation would reuse")
    for seed in (11, 22, 33):
        for role in ("ep0", "ep128"):
            spec = _expected_identity_row(seed, role)
            fname = ("v2_train_s%d_before_b0.pt" if role == "ep0"
                     else "v2_train_s%d_ep128.pt") % seed
            entries.append({
                "role": f"r60_{role}_model_source_s{seed}",
                "path": str(R60_DIR / "raw" / "checkpoints" / fname),
                "exists": (R60_DIR / "raw" / "checkpoints" / fname).is_file(),
                "expected_file_sha256": spec["file_sha256"],
                "expected_parameter_sha256": spec["parameter_sha256"],
                "expected_episodes_done": spec["episodes_done"],
                "expected_updates": spec["updates"],
                "note": ("model-only INIT source" if role == "ep0"
                         else "read-only CONTROL comparison source"),
            })
    return {"kind": "round71-real-runtime-binding/source-manifest/1",
            "task_id": TASK_ID, "branch": BRANCH,
            "note": ("read-only manifest; .pt files are NOT opened here, only their registered "
                     "expected identities are recorded"),
            "entries": entries}


def verify_source_manifest(manifest: dict | None = None) -> dict:
    """Recompute every registered identity read-only and report drift.  Never opens a .pt."""
    manifest = manifest or build_source_manifest()
    fresh = build_source_manifest()
    checks, drift = {}, []
    old = {e["role"]: e for e in manifest["entries"]}
    for e in fresh["entries"]:
        prev = old.get(e["role"])
        if prev is None:
            drift.append({"role": e["role"], "problem": "absent from the registered manifest"})
            continue
        if e["exists"] != prev.get("exists"):
            drift.append({"role": e["role"], "problem": "existence changed"})
        if "sha256" in e:
            # a source file: its bytes and its registered function/line resolution must hold
            if e["exists"]:
                if e["sha256"] != prev.get("sha256"):
                    drift.append({"role": e["role"], "problem": "source sha256 drifted"})
                if e.get("functions") != prev.get("functions"):
                    drift.append({"role": e["role"], "problem": "function/line resolution changed"})
            checks[e["role"]] = {
                "exists": e["exists"],
                "sha256_matches": (not e["exists"]) or e["sha256"] == prev.get("sha256")}
        else:
            # a registered model-source identity: the .pt bytes are deliberately NEVER hashed
            # here; only the registered expected identity may be compared
            if e.get("expected_file_sha256") != prev.get("expected_file_sha256"):
                drift.append({"role": e["role"], "problem": "registered expected identity changed"})
            checks[e["role"]] = {
                "exists": e["exists"],
                "registered_identity_unchanged":
                    e.get("expected_file_sha256") == prev.get("expected_file_sha256")}
    # the R68 training-version literal must equal the frozen ROUND68_FREEZE.json value
    freeze = json.loads((R68_DIR / "ROUND68_FREEZE.json").read_text(encoding="utf-8"))
    checks["r68_training_version_literal"] = {
        "equals_freeze_json": R68_TRAINING_VERSION_EXPECTED == freeze["training_version"]}
    if not checks["r68_training_version_literal"]["equals_freeze_json"]:
        drift.append({"role": "r68_training_version_literal",
                      "problem": "literal != ROUND68_FREEZE.json training_version"})
    # the offline input-count evidence is reused only when its identity is unchanged
    counts = json.loads((HERE / "ROUND71_OFFLINE_INPUT_COUNT.json").read_text(encoding="utf-8"))
    checks["offline_input_count_reuse"] = {
        "totals_unchanged": (counts["totals"]["joint_decision_rows"] == 32046
                             and counts["totals"]["target_level_19d_input_rows_AUTHORITATIVE"]
                             == 58928)}
    if not checks["offline_input_count_reuse"]["totals_unchanged"]:
        drift.append({"role": "offline_input_count_reuse", "problem": "counted totals drifted"})
    return {"kind": "round71-real-runtime-binding/source-verification/1",
            "status": "passed" if not drift else "failed",
            "checks": checks, "drift": drift,
            "offline_input_reuse": {
                "registered_from": "ROUND71_OFFLINE_INPUT_COUNT.json",
                "teacher_source": counts["teacher_data_source"],
                "joint_decision_rows": counts["totals"]["joint_decision_rows"],
                "target_level_19d_input_rows": counts["totals"]["target_level_19d_input_rows_AUTHORITATIVE"],
                "reuse_rule": ("reused only while the teacher source and counted totals are "
                               "unchanged; no number is ever guessed (no 58632-style invention)")}}

# ======================================================================================== 2. identity
from r71_critic_calibration_seam import (  # noqa: E402
    CENTER_SCALE_SCOPE, CONSTANT_COLUMN_SCALE, CRITIC_HEAD_MODULES, FROZEN_MODULES,
    INTERCEPT_PENALIZED, MODULES, TRAINABLE_MODULES, CriticHeadSet,
    R68_STRUCTURE as SEAM_R68_STRUCTURE,
    TRAINING_VERSION as SEAM_R68_TRAINING_VERSION, spy_license_verifier)

OFFLINE_INPUT_COUNT_PATH = HERE / "ROUND71_OFFLINE_INPUT_COUNT.json"
R68_FREEZE_PATH = R68_DIR / "ROUND68_FREEZE.json"
R68_CHECKPOINTS = R68_DIR / "raw" / "checkpoints"
R60_CHECKPOINTS = R60_DIR / "raw" / "checkpoints"

#: registered (NOT authorised) launch plan of a future licensed run; installing these caps is
#: itself part of the licensed configuration and never happens this round.
R71_NORMAL_CAP = 582
R71_RESERVE_CAP = 18
R71_HARD_CAP = 600

_EXPECTED_LIVE = {name: value for name, value, _note in LIVE_CONFIG_CONTRACT}
_FROZEN_SEEDS = (11, 22, 33)


@dataclass(frozen=True)
class R71RunIdentity:
    """The identity a licensed ROUND71 run would carry into every payload it writes."""
    run_id: str
    freeze_sha256: str
    input_identity_sha256: str
    quota_identity: str = R71_QUOTA_IDENTITY
    task_id: str = TASK_ID
    branch: str = BRANCH
    training_version: str = R71_TRAINING_VERSION

    def as_dict(self) -> dict:
        return {"run_id": self.run_id, "freeze_sha256": self.freeze_sha256,
                "input_identity_sha256": self.input_identity_sha256,
                "quota_identity": self.quota_identity, "task_id": self.task_id,
                "branch": self.branch, "training_version": self.training_version}

    def matches_block(self, block) -> bool:
        """True only when a payload block equals THIS live identity field for field."""
        want = self.as_dict()
        return isinstance(block, dict) and all(block.get(k) == v for k, v in want.items()) \
            and set(block) == set(want)


def register_offline_input_reuse() -> dict:
    """Read-only reuse registration for the already-counted offline teacher inputs.

    The counted totals and the teacher source are reused ONLY while the evidence file is
    unchanged; its sha256 is the input identity a licensed payload would have to declare.
    No number is ever re-estimated or guessed here (nothing like an invented 58632).
    """
    counts = json.loads(OFFLINE_INPUT_COUNT_PATH.read_text(encoding="utf-8"))
    return {
        "registered_from": OFFLINE_INPUT_COUNT_PATH.name,
        "input_identity_sha256": sha256_file(OFFLINE_INPUT_COUNT_PATH),
        "teacher_source": counts["teacher_data_source"],
        "joint_decision_rows": counts["totals"]["joint_decision_rows"],
        "target_level_19d_input_rows":
            counts["totals"]["target_level_19d_input_rows_AUTHORITATIVE"],
        "offline_input_row_cap": OFFLINE_INPUT_ROW_CAP,
        "batch_input_row_cap": BATCH_INPUT_ROW_CAP,
        "reuse_rule": ("reused only while the teacher source and the counted totals are "
                       "unchanged; the teacher evidence is read, never re-extracted"),
    }


def registered_run_identity(run_id: str) -> R71RunIdentity:
    """Build a run identity from the registered read-only evidence (nothing guessed)."""
    reuse = register_offline_input_reuse()
    return R71RunIdentity(run_id=str(run_id),
                          freeze_sha256=sha256_file(R68_FREEZE_PATH),
                          input_identity_sha256=reuse["input_identity_sha256"])


def _r68_init_identity(seed: int) -> dict:
    """The registered R68 late-init identity evidence for one seed (a JSON, never a .pt)."""
    path = R68_CHECKPOINTS / f"r68_init_identity_s{int(seed)}.json"
    if not path.is_file():
        raise RoleValidationError(f"registered R68 init-identity evidence is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ payload role validation
def validate_calibration_source(payload, *, role, run_identity=None) -> dict:
    """Payload-bound role validation for every model payload the candidate would touch.

    Descriptive only: no .pt is opened, nothing is restored.  A role is proven from the
    PAYLOAD's declared identity block plus the registered source identities, never from a
    caller flag or a bare ``source_kind`` string:

      * ``r68_ep0_model_only_init``: the qualified ROUND60 ep0 file/parameter identity and the
        ep0 cursor from the registered ROUND66 table, the exact frozen ROUND68 training-version
        literal, ``load_mode == "model_only"``, no optimizer / lambda / training RNG restored,
        and the registered R68 post-scale init identity (pre-scale must equal the ROUND60 ep0).
      * ``r68_ep128_readonly_control``: the same table's ep128 row (cursor 128/32), read-only
        load mode, no trained state restored, and an explicit refusal to serve as an INIT source.
      * ``r71_new_full_restore``: EXACTLY this run's training version (a prefix-only ROUND68
        version is refused), the ROUND71 structure and five-module set, source_kind this_run,
        a payload ``run_identity`` block equal to the live identity field for field, and the
        optimizer / lambda / advanced training RNG / cursor all present.
      * ``rule_control_readonly``: no trained state of any kind.
    """
    if role not in RESTORE_ROLES:
        raise RoleValidationError(f"unknown calibration source role {role!r}")
    if not isinstance(payload, dict):
        raise RoleValidationError("a calibration payload must be a declared dict")
    ident = payload.get("identity")
    if not isinstance(ident, dict):
        raise RoleValidationError(
            "a calibration payload must declare an 'identity' block; caller flags alone are "
            "not a declaration")
    seed = ident.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in _FROZEN_SEEDS:
        raise RoleValidationError(
            f"payload identity 'seed' must be one of the frozen seeds {_FROZEN_SEEDS}")

    if role == "rule_control_readonly":
        for flag in ("carries_trained_optimizer_state", "carries_trained_lambda",
                     "carries_advanced_training_rng"):
            if payload.get(flag):
                raise RoleValidationError(
                    f"a read-only rule control may carry no trained state ({flag})")
        return {"role": role, "seed": seed, "descriptive_only": True}

    if set(ident.get("module_set") or ()) != set(MODULES):
        raise RoleValidationError(
            f"payload module set {sorted(ident.get('module_set') or ())} != the five-module "
            f"contract {sorted(MODULES)}")

    if role in ("r68_ep0_model_only_init", "r68_ep128_readonly_control"):
        which = "ep0" if role == "r68_ep0_model_only_init" else "ep128"
        spec = _expected_identity_row(seed, which)
        problems = []
        if ident.get("structure") != SEAM_R68_STRUCTURE:
            problems.append(f"structure {ident.get('structure')!r} != the frozen ROUND68 "
                            f"structure {SEAM_R68_STRUCTURE!r}")
        if ident.get("training_version") != R68_TRAINING_VERSION_EXPECTED:
            problems.append("training_version must be the frozen ROUND68 literal exactly; this "
                            "source predates the ROUND71 suffix and a prefix/suffix mismatch "
                            "is refused")
        if ident.get("source_kind") != "r68_ep0":
            problems.append(f"source_kind {ident.get('source_kind')!r} != 'r68_ep0'")
        if payload.get("file_sha256") != spec["file_sha256"]:
            problems.append("file sha256 != the registered ROUND66 expected identity")
        if payload.get("parameter_sha256") != spec["parameter_sha256"]:
            problems.append("parameter sha256 != the registered ROUND66 expected identity")
        if int(payload.get("episodes_done", -1)) != int(spec["episodes_done"]):
            problems.append(f"episodes_done cursor != {spec['episodes_done']}")
        if int(payload.get("updates", -1)) != int(spec["updates"]):
            problems.append(f"updates cursor != {spec['updates']}")
        if payload.get("load_mode") != ("model_only" if which == "ep0" else "read_only"):
            problems.append(f"load_mode {payload.get('load_mode')!r} is not the required "
                            f"{'model_only' if which == 'ep0' else 'read_only'} mode")
        for flag in ("optimizer_restored", "lambda_restored", "rng_restored"):
            if payload.get(flag):
                problems.append(f"{flag} must be False: this source is model-only and a fresh "
                                f"optimizer / lambda / random state is built instead")
        if which == "ep0":
            if payload.get("from_trained_ep128"):
                problems.append("from_trained_ep128 is set: a trained ep128 is never an INIT "
                                "source")
        else:
            if payload.get("from_trained_ep128") is not True:
                problems.append("the ep128 control must declare from_trained_ep128: it IS the "
                                "trained control and may not pose as the ep0 INIT source")
            if payload.get("used_as_init_source"):
                problems.append("a trained ep128 payload refuses to serve as an INIT source")
        if which == "ep0":
            evidence = _r68_init_identity(seed)
            if not evidence.get("pre_scale_equals_r60_ep0"):
                problems.append("registered R68 init evidence says pre-scale != ROUND60 ep0")
            if evidence.get("late_init_alpha") != _EXPECTED_LIVE["r68.LATE_INIT_ALPHA"]:
                problems.append("registered R68 init evidence late_init_alpha drifted")
            post = payload.get("post_scale_parameter_sha256")
            if post is not None and post != evidence.get("post_scale_parameter_sha256"):
                problems.append("post_scale_parameter_sha256 != the registered R68 init identity")
            if payload.get("pre_scale_equals_r60_ep0") is False:
                problems.append("payload says its pre-scale parameters are not the ROUND60 ep0")
        if problems:
            raise RoleValidationError("; ".join(problems))
        return {"role": role, "seed": seed, "registered_identity": spec,
                "descriptive_only": True}

    # role == "r71_new_full_restore"
    if run_identity is None:
        raise RoleValidationError(
            "a live R71RunIdentity is required to validate this run's own restore")
    problems = []
    if ident.get("structure") != R71_STRUCTURE:
        problems.append(f"structure {ident.get('structure')!r} != {R71_STRUCTURE!r}")
    if ident.get("training_version") != R71_TRAINING_VERSION:
        problems.append("training_version must be EXACTLY the ROUND71 version; a prefix-only "
                        "(ROUND68) version or any other string is refused")
    if ident.get("source_kind") != "this_run":
        problems.append(f"source_kind {ident.get('source_kind')!r} != 'this_run'")
    if not run_identity.matches_block(payload.get("run_identity")):
        problems.append("payload run_identity block does not equal the live run identity field "
                        "for field; a declared source_kind alone is not an identity")
    for flag in ("carries_trained_optimizer_state", "carries_trained_lambda",
                 "carries_advanced_training_rng"):
        if not payload.get(flag):
            problems.append(f"this run's full restore must carry {flag}")
    if payload.get("from_trained_ep128"):
        problems.append("from_trained_ep128 is set: a trained ep128 is never a restore source")
    if problems:
        raise RoleValidationError("; ".join(problems))
    return {"role": role, "seed": seed, "run_identity": run_identity.as_dict(),
            "descriptive_only": True}


# ======================================================================================== 3. provider
#: where each of the five modules actually lives on the verified PartitionedController
_MODULE_ATTRS = {
    "trunk": ("inner", "trunk"),
    "early_head": ("inner", "early_head"),
    "late_head": ("inner", "late_head"),
    "value_task_head": ("inner", "value_head"),
    "value_cost_head": (None, "cost_head"),
}


def resolve_module(controller, name: str):
    """Resolve one of the five modules on the real controller (or a structural stand-in)."""
    if name not in _MODULE_ATTRS:
        raise SourceIdentityError(f"unknown module {name!r}")
    holder_name, attr = _MODULE_ATTRS[name]
    holder = controller if holder_name is None else getattr(controller, holder_name, None)
    if holder is None:
        raise SourceIdentityError(f"controller has no {holder_name!r} holder for {name}")
    module = getattr(holder, attr, None)
    if module is None:
        raise SourceIdentityError(f"controller exposes no module for {name}")
    return module


def _parameter_bytes(parameter) -> bytes:
    detach = getattr(parameter, "detach", None)
    if callable(detach):
        parameter = detach().cpu().numpy()
    return np.ascontiguousarray(parameter).tobytes()


def module_parameter_digest(controller) -> dict:
    """Read-only SHA256 of every module's parameters, keyed by module name.

    Works on real torch modules and on the numpy structural stand-ins used by the tests, so the
    same proof code runs in both places.  It reads parameters only: no forward, no autograd.
    """
    out = {}
    for name in MODULES:
        try:
            module = resolve_module(controller, name)
        except SourceIdentityError:
            out[name] = None
            continue
        digest = hashlib.sha256()
        for pname, parameter in module.named_parameters():
            digest.update(pname.encode("utf-8"))
            digest.update(_parameter_bytes(parameter))
        out[name] = digest.hexdigest()
    return out


def torch_copy_backend(module, weight, bias):
    """The REAL install backend: torch copy_ into the live critic parameters (no autograd)."""
    import torch
    with torch.no_grad():
        module.weight.copy_(torch.as_tensor(np.asarray(weight), dtype=module.weight.dtype))
        module.bias.copy_(torch.as_tensor(np.asarray(bias), dtype=module.bias.dtype))
    return module


torch_copy_backend.real_backend = True


def _real_trunk_callable(controller):
    """The production trunk delegate: the controller's own inner.trunk, never an echo."""
    inner = getattr(controller, "inner", controller)
    trunk = getattr(inner, "trunk", None)
    if not callable(trunk):
        raise SourceIdentityError(
            "the controller exposes no callable inner.trunk; the provider refuses to invent a "
            "default delegate (an echo is never the real trunk). Tests inject an explicit spy.")
    import torch

    def call(matrix):
        return trunk(torch.as_tensor(np.asarray(matrix), dtype=torch.float32))

    call.real_trunk = trunk
    return call


class CalibrationProvider:
    """The real binding between a fitted calibration and the live five-module controller.

    Construction takes the REAL controller; the trunk delegate defaults to that controller's
    own ``inner.trunk`` and may only be replaced by an explicitly injected spy in tests.  The
    provider never opens a checkpoint, never restores RNG and never trains: it forwards through
    the guarded entry, pools exactly the way ``r66_frozen.frozen_early_ppo_update`` pools critic
    values (mean over the target rows of each joint decision row, AFTER the trunk), fits the
    fixed dual-output ridge, and installs the resulting affine heads into the two critic
    parameters only.
    """

    def __init__(self, controller, ledger, trunk_callable=None):
        self.controller = controller
        self.ledger = ledger
        self.calls = 0
        self.charged_rows = 0
        if trunk_callable is None:
            self._trunk = _real_trunk_callable(controller)
            self._trunk_is_spy = False
        else:
            if not callable(trunk_callable):
                raise SourceIdentityError("an injected trunk callable must be callable")
            self._trunk = trunk_callable
            self._trunk_is_spy = True

    # ---------------------------------------------------------- the single REAL trunk entry
    def trunk_forward(self, matrix, n_joint_rows=1):
        """Validate, charge the REAL rows, then delegate once.  No refund on delegate failure."""
        m = np.asarray(matrix)
        if m.ndim != 2 or m.shape[1] != FEATURE_DIM:
            raise TrunkInputError(
                f"trunk input must be a 2-D (N, {FEATURE_DIM}) matrix, got shape {m.shape}")
        n = int(m.shape[0])
        if n <= 0:
            raise TrunkInputError("trunk input must carry at least one real row")
        if n > BATCH_INPUT_ROW_CAP:
            raise BatchTooLarge(
                f"trunk input batch of {n} real rows exceeds the per-batch cap "
                f"{BATCH_INPUT_ROW_CAP}; refused BEFORE any charge")
        self.ledger.debit(n, n_joint_rows=n_joint_rows)
        self.calls += 1
        self.charged_rows += n
        return self._trunk(m)

    # ---------------------------------------------------------- pooling, exactly as R66 does
    def mean_over_targets(self, hidden, counts):
        """Mean over each joint row's target rows, AFTER the trunk (the R66 pooling order)."""
        hidden = np.asarray(hidden, dtype=np.float64)
        counts = [int(c) for c in counts]
        if hidden.ndim != 2:
            raise ValueError("hidden must be a 2-D (rows, EMBED_DIM) matrix")
        if any(c <= 0 for c in counts):
            raise ValueError("every joint row must pool at least one target row")
        if hidden.shape[0] != sum(counts):
            raise ValueError(
                f"hidden has {hidden.shape[0]} rows but the target counts sum to {sum(counts)}")
        rows, offset = [], 0
        for k in counts:
            rows.append(hidden[offset:offset + k].mean(axis=0))
            offset += k
        return np.stack(rows)

    # ---------------------------------------------------------- the fixed dual-output ridge
    def fit_dual_output_ridge(self, X, Y) -> dict:
        """One dual-output ridge per seed on the standardised embedding; configuration LOCKED."""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != EMBED_DIM:
            raise ValueError(f"X must be (rows, {EMBED_DIM}), got {X.shape}")
        if Y.ndim != 2 or Y.shape[0] != X.shape[0] or Y.shape[1] != 2:
            raise ValueError(f"Y must be (rows, 2) aligned with X, got {Y.shape}")
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma = np.where(sigma == 0.0, CONSTANT_COLUMN_SCALE, sigma)
        design = np.concatenate([(X - mu) / sigma, np.ones((X.shape[0], 1))], axis=1)
        penalty = np.eye(design.shape[1]) * RIDGE
        penalty[-1, -1] = 0.0                      # the intercept is never penalised
        theta = np.linalg.solve(design.T @ design + penalty, design.T @ Y)
        return {"mu": mu, "sigma": sigma, "A": theta[:-1, :], "c": theta[-1, :],
                "ridge": RIDGE, "rows": int(X.shape[0]),
                "center_scale_scope": CENTER_SCALE_SCOPE,
                "constant_column_scale": CONSTANT_COLUMN_SCALE,
                "intercept_penalized": INTERCEPT_PENALIZED}

    # ---------------------------------------------------------- coefficients -> real heads
    def calibrated_heads(self, fit) -> CriticHeadSet:
        """The exact standardised->raw affine transform, expressed as the two critic heads."""
        B, b = standardized_to_raw_affine(fit["mu"], fit["sigma"], fit["A"], fit["c"])
        return CriticHeadSet(affine_to_critic_heads(B, b))

    # ---------------------------------------------------------- install into the NEW model
    def install_critic_heads(self, head_set, copy_backend=None):
        """copy_ the calibrated affine into the two critic parameters, and prove the rest stayed.

        The per-module parameter digests before and after must show EXACTLY the two critic
        modules changed; trunk, early_head and late_head are byte-identical afterwards.  No
        re-scaling happens here and no RNG is drawn.
        """
        if not isinstance(head_set, CriticHeadSet):
            raise TypeError("only a seam CriticHeadSet may be installed")
        backend = copy_backend or torch_copy_backend
        before = module_parameter_digest(self.controller)
        for module_name in CRITIC_HEAD_MODULES:
            module = resolve_module(self.controller, module_name)
            backend(module, head_set.heads[module_name]["weight"],
                    head_set.heads[module_name]["bias"])
        after = module_parameter_digest(self.controller)
        changed = sorted(m for m in MODULES if before[m] != after[m])
        if not set(CRITIC_HEAD_MODULES) <= set(changed):
            raise SourceIdentityError(
                f"install did not replace both critic heads (changed={changed})")
        outside = sorted(set(changed) - set(CRITIC_HEAD_MODULES))
        if outside:
            raise SourceIdentityError(
                f"install touched modules outside the two critic heads: {outside}")
        return {"changed_modules": changed, "digest_before": before, "digest_after": after,
                "backend": getattr(backend, "__name__", repr(backend)),
                "real_backend": bool(getattr(backend, "real_backend", False)),
                "rescaled": False}

    # ---------------------------------------------------------- guarded extraction + fit
    def extract_and_fit(self, teacher_rows) -> dict:
        """Extract the teacher embeddings through the guarded trunk, pool, then fit.

        ``teacher_rows`` yields ``(matrix, counts, targets)`` per batch: the real (N,19) input
        rows, the per-joint-row target counts, and the (joint_rows, 2) Monte-Carlo task/cost
        returns aligned with the pooled rows.  Every batch is charged at the trunk entry; an
        over-cap or malformed batch is refused before any charge.
        """
        pooled, targets = [], []
        for matrix, counts, target_rows in teacher_rows:
            m = np.asarray(matrix)
            counts = [int(c) for c in counts]
            hidden = np.asarray(self.trunk_forward(m, n_joint_rows=len(counts)),
                                dtype=np.float64)
            if hidden.shape[0] != m.shape[0]:
                raise SourceIdentityError(
                    f"trunk returned {hidden.shape[0]} rows for {m.shape[0]} input rows")
            if hidden.shape[1] != EMBED_DIM:
                raise SourceIdentityError(
                    f"trunk returned width {hidden.shape[1]}, expected {EMBED_DIM}")
            pooled.append(self.mean_over_targets(hidden, counts))
            targets.append(np.asarray(target_rows, dtype=np.float64))
        if not pooled:
            raise ValueError("no teacher rows were supplied")
        return self.fit_dual_output_ridge(np.concatenate(pooled), np.concatenate(targets))

# ======================================================================================== 4. licence
def verify_r71_licence(license) -> bool:
    """The REAL licence verifier: a licence must match the registered identity, not a token.

    A future licensed run's licence would have to declare this task, this quota identity, the
    frozen ROUND68 source-freeze sha256 and the registered offline input-identity sha256.  No
    such licence exists this round, so every gate fails closed; this function is deliberately
    NOT marked test_only, and the seam's spy verifier can never satisfy it.
    """
    if not isinstance(license, RealLicense):
        return False
    if license.test_only or not license.issued_by:
        return False
    try:
        reuse = register_offline_input_reuse()
        freeze_sha = sha256_file(R68_FREEZE_PATH)
    except OSError:
        return False
    return (license.task_id == TASK_ID
            and license.quota_identity == R71_QUOTA_IDENTITY
            and license.source_freeze_sha256 == freeze_sha
            and license.input_identity_sha256 == reuse["input_identity_sha256"])


def make_license_gate(licence_mapping=None):
    """Build the fail-closed gate: no mapping -> no licence; a bad mapping -> LicenseInvalid."""
    if licence_mapping is None:
        return LicenseGate()
    return LicenseGate(RealLicense.from_mapping(licence_mapping), verify_r71_licence)


# ======================================================================================== 5. checkpoint
#: the ROUND71 freeze identity, extending the verified ROUND68 one field for field
R71_FREEZE = {
    "kind": R71_FREEZE_KIND,
    "task_id": TASK_ID, "branch": BRANCH,
    "training_version": R71_TRAINING_VERSION,
    "structure": R71_STRUCTURE,
    "loss_version": R71_LOSS_VERSION,
    "calibration_route": R71_CALIBRATION_ROUTE,
    "frozen_modules": list(FROZEN_MODULES),
    "trainable_modules": list(TRAINABLE_MODULES),
    "gae_lambda": 1.0, "eta_lambda": 0.5, "gamma": 1.0, "delta": 0.0, "lambda_init": 1.0,
    "data_order_seed": 20260927, "b0_bytes": 10000.0,
    "init_source": ("R68 ep0 model-only five modules (bit-exact from the compliant ROUND60 ep0, "
                    "late_head scaled x0.10) with the dual critic affine heads replaced by the "
                    "seed's own ridge calibration; a fresh optimizer / lambda / RNG follows"),
    "inherits_from": ("round68-late-initial-confidence-ablation (structure, loss, config, routed "
                      "update, critic->trunk isolation); only the critic head INIT values differ"),
}

_FOREIGN_SCHEMAS = ("constrained-cost-state/2+partitioned-actor-head/1",)
_FOREIGN_TAGS = ("constrained-cost-v2-partitioned-early-late-actor-head",
                 "constrained-cost-v2-frozen-early-path",
                 "constrained-cost-v2-frozen-early-path-late-init-alpha")


def r71_save_checkpoint(controller, optimizer, state: dict, path) -> str:
    """Save the ROUND71 five modules + six-parameter optimizer + frozen mask + calibration id.

    Modelled field for field on the verified ``r68_frozen.save_checkpoint`` (source manifest
    entry ``r68_identity_and_checkpoint``) with the identity block extended to the ROUND71
    contract.  The calibrated critic heads are stored as-is, so a resume loads them and never
    re-fits.  Never called this round.
    """
    import torch
    import v2_trainer as trainer
    import r66_frozen as r66
    frozen_identity = r66.freeze_early_path(controller)
    payload = {
        "schema": R71_STATE_SCHEMA, "trainer_tag": R71_TRAINER_TAG,
        "training_version": R71_TRAINING_VERSION,
        "training_version_source": trainer.TRAINING_VERSION,
        "reward_timing_version": trainer.REWARD_TIMING_VERSION,
        "loss_version": R71_LOSS_VERSION,
        "gae_schema": trainer.GAE_SCHEMA, "optimizer_schema": R71_OPTIMIZER_SCHEMA,
        "reward_timing_schema": trainer.REWARD_TIMING_SCHEMA,
        "structure": R71_STRUCTURE,
        "calibration_route": R71_CALIBRATION_ROUTE,
        "late_init_alpha": float(state.get("late_init_alpha", 0.10)),
        "freeze": R71_FREEZE,
        "controller": {key: value for key, value in controller.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "optimizer_param_groups": [len(g["params"]) for g in optimizer.state_dict()["param_groups"]],
        "frozen_mask": {"frozen_modules": list(FROZEN_MODULES),
                        "trainable_modules": list(TRAINABLE_MODULES),
                        "frozen_sha256": frozen_identity["sha256"],
                        "frozen_scalar_count": frozen_identity["scalar_count"],
                        "frozen_parameter_object_count": frozen_identity["tensor_count"],
                        "critic_heads_calibrated": True,
                        "note": "requires_grad is not serialised; the calibrated critic heads are "
                                "stored as-is; resume never re-fits or re-scales"},
        "training_state": state,
        "run_identity": state.get("run_identity"),
        "rng": trainer.rng_state(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return str(path)


def _r71_reject_foreign(path, payload: dict) -> None:
    """ROUND60/ROUND66/ROUND68 payloads are never a ROUND71 training-resume source."""
    name = Path(path).name
    if payload.get("schema") in _FOREIGN_SCHEMAS or payload.get("trainer_tag") in _FOREIGN_TAGS:
        raise RuntimeError(
            f"{name}: this is a ROUND60/ROUND66/ROUND68 checkpoint; those are model-only or "
            f"foreign-round sources and are REJECTED as a ROUND71 training-resume source")
    if payload.get("schema") != R71_STATE_SCHEMA or payload.get("trainer_tag") != R71_TRAINER_TAG:
        raise RuntimeError(f"{name}: not a ROUND71 critic-calibration checkpoint")


def _r71_validate_base(path, payload: dict) -> dict:
    _r71_reject_foreign(path, payload)
    if payload.get("loss_version") != R71_LOSS_VERSION:
        raise RuntimeError(f"{Path(path).name}: loss version mismatch")
    if payload.get("gae_schema") != "per-decision-next-value/1" or \
            payload.get("optimizer_schema") != R71_OPTIMIZER_SCHEMA:
        raise RuntimeError(f"{Path(path).name}: trainer contract schema mismatch")
    if payload.get("structure") != R71_STRUCTURE:
        raise RuntimeError(f"{Path(path).name}: structure mismatch")
    if payload.get("freeze") != R71_FREEZE:
        raise RuntimeError(f"{Path(path).name}: freeze identity mismatch; refusing to resume")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict) or (
            training_state.get("loss_version") != R71_LOSS_VERSION
            or training_state.get("state_schema") != R71_STATE_SCHEMA
            or training_state.get("optimizer_schema") != R71_OPTIMIZER_SCHEMA):
        raise RuntimeError(f"{Path(path).name}: training state version mismatch")
    return training_state


def _r71_restore_controller(payload: dict, training_state: dict):
    import r66_frozen as r66
    seed = int(training_state["training_seed"])
    controller = r66.fresh_partitioned_controller(seed, ep0_state=None, verify_ep0=False)
    saved = payload["controller"]
    if set(saved) != set(MODULES):
        raise RuntimeError("checkpoint controller modules are not the partitioned five-module schema")
    controller.load_state_dict(saved)
    identity = r66.freeze_early_path(controller)
    mask = payload.get("frozen_mask") or {}
    if mask.get("frozen_sha256") != identity["sha256"]:
        raise RuntimeError("restored controller does not match the frozen mask identity")
    return controller


def r71_load_checkpoint(path, *, run_identity=None):
    """Resume only a ROUND71 checkpoint of THIS run (payload-bound role validation)."""
    import torch
    import r66_frozen as r66
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    training_state = _r71_validate_base(path, payload)
    validate_calibration_source(
        {"identity": {"seed": int(training_state["training_seed"]),
                      "structure": payload.get("structure"),
                      "training_version": payload.get("training_version"),
                      "module_set": list(MODULES), "source_kind": "this_run"},
         "run_identity": payload.get("run_identity"),
         "carries_trained_optimizer_state": True, "carries_trained_lambda": True,
         "carries_advanced_training_rng": True,
         "from_trained_ep128": False},
        role="r71_new_full_restore", run_identity=run_identity)
    controller = _r71_restore_controller(payload, training_state)
    optimizer = r66.build_optimizer(controller, 3e-4)
    if [len(g["params"]) for g in payload["optimizer"]["param_groups"]] != [6]:
        raise RuntimeError("ROUND71 checkpoint optimizer is not the six-parameter trainable set")
    optimizer.load_state_dict(payload["optimizer"])
    import v2_trainer as trainer
    trainer.restore_rng(payload.get("rng"))
    return controller, optimizer, training_state


def r71_load_checkpoint_read_only(path, *, run_identity=None):
    """Read-only load of a ROUND71 checkpoint: weights and frozen mask only, never RNG or Adam."""
    import torch
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    training_state = _r71_validate_base(path, payload)
    controller = _r71_restore_controller(payload, training_state)
    import r66_frozen as r66
    identity = {"load_mode": "read_only", "training_version": payload.get("training_version"),
                "reward_timing_version": payload.get("reward_timing_version"),
                "optimizer_restored": False, "rng_restored": False,
                "structure": R71_STRUCTURE,
                "calibration_route": payload.get("calibration_route"),
                "frozen_mask": payload.get("frozen_mask"),
                "parameter_identity": r66.frozen_parameter_summary(controller)}
    return controller, training_state, identity


# ======================================================================================== 6. runtime
def capture_live_config(cfg) -> dict:
    """Read the CONFIGURED living values from the stood-up runtime and refuse any drift.

    Every field of LIVE_CONFIG_CONTRACT is read from the live module the trainer actually uses;
    an import default can never satisfy this check.
    """
    import sys
    common = cfg["common"]
    modules = {"r66": "r66_frozen", "r68": "r68_frozen"}
    captured, drift = {}, []
    for name, expected, _note in LIVE_CONFIG_CONTRACT:
        holder, _, attr = name.partition(".")
        if holder == "common":
            value = getattr(common, attr, None)
        else:
            module = sys.modules.get(modules[holder])
            value = getattr(module, attr, None) if module is not None else None
        captured[name] = value
        if value is None or float(value) != float(expected):
            drift.append({"field": name, "expected": expected, "captured": value})
    if drift:
        raise RuntimeError(f"ROUND71 live-config drift from the frozen contract: {drift}")
    return captured


def configure_r71_runtime(raw_root, gate, run_identity, *, model_read_approval=None):
    """Stand up the REAL ROUND68 runtime and rebind ONLY quota/version/checkpoint to ROUND71.

    The gates run BEFORE anything is imported and before any directory is created: without a
    verified real licence this raises LicenseMissing/LicenseInvalid, and without an explicit
    model-read approval it raises ModelReadBoundary.  The routed PPO update (R53 critic->trunk
    isolation + the UNCHANGED R66/R68 update math), the sequential mask, the loss and the dual
    update are left exactly as the verified ROUND68 chain installed them.
    """
    if not isinstance(gate, LicenseGate):
        raise LicenseInvalid("a LicenseGate is required to configure the runtime")
    gate.require("environment_build")          # precise fail-closed error, no side effects
    if not model_read_approval:
        raise ModelReadBoundary(
            "ROUND71 stops at the model read-in boundary: an explicit model-read approval id "
            "is required before the runtime is stood up or any model is read")
    import sys
    if str(R68_DIR) not in sys.path:
        sys.path.insert(0, str(R68_DIR))
    import r68_execute                           # the REAL executor (never imported at module load)
    cfg = r68_execute.configure_and_inject(Path(raw_root))
    trainer = cfg["trainer"]
    common = cfg["common"]
    r40_common = cfg["runtime"]["r40_common"]
    # quota: the registered ROUND71 plan identity (installed only inside a licensed run)
    r40_common.NORMAL_CAP = R71_NORMAL_CAP
    r40_common.RESERVE_CAP = R71_RESERVE_CAP
    r40_common.HARD_CAP = R71_HARD_CAP
    common.NORMAL_LAUNCHES = R71_NORMAL_CAP
    common.RESERVE_LAUNCHES = R71_RESERVE_CAP
    common.LAUNCH_CAP = R71_HARD_CAP
    r40_common.plan = lambda: {"normal_total": R71_NORMAL_CAP,
                               "technical_reserve_max": R71_RESERVE_CAP,
                               "hard_cap": R71_HARD_CAP, "task_id": TASK_ID}
    r40_common.ROUND_DIR = Path(raw_root).resolve()
    # version: the ROUND71 training-version literal on the live versions module
    try:
        cfg["runtime"]["r42_versions"].TRAINING_VERSION = R71_TRAINING_VERSION
    except Exception as exc:                     # pragma: no cover - identity must never drift
        raise SourceIdentityError(f"could not rebind the live training version: {exc}") from exc
    # checkpoint: the ROUND71 five-module contract (foreign resumes still refused)
    def r71_save(controller, optimizer, state, path, *, freeze=None):
        return r71_save_checkpoint(controller, optimizer, state, path)

    def r71_load(path, *, freeze=None):
        return r71_load_checkpoint(path, run_identity=run_identity)

    def r71_load_ro(path, *, freeze=None):
        return r71_load_checkpoint_read_only(path, run_identity=run_identity)

    ppo_update_before = trainer.ppo_update
    trainer.save_checkpoint = r71_save
    trainer.load_checkpoint = r71_load
    trainer.load_checkpoint_read_only = r71_load_ro
    if trainer.ppo_update is not ppo_update_before:
        raise SourceIdentityError(
            "the routed PPO update must stay the verified ROUND68 update (critic->trunk "
            "isolation + frozen-early-path math); ROUND71 never rebinds it")
    captured = capture_live_config(cfg)
    identity = dict(cfg["identity"])
    identity.update({
        "r71_training_version": R71_TRAINING_VERSION,
        "r71_quota": {"normal": R71_NORMAL_CAP, "reserve": R71_RESERVE_CAP,
                      "hard": R71_HARD_CAP},
        "ppo_update_unchanged": trainer.ppo_update is ppo_update_before,
        "save_is_r71": trainer.save_checkpoint is r71_save,
        "load_is_r71": trainer.load_checkpoint is r71_load,
        "load_ro_is_r71": trainer.load_checkpoint_read_only is r71_load_ro,
        "live_config": captured,
        "run_identity": run_identity.as_dict(),
        "model_read_approval": model_read_approval,
    })
    cfg["r71"] = {"identity": identity, "run_identity": run_identity.as_dict(),
                  "freeze": R71_FREEZE}
    return cfg


# ======================================================================================== 7. init
def build_calibrated_initialisation(*, seed, gate, model_read_approval, teacher_rows,
                                    run_identity=None, copy_backend=None,
                                    ledger_state_path=None) -> dict:
    """The full candidate initialisation, end to end, behind the real gates.

    NEVER CALLED THIS ROUND.  Without a verified real licence it raises LicenseMissing /
    LicenseInvalid, and without an explicit model-read approval it raises ModelReadBoundary
    BEFORE any .pt is opened.  The body is the real sequence a licensed run would execute:

      1. payload-bound role validation of the ROUND68 ep0 model-only INIT source (registered
         file/parameter/cursor identity; no optimizer, lambda or training RNG restored);
      2. build the ROUND68 initial controller through the verified chain (ROUND60 ep0 five
         modules bit-exact, early path frozen, late_head scaled x0.10 once);
      3. verify the frozen mask and the registered post-scale init identity;
      4. guarded teacher extraction through the REAL trunk (ForwardLedger, 1024 guard);
      5. the fixed dual-output ridge on the mean-over-targets representation;
      6. install the calibrated critic heads into the model (trunk/early/late proven untouched);
      7. only THEN the fresh six-parameter optimizer and the fresh training state carrying the
         ROUND71 version contract (fresh lambda, frozen initial random state).
    """
    if not isinstance(gate, LicenseGate):
        raise LicenseInvalid("a LicenseGate is required for the initialisation")
    gate.require("fit_real_data")
    gate.require("model_forward")
    gate.require("checkpoint_write")
    if not model_read_approval:
        raise ModelReadBoundary(
            "build_calibrated_initialisation stops at the model read-in boundary: an explicit "
            "model-read approval is required before the ep0 model is read")
    import sys
    for directory in (R66_DIR, R68_DIR):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import r66_frozen as r66
    import r68_frozen as r68

    # 1+2: the verified ROUND68 initial controller (model-only ROUND60 ep0 + late scale)
    controller, ep0, frozen_identity, post, scale_identity = r68.fresh_r68_controller(int(seed))
    init_payload = {
        "identity": {"seed": int(seed), "structure": SEAM_R68_STRUCTURE,
                     "training_version": R68_TRAINING_VERSION_EXPECTED,
                     "module_set": list(MODULES), "source_kind": "r68_ep0"},
        "file_sha256": ep0["file_sha256"],
        "parameter_sha256": ep0["parameter_sha256"],
        "post_scale_parameter_sha256": post["parameter_sha256"],
        "pre_scale_equals_r60_ep0": scale_identity["pre_scale_equals_r60_ep0"],
        "episodes_done": ep0["episodes_done"], "updates": ep0["updates"],
        "load_mode": ep0["load_mode"],
        "optimizer_restored": bool(ep0["optimizer_restored"]),
        "lambda_restored": False, "rng_restored": bool(ep0["rng_restored"]),
        "from_trained_ep128": False}
    init_role = validate_calibration_source(init_payload, role="r68_ep0_model_only_init")

    # 3: frozen mask + registered post-scale identity
    mask = r66.assert_frozen_mask(controller)
    if not scale_identity["pre_scale_equals_r60_ep0"]:
        raise SourceIdentityError(f"seed {seed}: pre-scale parameters are not the ROUND60 ep0")

    # 4-6: guarded extraction, fixed ridge, install into the two critic parameters only
    ledger = ForwardLedger(state_path=ledger_state_path)
    provider = CalibrationProvider(controller, ledger)
    fit = provider.extract_and_fit(teacher_rows)
    head_set = provider.calibrated_heads(fit)
    install = provider.install_critic_heads(head_set, copy_backend=copy_backend)
    if not install["real_backend"] and copy_backend is None:
        raise SourceIdentityError("the real install backend was not used")

    # 7: fresh optimizer and fresh training state with the ROUND71 version contract
    optimizer = r66.build_optimizer(controller, 3e-4)
    state = r68.fresh_training_state(
        int(seed), gae_lambda=_EXPECTED_LIVE["common.GAE_LAMBDA"],
        eta_lambda=_EXPECTED_LIVE["common.ETA_LAMBDA"], gamma=_EXPECTED_LIVE["common.GAMMA"],
        delta=0.0, lambda_init=1.0,
        data_order_seed=_EXPECTED_LIVE["common.DATA_ORDER_SEED"],
        b0_bytes=_EXPECTED_LIVE["common.B0_BYTES"],
        alpha=_EXPECTED_LIVE["r68.LATE_INIT_ALPHA"])
    state["training_version"] = R71_TRAINING_VERSION
    state["loss_version"] = R71_LOSS_VERSION
    state["state_schema"] = R71_STATE_SCHEMA
    state["optimizer_schema"] = R71_OPTIMIZER_SCHEMA
    state["structure"] = R71_STRUCTURE
    state["calibration_route"] = R71_CALIBRATION_ROUTE
    state["run_identity"] = run_identity.as_dict() if run_identity is not None else None
    state["critic_heads_calibration"] = {
        "ridge": fit["ridge"], "rows": fit["rows"],
        "joint_rows": ledger.joint_rows, "input_rows_charged": ledger.rows,
        "changed_modules": install["changed_modules"],
        "digest_before": install["digest_before"], "digest_after": install["digest_after"]}
    return {"seed": int(seed), "controller": controller, "optimizer": optimizer,
            "state": state, "init_role": init_role, "fit": fit, "install": install,
            "frozen_mask": mask, "post_scale_identity": post,
            "ledger": {"rows": ledger.rows, "calls": ledger.calls,
                       "joint_rows": ledger.joint_rows, "remaining": ledger.remaining()}}


# ======================================================================================== 8. coverage
def runtime_binding_coverage() -> dict:
    """What the real runtime binding covers this round, and what still awaits approval."""
    return {
        "real_source_binding": {
            "covered": ("read-only source manifest with live SHA256 identities and static "
                        "function/line resolution for the canonical trainer/common/controller, "
                        "the frozen PPO primitives, the partitioned controller + frozen-early-"
                        "path PPO update, the critic->trunk detach, the ROUND68 identity/"
                        "checkpoint contract, the ROUND68 executor (claim/slot/quota/permit), "
                        "the ROUND58 base-runner seams and the evaluation seam; the ROUND60 "
                        "ep0/ep128 model sources are registered by expected identity only and "
                        "no .pt is opened"),
            "verified_by": "verify_source_manifest (recompute + drift + freeze-literal + reuse)",
        },
        "provider_binding": {
            "covered": ("the real trunk entry (N,19) with the 1024 guard and the ForwardLedger "
                        "debit BEFORE the forward; mean-over-targets pooling to joint rows in "
                        "the R66 order (after the trunk, per joint row); the fixed dual-output "
                        "ridge (1e-3, all-row centre/scale, constant column 1, intercept "
                        "unpenalised); the exact standardised->raw affine transform to the two "
                        "Linear(64,1) critic heads; copy_ installation into ONLY the two critic "
                        "parameters with before/after per-module digests proving trunk/early/"
                        "late untouched; no re-scale and no RNG draw at install"),
            "not_covered": ("no real trunk forward is run and no real model is constructed this "
                            "round; the production delegate is the controller's own inner.trunk "
                            "and tests inject an explicit spy - an echo delegate does not exist "
                            "in production"),
        },
        "payload_role_validation": {
            "covered": ("payload-bound roles: R68 ep0 model-only INIT (registered file/parameter/"
                        "cursor identity, exact R68 version literal, no optimizer/lambda/RNG), "
                        "R68 ep128 read-only CONTROL (refused as INIT), this run's NEW full "
                        "restore (exact ROUND71 version, live run-identity block equality, "
                        "trained state present), rule control read-only; a declared source_kind "
                        "or a prefix-only version is never sufficient"),
        },
        "runtime_and_cli": {
            "covered": ("configure_r71_runtime stands up the REAL runtime through "
                        "r68_execute.configure_and_inject and rebinds ONLY quota/version/"
                        "checkpoint save-load to the ROUND71 contract, leaving the routed PPO "
                        "update, mask, loss and dual update untouched; live-config capture "
                        "refuses drift from the frozen fields; the CLI validates licence, phase, "
                        "model-read approval, source/input identity and quota BEFORE any model "
                        "load and fails closed with no raw-root side effects"),
            "not_covered": ("no licence exists this round, so no runtime is ever stood up; the "
                            "model read-in point is a controlled stop boundary and every phase "
                            "except binding-acceptance fails closed"),
        },
        "real_gates_awaiting_approval": [
            "real initialisation equivalence (calibrated critic heads vs the R68 ep0 heads)",
            "real gradient gate (critic->trunk isolation with the calibrated heads)",
            "real restore/resume gate (fresh optimizer, lambda and RNG; foreign resumes refused)",
            "the actual offline ridge fit and teacher extraction on real data",
            "first-batch gate, per-seed 128-episode training and the fixed-ep128 evaluations",
        ],
    }


def binding_acceptance() -> dict:
    """Zero-consumption acceptance of the real runtime binding (static + live attributes).

    Verifies the source manifest against the live tree, proves the registered functions exist
    on the LIVE modules (import only: no model, no forward, no fit, no checkpoint), and
    demonstrates the fail-closed gates.  Writes nothing and consumes nothing.
    """
    receipt = {"kind": "round71-real-runtime-binding/acceptance/1", "task_id": TASK_ID,
               "branch": BRANCH, "consumption": 0}
    verification = verify_source_manifest()
    receipt["source_manifest"] = {
        "status": verification["status"],
        "source_entries": len(build_source_manifest()["entries"]),
        "checks": len(verification["checks"]),
        "drift": verification["drift"],
        "offline_input_reuse": verification["offline_input_reuse"]}
    # live module attribute presence (import only; r68_frozen prints on import by design)
    import io
    import contextlib
    import sys
    if str(R68_DIR) not in sys.path:
        sys.path.insert(0, str(R68_DIR))
    if str(R66_DIR) not in sys.path:
        sys.path.insert(0, str(R66_DIR))
    if str(R53_DIR) not in sys.path:
        sys.path.insert(0, str(R53_DIR))
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        import r68_execute
        import r68_frozen
        import r66_frozen as r66
        from round53_trial_adapter import critic_input_detached, install_on_verified_trainer
    receipt["live_modules"] = {
        "imported": ["r68_execute", "r68_frozen", "r66_frozen", "round53_trial_adapter"],
        "import_stdout": buffer.getvalue().strip(),
        "attributes": {
            "r68_execute.configure_and_inject": callable(r68_execute.configure_and_inject),
            "r68_frozen.save_checkpoint": callable(r68_frozen.save_checkpoint),
            "r68_frozen.load_checkpoint": callable(r68_frozen.load_checkpoint),
            "r66_frozen.frozen_early_ppo_update": callable(r66.frozen_early_ppo_update),
            "r66_frozen.load_r60_model": callable(r66.load_r60_model),
            "r66_frozen.freeze_early_path": callable(r66.freeze_early_path),
            "round53_trial_adapter.critic_input_detached": callable(critic_input_detached),
            "round53_trial_adapter.install_on_verified_trainer":
                callable(install_on_verified_trainer),
        },
        "r68_training_version_literal": r68_frozen.R68_TRAINING_VERSION ==
        R68_TRAINING_VERSION_EXPECTED,
        "seam_training_version_literal": SEAM_R68_TRAINING_VERSION == R68_TRAINING_VERSION_EXPECTED,
        "seam_structure_literal": SEAM_R68_STRUCTURE ==
        "partitioned-early-late-actor-head/1+frozen-early-path-late-adaptation/1",
    }
    # fail-closed demonstrations (no runtime is stood up, nothing is created)
    gates = {}
    try:
        configure_r71_runtime(Path("/nonexistent-r61-raw-root"), LicenseGate(), None)
        gates["no_licence"] = "UNEXPECTEDLY PROCEEDED"
    except LicenseMissing as exc:
        gates["no_licence"] = f"LicenseMissing: {exc}"
    except Exception as exc:                      # pragma: no cover - reported, not raised
        gates["no_licence"] = f"{type(exc).__name__}: {exc}"
    test_only = LicenseGate(RealLicense(approval_id="test", task_id=TASK_ID,
                                        source_freeze_sha256="x", input_identity_sha256="y",
                                        quota_identity=R71_QUOTA_IDENTITY, issued_by="test",
                                        test_only=True), spy_license_verifier)
    try:
        configure_r71_runtime(Path("/nonexistent-r61-raw-root"), test_only, None)
        gates["test_only_licence"] = "UNEXPECTEDLY PROCEEDED"
    except (LicenseInvalid, LicenseMissing) as exc:
        gates["test_only_licence"] = f"{type(exc).__name__}: refused"
    except Exception as exc:                      # pragma: no cover - reported, not raised
        gates["test_only_licence"] = f"{type(exc).__name__}: {exc}"
    receipt["fail_closed"] = gates
    receipt["status"] = ("passed" if verification["status"] == "passed"
                         and all(v is True for v in receipt["live_modules"]["attributes"].values())
                         and receipt["live_modules"]["r68_training_version_literal"]
                         and receipt["live_modules"]["seam_training_version_literal"]
                         and "UNEXPECTEDLY" not in json.dumps(gates)
                         else "failed")
    return receipt
