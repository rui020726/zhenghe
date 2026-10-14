# -*- coding: utf-8 -*-
"""ROUND73 pure-cost actor ablation: identity + licence binding.

ROUND73 is an S3 development diagnostic on the verified ROUND71 runtime.  Its ONE
structural intervention is the actor objective:

    CONTROL (ROUND71, verified): A = lambda_dual * A_task - A_cost
    NEW      (ROUND73):          A = -A_cost

Everything else is inherited field-for-field from the accepted ROUND71 living
configuration (dual critics, GAE lambda=1.0, reward timing, sequential budget mask,
entropy bonus, global clip, dual-lambda update eta=0.5, frozen trunk + early_head,
trainable late_head + both critic heads, five-module checkpoint, 19-dim observation,
one-shot SEND/SKIP, 6 goals, 29300B budget, ACK off, seeds 11/22/33 and their
128-episode order, greedy tie rule and sampled eval RNG scheme).

This module binds the human authorization (archived in ROUND73_EXECUTION_BASIS.md,
machine-readable in ROUND73_LICENCE.json) to the registered identities and nothing
else.  Importing it loads no model, opens no .pt and runs no forward.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_ROOT = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
READINESS_DIR = REPO / "analysis" / "round71-critic-calibration-readiness"
R71_DIR = RUN_ROOT / "round71_critic_initialization_calibration"
for _d in (str(READINESS_DIR),):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import r71_runtime_binding as readiness                        # noqa: E402
import r73_pure_cost_objective as r73                          # noqa: E402

# ------------------------------------------------------------------ trial identity ----
R73_TRIAL_TASK_ID = "ROUND73-PURE-COST-ACTOR-ABLATION"
R73_TRIAL_BRANCH = "exp/round73-pure-cost-actor-ablation"
TRIAL_APPROVAL_ID = "ROUND73-APPROVED-PURE-COST-ACTOR-FIXED-TRIAL"
TRIAL_RUN_ID = "r73-pure-cost-actor-20261014"
MODEL_READ_APPROVAL = TRIAL_APPROVAL_ID
LICENCE_PATH = HERE / "ROUND73_LICENCE.json"
EXECUTION_BASIS_PATH = HERE / "ROUND73_EXECUTION_BASIS.md"
R71_CHECKPOINT_INDEX_PATH = R71_DIR / "ROUND71_CHECKPOINT_INDEX.json"

# the trial's own output contract (R68/R71 runtime, ROUND73 identity)
ROW_TAG = "r73train"                    # training rows / tapes
EVAL_ROW_TAG = "r73eval"                # evaluation rows
MODEL_ARMS = ("NEW_EP128",)             # the 192-cell block evaluates the trained arm only
FOCAL_SCENE = 1_700_027

#: registered (NOT authorised) plan identity of THIS round; any licensed run is counted
#: separately and in full.  ROUND73 uses a fresh independent quota and borrows nothing
#: from R60/R61/R71.
R73_QUOTA_IDENTITY = ("round73-pure-cost-actor/plan-579-normal+21-reserve-hard-600/"
                      "offline-6000-input-rows-synth-8/1")


def verify_r73_trial_licence(license) -> bool:
    """The REAL trial verifier: the archived authorization, bound to the registered identities.

    A licence counts only when it declares the ROUND73 approval id and task id, the
    registered ROUND73 quota identity (579 normal + 21 reserve, hard 600), the frozen
    ROUND68 source-freeze sha256 (the runtime ROUND73 inherits) and the ROUND71
    checkpoint-index sha256 (the model source ROUND73 trains from), is not test-only,
    and names its issuer.  A raw token, a test-only spy licence and any other round's
    licence are all refused.
    """
    if not isinstance(license, readiness.RealLicense):
        return False
    if license.test_only or not license.issued_by:
        return False
    try:
        freeze_sha = readiness.sha256_file(readiness.R68_FREEZE_PATH)
        model_source_sha = readiness.sha256_file(R71_CHECKPOINT_INDEX_PATH)
    except OSError:
        return False
    return (license.approval_id == TRIAL_APPROVAL_ID
            and license.task_id == R73_TRIAL_TASK_ID
            and license.quota_identity == R73_QUOTA_IDENTITY
            and license.source_freeze_sha256 == freeze_sha
            and license.input_identity_sha256 == model_source_sha)


def make_trial_license_gate(licence_mapping=None):
    """Fail-closed gate: no mapping -> no licence; a bad mapping -> LicenseInvalid."""
    if licence_mapping is None:
        return readiness.LicenseGate()
    return readiness.LicenseGate(
        readiness.RealLicense.from_mapping(licence_mapping), verify_r73_trial_licence)


def load_trial_licence(path=None) -> dict:
    """Read the declared licence mapping (the archived authorization, machine-readable)."""
    path = Path(path) if path else LICENCE_PATH
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise RuntimeError("a licence must be a declared mapping")
    return mapping


def trial_gate(path=None):
    """The live trial gate from the archived licence (fails closed without it)."""
    return make_trial_license_gate(load_trial_licence(path))


def trial_run_identity(run_id: str = TRIAL_RUN_ID):
    """The identity every ROUND73 payload carries: trial task/branch + registered evidence."""
    return readiness.R71RunIdentity(
        run_id=str(run_id),
        freeze_sha256=readiness.sha256_file(readiness.R68_FREEZE_PATH),
        input_identity_sha256=readiness.sha256_file(R71_CHECKPOINT_INDEX_PATH),
        quota_identity=R73_QUOTA_IDENTITY,
        task_id=R73_TRIAL_TASK_ID,
        branch=R73_TRIAL_BRANCH,
        training_version=r73.R73_TRAINING_VERSION)


def licence_receipt() -> dict:
    """Zero-consumption proof that the archived licence opens the gate and lookalikes do not."""
    out = {"kind": "round73-trial-licence-receipt/1", "task_id": R73_TRIAL_TASK_ID,
           "approval_id": TRIAL_APPROVAL_ID, "consumption": 0}
    good = trial_gate()
    out["archived_licence_opens_gate"] = bool(good.present())
    out["gate_requires_environment_build"] = _require_ok(good, "environment_build")
    out["no_licence_fails_closed"] = _require_refuses(make_trial_license_gate, "environment_build")
    tampered = load_trial_licence()
    tampered = dict(tampered, task_id="ROUND71-APPROVED-CALIBRATION-FIXED-TRIAL")
    out["foreign_task_id_refused"] = _require_refuses(
        lambda: make_trial_license_gate(tampered), "environment_build")
    wrong_quota = load_trial_licence()
    wrong_quota = dict(wrong_quota, quota_identity="round73/plan-999-normal/1")
    out["foreign_quota_identity_refused"] = _require_refuses(
        lambda: make_trial_license_gate(wrong_quota), "environment_build")
    raw = load_trial_licence()
    raw = dict(raw, test_only=True)
    out["test_only_refused"] = _require_refuses(
        lambda: make_trial_license_gate(raw), "environment_build")
    missing = load_trial_licence()
    missing = dict(missing)
    missing.pop("input_identity_sha256")
    out["missing_field_refused"] = _require_refuses(
        lambda: make_trial_license_gate(missing), "environment_build")
    out["status"] = "passed" if all(v is True for k, v in out.items()
                                    if isinstance(v, bool)) else "failed"
    return out


def _require_ok(gate, action) -> bool:
    try:
        gate.require(action)
        return True
    except Exception:
        return False


def _require_refuses(gate_or_factory, action) -> bool:
    # A factory is required whenever construction itself must fail closed: the gate
    # verifies its licence at construction, so a tampered mapping raises there.
    try:
        gate = gate_or_factory() if callable(gate_or_factory) else gate_or_factory
        gate.require(action)
        return False
    except Exception:
        return True
