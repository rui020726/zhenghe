#!/usr/bin/env python
"""ROUND71 approved fixed trial: identity + licence binding.

The readiness branch (``analysis/round71-critic-calibration-readiness``) registered the
candidate plan and built the zero-consumption guards; the human authorization of 2026-10-13
(archived verbatim in ``ROUND71_TRIAL_EXECUTION_BASIS.md``, declared in
``ROUND71_TRIAL_LICENCE.json``) licenses THIS fixed trial.  This module binds that
authorization to the registered identities and nothing else:

  * the frozen ROUND68 source-freeze sha256 (the trial inherits the ROUND68 runtime);
  * the registered offline input-count identity (the teacher-data evidence);
  * the registered quota identity (582 normal + 18 reserve, hard 600; 72,000 offline rows;
    16 synthetic optimizer steps);
  * the trial task id and branch.

Every gated action still fails closed without a licence that satisfies
``verify_r71_trial_licence``; the readiness verifier is untouched and still refuses the
readiness task.  Importing this module loads no model, opens no .pt and runs no forward.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
READINESS_DIR = REPO / "analysis" / "round71-critic-calibration-readiness"
if str(READINESS_DIR) not in sys.path:
    sys.path.insert(0, str(READINESS_DIR))

import r71_critic_calibration_seam as seam                      # noqa: E402
import r71_runtime_binding as readiness                         # noqa: E402

# ------------------------------------------------------------------ trial identity ----
R71_TRIAL_TASK_ID = "ROUND71-APPROVED-CALIBRATION-FIXED-TRIAL"
R71_TRIAL_BRANCH = "exp/round71-critic-initialization-calibration"
TRIAL_APPROVAL_ID = "ROUND71-APPROVED-CALIBRATION-FIXED-TRIAL"
TRIAL_RUN_ID = "r71-trial-20261013"
MODEL_READ_APPROVAL = TRIAL_APPROVAL_ID
LICENCE_PATH = HERE / "ROUND71_TRIAL_LICENCE.json"
EXECUTION_BASIS_PATH = HERE / "ROUND71_TRIAL_EXECUTION_BASIS.md"

# the trial's own output contract (R68 runtime, R71 identity)
ROW_TAG = "r71train"                    # training rows / tapes
EVAL_ROW_TAG = "r71eval"                # evaluation rows
MODEL_ARMS = ("NEW_INIT", "NEW_EP128")
FOCAL_SCENE = 1_700_027


def verify_r71_trial_licence(license) -> bool:
    """The REAL trial verifier: the archived authorization, bound to the registered identities.

    A licence counts only when it declares the trial approval id, the trial task id, the
    registered quota identity, the frozen ROUND68 source-freeze sha256 and the registered
    offline input-count identity, is not test-only, and names its issuer.  A raw token, a
    test-only spy licence and the readiness task's licence are all refused.
    """
    if not isinstance(license, readiness.RealLicense):
        return False
    if license.test_only or not license.issued_by:
        return False
    try:
        reuse = readiness.register_offline_input_reuse()
        freeze_sha = readiness.sha256_file(readiness.R68_FREEZE_PATH)
    except OSError:
        return False
    return (license.approval_id == TRIAL_APPROVAL_ID
            and license.task_id == R71_TRIAL_TASK_ID
            and license.quota_identity == readiness.R71_QUOTA_IDENTITY
            and license.source_freeze_sha256 == freeze_sha
            and license.input_identity_sha256 == reuse["input_identity_sha256"])


def make_trial_license_gate(licence_mapping=None):
    """Fail-closed gate: no mapping -> no licence; a bad mapping -> LicenseInvalid."""
    if licence_mapping is None:
        return readiness.LicenseGate()
    return readiness.LicenseGate(
        readiness.RealLicense.from_mapping(licence_mapping), verify_r71_trial_licence)


def load_trial_licence(path=None) -> dict:
    """Read the declared licence mapping (the archived authorization, machine-readable)."""
    path = Path(path) if path else LICENCE_PATH
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise seam.LicenseInvalid("a licence must be a declared mapping")
    return mapping


def trial_gate(path=None):
    """The live trial gate from the archived licence (fails closed without it)."""
    return make_trial_license_gate(load_trial_licence(path))


def trial_run_identity(run_id: str = TRIAL_RUN_ID):
    """The identity every ROUND71 payload carries: trial task/branch + registered evidence."""
    reuse = readiness.register_offline_input_reuse()
    return readiness.R71RunIdentity(
        run_id=str(run_id),
        freeze_sha256=readiness.sha256_file(readiness.R68_FREEZE_PATH),
        input_identity_sha256=reuse["input_identity_sha256"],
        quota_identity=readiness.R71_QUOTA_IDENTITY,
        task_id=R71_TRIAL_TASK_ID,
        branch=R71_TRIAL_BRANCH,
        training_version=readiness.R71_TRAINING_VERSION)


def licence_receipt() -> dict:
    """Zero-consumption proof that the archived licence opens the gate and lookalikes do not."""
    out = {"kind": "round71-trial-licence-receipt/1", "task_id": R71_TRIAL_TASK_ID,
           "approval_id": TRIAL_APPROVAL_ID, "consumption": 0}
    good = trial_gate()
    out["archived_licence_opens_gate"] = bool(good.present())
    out["gate_requires_environment_build"] = _require_ok(good, "environment_build")
    out["no_licence_fails_closed"] = _require_refuses(make_trial_license_gate, "environment_build")
    tampered = load_trial_licence()
    tampered = dict(tampered, task_id="ROUND71-REAL-RUNTIME-BINDING-ZERO-EXEC")
    out["foreign_task_id_refused"] = _require_refuses(
        lambda: make_trial_license_gate(tampered), "environment_build")
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