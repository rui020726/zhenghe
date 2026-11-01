#!/usr/bin/env python
"""Narrow ROUND71 real-runtime-binding tests: STATIC / SPY / PURE ARITHMETIC only.

Zero consumption by construction: no real or synthetic neural forward (the trunk is a spy), no
optimizer step, no autograd, no real-data fit, no model load, no .pt open, no RNG restore, no
checkpoint write, no environment and no episode.  The only file reads are registered JSON
evidence and source hashes; the only writes are private scratch receipts.  A spy here is NOT a
real update and proves nothing about a licensed run.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_r71_readiness import _ephemeral, _scratch_dir  # noqa: E402

import r71_runtime_binding as binding  # noqa: E402
import r71_run_calibration as cli  # noqa: E402
from r71_critic_calibration_seam import (  # noqa: E402
    EMBED_DIM, MODULES, BudgetExceeded, ForwardLedger, LicenseInvalid, LicenseMissing,
    RealLicense, BatchTooLarge, TrunkInputError)
from r71_runtime_binding import (  # noqa: E402
    CalibrationProvider, ModelReadBoundary, RoleValidationError, SourceIdentityError,
    R71_TRAINING_VERSION, R68_TRAINING_VERSION_EXPECTED, SEAM_R68_STRUCTURE,
    registered_run_identity, register_offline_input_reuse, runtime_binding_coverage,
    torch_copy_backend, validate_calibration_source, verify_source_manifest)


# ------------------------------------------------------------------ structural stand-ins
class _FakeModule:
    def __init__(self, weight, bias):
        self.weight = np.array(weight, dtype=np.float64)
        self.bias = np.array(bias, dtype=np.float64)

    def named_parameters(self):
        return [("weight", self.weight), ("bias", self.bias)]


class _FakeInner:
    def __init__(self):
        rng = np.random.default_rng(5)
        self.trunk = _FakeModule(rng.normal(size=(EMBED_DIM, 19)), rng.normal(size=(EMBED_DIM,)))
        self.early_head = _FakeModule(rng.normal(size=(2, EMBED_DIM)), rng.normal(size=(2,)))
        self.late_head = _FakeModule(rng.normal(size=(2, EMBED_DIM)), rng.normal(size=(2,)))
        self.value_head = _FakeModule(rng.normal(size=(1, EMBED_DIM)), rng.normal(size=(1,)))
        self.cost_head = _FakeModule(rng.normal(size=(1, EMBED_DIM)), rng.normal(size=(1,)))


class _FakeController:
    def __init__(self):
        self.inner = _FakeInner()
        self.cost_head = self.inner.cost_head


def _numpy_copy_backend(module, weight, bias):
    """Spy install backend: the same in-place contract as the real torch copy_ backend."""
    module.weight[...] = np.asarray(weight, dtype=module.weight.dtype)
    module.bias[...] = np.asarray(bias, dtype=module.bias.dtype)
    return module


def _real_licence_mapping():
    reuse = register_offline_input_reuse()
    return {"approval_id": "r71-test-approval", "task_id": binding.TASK_ID,
            "source_freeze_sha256": binding.sha256_file(binding.R68_FREEZE_PATH),
            "input_identity_sha256": reuse["input_identity_sha256"],
            "quota_identity": binding.R71_QUOTA_IDENTITY, "issued_by": "test-issuer"}


# ------------------------------------------------------------------ 1. source manifest
def test_source_manifest_matches_the_live_tree():
    verification = verify_source_manifest()
    assert verification["status"] == "passed", verification["drift"]
    assert len(verification["checks"]) >= 20
    # the canonical live trainer/common/controller must equal the frozen EXPECTED_* constants
    src = (binding.R66_DIR / "r66_frozen.py").read_text(encoding="utf-8")
    import re
    expected = dict(re.findall(r"^(EXPECTED_[A-Z0-9_]+)\s*=\s*\"([0-9a-f]{64})\"", src, re.M))
    assert expected, "the frozen EXPECTED_* constants were not found"
    manifest = binding.build_source_manifest()
    by_role = {e["role"]: e for e in manifest["entries"]}
    for role, key in (("canonical_trainer", "EXPECTED_TRAINER_SHA256"),
                      ("canonical_common", "EXPECTED_COMMON_SHA256"),
                      ("canonical_controller", "EXPECTED_CONTROLLER_SHA256")):
        if key in expected:
            assert by_role[role]["sha256"] == expected[key], role


def test_manifest_registers_model_sources_without_opening_them():
    manifest = binding.build_source_manifest()
    sources = [e for e in manifest["entries"] if e["role"].startswith("r60_")]
    assert len(sources) == 6
    for entry in sources:
        assert entry["exists"], entry["role"]
        assert "sha256" not in entry, "a .pt model source must never be hashed by the manifest"
        assert len(entry["expected_file_sha256"]) == 64
        assert entry["expected_episodes_done"] in (0, 128)


def test_r71_version_extends_r68_and_matches_the_freeze_json():
    freeze = json.loads((binding.R68_DIR / "ROUND68_FREEZE.json").read_text(encoding="utf-8"))
    assert R68_TRAINING_VERSION_EXPECTED == freeze["training_version"]
    assert R71_TRAINING_VERSION.startswith(R68_TRAINING_VERSION_EXPECTED)
    assert R71_TRAINING_VERSION.endswith("+r71-critic-calibration-init/1")
    assert R71_TRAINING_VERSION != R68_TRAINING_VERSION_EXPECTED
    assert binding.R71_STRUCTURE.endswith("+r71-critic-calibration-heads/1")


def test_offline_input_reuse_registration():
    reuse = register_offline_input_reuse()
    assert reuse["joint_decision_rows"] == 32046
    assert reuse["target_level_19d_input_rows"] == 58928
    assert len(reuse["input_identity_sha256"]) == 64
    assert "never re-extracted" in reuse["reuse_rule"]
    identity = registered_run_identity("r71-test-run")
    assert identity.matches_block(identity.as_dict())
    assert not identity.matches_block({"run_id": "someone-else"})


# ------------------------------------------------------------------ 2. pooling and ridge
def test_pooling_matches_the_r66_mean_over_targets_order():
    provider = CalibrationProvider(_FakeController(), ForwardLedger(),
                                   trunk_callable=lambda m: m)
    rng = np.random.default_rng(11)
    hidden = rng.normal(size=(7, EMBED_DIM))
    counts = [1, 3, 3]
    pooled = provider.mean_over_targets(hidden, counts)
    assert pooled.shape == (3, EMBED_DIM)
    assert np.allclose(pooled[0], hidden[0])
    assert np.allclose(pooled[1], hidden[1:4].mean(axis=0))
    assert np.allclose(pooled[2], hidden[4:7].mean(axis=0))
    for bad, counts_bad in ((hidden, [2, 2]), (hidden, [0, 7])):
        try:
            provider.mean_over_targets(bad, counts_bad)
            raise AssertionError("a malformed pooling request must be refused")
        except ValueError:
            pass


def test_ridge_roundtrip_on_synthetic_data():
    rng = np.random.default_rng(20261013)
    B_true = rng.normal(size=(EMBED_DIM, 2)) * 0.1
    b_true = rng.normal(size=2) * 0.1
    X = rng.normal(size=(256, EMBED_DIM))
    Y = X @ B_true + b_true
    provider = CalibrationProvider(_FakeController(), ForwardLedger(),
                                   trunk_callable=lambda m: m)
    fit = provider.fit_dual_output_ridge(X, Y)
    assert fit["rows"] == 256 and fit["ridge"] == 1e-3
    heads = provider.calibrated_heads(fit)
    X_hold = rng.normal(size=(32, EMBED_DIM))
    got = (X_hold @ heads.heads["value_task_head"]["weight"].T
           + heads.heads["value_task_head"]["bias"])
    want = X_hold @ B_true + b_true
    assert np.abs(got[:, 0] - want[:, 0]).max() < 0.05
    # a constant column must get scale 1, never a division by zero
    Xc = X.copy(); Xc[:, 3] = 0.5
    fit_c = provider.fit_dual_output_ridge(Xc, Y)
    assert fit_c["sigma"][3] == 1.0


def test_install_critic_heads_touches_only_the_two_critic_modules():
    controller = _FakeController()
    provider = CalibrationProvider(controller, ForwardLedger(),
                                   trunk_callable=lambda m: m)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(64, EMBED_DIM))
    Y = rng.normal(size=(64, 2))
    head_set = provider.calibrated_heads(provider.fit_dual_output_ridge(X, Y))
    before = binding.module_parameter_digest(controller)
    install = provider.install_critic_heads(head_set, copy_backend=_numpy_copy_backend)
    after = binding.module_parameter_digest(controller)
    assert install["changed_modules"] == ["value_cost_head", "value_task_head"]
    for module in ("trunk", "early_head", "late_head"):
        assert before[module] == after[module], module
    assert after["value_task_head"] != before["value_task_head"]
    assert after["value_cost_head"] != before["value_cost_head"]
    assert np.allclose(controller.inner.value_head.weight,
                       head_set.heads["value_task_head"]["weight"])
    assert np.allclose(controller.cost_head.bias, head_set.heads["value_cost_head"]["bias"])
    assert install["rescaled"] is False
    # the DEFAULT backend is the real torch one, and a spy is never silently promoted
    assert torch_copy_backend.real_backend is True
    assert install["real_backend"] is False
    # only a seam head set may be installed
    try:
        provider.install_critic_heads({"value_task_head": {}}, copy_backend=_numpy_copy_backend)
        raise AssertionError("a raw dict must not be installable")
    except TypeError:
        pass


def test_default_trunk_delegate_is_never_an_echo():
    ledger = ForwardLedger()
    try:
        CalibrationProvider(object(), ledger)
        raise AssertionError("a controller without a trunk must be refused, not echoed")
    except SourceIdentityError:
        pass
    spy = lambda m: m
    provider = CalibrationProvider(_FakeController(), ledger, trunk_callable=spy)
    assert provider._trunk is spy


# ------------------------------------------------------------------ 3. the guarded trunk entry
def test_trunk_entry_guards_charge_and_delegate_exactly_once():
    ledger = ForwardLedger()
    seen = []

    def spy(matrix):
        seen.append(np.array(matrix))
        return np.zeros((matrix.shape[0], EMBED_DIM))

    provider = CalibrationProvider(_FakeController(), ledger, trunk_callable=spy)
    for bad in (np.zeros((3, 5)), np.zeros((0, 19)), np.zeros(19)):
        try:
            provider.trunk_forward(bad)
            raise AssertionError("a malformed matrix must be refused before any charge")
        except TrunkInputError:
            pass
    assert ledger.rows == 0 and not seen
    try:
        provider.trunk_forward(np.zeros((1025, 19)))
        raise AssertionError("an over-cap batch must be refused before any charge")
    except BatchTooLarge:
        pass
    assert ledger.rows == 0 and not seen
    matrix = np.ones((7, 19))
    provider.trunk_forward(matrix, n_joint_rows=2)
    assert ledger.rows == 7 and ledger.calls == 1 and ledger.joint_rows == 2
    assert len(seen) == 1 and np.array_equal(seen[0], matrix)
    provider.trunk_forward(matrix)                       # a repeat is charged again
    assert ledger.rows == 14 and ledger.calls == 2

    def exploding(matrix):
        raise RuntimeError("delegate failure")

    provider2 = CalibrationProvider(_FakeController(), ledger, trunk_callable=exploding)
    try:
        provider2.trunk_forward(np.ones((3, 19)))
        raise AssertionError("the delegate failure must surface")
    except RuntimeError:
        pass
    assert ledger.rows == 17, "a charged call whose delegate failed is never refunded"


def test_extract_and_fit_charges_real_rows_and_pools():
    ledger = ForwardLedger()
    rng = np.random.default_rng(9)

    def spy(matrix):
        return rng.normal(size=(matrix.shape[0], EMBED_DIM))

    provider = CalibrationProvider(_FakeController(), ledger, trunk_callable=spy)
    rows = [(rng.normal(size=(5, 19)), [2, 3], rng.normal(size=(2, 2)), {}),
            (rng.normal(size=(4, 19)), [1, 1, 2], rng.normal(size=(3, 2)), {})]
    fit = provider.extract_and_fit(rows)
    assert ledger.rows == 9 and ledger.joint_rows == 5
    assert fit["rows"] == 5


# ------------------------------------------------------------------ 4. payload role validation
def _ep0_payload(seed=11, **over):
    spec = binding._expected_identity_row(seed, "ep0")
    payload = {
        "identity": {"seed": seed, "structure": SEAM_R68_STRUCTURE,
                     "training_version": R68_TRAINING_VERSION_EXPECTED,
                     "module_set": list(MODULES), "source_kind": "r68_ep0"},
        "file_sha256": spec["file_sha256"], "parameter_sha256": spec["parameter_sha256"],
        "episodes_done": spec["episodes_done"], "updates": spec["updates"],
        "load_mode": "model_only", "optimizer_restored": False, "lambda_restored": False,
        "rng_restored": False, "from_trained_ep128": False}
    payload.update(over)
    return payload


def _restore_payload(seed=11, **over):
    identity = registered_run_identity("r71-test-run")
    payload = {
        "identity": {"seed": seed, "structure": binding.R71_STRUCTURE,
                     "training_version": R71_TRAINING_VERSION,
                     "module_set": list(MODULES), "source_kind": "this_run"},
        "run_identity": identity.as_dict(),
        "carries_trained_optimizer_state": True, "carries_trained_lambda": True,
        "carries_advanced_training_rng": True, "from_trained_ep128": False}
    payload.update(over)
    return payload


def test_r68_ep0_model_only_init_payload_is_accepted():
    out = validate_calibration_source(_ep0_payload(), role="r68_ep0_model_only_init")
    assert out["role"] == "r68_ep0_model_only_init" and out["descriptive_only"] is True


def test_ep0_init_counterexamples_are_refused():
    cases = {
        "wrong file sha": {"file_sha256": "0" * 64},
        "wrong parameter sha": {"parameter_sha256": "0" * 64},
        "advanced cursor": {"episodes_done": 128, "updates": 32},
        "optimizer restored": {"optimizer_restored": True},
        "lambda restored": {"lambda_restored": True},
        "rng restored": {"rng_restored": True},
        "wrong load mode": {"load_mode": "read_only"},
        "r71 version on an r68 source": {"identity": {
            "seed": 11, "structure": SEAM_R68_STRUCTURE,
            "training_version": R71_TRAINING_VERSION, "module_set": list(MODULES),
            "source_kind": "r68_ep0"}},
        "declared kind only": {"identity": {"seed": 11, "source_kind": "r68_ep0"}},
        "from trained ep128": {"from_trained_ep128": True},
    }
    for label, over in cases.items():
        try:
            validate_calibration_source(_ep0_payload(**over), role="r68_ep0_model_only_init")
            raise AssertionError(f"refusal expected: {label}")
        except RoleValidationError:
            pass
    try:
        validate_calibration_source({"file_sha256": "x"}, role="r68_ep0_model_only_init")
        raise AssertionError("a payload without an identity block must be refused")
    except RoleValidationError:
        pass


def test_ep128_is_a_readonly_control_and_never_an_init():
    spec = binding._expected_identity_row(11, "ep128")
    control = {
        "identity": {"seed": 11, "structure": SEAM_R68_STRUCTURE,
                     "training_version": R68_TRAINING_VERSION_EXPECTED,
                     "module_set": list(MODULES), "source_kind": "r68_ep0"},
        "file_sha256": spec["file_sha256"], "parameter_sha256": spec["parameter_sha256"],
        "episodes_done": spec["episodes_done"], "updates": spec["updates"],
        "load_mode": "read_only", "optimizer_restored": False, "lambda_restored": False,
        "rng_restored": False, "from_trained_ep128": True}
    out = validate_calibration_source(control, role="r68_ep128_readonly_control")
    assert out["role"] == "r68_ep128_readonly_control"
    control["used_as_init_source"] = True
    try:
        validate_calibration_source(control, role="r68_ep128_readonly_control")
        raise AssertionError("a trained ep128 must refuse to serve as an INIT source")
    except RoleValidationError:
        pass
    try:
        validate_calibration_source(control, role="r68_ep0_model_only_init")
        raise AssertionError("the ep128 payload must fail the ep0 INIT role")
    except RoleValidationError:
        pass


def test_r71_new_full_restore_requires_this_runs_exact_identity():
    identity = registered_run_identity("r71-test-run")
    out = validate_calibration_source(_restore_payload(), role="r71_new_full_restore",
                                      run_identity=identity)
    assert out["run_identity"]["run_id"] == "r71-test-run"
    cases = {
        "prefix-only version": {"identity": {
            "seed": 11, "structure": binding.R71_STRUCTURE,
            "training_version": R68_TRAINING_VERSION_EXPECTED, "module_set": list(MODULES),
            "source_kind": "this_run"}},
        "foreign run identity": {"run_identity": dict(identity.as_dict(),
                                                      run_id="someone-elses-run")},
        "declared kind without identity block": {"run_identity": None},
        "missing trained optimizer": {"carries_trained_optimizer_state": False},
        "missing trained rng": {"carries_advanced_training_rng": False},
        "foreign source kind": {"identity": {
            "seed": 11, "structure": binding.R71_STRUCTURE,
            "training_version": R71_TRAINING_VERSION, "module_set": list(MODULES),
            "source_kind": "r68_trained"}},
    }
    for label, over in cases.items():
        try:
            validate_calibration_source(_restore_payload(**over), role="r71_new_full_restore",
                                        run_identity=identity)
            raise AssertionError(f"refusal expected: {label}")
        except RoleValidationError:
            pass
    try:
        validate_calibration_source(_restore_payload(), role="r71_new_full_restore")
        raise AssertionError("a live run identity is required for this run's own restore")
    except RoleValidationError:
        pass


def test_rule_control_and_unknown_roles():
    clean = {"identity": {"seed": 11}, "carries_trained_optimizer_state": False}
    assert validate_calibration_source(clean, role="rule_control_readonly")["descriptive_only"]
    dirty = dict(clean, carries_trained_lambda=True)
    try:
        validate_calibration_source(dirty, role="rule_control_readonly")
        raise AssertionError("a rule control may carry no trained state")
    except RoleValidationError:
        pass
    try:
        validate_calibration_source(clean, role="not_a_role")
        raise AssertionError("an unknown role must be refused")
    except RoleValidationError:
        pass


# ------------------------------------------------------------------ 5. gates and CLI
def test_configure_refuses_before_importing_the_runtime_or_touching_disk():
    saved = sys.modules.pop("r68_execute", None)
    try:
        with _ephemeral(_scratch_dir("cfg_gate")) as d:
            raw_root = d / "raw"
            try:
                binding.configure_r71_runtime(raw_root, binding.LicenseGate(), None)
                raise AssertionError("no licence must fail closed")
            except LicenseMissing:
                pass
            assert not raw_root.exists(), "nothing may be created before the gates pass"
            assert "r68_execute" not in sys.modules, "the runtime must not be imported"
            test_only = binding.LicenseGate(
                RealLicense(approval_id="t", task_id=binding.TASK_ID,
                            source_freeze_sha256="x", input_identity_sha256="y",
                            quota_identity=binding.R71_QUOTA_IDENTITY, issued_by="t",
                            test_only=True),
                __import__("r71_critic_calibration_seam").spy_license_verifier)
            try:
                binding.configure_r71_runtime(raw_root, test_only, None)
                raise AssertionError("a test-only licence must fail closed")
            except LicenseInvalid:
                pass
            assert not raw_root.exists()
    finally:
        if saved is not None:
            sys.modules["r68_execute"] = saved


def test_model_read_boundary_requires_an_explicit_approval():
    with _ephemeral(_scratch_dir("read_boundary")) as d:
        raw_root = d / "raw"
        gate = binding.make_license_gate(_real_licence_mapping())
        identity = registered_run_identity("r71-test-run")
        assert gate.present(), "a correctly declared licence must open the gate"
        try:
            binding.configure_r71_runtime(raw_root, gate, identity, model_read_approval=None)
            raise AssertionError("the model read-in point must be a controlled stop boundary")
        except ModelReadBoundary:
            pass
        assert not raw_root.exists(), "the stop boundary precedes every side effect"


def test_cli_binding_acceptance_passes_with_zero_consumption():
    with _ephemeral(_scratch_dir("cli_accept")) as d:
        receipt = d / "acceptance.json"
        code = cli.main(["--phase", "binding-acceptance", "--receipt", str(receipt)])
        assert code == 0
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        assert payload["status"] == "passed"
        assert payload["consumption"] == 0
        assert payload["acceptance"]["live_modules"]["r68_training_version_literal"] is True
        assert payload["acceptance"]["fail_closed"]["no_licence"].startswith("LicenseMissing")


def test_cli_fails_closed_on_every_licensed_phase():
    with _ephemeral(_scratch_dir("cli_failclosed")) as d:
        raw_root = d / "raw"
        # no licence at all
        receipt = d / "r1.json"
        code = cli.main(["--phase", "train", "--raw-root", str(raw_root),
                         "--receipt", str(receipt)])
        assert code == 3
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        assert payload["status"] == "fail_closed"
        assert payload["refusal"].startswith("LicenseMissing")
        assert not raw_root.exists(), "no raw root may exist before the gates pass"
        # a correctly declared licence still stops at the model read-in boundary
        licence = d / "licence.json"
        licence.write_text(json.dumps(_real_licence_mapping()), encoding="utf-8")
        receipt2 = d / "r2.json"
        code = cli.main(["--phase", "train", "--raw-root", str(raw_root),
                         "--licence", str(licence), "--receipt", str(receipt2)])
        assert code == 3
        payload2 = json.loads(receipt2.read_text(encoding="utf-8"))
        assert payload2["status"] == "stopped_at_model_read_boundary"
        assert payload2["refusal"].startswith("ModelReadBoundary")
        assert not raw_root.exists()
        # a test-only licence is refused at construction
        test_only = dict(_real_licence_mapping(), test_only=True)
        receipt3 = d / "r3.json"
        code = cli.main(["--phase", "offline-fit", "--licence", json.dumps(test_only),
                         "--receipt", str(receipt3)])
        assert code == 2
        assert "LicenseInvalid" in json.loads(receipt3.read_text(encoding="utf-8"))["refusal"]


def test_cli_refuses_to_silently_overwrite_a_receipt():
    with _ephemeral(_scratch_dir("cli_overwrite")) as d:
        receipt = d / "acceptance.json"
        assert cli.main(["--phase", "binding-acceptance", "--receipt", str(receipt)]) == 0
        try:
            cli.main(["--phase", "binding-acceptance", "--receipt", str(receipt)])
            raise AssertionError("an existing receipt must not be silently replaced")
        except SystemExit:
            pass
        assert cli.main(["--phase", "binding-acceptance", "--receipt", str(receipt),
                         "--overwrite-receipt"]) == 0


def test_cli_phase_requirements_are_checked_before_anything_else():
    with _ephemeral(_scratch_dir("cli_usage")) as d:
        receipt = d / "r.json"
        code = cli.main(["--phase", "train", "--receipt", str(receipt)])
        assert code == 5
        assert json.loads(receipt.read_text(encoding="utf-8"))["problems"]


def test_coverage_block_declares_what_is_not_covered():
    coverage = runtime_binding_coverage()
    assert "an echo delegate does not exist" in coverage["provider_binding"]["not_covered"]
    assert "no licence exists this round" in coverage["runtime_and_cli"]["not_covered"]
    assert coverage["real_gates_awaiting_approval"]
    assert binding.CANDIDATE_STATUS == "APPROVAL_PENDING"
    assert binding.R71_QUOTA_IDENTITY.endswith("hard-600/offline-72000-input-rows-batch-1024/1")