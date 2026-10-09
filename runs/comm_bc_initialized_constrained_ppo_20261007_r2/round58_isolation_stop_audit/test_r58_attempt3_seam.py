"""ROUND58 attempt3 seam narrow acceptance tests (cold-start, pre-claim).

Run:  D:\\project_env\\python.exe test_r58_attempt3_seam.py
Writes nothing into real claim/slot/journal namespaces; all allocator fixtures
run in an isolated temporary directory.  The recovery-gate test is read-only
against the real round58_gen3 raw history.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
GEN2 = RUN_ROOT / "round58_gen2"
RAW = RUN_ROOT / "round58_gen3" / "raw"

for path in (GEN2,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import r58_contracts as frozen_contracts  # noqa: E402
import r58_attempt3_contract as attempt3_contract  # noqa: E402

# R45-global stand-ins so the fake guarded_execute exposes the same module
# surface the seam reads on the real live module.
import contextvars  # noqa: E402
CURRENT_CONTEXT = contextvars.ContextVar("r58_attempt3_test_context", default=None)


def _context_for_cell(kind, logical_job, scenario_seed, metadata, runner_kwargs):
    return {"logical_job": logical_job, "kind": kind, "scenario_seed": int(scenario_seed),
            "tag": str((runner_kwargs or {}).get("tag", ""))}


def _planned_key(context):
    return None

RESULTS: list[dict] = []


def check(name: str, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        RESULTS.append({"name": name, "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}"})
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    else:
        RESULTS.append({"name": name, "status": "passed"})
        print(f"PASS {name}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_plan() -> dict:
    return {"jobs": [
        {"scope": "ppo_training", "logical_job": "new_training_s11_e0", "seed": 11,
         "episode_index": 0, "scene": 1300000, "normal_slot": 1},
        {"scope": "ppo_training", "logical_job": "new_training_s22_e4", "seed": 22,
         "episode_index": 4, "scene": 1300003, "normal_slot": 3},
        {"scope": "ppo_training", "logical_job": "new_training_s11_e4", "seed": 11,
         "episode_index": 4, "scene": 1300003, "normal_slot": 2},
    ]}


def base_claims() -> list[dict]:
    return [
        {"logical_job": "new_training_s11_e0", "attempt": 1, "category": "normal",
         "status": "complete", "launch_slot": 1},
        {"logical_job": "new_training_s11_e4", "attempt": 1, "category": "normal",
         "status": "failed_technical", "launch_slot": 2,
         "environment_constructor_calls": 0, "simulation_started": False},
        {"logical_job": "new_training_s11_e4", "attempt": 2, "category": "reserve",
         "status": "failed_technical", "launch_slot": 1,
         "environment_constructor_calls": 0, "simulation_started": False},
    ]


def attempt3_claim(**overrides):
    claim = {"logical_job": "new_training_s11_e4", "attempt": 3, "category": "reserve",
             "status": "claimed_before_environment", "launch_slot": 2}
    claim.update(overrides)
    return claim


# ---------------------------------------------------------------------------
# 1. The frozen contract must reject attempt3 (old implementation fails).
def test_frozen_rejects_attempt3():
    claims = base_claims() + [attempt3_claim()]
    try:
        frozen_contracts.validate_claims(synthetic_plan(), claims, strict=False)
    except ValueError as exc:
        assert "category/attempt mismatch" in str(exc), str(exc)
    else:
        raise AssertionError("frozen contract unexpectedly accepted attempt3")


# 2. The extended contract accepts the exact legal 12+2+attempt3 history.
def test_extended_accepts_legal_history():
    plan = synthetic_plan()
    claims = base_claims() + [attempt3_claim()]
    closure = attempt3_contract.validate_claims(plan, claims, strict=False)
    assert closure["counts"] == {"normal": 2, "reserve": 2}, closure
    assert closure["attempt3_claims"] == 1


# 3. Real 14-claim gen3 history (12 complete + 2 failed) validates.
def test_real_history_validates():
    claims = [json.loads(p.read_text(encoding="utf-8"))
              for p in sorted((RAW / "launch_claims").glob("*/*.json"))]
    assert len(claims) >= 14, len(claims)
    plan = json.loads((RAW / "ROUND53_FIXED_PLAN.json").read_text(encoding="utf-8"))
    closure = attempt3_contract.validate_claims(plan, claims, strict=False)
    assert closure["counts"]["normal"] >= 13, closure
    assert closure["counts"]["reserve"] == 2, closure
    assert closure["completed_logical_jobs"] >= 12, closure


# 4. Rejection matrix for the unique attempt3 special case.
def test_rejection_matrix():
    plan = synthetic_plan()
    good = base_claims() + [attempt3_claim()]
    attempt3_contract.validate_claims(plan, good)  # sanity

    cases = {
        "other_task_attempt3": base_claims() + [attempt3_claim(logical_job="new_training_s22_e4", launch_slot=2)],
        "normal_attempt3": base_claims() + [attempt3_claim(category="normal", launch_slot=3)],
        "attempt4": base_claims() + [attempt3_claim(attempt=4, launch_slot=3)],
        "wrong_slot": base_claims() + [attempt3_claim(launch_slot=5)],
        "duplicate_slot": base_claims() + [attempt3_claim(launch_slot=1)],
        "unplanned_task": base_claims() + [attempt3_claim(logical_job="unplanned_job", launch_slot=2)],
        "forged_predecessor": (
            [dict(base_claims()[0]),
             dict(base_claims()[1], status="complete"),
             dict(base_claims()[2])]
            + [attempt3_claim()]),
        "duplicate_attempt3": base_claims() + [attempt3_claim(), attempt3_claim()],
        "completed_task_retry": (
            [dict(base_claims()[0]),
             dict(base_claims()[1], status="complete"),
             dict(base_claims()[2])]
            + [attempt3_claim()]),
    }
    for name, claims in cases.items():
        try:
            attempt3_contract.validate_claims(plan, claims)
        except ValueError:
            continue
        raise AssertionError(f"rejection case accepted: {name}")


# 5. Recovery gate through the real gen2 module attribute (read-only).
def test_recovery_gate_real_history():
    legacy = load_module("round58_legacy_gate_test", GEN2 / "r58_execute.py")
    plan = json.loads((RAW / "ROUND53_FIXED_PLAN.json").read_text(encoding="utf-8"))
    claims = [json.loads(p.read_text(encoding="utf-8"))
              for p in sorted((RAW / "launch_claims").glob("*/*.json"))]
    legacy.TRAINING_VERSION = json.loads((RAW / "ROUND46_FIRST_FOUR_s11.json").read_text(encoding="utf-8"))["training_version"]
    base = SimpleNamespace(V6_CHECKPOINTS_DIR=RAW / "checkpoints")
    legacy.verify_persisted_first_batch_state = attempt3_contract.verify_persisted_first_batch_state
    result = legacy.verify_persisted_first_batch_state(base, RAW, plan, legacy=legacy,
                                                       training_version=legacy.TRAINING_VERSION)
    assert result["status"] == "passed", result
    assert result["approved_attempt3_predecessors_verified"] is True
    assert set(result["audited_failed_claims"]) == {
        "new_training_s11_e4#1", "new_training_s11_e4#2"}, result["audited_failed_claims"]
    if result.get("interrupted_attempt3_bookkeeping"):
        assert result["interrupted_attempt3_bookkeeping"]["completed_steps"] == 1000
    else:
        corrected = [c for c in claims
                     if c.get("logical_job") == "new_training_s11_e4" and c.get("attempt") == 3]
        assert corrected and corrected[0]["status"] == "complete", corrected
        assert corrected[0]["round58_receipt_failure_correction"]["kind"] == \
            "round58-attempt3-receipt-failure-correction/1"



def _copy_history(destination: Path) -> Path:
    raw = destination / "raw"
    raw.mkdir(parents=True)
    for name in ("launch_claims", "launch_slots"):
        shutil.copytree(RAW / name, raw / name)
    for seed in (11, 22, 33):
        gate = RAW / f"ROUND45_R2_FIRST_BATCH_GATE_R3_s{seed}.json"
        if gate.is_file():
            shutil.copy2(gate, raw / gate.name)
        shutil.copy2(RAW / f"ROUND46_FIRST_FOUR_s{seed}.json", raw / f"ROUND46_FIRST_FOUR_s{seed}.json")
    return raw


def _tamper(directory: Path, path: Path, mutate):
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_gate(raw: Path, plan: dict, legacy):
    base = SimpleNamespace(V6_CHECKPOINTS_DIR=RAW / "checkpoints")
    saved = legacy.verify_persisted_first_batch_state
    legacy.verify_persisted_first_batch_state = attempt3_contract.verify_persisted_first_batch_state
    try:
        return legacy.verify_persisted_first_batch_state(base, raw, plan, legacy=legacy)
    finally:
        legacy.verify_persisted_first_batch_state = saved


# 6. Recovery gate rejects an extra complete claim (13th source record).
def test_recovery_gate_rejects_extra_complete(tmp: Path):
    legacy = load_module("round58_legacy_gate_test_extra", GEN2 / "r58_execute.py")
    plan = json.loads((RAW / "ROUND53_FIXED_PLAN.json").read_text(encoding="utf-8"))
    raw = _copy_history(tmp / "extra_complete")
    target = raw / "launch_claims" / "new_training_s11_e4" / "attempt_0002.json"
    _tamper(raw, target, lambda p: p.update({"status": "complete"}))
    try:
        _run_gate(raw, plan, legacy)
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("gate accepted a 13th complete source record")


# 7. Recovery gate rejects filtering the failure history out.
def test_recovery_gate_rejects_removed_failures(tmp: Path):
    legacy = load_module("round58_legacy_gate_test_filtered", GEN2 / "r58_execute.py")
    plan = json.loads((RAW / "ROUND53_FIXED_PLAN.json").read_text(encoding="utf-8"))
    raw = _copy_history(tmp / "filtered")
    shutil.rmtree(raw / "launch_claims" / "new_training_s11_e4")
    (raw / "launch_slots" / "normal" / "0013.json").unlink()
    (raw / "launch_slots" / "reserve" / "0001.json").unlink()
    try:
        _run_gate(raw, plan, legacy)
    except RuntimeError as exc:
        assert "predecessor pair" in str(exc), str(exc)
    else:
        raise AssertionError("gate accepted a filtered failure history")


# 8. Recovery gate rejects a forged zero-consumption counter.
def test_recovery_gate_rejects_env_consumption_on_failure(tmp: Path):
    legacy = load_module("round58_legacy_gate_test_env", GEN2 / "r58_execute.py")
    plan = json.loads((RAW / "ROUND53_FIXED_PLAN.json").read_text(encoding="utf-8"))
    raw = _copy_history(tmp / "env")
    target = raw / "launch_claims" / "new_training_s11_e4" / "attempt_0001.json"
    _tamper(raw, target, lambda p: p.update({"environment_constructor_calls": 1}))
    try:
        _run_gate(raw, plan, legacy)
    except (RuntimeError, ValueError) as exc:
        assert "environment_constructor_calls" in str(exc), str(exc)
    else:
        raise AssertionError("gate accepted a failed claim with environment consumption")


# 9. Real allocator + attempt3 seam in an isolated directory.
def test_allocator_seam_isolated(tmp: Path):
    import r58_attempt3_claim_seam as attempt3_seam
    r40_common = load_module("r40_common_attempt3_test",
                             REPO_ROOT / "runs" / "comm_first_report_training_ablation_20261003" / "r40_common.py")
    work = tmp / "allocator"
    work.mkdir()
    r40_common.ROUND_DIR = work
    r40_common.NORMAL_CAP = 1024
    r40_common.RESERVE_CAP = 24
    r40_common.HARD_CAP = 1064
    plan = synthetic_plan()
    claim_launch = r40_common.claim_launch

    def _claim(logical_job, *, attempt, metadata):
        return claim_launch(logical_job, category="normal" if int(attempt) == 1 else "reserve",
                            attempt=int(attempt), metadata=metadata)

    # Mirror the frozen planned_claim shape so the seam binds the delegation cell.
    def planned_claim(logical_job, *args, **kwargs):
        attempt = int(kwargs.get("attempt", args[0] if args else 1))
        if attempt not in (1, 2):
            raise RuntimeError(f"attempt number outside frozen one-retry contract: {attempt}")
        return _claim(logical_job, *args, **kwargs)

    planned_claim.__code__ = planned_claim.__code__.replace(
        co_filename=str(tmp / "fake_r58_execute.py"), co_name="planned_claim")

    def _execute_cell(label, logical_job, kind, scenario, metadata, *,
                      collector=None, runner_kwargs=None, row_updates=None,
                      identity=None):
        return {"executed": logical_job, "kind": kind, "scenario": int(scenario)}

    def make_guarded(base_execute):
        def guarded_execute(label, logical_job, *args, **kwargs):
            if kwargs.get("identity") == "USE_BASE":
                return base_execute(label, logical_job, *args, **kwargs)
            return {"frozen_gate": True, "logical_job": logical_job}
        guarded_execute.__code__ = guarded_execute.__code__.replace(
            co_filename=str(tmp / "fake_r58_execute.py"), co_name="guarded_execute")
        return guarded_execute

    run_module = SimpleNamespace(RUN_DIR=work,
                                 _execute_cell=make_guarded(_execute_cell))

    # Legal history in the isolated root: one complete normal + the failed pair.
    c1 = claim_launch("new_training_s11_e0", category="normal", attempt=1, metadata={})
    r40_common.finish_launch(c1, ok=True, detail="fixture")
    c2 = claim_launch("new_training_s11_e4", category="normal", attempt=1, metadata={})
    r40_common._update_claim(c2, status="failed_technical", environment_constructor_calls=0,
                             completed_steps=0, simulation_started=False)
    c3 = claim_launch("new_training_s11_e4", category="reserve", attempt=2, metadata={})
    r40_common._update_claim(c3, status="failed_technical", environment_constructor_calls=0,
                             completed_steps=0, simulation_started=False)
    (work / "failures").mkdir(exist_ok=True)
    (work / "failures" / "train_s11_e4_r44_a1.json").write_text("{}", encoding="utf-8")
    (work / "failures" / "train_s11_e4_r44_a2.json").write_text("{}", encoding="utf-8")

    installed = attempt3_seam.install(
        run_module, plan=plan, raw_root=work, frozen_wrapper=planned_claim,
        normal_cap=1024, reserve_cap=24, hard_cap=1064, historical_consumed=16)
    assert installed["status"] == "installed"

    seam_claim = run_module._claim
    # Rejections first (no durable write may happen for these).
    for logical, attempt in (("new_training_s22_e4", 3), ("new_training_s11_e0", 3)):
        try:
            seam_claim(logical, attempt=attempt, metadata={})
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"seam accepted non-approved attempt3: {logical}")
        assert not (work / "launch_claims" / logical / "attempt_0003.json").exists()
    try:
        seam_claim("new_training_s11_e4", attempt=4, metadata={})
    except RuntimeError as exc:
        assert "outside frozen one-retry contract" in str(exc), str(exc)
    else:
        raise AssertionError("seam accepted attempt4")

    # execute seam: the approved key bypasses the frozen one-retry gate exactly once
    executed = run_module._execute_cell("lbl", "new_training_s11_e4", "train", 1300003, {})
    assert executed == {"executed": "new_training_s11_e4", "kind": "train",
                        "scenario": 1300003}, executed
    # any other key keeps the frozen gate
    other = run_module._execute_cell("lbl", "new_training_s22_e4", "train", 1300003, {})
    assert other == {"frozen_gate": True, "logical_job": "new_training_s22_e4"}, other

    # Approved attempt3 allocation through the real allocator.
    claim_path = seam_claim("new_training_s11_e4", attempt=3, metadata={"retry_of": "approved"})
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    assert claim["category"] == "reserve" and claim["attempt"] == 3, claim
    assert claim["launch_slot"] == 2, claim
    slot = json.loads((work / "launch_slots" / "reserve" / "0002.json").read_text(encoding="utf-8"))
    assert slot["logical_job"] == "new_training_s11_e4" and slot["attempt"] == 3
    closure = attempt3_contract.validate_claims(plan, [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((work / "launch_claims").glob("*/*.json"))], strict=False)
    assert closure["counts"] == {"normal": 2, "reserve": 2}, closure
    # once the attempt3 file exists, the approved key also keeps the frozen gate
    replay = run_module._execute_cell("lbl", "new_training_s11_e4", "train", 1300003, {})
    assert replay == {"frozen_gate": True, "logical_job": "new_training_s11_e4"}, replay
    # Duplicate attempt3 must refuse atomically.
    try:
        claim_launch("new_training_s11_e4", category="reserve", attempt=3, metadata={})
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate attempt3 claim was accepted")


# 10. Occupied reserve slot 2 blocks allocation before any write.
def test_slot2_occupied_blocks(tmp: Path):
    import r58_attempt3_claim_seam as attempt3_seam
    work = tmp / "occupied"
    work.mkdir()
    plan = synthetic_plan()
    (work / "launch_slots" / "reserve").mkdir(parents=True)
    (work / "launch_slots" / "reserve" / "0002.json").write_text("{}", encoding="utf-8")
    try:
        attempt3_contract.preflight_attempt3_allocation(work, plan)
    except RuntimeError as exc:
        assert "slot 2" in str(exc), str(exc)
    else:
        raise AssertionError("preflight accepted an occupied attempt3 slot")


def main() -> int:
    tmp = HERE / "_attempt3_test_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        check("frozen_contract_rejects_attempt3", test_frozen_rejects_attempt3)
        check("extended_contract_accepts_legal_history", test_extended_accepts_legal_history)
        check("real_14_claim_history_validates", test_real_history_validates)
        check("attempt3_rejection_matrix", test_rejection_matrix)
        check("recovery_gate_real_history_readonly", test_recovery_gate_real_history)
        check("recovery_gate_rejects_extra_complete", lambda: test_recovery_gate_rejects_extra_complete(tmp))
        check("recovery_gate_rejects_removed_failures", lambda: test_recovery_gate_rejects_removed_failures(tmp))
        check("recovery_gate_rejects_env_consumption_on_failure", lambda: test_recovery_gate_rejects_env_consumption_on_failure(tmp))
        check("allocator_seam_isolated_end_to_end", lambda: test_allocator_seam_isolated(tmp))
        check("occupied_slot2_blocks_allocation", lambda: test_slot2_occupied_blocks(tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [r for r in RESULTS if r["status"] != "passed"]
    print(f"total={len(RESULTS)} passed={len(RESULTS) - len(failed)} failed={len(failed)}")
    (HERE / "ROUND58_ATTEMPT3_SEAM_TEST_RESULTS.json").write_text(
        json.dumps({"kind": "round58-attempt3-seam-test-results/1",
                    "results": RESULTS,
                    "passed": not failed}, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())




