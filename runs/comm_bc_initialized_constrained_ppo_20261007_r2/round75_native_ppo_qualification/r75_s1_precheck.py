# -*- coding: utf-8 -*-
"""ROUND75 S1 freeze precheck (request R75_S0_FINAL_PREP_20261014).

Pure stdlib. Zero torch, zero model load/construction, zero forward/backward, zero optimizer or
PPO step, zero environment, zero sampling, zero checkpoint write, zero canonical-ledger write.
It only READS files and re-derives every identity the S1 runner depends on from
ROUND75_S1_RUN_PLAN.json, so a drifted source, a changed batch, a non-zero counter or a rebuilt
ledger fails BEFORE the paid path is allowed to start.

Run:  python r75_s1_precheck.py            (exit 0 = the frozen S1 plan is intact)
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r75_guard as guard_mod                      # noqa: E402  (stdlib-only reuse chain)
import r75_replay_adapter as adapter_mod           # noqa: E402  (stdlib-only)
from r75_s1g_outer_exec import ledger_snapshot     # noqa: E402  verified stdlib parser


def _repo_root() -> Path:
    """The repository root, found by walking up. This precheck must NEVER depend on the CWD: the
    frozen plan stores repo-relative paths and the decision side may launch it from anywhere, so a
    path resolved against the working directory silently fails (or, worse, resolves elsewhere)."""
    for candidate in (HERE, *HERE.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"cannot locate the repository root (.git) above {HERE}")


REPO_ROOT = _repo_root()
PLAN_PATH = HERE / "ROUND75_S1_RUN_PLAN.json"
FROZEN_SRC = (HERE / "r75_frozen.py").read_text(encoding="utf-8")
GUARD_SRC = (HERE / "r75_guard.py").read_text(encoding="utf-8")
RUNNER_SRC = (HERE / "r75_s1_runner.py").read_text(encoding="utf-8")
R66_DIR = HERE.parent / "round66_bc_early_path_preserving_late_adaptation"
R66_SRC = (R66_DIR / "r66_frozen.py").read_text(encoding="utf-8")


def resolve_path(raw) -> Path:
    """Resolve a plan-stored path against the repository root, never against the CWD."""
    path = Path(str(raw))
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def passed(self) -> bool:
        return all(item["ok"] for item in self.checks)


def check_source_identity(plan: dict, report: Report) -> None:
    identity = plan["source_identity"]
    for key in ("canonical_trainer", "canonical_common", "canonical_controller",
                "canonical_onpolicy", "r66_frozen", "r75_guard", "r75_frozen",
                "r75_replay_adapter"):
        entry = identity[key]
        resolved = resolve_path(entry["path"])
        actual = sha256_file(resolved)
        expected = entry.get("expected", entry["sha256"])
        report.check(f"source:{key}", actual == expected,
                     f"{resolved} sha256 {actual[:16]} expected {expected[:16]}")
    for name, entry in identity["reused_verified_sources"].items():
        resolved = resolve_path(entry["path"])
        actual = sha256_file(resolved)
        report.check(f"reused:{name}", actual == entry["expected"],
                     f"{resolved} sha256 {actual[:16]} expected {entry['expected'][:16]}")
    # the reused sources must also pass the guard's own live byte-identity check
    try:
        guard_mod.verify_reused_source_identity()
        report.check("reused:live_verify", True, "r75_guard.verify_reused_source_identity() passed")
    except Exception as error:                                  # pragma: no cover - defensive
        report.check("reused:live_verify", False, str(error))


def check_guard_pin(plan: dict, report: Report) -> None:
    pinned = re.search(r'EXPECTED_R75_GUARD_SOURCE_SHA256 = "([0-9a-f]{64})"', FROZEN_SRC)
    on_disk = sha256_file(HERE / "r75_guard.py")
    report.check("guard:pin_constant_present", pinned is not None,
                 "r75_frozen pins the guard source sha256")
    if pinned:
        report.check("guard:pin_matches_disk", pinned.group(1) == on_disk,
                     f"pinned {pinned.group(1)[:16]} vs disk {on_disk[:16]}")
        report.check("guard:pin_matches_plan", pinned.group(1) == plan["source_identity"]["r75_guard"]["expected"],
                     "the pinned value equals the frozen plan expectation")


def check_input_identity(plan: dict, report: Report) -> None:
    identity = plan["input_identity"]
    for key in ("row", "tape", "r58_ep0_model"):
        entry = identity[key]
        resolved = resolve_path(entry["path"])
        actual = sha256_file(resolved)
        report.check(f"input:{key}", actual == entry["sha256"],
                     f"{resolved} sha256 {actual[:16]} expected {entry['sha256'][:16]}")
    model = identity["r58_ep0_model"]
    report.check("input:model_matches_registry",
                 model["sha256"] == model["registry_checkpoint_sha256"],
                 "the R58 ep0 checkpoint equals the r66 registry entry")
    report.check("input:behaviour_parameters_match_registry",
                 identity["behaviour_parameters"]["matches_registry"] is True,
                 f"row behaviour parameter sha256 {identity['behaviour_parameters']['value'][:16]}")


def check_batch_shape(plan: dict, report: Report) -> None:
    identity = plan["input_identity"]
    try:
        # Absolute paths: the adapter reads the files as given, and the precheck must not depend
        # on the CWD. The adapter's own bytes stay untouched and its pinned identity unchanged.
        episode, facts = adapter_mod.load_qualification_batch(
            resolve_path(identity["row"]["path"]), resolve_path(identity["tape"]["path"]))
    except Exception as error:
        report.check("batch:loads_and_validates", False, f"adapter refused: {error}")
        return
    report.check("batch:loads_and_validates", True, "the qualified adapter rebuilt the episode")
    required = plan["input_identity"]["required_batch_shape"]
    expectations = {"records": required["joint_records"],
                    "candidate_rows": required["candidate_rows"],
                    "horizon": required["horizon_ticks"],
                    "free_early_rows": required["free_early"],
                    "free_late_rows": required["free_late"]}
    for key, expected in expectations.items():
        actual = facts.get(key)
        report.check(f"batch:{key}", actual == expected, f"{key} = {actual}, expected {expected}")
    dims = {len(row) for record in episode["collector"].records for row in record["features"]}
    report.check("batch:feature_dim", dims == {required["feature_dim"]},
                 f"feature dims seen {sorted(dims)}")
    report.check("batch:lambda", facts.get("lambda") == 1.0, f"lambda = {facts.get('lambda')}")
    report.check("batch:row_tape_sha_match_plan",
                 facts["row_sha256"] == identity["row"]["sha256"]
                 and facts["tape_sha256"] == identity["tape"]["sha256"],
                 "the adapter reports the same row/tape identities as the plan")


def check_ledger_snapshot(plan: dict, report: Report) -> None:
    snapshot = plan["canonical_ledger_frozen_snapshot"]
    root = resolve_path(snapshot["root"])
    for name, entry in snapshot["files"].items():
        path = root / name
        if not path.is_file():
            report.check(f"ledger:{name}", False, "missing")
            continue
        actual = sha256_file(path)
        report.check(f"ledger:{name}", actual == entry["sha256"],
                     f"{name} sha256 {actual[:16]} expected {entry['sha256'][:16]} "
                     f"({path.stat().st_size} bytes)")
    text = (root / "r75_total_ledger.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    report.check("ledger:event_count", len(events) == snapshot["total_ledger_events"],
                 f"{len(events)} events, frozen {snapshot['total_ledger_events']} "
                 f"(no bootstrap append and no rebuild)")
    charged = [event for event in events if event.get("event") == "charged"]
    report.check("ledger:no_charged_events", len(charged) == snapshot["charged_events"] == 0,
                 f"{len(charged)} charged events")
    for name in ("r75_physical_optimizer_step_guard.json",
                 "r75_offline_trunk_target_row_guard.json",
                 "r75_joint_record_guard.json",
                 "r75_environment_construction_guard.json"):
        state = json.loads((root / name).read_text(encoding="utf-8"))
        used = int(state.get("used_normal", 0)) + int(state.get("used_reserve", 0))
        report.check(f"ledger:zero:{name}", used == 0 and int(state.get("refusals", 0)) == 0,
                     f"used={used} refusals={state.get('refusals')}")
    locks = sorted(p.name for p in root.glob("*.lock"))
    report.check("ledger:no_residual_lock", not locks, f"lock files: {locks}")


def check_active_production_baseline(plan: dict, report: Report) -> None:
    """Verify the real S1G start ledger instead of a rebuilt or zero ledger."""
    contract = plan["active_production_baseline"]
    root = resolve_path(contract["root"])
    snapshot = ledger_snapshot(root, contract["files"])
    report.check("ledger:baseline_files", snapshot["frozen_snapshot_match"] is True,
                 "the nine runtime files match the active S1G baseline byte-for-byte")
    report.check("ledger:baseline_event_count",
                 snapshot["total_ledger_events"] == contract["total_ledger_events"],
                 f"{snapshot['total_ledger_events']} events, expected "
                 f"{contract['total_ledger_events']}")
    report.check("ledger:baseline_reconciliation",
                 snapshot["guard_ledger_reconciliation"]["ok"] is True,
                 "each durable guard reconciles against the append-only ledger")
    events = [json.loads(line) for line in (root / "r75_total_ledger.jsonl")
              .read_text(encoding="utf-8").splitlines() if line.strip()]
    sums: dict[str, dict[str, int]] = {}
    for event in events:
        sums.setdefault(event.get("guard", ""),
                        {"charged_events": 0, "sum_normal": 0, "sum_reserve": 0})
        if event.get("event") == "charged":
            sums[event.get("guard", "")]["charged_events"] += 1
            if event.get("category") == "normal":
                sums[event.get("guard", "")]["sum_normal"] += int(event.get("n", 0))
            elif event.get("category") == "reserve":
                sums[event.get("guard", "")]["sum_reserve"] += int(event.get("n", 0))
    kinds = {
        "physical_optimizer_step": "r75-physical-optimizer-step-guard/1",
        "offline_trunk_target_row": "r75-offline-trunk-target-row-guard/1",
        "joint_record": "r75-joint-record-guard/1",
        "environment_construction": "r75-environment-construction-guard/1",
    }
    for unit, want in contract["guards"].items():
        got = snapshot["per_unit_last_known"].get(unit, {})
        for field in ("used_normal", "used_reserve", "refusals", "charged_events"):
            report.check(f"ledger:baseline:{unit}:{field}", got.get(field) == want[field],
                         f"{field} = {got.get(field)!r}, expected {want[field]!r}")
        event_sum = sums.get(kinds[unit], {})
        report.check(f"ledger:baseline:{unit}:event_sums",
                     event_sum.get("charged_events") == want["charged_events"]
                     and event_sum.get("sum_normal") == want.get("sum_normal", 0)
                     and event_sum.get("sum_reserve") == want.get("sum_reserve", 0),
                     f"event sums {event_sum!r}, expected {want!r}")
    locks = sorted(path.name for path in root.glob("*.lock"))
    report.check("ledger:baseline_no_residual_lock", not locks, f"lock files: {locks}")


def check_s1g_headroom(plan: dict, report: Report) -> None:
    """Check old production usage plus S1G by exact category, with no fallback."""
    baseline = plan["active_production_baseline"]["guards"]
    attempt = plan["s1g_execution_contract"]["attempt_budget"]["totals"]
    cumulative = plan["s1g_execution_contract"]["attempt_budget"]["cumulative_caps_after_attempt"]
    physical = {
        "normal": baseline["physical_optimizer_step"]["used_normal"]
                  + attempt["physical_optimizer_steps"]["normal"],
        "reserve": baseline["physical_optimizer_step"]["used_reserve"]
                   + attempt["physical_optimizer_steps"]["technical"],
        "total": baseline["physical_optimizer_step"]["used_normal"]
                 + baseline["physical_optimizer_step"]["used_reserve"]
                 + attempt["physical_optimizer_steps"]["hard"],
    }
    report.check("headroom:physical_by_category",
                 physical == {key: cumulative["physical_optimizer_step"][key]
                              for key in ("normal", "reserve", "total")},
                 f"physical baseline+attempt {physical}, cumulative "
                 f"{cumulative['physical_optimizer_step']}")
    report.check("headroom:physical_within_caps",
                 physical["normal"] <= 12 and physical["reserve"] <= 4
                 and physical["total"] <= 16,
                 "physical normal<=12, reserve<=4, total<=16")
    rows = {
        "normal": baseline["offline_trunk_target_row"]["used_normal"]
                  + attempt["offline_trunk_target_row"]["normal"],
        "reserve": baseline["offline_trunk_target_row"]["used_reserve"]
                   + attempt["offline_trunk_target_row"]["reserve"],
    }
    report.check("headroom:trunk_by_category",
                 rows == {key: cumulative["offline_trunk_target_row"][key]
                          for key in ("normal", "reserve")},
                 f"trunk baseline+attempt {rows}, cumulative "
                 f"{cumulative['offline_trunk_target_row']}")
    report.check("headroom:trunk_within_caps",
                 rows["normal"] <= 5000 and rows["reserve"] <= 1000
                 and rows["normal"] + rows["reserve"] <= 6000,
                 "trunk normal<=5000, reserve<=1000, hard<=6000")
    report.check("headroom:trunk_rows_stay_normal",
                 attempt["offline_trunk_target_row"]["reserve"] == 0,
                 "A2/A3 trunk rows and calls stay normal; only A3 physical steps are technical")
    per_charge = verified_per_charge_cap(plan, report)
    largest = plan["accounting"]["per_native_update"]["largest_single_row_charge"]
    report.check("headroom:largest_single_row_charge",
                 per_charge > 0 and largest <= per_charge,
                 f"largest single charge {largest} vs verified per-charge cap {per_charge}")
    env_root = resolve_path(plan["active_production_baseline"]["root"])
    env_state = json.loads((env_root / "r75_environment_construction_guard.json")
                           .read_text(encoding="utf-8"))
    report.check("headroom:environment_forbidden",
                 int(env_state["normal_cap"]) == 0 and int(env_state["hard_cap"]) == 0,
                 "environment construction stays 0/0/0 and is refused by source")


def verified_per_charge_cap(plan: dict, report: Report) -> int:
    """The per-charge cap must come from a VERIFIED source, never from a missing JSON key.

    ``per_charge_cap`` is a LIVE guard attribute (``R75OfflineRowGuard.__init__`` sets it), so the
    durable counter file does not carry it: reading it from that JSON yields ``None`` and the
    comparison raises TypeError. Two independent verified sources are required instead - the guard
    source attribute and the frozen plan contract - and the old counter is never rewritten and no
    bootstrap is appended to make this pass. A raw value that IS present must agree, never
    silently override.
    """
    guard_match = re.search(r"self\.per_charge_cap\s*=\s*(\d+)", GUARD_SRC)
    if not report.check("quota:per_charge_cap_from_guard_source", guard_match is not None,
                        "r75_guard.py declares self.per_charge_cap"):
        return 0
    guard_cap = int(guard_match.group(1))
    contract_text = plan["accounting"]["caps"]["offline_trunk_target_row"]
    contract_match = re.search(r"per-charge\s+(\d+)", contract_text)
    if not report.check("quota:per_charge_cap_from_frozen_contract", contract_match is not None,
                        f"the frozen plan contract states the per-charge cap: {contract_text!r}"):
        return 0
    contract_cap = int(contract_match.group(1))
    report.check("quota:per_charge_cap_sources_agree", guard_cap == contract_cap,
                 f"guard source {guard_cap} vs frozen contract {contract_cap}")
    root = resolve_path(plan.get("active_production_baseline",
                                  plan["canonical_ledger_frozen_snapshot"])["root"])
    raw_state = json.loads((root / "r75_offline_trunk_target_row_guard.json").read_text(encoding="utf-8"))
    raw_value = raw_state.get("per_charge_cap")
    report.check("quota:per_charge_cap_raw_json_not_contradicting",
                 raw_value is None or int(raw_value) == guard_cap,
                 f"raw counter per_charge_cap={raw_value!r}; absent is expected because it is a "
                 f"live guard attribute - it is never read as None and never compared against one")
    return guard_cap if guard_cap == contract_cap else 0


def check_quota_headroom(plan: dict, report: Report) -> None:
    accounting = plan["accounting"]
    root = resolve_path(plan["canonical_ledger_frozen_snapshot"]["root"])
    caps = {}
    for name, unit in (("r75_physical_optimizer_step_guard.json", "physical_optimizer_step"),
                       ("r75_offline_trunk_target_row_guard.json", "offline_trunk_target_row")):
        state = json.loads((root / name).read_text(encoding="utf-8"))
        caps[unit] = {"normal_cap": int(state["normal_cap"]), "reserve_cap": int(state["reserve_cap"]),
                      "hard_cap": int(state["hard_cap"])}
    steps = accounting["planned_totals"]["physical_optimizer_steps"]
    rows = accounting["planned_totals"]["offline_trunk_target_rows"]
    report.check("quota:physical_steps_within_normal_cap",
                 steps <= caps["physical_optimizer_step"]["normal_cap"],
                 f"{steps} planned vs normal cap {caps['physical_optimizer_step']['normal_cap']} "
                 f"(hard {caps['physical_optimizer_step']['hard_cap']})")
    report.check("quota:offline_rows_within_normal_cap",
                 rows <= caps["offline_trunk_target_row"]["normal_cap"],
                 f"{rows} planned vs normal cap {caps['offline_trunk_target_row']['normal_cap']} "
                 f"(hard {caps['offline_trunk_target_row']['hard_cap']})")
    per_charge = verified_per_charge_cap(plan, report)
    largest = accounting["per_native_update"]["largest_single_row_charge"]
    if per_charge <= 0:
        report.check("quota:largest_single_charge", False,
                     "the per-charge cap could not be verified from a verified source; refusing "
                     "to compare against an unverified value")
        return
    report.check("quota:largest_single_charge", largest <= per_charge,
                 f"largest single charge {largest} vs verified per-charge cap {per_charge}")
    env_state = json.loads((root / "r75_environment_construction_guard.json").read_text(encoding="utf-8"))
    report.check("quota:environment_forbidden",
                 int(env_state["normal_cap"]) == 0 and int(env_state["hard_cap"]) == 0,
                 "environment construction stays 0/0/0 and is refused by source")


def check_living_config_is_not_literal(plan: dict, report: Report) -> None:
    """The future S1 record must carry the LIVE module values, not runner-chosen literals.

    A literal in the driver is only acceptable when it is the frozen contract value AND the
    precheck re-derives it from the canonical source here. Everything else must be read from the
    live module namespace at run time (``common.EPOCHS_PER_BATCH`` and friends), so a drifted
    default can never be mistaken for a verified living value.
    """
    tree = ast.parse(RUNNER_SRC)
    module_level_literals: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, (int, float)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    module_level_literals[target.id] = node.value.value
    forbidden = ("EPOCHS", "MINIBATCH", "CLIP", "GAE", "ETA", "GAMMA", "BETA", "READY_CUT",
                 "MAX_GRAD", "DATA_ORDER", "B0", "LOG_PROB", "ENTROPY", "VALUE_COEFFICIENT")
    literal_config = sorted(name for name in module_level_literals
                            if any(token in name.upper() for token in forbidden))
    report.check("config:no_literal_config_in_runner", not literal_config,
                 f"module-level config literals in the runner: {literal_config or 'none'} "
                 f"(LR is the only permitted literal and is re-derived below)")
    live_reads = ("common.EPOCHS_PER_BATCH", "common.MINIBATCH_STEPS", "common.CLIP",
                  "common.GAE_LAMBDA", "common.ETA_LAMBDA", "common.GAMMA",
                  "common.LOG_PROB_TOLERANCE")
    missing = [token for token in live_reads if token not in RUNNER_SRC]
    report.check("config:runner_reads_live_values", not missing,
                 f"live reads missing from the runner: {missing or 'none'}")
    report.check("config:runner_builds_configuration_from_live_values",
                 "def _live_configuration(" in RUNNER_SRC
                 and "configuration = _live_configuration(" in RUNNER_SRC,
                 "the S1 record's configuration is built by _live_configuration() from the live "
                 "module namespace, not written out as literals")
    # the one permitted literal, LR, must equal the frozen contract value re-derived from r66
    contract_lr = re.search(r"\blr=([0-9.eE+-]+)", R66_SRC)
    if report.check("config:frozen_lr_derivable", contract_lr is not None,
                    "the r66 frozen contract states lr"):
        expected = float(contract_lr.group(1))
        actual = module_level_literals.get("LR")
        report.check("config:runner_lr_equals_frozen_contract", actual == expected,
                     f"runner LR {actual!r} vs frozen contract lr {expected!r}")
    facts = plan["input_identity"]["batch_facts"]
    for key, expected in (("lambda", 1.0), ("lambda_at_rollout", 1.0), ("eta_lambda", 0.5),
                          ("b0_bytes", 10000.0), ("budget_bytes", 29300)):
        report.check(f"config:batch_fact:{key}", facts.get(key) == expected,
                     f"batch fact {key} = {facts.get(key)!r}, expected {expected!r}")


def check_living_config(plan: dict, report: Report) -> None:
    distinction = plan["source_identity"]["gae_default_distinction"]
    common_path = resolve_path(plan["source_identity"]["canonical_common"]["path"])
    common_src = common_path.read_text(encoding="utf-8")
    report.check("config:common_defaults_are_the_old_ones",
                 "GAE_LAMBDA = 0.95" in common_src and "ETA_LAMBDA = 0.05" in common_src,
                 distinction["v2_common_default"])
    report.check("config:r66_overrides_to_the_living_values",
                 "common.GAE_LAMBDA = 1.0" in R66_SRC and "common.ETA_LAMBDA = 0.5" in R66_SRC,
                 distinction["r66_living_override"])
    report.check("config:r75_does_not_redefine_gae",
                 "GAE_LAMBDA =" not in FROZEN_SRC and "ETA_LAMBDA =" not in FROZEN_SRC,
                 "r75_frozen inherits the living values; it defines no GAE/ETA constant of its own")


def check_final_prep_fixes(plan: dict, report: Report) -> None:
    """The two S0 final-prep fixes must be present before the paid path may run."""
    report.check("fix:kl_return_reads_a_real_accumulator",
                 'totals["kl"] / minibatches' in FROZEN_SRC
                 and 'totals["approx_kl"]' not in FROZEN_SRC,
                 'the returned approx_kl reads totals["kl"] (r66_frozen.py:875 reads totals["kl"])')
    report.check("fix:epoch_meter_differences_cumulative_snapshots",
                 "alignment_end = meter.snapshot()" in FROZEN_SRC
                 and "_meter_delta(alignment_end, meter.snapshot())" in FROZEN_SRC
                 and "_meter_delta(alignment_meter, meter.snapshot())" not in FROZEN_SRC,
                 "the epoch phase differences the cumulative alignment-end snapshot")
    report.check("fix:phase_additivity_invariant_holds",
                 "alignment + epochs == total for EVERY component"
                 in json.dumps(plan["metering_contract"]["phase_split"]),
                 "the frozen metering contract states the invariant")


def check_import_bootstrap(plan: dict, report: Report) -> None:
    """The canonical import bootstrap must be wired BEFORE the canonical modules are used.

    The first S1 attempt failed at r75_frozen.py:61 with ModuleNotFoundError: No module named
    'v2_trainer', because the canonical directories were never put on sys.path. The repair routes
    the EXISTING R45/R58 wiring through r66_frozen.configure_source_imports() and then asserts the
    bindings. This check is a pure AST pass over the source on disk: it imports nothing, so it
    cannot itself be mistaken for an import qualification.
    """
    identity = plan["source_identity"]
    tree = ast.parse(FROZEN_SRC)
    helper_line = None
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CANONICAL_SOURCES":
                    helper_line = node.lineno
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("v2_trainer", "v2_common", "onpolicy", "cost_math",
                                  "v2_controller"):
                    imports.append((node.lineno, alias.name))
        if isinstance(node, ast.ImportFrom) and node.module == "onpolicy":
            imports.append((node.lineno, "onpolicy"))
    first_import = min((line for line, _ in imports), default=None)
    report.check("bootstrap:helper_called_before_canonical_imports",
                 helper_line is not None and bool(imports) and helper_line < first_import,
                 f"CANONICAL_SOURCES = r66.configure_source_imports() at line {helper_line}; "
                 f"first canonical import at line {first_import}")
    verify_called = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "_verify_canonical_bindings"
                        for node in ast.walk(tree))
    report.check("bootstrap:bindings_verified_at_import", verify_called,
                 "_verify_canonical_bindings() is invoked at module level")
    pinned = re.search(r'EXPECTED_ONPOLICY_SHA256 = "([0-9a-f]{64})"', FROZEN_SRC)
    report.check("bootstrap:onpolicy_pin_present", pinned is not None,
                 "r75_frozen pins the onpolicy source sha256")
    if pinned:
        onpolicy_path = resolve_path(identity["canonical_onpolicy"]["path"])
        report.check("bootstrap:onpolicy_pin_matches_disk_and_plan",
                     pinned.group(1) == sha256_file(onpolicy_path)
                     == identity["canonical_onpolicy"]["expected"],
                     f"pinned {pinned.group(1)[:16]} vs disk and plan")
    report.check("bootstrap:helper_source_is_the_pinned_r66",
                 sha256_file(resolve_path(identity["r66_frozen"]["path"]))
                 == identity["r66_frozen"]["expected"],
                 "the helper is the pinned r66_frozen.py, not a local reimplementation")


def main(plan_path=None) -> int:
    global PLAN_PATH
    if plan_path is not None:
        PLAN_PATH = Path(plan_path).resolve()
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    report = Report()
    check_source_identity(plan, report)
    check_guard_pin(plan, report)
    check_input_identity(plan, report)
    check_batch_shape(plan, report)
    if plan.get("revision_id") == "R75_S1G_AR1_PRECHECK_BASELINE_REPAIR_20261015":
        check_active_production_baseline(plan, report)
        check_s1g_headroom(plan, report)
    else:
        check_ledger_snapshot(plan, report)
        check_quota_headroom(plan, report)
    check_living_config(plan, report)
    check_living_config_is_not_literal(plan, report)
    check_final_prep_fixes(plan, report)
    check_import_bootstrap(plan, report)

    verdict = {
        "kind": "round75-s1-precheck/1",
        "revision_id": plan.get("revision_id"),
        "plan_path": str(PLAN_PATH),
        "request_id": plan["request_id"],
        "plan_status": plan["status"],
        "head_at_writing": plan["head_at_writing"],
        "passed": report.passed,
        "checks_run": len(report.checks),
        "checks_failed": [item for item in report.checks if not item["ok"]],
        "costs": {"torch_imported": False, "model_load_or_construction": 0, "forward": 0,
                  "backward": 0, "optimizer_step": 0, "ppo_update": 0, "environment": 0,
                  "sampling": 0, "checkpoint_write": 0, "ledger_charged_events": 0},
        "meaning": ("PASS means the frozen S1 plan, its sources, its batch and its zero ledger are "
                    "intact; it does NOT authorize the paid S1 execution."),
    }
    print(json.dumps(verdict, indent=2))
    for item in report.checks:
        if not item["ok"]:
            print(f"FAILED: {item['check']}: {item['detail']}", file=sys.stderr)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
