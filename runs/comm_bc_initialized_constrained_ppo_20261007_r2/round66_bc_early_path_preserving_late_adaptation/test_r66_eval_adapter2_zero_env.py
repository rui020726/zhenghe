"""ROUND66 zero-environment acceptance for the corrected evaluation adapter v2 (r66_eval_fixed2.py).

Pure computation only: no builder, no environment constructor, no optimizer step, no checkpoint
write, and no claim or slot in the real R66 raw root. The retry / cap / slot seams are exercised
against the REAL r40_common claim primitives pointed at a scratch directory, so the discipline under
test is the same code that will run, not a re-implementation.

What this file deliberately does NOT claim:
  * passing these checks qualifies no evaluation cell; only a real claim + builder does that;
  * the corrected matrix is a plan correction, not a change to the scientific configuration;
  * no gradient, likelihood or checkpoint behaviour is re-proved here.
"""
from __future__ import annotations
import contextlib, json, os, shutil, sys
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RUN_ROOT.parent / "comm_first_report_training_ablation_20261003"))
import r66_eval_fixed2 as v2
import r66_execute as base
import r66_frozen as r66mod
import r40_common

RESULTS: list[tuple[str, bool, dict]] = []
SYNTHETIC_OPTIMIZER_UPDATES = 0


def check(name, cond, **detail):
    cond = bool(cond)
    RESULTS.append((name, cond, detail))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  " + json.dumps(detail, default=str)),
          flush=True)
    return cond


@contextlib.contextmanager
def scratch():
    p = HERE / "_eval_adapter2_scratch"
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
    try:
        yield p
    finally:
        shutil.rmtree(p, ignore_errors=True)


# ================================================================ E3: plan vs request order ==
plan = v2.build_eval_plan()
cells = list(v2.eval_job_order())
check("plan_matches_the_single_order_generator",
      [j["logical_job"] for j in plan["jobs"]] == [v2.logical_job(c) for c in cells])
check("matrix_is_96_576_6",
      plan["matrix"] == {"rule_eval": 96, "model_eval": 576, "focal": 6, "total": 678},
      matrix=plan["matrix"])
scopes = [j["scope"] for j in plan["jobs"]]
check("executor_order_is_rules_then_model_then_focal",
      scopes == ["rule_eval"] * 96 + ["model_eval"] * 576 + ["focal"] * 6)
model_cells = [j for j in plan["jobs"] if j["scope"] == "model_eval"]
expected_model = [(int(seed), arm, mode, int(scene))
                  for seed in v2.SEEDS for arm in v2.MODEL_ARMS for mode in v2.MODES
                  for scene in v2.SCENES_NEW]
check("model_cells_iterate_seed_then_arm_then_mode_then_scene",
      [(j["seed"], j["arm"], j["mode"], j["scene"]) for j in model_cells] == expected_model,
      first=[(j["seed"], j["arm"], j["mode"], j["scene"]) for j in model_cells[:3]],
      expected_first=expected_model[:3])
check("rule_cells_come_first_and_are_greedy_only",
      all(j["mode"] == "greedy" and j["seed"] is None for j in plan["jobs"][:96]))
check("focal_cells_are_new_and_bc_greedy_on_1700027",
      all(j["mode"] == "greedy" and j["scene"] == 1700027 and j["arm"] in ("NEW", "BC")
          for j in plan["jobs"][-6:]))
planned = [j["planned_normal_slot"] for j in plan["jobs"]]
check("planned_slots_are_the_contiguous_range_433_to_1110", planned == list(range(433, 1111)))
check("slot_mechanism_is_declared_as_pool_allocation_not_a_binding",
      plan["slot_allocation"]["mechanism"].startswith("r40_common._reserve_slot")
      and plan["slot_allocation"]["planned_field"].startswith("planned_normal_slot is a plan"))
check("already_numbered_cells_plus_eval_equals_the_normal_cap",
      plan["already_numbered_cells"] + plan["matrix"]["total"] == v2.NORMAL_CAP == 1110)

# ================================================================ E1: reserve cap / hard cap =
check("booked_reserve_ceiling_is_39_not_the_inherited_40",
      v2.EVAL_RESERVE_CAP == 39 and v2.RESERVE_CAP_INHERITED == 40 and v2.NORMAL_CAP == 1110
      and v2.HARD_CAP == 1150,
      booked=v2.EVAL_RESERVE_CAP, inherited=v2.RESERVE_CAP_INHERITED)
check("d1_extra_construction_is_counted_in_the_arithmetic",
      v2.UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS == 1
      and v2.NORMAL_CAP + v2.EVAL_RESERVE_CAP + v2.UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS
      == v2.HARD_CAP,
      arithmetic=(v2.NORMAL_CAP, v2.EVAL_RESERVE_CAP,
                  v2.UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS, v2.HARD_CAP))


class _StubCommon:
    GAE_LAMBDA = 1.0
    ETA_LAMBDA = 0.5
    RESERVE_LAUNCHES = 40
    NORMAL_LAUNCHES = 1110


class _StubR40:
    def __init__(self, normal_used=0, reserve_used=0):
        self.NORMAL_CAP = 1110
        self.RESERVE_CAP = 40
        self.HARD_CAP = 1150
        self.normal_used = normal_used
        self.reserve_used = reserve_used

    def plan(self):
        return {"normal_total": 1110, "technical_reserve_max": 40, "hard_cap": 1150}

    def ledger_snapshot(self):
        if self.normal_used > self.NORMAL_CAP or self.reserve_used > self.RESERVE_CAP:
            raise RuntimeError("stub ledger exceeds its immutable launch cap")
        return {"normal_used": self.normal_used, "reserve_used": self.reserve_used,
                "total_used": self.normal_used + self.reserve_used}


def _stub_cfg(normal_used=0, reserve_used=0):
    r40 = _StubR40(normal_used, reserve_used)
    return {"runtime": {"r40_common": r40}, "common": _StubCommon()}, r40


def _configure(normal_used=0, reserve_used=0):
    cfg, r40 = _stub_cfg(normal_used, reserve_used)
    saved = base.configure_and_inject
    base.configure_and_inject = lambda _raw: cfg
    try:
        out = v2.configure_r66_eval(Path("unused"))
    finally:
        base.configure_and_inject = saved
    return out["r66_eval_quota_guard"], r40


guard, r40 = _configure(0, 0)
check("configure_r66_eval_rebinds_reserve_on_the_live_modules",
      int(r40.RESERVE_CAP) == 39 and int(r40.plan()["technical_reserve_max"]) == 39,
      r40_RESERVE_CAP=int(r40.RESERVE_CAP), plan_reserve=r40.plan()["technical_reserve_max"])
check("quota_guard_counts_the_d1_extra_construction",
      guard["booked_total"] == 0 and guard["actual_environment_constructions"] == 1
      and guard["hard_headroom"] == 1149 and guard["booked_reserve_remaining"] == 39,
      guard={k: guard[k] for k in ("booked_total", "actual_environment_constructions",
                                   "hard_headroom", "booked_reserve_remaining")})
check("quota_guard_records_live_bindings",
      guard["live_bindings"]["r40_RESERVE_CAP"] == 39
      and guard["live_bindings"]["common_RESERVE_LAUNCHES"] == 39
      and guard["live_bindings"]["r40_HARD_CAP"] == 1150,
      live=guard["live_bindings"])

full_cfg, full_r40 = _stub_cfg(1110, 39)
full_r40.RESERVE_CAP = 39
full = v2.quota_guard(full_cfg)
check("quota_guard_accepts_the_full_1110_plus_39_booked_reserve",
      full["actual_environment_constructions"] == 1150 and full["hard_headroom"] == 0,
      actual=full["actual_environment_constructions"], headroom=full["hard_headroom"])
check("configure_r66_eval_itself_accepts_the_fully_booked_quota",
      _configure(1110, 39)[0]["actual_environment_constructions"] == 1150)

over_cfg, over_r40 = _stub_cfg(1110, 40)
over_r40.RESERVE_CAP = 39
try:
    v2.quota_guard(over_cfg)
    refused = False
except RuntimeError:
    refused = True
check("quota_guard_refuses_a_40th_booked_reserve_slot", refused)

saved_extra = v2.UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS
v2.UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS = 2
try:
    v2.quota_guard(full_cfg)
    refused = False
except RuntimeError:
    refused = True
finally:
    v2.UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS = saved_extra
check("quota_guard_refuses_when_the_unregistered_extra_would_break_the_hard_cap", refused)


# ================================================================ E2: retry discipline =======
def _fresh_r40(root: Path, normal_cap=1110, reserve_cap=39):
    r40_common.ROUND_DIR = root
    r40_common.NORMAL_CAP = normal_cap
    r40_common.RESERVE_CAP = reserve_cap
    r40_common.HARD_CAP = normal_cap + reserve_cap
    return r40_common


def _cfg_with(root: Path):
    return {"runtime": {"r40_common": _fresh_r40(root)},
            "r42_run": SimpleNamespace(_ACTIVE_CLAIM=None, _ACTIVE_PROGRESS=None,
                                       _new_run_progress=lambda: {"steps": 0})}


def _claim(root: Path, logical: str, attempt: int, status: str, category: str = "normal"):
    d = root / "launch_claims" / r40_common._slug(logical)
    d.mkdir(parents=True, exist_ok=True)
    payload = {"logical_job": logical, "attempt": attempt, "category": category, "status": status}
    (d / f"attempt_{attempt:04d}.json").write_text(json.dumps(payload), encoding="utf-8")


def _make_row(root: Path, tag: str, *, arm="NEW", mode="greedy", scene=1800000, seed=11,
              omit=(), empty=(), ids=None):
    art = {}
    for key, sub in (("row", "rows"), ("lifecycle", "logs"), ("transcript", "logs"), ("tape", "logs")):
        if key in omit:
            continue
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        art[key] = str(d / f"{tag}_{key}.json")
    for key, path in art.items():
        Path(path).write_text("" if key in empty else '{"x": 1}\n', encoding="utf-8")
    if ids is None:
        ids = ["t1", "target-5"] if scene == 1800001 else ["t1"]
    return ({"artefacts": art, "on_time": len(ids), "spent_bytes": 15000, "steps": 1000,
             "model_arm": arm, "eval_mode": mode, "scenario_seed": int(scene),
             "training_seed": int(seed) if seed is not None else None,
             "task_reward_events": [{"task_id": t, "on_time": True} for t in ids],
             "on_time_target_ids": ids}, art)


def _rows_cfg(root: Path):
    rows = root / "rows"
    rows.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return {"common": SimpleNamespace(ROWS=rows),
            "runtime": {"r40_common": _fresh_r40(root)}}


def _raises(fn):
    try:
        fn()
        return False
    except RuntimeError:
        return True


def _write_claim(root: Path, logical: str, status: str, receipt, *, garbage: bool = False,
                 attempt: int = 1, category: str = "normal"):
    d = root / "launch_claims" / r40_common._slug(logical)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"attempt_{attempt:04d}.json"
    if garbage:
        path.write_text("{ this is not json", encoding="utf-8")
        return path
    path.write_text(json.dumps({"logical_job": logical, "attempt": attempt, "category": category,
                                "status": status, "result_receipt": receipt}), encoding="utf-8")
    return path


def _completed_claim(root: Path, cfg, logical: str, cell: dict, tamper=None, *,
                     attempt: int = 1, category: str = "normal", ids=None, tag=None):
    tag = tag or f"cellrow_a{attempt}"
    row, _art = _make_row(root, tag, arm=cell["arm"], mode=cell["mode"],
                          scene=cell["scene"], seed=cell.get("seed"), ids=ids)
    receipt = v2.persist_result(cfg, row, tag)
    if tamper == "no_receipt":
        receipt = None
    elif tamper == "drop_row":
        os.remove(receipt["row_path"])
    elif tamper == "tamper_row":
        Path(receipt["row_path"]).write_text('{"on_time": 999}\n', encoding="utf-8")
    elif tamper == "drop_companion":
        os.remove(next(iter(receipt["companions"].values()))["path"])
    elif tamper == "empty_companion":
        Path(next(iter(receipt["companions"].values()))["path"]).write_text("", encoding="utf-8")
    elif tamper in ("identity", "no_steps"):
        reread = base._read_json(receipt["row_path"])
        reread["model_arm" if tamper == "identity" else "steps"] = "BC" if tamper == "identity" else 0
        Path(receipt["row_path"]).write_text(json.dumps(reread), encoding="utf-8")
        receipt = dict(receipt, row_sha256=base._sha256_file(receipt["row_path"]))
    _write_claim(root, logical, "complete", receipt, attempt=attempt, category=category)


with scratch() as root:
    cfg = _cfg_with(root)
    logical = "r66eval2_model_new_s11_greedy_scene1800000"
    cp, attempt, run_tag = v2._prepare_claim(cfg, logical, {"arm": "NEW"}, "t_model_new_s11_greedy")
    check("prepare_claim_on_a_fresh_cell_takes_attempt_1_in_the_normal_category",
          attempt == 1 and run_tag == "t_model_new_s11_greedy_a1"
          and json.loads(cp.read_text(encoding="utf-8"))["category"] == "normal")
    check("prepare_claim_on_a_fresh_cell_reserves_one_normal_slot",
          len(list((root / "launch_slots" / "normal").glob("*.json"))) == 1
          and len(list((root / "launch_slots" / "reserve").glob("*.json"))) == 0)

with scratch() as root:
    cfg = _cfg_with(root)
    logical = "r66eval2_model_new_s11_greedy_scene1800000"
    cfg["common"] = SimpleNamespace(ROWS=(root / "rows"))
    seen = {}

    def _record(tag):
        seen["tag"] = tag
        row, _art = _make_row(root, tag)
        return row

    row = v2._safe_atomic_run(cfg, logical, {"arm": "NEW"}, _record,
                              detail=lambda r: "ok", tag="t_model_new_s11_greedy",
                              persist=lambda r, t: v2.persist_result(cfg, r, t))
    claim = json.loads((root / "launch_claims" / r40_common._slug(logical)
                        / "attempt_0001.json").read_text(encoding="utf-8"))
    check("safe_run_on_a_fresh_cell_runs_attempt_1_normal_and_completes_it",
          claim["attempt"] == 1 and claim["category"] == "normal"
          and claim["status"] == "complete" and row["claim_attempt"] == 1,
          claim_attempt=claim["attempt"], category=claim["category"], status=claim["status"])
    check("safe_run_passes_the_unmarked_attempt_1_tag_to_the_builder",
          seen.get("tag") == "t_model_new_s11_greedy_a1", tag=seen.get("tag"))
    check("safe_run_leaves_exactly_one_claim_and_one_normal_slot",
          len(list((root / "launch_claims" / r40_common._slug(logical)).glob("attempt_*.json"))) == 1
          and len(list((root / "launch_slots" / "normal").glob("*.json"))) == 1
          and len(list((root / "launch_slots" / "reserve").glob("*.json"))) == 0)
    check("a_complete_cell_is_accepted_read_only_without_rerunning",
          v2._cell_complete(cfg, logical) is True)

with scratch() as root:
    cfg = _cfg_with(root)
    logical = "r66eval2_rule_immediate_greedy_scene1800000"
    _claim(root, logical, 1, "claimed_before_environment")
    original = (root / "launch_claims" / r40_common._slug(logical) / "attempt_0001.json").read_bytes()
    check("cell_with_a_complete_attempt_is_skipped_read_only", v2._cell_complete(cfg, logical) is False)
    cp, attempt, run_tag = v2._prepare_claim(cfg, logical, {"arm": "IMMEDIATE"}, "t_rule_immediate")
    check("claimed_before_environment_attempt_is_never_reused",
          attempt == 2 and run_tag == "t_rule_immediate_a2"
          and json.loads(cp.read_text(encoding="utf-8"))["category"] == "reserve",
          attempt=attempt, run_tag=run_tag,
          category=json.loads(cp.read_text(encoding="utf-8"))["category"])
    check("the_original_claim_is_preserved_byte_identical",
          (root / "launch_claims" / r40_common._slug(logical) / "attempt_0001.json").read_bytes()
          == original)
    meta = json.loads(cp.read_text(encoding="utf-8"))["metadata"]
    check("the_retry_records_what_it_retries_and_why",
          meta["retry_of_attempt"] == 1 and meta["prior_statuses"] == ["claimed_before_environment"]
          and meta["retry_artifact_isolated"] is True, metadata=meta)
    check("the_retry_reserves_a_reserve_slot_not_a_normal_one",
          len(list((root / "launch_slots" / "reserve").glob("*.json"))) == 1
          and len(list((root / "launch_slots" / "normal").glob("*.json"))) == 0)

for status in ("failed_technical", "not_started_quota_exhausted", "claiming", "unreadable"):
    with scratch() as root:
        cfg = _cfg_with(root)
        logical = f"r66eval2_model_old_s22_sampled_scene1800005_{status}"
        _claim(root, logical, 1, status)
        cp, attempt, run_tag = v2._prepare_claim(cfg, logical, {"arm": "OLD"}, "t_old_s22")
        check(f"status_{status}_also_takes_a_new_reserve_attempt",
              attempt == 2 and run_tag == "t_old_s22_a2"
              and json.loads(cp.read_text(encoding="utf-8"))["category"] == "reserve",
              attempt=attempt)

with scratch() as root:
    cfg = _cfg_with(root)
    cfg["common"] = SimpleNamespace(ROWS=(root / "rows"))
    logical = "r66eval2_focal_bc_s33_scene1700027"
    _claim(root, logical, 1, "failed_technical")
    _completed_claim(root, cfg, logical, {"arm": "BC", "mode": "greedy",
                                         "scene": 1700027, "seed": 33}, attempt=2,
                     category="reserve")
    check("a_complete_retry_attempt_makes_the_cell_complete", v2._cell_complete(cfg, logical))
    try:
        v2._prepare_claim(cfg, logical, {"arm": "BC"}, "t_focal_bc_s33")
        raised = False
    except v2.AlreadyComplete:
        raised = True
    check("prepare_claim_refuses_to_open_a_third_attempt_when_one_is_complete", raised)

with scratch() as root:
    cfg = _cfg_with(root)
    logical = "r66eval2_model_bc_s11_greedy_scene1800009"
    _claim(root, logical, 1, "claimed_before_environment")
    cfg["r42_run"]._ACTIVE_CLAIM = Path("busy")
    try:
        v2._safe_atomic_run(cfg, logical, {}, lambda tag: {"on_time": 1}, detail=lambda r: "x",
                            tag="t_bc_s11")
        raised = False
    except RuntimeError:
        raised = True
    finally:
        cfg["r42_run"]._ACTIVE_CLAIM = None
    check("safe_run_refuses_nested_claims", raised)

with scratch() as root:
    cfg = _cfg_with(root)
    logical = "r66eval2_model_new_s22_sampled_scene1800011"
    seen = {}

    def _record(tag):
        seen["tag"] = tag
        return {"on_time": 5}

    row = v2._safe_atomic_run(cfg, logical, {"arm": "NEW"}, _record,
                              detail=lambda r: "ok", tag="t_new_s22")
    check("safe_run_passes_the_attempt_isolated_tag_to_the_builder",
          seen.get("tag") == "t_new_s22_a1" and row["claim_attempt"] == 1, seen=seen)

with scratch() as root:
    cfg = _cfg_with(root)
    logical = "r66eval2_model_old_s33_greedy_scene1800013"
    try:
        v2._safe_atomic_run(cfg, logical, {"arm": "OLD"},
                            lambda tag: (_ for _ in ()).throw(ValueError("boom")),
                            detail=lambda r: "never", tag="t_old_s33")
        raised = False
    except ValueError:
        raised = True
    finally:
        pass
    claim = json.loads((root / "launch_claims" / r40_common._slug(logical)
                        / "attempt_0001.json").read_text(encoding="utf-8"))
    check("a_failed_builder_marks_the_claim_failed_technical_and_propagates",
          raised and claim["status"] == "failed_technical", claim=claim.get("status"))

# ================================================================ order assertion ============
expected = [j["logical_job"] for j in plan["jobs"]]
v2.VISITED[:] = expected
check("assert_visited_order_accepts_the_planned_order",
      v2._assert_visited_order(plan)["status"] == "passed")
v2.VISITED[:] = expected[:-1] + ["wrong"]
try:
    v2._assert_visited_order(plan)
    raised = False
except AssertionError:
    raised = True
check("assert_visited_order_rejects_a_divergent_execution_order", raised)
v2.VISITED[:] = []

# ================================================================ runtime binding ============
with scratch() as root:
    def _boom(_raw_root):
        raise AssertionError("the corrected plan phase must not configure the simulation runtime")

    saved = base.configure_and_inject
    base.configure_and_inject = _boom
    try:
        rc = v2.main(["--raw-root", str(root), "--phase", "plan"])
    finally:
        base.configure_and_inject = saved
    check("plan_phase_returns_zero_without_configuring_the_runtime", rc == 0)
    written = json.loads((root / "ROUND66_EVAL_PLAN_CORRECTED_2.json").read_text(encoding="utf-8"))
    check("plan_phase_writes_the_v2_plan_file",
          written["matrix"]["total"] == 678 and written["kind"].endswith("/2"))

    reached = False

    def _boom2(_raw_root):
        raise AssertionError("runtime binding reached")

    base.configure_and_inject = _boom2
    try:
        v2.main(["--raw-root", str(root / "missing"), "--phase", "eval"])
    except AssertionError:
        reached = True
    except Exception:
        reached = True
    finally:
        base.configure_and_inject = saved
    check("eval_phase_binds_the_live_runtime_before_anything_else", reached)

    # the eval path must go through configure_r66_eval, i.e. the quota guard cannot be bypassed
    calls = {"n": 0}
    real_configure = v2.configure_r66_eval

    def _spy(raw_root):
        calls["n"] += 1
        raise AssertionError("configure_r66_eval reached")

    v2.configure_r66_eval = _spy
    try:
        v2.main(["--raw-root", str(root / "missing2"), "--phase", "eval"])
    except AssertionError:
        reached = True
    except Exception:
        reached = True
    finally:
        v2.configure_r66_eval = real_configure
    check("eval_phase_cannot_bypass_the_quota_guard", reached and calls["n"] == 1)

# ================================================================ namespace / identity =======
v2_names = ([f"v2_{v2.ROW_TAG}_model_{a.lower()}_s{s}_{m}_scene{sc}_a1.json"
             for a in v2.MODEL_ARMS for s in v2.SEEDS for m in v2.MODES for sc in v2.SCENES_NEW]
            + [f"v2_{v2.ROW_TAG}_rule_{a}_greedy_scene{sc}_a1.json"
               for a in v2.RULE_ARMS for sc in v2.SCENES_NEW]
            + [f"v2_{v2.ROW_TAG}_focal_{a.lower()}_s{s}_scene{v2.FOCAL_SCENE}_a1.json"
               for s in v2.SEEDS for a in ("NEW", "BC")])
check("v2_row_names_are_678", len(v2_names) == 678, n=len(v2_names))
older = ["v2_r66_eval_new_*.json", "v2_r66_eval_bc_*.json", "v2_r66_rule_*.json",
         "v2_r66_focal_*.json", "v2_r66eval_model_*.json", "v2_r66eval_rule_*.json",
         "v2_r66eval_focal_*.json"]
leak = [n for n in v2_names if any(fnmatch(n, g) for g in older)]
check("v2_rows_cannot_match_any_earlier_namespace", not leak, leaked=leak[:3])
v2_globs = [f"v2_{v2.ROW_TAG}_model_*.json", f"v2_{v2.ROW_TAG}_rule_*.json", f"v2_{v2.ROW_TAG}_focal_*.json"]
old_names = ["v2_r66_eval_new_s11_greedy_scene1800000_a1.json",
             "v2_r66_rule_immediate_greedy_scene1800000_a1.json",
             "v2_r66_focal_new_s11_scene1700027_a1.json",
             "v2_r66eval_model_new_s11_greedy_scene1800000_a1.json",
             "v2_r66eval_rule_immediate_greedy_scene1800000_a1.json"]
back = [n for n in old_names if any(fnmatch(n, g) for g in v2_globs)]
check("earlier_namespaces_cannot_match_the_v2_globs", not back, matched=back)
check("v2_version_extends_the_r66_training_version",
      v2.EVAL_VERSION.startswith(r66mod.R66_TRAINING_VERSION)
      and v2.EVAL_VERSION.endswith("+r66-eval-matrix-correction/2"),
      version=v2.EVAL_VERSION[-40:])
check("v2_shares_the_r66_task_branch_and_hard_cap",
      base.TASK_ID == "ROUND66-BC-EARLY-PATH-PRESERVING-LATE-ADAPTATION"
      and base.BRANCH == "exp/round66-bc-early-path-preserving-late-adaptation"
      and v2.HARD_CAP == base.HARD_CAP == 1150)
idx, sha = v2._bootstrap_index()
check("bootstrap_index_is_the_frozen_pcg64_20261012_draw",
      idx.shape == (10000, 32) and sha == v2._bootstrap_index()[1], sha=sha[:16])

# ================================================================ E5: row before commit ====


with scratch() as root:
    cfg = _rows_cfg(root)
    row, _art = _make_row(root, "good")
    receipt = v2.persist_result(cfg, row, "good_a1")
    check("persist_result_writes_the_row_and_records_a_receipt",
          Path(receipt["row_path"]).is_file() and bool(receipt.get("row_sha256"))
          and sorted(receipt["companions"]) == ["lifecycle", "tape", "transcript"],
          companions=sorted(receipt.get("companions") or {}))

    bad, _ = _make_row(root, "no_tape", omit=("tape",))
    try:
        v2.persist_result(cfg, bad, "no_tape_a1")
        raised = False
    except RuntimeError:
        raised = True
    check("persist_result_refuses_a_row_missing_a_companion_key", raised)

    empty_row, _ = _make_row(root, "empty_tape", empty=("tape",))
    try:
        v2.persist_result(cfg, empty_row, "empty_tape_a1")
        raised = False
    except RuntimeError:
        raised = True
    check("persist_result_refuses_an_empty_companion", raised)

    ghost, art = _make_row(root, "ghost")
    art["tape"] = str(root / "logs" / "does_not_exist.json")
    try:
        v2.persist_result(cfg, ghost, "ghost_a1")
        raised = False
    except RuntimeError:
        raised = True
    check("persist_result_refuses_a_companion_that_is_not_on_disk", raised)

with scratch() as root:
    cfg = _cfg_with(root)
    cfg["common"] = SimpleNamespace(ROWS=(root / "rows"))
    logical = "r66eval2_model_new_s11_greedy_scene1800001"

    def _boom_persist(_row, _tag):
        raise RuntimeError("simulated row-write failure")

    try:
        v2._safe_atomic_run(cfg, logical, {"arm": "NEW"}, lambda tag: _make_row(root, tag)[0],
                            detail=lambda r: "never", tag="t_new_s11", persist=_boom_persist)
        raised = False
    except RuntimeError:
        raised = True
    claim = json.loads((root / "launch_claims" / r40_common._slug(logical)
                        / "attempt_0001.json").read_text(encoding="utf-8"))
    check("a_failing_row_write_leaves_failed_technical_and_never_complete",
          raised and claim["status"] == "failed_technical" and "result_receipt" not in claim,
          status=claim.get("status"))

with scratch() as root:
    cfg = _cfg_with(root)
    cfg["common"] = SimpleNamespace(ROWS=(root / "rows"))
    logical = "r66eval2_model_old_s22_sampled_scene1800002"
    cell = {"arm": "OLD", "mode": "sampled", "scene": 1800002, "seed": 22}
    row = v2._safe_atomic_run(cfg, logical, {"arm": "OLD"},
                              lambda tag: _make_row(root, tag, arm="OLD", mode="sampled",
                                                    scene=1800002, seed=22)[0],
                              detail=lambda r: "ok", tag="t_old_s22",
                              persist=lambda r, t: v2.persist_result(cfg, r, t))
    check("safe_run_commits_only_after_the_row_is_durable",
          row["claim_attempt"] == 1 and v2._cell_complete(cfg, logical, cell) is True)

# ================================================================ E4: completed admission ===


with scratch() as root:
    cfg = _rows_cfg(root)
    r40 = _fresh_r40(root)
    probe = {"runtime": {"r40_common": r40}}
    logical = "r66eval2_model_new_s11_greedy_scene1800000"
    cell = {"arm": "NEW", "mode": "greedy", "scene": 1800000, "seed": 11}

    _completed_claim(root, cfg, logical, cell)
    check("a_complete_claim_with_a_sound_receipt_is_accepted_read_only",
          v2._cell_complete(probe, logical, cell) is True)

    for tamper, label in (("no_receipt", "without_a_receipt"),
                          ("drop_row", "whose_row_is_missing"),
                          ("tamper_row", "whose_row_hash_does_not_match"),
                          ("drop_companion", "whose_companion_is_missing"),
                          ("empty_companion", "whose_companion_is_empty"),
                          ("identity", "whose_row_identity_is_wrong"),
                          ("no_steps", "whose_row_records_no_steps")):
        _completed_claim(root, cfg, logical, cell, tamper=tamper)
        check(f"a_complete_claim_{label}_is_not_skipped",
              v2._cell_complete(probe, logical, cell) is False)

    _write_claim(root, logical, "complete", {"row_path": "x"}, garbage=True)
    check("an_unreadable_claim_fails_closed", v2._cell_complete(probe, logical, cell) is False)

    # E4b: an unsound complete claim must not deadlock the cell
    deadlock_logical = "r66eval2_model_new_s33_greedy_scene1800004"
    deadlock_cell = {"arm": "NEW", "mode": "greedy", "scene": 1800004, "seed": 33}
    _completed_claim(root, cfg, deadlock_logical, deadlock_cell, tamper="drop_row")
    check("an_unsound_complete_claim_is_not_skipped",
          v2._cell_complete(probe, deadlock_logical, deadlock_cell) is False)
    cp, attempt, run_tag = v2._prepare_claim(cfg, deadlock_logical, {"arm": "NEW"},
                                             "t_new_s33", deadlock_cell)
    check("prepare_claim_opens_a_reserve_retry_beside_an_unsound_complete_claim",
          attempt == 2 and run_tag == "t_new_s33_a2"
          and json.loads(cp.read_text(encoding="utf-8"))["category"] == "reserve",
          attempt=attempt)
    preserved = json.loads((root / "launch_claims" / r40_common._slug(deadlock_logical)
                            / "attempt_0001.json").read_text(encoding="utf-8"))
    check("the_unsound_complete_claim_is_preserved_untouched",
          preserved["status"] == "complete" and "result_receipt" in preserved)

    # a companion whose CONTENT changed (same size) and one whose SIZE changed are both detected
    hash_logical = "r66eval2_model_bc_s11_sampled_scene1800006"
    hash_cell = {"arm": "BC", "mode": "sampled", "scene": 1800006, "seed": 11}
    _completed_claim(root, cfg, hash_logical, hash_cell)
    receipt = json.loads((root / "launch_claims" / r40_common._slug(hash_logical)
                          / "attempt_0001.json").read_text(encoding="utf-8"))["result_receipt"]
    companion = next(iter(receipt["companions"].values()))
    original_size = int(companion["bytes"])
    Path(companion["path"]).write_text('{"x": 2}\n', encoding="utf-8")
    check("a_companion_whose_content_changed_is_detected",
          v2._cell_complete(probe, hash_logical, hash_cell) is False)
    _completed_claim(root, cfg, hash_logical, hash_cell)
    receipt = json.loads((root / "launch_claims" / r40_common._slug(hash_logical)
                          / "attempt_0001.json").read_text(encoding="utf-8"))["result_receipt"]
    companion = next(iter(receipt["companions"].values()))
    with open(companion["path"], "a", encoding="utf-8") as fh:
        fh.write("x")
    check("a_companion_whose_size_changed_is_detected",
          int(Path(companion["path"]).stat().st_size) != original_size
          and v2._cell_complete(probe, hash_logical, hash_cell) is False)

    # a sound retry attempt still qualifies even when attempt 1 is unsound
    _completed_claim(root, cfg, logical, cell, tamper="drop_row")
    other = "r66eval2_model_bc_s33_greedy_scene1800003"
    _completed_claim(root, cfg, other, {"arm": "BC", "mode": "greedy", "scene": 1800003, "seed": 33})
    check("a_sound_complete_attempt_qualifies_even_if_another_attempt_is_unsound",
          v2._cell_complete(probe, other, {"arm": "BC", "mode": "greedy", "scene": 1800003,
                                           "seed": 33}) is True)

# ================================================================ E6: paired set diffs ======
with scratch() as root:
    cfg = _rows_cfg(root)
    for seed in v2.SEEDS:
        for mode in v2.MODES:
            for arm in v2.MODEL_ARMS:
                for scene in v2.SCENES_NEW:
                    special = (seed == 11 and mode == "greedy")
                    if special and arm == "NEW":
                        ids = ["t1", "target-5"] if scene == 1800001 else ["t1"]
                    elif special and arm == "OLD":
                        ids = ["t1", "target-5"] if scene == 1800000 else ["t1"]
                    else:
                        ids = ["t1"]
                    _completed_claim(root, cfg,
                                     f"r66eval2_model_{arm.lower()}_s{seed}_{mode}_scene{scene}",
                                     {"arm": arm, "mode": mode, "scene": scene, "seed": seed},
                                     ids=ids, tag=f"m_{arm}_{seed}_{mode}_{scene}")
    # the rule arms are greedy-only cells; without them the greedy entry cannot report a rule row
    for arm in v2.RULE_ARMS:
        for scene in v2.SCENES_NEW:
            rids = ["t1"] if scene != 1800002 else ["t1", "target-5"]
            _completed_claim(root, cfg, f"r66eval2_rule_{arm}_greedy_scene{scene}",
                             {"arm": arm.upper(), "mode": "greedy", "scene": scene},
                             ids=rids, tag=f"r_{arm}_{scene}")
    stats = v2.compute_stats(root)
    entry = stats["per_seed_mode"]["11/greedy"]
    check("e6_lost_is_reported_as_scene_target_pairs",
          entry["targets_lost_NEW_vs_OLD"] == [{"scene": 1800000, "target_id": "target-5"}],
          got=entry["targets_lost_NEW_vs_OLD"])
    check("e6_gained_is_reported_as_scene_target_pairs",
          entry["targets_gained_NEW_vs_OLD"] == [{"scene": 1800001, "target_id": "target-5"}],
          got=entry["targets_gained_NEW_vs_OLD"])
    check("e6_counts_are_nonzero_where_a_merged_set_would_cancel",
          entry["targets_lost_NEW_vs_OLD_count"] == 1
          and entry["targets_gained_NEW_vs_OLD_count"] == 1)
    merged_new = {t for s in v2.SCENES_NEW for t in
                  (["t1", "target-5"] if s == 1800001 else ["t1"])}
    merged_old = {t for s in v2.SCENES_NEW for t in
                  (["t1", "target-5"] if s == 1800000 else ["t1"])}
    check("e6_rule_comparisons_appear_only_in_the_greedy_entry",
          "immediate_on_time_total" in entry
          and "rule_arms_not_comparable_in_this_mode" in stats["per_seed_mode"]["11/sampled"]
          and "immediate_on_time_total" not in stats["per_seed_mode"]["11/sampled"],
          greedy_has_rule="immediate_on_time_total" in entry,
          sampled_marks_not_applicable="rule_arms_not_comparable_in_this_mode"
          in stats["per_seed_mode"]["11/sampled"])
    check("e6_new_vs_immediate_reports_task_count_and_target_set",
          "task_delta_NEW_minus_immediate_mean" in entry
          and "targets_lost_NEW_vs_immediate" in entry
          and "NEW_minus_immediate_bytes_B_per_scene" in entry)
    check("the_merged_set_would_have_cancelled_this_loss_so_the_test_is_meaningful",
          merged_new == merged_old and entry["task_delta_NEW_minus_OLD_mean"] == 0.0,
          merged_new=sorted(merged_new), merged_old=sorted(merged_old),
          task_delta=entry["task_delta_NEW_minus_OLD_mean"])
    check("stats_still_covers_all_three_seeds_and_both_modes",
          sorted(stats["per_seed_mode"]) == ["11/greedy", "11/sampled", "22/greedy", "22/sampled",
                                             "33/greedy", "33/sampled"]
          and stats["cells"] == 576, cells=stats.get("cells"))

# ================================================================ E7: evidenced target sets =
with scratch() as root:
    events = [{"task_id": "t1", "on_time": True}, {"task_id": "target-5", "on_time": True}]
    check("e7_extracts_the_evidenced_on_time_set",
          v2.on_time_target_ids({"task_reward_events": events, "on_time": 2}, 1800000)
          == ["t1", "target-5"])
    check("e7_ignores_the_absent_target_id_field",
          v2.on_time_target_ids({"task_reward_events": events, "on_time": 2,
                                 "on_time_target_ids": [], "on_time_ids": None}, 1800000)
          == ["t1", "target-5"])
    check("e7_skips_events_that_are_not_on_time",
          v2.on_time_target_ids({"task_reward_events": events + [{"task_id": "t9",
                                                                  "on_time": False}],
                                 "on_time": 2}, 1800000) == ["t1", "target-5"])
    cases = (
        ({"task_reward_events": [], "on_time": 2}, "no_events"),
        ({"on_time": 2}, "missing_events"),
        ({"task_reward_events": "nope", "on_time": 2}, "events_not_a_list"),
        ({"task_reward_events": events, "on_time": 3}, "count_mismatch"),
        ({"task_reward_events": [{"task_id": "t1", "on_time": True},
                                 {"task_id": "t1", "on_time": True}], "on_time": 2}, "duplicate_ids"),
        ({"task_reward_events": [{"task_id": "t1", "on_time": True},
                                 {"on_time": True}], "on_time": 2}, "event_without_task_id"),
        ({"task_reward_events": ["nope"], "on_time": 1}, "malformed_event"),
        ({"task_reward_events": events, "on_time": True}, "on_time_not_an_int"),
    )
    for bad, label in cases:
        try:
            v2.on_time_target_ids(bad, 1800000)
            raised = False
        except RuntimeError:
            raised = True
        check(f"e7_refuses_{label}_instead_of_writing_an_empty_set", raised)

    # the loader must surface an unclosed set instead of silently producing an empty one
    cfg = _rows_cfg(root)
    logical = "r66eval2_model_new_s11_greedy_scene1800000"
    cell = {"arm": "NEW", "mode": "greedy", "scene": 1800000, "seed": 11}
    _completed_claim(root, cfg, logical, cell, ids=["t1"])
    claim_path = root / "launch_claims" / r40_common._slug(logical) / "attempt_0001.json"
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    row_path = claim["result_receipt"]["row_path"]
    row = base._read_json(row_path)
    row["on_time"] = 6
    Path(row_path).write_text(json.dumps(row), encoding="utf-8")
    claim["result_receipt"]["row_sha256"] = base._sha256_file(row_path)
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    try:
        v2._load_model_rows(root)
        raised = False
    except RuntimeError:
        raised = True
    check("e7_the_row_loader_fails_closed_on_an_unclosed_target_set", raised)


# ============================================== E8: receipt-owned unique row source ======
with scratch() as root:
    rows = root / "rows"
    rows.mkdir(parents=True, exist_ok=True)
    cfg = _rows_cfg(root)
    logical = "r66eval2_model_new_s11_greedy_scene1800000"
    cell = {"arm": "NEW", "mode": "greedy", "scene": 1800000, "seed": 11}

    # attempt 1 failed AFTER writing a partial row; the orphan is named so that a naive
    # glob-order-last-wins loader would pick it over the real attempt-2 row
    orphan, orphan_art = _make_row(root, "orphan", arm="NEW", mode="greedy", scene=1800000, seed=11)
    orphan["on_time"] = 1
    orphan["artefacts"] = dict(orphan_art,
                               row=str(rows / f"v2_{v2.ROW_TAG}_model_new_s11_greedy_scene1800000_a9.json"))
    base._write_row(cfg, orphan)
    _claim(root, logical, 1, "failed_technical")
    _completed_claim(root, cfg, logical, cell, attempt=2, category="reserve")

    loaded = v2._load_model_rows(root)
    check("e8_an_orphan_row_from_a_failed_attempt_never_shadows_the_receipted_row",
          loaded[("NEW", 11, "greedy", 1800000)]["on_time"] == 1
          and loaded[("NEW", 11, "greedy", 1800000)]["logical_job"] == logical
          and loaded[("NEW", 11, "greedy", 1800000)]["row_path"].endswith("cellrow_a2_row.json"),
          got=loaded[("NEW", 11, "greedy", 1800000)]["row_path"])

    _rows2, audit = v2._load_qualified_rows(root, "model_eval", "model")
    check("e8_the_excluded_orphan_is_reported_rather_than_silently_dropped",
          audit["qualified_logical_jobs"] == 1
          and any("_a9.json" in q for q in audit["orphan_row_files_excluded_from_statistics"]),
          audit={k: audit[k] for k in ("qualified_logical_jobs",
                                       "orphan_row_files_excluded_from_statistics")})

    # a row on disk that NO complete claim names cannot enter the statistics at all
    stray, _ = _make_row(root, "stray_no_claim", arm="NEW", mode="greedy", scene=1800001, seed=11)
    base._write_row(cfg, stray)
    loaded = v2._load_model_rows(root)
    check("e8_an_unclaimed_row_file_is_not_admitted", ("NEW", 11, "greedy", 1800001) not in loaded)

    # two sound complete receipts naming different rows for one job is an ambiguity, not a tie
    ambiguous = "r66eval2_model_new_s11_greedy_scene1800002"
    _completed_claim(root, cfg, ambiguous, {"arm": "NEW", "mode": "greedy", "scene": 1800002,
                                            "seed": 11})
    other, _ = _make_row(root, "second_receipt", arm="NEW", mode="greedy", scene=1800002, seed=11)
    other_receipt = v2.persist_result(cfg, other, "second_receipt_a2")
    _write_claim(root, ambiguous, "complete", other_receipt, attempt=2, category="reserve")
    try:
        v2._load_model_rows(root)
        raised = False
    except RuntimeError:
        raised = True
    check("e8_two_sound_receipts_for_one_cell_are_refused_not_ordered", raised)

    # a receipted row whose own identity contradicts its claim name is refused
    liar = "r66eval2_model_new_s11_greedy_scene1800003"
    liar_row, _ = _make_row(root, "liar", arm="OLD", mode="greedy", scene=1800003, seed=22)
    liar_receipt = v2.persist_result(cfg, liar_row, "liar_a1")
    _write_claim(root, liar, "complete", liar_receipt)
    try:
        v2._load_model_rows(root)
        raised = False
    except RuntimeError:
        raised = True
    check("e8_a_receipted_row_whose_identity_contradicts_its_claim_is_refused", raised)

# ============================================== E7b: legitimate zero-task rows ==========
with scratch() as root:
    check("e7b_a_zero_task_row_with_an_empty_event_list_closes_to_the_empty_set",
          v2.on_time_target_ids({"task_reward_events": [], "on_time": 0}, 1800000) == [])
    check("e7b_a_zero_task_row_with_absent_events_also_closes_to_the_empty_set",
          v2.on_time_target_ids({"on_time": 0}, 1800000) == [])
    check("e7b_a_nonzero_claim_with_an_empty_event_list_is_still_a_data_error",
          _raises(lambda: v2.on_time_target_ids({"task_reward_events": [], "on_time": 3}, 1800000)))
    check("e7b_a_zero_task_row_with_an_off_time_event_closes_to_the_empty_set",
          v2.on_time_target_ids({"task_reward_events": [{"task_id": "t1", "on_time": False}],
                                 "on_time": 0}, 1800000) == [])

    cfg = _rows_cfg(root)
    for scene in (1800000, 1800001):
        _completed_claim(root, cfg, f"r66eval2_model_new_s11_greedy_scene{scene}",
                         {"arm": "NEW", "mode": "greedy", "scene": scene, "seed": 11},
                         ids=[], tag=f"zero_{scene}")
    loaded = v2._load_model_rows(root)
    check("e7b_a_legitimate_zero_row_loads_with_an_empty_target_set",
          loaded[("NEW", 11, "greedy", 1800000)]["on_time"] == 0
          and loaded[("NEW", 11, "greedy", 1800000)]["on_time_ids"] == [])

failed = [n for n, ok, _ in RESULTS if not ok]
print(json.dumps({"kind": "round66-eval-adapter2-zero-env/1",
                  "status": "passed" if not failed else "failed",
                  "checks": len(RESULTS), "failed": failed,
                  "synthetic_optimizer_updates": SYNTHETIC_OPTIMIZER_UPDATES,
                  "environment_constructions": 0, "real_claims_opened": 0, "real_slots_consumed": 0,
                  "bootstrap_index_sha256": sha, "eval_version": v2.EVAL_VERSION,
                  "seams": {"E1_reserve_cap_39_and_d1_extra_in_the_hard_cap": True,
                            "E2_no_reuse_of_incomplete_attempts_reserve_retry_isolated_artifact": True,
                            "E3_plan_order_equals_request_order_slots_declared_pool_allocated": True,
                            "E4_complete_claim_requires_a_durable_re_provable_row": True,
                            "E5_row_and_receipt_persisted_before_the_success_commit": True,
                            "E6_target_set_diffs_paired_per_scene": True,
                            "E7_on_time_target_sets_evidenced_and_closed_fail_closed": True},
                  "matrix": {"rule_eval": 96, "model_eval": 576, "focal": 6, "total": 678,
                             "planned_normal_slot_range": [433, 1110]}}, indent=1))
raise SystemExit(0 if not failed else 1)
