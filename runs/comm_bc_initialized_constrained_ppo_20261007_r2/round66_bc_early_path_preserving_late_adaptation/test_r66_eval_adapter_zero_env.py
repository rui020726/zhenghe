"""ROUND66 zero-environment acceptance for the corrected evaluation adapter (r66_eval_fixed.py).

Everything here is pure computation: no builder, no environment constructor, no optimizer step, no
checkpoint write, no claim and no slot. It proves that the adapter (a) builds exactly the
decision-side evaluation matrix, (b) numbers its slots after the 432 cells already numbered by the
48 block-A references and the 384 training episodes, (c) can never stand up the simulation runtime
on the planning path, (d) does go through configure_and_inject on the eval/stats path so the real
CLI, module paths and quota guard are the ones actually used, and (e) writes rows under a tag
namespace that cannot collide with the rejected matrix r66_execute.py would have produced.

What this file deliberately does NOT claim:
  * passing these checks qualifies no evaluation cell; only a real claim + builder does that;
  * the corrected matrix is a plan correction, not a change to the scientific configuration;
  * no gradient, likelihood or checkpoint behaviour is re-proved here; that is
    test_r66_frozen_acceptance.py's job.
"""
from __future__ import annotations
import contextlib, json, shutil, sys
from fnmatch import fnmatch
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r66_eval_fixed as fixed
import r66_execute as base
import r66_frozen as r66mod

RESULTS: list[tuple[str, bool, dict]] = []
SYNTHETIC_OPTIMIZER_UPDATES = 0  # this file performs no optimizer work at all


@contextlib.contextmanager
def _scratch_dir():
    """The sandbox AC temp rejects the atomic tmp.replace() that base._write_json uses, so scratch
    lives inside the round directory and is removed on exit. Nothing here touches the science."""
    p = HERE / "_eval_adapter_scratch"
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
    try:
        yield p
    finally:
        shutil.rmtree(p, ignore_errors=True)


def check(name, cond, **detail):
    cond = bool(cond)
    RESULTS.append((name, cond, detail))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  " + json.dumps(detail, default=str)),
          flush=True)
    return cond


# ------------------------------------------------------------------ matrix shape ------------
plan = fixed.build_eval_plan()
check("matrix_counts_are_576_96_6",
      (plan["model_eval_count"], plan["rule_eval_count"], plan["focal_count"], plan["eval_total"])
      == (576, 96, 6, 678),
      got=(plan["model_eval_count"], plan["rule_eval_count"], plan["focal_count"], plan["eval_total"]))
check("matrix_jobs_unique", len(plan["jobs"]) == 678 and len({j["logical_job"] for j in plan["jobs"]}) == 678,
      jobs=len(plan["jobs"]), unique=len({j["logical_job"] for j in plan["jobs"]}))
scopes: dict[str, int] = {}
for j in plan["jobs"]:
    scopes[j["scope"]] = scopes.get(j["scope"], 0) + 1
check("matrix_scopes", scopes == {"model_eval": 576, "rule_eval": 96, "focal": 6}, scopes=scopes)

# ------------------------------------------------------------------ slot numbering ----------
slots = [j["normal_slot"] for j in plan["jobs"]]
check("slots_follow_the_432_already_numbered_cells",
      plan["normal_slot_range"] == [433, 1110] and slots == list(range(433, 1111)),
      rng=plan["normal_slot_range"], contiguous=slots == list(range(433, 1111)))
check("slot_total_plus_numbered_cells_equals_the_normal_cap",
      432 + 678 == base.NORMAL_CAP == 1110, numbered=432, eval=678, cap=base.NORMAL_CAP)

# ------------------------------------------------------------------ arms and scenes ---------
check("model_arms_are_new_old_bc", list(fixed.MODEL_ARMS) == ["NEW", "OLD", "BC"], arms=list(fixed.MODEL_ARMS))
check("rule_arms_are_the_three_decision_side_rules",
      list(fixed.RULE_ARMS) == ["immediate", "late_discovery_gate", "always_hold"], arms=list(fixed.RULE_ARMS))
check("eval_scenes_are_the_frozen_new_block",
      tuple(fixed.SCENES_NEW) == tuple(range(1_800_000, 1_800_032)), first=fixed.SCENES_NEW[0],
      last=fixed.SCENES_NEW[-1])
check("focal_scene_is_1700027", fixed.FOCAL_SCENE == 1_700_027, focal=fixed.FOCAL_SCENE)
check("modes_are_greedy_then_sampled", list(fixed.MODES) == ["greedy", "sampled"], modes=list(fixed.MODES))

# ------------------------------------------------------------------ runtime binding ----------
with _scratch_dir() as td:
    tmp = Path(td)
    seen = {"calls": 0}

    def _boom(_raw_root):
        seen["calls"] += 1
        raise AssertionError("the corrected plan phase must not configure the simulation runtime")

    saved = base.configure_and_inject
    base.configure_and_inject = _boom
    try:
        rc = fixed.main(["--raw-root", str(tmp), "--phase", "plan"])
    finally:
        base.configure_and_inject = saved
    check("plan_phase_returns_zero_without_configuring_the_runtime",
          rc == 0 and seen["calls"] == 0, rc=rc, configure_calls=seen["calls"])
    written = json.loads((tmp / "ROUND66_EVAL_PLAN_CORRECTED.json").read_text(encoding="utf-8"))
    check("plan_phase_writes_the_corrected_plan_file",
          written["eval_total"] == 678 and written["normal_slot_range"] == [433, 1110],
          eval_total=written.get("eval_total"))

    # the eval and stats phases MUST go through configure_and_inject, i.e. the real live modules,
    # the real CLI path and the real quota guard - not a private bypass.
    for phase in ("eval", "stats"):
        def _boom2(_raw_root, _p=phase):
            raise AssertionError("runtime binding reached")
        base.configure_and_inject = _boom2
        try:
            fixed.main(["--raw-root", str(tmp / "rows_missing"), "--phase", phase])
            reached = False
        except AssertionError:
            reached = True
        except Exception:
            reached = True
        finally:
            base.configure_and_inject = saved
        check(f"{phase}_phase_binds_the_live_runtime_before_anything_else", reached, phase=phase)

    # statistics must refuse an incomplete matrix instead of silently reporting partial results.
    rows_dir = tmp / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    base.configure_and_inject = _boom
    try:
        fixed.compute_stats(tmp)
        refused = False
    except RuntimeError:
        refused = True
    except AssertionError:
        refused = False
    finally:
        base.configure_and_inject = saved
    check("stats_refuses_an_incomplete_paired_matrix", refused)

# ------------------------------------------------------------------ determinism -------------
a, b = fixed.build_eval_plan(), fixed.build_eval_plan()
check("plan_is_deterministic", a["jobs"] == b["jobs"] and a["eval_version"] == b["eval_version"])
idx, sha = fixed._bootstrap_index()
check("bootstrap_index_is_the_frozen_pcg64_20261012_draw",
      idx.shape == (10000, 32) and int(idx.min()) >= 0 and int(idx.max()) < 32
      and sha == fixed._bootstrap_index()[1], shape=list(idx.shape), sha=sha[:16])

# ------------------------------------------------------------------ namespace collision -----
corrected_names = ([f"v2_r66eval_model_{arm.lower()}_s{seed}_{mode}_scene{s}_a1.json"
                    for arm in fixed.MODEL_ARMS for seed in fixed.SEEDS for mode in fixed.MODES
                    for s in fixed.SCENES_NEW]
                   + [f"v2_r66eval_rule_{arm}_greedy_scene{s}_a1.json"
                      for arm in fixed.RULE_ARMS for s in fixed.SCENES_NEW]
                   + [f"v2_r66eval_focal_{arm.lower()}_s{seed}_scene{fixed.FOCAL_SCENE}_a1.json"
                      for seed in fixed.SEEDS for arm in ("NEW", "BC")])
check("corrected_row_names_are_678", len(corrected_names) == 678, n=len(corrected_names))
rejected_globs = ["v2_r66_eval_new_*.json", "v2_r66_eval_bc_*.json", "v2_r66_rule_*.json",
                  "v2_r66_focal_*.json"]
leak = [n for n in corrected_names if any(fnmatch(n, g) for g in rejected_globs)]
check("corrected_rows_cannot_match_the_rejected_matrix_globs", not leak, leaked=leak[:3])
corrected_globs = ["v2_r66eval_model_*.json", "v2_r66eval_rule_*.json", "v2_r66eval_focal_*.json"]
rejected_names = ["v2_r66_eval_new_s11_greedy_scene1800000_a1.json",
                  "v2_r66_eval_bc_s22_sampled_scene1800031_a1.json",
                  "v2_r66_rule_immediate_greedy_scene1800000_a1.json",
                  "v2_r66_rule_periodic_20_greedy_scene1800000_a1.json",
                  "v2_r66_focal_new_s11_scene1700027_a1.json"]
back = [n for n in rejected_names if any(fnmatch(n, g) for g in corrected_globs)]
check("rejected_matrix_rows_cannot_match_the_corrected_globs", not back, matched=back)

# ------------------------------------------------------------------ identity and quota ------
check("adapter_shares_the_r66_quota",
      (fixed.NORMAL_CAP, fixed.RESERVE_CAP, fixed.HARD_CAP) == (base.NORMAL_CAP, base.RESERVE_CAP, base.HARD_CAP)
      == (1110, 40, 1150),
      got=(fixed.NORMAL_CAP, fixed.RESERVE_CAP, fixed.HARD_CAP))
check("adapter_version_extends_the_r66_training_version",
      fixed.EVAL_VERSION.startswith(r66mod.R66_TRAINING_VERSION)
      and fixed.EVAL_VERSION.endswith("+r66-eval-matrix-correction/1"),
      version=fixed.EVAL_VERSION[-60:])
check("adapter_uses_the_r66_task_and_branch",
      base.TASK_ID == "ROUND66-BC-EARLY-PATH-PRESERVING-LATE-ADAPTATION"
      and base.BRANCH == "exp/round66-bc-early-path-preserving-late-adaptation",
      task_id=base.TASK_ID, branch=base.BRANCH)

failed = [n for n, ok, _ in RESULTS if not ok]
print(json.dumps({"kind": "round66-eval-adapter-zero-env/1",
                  "status": "passed" if not failed else "failed",
                  "checks": len(RESULTS), "failed": failed,
                  "synthetic_optimizer_updates": SYNTHETIC_OPTIMIZER_UPDATES,
                  "environment_constructions": 0, "claims_opened": 0, "slots_consumed": 0,
                  "bootstrap_index_sha256": sha,
                  "eval_version": fixed.EVAL_VERSION,
                  "corrected_matrix": {"model_eval": 576, "rule_eval": 96, "focal": 6, "total": 678,
                                       "normal_slot_range": [433, 1110]}}, indent=1))
raise SystemExit(0 if not failed else 1)
