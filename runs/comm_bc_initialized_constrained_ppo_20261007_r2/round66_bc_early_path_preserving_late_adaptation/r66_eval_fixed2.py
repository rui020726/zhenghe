"""ROUND66 corrected evaluation adapter, private version 2 (retry / cap / slot seams).

WHY A SECOND FILE EXISTS
------------------------
r66_eval_fixed.py (committed at 7e8014f) fixed the evaluation MATRIX but still inherited three real
seams from r66_execute.py that the decision side required to be landed before the first evaluation
builder. Originals are never edited, so the corrections live here under a new evaluation identity:

  E1  reserve cap and the hard-cap judgement
      r66_eval_fixed.py took RESERVE_CAP = 40 straight from base, and base.configure_and_inject
      rebinds that 40 onto the LIVE r40_common / common modules. Deviation D1 adds ONE environment
      construction that no slot records, so booked normal + booked reserve + that extra must stay
      <= 1150. This module books reserve at 39 and enforces the arithmetic on the live modules, in
      source, before any builder runs.

  E2  the open-claim rerun defect
      base._atomic_run falls through when a claim exists with status "claimed_before_environment",
      i.e. it silently re-runs a cell whose environment may already have been constructed, under the
      same claim and the same slot. That is exactly deviation D1. This module installs
      _prepare_claim / _safe_atomic_run: an incomplete, already-constructed, already-run or
      unreadable attempt is NEVER reused. A retry takes a NEW attempt number in the RESERVE
      category, the original claim / row / log is preserved untouched, and the retry writes to an
      attempt-isolated tag so it cannot overwrite the earlier artifact. Only a claim that is
      verifiably "complete" is accepted read-only, without a rerun.

  E3  planned slot numbers were not the real allocation
      r40_common._reserve_slot hands out the lowest free number in launch_slots/<category>, so the
      per-job "normal_slot" in the v1 plan was a plan, not a binding - and it was generated in
      model-then-rule order while the executor runs rule-then-model with a different inner loop
      order. Here the single generator eval_job_order() drives BOTH the plan and the executor loops,
      so plan order and request order are identical by construction, the slot mechanism is stated
      explicitly, and the executor asserts the visited order equals the plan order at the end.

Arms:
    NEW = R66 ep128 (frozen early path intact, read-only load)
    OLD = compliant R60 ep128, read-only, NOT frozen (runs exactly as R60 had it)
    BC  = compliant R60 ep0 initial value, read-only, NOT frozen

  E4  a "complete" claim is not by itself an admissible result
      _cell_complete used to accept any claim whose status was "complete". A complete claim whose
      row is missing, truncated, unreadable, hash-mismatched, identity-wrong or whose companions
      (lifecycle / transcript / tape) are absent would be skipped without a rerun, i.e. a cell could
      silently drop out of the statistics. Now a complete claim only qualifies when the durable
      receipt proves the row exists, reads back, hashes to the recorded value, carries companions,
      and matches the cell identity. Every parse or verification failure fails closed: the cell is
      NOT skipped and a reserve retry is opened instead. The same soundness test gates
      _prepare_claim, so an unsound complete claim can never deadlock a cell: it is preserved
      untouched and an independent reserve attempt is opened beside it.

  E5  the result row must be durable before the claim is committed
      base._atomic_run calls finish_launch(ok=True) and the caller writes the row afterwards, so a
      failing row write would leave a complete claim with no result. Here the row and its receipt
      are persisted and re-read/re-hashed INSIDE the success path, before finish_launch; a write or
      verification failure marks the attempt failed_technical and leaves the earlier artifacts alone.

  E6  target-ID set differences must be paired per scene
      Merging target IDs across the 32 scenes lets the same name (target-5) in two scenes cancel
      out, hiding a real loss. lost/gained are now reported as (scene, target_id) pairs.

  E7  on-time target sets must be evidenced, not defaulted to empty
      A real _run_episode row carries no on_time_target_ids / on_time_ids field. Reading that absent
      field with an "or []" fallback would fabricate an empty target set and turn every target
      difference - including the focal scene 1700027 / target-5 loss - into a silent zero. The sets
      are rebuilt from the row's own task_reward_events and closed against row["on_time"] per cell;
      anything unreadable or inconsistent raises instead of writing an empty set. Rule arms are
      greedy-only, so NEW-vs-rule comparisons appear in the greedy entry and are explicitly marked
      not-applicable in the sampled entry - a rule greedy number is never reported as a sampled
      same-mode result.

No training, no optimizer step, no checkpoint write, no RNG restore happens here.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Any, Iterator
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r66_frozen as r66
import r66_execute as base

SEEDS = (11, 22, 33)
MODES = ("greedy", "sampled")
SCENES_NEW = tuple(range(1_800_000, 1_800_032))
FOCAL_SCENE = 1_700_027
MODEL_ARMS = ("NEW", "OLD", "BC")
RULE_ARMS = ("immediate", "late_discovery_gate", "always_hold")
ROW_TAG = "r66eval2"
EVAL_VERSION = r66.R66_TRAINING_VERSION + "+r66-eval-matrix-correction/2"
NORMAL_CAP, RESERVE_CAP_INHERITED, HARD_CAP = base.NORMAL_CAP, base.RESERVE_CAP, base.HARD_CAP

# E1: the booked reserve ceiling for ROUND66. 1110 normal slots + 39 booked reserve + the ONE
# unregistered extra construction from deviation D1 = 1150 = the hard cap.
EVAL_RESERVE_CAP = 39
UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS = 1
D1_REFERENCE = "ROUND66_DEVIATION_REGISTER.json#D1-reaped-reference-attempt-claim-reuse"

# E3: the executor and the plan iterate this one generator, so their orders cannot drift.
def eval_job_order() -> Iterator[dict[str, Any]]:
    """Every evaluation cell in the exact order the executor will request it."""
    for arm in RULE_ARMS:
        for scene in SCENES_NEW:
            yield {"scope": "rule_eval", "arm": arm.upper(), "seed": None, "mode": "greedy",
                   "scene": int(scene)}
    for seed in SEEDS:
        for arm in MODEL_ARMS:
            for mode in MODES:
                for scene in SCENES_NEW:
                    yield {"scope": "model_eval", "arm": arm, "seed": int(seed), "mode": mode,
                           "scene": int(scene)}
    for seed in SEEDS:
        for arm in ("NEW", "BC"):
            yield {"scope": "focal", "arm": arm, "seed": int(seed), "mode": "greedy",
                   "scene": FOCAL_SCENE}


def logical_job(cell: dict[str, Any]) -> str:
    if cell["scope"] == "rule_eval":
        return f"{ROW_TAG}_rule_{cell['arm'].lower()}_greedy_scene{cell['scene']}"
    if cell["scope"] == "model_eval":
        return f"{ROW_TAG}_model_{cell['arm'].lower()}_s{cell['seed']}_{cell['mode']}_scene{cell['scene']}"
    return f"{ROW_TAG}_focal_{cell['arm'].lower()}_s{cell['seed']}_scene{cell['scene']}"


def build_eval_plan() -> dict[str, Any]:
    cells = list(eval_job_order())
    jobs = []
    for index, cell in enumerate(cells):
        jobs.append({"scope": cell["scope"], "arm": cell["arm"], "seed": cell["seed"],
                     "mode": cell["mode"], "scene": cell["scene"],
                     "logical_job": logical_job(cell),
                     # E3: a planned enumeration only. The real number comes from
                     # r40_common._reserve_slot, which takes the lowest free slot in the pool.
                     "planned_normal_slot": 433 + index})
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j["scope"]] = counts.get(j["scope"], 0) + 1
    if counts != {"rule_eval": 96, "model_eval": 576, "focal": 6} or len(jobs) != 678:
        raise AssertionError(f"corrected v2 eval plan is not 96+576+6=678 unique jobs: {counts}")
    if len({j["logical_job"] for j in jobs}) != 678:
        raise AssertionError("corrected v2 eval plan contains duplicate logical jobs")
    planned = [j["planned_normal_slot"] for j in jobs]
    if planned != list(range(433, 1111)):
        raise AssertionError("planned normal slots are not the contiguous range 433..1110")
    if 432 + 678 != NORMAL_CAP:
        raise AssertionError("432 already-numbered cells + 678 evaluation cells != the normal cap")
    return {"kind": "round66-corrected-eval-plan/2", "task_id": base.TASK_ID,
            "branch": base.BRANCH, "eval_version": EVAL_VERSION,
            "supersedes": ("raw/ROUND66_EVAL_PLAN_CORRECTED.json written by r66_eval_fixed.py at "
                           "7e8014f, which is preserved unchanged; and the model_eval/rule_eval "
                           "scopes of raw/ROUND66_FIXED_PLAN.json written from the erroneous "
                           "freeze at 7267fc7, also preserved unchanged"),
            "correction": "r66-eval-matrix-correction/2: retry discipline + reserve cap 39 + slot semantics",
            "matrix": {"rule_eval": 96, "model_eval": 576, "focal": 6, "total": 678},
            "already_numbered_cells": 432,
            "already_numbered_note": "48 block-A immediate references + 384 training episodes",
            "model_arms": list(MODEL_ARMS), "rule_arms": list(RULE_ARMS),
            "eval_scenes": list(SCENES_NEW), "focal_scene": FOCAL_SCENE,
            "execution_order": ["rule_eval", "model_eval", "focal"],
            "slot_allocation": {
                "mechanism": "r40_common._reserve_slot takes the lowest free number in "
                             "raw/launch_slots/<category>",
                "planned_field": "planned_normal_slot is a plan enumeration, NOT a binding",
                "planned_range": [433, 1110],
                "plan_order_equals_request_order": True,
                "enforced_by": "r66_eval_fixed2._assert_visited_order at the end of the eval phase"},
            "quota": {"normal_cap": NORMAL_CAP,
                      "reserve_cap_inherited_from_base": RESERVE_CAP_INHERITED,
                      "reserve_cap_booked_here": EVAL_RESERVE_CAP,
                      "hard_cap": HARD_CAP,
                      "unregistered_extra_environment_constructions":
                          UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS,
                      "arithmetic": ("booked_normal + booked_reserve + "
                                     f"{UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS} (D1) <= "
                                     f"{HARD_CAP}; with {NORMAL_CAP} normal slots the booked "
                                     f"reserve ceiling is {HARD_CAP - NORMAL_CAP - UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS}"),
                      "d1_reference": D1_REFERENCE},
            "row_namespace": {"tag_prefix": ROW_TAG,
                              "row_globs": [f"v2_{ROW_TAG}_model_*.json", f"v2_{ROW_TAG}_rule_*.json",
                                            f"v2_{ROW_TAG}_focal_*.json"],
                              "disjoint_from": ["v2_r66_eval_*.json", "v2_r66_rule_*.json",
                                                "v2_r66_focal_*.json", "v2_r66eval_*.json"]},
            "jobs": jobs}


# ------------------------------------------------------------------ E1: quota guard ---------
def configure_r66_eval(raw_root: Path) -> dict[str, Any]:
    """Bind the live runtime, then apply the R66-EVAL reserve cap and prove the hard-cap arithmetic
    on the LIVE modules before any builder can run."""
    cfg = base.configure_and_inject(raw_root)
    r40 = cfg["runtime"]["r40_common"]
    common = cfg["common"]
    # Rebind the booked reserve ceiling onto the modules the quota guard actually reads.
    r40.RESERVE_CAP = EVAL_RESERVE_CAP
    common.RESERVE_LAUNCHES = EVAL_RESERVE_CAP
    r40.plan = lambda: {"normal_total": NORMAL_CAP, "technical_reserve_max": EVAL_RESERVE_CAP,
                        "hard_cap": HARD_CAP, "task_id": base.TASK_ID,
                        "unregistered_extra_environment_constructions":
                            UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS}
    if int(r40.RESERVE_CAP) != EVAL_RESERVE_CAP or int(common.RESERVE_LAUNCHES) != EVAL_RESERVE_CAP:
        raise RuntimeError("R66-EVAL reserve cap did not bind on the live modules")
    if int(r40.NORMAL_CAP) != NORMAL_CAP or int(r40.HARD_CAP) != HARD_CAP:
        raise RuntimeError("R66-EVAL normal/hard cap drifted from the frozen R66 quota")
    guard = quota_guard(cfg)
    cfg["r66_eval_quota_guard"] = guard
    return cfg


def quota_guard(cfg: dict[str, Any]) -> dict[str, Any]:
    """The real hard-cap judgement: booked slots + the D1 unregistered extra construction."""
    r40 = cfg["runtime"]["r40_common"]
    common = cfg["common"]
    snapshot = r40.ledger_snapshot()  # itself raises if a live cap is already exceeded
    normal_used = int(snapshot["normal_used"])
    reserve_used = int(snapshot["reserve_used"])
    booked = normal_used + reserve_used
    actual = booked + UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS
    hard = int(r40.HARD_CAP)
    if normal_used > NORMAL_CAP:
        raise RuntimeError(f"R66-EVAL: booked normal {normal_used} exceeds the normal cap {NORMAL_CAP}")
    if reserve_used > EVAL_RESERVE_CAP:
        raise RuntimeError(f"R66-EVAL: booked reserve {reserve_used} exceeds the booked ceiling "
                           f"{EVAL_RESERVE_CAP}")
    if actual > hard:
        raise RuntimeError(f"R66-EVAL: actual environment constructions {actual} "
                           f"(booked {booked} + {UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS} "
                           f"unregistered D1 extra) exceed the hard cap {hard}")
    return {"kind": "round66-eval-quota-guard/2", "eval_version": EVAL_VERSION,
            "booked_normal": normal_used, "booked_reserve": reserve_used, "booked_total": booked,
            "unregistered_extra_environment_constructions":
                UNREGISTERED_EXTRA_ENVIRONMENT_CONSTRUCTIONS,
            "actual_environment_constructions": actual,
            "normal_cap": NORMAL_CAP, "reserve_cap_booked": EVAL_RESERVE_CAP,
            "reserve_cap_inherited_from_base": RESERVE_CAP_INHERITED,
            "hard_cap": hard, "hard_headroom": hard - actual,
            "booked_reserve_remaining": EVAL_RESERVE_CAP - reserve_used,
            "historical_non_plan_probe": int(base.HISTORICAL_PROBE),
            "note": ("R58/R60 historical constructions are NOT ROUND66 consumption; R66's own D1 "
                     "extra construction IS counted and is the reason the booked reserve ceiling "
                     "is 39 rather than the inherited 40."),
            "d1_reference": D1_REFERENCE,
            "live_bindings": {"r40_RESERVE_CAP": int(r40.RESERVE_CAP),
                              "common_RESERVE_LAUNCHES": int(common.RESERVE_LAUNCHES),
                              "r40_NORMAL_CAP": int(r40.NORMAL_CAP),
                              "r40_HARD_CAP": int(r40.HARD_CAP)}}


# ------------------------------------------------------------------ E2: safe claim seam -----
class AlreadyComplete(RuntimeError):
    pass


def _receipt_is_sound(claim: dict, cell: dict | None) -> bool:
    """E4: a complete claim only counts when its durable result can be re-proved from disk."""
    receipt = claim.get("result_receipt")
    if not isinstance(receipt, dict):
        return False
    row_path = receipt.get("row_path")
    if not row_path:
        return False
    try:
        path = Path(row_path)
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if base._sha256_file(path) != receipt.get("row_sha256"):
            return False
        row = base._read_json(path)
        if base._sha256_file(path) != receipt.get("row_sha256"):
            return False
    except Exception:
        return False
    for key, value in (receipt.get("companions") or {}).items():
        try:
            if not isinstance(value, dict) or not value.get("path"):
                return False
            companion = Path(value["path"])
            if not companion.is_file() or companion.stat().st_size == 0:
                return False
            if int(companion.stat().st_size) != int(value.get("bytes", -1)):
                return False
            if base._sha256_file(companion) != value.get("sha256"):
                return False
        except Exception:
            return False
    if cell is None:
        return True
    try:
        if str(row.get("model_arm") or "").upper() != str(cell["arm"]).upper():
            return False
        if str(row.get("eval_mode")) != str(cell["mode"]):
            return False
        if int(row.get("scenario_seed", -1)) != int(cell["scene"]):
            return False
        if cell.get("seed") is not None and int(row.get("training_seed", -1)) != int(cell["seed"]):
            return False
        if int(row.get("steps", -1)) <= 0:
            return False
    except Exception:
        return False
    return True


def _cell_complete(cfg: dict[str, Any], logical: str, cell: dict | None = None) -> bool:
    """True only when SOME attempt is complete AND its durable result re-proves from disk (E4).

    Any unreadable claim, missing receipt, missing/empty row or companion, hash mismatch or identity
    mismatch fails closed: the cell is treated as NOT complete so a reserve retry is opened rather
    than silently dropping the cell out of the statistics.
    """
    r40 = cfg["runtime"]["r40_common"]
    directory = r40.ROUND_DIR / "launch_claims" / r40._slug(logical)
    if not directory.is_dir():
        return False
    for path in sorted(directory.glob("attempt_*.json")):
        try:
            claim = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if claim.get("status") != "complete":
            continue
        if _receipt_is_sound(claim, cell):
            return True
    return False


def _prepare_claim(cfg: dict[str, Any], logical: str, metadata: dict, tag: str,
                    cell: dict | None = None):
    """Claim a cell without ever reusing an incomplete attempt.

    Returns (claim_path, attempt, run_tag). A retry takes the next attempt number in the RESERVE
    category and an attempt-isolated tag, so the original claim, row and log survive untouched.
    """
    r40 = cfg["runtime"]["r40_common"]
    directory = r40.ROUND_DIR / "launch_claims" / r40._slug(logical)
    prior = []
    if directory.is_dir():
        for path in sorted(directory.glob("attempt_*.json")):
            try:
                prior.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                prior.append({"attempt": None, "status": "unreadable"})
    if any(c.get("status") == "complete" and _receipt_is_sound(c, cell) for c in prior):
        raise AlreadyComplete(f"{logical}: a complete attempt with a sound receipt already exists; "
                              "caller must skip")
    if prior:
        attempt = r40.next_attempt(logical)
        category = "reserve"
        retry = {"retry_of_attempt": max(int(c.get("attempt") or 0) for c in prior),
                 "retry_reason": "prior attempt is not verifiably complete and is never reused",
                 "prior_statuses": [c.get("status") for c in prior],
                 "retry_artifact_isolated": True}
    else:
        attempt, category, retry = 1, "normal", {}
    claim_path = r40.claim_launch(logical, category=category, attempt=attempt,
                                  metadata={**metadata, **retry, "eval_version": EVAL_VERSION,
                                            "row_tag": f"{tag}_a{attempt}"})
    return claim_path, attempt, f"{tag}_a{attempt}"


def persist_result(cfg, row, run_tag: str) -> dict:
    """E5: write the result row and prove the durable artifacts exist, before anything is committed.

    Raises on any problem so the caller marks the attempt failed_technical instead of committing a
    complete claim with no result behind it.
    """
    base._write_row(cfg, row)
    artefacts = row.get("artefacts") or {}
    missing = [k for k in ("row", "lifecycle", "transcript", "tape") if not artefacts.get(k)]
    if missing:
        raise RuntimeError(f"{run_tag}: result row is missing artefact keys {missing}")
    for key in ("row", "lifecycle", "transcript", "tape"):
        path = Path(artefacts[key])
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{run_tag}: artefact {key} is missing or empty at {path}")
    row_path = Path(artefacts["row"])
    first = base._sha256_file(row_path)
    reread = base._read_json(row_path)
    if int(reread.get("steps", -1)) <= 0:
        raise RuntimeError(f"{run_tag}: persisted row records no steps")
    if int(reread.get("on_time", -1)) != int(row.get("on_time", -2)):
        raise RuntimeError(f"{run_tag}: persisted row does not read back consistently")
    if base._sha256_file(row_path) != first:
        raise RuntimeError(f"{run_tag}: persisted row hash is not stable across re-reads")
    companions = {}
    for key in ("lifecycle", "transcript", "tape"):
        companion = Path(artefacts[key])
        companions[key] = {"path": str(companion),
                           "sha256": base._sha256_file(companion),
                           "bytes": int(companion.stat().st_size)}
    return {"row_path": str(row_path), "row_sha256": first,
            "row_bytes": int(row_path.stat().st_size),
            "companions": companions,
            "run_tag": run_tag, "persisted_at_unix": time.time()}


def _safe_atomic_run(cfg, logical, metadata, run_fn, *, detail, tag, persist=None, cell=None):
    """base._atomic_run with the open-claim rerun defect removed. run_fn receives the run tag.

    When persist is given it runs INSIDE the success path, before finish_launch, so a complete claim
    always has a durable, re-provable result behind it (E5).
    """
    r40 = cfg["runtime"]["r40_common"]
    r42_run = cfg["r42_run"]
    if r42_run._ACTIVE_CLAIM is not None:
        raise RuntimeError("nested R66 evaluation claims are not allowed")
    claim_path, attempt, run_tag = _prepare_claim(cfg, logical, metadata, tag, cell)
    r42_run._ACTIVE_CLAIM = claim_path
    r42_run._ACTIVE_PROGRESS = r42_run._new_run_progress()
    try:
        row = run_fn(run_tag)
        receipt = persist(row, run_tag) if persist is not None else None
        if receipt is not None:
            r40._update_claim(claim_path, result_receipt=receipt)
        r40.finish_launch(claim_path, ok=True, detail=detail(row))
        if isinstance(row, dict):
            row["claim_attempt"] = int(attempt)
            row["claim_path"] = str(claim_path)
        return row
    except BaseException as exc:
        try:
            r40.finish_launch(claim_path, ok=False, detail=f"{type(exc).__name__}: {exc}")
        except BaseException:
            pass
        raise
    finally:
        r42_run._ACTIVE_CLAIM = None
        r42_run._ACTIVE_PROGRESS = None


# ------------------------------------------------------------------ arms --------------------
def _load_arm(cfg, seed: int, arm: str):
    """Read-only arm controllers. NEW keeps its frozen mask; OLD and BC run exactly as R60 had them."""
    import r42_one_shot  # noqa: F401  (ensures the frozen collector module is importable)
    common = cfg["common"]
    if arm == "NEW":
        ep128 = Path(common.CHECKPOINTS) / f"v2_train_s{seed}_ep128.pt"
        if not ep128.is_file():
            raise FileNotFoundError(f"seed {seed}: R66 ep128 checkpoint missing")
        controller, state, identity = r66.load_checkpoint_read_only(ep128, freeze=base.R66_FREEZE)
        if int(state.get("episodes_done", -1)) != base.EPISODES_PER_SEED:
            raise RuntimeError(f"seed {seed}: R66 ep128 is not {base.EPISODES_PER_SEED} episodes")
        if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
            raise RuntimeError(f"seed {seed}: NEW read-only eval restored state")
        if any(p.requires_grad for p in r66.frozen_parameters(controller)):
            raise RuntimeError(f"seed {seed}: NEW ep128 lost its frozen mask")
        return controller, identity
    role = "ep128" if arm == "OLD" else "ep0"
    return r66.load_r60_checkpoint_read_only(seed, role)


# ------------------------------------------------------------------ executors ---------------
VISITED: list[str] = []


def eval_rules(cfg, plan) -> dict:
    import v2_rules
    runner = cfg["runner"]
    for cell in eval_job_order():
        if cell["scope"] != "rule_eval":
            continue
        logical = logical_job(cell)
        VISITED.append(logical)
        if _cell_complete(cfg, logical, cell):
            continue
        arm, scene = cell["arm"].lower(), int(cell["scene"])
        tag = f"{ROW_TAG}_rule_{arm}"
        metadata = {"arm": arm.upper(), "mode": "greedy", "scenario_seed": scene,
                    "eval_version": EVAL_VERSION}

        def _do(run_tag, arm=arm, scene=scene, tag=tag):
            if arm == "immediate":
                row = runner._run_episode(run_tag, "rule", scene, policy="immediate", tag=run_tag)
            else:
                policy = v2_rules.build_rule(arm, scene)
                row = runner._run_episode(run_tag, "rule", scene, policy="learned",
                                          attach=policy, tag=run_tag)
                row["rule_identity"] = v2_rules.rule_identity(arm)
            row["model_arm"] = arm.upper(); row["eval_mode"] = "greedy"; row["scenario_seed"] = scene
            return row

        row = _safe_atomic_run(cfg, logical, metadata, _do,
                               detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}",
                               tag=tag, persist=lambda r, t: persist_result(cfg, r, t), cell=cell)
    return {"status": "rules_eval_complete", "arms": list(RULE_ARMS), "cells": 96}


def eval_model_arms(cfg, plan) -> dict:
    common = cfg["common"]; runner = cfg["runner"]
    import r42_one_shot
    for cell in eval_job_order():
        if cell["scope"] != "model_eval":
            continue
        seed, arm, mode, scene = int(cell["seed"]), cell["arm"], cell["mode"], int(cell["scene"])
        logical = logical_job(cell)
        VISITED.append(logical)
        if _cell_complete(cfg, logical, cell):
            continue
        controller, identity = _load_arm(cfg, seed, arm)
        collector = r42_one_shot.OneShotSendSkipCollector(controller, greedy=(mode == "greedy"))
        tag = f"{ROW_TAG}_model_{arm.lower()}_s{seed}_{mode}"
        metadata = {"arm": arm, "training_seed": seed, "mode": mode, "scenario_seed": scene,
                    "eval_version": EVAL_VERSION}

        def _do(run_tag, controller=controller, collector=collector, tag=tag, scene=scene,
                mode=mode, arm=arm, identity=identity, seed=seed):
            row = runner._run_episode(run_tag, "eval", scene, policy="learned",
                                      attach=collector.policy(), controller=controller,
                                      collector=collector,
                                      action_rng_seed=runner.eval_action_seed(seed, scene, mode),
                                      tag=run_tag, checkpoint_load_identity=identity)
            row["model_arm"] = arm; row["eval_mode"] = mode; row["training_seed"] = seed
            return row

        row = _safe_atomic_run(cfg, logical, metadata, _do,
                               detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}",
                               tag=tag, persist=lambda r, t: persist_result(cfg, r, t), cell=cell)
    return {"status": "model_arms_eval_complete", "arms": list(MODEL_ARMS), "cells": 576}


def eval_focal(cfg, plan) -> dict:
    runner = cfg["runner"]
    import r42_one_shot
    for cell in eval_job_order():
        if cell["scope"] != "focal":
            continue
        seed, arm = int(cell["seed"]), cell["arm"]
        logical = logical_job(cell)
        VISITED.append(logical)
        if _cell_complete(cfg, logical, cell):
            continue
        controller, identity = _load_arm(cfg, seed, arm)
        collector = r42_one_shot.OneShotSendSkipCollector(controller, greedy=True)
        tag = f"{ROW_TAG}_focal_{arm.lower()}_s{seed}"
        metadata = {"arm": arm, "training_seed": seed, "mode": "greedy",
                    "scenario_seed": FOCAL_SCENE, "block": "D", "eval_version": EVAL_VERSION}

        def _do(run_tag, controller=controller, collector=collector, tag=tag, arm=arm,
                identity=identity, seed=seed):
            row = runner._run_episode(run_tag, "eval", FOCAL_SCENE, policy="learned",
                                      attach=collector.policy(), controller=controller,
                                      collector=collector,
                                      action_rng_seed=runner.eval_action_seed(seed, FOCAL_SCENE, "greedy"),
                                      tag=run_tag, checkpoint_load_identity=identity)
            row["model_arm"] = arm; row["eval_mode"] = "greedy"; row["training_seed"] = seed
            row["focal"] = True
            return row

        row = _safe_atomic_run(cfg, logical, metadata, _do,
                               detail=lambda r: f"on_time={r['on_time']} bytes={r['spent_bytes']}",
                               tag=tag, persist=lambda r, t: persist_result(cfg, r, t), cell=cell)
    return {"status": "focal_complete", "cells": 6}


def _assert_visited_order(plan) -> dict:
    expected = [j["logical_job"] for j in plan["jobs"]]
    if VISITED != expected:
        for i, (a, b) in enumerate(zip(VISITED, expected)):
            if a != b:
                raise AssertionError(f"execution order diverges from the plan at cell {i}: "
                                     f"ran {a}, plan {b}")
        raise AssertionError(f"visited {len(VISITED)} cells, plan has {len(expected)}")
    return {"kind": "round66-eval-order-check/1", "status": "passed", "cells": len(VISITED),
            "note": "the executor visited every planned cell in exactly the planned order"}


# ------------------------------------------------------------------ statistics --------------
def _bootstrap_index():
    import hashlib
    gen = np.random.Generator(np.random.PCG64(20261012))
    idx = gen.integers(0, 32, size=(10000, 32))
    return idx, hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest()


def on_time_target_ids(row: dict, scene: int) -> list[str]:
    """E7: the evidenced (scene, target_id) on-time set, rebuilt from the row's own task events.

    A real row has no on_time_target_ids / on_time_ids field, so an "or []" fallback would fabricate
    an empty set and hide every target difference. The set is closed against row["on_time"] in two
    ways - the number of distinct on-time task ids and the number of on-time events must both equal
    the reported count - and anything unreadable or inconsistent raises rather than writing [].
    """
    on_time = row.get("on_time")
    if not isinstance(on_time, int) or isinstance(on_time, bool):
        raise RuntimeError(f"scene {scene}: row has no integer on_time count")
    events = row.get("task_reward_events")
    if events is None:
        events = []
    if not isinstance(events, list):
        raise RuntimeError(f"scene {scene}: task_reward_events is {type(events).__name__}, not a list")
    if not events:
        # E7b: a legitimate zero-task row evidences an EMPTY set. An empty list is only a data
        # error when the row still claims on-time targets, so the emptiness alone is never the test.
        if int(on_time) == 0:
            return []
        raise RuntimeError(f"scene {scene}: row reports on_time={on_time} but carries no "
                           "task_reward_events to evidence it")
    ids, on_time_events = set(), 0
    for event in events:
        if not isinstance(event, dict):
            raise RuntimeError(f"scene {scene}: malformed task reward event {event!r}")
        if event.get("on_time") is not True:
            continue
        on_time_events += 1
        task_id = event.get("task_id")
        if not task_id:
            raise RuntimeError(f"scene {scene}: on-time event without a task_id")
        ids.add(str(task_id))
    if len(ids) != int(on_time) or on_time_events != int(on_time):
        raise RuntimeError(f"scene {scene}: {len(ids)} evidenced on-time target ids and "
                           f"{on_time_events} on-time events, but the row reports on_time={on_time}")
    return sorted(ids)


def _paired_target_diff(new_ids_of, other_ids_of, scenes) -> tuple[list, list]:
    """E6: per-(scene, target_id) differences, so a same-named target in two scenes cannot cancel."""
    lost, gained = [], []
    for scene in scenes:
        new_ids = set(new_ids_of(scene))
        other_ids = set(other_ids_of(scene))
        lost += [{"scene": int(scene), "target_id": t} for t in sorted(other_ids - new_ids)]
        gained += [{"scene": int(scene), "target_id": t} for t in sorted(new_ids - other_ids)]
    return lost, gained


def _qualified_row_paths(raw_root: Path) -> dict[str, list[str]]:
    """E8: the only admissible row source is a row NAMED BY a complete claim whose receipt re-proves.

    A retry that died after writing a partial row leaves that orphan on disk under an
    attempt-isolated name. Globbing the rows directory and letting dict insertion order pick a
    winner would let the orphan shadow, or be shadowed by, the real row - so the receipt decides,
    never the glob order. Two sound complete receipts naming different rows for one logical job are
    an ambiguity to refuse, not a tie to break by order.
    """
    claims_root = Path(raw_root) / "launch_claims"
    out: dict[str, list[str]] = {}
    if not claims_root.is_dir():
        return out
    for directory in sorted(p for p in claims_root.iterdir() if p.is_dir()):
        for path in sorted(directory.glob("attempt_*.json")):
            try:
                claim = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if claim.get("status") != "complete":
                continue
            logical = claim.get("logical_job")
            receipt = claim.get("result_receipt")
            if not logical or not isinstance(receipt, dict) or not receipt.get("row_path"):
                continue
            if not _receipt_is_sound(claim, None):
                continue
            resolved = str(Path(receipt["row_path"]).resolve())
            bucket = out.setdefault(str(logical), [])
            if resolved not in bucket:
                bucket.append(resolved)
    return out


def _cell_of(row: dict, scope: str) -> dict:
    if scope == "model_eval":
        return {"scope": scope, "arm": row["model_arm"], "seed": int(row["training_seed"]),
                "mode": row["eval_mode"], "scene": int(row["scenario_seed"])}
    if scope == "rule_eval":
        return {"scope": scope, "arm": row["model_arm"], "seed": None, "mode": row["eval_mode"],
                "scene": int(row["scenario_seed"])}
    return {"scope": "focal", "arm": row["model_arm"], "seed": int(row["training_seed"]),
            "mode": row["eval_mode"], "scene": int(row["scenario_seed"])}


def _load_qualified_rows(raw_root: Path, scope: str, prefix: str) -> tuple[dict, dict]:
    """Rows for one scope, keyed by (logical_job, cell identity), each from a single sound receipt."""
    qualified = _qualified_row_paths(raw_root)
    on_disk = {str(q.resolve()) for q in (Path(raw_root) / "rows").glob(f"v2_{ROW_TAG}_{prefix}_*.json")}
    named = {r for paths in qualified.values() for r in paths
             if r.startswith(f"{ROW_TAG}_{prefix}_")}
    rows, audit = {}, {"scope": scope, "qualified_logical_jobs": 0,
                       "row_files_on_disk": len(on_disk),
                       "orphan_row_files_excluded_from_statistics": sorted(on_disk - named)}
    for logical in sorted(qualified):
        if not logical.startswith(f"{ROW_TAG}_{prefix}_"):
            continue
        paths = qualified[logical]
        if len(paths) != 1:
            raise RuntimeError(f"{logical}: {len(paths)} sound complete receipts name different rows "
                               f"({paths}); refusing to choose one by order")
        row = base._read_json(paths[0])
        cell = _cell_of(row, scope)
        if logical_job(cell) != logical:
            raise RuntimeError(f"{logical}: row identity {logical_job(cell)} does not match its claim")
        key = ((str(cell["arm"]).upper(), int(cell["scene"]))
               if scope == "rule_eval" else
               (str(cell["arm"]).upper(), int(cell["seed"]), str(cell["mode"]), int(cell["scene"])))
        if key in rows:
            raise RuntimeError(f"{logical}: row key {key} is already owned by {rows[key]['logical_job']}")
        rows[key] = {"on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
                     "on_time_ids": on_time_target_ids(row, int(cell["scene"])),
                     "row_path": paths[0], "logical_job": logical}
        audit["qualified_logical_jobs"] += 1
    return rows, audit


def _load_model_rows(raw_root: Path) -> dict:
    rows, _audit = _load_qualified_rows(raw_root, "model_eval", "model")
    return rows


def _load_rule_rows(raw_root: Path) -> dict:
    rows, _audit = _load_qualified_rows(raw_root, "rule_eval", "rule")
    return {(arm.lower(), "greedy", scene): value
            for (arm, scene), value in rows.items()}


def compute_stats(raw_root: Path) -> dict:
    idx, idx_sha = _bootstrap_index()
    rows, model_audit = _load_qualified_rows(raw_root, "model_eval", "model")
    rules, rule_audit = _load_qualified_rows(raw_root, "rule_eval", "rule")
    rules = {(arm.lower(), "greedy", scene): value
             for (arm, scene), value in rules.items()}
    scenes = list(SCENES_NEW)
    per: dict[str, Any] = {}
    for seed in SEEDS:
        for mode in MODES:
            def cell(arm, scene):
                return rows.get((arm, seed, mode, scene))
            missing = [(a, s) for a in MODEL_ARMS for s in scenes if cell(a, s) is None]
            if missing:
                raise RuntimeError(f"missing paired cells seed={seed} mode={mode}: "
                                   f"{missing[:4]} ({len(missing)})")
            entry: dict[str, Any] = {}
            for arm in MODEL_ARMS:
                entry[f"{arm}_on_time_total"] = int(sum(cell(arm, s)["on_time"] for s in scenes))
                entry[f"{arm}_bytes_per_scene"] = float(np.mean([cell(arm, s)["spent_bytes"] for s in scenes]))

            def _task_cost(other_of):
                d_task = np.array([cell("NEW", s)["on_time"] - other_of(s)["on_time"] for s in scenes],
                                  dtype=np.float64)
                d_cost = np.array([cell("NEW", s)["spent_bytes"] - other_of(s)["spent_bytes"]
                                   for s in scenes], dtype=np.float64)
                return d_task, d_cost

            for other in ("OLD", "BC"):
                d_task, d_cost = _task_cost(lambda s, o=other: cell(o, s))
                entry[f"task_delta_NEW_minus_{other}_mean"] = float(d_task.mean())
                entry[f"task_ci95_NEW_minus_{other}"] = [
                    float(np.quantile(d_task[idx].mean(axis=1), 0.025)),
                    float(np.quantile(d_task[idx].mean(axis=1), 0.975))]
                entry[f"cost_delta_NEW_minus_{other}_B_per_scene"] = float(d_cost.mean())
                entry[f"cost_ci95_NEW_minus_{other}_B_per_scene"] = [
                    float(np.quantile(d_cost[idx].mean(axis=1), 0.025)),
                    float(np.quantile(d_cost[idx].mean(axis=1), 0.975))]
                entry[f"scenes_NEW_better_task_than_{other}"] = int((d_task > 0).sum())
                entry[f"scenes_NEW_worse_task_than_{other}"] = int((d_task < 0).sum())
                entry[f"scenes_NEW_cheaper_than_{other}"] = int((d_cost < 0).sum())
                lost, gained = _paired_target_diff(lambda s: cell("NEW", s)["on_time_ids"],
                                                   lambda s, o=other: cell(o, s)["on_time_ids"],
                                                   scenes)
                entry[f"targets_lost_NEW_vs_{other}"] = lost
                entry[f"targets_gained_NEW_vs_{other}"] = gained
                entry[f"targets_lost_NEW_vs_{other}_count"] = len(lost)
                entry[f"targets_gained_NEW_vs_{other}_count"] = len(gained)

            rule_by_arm = {}
            for arm in RULE_ARMS:
                have = [rules.get((arm.lower(), "greedy", s)) for s in scenes]
                if all(h is not None for h in have):
                    rule_by_arm[arm] = dict(zip(scenes, have))
            if mode == "greedy":
                for arm, by_scene in rule_by_arm.items():
                    entry[f"{arm}_on_time_total"] = int(sum(by_scene[s]["on_time"] for s in scenes))
                    entry[f"{arm}_bytes_per_scene"] = float(np.mean([by_scene[s]["spent_bytes"]
                                                                    for s in scenes]))
                    entry[f"NEW_minus_{arm}_bytes_B_per_scene"] = float(
                        np.mean([cell("NEW", s)["spent_bytes"] for s in scenes])
                        - entry[f"{arm}_bytes_per_scene"])
                    d_task, d_cost = _task_cost(lambda s, a=arm: rule_by_arm[a][s])
                    entry[f"task_delta_NEW_minus_{arm}_mean"] = float(d_task.mean())
                    entry[f"task_ci95_NEW_minus_{arm}"] = [
                        float(np.quantile(d_task[idx].mean(axis=1), 0.025)),
                        float(np.quantile(d_task[idx].mean(axis=1), 0.975))]
                    entry[f"cost_delta_NEW_minus_{arm}_B_per_scene"] = float(d_cost.mean())
                    lost, gained = _paired_target_diff(lambda s: cell("NEW", s)["on_time_ids"],
                                                       lambda s, a=arm: rule_by_arm[a][s]["on_time_ids"],
                                                       scenes)
                    entry[f"targets_lost_NEW_vs_{arm}"] = lost
                    entry[f"targets_gained_NEW_vs_{arm}"] = gained
                    entry[f"targets_lost_NEW_vs_{arm}_count"] = len(lost)
                    entry[f"targets_gained_NEW_vs_{arm}_count"] = len(gained)
            else:
                entry["rule_arms_not_comparable_in_this_mode"] = {
                    "reason": ("rule arms are evaluated greedy-only on this block; no sampled rule "
                               "cells exist, so no NEW-vs-rule number is reported for the sampled "
                               "mode and no rule greedy figure is presented as a sampled result"),
                    "rule_arms": list(RULE_ARMS)}
            per[f"{seed}/{mode}"] = entry
    return {"kind": "round66-corrected-statistics/2", "eval_version": EVAL_VERSION,
            "primary_comparison": "NEW_minus_OLD",
            "secondary_comparison": "NEW_minus_BC",
            "comparison_labels_are_fixed_by_the_authorisation": {
                "NEW_minus_OLD": ("structural contrast: R66 ep128 (frozen early path) minus the "
                                  "compliant R60 ep128 it started from - isolates the restricted "
                                  "policy family, NOT an added-learning gain"),
                "NEW_minus_BC": ("the authorised added-learning reading: R66 ep128 minus the "
                                 "compliant R60 ep0 initial value it was initialised from"),
                "labels_fixed": True},
            "greedy_is_primary": True, "sampled_reported_separately": True,
            "scene_unit": 32, "seeds_not_pooled": True,
            "on_time_target_id_source": {
                "field_used": "task_reward_events[].task_id where on_time is true",
                "absent_field": "on_time_target_ids / on_time_ids are NOT present on a real row",
                "closure": ("distinct on-time task ids == on-time event count == row['on_time'], "
                            "per cell; any mismatch raises instead of writing an empty set"),
                "pairing": "differences are reported per (scene, target_id)"},
            "rule_comparison_scope": "greedy entries only; the sampled entry records non-applicability",
            "bootstrap": {"rng": "PCG64", "seed": 20261012, "draws": 10000, "index_sha256": idx_sha,
                          "quantile": "numpy linear 0.025/0.975"},
            "row_source_audit": {
                "model": model_audit, "rule": rule_audit,
                "admission_rule": ("every admitted row is named by a complete claim whose "
                                   "result_receipt re-proves from disk; orphan rows left by a failed "
                                   "retry are excluded, never chosen by glob order; two sound "
                                   "receipts for one logical job are refused as ambiguous")},
            "per_seed_mode": per, "cells": len(rows),
            "note": ("NEW=R66 ep128 frozen-early-path, OLD=compliant R60 ep128, BC=compliant R60 ep0 "
                     "initial value; all read-only on the fresh 32-scene block. CI containing 0 is "
                     "not equivalence. Restricted policy family, single development block, three "
                     "seeds not pooled.")}


# ------------------------------------------------------------------ CLI ---------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND66 corrected evaluation adapter (v2)")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--phase", choices=("plan", "eval", "stats"), required=True)
    args = ap.parse_args(argv)
    raw_root = Path(args.raw_root).resolve()
    if args.phase == "plan":
        plan = build_eval_plan()
        base._write_json(raw_root / "ROUND66_EVAL_PLAN_CORRECTED_2.json", plan)
        print(json.dumps({k: v for k, v in plan.items() if k != "jobs"}, indent=1))
        return 0
    cfg = configure_r66_eval(raw_root)
    base._write_json(raw_root / "runtime" / "ROUND66_EVAL_QUOTA_GUARD.json",
                     cfg["r66_eval_quota_guard"])
    if args.phase == "eval":
        plan = base._read_json(raw_root / "ROUND66_EVAL_PLAN_CORRECTED_2.json")
        VISITED.clear()
        print(json.dumps(eval_rules(cfg, plan), indent=1), flush=True)
        print(json.dumps(eval_model_arms(cfg, plan), indent=1), flush=True)
        print(json.dumps(eval_focal(cfg, plan), indent=1), flush=True)
        order = _assert_visited_order(plan)
        base._write_json(raw_root / "runtime" / "ROUND66_EVAL_ORDER_CHECK.json", order)
        print(json.dumps(order, indent=1), flush=True)
        base._write_json(raw_root / "runtime" / "ROUND66_EVAL_QUOTA_GUARD.json",
                         quota_guard(cfg))
        return 0
    stats = compute_stats(raw_root)
    base._write_json(raw_root / "ROUND66_CORRECTED_STATISTICS.json", stats)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
