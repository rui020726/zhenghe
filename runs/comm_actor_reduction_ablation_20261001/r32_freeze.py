"""ROUND32 冻结件生成：代码身份、配置哈希、两臂定义、数据顺序、启动计划与统计方案。

差异审计的参照物是 **ROUND30 的冻结件**（ROUND30 自己已与 ROUND24 逐项审计过，身份未变的
历史证据按执行依据 §二复用，不重跑）：

- 必须**完全相同**的块：`configuration_hash`、`ppo_hyperparameters`、`wiring_constants`、
  `data_order`、`rule_arms`、`scene_registry`、`train_seeds`、`inherited_behaviour_fields`、
  `statistics_plan`（除本轮新增的 `inherited_from` 标注键）、
  `code_identity` 的 production / imported_round_modules / checkpoints 三段；
- 允许差异的块必须是本轮身份、输出目录与命名、归约臂定义、诊断证据计划、启动计划五类之一。

配置哈希与行为字段直接复用 `r30_check` 的同一构造函数，不另写第二套。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import r32_common as common  # noqa: F401  (installs the shared sys.path before any archived-round import)
import r30_check as check30
import r30_common as c30
import r32_run as runner
import r32_trainer as trainer
import v2_common as v2c

BRANCH_START = "614df51709a8d82525405b46468a537f4ccd259e"      # ROUND31 delivery head
ROUND30_HEAD = "915868297c08d4b01b7dfeff0ea810d5394057a3"
ROUND30_MANIFEST = common.R30_DIR / "CONFIG_AND_MODEL_MANIFEST.json"
ROUND31_HEADS = {"execution_basis_commit": "c49a678e051832175a8980bd15b40b74a89a4552",
                 "delivery_commit": "614df51709a8d82525405b46468a537f4ccd259e"}

#: 与 ROUND30 冻结件必须逐值相同的块
IDENTICAL_BLOCKS = ("configuration_hash", "ppo_hyperparameters", "wiring_constants",
                    "data_order", "rule_arms", "scene_registry", "train_seeds",
                    "inherited_behaviour_fields")
IDENTICAL_CODE_SECTIONS = ("production", "imported_round_modules", "checkpoints")
#: 允许本轮新增键的块 -> 允许新增的键名
EXTENDABLE_BLOCKS = {"statistics_plan": {"inherited_from"}, "launch_plan": {
    "max_parallel_workers", "first_batch_gate_uses_official_episodes_only", "no_extra_smoke",
    "all_launches_counted_including_failures_and_retries"}}
#: 只做注记/命名的键：内容值仍严格比，这些键允许措辞不同（逐条列出，不笼统放行）
ANNOTATION_KEYS = {
    "data_order": {"order_sha256_note", "order_digest_sha256"},
    "scene_registry": {"note"},
    "statistics_plan": set()}


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def head_at_freeze() -> str:
    return git("rev-parse", "HEAD")


def current_branch() -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD")


def inherited_behaviour_fields() -> dict[str, Any]:
    return check30.behaviour_fields()


def configuration_hash() -> str:
    return check30.configuration_hash(check30.behaviour_fields())


def _diff_values(previous: Any, current: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(previous, dict) and isinstance(current, dict):
        for key in sorted(set(previous) | set(current)):
            if previous.get(key) != current.get(key):
                out[key] = {"round30": previous.get(key), "round32": current.get(key)}
    elif previous != current:
        out["<value>"] = {"round30": previous, "round32": current}
    return out


def round30_identity_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compare this freeze with ROUND30's frozen block; only declared categories may differ."""
    previous = common.read_json(ROUND30_MANIFEST)
    problems: list[str] = []
    compared: dict[str, Any] = {}

    for block in IDENTICAL_BLOCKS:
        delta = _diff_values(previous.get(block), manifest.get(block))
        annotations = {key: value for key, value in delta.items()
                       if key in ANNOTATION_KEYS.get(block, set())}
        load_bearing = {key: value for key, value in delta.items() if key not in annotations}
        compared[block] = {"identical_on_load_bearing_values": not load_bearing,
                           "differences": load_bearing,
                           "annotation_only_differences": sorted(annotations),
                           "annotation_keys_declared_in_this_round": sorted(
                               ANNOTATION_KEYS.get(block, set()))}
        if load_bearing:
            problems.append(f"block {block!r} differs from the ROUND30 freeze: "
                            f"{json.dumps(load_bearing, ensure_ascii=False)[:400]}")

    for block, allowed_new_keys in EXTENDABLE_BLOCKS.items():
        delta = _diff_values(previous.get(block), manifest.get(block))
        unexpected = sorted(key for key in delta if key not in allowed_new_keys)
        compared[block] = {"identical_except_declared": not unexpected,
                           "new_or_changed_keys": sorted(delta),
                           "unexpected_keys": unexpected,
                           "differences": delta}
        if unexpected:
            problems.append(f"block {block!r} changed outside the declared keys: {unexpected}")

    previous_ci, current_ci = previous.get("code_identity", {}), manifest["code_identity"]
    for section in IDENTICAL_CODE_SECTIONS:
        delta = _diff_values(previous_ci.get(section, {}), current_ci.get(section, {}))
        compared[f"code_identity.{section}"] = {"identical": not delta, "differences": delta}
        if delta:
            problems.append(f"code identity section {section!r} differs: "
                            f"{json.dumps(delta, ensure_ascii=False)[:600]}")

    registered = [
        {"difference": "round identity, branch, head and output directory (r32_* cells)",
         "where": "this round's own directory; ROUND24/30/31 rows, logs and weights are never "
                  "written"},
        {"difference": "the two arms are surrogate reductions (ALL_ROWS / ACTIVE_ROWS) instead "
                       "of encoder layouts",
         "where": "r32_trainer.actor_surrogate_mean, registered by the block-classified source "
                  "diff; the encoder is pinned to ROUND30's old arm for both arms"},
        {"difference": "diagnostic evidence added: per-minibatch N/M/loss/KL/clip/gradient-norm "
                       "reads and the pre-update checkpoints of batches 0/15/31",
         "where": "r32_trainer.ppo_update passive reads and r32_train.save_immutable; proved not "
                  "to change the update by REDUCTION_MATH.json T9"},
        {"difference": "evaluation arrangement: 2 reductions x 3 seeds x 2 modes x 16 dev "
                       "scenes = 192",
         "where": "the same shape as ROUND30, with reductions replacing encoder arms"},
    ]
    return {"reference": str(ROUND30_MANIFEST.relative_to(common.ROOT)),
            "reference_round30_head": ROUND30_HEAD,
            "reference_freeze_sha256": previous_ci.get("freeze_sha256"),
            "compared": compared,
            "registered_differences": registered,
            "problems": problems,
            "clean": not problems}


def build_manifest() -> dict[str, Any]:
    fields = inherited_behaviour_fields()
    order = common.scenario_order()
    identity = common.code_identity()
    order_repr = json.dumps(order, sort_keys=True)
    manifest = {
        "round": "32",
        "phase": "actor surrogate reduction paired ablation: pre-launch freeze",
        "branch": "exp/comm-actor-reduction-ablation",
        "branch_current_at_freeze": current_branch(),
        "branch_start": BRANCH_START,
        "head_at_freeze": head_at_freeze(),
        "round30_head": ROUND30_HEAD,
        "round31_heads": ROUND31_HEADS,
        "remote_start_head": None,
        "spec": "execution basis EXECUTION_BASIS.md (sha256 "
                "49d52f9511400e68085cc8872be7853e8b0239ee8821f9053bd2fb20b0638780)",
        "output_dir": str(common.OUTPUT_DIR.relative_to(common.ROOT)),
        "configuration_hash": configuration_hash(),
        "configuration_hash_construction": (
            "r30_check.configuration_hash(behaviour_fields) -- the same construction ROUND24/30 "
            "used, reused rather than re-implemented"),
        "train_seeds": list(common.TRAIN_SEEDS),
        "episodes_per_seed": common.EPISODES_PER_SEED,
        "ppo_hyperparameters": {
            "clip": v2c.CLIP, "value_coefficient": v2c.VALUE_COEFFICIENT,
            "entropy_bonus": v2c.ENTROPY_BONUS, "learning_rate": v2c.LEARNING_RATE,
            "max_grad_norm": v2c.MAX_GRAD_NORM, "epochs_per_batch": v2c.EPOCHS_PER_BATCH,
            "minibatch_steps": v2c.MINIBATCH_STEPS},
        "wiring_constants": {
            "b0_bytes": v2c.B0_BYTES, "delta": v2c.DELTA, "lambda_init": v2c.LAMBDA_INIT,
            "eta_lambda": v2c.ETA_LAMBDA, "gamma": v2c.GAMMA, "gae_lambda": v2c.GAE_LAMBDA,
            "log_prob_tolerance": v2c.LOG_PROB_TOLERANCE, "steps_per_episode": v2c.STEPS_PER_EPISODE,
            "episodes_per_update": v2c.EPISODES_PER_UPDATE,
            "episodes_per_seed": v2c.EPISODES_PER_SEED, "data_order_seed": v2c.DATA_ORDER_SEED},
        "objective": {
            "c": "C_measured / B0", "g": "E[Y_reference] - E[Y_policy] - delta",
            "L_max": "-E[c] - lambda*g", "A_actor": "lambda*A_task - A_positive_cost",
            "dual": "lambda_next = max(0, lambda + eta_lambda * g_hat)",
            "lambda_unit": "normalised cost per task",
            "advantage_standardisation": "none (the registered difference of the correctness-v2 "
                                         "trainer; neither component is standardised separately)",
            "delta_is_a_training_target_not_a_deployment_guarantee": True,
        },
        "inherited_behaviour_fields": fields,
        "data_order": {
            "algorithm": "random.Random(DATA_ORDER_SEED); the 48 training scenarios reshuffled "
                         "each of 3 rounds, 48 + 48 + 32 taken (frozen ROUND24 algorithm)",
            "episodes_per_seed": common.EPISODES_PER_SEED,
            "seed": v2c.DATA_ORDER_SEED,
            "order_first_8": order[:8],
            "order_sha256": order_repr,
            "order_sha256_note": "the ROUND24/30 field name is kept for comparability; the value "
                                 "is the sorted JSON dump of the 128-scenario order, not a digest",
            "order_digest_sha256": hashlib.sha256(order_repr.encode("utf-8")).hexdigest(),
        },
        "scene_registry": {
            "training": list(common.TRAINING_SCENARIOS),
            "development": list(common.DEVELOPMENT_SCENARIOS),
            "note": "already-used blocks; this is a development ablation, not an unseen test set "
                    "and not an independent confirmation; the ROUND25 evaluation block "
                    "1400000..1400063 is not used for parameter, checkpoint or stopping "
                    "decisions in this round; the sealed blocks 800000..800399 and "
                    "900000..900199 are not touched",
        },
        "rule_arms": {name: dict(spec) for name, spec in common.RULE_ARMS.items()},
        "encoder_identity": {
            "both_arms_use": runner.ENCODER_ARM,
            "feature_layout": c30.ARMS["old"]["feature_layout"],
            "deadline_column": c30.ARMS["old"]["deadline_column"],
            "encoder_object": "integrated_sar.comm_learning.features (production encoder)",
            "collector": "r30_encoding.scoped_training_collector(controller, 'old') -- the "
                         "ROUND30 collector whose tape was verified field-by-field against the "
                         "frozen parent collector",
            "new_deadline_encoding_used": False,
            "round30_scoped_collector_source_diff": enc_source_diff(),
        },
        "arms": {name: dict(spec) for name, spec in common.REDUCTIONS.items()},
        "reduction_definition": {
            "N": "number of logged decision rows in the minibatch",
            "M": "rows whose saved `actionable`/`free` set is non-empty",
            "ALL_ROWS": "-sum(s_i for i<N)/N  (the frozen expression, verbatim)",
            "ACTIVE_ROWS": "-sum(s_i*1[i active])/max(M,1)",
            "M_is_zero": "the numerator is already an exact zero and keeps a grad_fn, so no "
                         "special branch is needed and no NaN can appear",
            "active_row_rules": ["recorded rows, not environment steps and not candidates",
                                 "a legal sampled HOLD still counts as a free decision",
                                 "forced or masked candidates never make a row active",
                                 "the saved executed-action prefix decides activity; the PPO "
                                 "recompute does not re-sample",
                                 "several free candidates in one row still count as one row"],
            "gradient_identity": "grad(ACTIVE) = (N/M) * grad(ALL) holds for the ACTOR SURROGATE "
                                 "term only; the loss scalar does not scale that way because a "
                                 "masked row's surrogate value is a non-zero constant while its "
                                 "gradient is exactly zero",
            "unchanged": ["task/cost advantages and their normalisation", "GAE", "reward timing",
                          "both critics' samples, reduction and coefficients", "entropy reduction "
                          "and coefficient", "the lambda update and when it takes effect",
                          "PPO clipping, the KL field and the existing stop logic", "minibatch "
                          "construction and order", "optimizer, learning rate, global clipping",
                          "network, observation, mask and environment"],
            "explicitly_not_claimed": "that no other optimisation relationship is unaffected: the "
                                      "actor's weight relative to entropy and the critics is "
                                      "exactly what this ablation changes, and no fully masked "
                                      "row is removed from the tape, GAE, critics or minibatches",
        },
        "launch_plan": {
            "reference": common.REFERENCE_LAUNCHES, "training": common.TRAINING_LAUNCHES,
            "eval": common.EVAL_LAUNCHES, "late_rule": common.LATE_RULE_LAUNCHES,
            "normal_launches": common.NORMAL_LAUNCHES, "reserve_launches": common.RESERVE_LAUNCHES,
            "hard_cap": common.LAUNCH_CAP,
            "synthetic_update_cap": common.SYNTHETIC_UPDATE_CAP,
            "max_parallel_workers": common.MAX_PARALLEL_WORKERS,
            "first_batch_gate_uses_official_episodes_only": True,
            "no_extra_smoke": True,
            "all_launches_counted_including_failures_and_retries": True,
        },
        "evaluation_plan": {
            "checkpoint": "ep128 milestone of each (reduction, seed) run, fixed before launch",
            "modes": {"greedy": "argmax with ties to SEND (archived implementation)",
                      "sampled": "the archived masked multinomial sampler"},
            "rollouts_per_cell": 1,
            "eval_action_rng": common.EVAL_RNG_SCHEME,
            "training_action_rng": common.ACTION_RNG_SCHEME,
            "same_seed_does_not_mean_same_per_message_losses": (
                "channel RNG comes from the scenario seed and the streams diverge as soon as the "
                "send sequences differ"),
        },
        "diagnostic_evidence_plan": {
            "per_minibatch": ["n_rows N", "m_active_rows M", "M/N",
                              "actor surrogate mean (arm-specific)",
                              "arm-independent surrogate mean over all rows",
                              "mean over active rows (reference)", "entropy term",
                              "both critic losses", "approx_kl", "clip_fraction",
                              "ratio min/max", "advantage mean (all and active-only)",
                              "actual lambda", "pre-clip total gradient norm",
                              "post-clip total norm, clipping flag and scale",
                              "per-module pre-clip L1 and L2", "non-finite flag"],
            "per_batch": ["weight hash before and after", "module digests before and after",
                          "optimizer parameter shapes vs controller parameter shapes",
                          "frozen search/allocation checkpoint hashes", "g_hat and its pairing",
                          "dual record", "reduction mode and feature layout",
                          "aggregated N/M and gradient readings"],
            "pre_update_checkpoints_saved_for_batches": list(common.PRE_UPDATE_BATCHES),
            "pre_update_checkpoint_contents": ["model", "optimizer", "lambda and history",
                                               "episodes_done / updates / optimizer_steps",
                                               "python + numpy + torch RNG state",
                                               "data position (batch and its scenarios)",
                                               "freeze carrying reduction mode, feature layout "
                                               "and round identity"],
            "record_must_not_touch": "RNG, actions and the parameter update; proved by "
                                     "REDUCTION_MATH.json T9, where ALL_ROWS reproduces the "
                                     "frozen trainer's weight hash after every optimizer step",
            "why_b0_is_not_enough_any_more": "ROUND31 could only analyse batch 0 because only "
                                             "the initial milestone survived; this round saves "
                                             "the pre-update weights of batches 0, 15 and 31",
        },
        "statistics_plan": common.STATISTICS_PLAN,
        "code_identity": identity,
        "runner_seam_diff": runner.seam_diff(),
        "trainer_source_diff": trainer.source_diff(),
        "trainer_identity_assertions": trainer.identity_assertions(),
        "reduction_math_evidence": "REDUCTION_MATH.json (11 checks, all passed, 0 launches, "
                                   "24/64 synthetic optimizer steps)",
        "carry_forward_from_archived_rounds": [
            "ROUND23 correctness fixes: reused as archived, identity unchanged, not re-run",
            "ROUND24/30 reference behaviour identity: re-checked against both archives before "
            "training starts (block_reference compares behaviour fields with both)",
            "the clock rule now = tape step - 1: verified in ROUND30, reused not re-derived",
        ],
    }
    manifest["round30_identity_audit"] = round30_identity_audit(manifest)
    return manifest


def enc_source_diff() -> dict[str, Any]:
    import r30_encoding as enc

    return enc.source_diff()


def write_manifest() -> dict[str, Any]:
    manifest = build_manifest()
    target = common.OUTPUT_DIR / "CONFIG_AND_MODEL_MANIFEST.json"
    if target.exists():
        raise RuntimeError("the freeze manifest already exists; refusing to silently refreeze")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                      encoding="utf-8")
    return manifest


def freeze_record(reduction: str) -> dict[str, Any]:
    """The freeze record every checkpoint of this round is compared against."""
    manifest = common.read_json(common.OUTPUT_DIR / "CONFIG_AND_MODEL_MANIFEST.json")
    return {"freeze_sha256": manifest["code_identity"]["freeze_sha256"],
            "config_hash": manifest["configuration_hash"],
            "branch_start": manifest["branch_start"],
            "head_at_freeze": manifest["head_at_freeze"],
            "round": "32", "reduction": reduction,
            "encoder_arm": runner.ENCODER_ARM,
            "feature_layout": c30.ARMS[runner.ENCODER_ARM]["feature_layout"],
            "reduction_definition": common.REDUCTIONS[reduction]["actor_reduction"]}


def init_pairing_evidence() -> dict[str, Any]:
    """Pre-launch proof that both arms of one seed start from byte-identical state.

    Only a fresh Controller(seed), an Adam over its parameters and a fresh training state are
    constructed -- no environment, no rollout. r32_train re-verifies the same file at the first
    real fresh initialisation instead of rewriting it.
    """
    import torch
    from v2_controller import fresh_controller, parameter_identity
    from r32_train import module_digests, init_pairing_path

    per_cell: dict[str, Any] = {}
    for reduction in common.ARM_ORDER:
        for seed in common.TRAIN_SEEDS:
            controller = fresh_controller(int(seed))
            optimizer = torch.optim.Adam(controller.parameters, lr=common.v2c.LEARNING_RATE)
            state = common.v2c and __import__("v2_trainer").fresh_training_state(int(seed))
            payload = {
                "reduction": reduction, "arm": reduction, "training_seed": int(seed),
                "encoder_arm": runner.ENCODER_ARM,
                "feature_layout": c30.ARMS[runner.ENCODER_ARM]["feature_layout"],
                "controller_parameters": parameter_identity(controller),
                "module_digests": module_digests(controller),
                "optimizer_state_sha256": hashlib.sha256(json.dumps(
                    optimizer.state_dict(), sort_keys=True, default=str).encode("utf-8")).hexdigest(),
                "lambda": float(state["lambda"]),
                "fresh_state": {key: state[key] for key in
                                ("training_seed", "lambda", "updates", "episodes_done",
                                 "optimizer_steps")},
                "written_before_the_first_environment_launch": True,
            }
            per_cell[f"{reduction}_s{seed}"] = payload
            path = init_pairing_path(reduction, int(seed))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True)
                            + "\n", encoding="utf-8")
    pairing = {}
    for seed in common.TRAIN_SEEDS:
        left = per_cell[f"{common.ARM_ORDER[0]}_s{seed}"]
        right = per_cell[f"{common.ARM_ORDER[1]}_s{seed}"]
        pairing[f"s{seed}"] = {
            "parameter_sha256_equal": (left["controller_parameters"]["parameter_sha256"]
                                       == right["controller_parameters"]["parameter_sha256"]),
            "module_digests_equal": left["module_digests"] == right["module_digests"],
            "optimizer_state_equal": (left["optimizer_state_sha256"]
                                      == right["optimizer_state_sha256"]),
            "lambda_equal": left["lambda"] == right["lambda"] == common.LAMBDA_INIT,
            "fresh_state_equal": left["fresh_state"] == right["fresh_state"],
            "parameter_sha256": left["controller_parameters"]["parameter_sha256"],
            "module_digests": left["module_digests"],
        }
        pairing[f"s{seed}"]["all_equal"] = all(
            value is True for key, value in pairing[f"s{seed}"].items()
            if isinstance(value, bool))
    aggregate = {"round": "32",
                 "section": "initialisation pairing of the two reduction arms, written before the "
                            "first environment launch (0 launches)",
                 "arms": list(common.ARM_ORDER), "seeds": list(common.TRAIN_SEEDS),
                 "cells": per_cell, "pairing": pairing,
                 "identical_for_every_seed": all(pairing[f"s{seed}"]["all_equal"]
                                                 for seed in common.TRAIN_SEEDS),
                 "note": "the only difference between the two arms is the reduction mode recorded "
                         "in each checkpoint's freeze; the starting weights, optimizer state and "
                         "lambda are byte-identical per seed"}
    (common.OUTPUT_DIR / "INIT_PAIRING.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    return {"identical_for_every_seed": aggregate["identical_for_every_seed"],
            "pairing": {key: value for key, value in pairing.items()}}


def main(argv: list[str]) -> int:
    if argv and argv[0] == "initpairing":
        print(json.dumps(init_pairing_evidence(), ensure_ascii=False, indent=1))
        return 0
    if argv and argv[0] == "freeze":
        manifest = write_manifest()
        print(json.dumps({
            "freeze_sha256": manifest["code_identity"]["freeze_sha256"],
            "configuration_hash": manifest["configuration_hash"],
            "order_digest": manifest["data_order"]["order_digest_sha256"][:16],
            "audit_clean": manifest["round30_identity_audit"]["clean"],
            "audit_problems": manifest["round30_identity_audit"]["problems"],
            "runner_seam_clean": manifest["runner_seam_diff"]["passed"],
            "trainer_diff_clean": manifest["trainer_source_diff"]["passed"]},
            ensure_ascii=False, indent=1))
        return 0
    if argv and argv[0] == "manifest":
        print(json.dumps(build_manifest(), ensure_ascii=False, indent=1, sort_keys=True))
        return 0
    if argv and argv[0] == "record":
        print(json.dumps(freeze_record(argv[1]), ensure_ascii=False, indent=1))
        return 0
    print("usage: r32_freeze.py [freeze | manifest | record <reduction>]")
    return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
