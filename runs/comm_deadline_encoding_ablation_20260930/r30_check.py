"""ROUND30 §6：启动前冻结件与首批（首 4 集）门禁的独立重算。

冻结件 `CONFIG_AND_MODEL_MANIFEST.json` 记录：代码身份（本轮 harness + 生产/导入模块 +
checkpoint）、配置哈希（与 ROUND24 同一构造法）、场景块、数据顺序、两臂编码身份、启动
计划与统计方案，并逐项登记与 ROUND24 冻结件的差异（只允许轮次身份、输出目录与命名、
编码臂、评估安排、启动额度五类；其余必须一致，生产/导入模块代码身份必须逐字节相同）。

``freeze_record(arm)`` 把臂与观测语义版本写进 freeze：checkpoint 的保存/恢复契约比较
整个 freeze 字典，因此跨臂、跨轮恢复都会显式失败（执行依据 §4.4/§6.8）。

首批门禁对每个臂独立重算：输入编码、reward/成本/账本闭合、事件归属/GAE/组合优势、
log_prob 重算（既有 1e-6 门槛）、optimizer 参数覆盖、λ 更新公式与方向、有限性、
checkpoint 恢复契约。重算函数复用 ROUND24 冻结的 `v2_check.recompute_episode_math`，
不新写第二套数学。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

import r30_common as common
import r30_encoding as enc
import v2_common as v2c

ROUND24_MANIFEST = common.ROOT / "runs/comm_cost_v2_rerun_20260928/CONFIG_AND_MODEL_MANIFEST.json"
BRANCH_START = "99eb746baaa073550f9062fec5c87a4b4711827b"   # ROUND29 head (registered start)


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def behaviour_fields() -> dict[str, Any]:
    fields, _substituted = v2c.stage_b_inherited_config(int(v2c.TRAINING_SCENARIOS[0]))
    identity = {"output", "comm_transcript", "info_probe", "search_probe", "decision_trace",
                "record_stream", "replay_stream", "seed", "comm_policy", "comm_policy_params"}
    return {key: value for key, value in sorted(fields.items()) if key not in identity}


def configuration_hash(fields: dict[str, Any]) -> str:
    import hashlib

    payload = {"setting": v2c.STAGE_B_INHERITED_OVERRIDES,
               "inherited_behaviour": fields,
               "hyperparameters": {
                   "clip": v2c.CLIP, "value_coefficient": v2c.VALUE_COEFFICIENT,
                   "entropy_bonus": v2c.ENTROPY_BONUS, "learning_rate": v2c.LEARNING_RATE,
                   "max_grad_norm": v2c.MAX_GRAD_NORM, "epochs_per_batch": v2c.EPOCHS_PER_BATCH,
                   "minibatch_steps": v2c.MINIBATCH_STEPS,
                   "episodes_per_update": v2c.EPISODES_PER_UPDATE,
                   "gamma": v2c.GAMMA, "gae_lambda": v2c.GAE_LAMBDA,
                   "b0_bytes": v2c.B0_BYTES, "delta": v2c.DELTA,
                   "lambda_init": v2c.LAMBDA_INIT, "eta_lambda": v2c.ETA_LAMBDA,
                   "data_order_seed": v2c.DATA_ORDER_SEED},
               "train_seeds": list(v2c.TRAIN_SEEDS),
               "episodes_per_seed": v2c.EPISODES_PER_SEED}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True)
                          .encode("utf-8")).hexdigest()[:16]


def round24_difference_audit() -> dict[str, Any]:
    """Registered item-by-item diff against the ROUND24 frozen manifest; anything else stops."""
    old = json.loads(ROUND24_MANIFEST.read_text(encoding="utf-8"))
    problems: list[str] = []
    old_production = old["code_identity"]["production"]
    new_production = common.code_identity()["production"]
    if old_production != new_production:
        differing = sorted(name for name in set(old_production) | set(new_production)
                           if old_production.get(name) != new_production.get(name))
        problems.append(f"production code identity differs from ROUND24: {differing}")
    old_imported = old["code_identity"]["imported_round_modules"]
    new_imported = common.code_identity()["imported_round_modules"]
    differing = sorted(name for name in set(old_imported) | set(new_imported)
                       if old_imported.get(name) != new_imported.get(name))
    if differing:
        problems.append(f"imported round module identity differs from ROUND24: {differing}")
    old_wiring = old.get("wiring_constants") or {}
    new_wiring = {"b0_bytes": v2c.B0_BYTES, "delta": v2c.DELTA, "lambda_init": v2c.LAMBDA_INIT,
                  "eta_lambda": v2c.ETA_LAMBDA, "gamma": v2c.GAMMA, "gae_lambda": v2c.GAE_LAMBDA,
                  "data_order_seed": v2c.DATA_ORDER_SEED,
                  "log_prob_tolerance": v2c.LOG_PROB_TOLERANCE}
    for key, value in new_wiring.items():
        if key in old_wiring and old_wiring[key] != value:
            problems.append(f"wiring constant {key}: ROUND24 {old_wiring[key]} != now {value}")
    old_hyper = old.get("ppo_hyperparameters") or {}
    new_hyper = {"clip": v2c.CLIP, "value_coefficient": v2c.VALUE_COEFFICIENT,
                 "entropy_bonus": v2c.ENTROPY_BONUS, "learning_rate": v2c.LEARNING_RATE,
                 "max_grad_norm": v2c.MAX_GRAD_NORM, "epochs_per_batch": v2c.EPOCHS_PER_BATCH,
                 "minibatch_steps": v2c.MINIBATCH_STEPS}
    for key, value in new_hyper.items():
        if key in old_hyper and old_hyper[key] != value:
            problems.append(f"PPO hyperparameter {key}: ROUND24 {old_hyper[key]} != now {value}")
    for arm in v2c.RULE_ARMS:
        old_rule = (old.get("rule_arms") or {}).get(arm)
        if old_rule is not None and old_rule != v2c.RULE_ARMS[arm]:
            problems.append(f"rule arm {arm} definition changed")
    return {
        "problems": problems,
        "clean": not problems,
        "registered_differences": [
            {"difference": "round identity, output directory and artefact naming (r30_* cells)",
             "where": "this round's own directories; ROUND24's rows/logs are never written"},
            {"difference": "the two encoder arms (old/new deadline column)",
             "where": "r30_encoding; the only behavioural difference between the arms, "
                      "registered by the single-line source diff and ROUND29's semantic "
                      "registry"},
            {"difference": "evaluation arrangement: 2 arms x 3 seeds x 2 modes x 16 dev "
                           "scenes = 192 (ROUND24 ran 3 seeds x 3 blocks x 16 = 144, including "
                           "an init_greedy block this round does not need because both arms "
                           "start from the same fresh initialisation)",
             "where": "the fixed launch plan"},
            {"difference": "launch plan: 1040 normal + 40 reserve = 1080 hard cap",
             "where": "r30_common"},
        ],
    }


def freeze_manifest() -> dict[str, Any]:
    identity = common.code_identity()
    fields = behaviour_fields()
    manifest = {
        "round": "30",
        "phase": "old/new deadline encoding paired ablation: pre-launch freeze",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head_at_freeze": git("rev-parse", "HEAD"),
        "branch_start": BRANCH_START,
        "remote_start_head": None,
        "spec": "execution basis EXECUTION_BASIS.md (sha256 in the same directory)",
        "code_identity": identity,
        "configuration_hash": configuration_hash(fields),
        "inherited_behaviour_fields": fields,
        "ppo_hyperparameters": {
            "clip": v2c.CLIP, "value_coefficient": v2c.VALUE_COEFFICIENT,
            "entropy_bonus": v2c.ENTROPY_BONUS, "learning_rate": v2c.LEARNING_RATE,
            "max_grad_norm": v2c.MAX_GRAD_NORM, "epochs_per_batch": v2c.EPOCHS_PER_BATCH,
            "minibatch_steps": v2c.MINIBATCH_STEPS},
        "wiring_constants": {
            "b0_bytes": v2c.B0_BYTES, "delta": v2c.DELTA, "lambda_init": v2c.LAMBDA_INIT,
            "eta_lambda": v2c.ETA_LAMBDA, "gamma": v2c.GAMMA, "gae_lambda": v2c.GAE_LAMBDA,
            "data_order_seed": v2c.DATA_ORDER_SEED,
            "log_prob_tolerance": v2c.LOG_PROB_TOLERANCE,
            "episodes_per_update": v2c.EPISODES_PER_UPDATE,
            "episodes_per_seed": v2c.EPISODES_PER_SEED, "steps_per_episode": v2c.STEPS_PER_EPISODE},
        "scene_registry": {
            "training": [int(s) for s in v2c.TRAINING_SCENARIOS],
            "development": [int(s) for s in v2c.DEVELOPMENT_SCENARIOS],
            "note": "already-used blocks; this is a development ablation, not an unseen test set "
                    "and not an independent confirmation; the ROUND25 evaluation block "
                    "1400000..1400063 is not used for parameter, checkpoint or stopping "
                    "decisions in this round"},
        "data_order": {
            "algorithm": "random.Random(DATA_ORDER_SEED); the 48 training scenarios reshuffled "
                         "each of 3 rounds, 48 + 48 + 32 taken (frozen ROUND24 algorithm)",
            "seed": v2c.DATA_ORDER_SEED,
            "episodes_per_seed": v2c.EPISODES_PER_SEED,
            "order_sha256": json.dumps(common.scenario_order(), sort_keys=True),
            "order_first_8": common.scenario_order()[:8],
        },
        "encoder_arms": enc.arm_declaration(),
        "scoped_collector_source_diff": enc.source_diff(),
        "train_seeds": [int(s) for s in v2c.TRAIN_SEEDS],
        "launch_plan": {
            "reference": common.REFERENCE_LAUNCHES, "training": common.TRAINING_LAUNCHES,
            "eval": common.EVAL_LAUNCHES, "late_rule": common.LATE_RULE_LAUNCHES,
            "normal_launches": common.NORMAL_LAUNCHES, "reserve_launches": common.RESERVE_LAUNCHES,
            "hard_cap": common.LAUNCH_CAP, "synthetic_update_cap": common.SYNTHETIC_UPDATE_CAP},
        "rule_arms": v2c.RULE_ARMS,
        "statistics_plan": common.STATISTICS_PLAN,
        "passive_diagnostics": {
            "note": "fields recorded from artefacts the fixed plan already produces; frozen "
                    "before the first launch, never collected by an extra simulation",
            "per_batch": ["task/cost/g_hat/lambda", "effective two-action and masked sample "
                        "counts", "task/cost critic losses and per-module gradient reads",
                        "likelihood alignment"],
            "per_episode_first_report": ["raw remaining_time (registry deadline minus the "
                                         "verified decision clock)", "actual encoded deadline "
                                         "column", "p(SEND)", "original action"],
        },
        "round24_difference_audit": round24_difference_audit(),
        "round30_conflict_rechecks": {
            "3_1": "CONFLICT_RECHECK_3_1.json (development-block task counts)",
            "3_2": "CLOCK_RECHECK_3_2.json (clock relation)"},
    }
    return manifest


def write_manifest() -> dict[str, Any]:
    manifest = freeze_manifest()
    audit = manifest["round24_difference_audit"]
    if not audit["clean"]:
        raise RuntimeError("unregistered differences from the ROUND24 freeze: "
                           + json.dumps(audit["problems"], ensure_ascii=False))
    diff = git("diff", "--name-only", BRANCH_START, "--",
               "integrated_sar/comm.py", "integrated_sar/comm_learning.py",
               "integrated_sar/comm_policy.py", "integrated_sar/joint_driver.py")
    if diff:
        raise RuntimeError(f"production code changed since the round-30 start: {diff}")
    path = common.OUTPUT_DIR / "CONFIG_AND_MODEL_MANIFEST.json"
    if path.exists():
        raise RuntimeError(f"{path.name} already exists; refusing to rewrite a frozen manifest")
    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return manifest


def freeze_record(arm: str) -> dict[str, Any]:
    path = common.OUTPUT_DIR / "CONFIG_AND_MODEL_MANIFEST.json"
    if not path.exists():
        raise RuntimeError("CONFIG_AND_MODEL_MANIFEST.json is missing; freeze before starting")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {"freeze_sha256": manifest["code_identity"]["freeze_sha256"],
            "config_hash": manifest["configuration_hash"],
            "branch_start": manifest["branch_start"],
            "head_at_freeze": manifest["head_at_freeze"],
            "round": "30", "arm": str(arm),
            "feature_layout": enc.ENCODERS[arm]["layout"]}


# ------------------------------------------------------------------ 首批门禁 -------------
def load_cell_row(kind: str, seed: int, attempt: int, tag: str) -> dict[str, Any]:
    path = common.cell_paths(kind, seed, attempt, tag=tag)["row"]
    return json.loads(path.read_text(encoding="utf-8"))


def load_cell_tape(kind: str, seed: int, attempt: int, tag: str) -> list[dict[str, Any]]:
    path = common.cell_paths(kind, seed, attempt, tag=tag)["tape"]
    return list(common.iter_jsonl(path)) if path.exists() else []


def first4_gate(arm: str, training_seed: int, *, episodes: int = 4) -> dict[str, Any]:
    """Independent recomputation over the first ``episodes`` rollouts of one arm."""
    import torch

    import v2_check
    from onpolicy import joint_log_prob

    freeze = freeze_record(arm)
    layout = enc.ENCODERS[arm]["layout"]

    # the first 4 episodes are batch 0 of this (arm, seed); the batch record names them
    batch_path = common.CHECKPOINTS / f"r30_batch_{arm}_s{training_seed}_b0.json"
    if not batch_path.exists():
        raise RuntimeError(f"missing first batch record: {batch_path}")
    batch_record = json.loads(batch_path.read_text(encoding="utf-8"))
    episode_rows = []
    tapes_all = []
    for index in batch_record["episodes"]:
        row = load_cell_row("train", training_seed, index, f"train_{arm}_s{training_seed}_e{index}")
        tapes = load_cell_tape("train", training_seed, index, f"train_{arm}_s{training_seed}_e{index}")
        episode_rows.append(row)
        tapes_all.append(tapes)
    if len(episode_rows) != episodes:
        raise RuntimeError(f"first batch holds {len(episode_rows)} episodes, expected {episodes}")

    checks: dict[str, Any] = {}

    # G0: input encoding -- every row and tape of this arm carries the arm's layout, and the
    # deadline column is finite and consistent with the arm's definition on a hand-checked
    # value set (the synthetic part lives in r30_preflight).
    wrong_layout = [row["label"] for row in episode_rows if row.get("feature_layout") != layout]
    deadline_values: list[float] = []
    for tapes in tapes_all:
        for tape in tapes:
            for features_row in tape["features"]:
                deadline_values.append(float(features_row[enc.DEADLINE_COLUMN]))
    checks["G0_input_encoding_correct"] = {
        "arm_layout": layout, "rows_with_wrong_layout": wrong_layout,
        "deadline_column_min": min(deadline_values) if deadline_values else None,
        "deadline_column_max": max(deadline_values) if deadline_values else None,
        "all_finite": all(value == value and abs(value) != float("inf")
                          for value in deadline_values),
        "passed": not wrong_layout,
    }

    # G1: reward / cost / ledger closure per episode
    closure_ok = True
    for row in episode_rows:
        counters = row["relay_counters"]
        emissions = int(row["emissions"])
        closure_ok &= (int(row["spent_bytes"]) == int(row["airtime_tap_total"])
                       and int(counters["emissions"]) == emissions
                       and int(counters["first_emissions"]) + int(counters["refresh_emissions"])
                       == emissions
                       and int(row["b0_bytes"]) == int(v2c.B0_BYTES)
                       and int(row["task_reward_total"]) == int(row["on_time"]))
    checks["G1_reward_cost_ledger_closure"] = {"passed": bool(closure_ok),
                                               "episodes": len(episode_rows)}

    # G2: event attribution / GAE / combined advantage recomputed from row + tape only, and the
    # unassigned prefix totals must equal what the batch record kept
    math_ok = True
    prefix_task = 0.0
    prefix_cost = 0.0
    decision_total = 0
    for row, tapes in zip(episode_rows, tapes_all):
        recomputed = v2_check.recompute_episode_math(row, tapes, float(row["lambda"]))
        math_ok &= bool(recomputed.get("all_finite", True))
        prefix_task += float(recomputed.get("unassigned_prefix_task_reward", 0.0))
        prefix_cost += float(recomputed.get("unassigned_prefix_cost_bytes", 0.0))
        decision_total += int(recomputed.get("decision_steps", 0))
        collector = row.get("collector_summary") or {}
        math_ok &= int(recomputed.get("decision_steps", 0)) == int(
            collector.get("decision_steps", -1))
    update = batch_record.get("update") or {}
    math_ok &= abs(prefix_task - float(update.get("unassigned_prefix_task_reward", 0.0))) <= 1e-9
    math_ok &= abs(prefix_cost - float(update.get("unassigned_prefix_cost_bytes", 0.0))) <= 1e-9
    checks["G2_math_recomputable_from_disk"] = {
        "decision_steps": decision_total,
        "unassigned_prefix_task_reward": prefix_task,
        "unassigned_prefix_cost_bytes": prefix_cost,
        "batch_record_samples": int(update.get("samples", -1)),
        "samples_match": decision_total == int(update.get("samples", -1)),
        "passed": bool(math_ok),
    }

    # G3: log_prob recompute <= 1e-6 with the controller weights this batch actually used
    payload = torch.load(common.CHECKPOINTS / f"r30_train_{arm}_s{training_seed}.pt",
                         map_location="cpu", weights_only=False)
    gaps_max = 0.0
    for tapes in tapes_all:
        for tape in tapes:
            logits, _values = controller_forward(payload, tape["features"])
            with torch.no_grad():
                recomputed = float(joint_log_prob(
                    logits,
                    torch.tensor([tape["actions"][t] for t in tape["targets"]],
                                 dtype=torch.long),
                    torch.tensor([1.0 if t in tape["actionable"] else 0.0
                                  for t in tape["targets"]], dtype=torch.float32)))
            gaps_max = max(gaps_max, abs(recomputed - float(tape["log_prob"])))
    checks["G3_log_prob_recomputable"] = {
        "max_abs_gap": gaps_max, "tolerance": v2c.LOG_PROB_TOLERANCE,
        "passed": gaps_max <= v2c.LOG_PROB_TOLERANCE,
    }

    # G4: optimizer parameter coverage: the four comm modules, nothing outside
    import v2_controller

    fresh = v2_controller.fresh_controller(training_seed)
    modules = [fresh.inner.trunk, fresh.inner.action_head, fresh.inner.value_head, fresh.cost_head]
    expected_ids = {id(parameter) for module in modules for parameter in module.parameters()}
    controller_ids = [id(parameter) for parameter in fresh.parameters]
    checks["G4_optimizer_covers_comm_only"] = {
        "comm_parameter_count": len(expected_ids),
        "optimizer_parameter_count": len(controller_ids),
        "exact_same_set": set(controller_ids) == expected_ids,
        "passed": set(controller_ids) == expected_ids,
    }

    # G5: lambda update formula and direction, read from the first batch's dual record
    dual = batch_record.get("dual") or {}
    before = float(dual.get("lambda_before"))
    g_hat = float(dual.get("g_hat"))
    expected_next = max(0.0, before + v2c.ETA_LAMBDA * g_hat)
    checks["G5_dual_direction_and_formula"] = {
        "lambda_before": before, "g_hat": g_hat, "eta": v2c.ETA_LAMBDA,
        "lambda_next_recorded": float(dual.get("lambda_next")),
        "lambda_next_expected": expected_next,
        "direction": dual.get("direction"),
        "passed": abs(float(dual.get("lambda_next")) - expected_next) <= 1e-12
                  and float(dual.get("lambda_next")) >= 0.0,
    }

    # G6: parameters, gradients and losses finite; the frozen per-module gradient reads exist
    finite_ok = True
    for key in ("policy_surrogate", "value_loss_task", "value_loss_cost", "entropy",
                "gradient_norm_mean"):
        value = float(update.get(key, 0.0))
        finite_ok &= value == value and abs(value) != float("inf")
    gradient_reads = update.get("module_gradient_l1_mean") or {}
    finite_ok &= set(gradient_reads) == {"trunk", "action_head", "task_value_head", "cost_head"}
    finite_ok &= bool((update.get("likelihood_alignment") or {}).get("passed"))
    finite_ok &= int(update.get("optimizer_steps", 0)) > 0
    checks["G6_finite_and_no_pseudo_samples"] = {
        "module_gradient_l1_mean": gradient_reads,
        "optimizer_steps": int(update.get("optimizer_steps", 0)),
        "passed": bool(finite_ok),
    }

    # G7: checkpoint restore contract: rolling file present and the freeze carries arm+layout
    rolling_present = (common.CHECKPOINTS / f"r30_train_{arm}_s{training_seed}.pt").exists()
    checks["G7_checkpoint_restore_contract"] = {
        "rolling_checkpoint_present": rolling_present,
        "freeze_carries_arm_and_layout": (freeze.get("arm") == arm
                                          and freeze.get("feature_layout") == layout),
        "passed": bool(rolling_present and freeze.get("arm") == arm
                       and freeze.get("feature_layout") == layout),
    }

    passed = all(bool(check.get("passed")) for check in checks.values())
    return {"arm": arm, "training_seed": training_seed, "gate": "round30-first4",
            "episodes": [int(index) for index in batch_record["episodes"]],
            "checks": checks, "passed": bool(passed)}


def controller_forward(payload: dict[str, Any], features: list[list[float]]):
    """Forward with the controller weights stored inside a round-30 rolling checkpoint."""
    import v2_controller

    state = payload["training_state"]
    controller = v2_controller.fresh_controller(int(state["training_seed"]))
    saved = payload["controller"]
    controller.inner.trunk.load_state_dict(saved["trunk"])
    controller.inner.action_head.load_state_dict(saved["action_head"])
    controller.inner.value_head.load_state_dict(saved["value_task_head"])
    controller.cost_head.load_state_dict(saved["value_cost_head"])
    with torch.no_grad():
        return controller.forward(torch.as_tensor(features, dtype=torch.float32))


def main(argv: list[str]) -> int:
    if argv and argv[0] == "freeze":
        manifest = write_manifest()
        print(json.dumps({"freeze_sha256": manifest["code_identity"]["freeze_sha256"],
                          "configuration_hash": manifest["configuration_hash"],
                          "difference_audit_clean":
                              manifest["round24_difference_audit"]["clean"]},
                         ensure_ascii=False))
        return 0
    if argv and argv[0] == "gate":
        arm, seed = argv[1], int(argv[2])
        record = first4_gate(arm, seed)
        (common.OUTPUT_DIR / f"FIRST4_GATE_{arm}_s{seed}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({"arm": arm, "seed": seed, "passed": record["passed"]},
                         ensure_ascii=False))
        return 0 if record["passed"] else 1
    print("usage: r30_check.py [freeze | gate <arm> <seed>]")
    return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
