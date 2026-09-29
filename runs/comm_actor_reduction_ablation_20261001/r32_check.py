"""ROUND32 首批（正式训练的前 4 集）门禁：两臂各自独立重算，0 额外启动。

门禁输入是本轮批 0 的落盘件（batch 记录 + 4 条训练行 + 4 条 tape + ep0/更新前 checkpoint）。
重算函数复用 ROUND24 冻结的 `v2_check.recompute_episode_math`，不新写第二套数学。

除 ROUND30 同款的 G0–G7 外，本轮按执行依据 §六新增三项：
- G8 归约确实生效且 N/M 计数正确（从 tape 独立重数，并检查两臂各归约的定义式成立）；
- G9 批 0 更新前 checkpoint 契约（内容与身份可恢复、跨归约拒绝）；
- G10 两臂交叉算术比较（同一初始状态、同一批数据：未被触碰的量必须相等，被干预的量必须不同）。

表现差、出现全 HOLD、支出偏高或约束仍违背 **不是** 门禁失败项。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

import r32_common as common  # noqa: F401  (installs the shared sys.path before any archived-round import)
import r30_encoding as enc
import r32_run as runner
import r32_trainer as trainer
import v2_common as v2c
from r32_freeze import freeze_record
from r32_train import batch_record_path, pre_update_path, rolling_path

#: 两臂在 minibatch 0 上必须相等的字段（未被触碰的量）与必须不同的字段（被干预的量）
SHARED_AT_UPDATE_START = ("entropy_term", "value_loss_task", "value_loss_cost", "approx_kl",
                          "clip_fraction", "n_rows", "m_active_rows", "lambda", "ratio_min",
                          "ratio_max")
INTERVENED_AT_UPDATE_START = ("actor_surrogate_mean", "pre_clip_total_norm")


def cell_row(kind: str, seed: int, attempt: int, tag: str) -> dict[str, Any]:
    matches = sorted(common.ROWS.glob(f"r32_{tag}_s*_a{attempt}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one row for tag {tag} attempt {attempt}, "
                                f"found {len(matches)}: {[p.name for p in matches][:4]}")
    return common.read_json(matches[0])


def cell_tape(kind: str, seed: int, attempt: int, tag: str) -> list[dict[str, Any]]:
    matches = sorted(common.LOGS.glob(f"r32_{tag}_s*_a{attempt}_tape.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one tape for tag {tag} attempt {attempt}, "
                                f"found {len(matches)}: {[p.name for p in matches][:4]}")
    return list(common.iter_jsonl(matches[0]))


def recount_rows(tapes: list[dict[str, Any]]) -> tuple[int, int]:
    """N and M from the saved tape: rows, and rows with a non-empty saved free set."""
    return len(tapes), sum(1 for tape in tapes if tape["actionable"])


def first4_gate(reduction: str, training_seed: int, *, episodes: int = 4) -> dict[str, Any]:
    """Independent recomputation over the first ``episodes`` rollouts of one reduction arm."""
    import v2_check
    from onpolicy import joint_log_prob
    from v2_controller import fresh_controller, parameter_identity

    freeze = freeze_record(reduction)
    layout = enc.ENCODERS[runner.ENCODER_ARM]["layout"]
    batch_path = batch_record_path(reduction, training_seed, 0)
    if not batch_path.exists():
        raise RuntimeError(f"missing first batch record: {batch_path}")
    batch_record = common.read_json(batch_path)
    episode_rows: list[dict[str, Any]] = []
    tapes_all: list[list[dict[str, Any]]] = []
    for index in batch_record["episodes"]:
        tag = f"train_{reduction}_s{training_seed}_e{index}"
        episode_rows.append(cell_row("train", training_seed, int(index), tag))
        tapes_all.append(cell_tape("train", training_seed, int(index), tag))
    if len(episode_rows) != episodes:
        raise RuntimeError(f"first batch holds {len(episode_rows)} episodes, expected {episodes}")

    checks: dict[str, Any] = {}

    # G0 input encoding: both arms run the production old layout, and the deadline column is
    # the clip(remaining/50) encoding (bounded by 1.0), never the ROUND29/30 candidate column.
    wrong_layout = [row["label"] for row in episode_rows if row.get("feature_layout") != layout]
    wrong_encoder = [row["label"] for row in episode_rows
                     if row.get("encoder_arm") != runner.ENCODER_ARM]
    deadline_values = [float(features_row[enc.DEADLINE_COLUMN])
                       for tapes in tapes_all for tape in tapes for features_row in tape["features"]]
    all_finite = all(value == value and abs(value) != float("inf") for value in deadline_values)
    within_clip_range = bool(deadline_values
                             and max(abs(value) for value in deadline_values) <= 1.0)
    checks["G0_input_encoding_correct"] = {
        "arm_layout": layout, "rows_with_wrong_layout": wrong_layout,
        "rows_with_wrong_encoder_arm": wrong_encoder,
        "deadline_column_min": min(deadline_values) if deadline_values else None,
        "deadline_column_max": max(deadline_values) if deadline_values else None,
        "column_is_within_the_clip_range": within_clip_range,
        "all_finite": all_finite,
        "passed": bool(not wrong_layout and not wrong_encoder and all_finite
                       and within_clip_range),
    }

    # G1 reward / cost / ledger closure per episode
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

    # G2 event attribution / GAE / combined advantage recomputed from row + tape only; the
    # unassigned prefix totals must equal what the batch record kept. Fully masked rows are
    # counted as retained critic samples, not dropped.
    math_ok = True
    prefix_task = prefix_cost = 0.0
    decision_total = masked_rows_total = 0
    for row, tapes in zip(episode_rows, tapes_all):
        recomputed = v2_check.recompute_episode_math(row, tapes, float(row["lambda"]))
        math_ok &= bool(recomputed.get("all_finite", True))
        prefix_task += float(recomputed.get("unassigned_prefix_task_reward", 0.0))
        prefix_cost += float(recomputed.get("unassigned_prefix_cost_bytes", 0.0))
        decision_total += int(recomputed.get("decision_steps", 0))
        n_rows_local, m_rows_local = recount_rows(tapes)
        masked_rows_total += n_rows_local - m_rows_local
        collector = row.get("collector_summary") or {}
        math_ok &= int(recomputed.get("decision_steps", 0)) == int(
            collector.get("decision_steps", -1))
    update = batch_record.get("update") or {}
    math_ok &= abs(prefix_task - float(update.get("unassigned_prefix_task_reward", 0.0))) <= 1e-9
    math_ok &= abs(prefix_cost - float(update.get("unassigned_prefix_cost_bytes", 0.0))) <= 1e-9
    checks["G2_math_recomputable_from_disk"] = {
        "decision_steps": decision_total,
        "fully_masked_rows_retained_as_critic_samples": masked_rows_total,
        "unassigned_prefix_task_reward": prefix_task,
        "unassigned_prefix_cost_bytes": prefix_cost,
        "batch_record_samples": int(update.get("samples", -1)),
        "samples_match": decision_total == int(update.get("samples", -1)),
        "critic_keeps_masked_rows": masked_rows_total == 0 or decision_total > sum(
            sum(1 for tape in tapes if tape["actionable"]) for tapes in tapes_all),
        "passed": bool(math_ok and decision_total == int(update.get("samples", -1)))}

    # G3 log_prob recompute <= 1e-6 with the batch's PRE-update weights. For batch 0 those are
    # the registered fresh initialisation, available three independent ways: the seed, the ep0
    # milestone and the new pre-update checkpoint.
    initial = fresh_controller(training_seed)
    pre_checkpoint, _pre_optimizer, pre_state = trainer.load_checkpoint(
        pre_update_path(reduction, training_seed, 0), freeze=freeze, reduction=reduction)
    hashes = {"seed_reconstruction": parameter_identity(initial)["parameter_sha256"],
              "batch_record_before": batch_record["controller_parameters_before"]
              ["parameter_sha256"],
              "ep0_milestone": parameter_identity(
                  trainer.load_checkpoint(common.CHECKPOINTS / f"r32_train_{reduction}"
                                          f"_s{training_seed}_ep0.pt", freeze=freeze,
                                          reduction=reduction)[0])["parameter_sha256"],
              "pre_update_checkpoint": parameter_identity(pre_checkpoint)["parameter_sha256"]}
    gaps_max = 0.0
    with torch.no_grad():
        for tapes in tapes_all:
            for tape in tapes:
                logits, _values = pre_checkpoint.forward(
                    torch.as_tensor(tape["features"], dtype=torch.float32))
                recomputed = float(joint_log_prob(
                    logits,
                    torch.tensor([tape["actions"][t] for t in tape["targets"]], dtype=torch.long),
                    torch.tensor([1.0 if t in tape["actionable"] else 0.0
                                  for t in tape["targets"]], dtype=torch.float32)))
                gaps_max = max(gaps_max, abs(recomputed - float(tape["log_prob"])))
    checks["G3_log_prob_recomputable_pre_update"] = {
        "pre_update_weight_hash_sources": hashes,
        "all_four_sources_agree": len(set(hashes.values())) == 1,
        "max_abs_gap": gaps_max, "tolerance": v2c.LOG_PROB_TOLERANCE,
        "passed": bool(len(set(hashes.values())) == 1
                       and gaps_max <= v2c.LOG_PROB_TOLERANCE)}

    # G4 optimizer parameter coverage: the four comm modules, nothing outside
    fresh = fresh_controller(training_seed)
    modules = [fresh.inner.trunk, fresh.inner.action_head, fresh.inner.value_head, fresh.cost_head]
    expected = {id(parameter) for module in modules for parameter in module.parameters()}
    listed = [id(parameter) for parameter in fresh.parameters]
    checks["G4_optimizer_covers_comm_only"] = {
        "comm_parameter_count": len(expected), "listed_parameter_count": len(listed),
        "exact_same_set": set(listed) == expected, "passed": set(listed) == expected}

    # G5 dual formula, direction and use-within-batch timing
    dual = batch_record.get("dual") or {}
    before = float(dual.get("lambda_before"))
    g_hat = float(dual.get("g_hat"))
    expected_next = max(0.0, before + v2c.ETA_LAMBDA * g_hat)
    rows_lambda_ok = all(abs(float(row["lambda"]) - float(batch_record["lambda_at_rollout"]))
                         <= 1e-12 for row in episode_rows)
    checks["G5_dual_direction_and_formula"] = {
        "lambda_before": before, "g_hat": g_hat, "eta": v2c.ETA_LAMBDA,
        "lambda_next_recorded": float(dual.get("lambda_next")),
        "lambda_next_expected": expected_next, "direction": dual.get("direction"),
        "actor_used_the_pre_update_lambda": rows_lambda_ok,
        "passed": bool(abs(float(dual.get("lambda_next")) - expected_next) <= 1e-12
                       and float(dual.get("lambda_next")) >= 0.0 and rows_lambda_ok)}

    # G6 finite parameters/gradients/losses; the per-module gradient reads exist
    finite_ok = True
    for key in ("policy_surrogate", "value_loss_task", "value_loss_cost", "entropy",
                "gradient_norm_mean"):
        value = float(update.get(key, 0.0))
        finite_ok &= value == value and abs(value) != float("inf")
    gradient_reads = update.get("module_gradient_l1_mean") or {}
    finite_ok &= set(gradient_reads) == {"trunk", "action_head", "task_value_head", "cost_head"}
    finite_ok &= bool((update.get("likelihood_alignment") or {}).get("passed"))
    finite_ok &= int(update.get("optimizer_steps", 0)) > 0
    finite_ok &= int(batch_record.get("update_read_summary", {}).get("non_finite_minibatches", -1)
                     ) == 0
    checks["G6_finite_and_no_pseudo_samples"] = {
        "module_gradient_l1_mean": gradient_reads,
        "optimizer_steps": int(update.get("optimizer_steps", 0)),
        "non_finite_minibatches": int(batch_record.get("update_read_summary", {})
                                      .get("non_finite_minibatches", -1)),
        "passed": bool(finite_ok)}

    # G7 checkpoint restore contract: the freeze carries reduction + encoder + layout
    rolling = rolling_path(reduction, training_seed)
    restored_ok = False
    if rolling.exists():
        _controller, _optimizer, restored_state = trainer.load_checkpoint(
            rolling, freeze=freeze, reduction=reduction)
        restored_ok = (restored_state["reduction_mode"] == reduction
                       and int(restored_state["episodes_done"]) == episodes)
    cross_refused = False
    other = next(name for name in common.ARM_ORDER if name != reduction)
    try:
        trainer.load_checkpoint(rolling, freeze=freeze, reduction=other)
    except RuntimeError:
        cross_refused = True
    checks["G7_checkpoint_restore_contract"] = {
        "rolling_checkpoint_present": rolling.exists(),
        "restore_ok": restored_ok,
        "cross_reduction_resume_refused": cross_refused,
        "freeze_carries_reduction_encoder_and_layout": bool(
            freeze.get("reduction") == reduction and freeze.get("encoder_arm") == "old"
            and freeze.get("feature_layout") == layout),
        "passed": bool(rolling.exists() and restored_ok and cross_refused
                       and freeze.get("reduction") == reduction)}

    # G8 the reduction really took effect, and N/M are counted from the saved tapes
    reads = update.get("minibatch_reads") or []
    recount = [recount_rows(tapes) for tapes in tapes_all]
    total_n = sum(pair[0] for pair in recount)
    total_m = sum(pair[1] for pair in recount)
    # The reads are PER MINIBATCH and the batch runs EPOCHS_PER_BATCH epochs over the same
    # sample list, so the batch-level sums are epochs x rows: comparing them against the raw
    # tape counts was the gate's own arithmetic mistake, not a training defect. Compare
    # per epoch instead, and require every epoch to carry the same counts.
    epochs = int(common.v2c.EPOCHS_PER_BATCH)
    per_epoch: dict[int, tuple[int, int]] = {}
    for read in reads:
        e_key = int(read["epoch"])
        n_e, m_e = per_epoch.get(e_key, (0, 0))
        per_epoch[e_key] = (n_e + int(read["n_rows"]), m_e + int(read["m_active_rows"]))
    read_n = sum(int(read["n_rows"]) for read in reads)
    read_m = sum(int(read["m_active_rows"]) for read in reads)
    epoch_sums_equal_rows = (len(per_epoch) == epochs
                             and all(n == total_n for n, _m in per_epoch.values())
                             and all(m == total_m for _n, m in per_epoch.values()))
    definition_ok = True
    for read in reads:
        if read["reduction"] == "ALL_ROWS":
            definition_ok &= abs(read["actor_surrogate_mean"]
                                 - read["actor_surrogate_mean_all_rows_reference"]) < 1e-6
        else:
            if read["m_active_rows"]:
                definition_ok &= abs(read["actor_surrogate_mean"]
                                     - read["actor_surrogate_mean_active_only_reference"]) < 1e-6
            else:
                definition_ok &= read["actor_surrogate_mean"] == 0.0
    checks["G8_reduction_effective_and_counted"] = {
        "reduction": reduction,
        "tapes_recount_N": total_n, "tapes_recount_M": total_m,
        "recorded_N_sum_over_minibatches": read_n,
        "recorded_M_sum_over_minibatches": read_m,
        "epochs_per_batch": epochs,
        "per_epoch_N_M": {str(key): list(value) for key, value in sorted(per_epoch.items())},
        "every_epoch_carries_exactly_the_tape_counts": epoch_sums_equal_rows,
        "counts_match": bool(read_n == epochs * total_n and read_m == epochs * total_m
                             and epoch_sums_equal_rows),
        "reduction_can_differ_this_batch": 0 < total_m < total_n,
        "minibatches": len(reads),
        "each_arms_own_definition_holds": bool(definition_ok),
        "passed": bool(read_n == epochs * total_n and read_m == epochs * total_m
                       and epoch_sums_equal_rows and definition_ok and reads)}

    # G9 the batch-0 pre-update checkpoint is a complete, restorable update-start record
    payload = torch.load(pre_update_path(reduction, training_seed, 0), map_location="cpu",
                         weights_only=False)
    checks["G9_pre_update_checkpoint_contract"] = {
        "path": str(pre_update_path(reduction, training_seed, 0).relative_to(common.ROOT)),
        "contents": sorted(key for key in payload),
        "has_model_optimizer_rng_lambda": bool({"controller", "optimizer", "rng",
                                                "training_state"} <= set(payload)),
        "lambda_matches_batch_record": abs(float(payload["training_state"]["lambda"])
                                           - float(batch_record["lambda_at_rollout"])) <= 1e-12,
        "reduction_mode_recorded": payload["reduction_mode"] == reduction,
        "feature_layout_recorded": payload["freeze"]["feature_layout"] == layout,
        "episodes_done_at_this_start": int(payload["training_state"]["episodes_done"]),
        "parameter_sha_equals_batch_record_before": parameter_identity(pre_checkpoint)
        ["parameter_sha256"] == batch_record["controller_parameters_before"]["parameter_sha256"],
        "passed": bool({"controller", "optimizer", "rng", "training_state"} <= set(payload)
                       and payload["reduction_mode"] == reduction
                       and parameter_identity(pre_checkpoint)["parameter_sha256"]
                       == batch_record["controller_parameters_before"]["parameter_sha256"])}

    passed = all(bool(check.get("passed")) for check in checks.values())
    return {"reduction": reduction, "arm": reduction, "training_seed": training_seed,
            "encoder_arm": runner.ENCODER_ARM, "gate": "round32-first4",
            "episodes": [int(index) for index in batch_record["episodes"]],
            "scenarios": [int(value) for value in batch_record["scenarios"]],
            "checks": checks, "passed": bool(passed)}


def cross_arm_gate(training_seed: int) -> dict[str, Any]:
    """Same start, same batch: untouched fields equal, intervened fields different."""
    records = {reduction: common.read_json(batch_record_path(reduction, training_seed, 0))
               for reduction in common.ARM_ORDER}
    reads = {reduction: (record.get("update") or {}).get("minibatch_reads") or []
             for reduction, record in records.items()}
    if not reads["ALL_ROWS"] or not reads["ACTIVE_ROWS"]:
        return {"passed": False, "problem": "one arm has no minibatch reads"}
    first_all, first_active = reads["ALL_ROWS"][0], reads["ACTIVE_ROWS"][0]
    shared = {key: first_all[key] == first_active[key] for key in SHARED_AT_UPDATE_START}
    intervened = {key: first_all[key] != first_active[key]
                  for key in INTERVENED_AT_UPDATE_START}
    same_batch = (records["ALL_ROWS"]["episodes"] == records["ACTIVE_ROWS"]["episodes"]
                  and records["ALL_ROWS"]["scenarios"] == records["ACTIVE_ROWS"]["scenarios"])
    same_start = (records["ALL_ROWS"]["controller_parameters_before"]["parameter_sha256"]
                  == records["ACTIVE_ROWS"]["controller_parameters_before"]["parameter_sha256"])
    same_data = all(json.dumps(record["constraint"]) == json.dumps(records["ALL_ROWS"]["constraint"])
                    for record in records.values())
    return {
        "training_seed": training_seed,
        "both_arms_visited_the_same_scenarios_in_the_same_order": bool(same_batch),
        "both_arms_started_from_the_same_weights": bool(same_start),
        "both_arms_report_the_same_paired_constraint_block": bool(same_data),
        "untouched_fields_equal_at_update_start": shared,
        "intervened_fields_differ_at_update_start": intervened,
        "n_rows": first_all["n_rows"], "m_active_rows": first_all["m_active_rows"],
        "expected_amplification_N_over_M": (first_all["n_rows"]
                                            / max(first_all["m_active_rows"], 1)),
        "actor_surrogate_mean": {reduction: reads[reduction][0]["actor_surrogate_mean"]
                                 for reduction in common.ARM_ORDER},
        "pre_clip_total_norm": {reduction: reads[reduction][0]["pre_clip_total_norm"]
                                for reduction in common.ARM_ORDER},
        "reading": "the amplification is a statement about the ACTOR SURROGATE GRADIENT; the "
                   "observed pre-clip total norm also carries the two critics and the entropy "
                   "term, so N/M is not a parameter-step multiplier",
        "passed": bool(same_batch and same_start and all(shared.values())
                       and all(intervened.values()))}


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: r32_check.py [gate <reduction> <seed> | cross <seed>]")
        return 2
    if argv[0] == "gate":
        reduction, seed = argv[1], int(argv[2])
        record = first4_gate(reduction, seed)
        out = common.OUTPUT_DIR / f"FIRST4_GATE_{reduction}_s{seed}.json"
        out.write_text(json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(json.dumps({"reduction": reduction, "seed": seed, "passed": record["passed"],
                          "failed": [key for key, value in record["checks"].items()
                                     if not value.get("passed")]}, ensure_ascii=False))
        return 0 if record["passed"] else 1
    if argv[0] == "cross":
        seed = int(argv[1])
        record = cross_arm_gate(seed)
        out = common.OUTPUT_DIR / f"FIRST4_CROSS_s{seed}.json"
        out.write_text(json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=1))
        return 0 if record["passed"] else 1
    return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
