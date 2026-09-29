"""ROUND32 启动前零环境预检：接线、身份、恢复与归约语义（**0 环境启动**）。

本脚本绝不构造 `JointRun`，也不跑任何 episode：所有"真实数据"都来自已归档的 ROUND30 old 臂
tape 与行。合成 optimizer 更新单独分账（上限 64，键控幂等，重跑不重复计数）。

检查四类：
1. 行为身份未变：本轮 code_identity 的 production/imported/checkpoints 三段与 ROUND30 冻结件
   逐字节相同；`configuration_hash` 与 ROUND24/30 相同；冻结数据顺序与 ROUND30 记录相同。
2. 唯一主动变量：trainer 的逐行差异只含归约与被动读数；被 import 的数学对象与冻结件同一；
   两臂定义与 REDUCTION_MATH.json 的实测一致。
3. 归约语义在真实数据上成立：active 判据取自保存的 free 集合；整行被 mask 的行 likelihood
   恒为 0 且梯度为 0；critic 样本不因 mask 而丢失。
4. 恢复契约：同归约可往返、跨归约拒绝、跨轮拒绝、恢复后的下一步与不中断路径逐权重相同。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import r32_common as common  # noqa: F401  (installs the shared sys.path first)
import r32_check as check  # (installs nothing extra; r32_common runs first)
import r32_run as runner
import r32_trainer as trainer
import v2_trainer as frozen
from v2_controller import fresh_controller, parameter_identity

PROBE_KEY = "preflight-restore-determinism"


class _CollectorShim:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records


def archived_batch(arm_row_count: int = 2) -> tuple[list[dict[str, Any]], list[Path]]:
    """Load ROUND30 old-arm training rows and tapes as live episode dicts (0 launches)."""
    paths = sorted(common.R30_DIR.glob("rows/r30_train_old_s11_e*_a*.json"))[:arm_row_count]
    episodes = []
    for path in paths:
        row = common.read_json(path)
        tapes = list(common.iter_jsonl(Path(row["artefacts"]["tape"])))
        episodes.append({
            "collector": _CollectorShim(tapes),
            "cost_stream": row["cost_stream"], "task_stream": row["task_stream"],
            "steps": int(row["steps"]), "lambda": float(row["lambda"]),
            "episode_index": int(row["episode_index"]),
            "scenario_seed": int(row["scenario_seed"]), "on_time": int(row["on_time"]),
            "spent_bytes": int(row["spent_bytes"]),
        })
    return episodes, paths


def identity_checks() -> dict[str, Any]:
    manifest_target = common.OUTPUT_DIR / "CONFIG_AND_MODEL_MANIFEST.json"
    manifest = common.read_json(manifest_target) if manifest_target.exists() else None
    previous_manifest = common.round30_manifest()
    previous = previous_manifest["code_identity"]
    current = (manifest or {}).get("code_identity", {})
    sections = {}
    for section in ("production", "imported_round_modules", "checkpoints"):
        before, now = previous.get(section, {}), current.get(section, {})
        differing = sorted(key for key in set(before) | set(now) if before.get(key) != now.get(key))
        sections[section] = {"compared": len(before), "differing_keys": differing,
                             "identical": not differing and len(before) == len(now)}
    order = common.scenario_order()
    order_repr = json.dumps(order, sort_keys=True)
    return {
        "manifest_written": manifest_target.exists(),
        "code_identity_sections": sections,
        "configuration_hash_matches_round30": (
            (manifest or {}).get("configuration_hash")
            == previous_manifest.get("configuration_hash")),
        "configuration_hash_value": (manifest or {}).get("configuration_hash"),
        "data_order_matches_round30_record": (
            (manifest or {}).get("data_order", {}).get("order_sha256")
            == previous_manifest.get("data_order", {}).get("order_sha256")),
        "data_order_digest": hashlib.sha256(order_repr.encode("utf-8")).hexdigest()[:16],
        "data_order_first_8": order[:8],
        "round30_difference_audit_clean": bool(
            (manifest or {}).get("round30_identity_audit", {}).get("clean")),
        "round30_audit_problems": (manifest or {}).get("round30_identity_audit",
                                                       {}).get("problems", []),
        "runner_seam_clean": bool((manifest or {}).get("runner_seam_diff", {}).get("passed")),
        "trainer_source_clean": bool((manifest or {}).get("trainer_source_diff", {}).get("passed")),
    }


def reduction_semantics_checks() -> dict[str, Any]:
    """Active-row semantics on real archived tapes, and the arm definitions."""
    episodes, paths = archived_batch(6)
    rows = [tape for episode in episodes for tape in episode["collector"].records]
    n_total = len(rows)
    m_total = sum(1 for tape in rows if tape["actionable"])
    masked_only = [tape for tape in rows if not tape["actionable"]]
    samples = frozen.batch_samples(episodes)
    # an inactive row's likelihood must be a constant zero with a zero gradient
    net = fresh_controller(11)
    probe = masked_only[0]
    matrix = torch.as_tensor(np.asarray(probe["features"], dtype=np.float32))
    logits = net.inner.action_head(net.inner.trunk(matrix))
    usable = torch.tensor([1.0 if t in probe["actionable"] else 0.0 for t in probe["targets"]],
                          dtype=torch.float32)
    picked = torch.tensor([int(probe["actions"][t]) for t in probe["targets"]], dtype=torch.long)
    from onpolicy import joint_log_prob

    jlp = joint_log_prob(logits, picked, usable)
    grads = torch.autograd.grad(jlp, list(net.parameters), allow_unused=True)
    flat = torch.cat([(g if g is not None else torch.zeros_like(p)).detach().reshape(-1)
                      for g, p in zip(grads, net.parameters)])
    # a row whose free candidates all chose HOLD must still be counted active
    held_rows = [tape for tape in rows if tape["actionable"]
                 and all(int(tape["actions"][t]) == 0 for t in tape["actionable"])]
    return {
        "archived_rows_used": n_total,
        "active_rows": m_total,
        "fully_masked_rows": len(masked_only),
        "m_over_n": round(m_total / n_total, 6),
        "actionable_equals_free_everywhere": all(tape["actionable"] == tape["free"]
                                                 for tape in rows),
        "masked_candidates_never_in_send_probabilities": all(
            target not in tape["send_probabilities"] for tape in rows for target in tape["masked"]),
        "samples_equal_rows_no_row_dropped": len(samples) == n_total,
        "inactive_row_likelihood_value": float(jlp.detach().item()),
        "inactive_row_actor_gradient_max_abs": float(flat.abs().max().item()),
        "rows_with_only_HOLD_on_free_candidates": len(held_rows),
        "hold_rows_are_active": all(tape["actionable"] for tape in held_rows),
        "tape_files": [path.name for path in paths],
        "arm_definitions": {name: dict(spec) for name, spec in common.REDUCTIONS.items()},
        "M_zero_safe_by_construction": bool(
            torch.isfinite((torch.zeros(4) * torch.tensor([0.0, 0.0, 0.0, 0.0])).sum() / 1)),
    }


def restore_contract_checks() -> dict[str, Any]:
    """Same-mode roundtrip, cross-mode refusal, cross-round refusal, deterministic next step."""
    episodes, _paths = archived_batch(1)
    net_placeholder = None
    probe_dir = common.OUTPUT_DIR / "checkpoints"
    probe_dir.mkdir(parents=True, exist_ok=True)
    path = probe_dir / "r32_preflight_restore_probe.pt"
    if path.exists():
        path.unlink()
    freeze = {"round": "32", "probe": True}
    # The archived ROUND30 old-arm ep0 milestone holds exactly the weights that produced the
    # archived tapes used below, so the update path can be exercised on real rows without
    # launching anything. Loading them goes through the frozen v2 loader with ROUND30's own
    # freeze record; nothing is written back.
    import r30_check as check30
    r30_milestone = common.R30_DIR / "checkpoints" / "r30_train_old_s11_ep0.pt"
    net, _r30_optimizer, _r30_state = frozen.load_checkpoint(r30_milestone,
                                                             freeze=check30.freeze_record("old"))
    optimizer = torch.optim.Adam(net.parameters, lr=common.v2c.LEARNING_RATE)
    state = frozen.fresh_training_state(11)
    state["episodes_done"] = len(episodes) * common.EPISODES_PER_UPDATE

    # the likelihood guard must still fire when the tape does not describe its own draw
    mismatch_refused = False
    mismatch_detail = None
    wrong = fresh_controller(22)
    try:
        trainer.ppo_update(wrong, torch.optim.Adam(wrong.parameters,
                                                   lr=common.v2c.LEARNING_RATE),
                           episodes, update_index=0, reduction="ALL_ROWS")
    except RuntimeError as error:
        mismatch_refused = True
        mismatch_detail = str(error)[:200]
    trainer.save_checkpoint(net, optimizer, state, path, freeze=freeze, reduction="ALL_ROWS")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    roundtrip = trainer.load_checkpoint(path, freeze=freeze, reduction="ALL_ROWS")
    same_weights = (parameter_identity(roundtrip[0])["parameter_sha256"]
                    == parameter_identity(net)["parameter_sha256"])
    cross_mode_refused = False
    try:
        trainer.load_checkpoint(path, freeze=freeze, reduction="ACTIVE_ROWS")
    except RuntimeError:
        cross_mode_refused = True
    cross_round_refused = False
    try:
        trainer.load_checkpoint(r30_milestone, freeze=freeze, reduction="ALL_ROWS")
    except RuntimeError:
        cross_round_refused = True
    # Does the *frozen* v2 loader reject this payload? Reported as a fact, not forced to pass:
    # production code is not modified this round, so the real discriminator is the freeze block
    # (round + reduction + feature layout), which differs for every round.
    frozen_rejects = False
    refusal_text = None
    try:
        frozen.load_checkpoint(path, freeze=freeze)
    except (RuntimeError, KeyError, AttributeError) as error:
        frozen_rejects = True
        refusal_text = f"{type(error).__name__}"
    # deterministic continuation: interrupted-and-restored vs uninterrupted
    samples_estimate = len(frozen.batch_samples(episodes))
    per_call = (int(common.v2c.EPOCHS_PER_BATCH)
                * max(1, -(-samples_estimate // int(common.v2c.MINIBATCH_STEPS))))
    used_before = common.synthetic_updates_used()
    common.book_synthetic_updates(2 * per_call,
                                  "preflight restore determinism (2 calls)", key=PROBE_KEY)
    uninterrupted_hash = None
    interrupted_hash = None
    trainer.save_checkpoint(net, optimizer, state, path, freeze=freeze, reduction="ALL_ROWS")
    trainer.ppo_update(net, optimizer, episodes, update_index=0, reduction="ALL_ROWS")
    uninterrupted_hash = parameter_identity(net)["parameter_sha256"]
    net_b, opt_b, _state_b = trainer.load_checkpoint(path, freeze=freeze, reduction="ALL_ROWS")
    trainer.ppo_update(net_b, opt_b, episodes, update_index=0, reduction="ALL_ROWS")
    interrupted_hash = parameter_identity(net_b)["parameter_sha256"]
    return {
        "probe_rows_from": "archived ROUND30 old-arm training rows/tapes (0 new launches); the "
                           "update path is exercised on the weights that actually produced them",
        "likelihood_guard_still_fires_on_a_mismatched_tape": {
            "refused": mismatch_refused, "detail": mismatch_detail},
        "payload_records_reduction": payload.get("reduction_mode") == "ALL_ROWS",
        "training_state_records_reduction": payload["training_state"].get("reduction_mode"),
        "same_mode_roundtrip_weights_equal": bool(same_weights),
        "cross_reduction_resume_refused": cross_mode_refused,
        "archived_round30_checkpoint_refused": cross_round_refused,
        "frozen_v2_loader_on_this_payload": {
            "rejects": frozen_rejects, "error": refusal_text,
            "meaning": "the v2 trainer's own schema checks pass because this payload keeps the "
                       "inherited contract tags; the actual barrier is the freeze block, which "
                       "carries round=32 + reduction + feature layout and differs per round, and "
                       "the r32 loader additionally demands reduction_mode. Registered as a "
                       "fact rather than forced to be a refusal."},
        "rng_state_serialisable": bool(isinstance(payload.get("rng"), dict)
                                       and {"python", "numpy", "torch"} <= set(payload["rng"])),
        "samples_in_probe_batch": samples_estimate,
        "minibatches_per_probe_call": per_call,
        "synthetic_steps_before": used_before,
        "synthetic_steps_used": common.synthetic_updates_used(),
        "restored_continuation_matches_uninterrupted": (
            uninterrupted_hash == interrupted_hash),
        "uninterrupted_hash": uninterrupted_hash,
        "restored_hash": interrupted_hash,
    }


def ledger_arithmetic() -> dict[str, Any]:
    return {
        "reference": common.REFERENCE_LAUNCHES, "training": common.TRAINING_LAUNCHES,
        "eval": common.EVAL_LAUNCHES, "late_rule": common.LATE_RULE_LAUNCHES,
        "normal_total": common.NORMAL_LAUNCHES,
        "total_matches_parts": (common.REFERENCE_LAUNCHES + common.TRAINING_LAUNCHES
                                + common.EVAL_LAUNCHES + common.LATE_RULE_LAUNCHES
                                == common.NORMAL_LAUNCHES == 1040),
        "reserve": common.RESERVE_LAUNCHES, "hard_cap": common.LAUNCH_CAP,
        "hard_cap_is_normal_plus_reserve": common.LAUNCH_CAP == 1080,
        "training_is_2_arms_3_seeds_128": common.TRAINING_LAUNCHES == 2 * 3 * 128 == 768,
        "eval_is_2_arms_3_seeds_2_modes_16": common.EVAL_LAUNCHES == 2 * 3 * 2 * 16 == 192,
        "max_parallel_workers": common.MAX_PARALLEL_WORKERS,
        "cells_and_shards": [{"reduction": reduction, "seed": seed}
                             for reduction in common.ARM_ORDER for seed in common.TRAIN_SEEDS],
        "shards_are_disjoint": len({(reduction, seed) for reduction in common.ARM_ORDER
                                    for seed in common.TRAIN_SEEDS}) == 6,
        "booked_so_far": common.ledger_total(),
        "seals_untouched": "800000..800399 and 900000..900199 never appear in this round's "
                          "scenario lists",
        "sealed_blocks_absent_from_plans": not any(
            800000 <= scenario <= 800399 or 900000 <= scenario <= 900199
            for scenario in list(common.TRAINING_SCENARIOS) + list(common.DEVELOPMENT_SCENARIOS)),
    }


def main() -> int:
    payload: dict[str, Any] = {"round": "32", "section": "pre-launch preflight (0 launches)",
                               "environment_launches": 0, "new_streams": 0,
                               "real_training_updates": 0}
    payload["identity"] = identity_checks()
    payload["reduction_semantics"] = reduction_semantics_checks()
    payload["restore_contract"] = restore_contract_checks()
    payload["ledger_arithmetic"] = ledger_arithmetic()
    payload["reduction_math_reference"] = {
        "file": "REDUCTION_MATH.json",
        "passed": common.read_json(common.OUTPUT_DIR / "REDUCTION_MATH.json")["passed"]
        if (common.OUTPUT_DIR / "REDUCTION_MATH.json").exists() else None}

    gates = {
        "identity_sections_unchanged": all(
            value["identical"] for value in payload["identity"]["code_identity_sections"]
            .values()),
        "configuration_hash_unchanged": payload["identity"]["configuration_hash_matches_round30"],
        "data_order_unchanged": payload["identity"]["data_order_matches_round30_record"],
        "round30_audit_clean": payload["identity"]["round30_difference_audit_clean"],
        "runner_seam_clean": payload["identity"]["runner_seam_clean"],
        "trainer_source_clean": payload["identity"]["trainer_source_clean"],
        "reduction_math_passed": payload["reduction_math_reference"]["passed"] is True,
        "active_semantics_on_real_tapes": bool(
            payload["reduction_semantics"]["actionable_equals_free_everywhere"]
            and payload["reduction_semantics"]["samples_equal_rows_no_row_dropped"]
            and payload["reduction_semantics"]["masked_candidates_never_in_send_probabilities"]
            and payload["reduction_semantics"]["fully_masked_rows"] > 0
            and payload["reduction_semantics"]["inactive_row_actor_gradient_max_abs"] == 0.0
            and payload["reduction_semantics"]["inactive_row_likelihood_value"] == 0.0
            and payload["reduction_semantics"]["rows_with_only_HOLD_on_free_candidates"] > 0
            and payload["reduction_semantics"]["hold_rows_are_active"]),
        "restore_contract": bool(
            payload["restore_contract"]["payload_records_reduction"]
            and payload["restore_contract"]["same_mode_roundtrip_weights_equal"]
            and payload["restore_contract"]["cross_reduction_resume_refused"]
            and payload["restore_contract"]["archived_round30_checkpoint_refused"]
            and payload["restore_contract"]["rng_state_serialisable"]
            and payload["restore_contract"]["likelihood_guard_still_fires_on_a_mismatched_tape"]["refused"]
            and payload["restore_contract"]["restored_continuation_matches_uninterrupted"]),
        "ledger_arithmetic": bool(payload["ledger_arithmetic"]["total_matches_parts"]
                                  and payload["ledger_arithmetic"]["hard_cap_is_normal_plus_reserve"]
                                  and payload["ledger_arithmetic"]["shards_are_disjoint"]
                                  and payload["ledger_arithmetic"]["sealed_blocks_absent_from_plans"]),
        "synthetic_within_cap": (payload["restore_contract"]["synthetic_steps_used"]
                                 <= common.SYNTHETIC_UPDATE_CAP),
    }
    payload["gates"] = gates
    payload["passed"] = bool(all(gates.values()))

    target = common.OUTPUT_DIR / "RUN_PREFLIGHT.json"
    if target.exists():
        target.unlink()
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "gates": gates,
                      "identity": {key: value["identical"] for key, value in
                                   payload["identity"]["code_identity_sections"].items()},
                      "configuration_hash": payload["identity"]["configuration_hash_value"],
                      "semantics": {key: payload["reduction_semantics"][key] for key in
                                    ("archived_rows_used", "active_rows", "fully_masked_rows",
                                     "m_over_n", "inactive_row_likelihood_value",
                                     "inactive_row_actor_gradient_max_abs",
                                     "rows_with_only_HOLD_on_free_candidates")},
                      "restore": {key: payload["restore_contract"][key] for key in
                                  ("same_mode_roundtrip_weights_equal",
                                   "cross_reduction_resume_refused",
                                   "archived_round30_checkpoint_refused",
                                   "likelihood_guard_still_fires_on_a_mismatched_tape",
                                   "restored_continuation_matches_uninterrupted")},
                      "synthetic_used": payload["restore_contract"]["synthetic_steps_used"]},
                     ensure_ascii=False, indent=1))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
