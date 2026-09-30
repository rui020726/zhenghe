"""ROUND35 任务 6：checkpoint 内容层补验（0 环境启动、0 个旧文件改写）。

ROUND34 在同一台机器上因为缺 PyTorch，把 18 个 `.pt` 的"参数内容 / 模块摘要 / payload 元数据 /
forward / native likelihood"四层全部登记为**不可用或未运行**。那个环境豁免本轮不再成立，所以这里
把每一层真的做一遍，并**逐层**与 ROUND34 的"待复算声明"比对：

L1 文件字节哈希：本轮重算 vs ROUND34 观测 vs index 声明（三方）。
L2 payload 元数据：真实反序列化后的 schema/trainer_tag/loss/gae/optimizer/reward_timing_schema、
   freeze、training_state 自述（episodes/updates/optimizer_steps/lambda）。
L3 whole-controller 参数摘要：从反序列化张量重算 parameter_sha256，与 index 声明、批记录
   `controller_parameters_before` 比对。
L4 四模块 digest：用真实 `v2_train.module_digests` 重算，与 ROUND34 登记的 `module_digest_claims`
   比对。不同命名空间的哈希不互相替代（ROUND34 的 hash_namespace_rules 同样适用本轮）。
L5 forward + native likelihood：用该点自己的更新前权重对本批 tape 实跑前向，重算联合对数似然，
   与 ROUND32 批记录里归档的 `likelihood_alignment.max_abs_gap` 比对（侧车可比 → 比对）。

另外对本轮新产的 2 个 checkpoint 做同样的 L2/L3/L4 与一次真实 forward/likelihood。
全程只读：每个旧文件在处理前后各算一次字节哈希，证明未被改写。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

import r35_common as smoke
import r35_restore_boundary
import v2_run
import v2_train
import v2_trainer
from onpolicy import joint_log_prob
from v2_controller import parameter_identity

RESULTS: dict[str, Any] = {}
PROBLEMS: list[str] = []


def gate(name: str, payload: dict[str, Any]) -> None:
    payload["passed"] = bool(payload["passed"])
    RESULTS[name] = payload
    if not payload["passed"]:
        PROBLEMS.append(f"{name}: {payload.get('failure', 'layer check failed')}")


def tape_rows(directory: Path, stem: str) -> list[dict[str, Any]]:
    path = directory / f"{stem}_tape.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def cell_stem(tape_path: Path) -> str:
    """`v2_train_s11_e0_s1300027_a0_tape` -> `v2_train_s11_e0_s1300027_a0`."""
    name = tape_path.stem
    return name[:-len("_tape")] if name.endswith("_tape") else name


def likelihood_gap(controller: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A real forward pass over the batch's own tapes, recomputing each row's joint log-prob."""
    gaps = []
    with torch.no_grad():
        for row in rows:
            logits, _values = controller.forward(
                torch.as_tensor(row["features"], dtype=torch.float32))
            recomputed = float(joint_log_prob(
                logits,
                torch.tensor([row["actions"][t] for t in row["targets"]], dtype=torch.long),
                torch.tensor([1.0 if t in row["actionable"] else 0.0 for t in row["targets"]],
                             dtype=torch.float32)))
            gaps.append(abs(recomputed - float(row["log_prob"])))
    return {"tapes": len(gaps), "max_abs_gap": max(gaps) if gaps else 0.0,
            "tolerance": smoke.v2_common.LOG_PROB_TOLERANCE,
            "within_tolerance": (not gaps) or max(gaps) <= smoke.v2_common.LOG_PROB_TOLERANCE}


def tensor_layer(payload: dict[str, Any]) -> dict[str, Any]:
    controller = v2_trainer._restore_controller(payload, payload["training_state"])
    identity = parameter_identity(controller)
    saved = payload["controller"]
    finite = all(bool(torch.is_tensor(value) and torch.isfinite(value).all())
                 for module in saved.values() for value in
                 (module.values() if isinstance(module, dict) else [module]))
    return {"parameter_sha256_recomputed": identity["parameter_sha256"],
            "parameter_count_recomputed": identity["parameter_count"],
            "shapes_recomputed": identity["shapes"],
            "module_digests_recomputed": v2_train.module_digests(controller),
            "all_saved_tensors_finite": finite}


def new_checkpoints() -> None:
    smoke.configure("launch1")
    freeze = smoke.freeze_identity()
    directory = smoke.HERE / "launch1" / "checkpoints"
    per_file = {}
    problems = []
    for name in ("v2_train_s11_ep0.pt", "v2_train_s11.pt"):
        path = directory / name
        before = smoke.sha256_file(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        layer = tensor_layer(payload)
        state = payload["training_state"]
        entry = {
            "L1_file_sha256": before, "bytes": path.stat().st_size,
            "L2_payload_metadata": {key: payload.get(key) for key in
                                    ("schema", "trainer_tag", "training_version",
                                     "reward_timing_version", "loss_version", "gae_schema",
                                     "optimizer_schema", "reward_timing_schema")},
            "L2_training_state_self_description": {key: state.get(key) for key in
                                                   ("training_version", "reward_timing_version",
                                                    "loss_version", "state_schema", "gae_schema",
                                                    "optimizer_schema", "reward_timing_schema",
                                                    "episodes_done", "updates", "optimizer_steps",
                                                    "lambda", "data_order_seed")},
            "L3_parameter_content": {key: value for key, value in layer.items()
                                     if key != "module_digests_recomputed"},
            "L4_module_digests": layer["module_digests_recomputed"],
            "L5_forward_likelihood": None,
            "file_byte_hash_after_processing": smoke.sha256_file(path)}
        if before != entry["file_byte_hash_after_processing"]:
            problems.append(f"{name}: this script changed the file bytes")
        per_file[name] = entry
    # a real forward + likelihood on this round's own batch-0 tapes, with the milestone weights
    record = json.loads((directory / "v2_batch_s11_b0.json").read_text("utf-8"))
    rows = []
    for episode in record["episodes"]:
        paths = sorted((smoke.HERE / "launch1" / "logs").glob(
            f"v2_train_s11_e{episode}_s*_a{episode}_tape.jsonl"))
        rows.extend(tape_rows(smoke.HERE / "launch1" / "logs", cell_stem(paths[0])))
    controller, _opt, _state = v2_trainer.load_checkpoint(directory / "v2_train_s11_ep0.pt",
                                                           freeze=freeze)
    measured = likelihood_gap(controller, rows)
    per_file["v2_train_s11_ep0.pt"]["L5_forward_likelihood"] = {
        "tapes_of_batch": measured["tapes"], "max_abs_gap_recomputed_here": measured["max_abs_gap"],
        "tolerance": measured["tolerance"], "within_tolerance": measured["within_tolerance"],
        "recorded_in_this_rounds_batch_record": record["update"]["likelihood_alignment"]}
    if not measured["within_tolerance"]:
        problems.append("the recomputed likelihoods do not match this round's own tapes")
    archived = record["update"]["likelihood_alignment"]
    if abs(archived["max_abs_gap"] - measured["max_abs_gap"]) > 1e-12:
        problems.append("the recomputed max gap differs from the gap the trainer recorded during "
                        "the run")
    batch_after = json.loads((directory / "v2_batch_s11_b1.json").read_text("utf-8"))
    rolling = per_file["v2_train_s11.pt"]
    if rolling["L3_parameter_content"]["parameter_sha256_recomputed"] != \
            batch_after["controller_parameters_after"]["parameter_sha256"]:
        problems.append("the rolling checkpoint's tensors do not match the last batch record's "
                        "post-update hash")
    if rolling["L4_module_digests"] != batch_after["controller_module_digests_after"]:
        problems.append("the rolling checkpoint's module digests do not match the batch record")
    gate("N_new_checkpoints_content_layer", {
        "files": per_file, "problems": problems,
        "note": "the whole-controller parameter digest and the four module digests are separate "
                "namespaces and are compared only against claims of the same kind",
        "passed": not problems})


def legacy_checkpoints() -> None:
    reconciliation = json.loads((smoke.ROUND34 / "CHECKPOINT_IDENTITY_RECONCILIATION.json")
                                 .read_text("utf-8"))
    entries = reconciliation["checkpoints"]
    per_file = {}
    problems = []
    for entry in entries:
        path = smoke.ROOT / entry["checkpoint_path"]
        if not path.exists():
            problems.append(f"{entry['checkpoint_path']}: file missing")
            continue
        before = smoke.sha256_file(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["training_state"]
        layer = tensor_layer(payload)
        claims = entry["parameter_identity_claims_not_recomputed_from_pt"]
        arm = entry["arm"]
        batch = int(entry["batch_index"])
        freeze_record = r35_restore_boundary.round32_freeze_record(arm)
        controller, _restored_state, identity = v2_trainer.load_checkpoint_read_only(
            path, freeze=freeze_record)
        # the batch's own tapes: the episodes this pre-update checkpoint was saved before
        batch_record = json.loads((smoke.ROUND32 / "checkpoints" /
                                   f"r32_batch_{arm}_s{entry['training_seed']}_b{batch}.json")
                                   .read_text("utf-8"))
        rows = []
        for episode in batch_record["episodes"]:
            paths = sorted((smoke.ROUND32 / "logs").glob(
                f"r32_train_{arm}_s{entry['training_seed']}_e{episode}_s*_a{episode}_tape.jsonl"))
            if len(paths) != 1:
                problems.append(f"{path.name}: no unique tape for e{episode}")
                continue
            rows.extend(tape_rows(smoke.ROUND32 / "logs", cell_stem(paths[0])))
        measured = likelihood_gap(controller, rows) if rows else {"tapes": 0, "max_abs_gap": None,
                                                                  "within_tolerance": False}
        recorded = batch_record["update"]["likelihood_alignment"]
        index_claims = entry["state_claims_from_index_and_mechanism"]
        result = {
            "L1_file_sha256": {
                "round35_recomputed": before,
                "round34_observed": entry["file_sha256"]["observed"],
                "index_claim": entry["file_sha256"]["index_claim"],
                "all_three_equal": before == entry["file_sha256"]["observed"]
                                   == entry["file_sha256"]["index_claim"]},
            "L2_payload_metadata": {
                "schemas": {key: payload.get(key) for key in
                            ("schema", "trainer_tag", "loss_version", "gae_schema",
                             "optimizer_schema", "reward_timing_schema")},
                "training_version": payload.get("training_version"),
                "reward_timing_version": payload.get("reward_timing_version"),
                "reduction_mode": payload.get("reduction_mode"),
                "round34_declared_unavailable": [
                    index_claims["checkpoint_payload_training_version"],
                    index_claims["reward_timing_version_in_checkpoint_payload"]],
                "episodes_done": int(state["episodes_done"]), "updates": int(state["updates"]),
                "optimizer_steps": int(state["optimizer_steps"]), "lambda": float(state["lambda"]),
                "lambda_matches_index_claim": float(state["lambda"]) == float(
                    index_claims["lambda"])},
            "L3_parameter_content": {
                "round35_recomputed": layer["parameter_sha256_recomputed"],
                "index_claim": claims["checkpoint_index_parameter_sha256_claim"],
                "batch_record_before_claim": claims["batch_record_controller_parameters_before"][
                    "parameter_sha256"],
                "parameter_count_recomputed": layer["parameter_count_recomputed"],
                "parameter_count_claimed": claims["batch_record_controller_parameters_before"][
                    "parameter_count"],
                "shapes_match": (layer["shapes_recomputed"]
                                 == claims["batch_record_controller_parameters_before"]["shapes"]),
                "round34_declared": "not recomputed (PyTorch unavailable)"},
            "L4_module_digests": {
                "round35_recomputed": layer["module_digests_recomputed"],
                "claim": claims.get("module_digest_claims_before"),
                "round34_declared": "not recomputed (PyTorch unavailable)"},
            "L5_forward_likelihood": {
                "tapes": measured["tapes"], "max_abs_gap_recomputed": measured["max_abs_gap"],
                "within_tolerance": measured["within_tolerance"],
                "archived_sidecar_recorded_gap": recorded["max_abs_gap"],
                "sidecar_comparable": bool(recorded.get("passed") is not None
                                           and measured["tapes"] == recorded["samples"]),
                "gap_difference_vs_sidecar": (None if measured["max_abs_gap"] is None else round(
                    abs(measured["max_abs_gap"] - float(recorded["max_abs_gap"])), 12)),
                "round34_declared": "not run (PyTorch unavailable)"},
            "read_only_identity": identity,
            "file_byte_hash_after_processing": smoke.sha256_file(path)}
        if result["L1_file_sha256"]["all_three_equal"] is False:
            problems.append(f"{path.name}: file hash differs across the three sources")
        if before != result["file_byte_hash_after_processing"]:
            problems.append(f"{path.name}: this script modified a legacy file")
        if result["L3_parameter_content"]["round35_recomputed"] != result["L3_parameter_content"][
                "index_claim"]:
            problems.append(f"{path.name}: recomputed parameter digest differs from the index "
                            "claim")
        if result["L3_parameter_content"]["round35_recomputed"] != result["L3_parameter_content"][
                "batch_record_before_claim"]:
            problems.append(f"{path.name}: recomputed digest differs from the batch record")
        if result["L4_module_digests"]["claim"] and (
                result["L4_module_digests"]["round35_recomputed"]
                != result["L4_module_digests"]["claim"]):
            problems.append(f"{path.name}: recomputed module digests differ from the claim")
        if not result["L5_forward_likelihood"]["within_tolerance"]:
            problems.append(f"{path.name}: the recomputed likelihoods exceed the tolerance")
        per_file[path.name] = result
    aligned = sum(1 for value in per_file.values()
                  if value["L3_parameter_content"]["round35_recomputed"]
                  == value["L3_parameter_content"]["index_claim"])
    forward_runs = sum(1 for value in per_file.values()
                       if value["L5_forward_likelihood"]["tapes"])
    gate("L_legacy_round32_checkpoints_content_layer", {
        "legacy_checkpoint_files": len(per_file),
        "per_file_records": per_file,
        "parameter_digests_matching_their_claims": aligned,
        "native_forward_runs": forward_runs,
        "all_four_declared_unavailable_layers_now_filled": (aligned == len(per_file)
                                                             == len(entries)
                                                             and forward_runs == len(entries)),
        "problems": problems,
        "does_not_rewrite_round34": True,
        "passed": not problems and aligned == len(per_file) == len(entries)})


def main() -> int:
    new_checkpoints()
    legacy_checkpoints()
    payload = {
        "schema": "round35-checkpoint-content-layer/1", "round": 35,
        "environment_launches_used_by_this_script": 0,
        "episode_constructions_by_this_script": 0,
        "optimizer_steps_by_this_script": 0,
        "legacy_files_written": 0,
        "round34_layers_it_declared_unavailable": ["checkpoint payload metadata",
                                                   "parameter content / module digests",
                                                   "native forward", "native likelihood"],
        "round35_status_of_those_layers": "all executed for real on this machine "
                                          "(torch 2.9.0+cu128), see the per-file records",
        "checks": RESULTS, "problems": PROBLEMS, "clean": not PROBLEMS,
        "summary_counts": {
            "legacy_checkpoints": RESULTS["L_legacy_round32_checkpoints_content_layer"]
            ["legacy_checkpoint_files"],
            "parameter_digests_matching_claims": RESULTS[
                "L_legacy_round32_checkpoints_content_layer"]
            ["parameter_digests_matching_their_claims"],
            "native_forward_and_likelihood_runs_on_legacy": RESULTS[
                "L_legacy_round32_checkpoints_content_layer"]["native_forward_runs"]},
        "note": "ROUND34's conclusions are left exactly as delivered; this file is the "
                "independent completion of the layers it could not reach",
    }
    out = smoke.write_json("CHECKPOINT_CONTENT_LAYER.json", payload, refresh=True)
    print(json.dumps({"clean": payload["clean"], "problems": PROBLEMS,
                      "summary": payload["summary_counts"]}, ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0 if payload["clean"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
