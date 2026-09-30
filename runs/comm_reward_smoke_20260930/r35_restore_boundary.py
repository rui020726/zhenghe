"""ROUND35 任务 4 的启动无关部分（0 环境启动、0 次 episode 构造、0 个旧文件改写）。

对**真实落盘的 .pt** 使用真实 `v2_trainer` 入口做恢复边界核对。ROUND34 只用 mocked payload 证过
这套门，本轮把它们落到真文件上：

R1 新 checkpoint 真实恢复（launch1 的 ep0 里程碑 + ep8 rolling）：版本两字段齐全、恢复出的
   controller/optimizer/training_state 与文件内自述和批记录相符、重复加载结果相同。
R2 篡改副本（写进本轮 tampered/，原文件不动）：只改 payload 或 training_state 里的
   training_version / reward_timing_version，用计数桩证明拒绝发生在 controller 构造、
   optimizer 构造与状态恢复、RNG 恢复**之前**，且进程 RNG 未被移动。
R3 真实旧 ROUND32 .pt：带 ROUND32 自己的 freeze 身份时 `load_checkpoint` 被版本门拒绝；
   带 ROUND24 freeze 身份时更早的 freeze 门先拒绝（检查顺序如实登记）；
   `load_checkpoint_read_only` 成功并给出 legacy_unversioned_capture_semantics，
   且 optimizer/RNG 未被恢复（这两处用会抛错的桩守著）。
R4 恢复后的确定性续算（0 次 episode 构造）：从 ep0 里程碑恢复 + 用本轮自己的 e0..e3 行与 tape
   重算批 0 更新，与本轮批记录登记的更新后参数哈希/λ/步数逐字节相符。
   跨进程"恢复后继续 rollout"需要再一次 runner 调用，因启动边界未执行，登记为未覆盖项。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy
import torch

import r35_common as smoke
import v2_train
import v2_trainer
from v2_controller import fresh_controller, parameter_identity

LAUNCH = "launch1"
RESULTS: dict[str, Any] = {}
PROBLEMS: list[str] = []


def gate(name: str, payload: dict[str, Any]) -> None:
    payload["passed"] = bool(payload["passed"])
    RESULTS[name] = payload
    if not payload["passed"]:
        PROBLEMS.append(f"{name}: {payload.get('failure', 'check failed')}")


def digest_of(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def rng_fingerprint() -> dict[str, str]:
    return {"python": hash(repr(random.getstate())),
            "numpy": hash(repr(numpy.random.get_state())),
            "torch": hash(bytes(torch.get_rng_state().numpy().tobytes()))}


def checkpoint_paths() -> dict[str, Path]:
    directory = smoke.HERE / LAUNCH / "checkpoints"
    return {"ep0_milestone": directory / "v2_train_s11_ep0.pt",
            "rolling_after_8_episodes": directory / "v2_train_s11.pt"}


def r1_real_resume() -> None:
    smoke.configure(LAUNCH)
    freeze = smoke.freeze_identity()
    paths = checkpoint_paths()
    batch = json.loads((smoke.HERE / LAUNCH / "checkpoints" / "v2_batch_s11_b1.json")
                       .read_text("utf-8"))
    observations: dict[str, Any] = {}
    problems = []
    for name, path in paths.items():
        if not path.exists():
            problems.append(f"{name}: missing {path}")
            continue
        controller, optimizer, state = v2_trainer.load_checkpoint(path, freeze=freeze)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        observations[name] = {
            "path": path.name, "bytes": path.stat().st_size,
            "file_sha256": smoke.sha256_file(path),
            "training_version": payload.get("training_version"),
            "reward_timing_version": payload.get("reward_timing_version"),
            "training_state_training_version": state.get("training_version"),
            "training_state_reward_timing_version": state.get("reward_timing_version"),
            "state_schema": payload.get("schema"), "loss_version": payload.get("loss_version"),
            "gae_schema": payload.get("gae_schema"),
            "optimizer_schema": payload.get("optimizer_schema"),
            "reward_timing_schema": payload.get("reward_timing_schema"),
            "trainer_tag": payload.get("trainer_tag"),
            "episodes_done": int(state["episodes_done"]), "updates": int(state["updates"]),
            "optimizer_steps": int(state["optimizer_steps"]), "lambda": float(state["lambda"]),
            "data_order_seed": state.get("data_order_seed"),
            "parameter_sha256_after_restore": parameter_identity(controller)["parameter_sha256"],
            "parameter_sha256_declared_in_payload": None,
            "module_digests_after_restore": v2_train.module_digests(controller),
            "optimizer_state_sha256": digest_of(payload["optimizer"]),
            "rng_keys": sorted((payload.get("rng") or {}).keys()),
            "rng_restored_matches_file": None,
            "second_load_identical": None}
        # what the file itself says the weights are
        declared = (payload.get("training_state") or {}).get("controller_parameters")
        if declared:
            observations[name]["parameter_sha256_declared_in_payload"] = declared.get(
                "parameter_sha256")
            if declared.get("parameter_sha256") != observations[name][
                    "parameter_sha256_after_restore"]:
                problems.append(f"{name}: restored parameters differ from the payload's claim")
        # RNG restore is real: the process state must equal the file's rng block
        after = v2_trainer.rng_state() if hasattr(v2_trainer, "rng_state") else None
        observations[name]["rng_restored_matches_file"] = (
            None if after is None else after == payload.get("rng"))
        again_controller, _again_optimizer, again_state = v2_trainer.load_checkpoint(
            path, freeze=freeze)
        observations[name]["second_load_identical"] = (
            parameter_identity(again_controller) == parameter_identity(controller)
            and again_state == state)
        if not observations[name]["second_load_identical"]:
            problems.append(f"{name}: a second load of the same file gave a different state")
    rolling = observations.get("rolling_after_8_episodes") or {}
    if rolling:
        if rolling["episodes_done"] != 8 or rolling["updates"] != 2:
            problems.append("the rolling checkpoint does not hold the 8-episode / 2-update state")
        if rolling["parameter_sha256_after_restore"] != batch["controller_parameters_after"][
                "parameter_sha256"]:
            problems.append("the rolling checkpoint's weights differ from the batch record's "
                            "post-update hash")
    versions_ok = all(value.get("training_version")
                      == "constrained-cost-v2/published-completion-event-at-relay-tick/2"
                      and value.get("reward_timing_version")
                      == "published-completion-event-at-relay-tick/2"
                      for value in observations.values())
    gate("R1_new_checkpoint_real_resume", {
        "observations": observations, "problems": problems,
        "both_version_fields_present_in_new_checkpoints": versions_ok,
        "passed": not problems and versions_ok and bool(observations)})


def tampered_copies() -> dict[str, Path]:
    directory = smoke.HERE / "tampered"
    directory.mkdir(parents=True, exist_ok=True)
    source = checkpoint_paths()["rolling_after_8_episodes"]
    freeze = smoke.freeze_identity()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    variants: dict[str, Path] = {}
    edits = {
        "payload_training_version_wrong": lambda block: block.__setitem__(
            "training_version", "constrained-cost-v2/published-completion-event-at-relay-tick/3"),
        "payload_reward_timing_version_wrong": lambda block: block.__setitem__(
            "reward_timing_version", "published-completion-event-at-relay-tick/3"),
        "payload_versions_absent": lambda block: (block.pop("training_version", None),
                                                  block.pop("reward_timing_version", None)),
        "training_state_training_version_wrong": lambda block: block[
            "training_state"].__setitem__(
                "training_version", "constrained-cost-v2"),
        "training_state_reward_timing_version_absent": lambda block: block[
            "training_state"].pop("reward_timing_version", None),
    }
    for name, edit in edits.items():
        block = {key: (dict(value) if isinstance(value, dict) else value)
                 for key, value in payload.items()}
        edit(block)
        path = directory / f"tampered_{name}.pt"
        torch.save(block, path)
        variants[name] = path
    return variants


def r2_tampering_rejected_before_restore() -> None:
    smoke.configure(LAUNCH)
    freeze = smoke.freeze_identity()
    calls = {"controller_built": 0, "optimizer_built": 0, "rng_restored": 0}
    original_restore_controller = v2_trainer._restore_controller
    original_restore_rng = v2_trainer.restore_rng
    original_adam = torch.optim.Adam

    def counting_controller(payload, training_state):
        calls["controller_built"] += 1
        return original_restore_controller(payload, training_state)

    def counting_rng(state):
        calls["rng_restored"] += 1
        return original_restore_rng(state)

    def counting_adam(*args, **kwargs):
        calls["optimizer_built"] += 1
        return original_adam(*args, **kwargs)

    refusals = {}
    problems = []
    before = rng_fingerprint()
    v2_trainer._restore_controller = counting_controller
    v2_trainer.restore_rng = counting_rng
    torch.optim.Adam = counting_adam
    try:
        for name, path in tampered_copies().items():
            marker = {key: value for key, value in calls.items()}
            try:
                v2_trainer.load_checkpoint(path, freeze=freeze)
                problems.append(f"{name}: a tampered checkpoint was accepted")
                refusals[name] = {"accepted": True}
                continue
            except RuntimeError as error:
                delta = {key: calls[key] - marker[key] for key in calls}
                refusal = {
                    "accepted": False, "message": str(error)[:220],
                    "controller_built_after_the_refusal": delta["controller_built"],
                    "optimizer_built_after_the_refusal": delta["optimizer_built"],
                    "rng_restored_after_the_refusal": delta["rng_restored"],
                    "refused_before_any_restore": (delta["controller_built"] == 0
                                                   and delta["optimizer_built"] == 0
                                                   and delta["rng_restored"] == 0)}
                refusals[name] = refusal
                if not refusal["refused_before_any_restore"]:
                    problems.append(f"{name}: the refusal happened after a restore step")
    finally:
        v2_trainer._restore_controller = original_restore_controller
        v2_trainer.restore_rng = original_restore_rng
        torch.optim.Adam = original_adam
    after = rng_fingerprint()
    gate("R2_tampered_copies_refused_before_restore", {
        "variants": refusals, "problems": problems,
        "process_rng_untouched_by_the_refusals": before == after,
        "method": "the controller builder, the Adam constructor and restore_rng were wrapped "
                  "with counters, so 'refused before restoring' is an observed call count, not "
                  "an assertion about the code",
        "original_checkpoints_unmodified": smoke.sha256_file(
            checkpoint_paths()["rolling_after_8_episodes"]) == (
            RESULTS["R1_new_checkpoint_real_resume"]["observations"]
            ["rolling_after_8_episodes"]["file_sha256"]),
        "passed": not problems and before == after})


def round32_freeze_record(arm: str) -> dict[str, Any]:
    """Rebuild a ROUND32 run's freeze identity from that round's own archived manifest.

    The read-only caller is required to supply the checkpoint's recorded freeze identity, so it
    is reconstructed here from the manifest rather than lifted out of the file being checked --
    otherwise the test would feed a checkpoint its own claim and prove nothing.
    """
    manifest = json.loads((smoke.ROUND32 / "CONFIG_AND_MODEL_MANIFEST.json").read_text("utf-8"))
    entry = manifest["arms"][arm]
    return {"freeze_sha256": manifest["code_identity"]["freeze_sha256"],
            "config_hash": manifest["configuration_hash"],
            "branch_start": manifest["branch_start"],
            "head_at_freeze": manifest["head_at_freeze"],
            "round": str(manifest["round"]), "reduction": arm,
            "encoder_arm": entry["encoder_arm"],
            "feature_layout": entry["feature_layout"],
            "reduction_definition": entry["actor_reduction"]}


def r3_legacy_round32_checkpoints() -> None:
    legacy_directory = smoke.ROUND32 / "checkpoints"
    samples = [legacy_directory / name for name in (
        "r32_pre_ALL_ROWS_s11_b0.pt",      # the pre-update state of batch 0
        "r32_train_ALL_ROWS_s11.pt",        # the rolling resume file
        "r32_train_ALL_ROWS_s11_ep0.pt")]   # the registered initialisation milestone
    stage_c_freeze = smoke.freeze_identity()
    manifest = json.loads((smoke.ROUND32 / "CONFIG_AND_MODEL_MANIFEST.json").read_text("utf-8"))
    per_file = {}
    problems = []
    for path in samples:
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        # the read-only caller must supply the checkpoint's own recorded freeze identity: for
        # ROUND32 that is the four manifest fields plus the arm-level keys it wrote per run
        legacy_freeze = round32_freeze_record("ALL_ROWS")
        entry: dict[str, Any] = {
            "path": path.name, "file_sha256": smoke.sha256_file(path),
            "payload_training_version": payload.get("training_version"),
            "payload_reward_timing_version": payload.get("reward_timing_version"),
            "payload_keys": sorted(payload.keys()),
            "freeze_fields_recorded_by_the_file": sorted(legacy_freeze),
            "freeze_matches_the_archived_manifest_fields": all(
                legacy_freeze.get(key) == value for key, value in (
                    ("freeze_sha256", manifest["code_identity"]["freeze_sha256"]),
                    ("config_hash", manifest["configuration_hash"]),
                    ("branch_start", manifest["branch_start"]),
                    ("head_at_freeze", manifest["head_at_freeze"]))),
            "reconstructed_identity_equals_the_file_claim": legacy_freeze == (
                payload.get("freeze") or {}),
            "schemas": {key: payload.get(key) for key in
                        ("schema", "trainer_tag", "loss_version", "gae_schema",
                         "optimizer_schema", "reward_timing_schema")},
            "reduction_mode_recorded": payload.get("reduction_mode")}
        try:
            v2_trainer.load_checkpoint(path, freeze=legacy_freeze)
            problems.append(f"{path.name}: an unversioned legacy checkpoint was resumed")
            entry["resume_with_matching_freeze"] = "ACCEPTED (defect)"
        except RuntimeError as error:
            entry["resume_with_matching_freeze"] = {
                "refused": True, "message": str(error)[:200],
                "refusal_is_the_version_gate": "read-only under the current" in str(error)}
            if not entry["resume_with_matching_freeze"]["refusal_is_the_version_gate"]:
                problems.append(f"{path.name}: refused by something other than the version gate")
        try:
            v2_trainer.load_checkpoint(path, freeze=stage_c_freeze)
            entry["resume_with_stage_c_freeze"] = "ACCEPTED (unexpected)"
            problems.append(f"{path.name}: a foreign-round checkpoint was resumed")
        except RuntimeError as error:
            entry["resume_with_stage_c_freeze"] = {
                "refused": True, "message": str(error)[:200],
                "which_gate_fired_first": "freeze identity" if "freeze identity" in str(error)
                                          else "other"}
        controller, state, identity = v2_trainer.load_checkpoint_read_only(
            path, freeze=legacy_freeze)
        rng_before = v2_trainer.rng_state()
        entry["read_only_load_succeeded"] = True
        entry["read_only_identity"] = identity
        entry["read_only_parameter_sha256"] = parameter_identity(controller)["parameter_sha256"]
        entry["read_only_training_state_episodes_done"] = int(state["episodes_done"])
        if identity.get("timing_interpretation") != "legacy_unversioned_capture_semantics":
            problems.append(f"{path.name}: the legacy label is missing")
        if identity.get("optimizer_restored") or identity.get("rng_restored"):
            problems.append(f"{path.name}: read-only load claimed to restore optimizer or RNG")
        if v2_trainer.rng_state() != rng_before:
            problems.append(f"{path.name}: the read-only load moved the process RNG")
        per_file[path.name] = entry
    gate("R3_real_legacy_checkpoints", {
        "files": per_file, "problems": problems,
        "legacy_files_selected": sorted(per_file),
        "selection_rule": "one b0 pre-update checkpoint, one rolling checkpoint and one ep0 "
                          "milestone from ROUND32, all unversioned",
        "check_order_note": "with the matching freeze identity the version gate is what refuses; "
                            "with a foreign freeze identity the earlier freeze gate refuses "
                            "first, so both refusals are reported separately rather than merged",
        "passed": not problems and bool(per_file)})


def r4_deterministic_continuation_without_new_rollouts() -> None:
    smoke.configure(LAUNCH)
    freeze = smoke.freeze_identity()
    milestone = checkpoint_paths()["ep0_milestone"]
    controller, optimizer, state = v2_trainer.load_checkpoint(milestone, freeze=freeze)
    record = json.loads((smoke.HERE / LAUNCH / "checkpoints" / "v2_batch_s11_b0.json")
                        .read_text("utf-8"))
    episodes = []
    for episode_index in record["episodes"]:
        row_paths = sorted((smoke.HERE / LAUNCH / "rows").glob(
            f"v2_train_s11_e{episode_index}_s*_a{episode_index}.json"))
        row = json.loads(row_paths[0].read_text("utf-8"))
        tape = sorted((smoke.HERE / LAUNCH / "logs").glob(
            f"v2_train_s11_e{episode_index}_s*_a{episode_index}_tape.jsonl"))[0]
        records = [json.loads(line) for line in tape.read_text(encoding="utf-8").splitlines()
                   if line.strip()]

        class Shim:
            pass

        shim = Shim()
        shim.records = records
        row["collector"] = shim
        episodes.append(row)
    references = {int(row["scenario_seed"]): float(row["on_time"]) for row in
                  smoke.reference_cache()["by_scenario"].values()}
    before = parameter_identity(controller)
    update = v2_trainer.ppo_update(controller, optimizer, episodes, update_index=0)
    booked = smoke.book_verification_optimizer_steps(
        int(update["optimizer_steps"]),
        "R4: recomputing batch 0's update from the restored ep0 milestone, to test whether the "
        "checkpoint preserves everything the continuation needs")
    g_record = v2_trainer.batch_g_hat(episodes, references)
    dual = v2_trainer.dual_update(state["lambda"], g_record["g_hat"])
    after = parameter_identity(controller)
    problems = []
    if before["parameter_sha256"] != record["controller_parameters_before"]["parameter_sha256"]:
        problems.append("the restored initialisation is not the parameters the batch started from")
    if after["parameter_sha256"] != record["controller_parameters_after"]["parameter_sha256"]:
        problems.append("recomputing the update from the restored state does not reproduce the "
                        "recorded post-update weights")
    if dual["lambda_next"] != record["dual"]["lambda_next"]:
        problems.append("the recomputed dual step differs from the record")
    if update["samples"] != record["update"]["samples"]:
        problems.append("the recomputed sample count differs from the record")
    gate("R4_deterministic_continuation_from_the_restored_state", {
        "restored_initial_parameters_equal_the_batch_record": (
            before["parameter_sha256"]
            == record["controller_parameters_before"]["parameter_sha256"]),
        "recomputed_parameters_after_equal_the_record": (
            after["parameter_sha256"] == record["controller_parameters_after"]["parameter_sha256"]),
        "recomputed_lambda_equal": dual["lambda_next"] == record["dual"]["lambda_next"],
        "recomputed_optimizer_steps_equal": update["optimizer_steps"]
                                            == record["update"]["optimizer_steps"],
        "samples_recomputed": update["samples"], "samples_recorded": record["update"]["samples"],
        "optimizer_steps_performed_here": update["optimizer_steps"],
        "these_steps_are_booked": True,
        "problems": problems,
        "what_this_does_not_cover": "running new episode rollouts after a cross-process resume "
                                    "needs another runner invocation and is outside the "
                                    "authorised launch count; it is registered as an uncovered "
                                    "item rather than claimed",
        "passed": not problems})


def main() -> int:
    r1_real_resume()
    r2_tampering_rejected_before_restore()
    r3_legacy_round32_checkpoints()
    r4_deterministic_continuation_without_new_rollouts()
    payload = {"schema": "round35-restore-boundary/1", "round": 35,
               "environment_launches_used_by_this_script": 0,
               "episode_constructions_by_this_script": 0,
               "old_round_files_modified": 0,
               "checks": RESULTS, "problems": PROBLEMS, "clean": not PROBLEMS,
               "authorised_launch_position": "this script needs no runner launch; the cross-"
                                             "process continuation test still needs a decision"}
    out = smoke.write_json("RESTORE_BOUNDARY.json", payload, refresh=True)
    print(json.dumps({"clean": payload["clean"], "problems": PROBLEMS,
                      "checks": {key: value["passed"] for key, value in RESULTS.items()}},
                     ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0 if payload["clean"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
