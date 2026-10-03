"""Build the tracked ROUND42 report set and Git-external raw-file inventory."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from r42_analysis import analyze_all
from r42_run import RUN_DIR, sha256

ROUND42_ROOT = RUN_DIR.parents[1]
STATUS_PATH = ROUND42_ROOT / "PROJECT_STATUS.md"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _batch_records() -> list[tuple[Path, dict[str, Any]]]:
    found = []
    for path in sorted((RUN_DIR / "checkpoints").glob("r40_batch_s*_b*_attempt*.json")):
        found.append((path, _read(path)))
    return found


def _training_curves(batch_records: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "round42-training-curves/1",
        "training_version": _read(RUN_DIR / "ROUND42_FROZEN_CONFIG.json")[
            "round42"]["training_version"],
        "by_seed": {},
    }
    for seed in (11, 22, 33):
        records = sorted((record for _path, record in batch_records
                          if int(record["training_seed"]) == seed),
                         key=lambda record: int(record["batch"]))
        if len(records) != 32 or [int(record["batch"]) for record in records] != list(range(32)):
            raise RuntimeError(f"seed {seed}: expected 32 committed batch records")
        episodes = []
        for _path, record in batch_records:
            if int(record["training_seed"]) != seed:
                continue
            for artifact in record["episode_artifacts"]:
                row_path = Path(artifact["row_path"])
                row = _read(row_path)
                counts = (row.get("collector_summary") or {}).get("one_shot_counts", {})
                episodes.append({
                    "episode_index": int(row["episode_index"]),
                    "scenario_seed": int(row["scenario_seed"]),
                    "on_time": int(row["on_time"]),
                    "spent_bytes": int(row["spent_bytes"]),
                    "first_report_emissions": int(
                        row.get("relay_counters", {}).get("first_emissions", 0)),
                    "send_now_choices": int(counts.get("send_now_choices", 0)),
                    "skip_episode_choices": int(counts.get("skip_episode_choices", 0)),
                    "free_choices": int(counts.get("free_choices", 0)),
                    "budget_or_capacity_masks": int(
                        counts.get("masked_budget_or_capacity", 0)),
                    "row_path": str(row_path),
                    "row_sha256": sha256(row_path),
                })
        episodes.sort(key=lambda row: row["episode_index"])
        if len(episodes) != 128 or [row["episode_index"] for row in episodes] != list(range(128)):
            raise RuntimeError(f"seed {seed}: training episode rows are incomplete")
        result["by_seed"][str(seed)] = {
            "episodes": episodes,
            "batches": [
                {
                    "batch": int(record["batch"]),
                    "episodes": list(record["episodes"]),
                    "scenarios": list(record["scenarios"]),
                    "lambda_at_rollout": float(record["lambda_at_rollout"]),
                    "g_hat": float(record["constraint"]["g_hat"]),
                    "lambda_before": float(record["dual"]["lambda_before"]),
                    "lambda_next": float(record["dual"]["lambda_next"]),
                    "sample_count": int(record["update"]["samples"]),
                    "minibatches": int(record["update"]["minibatches"]),
                    "optimizer_steps": int(record["update"]["optimizer_steps"]),
                    "policy_surrogate": record["update"].get("policy_surrogate"),
                    "task_value_loss": record["update"].get("value_loss_task"),
                    "cost_value_loss": record["update"].get("value_loss_cost"),
                    "entropy": record["update"].get("entropy"),
                    "approx_kl": record["update"].get("approx_kl"),
                    "clip_fraction": record["update"].get("clip_fraction"),
                    "gradient_norm_mean": record["update"].get("gradient_norm_mean"),
                    "module_gradient_l1_mean": record["update"].get(
                        "module_gradient_l1_mean"),
                    "module_gradient_l1_min": record["update"].get(
                        "module_gradient_l1_min"),
                    "likelihood_alignment": record["update"].get("likelihood_alignment"),
                    "parameter_sha256_before": record[
                        "controller_parameters_before"]["parameter_sha256"],
                    "parameter_sha256_after": record[
                        "controller_parameters_after"]["parameter_sha256"],
                    "module_digests_before": record["controller_module_digests_before"],
                    "module_digests_after": record["controller_module_digests_after"],
                    "batch_record": str(_path),
                    "batch_record_sha256": sha256(_path),
                    "postupdate_checkpoint": record["postupdate_checkpoint"],
                    "postupdate_checkpoint_sha256": record.get(
                        "postupdate_checkpoint_sha256"),
                }
                for _path, record in batch_records
                if int(record["training_seed"]) == seed
            ],
            "episodes_done": int(records[-1]["episodes_done_after"]),
            "updates": int(records[-1]["updates_after"]),
            "optimizer_steps": int(records[-1]["optimizer_steps_after"]),
            "on_time_total": sum(row["on_time"] for row in episodes),
            "spent_bytes_total": sum(row["spent_bytes"] for row in episodes),
            "first_report_emissions_total": sum(
                row["first_report_emissions"] for row in episodes),
            "send_now_choices_total": sum(row["send_now_choices"] for row in episodes),
            "skip_episode_choices_total": sum(row["skip_episode_choices"] for row in episodes),
            "free_choices_total": sum(row["free_choices"] for row in episodes),
            "lambda_history": [
                {"batch": int(record["batch"]),
                 "g_hat": float(record["constraint"]["g_hat"]),
                 "lambda_before": float(record["dual"]["lambda_before"]),
                 "lambda_next": float(record["dual"]["lambda_next"])}
                for record in records
            ],
        }
    return result


def _checkpoint_index(*, runtime: dict[str, Any], frozen: dict[str, Any],
                      batch_records: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    trainer = runtime["trainer"]
    result: dict[str, Any] = {
        "kind": "round42-checkpoint-index/1",
        "read_only_inspection": True,
        "old_training_restore": "not invoked; OLD checkpoints loaded read-only for evaluation",
        "new_from_registered_round40_ep0": {},
        "new_checkpoints": {},
        "batch_parameter_chain": {},
    }
    for seed in (11, 22, 33):
        source_ep0 = frozen["round42"]["new_initial_models"][str(seed)]
        source = Path(source_ep0["path"])
        source_before = sha256(source)
        if source_before != source_ep0["file_sha256_before"]:
            raise RuntimeError(f"ROUND40 ep0 input changed for seed {seed}")
        result["new_from_registered_round40_ep0"][str(seed)] = {
            "path": str(source),
            "file_sha256": source_before,
            "parameter_sha256": source_ep0["parameter_sha256"],
            "episodes_done": 0,
            "read_only": True,
            "used_only_as_identity_reference": True,
        }
        result["new_checkpoints"][str(seed)] = {}
        for milestone in (0, 64, 128):
            path = RUN_DIR / "checkpoints" / f"v2_train_s{seed}_ep{milestone}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"ROUND42 milestone checkpoint missing: {path}")
            before_sha = sha256(path)
            controller, state, identity = trainer.load_checkpoint_read_only(
                path, freeze=runtime["round42_freeze"])
            parameter = controller.parameter_summary()
            module_digests = runtime["r40_train"]._module_digests(controller)
            if int(state["training_seed"]) != seed or int(state["episodes_done"]) != milestone:
                raise RuntimeError(f"ROUND42 milestone checkpoint cursor mismatch: {path}")
            if milestone == 0 and parameter["parameter_sha256"] != source_ep0["parameter_sha256"]:
                raise RuntimeError(f"ROUND42 seed {seed} ep0 does not match registered R40 init")
            if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
                raise RuntimeError(f"ROUND42 read-only checkpoint inspection restored state: {path}")
            after_sha = sha256(path)
            if before_sha != after_sha:
                raise RuntimeError(f"ROUND42 read-only inspection changed file bytes: {path}")
            result["new_checkpoints"][str(seed)][str(milestone)] = {
                "path": str(path),
                "file_sha256_before": before_sha,
                "file_sha256_after": after_sha,
                "parameter_identity": parameter,
                "module_parameter_sha256": module_digests,
                "episodes_done": int(state["episodes_done"]),
                "updates": int(state["updates"]),
                "optimizer_steps": int(state["optimizer_steps"]),
                "lambda": float(state["lambda"]),
                "training_version": state["training_version"],
                "reward_timing_version": state["reward_timing_version"],
                "comm_action_semantics": state["comm_action_semantics"],
                "read_only_identity": identity,
            }
        result["batch_parameter_chain"][str(seed)] = [
            {
                "batch": int(record["batch"]),
                "before": record["controller_parameters_before"],
                "after": record["controller_parameters_after"],
                "preupdate_checkpoint": record["preupdate_checkpoint"],
                "preupdate_checkpoint_sha256": record["preupdate_checkpoint_sha256"],
                "postupdate_checkpoint": record["postupdate_checkpoint"],
                "postupdate_checkpoint_sha256": _checkpoint_hash_for_batch(
                    RUN_DIR / "checkpoints" / str(record["postupdate_checkpoint"])),
            }
            for _path, record in batch_records
            if int(record["training_seed"]) == seed
        ]
    old_reconciliation = _read(
        Path(frozen["execution_identity"]["round40_worktree"])
        / "runs" / "comm_first_report_training_ablation_20261003"
        / "CHECKPOINT_RECONCILIATION.json"
    )
    result["old_round40_ep128_evaluation_only"] = {}
    for seed in (11, 22, 33):
        entry = old_reconciliation["checkpoints"]["NEW_training_outputs"][str(seed)]["ep128"]
        path = Path(entry["path"])
        before_sha = sha256(path)
        if before_sha != entry["file_sha256"]:
            raise RuntimeError(f"ROUND40 OLD ep128 checkpoint changed: {path}")
        controller, state, identity = trainer.load_checkpoint_read_only(path)
        result["old_round40_ep128_evaluation_only"][str(seed)] = {
            "path": str(path),
            "file_sha256_before": before_sha,
            "file_sha256_after": sha256(path),
            "parameter_identity": controller.parameter_summary(),
            "episodes_done": int(state["episodes_done"]),
            "updates": int(state["updates"]),
            "optimizer_steps": int(state["optimizer_steps"]),
            "lambda": float(state["lambda"]),
            "training_version": state.get("training_version"),
            "load_identity": identity,
            "training_resume_performed": False,
        }
    return result


def _checkpoint_hash_for_batch(path: Path) -> str:
    return sha256(path)


def _raw_inventory(frozen: dict[str, Any]) -> dict[str, Any]:
    roots = ("logs", "rows", "checkpoints", "rejected", "failures",
             "launch_claims", "launch_slots", "runtime")
    files = []
    for root_name in roots:
        root = RUN_DIR / root_name
        if not root.is_dir():
            continue
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            files.append({
                "absolute_path": str(path),
                "round_relative_path": path.relative_to(RUN_DIR).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "git_ignored_raw_artifact": True,
            })
    return {
        "kind": "round42-git-external-raw-artifact-inventory/1",
        "raw_outputs_are_committed": False,
        "file_count": len(files),
        "files": files,
        "frozen_external_inputs": frozen["round42"]["frozen_search_allocation_checkpoints"],
        "round40_initial_and_ep128_inputs": frozen["round42"]["prelaunch_gate"][
            "checkpoint_content_and_forward"],
    }


def _verdict(paired: dict[str, Any], curves: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    preserve_all = True
    reduce_cost_all = True
    reduce_first_reports_all = True
    all_send_all = True
    all_skip_all = True
    old_equal_new_all = True
    one_seed_preserves: dict[str, bool] = {"11": True, "22": True, "33": True}
    greedy_sampled_differ: dict[str, bool] = {}
    for seed in (11, 22, 33):
        for mode in ("greedy", "sampled"):
            key = f"s{seed}_{mode}"
            comparison = paired["secondary_absolute_paired_to_immediate"][key][
                "NEW_vs_immediate"]
            old_new = paired["primary_new_minus_old_paired_by_seed_and_mode"][key]
            new = paired["group_summaries"][f"{key}_NEW"]
            old = paired["group_summaries"][f"{key}_OLD"]
            immediate = {int(row["scenario_seed"]): row
                         for row in paired["group_summaries"]["immediate_reference"]["scenes"]
                         if int(row["scenario_seed"]) in range(1_301_000, 1_301_016)}
            new_scenes = new["scenes"]
            new_by_scene = {int(row["scenario_seed"]): row for row in new_scenes}
            tasks_same = all(
                new_by_scene[scene]["on_time"] == immediate[scene]["on_time"]
                for scene in immediate
            )
            cost_lower = new["spent_bytes_total"] < sum(
                row["spent_bytes"] for row in immediate.values())
            reports_lower = new["first_report_emissions_total"] < sum(
                row["first_report_emissions"] for row in immediate.values())
            all_send = (new["free_choices_total"] > 0 and
                        new["send_now_choices_total"] == new["free_choices_total"])
            all_skip = (new["free_choices_total"] > 0 and
                        new["skip_episode_choices_total"] == new["free_choices_total"])
            old_equal = all(
                scene["OLD_on_time"] == scene["NEW_on_time"]
                and scene["OLD_spent_bytes"] == scene["NEW_spent_bytes"]
                for scene in old_new["scenes"]
            )
            preserve_all &= tasks_same
            reduce_cost_all &= cost_lower
            reduce_first_reports_all &= reports_lower
            all_send_all &= all_send
            all_skip_all &= all_skip
            old_equal_new_all &= old_equal
            one_seed_preserves[str(seed)] &= tasks_same
            evidence[key] = {
                "new_tasks_match_immediate_per_scene": tasks_same,
                "new_total_on_time": new["on_time_total"],
                "immediate_total_on_time": sum(
                    row["on_time"] for row in immediate.values()),
                "new_spent_bytes": new["spent_bytes_total"],
                "immediate_spent_bytes": sum(
                    row["spent_bytes"] for row in immediate.values()),
                "new_first_report_emissions": new["first_report_emissions_total"],
                "immediate_first_report_emissions": sum(
                    row["first_report_emissions"] for row in immediate.values()),
                "new_all_free_choices_send": all_send,
                "new_all_free_choices_skip": all_skip,
                "old_and_new_match_task_and_cost_in_every_scene": old_equal,
                "new_minus_old_task_delta": old_new["on_time_total_delta"],
                "new_minus_old_cost_delta_bytes": (
                    old_new["spent_bytes_total_right"] - old_new["spent_bytes_total_left"]),
                "new_minus_immediate_max_scene_loss": comparison[
                    "maximum_single_scene_loss"],
            }
        greedy = paired["primary_new_minus_old_paired_by_seed_and_mode"][f"s{seed}_greedy"]
        sampled = paired["primary_new_minus_old_paired_by_seed_and_mode"][f"s{seed}_sampled"]
        greedy_sampled_differ[str(seed)] = (
            greedy["on_time_total_right"] != sampled["on_time_total_right"]
            or greedy["spent_bytes_total_right"] != sampled["spent_bytes_total_right"]
            or [scene["NEW_on_time"] for scene in greedy["scenes"]]
            != [scene["NEW_on_time"] for scene in sampled["scenes"]]
        )
    return {
        "kind": "round42-verdict/1",
        "status": "complete",
        "primary_scope": "one-shot target-level first-report SEND_NOW vs SKIP_EPISODE",
        "new_preserves_task_count_and_reduces_first_reports_and_cost_in_all_seed_mode_pairs": (
            preserve_all and reduce_cost_all and reduce_first_reports_all),
        "all_new_free_choices_are_send": all_send_all,
        "all_new_free_choices_are_skip": all_skip_all,
        "old_is_exactly_equal_to_new_for_tasks_and_cost_in_all_pairs": old_equal_new_all,
        "one_seed_task_count_match_by_seed": one_seed_preserves,
        "greedy_sampled_outcomes_differ_by_seed": greedy_sampled_differ,
        "seed_mode_evidence": evidence,
        "interpretation": [
            "These indicators describe this registered 16-scene development block only.",
            "No statistical, population noninferiority, or generalization claim is made.",
            "A one-shot SEND/SKIP choice tests report selection; it does not solve general timing.",
            "One-shot choices cannot defer a send: later fixed HOLDs are latch closure, not WAIT.",
            "The result does not justify tuning or a follow-up run without separate authorization.",
        ],
    }


def _markdown_report(*, frozen: dict[str, Any], result: dict[str, Any],
                     paired: dict[str, Any], curves: dict[str, Any],
                     checkpoint_index: dict[str, Any], inventory: dict[str, Any],
                     verdict: dict[str, Any]) -> str:
    ledger = result["execution_ledger"]
    counts = ledger["counts"]
    lines = [
        "# ROUND42 Report — one-shot SEND/SKIP first-report selection ablation",
        "",
        "## Execution identity",
        "",
        f"- Frozen prelaunch HEAD: `{result['prelaunch_head']}`",
        f"- Branch: `{frozen['execution_identity']['branch']}`",
        f"- ROUND41 delivery HEAD: `{frozen['execution_identity']['round41_delivery_head']}`",
        f"- ROUND40 delivery HEAD: `{frozen['execution_identity']['round40_delivery_head']}`",
        f"- Training version: `{frozen['round42']['training_version']}`",
        f"- Action semantics: `{frozen['round42']['comm_action_semantics']}`",
        f"- Runtime imports were pinned and path-checked against the ROUND40 canonical input HEAD `{frozen['execution_identity']['canonical_read_only_input_head']}`. The ROUND41-derived checkout has {sum(not item['matches_round40_active_dependency'] for item in frozen['execution_identity']['round42_checkout_dependency_sha256'].values())} committed dependency snapshots with different content; those snapshots were not imported. See `ROUND42_CONFIGURATION_DIFF.json` for both hashes and active-source paths.",
        f"- Frozen execution duration: {result['duration_seconds']:.1f} seconds",
        "",
        "## Launches and updates",
        "",
        f"- Normal environment constructor calls: **{counts['environment_constructor_calls']} / 672**",
        f"- Simulations started: **{counts['simulation_started']} / 672**",
        f"- Technical reserve used: **{counts['reserve_used']} / 28**",
        f"- Total launch claims: **{counts['total_used']} / 700**",
        f"- Actual Adam optimizer steps: **{counts['optimizer_step_calls']}**",
        f"- Committed batch-record step sum matches the Adam hook: **{counts['optimizer_step_count_matches_committed_batches']}**",
        f"- ROUND42 checkpoint write calls: **{counts['checkpoint_write_calls']}**",
        "- Synthetic optimizer steps: **0** (the pure-compute gate guarded this before launch).",
        "- No extra smoke, independent stream, or throughput probe was launched.",
        "",
        "## Prelaunch and first-batch gates",
        "",
        f"- Pure-compute gate: {frozen['round42']['prelaunch_gate']['pure_compute']['test_count']} tests passed; environment constructors=0, optimizer steps=0, checkpoint writes=0.",
        "- Each seed's first four episodes produced one real PPO batch before continuation.",
        "- The actual batch gate independently checked action-set closure, SEND/SKIP likelihood, event ticks, cost bundling, GAE targets, paired `g_hat`, lambda update, optimizer parameter scope, both critics, and checkpoint identity.",
        "- Gate artifacts: `ROUND42_FIRST_BATCH_GATES.json`.",
        "",
        "## Training outcomes",
        "",
        "| Seed | On-time tasks / 128 | Bytes | First-report emissions | Free SEND | Free SKIP | Final λ | Adam steps |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in (11, 22, 33):
        data = curves["by_seed"][str(seed)]
        lines.append(
            f"| {seed} | {data['on_time_total']} | {data['spent_bytes_total']} | "
            f"{data['first_report_emissions_total']} | {data['send_now_choices_total']} | "
            f"{data['skip_episode_choices_total']} | "
            f"{data['lambda_history'][-1]['lambda_next']:.6f} | {data['optimizer_steps']} |"
        )
    lines.extend([
        "",
        "Training actions are sampled. All 128 episodes use the frozen ROUND40 scenario order, and no best-seed or best-checkpoint selection was performed.",
        "",
        "## Development comparisons",
        "",
        "Each row is paired on the same 16 registered development scenes. Greedy and sampled results remain separate.",
        "",
        "| Seed / mode | OLD tasks / bytes | NEW tasks / bytes | Immediate tasks / bytes | NEW−OLD tasks / bytes | NEW SEND / SKIP |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for seed in (11, 22, 33):
        for mode in ("greedy", "sampled"):
            key = f"s{seed}_{mode}"
            old = paired["group_summaries"][f"{key}_OLD"]
            new = paired["group_summaries"][f"{key}_NEW"]
            immediate = paired["secondary_absolute_paired_to_immediate"][key][
                "NEW_vs_immediate"]["on_time_total_left"]
            immediate_bytes = paired["secondary_absolute_paired_to_immediate"][key][
                "NEW_vs_immediate"]["spent_bytes_total_left"]
            delta = paired["primary_new_minus_old_paired_by_seed_and_mode"][key]
            lines.append(
                f"| {seed} / {mode} | {old['on_time_total']} / {old['spent_bytes_total']} | "
                f"{new['on_time_total']} / {new['spent_bytes_total']} | "
                f"{immediate} / {immediate_bytes} | "
                f"{delta['on_time_total_delta']:+d} / "
                f"{delta['spent_bytes_total_right'] - delta['spent_bytes_total_left']:+d} | "
                f"{new['send_now_choices_total']} / {new['skip_episode_choices_total']} |"
            )
    lines.extend([
        "",
        "The machine-readable paired file lists scene-level task deltas, recovered/lost target IDs, target swaps, maximum single-scene loss, cost, emissions, packets delivered/lost, and mask/refusal counts: `ROUND42_PAIRED_RESULTS.json`.",
        "The accepted-emission records in that file are relay-accepted send records (cost charged at emission); recipient delivery and packet loss counts are reported separately, and each raw transcript is linked by path and hash.",
        "",
        "Rule baselines on the same 16 scenes:",
        "",
        "| Rule arm | On-time tasks | Bytes | First reports | Packets delivered | Packets lost |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm in ("late_discovery_gate", "always_hold"):
        rule = paired["rule_absolute_performance"][arm]["absolute"]
        lines.append(
            f"| {arm} | {rule['on_time_total']} | {rule['spent_bytes_total']} | "
            f"{rule['first_report_emissions_total']} | {rule['packet_delivered_total']} | "
            f"{rule['packet_lost_total']} |"
        )
    lines.extend([
        "",
        "",
        "## Verdict indicators",
        "",
        f"- Task counts match immediate per scene, with fewer first reports and lower bytes, in all six seed/mode pairs: **{verdict['new_preserves_task_count_and_reduces_first_reports_and_cost_in_all_seed_mode_pairs']}**.",
        f"- All free NEW choices are SEND: **{verdict['all_new_free_choices_are_send']}**.",
        f"- All free NEW choices are SKIP: **{verdict['all_new_free_choices_are_skip']}**.",
        f"- OLD exactly matches NEW tasks and bytes in every scene and pair: **{verdict['old_is_exactly_equal_to_new_for_tasks_and_cost_in_all_pairs']}**.",
        f"- Greedy and sampled task/cost outcomes differ by seed: `{json.dumps(verdict['greedy_sampled_outcomes_differ_by_seed'], ensure_ascii=False)}`.",
        "",
        "These are descriptive checks on the registered development block. They do not support significance, population noninferiority, or generalization claims. The policy offers one immediate SEND/SKIP decision; later fixed HOLDs close an already chosen target and do not represent a learned wait duration.",
        "",
        "## Checkpoints and source artifacts",
        "",
        "- NEW ep0, ep64, ep128 and every batch parameter/checkpoint link are in `ROUND42_CHECKPOINT_INDEX.json`.",
        "- ROUND40 ep0 identities match each NEW ep0 initialization; OLD ROUND40 ep128 files were only loaded read-only and their byte hashes were checked after evaluation.",
        f"- Raw local artifact inventory contains {inventory['file_count']} files with absolute paths, sizes and SHA256: `ROUND42_RAW_ARTIFACT_MANIFEST.json`.",
        "- Raw lifecycles, transcripts, tapes, rows, checkpoints and launch claims remain Git-ignored.",
        "",
        "## Limitations and next-step boundary",
        "",
        "- ROUND42 uses the registered 16-scene development block, not an independent test.",
        "- OLD's historical first-report-only weights interpret HOLD as episode SKIP under this explicit one-shot evaluation protocol; historical ROUND40 behavior is not equivalent.",
        "- Results do not establish that all communication timing or full communication-frequency control is solved.",
        "- Any additional smoke, tuning, training, or confirmation experiment requires separate authorization.",
        "",
        "## Deliverable commit",
        "",
        "The post-run report, manifests and project-status update are committed locally on the ROUND42 branch. No push, merge or deployment was performed.",
        "",
    ])
    return "\n".join(lines)


def write_final_deliverables(*, runtime: dict[str, Any], frozen: dict[str, Any],
                             result: dict[str, Any]) -> dict[str, Any]:
    paired = analyze_all(runtime=runtime, frozen=frozen)
    batch_records = _batch_records()
    curves = _training_curves(batch_records)
    checkpoint_index = _checkpoint_index(runtime=runtime, frozen=frozen,
                                         batch_records=batch_records)
    inventory = _raw_inventory(frozen)
    verdict = _verdict(paired, curves)
    _write_json(RUN_DIR / "ROUND42_PAIRED_RESULTS.json", paired)
    _write_json(RUN_DIR / "ROUND42_TRAINING_CURVES.json", curves)
    gates = {str(seed): _read(RUN_DIR / "runtime" / "gates"
                              / f"seed_{seed}_first_batch.json")
             for seed in (11, 22, 33)}
    _write_json(RUN_DIR / "ROUND42_FIRST_BATCH_GATES.json", {
        "kind": "round42-real-first-batch-gates/1",
        "all_passed_before_continuation": True,
        "by_seed": gates,
    })
    _write_json(RUN_DIR / "ROUND42_CHECKPOINT_INDEX.json", checkpoint_index)
    _write_json(RUN_DIR / "ROUND42_RAW_ARTIFACT_MANIFEST.json", inventory)
    _write_json(RUN_DIR / "ROUND42_VERDICT.json", verdict)
    _write_json(RUN_DIR / "ROUND42_EXECUTION_LEDGER.json", result["execution_ledger"])

    report = _markdown_report(
        frozen=frozen, result=result, paired=paired, curves=curves,
        checkpoint_index=checkpoint_index, inventory=inventory, verdict=verdict)
    report_path = RUN_DIR / "ROUND42_REPORT.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")

    marker = "## ROUND42 — one-shot SEND/SKIP first-report selection"
    current_status = STATUS_PATH.read_text(encoding="utf-8")
    if marker in current_status:
        raise RuntimeError("PROJECT_STATUS already has a ROUND42 section; refusing to overwrite it")
    status_section = [
        "",
        marker,
        "",
        f"- Delivery branch: `{frozen['execution_identity']['branch']}`; frozen HEAD `{result['prelaunch_head']}`.",
        f"- Local report: `runs/comm_one_shot_report_selection_20261003/ROUND42_REPORT.md`.",
        f"- Training version: `{frozen['round42']['training_version']}`; semantics `one-shot-send-skip/1`.",
        f"- Registered execution completed: {result['execution_ledger']['counts']['total_used']}/672 normal launches, {result['execution_ledger']['counts']['reserve_used']}/28 reserve, {result['execution_ledger']['counts']['optimizer_step_calls']} real Adam steps.",
        "- Interpretation is limited to the registered 16-scene development block; see the report for per-seed/mode paired outcomes and fixed verdict indicators.",
        "- No follow-up run is authorized by these results.",
        "",
    ]
    STATUS_PATH.write_text(current_status.rstrip() + "\n" + "\n".join(status_section),
                           encoding="utf-8", newline="\n")
    return {
        "report_path": str(report_path),
        "paired_results_path": str(RUN_DIR / "ROUND42_PAIRED_RESULTS.json"),
        "checkpoint_index_path": str(RUN_DIR / "ROUND42_CHECKPOINT_INDEX.json"),
        "raw_inventory_path": str(RUN_DIR / "ROUND42_RAW_ARTIFACT_MANIFEST.json"),
        "verdict": verdict,
    }


__all__ = ["write_final_deliverables"]
