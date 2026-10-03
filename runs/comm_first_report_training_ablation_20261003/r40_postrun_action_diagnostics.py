"""Pure post-run summaries of ROUND40 tapes and already-recorded PPO updates."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROUND_DIR = Path(__file__).resolve().parent
SEEDS = (11, 22, 33)
MODES = ("greedy", "sampled")
ARMS = ("OLD", "NEW")
SCENES = tuple(range(1301000, 1301016))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _empty() -> dict[str, Any]:
    return {
        "decision_rows": 0,
        "all_restricted_rows": 0,
        "free_first_report_decisions": 0,
        "sampled_or_recorded_send": 0,
        "sampled_or_recorded_hold": 0,
        "greedy_send": 0,
        "greedy_hold": 0,
        "masked_first_reports": 0,
        "fixed_hold_refresh_candidates": 0,
        "forced_candidates": 0,
        "p_send_sum": 0.0,
        "p_send_min": math.inf,
        "p_send_max": -math.inf,
        "greedy_action_mismatches": 0,
        "probability_candidate_count": 0,
    }


def _add_tape(summary: dict[str, Any], tape: dict[str, Any], *,
              greedy_run: bool | None = None) -> None:
    if tape.get("comm_action_eligibility") != "first-report-only/1":
        raise RuntimeError("tape is missing first-report-only action eligibility")
    free = tape.get("free_first_reports", tape.get("actionable", []))
    summary["decision_rows"] += 1
    if not free:
        summary["all_restricted_rows"] += 1
    summary["masked_first_reports"] += len(tape.get("masked", []))
    summary["fixed_hold_refresh_candidates"] += len(tape.get("fixed_hold_refresh", []))
    summary["forced_candidates"] += len(tape.get("forced", []))
    for target in free:
        p_send = float(tape["send_probabilities"][target])
        action = int(tape["actions"][target])
        if action not in (0, 1) or not 0.0 <= p_send <= 1.0:
            raise RuntimeError(f"invalid free first-report action/probability: {target}")
        greedy_action = 1 if p_send >= 0.5 else 0
        summary["free_first_report_decisions"] += 1
        summary["sampled_or_recorded_send"] += int(action == 1)
        summary["sampled_or_recorded_hold"] += int(action == 0)
        summary["greedy_send"] += int(greedy_action == 1)
        summary["greedy_hold"] += int(greedy_action == 0)
        summary["p_send_sum"] += p_send
        summary["p_send_min"] = min(summary["p_send_min"], p_send)
        summary["p_send_max"] = max(summary["p_send_max"], p_send)
        summary["probability_candidate_count"] += 1
        if greedy_run is True and action != greedy_action:
            summary["greedy_action_mismatches"] += 1


def _finalize(summary: dict[str, Any]) -> dict[str, Any]:
    count = int(summary["free_first_report_decisions"])
    rows = int(summary["decision_rows"])
    result = dict(summary)
    result["all_restricted_row_fraction"] = (
        int(summary["all_restricted_rows"]) / rows if rows else 0.0)
    result["recorded_send_fraction_of_free_decisions"] = (
        int(summary["sampled_or_recorded_send"]) / count if count else 0.0)
    result["greedy_send_fraction_of_free_decisions"] = (
        int(summary["greedy_send"]) / count if count else 0.0)
    result["mean_p_send_on_free_decisions"] = (
        float(summary["p_send_sum"]) / count if count else None)
    result["p_send_min"] = (float(summary["p_send_min"]) if count else None)
    result["p_send_max"] = (float(summary["p_send_max"]) if count else None)
    result.pop("p_send_sum")
    result.pop("probability_candidate_count")
    result.pop("greedy_action_mismatches", None)
    return result


def _read_train_batch(seed: int, batch_index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checkpoint_dir = ROUND_DIR / "checkpoints"
    commit = _json(checkpoint_dir / f"r40_batch_commit_s{seed}_b{batch_index}.json")
    batch_path = checkpoint_dir / commit["batch_record"]
    if _sha256(batch_path) != commit["batch_record_sha256"]:
        raise RuntimeError(f"batch record hash mismatch: {batch_path}")
    batch = _json(batch_path)
    tapes: list[dict[str, Any]] = []
    for artifact in batch.get("episode_artifacts", []):
        tape_path = Path(artifact["tape_path"])
        if _sha256(tape_path) != artifact["tape_sha256"]:
            raise RuntimeError(f"training tape hash mismatch: {tape_path}")
        tapes.extend(_read_jsonl(tape_path))
    if len(batch.get("episode_artifacts", [])) != 4:
        raise RuntimeError(f"batch {seed}/{batch_index} does not have four episode tapes")
    return batch, tapes


def build() -> dict[str, Any]:
    training: dict[str, Any] = {}
    for seed in SEEDS:
        batch_summaries = []
        total = _empty()
        batch_count = 0
        for batch_index in range(32):
            batch, tapes = _read_train_batch(seed, batch_index)
            current = _empty()
            for tape in tapes:
                _add_tape(current, tape)
            batch_count += len(tapes)
            for key in total:
                if key == "p_send_min":
                    total[key] = min(total[key], current[key])
                elif key == "p_send_max":
                    total[key] = max(total[key], current[key])
                else:
                    total[key] += current[key]
            update = batch["update"]
            batch_summaries.append({
                "batch": batch_index,
                "episodes": batch["episodes"],
                "scenarios": batch["scenarios"],
                **_finalize(current),
                "g_hat": float(batch["constraint"]["g_hat"]),
                "lambda_before": float(batch["dual"]["lambda_before"]),
                "lambda_after": float(batch["dual"]["lambda_next"]),
                "samples": int(update["samples"]),
                "minibatches": int(update["minibatches"]),
                "optimizer_steps": int(update["optimizer_steps"]),
                "likelihood_max_abs_gap": float(
                    update["likelihood_alignment"]["max_abs_gap"]),
                "gradient_norm_mean_preclip": float(update["gradient_norm_mean"]),
                "ppo_ratio_clip_fraction": float(update["clip_fraction"]),
                "module_gradient_l1_mean_preclip": update["module_gradient_l1_mean"],
                "module_gradient_l1_min_preclip": update["module_gradient_l1_min"],
            })
        training[str(seed)] = {
            "total": _finalize(total),
            "batch_count": len(batch_summaries),
            "tape_decision_row_count": batch_count,
            "batches": batch_summaries,
        }

    final_eval: dict[str, Any] = {}
    for seed in SEEDS:
        for arm in ARMS:
            for mode in MODES:
                pattern = f"v2_r40_eval_{arm.lower()}_s{seed}_ep128_{mode}_s*_a*.json"
                row_paths = sorted((ROUND_DIR / "rows").glob(pattern))
                if len(row_paths) != len(SCENES):
                    raise RuntimeError(f"expected 16 evaluation rows for {arm}/{seed}/{mode}, "
                                       f"found {len(row_paths)}")
                group = _empty()
                per_scene = []
                seen_scenes: set[int] = set()
                for row_path in row_paths:
                    row = _json(row_path)
                    if (row.get("model_arm") != arm or int(row.get("training_seed", -1)) != seed
                            or row.get("eval_mode") != mode
                            or int(row["scenario_seed"]) not in SCENES):
                        raise RuntimeError(f"evaluation row identity mismatch: {row_path}")
                    scene = int(row["scenario_seed"])
                    if scene in seen_scenes:
                        raise RuntimeError(f"duplicate scene in evaluation group: {arm}/{seed}/{mode}/{scene}")
                    seen_scenes.add(scene)
                    tape_path = Path(row["artefacts"]["tape"])
                    local = _empty()
                    for tape in _read_jsonl(tape_path):
                        _add_tape(local, tape, greedy_run=(mode == "greedy"))
                    for key in group:
                        if key == "p_send_min":
                            group[key] = min(group[key], local[key])
                        elif key == "p_send_max":
                            group[key] = max(group[key], local[key])
                        else:
                            group[key] += local[key]
                    scene_metrics = _finalize(local)
                    if mode == "greedy":
                        scene_metrics["greedy_action_mismatches"] = local[
                            "greedy_action_mismatches"]
                    per_scene.append({
                        "scenario_seed": scene,
                        "on_time": int(row["on_time"]),
                        "spent_bytes": int(row["spent_bytes"]),
                        "tape_sha256": _sha256(tape_path),
                        **scene_metrics,
                    })
                if seen_scenes != set(SCENES):
                    raise RuntimeError(f"evaluation scenes do not close for {arm}/{seed}/{mode}")
                if mode == "greedy" and group["greedy_action_mismatches"] != 0:
                    raise RuntimeError(f"greedy evaluation disagrees with saved P(SEND), {arm}/{seed}")
                key = f"{arm}/s{seed}/{mode}"
                group_metrics = _finalize(group)
                if mode == "greedy":
                    group_metrics["greedy_action_mismatches"] = group[
                        "greedy_action_mismatches"]
                final_eval[key] = {"totals": group_metrics, "per_scene": per_scene}

    return {
        "kind": "round40-training-and-final-action-diagnostics/1",
        "action_interpretation": {
            "HOLD": 0,
            "SEND": 1,
            "training_recorded_action": "sampled action saved on tape, counted only for free_first_reports",
            "greedy_action": "SEND iff P(SEND) >= P(HOLD); verified in r40_eligibility.py",
            "greedy_action_mismatches": "reported only for greedy evaluation tapes; sampled actions are not expected to match argmax",
            "masked_or_fixed_hold_candidates_in_actor_probability_counts": False,
        },
        "gradient_interpretation": {
            "gradient_norm_mean_preclip": "mean total norm reported by the frozen trainer before global clip_grad_norm_",
            "ppo_ratio_clip_fraction": "PPO ratio clipping fraction; not the global gradient clipping incidence",
            "global_gradient_clip_incidence": "not persisted at per-minibatch resolution; not inferred from means",
        },
        "training_by_seed": training,
        "final_ep128_evaluation_by_arm_seed_mode": final_eval,
        "source_hashes": {
            "r40_postrun_action_diagnostics.py": _sha256(Path(__file__).resolve()),
            "r40_eligibility.py": _sha256(ROUND_DIR / "r40_eligibility.py"),
            "r40_train.py": _sha256(ROUND_DIR / "r40_train.py"),
            "r40_analysis.py": _sha256(ROUND_DIR / "r40_analysis.py"),
        },
        "environment_constructed": False,
        "optimizer_steps": 0,
    }


def main() -> int:
    output = ROUND_DIR / "TRAINING_ACTION_DIAGNOSTICS.json"
    result = build()
    descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({
        "output": str(output),
        "training_seeds": {
            seed: {
                "episodes": 128,
                "batches": result["training_by_seed"][seed]["batch_count"],
                "free_first_report_decisions": result["training_by_seed"][seed]["total"][
                    "free_first_report_decisions"],
                "recorded_send": result["training_by_seed"][seed]["total"][
                    "sampled_or_recorded_send"],
                "recorded_hold": result["training_by_seed"][seed]["total"][
                    "sampled_or_recorded_hold"],
                "mean_p_send": result["training_by_seed"][seed]["total"][
                    "mean_p_send_on_free_decisions"],
            }
            for seed in result["training_by_seed"]
        },
        "evaluation_groups": len(result["final_ep128_evaluation_by_arm_seed_mode"]),
        "environment_constructed": False,
        "optimizer_steps": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
