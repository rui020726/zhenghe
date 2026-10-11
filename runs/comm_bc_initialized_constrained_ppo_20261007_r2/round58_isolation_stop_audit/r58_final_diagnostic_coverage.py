"""Read-only source mapping of the original 556 minibatches and nine gradient points."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOTS = [BASE / "round58_gen3/raw"] + [
    BASE / f"round58_gen3_resume_v{v}/raw" for v in (18, 22, 27, 29, 32)]
APPROVED_RELOCATED_ORIGINALS = HERE / "round58_resume_v18_superseded_identity_quarantine_20261010T0030Z"
SEEDS = (11, 22, 33)
POINT_BATCHES = (0, 15, 31)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor_sha(checkpoint: Path) -> str:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["controller"]
    actor = {**state["trunk"], **state["action_head"]}
    return hashlib.sha256(b"".join(t.detach().cpu().contiguous().numpy().tobytes()
                                    for _, t in sorted(actor.items()))).hexdigest()


def main() -> None:
    checkpoint_by_hash = {}
    for root in [*ROOTS, APPROVED_RELOCATED_ORIGINALS]:
        for payload in root.rglob("*.pt"):
            checkpoint_by_hash.setdefault(sha(payload), []).append(payload)

    def locate(expected_path: Path, expected_sha: str) -> tuple[Path, str]:
        if expected_path.is_file() and sha(expected_path) == expected_sha:
            return expected_path, "original_recorded_path"
        candidates = checkpoint_by_hash.get(expected_sha, [])
        if not candidates:
            raise RuntimeError(f"recorded checkpoint bytes unavailable: {expected_path}")
        # The original V18 files were relocated into a named, preserved quarantine.
        # Exact SHA, not filename or mtime, identifies the approved archived source.
        relocated = [path for path in candidates if APPROVED_RELOCATED_ORIGINALS in path.parents]
        if not relocated:
            raise RuntimeError(f"checkpoint is not present at recorded path or approved archive: {expected_path}")
        return sorted(relocated)[0], "sha_verified_v18_archive_relocation"

    batch_records = {}
    for root in ROOTS:
        for path in (root / "checkpoints").glob("r40_batch_s*_b*_attempt01.json"):
            row = read(path)
            key = (int(row["training_seed"]), int(row["batch"]))
            if key in batch_records:
                raise RuntimeError(f"duplicate batch source for {key}")
            commit_path = path.parent / f"r40_batch_commit_s{key[0]}_b{key[1]}.json"
            commit = read(commit_path)
            if (commit["batch_record_sha256"] != sha(path)
                    or int(commit["optimizer_steps"]) != int(row["optimizer_steps_after"])):
                raise RuntimeError(f"batch record/commit mismatch {key}")
            batch_records[key] = (path, row, commit_path)
    expected_keys = {(seed, batch) for seed in SEEDS for batch in range(32)}
    if set(batch_records) != expected_keys:
        raise RuntimeError(f"batch source graph incomplete: {sorted(expected_keys - set(batch_records))}")
    original = BASE / "round58_gen3/raw/runtime"
    records_path = original / "r58_minibatch_records.jsonl"
    diagnostics_path = original / "r58_gradient_diagnostics.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    diagnostics = [json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines()]
    if len(records) != 556 or len(diagnostics) != 9:
        raise RuntimeError("original passive minibatch/diagnostic record count differs from training chain")
    order = [(seed, 0) for seed in SEEDS]
    order += [(seed, batch) for seed in SEEDS for batch in range(1, 32)]
    cursor = 0
    per_batch = []
    checkpoint_index = []
    for seed, batch in order:
        path, source, commit_path = batch_records[(seed, batch)]
        commit = read(commit_path)
        recorded_pre = path.parent / source["preupdate_checkpoint"]
        recorded_post = path.parent / source["postupdate_checkpoint"]
        if recorded_post.name != commit["postupdate_checkpoint"]:
            raise RuntimeError(f"committed before/after checkpoint source mismatch at {seed}/{batch}")
        pre, pre_location = locate(recorded_pre, source["preupdate_checkpoint_sha256"])
        post, post_location = locate(recorded_post, commit["postupdate_checkpoint_sha256"])
        checkpoint_index.append({"seed": seed, "batch": batch,
                                 "recorded_before": str(recorded_pre),
                                 "actual_before": str(pre), "before_sha256": sha(pre),
                                 "before_location": pre_location,
                                 "recorded_after": str(recorded_post),
                                 "actual_after": str(post), "after_sha256": sha(post),
                                 "after_location": post_location,
                                 "commit": str(commit_path), "commit_sha256": sha(commit_path),
                                 "episodes_done": int(commit["episodes_done"]),
                                 "updates": int(commit["updates"]),
                                 "optimizer_steps": int(commit["optimizer_steps"])})
        n = int(source["update"]["minibatches"])
        segment = records[cursor:cursor+n]
        if (len(segment) != n or [int(r["update_index"]) for r in segment] != [batch] * n
                or [int(r["minibatch"]) for r in segment] != list(range(n))):
            raise RuntimeError(f"passive minibatch order/length differs at seed{seed} batch{batch}")
        if any(int(r["n_rows"]) <= 0 or int(r["gate_rows"]) < 0
               or int(r["gate_rows"]) > int(r["n_rows"]) for r in segment):
            raise RuntimeError("passive minibatch exposure fields are invalid")
        per_batch.append({"seed": seed, "batch": batch, "record_start": cursor,
                          "record_count": n, "batch_record": str(path),
                          "batch_record_sha256": sha(path),
                          "commit": str(commit_path), "commit_sha256": sha(commit_path),
                          "sum_n_rows": sum(int(r["n_rows"]) for r in segment),
                          "sum_early_rows": sum(int(r["gate_rows"]) for r in segment),
                          "early_fraction_over_all_minibatch_rows": (
                              sum(int(r["gate_rows"]) for r in segment)
                              / sum(int(r["n_rows"]) for r in segment))})
        cursor += n
    if cursor != 556:
        raise RuntimeError("passive minibatch record sequence has unassigned suffix")
    points = []
    diag_by_hash = {item.get("params_before_sha256"): item for item in diagnostics}
    if len(diag_by_hash) != 9:
        raise RuntimeError("nine diagnostic points are not uniquely keyed by preupdate actor")
    for seed in SEEDS:
        for batch in POINT_BATCHES:
            path, source, commit_path = batch_records[(seed, batch)]
            pre = path.parent / source["preupdate_checkpoint"]
            post = path.parent / source["postupdate_checkpoint"]
            item = source["update"].get("gate_diag")
            status = "recorded_and_source_verified"
            missing = []
            if not isinstance(item, dict):
                missing.append("batch_record.update.gate_diag")
            if not pre.is_file() or not post.is_file():
                missing.append("pre_or_post_checkpoint")
            if missing:
                status = "missing_original_evidence"
                points.append({"seed": seed, "batch": batch, "status": status,
                               "missing": missing, "batch_record": str(path)})
                continue
            if sha(pre) != source["preupdate_checkpoint_sha256"]:
                raise RuntimeError(f"preupdate checkpoint hash changed: {seed}/{batch}")
            before_hash, after_hash = actor_sha(pre), actor_sha(post)
            original_item = diag_by_hash.get(before_hash)
            if original_item is None or item != original_item:
                raise RuntimeError(f"original nine-point record differs from committed batch: {seed}/{batch}")
            if (int(item["update_index"]) != batch
                    or item["params_before_sha256"] != before_hash
                    or item["params_after_sha256"] != after_hash):
                raise RuntimeError(f"nine-point update-before/after actor identity mismatch: {seed}/{batch}")
            points.append({"seed": seed, "batch": batch, "status": status,
                           "batch_record": str(path), "batch_record_sha256": sha(path),
                           "original_diagnostic_sha256": sha(diagnostics_path),
                           "pre_checkpoint": str(pre), "pre_sha256": sha(pre),
                           "post_checkpoint": str(post), "post_sha256": sha(post),
                           "params_before_sha256": before_hash,
                           "params_after_sha256": after_hash,
                           "base_actor_grad_l2": item["base_actor_grad_l2"],
                           "prior_actor_grad_l2": item["prior_actor_grad_l2"],
                           "cosine": item["cosine"]})
    output = {"kind": "round58-final-passive-diagnostic-source-coverage/1",
              "passive_records": {"count": len(records), "path": str(records_path),
                                  "sha256": sha(records_path), "matched_96_committed_batches": True,
                                  "has_training_seed_field": False,
                                  "seed_assignment_basis": "frozen training order + per-batch update index and minibatch count + committed batch records"},
              "gradient_points": {"count": len(diagnostics), "path": str(diagnostics_path),
                                  "sha256": sha(diagnostics_path),
                                  "source_verified_points": sum(p["status"] == "recorded_and_source_verified" for p in points)},
              "per_batch": per_batch, "points": points,
              "limits": ["passive journal has no per-row seed field; assignment uses frozen chronology and batch count",
                         "the nine gradient observations are diagnostic, not independent training outcomes"]}
    (HERE / "ROUND58_FINAL_DIAGNOSTIC_COVERAGE.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "ROUND58_FINAL_CHECKPOINT_INDEX.json").write_text(
        json.dumps({"kind": "round58-final-96-batch-checkpoint-index/1",
                    "source": "original batch record and commit, verified against actual before/after payload SHA256",
                    "batches": checkpoint_index}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"passive_records": len(records), "batches": len(per_batch),
                      "nine_points": points}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
