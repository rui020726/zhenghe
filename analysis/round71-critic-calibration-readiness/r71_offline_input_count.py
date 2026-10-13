#!/usr/bin/env python
"""ROUND71 read-only offline input counter (ZERO consumption).

Counts the candidate offline calibration input volume from R68's EXISTING teacher data
only: the 96 batch commit records and the 384 training tapes.  It opens no .pt file,
restores no RNG, loads no model, runs no neural forward (not even a NumPy equivalent),
fits nothing and writes no checkpoint.  Pure file reading, hashing and integer counting.

The candidate (APPROVAL_PENDING, not licensed this round) fits one dual-output ridge per
seed on the frozen trunk's mean-over-targets embedding over ALL of that seed's training
teacher rows, so the offline neural input volume equals the target-level 19-D rows needed
to form every mean-over-targets embedding, counted once.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = (REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
       / "round68_late_initial_confidence_ablation")
RAW = RUN / "raw"
ROWS = RAW / "rows"
CKPTS = RAW / "checkpoints"
SEEDS = (11, 22, 33)
BATCH_CAP = 1024
OFFLINE_INPUT_ROW_CAP = 72000     # candidate offline cap (not a licence, just a budget)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def row_path(seed, ep):
    hits = sorted(ROWS.glob(f"v2_r68_train_s{seed}_e{ep}_s*_a1.json"))
    if len(hits) != 1:
        raise RuntimeError(f"row for seed {seed} episode {ep}: {len(hits)} matches")
    return hits[0]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def schedule(target_counts, cap):
    """Reproduce the batching schedule (pure counting; no trunk call)."""
    flushes, batch, bt = [], [], 0
    for k in target_counts:
        if batch and (bt + k > cap):
            flushes.append(batch)
            batch, bt = [], 0
        batch.append(k)
        bt += k
    if batch:
        flushes.append(batch)
    return flushes


def main():
    per_seed = {}
    all_counts = []
    source_files = []
    for seed in SEEDS:
        counts, scenes, seen = [], set(), set()
        for b in range(32):
            rec = load(CKPTS / f"r40_batch_s{seed}_b{b}_attempt01.json")
            for ep, scene in zip(rec["episodes"], rec["scenarios"]):
                key = (seed, int(ep))
                if key in seen:
                    continue
                seen.add(key)
                scenes.add(int(scene))
                rp = row_path(seed, int(ep))
                source_files.append(str(rp))
                row = load(rp)
                for line in Path(row["artefacts"]["tape"]).read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        t = json.loads(line)
                        counts.append(len(t["targets"]))
        flushes = schedule(counts, BATCH_CAP)
        per_seed[str(seed)] = {
            "joint_decision_rows": len(counts),
            "target_level_19d_input_rows": sum(counts),
            "max_targets_in_one_joint_row": max(counts),
            "episodes": len(seen),
            "scenes": len(scenes),
            "batches_at_cap_1024": len(flushes),
            "max_real_input_batch": max(sum(f) for f in flushes),
        }
        all_counts.extend((seed, k) for k in counts)

    total_joint = sum(v["joint_decision_rows"] for v in per_seed.values())
    total_target = sum(v["target_level_19d_input_rows"] for v in per_seed.values())
    all_flushes = schedule([k for _s, k in all_counts], BATCH_CAP)

    out = {
        "kind": "round71-critic-calibration-readiness/offline-input-count/1",
        "task_id": "ROUND71-CRITIC-CALIBRATION-ZERO-USE-READINESS",
        "zero_consumption": {
            "neural_forwards": 0, "pt_files_loaded": 0, "rng_restores": 0,
            "statistical_fits": 0, "model_loads": 0, "optimizer_steps": 0,
            "environment": 0, "episodes": 0, "claims": 0, "checkpoint_writes": 0,
            "note": "only existing JSON batch records and tape paths were read and hashed",
        },
        "teacher_data_source": {
            "round": "ROUND68-LATE-INITIAL-CONFIDENCE-ABLATION",
            "training_scene_range": "1300000..1300047 (real training scenes, not the 180 block)",
            "batch_records": 96,
            "episodes": 384,
            "row_files_read": len(source_files),
        },
        "per_seed": per_seed,
        "totals": {
            "joint_decision_rows": total_joint,
            "target_level_19d_input_rows_AUTHORITATIVE": total_target,
            "batches_at_cap_1024": len(all_flushes),
            "max_real_input_batch": max(sum(f) for f in all_flushes),
        },
        "candidate_offline_budget": {
            "input_row_cap": OFFLINE_INPUT_ROW_CAP,
            "batch_cap": BATCH_CAP,
            "authoritative_consumption": total_target,
            "within_cap": total_target <= OFFLINE_INPUT_ROW_CAP,
            "headroom": OFFLINE_INPUT_ROW_CAP - total_target,
            "note": ("cap constrains the offline calibration phase only; any future licensed "
                     "training/evaluation consumption is counted separately and in full"),
        },
    }
    dst = HERE / "ROUND71_OFFLINE_INPUT_COUNT.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"totals": out["totals"],
                      "per_seed_joint": {k: v["joint_decision_rows"] for k, v in per_seed.items()},
                      "per_seed_target": {k: v["target_level_19d_input_rows"]
                                          for k, v in per_seed.items()},
                      "budget": out["candidate_offline_budget"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())