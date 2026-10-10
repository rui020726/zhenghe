"""Append-only, read-only-derived source map for eval-only ROUND58 closure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = HERE.parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    old = HERE / "ROUND58_GEN3_APPROVED_RESUME_SOURCES_V32.json"
    target = HERE / "ROUND58_GEN3_APPROVED_RESUME_SOURCES_V33.json"
    if target.exists():
        raise FileExistsError(target)
    value = json.loads(old.read_text(encoding="utf-8"))
    value["derived_from_map_sha256"] = sha(old)
    value["execution_generation"] = "round58_gen3_resume_v33"
    value["target_checkpoint_root"] = str(
        Path("runs/comm_bc_initialized_constrained_ppo_20261007_r2")
        / "round58_gen3_resume_v33/raw/checkpoints")
    sources = {}
    for seed, generation in ((11, "v29"), (22, "v32"), (33, "v32")):
        raw = ROOT / f"round58_gen3_resume_{generation}" / "raw"
        summary = raw / f"ROUND46_TRAINING_s{seed}.json"
        checkpoint = raw / "checkpoints" / f"v2_train_s{seed}_ep128.pt"
        commit = raw / "checkpoints" / f"r40_batch_commit_s{seed}_b31.json"
        body = json.loads(summary.read_text(encoding="utf-8"))
        receipt = json.loads(commit.read_text(encoding="utf-8"))
        if (body["status"] != "complete" or body["seed"] != seed
                or body["fixed_ep128_report"]["episodes_done"] != 128
                or body["fixed_ep128_report"]["updates"] != 32
                or receipt["episodes_done"] != 128 or receipt["updates"] != 32
                or body["ep128_checkpoint_sha256"] != sha(checkpoint)
                or receipt["postupdate_checkpoint_sha256"] != sha(checkpoint)):
            raise RuntimeError(f"seed {seed} completed source not closed")
        sources[str(seed)] = {
            "summary_path": str(summary.relative_to(REPO)).replace("\\", "/"),
            "summary_sha256": sha(summary),
            "checkpoint_path": str(checkpoint.relative_to(REPO)).replace("\\", "/"),
            "checkpoint_sha256": sha(checkpoint),
            "commit_path": str(commit.relative_to(REPO)).replace("\\", "/"),
            "commit_sha256": sha(commit),
            "optimizer_steps": receipt["optimizer_steps"],
        }
    value["completed_training_sources"] = sources
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    with target.open("xb") as stream:
        stream.write(payload)
    print(json.dumps({"path": str(target), "sha256": sha(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
