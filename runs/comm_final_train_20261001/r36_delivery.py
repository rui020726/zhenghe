"""ROUND36 交付目录登记（任务 3d 的可执行部分，0 环境启动）：代码身份、模型清单、Git 外产物哈希表。

产出 `delivery_draft/ARTEFACT_HASH_REGISTRY.json`：
- Git 内交付件清单（`git ls-files` 本轮目录）；
- Git 外原件（每 seed 的 rows / tapes / lifecycles / transcripts / checkpoints / batch records）
  的相对路径、字节数与 SHA256，逐文件登记；
- 模型清单：每个里程碑与 rolling 的字节哈希、payload/training_state 两个版本字段、恢复后的
  whole-controller 参数摘要与四模块 digest（只读恢复，不建 optimizer、不动 RNG）；
- 本轮代码身份（Stage C harness、生产件、导入件、本轮脚本）与开工头。

    /d/project_env/python.exe r36_delivery.py
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import torch

import r36_common as r36
import v2_controller
import v2_train
import v2_trainer

RAW_SUFFIXES = (".json", ".jsonl", ".pt")


def raw_files(seed: int) -> dict[str, Any]:
    directory = r36.seed_directory(seed)
    entries: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix not in RAW_SUFFIXES:
            continue
        if "logs" in path.parts and path.name.startswith("START_LEDGER"):
            continue                     # 账本单独登记，见下
        key = str(path.relative_to(r36.HERE)).replace("\\", "/")
        entries[key] = r36.sha256_file(path)
        sizes[key] = path.stat().st_size
    ledgers: dict[str, Any] = {}
    for path in sorted((directory / "logs").glob("START_LEDGER_p*.json")):
        book = r36.read_json(path)
        ledgers[path.name] = {"starts": int(book.get("starts") or 0),
                              "results": len(book.get("results") or []),
                              "sha256": r36.sha256_file(path)}
    return {"files_with_sha256": entries, "bytes": sizes,
            "total_bytes": sum(sizes.values()), "file_count": len(entries),
            "episode_start_ledgers": ledgers}


def model_inventory(seed: int) -> dict[str, Any]:
    """Read-only content layer of this seed's checkpoints (no optimizer, no RNG restore)."""
    freeze = r36.freeze_identity()
    inventory: dict[str, Any] = {}
    for episodes in (0, 64, 128):
        path = r36.seed_directory(seed) / "checkpoints" / f"v2_train_s{seed}_ep{episodes}.pt"
        inventory[f"milestone_ep{episodes}"] = _one(path, freeze, episodes)
    rolling = v2_train.rolling_path(seed)
    inventory["rolling"] = _one(r36.seed_directory(seed) / "checkpoints" / rolling.name, freeze,
                                None)
    return inventory


def _one(path, freeze: dict[str, Any], episodes: int | None) -> dict[str, Any]:
    payload = r36.checkpoint_version_fields(path)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw["training_state"]
    # _restore_controller builds a fresh controller and loads the saved tensors: it creates no
    # optimizer and touches no RNG, so this is a read-only content-layer read.
    controller = v2_trainer._restore_controller(raw, state)
    identity = v2_controller.parameter_identity(controller)
    return {"file": path.name, "bytes_sha256": payload["bytes_sha256"],
            "training_version": payload["training_version"],
            "reward_timing_version": payload["reward_timing_version"],
            "both_fields_in_payload_and_state": payload["both_fields_complete"],
            "freeze_token_matches_this_round": raw.get("freeze") == freeze,
            "episodes_done": state.get("episodes_done"), "updates": state.get("updates"),
            "optimizer_steps": state.get("optimizer_steps"), "lambda": state.get("lambda"),
            "parameter_sha256": identity["parameter_sha256"],
            "parameter_count": identity.get("parameter_count"),
            "module_digests": v2_train.module_digests(controller),
            "declared_milestone_episodes": episodes,
            "optimizer_restored": False, "rng_restored": False}


def main(argv: list[str]) -> int:
    tracked = [line for line in r36.git("ls-files", "--",
                                        "runs/comm_final_train_20261001").splitlines() if line]
    manifest = r36.read_json(r36.HERE / "FROZEN_CONFIG_MANIFEST.json")
    harness_commit = r36.git("log", "--format=%H", "--diff-filter=A", "--",
                            "runs/comm_final_train_20261001/r36_common.py").strip()
    seeds_present = [seed for seed in r36.SEEDS
                     if (r36.seed_directory(seed) / "LAUNCH_RECORD.json").exists()]
    payload = {
        "kind": "round36-artefact-hash-registry", "schema": "round36-delivery-registry/1",
        "round": "36",
        "registered_at": time.strftime("%F %T"),
        "branch": r36.git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "head_at_registration": r36.git("rev-parse", "HEAD").strip(),
        "head_at_preflight_before_any_commit_of_this_round": manifest["head_at_preflight"],
        "commit_that_added_this_rounds_harness": harness_commit.splitlines()[-1] if harness_commit
        else None,
        "training_path_files_are_byte_identical_at_both_heads": (
            manifest["code_identity"]["stage_c_harness"]
            == r36.code_identity()["stage_c_harness"]
            and manifest["code_identity"]["production"]
            == r36.code_identity()["production"]),
        "code_identity": r36.code_identity(),
        "freeze_identity": r36.freeze_identity(),
        "training_version": r36.TRAINING_VERSION,
        "reward_timing_version": r36.REWARD_TIMING_VERSION,
        "git_tracked_deliverables": tracked,
        "raw_artefacts_kept_outside_git": {f"seed{seed}": raw_files(seed) for seed in seeds_present},
        "model_inventory": {f"s{seed}": model_inventory(seed) for seed in seeds_present},
        "frozen_search_and_allocation_weights": r36.code_identity()["frozen_checkpoints"],
        "reference_cache_read_only": r36.reference_cache()["files_read_only"],
        "quota": {"authorised_launches": r36.MAX_TRAINING_LAUNCHES,
                  "authorised_constructions": r36.MAX_EPISODE_CONSTRUCTIONS,
                  "book": r36.book()},
        "old_round_artefacts_rewritten": 0,
        "note": "everything listed under raw_artefacts_kept_outside_git is on disk only, named "
                "and hashed here, so a reviewer can verify the原件 byte for byte without them "
                "being in the repository"}
    target = r36.HERE / "delivery_draft" / "ARTEFACT_HASH_REGISTRY.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({"tracked_files": len(tracked),
                      "seeds": seeds_present,
                      "raw_files_per_seed": {f"seed{seed}": payload[
                          "raw_artefacts_kept_outside_git"][f"seed{seed}"]["file_count"]
                          for seed in seeds_present},
                      "registry": str(target.relative_to(r36.ROOT))}, ensure_ascii=False,
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
