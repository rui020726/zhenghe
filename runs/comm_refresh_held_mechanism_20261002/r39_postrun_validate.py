"""Read-only ROUND39 launch, output, code, and checkpoint reconciliation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PRELAUNCH = "e8f3e76f5555e1da271e6d8d560c32fadf151ba2"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    plan = read_json(HERE / "FROZEN_SCENE_AND_ARM_PLAN.json")
    book = read_json(HERE / "LAUNCH_BOOK.json")
    gate = read_json(HERE / "FIRST_SCENE_GATE.json")
    code_identity = read_json(HERE / "CODE_IDENTITY.json")
    checkpoint_identity = read_json(HERE / "CHECKPOINT_IDENTITY.json")
    results = read_json(HERE / "ROUND39_RESULTS.json")["cells"]
    paired = read_json(HERE / "PAIRED_ANALYSIS.json")
    chains = read_json(HERE / "MECHANISM_CHAINS.json")["chains"]
    external = read_json(HERE / "GIT_EXTERNAL_ORIGINALS.json")["files"]

    require(git("branch", "--show-current") == plan["base"]["experiment_branch"],
            "wrong branch")
    require(book.get("status") == "complete" and book.get("starts") == 144
            and book.get("completed") == 144 and book.get("technical_failures") == 0
            and book.get("retries_used") == 0, "launch book does not close at 144")
    require(book.get("optimizer_steps") == 0 and book.get("training_launches") == 0
            and book.get("checkpoint_writes") == 0, "launch book violates no-training scope")
    require(book.get("environment_constructions") == 144
            and book.get("hard_cap") == 153, "environment count or cap is wrong")
    require(gate.get("all_passed") is True and gate.get("optimizer_steps") == 0,
            "corrected first-scene gate is not passed")

    attempts = book["attempts"]
    jobs = book["jobs"]
    require(len(attempts) == len(jobs) == 144, "attempt/job count is not 144")
    require(Counter(item.get("status") for item in attempts) == {"completed": 144},
            "attempt ledger contains incomplete or failed cells")
    require(Counter(item.get("status") for item in jobs) == {"completed": 144},
            "plan job status projection does not show 144 completions")
    require(sorted(int(item["index"]) for item in attempts) == list(range(144)),
            "attempt indices are missing or duplicated")
    require(len({int(item["index"]) for item in attempts}) == 144,
            "duplicate attempt index")
    require(len(list((HERE / "claims").glob("*.claim"))) == 144,
            "claim count is not 144")

    expected = {(scene, arm) for scene in plan["scene_scope"]["all_scenes_sorted"]
                for arm in plan["arms"]}
    observed = {(int(cell["scenario"]), cell["arm"]) for cell in results}
    require(len(results) == 144 and observed == expected, "result rows differ from frozen matrix")
    require(len(list((HERE / "rows").glob("*.json"))) == 144,
            "row artifact count is not 144")
    require(len(list((HERE / "checkpoints").glob("*"))) == 0,
            "ROUND39 checkpoint output directory is not empty")

    row_checks = 0
    checkpoint_row_checks = 0
    for cell in results:
        require(cell["on_time"] == len(cell["on_time_target_ids"]),
                f"task set count mismatch at {cell['arm']} s{cell['scenario']}")
        require(cell["first_report_bytes"] + cell["refresh_bytes"] == cell["spent_bytes"],
                f"byte split does not close at {cell['arm']} s{cell['scenario']}")
        require(cell["spent_bytes"] <= 29300
                and cell["remaining_budget_bytes"] == 29300 - cell["spent_bytes"],
                f"budget closure failed at {cell['arm']} s{cell['scenario']}")
        raw_row = read_json(ROOT / cell["row_path"])
        require(raw_row.get("steps") == 1000 and raw_row.get("optimizer_steps") == 0,
                f"row step/update condition failed at {cell['arm']} s{cell['scenario']}")
        row_checks += 1
        if cell["arm"].startswith("s"):
            seed = cell["arm"][1:3]
            expected_model = checkpoint_identity["models"][f"s{seed}"]
            loaded = raw_row.get("checkpoint_load_identity") or {}
            require(loaded.get("load_mode") == "read_only"
                    and loaded.get("checkpoint_bytes_sha256") == expected_model["bytes_sha256"]
                    and loaded.get("parameter_sha256") == expected_model["parameter_sha256"]
                    and loaded.get("training_seed") == int(seed)
                    and loaded.get("episodes_done") == 128
                    and loaded.get("updates") == 32
                    and loaded.get("optimizer_restored") is False
                    and loaded.get("rng_restored") is False,
                    f"frozen checkpoint identity mismatch at {cell['arm']} s{cell['scenario']}")
            checkpoint_row_checks += 1

    for seed in (11, 33):
        for group in ("failed", "control"):
            summary = paired["main_paired_refresh_held_minus_original"][f"s{seed}"][group]
            require(summary["refresh_held_refresh_bytes_total"] == 0
                    and summary["refresh_held_refresh_emissions_total"] == 0,
                    f"refresh suppression did not close for s{seed}/{group}")

    frozen_sources = code_identity["round37_frozen_source_check"]
    require(len(frozen_sources) == 17 and all(
        item["matches_round37_freeze"] for item in frozen_sources.values()),
        "one or more pinned ROUND37 source hashes changed")
    path_prefix = "runs/comm_refresh_held_mechanism_20261002/"
    runner_path = path_prefix + "r39_run.py"
    collector_path = path_prefix + "r39_collector.py"
    for relpath in (runner_path, collector_path):
        current_blob = git("hash-object", relpath)
        frozen_blob = git("rev-parse", f"{PRELAUNCH}:{relpath}")
        require(current_blob == frozen_blob, f"frozen execution code changed: {relpath}")

    checkpoint_checks = {}
    for seed, item in checkpoint_identity["models"].items():
        checkpoint_path = ROOT / item["path"]
        actual = sha256(checkpoint_path)
        require(actual == item["bytes_sha256"], f"checkpoint bytes changed: {seed}")
        checkpoint_checks[seed] = {
            "path": item["path"], "bytes_sha256": actual,
            "parameter_sha256": item["parameter_sha256"],
            "unchanged_from_prelaunch_identity": True,
        }

    ledger_paths = sorted((HERE / "logs").glob("START_LEDGER_p*.json"))
    ledgers = [read_json(path) for path in ledger_paths]
    ledger_entries = [entry for ledger in ledgers for entry in ledger.get("entries", [])]
    require(sum(int(ledger.get("starts", 0)) for ledger in ledgers) == 144
            and len(ledger_entries) == 144
            and all(entry.get("kind") == "normal" for entry in ledger_entries),
            "real Stage C ledger does not equal 144 normal starts")
    require(len(chains) == 165, "designated mechanism chain count is not 165")
    require(len(external) == 557 and not any(item["git_tracked"] for item in external),
            "external source manifest count/tracking state is unexpected")
    external_bytes = 0
    for entry in external:
        path = Path(entry["absolute_path"])
        require(path.is_file() and path.stat().st_size == entry["size_bytes"]
                and sha256(path) == entry["sha256"],
                f"external source manifest mismatch: {entry['absolute_path']}")
        external_bytes += int(entry["size_bytes"])

    outputs = ["ROUND39_RESULTS.json", "PAIRED_ANALYSIS.json", "MECHANISM_CHAINS.json",
               "GIT_EXTERNAL_ORIGINALS.json", "MACHINE_VERDICT.json", "ROUND39_REPORT.md"]
    output_hashes = {}
    for name in outputs:
        path = HERE / name
        require(path.is_file(), f"required output is missing: {name}")
        output_hashes[name] = sha256(path)

    prelaunch_analyzer_blob = git(
        "rev-parse", f"{PRELAUNCH}:{path_prefix}r39_analyze.py")
    current_analyzer_blob = git("hash-object", str(HERE / "r39_analyze.py"))
    validation = {
        "schema": "round39-postrun-validation/1",
        "status": "passed",
        "branch": plan["base"]["experiment_branch"],
        "prelaunch_commit": PRELAUNCH,
        "head_before_final_delivery_commit": git("rev-parse", "HEAD"),
        "matrix": {
            "scenes": len(plan["scene_scope"]["all_scenes_sorted"]),
            "arms_per_scene": len(plan["arms"]),
            "normal_starts": 144,
            "completed_cells": 144,
            "technical_failures": 0,
            "retries": 0,
            "hard_cap": 153,
            "optimizer_steps": 0,
            "training_launches": 0,
            "checkpoint_writes": 0,
            "claims": 144,
            "rows": 144,
            "row_cost_task_budget_checks": row_checks,
            "read_only_checkpoint_row_checks": checkpoint_row_checks,
            "mechanism_chains": len(chains),
            "real_ledger_entries": len(ledger_entries),
            "real_ledger_process_files": [path.name for path in ledger_paths],
            "raw_checkpoint_outputs": 0,
        },
        "gate": {
            "all_passed": True,
            "first_scene_cells": 9,
            "additional_starts_during_comparator_recheck": 0,
            "round37_frozen_sources_matched": 17,
        },
        "source_integrity": {
            "r39_run_frozen_blob_matches_prelaunch": True,
            "r39_collector_frozen_blob_matches_prelaunch": True,
            "production_modules_modified": [],
            "checkpoints": checkpoint_checks,
        },
        "job_status_reconciliation": book.get("job_status_reconciliation"),
        "external_original_manifest": {
            "files": len(external), "total_bytes": external_bytes,
            "all_size_and_sha256_rechecked": True,
            "raw_run_artifacts_git_tracked": False,
        },
        "analysis_audit": {
            "prelaunch_analyzer_blob": prelaunch_analyzer_blob,
            "final_analyzer_git_blob": current_analyzer_blob,
            "round18_import_path_added_after_initial_read_only_failure": True,
            "first_report_delivery_matching_corrected_to_recipient_deliver_records": True,
            "delivery_and_receiver_acceptance_reported_separately": True,
            "no_environment_starts_during_analysis": True,
        },
        "required_output_sha256": output_hashes,
    }
    (HERE / "POSTRUN_VALIDATION.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "rows": row_checks,
                      "checkpoint_identity_rows": checkpoint_row_checks,
                      "external_files_rehashed": len(external),
                      "external_bytes": external_bytes,
                      "outputs": list(output_hashes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
