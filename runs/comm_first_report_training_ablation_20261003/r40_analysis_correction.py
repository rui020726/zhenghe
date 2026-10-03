"""Run ROUND40's frozen post-run analysis with its seed-scoped summary binding corrected.

The frozen analyzer accidentally reused the last training-loop summary while reconciling
per-seed ep0/64/128 checkpoints. This adapter changes that one local binding in memory.
It does not change or rewrite the frozen analyzer, training artifacts, checkpoints, or rows.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(input_root: str | Path) -> dict[str, Any]:
    import sys

    round_dir = Path(__file__).resolve().parent
    if str(round_dir) not in sys.path:
        sys.path.insert(0, str(round_dir))
    import r40_worker

    _root, common, _runner, _trainer, _controller, _freeze = r40_worker._load_context(
        input_root)
    import r40_analysis

    manifest_path = round_dir / "ROUND40_ANALYSIS_CORRECTION.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    analyzer_path = Path(r40_analysis.__file__).resolve()
    adapter_path = Path(__file__).resolve()
    if _sha256(analyzer_path) != manifest["frozen_analyzer_sha256"]:
        raise RuntimeError("frozen ROUND40 analyzer bytes differ from the correction manifest")
    if _sha256(adapter_path) != manifest["correction_script_sha256"]:
        raise RuntimeError("ROUND40 analysis correction script hash differs from its manifest")

    source = inspect.getsource(r40_analysis.analyze)
    original_source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    target = '        milestone_paths = summary.get("milestone_checkpoints", {})\n'
    replacement = ('        summary = training[str(seed)]["summary"]\n' + target)
    if source.count(target) != 1:
        raise RuntimeError("frozen analyzer no longer has the audited stale-summary line")
    patched_source = source.replace(target, replacement, 1)
    patched_source_sha256 = hashlib.sha256(patched_source.encode("utf-8")).hexdigest()
    if original_source_sha256 != manifest["analyze_function_source_sha256"]:
        raise RuntimeError("ROUND40 analyze() function source differs from the audited version")
    exec(compile(patched_source, str(analyzer_path), "exec"), r40_analysis.__dict__)

    ledger = r40_worker.r40.ledger_snapshot()
    if ledger["normal_used"] != 672 or ledger["reserve_used"] > 28:
        raise RuntimeError("ROUND40 environment ledger is incomplete or exceeds its cap")
    if sum(claim.get("status") == "complete" for claim in ledger["claims"]) != 672:
        raise RuntimeError("ROUND40 does not have 672 completed environment claims")
    use_record = {
        "kind": "round40-analysis-correction-use/1",
        "recorded_at_unix": time.time(),
        "correction_script_sha256": _sha256(adapter_path),
        "frozen_analyzer_sha256": _sha256(analyzer_path),
        "analyze_function_source_sha256_before": original_source_sha256,
        "analyze_function_source_sha256_after": patched_source_sha256,
        "exact_in_memory_patch": {
            "inserted_before": target.rstrip("\n"),
            "inserted_line": 'summary = training[str(seed)]["summary"]',
            "reason": "bind ep0/64/128 milestone paths to the seed currently being reconciled",
        },
        "normal_environment_claims_complete": ledger["normal_used"],
        "reserve_environment_claims_used": ledger["reserve_used"],
        "environment_constructed_by_correction": False,
        "optimizer_steps_by_correction": 0,
        "checkpoint_or_raw_row_mutation_by_correction": False,
        "training_or_evaluation_semantics_changed": False,
    }
    use_path = common.OUTPUT_DIR / "training_progress" / (
        f"analysis_correction_use_{time.time_ns()}_{os.getpid()}.json"
    )
    _write_once(use_path, use_record)
    result = r40_analysis.analyze(common, write_artifacts=True)
    verdict = json.loads((common.OUTPUT_DIR / "VERDICT.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "analysis_correction": str(use_path),
        "verdict": verdict,
        "normal_launches": ledger["normal_used"],
        "reserve_launches": ledger["reserve_used"],
        "optimizer_steps": sum(
            int(batch["optimizer_steps"])
            for seed_row in result["training_diagnostics"].values()
            for batch in seed_row["batches"]
        ),
        "analysis_outputs_written": True,
    }, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    args = parser.parse_args(argv)
    run(args.input_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
