"""ROUND37 场景占用审计（0 环境启动）：先核对全项目的"已有登记 / 实际产物 / 声称范围"，
再据此选定一个从未被使用的连续 64 场景块。

选择规则（授权边界 §3）：只看占用，不看结局。本脚本因此**不读任何行的 on_time / spent_bytes**，
它只登记"某个场景号是否在任何形式下被使用或声称过"。候选块按固定顺序试：先看 1600000..1600063，
若被占用则整块另选下一个 1 万对齐的块，不逐个挑场景。

产出 SCENE_USAGE_AUDIT.json。

    /d/project_env/python.exe r37_scene_audit.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

#: 所有场景号都是 6-8 位整数；块边界按 1 万对齐逐个试。
NUMBER = re.compile(r"\b(\d{6,8})\b")
CANDIDATE_BLOCKS = [(1_600_000, 1_600_063), (1_700_000, 1_700_063), (1_800_000, 1_800_063),
                    (1_900_000, 1_900_063)]
#: 明显不是场景号的数字（字节数、时间戳、步数等常见量级）在归类时按"是否落在已知场景块"判定。
KNOWN_BLOCKS = [(30_000, 99_999), (1_100_000, 1_199_999), (1_200_000, 1_299_999),
                (1_300_000, 1_399_999), (1_400_000, 1_499_999), (1_500_000, 1_599_999),
                (1_600_000, 1_699_999), (1_700_000, 1_799_999), (1_800_000, 1_899_999),
                (1_900_000, 1_999_999), (2_000_000, 2_099_999)]


def looks_like_scenario(value: int) -> bool:
    return any(low <= value <= high for low, high in KNOWN_BLOCKS)


def harvest_paths() -> dict[str, list[str]]:
    """Every scenario-looking number that appears in a path under runs/ (files and dirs)."""
    hits: dict[str, set[str]] = defaultdict(set)
    for path in sorted((ROOT / "runs").rglob("*")):
        text = str(path.relative_to(ROOT)).replace("\\", "/")
        for token in NUMBER.findall(text):
            value = int(token)
            if looks_like_scenario(value):
                hits[str(value)].add(text)
    return {key: sorted(value)[:6] for key, value in sorted(hits.items(), key=lambda kv: int(kv[0]))}


def harvest_tracked_text() -> dict[str, list[str]]:
    """Scenario numbers mentioned inside tracked files, kept only when the context says scenario.

    A bare 7-digit run can also occur inside a hex digest, so a textual hit counts only when the
    neighbourhood says so; that keeps the audit sensitive to real registrations and claims without
    turning random hashes into occupied scenarios.
    """
    import subprocess

    files = subprocess.run(("git", "ls-files"), cwd=str(ROOT), capture_output=True,
                           text=True, encoding="utf-8").stdout.splitlines()
    hits: dict[str, set[str]] = defaultdict(set)
    for name in files:
        path = ROOT / name
        if not path.is_file() or path.suffix not in (".json", ".md", ".py", ".txt", ".jsonl"):
            continue
        try:
            text = path.read_text("utf-8", errors="ignore").lower()
        except OSError:
            continue
        for match in NUMBER.finditer(text):
            value = int(match.group(1))
            if not looks_like_scenario(value):
                continue
            window = text[max(0, match.start() - 60):match.end() + 60]
            if any(word in window for word in ("scenario", "seed", "scene", "block", "goal")):
                hits[str(value)].add(name)
    return {key: sorted(value)[:6] for key, value in sorted(hits.items(), key=lambda kv: int(kv[0]))}


def registered_blocks() -> dict[str, Any]:
    """Declared scene blocks: every SCENE_REGISTRY.json and every manifest scene_registry block."""
    out: dict[str, Any] = {}
    for path in sorted((ROOT / "runs").glob("*/SCENE_REGISTRY.json")):
        payload = json.loads(path.read_text("utf-8"))
        out[str(path.relative_to(ROOT)).replace("\\", "/")] = _shrink(payload)
    for path in sorted((ROOT / "runs").glob("*/CONFIG_AND_MODEL_MANIFEST.json")):
        payload = json.loads(path.read_text("utf-8"))
        if "scene_registry" in payload:
            out[str(path.relative_to(ROOT)).replace("\\", "/")] = _shrink(payload["scene_registry"])
        if "data_order" in payload:
            order = payload["data_order"]
            out[str(path.relative_to(ROOT)).replace("\\", "/") + ":data_order"] = {
                key: order[key] for key in order if "scenario" in key or "digest" in key
                or key in ("episodes_per_seed", "algorithm")}
    return out


def _shrink(payload: Any, depth: int = 0) -> Any:
    """Summarise a registration without inlining 64-element lists repeatedly."""
    if isinstance(payload, list):
        numbers = [item for item in payload if isinstance(item, int)]
        if numbers and len(numbers) == len(payload):
            return {"count": len(numbers), "min": min(numbers), "max": max(numbers),
                    "first_3": numbers[:3]}
        return [_shrink(item, depth + 1) for item in payload[:6]]
    if isinstance(payload, dict) and depth < 4:
        return {key: _shrink(value, depth + 1) for key, value in sorted(payload.items())}
    return payload


def block_status(low: int, high: int, used: set[int]) -> dict[str, Any]:
    inside = [value for value in used if low <= value <= high]
    return {"block": [low, high], "size": high - low + 1,
            "occupied_members": sorted(inside),
            "occupied_count": len(inside),
            "free": not inside}


def history_checks(low: int, high: int) -> dict[str, Any]:
    """Every committed content and every out-of-runs path, checked for the candidate block.

    A block could have been used in a branch that no longer has artefacts on disk, so the audit
    also asks git history (pickaxe over all refs) for the block's endpoints and interior sample,
    and looks for any path containing a block number outside runs/.
    """
    import subprocess

    probes = [str(low), str(high), str(low + (high - low) // 2)]
    commits = {}
    for probe in probes:
        out = subprocess.run(("git", "log", "--all", "--format=%h", "-S", probe),
                             cwd=str(ROOT), capture_output=True, text=True).stdout.split()
        commits[probe] = out
    stray = sorted(str(path.relative_to(ROOT)).replace("\\", "/")
                   for directory in (ROOT / "integrated_sar", ROOT / "piloc",
                                     ROOT / "task_allocation", ROOT / "architecture")
                   for path in directory.rglob(f"*{low}*"))
    return {"pickaxe_all_refs_per_probe": {key: len(value) for key, value in commits.items()},
           "paths_outside_runs_containing_the_block": stray[:10],
           "clean": all(not value for value in commits.values()) and not stray}


def main(argv: list[str]) -> int:
    from_paths = harvest_paths()
    from_text = harvest_tracked_text()
    used = {int(key) for key in set(from_paths) | set(from_text)}
    blocks = [block_status(low, high, used) for low, high in CANDIDATE_BLOCKS]
    chosen = next((item for item in blocks if item["free"]), None)
    history = history_checks(*chosen["block"]) if chosen else {}
    payload = {
        "kind": "round37-scene-usage-audit", "schema": "round37-scene-audit/1", "round": "37",
        "purpose": "the prompt requires auditing every existing registration, every produced "
                   "artefact and every claimed range BEFORE choosing the new 64-scenario block, "
                   "and choosing the block only from occupancy -- never from outcomes",
        "method": {
            "path_harvest": "every 6-8 digit number in every path under runs/ (files and "
                            "directories), filtered to the known scenario magnitude bands",
            "tracked_text_harvest": "the same numbers inside every tracked .json/.md/.py/.txt/"
                                    ".jsonl file",
            "registrations_read": "all runs/*/SCENE_REGISTRY.json and every "
                                  "CONFIG_AND_MODEL_MANIFEST.json scene_registry and data_order "
                                  "block",
            "outcome_data_read": "none -- no on_time, spent_bytes or any other result field is "
                                 "read by this script, so the choice cannot be informed by "
                                 "performance",
            "known_limit": "a scenario number used only inside an ignored, non-textual artefact "
                           "and absent from every path would be missed; the path harvest covers "
                           "the naming convention every round actually writes rows under"},
        "registered_blocks": registered_blocks(),
        "distinct_scenario_numbers_seen_in_paths": len(from_paths),
        "distinct_scenario_numbers_seen_in_tracked_text": len(from_text),
        "bands_in_use": _bands(used),
        "candidate_blocks_tested_in_fixed_order": blocks,
        "history_and_outside_runs_checks": history,
        "chosen_block": chosen["block"] if chosen else None,
        "chosen_block_size": chosen["size"] if chosen else 0,
        "passed": chosen is not None and bool(history.get("clean"))}
    payload["per_number_examples"] = {key: {"paths": from_paths.get(key, [])[:3],
                                            "tracked": from_text.get(key, [])[:3]}
                                      for key in sorted(set(list(from_paths)[:12]
                                                            + list(from_text)[:12]),
                                                        key=int)}
    (HERE / "SCENE_USAGE_AUDIT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"bands_in_use": payload["bands_in_use"],
                      "candidate_blocks": [{k: b[k] for k in ("block", "free", "occupied_count")}
                                           for b in blocks],
                      "chosen_block": payload["chosen_block"]}, ensure_ascii=False, indent=1))
    return 0


def _bands(used: set[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for low, high in KNOWN_BLOCKS:
        member = sorted(value for value in used if low <= value <= high)
        if member:
            out[f"{low}-{high}"] = {"distinct": len(member), "min": member[0], "max": member[-1],
                                    "members_if_small": member if len(member) <= 80 else None}
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
