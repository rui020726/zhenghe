"""ROUND25 场景碰撞审计：首次场景生成前的全项目只读元数据核验。

执行依据 §三：首次生成前完成全项目 seed/用途/已 claim 块碰撞审计，只查元数据，不读取
保护集内容；优先块 1400000..1400063，若冲突则按预定算法从 1400000..1499999 升序取前 64 个
未占用 seed；先冻结完整列表再生成。800000..800399、900000..900199 不触碰。

本脚本零环境启动，只读：
- 版本库（zhenghe）与同级工作目录的文件名（元数据）；
- 文本内容中是否出现目标块号段（只做 grep 级定位，不读取保护集内容产物）。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]          # D:\\code\\整合
ROOT = HERE.parents[1]               # zhenghe repo
REPO = ROOT

PREFERRED_LOW, PREFERRED_HIGH = 1400000, 1400063
FALLBACK_LOW, FALLBACK_HIGH = 1400000, 1499999
SCENE_COUNT = 64

#: Registered claim blocks (ROUND24 SCENE_REGISTRY + this round's instruction). Only metadata of
#: these blocks is consulted; their contents are never read.
CLAIMED_RANGES = [
    ("sealed_test_set", 800000, 800399, "ROUND22 registry; instruction §3 repeats the do-not-touch rule"),
    ("learned_model_reported_stream", 900000, 900199, "ROUND22 registry; instruction §3"),
    ("systemloop_20260918", 31000, 31059, "ROUND22 registry"),
    ("task_pressure_g6", 1000000, 1000015, "ROUND22 registry"),
    ("task_pressure_g8", 1000100, 1000115, "ROUND22 registry"),
    ("task_pressure_g10", 1000200, 1000215, "ROUND22 registry"),
    ("task_pressure_g12", 1000300, 1000315, "ROUND22 registry"),
    ("D1_20260923", 1100000, 1100059, "ROUND22 registry"),
    ("search_trace_and_policy_20260925", 1200000, 1200059, "ROUND22 registry"),
    ("ROUND22_24_training_scenarios", 1300000, 1300047, "ROUND24 SCENE_REGISTRY"),
    ("ROUND22_24_development_scenarios", 1301000, 1301015, "ROUND24 SCENE_REGISTRY (development history)"),
]
OWN_DIR = "runs/comm_cost_greedy_replication_20260928"

STRONG_TOKEN = re.compile(r"(?<![0-9A-Za-z])s(\d{4,10})(?![0-9])")
ANY_NUMBER = re.compile(r"(?<![0-9])(\d{6,10})(?![0-9])")
DATE_LIKE = re.compile(r"^202[0-9](0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])$")
TARGET_MENTION = re.compile(r"(?<![0-9])1[45]0\d{4}(?![0-9])")
HEX_RUN = re.compile(r"[0-9a-fA-F]{16,}")
SEED_WORDS = ("seed", "scenario", "场景", "种子", "block", "流")

TEXT_SUFFIXES = {".md", ".py", ".json", ".txt", ".cfg", ".toml", ".yaml", ".yml"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
#: Per-episode artefact directories: their names carry the seeds (step 1 scans every filename);
#: their contents are volume data and are never read by this audit.
ARTEFACT_DIRS = {"rows", "logs", "checkpoints", "rejected", "streams", "paired", "artefacts",
                 "data", "tape", "tapes", "records"}


def iter_files(base: Path):
    for path in base.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def inside(seed: int, low: int, high: int) -> bool:
    return low <= seed <= high


def main() -> int:
    scopes = {"repo": REPO,
              "siblings": [p for p in sorted(WORKSPACE.iterdir())
                           if p.is_dir() and p.name != REPO.name]}
    findings: dict[str, object] = {}

    # ---- 1. filename-embedded seeds (metadata only) --------------------------------------
    strong: dict[int, list[str]] = {}
    weak_unclassified: dict[int, list[str]] = {}
    scanned_files = 0
    scan_targets = [("repo", REPO)] + [("sibling", path) for path in scopes["siblings"]]
    for scope, base in scan_targets:
        for path in iter_files(base):
            scanned_files += 1
            name = path.name
            rel = path.relative_to(base).as_posix()
            label = f"{scope}:{rel}" if scope == "sibling" else f"{scope}:{rel}"
            for token in STRONG_TOKEN.findall(name):
                strong.setdefault(int(token), []).append(label)
            for token in ANY_NUMBER.findall(name):
                value = int(token)
                if DATE_LIKE.match(token):
                    continue
                if value in strong:
                    continue
                weak_unclassified.setdefault(value, []).append(label)
    in_target_strong = {seed: files for seed, files in sorted(strong.items())
                        if inside(seed, PREFERRED_LOW, PREFERRED_HIGH)}
    in_target_weak = {seed: files for seed, files in sorted(weak_unclassified.items())
                      if inside(seed, PREFERRED_LOW, PREFERRED_HIGH)}
    findings["filename_seeds"] = {
        "files_scanned": scanned_files,
        "strong_tokens_in_target_block": in_target_strong,
        "weak_unclassified_tokens_in_target_block": in_target_weak,
        "strong_tokens_total": len(strong),
        "weak_unclassified_total": len(weak_unclassified),
        "note": "strong = s<digits> tokens in artefact names; weak = bare 6-10 digit numbers "
                "excluding date-like values; neither reads file contents here",
    }

    # ---- 2. claimed ranges: registered + anything new in text ---------------------------
    registered = [{"name": name, "low": low, "high": high, "source": source}
                  for name, low, high, source in CLAIMED_RANGES]
    mentions: list[dict[str, object]] = []
    for scope, base in scan_targets:
        for path in iter_files(base):
            if path.suffix not in TEXT_SUFFIXES:
                continue
            if any(part in ARTEFACT_DIRS for part in path.parts):
                continue
            # Per-episode artefacts are volume data whose names already entered step 1; only
            # scan human-authored text (docs, protocols, harness code).
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, ValueError, OSError):
                continue
            if not TARGET_MENTION.search(text):
                continue
            rel = path.relative_to(base).as_posix()
            own = scope == "repo" and rel.startswith(OWN_DIR + "/")
            samples = []
            range_like = False
            for match in list(TARGET_MENTION.finditer(text))[:5]:
                start, end = match.span()
                context = text[max(0, start - 40):min(len(text), end + 40)].replace("\n", " ")
                if HEX_RUN.search(context):
                    classification = "incidental_hex_digest"
                elif any(word in context.lower() for word in SEED_WORDS):
                    classification = "seed_vocabulary"
                    range_like = True
                else:
                    classification = "incidental_number"
                samples.append({"match": match.group(0), "classification": classification,
                                "context": context})
            mentions.append({"scope": scope, "path": rel, "own_round_artefact": bool(own),
                             "range_like": bool(range_like), "samples": samples})
    preexisting = [m for m in mentions if not m["own_round_artefact"]]
    preexisting_claims = [m for m in preexisting if m["range_like"]]
    # The round's own planning document (old spec, offline MISMATCH status on file) is where the
    # preferred block was first written down; it is this round's registration, not a rival claim.
    own_planning_docs = [m for m in preexisting_claims
                         if m["path"] == "ROUND25_COMM_COST_GREEDY_REPLICATION.md"]
    rival_claims = [m for m in preexisting_claims if m not in own_planning_docs]
    findings["claimed_ranges"] = {
        "registered": registered,
        "new_text_mentions_of_1.4m_block": mentions,
        "pre_existing_mentions_outside_this_round": preexisting,
        "pre_existing_seed_vocabulary_mentions": preexisting_claims,
        "own_planning_documents": own_planning_docs,
        "rival_pre_existing_claims": rival_claims,
        "note": "the block is registered here for the first time by this round; a pre-existing "
                "mention outside this round's own artefacts would be inspected before any scene "
                "is generated",
    }

    # ---- 3. overlap arithmetic -----------------------------------------------------------
    target = (PREFERRED_LOW, PREFERRED_HIGH)
    overlaps = [entry for entry in registered
                if not (entry["high"] < target[0] or entry["low"] > target[1])]
    fallback_free = True
    for seed in range(FALLBACK_LOW, FALLBACK_HIGH + 1):
        if any(entry["low"] <= seed <= entry["high"] for entry in registered):
            fallback_free = False
            break
        if seed in strong or seed in weak_unclassified:
            fallback_free = False
            break
        if seed >= FALLBACK_LOW + SCENE_COUNT - 1:
            break
    blocked = bool(in_target_strong or in_target_weak or overlaps or rival_claims
                   or not fallback_free)

    # ---- 4. the scene list, only if clean -------------------------------------------------
    scenes = list(range(PREFERRED_LOW, PREFERRED_HIGH + 1))
    findings["decision"] = {
        "preferred_block": [PREFERRED_LOW, PREFERRED_HIGH],
        "preferred_block_free": not blocked,
        "fallback_rule": f"ascending from {FALLBACK_LOW}, first {SCENE_COUNT} unclaimed seeds",
        "fallback_first_64_available": fallback_free,
        "scenes_selected": scenes if not blocked else [],
        "blocked": blocked,
        "stop_rule": "if blocked, stop and report before generating any scenario",
    }

    (HERE / "SEED_COLLISION_AUDIT.json").write_text(
        json.dumps(findings, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"blocked": blocked,
                      "strong_in_target": len(in_target_strong),
                      "weak_in_target": len(in_target_weak),
                      "overlaps_with_registered_ranges": len(overlaps),
                      "own_planning_documents": len(own_planning_docs),
                      "rival_pre_existing_claims": len(rival_claims),
                      "fallback_first_64_available": fallback_free}, ensure_ascii=False))
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
