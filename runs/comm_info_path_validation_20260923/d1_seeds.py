"""Four-source seed collision audit for the D1 block, before any of it is recorded.

The project's rule is that a fresh seed block has to be clean against all four sources it
could already have been used in — artefacts on disk, recorded streams, ranges another round
declared in prose, and seed lists written into earlier manifests. Only the third and fourth
can be skipped by a careless reader, and skipping them is how a "held-out" set turns into
one that was already looked at.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import d1_common as dc

REPO = dc.REPO_ROOT

#: Ranges another round has declared in its own text, whether or not anything is on disk.
DECLARED = {
    "四层任务压力（第十一、十二轮）": (1000000, 1000315),
    "S3 规则独立确认（第七轮，规则工程用，不得称学习未见测试集）": (900000, 900199),
    "修正环境 400 seed 独立块 + 在线块": (500000, 500459),
    "联盟闭环在线块": (31000, 31059),
    "协调器 dev/held-out 块": (10000, 10029),
    "heldout 分析块": (20000, 20099),
    "fresh 扩展块": (21000, 21059),
    "信息边界探针": (30000, 30011),
    "封存块（未打开，仍按已用处理）": (800000, 800399),
    "第五轮训练/验证块": (700000, 700071),
    "预算 pilot 与标定块": (400000, 400299),
}

RANGE_IN_TEXT = re.compile(r"(\d{4,7})\s*\.\.\s*(\d{4,7})")


def used_by_filenames() -> set[int]:
    """Seeds appearing in any artefact name under runs/, excluding this round's own files."""
    used: set[int] = set()
    for path in (REPO / "runs").rglob("*"):
        if not path.is_file() or dc.ROUND in path.parts:
            continue
        for token in re.split(r"[^0-9]+", path.name):
            if token and len(token) in (5, 6, 7):
                used.add(int(token))
    return used


def used_by_declared_ranges() -> dict[str, tuple[int, int]]:
    out = dict(DECLARED)
    for name in ("PROJECT_STATUS.md", "PROJECT_CHARTER.md", "README.md"):
        text = (REPO / name).read_text(encoding="utf-8")
        for low, high in RANGE_IN_TEXT.findall(text):
            start, stop = int(low), int(high)
            if start < 10000 or stop - start > 100_000:
                continue
            out[f"{name}:{start}..{stop}"] = (start, stop)
    return out


def used_by_manifest_lists() -> set[int]:
    """Seed lists that earlier rounds wrote into their own JSON artefacts."""
    used: set[int] = set()
    for path in sorted((REPO / "runs").rglob("*.json")):
        if dc.ROUND in path.parts:
            continue
        if path.stat().st_size > 4_000_000:
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        used |= _seeds_in(document)
    return used


def _seeds_in(node: Any) -> set[int]:
    if isinstance(node, dict):
        out: set[int] = set()
        for key, value in node.items():
            if key in ("seed", "seeds", "stream", "stream_seed"):
                out |= _seeds_in(value)
            else:
                out |= _seeds_in(value)
        return out
    if isinstance(node, list):
        return {item for value in node for item in _seeds_in(value)}
    if isinstance(node, int) and 10_000 <= node <= 2_000_000:
        return {node}
    if isinstance(node, str) and node.isdigit() and len(node) in (5, 6, 7):
        return {int(node)}
    return set()


def run() -> int:
    block = set(dc.SEEDS)
    names = used_by_filenames()
    manifests = used_by_manifest_lists()
    ranges = used_by_declared_ranges()
    hits_range = sorted(seed for seed in block
                        for _, (low, high) in ranges.items() if low <= seed <= high)
    payload = {
        "round": dc.ROUND,
        "proposed_block": [dc.SEEDS[0], dc.SEEDS[-1]],
        "block_size": len(block),
        "sources_audited": {
            "artefact_names_on_disk": len(names),
            "seed_lists_inside_json_artefacts": len(manifests),
            "ranges_declared_in_text": len(ranges),
            "recorded_stream_names": len([
                path for path in (REPO / "runs").rglob("rec_s*.jsonl")]),
        },
        "collisions": {
            "with_artefact_names": sorted(block & names),
            "with_manifest_seed_lists": sorted(block & manifests),
            "with_declared_ranges": hits_range,
        },
        "delivered": "passed" if not (block & names) and not (block & manifests)
        and not hits_range else "collision",
        "note": (
            "a collision is not negotiable: the block is replaced and re-audited, never "
            "trimmed, because a trimmed block is a block someone has already seen the edges of"
        ),
    }
    dc.write_json("SEED_COLLISION_AUDIT.json", payload)
    print(json.dumps({key: payload[key] for key in ("proposed_block", "collisions", "delivered")},
                     ensure_ascii=False, indent=1))
    return 0 if payload["delivered"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(run())
