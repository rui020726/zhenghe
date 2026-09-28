"""ROUND24 §0 起点等价核验：本地修复 tip 与远端修复头逐条目比对（0 环境启动）。

远端修复分支 `fix/comm-cost-stage-c-training-contracts` 的 tip
`64cddcffc6f8796f132df8b8e003f005d0bdd749` 由规划端核验；本会话 git 端口不通，因此远端根目录
各条目 SHA 经已授权 GitHub MCP 按固定提交读取并落盘为 `REMOTE_HEAD_ENTRIES.json`。

本脚本把该记录与本地树逐条目比对（内容寻址：文件比对 blob SHA、目录比对 tree SHA），一致则
写 `START_EQUIVALENCE.json`（供 v2_freeze 作为冻结前置），不一致则退出非 0 并列 diff。

    /d/project_env/python.exe runs/comm_cost_v2_rerun_20260928/v2_start_equivalence.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRANCH_START = "b7217d18b4e7f4e301d0b62a810892967adc6566"
REMOTE_START_HEAD = "64cddcffc6f8796f132df8b8e003f005d0bdd749"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=HERE.parents[1], capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def local_entries() -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for line in git("ls-tree", BRANCH_START).splitlines():
        meta, name = line.split("\t", 1)
        mode, kind, sha = meta.split()
        entries[name] = {"sha": sha, "type": "dir" if kind == "tree" else "file"}
    return entries


def main() -> int:
    recorded_path = HERE / "REMOTE_HEAD_ENTRIES.json"
    if not recorded_path.exists():
        raise SystemExit("REMOTE_HEAD_ENTRIES.json is missing; the remote head is unverified")
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    if recorded.get("remote_head") != REMOTE_START_HEAD:
        raise SystemExit("the recorded remote head is not the verified fix head")
    remote = {entry["name"]: {"sha": entry["sha"], "type": entry["type"]}
              for entry in recorded["entries"]}
    local = local_entries()
    only_remote = sorted(set(remote) - set(local))
    only_local = sorted(set(local) - set(remote))
    differing = sorted(name for name in set(remote) & set(local)
                       if remote[name] != local[name])
    payload = {
        "local_head": BRANCH_START,
        "local_tree": git("rev-parse", f"{BRANCH_START}^{{tree}}"),
        "remote_head": REMOTE_START_HEAD,
        "remote_branch": recorded.get("branch"),
        "remote_source": recorded.get("source"),
        "remote_entries": remote,
        "entries_compared": len(remote),
        "entries_only_remote": only_remote,
        "entries_only_local": only_local,
        "entries_with_different_sha": differing,
        "verified": not (only_remote or only_local or differing),
        "note": "tree/blob SHAs are content-addressed: identical entry SHAs at every top-level "
                "path (including the four subtree roots) prove the full recursive trees are "
                "identical; only commit metadata differs (ROUND24 §0 anticipated this)",
    }
    (HERE / "START_EQUIVALENCE.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({key: payload[key] for key in
                      ("local_head", "remote_head", "entries_compared", "verified",
                       "entries_only_remote", "entries_only_local",
                       "entries_with_different_sha")}, ensure_ascii=False))
    return 0 if payload["verified"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
