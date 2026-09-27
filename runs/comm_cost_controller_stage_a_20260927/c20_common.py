"""第二十轮 Stage A 的公共件：路径、哈希与写盘沙箱。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ROUND_DIR = Path(__file__).resolve().parent
SPECS_DIR = REPO_ROOT.parent / "specs"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(name: str, payload: Any) -> Path:
    """写一件本轮产物；拒绝覆盖已交付件（旧件隔离到 rejected/，不删除）。"""
    if os.sep in name or "/" in name or ".." in name:
        raise ValueError(f"refusing to write outside the round directory: {name!r}")
    target = ROUND_DIR / name
    if target.exists():
        raise RuntimeError(f"refusing to rewrite a delivered artefact: {name}")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                      encoding="utf-8")
    return target


def git(*args: str) -> str:
    import subprocess
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def path() -> str:
    return os.environ.get("PATH", "")
