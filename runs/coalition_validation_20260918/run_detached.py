"""Launch a long-running experiment process detached from this shell.

The parent it starts owns the shard manifest, the preflight and the aggregation, so the
single-writer discipline of the brief holds; detaching only keeps it alive past the
calling shell's timeout. Progress is polled from the artefacts it writes.

    D:/project_env/python.exe runs/coalition_validation_20260918/run_detached.py <script> [args...]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000  # so a killed job does not take the run with it


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: run_detached.py <script> [args...]")
        return 2
    script = HERE / argv[0] if not Path(argv[0]).is_absolute() else Path(argv[0])
    log = HERE / f"{script.stem}.log"
    command = [sys.executable, str(script), *argv[1:]]
    with log.open("a", encoding="utf-8") as sink:
        flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        try:
            process = subprocess.Popen(
                command, cwd=str(HERE.parent.parent), stdout=sink, stderr=subprocess.STDOUT,
                creationflags=flags | CREATE_BREAKAWAY_FROM_JOB, stdin=subprocess.DEVNULL,
            )
        except OSError:
            # breakaway is not always permitted; the other flags still detach the tree
            process = subprocess.Popen(
                command, cwd=str(HERE.parent.parent), stdout=sink, stderr=subprocess.STDOUT,
                creationflags=flags, stdin=subprocess.DEVNULL,
            )
    print(f"detached pid {process.pid} -> {log.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
