"""Pure-computation checkpoint restore probe (ROUND22 spec §7, 0 launches).

Run as a subprocess by sc_check.first4_gate: load this round's checkpoint in a fresh
process, forward one fixed synthetic observation matrix through the restored controller, and
print the outputs so the parent can compare them with its own in-process forward. No
environment is constructed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import sc_common as common
import sc_trainer as trainer


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="restore probe, pure computation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)

    manifest = json.loads((common.OUTPUT_DIR / "CONFIG_AND_MODEL_MANIFEST.json").read_text(
        encoding="utf-8"))
    freeze = {"freeze_sha256": manifest["code_identity"]["freeze_sha256"],
              "config_hash": manifest["configuration_hash"],
              "branch_start": manifest["branch_start"],
              "head_at_freeze": manifest["head_at_freeze"]}
    controller, _optimizer, state = trainer.load_checkpoint(Path(args.checkpoint), freeze=freeze)
    matrix = torch.arange(19 * 4, dtype=torch.float32).reshape(4, 19) / 19.0
    with torch.no_grad():
        logits, task_values = controller.forward(matrix)
        cost_values = controller.cost_values(matrix)
    print(json.dumps({"action_logits": logits.tolist(), "task_values": task_values.tolist(),
                      "cost_values": cost_values.tolist(), "lambda": state["lambda"],
                      "episodes_done": state["episodes_done"], "updates": state["updates"],
                      "optimizer_steps": state["optimizer_steps"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
