"""ROUND67 - read-only focal attribution probe (3 forward rows, guarded).

ROUND66 preserved scene 1700027 / target-5 for every seed, including seed 33, whose full ROUND60
training lost it. R61 froze that target's first legal free first report at ready_step = 134, which
routes to the EARLY head. ROUND66 froze trunk + early_head, so this decision could not move. This
probe tests the attribution directly: forward the ROUND60 ep128 controller (read-only) on exactly
those three first-report inputs and compare with the ROUND60 ep0 controller.

Zero environment, zero sampling, zero optimizer, zero PPO, zero checkpoint write. The round-level
forward-row ledger is read and updated here so the 5000-row cap is enforced across processes, not
just inside one script.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
R66_DIR = HERE.parent
sys.path.insert(0, str(R66_DIR))
import r66_frozen as r66  # noqa: E402

LOGS = R66_DIR / "raw" / "logs"
LEDGER = HERE / "r67_forward_row_ledger.json"
CAP = 5000
FOCAL_SCENE = 1700027
FOCAL_TARGET = "piloc-e1-target-5"
HORIZON, READY_CUT = 1000, 147


def ready_step(vec):
    return round(float(vec[18]) * HORIZON) - round(float(vec[0]))


def f32_key(vec):
    return struct.pack("<" + "f" * len(vec), *[float(x) for x in vec])


def main():
    ledger = {"cap": CAP, "events": []}
    if LEDGER.is_file():
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    used = sum(e["rows"] for e in ledger["events"])
    print("ledger rows already consumed:", used, "cap", CAP)

    mods = r66.configure_source_imports()

    # The three first-report inputs, taken from the persisted R66 BC focal tapes (R60 ep0 model).
    inputs = []
    for seed in (11, 22, 33):
        path = LOGS / f"v2_r66eval2_focal_bc_s{seed}_a1_s{FOCAL_SCENE}_a1_tape.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if FOCAL_TARGET not in set(rec.get("free_first_reports") or []):
                    continue
                idx = (rec.get("targets") or []).index(FOCAL_TARGET)
                vec = rec["features"][idx]
                inputs.append({"seed": seed, "step": rec.get("step"), "ready_step": ready_step(vec),
                               "route": "late" if ready_step(vec) > READY_CUT else "early",
                               "vec": [float(x) for x in vec],
                               "saved_p_ep0": (rec.get("send_probabilities") or {})[FOCAL_TARGET],
                               "action_ep0": (rec.get("actions") or {})[FOCAL_TARGET]})
    if len(inputs) != 3:
        raise RuntimeError(f"expected 3 focal first-report inputs, found {len(inputs)}")

    rows = 3 * 1  # one model (R60 ep128), three rows
    if used + rows > CAP:
        raise RuntimeError("forward row guard: probe would exceed the round cap")
    print("charging", rows, "rows -> total", used + rows)

    out = []
    with torch.no_grad():
        for item in inputs:
            controller, identity = r66.load_r60_checkpoint_read_only(item["seed"], "ep128")
            mat = torch.tensor([item["vec"]], dtype=torch.float32)
            logits, _ = controller.forward(mat)
            probs = torch.softmax(logits, dim=-1)
            p_send = float(probs[0, 1])
            out.append({"seed": item["seed"], "step": item["step"],
                        "ready_step": item["ready_step"], "route": item["route"],
                        "r60_ep0_p_send": item["saved_p_ep0"],
                        "r60_ep0_action": item["action_ep0"],
                        "r60_ep128_p_send": p_send,
                        "r60_ep128_greedy_action": 1 if p_send >= 0.5 else 0,
                        "abs_dp_ep0_to_ep128": abs(p_send - item["saved_p_ep0"]),
                        "r60_ep128_file_sha256": identity["source_file_sha256"],
                        "r60_ep128_episodes_done": identity["episodes_done"],
                        "r60_ep128_updates": identity["updates"]})
    ledger["events"].append({"kind": "round67-focal-attribution-probe/1", "rows": rows,
                             "models": "R60 ep128 (read-only), 3 rows",
                             "detail": [f"s{o['seed']}" for o in out]})
    LEDGER.write_text(json.dumps(ledger, indent=1, ensure_ascii=False), encoding="utf-8")
    dest = HERE / "r67_focal_attribution_probe.json"
    dest.write_text(json.dumps({"kind": "round67-focal-attribution-probe/1",
                                "canonical_modules": mods.get("canonical_modules"),
                                "scene": FOCAL_SCENE, "target": FOCAL_TARGET,
                                "rows_used": rows,
                                "round_forward_rows_after": used + rows,
                                "cap": CAP,
                                "records": out}, indent=1, ensure_ascii=False), encoding="utf-8")
    for o in out:
        print(f"s{o['seed']} step={o['step']} ready={o['ready_step']} route={o['route']} "
              f"R60ep0 p={o['r60_ep0_p_send']:.6f} act={o['r60_ep0_action']} | "
              f"R60ep128 p={o['r60_ep128_p_send']:.6f} act={o['r60_ep128_greedy_action']} "
              f"|dp|={o['abs_dp_ep0_to_ep128']:.6f}")
    print("wrote", dest)


if __name__ == "__main__":
    main()
