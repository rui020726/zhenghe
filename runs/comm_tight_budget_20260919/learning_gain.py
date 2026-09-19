"""Does the corrected trainer actually learn? Paired, against its own untrained weights.

The round's headline negative -- no seed beats the strongest rule -- is only informative if the
positive half is established too: that the policy changed because of training, not because of a
sampler artefact. Round 3 could not make that claim cleanly, because its behaviour policy was not
the policy it logged. So this compares each trained checkpoint with **its own initialisation**, on
the same 24 validation streams, paired per stream, at the same credit and the same setting.

The initialisation is reconstructed from ``Controller(seed)`` -- the recorded initialisation rule --
and is labelled as a reconstruction, not a replay of an archived file: no pre-training checkpoint
was ever saved.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import tight_common as common

import evaluate_tight as E
import train_pilot


def gain(tier: str, seeds: list[int], *, setting: tuple[str, float], directory: Path) -> dict[str, Any]:
    curve = json.loads(
        (E.evaluation_root(tier) / f"curve_{tier}.json").read_text(encoding="utf-8")
    )
    budget = common.BUDGETS_BYTES[tier]
    rows: list[dict[str, Any]] = []
    for training_seed, block in sorted(curve.items()):
        available = E.inventory(tier, int(training_seed))
        wanted = {
            "init": available.get("0"),
            "selected": available.get(str(block["selected"]["at_steps"])),
            "final": available.get(str(max(int(step) for step in block["curve"]))),
        }
        for label, entry in wanted.items():
            if entry is None:
                continue
            for seed in seeds:
                rows.append(
                    E.score_row(
                        f"seed{training_seed}_{label}",
                        "init" if label == "init" else "checkpoint",
                        seed,
                        controller=entry["controller"],
                        policy="learned",
                        params="",
                        budget=budget,
                        bound=None,
                        directory=directory,
                        setting=setting,
                        provenance=entry["provenance"],
                    )
                )
        print(f"[{tier} seed {training_seed}] {list(wanted)} scored", flush=True)
    out: dict[str, Any] = {"tier": tier, "budget_bytes": budget, "setting": list(setting), "seeds": {}}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)
    for training_seed in sorted({seed.split("_")[0] for seed in grouped}):
        init_arm, chosen_arm = f"{training_seed}_init", f"{training_seed}_selected"
        final_arm = f"{training_seed}_final"
        pair = [row for row in rows if row["arm"] in {init_arm, chosen_arm, final_arm}]
        if len(pair) < 2:
            continue
        completions = E.paired(pair, baseline=init_arm, key="on_time_targets_per_episode")
        bytes_vs = E.paired(pair, baseline=init_arm, key="airtime_bytes")
        mean = lambda arm, key: round(
            float(np.mean([row[key] for row in grouped[arm]])), 4
        )
        out["seeds"][training_seed] = {
            "arms": {
                arm: {
                    "on_time_per_episode": mean(arm, "on_time_targets_per_episode"),
                    "airtime_bytes": mean(arm, "airtime_bytes"),
                    "autonomous_share_of_bytes": mean(arm, "autonomous_share_of_bytes")
                    if grouped[arm][0]["autonomous_share_of_bytes"] is not None
                    else None,
                    "optional_emissions_share": mean(arm, "optional_emissions_share"),
                    "provenance": grouped[arm][0]["provenance"],
                }
                for arm in (init_arm, chosen_arm, final_arm)
                if arm in grouped
            },
            "selected_vs_init": {
                "on_time": completions.get(chosen_arm),
                "airtime_bytes": bytes_vs.get(chosen_arm),
            },
            "final_vs_init": {
                "on_time": completions.get(final_arm),
                "airtime_bytes": bytes_vs.get(final_arm),
            },
        }
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", nargs="*", default=list(common.BUDGET_TIERS))
    parser.add_argument("--information", default="isolated")
    parser.add_argument("--delivery-probability", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    import torch

    torch.set_num_threads(max(1, int(args.threads)))
    directory = common.HERE / "evaluation" / "gain"
    directory.mkdir(parents=True, exist_ok=True)
    setting = (args.information, float(args.delivery_probability))
    payload: dict[str, Any] = {
        "statistic": E.BOOTSTRAP_NOTE,
        "initialisation_note": (
            "reconstructed from Controller(seed); round 2 saved no pre-training checkpoint, so this "
            "is reproducible by rule and not a bit-for-bit replay of an archived file"
        ),
        "runs": [],
    }
    for tier in args.tiers:
        block = gain(tier, list(common.VALIDATION_SEEDS), setting=setting, directory=directory / tier)
        payload["runs"].append(block)
        print(json.dumps(block, ensure_ascii=False)[:400], flush=True)
    key = f"{args.information}_{args.delivery_probability}"
    common.write_json(common.json_path(f"COMM_LEARNING_GAIN_{key}.json"), payload)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
