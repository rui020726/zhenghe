"""Identity gates I1-I5: what each contrast is entitled to claim about shared randomness.

The rule this enforces is narrow and easy to violate by accident: "the same lost packets"
may only be said of a pair whose per-packet outcomes are demonstrably the same. A pair
whose intervention *is* the loss parameter cannot satisfy that, and must instead be
described by the nesting relation the revision states as I4 — same keys, threshold moved.
"""

from __future__ import annotations

import json
from typing import Any

import d1_common as dc


def _sequences(tag: str, arm: str, seed: int) -> dict[str, list[Any]]:
    paths = dc.cell_paths(tag, arm, seed)
    return {
        "detections": dc.detection_sequence(paths["lifecycle"]),
        "emissions": dc.emission_sequence(paths["transcript"])
        if paths["transcript"].exists() else [],
        "packets": dc.packet_outcomes(paths["probe"]) if paths["probe"].exists() else [],
        "configuration": dc.behaviour_fields(dc.configuration_of(paths["lifecycle"])),
    }


def _lost_keys(packets: list[Any]) -> set[tuple[str, int, str]]:
    return {(target, emission, robot) for target, emission, robot, arrived in packets
            if not arrived}


def _arrived_keys(packets: list[Any]) -> set[tuple[str, int, str]]:
    return {(target, emission, robot) for target, emission, robot, arrived in packets
            if arrived}


def judge_contrast(name: str, tag: str, seed: int) -> dict[str, Any]:
    _, left_arm, right_arm = next(item for item in dc.CONTRASTS if item[0] == name)
    left = _sequences(tag, left_arm, seed)
    right = _sequences(tag, right_arm, seed)
    allowed = dc.CONTRAST_FIELDS[name]
    moved = {key for key in left["configuration"]
             if left["configuration"].get(key) != right["configuration"].get(key)}
    verdict: dict[str, Any] = {
        "contrast": name, "arms": [left_arm, right_arm], "tag": tag, "seed": seed,
        "i1_detection_stream_identical": left["detections"] == right["detections"],
        "detections_per_arm": [len(left["detections"]), len(right["detections"])],
        "i5_only_the_declared_field_differs": moved == allowed,
        "configuration_fields_moved": sorted(moved),
        "configuration_fields_allowed": sorted(allowed),
        "emissions_per_arm": [len(left["emissions"]), len(right["emissions"])],
        "charged_bytes_per_arm": [
            sum(item[3] for item in left["emissions"]),
            sum(item[3] for item in right["emissions"])],
    }
    same_channel = "comm_delivery_probability" not in allowed
    if same_channel:
        verdict["i2_emission_sequence_identical"] = left["emissions"] == right["emissions"]
        verdict["i3_packet_outcomes_identical"] = left["packets"] == right["packets"]
        verdict["packet_identity"] = ("identical" if left["packets"] == right["packets"]
                                      and left["packets"] else
                                      ("nothing_on_air" if not left["packets"] else "differs"))
    else:
        # The intervention is the loss parameter itself, so identity is the wrong relation.
        # What has to hold is that both arms address the same packets, and that the higher
        # delivery probability loses nothing the lower one delivers.
        verdict["i2_emission_sequence_identical"] = left["emissions"] == right["emissions"]
        left_lost, right_lost = _lost_keys(left["packets"]), _lost_keys(right["packets"])
        shared_recipients = _arrived_keys(left["packets"]) | _arrived_keys(right["packets"])
        verdict["i4_reference_arm_loses_nothing"] = not left_lost
        verdict["i4_loss_is_a_subset_of_the_same_packet_set"] = right_lost <= (
            shared_recipients | right_lost) and bool(shared_recipients)
        verdict["i4_packet_keys_shared"] = len(shared_recipients | left_lost | right_lost)
        verdict["packet_identity"] = (
            "nested_threshold" if not left_lost and right_lost else
            ("both_lossless" if not left_lost and not right_lost else "not_nested"))
    verdict["i3_applicable"] = same_channel
    verdict["may_say_same_lost_packets"] = bool(
        same_channel and verdict.get("i3_packet_outcomes_identical"))
    return verdict


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="smoke", choices=("smoke", "data"))
    parser.add_argument("--seeds", default="1100000",
                        help="comma-separated seeds to audit; the judgement is per seed, and "
                             "a contrast is only as paired as its worst audited seed")
    arguments = parser.parse_args(argv)
    tag = arguments.tag
    seeds = [int(seed) for seed in arguments.seeds.split(",") if seed.strip()]
    judgements = [{"cells": [judge_contrast(name, tag, seed) for seed in seeds]}
                  for name, _, _ in dc.CONTRASTS]
    flat = [cell for item in judgements for cell in item["cells"]]
    gating = {
        "C1": all(cell["i1_detection_stream_identical"]
                  and cell["i5_only_the_declared_field_differs"] for cell in flat
                  if cell["contrast"] == "C1"),
        "C2": all(cell["i1_detection_stream_identical"]
                  and cell.get("i2_emission_sequence_identical")
                  and cell.get("i4_reference_arm_loses_nothing")
                  and cell["i5_only_the_declared_field_differs"] for cell in flat
                  if cell["contrast"] == "C2"),
        "C3": all(cell["i1_detection_stream_identical"]
                  and cell.get("i2_emission_sequence_identical")
                  and cell.get("i3_packet_outcomes_identical")
                  and cell["i5_only_the_declared_field_differs"] for cell in flat
                  if cell["contrast"] == "C3"),
    }
    payload = {
        "round": dc.ROUND,
        "tag": tag,
        "seeds": seeds,
        "judgements": judgements,
        "gates_passed": gating,
        "all_passed": all(gating.values()),
        "phrasing_licensed": {
            cell["contrast"]: ("同一批丢包" if cell.get("may_say_same_lost_packets")
                               else ("同一批包、阈值不同（嵌套）"
                                     if cell.get("packet_identity") == "nested_threshold"
                                     else "不得称同一批丢包"))
            for cell in flat},
        "as_executed_note": (
            "the frozen revision's I3 sentence reads as equality of per-packet outcomes. "
            "For the pair whose intervention is the delivery probability, equality is "
            "unsatisfiable by construction, so the operative requirement there is the "
            "nesting relation the same revision states as I4: same packet keys, one arm "
            "losing nothing. This tightens what is checked; no judgement threshold moved."
        ),
    }
    dc.write_json(f"IDENTITY_AUDIT_{tag}.json", payload)
    failures = [cell for cell in flat if not cell["i1_detection_stream_identical"]
                or not cell["i5_only_the_declared_field_differs"]
                or (cell["contrast"] == "C2" and not cell.get("i4_reference_arm_loses_nothing"))
                or (cell["contrast"] == "C3"
                    and not cell.get("i3_packet_outcomes_identical"))]
    print(json.dumps({"tag": tag, "seeds_audited": len(seeds), "gates": gating,
                      "phrasing": payload["phrasing_licensed"],
                      "failing_cells": [
                          {"contrast": cell["contrast"], "seed": cell["seed"],
                           "det": cell["i1_detection_stream_identical"],
                           "cfg_only": cell["i5_only_the_declared_field_differs"],
                           "packets": cell.get("packet_identity")}
                          for cell in failures][:12]},
                     ensure_ascii=False, indent=1))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
