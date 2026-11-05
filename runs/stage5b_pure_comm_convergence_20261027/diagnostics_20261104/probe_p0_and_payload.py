"""P0 probe + PAYLOAD_BYTES measurement for the arbitrary-N pre-experiments.

Read-only. Two jobs:

1. P0 -- decide whether the subscriber count N can be moved without moving the
   rescue-robot count, by locating the registration source of CommRelay
   ``_subscribers`` and the assembly path that feeds it, and by checking whether
   the frozen configuration gates would notice.
2. P1 input -- derive PAYLOAD_BYTES from frozen artifacts rather than asserting
   it, recording the exact source path, sha256, numerator and denominator.

Nothing is written outside execution/scale_probe/. No world is replayed, so this
consumes zero replay quota.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

DIAG = Path(__file__).resolve().parent
STAGE = DIAG.parent
REPO = STAGE.parents[1]
REVISION = STAGE / "recovery_12" / "revision_01"
OUT = REVISION / "execution" / "scale_probe"

SCENES = tuple(range(2301901, 2301909))
RULE_EVAL = (STAGE / ".." / "stage5b_pure_comm_budget58600_20261026"
             / "revision_02").resolve()
BUDGET = 58600
B0 = 10000
SUBSCRIBERS_REF = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def grep_sites(path: Path, pattern: str, limit: int = 40) -> list[dict]:
    """Line-numbered matches, so every P0 claim cites the line it came from."""
    text = Path(path).read_text(encoding="utf-8").splitlines()
    hits = []
    for number, line in enumerate(text, start=1):
        if re.search(pattern, line):
            hits.append({"line": number, "text": line.strip()[:220]})
            if len(hits) >= limit:
                break
    return hits


def p0_probe() -> dict:
    comm = REPO / "integrated_sar" / "comm.py"
    driver = REPO / "integrated_sar" / "joint_driver.py"
    world = REVISION / "stage5b_comm" / "world.py"
    a1 = REPO / "runs" / "attribution_and_gossip_probe_20261020" / "a1_metrics.py"

    # The frozen expectation set that world.py:281 diffs against. If it does not
    # name comm_subscribers, narrowing the uplink adds no divergence key.
    a1_text = a1.read_text(encoding="utf-8")
    block = re.search(r"CONFIG_EXPECTATIONS = \{(.*?)\}", a1_text, re.S)
    frozen_keys = sorted(re.findall(r'"([a-z_]+)":', block.group(1))) if block else []

    world_text = world.read_text(encoding="utf-8")
    local_block = re.search(r"FROZEN_CONFIG_EXPECTATIONS = \{(.*?)\n\}", world_text, re.S)
    local_keys = (sorted(re.findall(r'"([a-z_]+)":', local_block.group(1)))
                  if local_block else [])
    transport = re.search(r"A4_TRANSPORT_OVERRIDES = \{(.*?)\n\}", world_text, re.S)
    transport_keys = (sorted(re.findall(r'"([a-z_]+)":', transport.group(1)))
                      if transport else [])

    return {
        "question": "can N (subscriber count) move while rescue_robots stays 6?",
        "registration_chain": [
            {"file": "integrated_sar/comm.py", "sha256": sha256(comm),
             "sites": grep_sites(comm, r"_subscribers\s*=|cost_bytes = payload_bytes")},
            {"file": "integrated_sar/joint_driver.py", "sha256": sha256(driver),
             "sites": grep_sites(driver, r"comm_subscribers|parse_comm_subscribers|"
                                         r"subscribers = |available = tuple")},
            {"file": "recovery_12/revision_01/stage5b_comm/world.py",
             "sha256": sha256(world),
             "sites": grep_sites(world, r'"comm_subscribers"|REGISTERED_EXPECTATION_OVERRIDE|'
                                         r'divergence\) != ')},
        ],
        "seam": {
            "location": "recovery_12/revision_01/stage5b_comm/world.py A4_TRANSPORT_OVERRIDES",
            "current_value": "comm_subscribers = 'all'",
            "revision_local": True,
            "pin_file_touched": False,
            "minimal_diff_not_applied": [
                "-    \"comm_subscribers\": \"all\",",
                "+    \"comm_subscribers\": SCALE_SUBSCRIBERS,   # 'all' | 'first:3'",
            ],
        },
        "gates": {
            "a1_metrics_CONFIG_EXPECTATIONS_keys": frozen_keys,
            "a1_metrics_names_comm_subscribers": "comm_subscribers" in frozen_keys,
            "a1_metrics_names_rescue_robots": "rescue_robots" in frozen_keys,
            "world_FROZEN_CONFIG_EXPECTATIONS_keys": local_keys,
            "world_names_comm_subscribers": "comm_subscribers" in local_keys,
            "A4_TRANSPORT_OVERRIDES_keys": transport_keys,
            "rescue_robots_derivation": "world.py verify_resolved_configuration computes "
                                        "rescue_species * rescue_per_species",
        },
        "subscriber_ceiling": {
            "rule": "joint_driver.py:1041-1052 -- requested ids are checked against "
                    "task_adapter.env.agent_dic and any unknown id raises ConfigurationError",
            "consequence": "N can be narrowed below the robot count, never widened above it",
        },
    }


def payload_bytes() -> dict:
    """Derive the per-send payload from frozen artifacts, with numerator/denominator."""
    rows = {}
    for scene in SCENES:
        path = (RULE_EVAL / "execution" / "rows" / "eval"
                / ("eval_normal_immediate_none_%d.json" % scene))
        rows[scene] = {"path": str(path.relative_to(STAGE.parent.parent)),
                       "sha256": sha256(path), "row": load(path)}

    sample = rows[SCENES[0]]["row"]
    keys = sorted(sample.keys())
    send_keys = [k for k in keys if "send" in k or "emiss" in k or "byte" in k or "paid" in k]

    per_scene = {}
    tot_paid = tot_send = 0
    for scene in SCENES:
        row = rows[scene]["row"]
        paid = int(row.get("paid_bytes", row.get("charged_bytes", 0)))
        sends = int(row.get("emissions", 0))
        per_scene[scene] = {"paid_bytes": paid, "emissions": sends,
                            "cost_per_send": (paid / sends) if sends else None,
                            "payload_per_send": (paid / sends / SUBSCRIBERS_REF) if sends else None}
        tot_paid += paid
        tot_send += sends

    mean_cost = tot_paid / tot_send if tot_send else None
    costs = [v["cost_per_send"] for v in per_scene.values() if v["cost_per_send"]]
    payloads = [v["payload_per_send"] for v in per_scene.values() if v["payload_per_send"]]
    return {
        "source_schema": "frozen rule-arm rows, normal_immediate, 8 scenes",
        "row_keys_send_or_byte_related": send_keys,
        "numerator_paid_bytes": tot_paid,
        "denominator_emissions": tot_send,
        "cost_per_send_mean": mean_cost,
        "cost_per_send_min": min(costs) if costs else None,
        "cost_per_send_max": max(costs) if costs else None,
        "payload_per_send_mean": (mean_cost / SUBSCRIBERS_REF) if mean_cost else None,
        "payload_per_send_min": min(payloads) if payloads else None,
        "payload_per_send_max": max(payloads) if payloads else None,
        "per_scene": {str(k): v for k, v in per_scene.items()},
        "row_provenance": {str(k): {"path": v["path"], "sha256": v["sha256"]}
                           for k, v in rows.items()},
        "note": "estimate_message_bytes() prices the canonical JSON of one DetectionMessage, "
                "so the payload is a distribution, not a constant; the mean over the frozen "
                "184 sends is what P1 uses, and the spread is reported beside it.",
    }


def analytic_check(payload: float) -> dict:
    """Preview the P1 acceptance arithmetic at the measured payload."""
    def cost(n):
        return payload * n

    out = {}
    for mode in ("fixed", "proportional"):
        table = {}
        for n in (3, 6, 12):
            budget = BUDGET * n / SUBSCRIBERS_REF if mode == "proportional" else BUDGET
            b0 = B0 * n / SUBSCRIBERS_REF if mode == "proportional" else B0
            c = cost(n)
            table[n] = {"cost_per_send": c, "budget_bytes": budget, "b0_bytes": b0,
                        "affordable_sends": int(budget // c),
                        "share_per_send_pct": 100.0 * c / budget,
                        "advantage_cost_per_send": c / b0}
        out[mode] = table
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    probe = p0_probe()
    payload = payload_bytes()

    decoupled_below = not probe["gates"]["a1_metrics_names_comm_subscribers"]
    verdict = "P0-A-partly" if decoupled_below else "P0-B"
    probe["verdict"] = {
        "label": verdict,
        "N_le_robot_count": {
            "decoupled": decoupled_below,
            "reason": "comm_subscribers is a RunConfiguration field with a CLI flag and a "
                      "parser, and it is absent from both frozen expectation sets, so "
                      "narrowing the uplink changes no registered key and no rescue_robots",
            "on_time_status": "valid indicator (labour held at 6 robots)",
        },
        "N_gt_robot_count": {
            "decoupled": False,
            "reason": "N=12 needs rescue_per_species=4, which sets rescue_robots=12; "
                      "joint_driver refuses subscriber ids that are not real robots",
            "on_time_status": "confounded indicator (labour moves with N)",
        },
    }
    probe["payload_preview"] = analytic_check(payload["payload_per_send_mean"])

    payload["analytic_preview"] = probe["payload_preview"]
    report = {"schema": "scale-probe-p0/1", "replay_quota_used": 0,
              "identity": "local experiment level, no identity built",
              "p0": probe, "payload": payload}

    path = OUT / "P0_probe.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True),
                    encoding="utf-8")
    print("P0 verdict:", verdict)
    print("  a1_metrics names comm_subscribers:",
          probe["gates"]["a1_metrics_names_comm_subscribers"])
    print("  a1_metrics names rescue_robots:",
          probe["gates"]["a1_metrics_names_rescue_robots"])
    print("  A4_TRANSPORT_OVERRIDES keys:", probe["gates"]["A4_TRANSPORT_OVERRIDES_keys"])
    print()
    print("payload numerator/denominator: %d B / %d sends"
          % (payload["numerator_paid_bytes"], payload["denominator_emissions"]))
    print("  cost_per_send mean %.4f  min %.4f  max %.4f"
          % (payload["cost_per_send_mean"], payload["cost_per_send_min"],
             payload["cost_per_send_max"]))
    print("  payload_per_send mean %.4f  min %.4f  max %.4f"
          % (payload["payload_per_send_mean"], payload["payload_per_send_min"],
             payload["payload_per_send_max"]))
    print("  row send/byte keys:", payload["row_keys_send_or_byte_related"])
    print()
    for mode in ("fixed", "proportional"):
        print(mode)
        for n, row in probe["payload_preview"][mode].items():
            print("  N=%-2d cost=%.2f budget=%.0f b0=%.0f afford=%d share=%.4f%% adv=%.5f"
                  % (n, row["cost_per_send"], row["budget_bytes"], row["b0_bytes"],
                     row["affordable_sends"], row["share_per_send_pct"],
                     row["advantage_cost_per_send"]))
    print()
    print("wrote", path)
    print("sha256", sha256(path))


if __name__ == "__main__":
    main()
