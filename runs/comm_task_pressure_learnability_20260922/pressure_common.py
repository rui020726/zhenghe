"""Fixtures for the task-pressure learnability calibration: one variable, nine locked arms.

This round trains nothing. It asks a narrower question -- whether raising *only* the number of
targets turns the locked 29,300 B credit into a constraint that fixed rules cannot all cover --
so the fixtures are the previous rounds' frozen world with exactly one field changed.

Everything that could bend a cost/effect comparison is imported rather than restated: the byte
credit is read from the calibration file that fixed it, the nine arms come from the protocol-v2
arm tables that produced the archived rule matrix, the recording configuration is the frozen
block's own constant, and the paired bootstrap is round 2's. The only new declarations here are
this round's four seed blocks, its layer order, and the labels of the launch ledger.

    python pressure_common.py     # self-check: the frozen things are still frozen
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT_ROUND = ROOT / "runs" / "comm_budget_pilot_20260919"
TIGHT_ROUND = ROOT / "runs" / "comm_tight_budget_20260919"
PROTOCOL_ROUND = ROOT / "runs" / "comm_protocol_v2_20260920"
ACK_ROUND = ROOT / "runs" / "comm_ack_value_20260920"
CONFIRM_ROUND = ROOT / "runs" / "comm_rule_confirm_20260921"
MASK_FIX_ROUND = ROOT / "runs" / "comm_budget_action_mask_fix_20260922"
RETRAIN_ROUND = ROOT / "runs" / "comm_budget_masked_retrain_20260922"

for _directory in (
    str(ROOT),
    str(PILOT_ROUND),
    str(TIGHT_ROUND),
    str(PROTOCOL_ROUND),
    str(ACK_ROUND),
    str(CONFIRM_ROUND),
    str(MASK_FIX_ROUND),
    str(RETRAIN_ROUND),
    str(HERE),
):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

pilot = importlib.import_module("pilot_common")
protocol_common = importlib.import_module("protocol_common")
protocol_env = importlib.import_module("protocol_env")
tight_common = importlib.import_module("tight_common")
confirm_common = importlib.import_module("confirm_common")
confirm_env = importlib.import_module("confirm_env")

#: Every round above prepends its own directory when it is imported, and two of them ship modules
#: named the same as this round's (`rule_matrix` exists in protocol-v2). Put this round's directory
#: back in front, after those imports, so ``import rule_matrix`` cannot silently resolve to the
#: five-round-old script and mix two harnesses into one aggregate.
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

# ---------------------------------------------------------------- the single variable
#: num_goals -> the sixteen episode seeds registered for that layer. Blocks are disjoint, and
#: every one of them is above 1_000_000 so no earlier round's arithmetic can reach them.
LAYERS: dict[int, tuple[int, ...]] = {
    6: tuple(range(1_000_000, 1_000_016)),
    8: tuple(range(1_000_100, 1_000_116)),
    10: tuple(range(1_000_200, 1_000_216)),
    12: tuple(range(1_000_300, 1_000_316)),
}

#: Layers are executed in this order and the round stops at the first layer that passes L0-L4.
LAYER_ORDER: tuple[int, ...] = (6, 8, 10, 12)

STREAMS_PER_LAYER = 16

# ---------------------------------------------------------------- frozen setting
#: The locked 25 % credit, read from the calibration file exactly as rounds 5/7/10 read it.
TIER = protocol_common.PRIMARY_TIER
BUDGET_BYTES = int(protocol_common.tier_bytes(TIER))

#: The only information path this round measures: uplink isolated (gossip above the horizon),
#: lossless, no reverse channel. Rounds 1-10 established that the normal path bypasses the
#: sender, so a pressure calibration on it would measure the bypass.
INFORMATION = "isolated"
DELIVERY_PROBABILITY = 1.0
ACK_ENABLED = False

TIMING_MODE = protocol_common.POLICY_MODE
BOUND_PARAMS = protocol_common.BOUND_PARAMS
BOUND_MAX_WAIT_STEPS = protocol_common.BOUND_MAX_WAIT_STEPS
BOUND_URGENT_WAIT_STEPS = protocol_common.BOUND_URGENT_WAIT_STEPS

STREAM_STEPS = int(pilot.STREAM_STEPS)

#: The frozen recording world. ``num_goals`` is the only key a layer may change, and
#: :func:`recording_configuration` refuses any other difference from this dict.
FROZEN_RECORD_KEYS = (
    "steps",
    "device",
    "piloc_checkpoint",
    "task_checkpoint",
    "mock",
    "use_coordinator",
    "search_agents",
    "grid_size",
    "obstacle_density",
    "detection_radius",
    "deadline_tightness",
    "dt",
    "rescue_species",
    "rescue_per_species",
    "actor_kv_mode",
    "comm_delivery_probability",
    "comm_delay_ticks",
    "comm_subscribers",
    "quiet",
    "task_membership_consistency",
)

RECORD_CONFIGURATION = dict(pilot.RECORD_CONFIGURATION)
BASE_CONFIGURATION = dict(pilot.base.arms.ARMS["C"].configuration)

# ---------------------------------------------------------------- locked rule library
#: The nine arms in the brief's order. ``competition`` arms may be selected as the best fixed
#: rule and enter the per-stream hindsight envelope; ``always_hold`` is the communication floor
#: and ``event_unbudgeted`` an out-of-credit reference -- neither is ever a competitor.
ARM_ORDER: tuple[str, ...] = (
    "send_once",
    "event",
    "event_priority",
    "periodic_5",
    "periodic_20",
    "first_send_plus_bound_3_10",
    "bound_only_3_10",
    "always_hold",
    "event_unbudgeted",
)
COMPETITION_ARMS: tuple[str, ...] = ARM_ORDER[:7]
CONTROL_ARMS: tuple[str, ...] = ("always_hold",)
CEILING_ARMS: tuple[str, ...] = ("event_unbudgeted",)

#: The arm's role, written down before any result exists so "best fixed rule" cannot drift.
ARM_ROLES: dict[str, str] = {
    arm: ("ceiling_reference" if arm in CEILING_ARMS else "control" if arm in CONTROL_ARMS else "competitor")
    for arm in ARM_ORDER
}


def arm_spec(arm: str) -> dict[str, Any]:
    """The protocol-v2 table entry for one arm, plus the role this round assigned it.

    Parameters are not re-selected here: ``periodic_5`` is ``interval=5`` because round 5 ran it
    that way, and the bound-named rules carry round 5's declared ``BOUND_PARAMS``.
    """
    if arm not in ARM_ORDER:
        raise ValueError(f"unknown arm {arm!r}; this round locks {ARM_ORDER}")
    table = "ceiling" if arm in CEILING_ARMS else "v2"
    spec = protocol_common.arm_spec(table, arm)
    spec["arm"] = arm
    spec["table"] = table
    spec["role"] = ARM_ROLES[arm]
    spec["bounded"] = spec.get("bound") != "none"
    spec["unbudgeted"] = bool(spec.get("ceiling"))
    return spec


# ---------------------------------------------------------------- paths
STREAMS = HERE / "streams"
ROWS = HERE / "rows"
RUNS_DIR = HERE / "runs"


def layer_streams(goals: int) -> Path:
    _require_layer(goals)
    return STREAMS / f"g{goals}"


def layer_rows(goals: int) -> Path:
    _require_layer(goals)
    return ROWS / f"g{goals}"


def layer_runs(goals: int) -> Path:
    _require_layer(goals)
    return RUNS_DIR / f"g{goals}"


def stream_path(goals: int, seed: int) -> Path:
    return layer_streams(goals) / f"s{seed}.json"


def _require_layer(goals: int) -> None:
    if goals not in LAYERS:
        raise ValueError(f"unknown pressure layer {goals!r}; this round runs {LAYER_ORDER}")


# ---------------------------------------------------------------- configuration builders
def recording_configuration(goals: int) -> dict[str, Any]:
    """The frozen recording configuration with ``num_goals`` as the only change."""
    _require_layer(goals)
    values = dict(RECORD_CONFIGURATION)
    values["num_goals"] = int(goals)
    return values


#: The fields the recorded world and the replayed world must agree on. Everything else the two
#: configurations legitimately differ on belongs to the allocation side -- the frozen C arm runs
#: with the coordinator on while the recorder searched with it off, exactly as rounds 2-10 did.
SHARED_WORLD_KEYS = (
    "steps",
    "device",
    "mock",
    "search_agents",
    "num_goals",
    "grid_size",
    "obstacle_density",
    "detection_radius",
    "deadline_tightness",
    "dt",
    "rescue_species",
    "rescue_per_species",
    "actor_kv_mode",
    "comm_delivery_probability",
    "comm_delay_ticks",
    "comm_subscribers",
    "quiet",
    "task_membership_consistency",
    "task_checkpoint",
)

#: Declared, not asserted equal: the C arm's allocation-side settings that the recorder does not
#: use. Listing them keeps "the replay is the frozen C arm" checkable instead of implicit.
DECLARED_REPLAY_ONLY_KEYS = (
    "use_coordinator",
    "coordinator_policy",
    "allocation_interval",
    "coordination_interval",
    "command_ttl",
    "residual_scale",
    "sticky_commitments",
    "suppress_recommand_own_task",
    "release_redundant_members",
    "check_release_source_stage",
)


def assert_only_num_goals_changed(goals: int) -> dict[str, Any]:
    """Prove the claim above field by field instead of asserting it in prose."""
    values = recording_configuration(goals)
    differences = {
        key: (RECORD_CONFIGURATION.get(key), values.get(key))
        for key in set(RECORD_CONFIGURATION) | set(values)
        if RECORD_CONFIGURATION.get(key) != values.get(key)
    }
    if not set(differences) <= {"num_goals"}:
        raise RuntimeError(
            f"layer {goals} would change {sorted(differences)}; num_goals is the only "
            "independent variable this round may move"
        )
    base = dict(BASE_CONFIGURATION)
    # num_goals is moved on both sides at once: the layer's replay must answer to the layer's own
    # recorded world, and ``assert_bundle_matches_configuration`` enforces it again at runtime.
    base["num_goals"] = int(goals)
    world = {
        key: (base.get(key, "<absent>"), values.get(key, "<absent>"))
        for key in SHARED_WORLD_KEYS
        if base.get(key, "<absent>") != values.get(key, "<absent>")
    }
    if world:
        raise RuntimeError(
            f"the recorded world and the replayed world disagree on {sorted(world)}: a replay "
            "would be answering to a different scenario than the one that was searched"
        )
    return {"goals": int(goals), "changed": sorted(differences), "world_conflicts": world}


def replay_configuration(
    goals: int, seed: int, *, directory: Path, label: str, stream: Path | None = None
) -> dict[str, Any]:
    """The frozen replay arm's configuration, pointed at this layer's own stream.

    ``stream`` exists for one purpose: the agreement check replays an *archived* round-5 stream
    through this round's runner, which must not be able to reach a layer's data by accident.
    """
    _require_layer(goals)
    stream = stream_path(goals, seed) if stream is None else Path(stream)
    if not stream.exists():
        raise FileNotFoundError(
            f"no recorded stream for layer {goals} seed {seed}: {stream}; record the layer "
            "first -- a missing stream never falls back to another round's data"
        )
    values = dict(BASE_CONFIGURATION)
    values.update(
        replay_stream=str(stream),
        seed=int(seed),
        num_goals=int(goals),
        steps=STREAM_STEPS,
    )
    values["output"] = str(Path(directory) / f"rp_{label}_g{goals}_s{seed}.jsonl")
    values["decision_trace"] = str(Path(directory) / f"tr_{label}_g{goals}_s{seed}.jsonl")
    values["comm_transcript"] = str(Path(directory) / f"ct_{label}_g{goals}_s{seed}.jsonl")
    return values


# ---------------------------------------------------------------- json helpers
def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_path(name: str) -> Path:
    return HERE / name


def read_json(name: str) -> Any:
    return json.loads(json_path(name).read_text(encoding="utf-8"))


#: The delivered round's evidence. This branch may read it, re-derive it and quote it, but it may
#: not rewrite it into something else (extension protocol §2): a g12 result must never look like a
#: post-hoc edit of the gate that said no. A write that reproduces the file byte for byte is still
#: allowed, which makes every such write a recomputation check rather than a risk.
DELIVERED_ARTEFACTS: tuple[str, ...] = (
    "COMM_TASK_PRESSURE_LEARNABILITY_PROTOCOL.md",
    "COMM_TASK_PRESSURE_LEARNABILITY_RESULTS.md",
    "COMM_TASK_PRESSURE_LEARNABILITY_RESULTS.json",
    "COMM_TASK_PRESSURE_LEARNABILITY_MANIFEST.json",
    "SEED_COLLISION_AUDIT.json",
    "AGREEMENT_GATE.json",
    "GATE_SEQUENCE_EXPLANATION.json",
    "CODE_FREEZE.json",
    "RULE_MATRIX.json",
    "PRESSURE_METRICS.json",
    *(
        f"{prefix}_g{goals}.json"
        for goals in (6, 8, 10)
        for prefix in ("GATES", "STREAM_MANIFEST", "ROW_PROVENANCE", "RULE_MATRIX", "PRESSURE_METRICS")
    ),
)


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # No trailing newline: the delivered round's artefacts were written this way, and the guard
    # below compares against those bytes.
    text = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True)
    if path.name in DELIVERED_ARTEFACTS and path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(
                f"{path.name} is the delivered round's evidence and the new content differs: an "
                "extension round must write its own files, not rewrite that verdict "
                "(COMM_TASK_PRESSURE_G12_EXTENSION_PROTOCOL.md §2)"
            )
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def shard_of(items: list[Any], shards: int, shard: int) -> list[Any]:
    """The one and only place sharding is applied, so two shards cannot overlap."""
    return [item for index, item in enumerate(items) if index % max(1, shards) == shard % max(1, shards)]


# ---------------------------------------------------------------- statistics
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_SEED = 20260922

#: Labels whose order the bootstrap walks. Registered so a rerun cannot reorder the draws and
#: silently move a quantile (round 8's lesson: identical means, different CI boundaries).
BOOTSTRAP_LABEL_ORDER: tuple[str, ...] = ("best_fixed", "envelope", *ARM_ORDER)


def paired_differences(
    values: dict[str, dict[int, float]], *, baseline: str
) -> dict[str, Any]:
    """Round 2's seed-level paired bootstrap, fed by (label -> {seed: value}) maps.

    One call draws one RNG stream and answers every label against one baseline, so each
    (axis, baseline) statistic has exactly one confidence interval in this round. Callers must
    not re-draw a subset of the rows to get a number that "looks" on a table.
    """
    if baseline not in values:
        raise KeyError(f"baseline {baseline!r} is not among {sorted(values)}")
    rows: list[dict[str, Any]] = []
    for label in BOOTSTRAP_LABEL_ORDER:
        for seed, value in sorted(values.get(label, {}).items()):
            rows.append({"arm": label, "seed": int(seed), "value": float(value)})
    import evaluate_pilot

    evaluate_pilot.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    evaluate_pilot.BOOTSTRAP_REPLICATIONS = BOOTSTRAP_REPLICATIONS
    result = evaluate_pilot.paired(rows, baseline=baseline, extract=lambda row: float(row["value"]))
    result["bootstrap_seed"] = BOOTSTRAP_SEED
    result["replications"] = BOOTSTRAP_REPLICATIONS
    result["label_order"] = list(BOOTSTRAP_LABEL_ORDER)
    return result


# ---------------------------------------------------------------- launch accounting
#: 4 layers x (16 recordings + 16 streams x 9 arms) = 640 data launches. Gates and the harness's
#: own agreement checks are recorded under ``gate`` kinds against their own small ceiling, and
#: every one of them is reported: the ceiling is the brief's arithmetic, not a place to hide cost.
MATRIX_LAUNCH_CAP = 640
GATE_LAUNCH_CAP = 12
MATRIX_KINDS = ("record", "matrix")
GATE_KINDS = ("gate", "selftest")

#: The g12 extension's grant, stated as a difference so the original number survives on disk. The
#: fourth layer was pre-registered and never ran because 20 launches went to this round's own
#: defects, and failures spend the ceiling; the user then authorised exactly the 16 launches that
#: layer is short by. Every one of the 496 launches already spent still counts against 656.
G12_EXTENSION_GRANT = 16
MATRIX_LAUNCH_CAP_WITH_GRANT = MATRIX_LAUNCH_CAP + G12_EXTENSION_GRANT
G12_GOALS = 12

LEDGER_PATTERN = "LAUNCH_LEDGER*.json"
LEDGER_FILE = "LAUNCH_LEDGER.json"

#: Which freeze file governs a layer. The three delivered layers were produced under
#: ``CODE_FREEZE.json``; the extension registers its own file next to it rather than replacing it,
#: because a freeze is evidence about a specific set of episodes and rewriting one erases that.
FREEZE_FILES = {goals: "CODE_FREEZE.json" for goals in LAYER_ORDER}
FREEZE_FILES[G12_GOALS] = "G12_FREEZE.json"

#: Which seed audit authorises a layer's recording, for the same reason.
AUDIT_FILES = {goals: "SEED_COLLISION_AUDIT.json" for goals in LAYER_ORDER}
AUDIT_FILES[G12_GOALS] = "G12_SEED_COLLISION_AUDIT.json"


def audit_file(goals: int) -> str:
    _require_layer(goals)
    return AUDIT_FILES[int(goals)]


def freeze_file(goals: int) -> str:
    _require_layer(goals)
    return FREEZE_FILES[int(goals)]


def launch_grant_for(goals: int) -> dict[str, Any]:
    """The ceiling a layer runs under, and whether it needed an extension to run at all."""
    _require_layer(goals)
    extended = int(goals) == G12_GOALS
    return {
        "goals": int(goals),
        "brief_ceiling": MATRIX_LAUNCH_CAP,
        "effective_ceiling": MATRIX_LAUNCH_CAP_WITH_GRANT if extended else MATRIX_LAUNCH_CAP,
        "extension_grant": G12_EXTENSION_GRANT if extended else 0,
        "granted_by": (
            "the user's explicit order of 2026-09-23, recorded in "
            "COMM_TASK_PRESSURE_G12_EXTENSION_PROTOCOL.md §2"
            if extended
            else ""
        ),
    }


def ledger_totals() -> dict[str, Any]:
    """Every ledger file added up: the round's real cost across processes and reruns."""
    totals: dict[str, Any] = {
        "matrix_cap": MATRIX_LAUNCH_CAP,
        "matrix_cap_with_grant": MATRIX_LAUNCH_CAP_WITH_GRANT,
        "extension_grant": G12_EXTENSION_GRANT,
        "gate_cap": GATE_LAUNCH_CAP,
        "starts": 0,
        "matrix_starts": 0,
        "gate_starts": 0,
        "failures": 0,
        "by_kind": {},
        "files": [],
    }
    for path in sorted(HERE.glob(LEDGER_PATTERN)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        totals["files"].append(path.name)
        totals["starts"] += int(payload.get("starts", 0))
        totals["failures"] += int(payload.get("failures", 0))
        for kind, count in (payload.get("by_kind") or {}).items():
            totals["by_kind"][kind] = int(totals["by_kind"].get(kind, 0)) + int(count)
    totals["matrix_starts"] = sum(
        int(count) for kind, count in totals["by_kind"].items() if kind in MATRIX_KINDS
    )
    totals["gate_starts"] = sum(
        int(count) for kind, count in totals["by_kind"].items() if kind in GATE_KINDS
    )
    return totals


class LaunchLedger:
    """One file per process, one global ceiling.

    Shards run in parallel and the payload is read-modify-write, so a shared file would let each
    process overwrite the other's counts -- the round would under-count its own cost. The cap is
    still global because :meth:`reserve` reads every ledger file before it allows a launch.
    A launch that later failed is still counted: failures spend the ceiling.
    """

    def __init__(self, name: str = LEDGER_FILE) -> None:
        self.path = HERE / name
        self.payload = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists()
            else {"starts": 0, "by_kind": {}, "entries": [], "failures": 0}
        )

    @property
    def starts(self) -> int:
        return int(self.payload["starts"])

    def reserve(self, kind: str, goals: int, seed: int, *, arm: str = "") -> None:
        if kind not in MATRIX_KINDS + GATE_KINDS:
            raise ValueError(
                f"unknown launch kind {kind!r}; ledger kinds are {MATRIX_KINDS + GATE_KINDS}"
            )
        totals = ledger_totals()
        if kind in MATRIX_KINDS:
            grant = launch_grant_for(goals)
            cap, used = grant["effective_ceiling"], totals["matrix_starts"]
        else:
            cap, used = GATE_LAUNCH_CAP, totals["gate_starts"]
        if used >= cap:
            raise RuntimeError(
                f"the {kind} ceiling of {cap} launches is spent ({used} started in "
                f"{totals['files']}); refusing to start {arm or kind} g{goals} s{seed} rather "
                "than extending the round past the brief's arithmetic"
            )
        self.payload["starts"] = self.starts + 1
        self.payload["by_kind"][kind] = int(self.payload["by_kind"].get(kind, 0)) + 1
        self._write()

    def record(self, kind: str, goals: int, seed: int, ok: bool, detail: str, *, arm: str = "") -> None:
        self.payload["entries"].append(
            {
                "index": len(self.payload["entries"]) + 1,
                "kind": kind,
                "arm": arm,
                "goals": int(goals),
                "seed": int(seed),
                "ok": bool(ok),
                "detail": detail,
            }
        )
        self.payload["failures"] = sum(
            1 for entry in self.payload["entries"] if not entry["ok"]
        )
        self._write()

    def _write(self) -> None:
        write_json(self.path, self.payload)


# ---------------------------------------------------------------- code identity
def git(*arguments: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", *arguments], cwd=str(ROOT), capture_output=True, text=True, check=True
    ).stdout.strip()


def blob_identity(relative: str) -> dict[str, Any]:
    """Working tree versus HEAD, judged by git blob identity, not by raw bytes.

    ``checkout`` rewrites line endings, so a byte comparison calls a clean tree dirty; the blob
    ids are what the round manifests have always compared. Absence from HEAD is reported as
    absence, never as an empty-string match.
    """
    path = ROOT / relative
    if not path.exists():
        return {"exists": False, "in_head": False, "matches_head": False}
    try:
        blob = git("hash-object", "--", relative)
    except Exception:  # noqa: BLE001 - an unreadable path is a real difference, not a crash
        blob = "<unreadable>"
    try:
        at_head = git("rev-parse", f"HEAD:{relative}")
    except Exception:  # noqa: BLE001
        at_head = "<absent-at-head>"
    return {
        "exists": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "blob": blob,
        "blob_at_head": at_head,
        "in_head": at_head != "<absent-at-head>",
        "matches_head": blob == at_head,
    }


def production_defaults_ok() -> dict[str, Any]:
    """The defaults a reader would inherit if this round's scripts never ran must be unchanged."""
    from integrated_sar import joint_driver

    configuration = joint_driver.RunConfiguration(
        seed=1, output=str(RUNS_DIR / "_default.jsonl"), mock=True, quiet=True
    )
    return {
        "comm_ack": bool(configuration.comm_ack),
        "comm_timing_mode": str(configuration.comm_timing_mode),
        "comm_policy": str(configuration.comm_policy),
        "comm_budget_bytes": configuration.comm_budget_bytes,
        "num_goals": int(configuration.num_goals),
        "ok": (
            bool(configuration.comm_ack) is False
            and str(configuration.comm_timing_mode) == "legacy_shared_bound"
            and int(configuration.num_goals) == 6
        ),
    }


def sealed_block_artefacts() -> dict[str, Any]:
    """``800000..800399`` stays ungenerated and unread; prove it by scanning, not by promise."""
    sealed = set(range(800_000, 800_400))
    hits: list[str] = []
    for path in (ROOT / "runs").rglob("*"):
        if not path.is_file():
            continue
        import re

        match = re.search(r"(?:_s|^s)(\d+)\.jsonl?$", path.name)
        if match and int(match.group(1)) in sealed:
            hits.append(str(path.relative_to(ROOT)))
    return {"block": [800_000, 800_399], "artefacts": sorted(hits), "count": len(hits)}


__all__ = [
    "HERE",
    "ROOT",
    "pilot",
    "protocol_common",
    "protocol_env",
    "tight_common",
    "confirm_common",
    "confirm_env",
    "LAYERS",
    "LAYER_ORDER",
    "STREAMS_PER_LAYER",
    "TIER",
    "BUDGET_BYTES",
    "INFORMATION",
    "DELIVERY_PROBABILITY",
    "ACK_ENABLED",
    "TIMING_MODE",
    "BOUND_PARAMS",
    "BOUND_MAX_WAIT_STEPS",
    "BOUND_URGENT_WAIT_STEPS",
    "STREAM_STEPS",
    "FROZEN_RECORD_KEYS",
    "RECORD_CONFIGURATION",
    "BASE_CONFIGURATION",
    "ARM_ORDER",
    "COMPETITION_ARMS",
    "CONTROL_ARMS",
    "CEILING_ARMS",
    "ARM_ROLES",
    "arm_spec",
    "STREAMS",
    "ROWS",
    "RUNS_DIR",
    "layer_streams",
    "layer_rows",
    "layer_runs",
    "stream_path",
    "recording_configuration",
    "assert_only_num_goals_changed",
    "replay_configuration",
    "sha256",
    "shard_of",
    "json_path",
    "read_json",
    "write_json",
    "DELIVERED_ARTEFACTS",
    "BOOTSTRAP_REPLICATIONS",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_LABEL_ORDER",
    "paired_differences",
    "MATRIX_LAUNCH_CAP",
    "MATRIX_LAUNCH_CAP_WITH_GRANT",
    "G12_EXTENSION_GRANT",
    "G12_GOALS",
    "GATE_LAUNCH_CAP",
    "LEDGER_PATTERN",
    "LaunchLedger",
    "ledger_totals",
    "freeze_file",
    "audit_file",
    "launch_grant_for",
    "git",
    "blob_identity",
    "production_defaults_ok",
    "sealed_block_artefacts",
]


def _self_check() -> int:
    problems: list[str] = []
    if BUDGET_BYTES != 29_300:
        problems.append(f"the locked 25 % credit reads as {BUDGET_BYTES}, not 29300")
    if len(set(ARM_ORDER)) != 9 or len(COMPETITION_ARMS) != 7:
        problems.append("the rule library is not 9 arms with 7 competitors")
    for goals in LAYER_ORDER:
        report = assert_only_num_goals_changed(goals)
        if not set(report["changed"]) <= {"num_goals"} or report["world_conflicts"]:
            problems.append(f"layer {goals} changes {report['changed']}")
        if len(LAYERS[goals]) != STREAMS_PER_LAYER:
            problems.append(f"layer {goals} has {len(LAYERS[goals])} seeds, not 16")
    blocks = [seed for goals in LAYER_ORDER for seed in LAYERS[goals]]
    if len(set(blocks)) != len(blocks):
        problems.append("two pressure layers share a seed: the blocks must be disjoint")
    if MATRIX_LAUNCH_CAP != len(LAYER_ORDER) * STREAMS_PER_LAYER * (1 + len(ARM_ORDER)):
        problems.append("the 640 ceiling is no longer four layers of 16 + 16 x 9 launches")
    if G12_EXTENSION_GRANT != STREAMS_PER_LAYER * (1 + len(ARM_ORDER)) - (
        MATRIX_LAUNCH_CAP - ledger_totals()["matrix_starts"]
    ):
        problems.append(
            f"the extension grant {G12_EXTENSION_GRANT} no longer equals what layer "
            f"{G12_GOALS} is short by against the launches already spent"
        )
    if freeze_file(G12_GOALS) == freeze_file(6):
        problems.append("the extension would overwrite the delivered layers' freeze file")
    if not production_defaults_ok()["ok"]:
        problems.append("a production default moved")
    for arm in ARM_ORDER:
        spec = arm_spec(arm)
        if spec["unbudgeted"] != (arm in CEILING_ARMS):
            problems.append(f"{arm}: the unbudgeted flag disagrees with its role")
    print(json.dumps(
        {
            "budget_bytes": BUDGET_BYTES,
            "layers": {str(k): [v[0], v[-1]] for k, v in LAYERS.items()},
            "arms": {arm: arm_spec(arm)["role"] for arm in ARM_ORDER},
            "launch_ceiling": launch_grant_for(G12_GOALS),
            "launches_spent": {
                key: value
                for key, value in ledger_totals().items()
                if key in ("matrix_starts", "gate_starts", "failures")
            },
            "problems": problems,
        },
        ensure_ascii=False,
        indent=1,
    ))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(_self_check())
