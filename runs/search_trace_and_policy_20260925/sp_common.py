"""Shared machinery for the round-18 search-trace and policy comparison.

Three things are enforced rather than promised:

* every joint-driver episode launch is booked in a per-process ledger and the cap is judged on
  the sum, because a shared read-modify-write ledger breaks under Windows file locks;
* synthetic unit tests -- including the ones that step the real ``timemap`` environment on a
  hand-built map -- are not episode launches, and this module states the boundary in one place
  so no phase can quietly widen it;
* the step recorder used to prove B1 passive is installed identically in both conditions, so
  the probe-on / probe-off contrast contains no third difference.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
try:
    import torch

    torch.set_num_threads(1)
except Exception:  # torch is optional in dependency-free environments
    pass

ROUND = "search_trace_and_policy_20260925"
ROUND_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROUND_DIR.parent.parent
START_SHA = "80af16ba4b90ce0c2d5456dd8b8e57068b4513e4"
START_BRANCH = "analysis/real-search-undetected-attribution"

LOGS = ROUND_DIR / "logs"
ROWS = ROUND_DIR / "rows"

PIL_CHECKPOINT = REPO_ROOT / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth"
CTDE_CHECKPOINT = REPO_ROOT / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth"

#: The frozen round cap from SEARCH_TRACE_AND_POLICY_PROTOCOL §三, inclusive of failures,
#: smoke starts and early terminations.
LAUNCH_CAP = 160
PHASE_PLAN = {
    "b1": 8,        # two candidate seeds x probe on/off x two repeats
    "repro": 14,    # ten historical miss seeds + four no-miss controls
    "smoke": 8,     # S1 interface, repeatability and configuration checks
    "data": 120,    # 60 new paired seeds x two arms
}
RESERVE = 10

D1_MISSED_SEEDS = (1100008, 1100015, 1100017, 1100031, 1100036, 1100051, 1100053)
SYSTEMLOOP_MISSED_SEEDS = (31003, 31009, 31030)
B1_CANDIDATE_SEEDS = (1100051, 1100053)
S1_SEEDS = list(range(1200000, 1200060))
S1_ARMS = ("A", "B")
#: Arm identity: the only difference between them is which action provider drives the same
#: real PILOC environment. Everything else is inherited from the D1 A1 configuration.
S1_ARM_OVERRIDES: dict[str, dict[str, Any]] = {
    "A": {"search_action_policy": "checkpoint"},
    "B": {"search_action_policy": "frontier"},
}

#: Seed ranges earlier rounds claim. A collision stops the matrix; the numbers are not
#: renumbered to find a "clean" block.
CLAIMED_SEED_RANGES = (
    ("sealed_test_set", 800000, 800399),
    ("learned_model_reported_stream", 900000, 900199),
    ("D1_20260923", 1100000, 1100059),
    ("systemloop_20260918", 31000, 31059),
)

BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_RNG = 20260925

D1_BATCH = REPO_ROOT / "runs" / "comm_info_path_validation_20260923" / "logs"
SYSTEMLOOP_BATCH = REPO_ROOT / "runs" / "coalition_validation_20260918" / "systemloop"
ROUND17_ATTRIBUTION = (REPO_ROOT / "runs" / "real_search_undetected_attribution_20260923"
                       / "UNDETECTED_GOAL_ATTRIBUTION.json")

#: Artefact-name fields, read from the same list the B0 probe uses so "what counts as
#: behaviour" has one implementation.
def identity_fields() -> tuple[str, ...]:
    from integrated_sar.info_probe import IDENTITY_FIELDS

    return tuple(IDENTITY_FIELDS)


def behaviour_fields(configuration: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in configuration.items() if key not in identity_fields()}


def behaviour_view(record: dict[str, Any]) -> dict[str, Any]:
    from integrated_sar.info_probe import behaviour_view as _view

    return _view(record)


# ------------------------------------------------------------------------------ files --------
def cell_paths(phase: str, arm: str, seed: int, attempt: int = 1,
               probe: bool | None = None) -> dict[str, Path]:
    """Every artefact of one cell. The name carries phase, arm, seed and attempt, because an
    earlier round lost data when two dimensions wrote the same file."""
    tag = f"{phase}_{arm}_s{seed}_a{attempt}" + ("" if probe is None else ("pon" if probe else "poff"))
    return {
        "lifecycle": LOGS / f"{tag}_rp.jsonl",
        "transcript": LOGS / f"{tag}_ct.jsonl",
        "decision_trace": LOGS / f"{tag}_dt.jsonl",
        "search_probe": LOGS / f"{tag}_sp.jsonl",
        "step_record": LOGS / f"{tag}_sr.json",
        "row": ROWS / f"{tag}.json",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(name: str, payload: Any, *, refresh: bool = False) -> Path:
    """Write one artefact of this round, refusing to touch anything else.

    ``refresh`` is granted to exactly one file, the launch ledger's live tally: the plan block
    of that file is frozen and re-checked on every refresh, so the one file allowed to change
    can still not have its budget rewritten.
    """
    if os.sep in name or "/" in name or ".." in name:
        raise ValueError(f"refusing to write outside the round directory: {name!r}")
    target = ROUND_DIR / name
    if refresh and name != "RUN_LEDGER.json":
        raise RuntimeError("only RUN_LEDGER.json may be refreshed")
    rendered = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    if target.exists():
        if refresh:
            previous = json.loads(target.read_text(encoding="utf-8"))
            if previous.get("plan") != payload.get("plan"):
                raise RuntimeError(f"the frozen plan block of {name} changed")
        elif target.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"refusing to rewrite a delivered artefact: {name}")
        return target
    target.write_text(rendered, encoding="utf-8")
    return target


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{number}: unparsable record") from error


def quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None,
                "p90": None, "max": None, "mean": None}
    ordered = sorted(float(value) for value in values)

    def at(fraction: float) -> float:
        position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[position]

    return {"n": len(ordered), "min": ordered[0], "p25": at(0.25), "median": at(0.50),
            "p75": at(0.75), "p90": at(0.90), "max": ordered[-1],
            "mean": round(sum(ordered) / len(ordered), 5)}


def ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


# ----------------------------------------------------------------------------- ledger --------
def _ledger_path(kind: str) -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    return LOGS / f"ledger_{kind}_p{os.getpid()}.json"


def ledger_reserve(kind: str, tag: str, detail: str = "") -> int:
    """Book one episode launch against the round cap. Raises once the cap is spent."""
    path = _ledger_path(kind)
    book = read_json(path) if path.exists() else {"kind": kind, "starts": 0, "entries": []}
    used = ledger_total()
    if used + 1 > LAUNCH_CAP:
        raise RuntimeError(f"launch cap {LAUNCH_CAP} reached at {used}; refusing a {kind} start")
    book["starts"] += 1
    book["entries"].append({"n": book["starts"], "round_total_before_this": used,
                            "phase": kind, "cell": tag, "detail": detail})
    path.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
    return used + 1


def ledger_total() -> int:
    total = 0
    for path in sorted(LOGS.glob("ledger_*.json")):
        try:
            total += int(read_json(path).get("starts") or 0)
        except Exception:
            continue
    return total


def ledger_report() -> dict[str, Any]:
    per_kind: dict[str, int] = {}
    entries: list[dict[str, Any]] = []
    for path in sorted(LOGS.glob("ledger_*.json")):
        try:
            book = read_json(path)
        except Exception:
            continue
        kind = book.get("kind", "?")
        per_kind[kind] = per_kind.get(kind, 0) + int(book.get("starts") or 0)
        entries.extend(book.get("entries") or [])
    return {"cap": LAUNCH_CAP, "used": ledger_total(), "by_phase": per_kind,
            "plan": dict(PHASE_PLAN), "reserve": RESERVE,
            "normal_total": sum(PHASE_PLAN.values()), "entries": sorted(
                entries, key=lambda row: (row.get("round_total_before_this", 0),
                                          str(row.get("cell"))))}


# --------------------------------------------------------------------- historical configs ----
def archived_configuration(path: Path) -> dict[str, Any]:
    """The run_config record an archived batch wrote for one cell."""
    for record in iter_jsonl(path):
        if record.get("event_type") == "run_config":
            return dict((record.get("detail") or {}).get("configuration") or {})
    raise AssertionError(f"{path.name} holds no run_config record")


def archived_scenario(path: Path) -> dict[str, Any]:
    """The scenario an archived cell recorded next to its configuration: map hash and both
    initial layouts. This is the only identity a search-side contrast can be paired on."""
    for record in iter_jsonl(path):
        if record.get("event_type") == "run_config":
            return dict((record.get("detail") or {}).get("scenario") or {})
    return {}


def d1_archive_path(arm: str, seed: int) -> Path:
    return D1_BATCH / f"data_{arm}_s{seed}_rp.jsonl"


def systemloop_archive_path(arm: str, seed: int) -> Path:
    return SYSTEMLOOP_BATCH / f"{arm}_rp_s{seed}.jsonl"


#: Fields a historical batch recorded as null because the field did not exist yet. Today's
#: dataclass default is the documented historical value, so the replay substitutes it and the
#: identity report lists the substitution instead of hiding it.
def configuration_from_archive(archived: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    from dataclasses import fields as dataclass_fields

    from integrated_sar.joint_driver import RunConfiguration

    accepted = {field.name for field in dataclass_fields(RunConfiguration)}
    used: dict[str, Any] = {}
    substituted: list[str] = []
    for name, value in archived.items():
        if name in identity_fields():
            continue
        if name not in accepted:
            substituted.append(f"{name}: dropped, not a field of the current configuration")
            continue
        if value is None:
            substituted.append(f"{name}: archived null, current default "
                               f"{getattr(RunConfiguration, name, None)!r} used")
            continue
        used[name] = value
    for name in ("device", "quiet"):
        used.setdefault(name, archived.get(name, True if name == "quiet" else "cpu"))
    used["mock"] = False
    return used, substituted


# ------------------------------------------------------------------------ step recorder ------
class StepRecorder:
    """Per-step read-only capture, installed identically whether or not B1 is on.

    It wraps ``Environment.step`` after the original returns and records only what a pure read
    gives: the actions handed in, the rewards and done flags handed back, both position
    arrays, and digests of the environment's own rng states. Nothing here consumes a draw or
    writes state, and because it is attached in both conditions it cannot bias the contrast --
    it is the instrument the contrast is judged with, not a third arm.
    """

    def __init__(self, run: Any) -> None:
        self.rows: list[dict[str, Any]] = []
        self.env = run.search_adapter.env
        original = getattr(self.env, "step", None)
        if original is None:
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            outcome = original(*args, **kwargs)
            self.rows.append(self._row(args, outcome))
            return outcome

        self.env.step = wrapped

    def _row(self, args: tuple, outcome: tuple) -> dict[str, Any]:
        actions = dict(args[0]) if args else {}
        rewards = dict(outcome[1]) if len(outcome) > 1 and outcome[1] else {}
        done = dict(outcome[2]) if len(outcome) > 2 and outcome[2] else {}
        return {
            "step": int(getattr(self.env, "steps", 0)),
            "actions": {str(key): int(value) for key, value in actions.items()},
            "rewards": {str(key): round(float(value), 9) for key, value in rewards.items()},
            "done": {str(key): bool(value) for key, value in done.items()},
            "agents": [[int(p[0]), int(p[1])] for p in getattr(self.env, "agents_pos", ())],
            "goals": [[int(p[0]), int(p[1])] for p in getattr(self.env, "goals_pos", ())],
            "py_rng": _state_digest(getattr(self.env, "_py_rng", None), "py"),
            "np_rng": _state_digest(getattr(self.env, "_np_rng", None), "np"),
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"steps": len(self.rows), "rows": self.rows},
                                   ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return path


def _state_digest(generator: Any, kind: str) -> str | None:
    """A hash of an rng's state, taken without touching the stream."""
    if generator is None:
        return None
    try:
        if kind == "py":
            payload = json.dumps(generator.getstate(), sort_keys=True)
        else:
            state = generator.get_state()
            payload = json.dumps([state[0], int(state[1]), int(state[2]), int(state[3]),
                                  state[4].astype("uint32").tobytes().hex()])
    except Exception as error:  # a recorder that cannot read a state says so, never guesses
        return f"unreadable:{type(error).__name__}"
    return sha256_text(payload)[:32]


def enter_round() -> None:
    here = str(ROUND_DIR)
    while here in sys.path:
        sys.path.remove(here)
    sys.path.insert(0, here)
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(1, root)


enter_round()
