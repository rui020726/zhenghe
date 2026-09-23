"""Read-only helpers for the recipient/topology feasibility audit.

Nothing here may run an episode or write into another round's directory. The tripwire is installed
before the frozen round's modules are imported, so "0 new episodes" is an enforced property rather
than a statement in a document: each episode entry point raises and is logged, and a target that
cannot be resolved aborts the install instead of silently wrapping nothing.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
#: The episodes being audited, and the round whose audit-trail format we reuse.
SOURCE_ROUND = ROOT / "runs" / "comm_task_pressure_learnability_20260922"
PREVIOUS_ROUND = ROOT / "runs" / "comm_action_geometry_audit_20260923"

LAYERS = ("g6", "g8", "g10", "g12")
ARMS = (
    "send_once", "event", "event_priority", "periodic_5", "periodic_20",
    "first_send_plus_bound_3_10", "bound_only_3_10", "always_hold", "event_unbudgeted",
)
BUDGET_BYTES = 29_300

FORBIDDEN_CALLS = (
    ("pressure_env", "run_episode"),
    ("pressure_env", "run_arm"),
    ("pressure_env", "agreement_with_archived_rows"),
    ("record_layer", "record"),
    ("pressure_matrix", "run"),
)

_TRIPPED: list[str] = []
_installed = False


class EnvironmentCallForbidden(RuntimeError):
    """Raised the moment an analysis path tries to execute an episode."""


def enter_source_round() -> None:
    """Put the audited round's directory first on sys.path.

    Its own protocols record why this matters: each round's fixtures re-insert their directory at
    import time, so a bare ``import pressure_env`` can silently resolve to a different round's
    module of the same name. Inserting after those imports keeps this audit pointed at the right
    episodes.
    """
    for path in (SOURCE_ROUND, ROOT):
        text = str(path)
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def _resolve(module_name: str, attr: str) -> Any:
    module = importlib.import_module(module_name)
    if not callable(getattr(module, attr, None)):
        raise RuntimeError(f"tripwire target {module_name}.{attr} is not a callable")
    return module


def install_env_tripwire() -> list[str]:
    """Wrap every episode entry point so calling it raises. Returns the shared call log."""
    global _installed
    if _installed:
        return _TRIPPED
    enter_source_round()
    for module_name, attr in FORBIDDEN_CALLS:
        module = _resolve(module_name, attr)
        original = getattr(module, attr)
        if getattr(original, "_audit_tripwire", False):
            continue

        def _blocked(*args: Any, _name: str = f"{module_name}.{attr}", **kwargs: Any) -> Any:
            _TRIPPED.append(_name)
            raise EnvironmentCallForbidden(
                f"{_name}() would run an episode; this round is authorised for 0 new episodes"
            )

        _blocked._audit_tripwire = True  # type: ignore[attr-defined]
        _blocked._audit_original = original  # type: ignore[attr-defined]
        setattr(module, attr, _blocked)
    _installed = True
    return _TRIPPED


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> Path:
    """Write inside this round's directory only."""
    resolved = Path(path).resolve()
    if HERE not in resolved.parents and resolved.parent != HERE:
        raise ValueError(f"refusing to write outside this audit directory: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return resolved


def rows_of(layer: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((SOURCE_ROUND / "rows" / layer).glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def transcript_path(layer: str, arm: str, seed: int) -> Path:
    return SOURCE_ROUND / "runs" / layer / f"ct_{arm}_{layer}_s{seed}.jsonl"


def read_transcript(layer: str, arm: str, seed: str | int) -> list[dict[str, Any]]:
    path = transcript_path(layer, str(arm), int(seed))
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "p95": None,
                "max": None, "mean": None}
    ordered = sorted(values)

    def _at(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p25": round(_at(0.25), 6),
        "median": round(_at(0.50), 6),
        "p75": round(_at(0.75), 6),
        "p95": round(_at(0.95), 6),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 6),
    }


def source_facts() -> dict[str, Any]:
    """Read the subscriber and pricing chain out of the production files *as text*.

    The source map must be built from the current code rather than from prose, so the audit keeps
    the exact lines it relied on: each named anchor is located and returned with its line number,
    and a missing anchor is reported instead of being glossed over.
    """
    anchors = {
        "subscribers_assigned_once": ("integrated_sar/comm.py", "self._subscribers = tuple("),
        "constructor_kwarg": ("integrated_sar/comm.py", "        subscribers,"),
        "attempt_charge_uses_fanout": (
            "integrated_sar/comm.py",
            "charge = int(estimate_message_bytes(message) * len(self._subscribers))",
        ),
        "emit_cost_uses_fanout": (
            "integrated_sar/comm.py",
            "cost_bytes = payload_bytes * len(self._subscribers)",
        ),
        "observation_reports_count": ("integrated_sar/comm.py", "subscribers=len(self._subscribers),"),
        "delivery_loop": ("integrated_sar/comm.py", "for agent_id in self._subscribers:"),
        "payload_estimator": ("integrated_sar/comm.py", "def estimate_message_bytes("),
        "driver_resolves_subscribers": (
            "integrated_sar/joint_driver.py",
            "requested = parse_comm_subscribers(configuration.comm_subscribers)",
        ),
        "driver_available_set": (
            "integrated_sar/joint_driver.py",
            "available = tuple(str(agent_id) for agent_id in task_adapter.env.agent_dic)",
        ),
        "driver_default_all": (
            "integrated_sar/joint_driver.py",
            "comm_subscribers: str = \"all\"",
        ),
        "relay_constructed": ("integrated_sar/joint_driver.py", "comm_relay = CommRelay("),
        "belief_write": ("task_allocation/old_task_env.py", "def learn_task_view("),
        "known_tasks_add": ("task_allocation/old_task_env.py", 'knowledge["known_tasks"].add'),
        "action_list_from_belief": (
            "task_allocation/old_task_env.py",
            "pending_tasks = knowledge[\"known_tasks\"] - knowledge[\"known_finished\"]",
        ),
        "gossip_gate": (
            "task_allocation/old_task_env.py",
            "if self._step_counter % self.gossip_interval == 0",
        ),
    }
    out: dict[str, Any] = {}
    for name, (relative, needle) in anchors.items():
        path = ROOT / relative
        if not path.exists():
            out[name] = {"file": relative, "found": False}
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        hits = [index + 1 for index, line in enumerate(lines) if needle in line]
        out[name] = {
            "file": relative,
            "found": bool(hits),
            "lines": hits,
            "text": lines[hits[0] - 1].strip() if hits else None,
        }
    return out
