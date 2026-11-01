"""B1: a passive search-side probe that records what the detection step actually saw.

The archived rounds could answer "was this goal ever discovered", but not "at the step where
the environment judged discovery, where was each robot, where was each still-hidden target,
which targets did the environment itself delete, and which ``DetectionEvent`` did the adapter
publish". Without those, the "entered legal range yet produced no event" explanation of an
undetected goal stays undecidable, and so does the difference between a detector bug and a
trajectory that never looked at that cell.

The probe is built only when a path is configured, and it is limited to reading:

* it consumes no random number. This matters more here than for B0: the hidden targets walk
  themselves via ``Environment._py_rng`` (``timemap.step``), so one extra draw would replace
  the whole trajectory;
* it writes no environment, registry or belief state, so no robot can act on what it recorded;
* it is not an actor observation -- the trained policy reads the two-channel map and the
  pheromone window, neither of which the probe touches;
* it writes one side file, leaving the lifecycle JSONL byte-identical to a probe-less run,
  which is what makes the paired-replay acceptance check a comparison of files rather than of
  interpretations.

Identity discipline. ``Environment.goals_pos`` is delete-only, so an *array index* is not an
identity: removing one target shifts every later index left. The probe therefore names each
target by its **reset index** (``goal_index_at_reset``), the same value the adapter's
``piloc-e{episode}-target-{index}`` id encodes, and carries the drifting array index only as
an annotation inside the step it was valid for.

Three independent views of the same fact are recorded side by side, because a correctness
audit must not decide between them by interpretation:

``env_deleted_reset_indices``
    what the environment itself judged discovered and removed -- ``observed_indices`` in the
    v2/potential branch, the per-agent union in the legacy branch, translated back to reset
    indices by the environment's own alive bookkeeping;
``observed_pairs``
    every (robot, target) pair satisfying the environment's *own* predicate primitives at the
    detection instant (its ``_is_line_of_sight_blocked`` over its ``self.map``), which answers
    "who else could have seen it" that the set semantics of the first view throws away;
``adapter_event_target_ids``
    what the integration actually published downstream.

The analysis asserts these three agree. A pair-list that is larger than the deleted set is not
silently averaged away: it is the definition of category 1 in the round-18 protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

PROBE_SCHEMA = "sar-search-probe/1"

#: The detection rule's own literal. ``timemap`` hard-codes 5 in both branches
#: (``_observed_goal_indices`` and the legacy loop), so the probe reads it from the
#: environment attribute written by ``timemap`` rather than carrying a second copy.
RADIUS_SOURCE = "environment.detection_rule_radius"


class SearchProbe:
    """One side file per episode: placement, per-step detection snapshots, closing record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows: list[dict[str, Any]] = []
        self.placed: list[dict[str, Any]] = []
        self.steps_recorded = 0
        self.anomalies: list[str] = []
        # Closures the probe can only judge step by step, aggregated here so the file is
        # self-describing: which steps disagree between the three views, and how often.
        self.disagreements: list[dict[str, Any]] = []
        self._placed_written = False

    # ------------------------------------------------------------- placement --------------
    def note_placed(self, goals: Iterable[tuple[str, Any]]) -> None:
        """The goals present at reset, whether or not anything ever finds them."""
        if self._placed_written:
            return
        self._placed_written = True
        for index, (target_id, position) in enumerate(goals):
            self.placed.append({
                "kind": "placed",
                "goal_index_at_reset": index,
                "target_id": str(target_id),
                "cell": [int(position[0]), int(position[1])],
            })
        self.rows.extend(self.placed)

    # ----------------------------------------------------------------- step ---------------
    def note_step(self, record: dict[str, Any]) -> None:
        self.steps_recorded += 1
        self.rows.append(record)
        closure = record.get("closure") or {}
        if not closure.get("all_views_agree", True):
            self.disagreements.append({
                "step": record.get("step"),
                "views": {key: value for key, value in closure.items()
                          if key.startswith("view_") or key == "all_views_agree"},
            })

    # ---------------------------------------------------------------- final ---------------
    def note_final(self, **fields: Any) -> None:
        self.rows.append({"kind": "final", "schema": PROBE_SCHEMA,
                          "steps_recorded": self.steps_recorded,
                          "placed": len(self.placed),
                          "disagreement_steps": len(self.disagreements),
                          "anomalies": sorted(set(self.anomalies))[:20],
                          "anomaly_count": len(self.anomalies),
                          **fields})

    def anomaly(self, text: str) -> None:
        self.anomalies.append(text)

    # ----------------------------------------------------------------- output -------------
    def records(self) -> list[dict[str, Any]]:
        return list(self.rows)

    def write(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return self.path

    def summary(self) -> dict[str, Any]:
        """The row a runner books: enough to judge passivity without opening the file."""
        return {
            "search_probe": str(self.path),
            "search_probe_placed": len(self.placed),
            "search_probe_steps": self.steps_recorded,
            "search_probe_disagreements": len(self.disagreements),
            "search_probe_anomalies": len(self.anomalies),
        }

    def reset(self) -> None:
        """Drop one episode's worth of records; the caller writes the file before resetting."""
        self.__init__(self.path)


def closure_flags(*, env_ids: list[str], pair_ids: list[str], adapter_ids: list[str],
                  alive_views_agree: bool) -> dict[str, Any]:
    """Compare the three views as sets of stable ids, and say which ones differ.

    ``observed_pairs`` is a per-robot list: two robots seeing one target is one discovery, not
    two, so it is collapsed to the target ids before comparison. The pair *list* itself stays
    in the record for the "who else could see it" question.
    """
    env_set, pair_set, adapter_set = set(env_ids), set(pair_ids), set(adapter_ids)
    return {
        "view_env_deleted": sorted(env_set),
        "view_pair_observed": sorted(pair_set),
        "view_adapter_events": sorted(adapter_set),
        "env_matches_pairs": env_set == pair_set,
        "env_matches_adapter": env_set == adapter_set,
        "pairs_match_adapter": pair_set == adapter_set,
        "alive_views_agree_at_step_start": bool(alive_views_agree),
        "all_views_agree": (env_set == pair_set == adapter_set and bool(alive_views_agree)),
    }
