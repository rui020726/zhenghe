"""``step`` must return a reward that its own decomposition adds up to.

The cumulative episode reward collapses as the curriculum lengthens episodes: with
``E`` growing from 10 to 250 steps the mean episode reward went from +66 mid-run to
-193 by the end, while coverage and discoveries kept improving. Diagnosing that needs
the per-term breakdown rather than one scalar.

These terms are recorded *alongside* the existing reward arithmetic, so the sum
invariant asserted here is what proves the log describes the reward the learner
actually saw, and that adding the log changed no reward value.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PILOC_DIR = _REPO_ROOT / "piloc"

# timemap imports gym, matplotlib and Pillow at module scope, so a machine without
# any of them cannot import the environment at all. Only that specific condition is
# a skip: once the libraries are present, a failing import is a real regression.
TIMEMAP_IMPORT_NAMES = ("gym", "PIL", "matplotlib")


def _timemap_skip_reason() -> str | None:
    missing = [
        name for name in TIMEMAP_IMPORT_NAMES if importlib.util.find_spec(name) is None
    ]
    if missing:
        return f"the PILOC environment needs {', '.join(missing)}"
    return None


SKIP_REASON = _timemap_skip_reason()

# Terms recorded on every step for every agent.
_ALWAYS_PRESENT = {
    "pheromone_rel",
    "pheromone_abs",
    "exploration_new",
    "reexploration_shaped",
    "reexploration_baseline",
    "teammate_proximity",
}
# Terms recorded only when the triggering event occurs.
_CONDITIONAL = {"collision", "repeat", "finish", "finish_dropped_prior"}


def _environment(track: bool):
    for entry in (_PILOC_DIR, _PILOC_DIR / "src" / "DDQN"):
        if str(entry) not in sys.path:
            sys.path.append(str(entry))
    from src.Map.timemap import Environment  # type: ignore[import-not-found]

    return Environment(num_agents=2, num_goals=3, track_reward_terms=track)


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class RewardTermLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(3)
        np.random.seed(3)

    def _rollout(self, track: bool, steps: int = 25):
        """Step the real environment with random actions, capturing each breakdown."""
        environment = _environment(track)
        agents = list(environment.agents)
        observation = {
            agent: np.full((60, 60), 0.5, dtype=np.float32) for agent in agents
        }
        two_channelmap: dict = {}
        captured = []
        for _ in range(steps):
            actions = {agent: random.randrange(4) for agent in agents}
            observation, reward, _done, _marks, two_channelmap = environment.step(
                actions, observation, {}, two_channelmap, 60
            )
            captured.append(
                (dict(environment.last_reward_terms), {k: float(v) for k, v in reward.items()})
            )
            if len(environment.goals_pos) == 0:
                break
        return environment, captured

    def test_the_terms_sum_to_the_reward_the_learner_sees(self) -> None:
        _environment_instance, captured = self._rollout(track=True)

        self.assertTrue(captured, "the rollout produced no steps")
        recorded = [terms for terms, _reward in captured if terms]
        self.assertTrue(recorded, "tracking was on but nothing was recorded")

        for terms, reward in captured:
            self.assertEqual(set(terms), set(reward), "the same agents must be covered")
            for agent_id, parts in terms.items():
                self.assertTrue(parts, f"{agent_id} recorded no terms")
                total = sum(parts.values())
                self.assertTrue(
                    np.isclose(total, reward[agent_id], rtol=1e-4, atol=1e-3),
                    f"{agent_id}: terms sum to {total} but step returned {reward[agent_id]}",
                )

    def test_tracking_is_off_by_default(self) -> None:
        _environment_instance, captured = self._rollout(track=False)

        self.assertTrue(captured, "the rollout produced no steps")
        for terms, reward in captured:
            self.assertEqual(terms, {}, "the default must not record anything")
            self.assertTrue(reward, "the reward itself must still be returned")

    def test_the_recorded_names_are_the_known_ones(self) -> None:
        _environment_instance, captured = self._rollout(track=True)

        names: set[str] = set()
        for terms, _reward in captured:
            for parts in terms.values():
                names.update(parts)

        self.assertTrue(
            _ALWAYS_PRESENT <= names,
            f"missing per-step terms: {sorted(_ALWAYS_PRESENT - names)}",
        )
        self.assertEqual(
            set(), names - _ALWAYS_PRESENT - _CONDITIONAL, "unexpected term names"
        )

    def test_the_hidden_baseline_is_one_unit_per_step_and_agent(self) -> None:
        """Locks the finding that explains the reward collapsing with episode length.

        ``timemap``'s ``update_timestamp_map`` subtracts an unconditional 1 per agent
        per step, and its goal-in-view branch can never fire because ``step`` deletes
        every detected goal before calling it. So on every step that does not end the
        episode with a ``finish``, each agent's logged baseline is exactly -1, and its
        episode total is exactly -steps.
        """
        _environment_instance, captured = self._rollout(track=True)

        counted_steps = 0
        totals: dict[str, float] = {}
        for terms, _reward in captured:
            if "finish" in terms.get("agent_0", {}):
                continue
            counted_steps += 1
            for agent_id, parts in terms.items():
                totals[agent_id] = totals.get(agent_id, 0.0) + parts.get(
                    "reexploration_baseline", 0.0
                )

        self.assertGreater(counted_steps, 0, "every step finished the episode")
        for agent_id, total in totals.items():
            self.assertAlmostEqual(
                total,
                -float(counted_steps),
                places=6,
                msg=f"{agent_id} baseline {total} is not one unit per step",
            )


if __name__ == "__main__":
    unittest.main()
