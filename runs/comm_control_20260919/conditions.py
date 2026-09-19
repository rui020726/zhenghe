"""The conditions this round will run, written down before any of them was run.

Two settings and a list of send policies. Nothing here was chosen after seeing a number:
the intervals, thresholds and budgets are round values picked from what the pipeline already
uses as its own time scales (a coordinator pass is every 5 steps, a command lives 10 steps, a
deadline band is 0.5-2.0 of the estimated work time, and TaskEnv's own refresh period is 2).

THE DEPLOYED SETTING (``information="deployed"``) is the system as frozen: the uplink plus
TaskEnv's line-of-sight refresh and one-hop gossip every second step. Measured in
``probe_traffic.py``, the uplink is then *not* the only path a piece of task information takes
to a robot, so it cannot be used to ask what a send policy is worth.

THE ISOLATED SETTING (``information="isolated"``) turns the environment's own knowledge
channels off -- one switch, ``gossip_interval`` above the step count, which gates the refresh
and the gossip together because both live in ``TaskEnv.propagate_task_completion``. A robot
then learns a task only from a packet, and learns what has happened to it only from a packet
or from driving up to it. That is the condition the §3 requirement asks for: while the sender
holds, the receiver cannot pick the information up somewhere else.

Anything that narrows the channel further -- an emission budget, packet loss, a transport
delay -- is tagged ``research_stress``. The repository records no measured radio bandwidth,
latency or loss rate for this system, so those numbers are probes of the trade-off's shape,
not claims about a deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ISOLATED_GOSSIP_INTERVAL = 100000
DEPLOYED_GOSSIP_INTERVAL = 2


@dataclass(frozen=True)
class Condition:
    name: str
    information: str
    policy: str
    params: dict[str, Any] = field(default_factory=dict)
    delay_ticks: int = 0
    delivery_probability: float = 1.0
    sends_per_tick: int | None = None
    comm_range: float | None = None
    role: str = "baseline"
    why: str = ""

    def configuration_overrides(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "comm_policy": self.policy,
            "comm_policy_params": ",".join(
                f"{key}={value}" for key, value in sorted(self.params.items())
            ),
            "comm_delay_ticks": self.delay_ticks,
            "comm_delivery_probability": self.delivery_probability,
            "comm_sends_per_tick": self.sends_per_tick,
            "gossip_interval": (
                ISOLATED_GOSSIP_INTERVAL if self.information == "isolated" else DEPLOYED_GOSSIP_INTERVAL
            ),
            "quiet": True,
        }
        if self.comm_range is not None:
            values["comm_range"] = self.comm_range
        return values

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "information": self.information,
            "policy": self.policy,
            "policy_parameters": dict(self.params),
            "delay_ticks": self.delay_ticks,
            "delivery_probability": self.delivery_probability,
            "sends_per_tick": self.sends_per_tick,
            "comm_range": self.comm_range if self.comm_range is not None else 0.3,
            "gossip_interval": (
                ISOLATED_GOSSIP_INTERVAL if self.information == "isolated" else DEPLOYED_GOSSIP_INTERVAL
            ),
            "role": self.role,
            "why": self.why,
        }


DEPLOYED = Condition(
    name="deployed_immediate",
    information="deployed",
    policy="immediate",
    role="reference",
    why=(
        "the frozen pipeline itself, byte-identical to the last round's C arm; it is the "
        "measurement of what the uplink is worth when other channels carry the same news"
    ),
)

CONDITIONS: tuple[Condition, ...] = (
    DEPLOYED,
    Condition(
        name="isolated_immediate",
        information="isolated",
        policy="immediate",
        role="baseline",
        why="the uplink alone, emitting each accepted detection once: every other isolated row is compared to this one",
    ),
    Condition(
        name="isolated_periodic_2",
        information="isolated",
        policy="periodic",
        params={"interval": 2},
        role="baseline",
        why="as often as the environment used to refresh beliefs itself; the high-frequency end of the trade-off",
    ),
    Condition(
        name="isolated_periodic_5",
        information="isolated",
        policy="periodic",
        params={"interval": 5},
        role="baseline",
        why="one coordinator period per target",
    ),
    Condition(
        name="isolated_periodic_20",
        information="isolated",
        policy="periodic",
        params={"interval": 20},
        role="baseline",
        why="one quarter of a command lifetime: a genuinely low rate",
    ),
    Condition(
        name="isolated_periodic_100",
        information="isolated",
        policy="periodic",
        params={"interval": 100},
        role="baseline",
        why="near-silence: how much of the system's performance survives on discovery news alone",
    ),
    Condition(
        name="isolated_event",
        information="isolated",
        policy="event",
        params={"progress_threshold": 0.2, "urgency_time": 25.0},
        role="baseline",
        why="say it when something a robot would act on changed, or when its deadline is inside 25 steps",
    ),
    Condition(
        name="isolated_event_maxwait",
        information="isolated",
        policy="event_maxwait",
        params={"progress_threshold": 0.2, "urgency_time": 25.0, "max_wait_steps": 10, "max_urgent_steps": 3},
        role="baseline",
        why="the same events, with silence bounded at 10 steps and 3 steps while a task is urgent",
    ),
    Condition(
        name="stress_budget_1",
        information="isolated",
        policy="immediate",
        sends_per_tick=1,
        role="research_stress",
        why="one task view per step at most, so competing targets pay for the same airtime",
    ),
    Condition(
        name="stress_budget_1_periodic_5",
        information="isolated",
        policy="periodic",
        params={"interval": 5},
        sends_per_tick=1,
        role="research_stress",
        why="the same scarce airtime, but a policy that chooses which target to report on each slot",
    ),
    Condition(
        name="stress_loss_0p7",
        information="isolated",
        policy="immediate",
        delivery_probability=0.7,
        role="research_stress",
        why="an unreliable uplink with no retransmission: what a send-once rule costs when packets die",
    ),
    Condition(
        name="stress_loss_0p7_periodic_5",
        information="isolated",
        policy="periodic",
        params={"interval": 5},
        delivery_probability=0.7,
        role="research_stress",
        why="the same loss, with a rule that keeps offering the news",
    ),
    Condition(
        name="stress_delay_5",
        information="isolated",
        policy="immediate",
        delay_ticks=5,
        role="research_stress",
        why="five steps on the wire, on top of the travel time to a target",
    ),
    # ---------------------------------------------------------------- diagnostics, not claims
    # Why the primary table shows so little to gain: in the deployed setting the environment's
    # own line-of-sight refresh hands the same news to any robot within ``comm_range`` -- 0.3 of
    # the unit square, about 18 grid cells. These rows narrow that radius and ask how much of
    # the uplink's value had been displaced by free sensing. Note that the radius also defines
    # the neighbour graph the allocator observes, so a change here is not a single-factor
    # ablation; read it as a structural sensitivity, never as an effect of communication alone.
    Condition(
        name="range_0p10_immediate",
        information="deployed",
        policy="immediate",
        comm_range=0.10,
        role="diagnostic",
        why="the send-once rule with the free-sensing bypass cut to a third of its radius",
    ),
    Condition(
        name="range_0p10_event",
        information="deployed",
        policy="event",
        params={"progress_threshold": 0.2, "urgency_time": 25.0},
        comm_range=0.10,
        role="diagnostic",
        why="the event rule at the same narrower radius: what the uplink is worth once it carries more of the load",
    ),
    Condition(
        name="range_0p03_immediate",
        information="deployed",
        policy="immediate",
        comm_range=0.03,
        role="diagnostic",
        why="near-blind sensing: the bypass is left to robots that physically reach the task",
    ),
    Condition(
        name="range_0p03_event",
        information="deployed",
        policy="event",
        params={"progress_threshold": 0.2, "urgency_time": 25.0},
        comm_range=0.03,
        role="diagnostic",
        why="the event rule with near-blind sensing, where an uplink schedule should start to matter",
    ),
)

BY_NAME = {condition.name: condition for condition in CONDITIONS}
