# -*- coding: utf-8 -*-
"""ROUND73 pure-cost actor objective: the ONE structural intervention (zero environment).

CONTROL (R71, verified): the PPO actor advantage is
    A = lambda_dual * A_task - A_cost                (cost_math.actor_advantage)
NEW (ROUND73): the actor advantage drops the task term and lambda entirely:
    A = -A_cost

This module is the single, auditable override.  It does NOT touch the dual-critic
targets/losses, GAE, reward timing, the sequential budget mask, the entropy bonus,
global gradient clip, or the dual-lambda update.  The task critic still trains and the
real task reward is still collected/evaluated; lambda is still updated and reported,
it simply no longer enters the ACTOR surrogate.

Wiring: ``canonical_v2_trainer.batch_samples`` is the only place the combined actor
advantage is formed (line: ``combined = cm.actor_advantage(advantages_task,
advantages_cost, episode["lambda"])``); that scalar becomes ``sample["advantage"]`` and
is consumed ONLY by the actor surrogate in ``ppo_update`` (``ratio * advantage``).  The
critic targets (return_task/return_cost/behaviour_value_*) and the dual-lambda step do
not call ``cm.actor_advantage``.  Therefore a scoped swap of ``cm.actor_advantage`` that
is active only inside ``batch_samples`` changes the actor objective and nothing else.

Hand-checked cost example (gate C): two legal actions, p(SEND)=0.58, SEND cost c, SKIP
cost 0, action-independent V_cost = p*c.  The expected (score-function) actor gradient of
the pure-cost objective w.r.t. the SEND logit is
    sum_a pi(a) * A_actor(a) * d log pi(a) / d logit_SEND = -c * p * (1 - p) = -0.2436 c,
i.e. the ascent direction lowers the SEND logit (less costly sending); the loss that the
optimizer MINIMIZES has the opposite sign (+0.2436 c).  This is verified against the real
log-softmax in test_r73_pure_cost_gates.py, not by copying this function.
"""
from __future__ import annotations

from typing import Any

# ------------------------------------------------------------------ identity ----
R73_TASK_ID = "ROUND73-PURE-COST-ACTOR-ABLATION"
R73_BRANCH = "exp/round73-pure-cost-actor-ablation"

# The ONLY intervention: actor advantage = -A_cost (task term and lambda removed).
R73_ACTOR_OBJECTIVE = "pure-cost/1"
R73_OBJECTIVE_VERSION = "actor-objective/pure-cost-minus-A_cost/1"

# New training/checkpoint version for NEW.  The base is the FULL verified ROUND71 training
# version (the R44 base plus the r53/r56/r58/r60/r66/r68/r71 tags) and the appended
# r73 tag is what makes a NEW checkpoint refuse to be resumed by (or confused with) an old
# single-objective ROUND71 training state.  The literal is mirrored by an assert in
# r73_checkpoint.py against the live ROUND71 version, so drift can never pass silently.
R73_TRAINING_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/"
    "dual-eta-lambda-0p5/1/rule-bc-late147-init/1/published-completion-event-at-relay-tick/2"
    "+r53-critic-representation-stop-gradient/1+r56-soft-early-report-prior-beta0p10/1"
    "+r58-prior-strength-beta1p00/1+r60-partitioned-early-late-actor-head/1"
    "+r66-frozen-early-path-late-adaptation/1+r68-late-initial-confidence-alpha0.10/1"
    "+r71-critic-calibration-init/1+r73-pure-cost-actor-objective/1"
)

# Live values that MUST be preserved (verified from the ROUND71 frozen checkpoint index;
# NOT the v2_common module defaults gae_lambda=0.95 / eta=0.05).
LIVE_GAMMA = 1.0
LIVE_GAE_LAMBDA = 1.0
LIVE_ETA_LAMBDA = 0.5
LIVE_LAMBDA_INIT = 1.0
LIVE_BETA = 1.00
READY_THRESHOLD = 147


def pure_cost_actor_advantage(advantage_task, advantage_cost, lam):
    """Return the ROUND73 actor advantage ``-A_cost`` (task term and lambda dropped).

    ``advantage_task`` and ``lam`` are accepted for signature compatibility with
    ``cost_math.actor_advantage`` and are intentionally ignored: with A_cost and every
    other quantity fixed, changing a NON-ZERO A_task or lambda cannot change this value,
    hence cannot change the actor surrogate or the actor gradient.
    """
    return -advantage_cost


def install_pure_cost_objective(trainer: Any, *, cm: Any, freeze: dict) -> dict:
    """Install the pure-cost actor objective into a live trainer, gated by the freeze.

    Refuses unless ``freeze["actor_objective"] == R73_ACTOR_OBJECTIVE``.  The swap of
    ``cm.actor_advantage`` is scoped to ``batch_samples`` calls only and restored
    afterwards, so no other consumer of cost_math is affected and the dual-lambda /
    critic paths are untouched.
    """
    if not isinstance(freeze, dict) or freeze.get("actor_objective") != R73_ACTOR_OBJECTIVE:
        raise RuntimeError(
            "ROUND73 install requires freeze actor_objective="
            f"{R73_ACTOR_OBJECTIVE!r}; got {freeze.get('actor_objective')!r}")
    if getattr(trainer, "R73_ACTOR_OBJECTIVE", None) == R73_ACTOR_OBJECTIVE and \
            getattr(getattr(trainer, "batch_samples", None), "__name__", "") == "_scoped_batch_samples":
        # already installed in this process: never stack a second wrapper around the same original
        return {"actor_objective": R73_ACTOR_OBJECTIVE,
                "objective_version": R73_OBJECTIVE_VERSION,
                "training_version": R73_TRAINING_VERSION,
                "patched": "trainer.batch_samples (already installed; not re-wrapped)",
                "untouched": ["dual-critic targets/losses", "GAE", "reward timing",
                              "sequential budget mask", "entropy bonus", "global grad clip",
                              "dual-lambda update", "task reward collection/eval"]}
    original_batch_samples = trainer.batch_samples
    original_actor_advantage = cm.actor_advantage

    def _scoped_batch_samples(episodes):
        cm.actor_advantage = pure_cost_actor_advantage
        try:
            return original_batch_samples(episodes)
        finally:
            cm.actor_advantage = original_actor_advantage

    trainer.batch_samples = _scoped_batch_samples
    trainer.R73_ACTOR_OBJECTIVE = R73_ACTOR_OBJECTIVE
    trainer.R73_OBJECTIVE_VERSION = R73_OBJECTIVE_VERSION
    return {
        "actor_objective": R73_ACTOR_OBJECTIVE,
        "objective_version": R73_OBJECTIVE_VERSION,
        "training_version": R73_TRAINING_VERSION,
        "patched": "trainer.batch_samples -> combined advantage = -A_cost (scoped)",
        "untouched": ["dual-critic targets/losses", "GAE", "reward timing",
                      "sequential budget mask", "entropy bonus", "global grad clip",
                      "dual-lambda update", "task reward collection/eval"],
    }


__all__ = [
    "R73_TASK_ID", "R73_BRANCH", "R73_ACTOR_OBJECTIVE", "R73_OBJECTIVE_VERSION",
    "R73_TRAINING_VERSION", "LIVE_GAMMA", "LIVE_GAE_LAMBDA", "LIVE_ETA_LAMBDA",
    "LIVE_LAMBDA_INIT", "LIVE_BETA", "READY_THRESHOLD",
    "pure_cost_actor_advantage", "install_pure_cost_objective",
]