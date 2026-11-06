"""PPO update for revision_02: ``A = A_task - lambda * A_cost`` with a dual lambda.

The update mechanics are the audited Stage C ones unchanged -- clip 0.2, four epochs,
minibatch 256, separate task and cost critics at 0.5 each, entropy 0.01, gradient clip 0.5,
``gamma = 0`` bandit advantages from the causal credit stream.

What differs from the inherited trainer:

* the cost reward is ``charged_bytes / (N * G)`` rather than ``charged_bytes / 10000``, so a
  byte means the same thing in every cell;
* ``lambda`` is an input, not a module constant, and moves at this boundary via
  ``dual_step``; and
* the batch records its own success rate and the reference it was compared against, so the
  lambda trace can be reconstructed rather than trusted.

The GAE recursion, log-probability and entropy helpers are the frozen implementations; this
module only re-assembles their output.
"""

from __future__ import annotations

import hashlib
import os
import random
from bisect import bisect_right
from pathlib import Path

from . import constants as C
from . import objective
from . import unlimited_budget

CHECKPOINT_SCHEMA = C.STATE_SCHEMA


def deps():
    """Import the frozen numerical helpers lazily so module import stays world-free."""
    import torch
    import numpy as np
    import cost_math as cm
    from onpolicy import action_entropy, joint_log_prob, rng_state, restore_rng
    return torch, np, cm, action_entropy, joint_log_prob, rng_state, restore_rng


def _bundle(stream, decision_steps, horizon):
    """Assign each tick reward to the latest decision at or before that tick."""
    bundled = {step: 0.0 for step in decision_steps}
    prefix = 0.0
    for step in range(0, int(horizon) + 1):
        value = float(stream.get(step, 0.0))
        previous = bisect_right(decision_steps, step) - 1
        if previous < 0:
            prefix += value
        else:
            bundled[decision_steps[previous]] += value
    return bundled, prefix


def _subscribers_of(episode):
    """The live fan-out as the relay saw it, for a row that did not record one."""
    value = episode.get("subscribers_live")
    if value is None:
        raise ValueError("episode %r carries neither subscribers_configured nor "
                         "subscribers_live" % (episode.get("label"),))
    return int(value)


def _episode_horizon(episode):
    """The episode's own step count; the tap streams are indexed by that horizon."""
    horizon = int(episode["steps"])
    if horizon < 1:
        raise ValueError("episode steps must be >= 1, got %r" % (episode["steps"],))
    return horizon


def _apply_transform(stacked, chunk):
    """Apply each sample's own cell transform before the trunk.

    The transform is **episode-scoped**: ``world`` restores it when the episode ends and the
    PPO update runs afterwards. Without this, the update recomputes log-probabilities from raw
    rows while the policy sampled from rewritten ones. Measured on this revision: max
    |dlogp| = 0.109 at update 0, i.e. every sample was off-policy by construction.

    A minibatch can mix cells and each cell divides #2/#14 by its own ``G``, so the transform
    cannot be one matrix-wide constant. Every sample carries its own ``cell_transform``.
    """
    from . import scale_features as scale_features_mod
    import torch as _torch

    specs = [item.get("cell_transform") for item in chunk]
    if all(spec is None for spec in specs):
        raise RuntimeError("every sample is missing cell_transform; a revision_02 update must "
                           "reproduce the row the policy sampled from")
    if any(spec is None for spec in specs):
        raise RuntimeError("a minibatch mixes transformed and untransformed samples")
    # Row ranges are computed **within this minibatch**: ``stacked`` holds only the chunk's
    # rows, so a global per-sample offset would index past the end. Each sample contributes
    # ``len(targets)`` rows, in chunk order.
    groups = {}
    cursor = 0
    for item, spec in zip(chunk, specs):
        count = len(item["targets"])
        key = (spec["mode"], int(spec["goals"]), int(spec["subscribers"]))
        groups.setdefault(key, []).extend(range(cursor, cursor + count))
        cursor += count
    if cursor != int(stacked.shape[0]):
        raise RuntimeError("minibatch row count %d != sum of sample rows %d"
                           % (int(stacked.shape[0]), cursor))
    if len(groups) == 1:
        (mode, goals, subscribers), rows = next(iter(groups.items()))
        return scale_features_mod.apply_rewrite(stacked[rows], mode=mode, goals=goals,
                                                subscribers=subscribers)
    out = _torch.empty_like(stacked)
    for (mode, goals, subscribers), rows in groups.items():
        out[rows] = scale_features_mod.apply_rewrite(stacked[rows], mode=mode, goals=goals,
                                                     subscribers=subscribers)
    return out


def build_samples(episodes, arm=None, *, lam=None):
    """One row per decision step with the registered combined advantage.

    Two revision_02 changes from the inherited body:

    * the charged bytes of each episode are divided by that episode's ``N * G`` before the
      advantage is formed, so ``A_cost`` is in scaled-byte units and one lambda spans the
      whole (G, R, N) grid. Each episode carries its own ``scale`` because a batch can mix
      cells -- resampling per episode is the point, not a side effect; and
    * the combination takes ``lam`` explicitly. Lambda is an input here, never a module
      constant, because it moves at update boundaries and a stale read would silently apply
      the previous step's price.
    """
    torch, np, cm, _entropy, _log_prob, _rng_state, _restore = deps()
    if lam is None:
        raise ValueError("build_samples requires lambda; it is a dual variable, not a "
                         "module constant")
    samples = []
    for episode in episodes:
        tapes = sorted(episode["collector"].records, key=lambda tape: int(tape["step"]))
        if not tapes:
            continue
        decision_steps = [int(tape["step"]) for tape in tapes]
        horizon = _episode_horizon(episode)
        # v2_run._install_taps stores {tick: {airtime_bytes: n, ...}}. A missing key or a
        # bare number is a different schema and must fail instead of becoming a zero-cost tick.
        cost_stream = {}
        for key, value in episode["cost_stream"].items():
            if not isinstance(value, dict):
                raise TypeError("cost_stream[%r] is %r" % (key, type(value).__name__))
            if "airtime_bytes" not in value:
                raise TypeError("cost_stream[%r] has no airtime_bytes" % (key,))
            cost_stream[int(key)] = float(value["airtime_bytes"])
        task_stream = {int(k): float(v) for k, v in episode["task_stream"].items()}
        bundled_cost, prefix_cost = _bundle(cost_stream, decision_steps, horizon)
        bundled_task, prefix_task = _bundle(task_stream, decision_steps, horizon)
        # The cross-cell divisor, rescaled so the frozen (6, 24) cell keeps ``bytes / B0``.
        # Without the rescale a send costs ~33 scaled units against a task advantage of ~0.41,
        # so "never send" is optimal at every lambda in the registered band -- measured: 342 of
        # 366 updates sent nothing and lambda sat on the floor.
        scale = episode.get("cost_scale")
        if scale is None or float(scale) <= 0.0:
            raise ValueError("episode %r carries no positive cost_scale; every revision_02 "
                             "row must know its own N*G" % (episode.get("label"),))
        factor = episode.get("cost_reward_factor")
        if factor is None or float(factor) <= 0.0:
            raise ValueError("episode %r carries no positive cost_reward_factor"
                             % (episode.get("label"),))
        scaled = float(factor)
        rewards_cost = torch.tensor([bundled_cost[s] * scaled for s in decision_steps],
                                    dtype=torch.float64)
        rewards_task = torch.tensor([bundled_task[s] for s in decision_steps],
                                    dtype=torch.float64)
        values_task = torch.tensor([float(t["value"]) for t in tapes], dtype=torch.float64)
        values_cost = torch.tensor([float(t["cost_value"]) for t in tapes], dtype=torch.float64)
        terminal = torch.zeros(len(decision_steps), dtype=torch.bool)
        terminal[-1] = True
        bootstrap_task = torch.zeros_like(values_task)
        bootstrap_cost = torch.zeros_like(values_cost)
        if len(decision_steps) > 1:
            bootstrap_task[:-1] = values_task[1:]
            bootstrap_cost[:-1] = values_cost[1:]
        if C.ADVANTAGE_MODE == "causal_step":
            # The causal stream already carries each credit on its own decision step, so a
            # multi-step GAE would only re-import cross-target crosstalk and re-discount long
            # tasks.  At gamma = 0 the advantage is the contextual-bandit form A = r - V(s).
            zero_boot = torch.zeros_like(values_task)
            advantages_task, returns_task = cm.gae(rewards_task, values_task,
                                                   gamma=C.CAUSAL_GAMMA, gae_lambda=0.0,
                                                   terminal=terminal, bootstrap=zero_boot)
            advantages_cost, returns_cost = cm.gae(rewards_cost, values_cost,
                                                   gamma=C.CAUSAL_GAMMA, gae_lambda=0.0,
                                                   terminal=terminal,
                                                   bootstrap=torch.zeros_like(values_cost))
        else:
            advantages_task, returns_task = cm.gae(rewards_task, values_task, gamma=C.GAMMA,
                                                   gae_lambda=C.GAE_LAMBDA, terminal=terminal,
                                                   bootstrap=bootstrap_task)
            advantages_cost, returns_cost = cm.gae(rewards_cost, values_cost, gamma=C.GAMMA,
                                                   gae_lambda=C.GAE_LAMBDA, terminal=terminal,
                                                   bootstrap=bootstrap_cost)
        combined = objective.combine_many([float(v) for v in advantages_task],
                                          [float(v) for v in advantages_cost], lam)
        cell_transform = {
            "mode": str(C.SCALE_FEATURES),
            "goals": int(episode.get("goals") or C.NUM_GOALS),
            "subscribers": int(episode.get("subscribers_configured")
                               or _subscribers_of(episode)),
        }
        for index, tape in enumerate(tapes):
            samples.append({
                "features": np.asarray(tape["features"], dtype=np.float32),
                "targets": list(tape["targets"]),
                "actions": dict(tape["actions"]),
                "free": set(tape["actionable"]),
                "log_prob": float(tape["log_prob"]),
                "advantage": float(combined[index]),
                "advantage_task": float(advantages_task[index]),
                "advantage_cost": float(advantages_cost[index]),
                "return_task": float(returns_task[index]),
                "return_cost": float(returns_cost[index]),
                "step": int(tape["step"]),
                "unassigned_prefix_task_reward": float(prefix_task if index == 0 else 0.0),
                "unassigned_prefix_cost_bytes": float(prefix_cost if index == 0 else 0.0),
                #: What the scale actually was, so a report can audit the divisor.
                "cost_scale": float(scale),
                #: Which rewrite this row needs at update time. It travels with the sample
                #: so the update can reproduce the exact input the policy sampled from.
                "cell_transform": cell_transform,
            })
    return samples


def minibatch_seed(update_index):
    """The minibatch stream depends on the update index only, never on the arm.

    T1 and T2 are paired arms: they consume the same samples and must see the same shuffling
    substream. Including the arm name would silently give each arm its own ordering.
    """
    return int(hashlib.sha256(("20261026|%d" % int(update_index)).encode("utf-8"))
               .hexdigest()[:16], 16)


def minibatch_order(count, update_index):
    """Deterministic shuffled order, independent of the objective arm."""
    order = list(range(int(count)))
    random.Random(minibatch_seed(update_index)).shuffle(order)
    return order


def ppo_update(controller, optimizer, episodes, *, lam, update_index, q_ref=None,
               arm="dual"):
    """One PPO update over a batch under ``A = A_task - lambda * A_cost``.

    The dual step happens here and only here: ``q`` is this batch's
    ``sum(on_time) / sum(generated_goals)``, ``q_ref`` is the ``normal_immediate`` rate for the
    same cells, and ``lambda`` moves by ``2 * (q - 0.95 * q_ref)`` clipped to [0.25, 8].

    A batch can span several cells, so ``q_ref`` may be a mapping from cell label to rate; the
    reference used is the on-time-weighted mean over the cells actually present. That keeps the
    comparison like for like while still allowing one shared lambda. When no cell in the batch
    has a reference the step is skipped and the reason is recorded instead of inventing a q_ref.
    """
    torch, np, _cm, action_entropy, joint_log_prob, _rng_state, _restore = deps()
    samples = build_samples(episodes, arm, lam=lam)
    if not samples:
        return {"samples": 0, "minibatches": 0, "optimizer_steps": 0, "arm": arm,
                "lambda": float(lam)}
    order = list(range(len(samples)))
    shuffle = random.Random(minibatch_seed(update_index))
    totals = {"surrogate": 0.0, "value_task": 0.0, "value_cost": 0.0, "entropy": 0.0,
              "kl": 0.0, "grad": 0.0}
    minibatches = 0
    for _epoch in range(C.EPOCHS_PER_BATCH):
        shuffle.shuffle(order)
        for start in range(0, len(order), C.MINIBATCH_STEPS):
            chunk = [samples[index] for index in order[start:start + C.MINIBATCH_STEPS]]
            stacked = torch.as_tensor(np.concatenate([item["features"] for item in chunk], axis=0))
            # The transform is episode-scoped, so it is NOT alive here. Reproduce the row the
            # policy actually sampled from, or the importance ratio is computed against a
            # different input. Measured without this: max |dlogp| = 0.109 at update 0.
            stacked = _apply_transform(stacked, chunk)
            hidden = controller.inner.trunk(stacked)
            logits = controller.inner.action_head(hidden)
            values_task_c = controller.inner.value_head(hidden).squeeze(-1)
            values_cost_c = controller.cost_head(hidden).squeeze(-1)
            row = 0
            new_log_prob = []
            entropy = []
            pooled_task = []
            pooled_cost = []
            for item in chunk:
                count = len(item["targets"])
                rows = torch.arange(row, row + count)
                row += count
                picked = torch.tensor([item["actions"][t] for t in item["targets"]],
                                      dtype=torch.long)
                usable = torch.tensor([1.0 if t in item["free"] else 0.0
                                       for t in item["targets"]], dtype=torch.float32)
                new_log_prob.append(joint_log_prob(logits[rows], picked, usable))
                entropy.append(action_entropy(logits[rows], usable))
                pooled_task.append(values_task_c[rows].mean())
                pooled_cost.append(values_cost_c[rows].mean())
            new_log_prob_t = torch.stack(new_log_prob)
            entropy_t = torch.stack(entropy)
            old_log_prob = torch.tensor([item["log_prob"] for item in chunk], dtype=torch.float32)
            advantage = torch.tensor([item["advantage"] for item in chunk], dtype=torch.float32)
            target_task = torch.tensor([item["return_task"] for item in chunk], dtype=torch.float32)
            target_cost = torch.tensor([item["return_cost"] for item in chunk], dtype=torch.float32)
            ratio = torch.exp(new_log_prob_t - old_log_prob)
            surrogate = torch.min(ratio * advantage,
                                  torch.clamp(ratio, 1.0 - C.CLIP, 1.0 + C.CLIP) * advantage)
            value_task_loss = ((torch.stack(pooled_task) - target_task) ** 2).mean()
            value_cost_loss = ((torch.stack(pooled_cost) - target_cost) ** 2).mean()
            if C.NORMALIZE_ADVANTAGE == "std_only":
                # Scale only: no mean subtraction, so the sign structure of the advantage is
                # preserved while a shrinking critic residual can no longer let a constant
                # offset drive the action logits monotonically negative.
                advantage = advantage / advantage.std().clamp_min(1e-6)
            loss = (-surrogate.mean()
                    + C.CRITIC_LOSS_WEIGHT * (value_task_loss + value_cost_loss)
                    - C.ENTROPY_BONUS * entropy_t.mean())
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(controller.parameters, C.MAX_GRAD_NORM)
            optimizer.step()
            if not (torch.isfinite(loss) and torch.isfinite(norm)):
                raise RuntimeError("update %d: non-finite loss or gradient norm" % int(update_index))
            totals["surrogate"] += float(surrogate.mean().item())
            totals["value_task"] += float(value_task_loss.item())
            totals["value_cost"] += float(value_cost_loss.item())
            totals["entropy"] += float(entropy_t.mean().item())
            totals["kl"] += float((old_log_prob - new_log_prob_t).mean().item())
            totals["grad"] += float(norm)
            minibatches += 1
    batch_advantage = np.asarray([item["advantage"] for item in samples], dtype=np.float64)
    if not np.all(np.isfinite(batch_advantage)):
        raise RuntimeError("update %d: non-finite advantage in the batch" % int(update_index))
    dual = dual_step(episodes, lam=lam, q_ref=q_ref)
    return {
        "arm": arm, "objective": objective.describe(lam),
        "lambda": float(lam),
        "dual": dual,
        "samples": len(samples), "minibatches": minibatches,
        "optimizer_steps": minibatches, "epochs": C.EPOCHS_PER_BATCH,
        "policy_surrogate": round(totals["surrogate"] / minibatches, 6),
        "value_loss_task": round(totals["value_task"] / minibatches, 6),
        "value_loss_cost": round(totals["value_cost"] / minibatches, 6),
        "entropy": round(totals["entropy"] / minibatches, 6),
        "approx_kl": round(totals["kl"] / minibatches, 6),
        "gradient_norm_mean": round(totals["grad"] / minibatches, 4),
        "advantage_normalisation": str(C.NORMALIZE_ADVANTAGE),
    }


def batch_success(episodes):
    """``sum(on_time) / sum(generated_goals)`` over a batch, and the per-cell breakdown.

    Returns ``(q, per_cell)`` where ``per_cell`` maps cell label to
    ``{on_time, generated, q}``. Summing across cells rather than averaging per-cell rates is
    deliberate: a 48-goal world contributes more observations, so it should weigh more.
    """
    on_time = 0
    generated = 0
    per_cell: dict[str, dict] = {}
    for episode in episodes:
        goals = int(episode.get("generated_goals") or 0)
        on = int(episode.get("on_time") or 0)
        on_time += on
        generated += goals
        label = str(episode.get("cell_label") or "unknown")
        entry = per_cell.setdefault(label, {"on_time": 0, "generated": 0})
        entry["on_time"] += on
        entry["generated"] += goals
    for entry in per_cell.values():
        entry["q"] = (entry["on_time"] / entry["generated"]) if entry["generated"] else None
    if generated <= 0:
        raise ValueError("a batch with no generated goals cannot define q")
    return on_time / generated, per_cell


def reference_rate(episodes, q_ref):
    """The on-time-weighted ``normal_immediate`` rate for the cells this batch contains.

    ``q_ref`` may be a scalar (one cell in the batch) or a mapping from cell label to rate.
    A cell with no recorded reference is skipped and named in the detail rather than being
    substituted with a guess, so a missing baseline shows up as a smaller denominator.
    """
    per_cell = {}
    for episode in episodes:
        label = str(episode.get("cell_label") or "unknown")
        goals = int(episode.get("generated_goals") or 0)
        entry = per_cell.setdefault(label, {"generated": 0})
        entry["generated"] += goals
    if isinstance(q_ref, (int, float)):
        return float(q_ref), {"cells": sorted(per_cell), "missing": [],
                              "mode": "scalar_for_whole_batch"}
    if not isinstance(q_ref, dict):
        raise TypeError("q_ref must be a rate or a {cell_label: rate} mapping")
    weighted = 0.0
    used = 0
    missing = []
    for label, entry in sorted(per_cell.items()):
        rate = q_ref.get(label)
        if rate is None:
            missing.append(label)
            continue
        weighted += float(rate) * entry["generated"]
        used += entry["generated"]
    if used <= 0:
        return None, {"cells": sorted(per_cell), "missing": missing,
                      "mode": "no_reference_for_any_cell_in_batch"}
    return weighted / used, {"cells": sorted(per_cell), "missing": missing,
                             "mode": "weighted_by_generated_goals"}


def dual_step(episodes, *, lam, q_ref):
    """The update-boundary dual move, with everything it read recorded.

    Called only from :func:`ppo_update`, which is the only place a batch boundary exists.
    """
    q, per_cell = batch_success(episodes)
    reference, reference_detail = reference_rate(episodes, q_ref)
    if reference is None:
        return {"applied": False, "reason": "no q_ref for any cell in this batch",
                "q_batch": q, "per_cell": per_cell,
                "lambda_before": float(lam), "lambda_after": float(lam),
                "reference": reference_detail}
    new_lambda, detail = unlimited_budget.update_lambda(lam, q, reference)
    detail.update({"applied": True, "per_cell": per_cell, "reference": reference_detail})
    return detail


def fresh_controller(seed):
    import v2_controller
    return v2_controller.fresh_controller(int(seed))


def save_checkpoint(path, controller, optimizer, state, *, identity_sha256):
    """Exclusive checkpoint write: an existing milestone is never overwritten."""
    torch, _np, _cm, _e, _l, rng_state, _r = deps()
    target = Path(path)
    if target.exists():
        raise RuntimeError("refusing to overwrite checkpoint %s" % (target,))
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "loss_schema": C.LOSS_SCHEMA,
        "identity_sha256": str(identity_sha256),
        "arm": state.get("arm"),
        "training_seed": int(state.get("training_seed")),
        "episode": int(state.get("episode", 0)),
        "controller": controller.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": dict(state),
        "rng": rng_state(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return target


def validate_checkpoint_payload(payload, *, arm, identity_sha256,
                                expected_seed=None, expected_episode=None):
    """Pure payload contract, preserving the original strict identity/arm check."""
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("not a Stage 5B checkpoint")
    if payload.get("identity_sha256") != str(identity_sha256):
        raise RuntimeError("checkpoint belongs to another experiment identity")
    if payload.get("arm") != arm:
        raise RuntimeError("checkpoint arm mismatch")
    state = payload.get("state") or {}
    if expected_seed is not None and (payload.get("training_seed") != expected_seed
                                     or state.get("training_seed") != expected_seed):
        raise RuntimeError("checkpoint seed mismatch")
    if expected_episode is not None and (payload.get("episode") != expected_episode
                                        or state.get("episode") != expected_episode):
        raise RuntimeError("checkpoint is not the fixed selected episode")
    return True


def load_checkpoint(path, *, arm, identity_sha256, expected_seed=None, expected_episode=None):
    """Restore controller, optimizer and state; refuse any other arm or identity."""
    torch, _np, _cm, _e, _l, _rng_state, restore_rng = deps()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("%s: not a Stage 5B checkpoint" % (path,))
    if payload.get("identity_sha256") != str(identity_sha256):
        raise RuntimeError("%s: checkpoint belongs to another experiment identity" % (path,))
    if payload.get("arm") != arm:
        raise RuntimeError("%s: checkpoint arm %r != %r" % (path, payload.get("arm"), arm))
    validate_checkpoint_payload(payload, arm=arm, identity_sha256=identity_sha256,
                                expected_seed=expected_seed, expected_episode=expected_episode)
    state = dict(payload["state"])
    controller = fresh_controller(int(state["training_seed"]))
    saved = payload["controller"]
    controller.inner.trunk.load_state_dict(saved["trunk"])
    controller.inner.action_head.load_state_dict(saved["action_head"])
    controller.inner.value_head.load_state_dict(saved["value_task_head"])
    controller.cost_head.load_state_dict(saved["value_cost_head"])
    optimizer = torch.optim.Adam(controller.parameters, lr=C.LEARNING_RATE)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload.get("rng"))
    return controller, optimizer, state


def parameter_identity(controller):
    return controller.parameter_summary()


def assert_restored_matches_pins(parameters, full_state, *, expected_parameter_sha256=None,
                                 expected_actor_sha256=None,
                                 expected_full_state_sha256=None):
    """Pure pin check for a restored base checkpoint; a mismatch never reaches a world.

    The parameter digest, the actor digest and the combined full-state digest are three
    different numbers. A parent checkpoint whose parameter-only digest was not recorded (the
    replay anchor) is still pinned by its actor and full-state digests.
    """
    if expected_parameter_sha256 is not None:
        actual = (parameters or {}).get("parameter_sha256")
        if actual != str(expected_parameter_sha256):
            raise RuntimeError("restored parameters %r != pinned %s"
                               % (actual, expected_parameter_sha256))
    if expected_actor_sha256 is not None:
        actual = (full_state or {}).get("actor")
        if actual != str(expected_actor_sha256):
            raise RuntimeError("restored actor %r != pinned %s"
                               % (actual, expected_actor_sha256))
    if expected_full_state_sha256 is not None:
        actual = (full_state or {}).get("sha256")
        if actual != str(expected_full_state_sha256):
            raise RuntimeError("restored full state %r != pinned %s"
                               % (actual, expected_full_state_sha256))
    return True


def load_base_checkpoint(path, *, arm, seed, episode, base_identity_sha256,
                         expected_parameter_sha256=None, expected_full_state_sha256=None,
                         expected_actor_sha256=None):
    """Restore one frozen parent checkpoint under the *parent* identity.

    The base files were written by the base experiment, so their payload carries the parent
    identity hash. That value is passed in explicitly for this one call; the global identity
    check is not disabled, relaxed or defaulted. The payload must match arm, seed and episode
    exactly, and the restored state must reproduce the registered parameter and full-state
    fingerprints before it may be used.
    """
    torch, _np, _cm, _e, _l, _rng_state, restore_rng = deps()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("%s: not a Stage 5B checkpoint" % (path,))
    if payload.get("identity_sha256") != str(base_identity_sha256):
        raise RuntimeError("%s: checkpoint belongs to another experiment identity" % (path,))
    if payload.get("arm") != arm:
        raise RuntimeError("%s: checkpoint arm %r != %r" % (path, payload.get("arm"), arm))
    validate_checkpoint_payload(payload, arm=arm, identity_sha256=str(base_identity_sha256),
                                expected_seed=int(seed), expected_episode=int(episode))
    state = dict(payload["state"])
    controller = fresh_controller(int(state["training_seed"]))
    saved = payload["controller"]
    controller.inner.trunk.load_state_dict(saved["trunk"])
    controller.inner.action_head.load_state_dict(saved["action_head"])
    controller.inner.value_head.load_state_dict(saved["value_task_head"])
    controller.cost_head.load_state_dict(saved["value_cost_head"])
    optimizer = torch.optim.Adam(controller.parameters, lr=C.LEARNING_RATE)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload.get("rng"))
    parameters = controller.parameter_summary()
    full_state = full_state_fingerprint(controller, optimizer)
    try:
        assert_restored_matches_pins(
            parameters, full_state, expected_parameter_sha256=expected_parameter_sha256,
            expected_actor_sha256=expected_actor_sha256,
            expected_full_state_sha256=expected_full_state_sha256)
    except RuntimeError as error:
        raise RuntimeError("%s: %s" % (path, error)) from None
    return {"controller": controller, "optimizer": optimizer, "state": state,
            "parameters": parameters, "full_state": full_state,
            "payload_episode": int(payload.get("episode")),
            "payload_seed": int(payload.get("training_seed")),
            "payload_arm": payload.get("arm"), "payload": payload}



def _digest_tensor(digest, tensor):
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())


def full_state_fingerprint(controller, optimizer):
    """Passive fingerprint of the communication network, optimizer and all live RNGs.

    Reads state without sampling, stepping an optimizer or changing any RNG. PILOC and CTDE
    internals are not inspected: only the communication controller's own parameters, the
    optimizer state and the three global RNG states are hashed.
    """
    torch, np, _cm, _e, _l, _rng_state, _restore = deps()
    actor = hashlib.sha256()
    for prefix, module in (("trunk", controller.inner.trunk),
                           ("action_head", controller.inner.action_head),
                           ("value_task_head", controller.inner.value_head),
                           ("value_cost_head", controller.cost_head)):
        for name, tensor in sorted(module.named_parameters()):
            actor.update(("%s.%s" % (prefix, name)).encode("utf-8"))
            _digest_tensor(actor, tensor)
    opt = hashlib.sha256()
    state = optimizer.state_dict()
    for param_id in sorted(state.get("state", {}), key=lambda value: int(value)):
        values = state["state"][param_id]
        for name in sorted(values):
            opt.update(("%s|%s" % (param_id, name)).encode("utf-8"))
            value = values[name]
            if torch.is_tensor(value):
                _digest_tensor(opt, value)
            else:
                opt.update(repr(value).encode("utf-8"))
    for group in state.get("param_groups", []):
        for key in sorted(group):
            opt.update(("%s=%r" % (key, group[key])).encode("utf-8"))
    rng_python = hashlib.sha256(repr(random.getstate()).encode("utf-8")).hexdigest()
    np_state = np.random.get_state()
    rng_numpy = hashlib.sha256(
        (str(np_state[0]) + "|" + str(np_state[2:])).encode("utf-8")
        + np.asarray(np_state[1]).tobytes()).hexdigest()
    rng_torch = hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest()
    combined = hashlib.sha256(
        ("|".join([actor.hexdigest(), opt.hexdigest(), rng_python, rng_numpy,
                   rng_torch])).encode("utf-8")).hexdigest()
    return {"sha256": combined, "actor": actor.hexdigest(), "optimizer": opt.hexdigest(),
            "rng": {"python": rng_python, "numpy": rng_numpy, "torch": rng_torch},
            "note": ("passive fingerprint; no sample, no forward, no RNG advance, no "
                     "PILOC/CTDE internals")}

