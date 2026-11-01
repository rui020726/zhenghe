"""Pure recovery action-seed mapping and rollout-key collision guard.

Runtime source contract (frozen R1 source): episode.py derives ``scene`` with
constants.scene_for(training_seed, episode_index) and derives the action seed from
v2_run.action_seed(training_seed, episode_index). It passes both to world.run_episode;
world.py puts ``scene`` into inherited config/RuntimeSpec and calls
``torch.manual_seed(action_rng_seed)``. Thus the run-scoped physical rollout key is
(arm, training_seed, scene_seed, action_rng_seed). Episode index and optional branch are
retained as provenance only and never make an otherwise identical key distinct.

This helper derives only the recovery action_rng_seed. It creates no RNG and does not prove
that prior_rollouts is complete or authentic. A future admission/frozen plan must supply a
complete, pinned snapshot of closed and unclosed claims, with historical runtime fields
reconstructed only from verified source evidence. Missing snapshot evidence fails closed.
"""

import hashlib

from . import constants as C
from . import recovery_protocol as R

_ALLOWED_ARMS = ("T1", "T2")
_REGISTERED_RUNS = frozenset((arm, seed) for arm, seed, _ in R.RUN_CURSORS)
_REQUIRED_ROLLOUT_FIELDS = frozenset((
    "arm", "training_seed", "episode_index", "scene_seed", "action_rng_seed"))
_ALLOWED_ROLLOUT_FIELDS = _REQUIRED_ROLLOUT_FIELDS | frozenset(("branch",))
_UINT64_MAX = (1 << 64) - 1


def _require_run(arm, training_seed):
    if type(arm) is not str or arm not in _ALLOWED_ARMS:
        raise ValueError("arm must be exactly 'T1' or 'T2'")
    if type(training_seed) is not int or (arm, training_seed) not in _REGISTERED_RUNS:
        raise ValueError("unregistered recovery run")


def _require_uint64(value, name):
    if type(value) is not int or not 0 <= value <= _UINT64_MAX:
        raise ValueError("%s must be an exact unsigned 64-bit integer" % name)


def _validated_rollout_key(record):
    if type(record) is not dict or not _REQUIRED_ROLLOUT_FIELDS <= set(record):
        raise ValueError("prior rollout record is missing required typed fields")
    if not set(record) <= _ALLOWED_ROLLOUT_FIELDS:
        raise ValueError("prior rollout record has unsupported fields")
    arm, seed = record["arm"], record["training_seed"]
    _require_run(arm, seed)
    episode = record["episode_index"]
    if type(episode) is not int or episode < 0:
        raise ValueError("prior episode_index must be an exact nonnegative integer")
    scene = record["scene_seed"]
    if type(scene) is not int or scene != C.scene_for(seed, episode):
        raise ValueError("prior scene_seed does not match the frozen scene schedule")
    _require_uint64(record["action_rng_seed"], "prior action_rng_seed")
    if "branch" in record and type(record["branch"]) is not str:
        raise ValueError("prior branch provenance must be an exact string")
    return (arm, seed, scene, record["action_rng_seed"])


def reject_rollout_collision(arm, training_seed, episode_index, scene_seed,
                             action_rng_seed, prior_rollouts):
    """Reject a repeated same-run physical scene/action-RNG key; ignore episode/branch labels."""
    _require_run(arm, training_seed)
    if type(episode_index) is not int or episode_index < R.run_cursor(arm, training_seed):
        raise ValueError("episode_index must be an exact integer at/after the run cursor")
    if type(scene_seed) is not int or scene_seed != C.scene_for(training_seed, episode_index):
        raise ValueError("scene_seed does not match the frozen scene schedule")
    _require_uint64(action_rng_seed, "action_rng_seed")
    if type(prior_rollouts) not in (tuple, list) or not prior_rollouts:
        raise ValueError("a nonempty concrete prior rollout snapshot is required")
    proposed = (arm, training_seed, scene_seed, action_rng_seed)
    for record in prior_rollouts:
        if _validated_rollout_key(record) == proposed:
            raise ValueError("same-run scene/action-RNG rollout key was already claimed")
    return True


def action_rng_seed(arm, training_seed, episode_index, scene_seed, prior_rollouts):
    """Map fixed-namespace SHA256 to uint64, then require a noncolliding full-key snapshot."""
    _require_run(arm, training_seed)
    if type(episode_index) is not int or episode_index < R.run_cursor(arm, training_seed):
        raise ValueError("episode_index must be an exact integer at/after the run cursor")
    if type(scene_seed) is not int or scene_seed != C.scene_for(training_seed, episode_index):
        raise ValueError("scene_seed does not match the frozen scene schedule")
    payload = "%s:sc-action:%d:%d" % (
        R.WORLD_SEED_NAMESPACE, training_seed, episode_index)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    result = int(digest[:16], 16)
    reject_rollout_collision(arm, training_seed, episode_index, scene_seed,
                             result, prior_rollouts)
    return result