"""Pure recovery cursor and convergence-check qualification data.

This module imports only frozen constants and pure convergence arithmetic. It does not load
checkpoints, instantiate runtime/model/World objects, draw RNG, or train.
"""

from . import constants as C
from . import convergence

RECOVERY_STAGE_ID = "stage5b_pure_comm_convergence_20261027_recovery_01_revision_01"
WORLD_SEED_NAMESPACE = "stage5b-convergence-recovery-v1-20261028"
EPISODES_PER_RUN_PER_ROOT = 120
RUNS_PER_ROOT = 6
WORLD_CAP_PER_ROOT = 720

# Ordered exactly as the preregistered arm/seed roster. These are absolute episode cursors.
RUN_CURSORS = (
    ("T1", 91001, 1320),
    ("T1", 91002, 1320),
    ("T1", 91003, 960),
    ("T2", 91001, 960),
    ("T2", 91002, 960),
    ("T2", 91003, 600),
)

# Half-open episode and absolute PPO-update intervals. Only these three runs have the
# accepted missing-diagnostics exception; all other missing data remains a hard error.
KNOWN_GAPS = {
    ("T1", 91003): {"episodes": (600, 960), "updates": (150, 240)},
    ("T2", 91001): {"episodes": (600, 960), "updates": (150, 240)},
    ("T2", 91002): {"episodes": (600, 960), "updates": (150, 240)},
}

CHECK_NOT_EVALUABLE = "NOT_EVALUABLE_KNOWN_HISTORY_GAP"
CHECK_EVALUABLE = "EVALUABLE"


def run_cursor(arm, seed):
    """Return the fixed absolute cursor; reject every unregistered run."""
    key = (str(arm), seed)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    for expected_arm, expected_seed, cursor in RUN_CURSORS:
        if key == (expected_arm, expected_seed):
            return cursor
    raise ValueError("unregistered recovery run %r/%r" % key)


def stage_endpoint(arm, seed):
    """The fixed first recovery-root endpoint, exactly 120 episodes after its cursor."""
    return run_cursor(arm, seed) + EPISODES_PER_RUN_PER_ROOT


def absolute_check(end_episode):
    """Validate the unchanged absolute 1560 + 480*k check ladder."""
    if type(end_episode) is not int:
        raise ValueError("check endpoint must be an integer")
    first = C.CONVERGENCE_FIRST_CHECK_EPISODE
    step = C.CONVERGENCE_CHECK_STEP
    if end_episode < first or (end_episode - first) % step:
        raise ValueError("episode %d is not on the absolute convergence check ladder"
                         % end_episode)
    return True


def known_gap(arm, seed):
    """Return the applicable registered update-diagnostic gap; never weaken the evaluator."""
    from . import recovery_cutover as X
    key = (str(arm), seed)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    run_cursor(*key)
    return KNOWN_GAPS.get(key) or X.CUTOVER_GAPS.get(key)


def known_row_gap(arm, seed):
    """Only the explicit recovery_05 cutover lacks successful episode rows."""
    from . import recovery_cutover as X
    key = (str(arm), seed)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    run_cursor(*key)
    return X.CUTOVER_GAPS.get(key)


def check_eligibility(arm, seed, end_episode):
    """Return a disposition only; callers invoke the original evaluator when eligible.

    Known gaps are qualification-only. This function never returns a certificate or a
    `passed`/streak value and never weakens convergence.evaluate.
    """
    key = (str(arm), seed)
    run_cursor(*key)  # fail closed on an unregistered run or malformed seed
    absolute_check(end_episode)
    windows = convergence.window_bounds(end_episode)
    required = []
    for window in windows:
        start = int(window["start"])
        end = int(window["end"])
        first_update = start // C.EPISODES_PER_UPDATE
        last_update = end // C.EPISODES_PER_UPDATE
        required.append({"start": start, "end": end,
                         "updates": [first_update, last_update]})

    gap = known_gap(*key)
    intersections = []
    if gap is not None:
        gap_start, gap_end = gap["updates"]
        for index, item in enumerate(required):
            start, end = item["updates"]
            left, right = max(start, gap_start), min(end, gap_end)
            if left < right:
                intersections.append({"window": index, "updates": [left, right]})

    if intersections:
        return {"status": CHECK_NOT_EVALUABLE,
                "arm": key[0], "seed": key[1],
                "end_episode": end_episode,
                "known_gap_episodes": list(gap["episodes"]),
                "known_gap_updates": list(gap["updates"]),
                "windows": required, "intersections": intersections}
    return {"status": CHECK_EVALUABLE,
            "arm": key[0], "seed": key[1],
            "end_episode": end_episode, "windows": required}
