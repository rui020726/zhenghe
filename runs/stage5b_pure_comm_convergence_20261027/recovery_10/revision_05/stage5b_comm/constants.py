"""Stage 2 draft: global episodes [1320,2040), pending real parent admission.

Only the finite continuation interval and its explicit ancestry change. Scientific constants
and the absolute 1560 + 480*k check ladder are inherited. No parent terminal hash is guessed.
The admission file must be completed from real terminal evidence and main-model acceptance
before any plan/identity, model load or world may be created.
"""

from __future__ import annotations
from pathlib import Path

RUN_ID = "stage5b_pure_comm_convergence_20261027_stage_02_revision_01"
REVISION_ID = "revision_01"
STAGE_INDEX = 2
IMPLEMENTATION_STATUS = "PENDING_PARENT_ADMISSION"
PROTOCOL_SCHEMA = "stage5b-comm-convergence-protocol/1"
IDENTITY_SCHEMA = "stage5b-comm-convergence-identity/1"
ROW_SCHEMA = "stage5b-comm-convergence-row/1"
LEDGER_SCHEMA = "stage5b-comm-convergence-ledger/1"
CONVERGENCE_SCHEMA = "stage5b-comm-convergence-certificate/1"
CONVERGENCE_REPORT_SCHEMA = "stage5b-comm-convergence-report/1"
BASE_ROW_SCHEMA = "stage5b-comm-convergence-row/1"

EXPERIMENT_BASE_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = EXPERIMENT_BASE_ROOT.parent
BASE_EXPERIMENT_ID = "stage5b_pure_comm_convergence_20261027_revision_01"
BASE_REVISION_ID = "revision_01"
BASE_ROOT = (EXPERIMENT_BASE_ROOT / "revision_01").as_posix()
PARENT_CHECKPOINT_NAME = "milestone_1320.pth"
LEGACY_LONGER_IDENTITY_SHA256 = "273aa9100fae44f9a5fa431228568fb5280e47019507f6c0058a2bc5e747cdfd"
LEGACY_LONGER_STOP_SHA256 = "278f0958d82411f8f6c152ed1dae9a109c9e51d580807b3f947ad36507764b2d"
LEGACY_LONGER_STOP_BYTES = 547
PARENT_TRAIN_STATUS = "PENDING_PARENT_ADMISSION"
PARENT_PROVENANCE = {
    "status": "PENDING_PARENT_ADMISSION",
    "required_parent_rows": 4320,
    "required_parent_terminal_episode": 1320,
    "required_parent_stop_absent": True,
    "note": "A normal stage-1 training terminal is required; legacy evaluation STOP stays old evidence.",
}

ATTACHMENT_SHA256 = "7b0419469644573305dcd18c1f3f413384c952c4752c62ebd486d72c21085cf7"

FEATURE_LAYOUT = "comm-send-state/1"
FEATURE_DIM = 19
FEATURE_NAMES = (
    "age_steps", "since_last_emission_time", "sent_count", "pending_event", "changed_view",
    "first_emission", "stage_changed", "requirements_changed", "members_changed",
    "progress_delta_abs", "revision_delta", "remaining_time_over_horizon", "urgency_now",
    "has_future_requirements", "queue_backlog_targets", "budget_remaining_fraction",
    "capacity_remaining_fraction", "subscribers", "step_over_horizon",
)
IDX_FIRST_EMISSION = 5
IDX_REMAINING_TIME = 11
IDX_HAS_FUTURE = 13
IDX_BUDGET_FRACTION = 15
IDX_STEP_FRACTION = 18

TRAIN_SCENES = (2201901, 2201902, 2201903, 2201904, 2201905, 2201906,
                2201907, 2201908, 2201909, 2201910, 2201911, 2201912)
EVAL_SCENES = (2301901, 2301902, 2301903, 2301904, 2301905, 2301906, 2301907, 2301908)
SMOKE_SCENE = 2201900
EVAL_CLASS = "DEV_REUSED_EVAL"
SCENE_CYCLE = len(TRAIN_SCENES)

TRAIN_SEEDS = (91001, 91002, 91003)
EVAL_WORLD_SEED = 20261026
SEED_OFFSET = {91001: 0, 91002: 4, 91003: 8}

#: Absolute episode indexes; resume after the parent's update 329 closes at episode 1319.
STAGE_START_EPISODE = 1320
STAGE_END_EPISODE = 2040
EPISODES_PER_UPDATE = 4
RESUME_FROM_EPISODE = STAGE_START_EPISODE
RESUME_FROM_UPDATE = RESUME_FROM_EPISODE // EPISODES_PER_UPDATE
NEW_EPISODES_PER_STAGE = STAGE_END_EPISODE - STAGE_START_EPISODE
NEW_UPDATES_PER_STAGE = NEW_EPISODES_PER_STAGE // EPISODES_PER_UPDATE
#: Resume checkpoints every 120 episodes so an interrupted stage can be restored and never
#: silently re-run. They are exclusive creates and are never overwritten.
CHECKPOINT_INTERVAL_EPISODES = 120
STAGE_CHECKPOINT_EPISODES = tuple(range(STAGE_START_EPISODE + CHECKPOINT_INTERVAL_EPISODES,
                                       STAGE_END_EPISODE + 1,
                                       CHECKPOINT_INTERVAL_EPISODES))
#: Historical name used by the monitor; the stage's own milestones are the six above.
NEW_CHECKPOINT_EPISODES = STAGE_CHECKPOINT_EPISODES

#: The single transport budget of this line of experiments, inherited unchanged.
BUDGET_BYTES = 58600
STEPS = 1000
SUBSCRIBERS = 6
NUM_GOALS = 24
B0_BYTES = 10000.0
FROZEN_AUTHORITY_BUDGET_BYTES = 29300

GAMMA = 1.0
GAE_LAMBDA = 0.95
CLIP = 0.2
ENTROPY_BONUS = 0.01
LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 0.5
EPOCHS_PER_BATCH = 4
MINIBATCH_STEPS = 256
STATE_SCHEMA = "stage5b-pacing-ppo-checkpoint/1"
LOSS_SCHEMA = "stage5b-pacing-advantage/1"
#: The frozen loop keeps a short final minibatch, so one update performs
#: 4 * ceil(samples / 256) optimizer steps: 64 for a full 4000-sample batch, 60 for 3828.
EXPECTED_OPTIMIZER_STEPS_PER_UPDATE = 64

ARMS_TRAIN = ("T1", "T2")
TASK_MULTIPLIER = 1.0
COST_MULTIPLIER = {"T1": 0.0, "T2": 0.1}
LAMBDA_FIXED = 1.0
CRITIC_LOSS_WEIGHT = 0.5

BASELINE_ARMS = ("normal_immediate", "always_hold", "late147")
PACING_ARMS = ("comm_windowed_pacing", "comm_priority_pacing")
BUDGET_FLOORS = ((0.05, 0.50), (0.15, 0.25), (1.00, 0.00))
PRIORITY_BAND = 0.10
PRIORITY_TAIL_CUTOFF = 0.15
PRIORITY_MIN_REMAINING = 0.5

#: ------------------------------------------------------------- convergence criterion
#: One window is 480 episodes = 40 scene cycles of 12 = 120 PPO updates. The window was
#: widened from 240 to 480 before freeze because 240 left too little CI margin on noisy
#: history; the equivalence bound delta itself is NOT relaxed. Three windows = 1440 episodes.
CONVERGENCE_WINDOW_EPISODES = 480
CONVERGENCE_WINDOW_CYCLES = CONVERGENCE_WINDOW_EPISODES // SCENE_CYCLE
CONVERGENCE_UPDATES_PER_WINDOW = CONVERGENCE_WINDOW_EPISODES // EPISODES_PER_UPDATE
CONVERGENCE_WINDOWS = 3
#: All three unordered pairs of the three windows must satisfy the equivalence bound.
CONVERGENCE_PAIRS = ((0, 1), (1, 2), (0, 2))
#: A run is converged only after two consecutive passing checks; the second pass must rest on
#: a further 480 brand-new episodes. 480 == the window width, so the second window set is
#: disjoint from the first except for one overlapping middle window.
CONVERGENCE_REQUIRED_CONSECUTIVE = 2
#: The first check with enough history that is also aligned on the 600 resume boundary:
#: windows [120,600), [600,1080), [1080,1560). Then 2040, 2520, 3000, ...
CONVERGENCE_FIRST_CHECK_EPISODE = 1560
CONVERGENCE_CHECK_STEP = CONVERGENCE_WINDOW_EPISODES
#: 90% equivalent interval from 10000 block-bootstrap resamples (orders 500 / 9499).
CONVERGENCE_INTERVAL_LEVEL = 0.90
CONVERGENCE_BOOTSTRAP_REPS = 10000
CONVERGENCE_LOWER_ORDER = 500
CONVERGENCE_UPPER_ORDER = 9499
#: Every replicate seed is derived from this label plus arm, seed, end episode and pair. It
#: never touches random/numpy/torch, so a check cannot perturb training.
CONVERGENCE_SEED_LABEL = "stage5b-convergence-equivalence/1"
CONVERGENCE_PAIR_BOUND = 0.025
CONVERGENCE_MEAN_RANGE_BOUND = 0.025
CONVERGENCE_SLOPE_BOUND = 0.0125
CONVERGENCE_BUDGET_RANGE_BOUND = 0.01
#: Three explicit sources, no recursive ancestor discovery or unpinned partial history.
HISTORY_EPISODES = (0, STAGE_START_EPISODE)
GRANDPARENT_HISTORY_EPISODES = (0, 200)
OLDER_PARENT_HISTORY_EPISODES = (200, 600)
PARENT_OWN_HISTORY_EPISODES = (600, STAGE_START_EPISODE)
PARENT_HISTORY_EPISODES = HISTORY_EPISODES
HISTORY_ROWS_PINNED = len(ARMS_TRAIN) * len(TRAIN_SEEDS) * (STAGE_START_EPISODE - 0)
#: Roots whose convergence certificates may be chained across stages. Stage 1 has none (the
#: parent round produced no check), but the interface is fixed here, mirrored in the protocol
#: and pinned in the identity, so the next stage can list its ancestor without ad-hoc globbing.
CONVERGENCE_ANCESTOR_ROOTS = ()
GRANDPARENT_EXPERIMENT_ID = "stage5b_pure_comm_budget58600_20261026_revision_02"
GRANDPARENT_ROOT = (RUNS_ROOT / "stage5b_pure_comm_budget58600_20261026" / "revision_02").as_posix()
GRANDPARENT_IDENTITY_SHA256 = "6fcba082c9fc23e87a01002fce94bcb3a10d07b8dc44b90a54ca4c03e657d70a"
GRANDPARENT_ROW_SCHEMA = "stage5b-comm-budget58600-row/1"
HISTORY_SOURCES = (
    {"source_id": "budget58600", "root": GRANDPARENT_ROOT,
     "experiment_id": GRANDPARENT_EXPERIMENT_ID, "episodes": [0, 200],
     "row_schema": GRANDPARENT_ROW_SCHEMA, "row_digits": 3,
     "telemetry_schema": "stage5b-ppo-telemetry/1",
     "identity_sha256": GRANDPARENT_IDENTITY_SHA256},
    {"source_id": "longer_training",
     "root": (RUNS_ROOT / "stage5b_pure_comm_longer_training_20261026" / "revision_02").as_posix(),
     "experiment_id": "stage5b_pure_comm_longer_training_20261026_revision_02",
     "episodes": [200, 600], "row_schema": "stage5b-comm-longer-training-row/1",
     "row_digits": 3, "telemetry_schema": "stage5b-ppo-telemetry/1",
     "identity_sha256": LEGACY_LONGER_IDENTITY_SHA256},
    {"source_id": "stage_1", "root": BASE_ROOT, "experiment_id": BASE_EXPERIMENT_ID,
     "episodes": [600, 1320], "row_schema": ROW_SCHEMA, "row_digits": 4,
     "telemetry_schema": "stage5b-ppo-telemetry/2", "identity_sha256": None},
)
#: The OLS slope of the criterion regresses the three WINDOW MEANS on x = (0, 1, 2) and is
#: expressed per full 480-episode window. Fitting the 40 in-window cycle points is explicitly
#: not part of this criterion.
CONVERGENCE_SLOPE_BASIS = "OLS_OF_THREE_WINDOW_MEANS_X_0_1_2_PER_480_EPISODE_WINDOW"
#: The byte-budget mean range is a required additional gate on cost-curve stability: exceeding
#: it means the run has NOT converged yet. It is never a data-validity error, never a STOP, and
#: never an independent claim about performance; a saturated budget is reported as saturation
#: and is legal.
CONVERGENCE_BUDGET_RANGE_ROLE = (
    "REQUIRED_COST_CURVE_STABILITY_GATE_FOR_CONVERGENCE_NOT_A_PERFORMANCE_CLAIM")
#: Stage 1 has no valid check; stage 2 can first confirm the two-pass streak at 2040.
STAGE_VERDICT_IF_NO_CERTIFICATE = "NOT_YET_CONVERGED"
STAGE_VERDICT_NOTE = "Stage 2 checks at absolute episodes 1560 and 2040; incomplete convergence keeps the goal active."

#: The registered 95% scene-paired dev-evaluation bootstrap of this family (unchanged).
BOOTSTRAP_SEED = 20261020
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_LOWER_ORDER = 249
BOOTSTRAP_UPPER_ORDER = 9749

#: Every launch of stage 2 and nothing else: no replay, no pacing, no evaluation world.
TRAIN_WORLDS = len(ARMS_TRAIN) * len(TRAIN_SEEDS) * NEW_EPISODES_PER_STAGE
REPLAY_WORLDS = 0
PACING_WORLDS = 0
SCIENCE_WORLDS = TRAIN_WORLDS
WORLD_CAP_TOTAL = TRAIN_WORLDS
STAGE_TRAIN_CAP = TRAIN_WORLDS
#: Declared for the later, separately frozen greedy evaluation of the six converged
#: checkpoints. Not planned, not claimed and not counted in this stage.
EVAL_WORLDS_AFTER_CONVERGENCE = len(ARMS_TRAIN) * len(TRAIN_SEEDS) * len(EVAL_SCENES)
NEW_EVAL_WORLDS = 0

#: Historical launch accounting. Nothing is re-claimed and no failure is relabelled.
PRIOR_APPROVED_CUMULATIVE_CAP = 8135
LEGACY_RECORDED_ACTUAL_CLAIMS = 3720
PRIOR_RECORDED_ACTUAL_CLAIMS = None  # populated from admitted real stage-1 evidence, never projected
#: Recorded parent actuals remain unknown in this draft. Admission derives them only from
#: real stage-1 claims/results. The cumulative registered cap12455 is not an actual count.
PROJECTED_ACTUAL_CUMULATIVE_IF_STAGE_FULL = None
PRIOR_APPROVED_STATUS = "APPROVED_CAP_8135_PARENT_ACTUAL_PENDING_ADMISSION"
OWN_CLAIM_CAP = WORLD_CAP_TOTAL
APPROVED_CUMULATIVE_LAUNCH_CAP = PRIOR_APPROVED_CUMULATIVE_CAP + WORLD_CAP_TOTAL
CUMULATIVE_LAUNCH_CAP = APPROVED_CUMULATIVE_LAUNCH_CAP
LAUNCH_CAP_APPROVED = True
CANDIDATE_STATUS = "PENDING_PARENT_ADMISSION"
STAGE_AUTHORITY = "USER_AUTHORISED_STAGE_2_FINITE_CAP_4320_CONDITIONAL_ON_PARENT_ADMISSION"

#: Parent checkpoint/state pins are intentionally absent from this draft. They are loaded
#: only from the admitted, main-model-accepted PARENT_ADMISSION.json evidence bundle.

WORKERS_MAX = 3
STARTUP_GATE_ARM = "T1"
STARTUP_GATE_SEED = 91001
THREADS_PER_WORKER = 1
EPISODE_WALL_CAP_SECONDS = 240
FILE_SIZE_CAP_BYTES = 10 * 1024 * 1024

#: Stage 2 produces one aggregated statement about the six runs. The five formal deliverables
#: belong to the later greedy-evaluation revision and are not produced here.
DELIVERABLES = ("CONVERGENCE_REPORT.json",)


def scene_for(seed, episode_index):
    """The frozen zero-based rotation: each seed owns an offset and the list wraps."""
    offset = SEED_OFFSET[int(seed)]
    return TRAIN_SCENES[(offset + int(episode_index)) % len(TRAIN_SCENES)]


def budget_floor(tau):
    """Piecewise budget floor of the pacing rule; tau is index 18 of the frozen vector."""
    value = float(tau)
    for cutoff, floor in BUDGET_FLOORS:
        if value < cutoff:
            return float(floor)
    return 0.0


def convergence_check_episodes(stage_end=None):
    """The check ladder 1560, 2040, 2520, ... truncated at this stage's end."""
    end = int(STAGE_END_EPISODE if stage_end is None else stage_end)
    episodes = []
    episode = CONVERGENCE_FIRST_CHECK_EPISODE
    while episode <= end:
        if episode > STAGE_START_EPISODE:
            episodes.append(episode)
        episode += CONVERGENCE_CHECK_STEP
    return tuple(episodes)


def train_plan():
    """Every stage-2 world: global [1320,2040), never renumbered to a local index."""
    plan = []
    for arm in ARMS_TRAIN:
        for seed in TRAIN_SEEDS:
            for index in range(STAGE_START_EPISODE, STAGE_END_EPISODE):
                plan.append({"slot": "train:%s:%d:%04d" % (arm, seed, index),
                             "phase": "train", "arm": arm, "seed": int(seed),
                             "episode": int(index),
                             "scene": int(scene_for(seed, index))})
    if len(plan) != TRAIN_WORLDS:
        raise RuntimeError("training plan holds %d worlds, expected %d"
                           % (len(plan), TRAIN_WORLDS))
    return plan


def slot_plan():
    """The one launch plan of this stage: 6 runs x 720 episodes = 4320 unique slots."""
    plan = train_plan()
    if len(plan) != WORLD_CAP_TOTAL:
        raise RuntimeError("slot plan holds %d worlds, expected %d"
                           % (len(plan), WORLD_CAP_TOTAL))
    if len({entry["slot"] for entry in plan}) != len(plan):
        raise RuntimeError("slot plan contains duplicate slot ids")
    return plan


def eval_arm_name(arm, seed):
    """The logical name of one converged run's evaluation arm (later stage only)."""
    return "%s_conv_s%d" % (arm, int(seed))


EVAL_LEARNING_ARMS = tuple(eval_arm_name(arm, seed)
                           for arm in ARMS_TRAIN for seed in TRAIN_SEEDS)


def all_eval_plan():
    """The later greedy evaluation: six converged runs x eight scenes = 48 worlds.

    It is declared so the future revision can freeze one plan without inventing a naming
    scheme after seeing the results. Nothing here is claimed in this training-only stage.
    """
    plan = []
    for arm in ARMS_TRAIN:
        for seed in TRAIN_SEEDS:
            for scene in EVAL_SCENES:
                plan.append({"slot": "eval:%s:%d:%d" % (arm, int(seed), int(scene)),
                             "phase": "eval", "arm": arm, "seed": int(seed),
                             "episode": None, "scene": int(scene)})
    if len(plan) != EVAL_WORLDS_AFTER_CONVERGENCE:
        raise RuntimeError("declared evaluation plan holds %d worlds, expected %d"
                           % (len(plan), EVAL_WORLDS_AFTER_CONVERGENCE))
    return plan


def base_checkpoint_path(item):
    """Absolute path of one pinned parent checkpoint, built from the frozen base root."""
    return (BASE_ROOT + "\\execution\\checkpoints\\%s_s%d\\%s"
            % (item["arm"], int(item["seed"]), item["name"]))


def base_train_row_path(arm, seed, episode_index):
    """Absolute path of one parent training row read as convergence history."""
    return (BASE_ROOT + "\\execution\\rows\\train\\train_%s_%d_%04d.json"
            % (arm, int(seed), int(episode_index)))


def history_source_for_episode(index):
    matches = [source for source in HISTORY_SOURCES
               if source["episodes"][0] <= int(index) < source["episodes"][1]]
    if len(matches) != 1:
        raise ValueError("history episode %r has %d explicit sources" % (index, len(matches)))
    return matches[0]


def history_row_path(arm, seed, index):
    if arm not in ARMS_TRAIN or seed not in TRAIN_SEEDS:
        raise ValueError("unregistered history run %r/%r" % (arm, seed))
    source = history_source_for_episode(index)
    name = "train_%s_%d_%0*d.json" % (arm, int(seed), source["row_digits"], int(index))
    return Path(source["root"]) / "execution" / "rows" / "train" / name

