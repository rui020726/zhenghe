"""The (G, R, N) scale programme: which cell an episode belongs to, and how it is seeded.

Three axes are crossed:

* ``G``  goals per world -> ``RunConfiguration.num_goals``
* ``R``  rescue robots   -> ``rescue_species * rescue_per_species``
* ``N``  subscribers     -> ``RunConfiguration.comm_subscribers``, always ``N <= R``

``N <= R`` is a hard rule, not a preference: ``joint_driver`` validates every requested
subscriber id against ``task_adapter.env.agent_dic`` and raises on an unknown one, so a
fan-out larger than the roster would either crash or silently overstate the cost.

Cells are sampled per episode from the deterministic grid below, seeded only by
``(scene, G, R, N, episode_index)``. Two consequences, both deliberate:

1. A cell is visited by many episodes across the run, so ``q_ref`` for a cell is estimated
   from its own repeated draws rather than from one world.
2. Re-running the same episode index reproduces the same cell and the same action seed, so
   an evaluation can be paired with training exactly.

Nothing here reads or writes a file, and nothing imports torch: this module is pure
arithmetic on purpose, so the unit tests can check the whole grid without a world.
"""

from __future__ import annotations

import hashlib

#: The registered training grid. ``N`` never exceeds ``R``; the pairs are the cross product
#: minus the four impossible combinations (R=6 with N=12).
GOALS = (12, 24, 48)
ROBOTS = (6, 12)
SUBSCRIBERS = (3, 6, 12)

#: The G-scaling sweep: the same five (R, N) pairs at ten goal counts, step 6, so the load
#: ratio G/R moves in equal increments. Only 12/24/48 are *training* cells; the other seven
#: are evaluation-only, and nothing here may feed training (``CELLS`` below stays the 15).
#: ``GOALS_EXTENDED`` is what the configuration parser admits, so a world at G=30 is a
#: registered world rather than a relaxed check.
GOALS_EXTENDED = (12, 18, 24, 30, 36, 42, 48, 54, 60, 72)
GOALS_ACCEPTED = tuple(sorted(set(GOALS) | set(GOALS_EXTENDED)))

#: rescue_robots = rescue_species * rescue_per_species, and rescue_species is 3 in every run.
RESCUE_SPECIES = 3


def rescue_per_species(robots):
    """The per-species count that makes ``3 * per_species == robots``; refuses anything else."""
    robots = int(robots)
    if robots % RESCUE_SPECIES:
        raise ValueError("rescue robots %d is not a multiple of species %d"
                         % (robots, RESCUE_SPECIES))
    return robots // RESCUE_SPECIES


def cells():
    """Every valid ``(G, R, N)`` cell, in a fixed order so reports can key on the index."""
    out = []
    for goals in GOALS:
        for robots in ROBOTS:
            for subscribers in SUBSCRIBERS:
                if subscribers <= robots:
                    out.append((goals, robots, subscribers))
    return tuple(out)


CELLS = cells()
#: cell index -> (G, R, N). 15 cells: 3 goals x 5 realisable (R, N) pairs.
CELL_BY_INDEX = {index: cell for index, cell in enumerate(CELLS)}


def cell_label(cell):
    """``G24_R12_N6`` -- the grouping key every conclusion must be reported under."""
    goals, robots, subscribers = cell
    return "G%d_R%d_N%d" % (int(goals), int(robots), int(subscribers))


#: The five (R, N) pairs, in the order every table reports them.
PAIRS = tuple((robots, subscribers) for robots in ROBOTS for subscribers in SUBSCRIBERS
              if subscribers <= robots)


def g_scaling_cells(goals=GOALS_EXTENDED):
    """``(G, R, N)`` for the G sweep: ten goal counts crossed with the same five pairs.

    This is an **evaluation** grid. ``CELLS`` stays the 15 training cells, so nothing that
    samples cells for training can reach a G that was never trained on; a caller that wants an
    unseen goal count has to ask for it explicitly here.
    """
    return tuple((int(goal), robots, subscribers)
                 for goal in goals for robots, subscribers in PAIRS)


def validate_cell(cell, *, allow_extended_goals=False):
    """Check one cell. ``allow_extended_goals`` admits the G sweep's seven new counts.

    Training paths keep the default, so a G=30 cell is refused where it would change the
    sampling grid; evaluation paths pass ``True`` and get the registered extended world
    instead of a relaxed check.
    """
    goals, robots, subscribers = (int(value) for value in cell)
    admitted = GOALS_ACCEPTED if allow_extended_goals else GOALS
    if goals not in admitted:
        raise ValueError("goals %d not in %r" % (goals, admitted))
    if robots not in ROBOTS:
        raise ValueError("robots %d not in %r" % (robots, ROBOTS))
    if subscribers not in SUBSCRIBERS:
        raise ValueError("subscribers %d not in %r" % (subscribers, SUBSCRIBERS))
    if subscribers > robots:
        raise ValueError("subscribers N=%d must be <= rescue robots R=%d"
                         % (subscribers, robots))
    return (goals, robots, subscribers)


def _digest(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def cell_for(scene, episode_index, *, salt=""):
    """The cell this episode runs under: a stable function of the episode, nothing else.

    The cell is drawn from ``CELLS`` by the episode's own digest, so the grid is covered
    without any mutable counter and a replay of one episode lands on the same cell.
    """
    index = _digest("cell", int(scene), int(episode_index), salt) % len(CELLS)
    return CELL_BY_INDEX[index]


def episode_seed(scene, cell, episode_index, *, salt=""):
    """The action-RNG seed for one episode. Depends on the scene, the cell and the index.

    Including the cell is what makes two cells with the same scene genuinely different
    worlds rather than a relabelled copy of one draw.
    """
    goals, robots, subscribers = validate_cell(cell)
    return _digest("seed", int(scene), goals, robots, subscribers, int(episode_index), salt)


def eval_seed(scene, cell, *, salt=""):
    """The evaluation seed for a cell. Same inputs, same seed, every time.

    Extended goal counts are admitted: an evaluation at G=30 is a registered world, and its
    seed must be a function of the cell like any other. The 15 training cells' seeds are
    unchanged (the digest never saw the admitted set).
    """
    goals, robots, subscribers = validate_cell(cell, allow_extended_goals=True)
    return _digest("eval", int(scene), goals, robots, subscribers, salt)


def subscriber_spec_for(cell):
    """``comm_subscribers`` for a whole cell, using that cell's own roster as the reference.

    ``all`` when N equals the roster keeps the exact configuration earlier revisions ran;
    ``first:N`` narrows the fan-out without touching the roster, which is what lets R and N
    move independently.
    """
    goals, robots, subscribers = validate_cell(cell, allow_extended_goals=True)
    return "all" if subscribers == robots else "first:%d" % subscribers


def cost_scale(subscribers, goals):
    """``N * G``: the divisor that makes one byte comparable across cells.

    ``N`` is the fan-out and is what actually prices a send; ``R`` is deliberately absent,
    because execution capability is not a communication cost. Dividing by this is the only
    change that lets one lambda mean the same thing in a 12-goal/3-subscriber world and a
    48-goal/12-subscriber one.
    """
    subscribers = int(subscribers)
    goals = int(goals)
    if subscribers < 1 or goals < 1:
        raise ValueError("subscribers and goals must both be >= 1, got %d and %d"
                         % (subscribers, goals))
    return float(subscribers * goals)
