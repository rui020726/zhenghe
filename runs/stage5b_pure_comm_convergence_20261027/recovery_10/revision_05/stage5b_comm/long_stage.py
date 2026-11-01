"""Pure planning rules for one long continuation to a fixed completed-episode target."""
from __future__ import annotations

TARGET_COMPLETED_EPISODES = 5000
CHECKPOINT_INTERVAL_EPISODES = 120
ROSTER = (("T1", 91001), ("T1", 91002), ("T1", 91003),
          ("T2", 91001), ("T2", 91002), ("T2", 91003))


def _int(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(label + " must be an exact integer >= %d" % minimum)
    return value


def validate_parameters(target=TARGET_COMPLETED_EPISODES,
                        interval=CHECKPOINT_INTERVAL_EPISODES):
    _int(target, "target completed episodes", 1)
    _int(interval, "checkpoint interval", 1)
    if target % 4:
        raise ValueError("target completed episodes must close four-episode updates")
    if interval % 4:
        raise ValueError("checkpoint interval must close four-episode updates")
    return target, interval


def completed_target_from_payload(payload):
    """Return the frozen long-stage target, or None for an ordinary finite stage."""
    if type(payload) is not dict:
        raise ValueError("runtime context payload must be a mapping")
    target = payload.get("target_completed_episodes")
    interval = payload.get("checkpoint_interval_episodes")
    if target is None:
        if interval is not None:
            raise ValueError("checkpoint interval requires a completed-episode target")
        return None
    target, interval = validate_parameters(target, interval)
    if interval != CHECKPOINT_INTERVAL_EPISODES:
        raise ValueError("long stage must reuse the frozen 120-episode checkpoint interval")
    return target


def certificate_stop_required(target, certificate_end, completed_episode):
    """Ordinary stages stop on a new certificate; a long target always completes."""
    if target is not None:
        target, _interval = validate_parameters(target)
    if type(completed_episode) is not int or completed_episode < 0:
        raise ValueError("completed episode must be an exact non-negative integer")
    if certificate_end is not None and (
            type(certificate_end) is not int or certificate_end < 0):
        raise ValueError("certificate endpoint must be an exact non-negative integer")
    return target is None and certificate_end == completed_episode


def plan_run(cursor, completed, target=TARGET_COMPLETED_EPISODES):
    cursor = _int(cursor, "episode cursor")
    completed = _int(completed, "completed episode count")
    target, _ = validate_parameters(target)
    if completed > cursor:
        raise ValueError("completed count cannot exceed episode cursor")
    remaining = max(0, target - completed)
    return {
        "episode_cursor_before_stage": cursor,
        "completed_episode_count_before_stage": completed,
        "registered_missing_episode_count_before_stage": cursor - completed,
        "target_completed_episodes": target,
        "new_world_cap": remaining,
        "end_episode_cap": cursor + remaining,
        "active": remaining > 0,
    }


def plan_roster(states, target=TARGET_COMPLETED_EPISODES):
    target, _ = validate_parameters(target)
    if set(states) != set(ROSTER):
        raise ValueError("state mapping must contain the exact six-run roster")
    result = {}
    for key in ROSTER:
        state = states[key]
        result[key] = plan_run(state["cursor"], state["completed"], target)
    expected = 6 * target - sum(states[key]["completed"] for key in ROSTER)
    if sum(item["new_world_cap"] for item in result.values()) != expected:
        raise ValueError("long-stage world total does not close the target")
    return result


def checkpoint_schedule(completed_before, target=TARGET_COMPLETED_EPISODES,
                        interval=CHECKPOINT_INTERVAL_EPISODES):
    completed_before = _int(completed_before, "completed before stage")
    target, interval = validate_parameters(target, interval)
    if completed_before >= target:
        return []
    first = (completed_before // interval + 1) * interval
    schedule = list(range(first, target, interval))
    if target not in schedule:
        schedule.append(target)
    if any(value <= completed_before or value > target for value in schedule):
        raise ValueError("checkpoint schedule escaped its long-stage target")
    return schedule


def resume_after_checkpoint(checkpoint_completed, next_cursor,
                            target=TARGET_COMPLETED_EPISODES,
                            interval=CHECKPOINT_INTERVAL_EPISODES):
    checkpoint_completed = _int(checkpoint_completed, "checkpoint completed count")
    next_cursor = _int(next_cursor, "checkpoint next cursor")
    target, interval = validate_parameters(target, interval)
    if checkpoint_completed > target or next_cursor < checkpoint_completed:
        raise ValueError("checkpoint resume position exceeds target/cursor")
    if checkpoint_completed not in checkpoint_schedule(0, target, interval) and checkpoint_completed != target:
        raise ValueError("checkpoint is not on the frozen 120-episode-or-target schedule")
    remaining = target - checkpoint_completed
    return {
        "resume_completed_episode_count": checkpoint_completed,
        "resume_episode_cursor": next_cursor,
        "remaining_episodes_to_target": remaining,
        "end_episode_cap": next_cursor + remaining,
        "target_reached": remaining == 0,
    }
def resume_from_stage_checkpoint(stage_run_plan, checkpoint_episode,
                                 interval=CHECKPOINT_INTERVAL_EPISODES):
    cursor = _int(stage_run_plan["episode_cursor_before_stage"], "stage run cursor")
    completed_before = _int(stage_run_plan["completed_episode_count_before_stage"],
                            "stage completed count")
    target = _int(stage_run_plan["target_completed_episodes"], "stage target")
    checkpoint_episode = _int(checkpoint_episode, "checkpoint episode")
    if checkpoint_episode < cursor or (checkpoint_episode - cursor) % 4:
        raise ValueError("checkpoint is outside a complete four-episode prefix")
    executed = checkpoint_episode - cursor
    if executed and executed % interval:
        raise ValueError("checkpoint is not on the 120-episode stage interval")
    completed = completed_before + executed
    if completed > target:
        raise ValueError("checkpoint completed count exceeds target")
    return resume_after_checkpoint(completed, checkpoint_episode, target, interval)