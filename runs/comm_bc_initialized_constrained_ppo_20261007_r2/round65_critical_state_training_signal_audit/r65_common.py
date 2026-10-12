"""ROUND65 read-only audit: pure helpers shared by the analyzer, probe and tests.

Zero-environment by construction: these functions only do float/byte classification of a
19-dim comm-send-state/1 feature vector. No environment, provider, claim/slot, optimizer, PPO,
RNG or checkpoint access lives here.

The AUTHORITATIVE focal encoding is parsed at runtime from the ROUND64 L baseline tape
(see r65_audit.parse_focal). The module-level FOCAL_FEATURES / FOCAL_F32_HEX below are the
independently hand-entered TEST EXPECTATION: the analyzer must validate the parsed focal against
them and ABORT on mismatch; it must never silently fall back to the constant. Matching functions
take the parsed focal explicitly (focal32 / focal_bytes) so the constant cannot drift into science.
"""
from __future__ import annotations
import numpy as np

FEATURE_LAYOUT = "comm-send-state/1"
FEATURE_NAMES: tuple[str, ...] = (
    "age_steps", "since_last_emission_time", "sent_count", "pending_event", "changed_view",
    "first_emission", "stage_changed", "requirements_changed", "members_changed",
    "progress_delta_abs", "revision_delta", "remaining_time_over_scale", "urgency_now",
    "has_future_requirements", "queue_backlog_targets", "budget_remaining_fraction",
    "capacity_remaining_fraction", "subscribers", "step_over_horizon",
)
IDX_AGE, IDX_FIRST_EMISSION, IDX_HAS_FUTURE, IDX_BACKLOG, IDX_BUDGET, IDX_STEP = 0, 5, 13, 14, 15, 18
READY_CUT = 147.0
EPISODE_HORIZON = 1000.0

# Independently hand-entered TEST EXPECTATION (NOT the production anchor).
FOCAL_FEATURES = [0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,6.0,0.134]
FOCAL_F32_HEX = "0000000000000000000000000000803f0000803f0000803f000000000000000000000000000000000000803f0000803f000000000000803f0000803f0000803f0000803f0000c0404c37093e"


def f32(feats) -> np.ndarray:
    return np.asarray(feats, dtype=np.float32)


def f32_bytes(feats) -> bytes:
    return f32(feats).tobytes()


def ready_step(feats) -> int:
    """ready_step = round(f[18]*1000) - round(f[0]) from the float32 legal input only."""
    v = f32(feats)
    return int(round(float(v[IDX_STEP]) * 1000.0) - round(float(v[IDX_AGE])))


def is_early(feats) -> bool:
    return ready_step(feats) <= READY_CUT


def differing_dims(feats, focal32) -> list[dict]:
    a = f32(focal32); b = f32(feats); out = []
    for i in range(len(FEATURE_NAMES)):
        if a[i].tobytes() != b[i].tobytes():
            out.append({"index": i, "name": FEATURE_NAMES[i],
                        "focal": float(a[i]), "candidate": float(b[i])})
    return out


def is_exact_match(feats, focal_bytes) -> bool:
    """T0: exact 19-dim float32 byte identity with the parsed focal."""
    return f32_bytes(feats) == focal_bytes


def partial_t1(feats, focal32) -> tuple[bool, list[dict]]:
    """T1 PARTIAL match: same early route AND ready_step AND queue_backlog AND
    has_future_requirements as the parsed focal. Never upgraded to same-state."""
    focal32 = f32(focal32)
    if not is_early(feats):
        return False, []
    if ready_step(feats) != ready_step(focal32):
        return False, []
    v = f32(feats)
    if v[IDX_BACKLOG].tobytes() != focal32[IDX_BACKLOG].tobytes():
        return False, []
    if v[IDX_HAS_FUTURE].tobytes() != focal32[IDX_HAS_FUTURE].tobytes():
        return False, []
    return True, differing_dims(feats, focal32)


def classify_free_first_report(feats, focal32, focal_bytes) -> str:
    if is_exact_match(feats, focal_bytes):
        return "T0_exact"
    ok, _ = partial_t1(feats, focal32)
    if ok:
        return "T1_partial"
    if is_early(feats):
        return "T2_early_other"
    return "late_or_other"
