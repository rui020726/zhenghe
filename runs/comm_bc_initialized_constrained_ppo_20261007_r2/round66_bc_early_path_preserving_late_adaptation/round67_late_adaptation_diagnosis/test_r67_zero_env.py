"""ROUND67 zero-environment verification (no forward rows, no environment, no writes to science).

Runs the parts of this round's analysis that are pure computation or pure bookkeeping:

  1. the forward-row guard refuses to plan past the cap and never refunds;
  2. the routing arithmetic (ready_step = round(f[18]*1000) - round(f[0])) and the 147 boundary;
  3. the float32 input key is deterministic and byte-exact, so panel dedup is stable;
  4. the softmax damping identity behind the mechanism claim, checked arithmetically;
  5. the delivered JSON artefacts are internally consistent and inside the round budget;
  6. the two protected ROUND46 files are byte-unchanged;
  7. the R66 NEW / R60 ep0 / R60 ep128 checkpoint bytes are unchanged (read-only hashing).

No torch forward is executed here, so this test consumes 0 of the round's 5000 forward rows.
"""
from __future__ import annotations

import hashlib
import json
import struct
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
R66_DIR = HERE.parent
RAW = R66_DIR / "raw"
RUN_ROOT = RAW.parents[1]
CAP = 5000
PROTECTED = {
    "ROUND46_CURRENT_TASK.json": "cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b",
    "ROUND46_R2_V6_STOP_STATE.json": "93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18",
}
R66_EP128 = {"s11": "d5c003c60b67716f4d25a4cdfa170f43c6fda3265a782a8e63b30ca2293da3d7",
             "s22": "77c2606d559455f0892e22989f94d1509dd2fd01855cc03106d32aed601656cf",
             "s33": "8c61ea7b82baf1833bd1afa4a9850ab14bae88d2e668d240dad2ce1626a9f06a"}
R60_EP0 = {"s11": "f267558d0cc8933987c09576b8d6d6ad512dbc3c73653d059104705c1faca437",
           "s22": "249f769770fd0bffcc8df345539088890f9323f1338a224e3c8195d35adc8555",
           "s33": "5f47f28b2678a0091e2370345b622aad7bdf1d8064b19d3ed57282bc8f005e93"}
R60_EP128 = {"s11": "1c765c27294c673aa82b1ab7cd8dd8f379699ede8f924cc3438bfcd73fa0c60e",
             "s22": "79bd3769d6382e5e908b22db2794ebfbb22cab430162fb211d87bb196550e234",
             "s33": "573e4c26a4fee6f01405d4386ca73b518eeb1adb9cd8409bc83661e4113f6c5a"}


class RowGuardUnderTest:
    """Mirror of the guard used by the analysis scripts, exercised with fake rows."""

    def __init__(self, cap):
        self.cap = cap
        self.used = 0

    def charge(self, rows):
        if self.used + rows > self.cap:
            raise RuntimeError("cap")
        self.used += rows


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ready_step(vec):
    return round(float(vec[18]) * 1000) - round(float(vec[0]))


def f32_key(vec):
    return struct.pack("<" + "f" * len(vec), *[float(x) for x in vec])


def softmax2(m_skip, m_send):
    import math
    hi = max(m_skip, m_send)
    e0, e1 = math.exp(m_skip - hi), math.exp(m_send - hi)
    return e1 / (e0 + e1)


class TestRound67ZeroEnv(unittest.TestCase):
    def test_guard_refuses_past_cap_and_never_refunds(self):
        guard = RowGuardUnderTest(20)
        guard.charge(19)
        guard.charge(1)
        with self.assertRaises(RuntimeError):
            guard.charge(1)
        self.assertEqual(guard.used, 20)

    def test_guard_counts_both_models(self):
        # 10 distinct inputs x 2 models = 20 rows, which is exactly the cap.
        guard = RowGuardUnderTest(20)
        guard.charge(2 * 10)
        with self.assertRaises(RuntimeError):
            guard.charge(1)

    def test_routing_boundary(self):
        early = [0.0] * 19
        early[18] = 0.147
        self.assertEqual(ready_step(early), 147)
        self.assertLessEqual(ready_step(early), 147)
        late = [0.0] * 19
        late[18] = 0.148
        self.assertEqual(ready_step(late), 148)
        self.assertGreater(ready_step(late), 147)
        # f[0] (the step) subtracts: ready = round(f18*1000) - round(f0)
        mixed = [0.0] * 19
        mixed[0] = 5.0
        mixed[18] = 0.152
        self.assertEqual(ready_step(mixed), 147)

    def test_f32_key_is_byte_exact_and_deterministic(self):
        vec = [0.1] * 19
        self.assertEqual(f32_key(vec), f32_key([0.1] * 19))
        self.assertEqual(len(f32_key(vec)), 19 * 4)
        # a double that rounds to the same float32 must collapse to the same key
        self.assertEqual(f32_key([0.1]), f32_key([0.1 + 1e-18]))

    def test_softmax_damping_identity(self):
        # The mechanism claim: at p ~= 0.999 the gain dp/dm = p(1-p) ~= 0.001, so a logit move
        # of ~1.4 changes p by ~0.003, while at p = 0.91 the same move is visible.
        m = 6.9  # ~ p = 0.999
        p0 = softmax2(0.0, m)
        self.assertGreater(p0, 0.998)
        gain = p0 * (1 - p0)
        self.assertLess(gain, 0.005)
        for delta in (0.5, 1.0, 1.4, 2.6):
            dp = abs(softmax2(0.0, m + delta) - p0)
            self.assertLessEqual(dp, gain * delta + 1e-9)
        # and the contrast case: a mid-range probability moves visibly
        m2 = 2.2  # ~ p = 0.90
        p2 = softmax2(0.0, m2)
        self.assertTrue(0.85 < p2 < 0.95)
        self.assertGreater(abs(softmax2(0.0, m2 + 1.4) - p2), 0.05)

    def test_delivered_artifacts_are_consistent_and_inside_budget(self):
        panel = json.loads((HERE / "r67_probability_panel.json").read_text(encoding="utf-8"))
        self.assertEqual(panel["kind"], "round67-task23-joint-probability-panel/1")
        self.assertLessEqual(panel["forward_rows_used"], CAP)
        planned = sum(v["forward_rows"] for v in panel["panel_sizes"].values())
        self.assertEqual(panel["forward_rows_used"], planned)
        self.assertTrue(panel["early_path_bit_identical_on_panel"])
        self.assertEqual(panel["recompute_vs_saved"]["mismatched"], 0)
        self.assertLessEqual(panel["recompute_vs_saved"]["max_abs_diff"], 1e-6)
        for route in ("early", "late"):
            self.assertEqual(panel["route_summary"][route]["argmax_flip_vs_BC"], 0)

        pair = json.loads((HERE / "r67_paired_probabilities.json").read_text(encoding="utf-8"))
        self.assertEqual(pair["blocks"]["all_early_free"]["n_action_diff"], 0)
        self.assertEqual(pair["blocks"]["all_late_free"]["n_action_diff"], 0)
        self.assertEqual(pair["blocks"]["all_early_free"]["n_prob_diff"], 0)
        self.assertGreater(pair["blocks"]["all_late_free"]["n_prob_diff"], 0)

        action = json.loads((HERE / "r67_action_equivalence.json").read_text(encoding="utf-8"))
        self.assertEqual(action["eval_cells_with_action_diffs"], [])
        self.assertEqual(action["focal_cells_with_action_diffs"], [])
        self.assertEqual(action["eval_cells_summary"]["row_missing"], 0)
        self.assertEqual(action["eval_cells_summary"]["tape_missing"], 0)

        exposure = json.loads((HERE / "r67_training_exposure.json").read_text(encoding="utf-8"))
        self.assertEqual(exposure["journals"]["ppo_update_calls"], 96)
        self.assertEqual(exposure["journals"]["optimizer_step_calls"], 540)
        for seed in ("s11", "s22", "s33"):
            self.assertEqual(exposure["per_seed"][seed]["updates"], 32)
            self.assertEqual(exposure["per_seed"][seed]["optimizer_steps_total"], 180)
            self.assertEqual(exposure["per_seed"][seed]["episode_count"], 128)
            self.assertTrue(exposure["per_seed"][seed]["parameters_moved_every_update"])
        for diag in exposure["gate_prior_diagnostics"]:
            self.assertEqual(diag["prior_trainable_grad_l1"], 0.0)

        ledger = json.loads((HERE / "r67_forward_row_ledger.json").read_text(encoding="utf-8"))
        total = sum(e["rows"] for e in ledger["events"])
        self.assertLessEqual(total, CAP)
        self.assertGreaterEqual(total, panel["forward_rows_used"])

    def test_protected_files_unchanged(self):
        for name, expected in PROTECTED.items():
            path = RUN_ROOT / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(sha256_file(path), expected, name)

    def test_model_bytes_unchanged(self):
        for tag, table, folder in (("r66_ep128", R66_EP128, RAW / "checkpoints"),
                                   ("r60_ep0", R60_EP0, RUN_ROOT / "round60_partitioned_actor_head_ablation" / "raw" / "checkpoints"),
                                   ("r60_ep128", R60_EP128, RUN_ROOT / "round60_partitioned_actor_head_ablation" / "raw" / "checkpoints")):
            for seed, expected in table.items():
                suffix = "ep128.pt" if tag != "r60_ep0" else "before_b0.pt"
                path = folder / f"v2_train_s{seed[1:]}_{suffix}"
                self.assertTrue(path.is_file(), str(path))
                self.assertEqual(sha256_file(path), expected, f"{tag} s{seed[1:]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
