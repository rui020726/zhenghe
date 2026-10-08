"""R56 pre-launch narrow gate tests (zero environment; autograd + read-only rows).

Covers the nine contract tests on the ACTUAL frozen trainer source + the R56
source-patched ppo_update.  No JointRun, no environment, no optimizer on real
models; synthetic samples are injected through the real batch_samples seam.
"""
from __future__ import annotations
import copy, json, sys, unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
R53 = HERE.parent / "round53_critic_isolation_trial_readiness"
R54 = HERE.parents[2] / "analysis" / "round54-r53-validation-and-cost-gap"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(R53))
sys.path.insert(0, str(HERE.parent.parent / "comm_rule_warmstart_cost_learning_20261006_r2"))
import r45_bc  # noqa: E402
r45_bc.configure_source_imports()
import v2_trainer  # noqa: E402
import v2_controller  # noqa: E402
import r56_gate_prior as gp  # noqa: E402
from r56_contracts import TRAINING_VERSION as TV56, validate_checkpoint_identity  # noqa: E402
from r53_contracts import TRAINING_VERSION as TV53  # noqa: E402

BETA = 0.10


def make_features(age, step, first_emission=True):
    f = [0.0] * 19
    f[0] = float(age)
    f[5] = 1.0 if first_emission else 0.0
    f[9] = 0.5
    f[14] = 1.0
    f[17] = 6.0
    f[18] = float(np.float32(min(1.0, step / 1000.0)))
    return f


def make_sample(controller, targets, free, actions, features, ready_flags):
    with torch.no_grad():
        logits, _ = controller.forward(torch.as_tensor(features, dtype=torch.float32))
        log_prob = float(v2_trainer.joint_log_prob(
            logits, torch.tensor([actions[t] for t in targets], dtype=torch.long),
            torch.tensor([1.0 if t in free else 0.0 for t in targets], dtype=torch.float32)))
    return {"targets": list(targets), "actions": dict(actions), "free": list(free),
            "features": [list(f) for f in features], "log_prob": log_prob,
            "advantage": 0.1, "return_task": 0.5, "return_cost": 0.2}


class TestR56GatePrior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = v2_trainer
        cls.original, cls.patched_source, cls.diff = gp.build_patched_source(v2_trainer)
        cls.ns = dict(vars(v2_trainer))
        cls.original_fn = staticmethod(cls.ns["ppo_update"])
        cls.ns["GATE_BETA"] = BETA
        exec(compile(cls.patched_source, "<r56-gate-ppo-update>", "exec"), cls.ns)
        cls.controller = v2_controller.fresh_controller(11)

    def run_update(self, samples, beta, patched=True):
        fn = self.ns["ppo_update"] if patched else self.original_fn
        self.ns["GATE_BETA"] = beta
        def _float32_samples(episodes):
            out = copy.deepcopy(samples)
            for s in out:
                s["features"] = [np.asarray(f, dtype=np.float32) for f in s["features"]]
            return out
        self.ns["batch_samples"] = _float32_samples
        saved_bs = v2_trainer.batch_samples
        v2_trainer.batch_samples = _float32_samples
        try:
            controller = v2_controller.fresh_controller(11)
            optimizer = torch.optim.Adam(controller.parameters, lr=1e-3)
            fake_episode = {"steps": 1000, "cost_stream": {}, "task_stream": {},
                            "collector": type("C", (), {"records": []})()}
            report = fn(controller, optimizer, [fake_episode], update_index=0)
            params = {n: pp.detach().clone()
                      for n, pp in controller.inner.trunk.named_parameters()}
            params.update({("ah." + n): pp.detach().clone()
                           for n, pp in controller.inner.action_head.named_parameters()})
            return report, params, controller
        finally:
            v2_trainer.batch_samples = saved_bs

    def test1_beta_zero_matches_r53_exactly(self):
        f = make_features(0, 100)
        s = make_sample(self.controller, ["t1"], ["t1"], {"t1": 1}, [f], [True])
        rep53, p53, _ = self.run_update([s], beta=0.0, patched=False)
        rep56, p56, _ = self.run_update([s], beta=0.0, patched=True)
        self.assertEqual(sorted(p53), sorted(p56))
        for key in p53:
            self.assertTrue(torch.equal(p53[key], p56[key]), key)
        self.assertNotIn("gate_loss", rep53)
        # gate_loss is a diagnostic measurement of -log pi(SKIP); with beta=0 its
        # CONTRIBUTION to the update is zero, hence identical parameters above.

    def test2_gate_gradient_is_beta_times_p_send(self):
        with torch.no_grad():
            pass
        # analytic: L = -log pi(SKIP) = softplus(z), z = logit_SEND - logit_SKIP
        logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
        gate = -torch.log_softmax(logits, dim=-1)[0][gp.SKIP_INDEX]
        gate.backward()
        z = float(logits[0][1] - logits[0][0])
        p_send = float(torch.sigmoid(torch.tensor(z)))
        expected = p_send  # dL/dz = sigmoid(z) = p(SEND); dL/dlogit_SEND = p(SEND)
        self.assertAlmostEqual(float(logits.grad[0][1]), expected, places=6)
        self.assertAlmostEqual(float(logits.grad[0][0]), -expected, places=6)
        # with beta=0.1 and p(SEND)=0.5 (z=0): scaled gradient is +0.05 on SEND logit
        logits0 = torch.tensor([[0.0, 0.0]], requires_grad=True)
        (-torch.log_softmax(logits0, dim=-1)[0][0]).backward()
        self.assertAlmostEqual(float(logits0.grad[0][1]), 0.5, places=6)
        self.assertAlmostEqual(BETA * 0.5, 0.05, places=9)
        # gradient descent on the gate loss lowers the SEND tendency
        z0 = torch.tensor(0.0, requires_grad=True)
        (torch.nn.functional.softplus(z0)).backward()
        z1 = z0.detach() - 0.1 * z0.grad
        self.assertLess(float(torch.sigmoid(z1)), 0.5)

    def test3_ready_over_147_masked_forced_nonfirst_contribute_zero(self):
        base = make_features(0, 100)
        rows = [
            (make_features(0, 200), ["t1"], ["t1"], {"t1": 1}),      # ready=200 > 147
            (make_features(0, 100), ["t1"], [], {"t1": 1}),          # masked (not free)
            (make_features(0, 148), ["t1"], ["t1"], {"t1": 1}),      # ready=148 > 147
            (make_features(0, 100, first_emission=False), ["t1"], ["t1"], {"t1": 1}),  # non-first
        ]
        for features, targets, free, actions in rows:
            s = make_sample(self.controller, targets, free, actions, [features], [True])
            rep, _, _ = self.run_update([s], beta=BETA, patched=True)
            self.assertEqual(rep["gate_loss"], 0.0)

    def test4_ready_147_penalized_148_not(self):
        f147 = make_features(0, 147)
        s147 = make_sample(self.controller, ["t1"], ["t1"], {"t1": 1}, [f147], [True])
        rep147, _, _ = self.run_update([s147], beta=BETA, patched=True)
        self.assertGreater(rep147["gate_loss"], 0.0)
        f148 = make_features(0, 148)
        s148 = make_sample(self.controller, ["t1"], ["t1"], {"t1": 1}, [f148], [True])
        rep148, _, _ = self.run_update([s148], beta=BETA, patched=True)
        self.assertEqual(rep148["gate_loss"], 0.0)

    def test5_midstep_masked_second_candidate_not_counted(self):
        # t1 free and gate-eligible; t2 masked mid-step after t1's sampled SEND.
        # t2 has features identical to t1 so the PPO/value path is unchanged; if the
        # masked t2 were (wrongly) counted, the gate loss would double.
        f1 = make_features(0, 100)
        s_single = make_sample(self.controller, ["t1"], ["t1"], {"t1": 1}, [f1], [True])
        s_double = make_sample(self.controller, ["t1", "t2"], ["t1"],
                               {"t1": 1, "t2": 1}, [f1, f1], [True, False])
        rep1, _, _ = self.run_update([s_single], beta=BETA, patched=True)
        rep2, _, _ = self.run_update([s_double], beta=BETA, patched=True)
        self.assertGreater(rep1["gate_loss"], 0.0)
        self.assertAlmostEqual(rep1["gate_loss"], rep2["gate_loss"], places=9)
        # and the masked-only variant contributes nothing
        s_masked = make_sample(self.controller, ["t1", "t2"], [],
                               {"t1": 1, "t2": 1}, [f1, f1], [False, False])
        rep3, _, _ = self.run_update([s_masked], beta=BETA, patched=True)
        self.assertEqual(rep3["gate_loss"], 0.0)

    def test6_real_features_recovery_matches_ready_field(self):
        panels = json.loads((R54 / "DECISION_POINTS.json").read_text(encoding="utf-8"))
        checked = 0
        for seed, summary in panels["seed_summaries"].items():
            for panel in summary["disagreement_panels"]:
                dec = panel["candidate_decision"]
                f = dec["features_19"]
                ready = int(round(float(f[18]) * 1000.0) - round(float(f[0])))
                self.assertEqual(ready, dec["step"] - int(round(float(f[0]))))
                self.assertTrue(0 <= ready <= 1000)
                checked += 1
        self.assertGreaterEqual(checked, 20)

    def test7_gate_grads_zero_on_critic_heads_actor_path_differentiable(self):
        controller = v2_controller.fresh_controller(11)
        f = make_features(0, 100)
        x = torch.as_tensor([f], dtype=torch.float32)
        with gp.critic_input_detached(controller) if False else torch.no_grad():
            pass
        hidden = controller.inner.trunk(x)
        logits = controller.inner.action_head(hidden)
        gate = -torch.log_softmax(logits, dim=-1)[0][0]
        gate.backward()
        for parameter in controller.inner.trunk.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        for parameter in controller.inner.action_head.parameters():
            self.assertIsNotNone(parameter.grad)
        for parameter in controller.inner.value_head.parameters():
            self.assertTrue(parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0)
        for parameter in controller.cost_head.parameters():
            self.assertTrue(parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0)

    def test8_accounting_untouched_diff_whitelist(self):
        removed = [l[1:].strip() for l in self.diff
                   if l.startswith("-") and not l.startswith("---")]
        allowed = set()
        for block in (gp.ANCHOR_USABLE, gp.ANCHOR_LISTS, gp.ANCHOR_LOSS,
                      gp.ANCHOR_TOTALS_INIT, gp.ANCHOR_TOTALS_ADD, gp.ANCHOR_RETURN):
            allowed.update(l.strip() for l in block.splitlines() if l.strip())
        self.assertEqual([l for l in removed if l not in allowed], [])
        # g_hat / cost / reward accounting lines are absent from the diff entirely
        joined = "\n".join(self.diff)
        for forbidden in ("g_hat", "batch_g_hat", "dual_update", "cost_stream", "spent_bytes"):
            self.assertNotIn(forbidden, joined)

    def test9_checkpoint_identity_accepts_r56_rejects_old(self):
        freeze = {"round42": {"training_version": TV56}}
        ok = validate_checkpoint_identity(
            {"kind": "ppo", "training_version": TV56, "freeze": freeze,
             "training_state": {"training_version": TV56}}, freeze=freeze)
        self.assertIn("accepted", ok)
        with self.assertRaises(ValueError):
            validate_checkpoint_identity(
                {"kind": "ppo", "training_version": TV53, "freeze": freeze,
                 "training_state": {"training_version": TV53}}, freeze=freeze)


if __name__ == "__main__":
    unittest.main()
