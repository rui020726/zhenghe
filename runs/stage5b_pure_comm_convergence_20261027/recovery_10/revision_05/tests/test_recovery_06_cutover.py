"""Pure recovery_06 tests: no checkpoint load, model, G3, World or training subprocess."""
import os
from pathlib import Path
import tempfile
import unittest

from stage5b_comm import recovery_cutover as X
from stage5b_comm import recovery_descendant as D
from stage5b_comm import recovery_protocol as R
from stage5b_comm import recovery_runtime as RT
from stage5b_comm import constants as C

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / X.CUTOVER_RELATIVE


def temporary():
    base = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP")
                or os.environ.get("TMP") or ROOT / "execution/puretests_tmp").resolve()
    RT.evidence.inside(ROOT, base)
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base)


def synthetic_runs():
    rows = []
    for (arm, seed), (donor_episode, _start, _end) in X.CUTOVER_RUNS.items():
        donor = {
            "path": str(ROOT / "synthetic" / ("%s_%d.pth" % (arm, seed))),
            "bytes": 1, "sha256": "1" * 64, "identity_sha256": "2" * 64,
            "cursor_from_closed_journal": donor_episode,
            "next_update": donor_episode // 4,
            "expected_full_state_sha256": "3" * 64,
            "expected_actor_sha256": "4" * 64,
        }
        rows.append({
            "arm": arm, "training_seed": seed, "start_episode": donor_episode,
            "end_episode_cap": donor_episode + 120, "new_world_cap": 120,
            "active": True, "next_update": donor_episode // 4, "donor": donor,
            "task_certified": False, "loss_stability_stable": False,
        })
    return rows


def update_item(index):
    return {
        "update_index": index, "episode_indices": list(range(index * 4, index * 4 + 4)),
        "samples": 384, "epochs": 4, "optimizer_steps": 8, "minibatches": 8,
        "policy_surrogate": 0.1, "value_loss_task": 0.2, "value_loss_cost": 0.3,
        "entropy": 0.4, "approx_kl": 0.001, "gradient_norm_mean": 0.5,
    }


def history_row(index):
    return {
        "episode_index": index, "arm": "T1", "seed": 91001,
        "scene": int(C.scene_for(91001, index)),
        "generated_goals": C.NUM_GOALS, "steps": C.STEPS,
    }


class CutoverPlanTests(unittest.TestCase):
    def test_donor_restart_cursors_and_exact_720_cap(self):
        runs = X.apply_donor_restart_runs(synthetic_runs())
        self.assertEqual(
            [(x["donor_episode"], x["start_episode"], x["end_episode_cap"]) for x in runs],
            [(1800, 1804, 1924), (1800, 1800, 1920), (1440, 1440, 1560),
             (1440, 1440, 1560), (1440, 1440, 1560), (1080, 1080, 1200)])
        self.assertEqual(sum(x["new_world_cap"] for x in runs), 720)
        self.assertEqual((runs[0]["next_update"], runs[0]["donor_next_update"]), (451, 450))
        self.assertEqual(runs[0]["known_history_gap"],
                         {"episodes": [1800, 1804], "updates": [450, 451]})
        self.assertTrue(all(x["branch_mode"] == X.BRANCH_MODE for x in runs))
        self.assertTrue(all(x["branch_mode"].startswith("DONOR_RNG_RESTART") for x in runs))

    def test_seed_plan_and_slot_plan_have_720_distinct_non_r5_keys(self):
        consumed = X.load_consumed_manifest(MANIFEST)
        self.assertEqual(len(consumed), 3)
        runs = X.apply_donor_restart_runs(synthetic_runs())
        seeds = D.seed_plan(runs, set(consumed))
        self.assertEqual(len(seeds), 720)
        keys = {(x["arm"], x["training_seed"], x["scene_seed"], x["action_rng_seed"])
                for x in seeds}
        self.assertEqual(len(keys), 720)
        self.assertFalse(keys & consumed)
        self.assertEqual([x["episode_index"] for x in seeds[:4]], [1804, 1805, 1806, 1807])
        plan = D.slot_body(ROOT, "stage5b_comm_recovery_recovery_06_revision_01", seeds)
        self.assertEqual(plan["world_cap"], 720)
        self.assertEqual(len(plan["slots"]), 720)
        self.assertTrue(all("/1800" not in x["slot"] and "/1801" not in x["slot"]
                            and "/1802" not in x["slot"] and "/1803" not in x["slot"]
                            for x in plan["slots"]))

    def test_real_r5_evidence_is_complete_and_read_only(self):
        body = X.validate_consumed_manifest(
            __import__("json").loads(MANIFEST.read_text(encoding="utf-8")))
        self.assertEqual(len(body), 3)
        self.assertEqual(sorted(x[2] for x in body), [2201901, 2201902, 2201903])


class GapAndEligibilityTests(unittest.TestCase):
    def test_cutover_checks_are_not_evaluable_until_3480(self):
        for endpoint in (2040, 2520, 3000):
            with self.subTest(endpoint=endpoint):
                result = R.check_eligibility("T1", 91001, endpoint)
                self.assertEqual(result["status"], R.CHECK_NOT_EVALUABLE)
                self.assertNotIn("passed", result)
                self.assertNotIn("streak", result)
        self.assertEqual(R.check_eligibility("T1", 91001, 3480)["status"],
                         R.CHECK_EVALUABLE)
        self.assertEqual(R.check_eligibility("T1", 91002, 1560)["status"],
                         R.CHECK_EVALUABLE)

    def test_material_allows_only_the_registered_gap(self):
        windows, required_rows, required_updates = RT._required_check_indices(2040)
        rows = {i: history_row(i) for i in required_rows if i not in range(1800, 1804)}
        updates = {i: update_item(i) for i in required_updates if i != 450}
        material = RT.validate_check_material("T1", 91001, 2040, rows, updates)
        self.assertEqual(material["allowed_known_gap_rows"], [1800, 1801, 1802, 1803])
        self.assertEqual(material["allowed_known_gap_updates"], [450])
        self.assertEqual(material["unexpected_missing_rows"], [])
        self.assertEqual(material["unexpected_missing_updates"], [])

        bad_rows = dict(rows)
        bad_rows.pop(1799)
        with self.assertRaisesRegex(RT.RecoveryRuntimeError, "unexpected missing history row"):
            RT.validate_check_material("T1", 91001, 2040, bad_rows, updates)
        bad_updates = dict(updates)
        bad_updates.pop(451)
        with self.assertRaisesRegex(RT.RecoveryRuntimeError,
                                    "outside the registered gap"):
            RT.validate_check_material("T1", 91001, 2040, rows, bad_updates)


class LockAndLedgerTests(unittest.TestCase):
    def test_root_phase_lock_rejects_second_owner_and_releases(self):
        with temporary() as temp:
            first = X.RootPhaseLock(temp)
            second = X.RootPhaseLock(temp)
            first.acquire()
            try:
                with self.assertRaisesRegex(X.CutoverError, "already held"):
                    second.acquire()
            finally:
                first.release()
            second.acquire()
            second.release()

    def test_recovery_error_accounting_uses_recovery_plan(self):
        with temporary() as temp:
            layout = __import__("types").SimpleNamespace(
                root=Path(temp), ledger=Path(temp) / "execution/ledger",
                train_rows=Path(temp) / "execution/rows/train")
            context = {
                "slot_plan": {"world_cap": 720, "slots": []},
                "payload": {"recovery_experiment_id": "r6", "namespace": R.WORLD_SEED_NAMESPACE},
            }
            accounting = RT.recovery_ledger_accounting(context, layout)
            self.assertEqual(accounting["basis"], "RECOVERY_CONTEXT_SLOT_PLAN")
            self.assertEqual(accounting["planned"], 720)
            self.assertEqual(accounting["claimed"], 0)
            self.assertFalse((Path(temp) / "protocol/WORLD_SLOT_PLAN.json").exists())


class RunPlanSchemaTests(unittest.TestCase):
    def test_planned_run_entry_returns_mapping_from_real_branch_plan(self):
        runs = X.apply_donor_restart_runs(synthetic_runs())
        context = __import__("types").SimpleNamespace(runs=runs)
        entry = RT.planned_run_entry(context, "T1", 91001)
        self.assertIsInstance(entry, dict)
        self.assertEqual((entry["donor_episode"], entry["start_episode"]), (1800, 1804))
        branches = RT.resume_branch_fields(entry, entry["donor_episode"],
                                           entry["start_episode"])
        self.assertTrue(branches["rng_restart_branch"])
        self.assertEqual(branches["branch_mode"], X.BRANCH_MODE)
        self.assertEqual(branches["history_gap"],
                         {"episodes": [1800, 1804], "updates": [450, 451]})

    def test_same_cursor_branch_has_no_gap_and_duplicate_run_rejects(self):
        runs = X.apply_donor_restart_runs(synthetic_runs())
        context = __import__("types").SimpleNamespace(runs=runs)
        entry = RT.planned_run_entry(context, "T1", 91002)
        fields = RT.resume_branch_fields(entry, entry["donor_episode"],
                                         entry["start_episode"])
        self.assertFalse(fields["rng_restart_branch"])
        self.assertIsNone(fields["history_gap"])
        with self.assertRaisesRegex(RT.RecoveryRuntimeError, "ambiguous"):
            RT.planned_run_entry(__import__("types").SimpleNamespace(runs=runs + [runs[0]]),
                                 "T1", 91001)



if __name__ == "__main__":
    unittest.main()
