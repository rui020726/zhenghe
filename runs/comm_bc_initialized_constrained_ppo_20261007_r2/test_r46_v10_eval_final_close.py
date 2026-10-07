from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
MODULE_PATH = RUN_DIR / "r46_v10_eval_final_close.py"
SPEC = importlib.util.spec_from_file_location("r46_v10_eval_final_close_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
V10 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V10
SPEC.loader.exec_module(V10)

import r46_v9_eval_runtime_close as V9


def synthetic_plan(entries: int = 3) -> dict:
    keys = [k for k in V10.fixed_eval_keys()
            if tuple(k) not in V10.ADOPTED_KEYS][:entries]
    normal = entries - 1
    items = []
    for index, key in enumerate(keys):
        reserve = index == entries - 1
        items.append({"key": list(key), "logical_job": V10.logical_job(key),
                      "attempt": 1, "category": "reserve" if reserve else "normal",
                      "slot": 2 if reserve else 389 + index,
                      "purpose": "technical-capacity-recovery" if reserve else "planned-eval"})
    return {"kind": "synthetic-plan", "branch": "test", "adopted_keys": [list(k) for k in V10.ADOPTED_KEYS],
            "new_claim_count": entries, "entries": items}


def write_claim(root: Path, item: dict, *, status: str, steps: int = 1000,
                slot: int | None = None, attempt: int = 1) -> Path:
    slug = item["logical_job"]
    claim_dir = root / "launch_claims" / slug
    claim_dir.mkdir(parents=True, exist_ok=True)
    path = claim_dir / f"attempt_{attempt:04d}.json"
    payload = {"logical_job": item["logical_job"], "attempt": attempt,
               "category": item["category"], "status": status,
               "launch_slot": item["slot"] if slot is None else slot,
               "environment_constructor_calls": 1 if status in ("complete", "failed_technical") else 0,
               "run_return_calls": 1 if status in ("complete", "failed_technical") else 0,
               "completed_steps": steps if status in ("complete", "failed_technical") else 0,
               "optimizer_step_calls": 0, "checkpoint_write_calls": 0}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def write_slot(root: Path, item: dict) -> Path:
    path = root / "launch_slots" / item["category"] / f"{item['slot']:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"slot": item["slot"], "category": item["category"]}) + "\n",
                    encoding="utf-8")
    return path


class LedgerStateMachineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=V10.HERE)
        self.root = Path(self._tmp.name) / "v10_like_root"
        (self.root / "launch_slots" / "normal").mkdir(parents=True)
        (self.root / "launch_slots" / "reserve").mkdir(parents=True)
        self.plan = synthetic_plan(3)

    def tearDown(self):
        self._tmp.cleanup()

    def _ledger(self, *, strict: bool):
        return V10.build_cumulative_ledger(self.plan, strict=strict, root=self.root)

    def test_zero_claims_partial_snapshot_is_legal_final_is_rejected(self):
        ledger = self._ledger(strict=False)
        self.assertEqual(ledger["new_claims"], 0)
        self.assertEqual(ledger["snapshot_mode"], "mid_run_partial")
        with self.assertRaises(V10.V10Error):
            self._ledger(strict=True)

    def test_partial_counts_one_and_two_are_legal_mid_run(self):
        entries = self.plan["entries"]
        for count in (1, 2):
            for item in entries[:count]:
                write_claim(self.root, item, status="complete")
                write_slot(self.root, item)
            ledger = self._ledger(strict=False)
            self.assertEqual(ledger["new_claims"], count)
        with self.assertRaises(V10.V10Error):
            self._ledger(strict=True)

    def test_pending_and_completed_coexist_mid_run(self):
        entries = self.plan["entries"]
        write_claim(self.root, entries[0], status="claimed_before_environment")
        write_claim(self.root, entries[1], status="complete")
        write_slot(self.root, entries[1])
        ledger = self._ledger(strict=False)
        self.assertEqual(ledger["new_claims"], 2)
        self.assertEqual(ledger["counts"]["completed_steps"],
                         V10.V7.BASE_STEPS + 3000 + 1000)

    def test_final_full_plan_passes_strict_gate(self):
        for item in self.plan["entries"]:
            write_claim(self.root, item, status="complete")
            write_slot(self.root, item)
        ledger = self._ledger(strict=True)
        self.assertEqual(ledger["new_claims"], 3)
        # Internal consistency for the synthetic plan: every completed claim is
        # counted exactly once, adopted consumption is separate, nothing refunded.
        self.assertEqual(ledger["total_used"],
                         ledger["normal_used"] + ledger["reserve_used"])
        self.assertEqual(ledger["normal_used"], 386 + 2 + 2)
        self.assertEqual(ledger["reserve_used"], 1 + 1)
        self.assertEqual(ledger["counts"]["completed_steps"],
                         V10.V7.BASE_STEPS + 3000 + 3000)

    def test_final_missing_one_is_rejected(self):
        entries = self.plan["entries"]
        for item in entries[:2]:
            write_claim(self.root, item, status="complete")
            write_slot(self.root, item)
        with self.assertRaises(V10.V10Error):
            self._ledger(strict=True)

    def test_duplicate_key_and_over_cap_are_rejected(self):
        entries = self.plan["entries"]
        for item in entries:
            write_claim(self.root, item, status="complete")
            write_slot(self.root, item)
        write_claim(self.root, entries[0], status="complete", attempt=2)
        with self.assertRaises(V10.V10Error):
            self._ledger(strict=False)

    def test_unplanned_key_is_rejected(self):
        item = dict(self.plan["entries"][0], logical_job="model_eval_new_s11_greedy_scene1301999")
        write_claim(self.root, item, status="complete")
        with self.assertRaises(V10.V10Error):
            self._ledger(strict=False)

    def test_over_cap_slots_are_rejected(self):
        entries = self.plan["entries"]
        for item in entries:
            write_claim(self.root, item, status="complete")
            write_slot(self.root, item)
        extra = dict(entries[0], slot=999)
        write_slot(self.root, extra)
        with self.assertRaises(V10.V10Error):
            self._ledger(strict=False)

    def test_failed_claims_do_not_refund_and_are_counted_separately(self):
        entries = self.plan["entries"]
        write_claim(self.root, entries[0], status="failed_technical")
        write_slot(self.root, entries[0])
        ledger = self._ledger(strict=False)
        self.assertEqual(ledger["normal_used"], 389)
        self.assertEqual(ledger["counts"]["technical_failed_claims"], 3)
        self.assertEqual(ledger["counts"]["environment_constructor_calls"], 389)

    def test_adopted_consumption_is_counted_separately_from_new_claims(self):
        ledger = self._ledger(strict=False)
        self.assertEqual(ledger["normal_used"], 388)
        self.assertEqual(ledger["reserve_used"], 1)
        self.assertEqual(len(ledger["base_identity"]["adopted_cells"]), 3)


class DerivedSnapshotGuardTests(unittest.TestCase):
    """A derived-snapshot failure must not flip already-persisted core claims."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=V10.HERE)
        self.root = Path(self._tmp.name) / "v10_like_root"
        (self.root / "launch_slots" / "normal").mkdir(parents=True)
        (self.root / "launch_slots" / "reserve").mkdir(parents=True)
        self.plan = synthetic_plan(3)
        self.claim_paths = [write_claim(self.root, item, status="complete")
                            for item in self.plan["entries"]]
        for item in self.plan["entries"]:
            write_slot(self.root, item)
        self.before = {str(p): V10.sha256_file(p) for p in self.claim_paths}

    def tearDown(self):
        self._tmp.cleanup()

    def test_snapshot_failure_records_warning_and_preserves_core_claims(self):
        original = V10.build_cumulative_ledger

        def broken(*args, **kwargs):
            raise RuntimeError("synthetic derived-snapshot failure")

        V10.build_cumulative_ledger = broken
        try:
            ledger = V10.write_ledger_closure(self.plan, self.root)
            self.assertEqual(ledger["status"], "deferred")
        finally:
            V10.build_cumulative_ledger = original
        for path, digest in self.before.items():
            self.assertEqual(V10.sha256_file(Path(path)), digest)
        warnings = list((self.root / "runtime" / "derived_snapshot_warnings").glob("*.json"))
        self.assertGreaterEqual(len(warnings), 1)

    def test_core_state_failure_still_stops(self):
        # Corrupt a core claim file: the raw ledger itself becomes unparsable.
        self.claim_paths[0].write_text("{not json", encoding="utf-8")
        with self.assertRaises(V10.V10Error):
            V10.write_ledger_closure(self.plan, self.root)


class AdoptionAndIdentityTests(unittest.TestCase):
    def test_preserved_1301002_bytes_match_registered_identity(self):
        expected = {
            "claim": "claim_sha256", "row": "row_sha256",
            "lifecycle": "lifecycle_sha256", "transcript": "transcript_sha256",
            "tape": "tape_sha256", "slot": "slot_record_sha256",
        }
        for rel_name, hash_name in expected.items():
            self.assertEqual(V10.sha256_file(V10.HERE / V10.CELL_1301002_REL[rel_name]),
                             V10.CELL_1301002_IDENTITY[hash_name], rel_name)
        self.assertEqual(V10.sha256_file(V10.HERE / V9.V9_ROOT / "ROUND46_R2_V9_STOP_STATE.json")
                         if False else
                         V10.sha256_file(V9.V9_ROOT / "ROUND46_R2_V9_STOP_STATE.json"),
                         V10.CELL_1301002_IDENTITY["v9_stop_state_sha256"])

    def test_adopted_receipts_and_rows_reusable_unchanged(self):
        self.assertEqual(V10.sha256_file(V9.V8.V8_ROOT / "receipts/ROUND46_R2_V8_ATTEMPT2_ADOPTION_RECEIPT.json"),
                         "d3570da3b31fe757d3a165bbd62e7236d9cecaee12667bc9ffd447cc3cbae22c")
        self.assertEqual(V10.sha256_file(V9.V9_ROOT / "receipts/ROUND46_R2_V9_CELL1301001_ADOPTION_RECEIPT.json"),
                         V9.sha256_file(V9.V9_ROOT / "receipts/ROUND46_R2_V9_CELL1301001_ADOPTION_RECEIPT.json"))
        self.assertEqual(V10.sha256_file(V9.V8.V8_ROOT / "adoption/attempt2_effective_row.json"),
                         "ea94ea06e239395be8754183b64549873e180dc38153b990f4b8c11427f79ffc")

    def test_quota_plan_matches_the_authorized_93_cell_map(self):
        plan = V10.build_quota_plan()
        self.assertEqual(plan["new_claim_count"], 93)
        self.assertEqual(plan["new_normal_slots"], list(range(389, 481)))
        self.assertEqual(plan["new_reserve_slots"], [2])
        self.assertEqual(plan["total_used_after"], 482)
        keys = [tuple(x["key"]) for x in plan["entries"]]
        for adopted in V10.ADOPTED_KEYS:
            self.assertNotIn(adopted, keys)
        self.assertEqual(len(set(keys)), 93)

    def test_no_performance_gate_in_the_adoption_audit(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('"performance_used_for_adoption": False', source)

    def test_synthetic_fixtures_never_touch_the_real_matrix(self):
        with tempfile.TemporaryDirectory(dir=V10.HERE) as directory:
            root = Path(directory) / "v10_like_root"
            (root / "launch_slots" / "normal").mkdir(parents=True)
            plan = synthetic_plan(3)
            write_claim(root, plan["entries"][0], status="complete")
            write_slot(root, plan["entries"][0])
        real_claims = list((V10.V10_ROOT / "launch_claims").glob("*/*.json")) \
            if V10.V10_ROOT.exists() else []
        for claim in real_claims:
            self.assertIn("model_eval_new_s11_greedy_scene1301002", claim.as_posix())


if __name__ == "__main__":
    unittest.main(verbosity=2)
