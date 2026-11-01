"""Small synthetic contracts only. No CP load, neural network, subprocess or World."""
import copy
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
from stage5b_comm import recovery_descendant as D, recovery_runtime as RT
from stage5b_comm import recovery_protocol as R, constants as C, convergence

ROOT = Path(__file__).resolve().parents[1]


def temporary():
    base = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP")
                or os.environ.get("TMP") or ROOT / "execution/descendant_test_tmp").resolve()
    RT.evidence.inside(ROOT, base)
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base)


def update(index):
    return {"update_index": index, "episode_indices": list(range(index*4, index*4+4)),
            "samples": 384, "epochs": 4, "optimizer_steps": 8, "minibatches": 8,
            "policy_surrogate": .1, "value_loss_task": .2, "value_loss_cost": .3,
            "entropy": .4, "approx_kl": .001, "gradient_norm_mean": .5}


def terminal():
    rid = "recovery-train-10-123"
    summary = {"schema": "stage5b-recovery-train-terminal-summary/1",
               "child_exit": 0, "tool_exit": 0, "cli_returncode": 0, "drift_count": 0,
               "stage_complete": True, "global_stop": False,
               "source_context_auth_drift": []}
    receipt = {"schema": "stage5b-phase-receipt/1", "phase": "recovery-train",
               "stage": "terminal", "returncode": 0, "receipt_id": rid,
               "detail": {"status": "FINITE_RECOVERY_STAGE_COMPLETE"}}
    raw = {"exit_code": 0, "output": "CHILD_EXIT=0\nDRIFT_COUNT=0\n" + json.dumps(
        {"phase": "recovery-train", "returncode": 0, "receipt_id": rid}) + "\n"}
    return summary, receipt, raw


def parent_fixture(root):
    runs, states = [], {}
    cursors = (1440, 1440, 1080, 1080, 1080, 720)
    for (arm, seed, initial), cursor in zip(R.RUN_CURSORS, cursors):
        run = {"arm": arm, "training_seed": seed, "start_episode": initial,
               "end_episode_cap": initial+120, "new_world_cap": 120}
        donor = {"path": str(root/"donor"/("%s_%d.pth" % (arm, seed))),
                 "bytes": 10, "sha256": "1"*64, "identity_sha256": "2"*64,
                 "observed_episode": cursor, "next_update": cursor//4,
                 "next_update_basis": "CLOSED_LAST_JOURNAL_REQUIRE_PAYLOAD_MATCH",
                 "expected_full_state_sha256": "3"*64,
                 "expected_actor_sha256": "4"*64,
                 "stored_adam_parameter_step": None,
                 "payload_observation": "NOT_LOADED_G3_REQUIRED"}
        runs.append(run)
        states[(arm, seed)] = {"cursor": cursor, "next_update": cursor//4,
                               "certified": False, "task_certified": False,
                               "task_certificate_end_episode": None,
                               "loss_stability_stable": False, "donor": donor}
    pin = {"path": str(root/"parent"/"base.json"), "bytes": 1, "sha256": "5"*64}
    native_path = str(root/"native.py")
    context = types.SimpleNamespace(
        stage_index=2,
        context_pin=pin,
        payload={"protocol": pin, "candidate_manifest": pin, "candidate_index": pin},
        source_manifest={"files": [{"category": "native", "path": native_path,
                                    "sha256": "6"*64}]})
    return types.SimpleNamespace(
        runs=runs, states=states, context=context, closure=[], run_directory=str(root/"run"),
        rows={key: {} for key in D.ROSTER}, updates={key: {} for key in D.ROSTER},
        old_keys={("T1", 91001, 2201901, 1)}, certificates=[])


class CompletedParentTests(unittest.TestCase):
    def test_actual_terminal_schema_is_accepted_without_marking_goal_complete(self):
        summary, receipt, raw = terminal()
        self.assertEqual(D.validate_terminal(summary, receipt, raw)["status"],
                         "FINITE_RECOVERY_STAGE_COMPLETE")
        self.assertNotIn("goal_complete", D.validate_terminal(summary, receipt, raw))

    def test_incomplete_failed_unknown_bool_and_stop_parent_reject(self):
        summary, receipt, raw = terminal()
        for key, value in (("child_exit", None), ("tool_exit", True),
                           ("cli_returncode", 1), ("drift_count", 1),
                           ("stage_complete", False), ("global_stop", True)):
            with self.subTest(key=key):
                bad = dict(summary, **{key: value})
                with self.assertRaises(RT.RecoveryRuntimeError):
                    D.validate_terminal(bad, receipt, raw)

    def test_train_receipt_id_and_real_outer_exit_are_required(self):
        summary, receipt, raw = terminal()
        for bad_raw in (dict(raw, exit_code=1), dict(raw, exit_code=None),
                        dict(raw, exit_code=True), {"other_zero": 0, "output": raw["output"]},
                        dict(raw, output=raw["output"].replace("10-123", "11-123"))):
            with self.subTest(raw=bad_raw), self.assertRaises(RT.RecoveryRuntimeError):
                D.validate_terminal(summary, receipt, bad_raw)

    def test_dynamic_cursor_comes_from_closed_parent_not_initial_inventory(self):
        parent = parent_fixture(ROOT)
        runs = D.next_runs(parent.runs, parent.states)
        self.assertEqual([x["start_episode"] for x in runs],
                         [1440,1440,1080,1080,1080,720])
        self.assertEqual([x["end_episode_cap"] for x in runs],
                         [1560,1560,1200,1200,1200,840])
        self.assertEqual(sum(x["new_world_cap"] for x in runs), 720)
        for x in runs:
            self.assertEqual(x["next_update"], x["start_episode"]//4)

    def test_task_certificate_without_loss_review_continues_and_preserves_donor(self):
        parent = parent_fixture(ROOT)
        state = parent.states[("T1", 91001)]
        state.update(certified=True, task_certified=True,
                     task_certificate_end_episode=1440)
        runs = D.next_runs(parent.runs, parent.states)
        self.assertTrue(runs[0]["active"])
        self.assertTrue(runs[0]["task_certified"])
        self.assertIs(runs[0]["donor"], state["donor"])
        self.assertEqual((runs[0]["start_episode"], runs[0]["end_episode_cap"],
                          runs[0]["new_world_cap"]), (1440, 1560, 120))
        self.assertEqual(len(D.seed_plan(runs, parent.old_keys)), 720)
    def test_parent_cursor_and_journal_next_update_must_agree(self):
        parent = parent_fixture(ROOT)
        parent.states[("T2",91003)]["next_update"] = 179
        with self.assertRaisesRegex(RT.RecoveryRuntimeError, "cursor/update"):
            D.next_runs(parent.runs, parent.states)

    def test_closed_run_requires_exact_row_update_prefix_and_terminal(self):
        run = {"arm":"T1","training_seed":91001,"start_episode":1440,
               "end_episode_cap":1444}
        result = {"episodes":4,"terminal_episode":1444,"status":"ok","converged":False}
        tele = {"schema":RT.TELEMETRY_SCHEMA,"arm":"T1","training_seed":91001,
                "start_episode":1440,"terminal_episode":1444,"end_episode_cap":1444,
                "episodes":[{"episode_index":x} for x in range(1440,1444)],
                "updates":[update(360)]}
        self.assertEqual(D.validate_closed_run(run,result,tele),1444)
        bad = copy.deepcopy(tele)
        bad["episodes"][1]["episode_index"] = 1440
        with self.assertRaisesRegex(RT.RecoveryRuntimeError,"gap/duplicate"):
            D.validate_closed_run(run,result,bad)
        bad = copy.deepcopy(tele)
        bad["updates"][0]["entropy"] = float("inf")
        with self.assertRaises(RuntimeError):
            D.validate_closed_run(run,result,bad)

    def test_prefix_merge_rejects_unexpected_gap_and_keeps_registered_gap_unchanged(self):
        rows = {i:{} for i in range(4)}
        ups = {0:update(0)}
        added = {i:{} for i in range(4,8)}
        new = {1:update(1)}
        merged, actual = D.merge_prefix(rows,ups,added,new,4,8,("T1",91001))
        self.assertEqual(sorted(merged),list(range(8)))
        self.assertEqual(sorted(actual),[0,1])
        self.assertEqual(R.KNOWN_GAPS[("T1",91003)]["updates"],(150,240))
        with self.assertRaisesRegex(RT.RecoveryRuntimeError,"row gap"):
            D.merge_prefix(rows,ups,{4:{},5:{},6:{}},new,4,8,("T1",91001))
        with self.assertRaisesRegex(RT.RecoveryRuntimeError,"journal update gap"):
            D.merge_prefix(rows,ups,added,{},4,8,("T1",91001))


def review_fixture(parent, key, decision="LOSS_STABLE_CONFIRMED"):
    cursor = parent.states[key]["next_update"]
    records = {}
    for index in range(cursor - 60, cursor):
        item = update(index)
        # Deliberately nonmonotonic values: the human decision is not a numeric veto rule.
        item["policy_surrogate"] = -1.0 if index % 2 else 2.0
        item["value_loss_task"] = float(index % 7)
        item["value_loss_cost"] = float((index * 3) % 11)
        records[index] = item
    parent.updates[key] = records
    ranges = [(cursor-60, cursor-40), (cursor-40, cursor-20), (cursor-20, cursor)]
    windows = [{"start_update": start, "end_update_exclusive": end}
               for start, end in ranges]
    flattened = []
    for start, end in ranges:
        for index in range(start, end):
            point = {"update_index": index}
            for metric in D.LOSS_METRICS:
                point[metric] = float(records[index][metric])
            flattened.append(point)
    entry = {
        "arm": key[0], "training_seed": key[1], "decision": decision,
        "reviewer": "reviewer-1", "reviewed_utc": "2026-10-28T12:00:00Z",
        "rationale": "Reviewed raw values and median/MAD in all three windows; PPO losses may be nonmonotonic.",
        "acknowledged_nonmonotonicity": True,
        "windows": windows,
    }
    return {
        "schema": D.LOSS_REVIEW_SCHEMA, "status": "HUMAN_REVIEW_COMPLETE",
        "parent_context_sha256": parent.context.context_pin["sha256"],
        "reviews": [entry],
    }


class LossStabilityContinuationTests(unittest.TestCase):
    def setUp(self):
        self.parent = parent_fixture(ROOT)
        self.key = ("T1", 91001)
        self.parent.states[self.key].update(
            certified=True, task_certified=True, task_certificate_end_episode=1440)

    def test_missing_review_defaults_to_continue(self):
        with temporary() as temp:
            pin, reviews = D.load_loss_stability_review(Path(temp), self.parent)
        self.assertIsNone(pin)
        self.assertEqual(reviews, {})
        run = D.next_runs(self.parent.runs, self.parent.states)[0]
        self.assertTrue(run["active"])
        self.assertFalse(run["loss_stability_stable"])
        self.assertEqual(run["loss_stability_review"]["status"], "REVIEW_REQUIRED")

    def test_three_window_human_decision_retires_only_task_certified_run(self):
        review = review_fixture(self.parent, self.key)
        decisions = D.validate_loss_stability_review(review, self.parent, ROOT)
        record = decisions[self.key]
        self.assertEqual([(x["start_update"], x["end_update_exclusive"])
                          for x in record["windows"]], [(300,320),(320,340),(340,360)])
        self.assertEqual(len(record["windows"]), 3)
        self.assertEqual(set(record["windows"][0]["metrics"]), set(D.LOSS_METRICS))
        self.assertEqual(record["decision"], "LOSS_STABLE_CONFIRMED")
        runs = D.next_runs(self.parent.runs, self.parent.states, decisions)
        self.assertFalse(runs[0]["active"])
        self.assertTrue(runs[0]["task_certified"])
        self.assertTrue(runs[0]["loss_stability_stable"])
        self.assertEqual(runs[0]["new_world_cap"], 0)
        self.assertEqual(sum(item["new_world_cap"] for item in runs), 600)

    def test_continue_decision_and_unearned_task_certificate_never_retire(self):
        review = review_fixture(self.parent, self.key, "CONTINUE_TRAINING")
        decisions = D.validate_loss_stability_review(review, self.parent, ROOT)
        self.assertTrue(D.next_runs(self.parent.runs, self.parent.states, decisions)[0]["active"])
        self.parent.states[self.key]["certified"] = False
        self.parent.states[self.key]["task_certified"] = False
        with self.assertRaises(RT.RecoveryRuntimeError):
            D.validate_loss_stability_review(review, self.parent, ROOT)
        self.assertTrue(D.next_runs(self.parent.runs, self.parent.states)[0]["active"])

    def test_review_is_bound_to_parent_and_complete_finite_windows(self):
        review = review_fixture(self.parent, self.key)
        bad = copy.deepcopy(review)
        bad["parent_context_sha256"] = "0" * 64
        with self.assertRaises(RT.RecoveryRuntimeError):
            D.validate_loss_stability_review(bad, self.parent, ROOT)
        bad = copy.deepcopy(review)
        bad["reviews"][0]["acknowledged_nonmonotonicity"] = False
        with self.assertRaises(RT.RecoveryRuntimeError):
            D.validate_loss_stability_review(bad, self.parent, ROOT)
        bad = copy.deepcopy(review)
        bad["reviews"][0]["windows"][0]["end_update_exclusive"] = 319
        with self.assertRaises(RT.RecoveryRuntimeError):
            D.validate_loss_stability_review(bad, self.parent, ROOT)
        del self.parent.updates[self.key][315]
        with self.assertRaises(RT.RecoveryRuntimeError):
            D.validate_loss_stability_review(review, self.parent, ROOT)

    def test_task_certificate_event_survives_later_failure_and_extension(self):
        checks = [{"end_episode": 1560, "passed": True},
                  {"end_episode": 2040, "passed": True},
                  {"end_episode": 2520, "passed": False}]
        original = copy.deepcopy(checks)
        run = {"task_certified": True, "task_certificate_end_episode": 2040,
               "end_episode_cap": 2160}
        result = {"converged": True, "task_certificate_end_episode": 2040,
                  "continued_after_task_certificate": True,
                  "stopped_on_task_certificate": False}
        self.assertEqual(D.validate_task_termination(run, result, checks, 2160, True),
                         (True, 2040))
        self.assertEqual(checks, original)
        later = checks + [{"end_episode": 3000, "passed": True},
                          {"end_episode": 3480, "passed": True}]
        later_result = {"converged": True, "task_certificate_end_episode": 2040,
                        "continued_after_task_certificate": True,
                        "stopped_on_task_certificate": True}
        self.assertEqual(D.validate_task_termination(run, later_result, later, 3480, True),
                         (True, 2040))

    def test_new_task_certificate_must_stop_at_its_confirming_check(self):
        checks = [{"end_episode": 1560, "passed": True},
                  {"end_episode": 2040, "passed": True}]
        run = {"task_certified": False, "end_episode_cap": 2160}
        result = {"converged": True, "task_certificate_end_episode": 2040,
                  "continued_after_task_certificate": False,
                  "stopped_on_task_certificate": True}
        self.assertEqual(D.validate_task_termination(run, result, checks, 2040, True),
                         (True, 2040))

    def test_dynamic_stage_markers_follow_frozen_context(self):
        context = types.SimpleNamespace(stage_index=3, experiment_id="r3", namespace="seed-v1")
        self.assertEqual(RT.recovery_stage_origin(context)["stage"], 3)
        state = RT._checkpoint_state(context, "T1", 91001, 1560, 390)
        self.assertEqual(state["stage"], 3)
        with self.assertRaises(RT.RecoveryRuntimeError):
            RT.recovery_stage_origin(types.SimpleNamespace(stage_index=True))
class FreezeSeedAndG3Tests(unittest.TestCase):
    def test_v1_seed_pairing_and_all_prior_consumed_keys_are_checked(self):
        parent = parent_fixture(ROOT)
        runs = D.next_runs(parent.runs,parent.states)
        seeds = D.seed_plan(runs,parent.old_keys)
        self.assertEqual(len(seeds),720)
        # Nonuniform real ranges need not overlap. Align only this synthetic pair.
        aligned = copy.deepcopy(runs)
        aligned[5]["start_episode"]=1080
        aligned[5]["end_episode_cap"]=1200
        seeds = D.seed_plan(aligned,parent.old_keys)
        left = next(x for x in seeds if x["arm"]=="T1" and x["training_seed"]==91003)
        right = next(x for x in seeds if x["arm"]=="T2" and x["training_seed"]==91003)
        self.assertEqual(left["action_rng_seed"],right["action_rng_seed"])
        self.assertEqual(left["scene_seed"],right["scene_seed"])
        old = set(parent.old_keys)|{(left["arm"],left["training_seed"],
                                   left["scene_seed"],left["action_rng_seed"])}
        with self.assertRaisesRegex(RT.RecoveryRuntimeError,"already claimed"):
            D.seed_plan(aligned,old)

    def test_synthetic_builder_is_callable_and_never_generates_g3_or_world_approval(self):
        with temporary() as temp:
            root=Path(temp).resolve()
            parent=parent_fixture(root)
            with mock.patch.object(D,"canonical",return_value=root), \
                 mock.patch.object(D,"completed_parent",return_value=parent), \
                 mock.patch.object(RT,"build_source_manifest",
                                   return_value=parent.context.source_manifest):
                bundle=D.build_next_stage(root/"parent",root)
            self.assertEqual(bundle["world_cap"],720)
            context=RT.strict_json_bytes(bundle["files"][RT.CONTEXT_RELATIVE],"context")
            self.assertIsNone(context["g3_report"])
            self.assertIsNone(context["world_authorization"])
            self.assertEqual(context["schema"],D.CONTEXT_SCHEMA)
            self.assertEqual(context["candidate_manifest"],parent.context.payload["candidate_manifest"])
            self.assertEqual(set(bundle["files"]),set(RT.GENERATED_PROTOCOL_PATHS))

    def test_builder_refuses_native_drift_and_canonical_copy_before_write(self):
        with self.assertRaisesRegex(RT.RecoveryRuntimeError,"canonical"):
            D.build_next_stage(ROOT/"parent",ROOT.parent/"copy")
        with temporary() as temp:
            root=Path(temp).resolve()
            parent=parent_fixture(root)
            bad=copy.deepcopy(parent.context.source_manifest)
            bad["files"][0]["sha256"]="0"*64
            with mock.patch.object(D,"canonical",return_value=root), \
                 mock.patch.object(D,"completed_parent",return_value=parent), \
                 mock.patch.object(RT,"build_source_manifest",return_value=bad):
                with self.assertRaisesRegex(RT.RecoveryRuntimeError,"native files differ"):
                    D.build_next_stage(root/"parent",root)

    def test_source_pin_tamper_rejects_without_reading_a_checkpoint(self):
        with temporary() as temp:
            path=Path(temp)/"journal.json"
            path.write_bytes(b'{"update_index":359}')
            pin=RT.pin_file(path)
            path.write_bytes(b'{"update_index":358}')
            with self.assertRaisesRegex(RT.RecoveryRuntimeError,"pin drift"):
                D.Reads().pinned(pin,Path(temp))

    def test_frozen_writer_never_overwrites_and_base_context_is_last(self):
        with temporary() as temp:
            root=Path(temp).resolve()
            parent=parent_fixture(root)
            native=root/"native.py"
            native.write_text("synthetic native data",encoding="utf-8")
            parent.context.source_manifest["files"][0].update(RT.pin_file(native))
            with mock.patch.object(D,"canonical",return_value=root), \
                 mock.patch.object(D,"completed_parent",return_value=parent), \
                 mock.patch.object(RT,"build_source_manifest",
                                   return_value=parent.context.source_manifest):
                bundle=D.build_next_stage(root/"parent",root)
                D.write_bundle(bundle,root)
                with self.assertRaisesRegex(RT.RecoveryRuntimeError,"refuse overwrite"):
                    D.write_bundle(bundle,root)

    def test_new_checkpoint_payload_cursor_and_full_state_must_match_its_own_journal(self):
        parent=parent_fixture(ROOT)
        donor=parent.states[("T1",91001)]["donor"]
        context=types.SimpleNamespace(is_descendant=True)
        restored={"state":{"episode":1440,"next_update":360},
                  "payload_episode":1440,
                  "full_state":{"sha256":"3"*64,"actor":"4"*64}}
        RT.validate_donor_state(context,donor,restored,1440)
        for key,value in (("next_update",330),("episode",1320)):
            bad=copy.deepcopy(restored)
            bad["state"][key]=value
            with self.assertRaisesRegex(RT.RecoveryRuntimeError,"payload next_update/cursor"):
                RT.validate_donor_state(context,donor,bad,1440)
        bad=copy.deepcopy(restored)
        bad["full_state"]["sha256"]="0"*64
        with self.assertRaisesRegex(RT.RecoveryRuntimeError,"loaded state differs"):
            RT.validate_donor_state(context,donor,bad,1440)

    def test_two_real_check_pins_carry_over_noncheck_end_and_fail_resets(self):
        with temporary() as temp:
            root=Path(temp).resolve()
            certs=[]
            for ep in (1560,2040):
                path=root/("CHECK_%d.json"%ep)
                path.write_text(json.dumps({"schema":C.CONVERGENCE_SCHEMA,
                    "experiment_id":"parent","arm":"T1","seed":91001,
                    "end_episode":ep,"passed":True,"status":convergence.STATUS_PASS}),
                    encoding="utf-8")
                certs.append({"arm":"T1","training_seed":91001,
                              "experiment_id":"parent",**RT.pin_file(path)})
            context=types.SimpleNamespace(experiment_id="child",
                                          payload={"certificate_sources":certs})
            checks=RT.read_check_certificates(context,types.SimpleNamespace(root=root),"T1",91001)
            self.assertTrue(convergence.is_converged(checks))
            self.assertEqual(convergence.consecutive_passes(checks),2)
            self.assertEqual(convergence.consecutive_passes(checks+[
                {"end_episode":2520,"passed":False}]),0)



class Recovery02ReconstructedAdmissionTests(unittest.TestCase):
    def fixture(self, root):
        cursors=(1560,1560,1200,1200,1200,840)
        runs,results,checks,certs=[],[],[],[]
        for (arm,seed,_),end in zip(R.RUN_CURSORS,cursors):
            start=end-120
            runs.append({"arm":arm,"training_seed":seed,"start_episode":start,
                         "end_episode_cap":end,"new_world_cap":120,"active":True})
            results.append({"slot":"recovery:%s:%d"%(arm,seed),"status":"ok",
                            "episodes":120,"terminal_episode":end,"converged":False})
            checks.append({"run":"%s_%d"%(arm,seed),"episode":end,
                           "bytes":1220,"sha256":"d"*64,"status":"NOT_YET_CONVERGED"})
            certs.append({"run":"%s_%d"%(arm,seed),"episode":end,
                          "bytes":1220,"sha256":"e"*64,"status":"NOT_YET_CONVERGED"})
        base={"path":str(root/"protocol/RECOVERY_RUNTIME_CONTEXT.json"),"bytes":9490,"sha256":"a"*64}
        g3={"path":str(root/"execution/recovery_runtime/G3_REPORT.json"),"bytes":25431,"sha256":"b"*64}
        auth={"path":str(root/"execution/recovery_world_authorization_01/WORLD_AUTHORIZATION.json"),
              "bytes":1270,"sha256":"c"*64}
        bindings={"base_context":base,"g3_report":g3,"world_authorization":auth}
        detail={"status":"FINITE_RECOVERY_STAGE_COMPLETE","worlds":720,
                "unused_world_slots":0,"unclaimed_slots":0,"convergence_complete":False,
                "goal_complete":False,"slot_counts":{"claimed":720,"ok":720,
                "planned":720,"remaining":0,"results":720},
                "execution_bindings":bindings,"results":results}
        argv=["stage5b.py","recovery-train","--root",str(root),"--workers","3"]
        rid=D.RECOVERY02_RECEIPT_ID
        started={"phase":"recovery-train","stage":"started","receipt_id":rid,
                 "argv":argv,"detail":{"execution_bindings":bindings}}
        child={"exit_code":0,"argv":["D:/project_env/python.exe","-B"]+argv}; original=copy.deepcopy(child)
        receipt={"schema":"stage5b-phase-receipt/1","phase":"recovery-train",
                 "stage":"terminal","receipt_id":rid,"returncode":0,
                 "outer_exit_code":None,"outer_exit_status":"UNKNOWN",
                 "argv":argv,"detail":detail}
        line=json.dumps({"phase":"recovery-train","returncode":0,"receipt_id":rid})
        stdout=line
        summary={"claims":720,"results":720,"update_journals":180,"stage_complete":True,
                 "global_stop":False,"convergence_complete":False,
                 "reconstruction_basis":{"pre_run_snapshot":"reconstructed; no historical wrapper snapshot claimed",
                                         "tool_exit":"actual session 26862 terminal poll exit_code 0"},
                 "checkpoints":checks,"certificates":certs}
        raw={"exit_code":0,"output":"CHILD_EXIT=0\nDRIFT_COUNT=0\n"+line,
              "recovery_kind":"assembled_from_existing_evidence_not_byte_exact_original",
              "observed_session_poll":{"session_id":26862,"chunk_id":"130ac2","exit_code":0,
                 "output":"CHILD_EXIT_CODE=0\n","basis":"actual terminal write_stdin response visible in execution history"},
              "normalization":"saved child logs plus verified markers; not claimed as original outer response"}
        files=[{"path":"synthetic/%02d.json"%i,"bytes":10,"sha256":str(i%10)*64}
               for i in range(11)]
        before={"files":files,"reconstruction_basis":{"kind":"frozen_source_manifest_context_and_started_receipt","not_original_wrapper_snapshot":True}}; after={"files":copy.deepcopy(files),
             "source_context_auth_drift":[],
             "reconstruction_basis":"current bytes match the reconstructed frozen 11-file baseline"}
        context=types.SimpleNamespace(context_pin=base,runs=runs)
        return context,started,child,original,receipt,stdout,summary,raw,before,after

    def validate(self, root, fixture):
        return D.validate_recovery02_reconstructed_terminal(
            root,root/"execution/recovery_train_run_01",*fixture[0:4],
            fixture[4],fixture[5],fixture[6],fixture[7],fixture[8],fixture[9])

    def test_reconstruction_marks_chat_poll_and_wrapper_unknowns_explicitly(self):
        with temporary() as temp:
            root=Path(temp).resolve(); fixture=self.fixture(root)
            with mock.patch.object(D,"RECOVERY02_ROOT",root):
                detail,bindings,evidence=self.validate(root,fixture)
            self.assertEqual(detail["worlds"],720)
            self.assertEqual(bindings["g3_report"]["sha256"],"b"*64)
            self.assertEqual(evidence["outer_tool_exit"]["exit_code"],0)
            self.assertEqual(evidence["outer_tool_exit"]["raw_tool_return"],"UNKNOWN_NOT_PERSISTED")
            self.assertEqual(evidence["source_context_auth_drift"],
                             "UNKNOWN_NOT_COMPUTABLE_FROM_PERSISTED_EVIDENCE")
            self.assertEqual(evidence["source_pre_start_snapshot"],"UNKNOWN_NOT_CAPTURED")
            self.assertEqual(evidence["wrapper_summary_raw_pins"],
                             "RECONSTRUCTED_FROM_CLOSED_EVIDENCE_NOT_ORIGINAL_WRAPPER")

    def test_reconstruction_rejects_exit_count_binding_and_provenance_tamper(self):
        with temporary() as temp:
            root=Path(temp).resolve(); base=self.fixture(root)
            with mock.patch.object(D,"RECOVERY02_ROOT",root):
                mutations=(
                    lambda f:f[4].update(outer_exit_code=0),
                    lambda f:f[2].update(exit_code=True),
                    lambda f:f[4]["detail"].update(worlds=719),
                    lambda f:f[4]["detail"]["slot_counts"].update(results=719),
                    lambda f:f[7].update(recovery_kind="original"),
                    lambda f:f[8]["reconstruction_basis"].update(not_original_wrapper_snapshot=False),
                    lambda f:f[9].update(files=[]),
                )
                for mutate in mutations:
                    bad=list(copy.deepcopy(base)); mutate(bad)
                    with self.assertRaises(RT.RecoveryRuntimeError):self.validate(root,bad)

    def test_reconstruction_is_fixed_to_registered_root_and_six_runs(self):
        with temporary() as temp:
            root=Path(temp).resolve(); fixture=self.fixture(root)
            with mock.patch.object(D,"RECOVERY02_ROOT",root):
                with self.assertRaisesRegex(RT.RecoveryRuntimeError,"restricted"):
                    D.validate_recovery02_reconstructed_terminal(
                        root/"copy",root/"copy/execution/recovery_train_run_01",
                        *fixture[0:4],fixture[4],fixture[5],fixture[6],fixture[7],fixture[8],fixture[9])
                bad=list(copy.deepcopy(fixture)); bad[6]["checkpoints"].pop()
                with self.assertRaises(RT.RecoveryRuntimeError):self.validate(root,bad)
class InheritedCertificateRuntimeTests(unittest.TestCase):
    def test_inherited_two_passes_and_first_new_pass_run_full_stage_without_overwrite(self):
        from stage5b_comm.paths import Layout

        arm, seed, start, end = "T1", 91001, 2400, 2520
        with temporary() as temp:
            root = Path(temp).resolve()
            layout = Layout(root).ensure()
            checks = [
                {"end_episode": 1560, "passed": True},
                {"end_episode": 2040, "passed": True},
            ]
            original_checks = copy.deepcopy(checks)
            run = {"arm": arm, "training_seed": seed, "start_episode": start,
                   "end_episode_cap": end, "new_world_cap": end-start, "active": True,
                   "task_certified": True, "task_certificate_end_episode": 2040}
            seed_plan = []
            for episode_index in range(start, end):
                seed_plan.append({"arm": arm, "training_seed": seed,
                                  "episode_index": episode_index,
                                  "scene_seed": int(C.scene_for(seed, episode_index)),
                                  "action_rng_seed": episode_index + 100000})
            donor = {"path": str(root/"donor.pth"), "bytes": 1, "sha256": "1"*64,
                     "identity_sha256": "2"*64, "expected_full_state_sha256": "3"*64,
                     "expected_actor_sha256": "4"*64}
            context = types.SimpleNamespace(
                runs=[run], seed_plan=seed_plan, old_runtime_keys=set(),
                payload={"identity": {"sha256": "5"*64}},
                experiment_id="synthetic-descendant", namespace="seed-v1", stage_index=3,
                is_descendant=True, loss_review_enabled=True,
                certificate_sources=[],
            )
            class FakeLedger:
                def __init__(self, _context, _layout):
                    pass
                def existing_runtime_keys(self, _arm, _seed):
                    return set()
                def claim(self, _slot, *, runtime_key_value):
                    return "claimed"
                def finish(self, _slot, _row_path):
                    return None

            def world_row(_layout, _arm, _seed, episode_index, scene, _controller,
                          *, action_rng_seed):
                return {"generated_goals": C.NUM_GOALS, "steps": C.STEPS,
                        "scene": scene, "action_rng_seed": action_rng_seed,
                        "on_time": 20, "spent_bytes": 10000, "emissions": 1,
                        "discovered": 2, "seconds": 0.01}

            def add_first_new_pass(*_args):
                checks.append({"end_episode": end, "passed": True})
                return {"evaluated": True, "status": convergence.STATUS_PASS,
                        "passed": True, "path": "synthetic-check", "streak": 3}

            state = {"sha256": "6"*64, "actor": "7"*64}
            update_value = {"samples": 256, "minibatches": 4,
                            "optimizer_steps": 4, "policy_surrogate": 0.0,
                            "value_loss_task": 1.0, "value_loss_cost": 1.0}

            def save_checkpoint(path, *_args, **_kwargs):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic checkpoint marker; never loaded")

            with mock.patch.object(RT, "load_context", return_value=context), \
                 mock.patch.object(RT, "load_execution_bindings", return_value={"g3": "pin", "auth": "pin"}), \
                 mock.patch.object(RT.episode, "native_import_preflight"), \
                 mock.patch.object(RT.episode, "ensure_import_paths"), \
                 mock.patch.object(RT, "_donor_pairs", return_value=[((arm, seed, start), donor)]), \
                 mock.patch.object(RT, "verify_pin", side_effect=lambda pin, *_a, **_k: dict(pin)), \
                 mock.patch.object(RT.trainer, "load_base_checkpoint",
                                   return_value={"controller": object(), "optimizer": object()}), \
                 mock.patch.object(RT, "validate_donor_state"), \
                 mock.patch.object(RT, "selected_history", return_value={}), \
                 mock.patch.object(RT, "selected_updates", return_value={}), \
                 mock.patch.object(RT, "RecoveryLedger", FakeLedger), \
                 mock.patch.object(RT.trainer, "full_state_fingerprint", return_value=state), \
                 mock.patch.object(RT.episode, "run_training_world", side_effect=world_row), \
                 mock.patch.object(RT.trainer, "ppo_update", return_value=update_value), \
                 mock.patch.object(RT, "write_update_journal"), \
                 mock.patch.object(RT.episode, "compact_row", side_effect=lambda row, **_kw: dict(row)), \
                 mock.patch.object(RT.trainer, "save_checkpoint", side_effect=save_checkpoint), \
                 mock.patch.object(RT, "evaluate_check", side_effect=add_first_new_pass), \
                 mock.patch.object(RT, "read_check_certificates", side_effect=lambda *_a: list(checks)):
                result = RT.run_group((str(root), "base-context", "g3-report",
                                       "world-auth", arm, seed))

            self.assertEqual(result["terminal_episode"], end)
            self.assertEqual(result["episodes"], end-start)
            self.assertFalse(result["stopped_on_task_certificate"])
            self.assertTrue(result["continued_after_task_certificate"])
            self.assertEqual(result["task_certificate_end_episode"], 2040)
            self.assertEqual(checks[:2], original_checks)
            telemetry = json.loads(Path(result["telemetry"]).read_text(encoding="utf-8"))
            self.assertEqual(telemetry["convergence"]["task_certificate_end_episode"], 2040)
            self.assertFalse(telemetry["convergence"]["stopped_on_task_certificate"])
