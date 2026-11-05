{
  "section": "14",
  "title": "P4 short learnability test at a binding budget (N=12, fixed 58,600 B)",
  "date": "2026-11-05",
  "verdict": "negative",
  "headline": "The learned policy did not beat the strong rule; it reproduced 'spend the whole budget at step 1' and lost 10 tasks.",

  "why_this_operating_point": {
    "N6_rejected": "At N=6 the 58,600 B budget covers 23 of 24 targets, so normal_immediate is already at the ceiling (P2 measured 23 emissions in every world). A policy cannot learn target selection where selection is unnecessary.",
    "N12_chosen": "At N=12 one send costs 12 subscribers' payload, so the same budget buys 11-12 sends against 24 targets. P2 measured normal_immediate at 11-12 emissions with 8/8 worlds at the cap. This is the pre-existing, same-caliber, same-revision strong-rule baseline.",
    "no_new_baseline_replay": "P2's N=12 normal_immediate rows were read from disk and asserted against its published totals (90/192, 90 sends, 450,192 B). Zero new baseline worlds were spent."
  },

  "seam_defect_found_and_fixed_before_training": {
    "what": "The P3 feature transform was installed by shadowing controller.forward. Three sites read the 19-dim row and only one goes through forward.",
    "sites": [
      "budget_masked._choose_masked -> controller.forward(matrix)  [covered]",
      "sb_collector.decide -> controller.cost_values(matrix)       [bypassed]",
      "trainer.build_samples -> controller.inner.trunk(tape['features'])  [bypassed]"
    ],
    "consequence_if_unfixed": "The stored behaviour log-prob came from the rewritten row while the PPO update recomputed it from the raw row. Measured importance-ratio error at update 0: max |new_logp - old_logp| = 1.589 (must be 0). With clip 0.2 the first update would have been silently wrong, and any 'cannot learn' verdict would have been an artefact of the seam.",
    "fix": "The transform now wraps controller.inner.trunk, the single node all three paths pass through. For the inference path the two placements are provably identical (forward -> trunk with no matmul between), so P3's archived rows still describe the same computation.",
    "evidence": "diagnostics_20261104/p4_unitcheck.py, 29 checks, all pass, exit 0.",
    "restore_bug_found_by_smoke": "The first 4-episode smoke died with 'Controller object has no attribute trunk': train_pilot.Controller is a plain class whose trunk lives in the instance __dict__, so popping the shadow deleted the real trunk. Restore now writes the real object back for plain classes and pops only for nn.Modules; unit check section G covers this shape."
  },

  "switches": {
    "CREDIT_MODE": "causal_task_id",
    "ADVANTAGE_MODE": "causal_step",
    "CAUSAL_GAMMA": 0.0,
    "CREDIT_RULE": "first_accepted_send",
    "SCALE_INVARIANT": "norm_only",
    "comm_subscribers": "all",
    "rescue_per_species": 4,
    "registered_expectation_override": {"rescue_robots": 12},
    "budget_bytes": 58600
  },

  "training": {
    "start_checkpoint": "recovery_09b/revision_01/execution/checkpoints/T2_s91002/milestone_5000.pth",
    "start_checkpoint_sha256": "645126abfbb91be7734eac618e869746eed380b633a89b0d6c1556fb7d8c77c6",
    "identity_sha256": "bf02580f38527c34c354aee14d7aad00c55151a800820d6e442a6c6a33973086",
    "episodes": 300,
    "start_episode": 5000,
    "end_episode": 5300,
    "updates": 75,
    "wall_seconds": 6226.4,
    "start_parameter_sha256": "1ae9893c0f4ce1daa6ef54340a44a430bea261110eb00510a2859994beeea029",
    "end_parameter_sha256": "95f39c3aa0990c4dc9904c910fed7b6b48f9061b4c5e09d8e693bedc8a18acf8",
    "checkpoints_saved": ["milestone_5100.pth", "milestone_5200.pth", "milestone_5300.pth"],
    "per_50_episode_blocks": [
      {"episodes": "0-49", "on_time_mean_per_world": 10.80, "sends_mean": 11.24, "paid_bytes_mean": 56115},
      {"episodes": "50-99", "on_time_mean_per_world": 10.74, "sends_mean": 11.22, "paid_bytes_mean": 56233},
      {"episodes": "100-149", "on_time_mean_per_world": 9.38, "sends_mean": 11.22, "paid_bytes_mean": 56183},
      {"episodes": "150-199", "on_time_mean_per_world": 8.82, "sends_mean": 11.26, "paid_bytes_mean": 56497},
      {"episodes": "200-249", "on_time_mean_per_world": 8.44, "sends_mean": 11.26, "paid_bytes_mean": 56252},
      {"episodes": "250-299", "on_time_mean_per_world": 8.34, "sends_mean": 11.26, "paid_bytes_mean": 56167}
    ],
    "trend": "on-time per world fell monotonically 10.80 -> 8.34 while sends stayed flat at ~11.25 and spend stayed pinned at the budget. Training did not move toward selection.",
    "causal_credit": {"moved": 2826, "unmatched": 0},
    "transform_live": {"scale_mode": "norm_only in 300/300 episodes", "subscribers_live": 12}
  },

  "evaluation": {
    "protocol": "greedy argmax, 8 DEV_REUSED scenes, eval_action_seed per scene, zero training",
    "checkpoint": "milestone_5300.pth",
    "learned": {
      "on_time": 80, "denominator": 192,
      "spent_bytes": 453924, "emissions": 91, "completed": 80,
      "completed_per_send": 0.8791,
      "bytes_per_completed_task": 5674.1,
      "masked_by_credit_total": 100844,
      "first_masked_credit_tick_per_scene": {"2301901": 4, "2301902": 2, "2301903": 2, "2301904": 3,
                                             "2301905": 1, "2301906": 3, "2301907": 5, "2301908": 1},
      "every_world_generated_24_goals": true,
      "transform_calls_per_world": 1000,
      "subscribers_live": 12
    },
    "baseline": {
      "arm": "normal_immediate", "n": 12,
      "on_time": 90, "denominator": 192,
      "spent_bytes": 450192, "emissions": 90,
      "source": "execution/scale_probe/P2_12_fixed_p2/RESULTS.json",
      "source_sha256_prefix": "c83d87726bb7e1fd",
      "totals_verified_against_P2": true
    },
    "paired": {
      "total_delta_on_time": -10,
      "mean_delta_per_scene": -1.25,
      "sd_of_deltas": 1.8323,
      "ci95_t7": [-2.782, 0.282],
      "ci95_as_tasks_over_8_scenes": [-22.26, 2.26],
      "sign_test": {"n": 5, "positive": 1, "negative": 4, "p_two_sided": 0.375},
      "total_delta_bytes": 3732,
      "per_scene": [
        {"scene": 2301901, "learned": 10, "baseline": 11, "delta_on_time": -1, "delta_bytes": 684},
        {"scene": 2301902, "learned": 11, "baseline": 11, "delta_on_time": 0, "delta_bytes": -504},
        {"scene": 2301903, "learned": 11, "baseline": 11, "delta_on_time": 0, "delta_bytes": 0},
        {"scene": 2301904, "learned": 9, "baseline": 12, "delta_on_time": -3, "delta_bytes": -864},
        {"scene": 2301905, "learned": 11, "baseline": 11, "delta_on_time": 0, "delta_bytes": -420},
        {"scene": 2301906, "learned": 8, "baseline": 12, "delta_on_time": -4, "delta_bytes": -120},
        {"scene": 2301907, "learned": 8, "baseline": 11, "delta_on_time": -3, "delta_bytes": 768},
        {"scene": "2301908", "learned": 12, "baseline": 11, "delta_on_time": 1, "delta_bytes": 4188}
      ]
    },
    "mechanism": {
      "observation": "The learned arm spent its budget at the start of the episode: 82.4% of its 91 sends fall on steps 1-2, median step 1, last send ever step 5.",
      "interpretation": "This is the same behaviour as normal_immediate: fire at every candidate that is pending at t=0, then run out. The policy did not learn to defer or skip.",
      "why_that_costs_tasks": "Spending every byte in the first ticks means the 12 that arrive later in the episode are unaffordable, so the tasks whose deadlines become tight later are the ones that expire. The rule arm sends the same 11-12 but on a slightly different set, which is why the difference is -10 tasks rather than 0."
    }
  },

  "can_say": [
    "Under these exact switches, three seeds not tested (single seed 91002), the learned arm did not beat normal_immediate at N=12/58,600 B: 80/192 vs 90/192, paired mean -1.25 tasks/scene.",
    "The paired interval includes zero (-2.78, +0.28 per scene; -22.3, +2.3 tasks over 8 scenes) and the sign test is not significant (p=0.375, 5 non-zero pairs). The correct statement is 'no evidence of improvement', not 'proven worse'.",
    "The policy's failure mode is visible and specific: it consumes the whole budget in the first 1-2 steps, matching the rule arm's shape rather than learning to select.",
    "The feature transform, causal credit and budget binding were all live and audited in every world (1000 forward calls per eval world, 12 subscribers, 24 goals, 2826 credits moved, 0 unmatched)."
  ],

  "cannot_say": [
    "That selection is unlearnable in general: one seed, 300 episodes, one checkpoint, 8 development scenes. This is a short screen, not a population estimate.",
    "That normalisation helps or hurts selectivity: the P2/P3 measurement already showed that at N=6 norm_only changes spend (34.4% -> 65.5% of budget) and the +22 on-time there was a spend artifact, not a strategy improvement. Here norm_only is on for both the learned arm and no rule arm, so no isolation of the transform's effect was attempted.",
    "That N=12's on-time rate is a clean indicator of selection: rescue_per_species=4 means the roster is 12 robots, so labour moved with fan-out (P2's inherited caveat).",
    "That the result transfers to N=6: different N is a different config_hash and behaviour_sha256, and the whole point of the operating point is that N=6 does not bind.",
    "Anything about cost savings: the learned arm spent 3,732 B MORE than the rule, not less."
  ],

  "evidence_table": [
    {"artifact": "diagnostics_20261104/p4_unitcheck.py", "sha256_note": "29 checks, exit 0", "command": "D:/project_env/python.exe -B diagnostics_20261104/p4_unitcheck.py"},
    {"artifact": "diagnostics_20261104/p4_tight_budget_train.py", "sha256_note": "driver", "command": "D:/project_env/python.exe -B diagnostics_20261104/p4_tight_budget_train.py --episodes 300 --tag p4_tight_n12 --save-every 100"},
    {"artifact": "diagnostics_20261104/p4_eval_and_score.py", "sha256_note": "driver", "command": "D:/project_env/python.exe -B diagnostics_20261104/p4_eval_and_score.py --tag p4_tight_n12 --episode 5300 --workers 3"},
    {"artifact": "recovery_12/revision_01/execution/p4_tight_n12/T2/TRAIN_SUMMARY.json", "sha256_note": "training summary", "command": "read"},
    {"artifact": "recovery_12/revision_01/execution/p4_tight_n12/eval_5300/EVAL.json", "sha256": "9794c764780f24f51979c063f23ac87b8c3a993744e0130e67029094c7515ac7", "command": "produced by the eval driver"},
    {"artifact": "execution/scale_probe/P2_12_fixed_p2/RESULTS.json", "sha256_note": "baseline source, totals verified", "command": "read-only"}
  ],

  "quota": {
    "authorised": "300-500 training episodes + 8 eval worlds (this round)",
    "spent": {"training_episodes": 300, "training_updates": 75, "eval_worlds": 8, "smoke_worlds": 4},
    "new_worlds_total": 312,
    "note": "The 4-episode smoke (tag p4_smoke4) died on the plain-class restore bug and consumed 0 worlds; p4_smoke4b completed 4 worlds and is kept as evidence."
  },

  "stop_loss_record": [
    {"rule": "same seam/import/gate error 3+ times without a fix",
     "triggered": true,
     "detail": "The p4_smoke4 run failed on 'Controller object has no attribute trunk'. Fixed by the plain-class branch in restore(). Root-caused within one attempt, not three."},
    {"rule": "pin file sha256 mismatch", "triggered": false,
     "detail": "No pin file was modified. scale_features.py and world.py are revision-local, not in the 98-file manifest."},
    {"rule": "off@N=6 not byte-reproducing frozen rows", "triggered": false,
     "detail": "Not applicable to P4; the unit check's section F proves the inference path is unchanged by the seam move."},
    {"rule": "replay count reaching 120", "triggered": false,
     "detail": "312 new worlds, but this round's authorisation is explicitly 300-500 training episodes plus 8 eval worlds, not the P0-P3 replay budget."}
  ],

  "next_step_if_continued": "The failure mode is specific and testable: the policy spends everything at t=0. A minimal next experiment is a scene pair where some targets' deadlines are tight enough that broadcasting at t=0 cannot finish them, so skipping them is strictly better. That would make deferral profitable and give the causal credit something to reward. Not run here; it needs a new authorisation."
}
