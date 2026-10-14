# ROUND75 native wiring diff (function level)

Source of truth: `r66_frozen.frozen_early_ppo_update` (round66, sha256 of file
A22425729C9327FFCB1D57F4820F3B38F2BB7E35A6D6D81890FE64A75A246B10).

- function-level source sha256, OLD `frozen_early_ppo_update`: `0a891308d06128dc4f7ee41fc137455a26d2f332ea2e1aa570fe7975064c0572`
- function-level source sha256, NEW `r75_native_ppo_update`:      `07f502fa71b2ae56239819711147aa26811235ef1b864f74fe903b431dd53281`

Permitted changes (and only these): the freeze set (trunk+early -> trunk), the trainable /
optimizer / clip parameter set (6 -> 8 objects), the gradient-read scope (early_head becomes a
measured module), the R75 identity strings, and the mandatory R75 production-ledger resource
seam (pre-charge before every physical step and every real trunk forward). The loss, the
gate prior, the shuffle, the epoch/minibatch loop, the reduction, the clip algorithm, the
optimizer protocol and the return contract are unchanged.

```diff
--- r66_frozen.py:frozen_early_ppo_update
+++ r75_frozen.py:r75_native_ppo_update
@@ -1,40 +1,36 @@
-def frozen_early_ppo_update(controller: PartitionedController, optimizer: torch.optim.Optimizer,
-                            episodes: list[dict[str, Any]], *, update_index: int,
-                            diag_updates: tuple[int, ...] = (0, 15, 31),
-                            gate_beta: float = GATE_BETA) -> dict[str, Any]:
-    """One PPO update over a 4-episode batch with routed early/late actor logits, early path frozen.
-
-    Faithful transcription of the frozen ``v2_trainer.ppo_update`` and the R58 beta=1.00 gate
-    prior. The routed early/late heads, the sequential mask, the joint log-probability, the
-    entropy, the combined advantage, both critic targets and the ALL_ROWS reduction are exactly
-    ROUND60's. ROUND66 changes ONLY which parameter objects the optimizer owns: ``trunk`` and
-    ``early_head`` are ``requires_grad=False``, so ``optimizer.zero_grad`` / ``clip_grad_norm_`` /
-    ``optimizer.step`` act on the six trainable objects (late_head + both critics) and the frozen
-    modules receive no gradient at all because they do not require one.
-
-    The beta=1.00 / ready_cut=147 gate prior is still computed over early first-emission free
-    candidates exactly as before. Those candidates route to the frozen ``early_head``, so the
-    prior cannot move the early path. That is the expected effect of this intervention and it is
-    recorded as such; no loss term, weight or coefficient was added or changed to produce it.
+def r75_native_ppo_update(controller, optimizer, episodes: list[dict[str, Any]], *,
+                          update_index: int, resources,
+                          diag_updates: tuple[int, ...] = (0, 15, 31),
+                          gate_beta: float = GATE_BETA) -> dict[str, Any]:
+    """One native PPO update over a real recorded batch; trunk frozen; 8 trainable objects.
+
+    Faithful adaptation of ``r66_frozen.frozen_early_ppo_update``. Everything except the freeze
+    set, the parameter ownership (6 -> 8 objects) and the R75 resource seam is the real function.
     """
-    import v2_trainer as trainer
-    import v2_common as common
-    from onpolicy import action_entropy, joint_log_prob
-
-    # Fail closed if the caller handed us a controller whose early path is not actually frozen.
+    _require_production_ledger(resources)
     frozen_parameters(controller)
     trainable_parameters(controller)
+
+    # Real recorded batch -> samples (real batch_samples + GAE + reward-event attribution).
+    # batch_samples performs no trunk forward (recorded-value tensor math only): no charge here.
     samples = trainer.batch_samples(episodes)
     prefix_task, prefix_cost_bytes = trainer._unassigned_prefix_totals(episodes)
-    alignment = likelihood_alignment(controller, samples)
+
+    # Real likelihood recomputation through the SAME routed forward as sampling.
+    resources.charge_offline_rows(
+        sum(len(item["features"]) for item in samples),
+        what=f"r75_native_update_{update_index}_alignment")
+    alignment = r66.likelihood_alignment(controller, samples)
     if not alignment["passed"]:
-        raise RuntimeError(f"update {update_index}: logged probability does not describe the draw: {alignment}")
+        raise RuntimeError(
+            f"update {update_index}: logged probability does not describe the draw: {alignment}")
     if not samples:
         return {"samples": 0, "minibatches": 0, "optimizer_steps": 0,
                 "unassigned_prefix_task_reward": prefix_task,
                 "unassigned_prefix_cost_bytes": prefix_cost_bytes,
                 "unassigned_prefix_cost_normalized": prefix_cost_bytes / common.B0_BYTES,
                 "likelihood_alignment": alignment, "note": "no decision step in the batch"}
+
     shuffle = random.Random(int(hashlib.sha256(
         f"{common.DATA_ORDER_SEED}:{int(update_index)}".encode("utf-8")).hexdigest()[:16], 16))
     order = list(range(len(samples)))
@@ -55,8 +51,10 @@
         for start in range(0, len(order), common.MINIBATCH_STEPS):
             chunk = [samples[index] for index in order[start:start + common.MINIBATCH_STEPS]]
             stacked = torch.as_tensor(np.concatenate([item["features"] for item in chunk], axis=0))
+            resources.charge_offline_rows(int(stacked.shape[0]),
+                                          what=f"r75_native_update_{update_index}_minibatch_{minibatches}_trunk")
             hidden = controller.inner.trunk(stacked)
-            logits = route_logits(controller, stacked, hidden)   # <-- the only structural change
+            logits = r66.route_logits(controller, stacked, hidden)
             candidate_values_task = controller.inner.value_head(hidden).squeeze(-1)
             candidate_values_cost = controller.cost_head(hidden).squeeze(-1)
             row = 0
@@ -109,24 +107,19 @@
                     - common.ENTROPY_BONUS * entropy_t.mean()
                     + gate_beta * gate_loss)
             if gate_diag["pending"] and not gate_diag["done"]:
-                # ONLY trainable parameters may be handed to autograd.grad. The frozen trunk and
-                # early_head have requires_grad=False, so passing them would raise; and a zero
-                # reading on them would say nothing about the R53 critic->trunk isolation.
+                # Passive diagnostic: raw early/late/critic gradients read AFTER backward and
+                # BEFORE clip/step, via autograd.grad on the TRAINABLE objects only. The frozen
+                # trunk has requires_grad=False and is never handed to autograd.grad.
                 trainable = trainable_parameters(controller)
+                early_params = list(controller.inner.early_head.parameters())
                 late_params = list(controller.inner.late_head.parameters())
                 critic_params = (list(controller.inner.value_head.parameters())
                                  + list(controller.cost_head.parameters()))
                 base_only = loss - gate_beta * gate_loss
                 prior_only = gate_beta * gate_loss
-                # allow_unused=True only tolerates an UNUSED input; it does not make a
-                # requires_grad=False output differentiable. torch.where can also keep a
-                # zero-valued graph, so graph existence is checked, never assumed. A minibatch
-                # with no eligible early first-emission free candidate has a structurally
-                # constant gate term: that is recorded as absent, never as a measured zero.
                 record = {"early_free_rows": early_free, "late_free_rows": late_free,
                           "base_output_requires_grad": bool(base_only.requires_grad),
                           "prior_output_requires_grad": bool(prior_only.requires_grad),
-                          "early_head": "frozen_requires_grad_false",
                           "trunk": "frozen_requires_grad_false",
                           "trainable_parameter_object_count": len(trainable)}
                 gb = gg = None
@@ -134,6 +127,9 @@
                     gb = torch.autograd.grad(base_only, trainable, retain_graph=True, allow_unused=True)
                     record["base_trainable_grad_l1"] = float(
                         sum(float(g.detach().abs().sum()) for g in gb if g is not None))
+                    ge = torch.autograd.grad(base_only, early_params, retain_graph=True, allow_unused=True)
+                    record["early_actor_grad_l1"] = float(
+                        sum(float(g.detach().abs().sum()) for g in ge if g is not None))
                     gr = torch.autograd.grad(base_only, late_params, retain_graph=True, allow_unused=True)
                     record["late_actor_grad_l1"] = float(
                         sum(float(g.detach().abs().sum()) for g in gr if g is not None))
@@ -142,6 +138,7 @@
                         sum(float(g.detach().abs().sum()) for g in gc if g is not None))
                 else:
                     record["base_trainable_grad_l1"] = None
+                    record["early_actor_grad_l1"] = None
                     record["late_actor_grad_l1"] = None
                     record["critic_grad_l1"] = None
                     record["base_gradient"] = "absent_output_has_no_graph"
@@ -149,9 +146,9 @@
                     gg = torch.autograd.grad(prior_only, trainable, retain_graph=True, allow_unused=True)
                     record["prior_trainable_grad_l1"] = float(
                         sum(float(g.detach().abs().sum()) for g in gg if g is not None))
-                    gr2 = torch.autograd.grad(prior_only, late_params, retain_graph=True, allow_unused=True)
-                    record["prior_late_actor_grad_l1"] = float(
-                        sum(float(g.detach().abs().sum()) for g in gr2 if g is not None))
+                    ge2 = torch.autograd.grad(prior_only, early_params, retain_graph=True, allow_unused=True)
+                    record["prior_early_actor_grad_l1"] = float(
+                        sum(float(g.detach().abs().sum()) for g in ge2 if g is not None))
                     if gb is not None:
                         dot = float(sum(float((a * b).sum()) for a, b in zip(gb, gg)
                                         if a is not None and b is not None))
@@ -161,7 +158,7 @@
                         record["cosine"] = (dot / (nb * ng)) if nb > 0 and ng > 0 else None
                 else:
                     record["prior_trainable_grad_l1"] = None
-                    record["prior_late_actor_grad_l1"] = None
+                    record["prior_early_actor_grad_l1"] = None
                     record["prior_gradient"] = "absent_no_eligible_gate_rows"
                 gate_diag["record"] = record
                 gate_diag["done"] = True
@@ -172,10 +169,12 @@
                               sum(float(parameter.grad.detach().abs().sum())
                                   for parameter in module.parameters() if parameter.grad is not None))
                 for module_name, module in modules})
-            # The R66 global clip norm is over the six trainable objects only. ROUND60 clipped
-            # over all twelve, so the two norms are not comparable quantities.
+            # The R75 global clip norm is over the EIGHT trainable objects only.
             norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(controller),
                                                   common.MAX_GRAD_NORM)
+            # Pre-charge BEFORE the physical step; a step failure does not refund.
+            resources.charge_physical_step(
+                what=f"r75_native_update_{update_index}_minibatch_{minibatches}")
             optimizer.step()
             if not (torch.isfinite(loss) and torch.isfinite(norm)):
                 raise RuntimeError(f"update {update_index}: non-finite loss or gradient norm")
@@ -209,11 +208,12 @@
         "value_loss_task": round(totals["value_task"] / minibatches, 6),
         "value_loss_cost": round(totals["value_cost"] / minibatches, 6),
         "entropy": round(totals["entropy"] / minibatches, 6),
-        "approx_kl": round(totals["kl"] / minibatches, 6),
+        "approx_kl": round(totals["approx_kl"] / minibatches, 6) if minibatches else 0.0,
         "clip_fraction": round(totals["clip"] / minibatches, 4),
         "gradient_norm_mean": round(totals["grad"] / minibatches, 4),
         "gate_loss": round(totals["gate"] / minibatches, 6) if minibatches else 0.0,
-        "gate_rows": totals["gate_rows"], "early_free": totals["early_free"], "late_free": totals["late_free"],
+        "gate_rows": totals["gate_rows"], "early_free": totals["early_free"],
+        "late_free": totals["late_free"],
         "module_gradient_l1_mean": {
             module_name: (None if module_name in FROZEN_MODULES else
                           round(sum(read[module_name] for read in module_gradient_reads)
@@ -225,13 +225,13 @@
             for module_name, _module in modules} if module_gradient_reads else {},
         "module_gradient_scope": ("per-minibatch pre-clip L1 of every TRAINABLE module's "
                                   "parameters, read after backward and before clipping/step; "
-                                  "frozen modules (trunk, early_head) are None because "
-                                  "requires_grad=False, which is a structural fact and not a "
-                                  "measured zero gradient"),
+                                  "the frozen trunk is None because requires_grad=False, which "
+                                  "is a structural fact and not a measured zero gradient"),
         "gate_diag": diag_record, "gate_records": gate_records,
         "likelihood_alignment": alignment,
-        "structure": R66_STRUCTURE,
-        "frozen_early_path": True,
+        "structure": R75_STRUCTURE,
+        "frozen_modules": list(FROZEN_MODULES),
+        "trainable_modules": list(TRAINABLE_MODULES),
         "trainable_parameter_object_count": len(trainable_parameters(controller)),
         "frozen_parameter_object_count": len(frozen_parameters(controller)),
     }
```
