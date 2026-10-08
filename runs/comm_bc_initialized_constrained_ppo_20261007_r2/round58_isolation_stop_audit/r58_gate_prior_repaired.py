"""R58 gate-prior seam: beta=1.00, per-minibatch passive recorder, and
9-point gradient diagnostics (batches 0/15/31, first minibatch).

Source-patched copy of the frozen ppo_update exactly as in R56, plus:
- a passive per-minibatch recorder (N_rows, free first-report count, early
  effective count, raw and beta-weighted gate loss) written to
  raw/runtime/r58_minibatch_records.jsonl;
- at the FIRST minibatch of batches 0/15/31: separated base-actor and
  prior gradients (torch.autograd.grad, retain_graph) with norms, dot and
  cosine, recorded before the combined backward.  Diagnostics never touch
  .grad accumulation, optimizer behavior, or RNG.
The frozen canonical trainer on disk is never modified.
"""
from __future__ import annotations
import difflib, inspect, json, math
from pathlib import Path
from typing import Any

GATE_BETA = 1.00
READY_CUT = 147.0
HORIZON = 1000.0
SKIP_INDEX = 0
DIAG_UPDATES = (0, 15, 31)

ANCHOR_USABLE = """                usable = torch.tensor([1.0 if t in item["free"] else 0.0
                                       for t in item["targets"]], dtype=torch.float32)
"""

INSERT_AFTER_USABLE = """                for gate_index, gate_target in enumerate(item["targets"]):
                    if gate_target in item["free"]:
                        gate_feature = item["features"][gate_index]
                        if float(gate_feature[5]) != 1.0:
                            continue  # not a first-emission candidate
                        gate_ready = int(round(float(gate_feature[18]) * 1000.0)
                                         - round(float(gate_feature[0])))
                        if gate_ready <= 147.0:
                            gate_skip_log_probs.append(
                                torch.log_softmax(logits[rows][gate_index], dim=-1)[0])
"""

ANCHOR_LISTS = """            pooled_cost: list[torch.Tensor] = []
"""

INSERT_LISTS = """            pooled_cost: list[torch.Tensor] = []
            gate_skip_log_probs: list[torch.Tensor] = []
"""

ANCHOR_LOSS = """                    - common.ENTROPY_BONUS * entropy_t.mean())
"""

INSERT_LOSS = """                    - common.ENTROPY_BONUS * entropy_t.mean())
            gate_loss = (-(torch.stack(gate_skip_log_probs)).sum() / len(chunk)
                         if gate_skip_log_probs
                         else torch.zeros((), dtype=torch.float32))
            loss = loss + GATE_BETA * gate_loss
            if GATE_DIAG["pending"] and not GATE_DIAG["done"]:
                actor_params = (list(controller.inner.trunk.parameters())
                                + list(controller.inner.action_head.parameters()))
                base_only = loss - GATE_BETA * gate_loss
                gb = torch.autograd.grad(base_only, actor_params, retain_graph=True,
                                         allow_unused=True)
                gg = torch.autograd.grad(GATE_BETA * gate_loss, actor_params,
                                         retain_graph=True, allow_unused=True)
                l1b = float(sum(float(g.detach().abs().sum()) for g in gb if g is not None))
                l1g = float(sum(float(g.detach().abs().sum()) for g in gg if g is not None))
                dot = float(sum(float((a * b).sum()) for a, b in zip(gb, gg)
                                if a is not None and b is not None))
                nb = math.sqrt(sum(float((g ** 2).sum()) for g in gb if g is not None))
                ng = math.sqrt(sum(float((g ** 2).sum()) for g in gg if g is not None))
                GATE_DIAG["record"] = {
                    "base_actor_grad_l1": l1b, "prior_grad_l1": l1g,
                    "dot": dot, "cosine": (dot / (nb * ng)) if nb > 0 and ng > 0 else None}
                GATE_DIAG["done"] = True
"""

ANCHOR_TOTALS_INIT = """              "kl": 0.0, "clip": 0.0, "grad": 0.0}
"""

PATCH_TOTALS_INIT = """              "kl": 0.0, "clip": 0.0, "grad": 0.0, "gate": 0.0, "gate_rows": 0}
"""

ANCHOR_TOTALS_ADD = """            totals["grad"] += float(norm)
"""

PATCH_TOTALS_ADD = """            totals["grad"] += float(norm)
            totals["gate"] += float(gate_loss.item())
            totals["gate_rows"] += len(gate_skip_log_probs)
"""

ANCHOR_RETURN = """        "samples": len(samples),
"""

PATCH_RETURN = """        "samples": len(samples),
        "gate_loss": round(totals["gate"] / minibatches, 6) if minibatches else 0.0,
        "gate_rows": totals["gate_rows"],
"""

ANCHOR_MINIB = """            minibatches += 1
"""

INSERT_MINIB = """            GATE_RECORDS.append({
                "update_index": int(update_index), "minibatch": minibatches - 1,
                "n_rows": len(chunk),
                "gate_rows": len(gate_skip_log_probs),
                "gate_loss_raw": float(gate_loss.item()),
                "gate_loss_weighted": float(GATE_BETA * gate_loss.item())})
"""


def build_patched_source(trainer_module) -> tuple[str, str, list]:
    original = inspect.getsource(trainer_module.ppo_update)
    patched = original
    for anchor, insert in ((ANCHOR_USABLE, ANCHOR_USABLE + INSERT_AFTER_USABLE),
                           (ANCHOR_LISTS, INSERT_LISTS),
                           (ANCHOR_LOSS, INSERT_LOSS),
                           (ANCHOR_TOTALS_INIT, PATCH_TOTALS_INIT),
                           (ANCHOR_TOTALS_ADD, PATCH_TOTALS_ADD),
                           (ANCHOR_MINIB, ANCHOR_MINIB + INSERT_MINIB),
                           (ANCHOR_RETURN, PATCH_RETURN)):
        assert patched.count(anchor) == 1, f"anchor not unique: {anchor[:60]!r}"
        patched = patched.replace(anchor, insert)
    diff = list(difflib.unified_diff(original.splitlines(), patched.splitlines(), lineterm="", n=0))
    removed = [l[1:].strip() for l in diff if l.startswith("-") and not l.startswith("---")]
    allowed = set()
    for block in (ANCHOR_USABLE, ANCHOR_LISTS, ANCHOR_LOSS, ANCHOR_TOTALS_INIT,
                  ANCHOR_TOTALS_ADD, ANCHOR_MINIB, ANCHOR_RETURN):
        allowed.update(l.strip() for l in block.splitlines() if l.strip())
    unexpected = [l for l in removed if l not in allowed]
    if unexpected:
        raise RuntimeError(f"patch removed unexpected lines: {unexpected}")
    return original, patched, diff


def install_gate_prior(trainer_module, controller_detach_context, raw_dir: Path) -> dict[str, Any]:
    original_source, patched_source, diff = build_patched_source(trainer_module)
    namespace = dict(vars(trainer_module))
    namespace["GATE_BETA"] = GATE_BETA
    namespace["READY_CUT"] = READY_CUT
    namespace["HORIZON"] = HORIZON
    namespace["import math"] = None
    exec(compile("import math", "<r58-math>", "exec"), namespace)
    GATE_RECORDS: list[dict[str, Any]] = []
    GATE_DIAG: dict[str, Any] = {"pending": False, "done": False, "record": None}
    namespace["GATE_RECORDS"] = GATE_RECORDS
    namespace["GATE_DIAG"] = GATE_DIAG
    exec(compile(patched_source, "<r58-gate-ppo-update>", "exec"), namespace)
    patched_fn = namespace["ppo_update"]
    previous = getattr(trainer_module, "ppo_update", None)
    record_path = Path(raw_dir) / "runtime" / "r58_minibatch_records.jsonl"
    record_path.parent.mkdir(parents=True, exist_ok=True)

    def r58_ppo_update(controller, optimizer, episodes, *, update_index):
        GATE_RECORDS.clear()
        GATE_DIAG["pending"] = update_index in DIAG_UPDATES
        GATE_DIAG["done"] = False
        before = {n: p.detach().clone() for n, p in
                  list(controller.inner.trunk.named_parameters())
                  + list(controller.inner.action_head.named_parameters())}
        report = patched_fn(controller, optimizer, episodes, update_index=update_index)
        with record_path.open("a", encoding="utf-8") as handle:
            for rec in GATE_RECORDS:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        diag_record = None
        try:
                    if GATE_DIAG["done"]:
                        diag_record = {"update_index": update_index, **GATE_DIAG["record"],
                                       "params_before_sha256": hashlib.sha256(b"".join(
                                           p.detach().numpy().tobytes() for _, p in sorted(before.items()))
                                           ).hexdigest(),
                                       "params_after_sha256": hashlib.sha256(b"".join(
                                           p.detach().numpy().tobytes() for _, p in sorted(
                                               {n: pp.detach().clone() for n, pp in
                                                list(controller.inner.trunk.named_parameters())
                                                + list(controller.inner.action_head.named_parameters())
                                            }.items()))).hexdigest()}
                        with (Path(raw_dir) / "runtime" / "r58_gradient_diagnostics.jsonl").open(
                                "a", encoding="utf-8") as handle:
                            handle.write(json.dumps(diag_record, ensure_ascii=False) + "\n")
        except Exception as exc:
            diag_record = {"update_index": update_index,
                           "diagnostics_error": f"{type(exc).__name__}: {exc}"}
            with (Path(raw_dir) / "runtime" / "r58_diagnostics_errors.jsonl").open(
                    "a", encoding="utf-8") as handle:
                handle.write(json.dumps(diag_record, ensure_ascii=False) + "\n")
        report["gate_diag"] = diag_record
        return report

    trainer_module.ppo_update = r58_ppo_update
    undone = {"flag": False}

    def undo() -> None:
        if undone["flag"]:
            return
        trainer_module.ppo_update = previous
        undone["flag"] = True

    return {"kind": "r58-gate-prior/1", "beta": GATE_BETA, "ready_cut": READY_CUT,
            "skip_index": SKIP_INDEX, "diag_updates": list(DIAG_UPDATES),
            "recorder_path": str(record_path),
            "lines_added": len(patched_source.splitlines()) - len(original_source.splitlines()),
            "diff": diff, "undo": undo}
