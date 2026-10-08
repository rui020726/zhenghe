"""R56 gate-prior seam: a source-patched copy of the frozen ppo_update.

The frozen canonical trainer (sha256 0550b99e...) is never modified on disk.
This module takes inspect.getsource of the frozen ``ppo_update``, applies an
exact minimal insertion that adds  beta * L_gate  to the per-minibatch loss,
verifies the patch against a whitelist, compiles it inside a copy of the
frozen module globals, and installs it wrapped in the same critic-isolation
detach context used by the R53 adapter.  beta is frozen at 0.10.
"""
from __future__ import annotations
import difflib, inspect
from typing import Any

GATE_BETA = 0.10
READY_CUT = 147.0
HORIZON = 1000.0
SKIP_INDEX = 0  # verified from the actual wiring: recorded send_probabilities == softmax[1]

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
"""

ANCHOR_TOTALS_INIT = """              "kl": 0.0, "clip": 0.0, "grad": 0.0}
"""

PATCH_TOTALS_INIT = """              "kl": 0.0, "clip": 0.0, "grad": 0.0, "gate": 0.0}
"""

ANCHOR_TOTALS_ADD = """            totals["grad"] += float(norm)
"""

PATCH_TOTALS_ADD = """            totals["grad"] += float(norm)
            totals["gate"] += float(gate_loss.item())
"""

ANCHOR_RETURN = """        "samples": len(samples),
"""

PATCH_RETURN = """        "samples": len(samples),
        "gate_loss": round(totals["gate"] / minibatches, 6) if minibatches else 0.0,
"""


def build_patched_source(trainer_module) -> tuple[str, str]:
    original = inspect.getsource(trainer_module.ppo_update)
    patched = original
    for anchor, insert in ((ANCHOR_USABLE, ANCHOR_USABLE + INSERT_AFTER_USABLE),
                           (ANCHOR_LISTS, INSERT_LISTS),
                           (ANCHOR_LOSS, INSERT_LOSS),
                           (ANCHOR_TOTALS_INIT, PATCH_TOTALS_INIT),
                           (ANCHOR_TOTALS_ADD, PATCH_TOTALS_ADD),
                           (ANCHOR_RETURN, PATCH_RETURN)):
        assert patched.count(anchor) == 1, f"anchor not unique: {anchor[:60]!r}"
        patched = patched.replace(anchor, insert)
    diff = list(difflib.unified_diff(original.splitlines(), patched.splitlines(),
                                     lineterm="", n=0))
    removed = [l[1:].strip() for l in diff if l.startswith("-") and not l.startswith("---")]
    allowed_removed = set()
    for block in (ANCHOR_USABLE, ANCHOR_LISTS, ANCHOR_LOSS, ANCHOR_TOTALS_INIT,
                  ANCHOR_TOTALS_ADD, ANCHOR_RETURN):
        for line in block.splitlines():
            if line.strip():
                allowed_removed.add(line.strip())
    unexpected = [l for l in removed if l not in allowed_removed]
    if unexpected:
        raise RuntimeError(f"patch removed unexpected lines: {unexpected}")
    return original, patched, diff


def install_gate_prior(trainer_module, controller_detach_context) -> dict[str, Any]:
    """Install the patched ppo_update wrapped in the critic-isolation detach context."""
    original_source, patched_source, diff = build_patched_source(trainer_module)
    namespace = dict(vars(trainer_module))
    namespace["GATE_BETA"] = GATE_BETA
    namespace["READY_CUT"] = READY_CUT
    namespace["HORIZON"] = HORIZON
    exec(compile(patched_source, "<r56-gate-ppo-update>", "exec"), namespace)
    patched_fn = namespace["ppo_update"]
    previous = getattr(trainer_module, "ppo_update", None)

    def r56_ppo_update(controller, optimizer, episodes, *, update_index):
        with controller_detach_context(controller):
            return patched_fn(controller, optimizer, episodes, update_index=update_index)

    trainer_module.ppo_update = r56_ppo_update
    undone = {"flag": False}

    def undo() -> None:
        if undone["flag"]:
            return
        trainer_module.ppo_update = previous
        undone["flag"] = True

    return {"kind": "r56-gate-prior/1", "beta": GATE_BETA, "ready_cut": READY_CUT,
            "skip_index": SKIP_INDEX,
            "lines_added": len(patched_source.splitlines()) - len(original_source.splitlines()),
            "diff": diff,
            "undo": undo}
