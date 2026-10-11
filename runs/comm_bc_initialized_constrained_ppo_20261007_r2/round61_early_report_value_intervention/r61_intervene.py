"""ROUND61 intervention interposer.

A subclass of the frozen ROUND42 OneShotSendSkipCollector whose _choose_masked is the frozen
source with exactly one injection: at a selected target's FIRST legal, payable, free one-shot
decision, AFTER the real sequential budget/capacity mask and the model's own action draw, and
BEFORE the one-shot latch / action finalization / emission and the shadow-gate charge, the action
is overridden to FORCE_SEND (SEND) or FORCE_SKIP (HOLD). The latch, requested flag, decision class
and gate charge all then use the overridden action, so the one-shot closed state is updated
correctly and a SEND is really billed on the same _emit path with no extra emission and no retry.
The original model probability and original greedy action are recorded. Override happens at most
once per episode for the selected target; afterwards the frozen model's original greedy is kept.
"""
from __future__ import annotations
import inspect
import sys
from typing import Any


_ANCHOR = "                action = int(torch.multinomial(probabilities[index], num_samples=1).item())\n\n"

_INJECTION = (
    "            if self._iv_enabled and target == self._iv_target and not self._iv_done:\n"
    "                self._iv_done = True\n"
    "                _orig = int(action)\n"
    "                action = int(self._iv_action)\n"
    "                self.iv_record = {\n"
    "                    \"target\": target, \"step\": int(observation.step),\n"
    "                    \"original_model_p_send\": float(probabilities[index, SEND].item()),\n"
    "                    \"original_model_p_hold\": float(probabilities[index, HOLD].item()),\n"
    "                    \"original_greedy_action\": _orig,\n"
    "                    \"intervened_action\": int(action),\n"
    "                    \"attempt_charge_bytes\": int(charge),\n"
    "                    \"sequentially_affordable\": True,\n"
    "                    \"override_reason\": self._iv_reason,\n"
    "                    \"pre_action_features\": [float(x) for x in rows[index]],\n"
    "                }\n")


def _build_patched_source(base_cls) -> str:
    src = inspect.getsource(base_cls._choose_masked)
    if src.count(_ANCHOR) != 1:
        raise RuntimeError("ROUND61 interposer anchor is not unique in the frozen _choose_masked")
    import textwrap
    patched = src.replace(_ANCHOR, _ANCHOR + _INJECTION, 1)
    return textwrap.dedent(patched)


def make_intervene_collector(base_cls):
    """Return an R61InterveneCollector subclass of the given frozen collector class."""
    from budget_masked import SEND, HOLD
    patched = _build_patched_source(base_cls)
    # Exec in a copy of the frozen module's globals so the patched method resolves shadow_gate,
    # features, SendObservation, TAPE_SCHEMA, etc. exactly as the frozen method does.
    namespace: dict = dict(vars(sys.modules[base_cls.__module__]))
    exec(compile(patched, "<r61-intervene-choose-masked>", "exec"), namespace)

    class R61InterveneCollector(base_cls):
        def __init__(self, controller, *, greedy, target, forced_action, reason,
                     override_enabled=True):
            super().__init__(controller, greedy=greedy)
            self._iv_target = str(target)
            self._iv_action = int(forced_action)  # SEND=1 or HOLD=0
            self._iv_reason = str(reason)
            self._iv_enabled = bool(override_enabled)
            self._iv_done = False
            self.iv_record = None

    R61InterveneCollector._choose_masked = namespace["_choose_masked"]
    return R61InterveneCollector
