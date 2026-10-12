"""Private ROUND64 action interposer at the frozen legal one-shot decision point."""
from __future__ import annotations

import inspect
import sys
import textwrap


ANCHOR = "                action = int(torch.multinomial(probabilities[index], num_samples=1).item())\n\n"
INJECTION = (
    "            if self._r64_enabled and target == self._r64_target and not self._r64_done:\n"
    "                self._r64_done = True\n"
    "                _r64_original_action = int(action)\n"
    "                action = int(self._r64_action)\n"
    "                _r64_item = items[index]\n"
    "                self.r64_record = {\n"
    "                    'target': target, 'decision_step': int(observation.step),\n"
    "                    'original_p_send': float(probabilities[index, SEND].item()),\n"
    "                    'original_p_skip': float(probabilities[index, HOLD].item()),\n"
    "                    'original_greedy_action': _r64_original_action,\n"
    "                    'actual_action': int(action), 'override_reason': self._r64_reason,\n"
    "                    'legal_initial': bool(executable), 'initial_block_reason': block_reason,\n"
    "                    'quote_bytes': int(charge), 'remaining_budget_before': float(remaining_before),\n"
    "                    'capacity_remaining': observation.capacity_remaining,\n"
    "                    'sequentially_affordable': bool(affordable),\n"
    "                    'deadline': float(_r64_item.deadline),\n"
    "                    'remaining_time': float(_r64_item.remaining_time),\n"
    "                    'current_time': float(observation.current_time),\n"
    "                    'ready_step': int(_r64_item.ready_step),\n"
    "                    'first_emission': bool(_r64_item.first_emission),\n"
    "                    'features_19d': [float(x) for x in rows[index]],\n"
    "                }\n"
)


def patched_choose_masked_source(base_cls):
    source = inspect.getsource(base_cls._choose_masked)
    if source.count(ANCHOR) != 1:
        raise RuntimeError("R64 action injection anchor is not unique")
    return textwrap.dedent(source.replace(ANCHOR, ANCHOR + INJECTION, 1))


def make_intervene_collector(base_cls):
    source = patched_choose_masked_source(base_cls)
    namespace = dict(vars(sys.modules[base_cls.__module__]))
    exec(compile(source, "<r64-private-one-shot-interposer>", "exec"), namespace)

    class R64InterveneCollector(base_cls):
        def __init__(self, controller, *, greedy, target, forced_action, reason,
                     override_enabled=True):
            super().__init__(controller, greedy=greedy)
            self._r64_target = str(target)
            self._r64_action = int(forced_action)
            self._r64_reason = str(reason)
            self._r64_enabled = bool(override_enabled)
            self._r64_done = False
            self.r64_record = None

    R64InterveneCollector._choose_masked = namespace["_choose_masked"]
    return R64InterveneCollector
