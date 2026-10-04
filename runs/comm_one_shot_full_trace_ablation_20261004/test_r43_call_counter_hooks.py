from __future__ import annotations

import unittest

import r43_call_counter_hooks as hooks


class _Sentinel(Exception):
    pass


class CallCounterHookTests(unittest.TestCase):
    def test_entry_count_and_argument_return_identity(self) -> None:
        seen = []
        token = object()
        result = object()

        def stub(*args, **kwargs):
            self.assertIs(args[0], token)
            self.assertEqual(kwargs, {"update_index": 7})
            return result

        wrapped = hooks.count_at_entry(stub, lambda *a, **k: seen.append((a, k)))
        self.assertIs(wrapped(token, update_index=7), result)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0][0], token)
        self.assertEqual(seen[0][1], {"update_index": 7})

    def test_entry_count_precedes_and_preserves_original_exception(self) -> None:
        seen = []
        error = _Sentinel("guarded real update boundary")

        def blocked(*_args, **_kwargs):
            raise error

        wrapped = hooks.count_at_entry(
            blocked, lambda *a, **k: seen.append((a, k)))
        with self.assertRaises(_Sentinel) as caught:
            wrapped(object(), batch=0)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1], {"batch": 0})
        self.assertEqual(hooks.COUNTER_HOOK_VERSION,
                         "round43-training-call-counter/2")


if __name__ == "__main__":
    unittest.main()
