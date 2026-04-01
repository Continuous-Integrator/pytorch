# Owner(s): ["module: dynamo"]
"""Tests for mp_subscript_impl: unified __getitem__ dispatch via vt_getitem in Dynamo."""

import operator

import torch
import torch._dynamo.test_case


class GetItemTests(torch._dynamo.test_case.TestCase):
    def test_object_without_getitem(self):
        class NoGetItem:
            pass

        def fn(x):
            obj = NoGetItem()
            return operator.getitem(obj, 0)

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
