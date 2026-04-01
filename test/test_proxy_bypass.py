# Owner(s): ["module: dispatch"]

"""
Tests for the proxy_call bypass optimization that directly calls
fake_mode.dispatch() for eligible ops, skipping proxy mode re-entry.
"""

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase

aten = torch.ops.aten


class TestProxyBypass(TestCase):
    def test_bypass_eligibility(self):
        # Pointwise ops: eligible (single return, not data-dependent)
        self.assertTrue(aten.cos.default._can_bypass_proxy_dispatch)
        self.assertTrue(aten.add.Tensor._can_bypass_proxy_dispatch)
        self.assertTrue(aten.relu.default._can_bypass_proxy_dispatch)
        # Multi-return ops: ineligible
        self.assertFalse(aten.var_mean.correction._can_bypass_proxy_dispatch)
        self.assertFalse(aten.topk.default._can_bypass_proxy_dispatch)
        # Data-dependent ops: ineligible
        self.assertFalse(aten.nonzero.default._can_bypass_proxy_dispatch)

    def test_bypass_correctness_pointwise(self):
        def f(x, y):
            return (x + y).cos() * x.sin()

        x, y = torch.randn(3, 4), torch.randn(3, 4)
        gm = make_fx(f, tracing_mode="fake")(x, y)
        self.assertEqual(gm(x, y), f(x, y))

    def test_bypass_correctness_linear(self):
        def f(x, w, b):
            return torch.nn.functional.relu(
                torch.nn.functional.linear(x, w, b)
            )

        x = torch.randn(4, 16)
        w = torch.randn(32, 16)
        b = torch.randn(32)
        gm = make_fx(f, tracing_mode="fake")(x, w, b)
        self.assertEqual(gm(x, w, b), f(x, w, b))

    def test_bypass_preserves_val_metadata(self):
        def f(x):
            return x.cos().sin()

        gm = make_fx(f, tracing_mode="fake")(torch.randn(5, 5))
        for n in gm.graph.nodes:
            if n.op not in ("output", "placeholder"):
                self.assertIn("val", n.meta, f"Node {n} missing 'val' metadata")

    def test_ineligible_ops_still_work(self):
        # Multi-return op should go through normal path and still produce correct graph
        def f(x):
            vals, indices = torch.topk(x, 3)
            return vals

        x = torch.randn(10)
        gm = make_fx(f, tracing_mode="fake")(x)
        self.assertEqual(gm(x), f(x))


if __name__ == "__main__":
    run_tests()
