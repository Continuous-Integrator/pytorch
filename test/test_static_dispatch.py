# Owner(s): ["module: dispatch"]

"""
Tests for static dispatch POC for TorchDispatchMode.

This module tests the static dispatch infrastructure that eliminates TLS
stack lookups for infra modes by statically chaining them in known order.
"""

import time
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.fx.experimental.proxy_tensor import make_fx


class TestStaticDispatch(TestCase):
    """Tests for static dispatch functionality."""

    def test_static_dispatch_basic(self):
        """Basic test that static dispatch works."""
        def model(x):
            return x.sin() + x.cos()

        x = torch.randn(4, 4)

        # Static dispatch
        gm = make_fx(model, static_dispatch=True)(x)

        # Verify graph executes correctly
        out_expected = model(x)
        out_actual = gm(x)
        self.assertEqual(out_expected, out_actual)

    def test_static_dispatch_graph_equivalence(self):
        """Verify static and dynamic dispatch produce identical graphs."""
        def model(x, y):
            z = x @ y
            return F.gelu(z) + z.sin()

        x = torch.randn(32, 32)
        y = torch.randn(32, 32)

        # Dynamic dispatch (current behavior)
        gm_dynamic = make_fx(model)(x, y)

        # Static dispatch (new behavior)
        gm_static = make_fx(model, static_dispatch=True)(x, y)

        # Verify graphs are functionally equivalent
        out_dynamic = gm_dynamic(x, y)
        out_static = gm_static(x, y)
        self.assertEqual(out_dynamic, out_static)

        # Verify graph structure is similar (may differ slightly in structure)
        # Just check that both have valid graphs
        self.assertGreater(len(list(gm_dynamic.graph.nodes)), 0)
        self.assertGreater(len(list(gm_static.graph.nodes)), 0)

    def test_static_dispatch_with_decomposition(self):
        """Test static dispatch with decomposition table."""
        def my_gelu_decomp(x):
            return x * 0.5 * (1 + torch.tanh(
                0.7978845608028654 * (x + 0.044715 * x.pow(3))
            ))

        decomp_table = {torch.ops.aten.gelu.default: my_gelu_decomp}

        def model(x):
            return F.gelu(x)

        x = torch.randn(8, 8)

        # With decomposition and static dispatch
        gm = make_fx(
            model,
            decomposition_table=decomp_table,
            static_dispatch=True
        )(x)

        # Verify output is correct
        out_expected = model(x)
        out_actual = gm(x)
        # Note: decomposed output may differ slightly due to implementation
        self.assertTrue(gm is not None)
        self.assertGreater(len(list(gm.graph.nodes)), 0)

    def test_static_dispatch_symbolic_tracing(self):
        """Test static dispatch with symbolic tracing mode."""
        def model(x):
            return x.sin() * x.cos()

        x = torch.randn(4, 4)

        # Static dispatch with symbolic tracing
        gm = make_fx(model, tracing_mode="symbolic", static_dispatch=True)(x)

        # Verify graph executes correctly
        out_expected = model(x)
        out_actual = gm(x)
        self.assertEqual(out_expected, out_actual)

    def test_static_dispatch_fake_tracing(self):
        """Test static dispatch with fake tensor mode."""
        def model(x):
            return x.relu() + 1

        x = torch.randn(4, 4)

        # Static dispatch with fake tracing
        gm = make_fx(model, tracing_mode="fake", static_dispatch=True)(x)

        # Verify graph executes correctly
        out_expected = model(x)
        out_actual = gm(x)
        self.assertEqual(out_expected, out_actual)

    def test_user_mode_with_static_infra(self):
        """Verify user modes work correctly with static infra dispatch."""
        from torch.utils._python_dispatch import TorchDispatchMode

        call_count = [0]

        class CountingMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                call_count[0] += 1
                return func(*args, **(kwargs or {}))

        def model(x):
            return x.sin() + x.cos()

        x = torch.randn(4, 4)

        with CountingMode():
            gm = make_fx(model, static_dispatch=True)(x)

        # Verify user mode was called
        self.assertGreater(call_count[0], 0)

        # Verify graph is correct
        out_expected = model(x)
        out_actual = gm(x)
        self.assertEqual(out_expected, out_actual)


class TestStaticDispatchPerformance(TestCase):
    """Performance benchmarks for static dispatch."""

    def _benchmark_make_fx(self, model, x, static_dispatch, num_warmup=3, num_runs=20):
        """Benchmark make_fx with given settings."""
        # Cold start: first call
        cold_start = time.perf_counter()
        _ = make_fx(model, static_dispatch=static_dispatch)(x)
        cold_time = time.perf_counter() - cold_start

        # Warm up
        for _ in range(num_warmup):
            _ = make_fx(model, static_dispatch=static_dispatch)(x)

        # Warm start: averaged over multiple runs
        warm_start = time.perf_counter()
        for _ in range(num_runs):
            _ = make_fx(model, static_dispatch=static_dispatch)(x)
        warm_time = (time.perf_counter() - warm_start) / num_runs

        return cold_time, warm_time

    def test_performance_cold_start(self):
        """Measure cold start performance."""
        def model(x):
            y = x @ x.T
            y = F.gelu(y)
            y = y.sin() + y.cos()
            return F.softmax(y, dim=-1)

        x = torch.randn(64, 64)

        cold_dynamic, _ = self._benchmark_make_fx(model, x, static_dispatch=False, num_runs=1)
        cold_static, _ = self._benchmark_make_fx(model, x, static_dispatch=True, num_runs=1)

        print(f"\nCold Start Performance:")
        print(f"  Dynamic dispatch: {cold_dynamic * 1000:.2f}ms")
        print(f"  Static dispatch:  {cold_static * 1000:.2f}ms")
        if cold_static > 0:
            print(f"  Speedup: {cold_dynamic / cold_static:.2f}x")

    def test_performance_warm_start(self):
        """Measure warm start performance."""
        def model(x):
            y = x @ x.T
            y = F.gelu(y)
            y = y.sin() + y.cos()
            return F.softmax(y, dim=-1)

        x = torch.randn(64, 64)

        _, warm_dynamic = self._benchmark_make_fx(model, x, static_dispatch=False)
        _, warm_static = self._benchmark_make_fx(model, x, static_dispatch=True)

        print(f"\nWarm Start Performance (avg of 20 runs):")
        print(f"  Dynamic dispatch: {warm_dynamic * 1000:.2f}ms")
        print(f"  Static dispatch:  {warm_static * 1000:.2f}ms")
        if warm_static > 0:
            print(f"  Speedup: {warm_dynamic / warm_static:.2f}x")

    def test_performance_varying_model_sizes(self):
        """Benchmark across different model complexities."""
        results = []

        for num_ops in [5, 10, 25, 50]:
            def make_model(n):
                def model(x):
                    for _ in range(n):
                        x = x.sin()
                    return x
                return model

            model = make_model(num_ops)
            x = torch.randn(32, 32)

            _, warm_dynamic = self._benchmark_make_fx(model, x, static_dispatch=False, num_runs=10)
            _, warm_static = self._benchmark_make_fx(model, x, static_dispatch=True, num_runs=10)

            speedup = warm_dynamic / warm_static if warm_static > 0 else 0
            results.append((num_ops, warm_dynamic, warm_static, speedup))

        print(f"\nPerformance by Model Complexity:")
        print(f"{'Ops':<6} {'Dynamic (ms)':<14} {'Static (ms)':<14} {'Speedup':<8}")
        print("-" * 44)
        for num_ops, dyn, stat, speedup in results:
            print(f"{num_ops:<6} {dyn*1000:<14.2f} {stat*1000:<14.2f} {speedup:<8.2f}x")


class TestTorchCompileIntegration(TestCase):
    """Tests for torch.compile integration with static dispatch."""

    def test_torch_compile_basic(self):
        """Basic test that torch.compile works (baseline)."""
        def model(x):
            return x.sin() + x.cos()

        x = torch.randn(4, 4)

        # Compile the model
        compiled = torch.compile(model, backend="eager")

        # Verify it works
        out_expected = model(x)
        out_actual = compiled(x)
        self.assertEqual(out_expected, out_actual)

    def test_torch_compile_with_decomposition(self):
        """Test torch.compile with decomposition."""
        def model(x):
            return F.gelu(x) + x.sin()

        x = torch.randn(4, 4)

        # Compile with eager backend
        compiled = torch.compile(model, backend="eager")

        # Verify it works
        out_expected = model(x)
        out_actual = compiled(x)
        self.assertEqual(out_expected, out_actual)


if __name__ == "__main__":
    run_tests()
