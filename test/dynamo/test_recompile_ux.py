# Owner(s): ["module: dynamo"]
import unittest
import weakref

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._logging
from torch._dynamo.exc import FailOnRecompileLimitHit
from torch.testing._internal.logging_utils import kwargs_to_settings, log_settings


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class RecompileUxTests(torch._dynamo.test_case.TestCase):
    # TODO(whc) dynamo actually recompiles one more time than the cache limit
    cache_limit = 1

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            torch._dynamo.config.patch("recompile_limit", cls.cache_limit)
        )

    def test_drop_cache_on_skip(self):
        def model(x, i):
            return x + i

        attached = False
        triggered = False

        def trigger():
            nonlocal triggered
            triggered = True

        def compiler(gm, input):
            nonlocal attached
            f = gm.forward
            if attached:
                raise AssertionError("Expected not attached")
            # NB: making this a weakref.ref causes the cycle to no
            # longer be promptly GC'ed
            weakref.finalize(f, trigger)
            attached = True
            return f

        x = torch.randn(2)
        for i in range(2):
            opt_model = torch.compile(model, backend=compiler)
            opt_model(x, i)

        self.assertTrue(triggered)

    def test_loop_torture(self):
        def loop_torture(input, iters):
            out = input
            # randint itself causes one graph break
            for _ in range(iters):
                out += input
            return out

        compile_counter = torch._dynamo.testing.CompileCounter()
        for _ in range(10):
            x = torch.randn(3)
            iters = torch.randint(low=0, high=1000, size=())
            opt_loop_torture = torch.compile(loop_torture, backend=compile_counter)
            opt_loop_torture(x, iters)

        # Currently, we recompile each time,
        # We'd probably like to bail out quickly and warn
        # TODO(whc) these checks fail on py37.  Why?
        # self.assertEqual(counters["frames"]["total"], 2 + self.cache_limit)
        # self.assertEqual(counters["frames"]["ok"], 1 + self.cache_limit)

        # compile_counter only sees frames that were fed to the backend compiler,
        # which is a subset of counters["frames"]["ok"] -- probably because
        # counters["frames"]["ok"] includes frames not containing torch ops?
        self.assertEqual(compile_counter.frame_count, self.cache_limit)

    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    def test_dynamic_input(self):
        def model(input):
            return input + input

        expected_recompiles = 2
        compile_counter = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch("recompile_limit", expected_recompiles):
            with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
                for _ in range(10):
                    bsz = torch.randint(low=0, high=1000, size=())
                    x = torch.randn((bsz, 3, 4))
                    opt_model = torch.compile(model, backend=compile_counter)
                    opt_model(x)

        self.assertEqual(compile_counter.frame_count, expected_recompiles)
        self.assertEqual(len(logs.records), 1)
        print(logs.records[0])
        self.assertTrue(
            logs.records[0]
            .getMessage()
            .startswith("torch._dynamo hit config.recompile_limit")
        )

    @unittest.skipIf(
        not torch.cuda.is_available() and not torch.xpu.is_available(),
        "requires cuda or xpu",
    )
    def test_nvfuser_guards(self):
        # we may want to model dynamo's guards sufficiently after nvfuser's ProfilingExecutor guards
        # such that we ensure dynamo is in charge of all the recompilations at the top level,
        # and we could thus simplify the underlying torchscript executor
        def func(a, b, c):
            return a + b * c

        a = torch.rand(3, 4, 5, device=device_type)
        b = torch.rand(3, 4, 5, device=device_type)
        b_v = torch.rand(3, 5, 4, device=device_type).view(3, 4, 5)
        b_p = torch.rand(3, 5, 4, device=device_type).permute(0, 2, 1)
        c = torch.rand(3, 4, 5, device=device_type)
        compile_counter = torch._dynamo.testing.CompileCounter()

        with torch._dynamo.config.patch("recompile_limit", 2):
            opt_func = torch.compile(func, backend=compile_counter)
            opt_func(a, b, c)  # warmup
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b, c)  # no guard fail or recompile
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b_v, c)  # a view should not cause nvfuser recompile
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b_p, c)  # a permutation should cause recompile
            self.assertEqual(compile_counter.frame_count, 2)

    def assert_single_log_contains(self, logs, contains_str):
        self.assertEqual(len(logs.records), 1)
        self.assertTrue(
            logs.records[0].getMessage().find(contains_str) > 0,
            msg=f'Expected to find "{contains_str}" in log "{logs.records[0].getMessage()}"',
        )

    def test_verbose_tensor_check(self):
        def func(a):
            # Warning: choose a function here whose meta implementation lives
            # entirely in C++.  If you do a Python one, Dynamo will dive into
            # torch._refs which is OK but it will muddy up the warnings
            return torch.add(a, 4)

        def cache_fail_test(cached_input, missed_input, expected_failure):
            # TODO(whc) maybe its hacky to have a 'test within a test' but this seemed convenient
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            opt_func = torch.compile(func, backend="eager")
            # warmup
            opt_func(cached_input)

            with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
                opt_func = torch.compile(func, backend="eager")
                opt_func(missed_input)
            self.assert_single_log_contains(logs, expected_failure)

        a = torch.rand(3, 4, 5)
        cache_fail_test(
            a,
            a[0:2, :, :],
            "tensor 'a' size mismatch at index 0. expected 3, actual 2",
        )
        cache_fail_test(
            a,
            a.clone().as_strided((3, 4, 5), stride=(1, 3, 12)),
            "tensor 'a' stride mismatch at index 0. expected 20, actual 1",
        )
        cache_fail_test(a, a[0, :, :], "tensor 'a' rank mismatch. expected 3, actual 2")
        cache_fail_test(a, a.to("meta"), "tensor 'a' dispatch key set mismatch.")
        cache_fail_test(
            a,
            a.to(torch.float16),
            "tensor 'a' dtype mismatch. expected Float, actual Half",
        )
        a_grad = a.clone()
        a_grad.requires_grad = True
        cache_fail_test(
            a,
            a_grad,
            "tensor 'a' requires_grad mismatch. expected requires_grad=0",
        )

    def test_mismatched_type(self):
        a = torch.rand(3, 4, 5)
        b = torch.rand(3, 4, 5)

        def func(a, b):
            return a + b

        opt_func = torch.compile(func, backend="eager")
        # warmup
        opt_func(a, b)

        with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
            opt_func = torch.compile(func, backend="eager")
            opt_func(a, 1)
        self.assert_single_log_contains(
            logs,
            "expected type of 'b' to be a tensor type, ' but found <class 'int'>",
        )

    @torch._dynamo.config.patch(recompile_limit=1, fail_on_recompile_limit_hit=True)
    def test_fail_on_recompile_limit_hit(self):
        @torch.compile(backend="eager")
        def func(b, a):
            if a:
                return b * 2
            else:
                return b + 1

        func(torch.randn(5), True)
        with self.assertRaises(FailOnRecompileLimitHit):
            func(torch.randn(5), False)

    @torch._dynamo.config.patch("recompile_limit", 32)
    def test_multiple_guard_fails(self):
        failure_reasons = []

        def guard_fail_fn(failure):
            failure_reasons.append(failure[0])

        def f(x):
            return torch.relu(x)

        opt_f = torch._dynamo.optimize(
            backend="eager", guard_fail_fn=guard_fail_fn, dynamic=False
        )(f)

        for i in range(5):
            failure_reasons.clear()
            opt_f(torch.randn(8 + i))

        failure_str = "\n".join(failure_reasons)
        for line in [
            "tensor 'x' size mismatch at index 0. expected 11, actual 12",
            "tensor 'x' size mismatch at index 0. expected 10, actual 12",
            "tensor 'x' size mismatch at index 0. expected 9, actual 12",
            "tensor 'x' size mismatch at index 0. expected 8, actual 12",
        ]:
            self.assertIn(
                line,
                failure_str,
            )

    @torch._dynamo.config.patch("recompile_limit", 32)
    def test_multiple_guard_fails_report_all(self):
        with log_settings(kwargs_to_settings(recompiles_verbose=True)):
            failure_reasons = []

            def guard_fail_fn(failure):
                failure_reasons.append(failure[0])

            def f(x):
                return torch.ones(len(x), x[-1])

            opt_f = torch._dynamo.optimize(
                backend="eager", guard_fail_fn=guard_fail_fn, dynamic=False
            )(f)

            opt_f([4, 5, 6])

            def filter_reasons():
                return "\n".join(
                    [
                        line
                        for line in "\n".join(failure_reasons).splitlines()
                        if not line.startswith("___check_type_id")
                    ]
                )

            failure_reasons.clear()
            opt_f([7, 8])

            for line in ["len(x) == 3"]:
                self.assertIn(line, filter_reasons())

            failure_reasons.clear()
            opt_f([9])

            for line in ["len(x) == 2", "len(x) == 3"]:
                self.assertIn(line, filter_reasons())

    @torch._dynamo.config.patch(recompile_limit=1)
    def test_recompile_child_run_only(self):
        def f(x, n):
            if torch.compiler.is_compiling():
                x = x + 1
            x = g(x)
            return h(x) + n

        def g(x):
            if torch.compiler.is_compiling():
                return x + 2
            return x

        def h(x):
            if torch.compiler.is_compiling():
                return x + 4
            return x

        torch.compile(g, backend="eager")(torch.randn(3))
        inp = torch.randn(3)
        opt_f = torch.compile(f, backend="eager")
        opt_f(inp, 0)

        # expect f to run eager, g compiled (from previous invocatino), h eager
        res = opt_f(inp, 1)

        self.assertEqual(res, inp + 3)


class RecompileLimitKwargTests(torch._dynamo.test_case.TestCase):
    @staticmethod
    def _num_cache_entries(code):
        return len(torch._dynamo.eval_frame._debug_get_cache_entry_list(code))

    def test_recompile_limit_basic(self):
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + y

        opt_f = torch.compile(f, backend=cnt, recompile_limit=2)

        opt_f(torch.randn(3), torch.randn(3))
        self.assertEqual(self._num_cache_entries(f), 1)

        opt_f(torch.randn(3, dtype=torch.float64), torch.randn(3, dtype=torch.float64))
        self.assertEqual(self._num_cache_entries(f), 2)

        # Third dtype should NOT trigger recompilation (recompile_limit=2)
        opt_f(torch.randn(3, dtype=torch.float16), torch.randn(3, dtype=torch.float16))
        self.assertEqual(self._num_cache_entries(f), 2)

    def test_recompile_limit_none_uses_global(self):
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + y

        # Without recompile_limit kwarg, uses global config (default 8)
        opt_f = torch.compile(f, backend=cnt)

        for i in range(10):
            dtype = [
                torch.float32,
                torch.float64,
                torch.float16,
                torch.bfloat16,
                torch.int32,
                torch.int64,
                torch.int16,
                torch.int8,
                torch.uint8,
                torch.complex64,
            ][i]
            opt_f(torch.ones(3, dtype=dtype), torch.ones(3, dtype=dtype))

        self.assertEqual(
            self._num_cache_entries(f), torch._dynamo.config.recompile_limit
        )

    def test_recompile_limit_stricter_than_global(self):
        """recompile_limit kwarg can be stricter than the global config."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        # Global default is 8, but this region only allows 1
        opt_f = torch.compile(f, backend=cnt, recompile_limit=1)

        opt_f(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        # Should stop — recompile_limit=1 reached
        opt_f(torch.randn(3, dtype=torch.float64))
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=True)
    def test_recompile_limit_resume_function_auto_dynamic(self):
        """With automatic dynamic shapes and recompile_limit=2, the resume
        function recompiles via dimension changes on a global tensor while
        the main function gets cache hits. The resume function should stop
        at 2 entries and fall back to eager."""
        cnt = torch._dynamo.testing.CompileCounter()

        y_holder = {"tensor": torch.randn(4, 8, 2)}

        def f(x):
            x.sin()
            print("graph break")
            return y_holder["tensor"].cos()

        opt_f = torch.compile(f, backend=cnt, recompile_limit=2)

        # Call 1: static compile
        y_holder["tensor"] = torch.randn(4, 8, 2)
        opt_f(torch.randn(4, 8, 2))

        # Call 2: y dim0 changes -> f cache hit, resume recompiles
        y_holder["tensor"] = torch.randn(5, 8, 2)
        opt_f(torch.randn(4, 8, 2))
        frame_count_after_2 = cnt.frame_count

        # Call 3: y dim1 changes -> resume should NOT recompile
        # (resume already has 2 entries = recompile_limit)
        y_holder["tensor"] = torch.randn(5, 9, 2)
        opt_f(torch.randn(4, 8, 2))
        self.assertEqual(cnt.frame_count, frame_count_after_2)

        # Verify f has 1 entry, resume has 2
        num_f_entries = len(torch._dynamo.eval_frame._debug_get_cache_entry_list(f))
        self.assertEqual(num_f_entries, 1)

        from torch._dynamo.resume_execution import ContinueExecutionCache

        resume_codes = list(ContinueExecutionCache.cache[f.__code__].values())
        self.assertTrue(len(resume_codes) > 0, "No resume functions found")
        for resume_code in resume_codes:
            num_resume_entries = len(
                torch._dynamo.eval_frame._debug_get_cache_entry_list(resume_code)
            )
            self.assertEqual(num_resume_entries, 2)


class IsolatedRegionTests(torch._dynamo.test_case.TestCase):
    """Tests for isolated_region=True on torch.compile(). Each compile region
    gets its own isolated cache via the per-region cache map."""

    @staticmethod
    def _num_cache_entries(code):
        return len(torch._dynamo.eval_frame._debug_get_cache_entry_list(code))

    @torch._dynamo.config.patch(recompile_limit=2)
    def test_isolated_region_basic(self):
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + y

        opt_f = torch.compile(f, backend=cnt, isolated_region=True)

        opt_f(torch.randn(3), torch.randn(3))
        self.assertEqual(self._num_cache_entries(f), 1)

        opt_f(torch.randn(3, dtype=torch.float64), torch.randn(3, dtype=torch.float64))
        self.assertEqual(self._num_cache_entries(f), 2)

        opt_f(torch.randn(3, dtype=torch.float16), torch.randn(3, dtype=torch.float16))
        self.assertEqual(self._num_cache_entries(f), 2)

    def test_isolated_region_same_function_different_regions(self):
        """Two torch.compile() calls on the same function with isolated_region
        get fully independent caches."""
        cnt1 = torch._dynamo.testing.CompileCounter()
        cnt2 = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + y

        opt_f = torch.compile(f, backend=cnt1, isolated_region=True)
        opt_g = torch.compile(f, backend=cnt2, isolated_region=True)

        opt_f(torch.randn(3), torch.randn(3))
        self.assertEqual(cnt1.frame_count, 1)

        opt_f(
            torch.randn(3, dtype=torch.float64),
            torch.randn(3, dtype=torch.float64),
        )
        self.assertEqual(cnt1.frame_count, 2)

        # opt_g can compile despite f.__code__ having 2 entries from opt_f
        opt_g(torch.randn(3, dtype=torch.float16), torch.randn(3, dtype=torch.float16))
        self.assertEqual(cnt2.frame_count, 1)

    @torch._dynamo.config.patch(recompile_limit=1, fail_on_recompile_limit_hit=True)
    def test_isolated_region_factory_pattern(self):
        """Factory creates multiple torch.compile wrappers around the same
        inner function. Each gets its own isolated cache."""
        from functools import cache

        def core(x):
            return x.sum()

        @cache
        def factory(key):
            @torch.compile(fullgraph=True, dynamic=False, isolated_region=True)
            def frontend(x, n):
                return core(x) + n

            return frontend

        factory("foo")(torch.ones(3), 3)
        factory("bar")(torch.ones(4), 3)
        factory("baz")(torch.ones(5), 3)

    def test_isolated_region_static_and_dynamic(self):
        """Two compile regions on the same function: one static, one dynamic.
        Their cache entries don't interfere."""
        cnt_static = torch._dynamo.testing.CompileCounter()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()

        def magic(x, y):
            return x.sum() - y.sum()

        magic_static = torch.compile(
            magic, backend=cnt_static, dynamic=False, isolated_region=True
        )
        magic_dynamic = torch.compile(
            magic, backend=cnt_dynamic, dynamic=True, isolated_region=True
        )

        magic_static(torch.randn(128, 32), torch.randn(128, 32))
        self.assertEqual(cnt_static.frame_count, 1)

        magic_dynamic(torch.randn(64, 16), torch.randn(64, 16))
        self.assertEqual(cnt_dynamic.frame_count, 1)

        magic_static(torch.randn(128, 32), torch.randn(128, 32))
        self.assertEqual(cnt_static.frame_count, 1)

        magic_dynamic(torch.randn(32, 8), torch.randn(32, 8))
        self.assertEqual(cnt_dynamic.frame_count, 1)

    @torch._dynamo.config.patch(recompile_limit=1)
    def test_isolated_region_fullgraph_raises(self):
        """With fullgraph=True, hitting the recompile limit raises
        FailOnRecompileLimitHit."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_f = torch.compile(f, backend=cnt, fullgraph=True, isolated_region=True)

        opt_f(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        with self.assertRaisesRegex(
            FailOnRecompileLimitHit,
            "fullgraph=True",
        ):
            opt_f(torch.randn(3, dtype=torch.float64))

    def test_isolated_region_two_models_independent(self):
        """Two compile regions wrapping the same model. Recompilations in
        one region don't affect the other."""
        cnt1 = torch._dynamo.testing.CompileCounter()
        cnt2 = torch._dynamo.testing.CompileCounter()

        def helper(x):
            return x.sin()

        def model(x):
            return helper(x).cos()

        opt_a = torch.compile(model, backend=cnt1, isolated_region=True)
        opt_b = torch.compile(model, backend=cnt2, isolated_region=True)

        opt_a(torch.randn(3))
        frame_count_a = cnt1.frame_count

        opt_b(torch.randn(4))
        frame_count_b = cnt2.frame_count

        opt_a(torch.randn(3, dtype=torch.float64))
        self.assertGreater(cnt1.frame_count, frame_count_a)

        opt_b(torch.randn(4))
        self.assertEqual(cnt2.frame_count, frame_count_b)

    @torch._dynamo.config.patch(recompile_limit=3, accumulated_recompile_limit=64)
    def test_global_recompile_limit_resume_function(self):
        """Documents existing behavior: global recompile_limit is per-code-object.
        Resume functions independently accumulate entries."""
        cnt = torch._dynamo.testing.CompileCounter()

        mode = {"value": "a"}

        def f(x):
            a = x.sin()
            print("graph break")
            if mode["value"] == "a":
                return a.cos()
            elif mode["value"] == "b":
                return a.tan()
            elif mode["value"] == "c":
                return a.exp()
            else:
                return a + 1

        opt_f = torch.compile(f, backend=cnt)

        opt_f(torch.randn(4, 8))
        frame_count_after_1 = cnt.frame_count

        mode["value"] = "b"
        opt_f(torch.randn(4, 8))
        self.assertGreater(cnt.frame_count, frame_count_after_1)

        mode["value"] = "c"
        opt_f(torch.randn(4, 8))
        frame_count_after_3 = cnt.frame_count

        mode["value"] = "d"
        opt_f(torch.randn(4, 8))
        self.assertEqual(cnt.frame_count, frame_count_after_3)

    def test_isolated_region_mark_dynamic_vs_static(self):
        """Two regions on the same function: one with mark_static, one with
        mark_dynamic. Their guards don't interfere."""
        cnt_static = torch._dynamo.testing.CompileCounter()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_static = torch.compile(f, backend=cnt_static, isolated_region=True)
        opt_dynamic = torch.compile(f, backend=cnt_dynamic, isolated_region=True)

        x_static = torch.randn(4, 8)
        torch._dynamo.mark_static(x_static, 0)
        opt_static(x_static)
        self.assertEqual(cnt_static.frame_count, 1)

        x_dynamic = torch.randn(4, 8)
        torch._dynamo.mark_dynamic(x_dynamic, 0)
        opt_dynamic(x_dynamic)
        self.assertEqual(cnt_dynamic.frame_count, 1)

        x_static2 = torch.randn(4, 8)
        torch._dynamo.mark_static(x_static2, 0)
        opt_static(x_static2)
        self.assertEqual(cnt_static.frame_count, 1)

        x_dynamic2 = torch.randn(7, 8)
        opt_dynamic(x_dynamic2)
        self.assertEqual(cnt_dynamic.frame_count, 1)

        x_static3 = torch.randn(7, 8)
        torch._dynamo.mark_static(x_static3, 0)
        opt_static(x_static3)
        self.assertEqual(cnt_static.frame_count, 2)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=True)
    def test_isolated_region_auto_dynamic_shared_pgo(self):
        """With isolated_region, PGO (frame_state) is shared. Region B
        benefits from region A's shape observations."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt_a, isolated_region=True)
        opt_b = torch.compile(f, backend=cnt_b, isolated_region=True)

        opt_a(torch.randn(3, 4))
        opt_a(torch.randn(5, 4))
        self.assertEqual(cnt_a.frame_count, 2)

        opt_b(torch.randn(7, 4))
        self.assertEqual(cnt_b.frame_count, 1)

        opt_b(torch.randn(9, 4))
        self.assertEqual(cnt_b.frame_count, 1)

    def test_isolated_region_explicit_dispatch(self):
        """Explicit dispatch between static and dynamic regions."""
        cnt_static = torch._dynamo.testing.CompileCounter()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()

        def magic(x, y):
            return x.sum() - y.sum()

        magic_static = torch.compile(
            magic, backend=cnt_static, dynamic=False, isolated_region=True
        )
        magic_dynamic = torch.compile(
            magic, backend=cnt_dynamic, dynamic=True, isolated_region=True
        )

        static_shape = (128, 32)
        magic_static(torch.randn(*static_shape), torch.randn(*static_shape))
        magic_static(torch.randn(*static_shape), torch.randn(*static_shape))
        self.assertEqual(cnt_static.frame_count, 1)

        magic_dynamic(torch.randn(64, 16), torch.randn(64, 16))
        magic_dynamic(torch.randn(32, 8), torch.randn(32, 8))
        magic_dynamic(torch.randn(128, 64), torch.randn(128, 64))
        self.assertEqual(cnt_dynamic.frame_count, 1)

        magic_static(torch.randn(*static_shape), torch.randn(*static_shape))
        self.assertEqual(cnt_static.frame_count, 1)

        magic_static(torch.randn(64, 16), torch.randn(64, 16))
        self.assertEqual(cnt_static.frame_count, 2)

        magic_dynamic(torch.randn(256, 128), torch.randn(256, 128))
        self.assertEqual(cnt_dynamic.frame_count, 1)

    def test_isolated_region_runtime_dispatch(self):
        """Runtime dispatch between two isolated regions."""
        cnt_small = torch._dynamo.testing.CompileCounter()
        cnt_large = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin() + x.cos()

        opt_small = torch.compile(f, backend=cnt_small, isolated_region=True)
        opt_large = torch.compile(f, backend=cnt_large, isolated_region=True)

        def dispatch(x):
            if x.shape[0] <= 16:
                return opt_small(x)
            else:
                return opt_large(x)

        dispatch(torch.randn(4, 8))
        self.assertEqual(cnt_small.frame_count, 1)
        self.assertEqual(cnt_large.frame_count, 0)

        dispatch(torch.randn(32, 8))
        self.assertEqual(cnt_small.frame_count, 1)
        self.assertEqual(cnt_large.frame_count, 1)

        dispatch(torch.randn(8, 16))
        self.assertEqual(cnt_small.frame_count, 2)
        self.assertEqual(cnt_large.frame_count, 1)

        dispatch(torch.randn(64, 4))
        self.assertEqual(cnt_small.frame_count, 2)
        self.assertEqual(cnt_large.frame_count, 2)

        dispatch(torch.randn(4, 8))
        dispatch(torch.randn(32, 8))
        self.assertEqual(cnt_small.frame_count, 2)
        self.assertEqual(cnt_large.frame_count, 2)

        x = torch.randn(4, 8)
        self.assertEqual(dispatch(x), x.sin() + x.cos())


    def test_per_region_cache_counts(self):
        """Each region accumulates entries independently; total is the sum."""
        cnt1 = torch._dynamo.testing.CompileCounter()
        cnt2 = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + y

        opt1 = torch.compile(f, backend=cnt1, isolated_region=True)
        opt2 = torch.compile(f, backend=cnt2, isolated_region=True)

        opt1(torch.randn(3), torch.randn(3))
        opt1(
            torch.randn(3, dtype=torch.float64),
            torch.randn(3, dtype=torch.float64),
        )
        self.assertEqual(cnt1.frame_count, 2)

        opt2(
            torch.randn(3, dtype=torch.float16),
            torch.randn(3, dtype=torch.float16),
        )
        self.assertEqual(cnt2.frame_count, 1)

        # Total cache entries across all regions = 3
        self.assertEqual(self._num_cache_entries(f), 3)

    def test_region_b_lookup_skips_region_a(self):
        """Region B lookup doesn't walk region A's entries at all."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt_a, isolated_region=True)
        opt_b = torch.compile(f, backend=cnt_b, isolated_region=True)

        # Region A compiles with float32
        opt_a(torch.randn(3))
        self.assertEqual(cnt_a.frame_count, 1)

        # Region B with same input compiles independently (doesn't reuse A)
        opt_b(torch.randn(3))
        self.assertEqual(cnt_b.frame_count, 1)

        # Cache hits — no new compilations
        opt_a(torch.randn(3))
        self.assertEqual(cnt_a.frame_count, 1)
        opt_b(torch.randn(3))
        self.assertEqual(cnt_b.frame_count, 1)

    @torch._dynamo.config.patch(recompile_limit=2)
    def test_per_region_recompile_limit(self):
        """Recompile limit is enforced per-region, not globally."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.cos()

        opt_a = torch.compile(f, backend=cnt_a, isolated_region=True)
        opt_b = torch.compile(f, backend=cnt_b, isolated_region=True)

        # Region A uses up its 2 compilations
        opt_a(torch.randn(3))
        opt_a(torch.randn(3, dtype=torch.float64))
        self.assertEqual(cnt_a.frame_count, 2)

        # Region A hits limit — no more compilations
        opt_a(torch.randn(3, dtype=torch.float16))
        self.assertEqual(cnt_a.frame_count, 2)

        # Region B can still compile (its own limit is independent)
        opt_b(torch.randn(3))
        opt_b(torch.randn(3, dtype=torch.float64))
        self.assertEqual(cnt_b.frame_count, 2)

    def test_non_isolated_entries_visible_to_isolated(self):
        """Non-isolated (region -1) cache entries are visible to isolated
        region lookups via the global fallback, provided the backend matches.
        This exercises the fallback path in lookup() that checks region -1
        when an isolated region has no hit in its own bucket."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.exp()

        # Compile without isolation — entry goes into region -1
        opt_global = torch.compile(f, backend=cnt)
        opt_global(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        # Compile with isolation using the SAME backend — the isolated
        # region's bucket is empty, so lookup falls back to region -1
        # and finds the matching entry (same backend, guards pass).
        opt_isolated = torch.compile(f, backend=cnt, isolated_region=True)
        opt_isolated(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)  # cache hit from global fallback


    def test_isolated_region_lru_independent(self):
        """LRU reordering within one region doesn't affect another region's
        ordering. Repeated hits in region A should not cause region B to
        recompile."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt_a, isolated_region=True)
        opt_b = torch.compile(f, backend=cnt_b, isolated_region=True)

        # Populate both regions
        opt_a(torch.randn(3))
        opt_a(torch.randn(3, dtype=torch.float64))
        opt_b(torch.randn(4))
        opt_b(torch.randn(4, dtype=torch.float64))
        self.assertEqual(cnt_a.frame_count, 2)
        self.assertEqual(cnt_b.frame_count, 2)

        # Repeatedly hit region A entries (triggers LRU move_to_front)
        for _ in range(5):
            opt_a(torch.randn(3))
            opt_a(torch.randn(3, dtype=torch.float64))

        # Region B should still get cache hits without recompilation
        opt_b(torch.randn(4))
        opt_b(torch.randn(4, dtype=torch.float64))
        self.assertEqual(cnt_a.frame_count, 2)
        self.assertEqual(cnt_b.frame_count, 2)

    def test_isolated_region_reset(self):
        """torch._dynamo.reset() clears all regions. After reset, both
        regions must recompile."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.cos()

        opt_a = torch.compile(f, backend=cnt_a, isolated_region=True)
        opt_b = torch.compile(f, backend=cnt_b, isolated_region=True)

        opt_a(torch.randn(3))
        opt_b(torch.randn(4))
        self.assertEqual(cnt_a.frame_count, 1)
        self.assertEqual(cnt_b.frame_count, 1)

        torch._dynamo.reset()

        # After reset, cache is cleared — both must recompile
        opt_a(torch.randn(3))
        opt_b(torch.randn(4))
        self.assertEqual(cnt_a.frame_count, 2)
        self.assertEqual(cnt_b.frame_count, 2)

    @torch._dynamo.config.patch(recompile_limit=2)
    def test_isolated_region_resume_function(self):
        """Resume functions from a graph break inside an isolated region
        inherit the region_id. Their cache entries land in the correct
        region bucket and respect the per-region recompile limit."""
        cnt = torch._dynamo.testing.CompileCounter()

        mode = {"value": "a"}

        def f(x):
            a = x.sin()
            print("graph break")
            if mode["value"] == "a":
                return a.cos()
            else:
                return a.tan()

        opt_f = torch.compile(f, backend=cnt, isolated_region=True)

        opt_f(torch.randn(4))
        frame_count_after_1 = cnt.frame_count

        mode["value"] = "b"
        opt_f(torch.randn(4))
        frame_count_after_2 = cnt.frame_count
        self.assertGreater(frame_count_after_2, frame_count_after_1)

        # Resume function has 2 entries now (= recompile_limit).
        # A third mode should NOT cause further recompilation.
        mode["value"] = "c"
        opt_f(torch.randn(4))
        self.assertEqual(cnt.frame_count, frame_count_after_2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
