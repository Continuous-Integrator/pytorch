# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.testing
from torch._dynamo.dynamic_spec import IntSpec, IntSpecType, TensorSpec
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestIntSpecConstruction(TestCase):
    """Direct constructor and property access."""

    def test_static_with_value(self):
        s = IntSpec("x", type=IntSpecType.STATIC, value=42)
        self.assertEqual(s.name, "x")
        self.assertEqual(s.type, IntSpecType.STATIC)
        self.assertEqual(s.value, 42)
        self.assertIsNone(s.min)
        self.assertIsNone(s.max)

    def test_static_without_value(self):
        s = IntSpec("x", type=IntSpecType.STATIC)
        self.assertIsNone(s.value)

    def test_backed_with_bounds(self):
        s = IntSpec("batch", type=IntSpecType.BACKED, min=1, max=64)
        self.assertEqual(s.type, IntSpecType.BACKED)
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 64)
        self.assertIsNone(s.value)

    def test_backed_with_hint(self):
        s = IntSpec("b", type=IntSpecType.BACKED, backed_hint=32)
        self.assertEqual(s.backed_hint, 32)

    def test_unbacked_with_bounds(self):
        s = IntSpec("seq", type=IntSpecType.UNBACKED, min=1, max=2048)
        self.assertEqual(s.type, IntSpecType.UNBACKED)
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 2048)

    def test_unbacked_with_hint(self):
        s = IntSpec("seq", type=IntSpecType.UNBACKED, optimization_hint=512)
        self.assertEqual(s.optimization_hint, 512)

    def test_no_name(self):
        s = IntSpec(type=IntSpecType.STATIC, value=10)
        self.assertIsNone(s.name)

    def test_no_type_allowed(self):
        s = IntSpec("x")
        self.assertIsNone(s.type)


class TestIntSpecValidation(TestCase):
    """Type-parameter cross-validation."""

    def test_static_rejects_min(self):
        with self.assertRaisesRegex(ValueError, "min/max.*STATIC"):
            IntSpec(type=IntSpecType.STATIC, min=1)

    def test_static_rejects_max(self):
        with self.assertRaisesRegex(ValueError, "min/max.*STATIC"):
            IntSpec(type=IntSpecType.STATIC, max=100)

    def test_static_rejects_optimization_hint(self):
        with self.assertRaisesRegex(ValueError, "optimization_hint.*UNBACKED"):
            IntSpec(type=IntSpecType.STATIC, optimization_hint=10)

    def test_static_rejects_backed_hint(self):
        with self.assertRaisesRegex(ValueError, "backed_hint.*BACKED"):
            IntSpec(type=IntSpecType.STATIC, backed_hint=10)

    def test_backed_rejects_value(self):
        with self.assertRaisesRegex(ValueError, "value.*STATIC"):
            IntSpec(type=IntSpecType.BACKED, value=42)

    def test_backed_rejects_optimization_hint(self):
        with self.assertRaisesRegex(ValueError, "optimization_hint.*UNBACKED"):
            IntSpec(type=IntSpecType.BACKED, optimization_hint=10)

    def test_unbacked_rejects_value(self):
        with self.assertRaisesRegex(ValueError, "value.*STATIC"):
            IntSpec(type=IntSpecType.UNBACKED, value=42)

    def test_unbacked_rejects_backed_hint(self):
        with self.assertRaisesRegex(ValueError, "backed_hint.*BACKED"):
            IntSpec(type=IntSpecType.UNBACKED, backed_hint=10)

    def test_backed_min_greater_than_max(self):
        with self.assertRaisesRegex(ValueError, "min must be <= max"):
            IntSpec(type=IntSpecType.BACKED, min=100, max=1)

    def test_unbacked_min_greater_than_max(self):
        with self.assertRaisesRegex(ValueError, "min must be <= max"):
            IntSpec(type=IntSpecType.UNBACKED, min=100, max=1)


class TestIntSpecFluentAPI(TestCase):
    """Fluent builder methods: .static() / .backed() / .unbacked()."""

    def test_static(self):
        s = IntSpec("x").static(10)
        self.assertEqual(s.type, IntSpecType.STATIC)
        self.assertEqual(s.value, 10)

    def test_static_no_value(self):
        s = IntSpec().static()
        self.assertEqual(s.type, IntSpecType.STATIC)
        self.assertIsNone(s.value)

    def test_backed(self):
        s = IntSpec("batch").backed(min=1, max=64)
        self.assertEqual(s.type, IntSpecType.BACKED)
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 64)

    def test_backed_with_hint(self):
        s = IntSpec("b").backed(hint=32)
        self.assertEqual(s.backed_hint, 32)

    def test_unbacked(self):
        s = IntSpec("seq").unbacked(min=1, max=2048)
        self.assertEqual(s.type, IntSpecType.UNBACKED)
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 2048)

    def test_unbacked_with_hint(self):
        s = IntSpec("seq").unbacked(hint=512)
        self.assertEqual(s.optimization_hint, 512)

    def test_returns_self(self):
        s = IntSpec("x")
        self.assertIs(s.backed(min=1), s)

    def test_clears_previous_fields(self):
        s = IntSpec("x").backed(min=1, max=64)
        s.static(10)
        self.assertIsNone(s.min)
        self.assertIsNone(s.max)
        self.assertEqual(s.value, 10)

    def test_backed_rejects_bad_bounds(self):
        with self.assertRaisesRegex(ValueError, "min must be <= max"):
            IntSpec().backed(min=100, max=1)


class TestIntSpecReprEq(TestCase):
    """__repr__, __eq__, __hash__."""

    def test_repr_static(self):
        r = repr(IntSpec("x", type=IntSpecType.STATIC, value=10))
        self.assertIn("name='x'", r)
        self.assertIn("type=static", r)
        self.assertIn("value=10", r)

    def test_repr_backed(self):
        r = repr(IntSpec("b", type=IntSpecType.BACKED, min=1, max=64))
        self.assertIn("type=backed", r)
        self.assertIn("min=1", r)
        self.assertIn("max=64", r)

    def test_eq(self):
        a = IntSpec("x", type=IntSpecType.BACKED, min=1, max=64)
        b = IntSpec("x", type=IntSpecType.BACKED, min=1, max=64)
        self.assertEqual(a, b)

    def test_neq_different_type(self):
        self.assertNotEqual(
            IntSpec("x", type=IntSpecType.BACKED),
            IntSpec("x", type=IntSpecType.STATIC),
        )

    def test_neq_different_name(self):
        self.assertNotEqual(
            IntSpec("x", type=IntSpecType.BACKED),
            IntSpec("y", type=IntSpecType.BACKED),
        )

    def test_eq_not_intspec(self):
        self.assertNotEqual(IntSpec().static(1), 1)

    def test_hashable(self):
        a = IntSpec("x", type=IntSpecType.BACKED, min=1)
        b = IntSpec("x", type=IntSpecType.BACKED, min=1)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)


class TestTensorSpecConstruction(TestCase):
    """Construction and list-like interface."""

    def test_basic(self):
        ts = TensorSpec(3)
        self.assertEqual(ts.rank, 3)
        self.assertEqual(len(ts), 3)
        for spec in ts:
            self.assertIsNone(spec)

    def test_zero_rank(self):
        ts = TensorSpec(0)
        self.assertEqual(ts.rank, 0)
        self.assertEqual(len(ts), 0)

    def test_negative_rank(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            TensorSpec(-1)

    def test_from_list(self):
        specs = [IntSpec().static(10), None, IntSpec().backed(min=1)]
        ts = TensorSpec.from_list(specs)
        self.assertEqual(ts.rank, 3)
        self.assertEqual(ts[0], IntSpec().static(10))
        self.assertIsNone(ts[1])

    def test_getitem_setitem(self):
        ts = TensorSpec(2)
        spec = IntSpec("batch").backed(min=1)
        ts[0] = spec
        self.assertEqual(ts[0], spec)
        self.assertIsNone(ts[1])

    def test_set_fluent(self):
        ts = TensorSpec(3)
        result = ts.set(0, IntSpec().static(10))
        self.assertIs(result, ts)
        self.assertEqual(ts[0], IntSpec().static(10))

    def test_iter(self):
        ts = TensorSpec(2)
        ts[0] = IntSpec().static(5)
        items = list(ts)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0], IntSpec().static(5))
        self.assertIsNone(items[1])

    def test_index_out_of_range(self):
        ts = TensorSpec(2)
        with self.assertRaises(IndexError):
            ts[5]

    def test_sparse_set(self):
        ts = TensorSpec(4)
        ts.set(1, IntSpec("h").backed())
        ts.set(3, IntSpec("w").backed())
        self.assertIsNone(ts[0])
        self.assertIsNotNone(ts[1])
        self.assertIsNone(ts[2])
        self.assertIsNotNone(ts[3])


class TestTensorSpecReprEq(TestCase):
    """__repr__, __eq__, __hash__."""

    def test_repr(self):
        ts = TensorSpec(3).set(0, IntSpec("b").backed())
        r = repr(ts)
        self.assertIn("rank=3", r)
        self.assertIn("0:", r)

    def test_eq(self):
        a = TensorSpec(2).set(0, IntSpec().static(10))
        b = TensorSpec(2).set(0, IntSpec().static(10))
        self.assertEqual(a, b)

    def test_neq_different_rank(self):
        self.assertNotEqual(TensorSpec(2), TensorSpec(3))

    def test_neq_different_specs(self):
        a = TensorSpec(2).set(0, IntSpec().static(10))
        b = TensorSpec(2).set(0, IntSpec().static(20))
        self.assertNotEqual(a, b)


class TestTensorSpecCompile(TestCase):
    """torch.compile(dynamic_shapes=...) with TensorSpec."""

    def test_tensorspec_backed_dim(self):
        torch._dynamo.reset()
        ts = TensorSpec(2).set(0, IntSpec("batch").backed())
        fn = torch.compile(
            lambda x: x.sum(0),
            backend="eager",
            dynamic_shapes={"x": ts},
        )
        for n in [4, 8, 16]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x.sum(0))

    def test_tensorspec_mixed_dims(self):
        torch._dynamo.reset()
        ts = TensorSpec(2).set(0, IntSpec("batch").backed()).set(1, IntSpec().static())
        fn = torch.compile(
            lambda x: x + 1,
            backend="eager",
            dynamic_shapes={"x": ts},
        )
        for n in [4, 8, 16]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x + 1)

    def test_tensorspec_partial_spec(self):
        torch._dynamo.reset()
        ts = TensorSpec(2).set(0, IntSpec("batch").backed())
        fn = torch.compile(
            lambda x: x.sum(0),
            backend="eager",
            dynamic_shapes={"x": ts},
        )
        for n in [4, 8]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x.sum(0))

    @skipIfTorchDynamo("frame_count unreliable when dynamo traces the test")
    def test_tensorspec_no_recompile(self):
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()
        ts = TensorSpec(2).set(0, IntSpec("batch").backed())
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=cnt,
            dynamic_shapes={"x": ts},
        )
        for n in [4, 8, 16, 32]:
            fn(torch.randn(n, 3))
        self.assertEqual(cnt.frame_count, 1)


class TestIntSpecCompile(TestCase):
    """torch.compile(dynamic_shapes=...) with IntSpec — correctness and behavior."""

    def test_static_correctness(self):
        torch._dynamo.reset()
        fn = torch.compile(
            lambda x: x + 1,
            backend="eager",
            dynamic_shapes={"x": {0: IntSpec().static()}},
        )
        for n in [4, 8]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x + 1)

    def test_backed(self):
        torch._dynamo.reset()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend="eager",
            dynamic_shapes={"x": {0: IntSpec("batch").backed()}},
        )
        for n in [4, 8, 16, 32]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x.sum(0))

    def test_unbacked(self):
        torch._dynamo.reset()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend="eager",
            dynamic_shapes={"x": {0: IntSpec("batch").unbacked()}},
        )
        for n in [4, 8, 16]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x.sum(0))

    def test_list_form(self):
        torch._dynamo.reset()
        fn = torch.compile(
            lambda x: x + 1,
            backend="eager",
            dynamic_shapes={"x": [IntSpec("batch").backed(), IntSpec().static()]},
        )
        for n in [4, 8, 16]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x + 1)

    @skipIfTorchDynamo("frame_count unreliable when dynamo traces the test")
    def test_static_recompiles_per_shape(self):
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(
            lambda x: x + 1,
            backend=cnt,
            dynamic_shapes={"x": {0: IntSpec().static()}},
        )
        fn(torch.randn(4, 3))
        fn(torch.randn(8, 3))
        fn(torch.randn(4, 3))  # cache hit
        self.assertEqual(cnt.frame_count, 2)

    @skipIfTorchDynamo("frame_count unreliable when dynamo traces the test")
    def test_backed_no_recompile(self):
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=cnt,
            dynamic_shapes={"x": {0: IntSpec("batch").backed()}},
        )
        for n in [4, 8, 16, 32, 64]:
            fn(torch.randn(n, 3))
        self.assertEqual(cnt.frame_count, 1)

    @skipIfTorchDynamo("frame_count unreliable when dynamo traces the test")
    def test_unbacked_no_recompile(self):
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=cnt,
            dynamic_shapes={"x": {0: IntSpec("batch").unbacked()}},
        )
        for n in [4, 8, 16, 32]:
            fn(torch.randn(n, 3))
        self.assertEqual(cnt.frame_count, 1)


if __name__ == "__main__":
    run_tests()
