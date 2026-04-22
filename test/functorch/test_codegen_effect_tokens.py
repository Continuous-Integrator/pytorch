# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing the EffectTokensWrapper in aot_autograd.

The codegen'd effect tokens wrapper inlines the number of tokens as a
compile-time constant, eliminating runtime conditionals. When num_tokens > 0,
the generated function prepends the exact number of None tokens and slices
the outputs with baked-in indices.

Tests verify that an "effect_tokens_wrapper" artifact is emitted via
trace_structured.
"""

import logging
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch._dynamo
from torch._higher_order_ops.effects import _register_effectful_op
from torch._library.effects import EffectType
from torch.testing._internal.common_utils import run_tests, TestCase


trace_log = logging.getLogger("torch.__trace")


def _make_effectful_op(name):
    @torch.library.custom_op(f"test::{name}", mutates_args=())
    def op(x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    @op.register_fake
    def _(x):
        return torch.empty_like(x)

    return op


class TestCodegenEffectTokens(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @contextmanager
    def _capture_codegen_source(self, artifact_name):
        captured: list[str] = []

        class _ArtifactHandler(logging.Handler):
            def emit(self, record):
                metadata = getattr(record, "metadata", {})
                if (
                    "artifact" in metadata
                    and metadata["artifact"].get("name") == artifact_name
                ):
                    payload = getattr(record, "payload", None)
                    if payload is not None:
                        captured.append(payload)

        handler = _ArtifactHandler()
        handler.setLevel(logging.DEBUG)
        old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)
        trace_log.addHandler(handler)
        try:
            yield captured
        finally:
            trace_log.removeHandler(handler)
            trace_log.setLevel(old_level)

    def test_single_effect_token(self):
        """
        Single effect token from one effectful op. Codegen should emit
        a wrapper that prepends one None and strips one token output.
        """
        op = _make_effectful_op("single_effect")
        handle = _register_effectful_op(op, EffectType.ORDERED)
        try:
            with self._capture_codegen_source("effect_tokens_wrapper") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    return torch.ops.test.single_effect(x)

                x = torch.randn(4)
                out = f(x)

            self.assertEqual(out, x)
            self.assertEqual(
                len(captured),
                1,
                "Expected effect_tokens_wrapper codegen artifact to be emitted",
            )
            source = captured[0]
            self.assertIn("None", source)
            self.assertIn("outs[1:]", source)
        finally:
            handle.destroy()

    def test_no_effect_tokens_no_codegen(self):
        """
        When there are no effect tokens, EffectTokensWrapper should
        return compiled_fn directly without emitting codegen.
        """
        with self._capture_codegen_source("effect_tokens_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2

            x = torch.randn(4)
            out = f(x)

        self.assertEqual(out, x * 2)
        self.assertEqual(
            len(captured),
            0,
            "No codegen artifact should be emitted when there are no effect tokens",
        )

    def test_effect_token_correctness(self):
        """
        Verify that the codegen'd wrapper produces correct results for
        a function with an effectful op.
        """
        op = _make_effectful_op("correctness_effect")
        handle = _register_effectful_op(op, EffectType.ORDERED)
        try:

            @torch.compile(backend="aot_eager")
            def f(x):
                y = torch.ops.test.correctness_effect(x)
                return y + 1

            x = torch.randn(3, 3)
            out = f(x)
            self.assertEqual(out, x + 1)
        finally:
            handle.destroy()

    def test_effect_token_with_mutation(self):
        """
        Effect tokens combined with input mutation. Both the effect
        token wrapping and mutation epilogue should work correctly.
        """
        op = _make_effectful_op("mutation_effect")
        handle = _register_effectful_op(op, EffectType.ORDERED)
        try:

            @torch.compile(backend="aot_eager")
            def f(x):
                y = torch.ops.test.mutation_effect(x)
                x.add_(1)
                return y

            x = torch.randn(4)
            x_ref = x.clone()
            out = f(x)

            self.assertEqual(out, x_ref)
            self.assertEqual(x, x_ref + 1)
        finally:
            handle.destroy()

    def test_effect_token_with_multiple_outputs(self):
        """
        Effect tokens with multiple outputs. The codegen should
        correctly strip token outputs while preserving user outputs.
        """
        op = _make_effectful_op("multi_out_effect")
        handle = _register_effectful_op(op, EffectType.ORDERED)
        try:

            @torch.compile(backend="aot_eager")
            def f(x):
                y = torch.ops.test.multi_out_effect(x)
                return y, x * 2, x + 1

            x = torch.randn(4)
            out1, out2, out3 = f(x)

            self.assertEqual(out1, x)
            self.assertEqual(out2, x * 2)
            self.assertEqual(out3, x + 1)
        finally:
            handle.destroy()

    def test_effect_token_training_path(self):
        """
        Effect tokens on the training path (requires_grad=True).
        Verify backward correctness with the codegen'd wrapper.
        """
        op = _make_effectful_op("training_effect")

        def setup_context(ctx, inputs, output):
            pass

        def backward(ctx, grad):
            return grad

        op.register_autograd(backward, setup_context=setup_context)
        handle = _register_effectful_op(op, EffectType.ORDERED)
        try:

            @torch.compile(backend="aot_eager")
            def f(x):
                y = torch.ops.test.training_effect(x)
                return y * 2

            x = torch.randn(3, requires_grad=True)
            out = f(x)
            out.sum().backward()

            self.assertEqual(out, x * 2)
            self.assertEqual(x.grad, torch.full((3,), 2.0))
        finally:
            handle.destroy()

    def _make_wrapper_via_post_compile(self, compiled_fn, num_tokens):
        """
        Call EffectTokensWrapper.post_compile with a mock runtime_metadata
        that has the given number of tokens, exercising the production
        codegen path.
        """
        from torch._functorch._aot_autograd.runtime_wrappers import (
            EffectTokensWrapper,
        )

        mock_meta = SimpleNamespace(tokens={i: None for i in range(num_tokens)})
        mock_aot_config = SimpleNamespace()
        return EffectTokensWrapper().post_compile(
            compiled_fn, mock_aot_config, runtime_metadata=mock_meta
        )

    def test_multiple_effect_tokens_codegen(self):
        """
        Verify the codegen template works for num_tokens > 1 by calling
        EffectTokensWrapper.post_compile directly. Currently only one
        EffectType (ORDERED) exists, so num_tokens > 1 can't arise
        through normal compilation, but the template should handle it
        correctly for future expansion.
        """
        call_log = []

        def mock_compiled_fn(args):
            call_log.append(list(args))
            return list(range(len(args)))

        wrapper = self._make_wrapper_via_post_compile(mock_compiled_fn, num_tokens=2)

        args = [10, 20, 30]
        result = wrapper(args)
        self.assertEqual(len(call_log), 1)
        self.assertEqual(call_log[0], [None, None, 10, 20, 30])
        self.assertEqual(args, [])
        self.assertEqual(list(result), [2, 3, 4])

    def test_multiple_effect_tokens_codegen_none_output(self):
        """
        Verify the codegen template for num_tokens > 1 handles None
        output (Inductor cache DummyModule can return None).
        """

        def mock_compiled_fn(args):
            return None

        wrapper = self._make_wrapper_via_post_compile(mock_compiled_fn, num_tokens=2)

        result = wrapper([10, 20])
        self.assertIsNone(result)

    def test_codegen_source_structure(self):
        """
        Verify the generated source has the expected structure:
        function definition, None prepend, compiled_fn call,
        None check, and output slicing.
        """
        op = _make_effectful_op("structure_effect")
        handle = _register_effectful_op(op, EffectType.ORDERED)
        try:
            with self._capture_codegen_source("effect_tokens_wrapper") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    return torch.ops.test.structure_effect(x)

                f(torch.randn(4))

            self.assertEqual(len(captured), 1)
            source = captured[0]
            self.assertIn("def _effect_tokens_wrapper", source)
            self.assertIn("_compiled_fn_", source)
            self.assertIn("args.clear()", source)
            self.assertIn("if outs is None:", source)
            self.assertIn("return None", source)
        finally:
            handle.destroy()


if __name__ == "__main__":
    run_tests()
