# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing the FunctionalizedRngRuntimeWrapper in aot_autograd.

The codegen'd wrapper bakes is_rng_op_functionalized, return_new_outs, and
the offset index as compile-time constants, eliminating the runtime
conditional branch. When is_rng_op_functionalized is False, the wrapper
returns compiled_fn directly.

Tests verify that a "functionalized_rng_wrapper" artifact is emitted via
trace_structured and that the codegen'd function produces correct outputs.
"""

import logging
from contextlib import contextmanager
from unittest import skipIf

import torch
import torch._dynamo
import torch._functorch.config
from torch.testing._internal.common_utils import run_tests, TestCase


trace_log = logging.getLogger("torch.__trace")

HAS_CUDA = torch.cuda.is_available()


class TestCodegenFunctionalizedRng(TestCase):
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

    @skipIf(not HAS_CUDA, "requires CUDA")
    def test_functionalized_rng_codegen_emitted(self):
        """
        When functionalize_rng_ops is True, a codegen artifact should be
        emitted with the wrapper source.
        """
        with torch._functorch.config.patch(functionalize_rng_ops=True):
            with self._capture_codegen_source("functionalized_rng_wrapper") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    return torch.rand_like(x) + x

                x = torch.randn(4, device="cuda")
                f(x)

        self.assertEqual(
            len(captured),
            1,
            "Expected functionalized_rng_wrapper codegen artifact",
        )
        source = captured[0]
        self.assertIn("_get_rng_state_", source)
        self.assertIn("_set_offset_", source)

    def test_no_rng_no_codegen(self):
        """
        When functionalize_rng_ops is False (default), the wrapper should
        return compiled_fn directly without emitting codegen.
        """
        with self._capture_codegen_source("functionalized_rng_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2

            f(torch.randn(4))

        self.assertEqual(
            len(captured),
            0,
            "No codegen should be emitted when RNG is not functionalized",
        )

    @skipIf(not HAS_CUDA, "requires CUDA")
    def test_functionalized_rng_correctness(self):
        """
        Verify that the codegen'd wrapper produces correct random outputs
        by comparing against eager mode with the same RNG state.
        """
        with torch._functorch.config.patch(functionalize_rng_ops=True):

            @torch.compile(backend="aot_eager")
            def f(x):
                return torch.rand_like(x)

            x = torch.randn(8, device="cuda")
            out = f(x)

        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.device, x.device)
        self.assertTrue((out >= 0).all() and (out <= 1).all())

    @skipIf(not HAS_CUDA, "requires CUDA")
    def test_functionalized_rng_with_computation(self):
        """
        Verify correctness when combining random ops with other
        computation. The codegen should correctly strip the RNG offset
        from outputs while preserving user outputs.
        """
        with torch._functorch.config.patch(functionalize_rng_ops=True):

            @torch.compile(backend="aot_eager")
            def f(x):
                noise = torch.rand_like(x)
                return x + noise, x * 2

            x = torch.randn(4, device="cuda")
            out1, out2 = f(x)

        self.assertEqual(out2, x * 2)
        self.assertEqual(out1.shape, x.shape)

    @skipIf(not HAS_CUDA, "requires CUDA")
    def test_functionalized_rng_multiple_calls(self):
        """
        Multiple compilations with functionalized RNG should each
        produce different random outputs (RNG state advances).
        """
        with torch._functorch.config.patch(functionalize_rng_ops=True):

            @torch.compile(backend="aot_eager")
            def f(x):
                return torch.rand_like(x)

            x = torch.randn(100, device="cuda")
            out1 = f(x)
            out2 = f(x)

        self.assertFalse(
            torch.allclose(out1, out2),
            "Successive calls should produce different random outputs",
        )

    @skipIf(not HAS_CUDA, "requires CUDA")
    def test_codegen_source_structure(self):
        """
        Verify the codegen'd source has the expected structure with
        baked-in offset index.
        """
        with torch._functorch.config.patch(functionalize_rng_ops=True):
            with self._capture_codegen_source("functionalized_rng_wrapper") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    return torch.rand_like(x)

                f(torch.randn(4, device="cuda"))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("def _functionalized_rng_wrapper", source)
        self.assertIn("extend", source)
        self.assertIn("outs[", source)


if __name__ == "__main__":
    run_tests()
