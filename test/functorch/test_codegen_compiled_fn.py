# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing the TensorAlias wrapping and non-differentiable
output marking in CompiledFunction.forward.

CompiledFunction.forward contains several loops over compile-time-known
index sets (metadata-only mutations, unsafe views, aliased outputs,
non-differentiable outputs). The codegen'd version emits targeted
statements only for the specific indices that need them, eliminating
the loop and conditional overhead.

Tests verify that a "compiled_fn_wrapper" artifact is emitted via
trace_structured.
"""

import importlib.util
import logging
from contextlib import contextmanager

import torch
import torch._functorch.config


_orig_find_spec = importlib.util.find_spec


def _no_numba_find_spec(name, *a, **kw):  # type: ignore[no-untyped-def]
    if name == "numba":
        return None
    return _orig_find_spec(name, *a, **kw)


importlib.util.find_spec = _no_numba_find_spec  # type: ignore[assignment]
from torch.testing._internal.common_utils import run_tests, TestCase  # noqa: E402


importlib.util.find_spec = _orig_find_spec  # type: ignore[assignment]


trace_log = logging.getLogger("torch.__trace")


class TestCodegenCompiledFn(TestCase):
    @contextmanager
    def _capture_codegen_source(self, artifact_name):
        """Capture codegen artifacts from the structured trace log."""
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

    def test_aliased_output_tensor_alias_wrapping(self):
        """
        Training mode: output that aliases input should get wrapped in
        TensorAlias inside CompiledFunction.forward. Codegen should emit
        the wrap for the specific output index.
        """
        with self._capture_codegen_source("compiled_fn_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2, x.view(-1)

            x = torch.randn(2, 3, requires_grad=True)
            out1, out2 = f(x)
            (out1.sum() + out2.sum()).backward()

        self.assertEqual(out1, x * 2)
        self.assertEqual(out2, x.view(-1))
        self.assertEqual(x.grad, torch.ones(2, 3) * 3)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected compiled_fn_wrapper codegen artifact to be emitted",
        )

    def test_non_differentiable_output_marking(self):
        """
        Output that doesn't require grad should be marked non-differentiable.
        Codegen should emit mark_non_differentiable for the specific index.
        """
        with self._capture_codegen_source("compiled_fn_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2, x.detach().clone()

            x = torch.randn(4, requires_grad=True)
            out1, out2 = f(x)
            out1.sum().backward()

        self.assertEqual(out1, x * 2)
        self.assertFalse(out2.requires_grad)
        self.assertEqual(x.grad, torch.ones(4) * 2)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected compiled_fn_wrapper codegen artifact to be emitted",
        )

    def test_metadata_mutation_tensor_alias_wrapping(self):
        """
        Training mode: metadata-only mutation (transpose_) should wrap
        the mutated input return in TensorAlias. Codegen should emit
        the wrap for the specific mutated input index.
        """
        with self._capture_codegen_source("compiled_fn_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.transpose_(0, 1)
                return x.contiguous() + 1

            # Use a non-leaf so transpose_ doesn't error on leaf requiring grad
            x = torch.randn(2, 3, requires_grad=True).clone()
            x.retain_grad()
            out = f(x)
            out.sum().backward()

        self.assertEqual(x.shape, (3, 2))
        self.assertIsNotNone(x.grad)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected compiled_fn_wrapper codegen artifact to be emitted",
        )


if __name__ == "__main__":
    run_tests()
