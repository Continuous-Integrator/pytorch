# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing the RuntimeWrapper orchestration in aot_autograd.

The codegen'd runtime wrapper collapses _RuntimeCompiledFnInvoker.run,
_RuntimeForwardEpilogue.capture_orig_inputs, increment_mutation_versions,
and finalize into a single generated function with all branches resolved
at compile time: trace_joint, detach indices, epilogue_args_idx, number
of mutated inputs, output arity, and dynamic dims are all baked in.

Tests verify that a "runtime_wrapper_orchestration" artifact is emitted
via trace_structured.
"""

import logging
from contextlib import contextmanager

import torch
import torch._dynamo
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


trace_log = logging.getLogger("torch.__trace")


class TestCodegenRuntimeWrapper(TestCase):
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

    def test_inference_simple(self):
        """
        Simple inference: no mutations, no aliases. Generated code should
        use the inference path (grad disabled) with empty orig_inputs.
        """
        with self._capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2

            x = torch.randn(4)
            out = f(x)

        self.assertEqual(out, x * 2)
        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("orig_inputs = {}", source)
        self.assertIn("torch._C._set_grad_enabled(False)", source)
        self.assertNotIn("_force_view_tracking_", source)

    def test_training_simple(self):
        """
        Simple training path: no mutations. Generated code should use
        the training path (enable_grad + force_view_tracking).
        """
        with self._capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2

            x = torch.randn(4, requires_grad=True)
            out = f(x)

        self.assertEqual(out, x * 2)
        out.sum().backward()
        self.assertEqual(x.grad, torch.full((4,), 2.0))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_force_view_tracking_", source)
        self.assertIn("torch.enable_grad()", source)

    def test_training_with_detach_indices(self):
        """
        Training path with inputs that should be detached. Generated code
        should contain inline detach calls for specific indices.
        """
        with self._capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                return x * y

            x = torch.randn(4, requires_grad=True)
            y = torch.randn(4)
            out = f(x, y)
            out.sum().backward()

        self.assertEqual(out, x * y)
        self.assertIsNotNone(x.grad)
        self.assertEqual(len(captured), 1)

    def test_inference_with_mutation(self):
        """
        Inference with input mutation. With keep_inference_input_mutations,
        mutations are kept in-graph so the runtime wrapper just increments
        versions (no runtime _apply_mutations_ needed).
        """
        with self._capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.add_(1)
                return x.clone()

            x = torch.randn(4)
            x_ref = x.clone()
            out = f(x)

        self.assertEqual(x, x_ref + 1)
        self.assertEqual(out, x_ref + 1)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_increment_version_", source)

    def test_inference_with_output_alias(self):
        """
        Inference with output aliased to input. Generated code should
        capture orig_inputs and call _replay_aliases_.
        """
        with self._capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x.view(-1)

            x = torch.randn(2, 3)
            out = f(x)

        self.assertEqual(out, x.view(-1))
        self.assertEqual(out.data_ptr(), x.data_ptr())

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_replay_aliases_", source)
        self.assertIn("orig_inputs = {0: args[0]}", source)

    def test_inference_with_mutation_and_alias(self):
        """
        Inference: input mutation + output alias. With
        keep_inference_input_mutations, mutations are in-graph. The
        runtime wrapper handles the alias replay.
        """
        with self._capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.add_(1)
                return x.view(-1)

            x = torch.randn(2, 3)
            x_ref = x.clone()
            out = f(x)

        self.assertEqual(x, x_ref + 1)
        self.assertEqual(out, (x_ref + 1).view(-1))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_replay_aliases_", source)

    def test_training_with_alias(self):
        """
        Training path with output alias and backward correctness.
        """
        with self._capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2, x.view(-1)

            x = torch.randn(2, 3, requires_grad=True)
            out1, out2 = f(x)

        self.assertEqual(out1, x * 2)
        self.assertEqual(out2, x.view(-1))

        out1.sum().backward()
        self.assertEqual(x.grad, torch.full((2, 3), 2.0))

        self.assertEqual(len(captured), 1)

    def test_multiple_inputs_mutation_version_increment(self):
        """
        Multiple inputs with mutations. Generated code should increment
        versions for all mutated inputs.
        """
        with self._capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                x.add_(1)
                y.mul_(2)
                return x + y

            x = torch.randn(4)
            y = torch.randn(4)
            x_ref, y_ref = x.clone(), y.clone()
            out = f(x, y)

        self.assertEqual(x, x_ref + 1)
        self.assertEqual(y, y_ref * 2)
        self.assertEqual(out, (x_ref + 1) + (y_ref * 2))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_increment_version_", source)

    def test_output_arity_validation_baked(self):
        """
        The expected output arity should be baked into the generated code
        as a constant, not computed at runtime.
        """
        with self._capture_codegen_source("runtime_wrapper_orchestration") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x + 1, x * 2, x - 1

            f(torch.randn(4))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("if len(all_outs) != 3:", source)

    @skipIfTorchDynamo("dynamo handles mutations in-graph")
    def test_split_index_baked(self):
        """
        When there are mutated inputs that produce runtime mutation
        indices, the split index between updated_inputs and fw_outs
        should be baked as a constant. Uses aot_function directly to
        avoid keep_inference_input_mutations.
        """
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        with self._capture_codegen_source("runtime_wrapper_orchestration") as captured:

            def f(x, y):
                x.add_(1)
                return y * 2

            compiled_f = aot_function(f, nop, keep_inference_input_mutations=False)
            compiled_f(torch.randn(4), torch.randn(4))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("updated_inputs = all_outs[:1]", source)
        self.assertIn("fw_outs = all_outs[1:]", source)
        self.assertIn("_apply_mutations_", source)

    @skipIfTorchDynamo("dynamo handles metadata mutations in-graph")
    def test_metadata_mutation(self):
        """
        Metadata-only mutation (transpose_). Verify the generated wrapper
        correctly applies metadata mutations.
        """
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        def f(x):
            x.transpose_(1, 0)
            return x + 1

        x = torch.randn(3, 4).add(0)
        compiled_f = aot_function(f, nop)
        out = compiled_f(x)

        self.assertEqual(x.shape, (4, 3))
        self.assertEqual(out.shape, (4, 3))


if __name__ == "__main__":
    run_tests()
