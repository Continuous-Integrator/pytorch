# Owner(s): ["module: inductor"]
"""
CPU-specific algorithm selection tests.

This test file contains CPU-specific tests that don't require GPU,
particularly for oneDNN/MKL operations like qlinear.
"""

import functools
import unittest
from unittest.mock import patch

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch._inductor.select_algorithm as select_algorithm
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_quantization import skipIfNoONEDNN
from torch.testing._internal.common_utils import IS_LINUX, TEST_MKL


def patches(fn):
    """Decorator to enable max_autotune and clear counters for CPU tests."""
    def skip_cache(self, choices, name, key, benchmark, hint_override=None):
        if benchmark is None:
            return {}
        return benchmark(choices)

    for patcher in [
        dynamo_config.patch(verbose=True),
        inductor_config.patch(
            debug=True,
            max_autotune=True,
            epilogue_fusion=True,
            max_autotune_gemm_backends="CPP,ATEN",
        ),
        patch.object(select_algorithm, "VERIFY", dict(atol=1e-4, rtol=1e-4)),
        patch.object(select_algorithm.AlgorithmSelectorCache, "lookup", skip_cache),
    ]:
        fn = patcher(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        counters.clear()
        torch.manual_seed(12345)
        return fn(*args, **kwargs)

    return wrapped


class TestSelectAlgorithmCPU(TestCase):
    """
    Test cases for CPU-specific algorithm selection (e.g., oneDNN qlinear).
    These tests don't require GPU and can run on CPU-only systems.
    """

    @skipIfNoONEDNN
    @unittest.skipUnless(
        IS_LINUX and TEST_MKL and torch._C._has_mkldnn,
        "Requires Linux, MKL, and oneDNN",
    )
    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    def test_qlinear_unary_with_computed_buffer_x_zp(self):
        """
        Test that qlinear_pointwise (unary) lowering correctly handles x_zp and 
        x_scale when they are ComputedBuffers.

        """
        torch._dynamo.reset()

        class QLinearUnaryModule(torch.nn.Module):
            def __init__(self, N, K):
                super().__init__()
                qw = torch.randint(-128, 127, (N, K), dtype=torch.int8)
                self.qw_packed = torch.ops.onednn.qlinear_prepack(qw, None)
                self.w_scales = torch.full((N,), 0.05)
                self.w_zps = torch.zeros(N, dtype=torch.int32)
                self.bias = torch.randn(N, dtype=torch.float32)
                self.output_scale = 1.0
                self.output_zp = 0

            def forward(self, qx):
                x_zp = torch.full([], 128, dtype=torch.int32)
                x_scale = torch.full([], 0.1, dtype=torch.float32)

                return torch.ops.onednn.qlinear_pointwise.tensor(
                    qx,
                    x_scale,
                    x_zp,
                    self.qw_packed,
                    self.w_scales,
                    self.w_zps,
                    self.bias,
                    self.output_scale,
                    self.output_zp,
                    torch.float32,  # output_dtype
                    "none",  # post_op
                    [],  # unary_post_op_args
                    "",  # post_op_algo
                )

        batch_size, in_features, out_features = 32, 64, 32

        x = torch.randn(batch_size, in_features, dtype=torch.float32)
        x_scale_val = 0.1
        qx = torch.quantize_per_tensor(x, x_scale_val, 128, torch.quint8).int_repr()

        mod = QLinearUnaryModule(out_features, in_features).eval()
        
        compiled_mod = torch.compile(mod)
        result = compiled_mod(qx)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (batch_size, out_features))

        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)


    @skipIfNoONEDNN
    @unittest.skipUnless(
        IS_LINUX and TEST_MKL and torch._C._has_mkldnn,
        "Requires Linux, MKL, and oneDNN",
    )
    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    def test_qlinear_binary_with_computed_buffer_x_zp(self):
        """
        Test that qlinear_pointwise.binary lowering correctly handles x_zp and 
        x_scale when they are ComputedBuffers.
        """
        torch._dynamo.reset()

        class QLinearBinaryModule(torch.nn.Module):
            def __init__(self, N, K):
                super().__init__()
                qw = torch.randint(-128, 127, (N, K), dtype=torch.int8)
                self.qw_packed = torch.ops.onednn.qlinear_prepack(qw, None)
                self.w_scales = torch.full((N,), 0.05)
                self.w_zps = torch.zeros(N, dtype=torch.int32)
                self.bias = torch.randn(N, dtype=torch.float32)
                self.output_scale = 1.0
                self.output_zp = 0

            def forward(self, qx, other):
                x_zp = torch.full([], 128, dtype=torch.int32)
                x_scale = torch.full([], 0.1, dtype=torch.float32)

                return torch.ops.onednn.qlinear_pointwise.binary_tensor(
                    qx,
                    x_scale,
                    x_zp,
                    self.qw_packed,
                    self.w_scales,
                    self.w_zps,
                    other,  # other tensor for binary op
                    self.bias,
                    self.output_scale,
                    self.output_zp,
                    torch.float32,  # output_dtype
                    1.0,  # other_scale
                    0,  # other_zp
                    "add",  # binary_post_op
                    1.0,  # binary_alpha
                    "none",  # unary_post_op
                    [],  # unary_post_op_args
                    "",  # unary_post_op_algo
                )

        batch_size, in_features, out_features = 32, 64, 32

        x = torch.randn(batch_size, in_features, dtype=torch.float32)
        x_scale_val = 0.1
        qx = torch.quantize_per_tensor(x, x_scale_val, 128, torch.quint8).int_repr()

        other = torch.randn(batch_size, out_features, dtype=torch.float32)

        mod = QLinearBinaryModule(out_features, in_features).eval()
        
        compiled_mod = torch.compile(mod)
        result = compiled_mod(qx, other)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (batch_size, out_features))

        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

if __name__ == "__main__":
    run_tests()
