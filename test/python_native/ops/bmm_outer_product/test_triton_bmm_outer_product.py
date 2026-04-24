# Owner(s): ["module: dsl-native-ops"]
"""
Tests for Triton BMM Outer Product implementation using OpInfo with DSL support.

This module provides an OpInfo definition for testing the Triton-optimized
batch matrix multiplication implementation specifically for outer product cases.
The op is non-elementwise (shape-contracting), so it uses plain OpInfo rather
than BinaryUfuncInfo — that way TestCommon drives it via op.sample_inputs(...)
and op.ref instead of TestBinaryUfuncs' elementwise shape generators.
"""

import unittest
from functools import partial

import numpy as np

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    onlyCUDA,
    skipCUDAIf,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.opinfo.core import DecorateInfo, OpInfo, SampleInput


def sample_inputs_triton_bmm_outer_product(
    op_info, device, dtype, requires_grad, **kwargs
):
    """
    Generate sample inputs for Triton BMM outer product testing.

    This generates inputs that match the outer product pattern:
    - a: shape (B, M, 1) - batch of column vectors
    - b: shape (B, 1, N) - batch of row vectors
    - output: (B, M, N) - batch of outer products
    """
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # Test various batch sizes and dimensions
    test_configs = [
        (2, 16, 32),  # Small: B=2, M=16, N=32
        (4, 64, 64),  # Medium: B=4, M=64, N=64
        (8, 128, 256),  # Large: B=8, M=128, N=256
        (1, 512, 128),  # Single batch, large dims
        (16, 32, 64),  # Many batches
        (3, 7, 11),  # Non-power-of-2 dimensions
    ]

    for batch_size, m_dim, n_dim in test_configs:
        # Create tensors with outer product pattern
        # a: (B, M, 1) - batch of column vectors
        a = make_arg((batch_size, m_dim, 1)).contiguous()

        # b: (B, 1, N) - batch of row vectors
        b = make_arg((batch_size, 1, n_dim)).contiguous()

        yield SampleInput(a, args=(b,))

        # Test with different value ranges
        yield SampleInput(
            make_arg((batch_size, m_dim, 1), low=-2.0, high=2.0).contiguous(),
            args=(make_arg((batch_size, 1, n_dim), low=-2.0, high=2.0).contiguous(),),
        )

        # Test with small positive values
        yield SampleInput(
            make_arg((batch_size, m_dim, 1), low=0.1, high=1.0).contiguous(),
            args=(make_arg((batch_size, 1, n_dim), low=0.1, high=1.0).contiguous(),),
        )


def reference_bmm_outer_product(a, b):
    """NumPy reference; inputs are ndarrays because TestCommon.test_numpy_ref
    converts SampleInputs via SampleInput.numpy() before calling this."""
    return np.matmul(a, b)


# Import the Triton kernel and availability check
try:
    from torch._native.ops.bmm_outer_product.triton_kernels import (
        bmm_outer_product as triton_bmm_outer_product_kernel,
    )

    HAS_TRITON_BMM = True
except ImportError:
    # Create a dummy function if Triton is not available
    def triton_bmm_outer_product_kernel(a, b):
        return torch.ops.aten.bmm(a, b)

    HAS_TRITON_BMM = False


def verify_outer_product_conditions(a, b):
    """
    Verify if tensors meet the conditions for Triton BMM outer product optimization.

    Returns True if Triton should be used, False otherwise.
    """
    return (
        a.ndim == 3
        and b.ndim == 3
        and a.shape[2] == 1  # a is (B, M, 1)
        and b.shape[1] == 1  # b is (B, 1, N)
        and a.numel() > 0
        and b.numel() > 0
        and not a.is_complex()
        and not b.is_complex()
        and a.is_cuda
        and b.is_cuda
        and a.is_contiguous()
        and b.is_contiguous()
    )


# Plain OpInfo (not BinaryUfuncInfo) because this op is shape-contracting rather
# than elementwise. TestCommon's numerics tests iterate op.sample_inputs(...) so
# the custom sample generator drives the run; TestBinaryUfuncs is bypassed.
triton_bmm_outer_product_opinfo = OpInfo(
    "triton_bmm_outer_product",
    op=triton_bmm_outer_product_kernel,
    dsl_name="triton",
    dtypes=(),  # CPU unsupported; Triton is CUDA-only
    dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    sample_inputs_func=sample_inputs_triton_bmm_outer_product,
    ref=reference_bmm_outer_product,
    supports_autograd=False,
    supports_forward_ad=False,
    supports_fwgrad_bwgrad=False,
    supports_out=False,
    inplace_variant=None,
    # Direct Triton kernel launches don't interoperate with tensor subclasses
    # (CompositeCompliantTensor, COW wrapper) or FakeTensor.
    supports_cow_input_no_materialize_forward=False,
    decorators=[
        DecorateInfo(
            toleranceOverride(
                {
                    torch.float32: tol(atol=1e-5, rtol=1e-5),
                    torch.float64: tol(atol=1e-8, rtol=1e-8),
                    torch.float16: tol(atol=1e-3, rtol=1e-3),
                    torch.bfloat16: tol(atol=1e-2, rtol=1e-2),
                }
            ),
            device_type="cuda",
        ),
        DecorateInfo(onlyCUDA),
    ],
    skips=(
        DecorateInfo(
            skipCUDAIf(not torch.cuda.is_available(), "CUDA not available"),
        ),
        DecorateInfo(
            unittest.skipIf(
                not HAS_TRITON_BMM, "Triton BMM outer product not available"
            ),
        ),
        # noncontiguous_like() on (B,M,1)/(B,1,N) breaks the shape contract the
        # kernel requires, so skip the generic non-contig test.
        DecorateInfo(
            unittest.skip("Outer-product kernel requires contiguous (B,M,1)/(B,1,N)"),
            "TestCommon",
            "test_noncontiguous_samples",
        ),
        # Triton kernels take raw pointers and can't dispatch through tensor
        # subclasses (CompositeCompliantTensor, CrossRefFakeMode, etc.).
        DecorateInfo(
            unittest.skip("Triton kernel incompatible with tensor subclasses"),
            "TestCompositeCompliance",
        ),
        DecorateInfo(
            unittest.skip("Triton kernel incompatible with FakeTensor"),
            "TestFakeTensor",
        ),
        DecorateInfo(
            unittest.skip("Triton kernel not introspectable for tag inference"),
            "TestTags",
        ),
        DecorateInfo(
            unittest.skip(
                "Triton kernel not introspectable for conjugate/negate views"
            ),
            "TestMathBits",
        ),
        # Sample generator only targets the primary CUDA device.
        DecorateInfo(
            unittest.skip("Sample generator allocates on the primary CUDA device"),
            "TestCommon",
            "test_multiple_devices",
        ),
    ),
    variant_test_name="triton_optimized",
)


def test_bmm_outer_product_conditions(self):
    """
    Test that Triton optimization conditions are properly detected.
    This is a custom test method for validating outer product pattern detection.
    """
    if not torch.cuda.is_available():
        self.skipTest("CUDA not available")

    device = torch.device("cuda")
    dtype = torch.float32

    # Test valid outer product pattern
    a_valid = make_tensor((4, 32, 1), device=device, dtype=dtype)
    b_valid = make_tensor((4, 1, 64), device=device, dtype=dtype)

    self.assertTrue(verify_outer_product_conditions(a_valid, b_valid))

    # Test invalid patterns
    # Wrong last dimension for a
    a_invalid = make_tensor((4, 32, 2), device=device, dtype=dtype)
    self.assertFalse(verify_outer_product_conditions(a_invalid, b_valid))

    # Wrong middle dimension for b
    b_invalid = make_tensor((4, 2, 64), device=device, dtype=dtype)
    self.assertFalse(verify_outer_product_conditions(a_valid, b_invalid))

    # Different batch sizes
    a_diff_batch = make_tensor((3, 32, 1), device=device, dtype=dtype)
    self.assertFalse(verify_outer_product_conditions(a_diff_batch, b_valid))


def create_outer_product_tensors(
    batch_size, m_dim, n_dim, device="cuda", dtype=torch.float32, **kwargs
):
    """
    Create tensors that meet Triton BMM outer product optimization conditions.
    """
    a = make_tensor((batch_size, m_dim, 1), device=device, dtype=dtype, **kwargs)
    b = make_tensor((batch_size, 1, n_dim), device=device, dtype=dtype, **kwargs)
    return a.contiguous(), b.contiguous()


if __name__ == "__main__":
    run_tests()


# Export the OpInfo for import by common_methods_invocations.py
__all__ = [
    "triton_bmm_outer_product_opinfo",
    "sample_inputs_triton_bmm_outer_product",
    "reference_bmm_outer_product",
    "verify_outer_product_conditions",
    "create_outer_product_tensors",
]
