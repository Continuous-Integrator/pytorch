# Owner(s): ["module: inductor"]
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch.testing._internal.inductor_utils import HAS_GPU, GPU_TYPE


class TestPhiloxRNG(TestCase):
    @unittest.skipIf(not HAS_GPU, "requires CUDA for Inductor RNG testing")
    def test_philox_strided_layout_alignment(self):
        """
        Verify RNG alignment for non-contiguous views. 
        Ensures philox_rand respects strides to avoid spatial misalignment.
        """
        def fn(x):
            # Slicing creates a view with non-standard physical strides
            y = x[::2, ::2]
            return torch.nn.functional.dropout(y, p=0.5, training=True)

        # Use GPU_TYPE from internal utils to respect standard test infrastructure
        x = torch.randn(20, 20, device=GPU_TYPE)

        # 1. Eager ground truth (runs normally without Inductor)
        torch.manual_seed(42)
        expected = fn(x.clone())

        # 2. Inductor compiled version MUST force fallback_random=False
        # This ensures it hits your new native FixedLayout compiler logic.
        with config.patch(fallback_random=False):
            torch.manual_seed(42)
            actual = torch.compile(fn)(x.clone())

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # Only run the test suite if a valid GPU environment is present
    if HAS_GPU:
        run_tests()