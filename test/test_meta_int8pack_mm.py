import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestWeightInt8PackMM(TestCase):
    def test_inner_dim_mismatch(self):
        """Should error when B's inner dim doesn't match A's."""
        A = torch.randn(4, 8, device="meta")
        B = torch.randint(-128, 127, (3, 16), device="meta", dtype=torch.int8)
        scales = torch.randn(3, device="meta")
        with self.assertRaises(RuntimeError):
            torch.ops.aten._weight_int8pack_mm(A, B, scales)

    def test_scales_shape_mismatch(self):
        """Should error when scales doesn't match B's output dim."""
        A = torch.randn(4, 8, device="meta")
        B = torch.randint(-128, 127, (3, 8), device="meta", dtype=torch.int8)
        scales = torch.randn(5, device="meta")
        with self.assertRaises(RuntimeError):
            torch.ops.aten._weight_int8pack_mm(A, B, scales)

    def test_valid_inputs(self):
        """Should work with valid inputs."""
        A = torch.randn(4, 8, device="meta")
        B = torch.randint(-128, 127, (3, 8), device="meta", dtype=torch.int8)
        scales = torch.randn(3, device="meta")
        result = torch.ops.aten._weight_int8pack_mm(A, B, scales)
        self.assertEqual(result.shape, (4, 3))


if __name__ == "__main__":
    run_tests()
