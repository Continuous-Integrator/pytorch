import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPreluMeta(TestCase):
    def test_prelu_dtype_mismatch_error(self):
        """prelu should reject mismatched dtypes between input and weight."""
        x = torch.randn(3, 4, device="meta", dtype=torch.float32)
        weight = torch.randn(4, device="meta", dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, ""):
            torch.nn.functional.prelu(x, weight)

    def test_prelu_same_dtype_ok(self):
        """prelu should work when dtypes match."""
        x = torch.randn(3, 4, device="meta", dtype=torch.float32)
        weight = torch.randn(4, device="meta", dtype=torch.float32)
        result = torch.nn.functional.prelu(x, weight)
        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(result.dtype, torch.float32)


if __name__ == "__main__":
    run_tests()
