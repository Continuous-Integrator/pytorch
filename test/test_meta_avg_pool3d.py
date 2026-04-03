import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestAvgPool3dMeta(TestCase):
    def test_avg_pool3d_divisor_zero_error(self):
        """avg_pool3d should error when divisor_override=0."""
        x = torch.randn(1, 1, 4, 4, 4, device="meta")
        with self.assertRaisesRegex(RuntimeError, "divisor"):
            torch.nn.functional.avg_pool3d(x, kernel_size=2, divisor_override=0)

    def test_avg_pool3d_backward_divisor_zero_error(self):
        """avg_pool3d_backward should error when divisor_override=0."""
        grad_output = torch.randn(1, 1, 2, 2, 2, device="meta")
        with self.assertRaisesRegex(RuntimeError, "divisor"):
            torch.ops.aten.avg_pool3d_backward(
                grad_output,
                torch.randn(1, 1, 4, 4, 4, device="meta"),
                [2, 2, 2],
                [2, 2, 2],
                [0, 0, 0],
                True,
                True,
                0,
            )


if __name__ == "__main__":
    run_tests()
