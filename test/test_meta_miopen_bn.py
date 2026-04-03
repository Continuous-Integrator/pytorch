import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMiopenBatchNormMeta(TestCase):
    def test_miopen_batch_norm_save_dtype(self):
        """save_mean and save_var should use weight's dtype, not input's."""
        input_t = torch.randn(2, 3, 4, 4, device="meta", dtype=torch.float16)
        weight = torch.randn(3, device="meta", dtype=torch.float32)
        bias = torch.randn(3, device="meta", dtype=torch.float32)
        running_mean = torch.randn(3, device="meta", dtype=torch.float32)
        running_var = torch.randn(3, device="meta", dtype=torch.float32)

        result = torch.ops.aten.miopen_batch_norm(
            input_t, weight, bias, running_mean, running_var, True, 0.1, 1e-5
        )
        output, save_mean, save_var = result
        # save_mean and save_var should be float32 (weight's dtype), not float16
        self.assertEqual(save_mean.dtype, torch.float32)
        self.assertEqual(save_var.dtype, torch.float32)


if __name__ == "__main__":
    run_tests()
