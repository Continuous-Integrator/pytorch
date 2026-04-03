import torch
from torch.testing._internal.common_utils import run_tests, TestCase

class TestMetaFill(TestCase):
    def test_fill_tensor_dim_check(self):
        """fill_ with a multi-dim value tensor should raise an error."""
        x = torch.randn(3, 4, device='meta')
        value = torch.randn(2, 3, device='meta')
        with self.assertRaisesRegex(RuntimeError, "0-dimension"):
            x.fill_(value)

    def test_fill_tensor_scalar_ok(self):
        """fill_ with a 0-dim value tensor should work fine."""
        x = torch.randn(3, 4, device='meta')
        value = torch.tensor(1.0, device='meta')
        result = x.fill_(value)
        self.assertEqual(result.shape, (3, 4))

    def test_fill_scalar_ok(self):
        """fill_ with a scalar should work fine."""
        x = torch.randn(3, 4, device='meta')
        result = x.fill_(1.0)
        self.assertEqual(result.shape, (3, 4))

if __name__ == "__main__":
    run_tests()
