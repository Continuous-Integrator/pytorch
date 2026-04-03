import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestRreluBackward(TestCase):
    def test_rrelu_backward_small_range(self):
        """When upper-lower is very small but training=True, should still use noise path."""
        x = torch.randn(5, requires_grad=True)
        lower, upper = 0.125, 0.125 + 1e-7  # very small gap
        noise = torch.rand(5)
        grad = torch.ones(5)

        # C++ always uses noise * grad when training=True
        expected = noise * grad

        # The decomposition should match
        from torch._decomp.decompositions import rrelu_with_noise_backward
        result = rrelu_with_noise_backward(grad, x, noise, lower, upper, True, False)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    run_tests()
