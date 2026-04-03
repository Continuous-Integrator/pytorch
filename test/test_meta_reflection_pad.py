import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestReflectionPad2dMeta(TestCase):
    def test_reflection_pad2d_channels_last(self):
        """reflection_pad2d should preserve channels_last memory format for 4D tensors."""
        x = torch.randn(1, 3, 4, 4, device="meta").to(
            memory_format=torch.channels_last
        )
        result = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")
        self.assertTrue(result.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(result.shape, (1, 3, 6, 6))

    def test_reflection_pad2d_contiguous(self):
        """reflection_pad2d should keep contiguous format for contiguous inputs."""
        x = torch.randn(1, 3, 4, 4, device="meta")
        result = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")
        self.assertTrue(result.is_contiguous())
        self.assertEqual(result.shape, (1, 3, 6, 6))


if __name__ == "__main__":
    run_tests()
