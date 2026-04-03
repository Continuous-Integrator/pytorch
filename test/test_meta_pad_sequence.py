import torch
from torch._decomp import decompositions
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPadSequence(TestCase):
    def test_pad_sequence_left(self):
        """pad_sequence with padding_side='left' should pad from the left."""
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        result = torch.nn.utils.rnn.pad_sequence([a, b], batch_first=True, padding_side='left')
        expected = torch.tensor([[1, 2, 3], [0, 4, 5]])
        self.assertEqual(result, expected)

    def test_pad_sequence_left_meta(self):
        """pad_sequence with padding_side='left' should work on meta tensors."""
        a = torch.randn(3, 4, device='meta')
        b = torch.randn(5, 4, device='meta')
        result = torch.nn.utils.rnn.pad_sequence([a, b], batch_first=True, padding_side='left')
        self.assertEqual(result.shape, (2, 5, 4))

    def test_pad_sequence_decomp_left(self):
        """The decomposition should accept and correctly handle padding_side='left'."""
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        result = decompositions.pad_sequence(
            [a, b], batch_first=True, padding_value=0.0, padding_side='left'
        )
        expected = torch.tensor([[1, 2, 3], [0, 4, 5]])
        self.assertEqual(result, expected)

    def test_pad_sequence_decomp_right(self):
        """The decomposition should still work correctly with padding_side='right'."""
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        result = decompositions.pad_sequence(
            [a, b], batch_first=True, padding_value=0.0, padding_side='right'
        )
        expected = torch.tensor([[1, 2, 3], [4, 5, 0]])
        self.assertEqual(result, expected)

    def test_pad_sequence_decomp_default(self):
        """The decomposition should default to right padding."""
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        result = decompositions.pad_sequence(
            [a, b], batch_first=True, padding_value=0.0
        )
        expected = torch.tensor([[1, 2, 3], [4, 5, 0]])
        self.assertEqual(result, expected)

    def test_pad_sequence_decomp_left_not_batch_first(self):
        """The decomposition with padding_side='left' and batch_first=False."""
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5])
        result = decompositions.pad_sequence(
            [a, b], batch_first=False, padding_value=0.0, padding_side='left'
        )
        expected = torch.tensor([[1, 0], [2, 4], [3, 5]])
        self.assertEqual(result, expected)


if __name__ == "__main__":
    run_tests()
