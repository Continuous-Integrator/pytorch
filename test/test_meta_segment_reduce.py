import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestSegmentReduce(TestCase):
    def test_segment_reduce_shape_with_extra_dims(self):
        """segment_reduce meta should use data's shape as base, replacing only the axis dim."""
        data = torch.randn(10, 5, device="meta")
        lengths = torch.tensor([3, 4, 3])  # 3 segments, sums to 10
        result = torch.segment_reduce(data, "sum", lengths=lengths, axis=0)
        self.assertEqual(result.shape, (3, 5))

    def test_segment_reduce_1d(self):
        """Basic 1D case."""
        data = torch.randn(10, device="meta")
        lengths = torch.tensor([3, 4, 3])
        result = torch.segment_reduce(data, "sum", lengths=lengths, axis=0)
        self.assertEqual(result.shape, (3,))

    def test_segment_reduce_offsets_shape(self):
        """segment_reduce with offsets should also use data's shape as base."""
        data = torch.randn(10, 5, device="meta")
        offsets = torch.tensor([0, 3, 7, 10])  # 3 segments
        result = torch.segment_reduce(data, "sum", offsets=offsets, axis=0)
        self.assertEqual(result.shape, (3, 5))

    def test_segment_reduce_2d_data_batched_lengths(self):
        """When data has dims before axis, meta must use data.shape[:axis] not lengths.shape[:axis].

        This is the key test that exposes the bug: the old meta used lengths.shape
        for the entire prefix, but C++ uses data.sizes() and only replaces the axis
        dim with the segment count from lengths.
        """
        data = torch.randn(10, 5, device="meta")
        # lengths shape (2, 3) with axis=1 means 2 batch dims and 3 segments
        # C++ would produce output_shape = data.sizes() = [10, 5], replace dim 1
        # with lengths.size(1) = 3 -> [10, 3]
        # Old meta would produce: lengths.shape + data.shape[2:] = (2, 3) + () = (2, 3)
        lengths = torch.ones(2, 3, dtype=torch.long)
        result = torch.segment_reduce(
            data, "sum", lengths=lengths, axis=1, unsafe=True
        )
        self.assertEqual(result.shape, (10, 3))

    def test_segment_reduce_offsets_2d_data(self):
        """Same shape mismatch test but with offsets instead of lengths."""
        data = torch.randn(10, 5, device="meta")
        # offsets shape (2, 4) with axis=1 means 2 batch dims and 3 segments
        # C++ would produce output_shape = data.sizes() = [10, 5], replace dim 1
        # with offsets.size(1)-1 = 3 -> [10, 3]
        # Old meta would produce: (2, 3) + () = (2, 3)
        offsets = torch.zeros(2, 4, dtype=torch.long)
        result = torch.segment_reduce(
            data, "sum", offsets=offsets, axis=1, unsafe=True
        )
        self.assertEqual(result.shape, (10, 3))


if __name__ == "__main__":
    run_tests()
