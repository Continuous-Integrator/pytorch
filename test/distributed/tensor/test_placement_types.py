# Owner(s): ["oncall: distributed"]
import copy
import itertools

import torch
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# Basic functionality test for Placement types.
class PlacementTypesTestCase(TestCase):
    def test_type_identification(self):
        shard = Shard(3)
        strided_shard = _StridedShard(dim=3, split_factor=7)
        partial_sum = Partial("sum")
        partial_max = Partial("max")
        replicate = Replicate()

        ident_tests = (
            (shard, True, False, False),
            (strided_shard, False, False, False),
            (partial_sum, False, True, False),
            (partial_max, False, True, False),
            (replicate, False, False, True),
        )
        for do_deepcopy in (False, True):
            for placement, is_shard, is_partial, is_replicate in ident_tests:
                if do_deepcopy:
                    placement = copy.deepcopy(placement)
                self.assertEqual(placement.is_shard(), is_shard)
                self.assertEqual(placement.is_partial(), is_partial)
                self.assertEqual(placement.is_replicate(), is_replicate)

    def test_equality(self):
        equivalence_classes = (
            (Shard(3),),
            (Shard(4),),
            (_StridedShard(dim=3, split_factor=1),),
            (_StridedShard(dim=3, split_factor=2),),
            (_StridedShard(dim=4, split_factor=9),),
            (Replicate(),),
            (Partial("sum"),),
            (Partial("max"),),
        )
        for eq_class in equivalence_classes:
            # Each item in the equivalence class should be equal to every other item in
            # its class.
            for lhs, rhs in itertools.product(eq_class, eq_class):
                self.assertEqual(lhs, rhs)

            # Each item in the equivalence class should not be equal to any item in any
            # other class.
            for other_class in equivalence_classes:
                if other_class is eq_class:
                    continue
                for lhs, rhs in itertools.product(eq_class, other_class):
                    self.assertNotEqual(lhs, rhs)

    def test_strided_shard_kwonly_argument(self):
        with self.assertRaises(TypeError):
            _StridedShard(3, 4)
        _StridedShard(3, split_factor=4)

    def test_select_split_tensor_matches_split_tensor(self):
        """
        Test that _select_split_tensor produces the same result as indexing
        into _split_tensor. This validates that any alternative implementation
        (e.g., the narrow-based SymInt path) matches the canonical _split_tensor.
        """
        # Test various tensor sizes and num_chunks combinations
        test_cases = [
            # (dim_size, num_chunks) - covers even splits, uneven splits, edge cases
            (16, 4),  # even split
            (17, 4),  # uneven split, last chunk smaller
            (15, 4),  # uneven split
            (7, 4),  # fewer elements than chunks would like
            (3, 4),  # very few elements
            (1, 4),  # single element
            (8, 1),  # single chunk
            (8, 8),  # one element per chunk
            (8, 16),  # more chunks than elements
        ]

        for dim in [0, 1]:
            shard = Shard(dim)
            for dim_size, num_chunks in test_cases:
                # Create a tensor with distinct values for easy debugging
                if dim == 0:
                    tensor = torch.arange(dim_size * 4).reshape(dim_size, 4)
                else:
                    tensor = torch.arange(4 * dim_size).reshape(4, dim_size)

                # Get ground truth from _split_tensor
                shards, _ = shard._split_tensor(
                    tensor, num_chunks, with_padding=False, contiguous=False
                )

                # Compare _select_split_tensor against _split_tensor for each index
                for idx in range(num_chunks):
                    selected = shard._select_split_tensor(
                        tensor,
                        num_chunks,
                        idx,
                        with_padding=False,
                        contiguous=False,
                        clone=False,
                    )
                    self.assertEqual(
                        selected,
                        shards[idx],
                        msg=f"Mismatch for dim={dim}, dim_size={dim_size}, "
                        f"num_chunks={num_chunks}, idx={idx}",
                    )

    def test_strided_shard_split_roundtrip(self):
        """Test that _StridedShard._split_tensor partitions all elements correctly."""
        for dim_size in [6, 12, 15, 31]:
            for split_factor in [2, 3, 4]:
                for num_chunks in [2, 3, 4]:
                    ss = _StridedShard(0, split_factor=split_factor)
                    tensor = torch.arange(dim_size)
                    shards, _ = ss._split_tensor(tensor, num_chunks, with_padding=False)
                    recovered = torch.cat(shards).sort().values
                    self.assertEqual(
                        recovered,
                        tensor,
                        msg=f"dim_size={dim_size}, sf={split_factor}, "
                        f"chunks={num_chunks}",
                    )
                    self.assertEqual(
                        sum(s.numel() for s in shards),
                        dim_size,
                        msg=f"element count mismatch: dim_size={dim_size}, "
                        f"sf={split_factor}, chunks={num_chunks}",
                    )

    def test_strided_shard_replicate_permutation(self):
        """Simulate _to_replicate_tensor's index_select logic without collectives.

        Verifies that the permutation algorithm correctly reconstructs the
        original tensor from padded all_gather output.
        """
        for dim_size in [6, 9, 12, 15, 31]:
            for split_factor in [2, 3]:
                for num_chunks in [2, 3, 4]:
                    ss = _StridedShard(0, split_factor=split_factor)
                    original = torch.arange(dim_size)
                    shards, pad_sizes = ss._split_tensor(
                        original, num_chunks, with_padding=True
                    )
                    # _split_tensor returns unpadded shards + pad_sizes;
                    # simulate pad + all_gather by padding each shard
                    max_chunk = max(s.size(0) for s in shards)
                    padded_shards = [
                        torch.nn.functional.pad(s, (0, max_chunk - s.size(0)))
                        for s in shards
                    ]
                    gathered = torch.cat(padded_shards)

                    # Build select_indices (mirrors _to_replicate_tensor)
                    indices = torch.arange(dim_size)
                    index_shards, _ = ss._split_tensor(
                        indices, num_chunks, with_padding=False
                    )
                    padded_pos = [
                        i * max_chunk + torch.arange(len(s))
                        for i, s in enumerate(index_shards)
                    ]
                    permutation = torch.cat(index_shards)
                    select_pos = torch.cat(padded_pos)
                    select_indices = select_pos.index_select(
                        0, torch.argsort(permutation)
                    )
                    recovered = gathered.index_select(0, select_indices)
                    self.assertEqual(
                        recovered,
                        original,
                        msg=f"dim_size={dim_size}, sf={split_factor}, "
                        f"chunks={num_chunks}",
                    )

    def test_select_split_tensor_symint_with_padding_raises(self):
        """
        Test that _select_split_tensor raises GuardOnDataDependentSymNode when
        called with a SymInt index and with_padding=True.

        This is expected because with_padding=True requires indexing into a
        Python list with the SymInt, which is not supported.
        """
        from torch.fx.experimental.symbolic_shapes import (
            GuardOnDataDependentSymNode,
            ShapeEnv,
        )

        shape_env = ShapeEnv()
        symint_index = shape_env.create_unbacked_symint()

        shard = Shard(0)
        tensor = torch.arange(16).reshape(4, 4)

        with self.assertRaises(GuardOnDataDependentSymNode):
            shard._select_split_tensor(
                tensor,
                num_chunks=4,
                index=symint_index,
                with_padding=True,
            )


if __name__ == "__main__":
    run_tests()
