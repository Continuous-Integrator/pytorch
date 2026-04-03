import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


class TestPaddedDenseToJagged(TestCase):
    def test_total_L_zero(self):
        """When total_L=0 is explicitly passed, should use 0 not an unbacked symint."""
        with FakeTensorMode():
            padded = torch.randn(2, 3, 4)
            offsets = [torch.tensor([0, 0, 0])]
            result = torch.ops.aten._padded_dense_to_jagged_forward(
                padded, offsets, total_L=0
            )
            # With total_L=0, output dim 0 should be exactly 0 (not a symint)
            self.assertEqual(result.shape[0], 0)

    def test_total_L_none(self):
        """When total_L=None, should still work (creates unbacked symint)."""
        shape_env = ShapeEnv(allow_dynamic_output_shape_ops=True)
        with FakeTensorMode(shape_env=shape_env):
            padded = torch.randn(2, 3, 4)
            offsets = [torch.tensor([0, 1, 3])]
            result = torch.ops.aten._padded_dense_to_jagged_forward(
                padded, offsets, total_L=None
            )
            self.assertEqual(len(result.shape), 2)


if __name__ == "__main__":
    run_tests()
