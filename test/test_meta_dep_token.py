import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMakeDepToken(TestCase):
    def test_make_dep_token_shape(self):
        """_make_dep_token should return a 0-dim scalar tensor, not a 1-dim [0] tensor."""
        result = torch.ops.aten._make_dep_token(device=torch.device("meta"))
        self.assertEqual(result.dim(), 0)
        self.assertEqual(result.shape, torch.Size([]))


if __name__ == "__main__":
    run_tests()
