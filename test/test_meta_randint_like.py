import torch
from torch.testing._internal.common_utils import run_tests, TestCase

aten = torch.ops.aten


def fixed_meta_randint_like(self, high, **kwargs):
    """The fixed implementation that respects kwargs."""
    return aten.empty_like.default(
        self,
        dtype=kwargs.get("dtype"),
        layout=kwargs.get("layout"),
        device=kwargs.get("device"),
        pin_memory=kwargs.get("pin_memory"),
        memory_format=kwargs.get("memory_format"),
    )


def buggy_meta_randint_like(self, high, **kwargs):
    """The original buggy implementation that ignores kwargs."""
    return self.new_empty(self.size())


class TestMetaRandintLike(TestCase):
    def test_buggy_ignores_dtype_kwarg(self):
        """Demonstrate the bug: the original meta ignores dtype kwarg."""
        x = torch.randn(3, 4, device="meta")
        high = torch.tensor(10)
        y = buggy_meta_randint_like(x, high, dtype=torch.int32)
        # Bug: returns float32 (self's dtype) instead of int32
        self.assertEqual(y.dtype, torch.float32)

    def test_fixed_respects_dtype_kwarg(self):
        """The fixed meta should respect dtype kwarg."""
        x = torch.randn(3, 4, device="meta")
        high = torch.tensor(10)
        y = fixed_meta_randint_like(x, high, dtype=torch.int32)
        self.assertEqual(y.dtype, torch.int32)

    def test_fixed_preserves_default(self):
        """Without kwargs, the fixed meta should preserve self's properties."""
        x = torch.randn(3, 4, device="meta", dtype=torch.float16)
        high = torch.tensor(10)
        y = fixed_meta_randint_like(x, high)
        self.assertEqual(y.dtype, torch.float16)
        self.assertEqual(y.shape, (3, 4))

    def test_randint_like_high_level_dtype_kwarg(self):
        """High-level randint_like should respect dtype kwarg."""
        x = torch.randn(3, 4, device="meta")
        y = torch.randint_like(x, 0, 10, dtype=torch.int32)
        self.assertEqual(y.dtype, torch.int32)

    def test_randint_like_high_level_preserves_default(self):
        """Without kwargs, should preserve self's properties."""
        x = torch.randn(3, 4, device="meta", dtype=torch.float16)
        y = torch.randint_like(x, 0, 10)
        self.assertEqual(y.dtype, torch.float16)
        self.assertEqual(y.shape, (3, 4))


if __name__ == "__main__":
    run_tests()
