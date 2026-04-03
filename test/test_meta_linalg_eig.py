import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestLinalgEigMeta(TestCase):
    def test_linalg_eig_strides_meta(self):
        """linalg_eig eigenvectors should always be column-major (Fortran contiguous)."""
        A = torch.randn(3, 3, device='meta')
        eigenvalues, eigenvectors = torch.linalg.eig(A)
        # Eigenvectors should be Fortran-contiguous (column-major)
        # i.e., stride(0) == 1, stride(1) == nrows
        self.assertEqual(eigenvectors.stride(-2), 1)
        self.assertEqual(eigenvectors.stride(-1), 3)


if __name__ == "__main__":
    run_tests()
