# Owner(s): ["module: random"]

import torch
import torch._dynamo.testing
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPhiloxKeySplit(TestCase):
    def test_basic_shape_and_dtype(self, device):
        key = torch.random.key(42, device=device)
        splits = torch.random.split(key, 4)
        self.assertEqual(splits.shape, (4, 2))
        self.assertEqual(splits.dtype, torch.uint64)
        self.assertEqual(splits.device, key.device)

    def test_single_split(self, device):
        key = torch.random.key(42, device=device)
        splits = torch.random.split(key, 1)
        self.assertEqual(splits.shape, (1, 2))

    def test_large_num_splits(self, device):
        key = torch.random.key(42, device=device)
        splits = torch.random.split(key, 10000)
        self.assertEqual(splits.shape, (10000, 2))

    def test_determinism(self, device):
        key = torch.random.key(42, device=device)
        splits1 = torch.random.split(key, 8)
        splits2 = torch.random.split(key, 8)
        self.assertEqual(splits1, splits2)

    def test_all_keys_unique(self, device):
        key = torch.random.key(42, device=device)
        splits = torch.random.split(key, 100)
        unique_keys = torch.unique(splits, dim=0)
        self.assertEqual(unique_keys.shape[0], 100)

    def test_different_seeds_produce_different_outputs(self, device):
        key1 = torch.random.key(42, device=device)
        key2 = torch.random.key(43, device=device)
        splits1 = torch.random.split(key1, 4)
        splits2 = torch.random.split(key2, 4)
        self.assertNotEqual(splits1, splits2)

    def test_different_offsets_produce_different_outputs(self, device):
        key1 = torch.random.key(42, device=device)
        key2 = torch.random.fold_in(key1, 1)
        splits1 = torch.random.split(key1, 4)
        splits2 = torch.random.split(key2, 4)
        self.assertNotEqual(splits1, splits2)

    def test_batched(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 4)  # (4, 2)
        num_splits = 3
        batched = torch.random.split(keys, num_splits)  # (3, 4, 2)
        self.assertEqual(batched.shape, (num_splits, 4, 2))
        for k in range(4):
            individual = torch.random.split(keys[k], num_splits)
            for s in range(num_splits):
                self.assertEqual(batched[s][k], individual[s])

    def test_multi_batch(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 12).reshape(3, 4, 2)  # (3, 4, 2)
        num_splits = 5
        batched = torch.random.split(keys, num_splits)  # (5, 3, 4, 2)
        self.assertEqual(batched.shape, (num_splits, 3, 4, 2))
        for i in range(3):
            for j in range(4):
                individual = torch.random.split(keys[i][j], num_splits)
                for s in range(num_splits):
                    self.assertEqual(batched[s][i][j], individual[s])

    def test_error_wrong_shape(self, device):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            torch.random.split(key, 4)

    def test_error_wrong_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaises(RuntimeError):
            torch.random.split(key, 4)

    def test_error_wrong_device(self, device):
        key = torch.random.key(42)  # CPU key
        with self.assertRaises(RuntimeError):
            torch.random.split(key, 4)

    def test_error_invalid_num_splits(self, device):
        key = torch.random.key(42, device=device)
        with self.assertRaises(RuntimeError):
            torch.random.split(key, 0)
        with self.assertRaises(RuntimeError):
            torch.random.split(key, -1)

    def test_error_batched_last_dim_not_2(self, device):
        key = torch.tensor([[42, 0, 1], [43, 0, 1]], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            torch.random.split(key, 4)


instantiate_device_type_tests(TestPhiloxKeySplit, globals(), only_for=("cuda"))


class TestPhiloxKeyFoldIn(TestCase):
    def test_basic_shape_and_dtype(self, device):
        key = torch.random.key(42, device=device)
        result = torch.random.fold_in(key, 7)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.dtype, torch.uint64)
        self.assertEqual(result.device, key.device)

    def test_determinism(self, device):
        key = torch.random.key(42, device=device)
        result1 = torch.random.fold_in(key, 7)
        result2 = torch.random.fold_in(key, 7)
        self.assertEqual(result1, result2)

    def test_different_data_produces_different_outputs(self, device):
        key = torch.random.key(42, device=device)
        result1 = torch.random.fold_in(key, 0)
        result2 = torch.random.fold_in(key, 1)
        self.assertNotEqual(result1, result2)

    def test_consistency_with_split(self, device):
        key = torch.random.key(42, device=device)
        splits = torch.random.split(key, 10)
        for i in range(10):
            folded = torch.random.fold_in(key, i)
            self.assertEqual(folded, splits[i])

    def test_batched(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 4)  # (4, 2)
        data = 7
        batched = torch.random.fold_in(keys, data)  # (4, 2)
        self.assertEqual(batched.shape, (4, 2))
        for k in range(4):
            individual = torch.random.fold_in(keys[k], data)
            self.assertEqual(batched[k], individual)

    def test_multi_batch(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 12).reshape(3, 4, 2)  # (3, 4, 2)
        data = 7
        batched = torch.random.fold_in(keys, data)  # (3, 4, 2)
        self.assertEqual(batched.shape, (3, 4, 2))
        for i in range(3):
            for j in range(4):
                individual = torch.random.fold_in(keys[i][j], data)
                self.assertEqual(batched[i][j], individual)

    def test_error_wrong_shape(self, device):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            torch.random.fold_in(key, 0)

    def test_error_wrong_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaises(RuntimeError):
            torch.random.fold_in(key, 0)

    def test_error_wrong_device(self, device):
        key = torch.random.key(42)  # CPU key
        with self.assertRaises(RuntimeError):
            torch.random.fold_in(key, 0)

    def test_error_batched_last_dim_not_2(self, device):
        key = torch.tensor([[42, 0, 1], [43, 0, 1]], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            torch.random.fold_in(key, 0)


instantiate_device_type_tests(TestPhiloxKeyFoldIn, globals(), only_for=("cuda"))


class TestPhiloxCompile(TestCase):
    def test_split_aot_eager(self, device):
        key = torch.random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return torch.random.split(key, 4)

        self.assertEqual(f(key), torch.random.split(key, 4))

    def test_fold_in_aot_eager(self, device):
        key = torch.random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return torch.random.fold_in(key, 7)

        self.assertEqual(f(key), torch.random.fold_in(key, 7))


instantiate_device_type_tests(TestPhiloxCompile, globals(), only_for=("cuda"))


if __name__ == "__main__":
    run_tests()
