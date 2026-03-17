# Owner(s): ["module: cuda"]

import unittest

import torch
import torch._dynamo.testing
from torch.testing._internal.common_utils import run_tests, TestCase

TEST_CUDA = torch.cuda.is_available()


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
class TestPhiloxKeySplit(TestCase):
    def test_basic_shape_and_dtype(self):
        key = torch.random.key(42, device="cuda")
        splits = torch.random.split(key, 4)
        self.assertEqual(splits.shape, (4, 2))
        self.assertEqual(splits.dtype, torch.uint64)
        self.assertEqual(splits.device, key.device)

    def test_single_split(self):
        key = torch.random.key(42, device="cuda")
        splits = torch.random.split(key, 1)
        self.assertEqual(splits.shape, (1, 2))

    def test_large_num_splits(self):
        key = torch.random.key(42, device="cuda")
        splits = torch.random.split(key, 10000)
        self.assertEqual(splits.shape, (10000, 2))

    def test_determinism(self):
        key = torch.random.key(42, device="cuda")
        splits1 = torch.random.split(key, 8)
        splits2 = torch.random.split(key, 8)
        self.assertEqual(splits1, splits2)

    def test_all_keys_unique(self):
        key = torch.random.key(42, device="cuda")
        splits = torch.random.split(key, 100)
        unique_keys = torch.unique(splits, dim=0)
        self.assertEqual(unique_keys.shape[0], 100)

    def test_different_seeds_produce_different_outputs(self):
        key1 = torch.random.key(42, device="cuda")
        key2 = torch.random.key(43, device="cuda")
        splits1 = torch.random.split(key1, 4)
        splits2 = torch.random.split(key2, 4)
        self.assertNotEqual(splits1, splits2)

    def test_different_offsets_produce_different_outputs(self):
        key1 = torch.random.key(42, device="cuda")
        key2 = torch.random.fold_in(key1, 1)
        splits1 = torch.random.split(key1, 4)
        splits2 = torch.random.split(key2, 4)
        self.assertNotEqual(splits1, splits2)

    def test_batched(self):
        key = torch.random.key(42, device="cuda")
        keys = torch.random.split(key, 4)  # (4, 2)
        num_splits = 3
        batched = torch.random.split(keys, num_splits)  # (3, 4, 2)
        self.assertEqual(batched.shape, (num_splits, 4, 2))
        for k in range(4):
            individual = torch.random.split(keys[k], num_splits)
            for s in range(num_splits):
                self.assertEqual(batched[s][k], individual[s])

    def test_multi_batch(self):
        key = torch.random.key(42, device="cuda")
        keys = torch.random.split(key, 12).reshape(3, 4, 2)  # (3, 4, 2)
        num_splits = 5
        batched = torch.random.split(keys, num_splits)  # (5, 3, 4, 2)
        self.assertEqual(batched.shape, (num_splits, 3, 4, 2))
        for i in range(3):
            for j in range(4):
                individual = torch.random.split(keys[i][j], num_splits)
                for s in range(num_splits):
                    self.assertEqual(batched[s][i][j], individual[s])

    def test_error_wrong_shape(self):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device="cuda")
        with self.assertRaises(RuntimeError):
            torch.random.split(key, 4)

    def test_error_wrong_dtype(self):
        key = torch.tensor([42, 0], dtype=torch.float32, device="cuda")
        with self.assertRaises(RuntimeError):
            torch.random.split(key, 4)

    def test_error_wrong_device(self):
        key = torch.random.key(42)  # CPU key
        with self.assertRaises(RuntimeError):
            torch.random.split(key, 4)

    def test_error_invalid_num_splits(self):
        key = torch.random.key(42, device="cuda")
        with self.assertRaises(RuntimeError):
            torch.random.split(key, 0)
        with self.assertRaises(RuntimeError):
            torch.random.split(key, -1)

    def test_error_batched_last_dim_not_2(self):
        key = torch.tensor([[42, 0, 1], [43, 0, 1]], dtype=torch.uint64, device="cuda")
        with self.assertRaises(RuntimeError):
            torch.random.split(key, 4)


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
class TestPhiloxKeyFoldIn(TestCase):
    def test_basic_shape_and_dtype(self):
        key = torch.random.key(42, device="cuda")
        result = torch.random.fold_in(key, 7)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.dtype, torch.uint64)
        self.assertEqual(result.device, key.device)

    def test_determinism(self):
        key = torch.random.key(42, device="cuda")
        result1 = torch.random.fold_in(key, 7)
        result2 = torch.random.fold_in(key, 7)
        self.assertEqual(result1, result2)

    def test_different_data_produces_different_outputs(self):
        key = torch.random.key(42, device="cuda")
        result1 = torch.random.fold_in(key, 0)
        result2 = torch.random.fold_in(key, 1)
        self.assertNotEqual(result1, result2)

    def test_consistency_with_split(self):
        key = torch.random.key(42, device="cuda")
        splits = torch.random.split(key, 10)
        for i in range(10):
            folded = torch.random.fold_in(key, i)
            self.assertEqual(folded, splits[i])

    def test_batched(self):
        key = torch.random.key(42, device="cuda")
        keys = torch.random.split(key, 4)  # (4, 2)
        data = 7
        batched = torch.random.fold_in(keys, data)  # (4, 2)
        self.assertEqual(batched.shape, (4, 2))
        for k in range(4):
            individual = torch.random.fold_in(keys[k], data)
            self.assertEqual(batched[k], individual)

    def test_multi_batch(self):
        key = torch.random.key(42, device="cuda")
        keys = torch.random.split(key, 12).reshape(3, 4, 2)  # (3, 4, 2)
        data = 7
        batched = torch.random.fold_in(keys, data)  # (3, 4, 2)
        self.assertEqual(batched.shape, (3, 4, 2))
        for i in range(3):
            for j in range(4):
                individual = torch.random.fold_in(keys[i][j], data)
                self.assertEqual(batched[i][j], individual)

    def test_error_wrong_shape(self):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device="cuda")
        with self.assertRaises(RuntimeError):
            torch.random.fold_in(key, 0)

    def test_error_wrong_dtype(self):
        key = torch.tensor([42, 0], dtype=torch.float32, device="cuda")
        with self.assertRaises(RuntimeError):
            torch.random.fold_in(key, 0)

    def test_error_wrong_device(self):
        key = torch.random.key(42)  # CPU key
        with self.assertRaises(RuntimeError):
            torch.random.fold_in(key, 0)

    def test_error_batched_last_dim_not_2(self):
        key = torch.tensor([[42, 0, 1], [43, 0, 1]], dtype=torch.uint64, device="cuda")
        with self.assertRaises(RuntimeError):
            torch.random.fold_in(key, 0)


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
class TestPhiloxCompile(TestCase):
    def test_split_aot_eager(self):
        key = torch.random.key(42, device="cuda")

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return torch.random.split(key, 4)

        self.assertEqual(f(key), torch.random.split(key, 4))

    def test_fold_in_aot_eager(self):
        key = torch.random.key(42, device="cuda")

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return torch.random.fold_in(key, 7)

        self.assertEqual(f(key), torch.random.fold_in(key, 7))


if __name__ == "__main__":
    run_tests()
