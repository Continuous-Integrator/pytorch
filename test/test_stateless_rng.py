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
class TestPhiloxNormal(TestCase):
    def test_basic_shape_and_dtype(self):
        key = torch.random.key(42, device="cuda")
        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            result = torch.random.normal(key, (100,), dtype=dtype)
            self.assertEqual(result.shape, (100,))
            self.assertEqual(result.dtype, dtype)

    def test_determinism(self):
        key = torch.random.key(42, device="cuda")
        a = torch.random.normal(key, (1000,))
        b = torch.random.normal(key, (1000,))
        self.assertEqual(a, b)

    def test_different_keys(self):
        key1 = torch.random.key(42, device="cuda")
        key2 = torch.random.key(43, device="cuda")
        a = torch.random.normal(key1, (1000,))
        b = torch.random.normal(key2, (1000,))
        self.assertNotEqual(a, b)

    def test_standard_normal_statistics(self):
        key = torch.random.key(42, device="cuda")
        result = torch.random.normal(key, (100000,))
        self.assertTrue(abs(result.mean().item()) < 0.05)
        self.assertTrue(abs(result.std().item() - 1.0) < 0.05)

    def test_custom_mean_std(self):
        key = torch.random.key(42, device="cuda")
        result = torch.random.normal(key, (100000,), mean=5.0, std=2.0)
        self.assertTrue(abs(result.mean().item() - 5.0) < 0.1)
        self.assertTrue(abs(result.std().item() - 2.0) < 0.1)

    def test_batched_keys(self):
        key = torch.random.key(42, device="cuda")
        keys = torch.random.split(key, 4)  # (4, 2)
        result = torch.random.normal(keys, (4, 100))
        for i in range(4):
            individual = torch.random.normal(keys[i], (100,))
            self.assertEqual(result[i], individual)

    def test_multi_batch(self):
        key = torch.random.key(42, device="cuda")
        keys = torch.random.split(key, 6).reshape(2, 3, 2)  # (2, 3, 2)
        result = torch.random.normal(keys, (2, 3, 50))
        for i in range(2):
            for j in range(3):
                individual = torch.random.normal(keys[i][j], (50,))
                self.assertEqual(result[i][j], individual)

    def test_broadcasting(self):
        key = torch.random.key(42, device="cuda").unsqueeze(0)  # (1, 2)
        result = torch.random.normal(key, (4, 100))
        for i in range(1, 4):
            self.assertEqual(result[0], result[i])

    def test_dtype_and_device(self):
        key = torch.random.key(42, device="cuda")
        result = torch.random.normal(
            key, (500,), mean=3.0, std=0.5, dtype=torch.float64
        )
        self.assertEqual(result.shape, (500,))
        self.assertEqual(result.dtype, torch.float64)

    def test_error_wrong_key_dtype(self):
        key = torch.tensor([42, 0], dtype=torch.float32, device="cuda")
        with self.assertRaises(RuntimeError):
            torch.random.normal(key, (100,))

    def test_error_wrong_device(self):
        key = torch.random.key(42)  # CPU key
        with self.assertRaises(RuntimeError):
            torch.random.normal(key, (100,), device="cuda")

    def test_offset_shift_consistency(self):
        """Box-Muller alignment: shifting key offset shifts the output stream."""
        seed = 42
        n = 100
        key0 = torch.tensor([seed, 0], dtype=torch.uint64, device="cuda")
        ref = torch.random.normal(key0, (n,))
        for offset in range(1, 4):
            key = torch.tensor([seed, offset], dtype=torch.uint64, device="cuda")
            result = torch.random.normal(key, (n - offset,))
            self.assertEqual(result, ref[offset:])

    def test_offset_shift_consistency_double(self):
        """Box-Muller alignment for double: offset shift of 2 = element shift of 1."""
        seed = 42
        n = 100
        key0 = torch.tensor([seed, 0], dtype=torch.uint64, device="cuda")
        ref = torch.random.normal(key0, (n,), dtype=torch.float64)
        key2 = torch.tensor([seed, 2], dtype=torch.uint64, device="cuda")
        result = torch.random.normal(key2, (n - 1,), dtype=torch.float64)
        self.assertEqual(result, ref[1:])

    def test_error_shape_mismatch(self):
        key = torch.random.key(42, device="cuda")
        keys = torch.random.split(key, 3)  # (3, 2)
        with self.assertRaises(RuntimeError):
            torch.random.normal(keys, (2, 100))  # batch dim 2 != 3

    def test_error_key_last_dim_not_2(self):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device="cuda")
        with self.assertRaises(RuntimeError):
            torch.random.normal(key, (100,))


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
class TestPhiloxUniform(TestCase):
    def test_basic_shape_and_dtype(self):
        key = torch.random.key(42, device="cuda")
        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            result = torch.random.uniform(key, (100,), dtype=dtype)
            self.assertEqual(result.shape, (100,))
            self.assertEqual(result.dtype, dtype)

    def test_determinism(self):
        key = torch.random.key(42, device="cuda")
        a = torch.random.uniform(key, (1000,))
        b = torch.random.uniform(key, (1000,))
        self.assertEqual(a, b)

    def test_different_keys(self):
        key1 = torch.random.key(42, device="cuda")
        key2 = torch.random.key(43, device="cuda")
        a = torch.random.uniform(key1, (1000,))
        b = torch.random.uniform(key2, (1000,))
        self.assertNotEqual(a, b)

    def test_standard_uniform_statistics(self):
        key = torch.random.key(42, device="cuda")
        result = torch.random.uniform(key, (100000,))
        self.assertTrue(abs(result.mean().item() - 0.5) < 0.05)
        self.assertTrue(result.min().item() > 0.0)
        self.assertTrue(result.max().item() <= 1.0)

    def test_custom_low_high(self):
        key = torch.random.key(42, device="cuda")
        result = torch.random.uniform(key, (100000,), low=2.0, high=5.0)
        self.assertTrue(abs(result.mean().item() - 3.5) < 0.1)
        self.assertTrue(result.min().item() >= 2.0)
        self.assertTrue(result.max().item() <= 5.0)

    def test_batched_keys(self):
        key = torch.random.key(42, device="cuda")
        keys = torch.random.split(key, 4)  # (4, 2)
        result = torch.random.uniform(keys, (4, 100))
        for i in range(4):
            individual = torch.random.uniform(keys[i], (100,))
            self.assertEqual(result[i], individual)

    def test_multi_batch(self):
        key = torch.random.key(42, device="cuda")
        keys = torch.random.split(key, 6).reshape(2, 3, 2)  # (2, 3, 2)
        result = torch.random.uniform(keys, (2, 3, 50))
        for i in range(2):
            for j in range(3):
                individual = torch.random.uniform(keys[i][j], (50,))
                self.assertEqual(result[i][j], individual)

    def test_broadcasting(self):
        key = torch.random.key(42, device="cuda").unsqueeze(0)  # (1, 2)
        result = torch.random.uniform(key, (4, 100))
        for i in range(1, 4):
            self.assertEqual(result[0], result[i])

    def test_dtype_and_device(self):
        key = torch.random.key(42, device="cuda")
        result = torch.random.uniform(key, (500,), low=2.0, high=5.0, dtype=torch.float64)
        self.assertEqual(result.shape, (500,))
        self.assertEqual(result.dtype, torch.float64)

    def test_error_wrong_key_dtype(self):
        key = torch.tensor([42, 0], dtype=torch.float32, device="cuda")
        with self.assertRaises(RuntimeError):
            torch.random.uniform(key, (100,))

    def test_error_wrong_device(self):
        key = torch.random.key(42)  # CPU key
        with self.assertRaises(RuntimeError):
            torch.random.uniform(key, (100,), device="cuda")

    def test_error_shape_mismatch(self):
        key = torch.random.key(42, device="cuda")
        keys = torch.random.split(key, 3)  # (3, 2)
        with self.assertRaises(RuntimeError):
            torch.random.uniform(keys, (2, 100))  # batch dim 2 != 3

    def test_error_key_last_dim_not_2(self):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device="cuda")
        with self.assertRaises(RuntimeError):
            torch.random.uniform(key, (100,))

    def test_offset_shift_consistency_double(self):
        """Float64 uniform: offset shift of 2 = element shift of 1."""
        seed = 42
        n = 100
        key0 = torch.tensor([seed, 0], dtype=torch.uint64, device="cuda")
        ref = torch.random.uniform(key0, (n,), dtype=torch.float64)
        key2 = torch.tensor([seed, 2], dtype=torch.uint64, device="cuda")
        result = torch.random.uniform(key2, (n - 1,), dtype=torch.float64)
        self.assertEqual(result, ref[1:])


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
class TestPhiloxCompile(TestCase):
    def test_uniform_aot_eager(self):
        key = torch.random.key(42, device="cuda")

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return torch.random.uniform(key, (100,))

        self.assertEqual(f(key), torch.random.uniform(key, (100,)))

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

    def test_normal_aot_eager(self):
        key = torch.random.key(42, device="cuda")

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return torch.random.normal(key, (100,))

        self.assertEqual(f(key), torch.random.normal(key, (100,)))

    def test_batched_normal_aot_eager(self):
        key = torch.random.key(42, device="cuda")
        keys = torch.random.split(key, 4)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(keys):
            return torch.random.normal(keys, (4, 50))

        self.assertEqual(f(keys), torch.random.normal(keys, (4, 50)))

    def test_split_then_normal_aot_eager(self):
        key = torch.random.key(42, device="cuda")

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            keys = torch.random.split(key, 4)
            return torch.random.normal(keys, (4, 100))

        self.assertEqual(
            f(key), torch.random.normal(torch.random.split(key, 4), (4, 100))
        )

    def test_fold_in_then_uniform_aot_eager(self):
        key = torch.random.key(42, device="cuda")

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            k = torch.random.fold_in(key, 3)
            return torch.random.uniform(k, (100,))

        self.assertEqual(f(key), torch.random.uniform(
            torch.random.fold_in(key, 3), (100,)))


if __name__ == "__main__":
    run_tests()
