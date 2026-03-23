# Owner(s): ["module: cuda"]

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

    def test_error_invalid_num_splits(self, device):
        key = torch.random.key(42, device=device)
        with self.assertRaises(RuntimeError):
            torch.random.split(key, 0)
        with self.assertRaises(RuntimeError):
            torch.random.split(key, -1)

    def test_error_batched_last_dim_not_2(self, device):
        key = torch.tensor(
            [[42, 0, 1], [43, 0, 1]], dtype=torch.uint64, device=device
        )
        with self.assertRaises(RuntimeError):
            torch.random.split(key, 4)

    @onlyCUDA
    def test_cross_device_consistency(self, device):
        key_cpu = torch.random.key(42)
        key_cuda = torch.random.key(42, device=device)
        self.assertEqual(
            torch.random.split(key_cpu, 100),
            torch.random.split(key_cuda, 100).cpu(),
        )


instantiate_device_type_tests(TestPhiloxKeySplit, globals(), only_for=("cpu", "cuda"))


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

    def test_error_batched_last_dim_not_2(self, device):
        key = torch.tensor(
            [[42, 0, 1], [43, 0, 1]], dtype=torch.uint64, device=device
        )
        with self.assertRaises(RuntimeError):
            torch.random.fold_in(key, 0)

    @onlyCUDA
    def test_cross_device_consistency(self, device):
        key_cpu = torch.random.key(42)
        key_cuda = torch.random.key(42, device=device)
        self.assertEqual(
            torch.random.fold_in(key_cpu, 7),
            torch.random.fold_in(key_cuda, 7).cpu(),
        )


instantiate_device_type_tests(
    TestPhiloxKeyFoldIn, globals(), only_for=("cpu", "cuda")
)


class TestPhiloxNormal(TestCase):
    def test_basic_shape_and_dtype(self, device):
        key = torch.random.key(42, device=device)
        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            result = torch.random.normal(key, (100,), dtype=dtype)
            self.assertEqual(result.shape, (100,))
            self.assertEqual(result.dtype, dtype)

    def test_determinism(self, device):
        key = torch.random.key(42, device=device)
        a = torch.random.normal(key, (1000,))
        b = torch.random.normal(key, (1000,))
        self.assertEqual(a, b)

    def test_different_keys(self, device):
        key1 = torch.random.key(42, device=device)
        key2 = torch.random.key(43, device=device)
        a = torch.random.normal(key1, (1000,))
        b = torch.random.normal(key2, (1000,))
        self.assertNotEqual(a, b)

    def test_standard_normal_statistics(self, device):
        key = torch.random.key(42, device=device)
        result = torch.random.normal(key, (100000,))
        self.assertTrue(abs(result.mean().item()) < 0.05)
        self.assertTrue(abs(result.std().item() - 1.0) < 0.05)

    def test_custom_mean_std(self, device):
        key = torch.random.key(42, device=device)
        result = torch.random.normal(key, (100000,), mean=5.0, std=2.0)
        self.assertTrue(abs(result.mean().item() - 5.0) < 0.1)
        self.assertTrue(abs(result.std().item() - 2.0) < 0.1)

    def test_batched_keys(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 4)  # (4, 2)
        result = torch.random.normal(keys, (4, 100))
        for i in range(4):
            individual = torch.random.normal(keys[i], (100,))
            self.assertEqual(result[i], individual)

    def test_multi_batch(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 6).reshape(2, 3, 2)  # (2, 3, 2)
        result = torch.random.normal(keys, (2, 3, 50))
        for i in range(2):
            for j in range(3):
                individual = torch.random.normal(keys[i][j], (50,))
                self.assertEqual(result[i][j], individual)

    def test_broadcasting(self, device):
        key = torch.random.key(42, device=device).unsqueeze(0)  # (1, 2)
        result = torch.random.normal(key, (4, 100))
        for i in range(1, 4):
            self.assertEqual(result[0], result[i])

    def test_dtype_and_device(self, device):
        key = torch.random.key(42, device=device)
        result = torch.random.normal(
            key, (500,), mean=3.0, std=0.5, dtype=torch.float64
        )
        self.assertEqual(result.shape, (500,))
        self.assertEqual(result.dtype, torch.float64)

    def test_error_wrong_key_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaises(RuntimeError):
            torch.random.normal(key, (100,))

    @onlyCUDA
    def test_error_wrong_device(self, device):
        key = torch.random.key(42)  # CPU key
        with self.assertRaises(RuntimeError):
            torch.random.normal(key, (100,), device="cuda")

    def test_offset_shift_consistency(self, device):
        """Box-Muller alignment: shifting key offset shifts the output stream."""
        seed = 42
        n = 100
        key0 = torch.tensor([seed, 0], dtype=torch.uint64, device=device)
        ref = torch.random.normal(key0, (n,))
        for offset in range(1, 4):
            key = torch.tensor([seed, offset], dtype=torch.uint64, device=device)
            result = torch.random.normal(key, (n - offset,))
            self.assertEqual(result, ref[offset:])

    def test_offset_shift_consistency_double(self, device):
        """Box-Muller alignment for double: offset shift of 2 = element shift of 1."""
        seed = 42
        n = 100
        key0 = torch.tensor([seed, 0], dtype=torch.uint64, device=device)
        ref = torch.random.normal(key0, (n,), dtype=torch.float64)
        key2 = torch.tensor([seed, 2], dtype=torch.uint64, device=device)
        result = torch.random.normal(key2, (n - 1,), dtype=torch.float64)
        self.assertEqual(result, ref[1:])

    def test_error_shape_mismatch(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 3)  # (3, 2)
        with self.assertRaises(RuntimeError):
            torch.random.normal(keys, (2, 100))  # batch dim 2 != 3

    def test_error_key_last_dim_not_2(self, device):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            torch.random.normal(key, (100,))


instantiate_device_type_tests(TestPhiloxNormal, globals(), only_for=("cpu", "cuda"))


class TestPhiloxUniform(TestCase):
    def test_basic_shape_and_dtype(self, device):
        key = torch.random.key(42, device=device)
        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            result = torch.random.uniform(key, (100,), dtype=dtype)
            self.assertEqual(result.shape, (100,))
            self.assertEqual(result.dtype, dtype)

    def test_determinism(self, device):
        key = torch.random.key(42, device=device)
        a = torch.random.uniform(key, (1000,))
        b = torch.random.uniform(key, (1000,))
        self.assertEqual(a, b)

    def test_different_keys(self, device):
        key1 = torch.random.key(42, device=device)
        key2 = torch.random.key(43, device=device)
        a = torch.random.uniform(key1, (1000,))
        b = torch.random.uniform(key2, (1000,))
        self.assertNotEqual(a, b)

    def test_standard_uniform_statistics(self, device):
        key = torch.random.key(42, device=device)
        result = torch.random.uniform(key, (100000,))
        self.assertTrue(abs(result.mean().item() - 0.5) < 0.05)
        self.assertTrue(result.min().item() > 0.0)
        self.assertTrue(result.max().item() <= 1.0)

    def test_custom_low_high(self, device):
        key = torch.random.key(42, device=device)
        result = torch.random.uniform(key, (100000,), low=2.0, high=5.0)
        self.assertTrue(abs(result.mean().item() - 3.5) < 0.1)
        self.assertTrue(result.min().item() >= 2.0)
        self.assertTrue(result.max().item() <= 5.0)

    def test_batched_keys(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 4)  # (4, 2)
        result = torch.random.uniform(keys, (4, 100))
        for i in range(4):
            individual = torch.random.uniform(keys[i], (100,))
            self.assertEqual(result[i], individual)

    def test_multi_batch(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 6).reshape(2, 3, 2)  # (2, 3, 2)
        result = torch.random.uniform(keys, (2, 3, 50))
        for i in range(2):
            for j in range(3):
                individual = torch.random.uniform(keys[i][j], (50,))
                self.assertEqual(result[i][j], individual)

    def test_broadcasting(self, device):
        key = torch.random.key(42, device=device).unsqueeze(0)  # (1, 2)
        result = torch.random.uniform(key, (4, 100))
        for i in range(1, 4):
            self.assertEqual(result[0], result[i])

    def test_dtype_and_device(self, device):
        key = torch.random.key(42, device=device)
        result = torch.random.uniform(
            key, (500,), low=2.0, high=5.0, dtype=torch.float64
        )
        self.assertEqual(result.shape, (500,))
        self.assertEqual(result.dtype, torch.float64)

    def test_error_wrong_key_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaises(RuntimeError):
            torch.random.uniform(key, (100,))

    @onlyCUDA
    def test_error_wrong_device(self, device):
        key = torch.random.key(42)  # CPU key
        with self.assertRaises(RuntimeError):
            torch.random.uniform(key, (100,), device="cuda")

    def test_error_shape_mismatch(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 3)  # (3, 2)
        with self.assertRaises(RuntimeError):
            torch.random.uniform(keys, (2, 100))  # batch dim 2 != 3

    def test_error_key_last_dim_not_2(self, device):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            torch.random.uniform(key, (100,))

    def test_offset_shift_consistency_double(self, device):
        """Float64 uniform: offset shift of 2 = element shift of 1."""
        seed = 42
        n = 100
        key0 = torch.tensor([seed, 0], dtype=torch.uint64, device=device)
        ref = torch.random.uniform(key0, (n,), dtype=torch.float64)
        key2 = torch.tensor([seed, 2], dtype=torch.uint64, device=device)
        result = torch.random.uniform(key2, (n - 1,), dtype=torch.float64)
        self.assertEqual(result, ref[1:])

    @onlyCUDA
    def test_cross_device_uniform_consistency(self, device):
        key_cpu = torch.random.key(42)
        key_cuda = torch.random.key(42, device=device)
        self.assertEqual(
            torch.random.uniform(key_cpu, (1000,)),
            torch.random.uniform(key_cuda, (1000,)).cpu(),
        )

    @onlyCUDA
    def test_cross_device_f64_uniform_consistency(self, device):
        key_cpu = torch.random.key(42)
        key_cuda = torch.random.key(42, device=device)
        self.assertEqual(
            torch.random.uniform(key_cpu, (1000,), dtype=torch.float64),
            torch.random.uniform(key_cuda, (1000,), dtype=torch.float64).cpu(),
        )


instantiate_device_type_tests(TestPhiloxUniform, globals(), only_for=("cpu", "cuda"))


class TestPhiloxCompile(TestCase):
    def test_uniform_aot_eager(self, device):
        key = torch.random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return torch.random.uniform(key, (100,))

        self.assertEqual(f(key), torch.random.uniform(key, (100,)))

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

    def test_normal_aot_eager(self, device):
        key = torch.random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return torch.random.normal(key, (100,))

        self.assertEqual(f(key), torch.random.normal(key, (100,)))

    def test_batched_normal_aot_eager(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 4)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(keys):
            return torch.random.normal(keys, (4, 50))

        self.assertEqual(f(keys), torch.random.normal(keys, (4, 50)))

    def test_split_then_normal_aot_eager(self, device):
        key = torch.random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            keys = torch.random.split(key, 4)
            return torch.random.normal(keys, (4, 100))

        self.assertEqual(
            f(key), torch.random.normal(torch.random.split(key, 4), (4, 100))
        )

    def test_fold_in_then_uniform_aot_eager(self, device):
        key = torch.random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            k = torch.random.fold_in(key, 3)
            return torch.random.uniform(k, (100,))

        self.assertEqual(
            f(key), torch.random.uniform(torch.random.fold_in(key, 3), (100,))
        )


instantiate_device_type_tests(TestPhiloxCompile, globals(), only_for=("cpu", "cuda"))


class TestPhiloxVmap(TestCase):
    def test_vmap_normal(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 10)

        result = torch.vmap(lambda k: torch.random.normal(k, 5))(keys)
        self.assertEqual(result.shape, (10, 5))
        for i in range(10):
            self.assertEqual(result[i], torch.random.normal(keys[i], 5))

    def test_vmap_uniform(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 10)

        result = torch.vmap(lambda k: torch.random.uniform(k, 5))(keys)
        self.assertEqual(result.shape, (10, 5))
        for i in range(10):
            self.assertEqual(result[i], torch.random.uniform(keys[i], 5))

    def test_vmap_split(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 10)

        result = torch.vmap(lambda k: torch.random.split(k, 3))(keys)
        self.assertEqual(result.shape, (10, 3, 2))
        for i in range(10):
            self.assertEqual(result[i], torch.random.split(keys[i], 3))

    def test_vmap_fold_in(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 10)

        result = torch.vmap(lambda k: torch.random.fold_in(k, 7))(keys)
        self.assertEqual(result.shape, (10, 2))
        for i in range(10):
            self.assertEqual(result[i], torch.random.fold_in(keys[i], 7))

    def test_vmap_inplace_batched_self(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 10)
        out = torch.empty(10, 5, device=device)

        def f(o, k):
            return torch.ops.aten._philox_normal_(o, k, 0.0, 1.0)

        result = torch.vmap(f)(out, keys)
        self.assertEqual(result.shape, (10, 5))
        for i in range(10):
            self.assertEqual(result[i], torch.random.normal(keys[i], 5))

    def test_vmap_split_then_normal(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 8)

        def f(k):
            subkeys = torch.random.split(k, 3)
            return torch.random.normal(subkeys, (3, 20))

        result = torch.vmap(f)(keys)
        self.assertEqual(result.shape, (8, 3, 20))
        for i in range(8):
            self.assertEqual(result[i], f(keys[i]))

    def test_vmap_normal_multidim(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 5)

        result = torch.vmap(lambda k: torch.random.normal(k, 4, 3))(keys)
        self.assertEqual(result.shape, (5, 4, 3))
        for i in range(5):
            self.assertEqual(result[i], torch.random.normal(keys[i], 4, 3))

    def test_vmap_compiled_normal(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 10)

        @torch.compile(backend="aot_eager")
        def f(keys):
            return torch.vmap(lambda k: torch.random.normal(k, 5))(keys)

        result = f(keys)
        self.assertEqual(result.shape, (10, 5))
        for i in range(10):
            self.assertEqual(result[i], torch.random.normal(keys[i], 5))

    def test_vmap_compiled_uniform(self, device):
        key = torch.random.key(42, device=device)
        keys = torch.random.split(key, 10)

        @torch.compile(backend="aot_eager")
        def f(keys):
            return torch.vmap(lambda k: torch.random.uniform(k, 5))(keys)

        result = f(keys)
        self.assertEqual(result.shape, (10, 5))
        for i in range(10):
            self.assertEqual(result[i], torch.random.uniform(keys[i], 5))


instantiate_device_type_tests(TestPhiloxVmap, globals(), only_for=("cpu", "cuda"))



class TestGridSplit(TestCase):
    def test_1d_shape(self, device):
        k = torch.random.key(42, device=device)
        keys = torch.random.grid_split(k, (100,), (10,))
        self.assertIsInstance(keys, torch.random.PhiloxKey)
        # 10 tiles, each of size 10
        self.assertEqual(keys.shape, (10, 2))

    def test_1d_uniform_reconstruction(self, device):
        k = torch.random.key(42, device=device)
        num_tiles = 10
        keys = torch.random.grid_split(k, (100,), (num_tiles,))
        full = torch.random.uniform(k, (100,), device=device)
        tile_size = 100 // num_tiles
        tiled = torch.cat(
            [
                torch.random.uniform(keys[i], (tile_size,), device=device)
                for i in range(num_tiles)
            ]
        )
        self.assertEqual(full, tiled)

    def test_1d_normal_reconstruction(self, device):
        k = torch.random.key(42, device=device)
        num_tiles = 10
        keys = torch.random.grid_split(k, (100,), (num_tiles,))
        full = torch.random.normal(k, (100,), device=device)
        tile_size = 100 // num_tiles
        tiled = torch.cat(
            [
                torch.random.normal(keys[i], (tile_size,), device=device)
                for i in range(num_tiles)
            ]
        )
        self.assertEqual(full, tiled)

    def test_1d_determinism(self, device):
        k = torch.random.key(42, device=device)
        keys1 = torch.random.grid_split(k, (100,), (10,))
        keys2 = torch.random.grid_split(k, (100,), (10,))
        self.assertEqual(keys1._data, keys2._data)

    def test_2d_shape(self, device):
        k = torch.random.key(42, device=device)
        splits = (10, 10)
        keys = torch.random.grid_split(k, (100, 200), splits)
        # splits=(10,10), tile_shape=(10,20), per-tile rows=10
        # -> shape (*splits, tile_shape[0], 2) = (10, 10, 10, 2)
        self.assertEqual(keys.shape, (10, 10, 10, 2))

    def test_2d_uniform_tile(self, device):
        k = torch.random.key(42, device=device)
        shape = (100, 200)
        splits = (10, 10)
        tile_shape = (10, 20)
        keys = torch.random.grid_split(k, shape, splits)
        full = torch.random.uniform(k, shape, device=device)
        tile = torch.random.uniform(keys[0, 0], tile_shape, device=device)
        self.assertEqual(tile, full[0:10, 0:20])

    def test_2d_uniform_reconstruction(self, device):
        k = torch.random.key(42, device=device)
        shape = (60, 80)
        splits = (6, 4)
        tile_shape = (10, 20)
        keys = torch.random.grid_split(k, shape, splits)
        full = torch.random.uniform(k, shape, device=device)
        tiles = []
        for r in range(splits[0]):
            row = []
            for c in range(splits[1]):
                row.append(
                    torch.random.uniform(keys[r, c], tile_shape, device=device)
                )
            tiles.append(torch.cat(row, dim=1))
        tiled = torch.cat(tiles, dim=0)
        self.assertEqual(full, tiled)

    def test_2d_normal_reconstruction(self, device):
        k = torch.random.key(42, device=device)
        shape = (60, 80)
        splits = (6, 4)
        tile_shape = (10, 20)
        keys = torch.random.grid_split(k, shape, splits)
        full = torch.random.normal(k, shape, device=device)
        tiles = []
        for r in range(splits[0]):
            row = []
            for c in range(splits[1]):
                row.append(
                    torch.random.normal(keys[r, c], tile_shape, device=device)
                )
            tiles.append(torch.cat(row, dim=1))
        tiled = torch.cat(tiles, dim=0)
        self.assertEqual(full, tiled)

    def test_2d_arbitrary_tile(self, device):
        """Verify a non-corner tile matches the full generation."""
        k = torch.random.key(123, device=device)
        shape = (100, 200)
        splits = (10, 10)
        tile_shape = (10, 20)
        keys = torch.random.grid_split(k, shape, splits)
        full = torch.random.uniform(k, shape, device=device)
        for tr, tc in [(3, 7), (9, 9), (0, 5)]:
            tile = torch.random.uniform(keys[tr, tc], tile_shape, device=device)
            expected = full[
                tr * tile_shape[0] : (tr + 1) * tile_shape[0],
                tc * tile_shape[1] : (tc + 1) * tile_shape[1],
            ]
            self.assertEqual(tile, expected)

    def test_3d_shape(self, device):
        k = torch.random.key(42, device=device)
        splits = (3, 4, 3)
        keys = torch.random.grid_split(k, (12, 20, 30), splits)
        # tile_shape = (4, 5, 10), per-tile rows = (4, 5)
        # -> shape (*splits, 4, 5, 2) = (3, 4, 3, 4, 5, 2)
        self.assertEqual(keys.shape, (3, 4, 3, 4, 5, 2))

    def test_3d_uniform_reconstruction(self, device):
        k = torch.random.key(77, device=device)
        shape = (12, 20, 30)
        splits = (3, 4, 3)
        tile_shape = tuple(s // sp for s, sp in zip(shape, splits))
        keys = torch.random.grid_split(k, shape, splits)
        full = torch.random.uniform(k, shape, device=device)
        reconstructed = torch.empty_like(full)
        for t0 in range(splits[0]):
            for t1 in range(splits[1]):
                for t2 in range(splits[2]):
                    tile = torch.random.uniform(
                        keys[t0, t1, t2], tile_shape, device=device
                    )
                    reconstructed[
                        t0 * tile_shape[0] : (t0 + 1) * tile_shape[0],
                        t1 * tile_shape[1] : (t1 + 1) * tile_shape[1],
                        t2 * tile_shape[2] : (t2 + 1) * tile_shape[2],
                    ] = tile
        self.assertEqual(full, reconstructed)

    def test_2d_row_only_split(self, device):
        """Splitting only along rows (single column tile) should work."""
        k = torch.random.key(42, device=device)
        splits = (10, 1)
        keys = torch.random.grid_split(k, (100, 200), splits)
        tile_shape = (10, 200)
        # shape: (*splits, tile_shape[0], 2) = (10, 1, 10, 2)
        self.assertEqual(keys.shape, (10, 1, 10, 2))
        full = torch.random.uniform(k, (100, 200), device=device)
        tiles = [
            torch.random.uniform(keys[i, 0], tile_shape, device=device)
            for i in range(10)
        ]
        tiled = torch.cat(tiles, dim=0)
        self.assertEqual(full, tiled)

    def test_error_requires_prng_key(self, device):
        plain = torch.tensor([42, 0], dtype=torch.uint64, device=device)
        with self.assertRaises(TypeError):
            torch.random.grid_split(plain, (100,), (10,))

    def test_error_uneven_split(self, device):
        k = torch.random.key(42, device=device)
        with self.assertRaisesRegex(ValueError, "does not evenly divide"):
            torch.random.grid_split(k, (100,), (3,))
        with self.assertRaisesRegex(ValueError, "does not evenly divide"):
            torch.random.grid_split(k, (100, 200), (10, 3))

    def test_error_mismatched_lengths(self, device):
        k = torch.random.key(42, device=device)
        with self.assertRaisesRegex(ValueError, "same length"):
            torch.random.grid_split(k, (100, 200), (10,))

    @onlyCUDA
    def test_cross_device_consistency_1d(self, device):
        k_cpu = torch.random.key(42)
        k_cuda = torch.random.key(42, device=device)
        keys_cpu = torch.random.grid_split(k_cpu, (100,), (10,))
        keys_cuda = torch.random.grid_split(k_cuda, (100,), (10,))
        self.assertEqual(keys_cpu._data, keys_cuda._data.cpu())

    @onlyCUDA
    def test_cross_device_consistency_2d(self, device):
        k_cpu = torch.random.key(42)
        k_cuda = torch.random.key(42, device=device)
        shape = (60, 80)
        splits = (6, 4)
        tile_shape = (10, 20)
        keys_cpu = torch.random.grid_split(k_cpu, shape, splits)
        keys_cuda = torch.random.grid_split(k_cuda, shape, splits)
        tile_cpu = torch.random.uniform(keys_cpu[2, 1], tile_shape)
        tile_cuda = torch.random.uniform(
            keys_cuda[2, 1], tile_shape, device=device
        )
        self.assertEqual(tile_cpu, tile_cuda.cpu())


instantiate_device_type_tests(TestGridSplit, globals(), only_for=("cpu", "cuda"))


class TestStatefulPRNG(TestCase):
    def test_uniform_basic(self, device):
        g = torch.random.StatefulPRNG(42, device=device)
        result = g.uniform(100)
        self.assertEqual(result.shape, (100,))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device(device))

    def test_normal_basic(self, device):
        g = torch.random.StatefulPRNG(42, device=device)
        result = g.normal(100)
        self.assertEqual(result.shape, (100,))

    def test_sequential_calls_differ(self, device):
        g = torch.random.StatefulPRNG(42, device=device)
        a = g.uniform(100)
        b = g.uniform(100)
        self.assertNotEqual(a, b)

    def test_manual_seed_resets(self, device):
        g = torch.random.StatefulPRNG(42, device=device)
        a = g.normal(100)
        g.manual_seed(42)
        b = g.normal(100)
        self.assertEqual(a, b)

    def test_matches_stateless_stream(self, device):
        """Sequential StatefulPRNG calls produce the same stream as one large stateless call."""
        g = torch.random.StatefulPRNG(42, device=device)
        first = g.uniform(100)
        second = g.uniform(100)

        k = torch.random.key(42, device=device)
        full = torch.random.uniform(k, (200,), device=device)
        self.assertEqual(torch.cat([first, second]), full)

    def test_matches_stateless_normal(self, device):
        g = torch.random.StatefulPRNG(42, device=device)
        first = g.normal(50)
        second = g.normal(50)

        k = torch.random.key(42, device=device)
        full = torch.random.normal(k, (100,), device=device)
        self.assertEqual(torch.cat([first, second]), full)

    def test_matches_stateless_f64_uniform(self, device):
        """Float64 uniform stream is contiguous at any split point."""
        g = torch.random.StatefulPRNG(42, device=device)
        first = g.uniform(17, dtype=torch.float64)
        second = g.uniform(33, dtype=torch.float64)

        k = torch.random.key(42, device=device)
        full = torch.random.uniform(k, (50,), dtype=torch.float64, device=device)
        self.assertEqual(torch.cat([first, second]), full)

    def test_matches_stateless_f64_normal(self, device):
        """Float64 normal stream is contiguous at any split point."""
        g = torch.random.StatefulPRNG(42, device=device)
        first = g.normal(17, dtype=torch.float64)
        second = g.normal(33, dtype=torch.float64)

        k = torch.random.key(42, device=device)
        full = torch.random.normal(k, (50,), dtype=torch.float64, device=device)
        self.assertEqual(torch.cat([first, second]), full)

    def test_multidim_shape(self, device):
        g = torch.random.StatefulPRNG(42, device=device)
        result = g.uniform(10, 20)
        self.assertEqual(result.shape, (10, 20))

        g.manual_seed(42)
        flat = g.uniform(200)
        self.assertEqual(result.flatten(), flat)

    def test_custom_params(self, device):
        g = torch.random.StatefulPRNG(42, device=device)
        result = g.uniform(1000, low=2.0, high=5.0)
        self.assertTrue(result.min().item() >= 2.0)
        self.assertTrue(result.max().item() <= 5.0)

        g.manual_seed(42)
        result = g.normal(1000, mean=5.0, std=0.5)
        self.assertTrue(abs(result.mean().item() - 5.0) < 0.1)

    def test_dtype(self, device):
        g = torch.random.StatefulPRNG(42, device=device)
        result = g.uniform(100, dtype=torch.float64)
        self.assertEqual(result.dtype, torch.float64)

    def test_key_property(self, device):
        g = torch.random.StatefulPRNG(42, device=device)
        self.assertIsInstance(g.key, torch.random.PhiloxKey)
        k_before = g.key
        g.uniform(10)
        k_after = g.key
        self.assertNotEqual(k_before._data, k_after._data)

    def test_different_seeds(self, device):
        g1 = torch.random.StatefulPRNG(42, device=device)
        g2 = torch.random.StatefulPRNG(43, device=device)
        self.assertNotEqual(g1.uniform(100), g2.uniform(100))

    @onlyCUDA
    def test_cross_device_consistency(self, device):
        g_cpu = torch.random.StatefulPRNG(42)
        g_cuda = torch.random.StatefulPRNG(42, device=device)
        self.assertEqual(g_cpu.uniform(100), g_cuda.uniform(100).cpu())
        self.assertEqual(g_cpu.normal(100), g_cuda.normal(100).cpu())

    @onlyCUDA
    def test_cross_device_f64_uniform(self, device):
        g_cpu = torch.random.StatefulPRNG(42)
        g_cuda = torch.random.StatefulPRNG(42, device=device)
        self.assertEqual(
            g_cpu.uniform(100, dtype=torch.float64),
            g_cuda.uniform(100, dtype=torch.float64).cpu(),
        )


instantiate_device_type_tests(TestStatefulPRNG, globals(), only_for=("cpu", "cuda"))


if __name__ == "__main__":
    run_tests()
