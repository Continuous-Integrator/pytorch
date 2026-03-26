# Owner(s): ["module: random"]

import torch
import torch._dynamo.testing
import torch.func._random as random
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_utils import run_tests, TestCase

all_floating_dtypes = floating_types_and(torch.half, torch.bfloat16)


class TestPhiloxKeySplit(TestCase):
    def test_basic_shape_and_dtype(self, device):
        key = random.key(42, device=device)
        splits = random.split(key, 4)
        self.assertEqual(splits.shape, (4, 2))
        self.assertEqual(splits.dtype, torch.uint64)
        self.assertEqual(splits.device, key.device)

    def test_single_split(self, device):
        key = random.key(42, device=device)
        splits = random.split(key, 1)
        self.assertEqual(splits.shape, (1, 2))

    def test_large_num_splits(self, device):
        key = random.key(42, device=device)
        splits = random.split(key, 10000)
        self.assertEqual(splits.shape, (10000, 2))

    def test_determinism(self, device):
        key = random.key(42, device=device)
        splits1 = random.split(key, 8)
        splits2 = random.split(key, 8)
        self.assertEqual(splits1, splits2)

    def test_all_keys_unique(self, device):
        key = random.key(42, device=device)
        splits = random.split(key, 100)
        unique_keys = torch.unique(splits, dim=0)
        self.assertEqual(unique_keys.shape[0], 100)

    def test_different_seeds_produce_different_outputs(self, device):
        key1 = random.key(42, device=device)
        key2 = random.key(43, device=device)
        splits1 = random.split(key1, 4)
        splits2 = random.split(key2, 4)
        self.assertNotEqual(splits1, splits2)

    def test_different_offsets_produce_different_outputs(self, device):
        key1 = random.key(42, device=device)
        key2 = random.fold_in(key1, 1)
        splits1 = random.split(key1, 4)
        splits2 = random.split(key2, 4)
        self.assertNotEqual(splits1, splits2)

    def test_batched(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 4)  # (4, 2)
        num_splits = 3
        batched = random.split(keys, num_splits)  # (3, 4, 2)
        self.assertEqual(batched.shape, (num_splits, 4, 2))
        for k in range(4):
            individual = random.split(keys[k], num_splits)
            for s in range(num_splits):
                self.assertEqual(batched[s][k], individual[s])

    def test_multi_batch(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 12).reshape(3, 4, 2)  # (3, 4, 2)
        num_splits = 5
        batched = random.split(keys, num_splits)  # (5, 3, 4, 2)
        self.assertEqual(batched.shape, (num_splits, 3, 4, 2))
        for i in range(3):
            for j in range(4):
                individual = random.split(keys[i][j], num_splits)
                for s in range(num_splits):
                    self.assertEqual(batched[s][i][j], individual[s])

    def test_error_wrong_shape(self, device):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            random.split(key, 4)

    def test_error_wrong_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaises(RuntimeError):
            random.split(key, 4)

    def test_error_invalid_num_splits(self, device):
        key = random.key(42, device=device)
        with self.assertRaises(RuntimeError):
            random.split(key, 0)
        with self.assertRaises(RuntimeError):
            random.split(key, -1)

    def test_error_batched_last_dim_not_2(self, device):
        key = torch.tensor(
            [[42, 0, 1], [43, 0, 1]], dtype=torch.uint64, device=device
        )
        with self.assertRaises(RuntimeError):
            random.split(key, 4)

    @onlyCUDA
    def test_cross_device_consistency(self, device):
        key_cpu = random.key(42)
        key_cuda = random.key(42, device=device)
        self.assertEqual(
            random.split(key_cpu, 100),
            random.split(key_cuda, 100).cpu(),
        )


instantiate_device_type_tests(TestPhiloxKeySplit, globals(), only_for=("cpu", "cuda"))


class TestPhiloxKeyFoldIn(TestCase):
    def test_basic_shape_and_dtype(self, device):
        key = random.key(42, device=device)
        result = random.fold_in(key, 7)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.dtype, torch.uint64)
        self.assertEqual(result.device, key.device)

    def test_determinism(self, device):
        key = random.key(42, device=device)
        result1 = random.fold_in(key, 7)
        result2 = random.fold_in(key, 7)
        self.assertEqual(result1, result2)

    def test_fold_in_produces_new_key_for_zero_data(self, device):
        key = random.key(42, device=device)
        folded = random.fold_in(key, 0)
        self.assertNotEqual(folded, key)

    def test_different_data_produces_different_outputs(self, device):
        key = random.key(42, device=device)
        result1 = random.fold_in(key, 0)
        result2 = random.fold_in(key, 1)
        self.assertNotEqual(result1, result2)

    def test_consistency_with_split(self, device):
        key = random.key(42, device=device)
        splits = random.split(key, 10)
        for i in range(10):
            folded = random.fold_in(key, i)
            self.assertEqual(folded, splits[i])

    def test_batched(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 4)  # (4, 2)
        data = 7
        batched = random.fold_in(keys, data)  # (4, 2)
        self.assertEqual(batched.shape, (4, 2))
        for k in range(4):
            individual = random.fold_in(keys[k], data)
            self.assertEqual(batched[k], individual)

    def test_multi_batch(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 12).reshape(3, 4, 2)  # (3, 4, 2)
        data = 7
        batched = random.fold_in(keys, data)  # (3, 4, 2)
        self.assertEqual(batched.shape, (3, 4, 2))
        for i in range(3):
            for j in range(4):
                individual = random.fold_in(keys[i][j], data)
                self.assertEqual(batched[i][j], individual)

    def test_error_wrong_shape(self, device):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            random.fold_in(key, 0)

    def test_error_wrong_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaises(RuntimeError):
            random.fold_in(key, 0)

    def test_error_batched_last_dim_not_2(self, device):
        key = torch.tensor(
            [[42, 0, 1], [43, 0, 1]], dtype=torch.uint64, device=device
        )
        with self.assertRaises(RuntimeError):
            random.fold_in(key, 0)

    @onlyCUDA
    def test_cross_device_consistency(self, device):
        key_cpu = random.key(42)
        key_cuda = random.key(42, device=device)
        self.assertEqual(
            random.fold_in(key_cpu, 7),
            random.fold_in(key_cuda, 7).cpu(),
        )


instantiate_device_type_tests(
    TestPhiloxKeyFoldIn, globals(), only_for=("cpu", "cuda")
)


class TestPhiloxNormal(TestCase):
    @dtypes(*all_floating_dtypes)
    def test_basic_shape(self, device, dtype):
        key = random.key(42, device=device)
        result = random.normal(key, (100,), dtype=dtype)
        self.assertEqual(result.shape, (100,))
        self.assertEqual(result.dtype, dtype)

    @dtypes(*all_floating_dtypes)
    def test_determinism(self, device, dtype):
        key = random.key(42, device=device)
        a = random.normal(key, (1000,), dtype=dtype)
        b = random.normal(key, (1000,), dtype=dtype)
        self.assertEqual(a, b)

    @dtypes(*all_floating_dtypes)
    def test_different_keys(self, device, dtype):
        key1 = random.key(42, device=device)
        key2 = random.key(43, device=device)
        a = random.normal(key1, (1000,), dtype=dtype)
        b = random.normal(key2, (1000,), dtype=dtype)
        self.assertNotEqual(a, b)

    @dtypes(*all_floating_dtypes)
    def test_standard_normal_statistics(self, device, dtype):
        key = random.key(42, device=device)
        result = random.normal(key, (100000,), dtype=dtype)
        self.assertTrue(abs(result.mean().item()) < 0.05)
        self.assertTrue(abs(result.std().item() - 1.0) < 0.05)

    @dtypes(*all_floating_dtypes)
    def test_custom_mean_std(self, device, dtype):
        key = random.key(42, device=device)
        result = random.normal(key, (100000,), mean=5.0, std=2.0, dtype=dtype)
        self.assertTrue(abs(result.mean().item() - 5.0) < 0.1)
        self.assertTrue(abs(result.std().item() - 2.0) < 0.1)

    @dtypes(*all_floating_dtypes)
    def test_batched_keys(self, device, dtype):
        key = random.key(42, device=device)
        keys = random.split(key, 4)  # (4, 2)
        result = random.normal(keys, (4, 100), dtype=dtype)
        for i in range(4):
            individual = random.normal(keys[i], (100,), dtype=dtype)
            self.assertEqual(result[i], individual)

    @dtypes(*all_floating_dtypes)
    def test_multi_batch(self, device, dtype):
        key = random.key(42, device=device)
        keys = random.split(key, 6).reshape(2, 3, 2)  # (2, 3, 2)
        result = random.normal(keys, (2, 3, 50), dtype=dtype)
        for i in range(2):
            for j in range(3):
                individual = random.normal(keys[i][j], (50,), dtype=dtype)
                self.assertEqual(result[i][j], individual)

    @dtypes(*all_floating_dtypes)
    def test_broadcasting(self, device, dtype):
        key = random.key(42, device=device).unsqueeze(0)  # (1, 2)
        result = random.normal(key, (4, 100), dtype=dtype)
        for i in range(1, 4):
            self.assertEqual(result[0], result[i])

    def test_error_wrong_key_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaises(RuntimeError):
            random.normal(key, (100,))

    @onlyCUDA
    def test_error_wrong_device(self, device):
        key = random.key(42)  # CPU key
        with self.assertRaises(RuntimeError):
            random.normal(key, (100,), device=device)

    @dtypes(torch.float32, torch.float64)
    def test_offset_shift_consistency(self, device, dtype):
        """Shifting key offset shifts the output stream."""
        seed = 42
        n = 100
        outputs_per_elem = 2 if dtype == torch.float64 else 1
        key0 = torch.tensor([seed, 0], dtype=torch.uint64, device=device)
        ref = random.normal(key0, (n,), dtype=dtype)
        for elem_offset in range(1, 4):
            offset = elem_offset * outputs_per_elem
            key = torch.tensor([seed, offset], dtype=torch.uint64, device=device)
            result = random.normal(key, (n - elem_offset,), dtype=dtype)
            self.assertEqual(result, ref[elem_offset:])

    def test_error_shape_mismatch(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 3)  # (3, 2)
        with self.assertRaises(RuntimeError):
            random.normal(keys, (2, 100))  # batch dim 2 != 3

    @dtypes(torch.float32, torch.float64)
    def test_offset_overflow(self, device, dtype):
        """After wrapping past 2^64, generation continues from offset 0."""
        seed = 42
        outputs_per_elem = 2 if dtype == torch.float64 else 1
        wrap_at = 5
        near_max = (1 << 64) - wrap_at * outputs_per_elem
        key = torch.tensor([seed, near_max], dtype=torch.uint64, device=device)
        result = random.normal(key, (20,), dtype=dtype)
        # First wrap_at elements come from the stream at near_max.
        self.assertEqual(
            result[:wrap_at],
            random.normal(key, (wrap_at,), dtype=dtype),
        )
        # After the wrap, elements come from the stream at offset 0.
        key_zero = torch.tensor([seed, 0], dtype=torch.uint64, device=device)
        self.assertEqual(
            result[wrap_at:],
            random.normal(key_zero, (20 - wrap_at,), dtype=dtype),
        )

    def test_error_key_last_dim_not_2(self, device):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            random.normal(key, (100,))


instantiate_device_type_tests(TestPhiloxNormal, globals(), only_for=("cpu", "cuda"))


class TestPhiloxUniform(TestCase):
    @dtypes(*all_floating_dtypes)
    def test_basic_shape(self, device, dtype):
        key = random.key(42, device=device)
        result = random.uniform(key, (100,), dtype=dtype)
        self.assertEqual(result.shape, (100,))
        self.assertEqual(result.dtype, dtype)

    @dtypes(*all_floating_dtypes)
    def test_determinism(self, device, dtype):
        key = random.key(42, device=device)
        a = random.uniform(key, (1000,), dtype=dtype)
        b = random.uniform(key, (1000,), dtype=dtype)
        self.assertEqual(a, b)

    @dtypes(*all_floating_dtypes)
    def test_different_keys(self, device, dtype):
        key1 = random.key(42, device=device)
        key2 = random.key(43, device=device)
        a = random.uniform(key1, (1000,), dtype=dtype)
        b = random.uniform(key2, (1000,), dtype=dtype)
        self.assertNotEqual(a, b)

    @dtypes(*all_floating_dtypes)
    def test_standard_uniform_statistics(self, device, dtype):
        key = random.key(42, device=device)
        result = random.uniform(key, (100000,), dtype=dtype)
        self.assertTrue(abs(result.mean().item() - 0.5) < 0.05)
        self.assertTrue(result.min().item() > 0.0)
        self.assertTrue(result.max().item() <= 1.0)

    @dtypes(*all_floating_dtypes)
    def test_custom_low_high(self, device, dtype):
        key = random.key(42, device=device)
        result = random.uniform(key, (100000,), low=2.0, high=5.0, dtype=dtype)
        self.assertTrue(abs(result.mean().item() - 3.5) < 0.1)
        self.assertTrue(result.min().item() >= 2.0)
        self.assertTrue(result.max().item() <= 5.0)

    @dtypes(*all_floating_dtypes)
    def test_batched_keys(self, device, dtype):
        key = random.key(42, device=device)
        keys = random.split(key, 4)  # (4, 2)
        result = random.uniform(keys, (4, 100), dtype=dtype)
        for i in range(4):
            individual = random.uniform(keys[i], (100,), dtype=dtype)
            self.assertEqual(result[i], individual)

    @dtypes(*all_floating_dtypes)
    def test_multi_batch(self, device, dtype):
        key = random.key(42, device=device)
        keys = random.split(key, 6).reshape(2, 3, 2)  # (2, 3, 2)
        result = random.uniform(keys, (2, 3, 50), dtype=dtype)
        for i in range(2):
            for j in range(3):
                individual = random.uniform(keys[i][j], (50,), dtype=dtype)
                self.assertEqual(result[i][j], individual)

    @dtypes(*all_floating_dtypes)
    def test_broadcasting(self, device, dtype):
        key = random.key(42, device=device).unsqueeze(0)  # (1, 2)
        result = random.uniform(key, (4, 100), dtype=dtype)
        for i in range(1, 4):
            self.assertEqual(result[0], result[i])

    def test_error_wrong_key_dtype(self, device):
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaises(RuntimeError):
            random.uniform(key, (100,))

    @onlyCUDA
    def test_error_wrong_device(self, device):
        key = random.key(42)  # CPU key
        with self.assertRaises(RuntimeError):
            random.uniform(key, (100,), device=device)

    def test_error_shape_mismatch(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 3)  # (3, 2)
        with self.assertRaises(RuntimeError):
            random.uniform(keys, (2, 100))  # batch dim 2 != 3

    def test_error_key_last_dim_not_2(self, device):
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaises(RuntimeError):
            random.uniform(key, (100,))

    @dtypes(torch.float32, torch.float64)
    def test_offset_shift_consistency(self, device, dtype):
        """Shifting key offset shifts the output stream."""
        seed = 42
        n = 100
        outputs_per_elem = 2 if dtype == torch.float64 else 1
        key0 = torch.tensor([seed, 0], dtype=torch.uint64, device=device)
        ref = random.uniform(key0, (n,), dtype=dtype)
        for elem_offset in range(1, 4):
            offset = elem_offset * outputs_per_elem
            key = torch.tensor([seed, offset], dtype=torch.uint64, device=device)
            result = random.uniform(key, (n - elem_offset,), dtype=dtype)
            self.assertEqual(result, ref[elem_offset:])

    @dtypes(torch.float32, torch.float64)
    def test_offset_overflow(self, device, dtype):
        """After wrapping past 2^64, generation continues from offset 0."""
        seed = 42
        outputs_per_elem = 2 if dtype == torch.float64 else 1
        wrap_at = 5
        near_max = (1 << 64) - wrap_at * outputs_per_elem
        key = torch.tensor([seed, near_max], dtype=torch.uint64, device=device)
        result = random.uniform(key, (20,), dtype=dtype)
        # First wrap_at elements come from the stream at near_max.
        self.assertEqual(
            result[:wrap_at],
            random.uniform(key, (wrap_at,), dtype=dtype),
        )
        # After the wrap, elements come from the stream at offset 0.
        key_zero = torch.tensor([seed, 0], dtype=torch.uint64, device=device)
        self.assertEqual(
            result[wrap_at:],
            random.uniform(key_zero, (20 - wrap_at,), dtype=dtype),
        )

    @onlyCUDA
    def test_cross_device_consistency(self, device):
        key_cpu = random.key(42)
        key_cuda = random.key(42, device=device)
        self.assertEqual(
            random.uniform(key_cpu, (1000,)),
            random.uniform(key_cuda, (1000,)).cpu(),
        )

    @onlyCUDA
    def test_cross_device_f64_consistency(self, device):
        key_cpu = random.key(42)
        key_cuda = random.key(42, device=device)
        self.assertEqual(
            random.uniform(key_cpu, (1000,), dtype=torch.float64),
            random.uniform(key_cuda, (1000,), dtype=torch.float64).cpu(),
        )


instantiate_device_type_tests(TestPhiloxUniform, globals(), only_for=("cpu", "cuda"))


class TestPhiloxCompile(TestCase):
    def test_uniform_aot_eager(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return random.uniform(key, (100,))

        self.assertEqual(f(key), random.uniform(key, (100,)))

    def test_split_aot_eager(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return random.split(key, 4)

        self.assertEqual(f(key), random.split(key, 4))

    def test_fold_in_aot_eager(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return random.fold_in(key, 7)

        self.assertEqual(f(key), random.fold_in(key, 7))

    def test_normal_aot_eager(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            return random.normal(key, (100,))

        self.assertEqual(f(key), random.normal(key, (100,)))

    def test_batched_normal_aot_eager(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 4)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(keys):
            return random.normal(keys, (4, 50))

        self.assertEqual(f(keys), random.normal(keys, (4, 50)))

    def test_split_then_normal_aot_eager(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            keys = random.split(key, 4)
            return random.normal(keys, (4, 100))

        self.assertEqual(
            f(key), random.normal(random.split(key, 4), (4, 100))
        )

    def test_fold_in_then_uniform_aot_eager(self, device):
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            k = random.fold_in(key, 3)
            return random.uniform(k, (100,))

        self.assertEqual(
            f(key), random.uniform(random.fold_in(key, 3), (100,))
        )


instantiate_device_type_tests(TestPhiloxCompile, globals(), only_for=("cpu", "cuda"))


class TestGridSplit(TestCase):
    def test_1d_shape(self, device):
        k = random.key(42, device=device)
        keys = random.grid_split(k, (100,), (10,))
        self.assertIsInstance(keys, random.Philox4x32_10Key)
        # 10 tiles, each of size 10
        self.assertEqual(keys.shape, (10, 2))

    def test_1d_uniform_reconstruction(self, device):
        k = random.key(42, device=device)
        num_tiles = 10
        keys = random.grid_split(k, (100,), (num_tiles,))
        full = random.uniform(k, (100,), device=device)
        tile_size = 100 // num_tiles
        tiled = torch.cat(
            [
                random.uniform(keys[i], (tile_size,), device=device)
                for i in range(num_tiles)
            ]
        )
        self.assertEqual(full, tiled)

    def test_1d_normal_reconstruction(self, device):
        k = random.key(42, device=device)
        num_tiles = 10
        keys = random.grid_split(k, (100,), (num_tiles,))
        full = random.normal(k, (100,), device=device)
        tile_size = 100 // num_tiles
        tiled = torch.cat(
            [
                random.normal(keys[i], (tile_size,), device=device)
                for i in range(num_tiles)
            ]
        )
        self.assertEqual(full, tiled)

    def test_1d_determinism(self, device):
        k = random.key(42, device=device)
        keys1 = random.grid_split(k, (100,), (10,))
        keys2 = random.grid_split(k, (100,), (10,))
        self.assertEqual(keys1, keys2)

    def test_2d_shape(self, device):
        k = random.key(42, device=device)
        splits = (10, 10)
        keys = random.grid_split(k, (100, 200), splits)
        # splits=(10,10), tile_shape=(10,20), per-tile rows=10
        # -> shape (*splits, tile_shape[0], 2) = (10, 10, 10, 2)
        self.assertEqual(keys.shape, (10, 10, 10, 2))

    def test_2d_uniform_tile(self, device):
        k = random.key(42, device=device)
        shape = (100, 200)
        splits = (10, 10)
        tile_shape = (10, 20)
        keys = random.grid_split(k, shape, splits)
        full = random.uniform(k, shape, device=device)
        tile = random.uniform(keys[0, 0], tile_shape, device=device)
        self.assertEqual(tile, full[0:10, 0:20])

    def test_2d_uniform_reconstruction(self, device):
        k = random.key(42, device=device)
        shape = (60, 80)
        splits = (6, 4)
        tile_shape = (10, 20)
        keys = random.grid_split(k, shape, splits)
        full = random.uniform(k, shape, device=device)
        tiles = []
        for r in range(splits[0]):
            row = []
            for c in range(splits[1]):
                row.append(
                    random.uniform(keys[r, c], tile_shape, device=device)
                )
            tiles.append(torch.cat(row, dim=1))
        tiled = torch.cat(tiles, dim=0)
        self.assertEqual(full, tiled)

    def test_2d_normal_reconstruction(self, device):
        k = random.key(42, device=device)
        shape = (60, 80)
        splits = (6, 4)
        tile_shape = (10, 20)
        keys = random.grid_split(k, shape, splits)
        full = random.normal(k, shape, device=device)
        tiles = []
        for r in range(splits[0]):
            row = []
            for c in range(splits[1]):
                row.append(
                    random.normal(keys[r, c], tile_shape, device=device)
                )
            tiles.append(torch.cat(row, dim=1))
        tiled = torch.cat(tiles, dim=0)
        self.assertEqual(full, tiled)

    def test_2d_arbitrary_tile(self, device):
        """Verify a non-corner tile matches the full generation."""
        k = random.key(123, device=device)
        shape = (100, 200)
        splits = (10, 10)
        tile_shape = (10, 20)
        keys = random.grid_split(k, shape, splits)
        full = random.uniform(k, shape, device=device)
        for tr, tc in [(3, 7), (9, 9), (0, 5)]:
            tile = random.uniform(keys[tr, tc], tile_shape, device=device)
            expected = full[
                tr * tile_shape[0] : (tr + 1) * tile_shape[0],
                tc * tile_shape[1] : (tc + 1) * tile_shape[1],
            ]
            self.assertEqual(tile, expected)

    def test_3d_shape(self, device):
        k = random.key(42, device=device)
        splits = (3, 4, 3)
        keys = random.grid_split(k, (12, 20, 30), splits)
        # tile_shape = (4, 5, 10), per-tile rows = (4, 5)
        # -> shape (*splits, 4, 5, 2) = (3, 4, 3, 4, 5, 2)
        self.assertEqual(keys.shape, (3, 4, 3, 4, 5, 2))

    def test_3d_uniform_reconstruction(self, device):
        k = random.key(77, device=device)
        shape = (12, 20, 30)
        splits = (3, 4, 3)
        tile_shape = tuple(s // sp for s, sp in zip(shape, splits))
        keys = random.grid_split(k, shape, splits)
        full = random.uniform(k, shape, device=device)
        reconstructed = torch.empty_like(full)
        for t0 in range(splits[0]):
            for t1 in range(splits[1]):
                for t2 in range(splits[2]):
                    tile = random.uniform(
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
        k = random.key(42, device=device)
        splits = (10, 1)
        keys = random.grid_split(k, (100, 200), splits)
        tile_shape = (10, 200)
        # shape: (*splits, tile_shape[0], 2) = (10, 1, 10, 2)
        self.assertEqual(keys.shape, (10, 1, 10, 2))
        full = random.uniform(k, (100, 200), device=device)
        tiles = [
            random.uniform(keys[i, 0], tile_shape, device=device)
            for i in range(10)
        ]
        tiled = torch.cat(tiles, dim=0)
        self.assertEqual(full, tiled)

    @dtypes(torch.float32, torch.float64)
    def test_1d_near_max_offset(self, device, dtype):
        """grid_split reconstruction holds when tile offsets wrap past 2^64."""
        seed = 42
        near_max_offset = (1 << 64) - 48
        k = torch.tensor(
            [seed, near_max_offset], dtype=torch.uint64, device=device
        )
        shape = (100,)
        num_tiles = 10
        tile_size = shape[0] // num_tiles
        keys = random.grid_split(k, shape, (num_tiles,), dtype=dtype)
        full_uniform = random.uniform(k, shape, dtype=dtype, device=device)
        tiled_uniform = torch.cat(
            [
                random.uniform(keys[i], (tile_size,), dtype=dtype, device=device)
                for i in range(num_tiles)
            ]
        )
        self.assertEqual(full_uniform, tiled_uniform)
        full_normal = random.normal(k, shape, dtype=dtype, device=device)
        tiled_normal = torch.cat(
            [
                random.normal(keys[i], (tile_size,), dtype=dtype, device=device)
                for i in range(num_tiles)
            ]
        )
        self.assertEqual(full_normal, tiled_normal)

    def test_error_uneven_split(self, device):
        k = random.key(42, device=device)
        with self.assertRaisesRegex(ValueError, "does not evenly divide"):
            random.grid_split(k, (100,), (3,))
        with self.assertRaisesRegex(ValueError, "does not evenly divide"):
            random.grid_split(k, (100, 200), (10, 3))

    def test_error_mismatched_lengths(self, device):
        k = random.key(42, device=device)
        with self.assertRaisesRegex(ValueError, "same length"):
            random.grid_split(k, (100, 200), (10,))

    @onlyCUDA
    def test_cross_device_consistency_1d(self, device):
        k_cpu = random.key(42)
        k_cuda = random.key(42, device=device)
        keys_cpu = random.grid_split(k_cpu, (100,), (10,))
        keys_cuda = random.grid_split(k_cuda, (100,), (10,))
        self.assertEqual(keys_cpu, keys_cuda.cpu())

    @onlyCUDA
    def test_cross_device_consistency_2d(self, device):
        k_cpu = random.key(42)
        k_cuda = random.key(42, device=device)
        shape = (60, 80)
        splits = (6, 4)
        tile_shape = (10, 20)
        keys_cpu = random.grid_split(k_cpu, shape, splits)
        keys_cuda = random.grid_split(k_cuda, shape, splits)
        tile_cpu = random.uniform(keys_cpu[2, 1], tile_shape)
        tile_cuda = random.uniform(
            keys_cuda[2, 1], tile_shape, device=device
        )
        self.assertEqual(tile_cpu, tile_cuda.cpu())


instantiate_device_type_tests(TestGridSplit, globals(), only_for=("cpu", "cuda"))


class TestPhiloxVmap(TestCase):
    def test_vmap_normal(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 10)

        result = torch.vmap(lambda k: random.normal(k, 5))(keys)
        self.assertEqual(result.shape, (10, 5))
        for i in range(10):
            self.assertEqual(result[i], random.normal(keys[i], 5))

    def test_vmap_uniform(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 10)

        result = torch.vmap(lambda k: random.uniform(k, 5))(keys)
        self.assertEqual(result.shape, (10, 5))
        for i in range(10):
            self.assertEqual(result[i], random.uniform(keys[i], 5))

    def test_vmap_split(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 10)

        result = torch.vmap(lambda k: random.split(k, 3))(keys)
        self.assertEqual(result.shape, (10, 3, 2))
        for i in range(10):
            self.assertEqual(result[i], random.split(keys[i], 3))

    def test_vmap_fold_in(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 10)

        result = torch.vmap(lambda k: random.fold_in(k, 7))(keys)
        self.assertEqual(result.shape, (10, 2))
        for i in range(10):
            self.assertEqual(result[i], random.fold_in(keys[i], 7))

    def test_vmap_inplace_batched_self(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 10)
        out = torch.empty(10, 5, device=device)

        def f(o, k):
            return torch.ops.aten._philox_normal_(o, k, 0.0, 1.0)

        result = torch.vmap(f)(out, keys)
        self.assertEqual(result.shape, (10, 5))
        for i in range(10):
            self.assertEqual(result[i], random.normal(keys[i], 5))

    def test_vmap_split_then_normal(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 8)

        def f(k):
            subkeys = random.split(k, 3)
            return random.normal(subkeys, (3, 20))

        result = torch.vmap(f)(keys)
        self.assertEqual(result.shape, (8, 3, 20))
        for i in range(8):
            self.assertEqual(result[i], f(keys[i]))

    def test_vmap_normal_multidim(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 5)

        result = torch.vmap(lambda k: random.normal(k, 4, 3))(keys)
        self.assertEqual(result.shape, (5, 4, 3))
        for i in range(5):
            self.assertEqual(result[i], random.normal(keys[i], 4, 3))

    def test_vmap_compiled_normal(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 10)

        @torch.compile(backend="aot_eager")
        def f(keys):
            return torch.vmap(lambda k: random.normal(k, 5))(keys)

        result = f(keys)
        self.assertEqual(result.shape, (10, 5))
        for i in range(10):
            self.assertEqual(result[i], random.normal(keys[i], 5))

    def test_vmap_compiled_uniform(self, device):
        key = random.key(42, device=device)
        keys = random.split(key, 10)

        @torch.compile(backend="aot_eager")
        def f(keys):
            return torch.vmap(lambda k: random.uniform(k, 5))(keys)

        result = f(keys)
        self.assertEqual(result.shape, (10, 5))
        for i in range(10):
            self.assertEqual(result[i], random.uniform(keys[i], 5))


instantiate_device_type_tests(TestPhiloxVmap, globals(), only_for=("cpu", "cuda"))


if __name__ == "__main__":
    run_tests()
