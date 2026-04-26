"""Benchmark to measure DTensor sharding propagation cache impact."""

import time
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh


def bench_op(fn, warmup=10, iters=1000):
    """Generic benchmark helper."""
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1e6  # microseconds per op


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    sizes = [64, 256, 1024, 4096]

    if rank == 0:
        print("Comparing local tensor ops vs DTensor cache hits")
        print(f"{'Size':>8} | {'Local (us)':>12} | {'DTensor (us)':>13} | {'Overhead (us)':>14} | {'Overhead %':>11}")
        print("-" * 75)

    for size in sizes:
        t1 = torch.randn(size, size, device="cuda")
        t2 = torch.randn(size, size, device="cuda")

        dt1 = distribute_tensor(t1.clone(), mesh, [Shard(0)])
        dt2 = distribute_tensor(t2.clone(), mesh, [Shard(0)])

        local1 = dt1.to_local()
        local2 = dt2.to_local()

        local_time = bench_op(lambda: local1 + local2)
        dt_time = bench_op(lambda: dt1 + dt2)
        overhead = dt_time - local_time
        overhead_pct = (overhead / local_time) * 100 if local_time > 0 else 0

        if rank == 0:
            print(f"{size:>8} | {local_time:>12.2f} | {dt_time:>13.2f} | {overhead:>14.2f} | {overhead_pct:>10.1f}%")

    if rank == 0:
        print("\n\nBenchmarking different ops (cache hit) at size 1024:")
        print(f"{'Op':>20} | {'Local (us)':>12} | {'DTensor (us)':>13} | {'Overhead (us)':>14}")
        print("-" * 70)

    size = 1024
    t1 = torch.randn(size, size, device="cuda")
    t2 = torch.randn(size, size, device="cuda")
    dt1 = distribute_tensor(t1.clone(), mesh, [Shard(0)])
    dt2 = distribute_tensor(t2.clone(), mesh, [Shard(0)])
    local1 = dt1.to_local()
    local2 = dt2.to_local()

    ops = [
        ("add", lambda l1, l2: l1 + l2),
        ("mul", lambda l1, l2: l1 * l2),
        ("relu", lambda l1, l2: torch.relu(l1)),
        ("sum", lambda l1, l2: l1.sum()),
        ("mean", lambda l1, l2: l1.mean()),
        ("transpose", lambda l1, l2: l1.t()),
        ("clone", lambda l1, l2: l1.clone()),
    ]

    for name, op in ops:
        local_time = bench_op(lambda: op(local1, local2))
        dt_time = bench_op(lambda: op(dt1, dt2))
        overhead = dt_time - local_time

        if rank == 0:
            print(f"{name:>20} | {local_time:>12.2f} | {dt_time:>13.2f} | {overhead:>14.2f}")

    # First-call overhead (cache miss vs subsequent cache hits)
    if rank == 0:
        print("\n\nFirst-call overhead (cache miss vs subsequent cache hits):")

    test_ops = [
        ("sin", lambda dt: torch.sin(dt)),
        ("cos", lambda dt: torch.cos(dt)),
        ("exp", lambda dt: torch.exp(dt)),
        ("log1p", lambda dt: torch.log1p(torch.abs(dt) + 1)),
        ("tanh", lambda dt: torch.tanh(dt)),
    ]

    t3 = torch.randn(size, size, device="cuda")
    dt3 = distribute_tensor(t3, mesh, [Shard(0)])

    if rank == 0:
        print(f"{'Op':>12} | {'First (us)':>12} | {'Subseq (us)':>12} | {'Miss overhead':>14}")
        print("-" * 60)

    for name, op in test_ops:
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = op(dt3)
        torch.cuda.synchronize()
        first_call = (time.perf_counter() - start) * 1e6

        subsequent_time = bench_op(lambda: op(dt3))

        if rank == 0:
            print(f"{name:>12} | {first_call:>12.2f} | {subsequent_time:>12.2f} | {first_call - subsequent_time:>14.2f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
