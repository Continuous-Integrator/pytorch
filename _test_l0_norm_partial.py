"""Test that L0 norm with Partial(sum) input produces correct results.

Run with: torchrun --nproc_per_node=2 /tmp/test_l0_norm_partial.py
"""
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Partial, Replicate, init_device_mesh

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

mesh = init_device_mesh("cuda", (2,))

# Each rank holds a partial value; sum across ranks = full tensor
# Full tensor: [0, 3, 0, 5] -> L0 norm = 2 (two nonzero elements)
if rank == 0:
    local = torch.tensor([0.0, 1.0, 0.0, 2.0], device="cuda")
else:
    local = torch.tensor([0.0, 2.0, 0.0, 3.0], device="cuda")

dt = DTensor.from_local(local, mesh, (Partial("sum"),))

# Verify the full tensor is what we expect
full = dt.full_tensor()
if rank == 0:
    print(f"Full tensor: {full}")
    print(f"Expected L0 norm: {torch.linalg.vector_norm(full, ord=0)}")

# Compute L0 norm on the Partial DTensor
result = torch.linalg.vector_norm(dt, ord=0)
result_full = result.full_tensor()

if rank == 0:
    correct = torch.linalg.vector_norm(full, ord=0)
    print(f"DTensor L0 norm: {result_full}")
    print(f"Correct L0 norm: {correct}")
    print(f"Match: {torch.allclose(correct, result_full)}")

dist.destroy_process_group()
