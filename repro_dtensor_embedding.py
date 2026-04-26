"""
Minimal repro: Compiling embedding layer with DTensor + aot_eager backend.
"""

import os
import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor, Replicate, Shard, DeviceMesh, distribute_tensor


class SimpleModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids):
        h = self.embedding(input_ids)
        h = self.linear(h)
        return h


def test_dtensor_embedding_compile():
    """
    Test compiling a model with DTensor embedding using aot_eager.
    """
    print("=" * 60)
    print("Test: DTensor + aot_eager with embedding")
    print("=" * 60)

    # Initialize distributed
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print(f"Rank {rank}/{world_size} on device {device}")

    # Create device mesh for TP
    mesh = DeviceMesh("cuda", list(range(world_size)))

    # Create model
    model = SimpleModel(vocab_size=1000, embed_dim=64).to(device)

    # Distribute embedding weight with TP (shard on vocab dimension)
    # This is similar to what vLLM does
    with torch.no_grad():
        model.embedding.weight = nn.Parameter(
            distribute_tensor(
                model.embedding.weight,
                mesh,
                [Shard(0)],  # Shard vocab dimension
            )
        )

    print(f"Embedding weight is DTensor: {isinstance(model.embedding.weight, DTensor)}")
    print(f"Embedding weight shape: {model.embedding.weight.shape}")

    # Compile with aot_eager
    compiled_model = torch.compile(model, backend="aot_eager")

    # Test with varying sizes
    sizes = [16, 32, 64]
    for size in sizes:
        # Create input with valid indices for sharded vocab
        # With Shard(0), each rank has vocab_size/world_size tokens
        local_vocab_size = 1000 // world_size
        input_ids = torch.randint(0, local_vocab_size, (1, size), device=device)

        print(f"Rank {rank}: Running with size {size}, max_idx={input_ids.max().item()}...")
        try:
            out = compiled_model(input_ids)
            # Handle DTensor output
            if isinstance(out, DTensor):
                out = out.to_local()
            print(f"  Output shape: {out.shape}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"✓ Rank {rank} completed")
    torch.distributed.barrier()


def test_replicated_embedding_compile():
    """
    Test compiling a model with replicated DTensor embedding using aot_eager.
    """
    print("\n" + "=" * 60)
    print("Test: Replicated DTensor + aot_eager with embedding")
    print("=" * 60)

    # Initialize distributed
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create device mesh
    mesh = DeviceMesh("cuda", list(range(world_size)))

    # Create model
    model = SimpleModel(vocab_size=1000, embed_dim=64).to(device)

    # Distribute embedding weight as Replicate
    with torch.no_grad():
        model.embedding.weight = nn.Parameter(
            distribute_tensor(
                model.embedding.weight,
                mesh,
                [Replicate()],
            )
        )

    print(f"Embedding weight is DTensor: {isinstance(model.embedding.weight, DTensor)}")

    # Compile with aot_eager
    compiled_model = torch.compile(model, backend="aot_eager")

    # Test with varying sizes
    sizes = [16, 32, 64]
    for size in sizes:
        input_ids = torch.randint(0, 1000, (1, size), device=device)

        print(f"Rank {rank}: Running with size {size}...")
        try:
            out = compiled_model(input_ids)
            if isinstance(out, DTensor):
                out = out.to_local()
            print(f"  Output shape: {out.shape}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"✓ Rank {rank} completed")
    torch.distributed.barrier()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "shard":
        test_dtensor_embedding_compile()
    else:
        test_replicated_embedding_compile()
