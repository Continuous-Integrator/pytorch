"""
Minimal repro: Compiling embedding layer with aot_eager backend
causes index out of range errors.

Focus: Simulate vLLM warmup pattern without mark_unbacked.
"""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=64):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids):
        # Simulate vLLM wrapper pattern: unsqueeze to 2D
        tokens_2d = input_ids.unsqueeze(0)
        h = self.tok_embeddings(tokens_2d)
        h = self.linear(h)
        # Flatten back
        if h.dim() == 3:
            b, s, d = h.shape
            h = h.view(b * s, d)
        return h


def test_basic():
    """Basic test with aot_eager."""
    print("=" * 60)
    print("Test 1: Basic aot_eager with varying 1D input sizes")
    print("=" * 60)

    model = SimpleModel(vocab_size=1000, embed_dim=64).cuda()
    compiled_model = torch.compile(model, backend="aot_eager")

    # 1D inputs like vLLM uses
    sizes = [16, 32, 64, 128]
    for size in sizes:
        input_ids = torch.randint(0, 1000, (size,), device="cuda")
        print(f"Input shape: {input_ids.shape}")
        out = compiled_model(input_ids)
        print(f"  Output shape: {out.shape}")

    print("✓ Passed")


def test_warmup_pattern():
    """
    Simulate vLLM warmup: first call with large dummy, then small real inputs.
    """
    print("\n" + "=" * 60)
    print("Test 2: vLLM warmup pattern - large then small")
    print("=" * 60)

    model = SimpleModel(vocab_size=1000, embed_dim=64).cuda()
    compiled_model = torch.compile(model, backend="aot_eager")

    # Large warmup (like vLLM dummy run)
    warmup_size = 16384
    warmup_input = torch.randint(0, 1000, (warmup_size,), device="cuda")
    print(f"Warmup input shape: {warmup_input.shape}")
    out = compiled_model(warmup_input)
    print(f"  Warmup output shape: {out.shape}")

    # Small real input
    real_size = 32
    real_input = torch.randint(0, 1000, (real_size,), device="cuda")
    print(f"Real input shape: {real_input.shape}")
    out = compiled_model(real_input)
    print(f"  Real output shape: {out.shape}")

    print("✓ Passed")


def test_with_positions():
    """
    Simulate the full vLLM wrapper pattern with positions and .item() call.
    """
    print("\n" + "=" * 60)
    print("Test 3: With positions and .item() (graph break)")
    print("=" * 60)

    class ModelWithPositions(nn.Module):
        def __init__(self, vocab_size=1000, embed_dim=64):
            super().__init__()
            self.tok_embeddings = nn.Embedding(vocab_size, embed_dim)
            self.linear = nn.Linear(embed_dim, embed_dim)

        def forward(self, input_ids, positions):
            tokens_2d = input_ids.unsqueeze(0)
            h = self.tok_embeddings(tokens_2d)

            # This causes a graph break (like in vLLM wrapper)
            if positions is not None:
                max_position = positions.max().item()
            else:
                max_position = 0

            positions_2d = positions.unsqueeze(0)
            h = self.linear(h)

            if h.dim() == 3:
                b, s, d = h.shape
                h = h.view(b * s, d)
            return h

    model = ModelWithPositions(vocab_size=1000, embed_dim=64).cuda()
    compiled_model = torch.compile(model, backend="aot_eager")

    sizes = [16, 32, 64]
    for size in sizes:
        input_ids = torch.randint(0, 1000, (size,), device="cuda")
        positions = torch.arange(size, device="cuda")
        print(f"Input shape: {input_ids.shape}, positions: {positions.shape}")
        out = compiled_model(input_ids, positions)
        print(f"  Output shape: {out.shape}")

    print("✓ Passed")


def test_dynamo_disable_embedding():
    """
    Test with embedding excluded from dynamo (like current vLLM wrapper).
    """
    print("\n" + "=" * 60)
    print("Test 4: Embedding excluded with @torch._dynamo.disable")
    print("=" * 60)

    class ModelWithDisabledEmbed(nn.Module):
        def __init__(self, vocab_size=1000, embed_dim=64):
            super().__init__()
            self.tok_embeddings = nn.Embedding(vocab_size, embed_dim)
            self.linear = nn.Linear(embed_dim, embed_dim)

        @torch._dynamo.disable
        def _embed_tokens(self, tokens):
            return self.tok_embeddings(tokens)

        def forward(self, input_ids, positions):
            tokens_2d = input_ids.unsqueeze(0)
            h = self._embed_tokens(tokens_2d)

            if positions is not None:
                max_position = positions.max().item()
            else:
                max_position = 0

            positions_2d = positions.unsqueeze(0)
            h = self.linear(h)

            if h.dim() == 3:
                b, s, d = h.shape
                h = h.view(b * s, d)
            return h

    model = ModelWithDisabledEmbed(vocab_size=1000, embed_dim=64).cuda()
    compiled_model = torch.compile(model, backend="aot_eager")

    # Warmup
    warmup_ids = torch.randint(0, 1000, (16384,), device="cuda")
    warmup_pos = torch.arange(16384, device="cuda")
    print(f"Warmup: {warmup_ids.shape}")
    out = compiled_model(warmup_ids, warmup_pos)
    print(f"  Output: {out.shape}")

    # Real
    sizes = [16, 32, 64]
    for size in sizes:
        input_ids = torch.randint(0, 1000, (size,), device="cuda")
        positions = torch.arange(size, device="cuda")
        print(f"Real input: {input_ids.shape}")
        out = compiled_model(input_ids, positions)
        print(f"  Output: {out.shape}")

    print("✓ Passed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_num = sys.argv[1]
        if test_num == "1":
            test_basic()
        elif test_num == "2":
            test_warmup_pattern()
        elif test_num == "3":
            test_with_positions()
        elif test_num == "4":
            test_dynamo_disable_embedding()
    else:
        test_basic()
        test_warmup_pattern()
        test_with_positions()
        test_dynamo_disable_embedding()
