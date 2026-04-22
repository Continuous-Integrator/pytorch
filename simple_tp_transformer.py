"""
Minimal tensor-parallel transformer with one all-gather and one reduce-scatter.

    torchrun --nproc_per_node=2 simple_tp_transformer.py
    python -m torch.distributed.torchmux --nproc 2 simple_tp_transformer.py

Architecture: embedding -> transformer block (attention + TP FFN) -> RMS norm -> linear
The FFN uses column-parallel w1 / row-parallel w2 with sequence parallelism,
giving exactly one all-gather (before w1) and one reduce-scatter (after w2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def rms_norm(x, weight, eps=1e-6):
    return weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def all_gather_seq(x, group):
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, x, group=group)
    return torch.cat(gathered, dim=1)


def reduce_scatter_seq(x, group):
    chunks = list(x.chunk(dist.get_world_size(group), dim=1))
    output = torch.empty_like(chunks[0])
    dist.reduce_scatter(output, chunks, op=dist.ReduceOp.SUM, group=group)
    return output


class SimpleTPTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, rank, world_size, group):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.attn_norm_w = nn.Parameter(torch.ones(d_model))
        self.ffn_norm_w = nn.Parameter(torch.ones(d_model))
        self.final_norm_w = nn.Parameter(torch.ones(d_model))

        assert d_ff % world_size == 0
        local_d_ff = d_ff // world_size
        self.w1 = nn.Linear(d_model, local_d_ff, bias=False)
        self.w2 = nn.Linear(local_d_ff, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

    def _attention(self, x):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))
        out = (scores.softmax(dim=-1) @ v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)

    def forward(self, tokens):
        x = self.tok_emb(tokens)

        # Attention block (replicated across ranks)
        x = x + self._attention(rms_norm(x, self.attn_norm_w))

        # Transition to sequence parallelism: each rank takes its chunk
        x_local = x.chunk(self.world_size, dim=1)[self.rank].contiguous()

        # FFN with tensor parallelism + sequence parallelism
        normed_local = rms_norm(x_local, self.ffn_norm_w)

        # All-gather: [B, S/W, D] -> [B, S, D]
        normed_full = all_gather_seq(normed_local, self.group)

        # Column-parallel w1 (each rank has d_ff/W output columns)
        h = F.silu(self.w1(normed_full))

        # Row-parallel w2 (each rank produces partial sum over full D)
        h = self.w2(h)

        # Reduce-scatter: sum partials across ranks + scatter along seq dim
        # [B, S, D] -> [B, S/W, D]
        h_local = reduce_scatter_seq(h, self.group)

        x_local = x_local + h_local

        # Final RMS norm + output projection on local chunk
        logits = self.out_proj(rms_norm(x_local, self.final_norm_w))
        return logits


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    vocab_size = 256
    d_model = 64
    n_heads = 4
    d_ff = 128
    batch_size = 2
    seq_len = 16
    lr = 1e-3

    torch.manual_seed(42)
    model = SimpleTPTransformer(
        vocab_size, d_model, n_heads, d_ff, rank, world_size, dist.group.WORLD
    ).to(device)

    torch.manual_seed(0)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    for i in range(5):
        logits = model(tokens)
        loss = logits.sum()
        loss.backward()
        print(f"[Rank {rank}] iter {i}: loss = {loss.item():.6f}")

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad
                    p.grad.zero_()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
