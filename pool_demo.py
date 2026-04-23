import torch
from torch._inductor import config
print("compile_threads:", config.compile_threads)


@torch.compile(mode="max-autotune-no-cudagraphs")
def f(x, y):
    # Mix of pointwise, reductions, matmul — each becomes a distinct kernel
    a = (x * y + x.sin()).relu()
    b = a.softmax(-1)
    c = b @ y
    d = c.mean(-1, keepdim=True)
    e = (d - c).pow(2).sum()
    return e + a.sum() + b.sum() + c.sum()


x = torch.randn(256, 256, device="cuda")
y = torch.randn(256, 256, device="cuda")
f(x, y)  # triggers compile
