"""Compare torchlite vs torch.compile across a broad model sample.

Inference uses torchlite's full inference pipeline:
trace -> inference_passes -> inference codegen.
This reflects the optimized torchlite path rather than raw FX execution.

Training uses torch._torchlite.compile() on explicit TrainStep wrappers.

Run with:
    python -m torch._torchlite.examples.bench_compare

Environment variables:
    TORCHLITE_BENCH_DEVICE=cuda   (default: cuda if available, else cpu)
    TORCHLITE_BENCH_WARMUP=5
    TORCHLITE_BENCH_ITERS=50
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch._torchlite import (
    codegen,
    compile as torchlite_compile,
    inference_passes,
    timed_run_passes,
    trace,
)


class SimpleLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return x @ self.weight + self.bias


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(in_dim, hidden_dim) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = nn.Parameter(torch.randn(hidden_dim, out_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        h = torch.relu(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2


class FourLayerMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.w2 = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.w3 = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.w4 = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def forward(self, x):
        x = torch.relu(x @ self.w1)
        x = torch.relu(x @ self.w2)
        x = torch.relu(x @ self.w3)
        return x @ self.w4


class EightLayerMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim) * 0.01) for _ in range(8)
        ])

    def forward(self, x):
        for i, w in enumerate(self.layers):
            x = x @ w
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.wq = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.wk = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.wv = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.wo = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.w1 = nn.Parameter(torch.randn(dim, dim * 4) * 0.01)
        self.w2 = nn.Parameter(torch.randn(dim * 4, dim) * 0.01)
        self.n_heads = n_heads

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        head_dim = dim // self.n_heads
        q = (x @ self.wq).reshape(bsz, seqlen, self.n_heads, head_dim).transpose(1, 2)
        k = (x @ self.wk).reshape(bsz, seqlen, self.n_heads, head_dim).transpose(1, 2)
        v = (x @ self.wv).reshape(bsz, seqlen, self.n_heads, head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bsz, seqlen, dim)
        x = x + out @ self.wo
        h = torch.relu(x @ self.w1)
        x = x + h @ self.w2
        return x


class LlamaFFN(nn.Module):
    def __init__(self, dim, hidden_mult=4):
        super().__init__()
        hidden = dim * hidden_mult
        self.norm = nn.RMSNorm(dim)
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class GQABlock(nn.Module):
    def __init__(self, dim, n_heads=8, n_kv_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        q = (
            self.q_proj(x)
            .view(bsz, seqlen, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(bsz, seqlen, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.out_proj(attn)


class SiLUMLP(nn.Module):
    def __init__(self, dim, hidden_mult=4):
        super().__init__()
        hidden = dim * hidden_mult
        self.w_gate = nn.Parameter(torch.randn(dim, hidden) * 0.01)
        self.w_up = nn.Parameter(torch.randn(dim, hidden) * 0.01)
        self.w_down = nn.Parameter(torch.randn(hidden, dim) * 0.01)

    def forward(self, x):
        gate = torch.sigmoid(x @ self.w_gate) * (x @ self.w_up)
        return gate @ self.w_down


class TrainStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, target):
        out = self.model(x)
        if out.dim() == 3:
            out = out.mean(dim=1)
        return ((out - target) ** 2).mean()


def _synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _time_fn(fn, args, warmup, iters, device):
    for _ in range(warmup):
        fn(*args)
    _synchronize(device)

    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(*args)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        elapsed_ms = (time.perf_counter() - t0) * 1000

    return elapsed_ms / iters


def _clone_inputs(inputs):
    return [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]


def bench_inference(name, model_fn, input_fn, device, warmup, iters):
    torch.manual_seed(0)
    model = model_fn().to(device).eval()
    inp = [x.to(device) for x in input_fn()]
    eager_fn = lambda *args: model(*args)

    with torch.no_grad():
        eager_ms = _time_fn(eager_fn, inp, warmup, iters, device)

    torch.manual_seed(0)
    model_tl = model_fn().to(device).eval()
    tl_inputs = _clone_inputs(inp)
    use_inference_codegen = device == "cuda"

    t0 = time.perf_counter()
    gm = trace(model_tl, tl_inputs)
    gm, pass_timings = timed_run_passes(
        gm,
        tl_inputs,
        pipeline=inference_passes(gm, tl_inputs),
    )
    tl_fn = (
        codegen(gm, inference_codegen=True, example_inputs=tl_inputs)
        if use_inference_codegen
        else gm
    )
    tl_runner = lambda *args: tl_fn(*args)
    tl_compile_ms = (time.perf_counter() - t0) * 1000

    with torch.no_grad():
        _synchronize(device)
        t0 = time.perf_counter()
        tl_out = tl_runner(*inp)
        _synchronize(device)
        tl_compile_ms += (time.perf_counter() - t0) * 1000
        tl_ms = _time_fn(tl_runner, inp, warmup, iters, device)
        ref = eager_fn(*inp)
    torch.testing.assert_close(tl_out, ref, atol=1e-3, rtol=1e-3)

    torch._dynamo.reset()
    torch.manual_seed(0)
    model_tc = model_fn().to(device).eval()
    tc_eager = lambda *args: model_tc(*args)
    t0 = time.perf_counter()
    tc_compiled = torch.compile(tc_eager, fullgraph=True)
    with torch.no_grad():
        tc_out = tc_compiled(*inp)
        _synchronize(device)
        tc_compile_ms = (time.perf_counter() - t0) * 1000
        tc_ms = _time_fn(tc_compiled, inp, warmup, iters, device)

    torch.testing.assert_close(tc_out, ref, atol=1e-3, rtol=1e-3)

    hottest_pass = max(pass_timings.items(), key=lambda kv: kv[1])[0] if pass_timings else "-"
    return {
        "name": name,
        "mode": "inference",
        "eager_ms": eager_ms,
        "tl_ms": tl_ms,
        "tl_compile_ms": tl_compile_ms,
        "tc_ms": tc_ms,
        "tc_compile_ms": tc_compile_ms,
        "note": hottest_pass,
    }


def bench_training(name, model_fn, input_fn, device, warmup, iters):
    torch.manual_seed(0)
    model_eager = model_fn().to(device)
    inp = [x.to(device) for x in input_fn()]
    opt = torch.optim.SGD(model_eager.parameters(), lr=0.01)

    def eager_step():
        opt.zero_grad()
        loss = model_eager(*inp)
        loss.backward()
        opt.step()
        return loss

    eager_ms = _time_fn(eager_step, [], warmup, iters, device)

    torch.manual_seed(0)
    model_tl = model_fn().to(device)
    inp_tl = [x.to(device) for x in input_fn()]
    t0 = time.perf_counter()
    tl_compiled = torchlite_compile(model_tl, inp_tl)
    tl_compile_ms = (time.perf_counter() - t0) * 1000

    _synchronize(device)
    t0 = time.perf_counter()
    tl_compiled(*inp_tl)
    _synchronize(device)
    tl_compile_ms += (time.perf_counter() - t0) * 1000
    tl_ms = _time_fn(tl_compiled, inp_tl, warmup, iters, device)

    torch._dynamo.reset()
    torch.manual_seed(0)
    model_tc = model_fn().to(device)
    inp_tc = [x.to(device) for x in input_fn()]
    opt_tc = torch.optim.SGD(model_tc.parameters(), lr=0.01)

    def tc_step_eager():
        opt_tc.zero_grad()
        loss = model_tc(*inp_tc)
        loss.backward()
        opt_tc.step()
        return loss

    t0 = time.perf_counter()
    tc_step = torch.compile(tc_step_eager, fullgraph=False)
    tc_step()
    _synchronize(device)
    tc_compile_ms = (time.perf_counter() - t0) * 1000
    tc_ms = _time_fn(tc_step, [], warmup, iters, device)

    return {
        "name": name,
        "mode": "training",
        "eager_ms": eager_ms,
        "tl_ms": tl_ms,
        "tl_compile_ms": tl_compile_ms,
        "tc_ms": tc_ms,
        "tc_compile_ms": tc_compile_ms,
        "note": "-",
    }


def print_results(results):
    hdr = (
        f"  {'Model':<34s}  {'Mode':<10s}  "
        f"{'Eager ms':>10s}  "
        f"{'TL ms':>10s}  {'TL spdup':>8s}  {'TL comp':>10s}  "
        f"{'TC ms':>10s}  {'TC spdup':>8s}  {'TC comp':>10s}  "
        f"{'TL/TC':>7s}  {'Note':<18s}"
    )
    print()
    print("=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for r in results:
        tl_spd = r["eager_ms"] / r["tl_ms"] if r["tl_ms"] > 0 else float("inf")
        tc_spd = r["eager_ms"] / r["tc_ms"] if r["tc_ms"] > 0 else float("inf")
        tl_tc = r["tl_ms"] / r["tc_ms"] if r["tc_ms"] > 0 else float("inf")
        print(
            f"  {r['name']:<34s}  {r['mode']:<10s}  "
            f"{r['eager_ms']:>10.3f}  "
            f"{r['tl_ms']:>10.3f}  {tl_spd:>7.2f}x  {r['tl_compile_ms']:>9.1f}ms  "
            f"{r['tc_ms']:>10.3f}  {tc_spd:>7.2f}x  {r['tc_compile_ms']:>9.1f}ms  "
            f"{tl_tc:>6.2f}x  {r['note']:<18s}"
        )
    print("=" * len(hdr))


INFERENCE_MODELS = {
    "SimpleLinear(128->64)": (
        lambda: SimpleLinear(128, 64),
        lambda: [torch.randn(16, 128)],
    ),
    "TwoLayerMLP(256->512->128)": (
        lambda: TwoLayerMLP(256, 512, 128),
        lambda: [torch.randn(32, 256)],
    ),
    "FourLayerMLP(512)": (
        lambda: FourLayerMLP(512),
        lambda: [torch.randn(32, 512)],
    ),
    "EightLayerMLP(256)": (
        lambda: EightLayerMLP(256),
        lambda: [torch.randn(32, 256)],
    ),
    "SiLUMLP(512)": (
        lambda: SiLUMLP(512),
        lambda: [torch.randn(32, 512)],
    ),
    "LlamaFFN(512)": (
        lambda: LlamaFFN(512),
        lambda: [torch.randn(4, 64, 512)],
    ),
    "GQABlock(256,8/2)": (
        lambda: GQABlock(256, n_heads=8, n_kv_heads=2),
        lambda: [torch.randn(2, 64, 256)],
    ),
    "TransformerBlock(256,4h)": (
        lambda: SimpleTransformerBlock(256, n_heads=4),
        lambda: [torch.randn(4, 64, 256)],
    ),
}

TRAINING_MODELS = {
    "Train SimpleLinear(128->64)": (
        lambda: TrainStep(SimpleLinear(128, 64)),
        lambda: [torch.randn(16, 128), torch.randn(16, 64)],
    ),
    "Train TwoLayerMLP(256->512->128)": (
        lambda: TrainStep(TwoLayerMLP(256, 512, 128)),
        lambda: [torch.randn(32, 256), torch.randn(32, 128)],
    ),
    "Train FourLayerMLP(256)": (
        lambda: TrainStep(FourLayerMLP(256)),
        lambda: [torch.randn(32, 256), torch.randn(32, 256)],
    ),
    "Train SiLUMLP(256)": (
        lambda: TrainStep(SiLUMLP(256)),
        lambda: [torch.randn(32, 256), torch.randn(32, 256)],
    ),
}


def main():
    device = os.environ.get("TORCHLITE_BENCH_DEVICE", "")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    warmup = int(os.environ.get("TORCHLITE_BENCH_WARMUP", "5"))
    iters = int(os.environ.get("TORCHLITE_BENCH_ITERS", "50"))

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print("torchlite vs torch.compile benchmark")
    print(f"  device={device}, warmup={warmup}, iters={iters}")
    print(f"  PyTorch: {torch.__version__}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    results = []

    print("\n--- INFERENCE ---")
    for name, (model_fn, input_fn) in INFERENCE_MODELS.items():
        print(f"  Running {name}...", end="", flush=True)
        try:
            results.append(bench_inference(name, model_fn, input_fn, device, warmup, iters))
            print(" done")
        except Exception as e:
            print(f" FAILED: {e}")

    print("\n--- TRAINING ---")
    for name, (model_fn, input_fn) in TRAINING_MODELS.items():
        print(f"  Running {name}...", end="", flush=True)
        try:
            results.append(bench_training(name, model_fn, input_fn, device, warmup, iters))
            print(" done")
        except Exception as e:
            print(f" FAILED: {e}")

    print_results(results)


if __name__ == "__main__":
    main()
