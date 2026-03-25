"""
Measure proxy_call dispatch latency: bypass (fake_mode.dispatch) vs normal (func()).

Runs a simple MLP (linear -> relu -> linear) through both make_fx and torch.compile,
measuring cold start and warm start separately with mean/p50/p90 stats.

Usage:
    python benchmarks/proxy_dispatch_latency.py
    python benchmarks/proxy_dispatch_latency.py --size 2048 --iterations 50
"""

import argparse
import statistics
import time

import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._stats import simple_call_counter


def mlp(x, w1, b1, w2, b2):
    h = torch.nn.functional.linear(x, w1, b1)
    h = torch.nn.functional.relu(h)
    return torch.nn.functional.linear(h, w2, b2)


class MLPModule(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)

    def forward(self, x):
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))


def disable_bypass():
    patched = []
    ns = torch.ops.aten
    for op_name in dir(ns):
        op = getattr(ns, op_name, None)
        if op is None or not hasattr(op, "overloads"):
            continue
        for ol_name in op.overloads():
            ol = getattr(op, ol_name, None)
            if ol and getattr(ol, "_can_bypass_proxy_dispatch", False):
                ol._can_bypass_proxy_dispatch = False
                patched.append(ol)
    return patched


def restore_bypass(patched):
    for ol in patched:
        ol._can_bypass_proxy_dispatch = True


# -- make_fx timing --

def timed_make_fx(fn, args, tracing_mode):
    simple_call_counter.clear()
    start = time.perf_counter_ns()
    gm = make_fx(fn, tracing_mode=tracing_mode)(*args)
    elapsed = time.perf_counter_ns() - start
    del gm
    return elapsed / 1000


# -- torch.compile timing --

def timed_compile(model, example_input, backend="aot_eager"):
    torch.compiler.reset()
    compiled = torch.compile(model, backend=backend)
    start = time.perf_counter_ns()
    compiled(example_input)
    elapsed = time.perf_counter_ns() - start
    return elapsed / 1000


def fmt_row(label, times):
    s = sorted(times)
    p90_idx = min(int(len(s) * 0.9), len(s) - 1)
    return (
        f"{label:<10}"
        f" {statistics.mean(times):>10.1f}"
        f" {statistics.median(times):>10.1f}"
        f" {s[p90_idx]:>10.1f}"
        f" {s[0]:>10.1f}"
        f" {s[-1]:>10.1f}"
    )


HEADER = f"{'Mode':<10} {'Mean (µs)':>10} {'P50 (µs)':>10} {'P90 (µs)':>10} {'Min':>10} {'Max':>10}"
SEP = "-" * len(HEADER)


def print_section(title, bypass_times, normal_times, count):
    print()
    print("=" * len(HEADER))
    print(f"{title} ({count} runs, interleaved)")
    print("=" * len(HEADER))
    print(HEADER)
    print(SEP)
    print(fmt_row("bypass", bypass_times))
    print(fmt_row("normal", normal_times))
    diff = (statistics.mean(normal_times) - statistics.mean(bypass_times)) / statistics.mean(normal_times) * 100
    print(f"  diff: {diff:>+.1f}%  ({'bypass faster' if diff > 0 else 'bypass slower'})")


def run_interleaved(trace_fn_bypass, trace_fn_normal, count):
    """Run bypass/normal interleaved to reduce ordering bias."""
    bypass_times = []
    normal_times = []
    for i in range(count):
        if i % 2 == 0:
            bypass_times.append(trace_fn_bypass())
            normal_times.append(trace_fn_normal())
        else:
            normal_times.append(trace_fn_normal())
            bypass_times.append(trace_fn_bypass())
    return bypass_times, normal_times


def main():
    parser = argparse.ArgumentParser(description="Proxy dispatch latency benchmark")
    parser.add_argument("--size", type=int, default=4096, help="MLP hidden size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--iterations", type=int, default=30, help="Warm traces per mode")
    parser.add_argument("--cold-runs", type=int, default=10, help="Cold-start repetitions")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup traces before warm measurement")
    parser.add_argument("--tracing-mode", default="fake", choices=["fake", "symbolic"])
    parser.add_argument("--compile-backend", default="aot_eager",
                        help="torch.compile backend (default: aot_eager)")
    args = parser.parse_args()

    x = torch.randn(args.batch, args.size)
    w1 = torch.randn(args.size, args.size)
    b1 = torch.randn(args.size)
    w2 = torch.randn(args.size, args.size)
    b2 = torch.randn(args.size)
    fn_args = (x, w1, b1, w2, b2)
    model = MLPModule(args.size).eval()

    print(f"MLP: linear({args.size}) -> relu -> linear({args.size})")
    print(f"batch={args.batch}, warmup={args.warmup}, iterations={args.iterations}, "
          f"cold_runs={args.cold_runs}")
    print(f"make_fx mode={args.tracing_mode}, compile backend={args.compile_backend}")
    print()

    patched_ops = None

    def make_fx_bypass():
        return timed_make_fx(mlp, fn_args, args.tracing_mode)

    def make_fx_normal():
        nonlocal patched_ops
        patched_ops = disable_bypass()
        t = timed_make_fx(mlp, fn_args, args.tracing_mode)
        restore_bypass(patched_ops)
        return t

    def compile_bypass():
        return timed_compile(model, x, args.compile_backend)

    def compile_normal():
        nonlocal patched_ops
        patched_ops = disable_bypass()
        t = timed_compile(model, x, args.compile_backend)
        restore_bypass(patched_ops)
        return t

    # ===== make_fx =====
    print("--- make_fx ---")

    # Cold
    print("Measuring cold starts (interleaved)...")
    mfx_cold_bypass, mfx_cold_normal = run_interleaved(
        make_fx_bypass, make_fx_normal, args.cold_runs
    )

    # Warm
    print("Warming up...")
    for _ in range(args.warmup):
        make_fx_bypass()
    print("Running bypass (warm)...")
    mfx_warm_bypass = [make_fx_bypass() for _ in range(args.iterations)]

    # Grab counter snapshot
    bypass_ok = simple_call_counter.get("proxy_call.bypass_succeeded", 0)
    bypass_fail = simple_call_counter.get("proxy_call.bypass_failed", 0)
    normal_stat = simple_call_counter.get("proxy_call.normal_dispatch", 0)

    print("Warming up...")
    for _ in range(args.warmup):
        make_fx_normal()
    print("Running normal (warm)...")
    mfx_warm_normal = [make_fx_normal() for _ in range(args.iterations)]

    print_section("make_fx Cold Start", mfx_cold_bypass, mfx_cold_normal, args.cold_runs)
    print_section("make_fx Warm Start", mfx_warm_bypass, mfx_warm_normal, args.iterations)

    # ===== torch.compile =====
    print()
    print("--- torch.compile ---")

    # Cold
    print("Measuring cold starts (interleaved)...")
    tc_cold_bypass, tc_cold_normal = run_interleaved(
        compile_bypass, compile_normal, args.cold_runs
    )

    # Warm
    print("Warming up...")
    for _ in range(args.warmup):
        compile_bypass()
    print("Running bypass (warm)...")
    tc_warm_bypass = [compile_bypass() for _ in range(args.iterations)]

    print("Warming up...")
    for _ in range(args.warmup):
        compile_normal()
    print("Running normal (warm)...")
    tc_warm_normal = [compile_normal() for _ in range(args.iterations)]

    print_section(f"torch.compile ({args.compile_backend}) Cold Start",
                  tc_cold_bypass, tc_cold_normal, args.cold_runs)
    print_section(f"torch.compile ({args.compile_backend}) Warm Start",
                  tc_warm_bypass, tc_warm_normal, args.iterations)

    # Bypass rate
    total_ops = bypass_ok + bypass_fail + normal_stat
    if total_ops > 0:
        print()
        print(f"Bypass rate (make_fx last trace): {bypass_ok}/{total_ops} ops "
              f"({bypass_ok / total_ops * 100:.0f}%)")
        print(f"  succeeded={bypass_ok}, failed={bypass_fail}, normal={normal_stat}")


if __name__ == "__main__":
    main()
